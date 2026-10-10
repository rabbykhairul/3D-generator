"""Component ownership, source pixels, optical topology and export regressions."""
from copy import deepcopy
import cv2
import numpy as np
import pytest
from shapely.geometry import Polygon,box
from backend.reconstruction.sam3_components import apply_front_components,apply_side_components,component_proposals
from backend.reconstruction.hybrid_detection import _annotation
from backend.reconstruction.anatomy import audit_photo


def scene():
    image=np.full((400,800,3),255,np.uint8)
    photo=dict(id='front',view='front',width=800,height=400)
    lenses=[box(100,100,350,320),box(450,100,700,320)]
    annotations=[_annotation(photo,'lens-'+side,p,1,1,'source-traced') for side,p in zip(('left','right'),lenses)]
    for p in lenses:cv2.polylines(image,[np.array(p.exterior.coords,np.int32)],True,(160,160,160),2)
    result=dict(annotations=annotations,warnings=[],hybrid_detection=dict(requested_backend='sam3',proposal_backend='sam3',inference_ran=True,proposals_used=True,status='partial',reason='Lenses only',runtime={}))
    return image,photo,result


def proposal(p,role,score=.9):
    return dict(role=role,score=score,points=list(p.exterior.coords)[:-1],holes=[])


def draw(image,p,color=(45,50,55)):
    cv2.fillPoly(image,[np.rint(p.exterior.coords).astype(np.int32)],color)


def run(image,photo,result,proposals):
    result=deepcopy(result);result['hybrid_detection']['runtime']['component_proposals']=proposals
    return apply_front_components(photo,image,result)


def test_visible_support_bridge_and_pads_remain_separate_unapproved_components():
    image,photo,result=scene()
    bridge=box(348,125,452,135)
    pads=[box(358,180,372,215),box(428,179,442,216)]
    for p in [bridge,*pads]:draw(image,p)
    output=run(image,photo,result,[proposal(bridge,'bridge'),*[proposal(p,'nose-pad') for p in pads]])
    parts={a['part']:a for a in output['annotations']}
    assert {'bridge-hardware','nose-pad-left','nose-pad-right'}<=parts.keys()
    assert 'front' not in parts and not any(a['approved'] for a in output['annotations'])
    report=audit_photo(photo,output['annotations'],'rimless')
    assert not report['blocking']
    assert next(n for n in report['nodes'] if n['id']=='bridge')['state']=='observed'
    assert not output['component_detection']['depth_measured']


def test_connected_brow_is_not_filled_into_an_outer_rim():
    image,photo,result=scene()
    support=box(150,88,650,99).union(box(390,98,410,138)).union(box(348,128,452,138))
    draw(image,support)
    output=run(image,photo,result,[proposal(support,'frame')])
    parts={a['part']:a for a in output['annotations']}
    assert 'front-support' in parts and 'front' not in parts
    assert output['construction']['category'] is None


def test_blank_pixels_and_peripheral_false_pads_are_rejected():
    image,photo,result=scene();pad=box(360,180,375,215)
    output=run(image,photo,result,[proposal(pad,'nose-pad')])
    assert len(output['annotations'])==2
    false_pad=box(85,170,100,205);draw(image,false_pad)
    output=run(image,photo,result,[proposal(false_pad,'nose-pad')])
    assert len(output['annotations'])==2
    assert any(d['reason']=='component-location-or-material-conflict' for d in output['component_detection']['rejected'])


def test_competing_pad_boundaries_do_not_select_the_highest_score_blindly():
    image,photo,result=scene();a=box(355,180,370,215);b=box(378,180,393,215)
    for p in [a,b]:draw(image,p)
    output=run(image,photo,result,[proposal(a,'nose-pad',.99),proposal(b,'nose-pad',.75)])
    assert not any(a['part']=='nose-pad-left' for a in output['annotations'])
    assert any(d['reason']=='competing-component-regions' for d in output['component_detection']['rejected'])


def test_whole_glasses_mask_cannot_be_treated_as_frame_material():
    image,photo,result=scene();outer=box(80,80,720,340);draw(image,outer)
    output=run(image,photo,result,[proposal(outer,'frame')])
    assert not any(a['part']=='front' for a in output['annotations'])


def test_frame_mask_preserves_holes_until_optical_material_checks():
    image,photo,result=scene();mask=np.zeros(image.shape[:2],np.uint8)
    cv2.rectangle(mask,(80,80),(720,340),1,-1)
    cv2.rectangle(mask,(100,100),(350,320),0,-1);cv2.rectangle(mask,(450,100),(700,320),0,-1)
    image[mask>0]=(35,45,55)
    proposals=component_proposals(np.array([mask]),[.9],['frame'])
    assert len(proposals)==1 and len(proposals[0]['holes'])==2
    output=run(image,photo,result,proposals)
    assert any(a['part']=='front' for a in output['annotations'])
    assert not audit_photo(photo,output['annotations'])['blocking']


def test_existing_reviewed_components_are_preserved():
    image,photo,result=scene();pad=box(360,180,375,215);draw(image,pad)
    saved=_annotation(photo,'nose-pad-left',pad,1,1,'merchant-reviewed');saved['approved']=True
    result['annotations'].append(saved)
    output=run(image,photo,result,[proposal(pad,'nose-pad')])
    assert next(a for a in output['annotations'] if a['part']=='nose-pad-left')==saved


@pytest.mark.parametrize('side',['left','right'])
def test_sloping_lens_rim_strip_is_not_a_separate_nose_pad(side):
    image,photo,result=scene()
    # Axis-aligned bounds are wide, but the actual strip is only 5 px thick.
    # The optical edge slopes with it; both real pads and rim strips can have
    # strong image edges, so source contrast alone cannot disambiguate them.
    lens=Polygon([(100,100),(350,100),(310,320),(100,320)])
    strip=Polygon([(338,158),(343,158),(325,258),(320,258)])
    if side=='right':
        from shapely.affinity import scale
        lens=scale(lens,xfact=-1,yfact=1,origin=(400,0))
        strip=scale(strip,xfact=-1,yfact=1,origin=(400,0))
    result['annotations'][0 if side=='left' else 1]=_annotation(photo,'lens-'+side,lens,1,1,'source-traced')
    draw(image,strip)
    output=run(image,photo,result,[proposal(strip,'nose-pad')])
    assert not any(a['part'].startswith('nose-pad') for a in output['annotations'])
    assert any(d['reason']=='rim-strip-is-not-separate-pad-evidence' for d in output['component_detection']['rejected'])


def test_side_hardware_updates_only_the_selected_arm_root():
    image=np.full((300,600,3),255,np.uint8);photo=dict(id='side',view='right',width=600,height=300)
    arm=box(40,125,510,139);hinge=box(490,124,510,140);draw(image,arm);draw(image,hinge,(20,20,20))
    ann=_annotation(photo,'temple-right',arm,1,1,'automatic-sam3-side-v1');ann['anchors']={'hinge':dict(x=504,y=132),'tip':dict(x=43,y=132)}
    result=dict(annotations=[ann],hybrid_detection=dict(runtime=dict(component_proposals=[proposal(hinge,'hinge')])))
    output=apply_side_components(photo,image,result)
    assert {'temple-right','hinge-right'}=={a['part'] for a in output['annotations']}
    assert output['annotations'][0]['anchor_provenance']['hinge'].startswith('observed')
    assert not audit_photo(photo,output['annotations'])['blocking']
    draw(image,box(290,125,310,145),(20,20,20))
    result['hybrid_detection']['runtime']['component_proposals']=[proposal(box(290,125,310,145),'hinge')]
    assert len(apply_side_components(photo,image,result)['annotations'])==1


def test_approved_support_and_pads_export_real_textured_closed_meshes(tmp_path):
    import trimesh
    from backend.contracts import Material
    from backend.reconstruction import reconstruct
    image,photo,result=scene()
    support=box(150,88,650,99).union(box(390,98,410,138)).union(box(348,128,452,138));pad=box(360,180,375,215)
    for p in [support,pad]:draw(image,p)
    output=run(image,photo,result,[proposal(support,'frame'),proposal(pad,'nose-pad')])
    path=tmp_path/'front.png';cv2.imwrite(str(path),image);photo.update(path=str(path),role='fitting')
    project=dict(category='rimless',photos=[photo],annotations=[dict(a,approved=True) for a in output['annotations']],material=Material().model_dump(),measurements={k:dict(value=v,source='inferred') for k,v in dict(frame_width=140,lens_width=58,lens_height=40,bridge_width=22,temple_length=140).items()})
    metadata=reconstruct(project,tmp_path/'model')
    assert {'front_support','nose_pad_left'}<=set(metadata['parts'])
    assert 'bridge' not in metadata['parts'],'No invented straight bridge over a source support'
    loaded=trimesh.load(tmp_path/'model/model.glb',force='scene',process=False)
    for name in ('front_support','nose_pad_left'):
        geometry=loaded.geometry[name].copy()
        # GLB keeps separate vertices at UV/normal seams. Weld equal positions
        # for the topology check while separately retaining the texture check.
        geometry.merge_vertices(merge_tex=True,merge_norm=True)
        assert geometry.is_watertight and np.isfinite(geometry.vertices).all()
        assert loaded.geometry[name].visual.material.baseColorTexture is not None
