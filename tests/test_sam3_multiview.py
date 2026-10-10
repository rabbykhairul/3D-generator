"""Registration bounds, source ownership and inferred cross-view evidence."""
from copy import deepcopy
import cv2
import numpy as np
import pytest
from pydantic import ValidationError
from shapely.geometry import box
from backend.contracts import Annotation,ComponentRegistration
from backend.reconstruction import sam3_multiview as mv
from backend.reconstruction.anatomy import audit_photo
from backend.reconstruction.hybrid_detection import _annotation
from test_sam3_components import scene,draw


def photos(tmp_path):
    image,front,result=scene();front['path']=str(tmp_path/'front.png');front['role']='fitting';cv2.imwrite(front['path'],image)
    source={**front,'id':'angle','view':'front-left','path':str(tmp_path/'angle.png')}
    pad=box(360,180,375,215);draw(image,pad);cv2.imwrite(source['path'],image)
    component=_annotation(source,'nose-pad-left',pad,1,1,'automatic-sam3-nose-pad-left-source-v1')
    return dict(photos=[front,source]),front,result,component


def test_registered_extra_view_adds_inferred_component_without_changing_project(tmp_path,monkeypatch):
    project,front,result,component=photos(tmp_path);before=deepcopy(project)
    row=dict(photo_id='angle',view='front-left',reason='registered',inliers=20,source_to_front_homography=np.eye(3).tolist())
    monkeypatch.setattr(mv,'analyze_multiview',lambda *a,**k:dict(views=[row]))
    calls=[]
    def inference(*a,**k):
        calls.append(a[0]['id']);return dict(annotations=[component],component_detection=dict(accepted=[dict(part='nose-pad-left',score=.9)]))
    monkeypatch.setattr(mv,'apply_hybrid',inference)
    output=mv.combine_reference_components(project,front,result)
    assert calls==['angle'] and project==before and len(result['annotations'])==2
    pad=next(a for a in output['annotations'] if a['part']=='nose-pad-left')
    assert not pad['approved'] and pad['registration']['source_photo_id']=='angle'
    assert pad['registration']['source_points']==component['points']
    Annotation(**pad)
    assert next(n for n in output['anatomy']['nodes'] if n['id']=='nose-pad-left')['state']=='inferred'


@pytest.mark.parametrize('view,registration',[
    ('front-left',dict(reason='appearance_consistent',inliers=30)),
    ('front-left',dict(reason='registered',inliers=3,source_to_front_homography=np.eye(3).tolist())),
])
def test_no_component_inference_without_reliable_registration(tmp_path,monkeypatch,view,registration):
    project,front,result,_=photos(tmp_path)
    monkeypatch.setattr(mv,'analyze_multiview',lambda *a,**k:dict(views=[dict(view=view,photo_id='angle',**registration)]))
    monkeypatch.setattr(mv,'apply_hybrid',lambda *a,**k:pytest.fail('No model should run for unsupported warp'))
    output=mv.combine_reference_components(project,front,result)
    assert [{k:v for k,v in a.items() if k!='evidence'} for a in output['annotations']]==result['annotations']


def test_local_registration_cannot_transfer_component_outside_inlier_hull(tmp_path,monkeypatch):
    project,front,result,component=photos(tmp_path)
    fit=dict(source_to_front_homography=np.eye(3).tolist(),inliers=20,front_coverage_polygon=[[50,50],[200,50],[200,100],[50,100]])
    monkeypatch.setattr(mv,'analyze_multiview',lambda *a,**k:dict(views=[dict(view='front-left',photo_id='angle',reason='matches_do_not_cover_frame',component_registration=fit)]))
    monkeypatch.setattr(mv,'apply_hybrid',lambda *a,**k:dict(annotations=[component]))
    output=mv.combine_reference_components(project,front,result)
    assert [{k:v for k,v in a.items() if k!='evidence'} for a in output['annotations']]==result['annotations']


def test_heldout_photo_is_never_sent_to_registration_or_inference(tmp_path,monkeypatch):
    project,front,result,_=photos(tmp_path);project['photos'][1]['validation']=True
    monkeypatch.setattr(mv,'analyze_multiview',lambda *a,**k:pytest.fail('Heldout source must be excluded'))
    monkeypatch.setattr(mv,'apply_hybrid',lambda *a,**k:pytest.fail('Heldout source must not enter model'))
    assert mv.combine_reference_components(project,front,result)['annotations']==result['annotations']


def test_component_registration_rejects_malformed_or_singular_transform():
    for matrix in ([[1,0],[0,1]],[[1,0,0],[0,0,0],[0,0,1]],[[1,0,0],[0,float('nan'),0],[0,0,1]]):
        with pytest.raises(ValidationError):
            ComponentRegistration(source_photo_id='source',source_points=[dict(x=x,y=y) for x,y in [(1,1),(5,1),(5,5)]],source_to_front_homography=matrix)


def test_registered_texture_uses_source_colours_and_rejects_validation(tmp_path):
    from backend.reconstruction.component_texture import component_photo
    project,front,_,component=photos(tmp_path)
    source=project['photos'][1];image=np.full((400,800,3),(30,75,160),np.uint8);cv2.imwrite(source['path'],image)
    component['registration']=dict(source_photo_id='angle',source_points=component['points'],source_to_front_homography=np.eye(3).tolist(),method='inferred-front-plane-registration')
    mapped=component_photo(component,front,project['photos'],tmp_path/'export')
    assert np.array_equal(cv2.imread(mapped['path'])[195,368],[30,75,160])
    source['role']='validation'
    with pytest.raises(ValueError,match='Validation photo'):
        component_photo(component,front,project['photos'],tmp_path/'other')


def test_registered_component_exports_with_inferred_provenance_and_source_texture(tmp_path,monkeypatch):
    from backend.reconstruction import reconstruct
    from backend.contracts import Material
    project,front,result,component=photos(tmp_path)
    monkeypatch.setattr(mv,'analyze_multiview',lambda *a,**k:dict(views=[dict(view='front-left',photo_id='angle',reason='registered',inliers=20,source_to_front_homography=np.eye(3).tolist())]))
    monkeypatch.setattr(mv,'apply_hybrid',lambda *a,**k:dict(annotations=[component]))
    r=mv.combine_reference_components(project,front,result)
    project.update(category='rimless',annotations=[Annotation(**dict(a,approved=True)).model_dump() for a in r['annotations']],material=Material().model_dump(),measurements={k:dict(value=v,source='inferred') for k,v in dict(frame_width=140,lens_width=58,lens_height=40,bridge_width=22,temple_length=140).items()})
    metadata=reconstruct(project,tmp_path/'model')
    assert 'nose_pad_left' in metadata['parts']
    node=next(n for n in metadata['anatomy']['nodes'] if n['id']=='nose-pad-left')
    assert node['state']=='inferred'
