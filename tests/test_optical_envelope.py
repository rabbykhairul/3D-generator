"""Optical-window geometry with occluders, not eyewear accuracy benchmarks."""
import cv2
import numpy as np
import pytest
from shapely.geometry import Polygon,Point
from shapely.ops import unary_union
from backend.reconstruction import hybrid_detection as hybrid
from backend.reconstruction.anatomy import attach_evidence


def optical_fixture(scale=1,concave=False):
    h,w=400*scale,900*scale
    image=np.full((h,w,3),255,np.uint8)
    scores=np.zeros((h,w),np.float32)
    theta=np.linspace(0,2*np.pi,720,endpoint=False)
    radial=1+.18*np.cos(theta*3) if concave else np.ones_like(theta)
    shapes=[Polygon(np.column_stack((cx+112*np.cos(theta)*radial,200+125*np.sin(theta)*radial))*scale)
            for cx in (230,670)]
    front=unary_union([p.buffer(20*scale) for p in shapes]+[
        Polygon(np.array([(330,155),(570,155),(570,180),(330,180)])*scale)])
    assert front.geom_type=='Polygon'
    cv2.fillPoly(image,[np.rint(front.exterior.coords).astype(np.int32)],(32,38,45))
    for p,cx in zip(shapes,(230,670)):
        contour=np.rint(p.exterior.coords).astype(np.int32)
        cv2.fillPoly(image,[contour],(180,190,205));cv2.fillPoly(scores,[contour],.96)
        start=((cx-113)*scale,200*scale);end=(cx*scale,257*scale)
        cv2.line(image,start,end,(45,50,58),15*scale)
        cv2.line(scores,start,end,0,15*scale)
    photo=dict(id='front',view='front',width=w,height=h,role='fitting')
    baseline=dict(annotations=[hybrid._annotation(photo,'front',front,1,1,'observed-source')],warnings=[])
    return image,scores,photo,shapes,baseline


@pytest.mark.parametrize('scale',[1,2])
def test_folded_arms_are_inside_optical_windows_instead_of_becoming_frame_material(scale,monkeypatch):
    image,scores,photo,truth,baseline=optical_fixture(scale)
    monkeypatch.setattr(hybrid,'infer_lenses',lambda _: (scores,dict(device='cpu')))
    result=hybrid.apply_hybrid(photo,image,image,baseline,1,1)
    assert result['hybrid_detection']['proposals_used']
    assert result['hybrid_detection']['status']=='applied'
    lenses=[a for a in result['annotations'] if a['part'].startswith('lens-')]
    assert len(lenses)==2 and all(not a['approved'] for a in lenses)
    for a,p,cx in zip(lenses,truth,(230,670)):
        q=Polygon([(pt['x'],pt['y']) for pt in a['points']])
        assert q.intersection(p).area/q.union(p).area>.96
        assert q.covers(Point(cx*scale,257*scale)), 'The arm silhouette must not cut into the optical window'
        assert 'occlusion-completion' in a['source']
    result=attach_evidence(photo,result,photo['width'],photo['height'])
    assert all(a['evidence']['state']=='inferred' for a in result['annotations'] if a['part'].startswith('lens-'))


def test_outer_envelope_preserves_broad_concave_lens_design_instead_of_convexifying_it():
    image,_,_,shapes,_=optical_fixture(concave=True)
    for shape in shapes:
        envelope,detail=hybrid.outer_lens_envelope(image,shape)
        assert envelope is not None and not detail['occlusion_completion']
        assert envelope.intersection(shape).area/envelope.union(shape).area>.99
        assert envelope.area/envelope.convex_hull.area<.99
        assert abs(envelope.area/envelope.convex_hull.area-shape.area/shape.convex_hull.area)<.005


def test_excessive_shape_completion_is_withheld():
    image,_,_,_,_=optical_fixture()
    # A deeply open/non-star-shaped outline does not establish a complete lens.
    shape=Polygon([(100,100),(300,100),(300,300),(100,300),(100,260),
                   (260,260),(260,140),(100,140)])
    envelope,detail=hybrid.outer_lens_envelope(image,shape)
    assert envelope is None


def test_support_band_prevents_the_outer_frame_background_edge_becoming_the_lens():
    image=np.full((300,400,3),255,np.uint8)
    cv2.ellipse(image,(200,150),(105,115),0,0,360,(60,60,60),-1)
    cv2.ellipse(image,(200,150),(99,109),0,0,360,(82,82,82),-1)
    mask=np.uint8(np.any(image!=255,axis=2))
    outer=hybrid._polygon(max(cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0],key=cv2.contourArea))
    candidate,detail=hybrid.refine_lens(image,outer.buffer(-1),support=outer)
    assert candidate is not None
    assert outer.boundary.distance(candidate.boundary)>=1.5
    assert detail['edge_support_fraction']>=.7
