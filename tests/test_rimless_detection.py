from pathlib import Path
import cv2
import numpy as np
import pytest
from shapely.geometry import Polygon,Point
from backend.reconstruction.engine import extract_annotations
from backend.reconstruction.rimless_detection import recover_rimless_front

REFERENCE=Path(__file__).parent/'fixtures/rimless/front.png'

@pytest.mark.parametrize('size',[540,1080,2160])
@pytest.mark.parametrize('mirror',[False,True])
def test_rimless_independent_lenses(tmp_path,size,mirror):
    im=cv2.resize(cv2.imread(str(REFERENCE)),(size,size),interpolation=cv2.INTER_AREA if size<1080 else cv2.INTER_CUBIC)
    if mirror:im=cv2.flip(im,1)
    path=tmp_path/'front.png';cv2.imwrite(str(path),im)
    result=extract_annotations({'id':'rimless','view':'front','path':str(path)})
    assert result['construction']['category']=='rimless'
    shapes={a['part']:Polygon([(p['x']*1080/size,p['y']*1080/size)for p in a['points']])for a in result['annotations']}
    assert 'front' not in shapes
    assert {'lens-left','lens-right','bridge-hardware','nose-pad-left','nose-pad-right'}<=shapes.keys()
    assert all(p.is_valid for p in shapes.values())
    assert shapes['lens-left'].contains(Point(200,610)) # rear arm is NOT a lens hole
    assert shapes['lens-right'].contains(Point(850,610))
    assert not shapes['lens-left'].contains(Point(280,745)) # ground shadow
    assert shapes['bridge-hardware'].area<5000
    assert not any(a['approved'] for a in result['annotations'])


def test_missing_rims_are_not_enough_to_claim_rimless():
    blank=np.full((1080,1080,3),255,np.uint8)
    assert recover_rimless_front(blank,{}) is None
    for x in (280,800):cv2.ellipse(blank,(x,600),(180,130),0,0,360,(40,40,40),5)
    assert recover_rimless_front(blank,{}) is None


def test_rimless_generation_uses_observed_hardware(tmp_path):
    from backend.reconstruction.engine import reconstruct
    from backend.contracts import Material
    import trimesh
    photos=[{'id':view,'view':view,'role':'fitting','path':str(REFERENCE.parent/f'{view}.png'),
             'width':1080,'height':1080}for view in ('front','right')]
    anns=[a for photo in photos for a in extract_annotations(photo)['annotations']]
    for a in anns:a['approved']=True
    project={'id':'isolated-rimless','revision':1,'category':'rimless','photos':photos,'annotations':anns,
             'material':Material().model_dump(),'measurements':{k:{'value':v,'source':'inferred'}for k,v in
             {'frame_width':140,'lens_width':54,'lens_height':34,'bridge_width':18,'temple_length':140}.items()}}
    result=reconstruct(project,tmp_path/'model')
    assert result['checks']['front_parts_watertight']
    scene=trimesh.load(tmp_path/'model/model.glb',force='scene',process=False)
    assert {'bridge','nose_pad_left','nose_pad_right','lens_left','lens_right','temple_right'}<=set(scene.geometry)
    assert 'front_frame' not in scene.geometry
    assert scene.geometry['bridge'].visual.material.baseColorTexture is not None
