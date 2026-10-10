from pathlib import Path
import cv2
import numpy as np
import pytest
from shapely.geometry import Polygon, LineString
from backend.reconstruction import extract_annotations

ROOT=Path(__file__).resolve().parents[1]
CASES=[('pink','916b16f20d294e36bf92f935881d8f30','fc13d1010f344cd982c58f41ae474bf8'),
       ('lavender','8dd590d016a44d268efcb319d3f86437','74bde624e654452390ec09168648706d')]


@pytest.mark.parametrize('name,project,photo',CASES)
@pytest.mark.parametrize('size',[64,100,128,256,512,1080,2160,4096])
def test_outer_arm_identity_and_shaft_envelope_across_resolutions(tmp_path,name,project,photo,size):
    source=ROOT/f'data/projects/{project}/photos/{photo}.png'
    if not source.exists(): pytest.skip('Local merchant reference unavailable')
    original=cv2.imread(str(source))
    image=cv2.resize(original,(size,size),interpolation=cv2.INTER_AREA if size<1080 else cv2.INTER_CUBIC)
    path=tmp_path/'side.png';cv2.imwrite(str(path),image)
    result=extract_annotations({'id':'test','view':'right','path':str(path),'width':size,'height':size})
    if name=='lavender' and size<256:
        assert not result['annotations']  # uncertain is preferable to a hybrid arm
        return
    assert len(result['annotations'])==1,result
    ann=result['annotations'][0]
    assert not ann['approved']
    assert ann['anchors']['tip']['x']/size < .08  # old false branch ended at .13/.24
    assert ann['anchors']['hinge']['x']/size > .80
    polygon=Polygon([(p['x']/size,p['y']/size) for p in ann['points']])
    assert polygon.is_valid
    assert all(0<=p['x']<=size and 0<=p['y']<=size for p in ann['points'])
    for x in np.linspace(.48,.78,12):
        section=polygon.intersection(LineString([(x,0),(x,1)]))
        assert not section.is_empty
        thickness=section.bounds[3]-section.bounds[1]
        assert .006 < thickness < .045 + 1/size  # one source-pixel quantization allowance


@pytest.mark.parametrize('name,project,photo',CASES)
def test_branch_identity_mirrors_without_switching_to_other_tip(tmp_path,name,project,photo):
    source=ROOT/f'data/projects/{project}/photos/{photo}.png'
    if not source.exists(): pytest.skip('Local merchant reference unavailable')
    original=cv2.imread(str(source))
    path=tmp_path/'mirror.png';cv2.imwrite(str(path),cv2.flip(original,1))
    normal=extract_annotations({'id':'a','view':'right','path':str(source)})['annotations'][0]
    flipped=extract_annotations({'id':'b','view':'left','path':str(path)})['annotations'][0]
    assert flipped['part']=='temple-left'
    for anchor in ['hinge','tip']:
        assert flipped['anchors'][anchor]['x']==pytest.approx(original.shape[1]-1-normal['anchors'][anchor]['x'],abs=1)
        assert flipped['anchors'][anchor]['y']==pytest.approx(normal['anchors'][anchor]['y'],abs=1)
