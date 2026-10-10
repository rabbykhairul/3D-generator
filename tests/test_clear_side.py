"""Neutral arm must reach its own tip and exclude the rear arm/front rim."""
from pathlib import Path
import cv2
import pytest
from shapely.geometry import Polygon, Point
from backend.store import get_project
from backend.reconstruction.engine import extract_annotations


@pytest.mark.parametrize('size', [540,1080,2160])
@pytest.mark.parametrize('mirror', [False,True])
def test_water_clear_outer_arm(tmp_path: Path, size, mirror):
    project=get_project('b0bc9b703db9463abab0a9a4d30c0b4d')
    if not project:
        pytest.skip('Local reference unavailable')
    photo=next(p for p in project['photos'] if p['view']=='right')
    image=cv2.resize(cv2.imread(photo['path']),(size,size))
    if mirror:
        image=cv2.flip(image,1)
    path=tmp_path/'side.png';cv2.imwrite(str(path),image)
    result=extract_annotations({'id':'clear','view':'left' if mirror else 'right','path':str(path)})
    assert len(result['annotations'])==1, result['warnings']
    ann=result['annotations'][0]
    scale=size/1080
    def mapped(x,y):return ((size-1-x)/scale if mirror else x/scale,y/scale)
    poly=Polygon([mapped(p['x'],p['y']) for p in ann['points']])
    assert poly.is_valid
    assert poly.contains(Point(600,535))
    assert poly.contains(Point(80,715))
    assert not poly.contains(Point(300,670)) # opposite arm
    assert not poly.contains(Point(920,600)) # front rim
    assert not poly.contains(Point(85,746)) # contact shadow
    hinge=mapped(**ann['anchors']['hinge']);tip=mapped(**ann['anchors']['tip'])
    assert 830<hinge[0]<885 and tip[0]<65
    assert poly.distance(Point(*hinge))<1
