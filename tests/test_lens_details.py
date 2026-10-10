import cv2
import numpy as np
import pytest
from shapely.geometry import Polygon,Point
from backend.reconstruction.lens_details import visible_lens_details
from backend.reconstruction.engine import extract_annotations
from backend.store import get_project


def test_small_print_survives_but_rear_arm_does_not(tmp_path):
    image=np.full((1080,1080,3),255,np.uint8)
    points=np.array([[100,200],[500,200],[500,700],[100,700]])
    cv2.fillPoly(image,[points],(100,100,100))
    cv2.circle(image,(155,275),9,(35,35,35),3)
    cv2.line(image,(150,400),(350,650),(25,25,25),25)
    path=tmp_path/'source.png';cv2.imwrite(str(path),image)
    details=visible_lens_details({'path':str(path)}, {'left':points},{})
    assert len(details)==1 and details[0]['kind']=='lens-print-appearance'
    assert details[0]['polygon'].contains(Point(155,275))
    assert not details[0]['polygon'].contains(Point(250,525))


def test_actual_silver_print_and_mounts():
    project=get_project('8758d7b8ecf343c4b5f321221757954d')
    if project is None:pytest.skip('Local reference unavailable')
    photo=next(p for p in project['photos'] if p['view']=='front')
    anns=extract_annotations(photo)['annotations']
    shapes={a['part']:Polygon([(p['x'],p['y'])for p in a['points']])for a in anns}
    lenses={side:np.array(shapes[f'lens-{side}'].exterior.coords)for side in ('left','right')}
    pads={side:shapes[f'nose-pad-{side}']for side in ('left','right')}
    details=visible_lens_details(photo,lenses,pads)
    assert any(d['kind']=='lens-print-appearance' and d['side']=='left' for d in details)
    assert {d['side']for d in details if d['kind']=='pad-mount-appearance'}=={'left','right'}
    assert not any(d['polygon'].contains(Point(200,600)) for d in details)
