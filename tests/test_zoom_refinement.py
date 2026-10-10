from copy import deepcopy
import cv2
import numpy as np
import pytest
from shapely.geometry import Polygon
from backend.store import get_project
from backend.reconstruction.engine import extract_annotations
from backend.reconstruction.zoom_refinement import refine_front_zoom


@pytest.mark.parametrize('size',[256,540,1080,2160])
def test_lavender_zoom_keeps_contained_lenses_and_source_coordinates(tmp_path,size):
    project=get_project('8dd590d016a44d268efcb319d3f86437')
    if project is None:pytest.skip('Local reference unavailable')
    photo=next(p for p in project['photos'] if p['view']=='front')
    original=cv2.imread(photo['path'])
    if original is None:pytest.skip('Local reference unavailable')
    path=tmp_path/'front.png';cv2.imwrite(str(path),cv2.resize(original,(size,size)))
    result=extract_annotations({**photo,'path':str(path)})
    outlines={a['part']:Polygon([(p['x'],p['y']) for p in a['points']]) for a in result['annotations']}
    assert {'front','lens-left','lens-right'}<=outlines.keys()
    assert Polygon(outlines['front'].exterior.coords,
                   [outlines[k].exterior.coords for k in ['lens-left','lens-right']]).is_valid
    assert all(0<=v<=size for poly in outlines.values() for xy in poly.exterior.coords for v in xy)
    assert result['zoom_refinement']['adds_source_detail'] is False
    if size==1080:
        assert result['zoom_refinement']['parts'][0]['applied']
        assert result['zoom_refinement']['parts'][0]['max_analysis_pixel_shift']<=3.5


def test_missing_lenses_are_not_invented():
    anns=[{'part':'front','points':[{'x':0,'y':0},{'x':100,'y':0},{'x':100,'y':100}]}]
    before=deepcopy(anns)
    assert not refine_front_zoom(anns)['applied']
    assert anns==before
