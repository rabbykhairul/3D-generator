from pathlib import Path

import cv2
import numpy as np
import pytest
from shapely.geometry import Polygon, Point, LineString

from backend.reconstruction import extract_annotations, reconstruct
from backend.reconstruction.nose_support import recover_visible_nose_support
from backend.contracts import Material

REFERENCE = Path(__file__).parent/'fixtures/semi_rimless/front.png'


def test_thin_brow_is_not_inflated_and_outer_tabs_set_attachments():
    result = extract_annotations(dict(id='front', view='front', path=str(REFERENCE)))
    shapes = {a['part']: Polygon([(p['x'],p['y']) for p in a['points']])
              for a in result['annotations']}
    solid = shapes['front'].difference(shapes['lens-left'].union(shapes['lens-right']))
    for x in (180,250,320,740,810,880):
        section = solid.intersection(LineString([(x,450),(x,510)]))
        assert 2 < section.length < 8  # Old uniform envelope widened these brows.
    front = next(a for a in result['annotations'] if a['part']=='front')
    for side,lo,hi in [('left',40,75),('right',1005,1040)]:
        anchor = front['anchors'][f'hinge-{side}']
        assert lo < anchor['x'] < hi and 515 < anchor['y'] < 555
        assert shapes['front'].covers(Point(anchor['x'],anchor['y']))


@pytest.mark.parametrize('size', [540, 1080, 2160])
@pytest.mark.parametrize('mirror', [False, True])
def test_visible_clear_pads_are_recovered_without_generic_completion(tmp_path, size, mirror):
    image = cv2.resize(cv2.imread(str(REFERENCE)), (size, size))
    if mirror:
        image = cv2.flip(image, 1)
    path = tmp_path/'front.png'
    cv2.imwrite(str(path), image)
    result = extract_annotations(dict(id='front', view='front', path=str(path)))
    shapes = {a['part']: Polygon([(p['x']*1080/size, p['y']*1080/size) for p in a['points']])
              for a in result['annotations']}
    assert {'nose-pad-left', 'nose-pad-right'} <= shapes.keys()
    for side in ('left', 'right'):
        pad = shapes[f'nose-pad-{side}']
        assert pad.is_valid and 200 < pad.area < 2000
        assert pad.bounds[1] > 575 and pad.bounds[3] < 685
        assert pad.convex_hull.area/pad.area < 1.65
        assert not pad.contains(Point(540, 660))
    assert result['nose_support']['depth_measured'] is False
    assert not any(a['approved'] for a in result['annotations'])


def test_absent_pads_are_not_manufactured(tmp_path):
    image = cv2.imread(str(REFERENCE))
    # Preserve rims and lens evidence, remove the independent pad pixels only.
    image[580:690, 482:507] = 255
    image[580:690, 572:597] = 255
    path = tmp_path/'without-pads.png'
    cv2.imwrite(str(path), image)
    result = extract_annotations(dict(id='front', view='front', path=str(path)))
    assert not any(a['part'].startswith('nose-pad-') for a in result['annotations'])


def test_existing_pads_are_not_overwritten_and_blank_photo_returns_none():
    image = cv2.imread(str(REFERENCE))
    result = extract_annotations(dict(id='front', view='front', path=str(REFERENCE)))
    assert recover_visible_nose_support(image, {'id':'front'}, result['annotations'])[0] == []
    assert recover_visible_nose_support(np.full((200, 300, 3), 255, np.uint8), {'id':'front'}, [])[0] == []


def test_recovered_pads_survive_actual_glb_generation(tmp_path):
    import trimesh
    result = extract_annotations(dict(id='front', view='front', path=str(REFERENCE)))
    project = dict(category='semi-rimless', annotations=[dict(a, approved=True) for a in result['annotations']],
                   photos=[dict(id='front', view='front', role='fitting', path=str(REFERENCE), width=1080, height=1080)],
                   material=Material().model_dump(), measurements={k:dict(value=v, source='inferred') for k,v in
                       dict(frame_width=140, lens_width=58, lens_height=40, bridge_width=22, temple_length=140).items()})
    metadata = reconstruct(project, tmp_path/'model')
    assert {'nose_pad_left', 'nose_pad_right'} <= set(metadata['parts'])
    assert metadata['checks']['front_parts_watertight']
    assert metadata['checks']['anatomy_geometry_consistent']
    scene = trimesh.load(tmp_path/'model/model.glb', force='scene', process=False)
    assert scene.geometry['nose_pad_left'].visual.material.baseColorTexture is not None
