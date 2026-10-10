"""Side extraction should follow physical arm boundaries through crossings."""

from pathlib import Path

import cv2
import numpy as np
import pytest
from shapely.geometry import LineString, Polygon

from backend.reconstruction import extract_annotations
from backend.reconstruction.engine import _foreground_mask
from backend.reconstruction.side_extraction import extract_side, recover_pale_side


def _section(annotation, x):
    polygon = Polygon([(p['x'], p['y']) for p in annotation['points']])
    line = polygon.intersection(LineString([(x, 0), (x, 1000)]))
    assert polygon.is_valid and not line.is_empty
    return line.bounds[1], line.bounds[3]


def test_crossing_does_not_make_a_hanging_block():
    mask = np.zeros((180, 440), np.uint8)
    # Slim main arm, a small genuine sleeve step, then a smooth bent ear tip.
    cv2.line(mask, (420, 48), (150, 68), 255, 9)
    cv2.line(mask, (150, 68), (70, 90), 255, 16)
    cv2.line(mask, (70, 90), (15, 145), 255, 17)
    cv2.rectangle(mask, (425, 20), (433, 100), 255, -1)
    # The other arm merges into the main silhouette before separating below.
    cv2.line(mask, (280, 59), (100, 136), 255, 12)
    result = extract_side({'id': 'crossing', 'view': 'right'}, mask, 1, 1)
    assert len(result['annotations']) == 1
    ann = result['annotations'][0]
    assert not ann['approved']
    assert ann['anchors']['hinge']['x'] > ann['anchors']['tip']['x']
    slim = _section(ann, 210)
    sleeve = _section(ann, 105)
    assert slim[1] - slim[0] < 18
    assert 12 < sleeve[1] - sleeve[0] < 24


def test_single_sleeve_and_tip_are_preserved():
    mask = np.zeros((140, 360), np.uint8)
    cv2.line(mask, (340, 35), (140, 49), 255, 7)
    cv2.line(mask, (140, 47), (75, 58), 255, 13)
    cv2.line(mask, (75, 58), (15, 115), 255, 14)
    cv2.rectangle(mask, (344, 15), (353, 90), 255, -1)
    result = extract_side({'id': 'single', 'view': 'left'}, mask, 1, 1)
    ann = result['annotations'][0]
    assert ann['part'] == 'temple-left'
    assert _section(ann, 100)[1] - _section(ann, 100)[0] > _section(ann, 250)[1] - _section(ann, 250)[0] + 4
    assert ann['anchors']['tip']['y'] > ann['anchors']['hinge']['y'] + 60


def test_thumbnail_keeps_two_pixel_metal_arm():
    mask = np.zeros((100, 100), np.uint8)
    cv2.line(mask, (85, 30), (15, 43), 255, 1)
    cv2.rectangle(mask, (88, 12), (94, 72), 255, -1)
    ann = extract_side({'id': 'thin', 'view': 'right'}, mask, 1, 1)['annotations'][0]
    assert ann['anchors']['tip']['x'] <= 20
    assert ann['anchors']['hinge']['x'] >= 75
    assert _section(ann, 45)[1] - _section(ann, 45)[0] < 4


def test_real_reference_crossing_has_no_merged_arm_block():
    path = (Path(__file__).resolve().parents[1] / 'data/projects'
            / '04863342af864ae0a4cabb8854273c49/photos'
            / '194e7d371e31448fadc30af0f17eeb14.png')
    if not path.exists():
        pytest.skip('Merchant reference image is not in this checkout')
    ann = extract_annotations({'id': 'reference', 'view': 'right', 'path': str(path)})['annotations'][0]
    # The other arm touches around x=405..390. Its merged mask is ~30 px
    # deep there; the visible main sleeve itself is about 18 px deep.
    for x in (405, 395, 390, 380, 370):
        top, bottom = _section(ann, x)
        assert bottom - top < 22
    # No narrow hanging tab remains anywhere through the whole crossing.
    sections = np.array([_section(ann, x) for x in range(350, 501)])
    assert np.max(sections[:, 1] - sections[:, 0]) < 21
    assert np.max(np.abs(np.diff(sections[:, 1]))) < 1.2
    assert _section(ann, 390)[1] - _section(ann, 390)[0] > _section(ann, 600)[1] - _section(ann, 600)[0] + 4
    assert len(ann['points']) < 100  # No per-pixel stair-step polygon.
    assert ann['source'] == 'single-visible-arm-branch-consensus-v1'


def test_real_crossing_remains_slender_at_100_pixels():
    path = (Path(__file__).resolve().parents[1] / 'data/projects'
            / '04863342af864ae0a4cabb8854273c49/photos'
            / '194e7d371e31448fadc30af0f17eeb14.png')
    if not path.exists():
        pytest.skip('Merchant reference image is not in this checkout')
    mask, _ = _foreground_mask(cv2.imread(str(path)))
    thumbnail = np.uint8(cv2.resize(mask, (100, 100), interpolation=cv2.INTER_AREA) > 127) * 255
    ann = extract_side({'id': 'thumbnail', 'view': 'right'}, thumbnail, 10.8, 10.8)['annotations'][0]
    assert ann['anchors']['tip']['x'] < 60  # Short gaps must not truncate the bent tip.
    for x in (405, 395, 390, 380, 370):
        top, bottom = _section(ann, x)
        assert bottom - top < 23


def test_pale_textured_side_recovers_continuous_shaft_and_unoccluded_root():
    path = (Path(__file__).resolve().parents[1] / 'data/projects'
            / 'ce81f4ffd03e4b8cac759b32ec5ee650/photos'
            / '071de32f5a964f5796f996d0fc31524c.png')
    if not path.exists():
        pytest.skip('New side reference is not in this checkout')
    image = cv2.imread(str(path))
    result = recover_pale_side(image, {'id': 'new', 'view': 'right'}, 1, 1)
    assert result is not None
    ann = result['annotations'][0]
    assert not ann['approved']
    assert 900 <= ann['anchors']['hinge']['x'] <= 913
    assert ann['anchors']['tip']['x'] < 50
    for x in (400, 500, 600, 700, 800, 900):
        top, bottom = _section(ann, x)
        assert 14 < bottom - top < 28
    # Genuine dark sleeve and bent tip remain thicker than the metal shaft.
    assert _section(ann, 300)[1] - _section(ann, 300)[0] > 20
    assert _section(ann, 100)[1] - _section(ann, 100)[0] > 30

    mirrored = recover_pale_side(cv2.flip(image, 1), {'id': 'mirrored', 'view': 'left'}, 1, 1)
    assert mirrored is not None
    mirror_ann = mirrored['annotations'][0]
    assert mirror_ann['part'] == 'temple-left'
    assert mirror_ann['anchors']['hinge']['x'] == pytest.approx(image.shape[1] - 1 - ann['anchors']['hinge']['x'])
    assert mirror_ann['anchors']['tip']['x'] == pytest.approx(image.shape[1] - 1 - ann['anchors']['tip']['x'])


def test_pale_recovery_declines_two_visible_distal_branches():
    path = Path(__file__).resolve().parents[1] / 'frontend/public/upload-guide/side.png'
    image = cv2.imread(str(path))
    assert recover_pale_side(image, {'id': 'lavender', 'view': 'right'}, 1, 1) is None
