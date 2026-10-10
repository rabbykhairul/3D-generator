"""Image-guided recovery of an opaque side arm with a mark and rear crossing."""

from pathlib import Path

import cv2
import numpy as np
import pytest
from shapely.geometry import LineString, Polygon

from backend.reconstruction.side_extraction import recover_opaque_side, _repair_short_lower_overlap


ROOT = Path(__file__).resolve().parents[1]
BLUE = (ROOT / 'data/projects/c8e4014e30044160ace24fe171a57b5e/photos'
        / 'e7304c95abdb452e811de55581bf5ac5.png')


def test_local_overlap_repair_requires_second_arm_and_keeps_wider_shaft():
    upper = 50 + np.arange(200) * .12
    lower = upper + 20
    lower[:50] += 15  # genuine broad shaft, outside the crossing
    clean = lower.copy()
    lower[84:100] += np.linspace(1, 14, 16)
    split = np.arange(200) >= 100
    _, repaired = _repair_short_lower_overlap(upper, lower, split, split)
    assert np.allclose(repaired, clean)
    _, unchanged = _repair_short_lower_overlap(upper, lower, np.zeros(200, bool), split)
    assert np.array_equal(unchanged, lower)  # no rear-arm evidence, no flattening
    corner = upper.copy()
    corner[91:] -= 8
    _, unchanged = _repair_short_lower_overlap(corner, lower, split, split)
    assert np.array_equal(unchanged, lower)  # preserve an intentional upper-edge step


def _section(polygon, x):
    line = polygon.intersection(LineString([(x, 0), (x, 1080)]))
    assert not line.is_empty
    return line.bounds[1], line.bounds[3]


def _outline(annotation):
    polygon = Polygon([(point['x'], point['y']) for point in annotation['points']])
    assert polygon.is_valid
    return polygon


def test_opaque_side_recovery_follows_physical_arm_and_joint_seam():
    if not BLUE.exists():
        pytest.skip('Blue local product photograph not installed')
    image = cv2.imread(str(BLUE))
    result = recover_opaque_side(image, {'id': 'blue', 'view': 'right'}, 1, 1)
    assert result is not None
    ann = result['annotations'][0]
    poly = _outline(ann)
    assert not ann['approved']
    assert ann['part'] == 'temple-right'
    assert 866 <= ann['anchors']['hinge']['x'] <= 869
    assert ann['anchors']['tip']['x'] < 50
    # The overlap below the main arm is a rear arm, not a hanging tab.
    bottoms = [_section(poly, x)[1] for x in range(400, 471)]
    assert max(bottoms) < 563
    assert max(abs(b - a) for a, b in zip(bottoms, bottoms[1:])) < 1.5
    assert 19 < _section(poly, 420)[1] - _section(poly, 420)[0] < 27
    # A smooth but contaminated repair used to retain a shallow 4px bump.
    # Between the clean shaft and the first separated rear-arm column, the
    # lower boundary must follow the local chord, not that residual triangle.
    for x in range(425, 486):
        expected = _section(poly, 414)[1] + (x - 414) / (497 - 414) * (
            _section(poly, 497)[1] - _section(poly, 414)[1])
        assert _section(poly, x)[1] <= expected + .6
    # A later, genuine thick shaft survives that local crossing repair.
    assert _section(poly, 550)[1] - _section(poly, 550)[0] > 35
    # The printed white mark is an interior cutout, not the outside edge.
    for x in (820, 830, 840, 850):
        top, bottom = _section(poly, x)
        assert top < 504 and bottom > 540

    mirror = recover_opaque_side(cv2.flip(image, 1),
                                 {'id': 'mirror', 'view': 'left'}, 1, 1)
    assert mirror is not None
    mirror_ann = mirror['annotations'][0]
    mirror_poly = _outline(mirror_ann)
    assert mirror_ann['part'] == 'temple-left'
    for anchor in ('hinge', 'tip'):
        assert mirror_ann['anchors'][anchor]['x'] == pytest.approx(
            image.shape[1] - 1 - ann['anchors'][anchor]['x'])
    for x in (420, 550, 830):
        assert _section(mirror_poly, image.shape[1] - 1 - x) == pytest.approx(
            _section(poly, x), abs=.01)


@pytest.mark.parametrize('relative_path', [
    'data/projects/04863342af864ae0a4cabb8854273c49/photos/194e7d371e31448fadc30af0f17eeb14.png',
    'data/projects/ce81f4ffd03e4b8cac759b32ec5ee650/photos/071de32f5a964f5796f996d0fc31524c.png',
    'frontend/public/upload-guide/side.png',
])
def test_opaque_route_leaves_other_side_styles_to_their_extractors(relative_path):
    path = ROOT / relative_path
    if not path.exists():
        pytest.skip('Local reference photograph not installed')
    image = cv2.imread(str(path))
    assert recover_opaque_side(image, {'id': 'other', 'view': 'right'}, 1, 1) is None
