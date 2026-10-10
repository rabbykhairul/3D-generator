"""Physical boundary regressions from the reported crossing/bridge photos."""
from pathlib import Path

import numpy as np
import pytest
from shapely.geometry import Polygon, LineString

from backend.reconstruction.engine import extract_annotations

ROOT = Path(__file__).resolve().parents[1] / 'data/projects'


def extract(project, photo, view):
    path = ROOT / project / 'photos' / (photo + '.png')
    if not path.exists():
        pytest.skip('Local merchant reference unavailable')
    return extract_annotations({'id': photo, 'view': view, 'path': str(path)})


def section(annotation, x):
    poly = Polygon([(p['x'], p['y']) for p in annotation['points']])
    bounds = poly.intersection(LineString([(x, 0), (x, 1080)])).bounds
    return np.array([bounds[1], bounds[3]])


def test_genuine_burgundy_sleeve_not_trimmed_at_crossing():
    ann = extract('3ca4f944b2214b72988db3bde88e0602',
                  '58acf228e33b4923ae85abfcb98fae8a', 'right')['annotations'][0]
    for x in (330, 350, 370, 390, 410):
        top, bottom = section(ann, x)
        assert 19 < bottom - top < 27
        assert 582 < bottom < 589


def test_small_lavender_rear_arm_spur_is_not_geometry():
    ann = extract('8dd590d016a44d268efcb319d3f86437',
                  '74bde624e654452390ec09168648706d', 'right')['annotations'][0]
    lower = np.array([section(ann, x)[1] for x in range(330, 411)])
    assert np.max(np.abs(np.diff(lower))) < .3
    assert 542 < section(ann, 350)[1] < 547


def test_semi_bridge_tracks_gold_bar_not_brow_height():
    ann = extract('3ca4f944b2214b72988db3bde88e0602',
                  '02f3fa42c9974879bf1ab62b8cc1e384', 'front')['annotations'][0]
    top, bottom = section(ann, 540)
    assert 537 < top < 544
    assert 545 < bottom < 552
    assert bottom - top < 12
