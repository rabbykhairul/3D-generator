from pathlib import Path

import cv2
import numpy as np
import pytest
from shapely.geometry import Polygon

from backend.reconstruction.full_rim_recovery import recover_double_bridge_front


ROOT = Path(__file__).resolve().parents[1]
GREEN = ROOT / "data/projects/7def97639b034aa4be2d224429cc9f45/photos/c4d5383dff7949dcac8ea1d5ad279b6c.png"
SEMI = ROOT / "data/projects/3ca4f944b2214b72988db3bde88e0602/photos/02f3fa42c9974879bf1ab62b8cc1e384.png"


def _read(path):
    if not path.exists():
        pytest.skip("Saved source photo unavailable")
    image = cv2.imread(str(path))
    assert image is not None
    return image


def _check(result, width, height):
    assert result is not None
    assert [a["part"] for a in result["annotations"]] == [
        "front", "lens-left", "lens-right", "bridge-opening"]
    polygons = []
    for annotation in result["annotations"]:
        points = [(p["x"],p["y"]) for p in annotation["points"]]
        assert all(0 <= x < width and 0 <= y < height for x,y in points)
        poly = Polygon(points)
        assert poly.is_valid and poly.area > 100
        polygons.append(poly)
    assert all(polygons[0].contains(p) for p in polygons[1:])
    assert all(not polygons[i].intersects(polygons[j]) for i in range(1,4) for j in range(i+1,4))
    assert polygons[1].centroid.x < polygons[3].centroid.x < polygons[2].centroid.x


def test_green_double_bridge_recovers_three_openings_at_source_scale():
    source = _read(GREEN)
    h,w = source.shape[:2]
    for factor in (.5,1,2):
        working = cv2.resize(source,(round(w*factor),round(h*factor)))
        result = recover_double_bridge_front(working,{"id":"green","view":"front"},1/factor,1/factor)
        _check(result,w,h)
        front = result["annotations"][0]["points"]
        assert min(p["y"] for p in front) < h*.37
        assert h*.66 < max(p["y"] for p in front) < h*.71


def test_neutral_ground_shadow_does_not_become_rim_or_bridge_opening():
    source = _read(GREEN)
    baseline = recover_double_bridge_front(source,{"view":"front"},1,1)
    shadowed = source.copy()
    cv2.ellipse(shadowed,(540,769),(460,14),0,0,360,(221,221,221),-1)
    result = recover_double_bridge_front(shadowed,{"view":"front"},1,1)
    _check(result,*source.shape[1::-1])
    before = Polygon([(p['x'],p['y']) for p in baseline['annotations'][0]['points']])
    after = Polygon([(p['x'],p['y']) for p in result['annotations'][0]['points']])
    assert before.hausdorff_distance(after) < 3
    # A semi-rimless front has no third observed opening; do not invent one.
    assert recover_double_bridge_front(_read(SEMI),{"view":"front"},1,1) is None
