"""Regression checks for conservative semi-rimless photo suggestions."""

from pathlib import Path

import cv2
import numpy as np
import pytest
from shapely.geometry import Polygon

from backend.reconstruction.construction_detection import analyze_construction


ROOT = Path(__file__).resolve().parents[1]
SEMI = ROOT / "data/projects/3ca4f944b2214b72988db3bde88e0602/photos/02f3fa42c9974879bf1ab62b8cc1e384.png"
FULL = ROOT / "data/projects/ce81f4ffd03e4b8cac759b32ec5ee650/photos/0614da10abb84db08c7ed2110a9f0993.png"


def _photo(path):
    if not path.exists():
        pytest.skip("Saved photo fixture is unavailable")
    image = cv2.imread(str(path))
    assert image is not None
    return image


def _geometry(result, width, height):
    assert result["construction"]["category"] == "semi-rimless"
    assert result["construction"]["lower_support"] == "unverified"
    assert [a["part"] for a in result["annotations"]] == ["front", "lens-left", "lens-right"]
    polygons = []
    for ann in result["annotations"]:
        points = [(p["x"], p["y"]) for p in ann["points"]]
        assert all(0 <= x < width and 0 <= y < height for x, y in points)
        polygon = Polygon(points)
        assert polygon.is_valid and polygon.area > 100
        polygons.append(polygon)
    assert all(polygons[0].contains(p) for p in polygons[1:])
    assert not polygons[1].intersects(polygons[2])
    assert set(result["annotations"][0]["anchors"]) == {"rim-end-left", "rim-end-right", "rim-outer-end-left", "rim-outer-end-right", "hinge-left", "hinge-right"}
    for side in ('left','right'):
        anchor=result['annotations'][0]['anchors'][f'hinge-{side}']
        from shapely.geometry import Point
        assert polygons[0].covers(Point(anchor['x'],anchor['y']))


def test_saved_semi_rimless_front_at_multiple_scales_and_layouts():
    image = _photo(SEMI)
    h, w = image.shape[:2]
    for factor in (.5, 1, 2):
        resized = cv2.resize(image, (round(w*factor), round(h*factor)))
        result = analyze_construction(resized, {"id": "semi", "view": "front"}, 1/factor, 1/factor)
        assert result is not None
        _geometry(result, w, h)
    padded = np.full((1300, 1500, 3), 255, dtype=np.uint8)
    padded[100:1180, 200:1280] = image
    result = analyze_construction(padded, {"view": "front"}, 1, 1)
    assert result is not None
    _geometry(result, 1500, 1300)
    # A wider crop places the product close to the image edges and changes
    # canvas aspect ratio, while retaining the actual lens pixels.
    cropped = padded[100:870, 175:1325]
    result = analyze_construction(cropped, {"view": "front"}, 1, 1)
    assert result is not None
    _geometry(result, cropped.shape[1], cropped.shape[0])


def test_lower_edge_must_be_present_and_not_colored_full_rim():
    image = _photo(SEMI)
    erased = image.copy()
    erased[650:] = 255
    assert analyze_construction(erased, {"view": "front"}, 1, 1) is None
    full = _photo(FULL)
    assert analyze_construction(full, {"view": "front"}, 1, 1) is None
