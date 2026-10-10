"""Exercise real contours at small and large derivative resolutions."""

from pathlib import Path

import pytest

from scripts.evaluate_resolution_matrix import default_sources, evaluate_one, validate_annotations, missing_required_parts


@pytest.mark.parametrize("family,view,size", [
    ("blue", "front", 64), ("blue", "front", 100),
    ("blue", "right", 100), ("blue", "right", 4096),
    ("transparent", "front", 100), ("transparent", "front", 4096),
    ("gold", "front", 256), ("gold", "right", 1080),
])
def test_real_photo_extraction_returns_only_source_space_valid_contours(
        tmp_path: Path, family: str, view: str, size: int):
    source = next((item for item in default_sources()
                   if item["family"] == family and item["view"] == view), None)
    if source is None:
        pytest.skip(f"Local {family} {view} reference unavailable")
    row = evaluate_one(source, size, tmp_path, tmp_path)
    # A tiny photo may legitimately be declined. It must never yield an
    # apparently usable contour outside the derivative's original bounds.
    assert row["status"] in {"passed", "partial", "declined"}, row
    assert not row["issues"], row
    assert row["derived_from_source"] and not row["added_source_detail"]
    assert row["runtime_seconds"] >= 0


def test_contour_validator_rejects_out_of_bounds_and_invalid_geometry():
    result = {"annotations": [
        {"part": "front", "points": [{"x": 0, "y": 0}, {"x": 10, "y": 10},
                                      {"x": 0, "y": 10}, {"x": 10, "y": 0}],
         "anchors": {}, "approved": False},
        {"part": "temple-right", "points": [{"x": 1, "y": 1}, {"x": 8, "y": 1},
                                             {"x": 8, "y": 8}],
         "anchors": {"hinge": {"x": 11, "y": 2}}, "approved": False},
    ]}
    issues = validate_annotations(result, 10, 10)
    assert any("invalid contour polygon" in issue for issue in issues)
    assert any("anchor outside source image" in issue for issue in issues)


def test_front_without_lens_openings_is_partial():
    result = {"annotations": [{"part": "front"}]}
    assert missing_required_parts(result, "front") == ["lens-left", "lens-right"]


@pytest.mark.parametrize("size", [128, 160])
def test_transparent_thumbnail_keeps_both_lens_openings_inside_rim(tmp_path, size):
    source = next((item for item in default_sources()
                   if item["family"] == "transparent" and item["view"] == "front"), None)
    if source is None:
        pytest.skip("Local transparent front reference unavailable")
    row = evaluate_one(source, size, tmp_path, tmp_path)
    assert row["status"] == "passed", row
    assert row["parts"] == ["front", "lens-left", "lens-right"]
    assert not row["issues"]
