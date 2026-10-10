from pathlib import Path

import cv2
import numpy as np
import pytest

from backend.reconstruction.multiview import analyze_multiview


def _photo(path, photo_id, view):
    return {"id": photo_id, "view": view, "path": str(path)}


def _front_annotation(width, height):
    return [{"part": "front", "approved": True, "points": [
        {"x": 70, "y": 100}, {"x": width-70, "y": 100},
        {"x": width-70, "y": height-100}, {"x": 70, "y": height-100}]}]


def test_textured_angled_image_reports_consistent_registration(tmp_path):
    height, width = 500, 800
    front = np.full((height, width, 3), 248, np.uint8)
    cv2.ellipse(front, (240, 250), (165, 110), 0, 0, 360, (28, 42, 63), 26)
    cv2.ellipse(front, (560, 250), (165, 110), 0, 0, 360, (28, 42, 63), 26)
    cv2.line(front, (400, 190), (400, 230), (28, 42, 63), 22)
    rng = np.random.default_rng(73)
    for x, y in rng.integers([90, 140], [710, 360], size=(500, 2)):
        if np.min(front[y, x]) < 100:
            cv2.circle(front, (int(x), int(y)), 2, (int(x % 220), int(y % 190), 50), -1)
    front_path = tmp_path / "front.png"
    source_path = tmp_path / "angle.png"
    cv2.imwrite(str(front_path), front)
    transform = cv2.getRotationMatrix2D((width/2, height/2), 6, .94)
    transform[:, 2] += [15, -8]
    source = cv2.warpAffine(front, transform, (width, height), borderValue=(248, 248, 248))
    cv2.imwrite(str(source_path), source)
    result = analyze_multiview([_photo(source_path, "angle", "angled")],
                               _photo(front_path, "front", "front"),
                               _front_annotation(width, height))["views"][0]
    assert result["status"] == "accepted", result
    assert result["usable_for_inner_uv"] is False
    recovered = np.asarray(result["source_to_front_homography"])
    point = np.float32([[[280., 230.]]])
    moved = cv2.transform(point, transform)
    returned = cv2.perspectiveTransform(moved, recovered)
    assert np.linalg.norm(returned-point) < 4
    mirror_path = tmp_path / "rear_mirror.png"
    cv2.imwrite(str(mirror_path), cv2.flip(source, 1))
    mirrored = analyze_multiview([_photo(mirror_path, "rear", "rear")],
                                 _photo(front_path, "front", "front"),
                                 _front_annotation(width, height))["views"][0]
    assert mirrored["status"] == "accepted", mirrored
    assert mirrored["mirrored"] is True
    assert mirrored["usable_for_inner_uv"] is True
    mirror_point = np.float32([[[width-1-moved[0, 0, 0], moved[0, 0, 1]]]])
    restored = cv2.perspectiveTransform(mirror_point,
                                       np.asarray(mirrored["source_to_front_homography"]))
    assert np.linalg.norm(restored-point) < 4


def test_unrelated_and_heldout_images_cannot_supply_warp(tmp_path):
    front = np.full((400, 600, 3), 250, np.uint8)
    source = np.full_like(front, 250)
    cv2.circle(front, (250, 200), 120, (30, 40, 50), 12)
    cv2.rectangle(source, (30, 30), (120, 100), (20, 30, 40), -1)
    f, p = tmp_path / "front.png", tmp_path / "other.png"
    cv2.imwrite(str(f), front)
    cv2.imwrite(str(p), source)
    other = _photo(p, "other", "inner")
    heldout = dict(other, id="heldout", validation=True)
    views = analyze_multiview([other, heldout], _photo(f, "front", "front"),
                              _front_annotation(600, 400))["views"]
    assert len(views) == 1
    assert views[0]["status"] == "review"
    assert views[0]["usable_for_inner_uv"] is False
    assert "source_to_front_homography" not in views[0]


def test_large_images_keep_original_pixel_coordinates(tmp_path):
    rng = np.random.default_rng(19)
    small = np.full((500, 600, 3), 245, np.uint8)
    for x, y in rng.integers([60, 120], [540, 380], size=(600, 2)):
        cv2.circle(small, (int(x), int(y)), 2, (int(x % 180), int(y % 170), 40), -1)
    large = cv2.resize(small, (3000, 2500), interpolation=cv2.INTER_NEAREST)
    f, p = tmp_path / "large_front.png", tmp_path / "large_rear.png"
    cv2.imwrite(str(f), large)
    cv2.imwrite(str(p), large)
    annotation = [{"part": "front", "approved": True, "points": [
        {"x": 300, "y": 600}, {"x": 2700, "y": 600},
        {"x": 2700, "y": 1900}, {"x": 300, "y": 1900}]}]
    row = analyze_multiview([_photo(p, "large", "angled")],
                            _photo(f, "front", "front"), annotation)["views"][0]
    assert row["status"] == "accepted", row
    matrix = np.asarray(row["source_to_front_homography"])
    mapped = cv2.perspectiveTransform(np.float32([[[2100, 1400]]]), matrix)
    assert np.linalg.norm(mapped[0, 0]-[2100, 1400]) < 3
    assert max(x for x, _ in row["front_coverage_polygon"]) > 2048


def test_real_cropped_rear_only_covers_one_lens_when_available():
    from backend.reconstruction.engine import extract_annotations
    from backend.reconstruction.inner_texture import build_registered_inner_texture, _frame_mask

    folder = (Path(__file__).resolve().parents[1] / "data/projects" /
              "c8e4014e30044160ace24fe171a57b5e/photos")
    front_path = folder / "be4512810ce847d5a61946b1c69725a4.png"
    rear_path = folder / "3a146a450f674cbf859da7ae31fc02e4.png"
    if not front_path.exists() or not rear_path.exists():
        pytest.skip("Local pp catalogue reference is not installed")
    front = _photo(front_path, "front", "front")
    rear = _photo(rear_path, "pp", "inner")
    annotations = [dict(a, approved=True) for a in extract_annotations(front)["annotations"]]
    record = analyze_multiview([rear], front, annotations)["views"][0]
    assert record["status"] == "accepted", record
    assert record["lens_part"] == "lens-right"
    assert record["mirrored"] is True
    assert np.asarray(record["source_to_front_homography"])[0, 0] < 0
    assert record["usable_for_inner_uv"]
    assert record["front_width_coverage"] < .4
    assert min(x for x, _ in record["front_coverage_polygon"]) > 580
    angle_path = folder / "de7ca5a1277b4010993ad859ba650477.png"
    if angle_path.exists():
        angle = analyze_multiview([_photo(angle_path, "angle", "front-left")],
                                  front, annotations)["views"][0]
        assert angle["status"] == "accepted"
        assert angle["reason"] == "appearance_consistent"
        assert angle["usage"] == "Consistency check only"
        assert angle["lens_openings_observed"] == 2
        assert "source_to_front_homography" not in angle
    texture, diagnostics = build_registered_inner_texture(front, annotations, [rear],
                                                         {"views": [record]})
    assert texture is not None
    assert record["status"] == "used"
    assert record["usage"] == "Partial registered inner-frame texture"
    assert .1 < diagnostics["observed_fraction"] < .5
    assert diagnostics["reason"] == "registered_inner_appearance"
    assert diagnostics["applied"][0]["photo_id"] == "pp"
    assert diagnostics["applied"][0]["frame_coverage_fraction"] == diagnostics["observed_fraction"]
    pixels = np.asarray(texture)
    material = _frame_mask(annotations, pixels.shape[1], pixels.shape[0]) > 0
    left = pixels[material & (np.indices(material.shape)[1] < 500)]
    right = pixels[material & (np.indices(material.shape)[1] > 600)]
    assert float(np.std(right)) > float(np.std(left)) + 1
    assert np.all(left == left[0])  # no copied front logo on unseen inside rim
    absent, no_use = build_registered_inner_texture(front, annotations, [rear], {"views": []})
    assert absent is None and no_use["observed_fraction"] == 0
