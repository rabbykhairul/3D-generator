"""Regression checks for frame-to-arm placement in exported geometry."""

import numpy as np
import pytest
import trimesh
from PIL import Image

from backend.reconstruction import NeedsCorrection, reconstruct


def test_deficient_delaunay_cap_falls_back_without_editing_outline(monkeypatch):
    from backend.reconstruction import engine
    from shapely.ops import triangulate
    from shapely.geometry import Polygon
    poly = Polygon([(0, 0), (120, 0), (130, -10), (140, -10),
                    (122, 5), (0, 5)])
    # Reproduce the GEOS covers-filter failure without storing a merchant photo:
    # one legitimate boundary triangle is absent from the initial cap candidate.
    triangles = [t for t in triangulate(poly) if poly.covers(t)]
    monkeypatch.setattr(engine, 'triangulate', lambda shape: triangles[:-1])
    _, mesh = engine._triangulated_plate(poly, 2, lambda p: [0, p[1]/1000, p[0]/1000],
                                         engine._material('#554433', 1, 0, .5),
                                         'temple_left', extrusion_axis=0)
    assert mesh.is_watertight
    assert len(mesh.vertices) == 2 * (len(poly.exterior.coords) - 1)
    mesh.fix_normals(multibody=True)
    assert mesh.volume == pytest.approx(poly.area * 2 / 1e9)


def _polygon(points):
    return [{"x": x, "y": y} for x, y in points]


def _snapshot(tmp_path):
    front_path = tmp_path / "front.png"
    side_path = tmp_path / "side.png"
    Image.new("RGB", (400, 160), "white").save(front_path)
    Image.new("RGB", (300, 130), "white").save(side_path)
    front = [(20, 40), (40, 20), (360, 20), (380, 40), (380, 60),
             (360, 140), (40, 140), (20, 60)]
    side = [(290, 45), (180, 55), (80, 65), (10, 100),
            (10, 112), (80, 77), (180, 67), (290, 55)]
    annotations = [
        {"photo_id": "front", "part": "front", "points": _polygon(front), "approved": True},
        {"photo_id": "front", "part": "lens-left", "points": _polygon([(55, 45), (175, 45), (175, 115), (55, 115)]), "approved": True},
        {"photo_id": "front", "part": "lens-right", "points": _polygon([(225, 45), (345, 45), (345, 115), (225, 115)]), "approved": True},
    ]
    for side_name in ("left", "right"):
        annotations.append({"photo_id": "side", "part": f"temple-{side_name}", "points": _polygon(side),
                            "approved": True, "source": "reviewed-mirrored-side" if side_name == "left" else "reviewed-side",
                            "anchors": {"hinge": {"x": 290, "y": 50}, "tip": {"x": 10, "y": 106}}})
    return {"category": "opaque-full-rim", "photos": [
        {"id": "front", "path": str(front_path), "width": 400, "height": 160, "view": "front", "role": "fitting"},
        {"id": "side", "path": str(side_path), "width": 300, "height": 130, "view": "right", "role": "fitting"}],
        "annotations": annotations,
        "measurements": {key: {"value": value, "source": "measured"} for key, value in {
            "frame_width": 140, "lens_width": 47, "lens_height": 28,
            "bridge_width": 20, "temple_length": 140}.items()}}


def test_joint_attaches_to_outer_shoulder_and_levels_straight_shaft(tmp_path):
    result = reconstruct(_snapshot(tmp_path), tmp_path / "out")
    geometry = trimesh.load(tmp_path / "out" / "model.glb", force="scene").geometry
    for side, sign in (("left", -1), ("right", 1)):
        hinge = geometry[f"hinge_{side}"].vertices
        arm = geometry[f"temple_{side}"].vertices
        center = np.mean(hinge, axis=0)
        assert center[1] > .007  # The silhouette's shoulder is above the old 7 mm cap.
        assert sign * center[0] > .06
        assert sign * np.median(arm[:, 0]) > .04  # Extrusion stays on the correct side.
        assert arm[:, 2].min() < center[2] - .10
        root = arm[arm[:, 2] > arm[:, 2].max() - .004]
        assert len(root) >= 2
        assert np.linalg.norm(np.median(root[:, :2], axis=0) - center[:2]) < .004
        assert root[:, 2].max() >= hinge[:, 2].min()  # Joint and arm overlap in depth.
        shaft = arm[(arm[:, 2] < arm[:, 2].max() - .045) &
                    (arm[:, 2] > arm[:, 2].max() - .065)]
        assert len(shaft) >= 2
        assert abs(np.median(shaft[:, 1]) - np.median(root[:, 1])) < .006
        tip = arm[arm[:, 2] < arm[:, 2].min() + .007]
        assert np.median(tip[:, 1]) < np.median(root[:, 1]) - .006
        joint = result["fit"]["joints"][side]
        assert joint["surface_part"] == "front_frame"
        assert joint["arm_root_contains_joint"]
        assert joint["connector_depth_mm"][0] > 0
        assert joint["connector_depth_mm"][1] > joint["connector_depth_mm"][0]
        x_mm = (joint["attachment_px"][0] - 200) * 140 / 360
        front_face_z = -4 * (x_mm / 70) ** 2 / 1000
        assert hinge[:, 2].max() < front_face_z - .001


def test_explicit_front_hinge_height_is_not_capped(tmp_path):
    data = _snapshot(tmp_path)
    data["annotations"][0]["anchors"] = {
        "hinge-left": {"x": 24, "y": 42}, "hinge-right": {"x": 376, "y": 42}}
    reconstruct(data, tmp_path / "anchored")
    geometry = trimesh.load(tmp_path / "anchored" / "model.glb", force="scene").geometry
    assert all(np.mean(geometry[f"hinge_{side}"].vertices[:, 1]) > .01
               for side in ("left", "right"))


@pytest.mark.parametrize("category", ["thin-metal", "semi-rimless", "rimless"])
def test_joint_meets_modeled_front_structure_across_categories(tmp_path, category):
    data = _snapshot(tmp_path)
    data["category"] = category
    if category == "rimless":
        data["annotations"] = [a for a in data["annotations"] if a["part"] != "front"]
    result = reconstruct(data, tmp_path / category)
    assert all(result["fit"]["joints"][side]["surface_part"]
               for side in ("left", "right"))


@pytest.mark.parametrize("category", ["opaque-full-rim", "thin-metal", "rimless"])
def test_arm_root_follows_local_rear_surface(tmp_path, category):
    data = _snapshot(tmp_path)
    data["category"] = category
    if category == "rimless":
        data["annotations"] = [a for a in data["annotations"] if a["part"] != "front"]
    result = reconstruct(data, tmp_path / f"root-{category}")
    geometry = trimesh.load(tmp_path / f"root-{category}" / "model.glb", force="scene").geometry
    for side in ("left", "right"):
        joint = result["fit"]["joints"][side]
        root_z = geometry[f"temple_{side}"].vertices[:, 2].max() * 1000
        assert joint["arm_root_overlap_mm"] == pytest.approx(.5)
        assert joint["arm_root_z_mm"] == pytest.approx(joint["rear_surface_z_mm"] + .5)
        assert root_z == pytest.approx(joint["arm_root_z_mm"], abs=.05)


def test_front_hinge_anchor_in_lens_opening_requires_correction(tmp_path):
    data = _snapshot(tmp_path)
    data["annotations"][0]["anchors"] = {"hinge-left": {"x": 100, "y": 70}}
    with pytest.raises(NeedsCorrection, match="outside the front frame material"):
        reconstruct(data, tmp_path / "hole-anchor")


def test_side_hinge_anchor_outside_root_requires_correction(tmp_path):
    data = _snapshot(tmp_path)
    data["annotations"][3]["anchors"]["hinge"]["y"] = 15
    with pytest.raises(NeedsCorrection, match="does not meet its outline"):
        reconstruct(data, tmp_path / "loose-root")


def test_explicit_front_depth_is_honored_and_missing_depth_is_labeled(tmp_path):
    data = _snapshot(tmp_path)
    inferred = reconstruct(data, tmp_path / "draft")
    assert inferred["dimensions_mm"]["frame_thickness"] == 4
    assert any("unverified 4 mm" in warning for warning in inferred["warnings"])

    data["measurements"]["frame_thickness"] = {"value": 2, "source": "measured"}
    measured = reconstruct(data, tmp_path / "measured")
    assert measured["dimensions_mm"]["frame_thickness"] == 2
    assert not any("Front material depth" in warning for warning in measured["warnings"])
    draft_frame = trimesh.load(tmp_path / "draft" / "model.glb", force="scene").geometry["front_frame"]
    measured_frame = trimesh.load(tmp_path / "measured" / "model.glb", force="scene").geometry["front_frame"]
    assert measured_frame.vertices[:, 2].min() - draft_frame.vertices[:, 2].min() == pytest.approx(.002)


@pytest.mark.parametrize('category',['semi-rimless','transparent-full-rim'])
def test_thin_front_does_not_use_a_four_mm_draft_slab(tmp_path, category):
    data=_snapshot(tmp_path)
    data['category']=category
    if category=='transparent-full-rim':
        data['annotations'][0]['points']=_polygon([(20,20),(380,20),(380,140),(20,140)])
        data['annotations'][1]['points']=_polygon([(22,22),(178,22),(178,138),(22,138)])
        data['annotations'][2]['points']=_polygon([(222,22),(378,22),(378,138),(222,138)])
    result=reconstruct(data,tmp_path/f'{category}-thin')
    assert .8 <= result['dimensions_mm']['frame_thickness'] <= 1.2
    assert result['provenance']['front_depth']['source']=='default'
    data['measurements']['frame_thickness']={'value':3,'source':'measured'}
    measured=reconstruct(data,tmp_path/f'{category}-measured')
    assert measured['dimensions_mm']['frame_thickness']==3
