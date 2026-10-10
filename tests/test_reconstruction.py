from pathlib import Path

import numpy as np
import pytest
import trimesh
from PIL import Image
from PIL import ImageDraw
from shapely.geometry import Polygon, Point

from backend.reconstruction import NeedsCorrection, extract_annotations, reconstruct
from backend.reconstruction.render_mesh import welded_geometry


def rectangle(x0, y0, x1, y1):
    return [{"x": x0, "y": y0}, {"x": x1, "y": y0},
            {"x": x1, "y": y1}, {"x": x0, "y": y1}]


@pytest.mark.parametrize('category',['opaque-full-rim','transparent-full-rim','thin-metal','semi-rimless','rimless','sports-shield'])
def test_separate_approved_pad_silhouettes_work_for_every_category(tmp_path,category):
    data=snapshot(tmp_path,category)
    for side,x in [('left',188),('right',208)]:
        data['annotations'].append({'photo_id':'front','part':f'nose-pad-{side}','approved':True,
            'source':'manual','points':[{'x':x,'y':82},{'x':x+7,'y':84},{'x':x+5,'y':108},{'x':x-3,'y':102}]})
    result=reconstruct(data,tmp_path/'pads')
    assert {'nose_pad_left','nose_pad_right'}<=set(result['parts'])
    scene=trimesh.load(tmp_path/'pads/model.glb',force='scene',process=False)
    for side in ('left','right'):
        mesh=scene.geometry[f'nose_pad_{side}']
        assert mesh.visual.material.baseColorTexture is not None
        assert welded_geometry(mesh).is_watertight

def test_occluded_lenses_suggest_completion_and_preserve_bridge_hole(tmp_path):
    image=Image.new('RGB',(800,320),'white');draw=ImageDraw.Draw(image)
    draw.rectangle((40,40,760,280),fill='#55583b')
    draw.rectangle((90,75,350,250),fill='white')
    draw.rectangle((450,75,710,250),fill='white')
    draw.rectangle((375,60,425,105),fill='white')
    draw.line((100,75,210,195),fill='#333333',width=20)
    draw.line((700,75,590,195),fill='#333333',width=20)
    path=tmp_path/'occluded.png';image.save(path)
    result=extract_annotations({'id':'front','path':str(path),'view':'front','width':800,'height':320})
    by_part={a['part']:a for a in result['annotations']}
    assert 'bridge-opening' in by_part
    for part,tip in [('lens-left',(210,195)),('lens-right',(590,195))]:
        ann=by_part[part]
        assert ann['source']=='convex-occlusion-completion'
        assert not ann['approved']
        assert Polygon([(p['x'],p['y']) for p in ann['points']]).contains(Point(*tip))

def test_bridge_opening_is_actual_mesh_hole(tmp_path):
    data=snapshot(tmp_path,'opaque-full-rim')
    data['annotations'].append({'photo_id':'front','part':'bridge-opening','approved':True,'points':rectangle(185,30,215,60)})
    result=reconstruct(data,tmp_path/'hole')
    assert result['bridge_openings']==1
    mesh=trimesh.load(tmp_path/'hole/model.glb',force='scene').geometry['front_frame']
    # Bridge x=200 -> x=0; lens-center y=77.5, scale y=.4 mm/px.
    opening=Point(0,(77.5-45)*.4/1000)
    assert not any(Polygon(tri[:,:2]).contains(opening) for tri in mesh.triangles)
    assert welded_geometry(mesh).is_watertight


def snapshot(tmp_path: Path, category: str):
    photo_path = tmp_path / "front.png"
    Image.new("RGB", (400, 160), (220, 225, 230)).save(photo_path)
    annotations = [
        {"photo_id": "front", "part": "front", "points": rectangle(20, 20, 380, 140), "approved": True, "source": "manual"},
        {"photo_id": "front", "part": "lens-left", "points": rectangle(55, 40, 175, 115), "approved": True, "source": "manual"},
        {"photo_id": "front", "part": "lens-right", "points": rectangle(225, 40, 345, 115), "approved": True, "source": "manual"},
    ]
    measurements = {k: {"value": v, "source": "measured"} for k, v in
                    {"frame_width": 140, "lens_width": 45, "lens_height": 30,
                     "bridge_width": 20, "temple_length": 135}.items()}
    if category == "rimless":
        measurements["lens_width"]["value"] = 58
        annotations = annotations[1:]
    if category == "sports-shield":
        annotations = annotations[:1]
        annotations[0]["part"] = "shield"
    return {"category": category, "photos": [{"id": "front", "path": str(photo_path), "width": 400,
                                                "height": 160, "view": "front", "role": "overview"}],
            "measurements": measurements, "annotations": annotations, "material": {"color": "#232630"}}


@pytest.mark.parametrize("category", ["opaque-full-rim", "thin-metal", "transparent-full-rim",
                                      "semi-rimless", "rimless", "sports-shield"])
def test_six_categories_export_finite_measured_glb(tmp_path, category):
    data = snapshot(tmp_path, category)
    result = reconstruct(data, tmp_path / "out")
    model = trimesh.load(tmp_path / "out" / "model.glb", force="scene")
    assert result["checks"]["frame_width_within_1_percent"]
    assert result["dimensions_mm"]["constructed_frame_width"] == pytest.approx(140)
    assert (tmp_path / "out" / "metadata.json").exists()
    assert len(model.geometry) == len(result["parts"])
    assert all(np.isfinite(mesh.vertices).all() and len(mesh.faces) for mesh in model.geometry.values())
    assert all(welded_geometry(mesh).is_watertight for mesh in model.geometry.values())
    assert result['checks']['consistent_outward_normals']
    assert all(mesh.is_winding_consistent and mesh.volume > 0 for mesh in model.geometry.values())
    front_parts = [mesh for name, mesh in model.geometry.items() if not name.startswith("temple_")]
    actual_width = (max(mesh.vertices[:, 0].max() for mesh in front_parts) -
                    min(mesh.vertices[:, 0].min() for mesh in front_parts)) * 1000
    assert abs(actual_width - 140) <= 1.4
    assert result["dimensions_mm"]["constructed_frame_width"] == pytest.approx(actual_width, abs=.01)
    assert "temple_left" not in result["parts"] and "temple_right" not in result["parts"]
    assert result['reconstruction_status']=='incomplete'


def test_contour_edit_changes_geometry(tmp_path):
    data = snapshot(tmp_path, "opaque-full-rim")
    reconstruct(data, tmp_path / "first")
    first = trimesh.load(tmp_path / "first" / "model.glb", force="scene").geometry["front_frame"]
    data["annotations"][1]["points"][0]["y"] += 9
    reconstruct(data, tmp_path / "second")
    second = trimesh.load(tmp_path / "second" / "model.glb", force="scene").geometry["front_frame"]
    assert not np.array_equal(first.vertices, second.vertices)

def test_meter_sized_values_produce_actionable_unit_error(tmp_path):
    data=snapshot(tmp_path,'opaque-full-rim')
    for key,value in {'frame_width':.2,'lens_width':.08,'lens_height':.06,'bridge_width':1}.items():
        data['measurements'][key]['value']=value
    with pytest.raises(NeedsCorrection,match='millimeters, not meters'):
        reconstruct(data,tmp_path/'bad-units')

def test_wire_hinge_centers_calibrate_to_actual_outer_width(tmp_path):
    data=snapshot(tmp_path,'thin-metal')
    data['annotations'][0]['anchors']={'hinge-left':{'x':30,'y':70},'hinge-right':{'x':370,'y':70},'bridge':{'x':200,'y':65}}
    result=reconstruct(data,tmp_path/'calibrated-wire')
    model=trimesh.load(tmp_path/'calibrated-wire/model.glb',force='scene')
    front=[mesh for name,mesh in model.geometry.items() if not name.startswith('temple_')]
    span=(max(m.vertices[:,0].max() for m in front)-min(m.vertices[:,0].min() for m in front))*1000
    assert span==pytest.approx(140,abs=.05)
    assert 'assembled_width' in result['fit']


def test_approved_side_photo_shapes_temple(tmp_path):
    data = snapshot(tmp_path, "opaque-full-rim")
    side_path = tmp_path / "side.png"
    Image.new("RGB", (300, 100), "white").save(side_path)
    data["photos"].append({"id": "side", "path": str(side_path), "width": 300, "height": 100,
                           "view": "left-side", "role": "detail"})
    data["annotations"].append({"photo_id": "side", "part": "temple-left", "approved": True,
                                "anchors": {"hinge": {"x": 10, "y": 30}, "tip": {"x": 290, "y": 70}},
                                "points": [{"x": 10, "y": 30}, {"x": 270, "y": 35},
                                           {"x": 290, "y": 70}, {"x": 10, "y": 46}]})
    result = reconstruct(data, tmp_path / "side-output")
    assert result["provenance"]["arms"]["left"] == "approved side outline"
    assert result["provenance"]["arms"]["right"] == "omitted — unverified"
    assert result["fit"]["side"]["left"]["calibration"] == "approved hinge/tip anchors"
    scene = trimesh.load(tmp_path / "side-output" / "model.glb", force="scene")
    assert np.ptp(scene.geometry["temple_left"].vertices[:, 1]) > .005


def test_top_view_anchors_fit_curve_and_spread(tmp_path):
    data = snapshot(tmp_path, "opaque-full-rim")
    top_path = tmp_path / "top.png"
    Image.new("RGB", (400, 300), "white").save(top_path)
    data["photos"].append({"id": "top", "path": str(top_path), "width": 400, "height": 300,
                           "view": "top", "role": "overview"})
    data["annotations"].append({"photo_id": "top", "part": "top", "approved": True,
                                "points": [{"x": x, "y": y} for x, y in
                                           [(40, 50), (360, 50), (325, 270), (305, 270),
                                            (330, 75), (70, 75), (95, 270), (75, 270)]],
                                "anchors": {k: {"x": x, "y": y} for k, (x, y) in
                                            {"bridge": (200, 40), "hinge-left": (40, 60),
                                             "hinge-right": (360, 60), "tip-left": (75, 270),
                                             "tip-right": (325, 270)}.items()}})
    fitted = reconstruct(data, tmp_path / "with-top")
    assert fitted["dimensions_mm"]["front_curve"] == pytest.approx(8.75)
    assert fitted["dimensions_mm"]["temple_spread"] == pytest.approx(15.3125)
    assert fitted["fit"]["source_photo_id"] == "top"
    assert fitted["fit"]["residuals_mm"]["front_curve_to_top"] == pytest.approx(0)
    baseline = snapshot(tmp_path, "opaque-full-rim")
    reconstruct(baseline, tmp_path / "without-top")
    a = trimesh.load(tmp_path / "with-top" / "model.glb", force="scene")
    b = trimesh.load(tmp_path / "without-top" / "model.glb", force="scene")
    assert np.min(a.geometry["front_frame"].vertices[:, 2]) < np.min(b.geometry["front_frame"].vertices[:, 2])
    assert 'temple_left' not in a.geometry and 'temple_left' not in b.geometry


def test_metal_wire_and_dual_sports_paths_are_distinct(tmp_path):
    metal = snapshot(tmp_path, "thin-metal")
    metal_result = reconstruct(metal, tmp_path / "metal")
    assert "metal_rim_left" in metal_result["parts"]
    assert "front_frame" not in metal_result["parts"]
    metal_scene = trimesh.load(tmp_path / "metal" / "model.glb", force="scene")
    assert welded_geometry(metal_scene.geometry["metal_rim_left"]).is_watertight
    sports = snapshot(tmp_path, "sports-shield")
    sports["annotations"] = snapshot(tmp_path, "opaque-full-rim")["annotations"]
    result = reconstruct(sports, tmp_path / "sports-dual")
    assert "front_frame" in result["parts"]
    assert "lens_left" in result["parts"] and "lens_right" in result["parts"]
    assert "shield_lens" not in result["parts"]


def test_opaque_bevel_and_anchored_metal_nose_pads(tmp_path):
    opaque = snapshot(tmp_path, "opaque-full-rim")
    result = reconstruct(opaque, tmp_path / "beveled")
    assert 0 < result["dimensions_mm"]["front_bevel"] <= .35
    frame = trimesh.load(tmp_path / "beveled" / "model.glb", force="scene").geometry["front_frame"]
    assert welded_geometry(frame).is_watertight
    normals = frame.face_normals
    assert np.any((np.abs(normals[:, 2]) > .2) & (np.abs(normals[:, 2]) < .95))
    metal = snapshot(tmp_path, "thin-metal")
    metal["annotations"][0]["anchors"] = {
        "nose-pad-left": {"x": 188, "y": 90}, "nose-pad-right": {"x": 212, "y": 90}}
    metal_result = reconstruct(metal, tmp_path / "with-pads")
    assert "nose_pad_left" in metal_result["parts"] and "nose_pad_right" in metal_result["parts"]
    assert metal_result["provenance"]["nose_pads"]["left"] == "approved front anchor"


def test_angle_only_front_photo_requires_correction(tmp_path):
    data = snapshot(tmp_path, "opaque-full-rim")
    data["photos"][0]["view"] = "front-left"
    with pytest.raises(NeedsCorrection, match="straight front photo"):
        reconstruct(data, tmp_path / "bad-angle")


def test_front_anchors_set_bridge_origin_and_hinges(tmp_path):
    data = snapshot(tmp_path, "opaque-full-rim")
    data['photos'].append(dict(data['photos'][0],id='side',view='left'))
    data['annotations'].append({'photo_id':'side','part':'temple-left','approved':True,'points':rectangle(20,30,350,40)})
    data["annotations"][0]["anchors"] = {
        "bridge": {"x": 195, "y": 70},
        "hinge-left": {"x": 25, "y": 65},
        "hinge-right": {"x": 375, "y": 65},
    }
    result = reconstruct(data, tmp_path / "anchored")
    assert result["provenance"]["front"]["bridge_anchor"]
    assert all(result["provenance"]["front"]["hinge_anchors"].values())
    scene = trimesh.load(tmp_path / "anchored" / "model.glb", force="scene")
    hinge_center = np.mean(scene.geometry["hinge_left"].vertices, axis=0)
    assert hinge_center[1] > 0


def test_validation_photo_cannot_supply_geometry(tmp_path):
    data = snapshot(tmp_path, "opaque-full-rim")
    data["photos"][0]["role"] = "validation"
    with pytest.raises(NeedsCorrection, match="front photo outline"):
        reconstruct(data, tmp_path / "excluded")


@pytest.mark.parametrize("view, expected", [("left", "temple-left"), ("right", "temple-right"), ('inner-left','temple-left'),('inner-right','temple-right')])
def test_side_extraction_classifies_short_view_names(tmp_path, view, expected):
    path = tmp_path / f"{view}.png"
    image = Image.new("RGB", (400, 160), "white")
    from PIL import ImageDraw
    ImageDraw.Draw(image).rectangle((20, 50, 380, 100), fill="black")
    image.save(path)
    result = extract_annotations({"id": view, "path": str(path), "view": view})
    assert result["annotations"][0]["part"] == expected


def test_large_photo_extraction_maps_back_to_original_pixels(tmp_path):
    path = tmp_path / "large.png"
    image = Image.new("RGB", (3200, 500), "white")
    from PIL import ImageDraw
    ImageDraw.Draw(image).rectangle((320, 100, 2880, 400), fill="black")
    image.save(path)
    result = extract_annotations({"id": "large", "path": str(path), "view": "left"})
    outline = result["annotations"][0]["points"]
    assert min(p["x"] for p in outline) == pytest.approx(320, abs=3)
    assert max(p["x"] for p in outline) == pytest.approx(2880, abs=3)

def test_inner_right_arm_is_geometry_input_and_supersedes_outer(tmp_path):
    data=snapshot(tmp_path,'opaque-full-rim')
    path=tmp_path/'arm.png';Image.new('RGB',(300,100),'gray').save(path)
    for view in ['right','inner-right']:
        data['photos'].append({'id':view,'view':view,'role':'fitting','width':300,'height':100,'path':str(path)})
        data['annotations'].append({'photo_id':view,'part':'temple-right','approved':True,'points':rectangle(10,30,280,40),'anchors':{'hinge':{'x':10,'y':35},'tip':{'x':280,'y':35}}})
    result=reconstruct(data,tmp_path/'inner-arm')
    assert 'temple_right' in result['parts'] and 'temple_left' not in result['parts']
    assert result['fit']['side']['right']['source_photo_id']=='inner-right'

def test_right_facing_frame_with_overlapping_arm_roots(tmp_path):
    image=Image.new('RGB',(400,200),'white');draw=ImageDraw.Draw(image)
    draw.rectangle((370,25,389,180),fill='black')
    draw.line([(380,65),(145,85),(20,160)],fill='black',width=7)
    draw.line([(250,76),(150,90),(80,140)],fill='black',width=5)
    path=tmp_path/'right-facing.png';image.save(path)
    result=extract_annotations({'id':'right','view':'right','path':str(path),'width':400,'height':200})
    ann=result['annotations'][0]
    assert ann['part']=='temple-right'
    assert ann['anchors']['hinge']['x']>ann['anchors']['tip']['x']
    assert ann['anchors']['tip']['x']<50
    assert max(p['x'] for p in ann['points'])<370

def test_side_photo_separates_two_visible_arms_and_exports_reviewed_mirror(tmp_path):
    image=Image.new('RGB',(100,100),'white');draw=ImageDraw.Draw(image)
    draw.rectangle((8,25,12,70),fill='black')
    draw.line([(10,30),(70,30),(88,43)],fill='black',width=4)
    draw.line([(10,50),(72,50),(94,65)],fill='black',width=5)
    path=tmp_path/'two-arms.png';image.save(path)
    photo={'id':'side','path':str(path),'view':'left','role':'fitting','width':100,'height':100}
    result=extract_annotations(photo);ann=result['annotations'][0]
    assert ann['approved'] is False and ann['anchors']['hinge']['x'] < ann['anchors']['tip']['x']
    assert min(p['y'] for p in ann['points'])>40 # Other arm and frame excluded.
    data=snapshot(tmp_path,'opaque-full-rim');data['photos'].append(photo)
    data['annotations'].append(dict(ann,approved=True,source='merchant-reviewed-side:'+ann['source']))
    data['annotations'].append(dict(ann,part='temple-right',approved=True,source='explicit-mirrored-side-assumption:'+ann['source']))
    exported=reconstruct(data,tmp_path/'arms')
    assert 'temple_left' in exported['parts'] and 'temple_right' in exported['parts']
    assert exported['provenance']['arms']['right']=='mirrored — explicit symmetry assumption'
    assert any('Explicit symmetry assumption' in w for w in exported['warnings'])


def test_missing_front_and_bad_polygon_request_correction(tmp_path):
    data = snapshot(tmp_path, "opaque-full-rim")
    data["annotations"][0]["approved"] = False
    with pytest.raises(NeedsCorrection, match="front photo outline"):
        reconstruct(data, tmp_path / "missing")
    data["annotations"][0]["approved"] = True
    data["annotations"][0]["points"] = [
        {"x": 20, "y": 20}, {"x": 380, "y": 140},
        {"x": 380, "y": 20}, {"x": 20, "y": 140},
    ]
    with pytest.raises(NeedsCorrection, match="self-intersecting"):
        reconstruct(data, tmp_path / "invalid")


def test_missing_scale_and_conflicting_measurements(tmp_path):
    data = snapshot(tmp_path, "opaque-full-rim")
    del data["measurements"]["frame_width"]
    with pytest.raises(NeedsCorrection, match="Measure frame_width"):
        reconstruct(data, tmp_path / "missing")
    data["measurements"]["frame_width"] = {"value": 100, "source": "manual"}
    data["measurements"]["lens_width"]["value"] = 60
    with pytest.raises(NeedsCorrection, match="exceed frame width"):
        reconstruct(data, tmp_path / "conflict")


def test_auto_extraction_is_unapproved(tmp_path):
    path = tmp_path / "isolated.png"
    image = Image.new("RGB", (400, 160), "white")
    from PIL import ImageDraw
    draw = ImageDraw.Draw(image)
    draw.rectangle((20, 20, 380, 140), fill="black")
    draw.rectangle((55, 40, 175, 115), fill="white")
    draw.rectangle((225, 40, 345, 115), fill="white")
    image.save(path)
    result = extract_annotations({"id": "p", "path": str(path), "view": "front"})
    assert {a["part"] for a in result["annotations"]} == {"front", "lens-left", "lens-right"}
    assert all(not a["approved"] for a in result["annotations"])


def test_pale_transparent_front_recovers_two_reviewable_apertures(tmp_path):
    image = Image.new('RGB', (800, 400), 'white')
    draw = ImageDraw.Draw(image)
    draw.ellipse((100, 305, 700, 328), fill='#dddddd')
    draw.rounded_rectangle((40, 80, 760, 315), radius=65, fill='#f1edf5')
    draw.rounded_rectangle((90, 105, 360, 290), radius=45, fill='white')
    draw.rounded_rectangle((440, 105, 710, 290), radius=45, fill='white')
    draw.line((100, 115, 260, 250), fill='#514669', width=18)
    draw.line((700, 115, 540, 250), fill='#514669', width=18)
    path = tmp_path / 'pale.png'
    image.save(path)
    result = extract_annotations({'id': 'pale', 'path': str(path), 'view': 'front'})
    by_part = {a['part']: a for a in result['annotations']}
    assert result['segmentation']['method'] == 'stable-faint-front'
    assert set(by_part) == {'front', 'lens-left', 'lens-right'}
    assert all(not a['approved'] for a in by_part.values())
    outer = Polygon([(p['x'], p['y']) for p in by_part['front']['points']])
    holes = [Polygon([(p['x'], p['y']) for p in by_part[f'lens-{side}']['points']])
             for side in ('left', 'right')]
    assert outer.is_valid and all(hole.is_valid and outer.contains(hole) for hole in holes)
    assert not outer.covers(Point(200, 321))  # neutral contact shadow below the rim
    assert holes[0].contains(Point(260, 250)) and holes[1].contains(Point(540, 250))
    assert Polygon(outer.exterior.coords, [hole.exterior.coords for hole in holes]).is_valid


def test_tinted_side_reference_suggests_reviewable_tip_branch():
    path = Path(__file__).resolve().parents[1] / 'frontend/public/upload-guide/side.png'
    result = extract_annotations({'id': 'side', 'path': str(path), 'view': 'right'})
    assert result['segmentation']['method'] == 'outer-branch-contrast-consensus'
    assert len(result['annotations']) == 1
    ann = result['annotations'][0]
    assert ann['part'] == 'temple-right' and ann['approved'] is False
    assert ann['source'] == 'single-visible-arm-branch-consensus-v1'
    assert ann['anchors']['hinge']['x'] > 850
    assert 30 < ann['anchors']['tip']['x'] < 55  # near arm, not the old far-arm tip at x=141
    assert any('continuous outer arm' in warning for warning in result['warnings'])
    assert Polygon([(p['x'], p['y']) for p in ann['points']]).is_valid
