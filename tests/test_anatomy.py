from copy import deepcopy

import cv2
import numpy as np
import pytest
from shapely.geometry import Polygon

from backend.contracts import Annotation
from backend.reconstruction.anatomy import audit_photo, audit_project
from backend.reconstruction.engine import extract_annotations
from backend.reconstruction.product_local import recover_product_local_front


def outline(part, coords, **overrides):
    return dict(photo_id='front', part=part, points=[dict(x=x, y=y) for x, y in coords],
                anchors=overrides.get('anchors', {}), approved=overrides.get('approved', False),
                source=overrides.get('source', 'manual'))


def front_shapes():
    return [outline('front', [(10, 10), (290, 10), (290, 140), (10, 140)]),
            outline('lens-left', [(20, 25), (125, 25), (125, 120), (20, 120)]),
            outline('lens-right', [(175, 25), (280, 25), (280, 120), (175, 120)])]


PHOTO = dict(id='front', view='front', width=300, height=150, role='fitting')


def test_structure_does_not_equate_unknown_pad_with_absence():
    report = audit_photo(PHOTO, front_shapes(), 'transparent-full-rim')
    assert report['status'] == 'supported-structure' and not report['blocking']
    nodes = {n['id']: n for n in report['nodes']}
    assert nodes['nose-pad-left']['state'] == 'uncertain'
    assert nodes['bridge']['state'] == 'inferred'
    assert nodes['front']['evidence'][0]['approved'] is False
    assert nodes['temple-left']['state'] == 'not-observed-in-view'
    assert all('probability' not in n for n in report['nodes'])


@pytest.mark.parametrize('coords', [
    [(0, 0), (70, 60), (0, 60), (70, 0)],
    [(-1, 20), (80, 20), (80, 50)],
    [(0, 0), (float('nan'), 5), (80, 50)],
    [(5, 5), (5, 5), (5, 5)],
])
def test_invalid_polygons_are_not_observed_structure(coords):
    report = audit_photo(PHOTO, [outline('front', coords)])
    assert report['blocking']
    assert next(n for n in report['nodes'] if n['id'] == 'front')['state'] == 'uncertain'


def test_overlapping_lenses_and_disconnected_bridge_are_blocking():
    annotations = front_shapes()
    annotations[2] = outline('lens-right', [(90, 30), (290, 30), (290, 135), (90, 135)])
    annotations += [outline('bridge-opening', [(0, 0), (15, 0), (15, 35), (0, 35)])]
    report = audit_photo(PHOTO, annotations)
    codes = {i['code'] for i in report['issues']}
    assert {'overlapping-optical-boundaries', 'invalid-bridge-opening'} <= codes
    assert report['blocking']


def test_two_optical_blobs_without_support_are_not_complete_eyewear():
    report = audit_photo(PHOTO, front_shapes()[1:])
    assert report['status'] == 'partial-structure'
    assert any(i['code'] == 'support-unverified' for i in report['issues'])
    assert not report['blocking']  # Partial evidence can still be saved/edited.


def test_shield_and_rimless_topology_have_no_forced_full_rim():
    shield = front_shapes()[0]
    shield['part'] = 'shield'
    report = audit_photo(PHOTO, [shield], 'sports-shield')
    assert report['status'] == 'supported-structure'
    assert next(n for n in report['nodes'] if n['id'] == 'lens-left')['state'] == 'not-applicable'
    hardware = outline('bridge-hardware', [(125, 50), (175, 50), (175, 60), (125, 60)])
    report = audit_photo(PHOTO, front_shapes()[1:] + [hardware], 'rimless')
    assert report['status'] == 'supported-structure'
    assert next(n for n in report['nodes'] if n['id'] == 'front')['state'] == 'not-applicable'


def test_side_branch_needs_matching_hinge_and_tip():
    photo = {**PHOTO, 'view': 'right'}
    annotation = outline('temple-right', [(20, 50), (280, 50), (280, 65), (20, 65)])
    report = audit_photo(photo, [annotation])
    assert report['blocking']
    annotation['anchors'] = dict(hinge=dict(x=278, y=58), tip=dict(x=25, y=58))
    assert not audit_photo(photo, [annotation])['blocking']
    annotation['anchors']['tip'] = dict(x=25, y=120)
    assert any(i['code'] == 'arm-anchor-disconnected' for i in audit_photo(photo, [annotation])['issues'])
    assert audit_photo(PHOTO, [annotation])['blocking']  # Folded arm is not a profile.


def test_reference_view_does_not_fake_frontal_or_inner_arm_evidence():
    report = audit_photo({**PHOTO, 'view': 'rear'}, [])
    assert report['status'] == 'reference-only'
    assert all(n['state'] == 'not-observed-in-view' for n in report['nodes'])


def test_project_excludes_held_out_and_unapproved_and_marks_mirroring():
    annotations = front_shapes()
    for a in annotations:
        a['approved'] = True
    arm = outline('temple-left', [(20, 50), (280, 50), (280, 65), (20, 65)],
                  anchors=dict(hinge=dict(x=278, y=58), tip=dict(x=25, y=58)),
                  approved=True, source='explicit-mirrored-side-assumption:automatic')
    arm['photo_id'] = 'side'
    pad = outline('nose-pad-left', [(120, 80), (130, 80), (130, 90), (120, 90)], approved=True)
    pad['photo_id'] = 'held-out'
    project = dict(category='thin-metal', annotations=annotations+[arm, pad], photos=[PHOTO,
                   {**PHOTO, 'id': 'side', 'view': 'right'}, {**PHOTO, 'id': 'held-out', 'role': 'validation'}])
    report = audit_project(project)
    nodes = {n['id']: n for n in report['nodes']}
    assert nodes['temple-left']['state'] == 'inferred'
    assert nodes['temple-right']['state'] == nodes['nose-pad-left']['state'] == 'uncertain'
    assert len(report['photos']) == 2
    assert 'symmetry' in ' '.join(report['guidance'])
    project['annotations'][0]['approved'] = False
    assert next(n for n in audit_project(project)['nodes'] if n['id'] == 'front')['state'] == 'uncertain'


def test_audit_recomputes_geometry_instead_of_trusting_stale_evidence():
    annotation = front_shapes()[0]
    annotation['evidence'] = dict(schema_version=1, state='observed', method='geometric-anatomy-audit',
                                  review_required=True, notes=[])
    assert Annotation.model_validate(annotation).model_dump()['evidence']['state'] == 'observed'
    annotation['points'][0]['x'] = -30
    assert audit_photo(PHOTO, [annotation])['blocking']


def synthetic_unknown_frame(style, width=720, height=340):
    image = np.full((height, width, 3), 255, np.uint8)
    for cx in (190, 530):
        if style == 'ellipse':
            cv2.ellipse(image, (cx, 170), (150, 130), 0, 0, 360, (175, 180, 190), -1)
            cv2.ellipse(image, (cx, 170), (140, 120), 0, 0, 360, (55, 60, 65), -1)
        else:
            # Asymmetric rounded polygon shape not taken from a SKU template.
            points = np.array([[cx-150, 90], [cx-85, 40], [cx+95, 55],
                               [cx+150, 105], [cx+100, 270], [cx-95, 295]])
            cv2.fillPoly(image, [points], (175, 180, 190))
            inner = np.round((points-[cx, 170])*.92+[cx, 170]).astype(np.int32)
            cv2.fillPoly(image, [inner], (55, 60, 65))
    cv2.line(image, (328, 130), (392, 130), (175, 180, 190), 12)
    return image


@pytest.mark.parametrize('style', ['ellipse', 'polygon'])
@pytest.mark.parametrize('offset', [(90, 90), (870, 540)])
def test_unknown_shapes_recover_in_product_coordinates(style, offset, tmp_path):
    source = synthetic_unknown_frame(style)
    x, y = offset
    canvas = np.full((1100, 1900, 3), 255, np.uint8)
    canvas[y:y+340, x:x+720] = source
    before = canvas.copy()
    result = recover_product_local_front(canvas, {'id': 'front'}, 1, 1)
    assert result is not None
    assert np.array_equal(before, canvas)
    annotations = result['annotations']
    shapes = {a['part']: Polygon([(p['x'], p['y']) for p in a['points']]) for a in annotations}
    assert {'front', 'lens-left', 'lens-right'} <= shapes.keys()
    assert shapes['lens-left'].centroid.x == pytest.approx(x+190, abs=15)
    assert shapes['lens-right'].centroid.y == pytest.approx(y+170, abs=25)
    assert all(a['source'].startswith('product-local:') and not a['approved'] for a in annotations)
    path = tmp_path/'offset.png'
    cv2.imwrite(str(path), canvas)
    integrated = extract_annotations({**PHOTO, 'path': str(path), 'width': 1900, 'height': 1100})
    assert integrated['anatomy']['status'] == 'supported-structure'
    assert integrated['product_local']['method'] == 'quiet-studio-product-crop'
    assert all(a['evidence']['review_required'] for a in integrated['annotations'])


@pytest.mark.parametrize('negative', ['blank', 'two-blobs', 'noisy-border', 'clipped'])
def test_product_local_never_substitutes_a_frame_for_negative_images(negative):
    image = np.full((800, 1300, 3), 255, np.uint8)
    if negative == 'two-blobs':
        for x in (300, 700):
            cv2.ellipse(image, (x, 400), (140, 120), 0, 0, 360, (55, 55, 55), -1)
    elif negative == 'noisy-border':
        image[:30] = np.random.default_rng(10).integers(100, 255, (30, 1300, 3), dtype=np.uint8)
    elif negative == 'clipped':
        image[:340, :720] = synthetic_unknown_frame('ellipse')
        image[:, :20] = (50, 50, 50)
    assert recover_product_local_front(image, {'id': 'front'}, 1, 1) is None


def test_product_local_respects_source_scaling():
    image = np.full((800, 1500, 3), 255, np.uint8)
    image[200:540, 400:1120] = synthetic_unknown_frame('ellipse')
    result = recover_product_local_front(image, {'id': 'front'}, .5, .75)
    assert result is not None
    left = next(a for a in result['annotations'] if a['part'] == 'lens-left')
    centroid = Polygon([(p['x'], p['y']) for p in left['points']]).centroid
    assert centroid.x == pytest.approx(590*.5, abs=5)
    assert centroid.y == pytest.approx(370*.75, abs=5)
