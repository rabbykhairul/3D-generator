import copy

import cv2
import numpy as np
import pytest
from shapely.geometry import Point, Polygon, box

from backend.reconstruction.outline_finish import finish_front_outlines, finish_side_outlines, trim_lower_shadow
from backend.reconstruction.smooth_curve import smooth_detected_curve


@pytest.mark.parametrize('scale', [1, 2])
def test_short_waves_removed_without_ellipse_fit_or_losing_design_corners(scale):
    theta = np.linspace(0, 2*np.pi, 1000, endpoint=False)
    # Broad non-circular shape with short segmentation ripples.
    base = 110 + 12*np.cos(3*theta)
    raw = Polygon(np.column_stack(((base+1.8*np.sin(65*theta))*np.cos(theta),
                                   (base+1.8*np.sin(65*theta))*np.sin(theta)))*scale)
    truth = Polygon(np.column_stack((base*np.cos(theta), base*np.sin(theta)))*scale)
    clean = smooth_detected_curve(raw, scale)
    assert clean.is_valid
    assert clean.symmetric_difference(truth).area < raw.symmetric_difference(truth).area*.55
    assert clean.boundary.hausdorff_distance(raw.boundary) <= 2.75*scale
    assert abs(clean.area-raw.area)/raw.area < .008
    rectangle = box(0, 0, 100*scale, 80*scale)
    cornered = smooth_detected_curve(rectangle, scale)
    assert cornered.boundary.distance(Point(0, 0)) < .3*scale
    assert cornered.intersection(rectangle).area/rectangle.area > .999


def shadow_fixture(color=(40, 40, 40), shadow=True, shadow_gray=225):
    image = np.full((300, 600, 3), 255, np.uint8)
    front = box(30, 40, 570, 250)
    lenses = [box(60, 70, 260, 240), box(340, 70, 540, 240)]
    cv2.rectangle(image, (30, 40), (570, 250), color, -1)
    for lens in lenses:
        x0, y0, x1, y1 = map(int, lens.bounds)
        cv2.rectangle(image, (x0, y0), (x1, y1), (235, 240, 245), -1)
    contaminated = front
    if shadow:
        # Attached pale ledges beneath both lower rims, already in the proposal.
        for x0, x1 in ((75, 245), (355, 525)):
            cv2.rectangle(image, (x0, 251), (x1, 261), (shadow_gray,)*3, -1)
            contaminated = contaminated.union(box(x0, 249, x1, 261))
    return image, front, contaminated, lenses


@pytest.mark.parametrize('shadow_gray', [180, 225])
def test_shadow_cut_uses_source_pixels_preserves_real_lower_rim_and_source(shadow_gray):
    image, truth, contaminated, lenses = shadow_fixture(shadow_gray=shadow_gray)
    original = image.copy()
    clean, info = trim_lower_shadow(image, contaminated, lenses)
    assert info['applied'] and info['removed_pixels'] > 0
    assert not clean.covers(Point(150, 257))
    assert clean.covers(Point(150, 249))
    assert clean.intersection(truth).area/truth.area > .999
    assert all(clean.contains(lens) for lens in lenses)
    assert np.array_equal(image, original)


@pytest.mark.parametrize('color', [(40, 40, 40), (205, 205, 205), (225, 210, 205)])
def test_real_frame_bottom_is_not_cut_just_because_it_is_low_or_pale(color):
    image, truth, _, lenses = shadow_fixture(color=color, shadow=False)
    clean, info = trim_lower_shadow(image, truth, lenses)
    assert not info['applied']
    assert clean.equals(truth)


def test_ambiguous_pale_rim_and_nonuniform_background_are_left_intact():
    image, _, contaminated, lenses = shadow_fixture(color=(210, 210, 210))
    clean, info = trim_lower_shadow(image, contaminated, lenses)
    assert not info['applied'] and clean.equals(contaminated)
    image, _, contaminated, lenses = shadow_fixture()
    image[:10] = np.linspace(170, 250, 600).astype(np.uint8)[None, :, None]
    clean, info = trim_lower_shadow(image, contaminated, lenses)
    assert not info['applied'] and clean.equals(contaminated)


def test_pale_tinted_lens_shadow_path_has_no_unbound_edge():
    image = np.full((300, 600, 3), 255, np.uint8)
    front = box(30, 40, 570, 250)
    lenses = [box(60, 70, 260, 240), box(340, 70, 540, 240)]
    cv2.rectangle(image, (30, 40), (570, 250), (245, 245, 245), -1)
    for lens in lenses:
        x0, y0, x1, y1 = map(int, lens.bounds)
        cv2.rectangle(image, (x0, y0), (x1, y1), (140, 140, 140), -1)
    for x0, x1 in ((75, 245), (355, 525)):
        cv2.rectangle(image, (x0, 241), (x1, 246), (225, 225, 225), -1)
        cv2.rectangle(image, (x0, 247), (x1, 250), (253, 253, 253), -1)
    clean, info = trim_lower_shadow(image, front, lenses)
    assert clean.is_valid and isinstance(info['applied'], bool)


def test_joint_finish_respects_reviewed_geometry_and_coordinate_scaling():
    image, _, contaminated, lenses = shadow_fixture()
    def annotation(part, poly, approved=False):
        return dict(part=part, approved=approved, source='automatic-sam2-lens-refined-v1',
                    points=[dict(x=x*2, y=y*3) for x, y in poly.exterior.coords[:-1]])
    annotations = [annotation('front', contaminated),
                   *[annotation(part, poly) for part, poly in zip(('lens-left', 'lens-right'), lenses)]]
    approved = annotation('shield', box(40, 40, 550, 240), True)
    annotations.append(approved)
    saved = copy.deepcopy(approved)
    info = finish_front_outlines(image, annotations, 2, 3)
    assert info['shadow']['applied']
    assert approved == saved
    polygons = {a['part']: Polygon([(p['x']/2, p['y']/3) for p in a['points']])
                for a in annotations}
    assert not polygons['front'].covers(Point(150, 257))
    assert all(polygons['front'].contains(polygons[p]) for p in ('lens-left', 'lens-right'))
    assert all(not a['approved'] for a in annotations[:3])


@pytest.mark.parametrize('source', ['automatic-test', 'product-local:automatic-test',
                                    'convex-occlusion-completion:automatic-test'])
def test_front_finishing_also_handles_fallback_sources(source):
    image, _, front, lenses = shadow_fixture(shadow=False)
    theta = np.linspace(0, 2*np.pi, 720, endpoint=False)
    lenses = [Polygon(np.column_stack((cx+(85+.9*np.sin(theta*45))*np.cos(theta),
                                       155+(73+.9*np.sin(theta*45))*np.sin(theta))))
              for cx in (160, 440)]
    annotations = [dict(part=part, approved=False, source=source,
                        points=[dict(x=x, y=y) for x, y in p.exterior.coords[:-1]])
                   for part, p in [('front', front), *zip(('lens-left', 'lens-right'), lenses)]]
    info = finish_front_outlines(image, annotations)
    assert 'lens-left' in info['smoothed_parts']


def test_side_ripples_are_suppressed_without_changing_anchor_positions():
    x = np.arange(30, 971, dtype=float)
    center = 85 + x*.035
    wave = 1.3*np.sin(x*.35)
    shape = Polygon(np.vstack((np.column_stack((x, center-8+wave)),
                               np.column_stack((x[::-1], (center+8+wave)[::-1])))))
    truth = Polygon(np.vstack((np.column_stack((x, center-8)),
                               np.column_stack((x[::-1], (center+8)[::-1])))))
    ann = dict(part='temple-left', approved=False, source='single-visible-arm-stable-faint-v1',
               anchors=dict(hinge=dict(x=30., y=86.05), tip=dict(x=970., y=118.95)),
               points=[dict(x=x, y=y) for x, y in shape.exterior.coords[:-1]])
    saved_anchors = copy.deepcopy(ann['anchors'])
    info = finish_side_outlines(np.full((300, 1080, 3), 255, np.uint8), [ann])
    clean = Polygon([(p['x'], p['y']) for p in ann['points']])
    assert info['smoothed_parts'] == ['temple-left']
    assert clean.is_valid
    assert clean.symmetric_difference(truth).area < shape.symmetric_difference(truth).area*.45
    assert ann['anchors'] == saved_anchors
    assert abs(clean.area-shape.area)/shape.area < .02


def test_short_sharp_mask_teeth_are_not_preserved_as_design_corners():
    from scipy.signal import sawtooth
    x = np.arange(1, 600, dtype=float)
    teeth = 3*sawtooth(2*np.pi*x/14, width=.5)
    raw = Polygon([(0,0), *zip(x, teeth), (600,0), (600,120), (0,120)])
    truth = box(0,0,600,120)
    clean = smooth_detected_curve(raw, max_displacement=4.)
    assert clean.is_valid
    assert clean.symmetric_difference(truth).area < raw.symmetric_difference(truth).area*.45
    assert clean.boundary.distance(Point(0,0)) < .3
    assert clean.boundary.distance(Point(600,0)) < .3
