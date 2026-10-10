"""Neutral transparent-frame recovery on actual and constructed white photos."""

from pathlib import Path
from copy import deepcopy

import cv2
import numpy as np
import pytest
from shapely.geometry import Polygon, Point

from backend.reconstruction.clear_front import recover_clear_front_mask
from backend.reconstruction.engine import extract_annotations, reconstruct
from backend.store import get_project
from scripts.evaluate_resolution_matrix import derivative, validate_annotations


def _neutral_frame():
    image = np.full((400, 800, 3), 255, np.uint8)
    for center_x in (230, 570):
        cv2.ellipse(image, (center_x, 205), (170, 112), 0, 0, 360,
                    (232, 232, 232), 13)
    cv2.rectangle(image, (390, 156), (410, 177), (232, 232, 232), -1)
    return image


def test_neutral_rim_requires_stable_paired_openings():
    frame = _neutral_frame()
    mask, threshold = recover_clear_front_mask(frame)
    assert mask is not None and threshold in {8, 10, 12, 16, 20, 24}
    assert mask.shape == frame.shape[:2]
    blank = np.full_like(frame, 255)
    shadow = blank.copy()
    cv2.ellipse(shadow, (400, 300), (320, 15), 0, 0, 360, (232, 232, 232), -1)
    noisy = np.clip(frame.astype(np.int16) +
                    np.random.default_rng(1).normal(0, 9, frame.shape),
                    0, 255).astype(np.uint8)
    for negative in (blank, shadow, noisy, frame[105:305, 50:750]):
        assert recover_clear_front_mask(negative)[0] is None


@pytest.mark.parametrize("size,expected", [(64, False), (1080, True), (4096, True)])
def test_actual_water_clear_front_only_returns_safe_contours(tmp_path: Path, size, expected):
    project = get_project('b0bc9b703db9463abab0a9a4d30c0b4d')
    if project is None:
        pytest.skip('Local water-clear reference unavailable')
    photo = next(p for p in project['photos'] if p['view'] == 'front')
    source = cv2.imread(photo['path'])
    if source is None:
        pytest.skip('Local water-clear image unavailable')
    image = derivative(source, size)
    path = tmp_path / f'clear-{size}.png'
    assert cv2.imwrite(str(path), image)
    result = extract_annotations({'id': f'clear-{size}', 'path': str(path),
                                  'view': 'front', 'role': 'fitting'})
    parts = [a['part'] for a in result['annotations']]
    assert not validate_annotations(result, image.shape[1], image.shape[0])
    if expected:
        assert result['segmentation']['method'] == 'stable-neutral-front'
        assert parts == ['front', 'lens-left', 'lens-right']
        assert all(not a['approved'] for a in result['annotations'])
    else:
        assert not parts


@pytest.mark.parametrize('include_side', [False, True])
def test_actual_water_clear_suggestion_survives_draft_reconstruction(tmp_path: Path, include_side):
    project = get_project('b0bc9b703db9463abab0a9a4d30c0b4d')
    if project is None:
        pytest.skip('Local water-clear reference unavailable')
    photo = next(p for p in project['photos'] if p['view'] == 'front')
    draft = deepcopy(project)
    draft['photos'] = [photo]
    draft['annotations'] = extract_annotations(photo)['annotations']
    assert [a['part'] for a in draft['annotations']] == [
        'front', 'lens-left', 'lens-right']
    if include_side:
        side = next(p for p in project['photos'] if p['view'] == 'right')
        draft['photos'].append(side)
        side_annotations = extract_annotations(side)['annotations']
        assert len(side_annotations) == 1
        draft['annotations'].extend(side_annotations)
    # Simulate review only in a disposable copy. These values are diagnostic
    # assumptions, not merchant measurements or an approval of the contours.
    for annotation in draft['annotations']:
        annotation['approved'] = True
    draft['category'] = 'transparent-full-rim'
    for name, value in {'frame_width': 140, 'lens_width': 51,
                        'lens_height': 40, 'bridge_width': 19,
                        'temple_length': 140}.items():
        draft['measurements'][name] = {'value': value, 'source': 'diagnostic-assumption'}
    result = reconstruct(draft, tmp_path / 'draft')
    assert result['checks']['front_parts_watertight']
    assert result['checks']['consistent_outward_normals']
    assert result['reconstruction_status'] == 'incomplete'
    if include_side:
        import trimesh
        from backend.reconstruction.render_mesh import welded_geometry
        scene=trimesh.load(tmp_path/'draft/model.glb',force='scene',process=False)
        arms=[mesh for name,mesh in scene.geometry.items() if name.startswith('temple_right')]
        assert arms
        assert all(welded_geometry(mesh).is_watertight for mesh in arms)


@pytest.mark.parametrize('size', [540, 1080, 2160])
def test_clear_brow_uses_faint_surface_not_internal_highlight(tmp_path, size):
    project = get_project('b0bc9b703db9463abab0a9a4d30c0b4d')
    if project is None:
        pytest.skip('Local water-clear reference unavailable')
    photo = next(p for p in project['photos'] if p['view'] == 'front')
    image = cv2.imread(photo['path'])
    path = tmp_path / 'scaled-clear.png'
    cv2.imwrite(str(path), cv2.resize(image, (size, size)))
    result = extract_annotations({'id': 'clear-brow', 'view': 'front', 'path': str(path)})
    ann = next(a for a in result['annotations'] if a['part'] == 'front')
    scale = size / 1080
    polygon = Polygon([(p['x']/scale, p['y']/scale) for p in ann['points']])
    assert polygon.is_valid
    # Genuine transparent material above the nasal opening must not be cut
    # away by the bright internal reflection. The opening must remain open.
    for x in (400, 470, 540, 610, 680):
        assert polygon.contains(Point(x, 415))
    assert not polygon.contains(Point(540, 475))
    # Low-threshold upper recovery must not import the ground contact shadow.
    assert not polygon.contains(Point(260, 744))
