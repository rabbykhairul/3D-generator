import numpy as np
import pytest
from PIL import Image
from shapely.geometry import Polygon

from backend.reconstruction.engine import _beveled_plate, _material, _split_t_junctions, _triangulated_plate


def test_flat_earcut_triangle_is_discarded_before_t_junction_splitting():
    points = np.array([[0., 0.], [5., 0.], [10., 0.]], dtype=float)
    assert _split_t_junctions(np.array([[0, 1, 2]], dtype=int), points) == []


def test_bevel_welds_numpy_and_python_float_coordinates_at_rounding_tie():
    # .15546875 rounds differently as np.float64 versus Python float at seven
    # decimals. The cap and inset walls must still share the same seam vertex.
    assert round(np.float64(.15546875), 7) != round(float(.15546875), 7)
    outline = Polygon([(0, 0), (5, 0), (10, 0), (10, 10), (0, 10)])
    _, mesh = _beveled_plate(
        outline, thickness_mm=1, bevel_mm=.15546875, scale_mm_per_px=1,
        xy_to_model=lambda p: np.array([p[0], p[1], 0.]),
        material=_material('#252a32', 1, 0, .35), name='test_bevel',
        photo={'path': 'no-photo', 'width': 10, 'height': 10},
    )
    assert mesh.is_watertight
    assert np.isfinite(mesh.vertices).all()
    assert np.isfinite(mesh.face_normals).all()


def test_narrow_bridge_limits_only_its_local_bevel():
    # The top bridge is one pixel tall; the broad lower rim has ample width.
    outline = Polygon([(0, 0), (40, 0), (40, 20), (30, 20), (10, 20), (0, 20)],
                      [[(2, 1), (38, 1), (38, 10), (2, 10)]])
    _, mesh = _beveled_plate(
        outline, thickness_mm=4, bevel_mm=.35, scale_mm_per_px=.1,
        xy_to_model=lambda p: np.array([p[0] / 10000, p[1] / 10000, 0.]),
        material=_material('#252a32', 1, 0, .35), name='test_local_bevel',
        photo={'path': 'no-photo', 'width': 40, 'height': 20},
    )
    assert mesh.is_watertight
    assert mesh.metadata['bevel_mm'] == .35
    assert mesh.metadata['bevel_min_mm'] < .1
    mesh.fix_normals(multibody=True)
    assert mesh.is_winding_consistent and mesh.volume > 0


def test_side_bevel_keeps_approved_shaft_sleeve_bounds_and_photo_uv(tmp_path):
    profile = Polygon([(0, -2), (70, -2), (70, -4), (100, -4),
                       (100, -2), (140, -2), (140, 2), (100, 2),
                       (100, 4), (70, 4), (70, 2), (0, 2)])
    photo_path = tmp_path / 'side.png'
    Image.new('RGB', (320, 100), '#514c48').save(photo_path)
    photo = {'path': str(photo_path), 'width': 320, 'height': 100}
    model = lambda p: np.array([0., p[1] / 1000, -p[0] / 1000])
    uv = lambda p: [(20 + p[0] * 2) / 320, 1 - (50 - p[1] * 4) / 100]
    material = _material('#514c48', 1, 0, .35)
    _, square = _triangulated_plate(profile, 2.2, model, material, 'square',
                                   photo, texture=True, extrusion_axis=0, uv_map=uv)
    _, beveled = _beveled_plate(profile, 2.2, .22, 1, model, material, 'beveled',
                               photo, extrusion_axis=0, uv_map=uv, bevel_back=True)
    assert beveled.is_watertight
    beveled.fix_normals(multibody=True)
    assert beveled.is_winding_consistent and beveled.volume > 0
    assert np.allclose(beveled.bounds, square.bounds)
    assert beveled.volume < square.volume
    assert beveled.metadata['bevel_mm'] <= .22 + 1e-9
    assert beveled.metadata['bevel_mm'] > .1
    assert len(np.unique(np.round(beveled.vertices[:, 0], 6))) == 4
    # The mapped source-photo UVs still cover the approved root and tip.
    assert beveled.visual.uv[:, 0].min() == pytest.approx(20 / 320)
    assert beveled.visual.uv[:, 0].max() == pytest.approx(300 / 320)
