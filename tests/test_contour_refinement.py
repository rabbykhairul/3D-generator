import numpy as np
from shapely.geometry import Polygon

from backend.reconstruction.contour_refinement import refine_closed_contour


def test_rectangular_construction_corners_and_straight_edges_survive():
    original = np.array([[10, 10], [210, 10], [210, 18], [10, 18]], dtype=float)
    refined, info = refine_closed_contour(original)
    assert info['applied']
    assert info['locked_corners'] == 4
    for corner in original:
        assert np.min(np.linalg.norm(refined - corner, axis=1)) < 1e-8
    assert Polygon(refined).symmetric_difference(Polygon(original)).area < 1e-8


def test_pixel_stair_steps_are_smoothed_with_bounded_silhouette_change():
    original = np.array([[0, 0], [20, 0], [21, 1], [40, 0], [41, 1],
                         [60, 0], [61, 1], [80, 0], [100, 0], [100, 20], [0, 20]], dtype=float)
    refined, info = refine_closed_contour(original)
    assert info['applied']
    assert info['max_boundary_displacement_px'] <= .75
    assert Polygon(refined).is_valid
    for x in (21, 41, 61):
        original_peak = original[np.where(original[:, 0] == x)[0][0], 1]
        refined_peak = refined[np.argmin(abs(refined[:, 0] - x)), 1]
        assert refined_peak < original_peak - .25
    assert np.allclose(refined.min(axis=0), original.min(axis=0), atol=1e-8)
    assert np.allclose(refined.max(axis=0), original.max(axis=0), atol=1e-8)


def test_smooth_arc_gets_dense_continuous_contour_without_corner_overshoot():
    angles = np.linspace(-np.pi, 0, 15)
    arc = np.column_stack((60 + 50*np.cos(angles), 60 + 35*np.sin(angles)))
    original = np.vstack((arc, [[110, 90], [10, 90]]))
    refined, info = refine_closed_contour(original)
    assert info['applied']
    assert len(refined) > len(original) * 2
    assert info['max_boundary_displacement_px'] <= .75
    assert Polygon(refined).is_valid
    assert np.allclose(refined.min(axis=0), original.min(axis=0), atol=1e-8)
    assert np.allclose(refined.max(axis=0), original.max(axis=0), atol=1e-8)


def test_ambiguous_self_intersection_is_left_for_review():
    bowtie = np.array([[0, 0], [20, 20], [0, 20], [20, 0]], dtype=float)
    refined, info = refine_closed_contour(bowtie)
    assert not info['applied']
    assert info['reason'] == 'invalid_source'
    np.testing.assert_array_equal(refined, bowtie)
