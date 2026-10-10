import numpy as np
from PIL import Image

from scripts.render_fidelity_cpu import camera_basis, raster_triangle, sample_texture


def test_camera_basis_orthogonal():
    basis = camera_basis(np.array([-.24, .06, .3]), np.array([0, 0, -.05]))
    assert np.allclose(basis @ basis.T, np.eye(3))


def test_z_buffer_keeps_nearest_triangle_independent_of_order():
    screen = np.array([[1, 1], [8, 1], [1, 8]])
    for order in ((1, 2), (2, 1)):
        image = np.full((10, 10, 3), 255, dtype=np.uint8)
        depth = np.full((10, 10), np.inf)
        for z in order:
            raster_triangle(image, depth, screen, np.full(3, z),
                            colors=[255 if z == 1 else 0, 0, 0], surface='photo')
        assert np.array_equal(image[2, 2], [255, 0, 0])
        assert depth[2, 2] == 1
        assert np.array_equal(image[9, 9], [255, 255, 255])


def test_texture_v_axis_and_bilinear_sampling():
    image = Image.fromarray(np.array([[[255, 0, 0], [0, 255, 0]],
                                     [[0, 0, 255], [255, 255, 255]]], dtype=np.uint8))
    samples = sample_texture(image, np.array([[.25, .75], [.25, .25], [.5, .5]]))
    assert np.allclose(samples, [[255, 0, 0], [0, 0, 255], [127.5]*3])
