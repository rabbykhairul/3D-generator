import numpy as np
import cv2

from scripts.score_heldout_silhouette import iou, reference_mask, projected_vertices, optimize_transform, render


def test_foreground_iou_does_not_reward_matching_white_background():
    a = np.zeros((100, 100), np.uint8)
    b = a.copy()
    a[10:20, 10:20] = 1
    b[40:50, 40:50] = 1
    assert iou(a, b) == 0
    assert iou(a, a) == 1
    assert iou(np.zeros_like(a), np.zeros_like(a)) == 0


def test_mask_keeps_clear_aperture_empty_and_rejects_light_shadow(tmp_path):
    photo = np.full((100, 100, 3), 255, np.uint8)
    cv2.rectangle(photo, (20, 20), (80, 80), (50, 50, 50), 4)
    photo[88:92, 20:80] = 240
    path = tmp_path / 'reference.png'
    cv2.imwrite(str(path), photo)
    mask = reference_mask(path, 100, 215)
    assert mask[20, 40] == 1
    assert mask[50, 50] == 0
    assert mask[90, 40] == 0


def test_pinhole_depth_changes_apparent_scale_without_geometry_deformation():
    front = np.array([[[.01, .01, .1], [.02, .01, .1], [.01, .02, .1]]])
    back = front.copy()
    back[:, :, 2] = -.1
    near, far = projected_vertices([front, back], 0, 0, 0, .5)
    assert np.allclose(near / far, 1.5)
    assert np.array_equal(front[:, :, 2], np.full((1, 3), .1))


def test_nonsquare_photo_is_letterboxed_without_stretching(tmp_path):
    photo = np.full((60, 120, 3), 255, np.uint8)
    cv2.rectangle(photo, (50, 20), (70, 40), (0, 0, 0), -1)
    path = tmp_path / 'wide.png'
    cv2.imwrite(str(path), photo)
    mask = reference_mask(path, 120, 215)
    ys, xs = np.where(mask)
    assert xs.min() == 50 and xs.max() == 70
    assert ys.min() == 50 and ys.max() == 70


def test_registration_refinement_can_improve_while_camera_is_pinned():
    triangles = [np.array([[[0., 0., 0.], [.02, 0., 0.], [0., .02, 0.]]])]
    camera = (0., 0., 0., .5)
    target = render(triangles, camera, 100, (500., 40., 40.))
    initial = (500., 45., 43.)
    improved = optimize_transform(triangles, camera, target, initial)
    assert iou(target, render(triangles, camera, 100, improved)) >= iou(
        target, render(triangles, camera, 100, initial))
