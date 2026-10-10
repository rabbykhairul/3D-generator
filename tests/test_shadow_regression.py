import copy

import cv2
import numpy as np
import pytest
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt
from shapely.geometry import Point, Polygon

from backend.reconstruction.engine import _foreground_mask, _frame_texture, extract_annotations, reconstruct, NeedsCorrection
from test_reconstruction import snapshot


def catalogue_photo(tmp_path, color='#555555'):
    image = Image.new('RGB', (800, 400), 'white')
    draw = ImageDraw.Draw(image)
    # A low-contrast cast shadow meets the lower rim, with lateral ledges.
    draw.ellipse((50, 297, 750, 317), fill='#dddddd')
    draw.rectangle((80, 70, 720, 305), fill=color)
    draw.rectangle((100, 90, 350, 285), fill='white')
    draw.rectangle((450, 90, 700, 285), fill='white')
    # Genuine, one-pixel dark hardware must survive. No morphological opening.
    draw.line((300, 60, 500, 60), fill='#333333', width=1)
    path = tmp_path / 'shadow.png'
    image.save(path)
    return dict(id='front', path=str(path), width=800, height=400, view='front')


def test_soft_shadow_not_contour_and_thin_dark_material_survives(tmp_path):
    photo = catalogue_photo(tmp_path)
    mask, info = _foreground_mask(cv2.imread(photo['path']))
    assert info['method'] == 'adaptive-light-background'
    assert mask[60, 400] == 255
    assert mask[303, 90] == 255  # actual lower rim
    assert mask[310, 400] == 0  # photographed shadow
    result = extract_annotations(photo)
    front = next(a for a in result['annotations'] if a['part'] == 'front')
    poly = Polygon([(p['x'], p['y']) for p in front['points']])
    assert poly.covers(Point(400, 304))
    assert not poly.covers(Point(400, 310))
    assert not poly.covers(Point(65, 305))
    assert not front['approved']


@pytest.mark.parametrize('color', ['#555555', '#704925', '#315e32'])
def test_texture_padding_uses_product_color_not_white_or_shadow(tmp_path, color):
    photo = catalogue_photo(tmp_path, color)
    # Include a small imperfect boundary margin to exercise color-sample rejection.
    poly = Polygon([(78, 68), (722, 68), (722, 307), (78, 307)],
                   [[(100, 90), (350, 90), (350, 285), (100, 285)],
                    [(450, 90), (700, 90), (700, 285), (450, 285)]])
    image = np.asarray(_frame_texture(photo, poly))
    expected = np.array([int(color[i:i+2], 16) for i in (1, 3, 5)])
    assert np.array_equal(image[310, 400, :3], expected)
    assert np.array_equal(image[0, 0, :3], expected)
    assert (image[:, :, 3] == 255).all()


def test_texture_retains_pale_tinted_acetate_inside_reviewed_frame(tmp_path):
    source = Image.new('RGB', (400, 180), 'white')
    draw = ImageDraw.Draw(source)
    draw.rectangle((20, 160, 380, 170), fill='#dddddd')
    draw.rectangle((20, 20, 380, 160), fill='#f1edf5')
    draw.rectangle((50, 40, 175, 135), fill='white')
    draw.rectangle((225, 40, 350, 135), fill='white')
    draw.rectangle((185, 60, 190, 75), fill='#33253e')
    path = tmp_path / 'tinted.png'
    source.save(path)
    poly = Polygon([(20, 20), (380, 20), (380, 160), (20, 160)],
                   [[(50, 40), (175, 40), (175, 135), (50, 135)],
                    [(225, 40), (350, 40), (350, 135), (225, 135)]])
    texture = np.asarray(_frame_texture({'path': str(path), 'width': 400, 'height': 180}, poly))
    assert np.array_equal(texture[90, 200, :3], [241, 237, 245])
    assert np.array_equal(texture[65, 187, :3], [51, 37, 62])
    assert not np.array_equal(texture[165, 200, :3], [221, 221, 221])


def test_narrow_arm_padding_smooths_color_bands_without_blurring_photo(tmp_path):
    # A narrow diagonal arm carries alternating observed pixels. Extending
    # each interior pixel to the entire atlas created dark vertical scallops
    # when the render sampled just outside the cap boundary.
    source = np.full((130, 360, 3), 255, np.uint8)
    for x in range(25, 335):
        center = 30 + round((x - 25) * .15)
        source[center - 4:center + 5, x] = 55 if (x // 3) % 2 else 150
    path = tmp_path / 'narrow-arm.png'
    Image.fromarray(source).save(path)
    photo = {'path': str(path), 'width': 360, 'height': 130}
    polygon = Polygon([(24, 24), (335, 71), (335, 83), (24, 36)])
    texture = np.asarray(_frame_texture(photo, polygon))

    mask = np.zeros(source.shape[:2], np.uint8)
    cv2.fillPoly(mask, [np.round(np.asarray(polygon.exterior.coords)).astype(np.int32)], 255)
    foreground, _ = _foreground_mask(cv2.cvtColor(source, cv2.COLOR_RGB2BGR))
    interior = cv2.erode(np.uint8((mask > 0) & (foreground > 0)),
                         np.ones((3, 3), np.uint8)) > 0
    _, nearest = distance_transform_edt(~interior, return_indices=True)
    old_padding = source[nearest[0], nearest[1]]

    # Interior cap pixels retain real printed/highlight detail exactly.
    assert np.array_equal(texture[interior, :3], source[interior])
    assert np.all(texture[:, :, 3] == 255)
    # Both sides of the arm receive quieter padding than nearest-pixel bands.
    for y in (5, 115):
        old_edges = np.diff(old_padding[y, 50:310, 0].astype(float))
        new_edges = np.diff(texture[y, 50:310, 0].astype(float))
        assert np.std(new_edges) < np.std(old_edges) * .65
        assert np.max(texture[y, 50:310, 0]) < 200  # No white background bleed.


def test_old_automatic_outline_requires_review_not_silent_reuse(tmp_path):
    data = snapshot(tmp_path, 'opaque-full-rim')
    original = copy.deepcopy(data)
    data['annotations'][0]['source'] = 'merchant-reviewed:automatic'
    with pytest.raises(NeedsCorrection, match='shadow-aware'):
        reconstruct(data, tmp_path / 'out')
    # Manual outlines remain authoritative; version changes don't rewrite them.
    reconstruct(original, tmp_path / 'manual')
