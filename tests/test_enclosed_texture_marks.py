"""White printing is appearance, while outside background isn't material."""
import numpy as np
from PIL import Image, ImageDraw
from shapely.geometry import Polygon

from backend.reconstruction.engine import _frame_texture


def test_enclosed_white_mark_kept_and_background_padding_excluded(tmp_path):
    image = Image.new('RGB',(300,120),'white')
    draw = ImageDraw.Draw(image)
    draw.rectangle((20,40,280,80),fill=(32,54,75))
    draw.ellipse((224,49,235,61),fill='white')
    path = tmp_path/'printed.png'
    image.save(path)
    texture = np.asarray(_frame_texture(dict(path=str(path),width=300,height=120),
                                       Polygon([(20,40),(280,40),(280,80),(20,80)])))
    assert np.all(texture[55,230,:3] > 245), 'Enclosed white printing must survive'
    assert np.all(texture[65,200,:3] == [32,54,75])
    assert np.max(texture[35,200,:3]) < 100, 'Background outside approved frame must not bleed into its edge'
