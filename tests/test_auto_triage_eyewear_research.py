from PIL import Image

from scripts.auto_triage_eyewear_research import clean_background, paired_front


def polygon(part, coords):
    return {'part': part, 'points': [{'x': x, 'y': y} for x, y in coords]}


def test_background_filter_rejects_lifestyle_color(tmp_path):
    white = tmp_path / 'white.png'
    dark = tmp_path / 'dark.png'
    Image.new('RGB', (100, 100), 'white').save(white)
    Image.new('RGB', (100, 100), '#624a38').save(dark)
    assert clean_background(white)
    assert not clean_background(dark)


def test_front_pair_requires_bilateral_lenses():
    outer = polygon('front', [(5, 20), (95, 20), (95, 80), (5, 80)])
    left = polygon('lens-left', [(12, 30), (43, 30), (43, 70), (12, 70)])
    right = polygon('lens-right', [(57, 30), (88, 30), (88, 70), (57, 70)])
    assert paired_front([outer, left, right], 100, 100)
    asymmetric = polygon('lens-right', [(67, 30), (88, 30), (88, 54), (67, 54)])
    assert not paired_front([outer, left, asymmetric], 100, 100)
