from pathlib import Path

import cv2
import numpy as np
import pytest
from shapely.geometry import Polygon

from backend.reconstruction.detection_background import prepare_detection_background
from backend.reconstruction.engine import extract_annotations

REFERENCE = Path(__file__).parent / 'fixtures/rimless/front.png'


@pytest.mark.parametrize('background', [(210,210,210), (245,245,245), (225,235,245)])
def test_flat_background_gain_preserves_relative_contrast_and_source(background):
    image = np.full((80,120,3), background, dtype=np.uint8)
    image[30:50,40:80] = np.rint(np.array(background)*.8)
    original = image.copy()
    work, info = prepare_detection_background(image)
    assert info['applied']
    assert not info['source_pixels_changed']
    assert np.array_equal(image, original)
    assert np.all(work[0,0] == 255)
    assert np.max(np.abs(work[40,60].astype(int)-204)) <= 1


def test_unsafe_backgrounds_are_not_forced_white():
    white = np.full((80,120,3),255,np.uint8)
    dark = np.full_like(white,80)
    gradient = np.tile(np.linspace(180,245,120).astype(np.uint8)[None,:,None], (80,1,3))
    rng = np.random.default_rng(0)
    textured = rng.integers(190,245,white.shape,dtype=np.uint8)
    for image in (white,dark,gradient,textured):
        work, info = prepare_detection_background(image)
        assert not info['applied']
        assert work is image


@pytest.mark.parametrize('background', [(210,210,210),(245,245,245),(225,235,245)])
def test_rimless_detection_on_dim_and_tinted_flat_backgrounds(tmp_path,background):
    image=cv2.imread(str(REFERENCE))
    image=np.clip(np.rint(image.astype(float)*np.array(background)/255),0,255).astype(np.uint8)
    path=tmp_path/'photo.png'
    cv2.imwrite(str(path),image)
    result=extract_annotations({'id':'dim-rimless','path':str(path),'view':'front'})
    assert result['background_calibration']['applied']
    assert result['construction']['category']=='rimless'
    parts={a['part'] for a in result['annotations']}
    assert {'lens-left','lens-right','bridge-hardware'} <= parts
    assert np.array_equal(cv2.imread(str(path)),image)
    assert not any(a['approved'] for a in result['annotations'])


def test_small_rimless_recovers_visible_edges_without_inventing_missing_pad(tmp_path):
    image=cv2.resize(cv2.imread(str(REFERENCE)),(320,320),interpolation=cv2.INTER_AREA)
    path=tmp_path/'small.png'
    cv2.imwrite(str(path),image)
    result=extract_annotations({'id':'small-rimless','path':str(path),'view':'front'})
    assert result['construction']['category']=='rimless'
    parts={a['part'] for a in result['annotations']}
    assert {'lens-left','lens-right','bridge-hardware'} <= parts
    assert 'nose-pad-right' not in parts
    assert any('Could not verify nose-pad-right' in w for w in result['warnings'])
    for a in result['annotations']:
        poly=Polygon([(p['x'],p['y']) for p in a['points']])
        assert poly.is_valid
        assert all(0<=v<=320 for xy in poly.exterior.coords for v in xy)
    # Compare to source evidence, rather than only counting detected parts.
    baseline=extract_annotations({'id':'original','path':str(REFERENCE),'view':'front'})
    original={a['part']:Polygon([(p['x']*320/1080,p['y']*320/1080) for p in a['points']])
              for a in baseline['annotations']}
    for a in result['annotations']:
        if not a['part'].startswith('lens-'):
            continue
        poly=Polygon([(p['x'],p['y']) for p in a['points']])
        expected=original[a['part']]
        assert poly.intersection(expected).area/poly.union(expected).area > .95


@pytest.mark.parametrize('background',[(210,210,210),(245,245,245),(225,235,245)])
def test_background_calibration_keeps_outer_arm_identity(tmp_path,background):
    source=REFERENCE.parent/'right.png'
    image=cv2.imread(str(source))
    baseline=extract_annotations({'id':'baseline','view':'right','path':str(source)})
    dim=np.clip(np.rint(image.astype(float)*np.array(background)/255),0,255).astype(np.uint8)
    path=tmp_path/'side.png'
    cv2.imwrite(str(path),dim)
    result=extract_annotations({'id':'dim','view':'right','path':str(path)})
    assert result['background_calibration']['applied']
    assert len(result['annotations'])==1
    expected=baseline['annotations'][0]
    found=result['annotations'][0]
    assert found['part']==expected['part']=='temple-right'
    a,b=[Polygon([(p['x'],p['y']) for p in ann['points']]) for ann in (expected,found)]
    assert a.intersection(b).area/a.union(b).area>.98
    for anchor in ('tip','hinge'):
        assert found['anchors'][anchor]['x']==pytest.approx(expected['anchors'][anchor]['x'],abs=2)
    assert np.array_equal(cv2.imread(str(path)),dim)


@pytest.mark.parametrize('size',[160,256,320])
def test_weak_threshold_retry_does_not_invent_blank_or_dark_rim_lenses(size):
    from backend.reconstruction.rimless_detection import recover_rimless_front
    image=np.full((size,size,3),255,np.uint8)
    assert recover_rimless_front(image,{}) is None
    for x in (size*.27,size*.73):
        cv2.ellipse(image,(round(x),round(size*.6)),(round(size*.18),round(size*.13)),
                    0,0,360,(40,40,40),max(2,round(size*.005)))
    assert recover_rimless_front(image,{}) is None
