from pathlib import Path
import cv2
import numpy as np
import pytest
from backend.store import get_project
from backend.reconstruction.engine import extract_annotations
from backend.reconstruction.opposite_arm import opposite_arm_evidence, clean_appearance_mask


@pytest.mark.parametrize('size,mirror', [(540, False), (1080, False), (2160, False), (1080, True)])
def test_olive_hinge_excludes_front_and_retains_rear_appearance(tmp_path, size, mirror):
    project = get_project('7def97639b034aa4be2d224429cc9f45')
    if project is None:
        pytest.skip('Local olive reference unavailable')
    photo = next(p for p in project['photos'] if p['view'] == 'right')
    original = cv2.imread(photo['path'])
    if original is None:
        pytest.skip('Local olive reference unavailable')
    image = cv2.resize(original, (size, size))
    if mirror:
        image = cv2.flip(image, 1)
    path = tmp_path/'side.png'
    cv2.imwrite(str(path), image)
    result = extract_annotations({**photo, 'path': str(path), 'view': 'left' if mirror else 'right'})
    arm, = result['annotations']
    hinge = arm['anchors']['hinge']
    x = size-1-hinge['x'] if mirror else hinge['x']
    assert .80 < x/size < .835
    assert .475 < hinge['y']/size < .515  # not the former .557 vertical-rim midpoint
    evidence, = result['appearance_evidence']
    assert evidence['part'] != arm['part']
    assert evidence['applied'] is False
    assert evidence['possible_mark_regions'] > 0
    # Opposite-arm evidence never becomes an additional geometry annotation.
    assert not arm['approved']


def test_blank_photo_has_no_opposite_arm_evidence():
    annotation = {'photo_id':'test', 'part':'temple-right',
                  'points':[{'x':20,'y':40},{'x':180,'y':40},{'x':180,'y':50},{'x':20,'y':50}],
                  'anchors':{'hinge':{'x':180,'y':45},'tip':{'x':20,'y':45}}}
    assert opposite_arm_evidence(np.full((200,200,3),255,np.uint8), annotation) == []


def test_attached_shadow_is_removed_without_cutting_enclosed_printing():
    contrast=np.zeros((80,180),float)
    contrast[20:41,20:150]=170
    contrast[40:45,30:65]=45
    contrast[27:33,35:41]=0  # white print, restored as texture by caller
    mask=np.uint8(contrast>28)*255
    result=clean_appearance_mask(mask,contrast)
    assert result[43,60]==0
    assert result[25,60]==255
    assert result[30,38]==0
    # Thin pale material isn't classified as shadow by absolute brightness.
    pale=np.zeros((50,160),float);pale[20:28,10:150]=35
    clean=clean_appearance_mask(np.uint8(pale>28)*255,pale)
    assert clean[23,50]==255
    assert np.count_nonzero(clean)>np.count_nonzero(pale)*.9


@pytest.mark.parametrize('mirror',[False,True])
def test_exact_gray_reference_excludes_tip_shadow(mirror):
    project=get_project('934325061be045f0abda9fb2d233aa46')
    if project is None:
        pytest.skip('Local gray reference unavailable')
    photo=next(p for p in project['photos'] if p['view']=='right')
    image=cv2.imread(photo['path'])
    if image is None:
        pytest.skip('Local gray reference unavailable')
    arm=extract_annotations(photo)['annotations'][0]
    if mirror:
        from copy import deepcopy
        arm=deepcopy(arm);image=cv2.flip(image,1)
        for point in arm['points']+list(arm['anchors'].values()):
            point['x']=image.shape[1]-1-point['x']
        arm['part']='temple-left'
    evidence,=opposite_arm_evidence(image,arm)
    from shapely.geometry import Polygon,Point
    polygon=Polygon([(p['x'],p['y']) for p in evidence['points']])
    x=lambda value:image.shape[1]-1-value if mirror else value
    assert not polygon.covers(Point(x(225),681))
    assert polygon.covers(Point(x(210),665))  # genuine black end retained
    assert evidence['contour_refinement']['max_boundary_displacement_px'] <= .851
