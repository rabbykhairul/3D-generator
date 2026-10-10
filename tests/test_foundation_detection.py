import io
import subprocess
import cv2
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient
from backend import store
from backend.main import app
from backend.reconstruction import foundation_detection as foundation
from backend.reconstruction.detection_report import detection_report
from backend.reconstruction.hybrid_detection import apply_hybrid
from test_hybrid_detection import fixture
import pytest


def test_general_masks_assign_optical_pair_without_background_or_small_hardware():
    image,_,_,lenses=fixture()
    masks=[np.ones(image.shape[:2],np.uint8)]
    for polygon in lenses:
        mask=np.zeros(image.shape[:2],np.uint8)
        cv2.fillPoly(mask,[np.rint(polygon.exterior.coords).astype(np.int32)],1)
        masks.append(mask)
    hardware=np.zeros(image.shape[:2],np.uint8);cv2.circle(hardware,(290,170),5,1,-1)
    masks.append(hardware)
    probabilities,diagnostic=foundation.select_lenses(masks,[.99,.94,.95,.99],image.shape[:2])
    assert diagnostic['optical_pair_selected'] and diagnostic['optical_candidates']==2
    assert not probabilities[0,0] and not probabilities[170,290]
    assert probabilities[140,130] and probabilities[140,440]


def test_single_general_region_cannot_invent_a_second_lens():
    image,_,_,lenses=fixture()
    mask=np.zeros(image.shape[:2],np.uint8)
    cv2.fillPoly(mask,[np.rint(lenses[0].exterior.coords).astype(np.int32)],1)
    probabilities,diagnostic=foundation.select_lenses([mask],[.95],image.shape[:2])
    assert not diagnostic['optical_pair_selected'] and not probabilities.any()


def test_sam2_actual_backend_is_reported_and_suggestions_remain_unapproved(monkeypatch):
    image,scores,photo,_=fixture()
    monkeypatch.setattr(foundation,'infer_lenses',lambda _: (scores,dict(model='sam2.1-hiera-small',device='cpu',prompt_source='automatic-photo-grid')))
    result=apply_hybrid(photo,image,image,dict(annotations=[],warnings=[]),1,1,model='sam2')
    report=detection_report(photo,result)
    assert report['recognizer']['backend']=='sam2-source-proposals'
    assert report['recognizer']['model']=='sam2.1-hiera-small'
    assert report['recognizer']['prompt_source']=='automatic-photo-grid'
    assert all(not a['approved'] for a in result['annotations'])
    assert all('sam2' in a['source'] for a in result['annotations'] if a['part'].startswith('lens-'))


def test_sam2_api_is_explicit_and_does_not_change_saved_project(tmp_path,monkeypatch):
    image,scores,_,_=fixture()
    monkeypatch.setattr(store,'DATA',tmp_path)
    monkeypatch.setenv('EYEWEAR_START_WORKER','0')
    monkeypatch.setattr(foundation,'infer_lenses',lambda _: (scores,dict(model='sam2.1-hiera-small',device='cpu',prompt_source='coarse-lens-locator')))
    raw=io.BytesIO();Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB)).save(raw,'PNG')
    with TestClient(app) as client:
        project=client.post('/api/projects',json={'name':'Foundation inference'}).json()
        url='/api/projects/'+project['id']
        project=client.post(url+'/photos',data={'view':'front'},files={'file':('front.png',raw.getvalue(),'image/png')}).json()
        before=store.get_project(project['id'])
        result=client.post(url+'/photos/'+project['photos'][0]['id']+'/extract?backend=sam2').json()
        assert result['detection_report']['recognizer']['model']=='sam2.1-hiera-small'
        assert store.get_project(project['id'])==before
        assert all(not a['approved'] for a in result['annotations'])


def side_regions(two_branches=False):
    h,w=300,600
    arm=np.zeros((h,w),np.uint8)
    cv2.rectangle(arm,(40,125),(510,139),1,-1)
    if two_branches:
        cv2.rectangle(arm,(60,170),(260,183),1,-1)
        cv2.line(arm,(260,177),(350,132),1,10)
    lens=np.zeros_like(arm);cv2.ellipse(lens,(530,145),(18,95),0,0,360,1,-1)
    photo=dict(id='side',view='right',width=w,height=h,role='fitting')
    return [np.ones_like(arm),arm,lens],[.99,.95,.94],photo


def test_side_sam_selects_arm_rejects_background_and_preserves_inferred_attachment():
    masks,scores,photo=side_regions()
    result,diagnostic=foundation.select_side(masks,scores,photo,1,1)
    assert diagnostic['single_arm_selected']
    ann=result['annotations'][0]
    assert ann['part']=='temple-right' and not ann['approved']
    assert ann['anchors']['hinge']['x']>ann['anchors']['tip']['x']
    from backend.contracts import Annotation
    from backend.reconstruction.anatomy import audit_photo
    saved=Annotation(**ann).model_dump()
    nodes={n['id']:n for n in audit_photo(photo,[saved])['nodes']}
    assert nodes['temple-right']['state']=='observed'
    assert nodes['hinge-right']['state']=='inferred'
    assert nodes['temple-left']['state']=='not-observed-in-view'


def test_sam3_concept_scores_are_not_filtered_as_sam2_mask_iou(monkeypatch):
    from backend.reconstruction import sam3_detection, sam3_components
    masks,_,photo=side_regions()
    scores=[.99,.66,.88]
    candidate,_=foundation.select_side(masks,scores,photo,1,1)
    assert candidate is None  # SAM 2's .75 IoU policy remains unchanged.
    monkeypatch.setattr(sam3_detection,'infer_side',lambda _: (masks,scores,dict(model='sam3')))
    monkeypatch.setattr(sam3_components,'apply_side_components',lambda photo,image,result,sx,sy:result)
    result=foundation.apply_side(photo,np.zeros((300,600,3),np.uint8),dict(annotations=[],warnings=[]),1,1,model='sam3')
    assert result['hybrid_detection']['proposals_used']
    assert result['hybrid_detection']['concept_presence_score']==.66
    assert 'mask_predicted_iou' not in result['hybrid_detection']
    assert not result['annotations'][0]['approved']
    assert result['annotations'][0]['anchor_provenance']['hinge']=='inferred-frame-side-endpoint'


def test_side_sam_does_not_present_two_joined_arms_as_one():
    masks,scores,photo=side_regions(two_branches=True)
    result,diagnostic=foundation.select_side(masks,scores,photo,1,1)
    assert result is None and not diagnostic['single_arm_selected']


def test_side_sam_failure_is_explicit_and_existing_arm_is_retained(monkeypatch):
    masks,scores,photo=side_regions()
    baseline,_=foundation.select_side(masks,scores,photo,1,1)
    monkeypatch.setattr(foundation,'infer_side',lambda _: (masks,scores,dict(model='sam2.1-hiera-small')))
    result=foundation.apply_side(photo,np.zeros((300,600,3),np.uint8),baseline,1,1)
    assert result['hybrid_detection']['status']=='retained'
    assert result['annotations']==baseline['annotations']
    assert result['hybrid_detection']['inference_ran']
    def unavailable(_):
        raise RuntimeError('Checkpoint unavailable')
    monkeypatch.setattr(foundation,'infer_side',unavailable)
    result=foundation.apply_side(photo,np.zeros((300,600,3),np.uint8),baseline,1,1)
    assert result['hybrid_detection']['status']=='unavailable'
    assert not result['hybrid_detection']['inference_ran']
    assert result['annotations']==baseline['annotations']


def test_side_extract_dispatches_sam_and_remains_review_only(tmp_path,monkeypatch):
    from backend.reconstruction.engine import extract_annotations
    masks,scores,photo=side_regions()
    path=tmp_path/'side.png';cv2.imwrite(str(path),np.full((300,600,3),255,np.uint8))
    photo['path']=str(path)
    calls=[]
    def proposals(image):
        calls.append(image.shape)
        return masks,scores,dict(model='sam2.1-hiera-small')
    monkeypatch.setattr(foundation,'infer_side',proposals)
    result=extract_annotations(photo,backend='sam2')
    assert calls==[(300,600,3)]
    assert result['hybrid_detection']['proposals_used']
    assert result['annotations'][0]['part']=='temple-right'
    assert not result['annotations'][0]['approved']


def overlapping_regions(mirror_x=False,mirror_y=False,scale=1):
    masks,scores,photo=side_regions()
    mask=np.zeros_like(masks[1])
    cv2.rectangle(mask,(40,125),(510,139),1,-1)
    cv2.rectangle(mask,(230,180),(300,193),1,-1)
    cv2.line(mask,(300,186),(390,132),1,10)
    masks[1]=mask
    for i,m in enumerate(masks):
        if mirror_x:m=cv2.flip(m,1)
        if mirror_y:m=cv2.flip(m,0)
        if scale!=1:m=cv2.resize(m,None,fx=scale,fy=scale,interpolation=cv2.INTER_NEAREST)
        masks[i]=m
    photo.update(width=masks[0].shape[1],height=masks[0].shape[0])
    return masks,scores,photo


@pytest.mark.parametrize('mirror_x',[False,True])
@pytest.mark.parametrize('mirror_y',[False,True])
@pytest.mark.parametrize('scale',[1,2])
def test_sam_overlap_completion_selects_distinct_longer_path_and_marks_inference(mirror_x,mirror_y,scale):
    from backend.reconstruction.anatomy import audit_photo,attach_evidence
    from backend.contracts import Annotation
    from shapely.geometry import Polygon,Point
    masks,scores,photo=overlapping_regions(mirror_x,mirror_y,scale)
    result,diagnostic=foundation.select_side(masks,scores,photo,1,1)
    assert result and diagnostic['overlap_completion']
    assert diagnostic['source_mask_support']>=.95
    assert diagnostic['branch_length_ratio']<.9
    ann=result['annotations'][0]
    assert 'occlusion-completion' in ann['source'] and not ann['approved']
    x=(600-1-230 if mirror_x else 230)*scale
    y=(300-1-186 if mirror_y else 186)*scale
    poly=Polygon([(p['x'],p['y']) for p in ann['points']])
    assert not poly.covers(Point(x,y)), 'The second tip must not be included'
    saved=Annotation(**ann).model_dump()
    report=audit_photo(photo,[saved])
    assert not report['blocking']
    nodes={n['id']:n for n in report['nodes']}
    assert nodes['temple-right']['state']=='inferred'
    assert nodes['hinge-right']['state']=='inferred'
    result=attach_evidence(photo,result,photo['width'],photo['height'])
    assert result['annotations'][0]['evidence']['state']=='inferred'


def test_overlap_without_identifiable_frame_or_distinct_longer_tip_remains_withheld():
    masks,scores,photo=overlapping_regions()
    assert foundation.select_side(masks[:2],scores[:2],photo,1,1)[0] is None
    mask=np.zeros_like(masks[1])
    cv2.rectangle(mask,(40,125),(510,139),1,-1)
    cv2.rectangle(mask,(40,180),(300,193),1,-1)
    cv2.line(mask,(300,186),(390,132),1,10)
    masks[1]=mask
    assert foundation.select_side(masks,scores,photo,1,1)[0] is None


def test_sam_timeout_reports_a_readable_resource_limit_without_command_paths(tmp_path,monkeypatch):
    python=tmp_path/'python';python.touch()
    manifest=tmp_path/'manifest.json';manifest.write_text('{}')
    monkeypatch.setenv('EYEWEAR_SAM2_PYTHON',str(python))
    monkeypatch.setenv('EYEWEAR_SAM2_MANIFEST',str(manifest))
    def timeout(command,**kwargs):
        raise subprocess.TimeoutExpired(command,kwargs['timeout'])
    monkeypatch.setattr(foundation.subprocess,'run',timeout)
    with pytest.raises(RuntimeError,match='120-second CPU limit') as caught:
        foundation._infer(np.full((100,100,3),255,np.uint8),task='side')
    assert str(tmp_path) not in str(caught.value)
