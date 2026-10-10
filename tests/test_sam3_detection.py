"""SAM 3 dispatch, provenance and boundary integration without gated weights."""
import io
import cv2
import numpy as np
import pytest
from PIL import Image
from fastapi.testclient import TestClient
from backend import store
from backend.main import app
from backend.reconstruction import sam3_detection as sam3
from backend.reconstruction.hybrid_detection import apply_hybrid
from backend.reconstruction.detection_report import detection_report
from test_hybrid_detection import fixture
from test_foundation_detection import side_regions


def test_sam3_api_uses_requested_model_and_preserves_saved_data(tmp_path,monkeypatch):
    image,scores,_,_=fixture()
    monkeypatch.setattr(store,'DATA',tmp_path)
    monkeypatch.setenv('EYEWEAR_START_WORKER','0')
    monkeypatch.setattr(sam3,'infer_lenses',lambda _: (scores,dict(model='sam3',device='cpu',prompt_source='automatic-eyewear-text-concepts')))
    raw=io.BytesIO();Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB)).save(raw,'PNG')
    with TestClient(app) as client:
        project=client.post('/api/projects',json={'name':'SAM3 CPU'}).json()
        url='/api/projects/'+project['id']
        project=client.post(url+'/photos',data={'view':'front'},files={'file':('front.png',raw.getvalue(),'image/png')}).json()
        before=store.get_project(project['id'])
        response=client.post(url+'/photos/'+project['photos'][0]['id']+'/extract?backend=sam3')
        assert response.status_code==200
        result=response.json();report=result['detection_report']['recognizer']
        assert report['model']=='sam3' and report['learned_model_active']
        assert report['backend']==('sam3-source-proposals' if report['proposals_used'] else 'geometric-heuristics')
        assert store.get_project(project['id'])==before
        assert all(not a['approved'] for a in result['annotations'])



def test_missing_checkpoint_never_claims_active_sam3(tmp_path,monkeypatch):
    monkeypatch.setenv('EYEWEAR_SAM3_MANIFEST',str(tmp_path/'missing.json'))
    image,_,photo,_=fixture()
    result=apply_hybrid(photo,image,image,dict(annotations=[],warnings=[]),1,1,model='sam3')
    recognizer=detection_report(photo,result)['recognizer']
    assert not recognizer['learned_model_active'] and not recognizer['proposals_used']
    assert 'weights are not ready' in recognizer['reason']


def test_only_lens_concepts_can_form_optical_pair(monkeypatch):
    image,_,_,lenses=fixture()
    masks=[]
    for polygon in lenses:
        mask=np.zeros(image.shape[:2],np.uint8)
        cv2.fillPoly(mask,[np.rint(polygon.exterior.coords).astype(np.int32)],1)
        masks.append(mask)
    monkeypatch.setattr(sam3,'infer_regions',lambda _: (np.array(masks),np.array([.95,.95]),np.array(['lens','frame']),{}))
    probabilities,detail=sam3.infer_lenses(image)
    assert not probabilities.any() and not detail['optical_pair_selected']


def test_side_dispatches_sam3_without_sam2(tmp_path,monkeypatch):
    from backend.reconstruction.engine import extract_annotations
    from backend.reconstruction import foundation_detection as foundation
    masks,scores,photo=side_regions()
    path=tmp_path/'side.png';cv2.imwrite(str(path),np.full((300,600,3),255,np.uint8));photo['path']=str(path)
    monkeypatch.setattr(sam3,'infer_side',lambda _: (masks,scores,dict(model='sam3',device='cpu')))
    monkeypatch.setattr(foundation,'infer_side',lambda _: pytest.fail('SAM2 must not run'))
    result=extract_annotations(photo,backend='sam3')
    assert result['hybrid_detection']['proposals_used']
    assert 'sam3' in result['annotations'][0]['source']
    assert not result['annotations'][0]['approved']


def test_sam3_source_labels_when_proposals_are_used(monkeypatch):
    image,scores,photo,_=fixture()
    monkeypatch.setattr(sam3,'infer_lenses',lambda _: (scores,dict(model='sam3',device='cpu')))
    result=apply_hybrid(photo,image,image,dict(annotations=[],warnings=[]),1,1,model='sam3')
    assert detection_report(photo,result)['recognizer']['backend']=='sam3-source-proposals'
    assert all('sam3' in a['source'] for a in result['annotations'] if a['part'].startswith('lens-'))


@pytest.mark.parametrize('role',['lens','invalid-hardware-role'])
def test_cpu_region_transport_validates_roles(tmp_path,monkeypatch,role):
    import subprocess
    from pathlib import Path
    manifest=tmp_path/'manifest.json';manifest.write_text('{}')
    python=tmp_path/'python';python.touch()
    monkeypatch.setenv('EYEWEAR_SAM3_MANIFEST',str(manifest))
    monkeypatch.setenv('EYEWEAR_SAM3_PYTHON',str(python))
    image=np.zeros((15,17,3),np.uint8)
    mask=np.zeros((15,17),np.uint8);mask[3:8,4:9]=1
    def run(command,**kwargs):
        assert command[command.index('--task')+1]=='lenses'
        output=Path(command[command.index('--output')+1])
        np.savez_compressed(output,masks=np.packbits(mask.ravel())[None],scores=np.array([.9]),roles=np.array([role]),shape=np.array(mask.shape))
        return subprocess.CompletedProcess(command,0,'{"model":"sam3","device":"cpu"}','')
    monkeypatch.setattr(sam3.subprocess,'run',run)
    if role=='lens':
        masks,scores,roles,runtime=sam3.infer_regions(image)
        assert np.array_equal(masks[0],mask) and roles.tolist()==['lens']
        assert runtime['device']=='cpu'
    else:
        with pytest.raises(ValueError,match='invalid region data'):
            sam3.infer_regions(image)
    assert sam3.INFERENCE_SLOT.acquire(timeout=.1)
    sam3.INFERENCE_SLOT.release()


def test_cpu_memory_stop_is_reported_and_releases_inference_slot(tmp_path,monkeypatch):
    import subprocess
    manifest=tmp_path/'manifest.json';manifest.write_text('{}')
    python=tmp_path/'python';python.touch()
    monkeypatch.setenv('EYEWEAR_SAM3_MANIFEST',str(manifest))
    monkeypatch.setenv('EYEWEAR_SAM3_PYTHON',str(python))
    monkeypatch.setattr(sam3.subprocess,'run',lambda command,**kw: subprocess.CompletedProcess(
        command,2,'','SAM 3 stopped: CPU process memory exceeded its 5.0 GiB limit.'))
    with pytest.raises(RuntimeError,match='CPU memory limit'):
        sam3.infer_regions(np.zeros((15,17,3),np.uint8))
    assert sam3.INFERENCE_SLOT.acquire(timeout=.1)
    sam3.INFERENCE_SLOT.release()
