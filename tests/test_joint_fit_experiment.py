from copy import deepcopy
import numpy as np
import pytest
import trimesh

from backend.reconstruction.joint_fit import fit_correspondences, project_front, fit_project_front
from backend.reconstruction.export_review import validate_export
from backend.reconstruction.source_zoom import zoom_diagnostics
from backend.reconstruction.engine import reconstruct
from test_reconstruction import snapshot, rectangle


def correspondences(sag=12.,count=80):
    rng=np.random.default_rng(9)
    xy=rng.uniform([-68,-22],[68,22],(count,2))
    result=[]
    for i,yaw in enumerate([.6,-.5]):
        camera=np.array([yaw,.12,.03,np.log(5),600,300])
        result.append({'xy':xy,'uv':project_front(xy,sag,70,camera)+rng.normal(0,.1,xy.shape),
                       'photo_id':str(i),'view':'front-left' if i==0 else 'front-right'})
    return result

@pytest.mark.parametrize('views',[1,2])
def test_joint_fit_recovers_known_curvature_and_heldout_camera_projection(views):
    result=fit_correspondences(correspondences()[:views],70,4)
    assert result['applied']
    assert result['fitted_front_curve_mm']==pytest.approx(12,abs=.3)
    assert result['withheld_median_error_px']<.3
    assert result['relative_error_improvement']>.8
    assert all(v['training_matches']+v['withheld_matches']==v['matches'] for v in result['views'])


def test_fit_withholds_underconstrained_and_already_matching_evidence():
    views=correspondences();views[0]['xy'][:,0]=1
    assert not fit_correspondences(views[:1],70,4)['applied']
    assert not fit_correspondences(correspondences(sag=4),70,4)['applied']
    assert not fit_correspondences([],70,4)['applied']


def test_fit_withholds_noisy_or_mismatched_evidence():
    views=correspondences();views[0]['uv']=np.random.default_rng(10).uniform(0,1000,(80,2))
    assert not fit_correspondences(views,70,4)['applied']


def test_withheld_oblique_photo_is_not_used(tmp_path):
    data=snapshot(tmp_path,'opaque-full-rim');f=data['photos'][0]
    photos=[f,{**f,'id':'holdout','view':'front-left','role':'validation'}]
    result=fit_project_front(photos,data['annotations'],f,(200,80),(.4,.4),140,4)
    assert not result['applied'] and not result.get('input_photo_ids')


def test_experiment_exports_diagnostics_without_changing_missing_view_geometry(tmp_path):
    base=snapshot(tmp_path,'opaque-full-rim')
    base['annotations'].append({'photo_id':'front','part':'nose-pad-left','approved':True,
        'points':rectangle(187,82,195,105),'source':'manual'})
    original=reconstruct(base,tmp_path/'base')
    trial=deepcopy(base);trial['reconstruction_settings']={'joint_front_fit':True}
    result=reconstruct(trial,tmp_path/'trial')
    assert result['fit']['joint_front']['status']=='withheld'
    assert result['dimensions_mm']['front_curve']==original['dimensions_mm']['front_curve']
    assert result['export_review']['component_preservation_passed']
    assert result['export_review']['independent_fidelity_verified'] is False
    assert (tmp_path/'trial/fit-front.png').exists()
    assert (tmp_path/'trial/fit-zoom-front.png').exists()
    assert result['export_review']['front_trace_iou']>.9
    assert 'nose_pad_left' in trimesh.load(tmp_path/'trial/model.glb',force='scene').geometry


def test_explicit_curvature_is_never_overwritten(tmp_path,monkeypatch):
    data=snapshot(tmp_path,'opaque-full-rim');data['reconstruction_settings']={'joint_front_fit':True}
    data['measurements']['front_curve']={'value':9,'source':'measured'}
    monkeypatch.setattr('backend.reconstruction.joint_fit.fit_project_front',lambda *args:pytest.fail('Measured curvature must not be fitted'))
    result=reconstruct(data,tmp_path/'measured')
    assert result['dimensions_mm']['front_curve']==9
    assert 'measurement' in result['fit']['joint_front']['reason']


def test_zoom_inspects_native_crop_without_mutating_outline(tmp_path):
    data=snapshot(tmp_path,'opaque-full-rim');before=deepcopy(data['annotations'])
    result=zoom_diagnostics(data['photos'][0],data['annotations'],tmp_path/'zooms.png')
    assert data['annotations']==before
    assert result['outline_changed'] is False and result['adds_source_detail'] is False
    assert result['regions'] and all(r['native_pixels_inspected'] for r in result['regions'])
    assert (tmp_path/'zooms.png').exists()

from test_api import client

def test_experiment_setting_round_trips_and_review_images_are_served(client):
    health=client.get('/api/health').json()
    assert health['engine_version']>=44 and health['joint_front_fit_available']
    project=client.post('/api/samples/synthetic-opaque-full-rim/import').json()
    url=f"/api/projects/{project['id']}"
    patched=client.patch(url,json={'reconstruction_settings':{'joint_front_fit':True}})
    assert patched.status_code==200 and patched.json()['reconstruction_settings']['joint_front_fit']
    assert patched.json()['revision']>project['revision']
    job=client.post(url+'/jobs').json()
    from backend.worker import process_next
    assert process_next()
    completed=client.get('/api/jobs/'+job['id']).json()
    assert completed['status']=='completed',completed
    review=completed['artifact']['metadata']['export_review']
    assert review['component_preservation_passed']
    for image in review['images']:
        response=client.get(f"/api/artifacts/{job['id']}/{image['file']}")
        assert response.status_code==200 and response.headers['content-type']=='image/png'
        assert response.content.startswith(b'\x89PNG')
    assert client.get(f"/api/artifacts/{job['id']}/unlisted.png").status_code==404


def test_accepted_fit_changes_serialized_front_geometry(tmp_path,monkeypatch):
    data=snapshot(tmp_path,'opaque-full-rim')
    before=reconstruct(data,tmp_path/'original')
    result=fit_correspondences(correspondences(),70,4)
    assert result['applied']
    monkeypatch.setattr('backend.reconstruction.joint_fit.fit_project_front',lambda *args:deepcopy(result))
    data['reconstruction_settings']={'joint_front_fit':True}
    after=reconstruct(data,tmp_path/'fitted')
    old=trimesh.load(tmp_path/'original/model.glb',force='scene').geometry['front_frame']
    new=trimesh.load(tmp_path/'fitted/model.glb',force='scene').geometry['front_frame']
    assert after['dimensions_mm']['front_curve']==pytest.approx(12,abs=.3)
    assert new.bounds[0,2]<old.bounds[0,2]-.007
    assert np.allclose(new.bounds[:,:2],old.bounds[:,:2])
    assert after['export_review']['component_preservation_passed']
