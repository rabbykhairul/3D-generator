import io
import json

from fastapi.testclient import TestClient
from PIL import Image

from backend import store
from backend.main import app
from backend.reconstruction.detection_report import detection_report


def test_empty_detection_is_not_absence_or_fidelity():
    photo = dict(id='p', view='front', width=600, height=300)
    report = detection_report(photo, dict(annotations=[], warnings=['not found']))
    assert report['status'] == 'needs-correction'
    assert report['missing_parts'] == ['lens-left', 'lens-right']
    assert not report['automatic_acceptance'] and not report['fidelity_verified']
    assert not report['recognizer']['learned_model_active']
    assert report['issues']
    side = detection_report(dict(photo, view='left'), dict(annotations=[]))
    assert side['missing_parts'] == ['temple-left']
    assert detection_report(dict(photo, view='inner'), dict(annotations=[]))['status'] == 'reference-only'


def test_report_storage_is_bounded_and_does_not_mutate_project(tmp_path, monkeypatch):
    monkeypatch.setattr(store, 'DATA', tmp_path)
    store.initialize()
    project = dict(id='project', photos=[], updated_at='before', revision=7)
    store.save_project(project)
    for index in range(103):
        store.record_detection('project', 'photo', dict(index=index))
    runs = store.detection_history('project', {'photo'})
    assert len(runs) == 100 and runs[0]['index'] == 102 and runs[-1]['index'] == 3
    assert store.detection_history('another-project', {'photo'}) == []
    assert store.detection_history('project', set()) == []
    assert store.get_project('project') == project


def test_extract_persists_failure_without_approval_or_revision_change(tmp_path, monkeypatch):
    from backend import reconstruction
    monkeypatch.setattr(store, 'DATA', tmp_path)
    monkeypatch.setenv('EYEWEAR_START_WORKER', '0')
    def fail(photo):
        raise reconstruction.NeedsCorrection('No continuous boundary')
    monkeypatch.setattr(reconstruction, 'extract_annotations', fail)
    raw = io.BytesIO()
    Image.new('RGB', (600, 300), 'white').save(raw, 'PNG')
    with TestClient(app) as client:
        project = client.post('/api/projects', json={'name':'Unseen','category':'thin-metal'}).json()
        url = '/api/projects/' + project['id']
        project = client.post(url+'/photos', data={'view':'front'},
                              files={'file':('front.png', raw.getvalue(), 'image/png')}).json()
        photo = project['photos'][0]
        before = store.get_project(project['id'])
        response = client.post(url+'/photos/'+photo['id']+'/extract')
        assert response.status_code == 200
        result = response.json()
        assert result['annotations'] == []
        report = result['detection_report']
        assert report['status'] == 'needs-correction'
        assert store.get_project(project['id']) == before
        assert client.get(url+'/detection-runs').json() == [report]
        assert client.get('/api/projects/missing/detection-runs').status_code == 404
        assert 'path' not in json.dumps(report)
        client.delete(url+'/photos/'+photo['id'])
        assert client.get(url+'/detection-runs').json() == []
