"""Photo suggestion, merchant approval, and export in an isolated API store."""
import io
from pathlib import Path

import pytest
import trimesh
from fastapi.testclient import TestClient
from shapely.geometry import Polygon

from backend import store
from backend.main import app
from backend.reconstruction.render_mesh import welded_geometry
from backend.worker import process_next


CLIPBOARD_FRONT = Path('/var/folders/nq/qjt_x3410kv3nm8r42j4njmw0000gn/T/codex-clipboard-BuRoOI.png')
SAVED_FRONT = (Path(__file__).resolve().parents[1] / 'data/projects'
               / '3ca4f944b2214b72988db3bde88e0602/photos'
               / '02f3fa42c9974879bf1ab62b8cc1e384.png')


def approximate_measurements(annotations):
    bounds = {a['part']: Polygon([(p['x'], p['y']) for p in a['points']]).bounds
              for a in annotations}
    front, left, right = (bounds[k] for k in ('front', 'lens-left', 'lens-right'))
    scale = 140 / (front[2] - front[0])
    values = {
        'frame_width': 140,
        'temple_length': 140,
        'lens_width': ((left[2]-left[0]) + (right[2]-right[0])) / 2 * scale,
        'lens_height': ((left[3]-left[1]) + (right[3]-right[1])) / 2 * scale,
        'bridge_width': (right[0]-left[2]) * scale,
    }
    assert all(value > 0 for value in values.values())
    return {key: {'value': value, 'source': 'inferred'} for key, value in values.items()}


def test_auto_semi_rimless_suggestion_requires_approval_and_exports_closed_front(tmp_path, monkeypatch):
    front = CLIPBOARD_FRONT if CLIPBOARD_FRONT.exists() else SAVED_FRONT
    if not front.exists():
        pytest.skip('Local semi-rimless reference photo not installed')
    monkeypatch.setattr(store, 'DATA', tmp_path / 'isolated-auto-construction')
    monkeypatch.setenv('EYEWEAR_START_WORKER', '0')
    with TestClient(app) as client:
        created = client.post('/api/projects', json={
            'name': 'Isolated auto-construction regression', 'category': 'opaque-full-rim'})
        assert created.status_code == 201, created.text
        url = f"/api/projects/{created.json()['id']}"
        uploaded = client.post(url + '/photos', data={'view': 'front', 'role': 'fitting'},
                               files={'file': ('front.png', front.read_bytes(), 'image/png')})
        assert uploaded.status_code == 200, uploaded.text
        photo_id = uploaded.json()['photos'][-1]['id']
        before = client.get(url).json()
        assert before['category'] == 'opaque-full-rim'
        assert not any(a['approved'] for a in before['annotations'])

        extracted = client.post(url + '/photos/' + photo_id + '/extract')
        assert extracted.status_code == 200, extracted.text
        suggestion = extracted.json()['construction']
        assert suggestion['category'] == 'semi-rimless', suggestion
        assert suggestion['confidence'] in ('high', 'medium')
        annotations = extracted.json()['annotations']
        assert {'front', 'lens-left', 'lens-right'} <= {a['part'] for a in annotations}
        assert all(not a['approved'] for a in annotations)
        assert all(len(a['points']) >= 3 for a in annotations)
        after_analysis = client.get(url).json()
        assert after_analysis['revision'] == before['revision']
        assert after_analysis['category'] == before['category']
        assert after_analysis['annotations'] == before['annotations']

        approved = [dict(a, approved=True, source='merchant-reviewed:' + a['source'])
                    for a in annotations]
        saved = client.patch(url, json={
            'category': suggestion['category'],
            'annotations': approved,
            'measurements': approximate_measurements(annotations),
        })
        assert saved.status_code == 200, saved.text
        assert saved.json()['category'] == 'semi-rimless'
        assert all(a['approved'] for a in saved.json()['annotations'])
        queued = client.post(url + '/jobs')
        assert queued.status_code == 202, queued.text
        assert process_next()
        job = client.get('/api/jobs/' + queued.json()['id']).json()
        assert job['status'] == 'completed', job
        metadata = client.get(job['artifact']['metadata_url']).json()
        assert metadata['checks']['front_parts_watertight']
        assert metadata['checks']['consistent_outward_normals']
        response = client.get(job['artifact']['model_url'])
        assert response.status_code == 200
        scene = trimesh.load(io.BytesIO(response.content), file_type='glb',
                             force='scene', process=False)
        assert not any(name.startswith('lower_wire') for name in scene.geometry)
        assert {'upper_frame_0', 'lens_left', 'lens_right'} <= set(scene.geometry)
        for name in ('upper_frame_0', 'lens_left', 'lens_right'):
            geometry = welded_geometry(scene.geometry[name])
            assert geometry.is_watertight and geometry.is_winding_consistent, name
        for name in ('lens_left', 'lens_right'):
            lens = scene.geometry[name]
            assert lens.visual.material.alphaMode == 'BLEND'
            assert lens.visual.material.baseColorTexture is None
            assert 0 < lens.visual.material.baseColorFactor[3] < 255
            # Cut-edge bevels survive export, not just two planar caps.
            normal_z = abs(lens.face_normals[:, 2])
            assert ((normal_z > .1) & (normal_z < .9)).any()
