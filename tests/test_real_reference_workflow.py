"""Optional local product regression; all writes use pytest's temporary database.

This exercises upload -> saved approvals -> worker -> download -> GLB reload.
It does not stand in for visual browser/AR verification or fresh auto-segmentation.
"""
import io
import json
from pathlib import Path

import numpy as np
import pytest
import trimesh
from fastapi.testclient import TestClient

from backend import store
from backend.engine_version import ENGINE_VERSION
from backend.main import app
from backend.worker import process_next


def test_actual_reviewed_product_upload_generate_and_reload(tmp_path, monkeypatch):
    baseline = Path(__file__).resolve().parents[1] / 'artifacts/fidelity/baseline-project.json'
    if not baseline.exists():
        pytest.skip('Local product snapshot not installed')
    snapshot = json.loads(baseline.read_text())
    if any(not Path(photo['path']).exists() for photo in snapshot['photos']):
        pytest.skip('Local product photos not installed')
    monkeypatch.setattr(store, 'DATA', tmp_path / 'isolated-real-product')
    monkeypatch.setenv('EYEWEAR_START_WORKER', '0')
    with TestClient(app) as client:
        response = client.post('/api/projects', json={
            'name': 'Isolated real-product regression', 'category': snapshot['category']})
        assert response.status_code == 201
        project_id = response.json()['id']
        url = f'/api/projects/{project_id}'
        remap = {}
        for photo in snapshot['photos']:
            response = client.post(url+'/photos', data={'view': photo['view'], 'role': photo['role']},
                                   files={'file': ('reference.png', Path(photo['path']).read_bytes(), 'image/png')})
            assert response.status_code == 200, response.text
            remap[photo['id']] = response.json()['photos'][-1]['id']
        annotations = [dict(a, photo_id=remap[a['photo_id']]) for a in snapshot['annotations']]
        response = client.patch(url, json={'measurements': snapshot['measurements'],
                                          'material': snapshot['material'], 'annotations': annotations})
        assert response.status_code == 200, response.text
        response = client.post(url+'/jobs')
        assert response.status_code == 202, response.text
        assert process_next()
        job = client.get('/api/jobs/'+response.json()['id']).json()
        assert job['status'] == 'completed', job
        metadata = client.get(job['artifact']['metadata_url']).json()
        assert metadata['engine_version'] == ENGINE_VERSION
        assert metadata['checks']['finite_vertices']
        assert metadata['checks']['front_parts_watertight']
        assert metadata['checks']['consistent_outward_normals']
        assert not client.get(url).json()['stale']
        model_response = client.get(job['artifact']['model_url'])
        assert model_response.status_code == 200
        scene = trimesh.load(io.BytesIO(model_response.content), file_type='glb', force='scene', process=False)
        assert set(scene.geometry) == {'front_frame', 'lens_left', 'lens_right',
                                       'temple_left', 'temple_right', 'hinge_left', 'hinge_right'}
        for side in ('left', 'right'):
            assert metadata['fit']['joints'][side]['arm_root_contains_joint']
            assert metadata['fit']['side'][side]['edge_bevel_mm'] > 0
            arm = scene.geometry[f'temple_{side}']
            assert arm.visual.material.baseColorTexture is not None
            assert np.isfinite(arm.vertices).all()
            lens = scene.geometry[f'lens_{side}'].visual.material
            assert lens.baseColorTexture is None
            from backend.reconstruction.engine import _color
            assert np.array_equal(lens.baseColorFactor[:3], _color('#b5d9e5')[:3])
            assert abs(lens.baseColorFactor[3]/255 - snapshot['material']['lens_opacity']) <= 1/255
        frame_texture = np.asarray(scene.geometry['front_frame'].visual.material.baseColorTexture)
        assert frame_texture[..., :3].mean() < 220, 'Reloaded frame texture must not be a white fallback'
