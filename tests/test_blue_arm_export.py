"""Exercise blue pp extraction through generation and an exported GLB reload."""
import copy
import io
from pathlib import Path

import numpy as np
import pytest
import trimesh
from PIL import Image
from fastapi.testclient import TestClient

from backend import store
from backend.main import app
from backend.worker import process_next
from backend.reconstruction.render_mesh import welded_geometry


def test_blue_product_side_api_to_connected_export(tmp_path, monkeypatch):
    original = store.get_project('c8e4014e30044160ace24fe171a57b5e')
    if original is None:
        pytest.skip('Local blue product not installed')
    original = copy.deepcopy(original)
    photos = [p for p in original['photos'] if p['view'] in ('front','right')]
    if any(not Path(p['path']).exists() for p in photos):
        pytest.skip('Local blue photos not installed')
    monkeypatch.setattr(store,'DATA',tmp_path/'blue-api')
    monkeypatch.setenv('EYEWEAR_START_WORKER','0')
    with TestClient(app) as client:
        project = client.post('/api/projects',json={'name':'Blue isolated QA','category':original['category']}).json()
        url = '/api/projects/'+project['id']
        remap = {}
        for photo in photos:
            r = client.post(url+'/photos',data={'view':photo['view'],'role':'fitting'},
                            files={'file':('photo.png',Path(photo['path']).read_bytes(),'image/png')})
            assert r.status_code == 200,r.text
            remap[photo['id']] = r.json()['photos'][-1]['id']
        side = next(p for p in photos if p['view']=='right')
        r = client.post(url+'/photos/'+remap[side['id']]+'/extract')
        assert r.status_code == 200
        ann = r.json()['annotations'][0]
        assert not ann['approved'] and ann['part']=='temple-right'
        assert 860 < ann['anchors']['hinge']['x'] < 878
        annotations = [dict(a,photo_id=remap[a['photo_id']]) for a in original['annotations']
                       if not a['part'].startswith('temple-') and a['photo_id'] in remap]
        annotations += [dict(ann,approved=True,source='merchant-reviewed-side:'+ann['source']),
                        dict(ann,part='temple-left',approved=True,source='explicit-mirrored-side-assumption:'+ann['source'])]
        r = client.patch(url,json={'annotations':annotations,'measurements':original['measurements'],'material':original['material']})
        assert r.status_code == 200,r.text
        r = client.post(url+'/jobs')
        assert r.status_code == 202,r.text
        assert process_next()
        job = client.get('/api/jobs/'+r.json()['id']).json()
        assert job['status']=='completed',job
        meta = client.get(job['artifact']['metadata_url']).json()
        for side in ('left','right'):
            assert meta['fit']['joints'][side]['arm_root_contains_joint']
            assert meta['fit']['joints'][side]['surface_part']=='front_frame'
        assert meta['checks']['consistent_outward_normals']
        exported = client.get(job['artifact']['model_url'])
        assert exported.status_code==200
        scene = trimesh.load(io.BytesIO(exported.content),file_type='glb',force='scene',process=False)
        assert len(scene.geometry)==7
        for name,mesh in scene.geometry.items():
            welded = welded_geometry(mesh)
            assert welded.is_watertight and welded.is_winding_consistent,name
            assert np.isfinite(mesh.vertices).all() and np.isfinite(mesh.vertex_normals).all()
        for side in ('left','right'):
            assert scene.geometry['temple_'+side].visual.material.baseColorTexture is not None
        original_side = next(p for p in photos if p['view']=='right')
        source = np.asarray(Image.open(original_side['path']).convert('RGB'))[510:535,815:850]
        texture = np.asarray(scene.geometry['temple_right'].visual.material.baseColorTexture)[510:535,815:850,:3]
        printing = np.all(source > 220,axis=2)
        assert printing.sum() > 200
        assert np.mean(np.all(texture[printing] > 220,axis=1)) > .95, 'White logo stays appearance, not a geometric notch or dark replacement'
