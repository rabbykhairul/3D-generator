"""New black-brow/pale-metal product: extraction to an exported GLB.

Uses the local uploaded reference, never writes its project or approvals.
"""
from pathlib import Path
import io

import numpy as np
import pytest
import trimesh
from fastapi.testclient import TestClient
from shapely.geometry import Point, Polygon

from backend.reconstruction import extract_annotations, reconstruct
from backend.reconstruction.render_mesh import welded_geometry

ROOT = Path(__file__).resolve().parents[1]
PHOTOS = ROOT / 'data/projects/ce81f4ffd03e4b8cac759b32ec5ee650/photos'


def new_photo(view):
    name = {'front': '0614da10abb84db08c7ed2110a9f0993',
            'right': '071de32f5a964f5796f996d0fc31524c'}[view]
    path = PHOTOS / (name + '.png')
    if not path.exists():
        pytest.skip('New local product photograph not installed')
    return dict(id=name, path=str(path), view=view, role='fitting', width=1080, height=1080)


def test_new_front_excludes_arms_pads_and_shadow_tabs():
    result = extract_annotations(new_photo('front'))
    polygons = {a['part']: Polygon([(p['x'], p['y']) for p in a['points']])
                for a in result['annotations']}
    assert all(p.is_valid for p in polygons.values())
    assert all(not a['approved'] for a in result['annotations'])
    left, right, outer = (polygons[p] for p in ('lens-left','lens-right','front'))
    assert outer.contains(left) and outer.contains(right) and left.disjoint(right)
    # Behind-lens arm pixels must be inside the lens aperture, NOT frame solid.
    assert left.contains(Point(235, 615)) and right.contains(Point(847, 607))
    assert left.area / right.area > .96
    # Tiny shadow-derived ledges in the previous green outline are not material.
    assert not outer.covers(Point(226, 732.9))
    assert not outer.covers(Point(465, 626))  # protruding nose-pad edge, not rim
    # Genuine thin lower metal rim survives cleanup.
    rim = outer.difference(left.union(right))
    assert rim.covers(Point(290, 730)) and rim.covers(Point(794, 730))


def test_new_front_generation_and_glb_reload(tmp_path):
    photo = new_photo('front')
    result = extract_annotations(photo)
    parts = {a['part']: Polygon([(p['x'],p['y']) for p in a['points']]) for a in result['annotations']}
    f, l, r = (parts[p].bounds for p in ('front','lens-left','lens-right'))
    scale = 140 / (f[2]-f[0])
    values = dict(frame_width=140, temple_length=140,
                  lens_width=((l[2]-l[0])+(r[2]-r[0]))/2*scale,
                  lens_height=((l[3]-l[1])+(r[3]-r[1]))/2*scale,
                  bridge_width=(r[0]-l[2])*scale)
    snapshot = dict(category='opaque-full-rim', photos=[photo],
        measurements={k:dict(value=v,source='inferred') for k,v in values.items()},
        annotations=[dict(a,approved=True,source='merchant-reviewed:'+a['source']) for a in result['annotations']],
        material=dict(color='#353336',opacity=1,lens_opacity=.08))
    metadata = reconstruct(snapshot, tmp_path/'model')
    assert metadata['checks']['front_parts_watertight']
    assert metadata['checks']['consistent_outward_normals']
    scene = trimesh.load(tmp_path/'model/model.glb',force='scene')
    # Both separate clear pads are visible in this reference. The general
    # nasal-component route now preserves them instead of silently omitting.
    assert set(scene.geometry) == {'front_frame','lens_left','lens_right','nose_pad_left','nose_pad_right'}
    assert scene.geometry['front_frame'].visual.material.baseColorTexture is not None


def test_new_product_api_extract_approve_generate_download(tmp_path, monkeypatch):
    from backend import store
    from backend.main import app
    from backend.worker import process_next
    originals = [new_photo(view) for view in ('front','right')]
    monkeypatch.setattr(store, 'DATA', tmp_path/'isolated-api')
    monkeypatch.setenv('EYEWEAR_START_WORKER', '0')
    with TestClient(app) as client:
        project = client.post('/api/projects',json={'name':'New outline QA','category':'opaque-full-rim'}).json()
        url = '/api/projects/'+project['id']
        annotations = []
        for photo in originals:
            upload = client.post(url+'/photos',data={'view':photo['view'],'role':'fitting'},
                                 files={'file':('photo.png',Path(photo['path']).read_bytes(),'image/png')})
            assert upload.status_code == 200
            photo_id = upload.json()['photos'][-1]['id']
            result = client.post(url+'/photos/'+photo_id+'/extract')
            assert result.status_code == 200
            assert all(not a['approved'] for a in result.json()['annotations'])
            annotations.extend(result.json()['annotations'])
        # Extraction must not have changed saved approvals or automatically generated.
        assert not client.get(url).json()['annotations']
        assert {a['part'] for a in annotations} == {'front','lens-left','lens-right','temple-right','nose-pad-left','nose-pad-right'}
        parts = {a['part']: Polygon([(p['x'],p['y']) for p in a['points']]) for a in annotations}
        front, left, right = (parts[k].bounds for k in ('front','lens-left','lens-right'))
        scale = 140 / (front[2]-front[0])
        values = dict(frame_width=140,temple_length=140,
                      lens_width=((left[2]-left[0])+(right[2]-right[0]))/2*scale,
                      lens_height=((left[3]-left[1])+(right[3]-right[1]))/2*scale,
                      bridge_width=(right[0]-left[2])*scale)
        approved = [dict(a,approved=True,source='merchant-reviewed:'+a['source']) for a in annotations]
        patch = client.patch(url,json={'annotations':approved,
            'measurements':{k:dict(value=v,source='inferred') for k,v in values.items()}})
        assert patch.status_code == 200, patch.text
        response = client.post(url+'/jobs')
        assert response.status_code == 202, response.text
        assert process_next()
        job = client.get('/api/jobs/'+response.json()['id']).json()
        assert job['status'] == 'completed', job
        metadata = client.get(job['artifact']['metadata_url']).json()
        joint = metadata['fit']['joints']['right']
        assert joint['surface_part'] == 'front_frame' and joint['arm_root_contains_joint']
        assert metadata['checks']['consistent_outward_normals']
        glb = client.get(job['artifact']['model_url'])
        assert glb.status_code == 200
        scene = trimesh.load(io.BytesIO(glb.content),file_type='glb',force='scene',process=False)
        assert set(scene.geometry) == {'front_frame','lens_left','lens_right','temple_right','hinge_right','nose_pad_left','nose_pad_right'}
        for name,mesh in scene.geometry.items():
            geometry = welded_geometry(mesh)
            assert geometry.is_watertight and geometry.is_winding_consistent, name
            assert np.isfinite(mesh.vertices).all() and np.isfinite(mesh.vertex_normals).all()
            if name in ('front_frame','temple_right'):
                assert mesh.visual.material.baseColorTexture is not None
        assert scene.geometry['lens_left'].visual.material.alphaMode == 'BLEND'
        assert not client.get(url).json()['stale']
