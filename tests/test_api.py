import io
import json
from pathlib import Path

import pytest
import trimesh
import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from backend import store
from backend.engine_version import ENGINE_VERSION
from backend.main import app
from backend.worker import process_next

@pytest.fixture
def client(tmp_path,monkeypatch):
    monkeypatch.setattr(store,'DATA',tmp_path/'data')
    monkeypatch.setenv('EYEWEAR_START_WORKER','0')
    with TestClient(app) as client:
        yield client

def png():
    raw=io.BytesIO();Image.new('RGB',(600,300),'white').save(raw,'PNG');return raw.getvalue()

def test_anatomy_review_is_read_only_and_detection_evidence_survives_patch(client):
    schema=client.get('/api/anatomy/schema').json()
    assert schema['schema_version']==1 and 'bridge-hardware' in schema['parts']
    assert schema['optical_topologies']==['separate-left-right','single-shield']
    project=client.post('/api/samples/synthetic-opaque-full-rim/import').json()
    url=f'/api/projects/{project["id"]}'
    photo=next(p for p in project['photos'] if p['view']=='front')
    annotations=[a for a in project['annotations'] if a['photo_id']==photo['id']]
    assert annotations
    annotations[0]['evidence']={'schema_version':1,'state':'observed',
                              'method':'geometric-anatomy-audit','review_required':True,'notes':[]}
    result=client.post(url+'/photos/'+photo['id']+'/anatomy',json={'annotations':annotations,'category':project['category']})
    assert result.status_code==200 and result.json()['schema_version']==1
    assert client.get(url).json()['revision']==project['revision']
    patched=client.patch(url,json={'annotations':annotations}).json()
    assert patched['annotations'][0]['evidence']['state']=='observed'
    invalid=json.loads(json.dumps(annotations));invalid[0]['points'][0]['x']=-10
    result=client.post(url+'/photos/'+photo['id']+'/anatomy',json={'annotations':invalid})
    assert result.status_code==200 and result.json()['blocking']
    invalid[0]['photo_id']='another-photo'
    assert client.post(url+'/photos/'+photo['id']+'/anatomy',json={'annotations':invalid}).status_code==422
    assert client.post(url+'/photos/missing/anatomy',json={'annotations':[]}).status_code==404


def test_engine_upgrade_marks_export_stale_without_discarding_approvals(client):
    project=client.post('/api/samples/synthetic-opaque-full-rim/import').json()
    job=client.post(f'/api/projects/{project["id"]}/jobs').json()
    assert process_next()
    completed=client.get('/api/jobs/'+job['id']).json()
    assert completed['status']=='completed'
    current=store.get_project(project['id'])
    assert current['artifact']['metadata']['engine_version']==ENGINE_VERSION
    assert not store.public_project(current)['stale']
    current['artifact']['metadata']['engine_version']=ENGINE_VERSION-1
    previous_annotations=json.loads(json.dumps(current['annotations']))
    assert store.public_project(current)['stale']
    assert current['annotations']==previous_annotations

def test_relabel_side_photo_and_overall_inner_reference(client):
    p=client.post('/api/samples/synthetic-opaque-full-rim/import').json();url=f'/api/projects/{p["id"]}'
    photo=p['photos'][0]
    r=client.patch(url+'/photos/'+photo['id'],json={'view':'right'})
    assert r.status_code==200 and r.json()['photos'][-1]['view']=='right'
    assert not any(a['approved'] for a in r.json()['annotations'])
    assert client.patch(url+'/photos/'+photo['id'],json={'view':'invalid'}).status_code==422
    uploaded=client.post(url+'/photos',data={'view':'inner'},files={'file':('inner.png',png(),'image/png')})
    assert uploaded.status_code==200
    inner=uploaded.json()['photos'][-1]
    result=client.post(url+'/photos/'+inner['id']+'/extract').json()
    assert result['annotations']==[] and 'registered inner-frame appearance' in result['warnings'][0]

def test_remove_and_restore_project_preserves_data(client):
    p=client.post('/api/samples/synthetic-opaque-full-rim/import').json()
    url=f'/api/projects/{p["id"]}'
    photo=p['photos'][0]
    assert client.delete(url).status_code==200
    assert p['id'] not in [x['id'] for x in client.get('/api/projects').json()]
    assert p['id'] in [x['id'] for x in client.get('/api/projects?archived=true').json()]
    assert client.get(photo['url']).status_code==200
    assert client.post(url+'/restore').status_code==200
    restored=client.get(url).json()
    assert restored['annotations']==p['annotations'] and restored['revision']==p['revision']
    assert p['id'] in [x['id'] for x in client.get('/api/projects').json()]
    assert client.delete('/api/projects/not-found').status_code==404

def test_thumbnail_upload_is_accepted_with_quality_warning(client):
    p=client.post('/api/projects',json={'name':'Thumbnail references'}).json()
    raw=io.BytesIO();Image.new('RGB',(100,100),'white').save(raw,'PNG')
    response=client.post(f'/api/projects/{p["id"]}/photos',data={'view':'left'},files={'file':('side.png',raw.getvalue(),'image/png')})
    assert response.status_code==200
    photo=response.json()['photos'][0]
    assert photo['width']==100 and photo['height']==100
    assert any('Thumbnail accepted' in w for w in photo['warnings'])
    tiny=io.BytesIO();Image.new('RGB',(32,32),'white').save(tiny,'PNG')
    response=client.post(f'/api/projects/{p["id"]}/photos',data={'view':'left'},files={'file':('tiny.png',tiny.getvalue(),'image/png')})
    assert response.status_code==422 and 'image size' in response.json()['detail']

def test_assumed_scale_warning_survives_export(client):
    p=client.post('/api/samples/synthetic-opaque-full-rim/import').json()
    j=client.post(f'/api/projects/{p["id"]}/jobs').json();assert process_next()
    result=client.get('/api/jobs/'+j['id']).json()
    assert result['status']=='completed'
    assert any('Approximate-size draft' in w for w in result['artifact']['metadata']['warnings'])

def test_new_front_cannot_reuse_old_model_geometry(client):
    p=client.post('/api/samples/synthetic-opaque-full-rim/import').json()
    url=f'/api/projects/{p["id"]}'
    old_annotations=p['annotations']
    first=client.post(url+'/jobs').json()
    assert process_next()
    first=client.get('/api/jobs/'+first['id']).json()
    old_bytes=client.get(first['artifact']['model_url']).content
    # Upload a genuinely different product silhouette, not just a changed filename.
    other=client.post('/api/samples/synthetic-transparent-full-rim/import').json()
    original=Image.open(io.BytesIO(client.get(other['photos'][0]['url']).content))
    changed=original.resize((original.width,original.height//2))
    changed=changed.transform(changed.size,Image.Transform.AFFINE,(1,-.3,0,0,1,0),fillcolor='white')
    encoded=io.BytesIO();changed.save(encoded,'PNG');raw=encoded.getvalue()
    p=client.post(url+'/photos',data={'view':'front'},files={'file':('new.png',raw,'image/png')}).json()
    current=p['photos'][-1]
    assert p['stale'] and not any(a['approved'] for a in p['annotations'])
    # Even explicitly restoring legacy approvals cannot select a superseded photo.
    client.patch(url,json={'annotations':old_annotations})
    rejected=client.post(url+'/jobs').json();assert process_next()
    assert client.get('/api/jobs/'+rejected['id']).json()['status']=='needs-correction'
    annotations=[dict(a,photo_id=current['id']) for a in other['annotations']]
    for a in annotations:
        a['points']=[dict(point,x=point['x']+.15*point['y'],y=point['y']/2) for point in a['points']]
        a['anchors']={key:dict(point,x=point['x']+.15*point['y'],y=point['y']/2) for key,point in a['anchors'].items()}
    client.patch(url,json={'annotations':annotations})
    second=client.post(url+'/jobs').json();assert process_next()
    second=client.get('/api/jobs/'+second['id']).json()
    assert second['status']=='completed',second
    assert second['artifact']['metadata']['provenance']['front']['photo_id']==current['id']
    new_bytes=client.get(second['artifact']['model_url']).content
    assert new_bytes != old_bytes
    old_scene=trimesh.load(io.BytesIO(old_bytes),file_type='glb')
    new_scene=trimesh.load(io.BytesIO(new_bytes),file_type='glb')
    assert any(mesh.vertices.shape != old_scene.geometry[name].vertices.shape or
               not np.allclose(mesh.vertices,old_scene.geometry[name].vertices)
               for name,mesh in new_scene.geometry.items())

def test_validation_upload_does_not_supersede_fitting_outlines(client):
    p=client.post('/api/samples/synthetic-opaque-full-rim/import').json()
    url=f'/api/projects/{p["id"]}'
    p=client.post(url+'/photos',data={'view':'front','role':'validation'},files={'file':('heldout.png',png(),'image/png')}).json()
    assert all(a['approved'] for a in p['annotations'])
    j=client.post(url+'/jobs').json();assert process_next()
    assert client.get('/api/jobs/'+j['id']).json()['status']=='completed'

def test_project_upload_validation_and_persistence(client):
    p=client.post('/api/projects',json={'name':'Real frame','category':'opaque-full-rim'}).json()
    url=f'/api/projects/{p["id"]}'
    bad=client.post(url+'/photos',data={'view':'front'},files={'file':('fake.jpg',b'not an image','image/jpeg')})
    assert bad.status_code==422
    upload=client.post(url+'/photos',data={'view':'front','role':'fitting'},files={'file':('front.png',png(),'image/png')})
    assert upload.status_code==200,upload.text
    p=upload.json();photo=p['photos'][0]
    assert 'path' not in photo and p['revision']==2
    assert client.get(photo['url']).status_code==200
    duplicate=client.post(url+'/photos',data={'view':'top'},files={'file':('same.png',png(),'image/png')}).json()
    assert any('Duplicate' in w for w in duplicate['photos'][-1]['warnings'])
    assert len(client.get(url).json()['photos'])==2
    assert client.patch(url,json={'measurements':{'frame_width':{'value':-3}}}).status_code==422
    assert client.delete(url+'/photos/'+photo['id']).status_code==200
    assert client.get(photo['url']).status_code==404
    assert client.get('/api/projects').json()[0]['id']==p['id']

def test_complete_cpu_job_and_stale_revision(client):
    p=client.post('/api/samples/synthetic-opaque-full-rim/import').json()
    assert len(p['annotations'])==3
    url=f'/api/projects/{p["id"]}'
    job=client.post(url+'/jobs').json()
    assert client.post(url+'/jobs').json()['id']==job['id']
    # A change during generation must not mutate the snapshot or mark its result current.
    edited=client.patch(url,json={'notes':'Changed after submit'}).json()
    assert edited['revision']>job['revision']
    assert process_next()
    finished=client.get('/api/jobs/'+job['id']).json()
    assert finished['status']=='completed',finished
    p=client.get(url).json()
    assert p['stale'] is True
    assert p['artifact']['revision']==job['revision']
    model=client.get(p['artifact']['model_url'])
    assert model.status_code==200 and model.content[:4]==b'glTF'
    metadata=client.get(p['artifact']['metadata_url']).json()
    assert metadata['input_revision']==job['revision']
    assert metadata['asset_bytes']==len(model.content)
    assert not process_next()

def test_missing_geometry_is_actionable_not_generic_fallback(client):
    p=client.post('/api/projects',json={'name':'Empty input'}).json()
    job=client.post(f'/api/projects/{p["id"]}/jobs').json()
    process_next()
    result=client.get('/api/jobs/'+job['id']).json()
    assert result['status']=='needs-correction'
    assert 'Measure' in result['error']
    assert client.get(f'/api/projects/{p["id"]}').json()['artifact'] is None

def test_validation_photo_excluded_and_external_origin_blocked(client):
    p=client.post('/api/projects',json={'name':'Withheld'}).json()
    url=f'/api/projects/{p["id"]}'
    p=client.post(url+'/photos',data={'view':'front','role':'validation'},files={'file':('front.png',png(),'image/png')}).json()
    photo=p['photos'][0]
    assert client.post(url+'/photos/'+photo['id']+'/extract').status_code==422
    blocked=client.post('/api/projects',json={'name':'Blocked'},headers={'origin':'https://example.com'})
    assert blocked.status_code==403
    assert client.get('/api/artifacts/nonexistent/model.glb').status_code==404

@pytest.mark.parametrize('category',['opaque-full-rim','thin-metal','transparent-full-rim','semi-rimless','rimless','sports-shield'])
def test_sample_workflow_each_category(client,category):
    p=client.post('/api/samples/synthetic-'+category+'/import').json()
    job=client.post(f'/api/projects/{p["id"]}/jobs').json()
    assert process_next()
    result=client.get('/api/jobs/'+job['id']).json()
    assert result['status']=='completed',result

def test_annotations_reject_outside_and_unknown_photos(client):
    p=client.post('/api/samples/synthetic-opaque-full-rim/import').json()
    annotations=p['annotations'];annotations[0]['points'][0]['x']=-1
    assert client.patch(f'/api/projects/{p["id"]}',json={'annotations':annotations}).status_code==422
    annotations[0]['points'][0]['x']=30;annotations[0]['photo_id']='missing'
    assert client.patch(f'/api/projects/{p["id"]}',json={'annotations':annotations}).status_code==422

def test_mask_brush_changes_boundary_and_rejects_internal_hole(client):
    polygon=[{'x':40,'y':40},{'x':160,'y':40},{'x':160,'y':160},{'x':40,'y':160}]
    body={'width':200,'height':200,'points':polygon,'strokes':[{'mode':'add','radius':15,'points':[{'x':165,'y':100}]}]}
    r=client.post('/api/masks/refine',json=body)
    assert r.status_code==200,r.text
    assert max(p['x'] for p in r.json()['points'])>170
    body['strokes']=[{'mode':'erase','radius':10,'points':[{'x':100,'y':100}]}]
    assert client.post('/api/masks/refine',json=body).status_code==422
    body['strokes']=[{'mode':'erase','radius':20,'points':[{'x':40,'y':100}]}]
    r=client.post('/api/masks/refine',json=body)
    assert r.status_code==200,r.text


@pytest.mark.parametrize('mode,color',[('RGB',(20,80,160)),('RGBA',(20,80,160,128))])
def test_webp_upload_normalizes_pixels_and_transparency_to_png(client,mode,color):
    project=client.post('/api/projects',json={'name':'WebP upload'}).json()
    original=Image.new(mode,(96,128),color)
    raw=io.BytesIO();original.save(raw,'WEBP',lossless=True)
    response=client.post(f'/api/projects/{project["id"]}/photos',data={'view':'front'},
                         files={'file':('product.webp',raw.getvalue(),'image/webp')})
    assert response.status_code==200,response.text
    photo=response.json()['photos'][0]
    assert (photo['width'],photo['height'])==(96,128)
    served=client.get(photo['url'])
    assert served.headers['content-type']=='image/png'
    with Image.open(io.BytesIO(served.content)) as decoded:
        assert decoded.format=='PNG' and decoded.mode=='RGB'
        expected=Image.new('RGB',original.size,'white')
        rgba=original.convert('RGBA');expected.paste(rgba,mask=rgba.getchannel('A'))
        assert np.array_equal(np.array(decoded),np.array(expected))


def test_webp_upload_respects_exif_orientation(client):
    project=client.post('/api/projects',json={'name':'Oriented WebP'}).json()
    exif=Image.Exif();exif[274]=6
    raw=io.BytesIO();Image.new('RGB',(96,128),'blue').save(raw,'WEBP',lossless=True,exif=exif)
    response=client.post(f'/api/projects/{project["id"]}/photos',data={'view':'front'},
                         files={'file':('rotated.webp',raw.getvalue(),'image/webp')})
    assert response.status_code==200,response.text
    photo=response.json()['photos'][0]
    assert (photo['width'],photo['height'])==(128,96)


def test_webp_filename_cannot_bypass_image_format_validation(client):
    project=client.post('/api/projects',json={'name':'Invalid WebP'}).json()
    raw=io.BytesIO();Image.new('RGB',(96,128),'blue').save(raw,'GIF')
    for content in (b'not an image',raw.getvalue()):
        response=client.post(f'/api/projects/{project["id"]}/photos',data={'view':'front'},
                             files={'file':('fake.webp',content,'image/webp')})
        assert response.status_code==422
    assert client.get(f'/api/projects/{project["id"]}').json()['photos']==[]
