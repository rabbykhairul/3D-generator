import hashlib
import io
import json
import os
import re
import subprocess
import sys
import warnings
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Request
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image, ImageOps, UnidentifiedImageError

from . import store
from .contracts import ProjectCreate, ProjectPatch, Material, MaskRefine, PhotoViewPatch, AnatomyReview
from .reconstruction.image_resolution import resolution_diagnostics

MAX_UPLOAD = 20 * 1024 * 1024
Image.MAX_IMAGE_PIXELS = 40_000_000
VIEWS = {'front','front-left','left','rear-left','rear','rear-right','right','front-right','top','bottom','elevated-front-left','elevated-front-right','hinge-left','hinge-right','bridge','nose-pads','inner-left','inner-right','detail'}
VIEWS.add('inner')

@asynccontextmanager
async def lifespan(app):
    store.initialize()
    process = None
    if os.environ.get('EYEWEAR_START_WORKER', '1') == '1':
        process = subprocess.Popen([sys.executable, '-m', 'backend.worker'], cwd=store.ROOT)
    yield
    if process:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()

app = FastAPI(title='Frame Lab local reconstruction API', version='0.1.0', lifespan=lifespan)

@app.middleware('http')
async def local_only(request: Request, call_next):
    # Origin protection is not authentication. Remote installations must also
    # protect every route with a trusted authenticated HTTPS reverse proxy.
    origin = request.headers.get('origin')
    if origin:
        from urllib.parse import urlparse
        allowed = {value.strip().rstrip('/') for value in os.environ.get('EYEWEAR_ALLOWED_ORIGINS','').split(',') if value.strip()}
        parsed = urlparse(origin)
        local = parsed.scheme in {'http','https'} and parsed.hostname in {'localhost','127.0.0.1','::1'}
        if origin.rstrip('/') not in allowed and not (local and not allowed):
            return JSONResponse({'detail':'Browser origin is not allowed.'}, status_code=403)
    response = await call_next(request)
    response.headers['X-Content-Type-Options'] = 'nosniff'
    return response

def project_or_404(project_id):
    p = store.get_project(project_id)
    if not p:
        raise HTTPException(404, 'Project not found')
    return p

def bump(project):
    project['revision'] += 1

def create_project(name, category):
    p = dict(id=store.uid(),name=name,category=category,revision=1,created_at=store.now(),updated_at=store.now(),measurements={},material=Material().model_dump(),annotations=[],photos=[],notes='',artifact=None)
    if category == 'thin-metal':
        p['material'].update(color='#78858e',metalness=0.85,roughness=0.24)
    if category == 'transparent-full-rim':
        p['material'].update(color='#c1dce7',opacity=0.4,roughness=0.15)
    if category == 'sports-shield':
        p['material'].update(lens_color='#355b78',lens_opacity=0.7)
    store.save_project(p)
    return p

@app.get('/api/health')
def health():
    from .engine_version import ENGINE_VERSION
    return {'status':'ok','mode':'local-cpu','camera_processing':'browser-only','engine_version':ENGINE_VERSION,'joint_front_fit_available':True}

@app.get('/api/anatomy/schema')
def anatomy_schema():
    from .reconstruction.anatomy import structure_schema
    return structure_schema()

@app.post('/api/masks/refine')
def refine_mask(data:MaskRefine):
    if data.width*data.height>40_000_000:raise HTTPException(422,'Mask exceeds the photo resolution limit.')
    all_points=list(data.points)+[p for stroke in data.strokes for p in stroke.points]
    if any(p.x<0 or p.y<0 or p.x>data.width or p.y>data.height for p in all_points):
        raise HTTPException(422,'Paint strokes must stay within the photograph.')
    scale=min(1,2048/max(data.width,data.height))
    mask=np.zeros((max(1,round(data.height*scale)),max(1,round(data.width*scale))),dtype=np.uint8)
    def xy(points):return np.round([[p.x*scale,p.y*scale] for p in points]).astype(np.int32)
    if len(data.points)>=3:cv2.fillPoly(mask,[xy(data.points)],255)
    for stroke in data.strokes:
        points=xy(stroke.points);radius=max(1,round(stroke.radius*scale));color=255 if stroke.mode=='add' else 0
        for point in points:cv2.circle(mask,tuple(point),radius,color,-1)
        if len(points)>1:cv2.polylines(mask,[points],False,color,2*radius)
    contours,hierarchy=cv2.findContours(mask,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
    if not contours:return {'points':[],'warnings':['The mask is empty; paint or draw a new outline.']}
    external=[i for i in range(len(contours)) if hierarchy[0][i][3]==-1 and cv2.contourArea(contours[i])>4]
    if len(external)!=1:raise HTTPException(422,'This stroke creates disconnected parts. Keep one connected outline per part, or undo and trace separate parts.')
    if hierarchy[0][external[0]][2]!=-1:raise HTTPException(422,'This stroke creates an internal hole. Trace lens openings as separate lens outlines; erase from the boundary to reshape this part.')
    contour=contours[external[0]]
    simplified=cv2.approxPolyDP(contour,max(.5,cv2.arcLength(contour,True)*.0005),True).reshape(-1,2)/scale
    return {'points':[{'x':min(data.width,float(x)),'y':min(data.height,float(y))} for x,y in simplified],'warnings':['Brush correction is unapproved until you review and approve the outline.']}

@app.get('/api/projects')
def projects(archived:bool=False):
    return [store.public_project(p) for p in store.list_projects() if bool(p.get('archived_at'))==archived]

@app.delete('/api/projects/{project_id}')
def archive_project(project_id:str):
    project_or_404(project_id)
    return store.public_project(store.mutate_project(project_id,lambda p:p.update(archived_at=store.now())))

@app.post('/api/projects/{project_id}/restore')
def restore_project(project_id:str):
    project_or_404(project_id)
    return store.public_project(store.mutate_project(project_id,lambda p:p.pop('archived_at',None)))

@app.post('/api/projects', status_code=201)
def new_project(data: ProjectCreate):
    return store.public_project(create_project(data.name, data.category))

@app.get('/api/projects/{project_id}')
def project(project_id: str):
    return store.public_project(project_or_404(project_id))

@app.patch('/api/projects/{project_id}')
def patch(project_id: str, data: ProjectPatch):
    project_or_404(project_id)
    changes = data.model_dump(exclude_none=True)
    def mutate(p):
        if 'annotations' in changes:
            ids = {photo['id'] for photo in p['photos']}
            if any(a['photo_id'] not in ids for a in changes['annotations']):
                raise HTTPException(422,'An annotation refers to a missing photo.')
            for annotation in changes['annotations']:
                photo = next(photo for photo in p['photos'] if photo['id']==annotation['photo_id'])
                if any(not np.isfinite(point[k]) for point in annotation['points'] for k in ('x','y')):
                    raise HTTPException(422,'Outline coordinates must be finite.')
                if any(point['x'] < 0 or point['y'] < 0 or point['x'] > photo['width'] or point['y'] > photo['height'] for point in annotation['points']):
                    raise HTTPException(422,'Outline points must stay inside the photo.')
                if any(point['x'] < 0 or point['y'] < 0 or point['x'] > photo['width'] or point['y'] > photo['height'] for point in annotation['anchors'].values()):
                    raise HTTPException(422,'Anchor points must stay inside the photo.')
        p.update(changes)
        bump(p)
    return store.public_project(store.mutate_project(project_id, mutate))

def ingest_photo(project_id, raw, view, role, source_url='', image_source_url=''):
    if len(raw)>MAX_UPLOAD:
        raise HTTPException(413,'Each image must be at most 20 MB.')
    if view not in VIEWS or role not in {'fitting','validation'}:
        raise HTTPException(422,'Choose a supported photo view and fitting/validation role.')
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(raw)) as original:
                if original.format not in {'JPEG','PNG','WEBP'}:
                    raise HTTPException(422,'Only genuine JPEG, PNG and WebP images are supported.')
                if original.width * original.height > 40_000_000:
                    raise HTTPException(413,'Images must contain at most 40 megapixels.')
                original.load()
                oriented = ImageOps.exif_transpose(original).convert('RGBA')
                image = Image.new('RGB', oriented.size, 'white')
                image.paste(oriented, mask=oriented.getchannel('A'))
    except HTTPException:
        raise
    except (UnidentifiedImageError,OSError,ValueError,Image.DecompressionBombError,Image.DecompressionBombWarning):
        raise HTTPException(422,'Could not decode this image safely. Upload a JPEG, PNG or WebP.')
    if min(image.size)<64:
        raise HTTPException(422,'Image resolution must be at least 64 pixels on each side. This is image size, not the frame measurements in millimeters.')
    sha=hashlib.sha256(raw).hexdigest()
    photo_id=store.uid()
    folder=store.DATA/'projects'/project_id/'photos'
    folder.mkdir(parents=True,exist_ok=True)
    path=folder/f'{photo_id}.png'
    image.save(path)
    small=image.copy();small.thumbnail((1024,1024))
    sharpness=float(cv2.Laplacian(cv2.cvtColor(np.asarray(small),cv2.COLOR_RGB2GRAY),cv2.CV_64F).var())
    resolution=resolution_diagnostics(image.width,image.height)
    notes=list(resolution['warnings'])
    if sharpness<30: notes.append('This photo may be blurry or low contrast; inspect the outlines.')
    photo=dict(id=photo_id,view=view,role=role,url=f'/api/projects/{project_id}/photos/{photo_id}',path=str(path),width=image.width,height=image.height,resolution=resolution,warnings=notes,sha256=sha,source_url=source_url,image_source_url=image_source_url)
    def mutate(p):
        if len(p['photos'])>=40: raise HTTPException(422,'A project can contain at most 40 photos.')
        if any(x['sha256']==sha for x in p['photos']): photo['warnings'].append('Duplicate image: this does not add an independent viewpoint.')
        if role == 'fitting' and view in {'front','left','right','top','inner-left','inner-right'}:
            superseded = {x['id'] for x in p['photos'] if x['view']==view and x['role']=='fitting'}
            for annotation in p['annotations']:
                if annotation['photo_id'] in superseded:
                    annotation['approved'] = False
        p['photos'].append(photo)
        bump(p)
    try:
        return store.mutate_project(project_id,mutate)
    except Exception:
        path.unlink(missing_ok=True)
        raise

@app.post('/api/projects/{project_id}/photos')
async def upload(project_id:str,file:UploadFile=File(...),view:str=Form(...),role:str=Form('fitting')):
    project_or_404(project_id)
    raw=await file.read(MAX_UPLOAD+1)
    return store.public_project(ingest_photo(project_id,raw,view,role))

@app.get('/api/projects/{project_id}/photos/{photo_id}')
def get_photo(project_id:str,photo_id:str):
    p=project_or_404(project_id)
    photo=next((x for x in p['photos'] if x['id']==photo_id),None)
    if not photo: raise HTTPException(404,'Photo not found')
    return FileResponse(photo['path'],media_type='image/png')

@app.delete('/api/projects/{project_id}/photos/{photo_id}')
def remove_photo(project_id:str,photo_id:str):
    p=project_or_404(project_id)
    if not any(x['id']==photo_id for x in p['photos']): raise HTTPException(404,'Photo not found')
    def mutate(p):
        p['photos']=[x for x in p['photos'] if x['id']!=photo_id]
        p['annotations']=[x for x in p['annotations'] if x['photo_id']!=photo_id]
        bump(p)
    # Keep underlying bytes for immutable historical job snapshots; unlink only on explicit dataset cleanup.
    return store.public_project(store.mutate_project(project_id,mutate))

@app.patch('/api/projects/{project_id}/photos/{photo_id}')
def relabel_photo(project_id:str,photo_id:str,data:PhotoViewPatch):
    project_or_404(project_id)
    if data.view not in VIEWS:raise HTTPException(422,'Unsupported photo view.')
    def mutate(p):
        photo=next((x for x in p['photos'] if x['id']==photo_id),None)
        if photo is None:raise HTTPException(404,'Photo not found')
        invalid={photo_id}
        if photo['role']=='fitting':invalid.update(x['id'] for x in p['photos'] if x['view']==data.view and x['role']=='fitting')
        for ann in p['annotations']:
            if ann['photo_id'] in invalid:ann['approved']=False
        photo['view']=data.view
        p['photos']=[x for x in p['photos'] if x['id']!=photo_id]+[photo]
        bump(p)
    return store.public_project(store.mutate_project(project_id,mutate))

@app.post('/api/projects/{project_id}/photos/{photo_id}/extract')
def extract(project_id:str,photo_id:str,backend:Literal['heuristics','cpu-metal','hybrid','sam2','sam3','eyewear-trained']|None=None):
    from .reconstruction import extract_annotations, NeedsCorrection
    p=project_or_404(project_id)
    photo=next((x for x in p['photos'] if x['id']==photo_id),None)
    if not photo: raise HTTPException(404,'Photo not found')
    if photo['role']=='validation': raise HTTPException(422,'Validation photos are withheld from fitting. Upload or select a fitting photo.')
    if backend == 'eyewear-trained' and not os.environ.get('EYEWEAR_TRAINED_MODEL'):
        raise HTTPException(503, 'No trained eyewear checkpoint is configured. Dataset audit and held-out validation are required first.')
    try:
        result = extract_annotations(photo) if backend is None else extract_annotations(photo, backend=backend)
    except NeedsCorrection as e:
        from .reconstruction.anatomy import audit_photo
        result = {'annotations':[], 'anatomy':audit_photo(photo, []), 'warnings':[str(e)]}
    except (RuntimeError, ValueError) as e:
        if backend != 'eyewear-trained':
            raise
        raise HTTPException(503, f'Eyewear candidate unavailable: {e}') from e
    if (backend or os.environ.get('EYEWEAR_DETECTION_BACKEND'))=='sam3' and photo.get('view')=='front' and result.get('hybrid_detection',{}).get('inference_ran'):
        from .reconstruction.sam3_multiview import combine_reference_components
        result=combine_reference_components(p,photo,result)
    from .reconstruction.detection_report import detection_report
    result['detection_report'] = store.record_detection(project_id, photo_id, detection_report(photo, result))
    return result

@app.get('/api/projects/{project_id}/detection-runs')
def detection_runs(project_id:str):
    project = project_or_404(project_id)
    return store.detection_history(project_id, {p['id'] for p in project['photos']})


@app.post('/api/projects/{project_id}/photos/{photo_id}/anatomy')
def review_anatomy(project_id:str, photo_id:str, payload: AnatomyReview):
    """Read-only audit of the current editor draft, not approval or persistence."""
    from .reconstruction.anatomy import audit_photo
    project = project_or_404(project_id)
    photo = next((p for p in project['photos'] if p['id'] == photo_id), None)
    if photo is None:
        raise HTTPException(404, 'Photo not found')
    if any(a.photo_id != photo_id for a in payload.annotations):
        raise HTTPException(422, 'Anatomy audit accepts only annotations for the selected photo.')
    return audit_photo(photo, [a.model_dump() for a in payload.annotations], payload.category or project['category'])

@app.post('/api/projects/{project_id}/jobs',status_code=202)
def submit(project_id:str):
    p=project_or_404(project_id)
    # One active job per immutable revision, serialized with worker claims.
    with store.db() as conn:
        conn.execute('BEGIN IMMEDIATE')
        row=conn.execute("SELECT payload FROM jobs WHERE project_id=? AND status IN ('queued','processing')",(project_id,)).fetchone()
        if row: return json.loads(row['payload'])
        job=dict(id=store.uid(),project_id=project_id,revision=p['revision'],status='queued',stage='Waiting for CPU worker',created_at=store.now())
        conn.execute('INSERT INTO jobs VALUES (?,?,?,?,?,?)',(job['id'],project_id,job['status'],json.dumps(job),json.dumps(p),job['created_at']))
    return job

@app.get('/api/jobs/{job_id}')
def job_status(job_id:str):
    job=store.get_job(job_id)
    if not job: raise HTTPException(404,'Job not found')
    return job

@app.get('/api/artifacts/{job_id}/{filename}')
def artifact(job_id:str,filename:str):
    if filename not in {'model.glb','metadata.json','fit-front.png','fit-view-0.png','fit-view-1.png','fit-zoom-front.png','fit-zoom-left.png','fit-zoom-right.png'}: raise HTTPException(404,'Artifact not found')
    job=store.get_job(job_id)
    if not job or job['status']!='completed': raise HTTPException(404,'Artifact is not ready')
    path=store.DATA/'artifacts'/job_id/filename
    if not path.is_file(): raise HTTPException(404,'Artifact file is missing')
    return FileResponse(path,media_type='model/gltf-binary' if filename.endswith('.glb') else 'image/png' if filename.endswith('.png') else 'application/json',filename=filename,headers={'Cache-Control':'private, max-age=31536000, immutable'})

@app.get('/api/samples')
def samples():
    from .samples import catalog
    return catalog()

@app.post('/api/samples/{sample_id}/import',status_code=201)
def import_sample(sample_id:str):
    from .samples import import_into_project
    return store.public_project(import_into_project(sample_id))
