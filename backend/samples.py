"""Local public references and explicitly synthetic numerical fixtures."""
import io
import json
import math
from PIL import Image, ImageDraw
from fastapi import HTTPException
from . import store

CATEGORIES=['opaque-full-rim','thin-metal','transparent-full-rim','semi-rimless','rimless','sports-shield']

def public_references():
    path=store.ROOT/'data'/'references'/'manifest.json'
    return json.loads(path.read_text()) if path.exists() else []

def catalog():
    public=[{k:p.get(k) for k in ('id','name','category','kind','description')} for p in public_references() if p.get('photos')]
    return public+[dict(id='synthetic-'+c,name=c.replace('-',' ').title()+' · synthetic',category=c,kind='synthetic',description='Known 2D outline fixture for workflow/numerical tests. Not a real product or proof of reconstruction accuracy.') for c in CATEGORIES]

def fixture(category):
    """An explicit drawn numerical shape; never passed off as a product photo."""
    image=Image.new('RGB',(1000,500),'white');draw=ImageDraw.Draw(image)
    def superellipse(cx,cy,rx,ry,n=3.4):
        result=[]
        for i in range(64):
            a=2*math.pi*i/64;c=math.cos(a);s=math.sin(a)
            result.append({'x':cx+rx*math.copysign(abs(c)**(2/n),c),'y':cy+ry*math.copysign(abs(s)**(2/n),s)})
        return result
    # Two distinct rims joined across the top of the bridge, represented as one outer polygon.
    from shapely.geometry import Polygon,box
    from shapely.ops import unary_union
    left=superellipse(300,240,155,112);right=superellipse(700,240,155,112)
    outer_left=superellipse(300,240,181,137);outer_right=superellipse(700,240,181,137)
    front=unary_union([Polygon([(p['x'],p['y']) for p in outer_left]),Polygon([(p['x'],p['y']) for p in outer_right]),box(470,170,530,196)])
    outlines=[('front',[{'x':float(x),'y':float(y)} for x,y in list(front.exterior.coords)[:-1]]),('lens-left',left),('lens-right',right)]
    if category=='sports-shield':
        shield=superellipse(500,240,381,137,3)
        outlines=[('shield',shield)]
        draw.polygon([(p['x'],p['y']) for p in shield],fill='#365a70')
    else:
        draw.polygon([(p['x'],p['y']) for p in outlines[0][1]],fill='#243b49')
        for points in (left,right):draw.polygon([(p['x'],p['y']) for p in points],fill='white')
    if category=='rimless':outlines=outlines[1:]
    raw=io.BytesIO();image.save(raw,format='PNG')
    return raw.getvalue(),outlines

def import_into_project(sample_id):
    from .main import create_project,ingest_photo
    category=sample_id.removeprefix('synthetic-')
    if sample_id.startswith('synthetic-') and category in CATEGORIES:
        p=create_project(category.replace('-',' ').title()+' — synthetic fixture',category)
        raw,outlines=fixture(category)
        p=ingest_photo(p['id'],raw,'front','fitting')
        photo=p['photos'][0]
        def finish(p):
            p['annotations']=[dict(photo_id=photo['id'],part=part,points=points,anchors={},approved=True,source='synthetic-ground-truth') for part,points in outlines]
            p['measurements']={k:dict(value=v,source='default') for k,v in {'frame_width':140,'lens_width':54,'lens_height':40,'bridge_width':18,'temple_length':140,'frame_thickness':4,'temple_thickness':3,'front_curve':6,'temple_spread':3}.items()}
            p['notes']='SYNTHETIC NUMERICAL FIXTURE. Known drawn front outlines; default dimensions. Not a real product photo and not evidence of real-world fidelity. Arms/curvature are inferred. Compare public samples separately.'
            p['revision']+=1
        return store.mutate_project(p['id'],finish)
    reference=next((p for p in public_references() if p['id']==sample_id and p.get('photos')),None)
    if not reference:raise HTTPException(404,'Sample unavailable. Run scripts/fetch_references.py to fetch public references.')
    p=create_project(reference['name'],reference['category'])
    for photo in reference['photos']:
        path=store.ROOT/photo['path']
        if path.is_file():
            p=ingest_photo(p['id'],path.read_bytes(),photo['view'],photo['role'],reference['source_url'],photo['image_source_url'])
    def finish(p):
        p['measurements']=reference.get('measurements',{})
        p['notes']='PUBLIC REFERENCE — '+reference['source_url']+'\n'+'\n'.join(reference.get('limitations',[]))+'\nProduct photography remains the source owner’s content. Local research only.'
        p['revision']+=1
    return store.mutate_project(p['id'],finish)
