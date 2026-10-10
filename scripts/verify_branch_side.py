"""Compare legacy/current side outlines and isolated generation; no saved writes."""
import copy,json,sys
from pathlib import Path
import cv2
import numpy as np
import trimesh
from PIL import Image,ImageDraw
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from backend import store
from backend.reconstruction import extract_annotations,reconstruct
from backend.reconstruction.engine import _foreground_mask,_faint_side_suggestion
from backend.reconstruction.side_extraction import extract_side,recover_pale_side,recover_opaque_side
from backend.reconstruction.render_mesh import welded_geometry
from render_fidelity_cpu import render

output=Path('artifacts/branch-side-fix');output.mkdir(parents=True,exist_ok=True)
for name,pid in [('pink','916b16f20d294e36bf92f935881d8f30'),('lavender','8dd590d016a44d268efcb319d3f86437')]:
    project=store.get_project(pid)
    photo=next(p for p in project['photos'] if p['view']=='right')
    im=cv2.imread(photo['path']);mask,_=_foreground_mask(im)
    old=recover_opaque_side(im,photo,1,1) or extract_side(photo,mask,1,1)
    if not old['annotations']:
        old=recover_pale_side(im,photo,1,1) or _faint_side_suggestion(im,photo,1,1) or old
    new=extract_annotations(photo)
    panels=[]
    front=next(p for p in project['photos'] if p['view']=='front')
    front_annotations=[dict(a,approved=True) for a in extract_annotations(front)['annotations']]
    for label,result in [('before',old),('after',new)]:
        dest=output/name/label;dest.mkdir(parents=True,exist_ok=True)
        (dest/'extraction.json').write_text(json.dumps(result,indent=2))
        overlay=im.copy()
        for ann in result['annotations']:
            points=np.round([[p['x'],p['y']] for p in ann['points']]).astype(np.int32)
            cv2.polylines(overlay,[points],True,(30,125,230),2,cv2.LINE_AA)
        panel=overlay[420:780,15:1055]
        panel=cv2.copyMakeBorder(panel,35,0,0,0,cv2.BORDER_CONSTANT,value=(255,255,255))
        cv2.putText(panel,label.upper(),(10,25),cv2.FONT_HERSHEY_SIMPLEX,.7,(20,20,20),1)
        panels.append(panel)
        snapshot=copy.deepcopy(project)
        snapshot['annotations']=front_annotations+[dict(a,approved=True) for a in result['annotations']]
        try:
            metadata=reconstruct(snapshot,dest)
            scene=trimesh.load(dest/'model.glb',force='scene')
            assert all(welded_geometry(m).is_watertight for m in scene.geometry.values())
            print(name,label,metadata['parts'])
        except Exception as exc:
            print(name,label,'generation declined:',exc)
    cv2.imwrite(str(output/name/'outline-comparison.png'),np.concatenate(panels,axis=0))
    paths=[output/name/label/'model.glb' for label in ['before','after']]
    if all(p.exists() for p in paths):
        target=trimesh.load(paths[1],force='scene').geometry['temple_right'].bounds.mean(axis=0)
        pair=Image.new('RGB',(1200,640),'white');draw=ImageDraw.Draw(pair)
        for i,(label,path) in enumerate(zip(['before','after'],paths)):
            draw.text((i*600+10,10),label+' / reloaded GLB clay; CPU diagnostic',fill='black')
            pair.paste(render(path,target+[-.28,0,0],target,'clay',600),(600*i,40))
        pair.save(output/name/'mesh-comparison.png')
