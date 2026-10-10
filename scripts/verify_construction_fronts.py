"""Isolated real-photo detection -> suggested construction -> GLB diagnostics."""
import copy,json,sys
from pathlib import Path
import cv2
import numpy as np
import trimesh
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from backend import store
from backend.reconstruction import extract_annotations,reconstruct
from backend.reconstruction.render_mesh import welded_geometry
from render_fidelity_cpu import render
from shapely.geometry import Polygon

output=Path('artifacts/construction-followup');output.mkdir(parents=True,exist_ok=True)
report=[]
for name,pid in [('semi','3ca4f944b2214b72988db3bde88e0602'),
                 ('water-clear','b0bc9b703db9463abab0a9a4d30c0b4d'),
                 ('double-bridge','7def97639b034aa4be2d224429cc9f45')]:
    project=store.get_project(pid)
    photo=next(p for p in project['photos'] if p['view']=='front')
    result=extract_annotations(photo)
    dest=output/name;dest.mkdir(exist_ok=True)
    (dest/'extraction.json').write_text(json.dumps(result,indent=2))
    assert result['construction']['category'],(name,result)
    overlay=cv2.imread(photo['path'])
    bounds={}
    for ann in result['annotations']:
        points=np.array([[p['x'],p['y']] for p in ann['points']])
        bounds[ann['part']]=Polygon(points).bounds
        cv2.polylines(overlay,[np.round(points).astype(np.int32)],True,
                      (40,160,30) if ann['part']=='front' else (20,120,240),2,cv2.LINE_AA)
    cv2.imwrite(str(dest/'outline.png'),overlay)
    f,l,r=[bounds[k] for k in ['front','lens-left','lens-right']]
    scale=140/(f[2]-f[0])
    values={'frame_width':140,'temple_length':140,
            'lens_width':((l[2]-l[0])+(r[2]-r[0]))*.5*scale,
            'lens_height':((l[3]-l[1])+(r[3]-r[1]))*.5*scale,
            'bridge_width':(r[0]-l[2])*scale}
    snapshot=copy.deepcopy(project)
    snapshot['category']=result['construction']['category']
    snapshot['measurements']={k:{'value':v,'source':'inferred'} for k,v in values.items()}
    snapshot['annotations']=[dict(a,approved=True,source='merchant-reviewed:'+a['source']) for a in result['annotations']]
    metadata=reconstruct(snapshot,dest)
    scene=trimesh.load(dest/'model.glb',force='scene',process=False)
    assert all(welded_geometry(m).is_watertight for m in scene.geometry.values())
    target=scene.bounds.mean(axis=0)
    render(dest/'model.glb',target+[0,0,.31],target,'clay',650).save(dest/'front-clay.png')
    render(dest/'model.glb',target+[.16,.035,.28],target,'photo',650).save(dest/'threequarter-photo.png')
    report.append({'case':name,'construction':result['construction'],'parts':metadata['parts'],
                   'checks':metadata['checks'],'front_only':True})
(output/'verification.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
