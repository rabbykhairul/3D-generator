"""Compare the same lavender source with and without zoom regularization."""
from pathlib import Path
import json
import cv2
import numpy as np
from copy import deepcopy
import trimesh
from backend.store import get_project
from backend.reconstruction.engine import extract_annotations,reconstruct
from backend.reconstruction import zoom_refinement
from backend.reconstruction.render_mesh import welded_geometry
from scripts.render_fidelity_cpu import render

def main():
    project=get_project('8dd590d016a44d268efcb319d3f86437')
    photo=next(p for p in project['photos'] if p['view']=='front')
    original=zoom_refinement.refine_front_zoom
    try:
        zoom_refinement.refine_front_zoom=lambda *a,**k:{'applied':False}
        before=extract_annotations(photo)
    finally:
        zoom_refinement.refine_front_zoom=original
    after=extract_annotations(photo)
    folder=Path('artifacts/front-zoom');folder.mkdir(parents=True,exist_ok=True)
    for name,result in [('before',before),('after',after)]:
        image=cv2.resize(cv2.imread(photo['path']),None,fx=3,fy=3,interpolation=cv2.INTER_CUBIC)
        for ann in result['annotations']:
            points=np.rint(np.array([[p['x'],p['y']] for p in ann['points']])*48).astype(np.int32)
            cv2.polylines(image,[points],True,(80,160,60) if ann['part']=='front' else (30,120,235),3,cv2.LINE_AA,shift=4)
        cv2.imwrite(str(folder/f'{name}.png'),image)
        cv2.imwrite(str(folder/f'{name}-brow.png'),image[1200:1560,300:1500])
        (folder/f'{name}.json').write_text(json.dumps(result,indent=2))
        draft=deepcopy(project)
        draft['annotations']=[dict(a,approved=True,source='diagnostic-reviewed:'+a['source']) for a in result['annotations']]
        coords={a['part']:np.array([[p['x'],p['y']] for p in a['points']]) for a in result['annotations']}
        left,right=coords['lens-left'],coords['lens-right'];scale=140/np.ptp(coords['front'][:,0])
        values={'frame_width':140,'temple_length':140,'lens_width':(np.ptp(left[:,0])+np.ptp(right[:,0]))*.5*scale,
                'lens_height':(np.ptp(left[:,1])+np.ptp(right[:,1]))*.5*scale,'bridge_width':(right[:,0].min()-left[:,0].max())*scale}
        draft['measurements']={k:{'value':float(v),'source':'inferred'} for k,v in values.items()}
        destination=folder/name
        reconstruct(draft,destination)
        scene=trimesh.load(destination/'model.glb',force='scene',process=False)
        assert all(welded_geometry(mesh).is_watertight for mesh in scene.geometry.values())
        center=scene.bounds.mean(axis=0)
        render(destination/'model.glb',center+[0,0,.32],center,'clay',900).save(folder/f'{name}-clay.png')
    print(after['zoom_refinement'])

if __name__=='__main__':main()
