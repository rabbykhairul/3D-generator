"""Read-only merchant reference; isolated post-fix contour visualization."""
from pathlib import Path
import json
from copy import deepcopy
import cv2
import numpy as np
import trimesh
from PIL import Image
from backend.store import get_project
from backend.reconstruction.engine import extract_annotations, reconstruct
from backend.reconstruction.render_mesh import welded_geometry
from scripts.render_fidelity_cpu import render

def main():
    project = get_project('7def97639b034aa4be2d224429cc9f45')
    photo = next(p for p in project['photos'] if p['view'] == 'right')
    result = extract_annotations(photo)
    image = cv2.imread(photo['path'])
    for item in result['annotations'] + result.get('appearance_evidence', []):
        points = np.rint([[p['x'],p['y']] for p in item['points']]).astype(np.int32)
        color = (30,130,230) if 'anchors' in item else (190,160,20)
        cv2.polylines(image,[points],True,color,2,cv2.LINE_AA)
        for name,point in item.get('anchors',{}).items():
            xy = (round(point['x']),round(point['y']))
            cv2.circle(image,xy,7,(60,170,70),-1)
            cv2.putText(image,name,xy,cv2.FONT_HERSHEY_SIMPLEX,.6,(0,0,0),1,cv2.LINE_AA)
    output = Path('artifacts/olive-side-root')
    output.mkdir(parents=True,exist_ok=True)
    cv2.imwrite(str(output/'after-outline.png'),image)
    (output/'analysis.json').write_text(json.dumps(result,indent=2))
    front = next(p for p in project['photos'] if p['view'] == 'front')
    front_result = extract_annotations(front)
    draft = deepcopy(project)
    draft['annotations'] = [dict(a, approved=True, source='diagnostic-review:'+a['source'])
                            for a in front_result['annotations']+result['annotations']]
    if any(a['part']=='temple-left' and a['source'].startswith('explicit-mirrored') for a in project['annotations']):
        draft['annotations'].append(dict(result['annotations'][0],part='temple-left',approved=True,
                                         source='explicit-mirrored-side-assumption:diagnostic'))
    # This disposable asset tests the observed right arm only. No opposite
    # arm or texture transfer is fabricated, and no database is changed.
    polygons = {a['part']:np.array([[p['x'],p['y']] for p in a['points']])
                for a in front_result['annotations']}
    scale = 140/np.ptp(polygons['front'][:,0])
    left,right = polygons['lens-left'],polygons['lens-right']
    values = {'frame_width':140,'temple_length':140,
              'lens_width':(np.ptp(left[:,0])+np.ptp(right[:,0]))*.5*scale,
              'lens_height':(np.ptp(left[:,1])+np.ptp(right[:,1]))*.5*scale,
              'bridge_width':(right[:,0].min()-left[:,0].max())*scale}
    draft['measurements'] = {k:{'value':float(v),'source':'inferred'} for k,v in values.items()}
    metadata = reconstruct(draft,output)
    scene = trimesh.load(output/'model.glb',force='scene',process=False)
    assert all(welded_geometry(m).is_watertight for m in scene.geometry.values())
    assert metadata['multiview']['opposite_arm_appearance'][0]['applied'] is True
    center = scene.bounds.mean(axis=0)
    render(output/'model.glb',center+[.30,.07,.22],center,'clay',850).save(output/'after-clay.png')
    arm = scene.geometry['temple_left']
    isolated=trimesh.Scene(arm)
    isolated.export(str(output/'inner-arm.glb'))
    center=arm.bounds.mean(axis=0)
    render(output/'inner-arm.glb',center+[.34,0,.02],center,'photo',1100).save(output/'inner-arm-photo.png')
    render(output/'inner-arm.glb',center+[-.34,0,.02],center,'photo',1100).save(output/'outer-arm-photo.png')
    baseline=arm.copy()
    baseline.visual.material=arm.visual.material.copy()
    atlas=np.array(baseline.visual.material.baseColorTexture)
    tile=atlas.shape[1]//3
    atlas[:,tile:tile*2]=atlas[:,:tile]
    baseline.visual.material.baseColorTexture=Image.fromarray(atlas)
    trimesh.Scene(baseline).export(str(output/'before-inner-arm.glb'))
    render(output/'before-inner-arm.glb',center+[.34,0,.02],center,'photo',1100).save(output/'before-inner-arm-photo.png')

if __name__ == '__main__':
    main()
