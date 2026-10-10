"""Isolated evidence; never changes merchant approvals or saved artifacts."""
from copy import deepcopy
from pathlib import Path
import json
import cv2
import numpy as np
from backend.store import get_project
from backend.reconstruction.engine import extract_annotations, _extract_image_annotations, reconstruct


def main():
    project = get_project('d6fec31607d84030849761a204c137db')
    photo = next(p for p in project['photos'] if p['view']=='front')
    image = cv2.imread(photo['path'])
    folder = Path('artifacts/dark-sunglasses')
    folder.mkdir(parents=True,exist_ok=True)
    result = extract_annotations(photo)
    for label,data in [('before',_extract_image_annotations(photo,image,1,1)),('after',result)]:
        overlay = image.copy()
        for a in data['annotations']:
            points = np.rint([[p['x'],p['y']] for p in a['points']]).astype('int32')
            cv2.polylines(overlay,[points],True,(220,140,20) if a['part'].startswith('nose-pad-') else (60,170,70) if a['part']=='front' else (20,130,230),1,cv2.LINE_AA)
        cv2.imwrite(str(folder/f'{label}.png'),overlay)
    (folder/'analysis.json').write_text(json.dumps(result,indent=2))
    side = next(p for p in project['photos'] if p['view']=='right')
    side_result = extract_annotations(side)
    side_image = cv2.imread(side['path'])
    for a in side_result['annotations']:
        cv2.polylines(side_image,[np.rint([[p['x'],p['y']] for p in a['points']]).astype('int32')],True,(20,130,230),2,cv2.LINE_AA)
    cv2.imwrite(str(folder/'arm-after.png'),side_image)
    draft = deepcopy(project)
    draft['photos'] = [photo,side]
    draft['annotations'] = deepcopy(result['annotations']+side_result['annotations'])
    for a in draft['annotations']:
        a['approved'] = True  # isolated diagnostic copy only
    for name,value in {'frame_width':140,'lens_width':54,'lens_height':46,'bridge_width':18,'temple_length':140}.items():
        draft['measurements'][name] = {'value':value,'source':'diagnostic-assumption'}
    generated = reconstruct(draft,folder/'draft')
    assert generated['checks']['front_parts_watertight']
    assert generated['checks']['consistent_outward_normals']
    import trimesh
    from backend.reconstruction.render_mesh import welded_geometry
    scene = trimesh.load(folder/'draft/model.glb',force='scene',process=False)
    arms = [mesh for name,mesh in scene.geometry.items() if name.startswith('temple_right')]
    assert arms and all(welded_geometry(mesh).is_watertight for mesh in arms)
    from scripts.render_fidelity_cpu import render
    target = scene.bounds.mean(axis=0)
    render(folder/'draft/model.glb',target+[0,0,.34],target,'photo',900,
           include_lenses=True).save(folder/'tint-pads.png')
    print(generated['checks'])


if __name__=='__main__':
    main()
