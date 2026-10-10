"""Reproduce blue pp arm defects without changing the saved project."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend import store
from backend.reconstruction import extract_annotations, reconstruct

OUT = Path('artifacts/blue-arm-fix')


def overlay(photo, annotation):
    image = cv2.imread(photo['path'])
    points = np.round([[p['x'],p['y']] for p in annotation['points']]).astype(np.int32)
    cv2.polylines(image,[points],True,(10,120,230),2,cv2.LINE_AA)
    for p in annotation['anchors'].values():
        cv2.circle(image,(round(p['x']),round(p['y'])),5,(180,20,200),-1)
    return image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline',action='store_true')
    args = parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    if args.baseline:
        project = store.get_project('c8e4014e30044160ace24fe171a57b5e')
        if (OUT/'before-project.json').exists():
            raise RuntimeError('Baseline already preserved; refusing overwrite')
        (OUT/'before-project.json').write_text(json.dumps(project,indent=2))
        shutil.copyfile(store.DATA/'artifacts'/project['artifact']['job_id']/'model.glb', OUT/'before.glb')
        photo = next(p for p in project['photos'] if p['view']=='right')
        result = extract_annotations(photo)
        (OUT/'before-extraction.json').write_text(json.dumps(result,indent=2))
        cv2.imwrite(str(OUT/'before.png'),overlay(photo,result['annotations'][0]))
        print('Preserved baseline',photo['path'])
        return
    project = json.loads((OUT/'before-project.json').read_text())
    photo = next(p for p in project['photos'] if p['view']=='right')
    result = extract_annotations(photo)
    ann = result['annotations'][0]
    (OUT/'after-extraction.json').write_text(json.dumps(result,indent=2))
    cv2.imwrite(str(OUT/'after.png'),overlay(photo,ann))
    panels = []
    for label in ('before','after'):
        image = cv2.imread(str(OUT/f'{label}.png'))
        # Identical crop in original 1080px photo coordinates, full glasses visible.
        panel = cv2.copyMakeBorder(image[445:760,20:1055],40,0,0,0,cv2.BORDER_CONSTANT,value=(255,255,255))
        cv2.putText(panel,label.upper()+' - same blue product photo',(12,25),cv2.FONT_HERSHEY_SIMPLEX,.65,(30,30,30),1,cv2.LINE_AA)
        panels.append(panel)
    cv2.imwrite(str(OUT/'outline-comparison.png'),np.concatenate(panels,axis=1))
    snapshot = copy.deepcopy(project)
    for index, old in enumerate(snapshot['annotations']):
        if old['part']=='temple-right':
            snapshot['annotations'][index]=dict(ann,approved=True,source='merchant-reviewed-side:'+ann['source'])
        elif old['part']=='temple-left' and old['source'].startswith('explicit-mirrored-side-assumption:'):
            # Preserve the existing merchant symmetry choice ONLY in this QA copy.
            snapshot['annotations'][index]=dict(ann,part='temple-left',approved=True,source='explicit-mirrored-side-assumption:'+ann['source'])
    destination = OUT/'after'
    destination.mkdir(exist_ok=True)
    (destination/'snapshot.json').write_text(json.dumps(snapshot,indent=2))
    metadata = reconstruct(snapshot,destination)
    (OUT/'source-hash.json').write_text(json.dumps({'sha256':hashlib.sha256(Path(photo['path']).read_bytes()).hexdigest(),'photo':photo['id']},indent=2))
    print(metadata['parts'],metadata['checks'],result['warnings'])


if __name__=='__main__':
    main()
