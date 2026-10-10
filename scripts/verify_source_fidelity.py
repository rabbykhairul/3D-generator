"""Source-fidelity before/after on an isolated project copy, no approval changes.

CPU diagnostic is not a browser/PBR or independent commercial-quality benchmark.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import trimesh
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.store import get_project  # noqa: E402
from backend.reconstruction import extract_annotations, reconstruct  # noqa: E402
from scripts.render_fidelity_cpu import render  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('project_id')
    parser.add_argument('--output', type=Path, default=ROOT/'artifacts/source-fidelity')
    parser.add_argument('--reanalyze-sides', action='store_true', help='Re-extract saved arm photos in the isolated after copy')
    args = parser.parse_args()
    project = get_project(args.project_id)
    if project is None:
        parser.error('Project not found')
    before = deepcopy(project)
    front = next(p for p in reversed(project['photos']) if p['view']=='front' and p['role']=='fitting')
    result = extract_annotations(front)
    after = deepcopy(project)
    # Automatic review here is confined to an isolated diagnostic copy. Never
    # persist these suggestions or replace the merchant's approved annotations.
    after['annotations'] = [a for a in after['annotations'] if a['photo_id']!=front['id']]
    after['annotations'] += [dict(a, approved=True) for a in result['annotations']]
    side_results={}
    if args.reanalyze_sides:
        for annotation in after['annotations']:
            if not annotation['part'].startswith('temple-'):
                continue
            photo=next(p for p in project['photos'] if p['id']==annotation['photo_id'])
            if photo['id'] not in side_results:
                side_results[photo['id']]=(photo,extract_annotations(photo))
            candidates=side_results[photo['id']][1]['annotations']
            arm=next((a for a in candidates if a['part'].startswith('temple-')),None)
            if arm is not None:
                annotation['points']=deepcopy(arm['points'])
                annotation['anchors']=deepcopy(arm['anchors'])
    output = args.output.resolve()
    before_meta = reconstruct(before, output/'before')
    after_meta = reconstruct(after, output/'after')
    image = cv2.imread(front['path'])
    for a in result['annotations']:
        points = np.rint([(p['x'], p['y']) for p in a['points']]).astype(np.int32)
        color = (220, 110, 20) if a['part'].startswith('nose-pad-') else (60, 180, 60)
        cv2.polylines(image, [points], True, color, 2, cv2.LINE_AA)
    cv2.imwrite(str(output/'source-outlines.png'), image)
    for photo,side_result in side_results.values():
        side_image=cv2.imread(photo['path'])
        for a in side_result['annotations']:
            points=np.rint([(p['x'],p['y']) for p in a['points']]).astype(np.int32)
            cv2.polylines(side_image,[points],True,(30,120,235),2,cv2.LINE_AA)
            for name,p in a['anchors'].items():
                cv2.circle(side_image,(round(p['x']),round(p['y'])),5,(60,160,60),-1)
        cv2.imwrite(str(output/f'side-outlines-{photo["id"]}.png'),side_image)
    before_path, after_path = output/'before/model.glb', output/'after/model.glb'
    target = trimesh.load(before_path, force='scene', process=False).bounds.mean(axis=0)
    eye = target + [0, 0, .37]
    pair = Image.new('RGB', (1440, 760), 'white')
    draw = ImageDraw.Draw(pair)
    draw.text((15, 10), 'BEFORE / AFTER — identical front camera; CPU diagnostic, not PBR', fill='black')
    for index, path in enumerate((before_path, after_path)):
        pair.paste(render(path, eye, target, 'photo', 720, include_lenses=True), (720*index, 40))
    pair.save(output/'front-before-after.png')
    if side_results:
        side_eye=target+[-.37,0,.02]
        draw.rectangle((0,0,1440,40),fill='white')
        draw.text((15,10),'SAVED / RE-ANALYZED outlines — identical side camera; CPU diagnostic, not PBR',fill='black')
        for index,path in enumerate((before_path,after_path)):
            pair.paste(render(path,side_eye,target,'photo',720,include_lenses=True),(720*index,40))
        pair.save(output/'side-before-after.png')
    report = dict(scope='Reviewed fixture regression; not unseen-SKU or commercial equivalence certification',
                  project_id=project['id'], source_photo_id=front['id'],
                  before_parts=before_meta['parts'], after_parts=after_meta['parts'],
                  checks=after_meta['checks'], nose_support=result.get('nose_support'),
                  camera=dict(eye=eye.tolist(), target=target.tolist()),
                  merchant_project_modified=False,
                  remaining=['Unknown depth, size defaults and mirrored opposite arm remain assumptions.',
                             'No stronger trained part recognizer or joint multi-view optimizer is introduced in this first delivery.',
                             'Clear material fidelity requires browser/PBR acceptance.'])
    (output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__=='__main__':
    main()
