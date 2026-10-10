"""Local, read-only project extraction evidence; never changes merchant approvals."""
import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend import store
from backend.reconstruction import extract_annotations, reconstruct


def render(photo, result, output):
    image = cv2.imread(photo['path'])
    for ann in result['annotations']:
        points = np.array([[p['x'], p['y']] for p in ann['points']], np.int32)
        color = (94, 161, 57) if ann['part'] == 'front' else (24, 116, 220)
        cv2.polylines(image, [points], True, color, 2, cv2.LINE_AA)
        for anchor in ann.get('anchors', {}).values():
            cv2.circle(image, (round(anchor['x']), round(anchor['y'])), 5, (180, 20, 200), -1)
    cv2.imwrite(str(output), image)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--label', default='after')
    parser.add_argument('--generate', action='store_true')
    args = parser.parse_args()
    project = store.get_project('ce81f4ffd03e4b8cac759b32ec5ee650')
    out = Path('artifacts/new-outline-fix')
    out.mkdir(parents=True, exist_ok=True)
    extracted = []
    hashes = {}
    for photo in project['photos']:
        if photo['view'] not in ('front', 'right'):
            continue
        result = extract_annotations(photo)
        extracted.extend(result['annotations'])
        hashes[photo['view']] = hashlib.sha256(Path(photo['path']).read_bytes()).hexdigest()
        (out / f'{photo["view"]}-{args.label}.json').write_text(json.dumps(result, indent=2))
        render(photo, result, out / f'{photo["view"]}-{args.label}.png')
        print(photo['view'], result.get('segmentation'), result['warnings'])
        if args.label == 'after':
            original = cv2.imread(photo['path'])
            foreground = np.any(original < 230, axis=2)
            ys, xs = np.where(foreground)
            y0, y1 = max(0, ys.min()-20), min(original.shape[0], ys.max()+21)
            x0, x1 = max(0, xs.min()-20), min(original.shape[1], xs.max()+21)
            tiles = []
            for label in ('before', 'after'):
                overlay = cv2.imread(str(out/f'{photo["view"]}-{label}.png'))
                crop = overlay[y0:y1, x0:x1]
                tile = cv2.copyMakeBorder(crop, 40, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255,255,255))
                cv2.putText(tile, label.upper()+' - exact same uploaded photo', (12,26), cv2.FONT_HERSHEY_SIMPLEX, .65, (30,30,30), 1, cv2.LINE_AA)
                tiles.append(tile)
            cv2.imwrite(str(out/f'{photo["view"]}-comparison.png'), np.concatenate(tiles, axis=1))
    if args.generate:
        # Explicit QA-only approval in an isolated artifact snapshot, never store.save_project.
        snapshot = copy.deepcopy(project)
        snapshot['annotations'] = [dict(a, approved=True, source='merchant-reviewed:'+a['source']) for a in extracted]
        snapshot['artifact'] = None
        destination = out/'generated'
        destination.mkdir(parents=True, exist_ok=True)
        (destination/'snapshot.json').write_text(json.dumps(snapshot, indent=2))
        metadata = reconstruct(snapshot, destination)
        (out/'source-hashes.json').write_text(json.dumps(hashes, indent=2))
        print('Generated isolated QA asset', metadata['parts'], metadata['checks'])
        from render_fidelity_cpu import render
        target = np.array([0., 0., -.05])
        views = [('Front', np.array([0., 0., .36])),
                 ('Right side', np.array([-.36, .01, -.05])),
                 ('Three-quarter', np.array([-.24, .055, .29]))]
        sheet = Image.new('RGB', (1440, 1050), 'white')
        draw = ImageDraw.Draw(sheet)
        draw.text((12, 10), 'EXPORTED / RELOADED GLB - CPU diagnostic, not browser or live AR. Clear lenses omitted for geometry inspection.', fill='black')
        for row, surface in enumerate(('photo','clay')):
            for col, (name, eye) in enumerate(views):
                draw.text((col*480+12, row*510+36), name+' / '+surface+' / only reviewed right arm', fill='black')
                sheet.paste(render(destination/'model.glb', eye, target, surface, 480), (col*480, row*510+60))
        sheet.save(out/'generated-views.png')
