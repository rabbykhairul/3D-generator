"""Isolated crossing regression evidence, without modifying saved projects."""
import argparse
import json
from pathlib import Path
import sys
import cv2
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.reconstruction import extract_annotations, reconstruct
from backend.reconstruction.render_mesh import welded_geometry
from render_fidelity_cpu import render
from PIL import Image, ImageDraw
import trimesh

parser = argparse.ArgumentParser()
parser.add_argument('--baseline', action='store_true')
args = parser.parse_args()
out = Path('artifacts/overlap-residue')
out.mkdir(parents=True, exist_ok=True)
snapshot = json.loads(Path('artifacts/blue-arm-fix/after/snapshot.json').read_text())
photo = next(p for p in snapshot['photos'] if p['view'] == 'right')
result = extract_annotations(photo)
label = 'before' if args.baseline else 'after'
destination = out/label
if args.baseline and destination.exists():
    raise RuntimeError('Baseline already preserved')
destination.mkdir(exist_ok=True)
(destination/'extraction.json').write_text(json.dumps(result, indent=2))
ann = result['annotations'][0]
for i, old in enumerate(snapshot['annotations']):
    if old['part'] == 'temple-right':
        snapshot['annotations'][i] = dict(ann, approved=True)
    elif old['part'] == 'temple-left' and old['source'].startswith('explicit-mirrored-side-assumption:'):
        snapshot['annotations'][i] = dict(ann, part='temple-left', approved=True,
            source='explicit-mirrored-side-assumption:'+ann['source'])
reconstruct(snapshot, destination)
if not args.baseline:
    before_ann = json.loads((out/'before/extraction.json').read_text())['annotations'][0]
    assert ann['anchors'] == before_ann['anchors']
    loaded = trimesh.load(destination/'model.glb', force='scene')
    for mesh in loaded.geometry.values():
        assert welded_geometry(mesh).is_watertight
        assert np.isfinite(mesh.vertices).all()
    original = cv2.imread(photo['path'])
    panels = []
    for name in ['before', 'after']:
        data = json.loads((out/name/'extraction.json').read_text())
        image = original.copy()
        points = np.round([[p['x'], p['y']] for p in data['annotations'][0]['points']]).astype(np.int32)
        cv2.polylines(image, [points], True, (35, 125, 230), 1, cv2.LINE_AA)
        panel = cv2.resize(image[505:600, 360:550], (760,380), interpolation=cv2.INTER_NEAREST)
        panel = cv2.copyMakeBorder(panel,35,0,0,0,cv2.BORDER_CONSTANT,value=(255,255,255))
        cv2.putText(panel,name.upper(),(10,25),cv2.FONT_HERSHEY_SIMPLEX,.65,(20,20,20),1)
        panels.append(panel)
    cv2.imwrite(str(out/'outline-comparison.png'), np.concatenate(panels,axis=1))
    target = np.array([-.066, -.025, -.074])
    # Side view chosen from generated geometry, shared between both assets.
    scene = trimesh.load(out/'before/model.glb', force='scene')
    target = scene.geometry['temple_right'].bounds.mean(axis=0)
    pair = Image.new('RGB',(1200,640),'white')
    draw=ImageDraw.Draw(pair)
    for i,name in enumerate(['before','after']):
        draw.text((i*600+10,10),name.upper()+' / reloaded GLB clay; CPU diagnostic',fill='black')
        pair.paste(render(out/name/'model.glb',target+[-.24,0,0],target,'clay',600),(i*600,40))
    pair.save(out/'mesh-comparison.png')
print(label, len(ann['points']), 'vertices', destination)
