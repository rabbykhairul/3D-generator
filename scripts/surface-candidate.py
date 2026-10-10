"""Offline candidate from fresh suggestions: never writes the saved project."""
import copy
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend import store
from backend.reconstruction import extract_annotations, reconstruct

project = copy.deepcopy(store.get_project(sys.argv[1]))
annotations = []
for view in ['front', 'right', 'left']:
    photo = next((p for p in reversed(project['photos']) if p['view'] == view), None)
    if photo is None:
        continue
    result = extract_annotations(photo)
    for ann in result['annotations']:
        ann.update(approved=True, source='offline-reviewed-candidate:' + ann['source'])
        annotations.append(ann)
        if view in ['right', 'left']:
            opposite = 'temple-left' if view == 'right' else 'temple-right'
            if any(a['part'] == opposite and 'mirrored' in a.get('source', '') for a in project['annotations']):
                mirrored = copy.deepcopy(ann)
                mirrored.update(part=opposite, source='explicit-mirrored-side-assumption:' + ann['source'])
                annotations.append(mirrored)
project['annotations'] = annotations
out = Path('artifacts/surface-fix/candidate')
metadata = reconstruct(project, out)
(out / 'snapshot.json').write_text(json.dumps(project, indent=2))
print(json.dumps({'parts': metadata['parts'], 'elapsed': metadata['elapsed_seconds'], 'checks': metadata['checks']}, indent=2))
