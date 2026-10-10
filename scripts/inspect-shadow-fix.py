"""Read-only comparison of saved approved front versus new extraction.

Usage: .venv/bin/python scripts/inspect-shadow-fix.py PROJECT_ID
Writes diagnostic assets only; does not approve outlines or modify the project.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PIL import Image, ImageDraw
from backend import store
from backend.reconstruction import extract_annotations

project = store.get_project(sys.argv[1])
photo = next(p for p in reversed(project['photos']) if p['view'] == 'front')
result = extract_annotations(photo)
out = Path('artifacts/assembly-fix')
out.mkdir(parents=True, exist_ok=True)
with Image.open(photo['path']) as source:
    for label, annotations in [('before', project['annotations']), ('after', result['annotations'])]:
        image = source.convert('RGB')
        draw = ImageDraw.Draw(image)
        for ann in annotations:
            if ann['photo_id'] == photo['id']:
                points = [(p['x'], p['y']) for p in ann['points']]
                draw.line([*points, points[0]], fill='#d63636' if label == 'before' else '#00a775', width=2)
        image.crop((20, 395, 1060, 760)).save(out / f'{label}-front-contours.png')
(out / 'shadow-extraction.json').write_text(json.dumps(result, indent=2))
print(json.dumps({k: v for k, v in result.items() if k != 'annotations'}, indent=2))
