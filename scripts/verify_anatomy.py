"""Reproducible, isolated CPU anatomy diagnostics; not a held-out SKU benchmark."""
from pathlib import Path
import json
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.reconstruction import extract_annotations  # noqa: E402


def main():
    output = ROOT / 'artifacts/anatomy'
    output.mkdir(parents=True, exist_ok=True)
    grey = cv2.imread(str(ROOT / 'tests/fixtures/grey_double_bridge/front.png'))
    offset = np.full((1900, 2500, 3), 255, np.uint8)
    offset[850:850+grey.shape[0], 1000:1000+grey.shape[1]] = grey
    cases = [('grey-front', 'front', grey), ('offset-grey-front', 'front', offset)]
    cases += [(f'rimless-{view}', view, cv2.imread(str(ROOT / f'tests/fixtures/rimless/{view}.png')))
              for view in ('front', 'right')]
    reports = []
    for name, view, image in cases:
        path = output / f'{name}-input.png'
        cv2.imwrite(str(path), image)
        photo = dict(id=name, view=view, path=str(path), width=image.shape[1], height=image.shape[0])
        start = time.perf_counter()
        result = extract_annotations(photo)
        elapsed = time.perf_counter() - start
        overlay = image.copy()
        for annotation in result['annotations']:
            points = np.rint([(p['x'], p['y']) for p in annotation['points']]).astype(np.int32)
            part = annotation['part']
            color = ((100, 170, 40) if part == 'front' else (230, 140, 20) if part.startswith('nose-pad')
                     else (180, 50, 200) if part == 'bridge-opening' else (30, 120, 230))
            cv2.polylines(overlay, [points], True, color, max(1, image.shape[1]//600), cv2.LINE_AA)
        cv2.imwrite(str(output / f'{name}-outline.png'), overlay)
        (output / f'{name}-report.json').write_text(json.dumps(result, indent=2) + '\n')
        report = dict(case=name, seconds=round(elapsed, 3), status=result['anatomy']['status'],
                      blocking=result['anatomy']['blocking'],
                      parts=[a['part'] for a in result['annotations']],
                      product_local=result.get('product_local'))
        reports.append(report)
        print(json.dumps(report))
    (output / 'summary.json').write_text(json.dumps(dict(
        scope='Existing-fixture derivatives; no production/generalization certification', cases=reports), indent=2) + '\n')
    if any(r['blocking'] or r['status'] != 'supported-structure' for r in reports):
        raise SystemExit('At least one diagnostic needs structural review; inspect the JSON and overlays.')


if __name__ == '__main__':
    main()
