"""Save before/after side-outline overlays and width measurements for review."""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from shapely.geometry import LineString, Polygon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.reconstruction.engine import _faint_side_suggestion, _foreground_mask
from backend.reconstruction.side_extraction import extract_side, recover_pale_side


OUT = ROOT / 'artifacts/new-outline-fix'
PHOTOS = {
    'new': ROOT / 'data/projects/ce81f4ffd03e4b8cac759b32ec5ee650/photos/071de32f5a964f5796f996d0fc31524c.png',
    'gray': ROOT / 'data/projects/d9cb85ea817b4a7d968e31135814cab4/photos/3b91344955784a8baec74adeeffebe4b.png',
    'lavender': ROOT / 'frontend/public/upload-guide/side.png',
}


def polygon(annotation):
    return Polygon([(p['x'], p['y']) for p in annotation['points']])


def section(poly, x, height):
    bounds = poly.intersection(LineString([(x, 0), (x, height)])).bounds
    return [round(bounds[1], 2), round(bounds[3], 2)] if len(bounds) == 4 else None


def overlay(image, annotation, label):
    result = image.copy()
    points = np.round([[p['x'], p['y']] for p in annotation['points']]).astype(np.int32)
    cv2.polylines(result, [points], True, (5, 100, 230), 2, cv2.LINE_AA)
    for name, point in annotation['anchors'].items():
        center = (round(point['x']), round(point['y']))
        cv2.circle(result, center, 5, (205, 20, 180), -1)
        cv2.putText(result, name, (center[0] + 6, center[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, .45, (70, 45, 110), 1, cv2.LINE_AA)
    cv2.putText(result, label, (24, 44), cv2.FONT_HERSHEY_SIMPLEX,
                .8, (30, 30, 30), 2, cv2.LINE_AA)
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    report = {}
    for name, path in PHOTOS.items():
        image = cv2.imread(str(path))
        if image is None:
            raise FileNotFoundError(path)
        height, width = image.shape[:2]
        photo = {'id': name, 'view': 'right', 'path': str(path),
                 'width': width, 'height': height}
        mask, _ = _foreground_mask(image)
        standard = extract_side(photo, mask, 1, 1)
        before = standard if standard['annotations'] else _faint_side_suggestion(image, photo, 1, 1)
        recovered = recover_pale_side(image, photo, 1, 1)
        after = standard if standard['annotations'] else recovered or before
        if before is None or after is None or not before['annotations'] or not after['annotations']:
            raise AssertionError(f'{name}: no comparison contour')
        old_ann, new_ann = before['annotations'][0], after['annotations'][0]
        old, new = polygon(old_ann), polygon(new_ann)
        if not old.is_valid or not new.is_valid:
            raise AssertionError(f'{name}: invalid polygon')
        sample_xs = (100, 300, 400, 500, 600, 700, 800, 850, 900)
        report[name] = {
            'photo': str(path.relative_to(ROOT)),
            'before_source': old_ann['source'], 'after_source': new_ann['source'],
            'before_anchors': old_ann['anchors'], 'after_anchors': new_ann['anchors'],
            'before_area_px': round(old.area, 1), 'after_area_px': round(new.area, 1),
            'sections': {str(x): {'before': section(old, x, height),
                                  'after': section(new, x, height)} for x in sample_xs},
        }
        panels = np.concatenate((overlay(image, old_ann, 'Before'),
                                 overlay(image, new_ann, 'After')), axis=1)
        cv2.imwrite(str(OUT / f'{name}-side-comparison.png'), panels)
    (OUT / 'side-comparison.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({name: {'before': item['before_anchors'],
                             'after': item['after_anchors'],
                             'area_ratio': round(item['after_area_px'] / item['before_area_px'], 2)}
                      for name, item in report.items()}, indent=2))


if __name__ == '__main__':
    main()
