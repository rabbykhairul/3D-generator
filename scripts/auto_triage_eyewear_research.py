"""Batch-generate conservative, unapproved component candidates.

This reduces review work but never turns proposals into training ground truth.
Clean-background and bilateral-geometry checks reject many lifestyle and
angled photos. The report records every reason a photo was withheld.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
from shapely.geometry import Polygon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.eyewear_dataset import PART_CLASS, load_manifest
from scripts.review_eyewear_collection import SOURCE, draft_path


def clean_background(path):
    with Image.open(path) as source:
        image = source.convert('RGB')
        image.thumbnail((256, 256))
        rgb = np.asarray(image)
    h, w = rgb.shape[:2]
    corners = np.concatenate([rgb[:h // 5, :w // 5].reshape(-1, 3),
                              rgb[:h // 5, -w // 5:].reshape(-1, 3),
                              rgb[-h // 5:, :w // 5].reshape(-1, 3),
                              rgb[-h // 5:, -w // 5:].reshape(-1, 3)])
    return float((corners.min(axis=1) >= 235).mean()) >= .82


def paired_front(annotations, width, height):
    parts = {entry.get('part'): entry for entry in annotations}
    if not {'front', 'lens-left', 'lens-right'} <= parts.keys():
        return False
    try:
        outer, left, right = [Polygon([(p['x'], p['y']) for p in parts[name]['points']])
                              for name in ('front', 'lens-left', 'lens-right')]
    except (KeyError, TypeError, ValueError):
        return False
    if not all(p.is_valid for p in (outer, left, right)):
        return False
    if not .7 <= left.area / max(right.area, 1) <= 1.43:
        return False
    if not left.centroid.x < width * .48 < right.centroid.x:
        return False
    if abs((left.centroid.x + right.centroid.x) / 2 - width / 2) > width * .07:
        return False
    if abs(left.centroid.y - right.centroid.y) > height * .05:
        return False
    if outer.bounds[2] - outer.bounds[0] < width * .55:
        return False
    return True


def propose(item):
    from backend.reconstruction import extract_annotations
    photo = {'id': item['photo_id'], 'path': item['image'], 'view': 'front'}
    with Image.open(item['image']) as image:
        width, height = image.size
    result = extract_annotations(photo, backend='heuristics')
    if paired_front(result.get('annotations', []), width, height):
        view = 'front'
    else:
        photo['view'] = 'right'
        result = extract_annotations(photo, backend='heuristics')
        if not any(a.get('part') == 'temple-right' for a in result.get('annotations', [])):
            return None, 'no-stable-front-or-side-outline'
        view = 'right'
    annotations = [{'part': a['part'], 'points': a['points']}
                   for a in result.get('annotations', []) if a.get('part') in PART_CLASS
                   and len(a.get('points', [])) >= 3]
    suggested = result.get('construction')
    construction = suggested.get('category') if isinstance(suggested, dict) else suggested
    return {'view': view, 'construction': construction or '', 'polygons': annotations,
            'warnings': result.get('warnings', []), 'source': 'automatic-cpu-triage',
            'approved': False}, None


def run(directory, limit=0):
    items = [item for item in load_manifest(directory)['items'] if item['source'] == SOURCE]
    if limit:
        items = items[:limit]
    rows = []
    for index, item in enumerate(items, 1):
        row = {'photo_id': item['photo_id'], 'product_id': item['product_id']}
        if item.get('reviewed') or item.get('review_state') == 'skipped':
            row['status'] = 'already-decided'
        elif draft_path(directory, item['photo_id']).is_file():
            row['status'] = 'existing-draft'
        elif not clean_background(item['image']):
            row['status'] = 'withheld'; row['reason'] = 'non-white-background'
        else:
            try:
                draft, reason = propose(item)
                if draft:
                    target = draft_path(directory, item['photo_id'])
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text(json.dumps(draft, indent=2) + '\n')
                    row.update(status='proposed', view=draft['view'],
                               parts=[p['part'] for p in draft['polygons']])
                else:
                    row.update(status='withheld', reason=reason)
            except Exception as error:
                row.update(status='error', reason=f'{type(error).__name__}: {error}'[:250])
        rows.append(row)
        print(f"{index}/{len(items)} {row['status']} {item['photo_id']}", flush=True)
    report = directory / 'auto-triage-report.json'
    report.write_text(json.dumps({'schema': 1, 'training_labels_created': 0,
        'note': 'Unapproved CPU proposals are not ground truth or an accuracy estimate.',
        'items': rows}, indent=2) + '\n')
    return report, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=ROOT / 'artifacts/eyewear-training')
    parser.add_argument('--limit', type=int, default=0, help='0 means all staged research photos')
    args = parser.parse_args()
    if args.limit < 0:
        parser.error('--limit cannot be negative')
    report, rows = run(args.dataset.resolve(), args.limit)
    from collections import Counter
    print(json.dumps({'report': str(report), 'statuses': dict(Counter(x['status'] for x in rows))}, indent=2))


if __name__ == '__main__':
    main()
