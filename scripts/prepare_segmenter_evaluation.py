"""Download publisher checkpoints and inventory saved photos for CPU evaluation.

No image uploads, project changes or automatic production integration.
Run using the app environment, then evaluate manifests with Python 3.12+.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend import store
from backend.reconstruction.anatomy import FRONT_VIEWS, SIDE_VIEWS

SOURCE = 'https://github.com/mantasu/glasses-detector/releases/download/v1.0.0/'
ARCHITECTURES = {'small': 'tinysegnet_v1', 'medium': 'lraspp_mobilenet_v3_large'}


def saved_cases(database):
    """Read-only inventory; keep duplicate photos from inflating sample counts."""
    with sqlite3.connect(database.resolve().as_uri() + '?mode=ro', uri=True) as conn:
        projects = [json.loads(row[0]) for row in conn.execute('SELECT payload FROM projects')]
    cases, seen = [], set()
    for project in sorted(projects, key=lambda p: p['updated_at'], reverse=True):
        if project.get('archived_at'):
            continue
        for photo in project['photos']:
            if photo.get('role') != 'fitting' or photo['view'] not in FRONT_VIEWS | SIDE_VIEWS:
                continue
            path = Path(photo['path']).resolve()
            if not path.is_file():
                continue
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            cases.append(dict(sku=project['id'], name=project['name'], photo_id=photo['id'],
                              view=photo['view'], path=str(path), sha256=digest,
                              family=project['category'], expected_masks={}))
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'artifacts/segmenter-evaluation')
    args = parser.parse_args()
    directory = args.output.resolve()
    weights = directory/'weights'
    weights.mkdir(parents=True, exist_ok=True)
    # Preserve the publisher notice alongside the local checkpoints.
    license_url = 'https://raw.githubusercontent.com/mantasu/glasses-detector/v1.0.4/LICENSE'
    with urllib.request.urlopen(license_url, timeout=60) as response:
        (directory/'PUBLISHER_LICENSE.txt').write_bytes(response.read())
    cases = saved_cases(store.DATA/'studio.sqlite3')
    for size, architecture in ARCHITECTURES.items():
        checkpoints = {}
        for kind in ('frames', 'lenses', 'legs'):
            filename = f'segmentation_{kind}_{architecture}.pth'
            path = weights/filename
            url = SOURCE + filename
            if not path.exists():
                temporary = path.with_suffix('.download')
                with urllib.request.urlopen(url, timeout=120) as response:
                    temporary.write_bytes(response.read())
                temporary.replace(path)
            checkpoints[kind] = dict(path=str(path.relative_to(directory)), size=size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(), source_url=url,
                license_review='Publisher repository MIT notice retained in PUBLISHER_LICENSE.txt. '
                    'Local research evaluation only; separate checkpoint/training-data rights '
                    'are not established for commercial redistribution.')
        manifest = dict(split='development', independent_labels=False,
                        candidate={'checkpoints': checkpoints}, cases=cases)
        (directory/f'{size}.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(dict(unique_photos=len(cases), output=str(directory)), indent=2))


if __name__ == '__main__':
    main()
