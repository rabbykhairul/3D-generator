"""Local browser queue for reviewing collected eyewear component masks.

Drafts are suggestions. Only an explicit Approve action writes training masks.
Run with .venv/bin/python scripts/review_eyewear_collection.py and open the
printed localhost URL. This server never binds to a public interface.
"""
import argparse
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
from urllib.parse import unquote

from PIL import Image, ImageDraw
from shapely.geometry import Polygon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.eyewear_dataset import CLASSES, PART_CLASS, load_manifest, save_manifest

SOURCE = 'lenskart-research'
VIEWS = {'front', 'left', 'right'}
CONSTRUCTIONS = {'opaque-full-rim', 'transparent-full-rim', 'thin-metal',
                 'semi-rimless', 'rimless', 'sports-shield'}
HTML = Path(__file__).with_name('review_eyewear_collection.html')


def item_key(item):
    return item['photo_id']


def find_item(directory, photo_id):
    matches = [item for item in load_manifest(directory)['items'] if
               item['source'] == SOURCE and item_key(item) == photo_id]
    if len(matches) != 1:
        raise ValueError('Research photo not found')
    return matches[0]


def draft_path(directory, photo_id):
    return directory / 'review-drafts' / f'{photo_id}.json'


def summarize(item):
    with Image.open(item['image']) as image:
        width, height = image.size
    return ({key: item.get(key) for key in ('photo_id', 'product_id', 'model_code',
        'product_page_url', 'view', 'construction', 'reviewed', 'review_state', 'labels', 'absent')}
        | {'width': width, 'height': height, 'image_url': f"/api/image/{item['photo_id']}"})


def suggest(directory, photo_id, view):
    if view not in VIEWS:
        raise ValueError('Choose front, left, or right before generating a draft')
    item = find_item(directory, photo_id)
    from backend.reconstruction import extract_annotations
    photo = {'id': photo_id, 'view': view, 'path': item['image']}
    result = extract_annotations(photo, backend='heuristics')
    polygons = [{'part': ann['part'], 'points': ann['points']}
                for ann in result.get('annotations', []) if ann.get('part') in PART_CLASS
                and len(ann.get('points', [])) >= 3]
    suggested = result.get('construction')
    construction = suggested.get('category') if isinstance(suggested, dict) else suggested
    draft = {'view': view, 'construction': construction if
             construction in CONSTRUCTIONS else '',
             'polygons': polygons, 'warnings': result.get('warnings', []),
             'source': 'existing-cpu-contour-proposal', 'approved': False}
    target = draft_path(directory, photo_id)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(draft, indent=2) + '\n')
    return draft


def _validate_polygons(polygons, width, height):
    if not isinstance(polygons, list) or len(polygons) > 40:
        raise ValueError('Expected at most 40 component polygons')
    grouped = {}
    for entry in polygons:
        part = entry.get('part')
        points = entry.get('points')
        if part not in PART_CLASS or not isinstance(points, list) or not 3 <= len(points) <= 2000:
            raise ValueError(f'Invalid component outline: {part}')
        xy = [(float(p['x']), float(p['y'])) for p in points]
        if any(not 0 <= x < width or not 0 <= y < height for x, y in xy):
            raise ValueError(f'{part} leaves the source image')
        polygon = Polygon(xy)
        if not polygon.is_valid or polygon.area < 8:
            raise ValueError(f'{part} is self-intersecting or too small')
        grouped.setdefault(PART_CLASS[part], []).append(xy)
    return grouped


def approve(directory, photo_id, payload):
    view = payload.get('view')
    construction = payload.get('construction')
    if view not in VIEWS or construction not in CONSTRUCTIONS:
        raise ValueError('Select a view and frame construction')
    absent = payload.get('absent', [])
    if not isinstance(absent, list) or any(part not in CLASSES for part in absent):
        raise ValueError('Invalid absent component')
    manifest = load_manifest(directory)
    matches = [item for item in manifest['items'] if item['source'] == SOURCE
               and item_key(item) == photo_id]
    if len(matches) != 1:
        raise ValueError('Research photo not found')
    item = matches[0]
    with Image.open(item['image']) as source:
        size = source.size
    grouped = _validate_polygons(payload.get('polygons', []), *size)
    parts = {entry['part'] for entry in payload['polygons']}
    existing_lens = 'lens' in item.get('labels', {}) and item.get('reviewed')
    if 'front' in parts and not ({'lens-left', 'lens-right'} <= parts or existing_lens):
        raise ValueError('Outer frame approval needs both lens openings')
    if set(absent) & (set(grouped) | set(item.get('labels', {}))):
        raise ValueError('A component cannot be both outlined and absent')
    if set(item.get('absent', [])) & set(grouped):
        raise ValueError('A component was previously confirmed absent; correct that decision first')
    if not grouped:
        raise ValueError('Draw at least one verified component before approval')
    if construction == 'rimless' and 'frame' in grouped:
        raise ValueError('A rimless product cannot have a frame mask')
    if construction == 'rimless' and 'frame' not in absent:
        raise ValueError('Confirm frame absent for a rimless product')
    target_dir = directory / 'masks' / SOURCE / photo_id
    target_dir.mkdir(parents=True, exist_ok=True)
    masks = {}
    for name, entries in grouped.items():
        mask = Image.new('L', size, 0)
        draw = ImageDraw.Draw(mask)
        for points in entries:
            draw.polygon(points, fill=255)
        masks[name] = mask
    if 'frame' in masks and 'lens' not in masks and existing_lens:
        with Image.open(item['labels']['lens']) as existing:
            if existing.size != size:
                raise ValueError('Existing lens mask size differs from source image')
            masks['lens'] = existing.convert('L').copy()
    # The front contour denotes the outer support; lens interiors are holes.
    if 'frame' in masks and 'lens' in masks:
        masks['frame'].paste(0, mask=masks['lens'])
    labels = dict(item.get('labels', {}))
    for name, mask in masks.items():
        if name == 'lens' and name not in grouped:
            continue
        if not mask.getbbox():
            continue
        destination = target_dir / f'{name}.png'
        mask.save(destination)
        labels[name] = str(destination.resolve())
    if not labels:
        raise ValueError('No nonempty masks remained after validation')
    item.update(view=view, construction=construction, labels=labels,
                absent=sorted(set(absent) | set(item.get('absent', []))),
                reviewed=True, review_state='approved')
    save_manifest(directory, manifest)
    target = draft_path(directory, photo_id)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({'view': view, 'construction': construction,
        'polygons': payload['polygons'], 'absent': absent, 'approved': True}, indent=2) + '\n')
    return summarize(item)


def skip(directory, photo_id):
    manifest = load_manifest(directory)
    matches = [item for item in manifest['items'] if item['source'] == SOURCE
               and item_key(item) == photo_id]
    if len(matches) != 1:
        raise ValueError('Research photo not found')
    item = matches[0]
    if item.get('reviewed'):
        raise ValueError('Cannot skip an approved photo')
    item['review_state'] = 'skipped'
    save_manifest(directory, manifest)
    return summarize(item)


def save_draft(directory, photo_id, payload):
    item = find_item(directory, photo_id)
    with Image.open(item['image']) as source:
        _validate_polygons(payload.get('polygons', []), *source.size)
    if payload.get('view') not in VIEWS:
        raise ValueError('Select the photo view')
    construction = payload.get('construction')
    if construction and construction not in CONSTRUCTIONS:
        raise ValueError('Invalid frame construction')
    absent = payload.get('absent', [])
    if not isinstance(absent, list) or any(part not in CLASSES for part in absent):
        raise ValueError('Invalid absent component')
    target = draft_path(directory, photo_id)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({'view': payload['view'], 'construction': construction,
        'polygons': payload['polygons'], 'absent': absent, 'approved': False,
        'source': 'reviewer-edited-draft'}, indent=2) + '\n')
    return {'saved': True}


class Handler(BaseHTTPRequestHandler):
    directory = None

    def send_json(self, value, status=200):
        body = json.dumps(value).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        try:
            if self.path == '/':
                body = HTML.read_bytes()
                self.send_response(200)
                self.send_header('Content-Type', 'text/html; charset=utf-8')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if self.path == '/api/items':
                items = [summarize(item) for item in load_manifest(self.directory)['items']
                         if item['source'] == SOURCE]
                self.send_json({'items': items, 'counts': dict(Counter(
                    'approved' if x['reviewed'] else x.get('review_state', 'pending') for x in items))})
                return
            if self.path.startswith('/api/item/'):
                photo_id = unquote(self.path.removeprefix('/api/item/'))
                item = find_item(self.directory, photo_id)
                target = draft_path(self.directory, photo_id)
                self.send_json({'item': summarize(item), 'draft': json.loads(target.read_text())
                                if target.is_file() else None})
                return
            if self.path.startswith('/api/image/'):
                photo_id = unquote(self.path.removeprefix('/api/image/'))
                item = find_item(self.directory, photo_id)
                body = Path(item['image']).read_bytes()
                self.send_response(200)
                with Image.open(item['image']) as image:
                    mime = Image.MIME.get(image.format, 'application/octet-stream')
                self.send_header('Content-Type', mime)
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            self.send_json({'error': 'Not found'}, 404)
        except (ValueError, KeyError, OSError) as error:
            self.send_json({'error': str(error)}, 400)

    def do_POST(self):
        try:
            length = int(self.headers.get('Content-Length', 0))
            if not 0 < length < 2_000_000:
                raise ValueError('Invalid request size')
            payload = json.loads(self.rfile.read(length))
            if self.path.startswith('/api/draft/'):
                result = suggest(self.directory, unquote(self.path.removeprefix('/api/draft/')),
                                 payload.get('view'))
            elif self.path.startswith('/api/approve/'):
                result = approve(self.directory, unquote(self.path.removeprefix('/api/approve/')),
                                 payload)
            elif self.path.startswith('/api/skip/'):
                result = skip(self.directory, unquote(self.path.removeprefix('/api/skip/')))
            elif self.path.startswith('/api/save-draft/'):
                result = save_draft(self.directory,
                                    unquote(self.path.removeprefix('/api/save-draft/')), payload)
            else:
                self.send_json({'error': 'Not found'}, 404)
                return
            self.send_json(result)
        except (ValueError, KeyError, TypeError, OSError) as error:
            self.send_json({'error': str(error)}, 400)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=ROOT / 'artifacts/eyewear-training')
    parser.add_argument('--port', type=int, default=5193)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error('Invalid port')
    Handler.directory = args.dataset.resolve()
    if not load_manifest(Handler.directory)['items']:
        parser.error('No staged photos; import a collection first')
    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    print(f'Review queue: http://127.0.0.1:{args.port}/', flush=True)
    server.serve_forever()


if __name__ == '__main__':
    main()
