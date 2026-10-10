"""Build and audit a product-split eyewear component dataset.

Public masks can be imported one class at a time. Missing labels stay unknown;
they are never converted into negative examples. Nothing here approves a model.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import sqlite3

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ('lens', 'frame', 'bridge', 'nose-pad', 'hinge', 'temple', 'lens-mount')
PRODUCT_CODE = re.compile(r'(?i)(?:vc-s\d+(?:-c\d+)?|jj-s\d+-c\d+|l[rb]-e\d+)')
PART_CLASS = {'lens-left': 'lens', 'lens-right': 'lens', 'front': 'frame',
              'front-support': 'frame', 'bridge': 'bridge',
              'bridge-hardware': 'bridge', 'nose-pad-left': 'nose-pad',
              'nose-pad-right': 'nose-pad', 'hinge-left': 'hinge',
              'hinge-right': 'hinge', 'temple-left': 'temple',
              'temple-right': 'temple', 'lens-mount-left': 'lens-mount',
              'lens-mount-right': 'lens-mount'}


def split_for(product_id):
    bucket = int(hashlib.sha256(product_id.encode()).hexdigest()[:8], 16) % 10
    return 'train' if bucket < 7 else 'val' if bucket < 9 else 'test'


def load_manifest(directory):
    path = directory / 'manifest.json'
    if path.is_file():
        data = json.loads(path.read_text())
        if data.get('schema') == 1 and data.get('classes') == list(CLASSES[:-1]):
            data['classes'] = list(CLASSES)
        if data.get('schema') != 1 or data.get('classes') != list(CLASSES):
            raise ValueError('Dataset schema or classes do not match this trainer')
        return data
    return {'schema': 1, 'classes': list(CLASSES), 'items': []}


def save_manifest(directory, manifest):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


def _mask(size, polygons):
    output = Image.new('L', size, 0)
    draw = ImageDraw.Draw(output)
    for polygon in polygons:
        if len(polygon) >= 3:
            draw.polygon([(float(p['x']), float(p['y'])) for p in polygon], fill=255)
    return output


def export_approved(database, directory):
    with sqlite3.connect(f'file:{database.resolve()}?mode=ro', uri=True) as conn:
        projects = [json.loads(row[0]) for row in conn.execute('SELECT payload FROM projects')]
    manifest = load_manifest(directory)
    existing = {(item['source'], item['product_id'], item['photo_id']) for item in manifest['items']}
    added = 0
    for project in projects:
        if project.get('archived_at'):
            continue
        for photo in project.get('photos', []):
            if photo.get('role') != 'fitting' or photo.get('view') not in {'front', 'left', 'right'}:
                continue
            source_path = Path(photo.get('path', ''))
            if not source_path.is_file():
                continue
            key = ('approved-local', project['id'], photo['id'])
            if key in existing:
                continue
            annotations = [a for a in project.get('annotations', [])
                           if a.get('photo_id') == photo['id'] and a.get('approved')
                           and a.get('part') in PART_CLASS and len(a.get('points', [])) >= 3]
            if not annotations:
                continue
            with Image.open(source_path) as image:
                size = image.size
            grouped = defaultdict(list)
            for annotation in annotations:
                grouped[PART_CLASS[annotation['part']]].append(annotation['points'])
            # A front silhouette includes the lenses. Remove their interiors
            # only when both optical outlines were actually approved.
            parts = {a['part'] for a in annotations}
            if 'frame' in grouped and not {'lens-left', 'lens-right'} <= parts:
                del grouped['frame']
            if not grouped:
                continue
            mask_dir = directory / 'masks' / project['id'] / photo['id']
            mask_dir.mkdir(parents=True, exist_ok=True)
            labels = {}
            for name, polygons in grouped.items():
                mask = _mask(size, polygons)
                if name == 'frame':
                    lens = _mask(size, grouped.get('lens', []))
                    mask.paste(0, mask=lens)
                if not mask.getbbox():
                    continue
                path = mask_dir / f'{name}.png'
                mask.save(path)
                labels[name] = str(path.resolve())
            if not labels:
                continue
            manifest['items'].append({'source': 'approved-local', 'product_id': project['id'],
                'photo_id': photo['id'], 'view': photo['view'], 'construction': project.get('category'),
                'image': str(source_path.resolve()), 'labels': labels, 'absent': [],
                'reviewed': True, 'split': split_for(project['id'])})
            added += 1
    save_manifest(directory, manifest)
    return added


def import_public(directory, images, masks, component, source_name, license_note):
    if component not in CLASSES:
        raise ValueError(f'Unsupported class: {component}')
    if not license_note.strip():
        raise ValueError('Record the source dataset terms before importing')
    manifest = load_manifest(directory)
    image_by_stem = {p.stem: p for p in images.iterdir() if p.is_file() and p.suffix.lower() in
                     {'.jpg', '.jpeg', '.png', '.webp'}}
    existing = {(i['source'], i['photo_id']): i for i in manifest['items']}
    added = 0
    for mask in sorted(masks.iterdir()):
        image = image_by_stem.get(mask.stem)
        if not image or mask.suffix.lower() != '.png':
            continue
        with Image.open(image) as photo, Image.open(mask) as label:
            if photo.size != label.size or not label.convert('L').getbbox():
                continue
        product_id = f'{source_name}:{mask.stem}'
        current = existing.get((source_name, mask.stem))
        if current:
            if Path(current['image']).resolve() != image.resolve():
                raise ValueError(f'Image collision for {source_name}:{mask.stem}')
            if component not in current['labels']:
                current['labels'][component] = str(mask.resolve())
                current['reviewed'] = False
                added += 1
            continue
        item = {'source': source_name, 'product_id': product_id,
            'photo_id': mask.stem, 'view': 'unknown', 'construction': None,
            'image': str(image.resolve()), 'labels': {component: str(mask.resolve())},
            'absent': [], 'reviewed': False, 'split': split_for(product_id),
            'license_note': license_note}
        manifest['items'].append(item)
        existing[(source_name, mask.stem)] = item
        added += 1
    save_manifest(directory, manifest)
    return added


def import_coco(directory, images, annotations, category_map, source_name, license_note):
    """Import polygon COCO masks as unreviewed, partially labeled photos.

    COCO category names must be mapped explicitly. RLE annotations are skipped
    rather than silently treated as absent or incorrectly decoded.
    """
    if not license_note.strip():
        raise ValueError('Record the source dataset terms before importing')
    if not re.fullmatch(r'[A-Za-z0-9_-]+', source_name):
        raise ValueError('COCO source name must contain only letters, digits, underscores or hyphens')
    if not category_map or any(name not in CLASSES for name in category_map.values()):
        raise ValueError('Map at least one COCO category to a supported class')
    payload = json.loads(annotations.read_text())
    category_ids = {entry['id']: category_map[entry['name']]
                    for entry in payload.get('categories', [])
                    if entry['name'] in category_map}
    if not category_ids:
        raise ValueError('No mapped categories appear in COCO annotations')
    manifest = load_manifest(directory)
    existing = {(item['source'], item['photo_id']): item for item in manifest['items']}
    polygons_by_image = defaultdict(lambda: defaultdict(list))
    unsupported_rle = 0
    for annotation in payload.get('annotations', []):
        component = category_ids.get(annotation.get('category_id'))
        if not component:
            continue
        segmentation = annotation.get('segmentation')
        if not isinstance(segmentation, list):
            unsupported_rle += 1
            continue
        for flat in segmentation:
            if len(flat) >= 6 and len(flat) % 2 == 0:
                polygons_by_image[annotation['image_id']][component].append(
                    list(zip(flat[0::2], flat[1::2])))
    added = 0
    image_root = images.resolve()
    for entry in payload.get('images', []):
        grouped = polygons_by_image.get(entry['id'])
        if not grouped:
            continue
        image = (images / entry['file_name']).resolve()
        if not image.is_relative_to(image_root) or not image.is_file():
            raise ValueError(f"Invalid COCO image path: {entry['file_name']}")
        photo_id = f"{entry['id']}-{Path(entry['file_name']).stem}"
        if (source_name, photo_id) in existing:
            continue
        with Image.open(image) as photo:
            size = photo.size
        if size != (entry['width'], entry['height']):
            raise ValueError(f'COCO dimensions disagree with photo: {image}')
        labels = {}
        mask_dir = directory / 'masks' / source_name / photo_id
        mask_dir.mkdir(parents=True, exist_ok=True)
        for component, polygons in grouped.items():
            mask = Image.new('L', size, 0)
            draw = ImageDraw.Draw(mask)
            for polygon in polygons:
                draw.polygon(polygon, fill=255)
            if mask.getbbox():
                path = mask_dir / f'{component}.png'
                mask.save(path)
                labels[component] = str(path.resolve())
        if not labels:
            continue
        product_id = f'{source_name}:{photo_id}'
        manifest['items'].append({
            'source': source_name, 'product_id': product_id, 'photo_id': photo_id,
            'view': 'unknown', 'construction': None, 'image': str(image),
            'labels': labels, 'absent': [], 'reviewed': False,
            'split': split_for(product_id), 'license_note': license_note,
            'coco_annotations': str(annotations.resolve()),
        })
        added += 1
    save_manifest(directory, manifest)
    return added, unsupported_rle


def stage_local_photos(directory, images, source_name='local-candidate'):
    """Inventory eyewear photos without treating filenames or pixels as labels.

    Product codes only group candidate views. They need human verification
    before any item is marked reviewed or included in model training.
    """
    manifest = load_manifest(directory)
    existing_hashes = {item.get('sha256') for item in manifest['items'] if item.get('sha256')}
    existing_keys = {(item['source'], item['photo_id']) for item in manifest['items']}
    added = 0
    skipped = []
    for path in sorted(images.iterdir()):
        if not path.is_file() or path.suffix.lower() not in {'.jpg', '.jpeg', '.png', '.webp'}:
            continue
        product = PRODUCT_CODE.search(path.name)
        if not product:
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest in existing_hashes:
            skipped.append(path.name)
            continue
        try:
            with Image.open(path) as image:
                image.verify()
        except Exception:
            skipped.append(path.name)
            continue
        photo_id = digest[:24]
        if (source_name, photo_id) in existing_keys:
            continue
        product_id = f'catalog:{product.group(0).lower()}'
        manifest['items'].append({
            'source': source_name, 'product_id': product_id, 'photo_id': photo_id,
            'view': 'unknown', 'construction': None, 'image': str(path.resolve()),
            'labels': {}, 'absent': [], 'reviewed': False,
            'split': split_for(product_id), 'sha256': digest,
            'license_note': 'Rights for training and commercial use not yet verified',
        })
        existing_hashes.add(digest)
        existing_keys.add((source_name, photo_id))
        added += 1
    save_manifest(directory, manifest)
    return added, skipped


def import_research_collection(directory, collection, source_name='lenskart-research'):
    """Stage collected product views without inventing component labels."""
    payload = json.loads(collection.read_text())
    if payload.get('schema') != 1 or not isinstance(payload.get('products'), list):
        raise ValueError('Unsupported research collection manifest')
    manifest = load_manifest(directory)
    existing = {(item['source'], item['photo_id']) for item in manifest['items']}
    product_splits = {item['product_id']: item['split'] for item in manifest['items']}
    added = 0
    for product in payload['products']:
        if product.get('status') != 'collected':
            continue
        code = product['product_id']
        if not re.fullmatch(r'[a-z0-9_-]+', code):
            raise ValueError(f'Invalid research product ID: {code}')
        product_id = f'catalog:{code}'
        model_code = product.get('model_code', code)
        if not re.fullmatch(r'[a-z0-9_-]+', model_code):
            raise ValueError(f'Invalid research model code: {model_code}')
        split = product_splits.get(product_id, split_for(f'catalog:{model_code}'))
        for photo in product.get('photos', []):
            path = Path(photo['path']).resolve()
            if not path.is_file():
                raise ValueError(f'Missing collected product photo: {path}')
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != photo['sha256']:
                raise ValueError(f'Collected product photo changed: {path}')
            photo_id = digest[:24]
            if (source_name, photo_id) in existing:
                continue
            manifest['items'].append({
                'source': source_name, 'product_id': product_id, 'photo_id': photo_id,
                'view': 'unknown', 'construction': None, 'image': str(path),
                'labels': {}, 'absent': [], 'reviewed': False, 'split': split,
                'sha256': digest, 'source_url': photo['source_url'],
                'product_page_url': product['page_url'],
                'model_code': model_code,
                'license_note': 'Research cache; commercial training rights unreviewed',
            })
            existing.add((source_name, photo_id))
            added += 1
    save_manifest(directory, manifest)
    return added


def review_item(directory, source, photo_id, product_id, view, construction, absent):
    manifest = load_manifest(directory)
    matches = [item for item in manifest['items'] if
               item['source'] == source and item['photo_id'] == photo_id]
    if len(matches) != 1:
        raise ValueError('Expected one imported item with that source and photo ID')
    item = matches[0]
    if set(absent) & set(item['labels']):
        raise ValueError('A class cannot be both outlined and absent')
    item.update(product_id=product_id, view=view, construction=construction,
                absent=sorted(set(absent)), reviewed=True, split=split_for(product_id))
    save_manifest(directory, manifest)


def audit(manifest):
    products = defaultdict(set)
    all_product_splits = defaultdict(set)
    classes = defaultdict(set)
    classes_by_split = defaultdict(lambda: defaultdict(set))
    views = Counter()
    construction = Counter()
    rimless_by_split = Counter()
    reviewed = 0
    for item in manifest['items']:
        all_product_splits[item['product_id']].add(item['split'])
        if item.get('reviewed'):
            reviewed += 1
            products[item['split']].add(item['product_id'])
            for name in item.get('labels', {}):
                classes[name].add(item['product_id'])
                classes_by_split[item['split']][name].add(item['product_id'])
            views[item['view']] += 1
            construction[item.get('construction') or 'unknown'] += 1
            if item.get('construction') == 'rimless':
                rimless_by_split[item['split']] += 1
    missing = [name for name in CLASSES if not classes[name]]
    rimless = sum(1 for item in manifest['items'] if item.get('reviewed') and
                  item.get('construction') == 'rimless')
    split_conflicts = sorted(product_id for product_id, splits in all_product_splits.items()
                             if len(splits) > 1)
    return {'items': len(manifest['items']), 'reviewed_items': reviewed,
            'products_by_split': {split: len(products[split]) for split in ('train', 'val', 'test')},
            'products_per_class': {name: len(classes[name]) for name in CLASSES},
            'products_per_class_by_split': {split: {name: len(classes_by_split[split][name])
                for name in CLASSES} for split in ('train', 'val', 'test')},
            'views': dict(views), 'construction': dict(construction),
            'missing_classes': missing, 'rimless_photos': rimless,
            'split_conflicts': split_conflicts,
            'rimless_photos_by_split': {split: rimless_by_split[split]
                for split in ('train', 'val', 'test')},
            'ready_for_training': not missing and not split_conflicts and all(
                rimless_by_split[split] and all(classes_by_split[split][name]
                for name in CLASSES) for split in ('train', 'val', 'test')),
            'note': 'This is a minimum completeness gate, not an accuracy claim. Public imports require review and product-level deduplication.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/eyewear-training')
    sub = parser.add_subparsers(dest='command', required=True)
    local = sub.add_parser('export-approved')
    local.add_argument('--database', type=Path, default=ROOT / 'data/studio.sqlite3')
    public = sub.add_parser('import-public')
    public.add_argument('--images', type=Path, required=True)
    public.add_argument('--masks', type=Path, required=True)
    public.add_argument('--class-name', choices=CLASSES, required=True)
    public.add_argument('--source-name', required=True)
    public.add_argument('--license-note', required=True)
    coco = sub.add_parser('import-coco')
    coco.add_argument('--images', type=Path, required=True)
    coco.add_argument('--annotations', type=Path, required=True)
    coco.add_argument('--category-map', action='append', required=True,
                      help='Exact COCO category name:eyewear class, repeat as needed')
    coco.add_argument('--source-name', required=True)
    coco.add_argument('--license-note', required=True)
    staged = sub.add_parser('stage-local')
    staged.add_argument('--images', type=Path, required=True,
                        help='Directory of candidate eyewear product photos')
    staged.add_argument('--source-name', default='local-candidate')
    research = sub.add_parser('import-research-collection')
    research.add_argument('--collection', type=Path, required=True)
    research.add_argument('--source-name', default='lenskart-research')
    review = sub.add_parser('review-item')
    review.add_argument('--source-name', required=True)
    review.add_argument('--photo-id', required=True)
    review.add_argument('--product-id', required=True, help='Same physical product across views')
    review.add_argument('--view', choices=('front', 'left', 'right'), required=True)
    review.add_argument('--construction', choices=('opaque-full-rim', 'transparent-full-rim',
                        'thin-metal', 'semi-rimless', 'rimless', 'sports-shield'), required=True)
    review.add_argument('--absent', choices=CLASSES, action='append', default=[])
    sub.add_parser('audit')
    args = parser.parse_args()
    if args.command == 'export-approved':
        print(f'Exported {export_approved(args.database, args.output)} approved photos')
    elif args.command == 'import-public':
        print(f'Imported {import_public(args.output, args.images, args.masks, args.class_name, args.source_name, args.license_note)} unreviewed photos')
    elif args.command == 'import-coco':
        mapping = {}
        for pair in args.category_map:
            category, separator, component = pair.partition(':')
            if not separator or not category or not component:
                raise ValueError(f'Invalid category map: {pair}')
            mapping[category] = component
        added, unsupported = import_coco(args.output, args.images, args.annotations,
                                         mapping, args.source_name, args.license_note)
        print(f'Imported {added} unreviewed COCO photos; skipped {unsupported} RLE annotations')
    elif args.command == 'stage-local':
        added, skipped = stage_local_photos(args.output, args.images, args.source_name)
        print(f'Staged {added} unreviewed photos; skipped {len(skipped)} duplicates or invalid files')
    elif args.command == 'import-research-collection':
        added = import_research_collection(args.output, args.collection, args.source_name)
        print(f'Staged {added} unreviewed research photos')
    elif args.command == 'review-item':
        review_item(args.output, args.source_name, args.photo_id, args.product_id,
                    args.view, args.construction, args.absent)
    print(json.dumps(audit(load_manifest(args.output)), indent=2))


if __name__ == '__main__':
    main()
