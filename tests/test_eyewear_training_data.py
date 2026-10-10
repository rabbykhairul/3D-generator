import json
import sqlite3

import numpy as np
from PIL import Image
import pytest

from scripts.eyewear_dataset import (audit, export_approved, import_coco, import_public,
                                     import_research_collection, load_manifest, review_item,
                                     stage_local_photos)
from scripts.train_eyewear_components import load_example, train


def test_rimless_approved_export_has_no_invented_frame(tmp_path):
    image = tmp_path / 'photo.png'
    Image.new('RGB', (100, 60), 'white').save(image)
    lens = [{'x': 10, 'y': 10}, {'x': 40, 'y': 10}, {'x': 40, 'y': 45}, {'x': 10, 'y': 45}]
    other = [{**p, 'x': p['x'] + 50} for p in lens]
    project = {'id': 'rimless-pair', 'category': 'rimless', 'photos': [
        {'id': 'front', 'path': str(image), 'view': 'front', 'role': 'fitting'}],
        'annotations': [
            {'photo_id': 'front', 'part': 'lens-left', 'approved': True, 'points': lens},
            {'photo_id': 'front', 'part': 'lens-right', 'approved': True, 'points': other}]}
    database = tmp_path / 'studio.sqlite3'
    with sqlite3.connect(database) as conn:
        conn.execute('CREATE TABLE projects(payload TEXT)')
        conn.execute('INSERT INTO projects VALUES (?)', (json.dumps(project),))
    directory = tmp_path / 'dataset'
    assert export_approved(database, directory) == 1
    item = load_manifest(directory)['items'][0]
    assert set(item['labels']) == {'lens'}
    assert item['absent'] == []  # Photo evidence does not prove a missing mask is absent.
    assert audit(load_manifest(directory))['ready_for_training'] is False


def test_public_single_class_mask_keeps_other_classes_unknown(tmp_path):
    images, masks = tmp_path / 'images', tmp_path / 'masks'
    images.mkdir(); masks.mkdir()
    Image.new('RGB', (64, 64), 'white').save(images / 'sample.png')
    raw = np.zeros((64, 64), np.uint8); raw[10:40, 10:40] = 255
    Image.fromarray(raw).save(masks / 'sample.png')
    directory = tmp_path / 'dataset'
    assert import_public(directory, images, masks, 'lens', 'public-source', 'Dataset terms checked') == 1
    item = load_manifest(directory)['items'][0]
    assert not item['reviewed']
    image, target, known = load_example(item, 256)
    assert image.shape == (3, 256, 256)
    assert target[0].sum() > 0
    assert known[0, 0, 0] == 1
    assert known[1:].sum() == 0
    review_item(directory, 'public-source', 'sample', 'physical-sku', 'front', 'rimless', ['frame'])
    item = load_manifest(directory)['items'][0]
    assert item['reviewed'] and item['absent'] == ['frame']
    with pytest.raises(ValueError, match='Dataset is not ready'):
        train(directory, tmp_path / 'run', 1, 256, 'cpu')
    with pytest.raises(ValueError, match='product holdouts'):
        train(directory, tmp_path / 'run', 1, 256, 'cpu', exploratory=True)


def test_staged_local_photos_group_product_and_deduplicate_without_labels(tmp_path):
    images = tmp_path / 'photos'
    images.mkdir()
    Image.new('RGB', (64, 64), 'red').save(images / 'vc-s19329-front.png')
    Image.new('RGB', (64, 64), 'blue').save(images / 'vc-s19329-side.png')
    (images / 'vc-s19329-copy.png').write_bytes((images / 'vc-s19329-front.png').read_bytes())
    Image.new('RGB', (64, 64), 'green').save(images / 'unrelated.png')
    dataset = tmp_path / 'dataset'
    added, skipped = stage_local_photos(dataset, images)
    assert added == 2 and len(skipped) == 1
    items = load_manifest(dataset)['items']
    assert len({item['product_id'] for item in items}) == 1
    assert all(item['view'] == 'unknown' and not item['reviewed'] and
               item['labels'] == {} and item['absent'] == [] for item in items)
    assert stage_local_photos(dataset, images)[0] == 0
    assert audit(load_manifest(dataset))['reviewed_items'] == 0


def test_audit_flags_same_product_in_multiple_splits():
    item = {'product_id': 'catalog:one', 'split': 'train', 'reviewed': False}
    other = dict(item, split='test')
    result = audit({'items': [item, other]})
    assert result['split_conflicts'] == ['catalog:one']
    assert not result['ready_for_training']


def test_coco_import_maps_only_known_polygons_and_stays_unreviewed(tmp_path):
    images = tmp_path / 'photos'
    images.mkdir()
    Image.new('RGB', (64, 64), 'white').save(images / 'pair.png')
    coco = {'images': [{'id': 4, 'file_name': 'pair.png', 'width': 64, 'height': 64}],
            'categories': [{'id': 1, 'name': 'glasses lens'},
                           {'id': 2, 'name': 'glasses shadow'}],
            'annotations': [
                {'image_id': 4, 'category_id': 1,
                 'segmentation': [[5, 5, 25, 5, 25, 25, 5, 25]]},
                {'image_id': 4, 'category_id': 1,
                 'segmentation': {'counts': 'RLE', 'size': [64, 64]}},
                {'image_id': 4, 'category_id': 2,
                 'segmentation': [[30, 30, 40, 30, 40, 40]]}]}
    annotation_path = tmp_path / 'annotations.json'
    annotation_path.write_text(json.dumps(coco))
    dataset = tmp_path / 'dataset'
    assert import_coco(dataset, images, annotation_path, {'glasses lens': 'lens'},
                       'public-v1', 'Terms checked') == (1, 1)
    assert import_coco(dataset, images, annotation_path, {'glasses lens': 'lens'},
                       'public-v1', 'Terms checked') == (0, 1)
    item = load_manifest(dataset)['items'][0]
    assert item['labels'].keys() == {'lens'}
    assert not item['reviewed'] and item['absent'] == []
    assert Image.open(item['labels']['lens']).getbbox() == (5, 5, 26, 26)


def test_research_collection_keeps_product_group_and_unknown_labels(tmp_path):
    import hashlib
    photo = tmp_path / 'photo.jpg'
    Image.new('RGB', (100, 100), 'red').save(photo)
    digest = hashlib.sha256(photo.read_bytes()).hexdigest()
    collection = tmp_path / 'collection.json'
    collection.write_text(json.dumps({'schema': 1, 'products': [{
        'product_id': 'vc-s19329', 'page_url': 'https://www.lenskart.com/p.html',
        'status': 'collected', 'photos': [{'path': str(photo), 'sha256': digest,
        'source_url': 'https://static5.lenskart.com/media/catalog/product/photo.jpg'}]}]}))
    dataset = tmp_path / 'dataset'
    assert import_research_collection(dataset, collection) == 1
    assert import_research_collection(dataset, collection) == 0
    item = load_manifest(dataset)['items'][0]
    assert item['product_id'] == 'catalog:vc-s19329'
    assert item['view'] == 'unknown' and item['labels'] == {} and not item['reviewed']
    assert audit(load_manifest(dataset))['reviewed_items'] == 0


def test_research_color_variants_share_model_split(tmp_path):
    import hashlib
    products = []
    for color, pixels in (('blaze', 'red'), ('flame', 'blue')):
        photo = tmp_path / f'{color}.jpg'
        Image.new('RGB', (100, 100), pixels).save(photo)
        digest = hashlib.sha256(photo.read_bytes()).hexdigest()
        products.append({
            'product_id': f'la-e15417-{color}', 'model_code': 'la-e15417',
            'page_url': f'https://www.lenskart.com/{color}.html',
            'status': 'collected', 'photos': [{'path': str(photo), 'sha256': digest,
                'source_url': f'https://static5.lenskart.com/media/catalog/product/{color}.jpg'}],
        })
    collection = tmp_path / 'collection.json'
    collection.write_text(json.dumps({'schema': 1, 'products': products}))
    dataset = tmp_path / 'dataset'
    assert import_research_collection(dataset, collection) == 2
    items = load_manifest(dataset)['items']
    assert len({item['product_id'] for item in items}) == 2
    assert len({item['split'] for item in items}) == 1
