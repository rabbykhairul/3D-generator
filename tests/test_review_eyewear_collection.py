import json

from PIL import Image
import pytest

from scripts.review_eyewear_collection import approve, save_draft, skip


def fixture(tmp_path):
    photo = tmp_path / 'photo.png'
    Image.new('RGB', (100, 80), 'white').save(photo)
    (tmp_path / 'manifest.json').write_text(json.dumps({
        'schema': 1,
        'classes': ['lens', 'frame', 'bridge', 'nose-pad', 'hinge', 'temple', 'lens-mount'],
        'items': [{'source': 'lenskart-research', 'product_id': 'catalog:one',
                   'photo_id': 'photo001', 'image': str(photo), 'view': 'unknown',
                   'construction': None, 'labels': {}, 'absent': [], 'reviewed': False,
                   'split': 'train'}],
    }))
    return tmp_path


def outline(part, points):
    return {'part': part, 'points': [{'x': x, 'y': y} for x, y in points]}


def test_draft_is_not_a_label_and_approval_cuts_lens_hole(tmp_path):
    root = fixture(tmp_path)
    payload = {'view': 'front', 'construction': 'opaque-full-rim', 'absent': [],
               'polygons': [outline('front', [(10, 10), (90, 10), (90, 70), (10, 70)]),
                            outline('lens-left', [(20, 20), (45, 20), (45, 50), (20, 50)]),
                            outline('lens-right', [(55, 20), (80, 20), (80, 50), (55, 50)])]}
    save_draft(root, 'photo001', payload)
    assert not json.loads((root / 'manifest.json').read_text())['items'][0]['reviewed']
    result = approve(root, 'photo001', payload)
    assert result['reviewed'] and set(result['labels']) == {'frame', 'lens'}
    with Image.open(result['labels']['frame']) as frame:
        assert frame.getpixel((30, 30)) == 0
        assert frame.getpixel((50, 30)) == 255


def test_rimless_requires_confirmed_absent_frame(tmp_path):
    root = fixture(tmp_path)
    payload = {'view': 'front', 'construction': 'rimless', 'absent': [],
               'polygons': [outline('lens-left', [(20, 20), (45, 20), (45, 50), (20, 50)])]}
    with pytest.raises(ValueError, match='Confirm frame absent'):
        approve(root, 'photo001', payload)
    payload['absent'] = ['frame']
    assert approve(root, 'photo001', payload)['absent'] == ['frame']


def test_skip_keeps_unreviewed_photo_out_of_training(tmp_path):
    root = fixture(tmp_path)
    result = skip(root, 'photo001')
    assert result['review_state'] == 'skipped' and not result['reviewed']


def test_adding_component_preserves_approved_lens(tmp_path):
    root = fixture(tmp_path)
    lens = {'view': 'front', 'construction': 'opaque-full-rim', 'absent': [],
            'polygons': [outline('lens-left', [(20, 20), (45, 20), (45, 50), (20, 50)]),
                         outline('lens-right', [(55, 20), (80, 20), (80, 50), (55, 50)])]}
    before = approve(root, 'photo001', lens)
    lens_path = before['labels']['lens']
    frame = {'view': 'front', 'construction': 'opaque-full-rim', 'absent': [],
             'polygons': [outline('front', [(10, 10), (90, 10), (90, 70), (10, 70)])]}
    after = approve(root, 'photo001', frame)
    assert after['labels']['lens'] == lens_path
    with Image.open(after['labels']['frame']) as mask:
        assert mask.getpixel((30, 30)) == 0
        assert mask.getpixel((50, 30)) == 255
