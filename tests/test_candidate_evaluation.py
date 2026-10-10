import hashlib
import numpy as np
import pytest
from scripts.evaluate_glasses_segmenter import checked_checkpoint, mask_metrics
from scripts.benchmark_detection import holdout_failures
import json
import sqlite3
from scripts.prepare_segmenter_evaluation import saved_cases


def test_candidate_requires_trusted_local_checkpoint_provenance(tmp_path):
    checkpoint = tmp_path/'model.pth'
    checkpoint.write_bytes(b'contract-test-not-a-model')
    spec = dict(path='model.pth', sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                source_url='https://example.test/artifact', license_review='test only')
    assert checked_checkpoint(spec, tmp_path) == checkpoint
    with pytest.raises(ValueError, match='SHA256'):
        checked_checkpoint(dict(spec, sha256='wrong'), tmp_path)
    with pytest.raises(ValueError, match='license'):
        checked_checkpoint(dict(spec, license_review=''), tmp_path)


def test_mask_evaluation_penalizes_extra_pixels():
    truth = np.array([[True, False], [False, False]])
    prediction = np.array([[True, True], [False, False]])
    result = mask_metrics(prediction, truth)
    assert result == dict(iou=.5, false_positive_pixels=1, false_negative_pixels=0)
    with pytest.raises(ValueError, match='dimensions'):
        mask_metrics(prediction, truth[:1])


def test_holdout_gate_rejects_renamed_duplicate_and_development_product(tmp_path):
    (tmp_path/'dev.png').write_bytes(b'development-image')
    (tmp_path/'test.png').write_bytes(b'new-image')
    (tmp_path/'dev.json').write_text(json.dumps({'cases':[{'sku':'used-product','path':'dev.png'}]}))
    case = dict(sku='new-product', path='test.png',
                sha256=hashlib.sha256(b'new-image').hexdigest(),
                label_source='independent-human', label_reviewer='reviewer-1')
    manifest = dict(development_manifests=['dev.json'], required_families=['thin-metal'], cases=[case])
    assert holdout_failures(manifest, tmp_path) == []
    manifest['cases'] = [dict(case, sku='used-product')]
    assert any('product appears' in x for x in holdout_failures(manifest, tmp_path))
    manifest['cases'] = [dict(case, path='dev.png', sha256=hashlib.sha256(b'development-image').hexdigest())]
    assert any('source image appears' in x for x in holdout_failures(manifest, tmp_path))
    manifest['cases'] = [dict(case, sha256='changed', label_source='engine-generated')]
    failures = holdout_failures(manifest, tmp_path)
    assert any('hash' in x for x in failures) and any('provenance' in x for x in failures)


def test_empty_coverage_or_missing_development_history_cannot_pass(tmp_path):
    failures = holdout_failures(dict(cases=[]), tmp_path)
    assert len(failures) == 2


def test_saved_inventory_deduplicates_and_excludes_reference_and_archived(tmp_path):
    image = tmp_path/'photo.png'
    image.write_bytes(b'same-photo')
    database = tmp_path/'studio.sqlite3'
    photo = dict(id='front',path=str(image),view='front',role='fitting')
    projects = [dict(id='active',name='Active',category='thin-metal',updated_at='2026-10-05',
                     photos=[photo,dict(photo,id='duplicate'),dict(photo,id='reference',role='validation')]),
                dict(id='archived',name='Archived',category='thin-metal',updated_at='2026-10-04',
                     archived_at='2026-10-04',photos=[photo])]
    with sqlite3.connect(database) as conn:
        conn.execute('CREATE TABLE projects(payload TEXT)')
        conn.executemany('INSERT INTO projects VALUES (?)',[(json.dumps(p),) for p in projects])
    before = database.read_bytes()
    cases = saved_cases(database)
    assert len(cases) == 1 and cases[0]['sku'] == 'active'
    assert cases[0]['expected_masks'] == {}  # Never manufacture ground truth.
    assert database.read_bytes() == before
