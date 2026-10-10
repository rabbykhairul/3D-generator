"""Novel procedural contours exercise refinement, rejection and API provenance.

These are geometry regressions, not an unseen-product accuracy benchmark.
"""
from copy import deepcopy
import io

import cv2
import numpy as np
from PIL import Image
from shapely.geometry import Polygon
from fastapi.testclient import TestClient

from backend import store
from backend.main import app
from backend.reconstruction import hybrid_detection as hybrid
from backend.reconstruction.detection_report import detection_report


def fixture():
    image = np.full((280,600,3),255,np.uint8)
    outer = np.array([(30,70),(260,58),(283,110),(318,110),(340,62),(573,82),
                      (565,210),(338,228),(316,150),(285,150),(259,220),(44,213)])
    left = Polygon([(59,91),(235,83),(255,124),(228,190),(71,185)])
    right = Polygon([(355,88),(543,104),(538,189),(362,202),(340,130)])
    cv2.fillPoly(image,[outer],(35,42,38))
    scores = np.zeros(image.shape[:2],np.float32)
    for polygon in (left,right):
        cv2.fillPoly(image,[np.array(polygon.exterior.coords,np.int32)],(255,255,255))
        # Coarse model proposals deliberately overshoot the source boundary.
        cv2.fillPoly(scores,[np.array(polygon.buffer(3).exterior.coords,np.int32)],.96)
    photo = dict(id='photo',view='front',width=600,height=280,role='fitting')
    return image,scores,photo,(left,right)


def run(image,scores,photo,monkeypatch,baseline=None):
    monkeypatch.setattr(hybrid,'infer_lenses',lambda _: (scores,{'device':'cpu'}))
    return hybrid.apply_hybrid(photo,image,image,baseline or dict(annotations=[],warnings=[]),1,1)


def test_refined_freeform_lenses_follow_source_and_remain_unapproved(monkeypatch):
    image,scores,photo,expected = fixture()
    result = run(image,scores,photo,monkeypatch)
    assert result['hybrid_detection']['status'] == 'applied'
    assert result['hybrid_detection']['proposals_used']
    assert {a['part'] for a in result['annotations']} == {'front','lens-left','lens-right'}
    polygons = {a['part']:Polygon([(p['x'],p['y']) for p in a['points']]) for a in result['annotations']}
    for part,truth in zip(('lens-left','lens-right'),expected):
        polygon = polygons[part]
        assert polygon.is_valid and polygons['front'].contains(polygon)
        assert polygon.intersection(truth).area/polygon.union(truth).area > .94
    assert all(not a['approved'] for a in result['annotations'])
    report = detection_report(photo,result)
    assert report['recognizer']['learned_model_active'] and report['recognizer']['proposals_used']
    assert not report['automatic_acceptance'] and not report['fidelity_verified']


def test_blank_source_cannot_be_filled_with_predicted_geometry(monkeypatch):
    image,scores,photo,_ = fixture()
    image[:] = 255
    result = run(image,scores,photo,monkeypatch)
    assert result['annotations'] == [] and result['hybrid_detection']['status'] == 'rejected'


def test_incomplete_threshold_front_does_not_veto_independent_source_lenses(monkeypatch):
    image, scores, photo, expected = fixture()
    baseline = dict(annotations=[hybrid._annotation(photo, 'front',
        Polygon([(100,70),(500,70),(500,220),(100,220)]), 1, 1, 'automatic-threshold')], warnings=[])
    result = run(image, scores, photo, monkeypatch, baseline)
    assert result['hybrid_detection']['baseline_front_support_rejected']
    assert result['hybrid_detection']['proposals_used']
    polygons = {a['part']: Polygon([(p['x'],p['y']) for p in a['points']]) for a in result['annotations']}
    for key, truth in zip(('lens-left', 'lens-right'), expected):
        assert polygons[key].is_valid
        assert polygons[key].intersection(truth).area/polygons[key].union(truth).area > .94
    assert all(not a['approved'] for a in result['annotations'])


def test_hybrid_coordinates_map_back_to_the_original_photo(monkeypatch):
    image,scores,photo,expected = fixture()
    monkeypatch.setattr(hybrid,'infer_lenses',lambda _: (scores,{'device':'cpu'}))
    photo = dict(photo,width=1200,height=560)
    result = hybrid.apply_hybrid(photo,image,image,dict(annotations=[],warnings=[]),2,2)
    assert result['hybrid_detection']['proposals_used']
    lens = next(a for a in result['annotations'] if a['part']=='lens-left')
    polygon = Polygon([(p['x']/2,p['y']/2) for p in lens['points']])
    assert polygon.intersection(expected[0]).area/polygon.union(expected[0]).area > .94


def test_separate_lenses_do_not_invent_a_front_or_bridge(monkeypatch):
    image,scores,photo,expected = fixture()
    image[:] = 255
    for polygon in expected:
        cv2.polylines(image,[np.array(polygon.exterior.coords,np.int32)],True,(30,30,30),5)
    result = run(image,scores,photo,monkeypatch)
    assert {a['part'] for a in result['annotations']} == {'lens-left','lens-right'}
    assert result['hybrid_detection']['proposals_used']
    assert result['hybrid_detection']['status']=='partial'
    report=detection_report(photo,result)
    assert report['status']=='partial' and report['missing_parts']==[]
    from backend.reconstruction.anatomy import audit_photo
    states={n['id']:n['state'] for n in audit_photo(photo,result['annotations'])['nodes']}
    assert states['lens-left']==states['lens-right']=='observed'
    assert states['front']=='uncertain'


def test_pale_source_front_requires_threshold_consensus_and_a_real_connection():
    image,_,_,lenses=fixture()
    # Pale material remains below the usual contrast detector's threshold.
    # The same arbitrary polygon is retained, without a template or filled
    # lens proposals becoming synthetic material in the source mask.
    image[:]=255
    for lens in lenses:
        cv2.polylines(image,[np.rint(lens.exterior.coords).astype(np.int32)],True,(245,245,245),5)
    cv2.line(image,(255,124),(340,130),(245,245,245),5)
    outer=hybrid._faint_source_front(image,lenses)
    assert outer is not None and all(outer.contains(p) for p in lenses)
    # Remove the photographed bridge. Independent rims must not be joined.
    image[:,278:324]=255
    assert hybrid._faint_source_front(image,lenses) is None
    # Quiet background verification also rejects a textured backdrop.
    rng=np.random.default_rng(25)
    image[:12]=rng.integers(200,256,image[:12].shape,dtype=np.uint8)
    assert hybrid._faint_source_front(image,lenses) is None


def test_extra_lens_instance_is_rejected(monkeypatch):
    image,scores,photo,_ = fixture()
    cv2.rectangle(scores,(278,200),(321,263),.96,-1)
    result = run(image,scores,photo,monkeypatch)
    assert result['hybrid_detection']['status'] == 'rejected'
    assert not result['annotations']


def test_existing_complete_geometry_is_not_overwritten(monkeypatch):
    image,scores,photo,expected = fixture()
    annotations = [hybrid._annotation(photo,'front',Polygon([(20,40),(580,40),(580,240),(20,240)]),1,1,'existing')]
    annotations += [hybrid._annotation(photo,part,p,1,1,'existing') for part,p in zip(('lens-left','lens-right'),expected)]
    baseline = dict(annotations=annotations,warnings=['existing warning'])
    before = deepcopy(baseline)
    result = run(image,scores,photo,monkeypatch,baseline)
    assert result['annotations'] == annotations and baseline == before
    assert result['hybrid_detection']['status'] == 'retained'
    report = detection_report(photo,result)
    assert report['recognizer']['backend'] == 'geometric-heuristics'
    assert report['recognizer']['learned_model_active'] and not report['recognizer']['proposals_used']


def test_unavailable_runtime_is_explicit_and_preserves_baseline(monkeypatch):
    image,_,photo,_ = fixture()
    baseline = dict(annotations=[],warnings=['existing'])
    def fail(_):
        raise RuntimeError('Local checkpoint missing')
    monkeypatch.setattr(hybrid,'infer_lenses',fail)
    result = hybrid.apply_hybrid(photo,image,image,baseline,1,1)
    assert baseline == dict(annotations=[],warnings=['existing'])
    assert result['hybrid_detection']['status'] == 'unavailable'
    assert not detection_report(photo,result)['recognizer']['learned_model_active']
    assert 'missing' in result['warnings'][-1]


def test_hybrid_api_option_is_review_only_and_reports_actual_backend(tmp_path,monkeypatch):
    image,scores,_,_ = fixture()
    monkeypatch.setattr(store,'DATA',tmp_path)
    monkeypatch.setenv('EYEWEAR_START_WORKER','0')
    monkeypatch.setattr(hybrid,'infer_lenses',lambda _: (scores,{'device':'cpu'}))
    raw = io.BytesIO()
    Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB)).save(raw,'PNG')
    with TestClient(app) as client:
        project = client.post('/api/projects',json={'name':'Novel contour','category':'opaque-full-rim'}).json()
        url = '/api/projects/'+project['id']
        project = client.post(url+'/photos',data={'view':'front'},files={'file':('front.png',raw.getvalue(),'image/png')}).json()
        endpoint = url+'/photos/'+project['photos'][0]['id']+'/extract'
        before = store.get_project(project['id'])
        assert client.post(endpoint+'?backend=unsupported').status_code == 422
        response = client.post(endpoint+'?backend=hybrid')
        assert response.status_code == 200
        result = response.json()
        assert result['detection_report']['recognizer']['learned_model_active']
        assert store.get_project(project['id']) == before
        assert all(not a['approved'] for a in result['annotations'])
