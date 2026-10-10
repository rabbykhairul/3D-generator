import numpy as np
import pytest

from backend.reconstruction import eyewear_trained


def test_candidate_requires_configured_checkpoint(monkeypatch):
    monkeypatch.delenv('EYEWEAR_TRAINED_MODEL', raising=False)
    with pytest.raises(RuntimeError, match='No trained eyewear checkpoint'):
        eyewear_trained.infer(np.zeros((100, 100, 3), np.uint8))


def test_candidate_does_not_invent_components_from_blank_photo(monkeypatch):
    image = np.full((240, 600, 3), 255, np.uint8)
    scores = np.zeros((7, 240, 600), np.float32)
    scores[0, 60:185, 45:245] = .9
    scores[0, 60:185, 355:555] = .9
    scores[4, 70:90, 20:40] = .9
    monkeypatch.setattr(eyewear_trained, 'infer', lambda _: (scores, {'model': 'candidate'}))
    photo = {'id': 'blank', 'view': 'front', 'role': 'fitting', 'width': 600, 'height': 240}
    result = eyewear_trained.apply_candidate(photo, image, image,
                                              {'annotations': [], 'warnings': []}, 1, 1)
    assert not result['annotations']
    assert result['hybrid_detection']['status'] == 'rejected'
