"""Optional candidate-model proposals; source checks still own final geometry."""
import json
import os
from pathlib import Path
import subprocess
import tempfile

import cv2
import numpy as np

from .hybrid_detection import ROOT, INFERENCE_SLOT, _apply_lens_hybrid
from .sam3_components import component_proposals, apply_front_components, apply_side_components
from .anatomy import FRONT_VIEWS, SIDE_VIEWS, audit_photo

CLASSES = ('lens', 'frame', 'bridge', 'nose-pad', 'hinge', 'temple', 'lens-mount')


def infer(image):
    checkpoint = Path(os.environ.get('EYEWEAR_TRAINED_MODEL', ''))
    python = Path(os.environ.get('EYEWEAR_TRAINED_PYTHON', ROOT / 'artifacts/segmenter-venv/bin/python'))
    if not os.environ.get('EYEWEAR_TRAINED_MODEL') or not checkpoint.is_file() or not python.is_file():
        raise RuntimeError('No trained eyewear checkpoint is configured; current detection is unchanged.')
    if not INFERENCE_SLOT.acquire(timeout=1):
        raise RuntimeError('The CPU eyewear model worker is busy.')
    try:
        with tempfile.TemporaryDirectory(prefix='eyewear-trained-') as temporary:
            source, output = Path(temporary) / 'image.png', Path(temporary) / 'scores.npz'
            if not cv2.imwrite(str(source), image):
                raise RuntimeError('Could not prepare eyewear inference image')
            try:
                process = subprocess.run([str(python), str(ROOT / 'scripts/eyewear_trained_infer.py'),
                    '--checkpoint', str(checkpoint), '--image', str(source), '--output', str(output)],
                    cwd=ROOT, capture_output=True, text=True, timeout=180)
            except subprocess.TimeoutExpired as error:
                raise RuntimeError('Eyewear candidate exceeded its three-minute CPU limit') from error
            if process.returncode or not output.is_file():
                raise RuntimeError('Eyewear candidate inference failed; inspect the local checkpoint and worker')
            runtime = json.loads(process.stdout)
            with np.load(output, allow_pickle=False) as data:
                scores = data['scores'].astype(np.float32)
                if (tuple(data['classes'].tolist()) != CLASSES or
                    scores.shape != (len(CLASSES), *image.shape[:2]) or
                    not np.isfinite(scores).all() or scores.min() < 0 or scores.max() > 1):
                    raise ValueError('Eyewear candidate returned invalid probability maps')
        return scores, runtime
    finally:
        INFERENCE_SLOT.release()


def apply_candidate(photo, image, detection_image, baseline, sx, sy):
    scores, runtime = infer(image)
    view = str(photo.get('view', '')).lower()
    if view in FRONT_VIEWS:
        result = _apply_lens_hybrid(photo, image, detection_image, baseline, sx, sy,
            model='eyewear-trained', precomputed=(scores[0], runtime))
        proposals = component_proposals(
            np.asarray(scores[[1, 2, 3, 4, 6]] >= .5, np.uint8),
            np.ones(5, np.float32), np.asarray(['frame', 'bridge', 'nose-pad', 'hinge', 'lens-mount']))
        result['hybrid_detection'].setdefault('runtime', {})['component_proposals'] = proposals
        result = apply_front_components(photo, image, result, sx, sy,
                                        source_prefix='eyewear-trained')
        result['hybrid_detection']['runtime'].pop('component_proposals', None)
        result.setdefault('warnings', []).append(
            'Experimental trained-component candidate: review every identity and boundary. '
            'Rimless construction needs visible mounting evidence; hidden depth is unmeasured.')
        return result
    if view in SIDE_VIEWS:
        from .side_extraction import extract_side
        mask = np.uint8(scores[5] >= .5) * 255
        side = extract_side(photo, mask, sx, sy)
        if len(side.get('annotations', [])) != 1 or audit_photo(photo, side['annotations'],
                width=image.shape[1] * sx, height=image.shape[0] * sy)['blocking']:
            baseline.setdefault('warnings', []).append(
                'The trained candidate did not establish one continuous hinge-to-tip arm.')
            return baseline
        side['annotations'][0]['source'] = 'automatic-eyewear-trained-side-review-v1'
        side['hybrid_detection'] = dict(requested_backend='eyewear-trained',
            proposal_backend='eyewear-components-lraspp', inference_ran=True,
            proposals_used=True, experimental=True, status='partial',
            reason='Candidate arm mask passed side geometry checks; hardware depth remains inferred.',
            runtime=runtime)
        side['hybrid_detection']['runtime']['component_proposals'] = component_proposals(
            np.asarray(scores[[4]] >= .5, np.uint8), np.ones(1, np.float32),
            np.asarray(['hinge']))
        side = apply_side_components(photo, image, side, sx, sy,
                                     source_prefix='eyewear-trained')
        return side
    return baseline
