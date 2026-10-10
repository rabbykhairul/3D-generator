"""Local CPU SAM 3 text proposals; source/topology audits remain mandatory."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time

import cv2
import numpy as np

from .hybrid_detection import ROOT, INFERENCE_SLOT
from .foundation_detection import select_lenses


def infer_regions(image, task='lenses'):
    python = Path(os.environ.get('EYEWEAR_SAM3_PYTHON', ROOT/'artifacts/sam3-venv/bin/python'))
    manifest = Path(os.environ.get('EYEWEAR_SAM3_MANIFEST', ROOT/'artifacts/sam3/manifest.json'))
    if not python.is_file() or not manifest.is_file():
        raise RuntimeError('SAM 3 CPU weights are not ready. Request facebook/sam3 access on Hugging Face, '
                           'run artifacts/sam3-venv/bin/hf auth login locally, then run '
                           'artifacts/sam3-venv/bin/python scripts/setup_sam3.py.')
    if not INFERENCE_SLOT.acquire(timeout=1):
        raise RuntimeError('The CPU segmentation worker is busy. Retry after the current analysis finishes.')
    try:
        timeout = min(1800, max(120, int(os.environ.get('EYEWEAR_SAM3_TIMEOUT', '900'))))
        with tempfile.TemporaryDirectory(prefix='eyewear-sam3-') as directory:
            directory = Path(directory)
            photo, output = directory/'analysis.png', directory/'regions.npz'
            if not cv2.imwrite(str(photo), image):
                raise RuntimeError('Could not prepare the local SAM 3 image')
            started = time.perf_counter()
            try:
                run = subprocess.run([str(python), str(ROOT/'scripts/sam3_infer.py'),
                    '--manifest', str(manifest), '--image', str(photo), '--output', str(output),
                    '--task', task], cwd=ROOT, capture_output=True, text=True, timeout=timeout)
            except subprocess.TimeoutExpired as error:
                raise RuntimeError(f'SAM 3 exceeded its {timeout}-second CPU limit.') from error
            if run.returncode != 0 or not output.is_file():
                if run.returncode == 2 and 'CPU process memory exceeded' in run.stderr:
                    raise RuntimeError('SAM 3 stopped at its CPU memory limit to reduce memory pressure. '
                                       'Close memory-heavy applications or use a machine with more RAM.')
                raise RuntimeError('SAM 3 CPU inference failed. Check the isolated runtime and official checkpoint.')
            runtime = json.loads(run.stdout)
            if runtime.get('model') != 'sam3' or runtime.get('device') != 'cpu':
                raise ValueError('SAM 3 did not report the requested CPU model')
            with np.load(output, allow_pickle=False) as data:
                shape = tuple(data['shape'].tolist())
                packed, scores, roles = data['masks'], data['scores'], data['roles']
                allowed = {'lens','frame','bridge','nose-pad','hinge'} if task=='lenses' else {'temple','hinge','lens'}
                if (shape != image.shape[:2] or packed.dtype != np.uint8 or packed.ndim != 2 or
                        len(packed)>48 or scores.shape!=(len(packed),) or roles.shape!=(len(packed),) or
                        packed.shape[1]!=(shape[0]*shape[1]+7)//8 or not np.isfinite(scores).all() or
                        np.any(scores<0) or np.any(scores>1) or not set(roles.tolist())<=allowed):
                    raise ValueError('SAM 3 returned invalid region data')
                masks = np.unpackbits(packed, axis=1, count=shape[0]*shape[1]).reshape((-1,*shape))
            runtime['request_seconds'] = round(time.perf_counter()-started,3)
            return masks, scores, roles, runtime
    finally:
        INFERENCE_SLOT.release()


def infer_lenses(image):
    masks, scores, roles, runtime = infer_regions(image)
    selected = roles == 'lens'
    probabilities, detail = select_lenses(masks[selected], scores[selected], image.shape[:2])
    runtime.update(detail, concept_region_counts={role:int(np.sum(roles==role)) for role in set(roles.tolist())})
    from .sam3_components import component_proposals
    runtime['component_proposals']=component_proposals(masks,scores,roles)
    return probabilities, runtime


def infer_side(image):
    masks, scores, roles, runtime = infer_regions(image, 'side')
    selected = (roles == 'temple') | (roles == 'lens')
    from .sam3_components import component_proposals
    runtime['component_proposals']=component_proposals(masks,scores,roles)
    runtime['hinge_concept_regions'] = int(np.sum(roles=='hinge'))
    return masks[selected], scores[selected], runtime
