"""Bounded experimental shared front-curvature / weak-perspective camera fit.

The straight front fixes XY and scale. Oblique views constrain one shared
parabolic wrap parameter and independent cameras. This is not full eyewear
reconstruction: arms, optical depth and hidden hardware remain outside the fit.
"""
from pathlib import Path
import time

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .multiview import _excluded

VIEWS = {'front-left', 'front-right', 'front_left', 'front_right',
         'elevated-front-left', 'elevated-front-right'}
LIMITATIONS = [
    'Experimental front-only fit; does not optimize arms, thickness, pads or hinge mechanics.',
    'Straight-front XY and supplied dimensions are fixed; weak perspective and parabolic wrap are assumptions.',
    'Matching details can be reflections or repeated patterns; residuals are not an independent accuracy score.',
    'Withheld feature matches test interpolation within fitted photos, not unseen products or physical geometry.',
]


def project_front(xy, sag, half_width, camera):
    """Millimetre front coordinates to image pixels (image Y points down)."""
    xyz = np.column_stack((xy, -sag * np.minimum(1., (xy[:, 0] / half_width)**2)))
    rotation = Rotation.from_euler('yxz', camera[:3]).as_matrix()
    rotated = xyz @ rotation.T
    return rotated[:, :2] * np.array([1., -1.]) * np.exp(camera[3]) + camera[4:6]


def _camera_seed(xy, uv):
    matrix = np.linalg.lstsq(np.column_stack((xy, np.ones(len(xy)))), uv, rcond=None)[0]
    scale = max(.05, np.linalg.norm(matrix[1]))
    ratio = np.clip(np.linalg.norm(matrix[0]) / scale, .2, .98)
    yaw = np.arccos(ratio)
    roll = np.arctan2(-matrix[0, 1], matrix[0, 0])
    return np.array([yaw, 0., roll, np.log(scale), *matrix[2]])


def fit_correspondences(views, half_width, initial_sag=4., max_seconds=20.):
    """Public numerical core, also tested against known synthetic 3D points."""
    report = {'schema_version': 1, 'status': 'withheld', 'applied': False,
              'method': 'shared parabolic front + weak-perspective cameras',
              'initial_front_curve_mm': float(initial_sag), 'views': [],
              'limitations': LIMITATIONS}
    valid = []
    for view in views:
        xy, uv = np.asarray(view['xy'], float), np.asarray(view['uv'], float)
        if (xy.ndim != 2 or xy.shape[1] != 2 or xy.shape != uv.shape or len(xy) < 20
                or not np.isfinite(xy).all() or not np.isfinite(uv).all()):
            continue
        if np.ptp(xy[:, 0]) < half_width * 1.1 or np.ptp(xy[:, 1]) < 8:
            continue
        # Spatially distributed deterministic split, kept out of optimization.
        order = np.lexsort((xy[:, 1], xy[:, 0]))
        hold = order[2::5]
        train = np.setdiff1d(np.arange(len(xy)), hold)
        valid.append({**view, 'xy': xy, 'uv': uv, 'train': train, 'hold': hold})
    if not valid:
        report['reason'] = 'Need an oblique photo with at least 20 broad, consistent front-surface matches.'
        return report
    valid = valid[:2]
    seeds = [_camera_seed(v['xy'][v['train']], v['uv'][v['train']]) for v in valid]
    lower = np.array([-.1, -.8, -1.2, -5., -5000., -5000.])
    upper = np.array([1.35, .8, 1.2, 6., 10000., 10000.])
    # Both yaw directions are permitted. No view-name-based mirror assumption.
    lower[0] = -1.35
    begin = time.monotonic()
    def residual(parameters, fixed=None):
        if time.monotonic() - begin > max_seconds:
            raise TimeoutError('Joint fit CPU time budget exceeded')
        sag = parameters[0] if fixed is None else fixed
        cameras = parameters[1:] if fixed is None else parameters
        chunks = []
        for i, v in enumerate(valid):
            ids = v['train']
            chunks.append((project_front(v['xy'][ids], sag, half_width, cameras[i*6:(i+1)*6]) - v['uv'][ids]).ravel())
        return np.concatenate(chunks)
    best = None
    try:
        from itertools import product
        for signs in product((1, -1), repeat=len(valid)):
            cameras = np.concatenate([seed * np.array([sign, 1, 1, 1, 1, 1]) for seed, sign in zip(seeds, signs)])
            cameras = np.clip(cameras, np.tile(lower, len(valid))+1e-7, np.tile(upper, len(valid))-1e-7)
            result = least_squares(residual, np.r_[np.clip(initial_sag, .001, 19.999), cameras],
                bounds=(np.r_[0., np.tile(lower, len(valid))], np.r_[20., np.tile(upper, len(valid))]),
                loss='soft_l1', f_scale=2., max_nfev=130)
            if best is None or result.cost < best.cost:
                best = result
        baseline = least_squares(lambda c: residual(c, initial_sag), best.x[1:],
            bounds=(np.tile(lower, len(valid)), np.tile(upper, len(valid))), loss='soft_l1', f_scale=2., max_nfev=100)
    except TimeoutError as exc:
        report['reason'] = str(exc)
        return report
    sag = float(best.x[0])
    errors, old_errors = [], []
    for i, v in enumerate(valid):
        camera = best.x[1+i*6:7+i*6]
        ids = v['hold']
        predicted = project_front(v['xy'][ids], sag, half_width, camera)
        error = np.linalg.norm(predicted - v['uv'][ids], axis=1)
        old = np.linalg.norm(project_front(v['xy'][ids], initial_sag, half_width, baseline.x[i*6:(i+1)*6]) - v['uv'][ids], axis=1)
        errors.extend(error); old_errors.extend(old)
        report['views'].append({'photo_id': v.get('photo_id'), 'view': v.get('view'),
            'matches': len(v['xy']), 'training_matches': len(v['train']), 'withheld_matches': len(ids),
            'withheld_median_error_px': round(float(np.median(error)), 3),
            'baseline_withheld_median_error_px': round(float(np.median(old)), 3),
            'camera': {'model': 'weak-perspective', 'yaw_pitch_roll_radians': camera[:3].tolist(),
                       'pixels_per_mm': float(np.exp(camera[3])), 'translation_px': camera[4:6].tolist()},
            'withheld_observed_px': v['uv'][ids].tolist(), 'withheld_predicted_px': predicted.tolist()})
    # Curvature sensitivity after eliminating camera freedoms. Small sensitivity
    # means cameras can explain changes to sag: do not mistake this for evidence.
    j = best.jac
    independent = j[:, 0] - j[:, 1:] @ np.linalg.lstsq(j[:, 1:], j[:, 0], rcond=None)[0]
    sensitivity = float(np.linalg.norm(independent))
    uncertainty = max(.5, float(np.sqrt(np.mean(np.square(errors))))) / max(sensitivity, 1e-9)
    current = float(np.median(errors)); old = float(np.median(old_errors))
    improvement = (old-current)/max(old, .1)
    camera_valid = all(.18 < abs(v['camera']['yaw_pitch_roll_radians'][0]) < 1.30 for v in report['views'])
    per_view_valid = all(v['withheld_median_error_px'] <= 3. and
                         v['withheld_median_error_px'] <= v['baseline_withheld_median_error_px'] + .25
                         for v in report['views'])
    accepted = (best.success and .1 < sag < 19.9 and camera_valid and per_view_valid
                and current <= 3. and uncertainty <= 3. and improvement >= .15 and abs(sag-initial_sag) >= .25)
    report.update(status='applied' if accepted else 'withheld', applied=bool(accepted),
                  fitted_front_curve_mm=round(sag, 4),
                  withheld_median_error_px=round(current, 3),
                  baseline_withheld_median_error_px=round(old, 3),
                  relative_error_improvement=round(improvement, 4),
                  local_sensitivity_uncertainty_mm=round(min(uncertainty, 1e9), 3),
                  elapsed_seconds=round(time.monotonic()-begin, 3),
                  reason='Front wrap updated from front and oblique evidence; review diagnostics.' if accepted else
                  'Fit did not establish sufficient curvature sensitivity and withheld-match improvement; existing geometry retained.')
    return report


def _load(photo):
    from PIL import Image
    with Image.open(photo['path']) as image:
        image = image.convert('RGB')
        image.thumbnail((1200, 1200))
        return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2GRAY), np.array([image.width/photo['width'], image.height/photo['height']])


def fit_project_front(photos, annotations, front_photo, center, scales, frame_width, initial_sag):
    candidates = [p for p in photos if p.get('view') in VIEWS and not _excluded(p)]
    latest = {p['view']: p for p in candidates}
    candidates = list(latest.values())[:2]
    skipped = {'schema_version': 1, 'status': 'withheld', 'applied': False, 'views': [], 'limitations': LIMITATIONS}
    if not candidates:
        return {**skipped, 'reason': 'Add a front-left or front-right fitting photo; side-only arm photos do not constrain front wrap.'}
    gray, shrink = _load(front_photo)
    mask = np.zeros(gray.shape, np.uint8)
    front = next((a for a in annotations if a.get('photo_id') == front_photo['id'] and a.get('part') in {'front', 'shield'}), None)
    lenses = [a for a in annotations if a.get('photo_id') == front_photo['id'] and a.get('part') in {'lens-left', 'lens-right'}]
    if front:
        cv2.fillPoly(mask, [np.round(np.array([[p['x'],p['y']] for p in front['points']])*shrink).astype(np.int32)], 255)
        for lens in lenses:
            cv2.fillPoly(mask, [np.round(np.array([[p['x'],p['y']] for p in lens['points']])*shrink).astype(np.int32)], 0)
    else:
        # Only boundary bands on rimless lenses; interiors/arms are excluded.
        for lens in lenses:
            pts = np.round(np.array([[p['x'],p['y']] for p in lens['points']])*shrink).astype(np.int32)
            cv2.polylines(mask, [pts], True, 255, 5)
    detector = cv2.SIFT_create(nfeatures=2000, contrastThreshold=.015)
    keys, desc = detector.detectAndCompute(gray, mask)
    if desc is None or len(keys) < 20:
        return {**skipped, 'reason': 'Too few stable details on approved front material; no curvature change.'}
    views=[]; matching=[]
    for photo in candidates:
        try:
            other, other_shrink = _load(photo)
        except (OSError, ValueError):
            matching.append({'photo_id':photo['id'], 'reason':'Photo cannot be read.'}); continue
        target, target_desc = detector.detectAndCompute(other, None)
        if target_desc is None:
            continue
        matcher=cv2.BFMatcher(cv2.NORM_L2)
        pairs=matcher.knnMatch(desc,target_desc,k=2)
        reverse=matcher.match(target_desc,desc)
        good=[a for pair in pairs if len(pair)==2 for a,b in [pair] if a.distance<.70*b.distance and reverse[a.trainIdx].trainIdx==a.queryIdx]
        if len(good)<20:
            matching.append({'photo_id':photo['id'],'reason':'Too few mutual distinctive front-surface matches.','matches':len(good)});continue
        ref=np.array([keys[m.queryIdx].pt for m in good],np.float32)
        dst=np.array([target[m.trainIdx].pt for m in good],np.float32)
        _, inliers=cv2.findHomography(ref,dst,cv2.RANSAC,4.,maxIters=2000)
        if inliers is None: continue
        keep=inliers.ravel().astype(bool)
        ref=ref[keep]/shrink;dst=dst[keep]/other_shrink
        if len(ref)>100:
            take=np.linspace(0,len(ref)-1,100).astype(int);ref=ref[take];dst=dst[take]
        xy=(ref-np.asarray(center))*np.asarray(scales)*np.array([1.,-1.])
        views.append({'photo_id':photo['id'],'view':photo['view'],'xy':xy,'uv':dst})
    report=fit_correspondences(views,frame_width/2,initial_sag)
    report['matching']=matching
    report['reference_photo_id']=front_photo['id']
    report['input_photo_ids']=[p['id'] for p in candidates]
    return report
