"""Conservative, photo-derived registration of extra catalogue views.

The homography maps *source pixels* to approved front-photo pixels. It is a
local appearance aid for the front frame, not a 3-D camera calibration or a
claim that temples/occluded surfaces share the front plane.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image


_OPTIONAL_VIEWS = {"inner", "rear", "back", "angled", "angle", "three-quarter", "three_quarter",
                   "oblique", "front-left", "front-right", "front_left", "front_right",
                   "elevated", "high-angle", "high_angle"}
_REAR_VIEWS = {"inner", "rear", "back"}


def _excluded(photo: dict) -> bool:
    labels = " ".join(str(photo.get(k, "")).lower() for k in ("role", "view", "split", "purpose"))
    return bool(photo.get("validation") or photo.get("is_validation") or photo.get("withheld") or
                photo.get("use_for_reconstruction") is False or
                any(x in labels for x in ("validation", "withheld", "holdout", "test-only")))


def _load(photo: dict):
    path = photo.get("path")
    if not path or not Path(path).is_file():
        return None, None
    try:
        with Image.open(path) as handle:
            original_size = handle.size
    except (OSError, ValueError):
        return None, None
    factor = max(original_size) / 2048
    reduction = 8 if factor > 4 else 4 if factor > 2 else 2 if factor > 1 else 1
    flag = {1: cv2.IMREAD_COLOR, 2: cv2.IMREAD_REDUCED_COLOR_2,
            4: cv2.IMREAD_REDUCED_COLOR_4, 8: cv2.IMREAD_REDUCED_COLOR_8}[reduction]
    image = cv2.imread(str(path), flag)
    if image is None:
        return None, None
    if max(image.shape[:2]) > 2048:
        scale = 2048/max(image.shape[:2])
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return image, original_size


def _scale_matrix(image, original_size):
    return np.diag([image.shape[1]/original_size[0], image.shape[0]/original_size[1], 1.])


def _scaled_annotations(annotations, matrix):
    sx, sy = matrix[0, 0], matrix[1, 1]
    return [{**a, "points": [{**p, "x": p["x"]*sx, "y": p["y"]*sy}
                            for p in a.get("points", [])]} for a in annotations]


def _front_bounds(annotations: list[dict], width: int, height: int, *, allow_unapproved=False):
    front = next((a for a in annotations if a.get("part") in ("front", "shield") and
                  (a.get("approved") or allow_unapproved) and len(a.get("points", [])) >= 3), None)
    if front is None:
        # Rimless products have two optical boundaries, not an outer frame.
        lenses = [a for a in annotations if (a.get('approved') or allow_unapproved) and
                  a.get('part') in {'lens-left', 'lens-right'} and len(a.get('points', [])) >= 3]
        if {a['part'] for a in lenses} != {'lens-left', 'lens-right'}:
            return None
        front = {'points': [p for a in lenses for p in a['points']]}
    try:
        pts = np.asarray([(float(p["x"]), float(p["y"])) for p in front["points"]], np.float32)
    except (KeyError, TypeError, ValueError):
        return None
    if not np.isfinite(pts).all() or (pts < 0).any() or (pts[:, 0] > width).any() or (pts[:, 1] > height).any():
        return None
    x, y, w, h = cv2.boundingRect(pts)
    if w < 40 or h < 20:
        return None
    return x, y, w, h


def _features(image, bounds=None):
    height, width = image.shape[:2]
    scale = min(1.0, 1600 / max(height, width))
    small = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else image
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    mask = None
    if bounds is not None:
        x, y, w, h = bounds
        margin = .08 * max(w, h)
        mask = np.zeros(gray.shape, np.uint8)
        cv2.rectangle(mask, (max(0, round((x-margin)*scale)), max(0, round((y-margin)*scale))),
                      (min(gray.shape[1]-1, round((x+w+margin)*scale)),
                       min(gray.shape[0]-1, round((y+h+margin)*scale))), 255, -1)
    detector = cv2.SIFT_create(nfeatures=3500, contrastThreshold=.015)
    keypoints, descriptors = detector.detectAndCompute(gray, mask)
    return keypoints, descriptors, scale


def _hypothesis(source, front, bounds, mirrored, front_features, *, local_components=False):
    work = cv2.flip(source, 1) if mirrored else source
    src_points, src_desc, src_scale = _features(work)
    dst_points, dst_desc, dst_scale = front_features
    if src_desc is None or dst_desc is None or len(src_desc) < 10 or len(dst_desc) < 10:
        return {"reason": "too_few_image_features", "matches": 0, "inliers": 0}
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(src_desc, dst_desc, k=2)
    good = [a for a, b in pairs if a.distance < .72 * b.distance]
    if len(good) < 10:
        return {"reason": "too_few_distinct_matches", "matches": len(good), "inliers": 0}
    src = np.float32([src_points[m.queryIdx].pt for m in good]) / src_scale
    dst = np.float32([dst_points[m.trainIdx].pt for m in good]) / dst_scale
    x, y, w, h = bounds
    threshold = max(2.5, min(8.0, .012*w))
    matrix, inlier_mask = cv2.findHomography(src, dst, cv2.RANSAC, threshold,
                                             maxIters=5000, confidence=.995)
    if matrix is None or inlier_mask is None:
        return {"reason": "homography_failed", "matches": len(good), "inliers": 0}
    inliers = inlier_mask.ravel().astype(bool)
    count = int(inliers.sum())
    result = {"matches": len(good), "inliers": count, "inlier_ratio": round(count / len(good), 3)}
    if count < 10 or count / len(good) < .35:
        return {**result, "reason": "insufficient_consensus"}
    src_in, dst_in = src[inliers], dst[inliers]
    projected = cv2.perspectiveTransform(src_in[:, None, :], matrix)[:, 0, :]
    error = np.linalg.norm(projected - dst_in, axis=1)
    median_error = float(np.median(error))
    result["median_error_px"] = round(median_error, 2)
    spread_x = float(np.ptp(dst_in[:, 0]) / w)
    spread_y = float(np.ptp(dst_in[:, 1]) / h)
    hull_area = cv2.contourArea(cv2.convexHull(dst_in)) / (w*h)
    result.update(front_width_coverage=round(spread_x, 3),
                  front_height_coverage=round(spread_y, 3),
                  matched_area_coverage=round(float(hull_area), 3))
    if median_error > threshold*.7:
        return {**result, "reason": "registration_residual_too_large"}
    if local_components:
        if min(np.ptp(dst_in[:,0]),np.ptp(dst_in[:,1]))<8 or cv2.contourArea(cv2.convexHull(dst_in))<100:
            return {**result,"reason":"insufficient_local_component_coverage"}
    elif spread_x < .22 or spread_y < .22 or hull_area < .035:
        return {**result, "reason": "matches_do_not_cover_frame"}
    # A source->front mapping must remain finite, preserve local orientation,
    # and have a plausible frame-scale Jacobian at the matched region.
    center = np.mean(src_in, axis=0)
    sample = np.float32([[center, center+[1, 0], center+[0, 1]]])
    mapped = cv2.perspectiveTransform(sample, matrix)[0]
    jac = np.column_stack((mapped[1]-mapped[0], mapped[2]-mapped[0]))
    singular = np.linalg.svd(jac, compute_uv=False)
    if not np.isfinite(matrix).all() or not np.isfinite(mapped).all() or singular[0] > 5 or singular[-1] < .12:
        return {**result, "reason": "implausible_local_warp"}
    # SIFT matching on a nearly symmetric frame can swap left and right.
    # Require matching ordering on the frame unless the explicit mirror
    # hypothesis accounts for the reversal.
    if np.linalg.det(jac) <= 0:
        return {**result, "reason": "reversed_local_warp"}
    if mirrored:
        flip = np.array([[-1., 0., source.shape[1]-1.], [0., 1., 0.], [0., 0., 1.]])
        matrix = matrix @ flip
    matrix = matrix / matrix[2, 2]
    confidence = min(1., .25*(count/24) + .25*(count/len(good)) +
                     .25*min(1., spread_x/.7) + .25*min(1., spread_y/.5))
    result.update(reason="registered", confidence=round(float(confidence), 3),
                  source_to_front_homography=np.round(matrix, 9).tolist(), mirrored=mirrored,
                  front_coverage_polygon=np.round(cv2.convexHull(dst_in).reshape(-1, 2), 1).tolist())
    return result


def _partial_lens_registration(source, annotations, front_shape):
    """Align a closed lens opening in a cropped rear catalogue photo.

    A visible second opening establishes which side of the bridge the full
    lens occupies. Without that context a near-symmetric pair is ambiguous.
    """
    height, width = source.shape[:2]
    gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    _, white = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(white, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    area = width * height
    closed = [c for c in contours if .025*area < cv2.contourArea(c) < .22*area and
              all(v > 2 for v in cv2.boundingRect(c)[:2]) and
              cv2.boundingRect(c)[0] + cv2.boundingRect(c)[2] < width-2 and
              cv2.boundingRect(c)[1] + cv2.boundingRect(c)[3] < height-2]
    if len(closed) != 1:
        return {"reason": "lens_opening_ambiguous", "candidate_lenses": len(closed)}
    lens = closed[0]
    sx, sy, sw, sh = cv2.boundingRect(lens)
    partial = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if c is lens or not .012*area < cv2.contourArea(c) < .12*area:
            continue
        if min(y+h, sy+sh)-max(y, sy) < .4*min(h, sh):
            continue
        if x <= 2 and x+w < sx:
            partial.append("left")
        if x+w >= width-2 and x > sx+sw:
            partial.append("right")
    if len(set(partial)) != 1:
        return {"reason": "lens_side_ambiguous", "candidate_lenses": 1}
    # Rear camera viewpoint reverses wearer sides relative to the front view.
    # A fully visible rear image-left lens (neighbor on its right) is the
    # front image-right lens; the x mapping must be reflected as well.
    part = "lens-right" if partial[0] == "right" else "lens-left"
    target = next((a for a in annotations if a.get("part") == part and a.get("approved") and
                   len(a.get("points", [])) >= 8), None)
    if target is None:
        return {"reason": "approved_lens_outline_missing", "candidate_lenses": 1}
    points = np.float32([(p["x"], p["y"]) for p in target["points"]])
    tx, ty, tw, th = cv2.boundingRect(points)
    shape_error = float(cv2.matchShapes(lens, points, cv2.CONTOURS_MATCH_I1, 0))
    matrix = np.array([[-tw/sw, 0., tx+tw+sx*tw/sw],
                       [0., th/sh, ty-sy*th/sh], [0., 0., 1.]])
    warped = cv2.perspectiveTransform(lens.astype(np.float32), matrix)
    mask_a = np.zeros(front_shape[:2], np.uint8)
    mask_b = np.zeros_like(mask_a)
    cv2.fillPoly(mask_a, [np.round(warped).astype(np.int32)], 1)
    cv2.fillPoly(mask_b, [np.round(points).astype(np.int32)], 1)
    union = int(np.logical_or(mask_a, mask_b).sum())
    overlap = float(np.logical_and(mask_a, mask_b).sum() / union) if union else 0.
    diagnostics = {"registration_method": "single_lens_contour", "lens_part": part,
                   "lens_shape_error": round(shape_error, 3), "lens_overlap": round(overlap, 3)}
    if shape_error > .07 or overlap < .68 or not .55 < tw/sw < 2.2 or not .55 < th/sh < 2.2:
        return {**diagnostics, "reason": "lens_contour_inconsistent"}
    # A dilated *lens outline* follows the local rim. A rectangular crop would
    # include the nearby hinge/arm and bridge, which this lens alone cannot
    # register reliably.
    support = np.zeros(front_shape[:2], np.uint8)
    cv2.fillPoly(support, [np.round(points).astype(np.int32)], 255)
    margin = max(8, round(.055*max(tw, th)))
    support = cv2.dilate(support, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                           (2*margin+1, 2*margin+1)))
    support_contours, _ = cv2.findContours(support, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boundary = max(support_contours, key=cv2.contourArea)
    coverage = cv2.approxPolyDP(boundary, 1.5, True).reshape(-1, 2)
    return {**diagnostics, "reason": "registered", "confidence": round(min(.8, overlap*.8), 3),
            "source_to_front_homography": np.round(matrix, 9).tolist(),
            "front_coverage_polygon": np.round(coverage, 1).tolist(),
            "mirrored": True, "front_width_coverage": round(tw/front_shape[1], 3),
            "front_height_coverage": round(th/front_shape[0], 3),
            "orientation_basis": "rear_view_reverses_screen_sides; visible_adjacent_lens_on_" + partial[0]}


def _angled_consistency(source, front):
    """Low-strength appearance/topology check; never supplies a pixel warp."""
    def material_color(image):
        height, width = image.shape[:2]
        n = max(3, min(height, width)//35)
        border = np.concatenate((image[:n].reshape(-1, 3), image[-n:].reshape(-1, 3),
                                 image[:, :n].reshape(-1, 3), image[:, -n:].reshape(-1, 3)))
        border_lab = cv2.cvtColor(border.reshape(-1, 1, 3), cv2.COLOR_BGR2LAB).reshape(-1, 3)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        distance = np.linalg.norm(lab.astype(np.float32)-np.median(border_lab, axis=0), axis=2)
        foreground = lab[distance > 20]
        if len(foreground) < .01*width*height:
            return None
        return np.median(foreground, axis=0)

    front_color = material_color(front)
    source_color = material_color(source)
    if front_color is None or source_color is None:
        return {"reason": "frame_material_not_isolated"}
    color_delta = float(np.linalg.norm(front_color-source_color))
    gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    _, white = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(white, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    height, width = source.shape[:2]
    apertures = []
    for contour in contours:
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        if (.015*width*height < area < .18*width*height and
                x > 2 and y > 2 and x+w < width-2 and y+h < height-2 and
                .09*width < w < .55*width and .1*height < h < .7*height):
            apertures.append(contour)
    result = {"color_delta_lab": round(color_delta, 2), "lens_openings_observed": len(apertures),
              "registration_method": "angled_color_and_aperture_check"}
    if color_delta > 18 or len(apertures) != 2:
        return {**result, "reason": "angled_evidence_inconsistent"}
    return {**result, "reason": "appearance_consistent", "confidence": round(max(.35, .7-color_delta/60), 3)}


def analyze_multiview(photos: list[dict], front_photo: dict,
                      front_annotations: list[dict], *, allow_unapproved=False, component_registration=False) -> dict:
    """Inspect the latest eligible optional views, with no new-photo requirement.

    Each accepted rear/inner registration can guide front-frame inner-surface
    appearance. An angled registration is a consistency observation only.
    Uncertain photos retain diagnostics but never expose a usable warp.
    """
    front, front_size = _load(front_photo)
    if front is None:
        return {"front_photo_id": str(front_photo.get("id", "")), "views": [],
                "warning": "Front photo cannot be decoded for multiview registration"}
    front_scale = _scale_matrix(front, front_size)
    scaled_annotations = _scaled_annotations(front_annotations, front_scale)
    bounds = _front_bounds(scaled_annotations, front.shape[1], front.shape[0], allow_unapproved=allow_unapproved)
    if bounds is None:
        return {"front_photo_id": str(front_photo.get("id", "")), "views": [],
                "warning": "Approved front outline is unavailable for multiview registration"}
    latest = {}
    for photo in photos:
        view = str(photo.get("view", "")).lower()
        if view in _OPTIONAL_VIEWS and not _excluded(photo):
            latest[view] = photo
    output = []
    front_features = _features(front, bounds)
    for view, photo in latest.items():
        photo_id = str(photo.get("id", ""))
        source, source_size = _load(photo)
        if source is None:
            output.append({"photo_id": photo_id, "view": view, "status": "skipped",
                           "reason": "photo_cannot_be_decoded", "usable_for_inner_uv": False})
            continue
        attempts = [_hypothesis(source, front, bounds, False, front_features)]
        if view in _REAR_VIEWS:
            attempts.append(_hypothesis(source, front, bounds, True, front_features))
        accepted = [a for a in attempts if a["reason"] == "registered"]
        if view in _REAR_VIEWS:
            # A rear-facing camera reverses wearer-side correspondence.
            # An unmirrored feature fit may be a visual coincidence or an
            # editorially flipped product photo; it cannot establish side.
            accepted = [a for a in accepted if a.get("mirrored")]
        best = max(accepted or attempts, key=lambda a: (a.get("inliers", 0),
                                                        a.get("front_width_coverage", 0),
                                                        -a.get("median_error_px", 1e9)))
        if view in _REAR_VIEWS and best["reason"] == "registered" and not best.get("mirrored"):
            best = {k: v for k, v in best.items() if k not in
                    ("source_to_front_homography", "front_coverage_polygon")}
            best["reason"] = "rear_orientation_unverified"
        if len(accepted) > 1:
            runner_up = min(accepted, key=lambda a: a.get("inliers", 0))
            if runner_up["inliers"] >= .8 * best["inliers"]:
                best = {k: v for k, v in best.items() if k not in
                        ("source_to_front_homography", "front_coverage_polygon")}
                best["reason"] = "mirror_orientation_ambiguous"
        if best["reason"] != "registered" and view in _REAR_VIEWS:
            contour = _partial_lens_registration(source, scaled_annotations, front.shape)
            if contour["reason"] == "registered":
                best = contour
        if best["reason"] != "registered" and view not in _REAR_VIEWS:
            consistency = _angled_consistency(source, front)
            if consistency["reason"] == "appearance_consistent":
                best = consistency
        if best["reason"] == "registered":
            source_scale = _scale_matrix(source, source_size)
            matrix = np.asarray(best["source_to_front_homography"], np.float64)
            original_matrix = np.linalg.inv(front_scale) @ matrix @ source_scale
            original_matrix /= original_matrix[2, 2]
            best["source_to_front_homography"] = np.round(original_matrix, 9).tolist()
            polygon = np.asarray(best["front_coverage_polygon"], np.float64)
            polygon[:, 0] /= front_scale[0, 0]
            polygon[:, 1] /= front_scale[1, 1]
            best["front_coverage_polygon"] = np.round(polygon, 1).tolist()
        registered = best["reason"] in ("registered", "appearance_consistent")
        result = {"photo_id": photo_id, "view": view,
                  "status": "accepted" if registered else "review",
                  "reason": best["reason"], "usable_for_inner_uv": best["reason"] == "registered" and view in _REAR_VIEWS,
                  "geometry_role": "front_frame_appearance" if view in _REAR_VIEWS else "angled_consistency_only",
                  "usage": ("Eligible for partial inner-frame texture" if registered and view in _REAR_VIEWS else
                            "Consistency check only" if registered else "No photo pixels used"),
                  **{k: v for k, v in best.items() if k != "reason"}}
        # An uncertain hypothesis must never become an accidental texture warp.
        if not registered:
            result.pop("source_to_front_homography", None)
            result["confidence"] = 0.0
        if component_registration and not result.get('source_to_front_homography') and view not in _REAR_VIEWS:
            local=_hypothesis(source,front,bounds,False,front_features,local_components=True)
            if local.get('reason')=='registered':
                source_scale=_scale_matrix(source,source_size)
                matrix=np.linalg.inv(front_scale) @ np.asarray(local['source_to_front_homography']) @ source_scale
                matrix/=matrix[2,2]
                polygon=np.asarray(local['front_coverage_polygon'])
                polygon[:,0]/=front_scale[0,0];polygon[:,1]/=front_scale[1,1]
                result['component_registration']=dict(source_to_front_homography=matrix.tolist(),
                    front_coverage_polygon=polygon.tolist(),inliers=local['inliers'],
                    median_error_px=local['median_error_px'],method='local-inlier-hull-only')
        output.append(result)
    return {"front_photo_id": str(front_photo.get("id", "")), "views": output,
            "method": "SIFT/RANSAC or guarded partial lens contour for planar appearance; angled color/aperture consistency"}


def save_registration_overlays(report: dict, photos: list[dict], front_photo: dict,
                               output_dir: str | Path) -> dict[str, str]:
    """Write inspectable front/source overlays for accepted registrations.

    The yellow boundary is the *supported* coverage region, not an assertion
    that the whole source image should texture the frame.
    """
    front, front_size = _load(front_photo)
    if front is None:
        return {}
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    paths = {}
    for record in report.get("views", []):
        if record.get("status") not in ("accepted", "used") or "source_to_front_homography" not in record:
            continue
        photo = next((p for p in photos if str(p.get("id")) == record.get("photo_id")), None)
        source, source_size = _load(photo) if photo else (None, None)
        if source is None:
            continue
        matrix = (_scale_matrix(front, front_size) @
                  np.asarray(record["source_to_front_homography"], np.float64) @
                  np.linalg.inv(_scale_matrix(source, source_size)))
        warped = cv2.warpPerspective(source, matrix, (front.shape[1], front.shape[0]),
                                     borderValue=(255, 255, 255))
        polygon = np.asarray(record["front_coverage_polygon"], np.float64)
        polygon[:, 0] *= front.shape[1]/front_size[0]
        polygon[:, 1] *= front.shape[0]/front_size[1]
        polygon = np.round(polygon).astype(np.int32)
        mask = np.zeros(front.shape[:2], np.uint8)
        cv2.fillPoly(mask, [polygon], 255)
        blended = cv2.addWeighted(front, .5, warped, .5, 0)
        overlay = np.where(mask[:, :, None] > 0, blended, front)
        cv2.polylines(overlay, [polygon], True, (0, 210, 255), 2)
        safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in record["photo_id"])
        path = directory / f"{safe_id}-registration.png"
        if cv2.imwrite(str(path), overlay):
            paths[record["photo_id"]] = str(path)
    return paths
