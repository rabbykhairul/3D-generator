"""Build a conservative inside-frame appearance texture from registered views."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def _polygon(annotation, width, height, scale_x=1., scale_y=1.):
    try:
        points = np.asarray([(p["x"]*scale_x, p["y"]*scale_y)
                             for p in annotation["points"]], np.float32)
    except (KeyError, TypeError, ValueError):
        return None
    if (len(points) < 3 or not np.isfinite(points).all() or (points < 0).any() or
            (points[:, 0] > width).any() or (points[:, 1] > height).any()):
        return None
    return np.round(points).astype(np.int32)


def _frame_mask(annotations, width, height, scale_x=1., scale_y=1.):
    mask = np.zeros((height, width), np.uint8)
    outer = next((a for a in annotations if a.get("approved") and
                  a.get("part") in ("front", "shield")), None)
    polygon = _polygon(outer, width, height, scale_x, scale_y) if outer else None
    if polygon is None:
        return mask
    cv2.fillPoly(mask, [polygon], 255)
    for annotation in annotations:
        if annotation.get("approved") and annotation.get("part") in (
                "lens-left", "lens-right", "bridge-opening"):
            polygon = _polygon(annotation, width, height, scale_x, scale_y)
            if polygon is not None:
                cv2.fillPoly(mask, [polygon], 0)
    return mask


def _source_material_mask(image):
    height, width = image.shape[:2]
    n = max(3, min(height, width)//35)
    border = np.concatenate((image[:n].reshape(-1, 3), image[-n:].reshape(-1, 3),
                             image[:, :n].reshape(-1, 3), image[:, -n:].reshape(-1, 3)))
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    border_lab = cv2.cvtColor(border.reshape(-1, 1, 3), cv2.COLOR_BGR2LAB).reshape(-1, 3)
    background = np.median(border_lab, axis=0)
    noise = np.percentile(np.linalg.norm(border_lab-background, axis=1), 85)
    distance = np.linalg.norm(lab-background, axis=2)
    return np.uint8(distance > max(14., noise*2.5)) * 255


def _load_bounded(path, limit=2048):
    try:
        with Image.open(path) as im:
            original_width, original_height = im.size
    except (OSError, ValueError):
        return None, None
    factor = max(original_width, original_height) / limit
    reduction = 8 if factor > 4 else 4 if factor > 2 else 2 if factor > 1 else 1
    flag = {1: cv2.IMREAD_COLOR, 2: cv2.IMREAD_REDUCED_COLOR_2,
            4: cv2.IMREAD_REDUCED_COLOR_4, 8: cv2.IMREAD_REDUCED_COLOR_8}[reduction]
    image = cv2.imread(str(path), flag)
    if image is None:
        return None, None
    if max(image.shape[:2]) > limit:
        scale = limit/max(image.shape[:2])
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return image, (original_width, original_height)


def build_registered_inner_texture(front_photo: dict, front_annotations: list[dict],
                                   photos: list[dict], report: dict) -> tuple[Image.Image | None, dict]:
    """Return (RGB texture, coverage diagnostics) for the inside front frame.

    The texture uses only registered inner/rear pixels on the approved frame
    material. It does not project rear arms, white background, or unseen surface
    detail. Unobserved material gets a smooth median front-frame color.
    """
    front_path = front_photo.get("path")
    front, original_front_size = (_load_bounded(front_path) if front_path and Path(front_path).is_file()
                                  else (None, None))
    if front is None:
        return None, {"reason": "front_photo_unavailable", "applied": []}
    height, width = front.shape[:2]
    front_sx, front_sy = width/original_front_size[0], height/original_front_size[1]
    frame = _frame_mask(front_annotations, width, height, front_sx, front_sy)
    total = int(np.count_nonzero(frame))
    if total == 0:
        return None, {"reason": "approved_front_material_unavailable", "applied": []}
    # A solid, low-frequency base avoids copying outer logos or arm markings to
    # the interior where no inner photo observed them.
    frame_pixels = front[frame > 0]
    material = np.median(frame_pixels, axis=0).astype(np.uint8)
    texture = np.empty_like(front)
    texture[:] = material
    covered = np.zeros((height, width), np.uint8)
    applied = []
    for record in report.get("views", []):
        if record.get("status") not in ("accepted", "used") or not record.get("usable_for_inner_uv"):
            continue
        photo_id = str(record.get("photo_id", ""))
        photo = next((p for p in photos if str(p.get("id")) == photo_id), None)
        path = photo.get("path") if photo else None
        source, original_source_size = (_load_bounded(path) if path and Path(path).is_file()
                                        else (None, None))
        if source is None:
            continue
        try:
            matrix = np.asarray(record["source_to_front_homography"], np.float64)
            polygon = np.asarray(record["front_coverage_polygon"], np.float32)
            if matrix.shape != (3, 3) or polygon.shape[0] < 3 or not np.isfinite(matrix).all():
                continue
        except (KeyError, TypeError, ValueError):
            continue
        source_sx = source.shape[1]/original_source_size[0]
        source_sy = source.shape[0]/original_source_size[1]
        matrix = (np.diag([front_sx, front_sy, 1.]) @ matrix @
                  np.diag([1/source_sx, 1/source_sy, 1.]))
        polygon[:, 0] *= front_sx
        polygon[:, 1] *= front_sy
        polygon = np.round(polygon).astype(np.int32)
        coverage = np.zeros((height, width), np.uint8)
        cv2.fillPoly(coverage, [polygon], 255)
        source_mask = _source_material_mask(source)
        warp = cv2.warpPerspective(source, matrix, (width, height), flags=cv2.INTER_LINEAR)
        valid = cv2.warpPerspective(source_mask, matrix, (width, height),
                                    flags=cv2.INTER_NEAREST) > 0
        valid &= (frame > 0) & (coverage > 0)
        # Prefer the first accepted view for overlapping evidence. No averaging
        # of inconsistent photos, since their lighting may differ substantially.
        valid &= covered == 0
        pixels = int(np.count_nonzero(valid))
        if pixels:
            texture[valid] = warp[valid]
            covered[valid] = 255
            record["status"] = "used"
            record["usage"] = "Partial registered inner-frame texture"
            record["applied_frame_coverage_fraction"] = round(pixels/total, 4)
            applied.append({"photo_id": photo_id, "applied_pixels": pixels,
                            "frame_coverage_fraction": round(pixels/total, 4),
                            "registration_method": record.get("registration_method", "feature_homography")})
        else:
            record["usage"] = "Registered, but no front-frame material pixels covered"
    if not np.any(covered):
        return None, {"reason": "no_usable_inner_registration", "applied": [],
                      "frame_material_pixels": total, "observed_fraction": 0.0}
    diagnostics = {"reason": "registered_inner_appearance" if applied else "no_usable_inner_registration",
                   "applied": applied, "frame_material_pixels": total,
                   "observed_fraction": round(int(np.count_nonzero(covered))/total, 4),
                   "unobserved_fill": "median_front_material_color"}
    return Image.fromarray(cv2.cvtColor(texture, cv2.COLOR_BGR2RGB)), diagnostics
