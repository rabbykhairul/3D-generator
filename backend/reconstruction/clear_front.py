"""Conservative luminance-only recovery of a water-clear frame on white.

This is a fallback for an otherwise undetected front photo. Its return value is
an unapproved pixel mask; the ordinary contour and geometry validation still
decide which suggestions may be displayed.
"""

import cv2
import numpy as np
from shapely.geometry import Polygon, box


def _paired_apertures(mask):
    height, width = mask.shape
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return None
    parents = hierarchy[0, :, 3]
    outers = sorted(((cv2.contourArea(c), i) for i, c in enumerate(contours)
                     if parents[i] == -1), reverse=True)
    if not outers:
        return None
    area, outer_i = outers[0]
    x, y, w, h = cv2.boundingRect(contours[outer_i])
    if not (.08 <= area / (width * height) <= .65 and
            w >= width * .45 and h >= height * .12 and
            x > 2 and y > 2 and x + w < width - 2 and y + h < height - 2):
        return None
    holes = sorted(((cv2.contourArea(c), i) for i, c in enumerate(contours)
                    if parents[i] == outer_i), reverse=True)
    if len(holes) < 2 or min(holes[0][0], holes[1][0]) < area * .2:
        return None
    # Extra small holes from reflections or clear arms are tolerable. A third
    # aperture of comparable size means this is not a stable two-lens frame.
    if len(holes) > 2 and holes[2][0] > holes[1][0] * .1:
        return None
    pair = sorted(holes[:2], key=lambda item: cv2.boundingRect(contours[item[1]])[0])
    left, right = [cv2.boundingRect(contours[i]) for _, i in pair]
    if (pair[0][0] / pair[1][0] < .65 or
            right[0] <= left[0] + left[2] or
            min(left[2], right[2]) < w * .2 or
            min(left[3], right[3]) < h * .45 or
            abs((left[1] + left[3] / 2) - (right[1] + right[3] / 2)) > h * .15):
        return None
    return (x, y, w, h), (left, right)


def _simple_rim_mask(mask, faint_distance=None):
    """Remove one-pixel contour returns without changing the observed shape.

    Transparent edges often form loops through a photographed arm or nose pad.
    Keep only a dominant repaired outer component, complete the two measured
    apertures across the arm occlusions, and leave a two-pixel pixel-space rim.
    Refuse the repair if it changes appreciable area or loses a simple hole.
    """
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return None
    parents = hierarchy[0, :, 3]
    outer_i = max((i for i, parent in enumerate(parents) if parent == -1),
                  key=lambda i: cv2.contourArea(contours[i]), default=None)
    if outer_i is None:
        return None
    original_outer = Polygon(contours[outer_i].reshape(-1, 2))
    outer = original_outer.buffer(0)
    if outer.geom_type == 'MultiPolygon':
        outer = max(outer.geoms, key=lambda part: part.area)
    if (outer.geom_type != 'Polygon' or not outer.is_valid or
            outer.area < original_outer.area * .995):
        return None
    # Clear material can disappear at a specular highlight. A threshold then
    # returns down an INTERNAL reflection and back out again, despite stable
    # aperture boxes. Close only narrow returns in the filled silhouette,
    # never in the thin frame annulus (which would erase the rim). Bound displacement
    # and area; this cannot recover large missing or hidden physical edges.
    radius = max(1., min(mask.shape) * .01)
    candidate = outer.buffer(radius).buffer(-radius)
    if (candidate.geom_type == 'Polygon' and candidate.is_valid
            and abs(candidate.area - outer.area) < outer.area * .045
            and candidate.difference(outer.buffer(radius * 1.1)).is_empty
            and outer.difference(candidate.buffer(radius * 1.1)).is_empty):
        outer = candidate
    aperture_contours = sorted((contours[i] for i, parent in enumerate(parents)
                                if parent == outer_i), key=cv2.contourArea, reverse=True)[:2]
    if len(aperture_contours) != 2:
        return None
    if faint_distance is not None:
        # The stable apertures establish topology, but a strong material
        # threshold cuts through water-clear plastic. Independently recover
        # its faint upper silhouette. Do not extend this low threshold into
        # the lower contact-shadow band.
        lens_boxes = [cv2.boundingRect(c) for c in aperture_contours]
        cutoff = min(y + h*.82 for x,y,w,h in lens_boxes)
        region = box(0, 0, mask.shape[1], cutoff)
        candidates = []
        for threshold in (4., 5., 6.):
            weak = np.uint8(faint_distance > threshold)*255
            contours_weak, _ = cv2.findContours(weak, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours_weak:
                continue
            p = Polygon(max(contours_weak, key=cv2.contourArea).reshape(-1,2)).buffer(0)
            if p.geom_type == 'MultiPolygon':
                largest = max(p.geoms, key=lambda item: item.area)
                if largest.area >= p.area*.995:
                    p = largest
            if p.geom_type != 'Polygon':
                continue
            cleaned = p.buffer(radius*1.2).buffer(-radius*1.2).buffer(-radius*.3).buffer(radius*.3)
            if cleaned.geom_type == 'Polygon' and cleaned.area >= p.area*.995:
                p = cleaned
            combined = outer.difference(region).union(p.intersection(region)).buffer(0)
            if (combined.geom_type == 'Polygon' and combined.is_valid
                    and .98 < combined.area/outer.area < 1.18
                    and max(abs(a-b) for a,b in zip(combined.bounds,outer.bounds)) < radius):
                candidates.append(combined)
        # Require agreement of the actual boundary, not merely bounding boxes.
        stable_region = box(0,0,mask.shape[1],min(y+h*.62 for x,y,w,h in lens_boxes))
        if len(candidates) == 3 and sum(candidates[1].intersection(stable_region).boundary.hausdorff_distance(
                p.intersection(stable_region).boundary) < radius*.6 for p in candidates) >= 2:
            outer = candidates[1]
    inner_limit = outer.buffer(-max(1., min(mask.shape) * .002))
    if inner_limit.is_empty:
        return None
    apertures = []
    for contour in aperture_contours:
        observed = Polygon(contour.reshape(-1, 2))
        completed = observed.convex_hull
        clipped = completed.intersection(inner_limit)
        if (clipped.geom_type != 'Polygon' or not clipped.is_valid or
                clipped.area < completed.area * .99 or
                completed.area > observed.area * 1.15):
            return None
        apertures.append(clipped)
    if apertures[0].intersects(apertures[1]):
        return None
    repaired = np.zeros_like(mask)
    cv2.fillPoly(repaired, [np.rint(outer.exterior.coords).astype(np.int32)], 255)
    for aperture in apertures:
        cv2.fillPoly(repaired, [np.rint(aperture.exterior.coords).astype(np.int32)], 0)
    if _paired_apertures(repaired) is None:
        return None
    return repaired


def recover_clear_front_mask(image):
    """Use a common analysis scale; return a mask in the caller's coordinates."""
    height, width = image.shape[:2]
    if max(height, width) != 1080 and min(height, width) >= 320:
        factor = 1080 / max(height, width)
        work = cv2.resize(image, (round(width*factor), round(height*factor)),
                          interpolation=cv2.INTER_AREA if factor < 1 else cv2.INTER_CUBIC)
        mask, info = _recover_clear_front_mask(work)
        if mask is not None:
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        return mask, info
    return _recover_clear_front_mask(image)


def _recover_clear_front_mask(image):
    """Return a neutral-rim mask only when outer and two holes persist.

    Requires a nearly uniform white background and four neighboring LAB
    distance thresholds with matching boxes. This intentionally declines
    cropped frames and less structured shadows.
    """
    height, width = image.shape[:2]
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    border_size = max(3, min(height, width) // 30)
    border = np.concatenate((lab[:border_size].reshape(-1, 3),
                             lab[-border_size:].reshape(-1, 3),
                             lab[:, :border_size].reshape(-1, 3),
                             lab[:, -border_size:].reshape(-1, 3)))
    background = np.median(border, axis=0)
    noise = float(np.percentile(np.linalg.norm(border - background, axis=1), 90))
    if background[0] < 245 or noise > 4:
        return None, 'The background is too varied for neutral clear-frame extraction.'
    distance = np.linalg.norm(lab - background, axis=2)
    candidates = []
    for threshold in (8, 10, 12, 16, 20, 24, 30):
        if threshold < max(8, noise * 2.5):
            continue
        mask = np.uint8(distance > threshold) * 255
        geometry = _paired_apertures(mask)
        if geometry is not None:
            candidates.append((threshold, mask, *geometry))
    tolerance = max(width, height) * .025
    stable = []
    for threshold, mask, box, holes in candidates:
        matching = 0
        for other_threshold, _, other_box, other_holes in candidates:
            if other_threshold == threshold or abs(other_threshold - threshold) > 12:
                continue
            if (max(abs(a - b) for a, b in zip(box, other_box)) <= tolerance and
                    all(max(abs(a - b) for a, b in zip(hole, other)) <= tolerance
                        for hole, other in zip(holes, other_holes))):
                matching += 1
        if matching >= 3:
            stable.append((threshold, mask))
    if stable:
        threshold, mask = min(stable, key=lambda item: abs(item[0] - 16))
        # Suppress pixel noise before measuring very faint neutral contrast.
        sigma = max(.6, min(height, width)/720.)
        faint_distance = cv2.GaussianBlur(distance, (0,0), sigma)
        repaired = _simple_rim_mask(mask, faint_distance)
        if repaired is not None:
            return repaired, threshold
        return None, 'Neutral rim topology crosses photographed hardware; review the outlines manually.'
    return None, 'Neutral rim and paired lens openings are not stable across contrast levels; review the outlines manually.'
