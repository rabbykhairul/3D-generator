"""Bounded smoothing for *suggested* closed photo contours.

Coordinates are in original photograph pixels.  This module deliberately has no
knowledge of saved/approved annotations: callers must apply it only while
creating a new, editable extraction suggestion.
"""

from __future__ import annotations

import math

import numpy as np
from shapely.geometry import Point, Polygon


def _sample_at(points: np.ndarray, cumulative: np.ndarray, distance: float) -> np.ndarray:
    perimeter = cumulative[-1]
    distance %= perimeter
    index = min(int(np.searchsorted(cumulative, distance, side="right") - 1), len(points) - 1)
    length = cumulative[index + 1] - cumulative[index]
    fraction = (distance - cumulative[index]) / length
    return points[index] * (1 - fraction) + points[(index + 1) % len(points)] * fraction


def _turn_degrees(before: np.ndarray, at: np.ndarray, after: np.ndarray) -> float:
    incoming, outgoing = at - before, after - at
    lengths = np.linalg.norm(incoming) * np.linalg.norm(outgoing)
    if lengths < 1e-10:
        return 0.0
    return math.degrees(math.acos(float(np.clip(np.dot(incoming, outgoing) / lengths, -1, 1))))


def refine_closed_contour(
    points: np.ndarray | list,
    *,
    max_displacement_px: float = 0.75,
    sample_spacing_px: float = 5.0,
    window_px: float = 5.0,
) -> tuple[np.ndarray, dict]:
    """Refine a pixel contour without inventing a different physical silhouette.

    Returns ``(Nx2 points, diagnostics)``.  The tolerance is a *boundary*
    displacement in original image pixels. Corners, extrema and straight bars
    are kept. Invalid or excessive changes return the input unchanged.
    """
    raw = np.asarray(points, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] != 2 or len(raw) < 3 or not np.isfinite(raw).all():
        raise ValueError("contour must contain at least three finite 2D points")
    if np.linalg.norm(raw[0] - raw[-1]) < 1e-9:
        raw = raw[:-1]
    if len(raw) < 3:
        raise ValueError("contour must contain at least three distinct points")
    source_poly = Polygon(raw)
    if not source_poly.is_valid or source_poly.area <= 0:
        return raw.copy(), {"applied": False, "reason": "invalid_source", "max_boundary_displacement_px": 0.0}
    lengths = np.linalg.norm(np.roll(raw, -1, axis=0) - raw, axis=1)
    if np.any(lengths < 1e-8):
        return raw.copy(), {"applied": False, "reason": "duplicate_point", "max_boundary_displacement_px": 0.0}
    cumulative = np.r_[0.0, np.cumsum(lengths)]
    perimeter = float(cumulative[-1])
    if perimeter < 3 * window_px or max_displacement_px <= 0 or sample_spacing_px <= 0:
        return raw.copy(), {"applied": False, "reason": "insufficient_scale", "max_boundary_displacement_px": 0.0}

    # Inspect turning over several pixels. One-pixel staircase corners disappear
    # at this scale, while a genuine bridge or endpiece corner remains sharp.
    turns = np.zeros(len(raw))
    for i, distance in enumerate(cumulative[:-1]):
        half_window = min(window_px, perimeter / 12)
        before = _sample_at(raw, cumulative, distance - half_window)
        after = _sample_at(raw, cumulative, distance + half_window)
        turns[i] = _turn_degrees(before, raw[i], after)
    locked = turns >= 32.0
    for dimension in (0, 1):
        locked[np.argmin(raw[:, dimension])] = True
        locked[np.argmax(raw[:, dimension])] = True
    # Preserve short neighborhoods around construction corners. Without this,
    # adjacent smoothing can round a corner despite keeping its apex fixed.
    for i in np.flatnonzero(locked).tolist():
        travel = 0.0
        for step in (-1, 1):
            travel = 0.0
            j = i
            while travel < 0.5 * window_px:
                next_j = (j + step) % len(raw)
                travel += lengths[j if step > 0 else next_j]
                if travel <= 0.5 * window_px:
                    locked[next_j] = True
                j = next_j
                if j == i:
                    break

    moved = raw.copy()
    for i, distance in enumerate(cumulative[:-1]):
        if locked[i]:
            continue
        before = _sample_at(raw, cumulative, distance - window_px)
        after = _sample_at(raw, cumulative, distance + window_px)
        chord = after - before
        length_sq = float(chord @ chord)
        if length_sq < 1e-8:
            continue
        projection = before + np.clip(float((raw[i] - before) @ chord) / length_sq, 0, 1) * chord
        shift = projection - raw[i]
        magnitude = float(np.linalg.norm(shift))
        if magnitude > 1e-10:
            moved[i] += shift * min(0.70, max_displacement_px / magnitude)

    # Cubic subdivision smooths broad arcs without changing the straight side
    # of a thin bridge. A segment is flattened if its cubic strays too far from
    # the original boundary. That also prevents spline overshoot at tight joins.
    result: list[np.ndarray] = []
    for i in range(len(raw)):
        next_i = (i + 1) % len(raw)
        start, end = moved[i], moved[next_i]
        segment = end - start
        count = max(1, int(math.ceil(float(np.linalg.norm(segment)) / sample_spacing_px)))
        tangent_start = (moved[next_i] - moved[(i - 1) % len(raw)]) * 0.5 if not locked[i] else segment
        tangent_end = (moved[(i + 2) % len(raw)] - moved[i]) * 0.5 if not locked[next_i] else segment
        samples = []
        for j in range(count):
            t = j / count
            cubic = ((2*t**3 - 3*t**2 + 1)*start + (t**3 - 2*t**2 + t)*tangent_start
                     + (-2*t**3 + 3*t**2)*end + (t**3 - t**2)*tangent_end)
            samples.append(cubic)
        # A cubic segment may not create a physical protrusion from photograph
        # aliasing. The per-segment cap is additionally checked globally below.
        source_boundary = source_poly.boundary
        if any(source_boundary.distance(Point(p)) > max_displacement_px * .95 for p in samples):
            samples = [start + (j / count) * segment for j in range(count)]
        result.extend(samples)
    refined = np.asarray(result)
    candidate = Polygon(refined)
    displacement = float(source_poly.boundary.hausdorff_distance(candidate.boundary)) if candidate.is_valid else math.inf
    if (not candidate.is_valid or candidate.area <= 0 or displacement > max_displacement_px + 1e-7
            or abs(candidate.area - source_poly.area) > max(source_poly.area * .025, perimeter * max_displacement_px)):
        return raw.copy(), {"applied": False, "reason": "topology_or_displacement_guard", "max_boundary_displacement_px": 0.0}
    return refined, {"applied": True, "max_boundary_displacement_px": displacement,
                     "locked_corners": int(locked.sum()), "source_points": len(raw), "refined_points": len(refined)}
