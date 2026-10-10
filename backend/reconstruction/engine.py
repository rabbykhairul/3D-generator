from __future__ import annotations

import json
import math
import re
import time
from pathlib import Path

import cv2
import mapbox_earcut
import numpy as np
from PIL import Image
from scipy.optimize import least_squares
from shapely.geometry import Polygon, LineString, Point
from shapely.ops import triangulate
from shapely.validation import explain_validity
import trimesh

from ..engine_version import ENGINE_VERSION


class NeedsCorrection(ValueError):
    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message)
        self.details = details or {}


def _points(annotation: dict, width: int, height: int) -> np.ndarray:
    pts = np.asarray([[p["x"], p["y"]] for p in annotation.get("points", [])], dtype=float)
    part = annotation.get("part", "outline")
    if pts.ndim != 2 or pts.shape[0] < 3 or pts.shape[1] != 2:
        raise NeedsCorrection(f"{part} needs at least three outline points", {"part": part})
    if not np.isfinite(pts).all() or np.any(pts < 0) or np.any(pts[:, 0] > width) or np.any(pts[:, 1] > height):
        raise NeedsCorrection(f"{part} contains points outside its photo", {"part": part})
    poly = Polygon(pts)
    if not poly.is_valid or poly.area < 8:
        raise NeedsCorrection(f"{part} is self-intersecting or too small", {"part": part})
    return pts


def _color(s: str, opacity: float = 1.0) -> np.ndarray:
    try:
        h = s.lstrip("#")
        rgb = [int(h[i:i + 2], 16) for i in (0, 2, 4)]
        if len(h) != 6:
            raise ValueError
    except (ValueError, IndexError):
        rgb = [40, 44, 52]
    srgb = np.asarray(rgb, dtype=float)/255
    linear = np.where(srgb<=.04045,srgb/12.92,((srgb+.055)/1.055)**2.4)
    # Texture pixels remain sRGB; glTF material multipliers are linear.
    return np.array([*np.rint(linear*255), round(255 * opacity)], dtype=np.uint8)


def _material(color: str, opacity: float, metal: float, rough: float, image=None):
    kwargs = dict(baseColorFactor=_color(color, opacity), metallicFactor=float(metal),
                  roughnessFactor=float(rough), alphaMode="BLEND" if opacity < 0.999 else "OPAQUE",
                  doubleSided=True)
    if image is not None:
        kwargs["baseColorTexture"] = image
    return trimesh.visual.material.PBRMaterial(**kwargs)


def _curve_z(x_mm: float, half_width_mm: float, sag_mm: float) -> float:
    # Front plane remains at the bridge; wrap moves outer edges rearward.
    return -sag_mm * min(1.0, (x_mm / max(half_width_mm, 1e-6)) ** 2)


def _split_t_junctions(triangles: np.ndarray, points: np.ndarray) -> list[tuple[int, int, int]]:
    """Keep all contour vertices on triangulated boundary edges for closed walls."""
    stack = [tuple(map(int, t)) for t in triangles]
    complete = []
    while stack:
        tri = stack.pop()
        triangle = points[list(tri)]
        first = triangle[1] - triangle[0]
        second = triangle[2] - triangle[0]
        if abs(float(first[0] * second[1] - first[1] * second[0])) < 1e-9:
            # Earcut can emit a flat triangle along a collinear contour run.
            # It adds a third face to otherwise shared edges and has no area.
            continue
        for edge in range(3):
            a, b, c = tri[edge], tri[(edge + 1) % 3], tri[(edge + 2) % 3]
            vector = points[b] - points[a]
            length_sq = float(vector @ vector)
            if length_sq == 0:
                raise NeedsCorrection("Contour contains duplicate consecutive points")
            relative = points - points[a]
            progress = relative @ vector / length_sq
            cross = np.abs(relative[:, 0] * vector[1] - relative[:, 1] * vector[0])
            candidates = np.flatnonzero((progress > 1e-9) & (progress < 1 - 1e-9) &
                                        (cross < 1e-8 * max(1., math.sqrt(length_sq))))
            candidates = candidates[~np.isin(candidates, tri)]
            if len(candidates):
                line = [a, *sorted((int(i) for i in candidates), key=lambda i: progress[i]), b]
                stack.extend((line[i], line[i + 1], c) for i in range(len(line) - 1))
                break
        else:
            complete.append(tri)
    return complete


def _foreground_mask(image):
    """Contrast segmentation for isolated catalogue photos, not a universal matte.

    A fixed 16-Lab-unit cutoff accepted soft neutral contact shadows. For a
    uniform light backdrop use the image's foreground/background split instead.
    No erosion/opening of the silhouette: a dark one-pixel wire is still material.
    Low-contrast or busy captures keep the conservative fallback and need review.
    """
    height, width = image.shape[:2]
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    border = np.concatenate((lab[:max(3, height // 30)].reshape(-1, 3),
                             lab[-max(3, height // 30):].reshape(-1, 3),
                             lab[:, :max(3, width // 30)].reshape(-1, 3),
                             lab[:, -max(3, width // 30):].reshape(-1, 3)))
    background = np.median(border, axis=0)
    noise = float(np.percentile(np.linalg.norm(border - background, axis=1), 90))
    distance = np.linalg.norm(lab - background, axis=2)
    threshold = max(16.0, noise * 2.2)
    separated = False
    if background[0] > 210 and noise < 12:
        split, _ = cv2.threshold(np.uint8(np.clip(distance, 0, 255)), 0, 255,
                                 cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Only use a clear contrast split. Transparent/white frames cannot be
        # confidently distinguished from their shadows by this heuristic.
        if split > threshold * 1.5:
            threshold = float(split)
            separated = True
    return np.uint8(distance > threshold) * 255, {
        'method': 'adaptive-light-background' if separated else 'border-contrast',
        'threshold_lab': threshold,
        'review_required': True,
    }


def _transparent_front_mask(image):
    """Recover a faint catalogue front only when its two apertures persist.

    A pale rim can lie below the Otsu split while dark photographed arms do
    not. Require the same outer component and paired holes at several nearby
    contrast levels; a single threshold is too easily fooled by a shadow.
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
    if background[0] < 225 or noise >= 7:
        return None, 'The background is too varied for faint-frame extraction.'
    distance = np.linalg.norm(lab - background, axis=2)
    # Neutral contact shadows can persist at exactly the contrast levels that
    # reveal a tinted transparent rim. Keep tinted material and dark hardware;
    # a neutral pale rim remains ambiguous and is left for review.
    chroma = np.linalg.norm(lab[:, :, 1:] - background[1:], axis=2)
    candidates = []
    for threshold in (6, 8, 10, 12, 16, 20, 24, 30):
        if threshold < max(6, noise * 2.2):
            continue
        mask = np.uint8((distance > threshold) & ((chroma > 3) | (distance > 40))) * 255
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is None:
            continue
        parents = hierarchy[0, :, 3]
        outer = sorted(((cv2.contourArea(c), i) for i, c in enumerate(contours)
                        if parents[i] == -1), reverse=True)
        if not outer:
            continue
        area, outer_i = outer[0]
        x, y, w, h = cv2.boundingRect(contours[outer_i])
        if not (.08 <= area / (width * height) <= .75 and
                w >= width * .45 and h >= height * .12 and
                x > 1 and y > 1 and x + w < width - 1 and y + h < height - 1):
            continue
        holes = sorted(((cv2.contourArea(c), i) for i, c in enumerate(contours)
                        if parents[i] == outer_i and cv2.contourArea(c) > area * .12),
                       reverse=True)
        if len(holes) != 2:
            continue
        boxes = sorted((cv2.boundingRect(contours[i]) for _, i in holes), key=lambda b: b[0])
        left, right = boxes
        hole_areas = [a for a, _ in holes]
        if (min(hole_areas) / max(hole_areas) < .65 or
                right[0] <= left[0] + left[2] or
                min(left[2], right[2]) < w * .2 or
                min(left[3], right[3]) < h * .45 or
                abs((left[1] + left[3] / 2) - (right[1] + right[3] / 2)) > h * .15):
            continue
        candidates.append((threshold, mask, (x, y, w, h), boxes))
    # Require a run of nearby thresholds with the same observed geometry.
    stable_candidates = []
    for threshold, mask, box, holes in candidates:
        neighbors = [candidate for candidate in candidates
                     if abs(candidate[0] - threshold) <= 10 and candidate[0] != threshold]
        stable = 0
        for _, _, other_box, other_holes in neighbors:
            if (max(abs(a - b) for a, b in zip(box, other_box)) <= max(width, height) * .025 and
                    all(max(abs(a - b) for a, b in zip(hole, other)) <= max(width, height) * .025
                        for hole, other in zip(holes, other_holes))):
                stable += 1
        if stable >= 3:
            stable_candidates.append((threshold, mask))
    if stable_candidates:
        threshold, mask = min(stable_candidates, key=lambda item: abs(item[0] - 16))
        return mask, threshold
    return None, 'Faint frame and two lens openings are not stable across contrast levels; draw or correct the outlines manually.'


def _faint_side_suggestion(image, photo, scale_x, scale_y):
    """Try a tinted side arm only when its hinge, tip, and body stay stable."""
    from .side_extraction import extract_side
    height, width = image.shape[:2]
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    border_size = max(3, min(height, width) // 30)
    border = np.concatenate((lab[:border_size].reshape(-1, 3),
                             lab[-border_size:].reshape(-1, 3),
                             lab[:, :border_size].reshape(-1, 3),
                             lab[:, -border_size:].reshape(-1, 3)))
    background = np.median(border, axis=0)
    noise = float(np.percentile(np.linalg.norm(border - background, axis=1), 90))
    if background[0] < 225 or noise >= 7:
        return None
    distance = np.linalg.norm(lab - background, axis=2)
    chroma = np.linalg.norm(lab[:, :, 1:] - background[1:], axis=2)
    candidates = []
    for threshold in (12, 16, 20, 24, 30):
        if threshold < max(6, noise * 2.2):
            continue
        mask = np.uint8((distance > threshold) & ((chroma > 3) | (distance > 40))) * 255
        result = extract_side(photo, mask, scale_x, scale_y)
        if len(result['annotations']) != 1:
            continue
        ann = result['annotations'][0]
        poly = Polygon([(p['x'], p['y']) for p in ann['points']])
        if not poly.is_valid or poly.area < width * height * scale_x * scale_y * .002:
            continue
        candidates.append((threshold, result, poly))
    if len(candidates) < 3:
        return None
    for threshold, result, poly in candidates:
        if threshold != 24:
            continue
        stable = 0
        for other_threshold, other, other_poly in candidates:
            if other_threshold == threshold or abs(other_threshold - threshold) > 8:
                continue
            anchor_error = max(
                abs(result['annotations'][0]['anchors'][key][axis] -
                    other['annotations'][0]['anchors'][key][axis])
                for key in ('hinge', 'tip') for axis in ('x', 'y'))
            overlap = poly.intersection(other_poly).area / poly.union(other_poly).area
            if anchor_error <= max(width * scale_x, height * scale_y) * .025 and overlap >= .7:
                stable += 1
        if stable >= 2:
            result['annotations'][0]['source'] = 'single-visible-arm-stable-faint-v1'
            result['warnings'] = [
                'A faint tinted side contour was suggested from consistent boundaries. Two arms overlap near the shaft; the selected tip branch may join a shared or obscured section. Check that it follows one physical arm from hinge to tip before approval.',
                'Side extrusion approximates cross-section; it is not recovered manufacturing geometry.']
            result['segmentation'] = {'method': 'stable-faint-side', 'threshold_lab': threshold,
                                      'review_required': True}
            return result
    return None


def _frame_texture(photo, poly, preserve_clear=False):
    """Extend actual frame colors over UV boundary bleed, not white background."""
    from scipy.ndimage import distance_transform_edt, binary_fill_holes
    with Image.open(photo['path']) as source:
        image = source.convert('RGBA')
        image.thumbnail((2048, 2048))
    pixels = np.asarray(image).copy()
    mask = np.zeros(pixels.shape[:2], dtype=np.uint8)
    scale = np.array([image.width/photo['width'],image.height/photo['height']])
    for i, ring in enumerate([poly.exterior,*poly.interiors]):
        points = np.round(np.asarray(ring.coords)*scale).astype(np.int32)
        cv2.fillPoly(mask,[points],255 if i==0 else 0)
    foreground, _ = _foreground_mask(cv2.cvtColor(pixels[:, :, :3], cv2.COLOR_RGB2BGR))
    lab = cv2.cvtColor(pixels[:, :, :3], cv2.COLOR_RGB2LAB).astype(np.float32)
    border_size = max(3, min(image.height, image.width) // 30)
    border = np.concatenate((lab[:border_size].reshape(-1, 3),
                             lab[-border_size:].reshape(-1, 3),
                             lab[:, :border_size].reshape(-1, 3),
                             lab[:, -border_size:].reshape(-1, 3)))
    background = np.median(border, axis=0)
    noise = float(np.percentile(np.linalg.norm(border - background, axis=1), 90))
    faint_chromatic = np.zeros(mask.shape, dtype=bool)
    if background[0] > 225 and noise < 7:
        distance = np.linalg.norm(lab - background, axis=2)
        chroma = np.linalg.norm(lab[:, :, 1:] - background[1:], axis=2)
        faint_chromatic = (distance > max(6, noise * 2.2)) & (chroma > 3)
    # The approved polygon limits samples to frame material. Faint colored
    # acetate is real texture, while neutral contact shadows stay excluded.
    safe = (mask > 0) & ((foreground > 0) | faint_chromatic)
    if preserve_clear:
        # In clear acetate, white/neutral pixels INSIDE the reviewed material
        # outline are transmitted background/highlights, not missing texture.
        # Rejecting them spreads black hinge/edge samples across the bridge.
        safe = mask > 0
    # A small light mark enclosed by dark material is photographic appearance,
    # not a background hole. Preserve it within the approved solid only; large
    # apertures and edge-connected shadows remain excluded from sampling.
    enclosed = np.uint8(binary_fill_holes(safe) & ~safe)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(enclosed, 8)
    solid_interior = cv2.erode(mask, np.ones((3,3),np.uint8)) > 0
    outside_counts = np.bincount(labels[~solid_interior], minlength=count)
    eligible = ((stats[:, cv2.CC_STAT_AREA] <= max(64, np.count_nonzero(mask) * .03)) &
                (outside_counts == 0))
    eligible[0] = False
    safe |= eligible[labels]
    interior = cv2.erode(np.uint8(safe), np.ones((3,3),np.uint8)) > 0
    thickness = cv2.distanceTransform(np.uint8(mask>0),cv2.DIST_L2,5)
    if np.any(mask) and np.percentile(thickness[mask>0],90)<4:
        # Thin metallic rims have real dark/light edge detail across most of
        # their width. Eroding it away spreads pale center pixels to both edges.
        interior = safe
    # Thin real material must not vanish just because it cannot survive erosion.
    # The mask only chooses color samples; it never changes geometry or alpha.
    if not interior.any():
        interior = safe
    if interior.any():
        _, nearest = distance_transform_edt(~interior,return_indices=True)
        pixels[~safe] = pixels[nearest[0][~safe],nearest[1][~safe]]
        # Nearest-neighbour extension of a narrow diagonal arm creates Voronoi
        # color bands at its edges. Filter only the synthetic padding, never the
        # observed interior (logos, seams and material transitions stay intact).
        sigma = max(.7, max(image.size) / 360.)
        padded = cv2.GaussianBlur(pixels, (0, 0), sigmaX=sigma, sigmaY=sigma)
        pixels[~safe, :3] = padded[~safe, :3]
    return Image.fromarray(pixels)


def _triangulated_plate(poly: Polygon, thickness_mm: float, xy_to_model, material, name: str,
                        front_photo: dict | None = None, texture: bool = False,
                        extrusion_axis: int = 2, uv_map=None):
    if poly.is_empty or poly.area < 1:
        raise NeedsCorrection(f"{name} has no usable area")
    vertices: list[list[float]] = []
    uv: list[list[float]] = []
    faces: list[list[int]] = []
    index: dict[tuple[float, float, int], int] = {}

    def add(p, back: int):
        key = (round(p[0], 7), round(p[1], 7), back)
        if key not in index:
            xyz = np.asarray(xy_to_model(p), dtype=float)
            xyz[extrusion_axis] -= (thickness_mm / 1000) * back
            index[key] = len(vertices)
            vertices.append(xyz.tolist())
            if front_photo:
                uv.append(uv_map(p) if uv_map else
                          [p[0] / front_photo["width"], 1 - p[1] / front_photo["height"]])
        return index[key]

    rings = [poly.exterior, *poly.interiors]
    polygon_points = [tuple(p) for ring in rings for p in list(ring.coords)[:-1]]
    ring_ends = np.cumsum([len(list(ring.coords)) - 1 for ring in rings]).astype(np.uint32)
    point_array = np.asarray(polygon_points, dtype=np.float64)
    if len(rings) == 1:
        # GEOS retains exact clipped boundary vertices (including narrow end
        # wedges), which Earcut may omit when the cut is tangent to a curve.
        covered = [t for t in triangulate(poly) if poly.covers(t)]
        surface_triangles = [list(t.exterior.coords)[:3] for t in covered]
        if poly.area - sum(t.area for t in covered) > max(1e-6, poly.area * 1e-7):
            # A concave boundary may leave tiny Delaunay triangles just outside
            # GEOS's covers tolerance. Earcut triangulates the exact ring.
            indices = mapbox_earcut.triangulate_float64(point_array, ring_ends).reshape(-1, 3)
            if len(np.unique(indices)) == len(point_array):
                surface_triangles = [[polygon_points[i] for i in triangle] for triangle in indices]
            else:
                surface_triangles = [[polygon_points[i] for i in triangle]
                                     for triangle in _split_t_junctions(indices, point_array)]
    else:
        indices = mapbox_earcut.triangulate_float64(point_array, ring_ends)
        surface_triangles = [[polygon_points[i] for i in triangle]
                             for triangle in _split_t_junctions(indices.reshape(-1, 3), point_array)]
    if not surface_triangles:
        raise NeedsCorrection(f"{name} could not be triangulated")
    triangulated_area = sum(abs((b[0] - a[0]) * (c[1] - a[1]) -
                                (b[1] - a[1]) * (c[0] - a[0])) / 2
                            for a, b, c in surface_triangles)
    if abs(triangulated_area - poly.area) > max(1e-6, poly.area * 1e-7):
        raise NeedsCorrection(f"{name} could not be completely triangulated")
    for coords in surface_triangles:
        a, b, c = (add(p, 0) for p in coords)
        faces.append([a, b, c])
        a, b, c = (add(p, 1) for p in coords)
        faces.append([c, b, a])
    cap_faces = len(faces)
    for ring in rings:
        coords = list(ring.coords)
        for p, q in zip(coords[:-1], coords[1:]):
            a, b, c, d = add(p, 0), add(q, 0), add(q, 1), add(p, 1)
            faces.extend(([a, b, c], [a, c, d]))
    mesh = trimesh.Trimesh(vertices=np.asarray(vertices), faces=np.asarray(faces), process=False)
    mesh.metadata['wall_faces'] = list(range(cap_faces, len(faces)))
    mesh.metadata['back_faces'] = list(range(1, cap_faces, 2))
    if not mesh.is_watertight:
        edges, counts = np.unique(mesh.edges_sorted, axis=0, return_counts=True)
        open_edges = edges[counts != 2]
        reverse_index = {number: key for key, number in index.items()}
        raise NeedsCorrection(f"{name} produced an open mesh; correct its outline",
                              {"open_edges": len(open_edges), "sample_edges": [[reverse_index[int(v)] for v in e] for e in open_edges[:12]]})
    if texture and front_photo and Path(front_photo["path"]).exists():
        with Image.open(front_photo["path"]) as source:
            if uv_map:
                def photo_ring(ring):
                    return [(uv_map(p)[0] * front_photo['width'],
                             (1 - uv_map(p)[1]) * front_photo['height']) for p in ring.coords]
                photo_poly = Polygon(photo_ring(poly.exterior), [photo_ring(r) for r in poly.interiors])
            else:
                photo_poly = poly
            img = _frame_texture(front_photo, photo_poly, preserve_clear=material.name == 'clear-acetate')
            img.thumbnail((2048, 2048))
            # UVs use the complete photograph; only frame triangles receive it.
            textured = _material("#ffffff", float(material.baseColorFactor[3]) / 255,
                                 material.metallicFactor, material.roughnessFactor, img)
            textured.name = material.name
            mesh.visual = trimesh.visual.TextureVisuals(uv=np.asarray(uv), material=textured)
    else:
        mesh.visual = trimesh.visual.TextureVisuals(material=material)
    return name, mesh


def _inset_ring(ring, distance: float | np.ndarray, hole: bool) -> list[tuple[float, float]]:
    points = np.asarray(list(ring.coords)[:-1], dtype=float)
    signed_area = float(np.sum(points[:, 0] * np.roll(points[:, 1], -1) -
                               points[:, 1] * np.roll(points[:, 0], -1)))
    direction = (1 if signed_area > 0 else -1) * (-1 if hole else 1)
    distances = np.broadcast_to(np.asarray(distance, dtype=float), len(points))
    result = []
    for i, current in enumerate(points):
        before = current - points[(i - 1) % len(points)]
        after = points[(i + 1) % len(points)] - current
        before /= max(np.linalg.norm(before), 1e-12)
        after /= max(np.linalg.norm(after), 1e-12)
        n_before = direction * np.array([-before[1], before[0]])
        n_after = direction * np.array([-after[1], after[0]])
        bisector = n_before + n_after
        bisector /= max(np.linalg.norm(bisector), 1e-12)
        local_distance = float(distances[i])
        miter = min(3 * local_distance, local_distance / max(.2, float(bisector @ n_after)))
        result.append(tuple(current + bisector * miter))
    return result


def _beveled_plate(poly: Polygon, thickness_mm: float, bevel_mm: float, scale_mm_per_px: float,
                   xy_to_model, material, name: str, photo: dict,
                   extrusion_axis: int = 2, uv_map=None, bevel_back: bool = False, texture: bool = True):
    """Chamfer plate edges locally while retaining the photographed silhouette.

    A single inset cannot survive a narrow bridge without reducing the bevel
    everywhere. Each ring vertex instead uses its clearance to the other rings;
    the two approaching bevels consume at most 70 percent of a narrow strip.
    """
    original_rings = [poly.exterior, *poly.interiors]
    original_coords = [list(r.coords)[:-1] for r in original_rings]
    target_px = bevel_mm / scale_mm_per_px
    local_insets = []
    for i, points in enumerate(original_coords):
        others = [ring for j, ring in enumerate(original_rings) if i != j]
        clearances = np.array([
            min(geometry.distance(ring) for ring in others for geometry in
                (Point(p), LineString([points[(j - 1) % len(points)], p]),
                 LineString([p, points[(j + 1) % len(points)]])))
            if others else float('inf') for j, p in enumerate(points)])
        # Keeping the fraction below one half also leaves a face on strips whose
        # outer and inner rings are both chamfered.
        local_insets.append(np.minimum(target_px, .35 * clearances))
    top_poly = None
    inset_factor = 1.0
    # Pixel-stepped approved outlines may self-intersect after offsetting at
    # isolated one-pixel notches. Relax only vertices near the crossing so a
    # single notch does not erase the bevel around the entire front.
    for attempt in range(512):
        inset_coords = [_inset_ring(r, distances * inset_factor, i > 0)
                        for i, (r, distances) in enumerate(zip(original_rings, local_insets))]
        candidate = Polygon(inset_coords[0], inset_coords[1:])
        if candidate.is_valid and candidate.area > 0 and poly.covers(candidate):
            top_poly = candidate
            break
        issue = None
        if not candidate.is_valid:
            match = re.search(r'\[([^ ]+) ([^\]]+)\]', explain_validity(candidate))
            if match:
                issue = np.array([float(match.group(1)), float(match.group(2))])
        elif candidate.area > 0:
            outside = candidate.difference(poly)
            if not outside.is_empty:
                point = outside.representative_point()
                issue = np.array([point.x, point.y])
        if issue is not None and attempt < 480:
            # Select the inset edges crossing the problem point, not nearby
            # original vertices: an edge can be long and have distant ends.
            nearby = set()
            for ring_index, points in enumerate(inset_coords):
                starts = np.asarray(points)
                vectors = np.roll(starts, -1, axis=0) - starts
                lengths_sq = np.einsum('ij,ij->i', vectors, vectors)
                t = np.clip(np.einsum('ij,ij->i', issue - starts, vectors) /
                            np.maximum(lengths_sq, 1e-12), 0, 1)
                closest = starts + t[:, None] * vectors
                distances = np.linalg.norm(closest - issue, axis=1)
                for vertex_index in np.flatnonzero(distances < max(.05, target_px * .02)):
                    nearby.add((ring_index, int(vertex_index)))
                    nearby.add((ring_index, int((vertex_index + 1) % len(points))))
            if nearby:
                for ring_index, vertex_index in nearby:
                    local_insets[ring_index][vertex_index] *= .5
                continue
        inset_factor *= .5
    if top_poly is None:
        raise NeedsCorrection(f"{name} contour is too narrow or irregular for a conservative bevel")
    local_depths = [distances * inset_factor * scale_mm_per_px for distances in local_insets]
    bevel_mm = float(max(np.max(depths) for depths in local_depths))
    vertices, faces, uv = [], [], []
    wall_faces = []
    indices = {}

    def add(p, layer, shoulder_depth=0.0):
        # Shapely rings yield Python floats while the inset builder yields
        # numpy floats. At a decimal tie round(np.float64) and round(float)
        # can choose different last digits for the same binary coordinate.
        key = (round(float(p[0]), 7), round(float(p[1]), 7), layer)
        if key not in indices:
            point = np.asarray(xy_to_model(p), dtype=float)
            depth = (0 if layer == 0 else shoulder_depth if layer == 1 else
                     thickness_mm - shoulder_depth if layer == 2 and bevel_back else thickness_mm)
            point[extrusion_axis] -= depth / 1000
            indices[key] = len(vertices)
            vertices.append(point)
            uv.append(uv_map(p) if uv_map else
                      [p[0] / photo["width"], 1 - p[1] / photo["height"]])
        return indices[key]

    def cap(shape, layer, reverse):
        rings = [shape.exterior, *shape.interiors]
        points = [tuple(p) for ring in rings for p in list(ring.coords)[:-1]]
        ends = np.cumsum([len(list(ring.coords)) - 1 for ring in rings]).astype(np.uint32)
        coordinates = np.asarray(points, dtype=np.float64)
        triangles = mapbox_earcut.triangulate_float64(coordinates, ends).reshape(-1, 3)
        for tri in _split_t_junctions(triangles, coordinates):
            face = [add(points[i], layer) for i in tri]
            faces.append(face[::-1] if reverse else face)

    cap(top_poly, 0, False)
    back_start = len(faces)
    cap(top_poly if bevel_back else poly, 3 if bevel_back else 2, True)
    back_faces = list(range(back_start, len(faces)))
    for inset, original, depths in zip(inset_coords, original_coords, local_depths):
        for i in range(len(original)):
            j = (i + 1) % len(original)
            a, b = add(inset[i], 0), add(inset[j], 0)
            c, d = add(original[j], 1, depths[j]), add(original[i], 1, depths[i])
            faces.extend(([a, b, c], [a, c, d]))
            a, b = add(original[i], 1, depths[i]), add(original[j], 1, depths[j])
            c, d = add(original[j], 2, depths[j]), add(original[i], 2, depths[i])
            wall_faces.extend([len(faces), len(faces) + 1])
            faces.extend(([a, b, c], [a, c, d]))
            if bevel_back:
                a, b = add(original[i], 2, depths[i]), add(original[j], 2, depths[j])
                c, d = add(inset[j], 3), add(inset[i], 3)
                back_faces.extend([len(faces), len(faces) + 1])
                faces.extend(([a, b, c], [a, c, d]))
    mesh = trimesh.Trimesh(vertices=np.asarray(vertices), faces=np.asarray(faces), process=False)
    if not mesh.is_watertight:
        raise NeedsCorrection(f"{name} bevel produced an open mesh")
    if texture and Path(photo["path"]).exists():
        with Image.open(photo["path"]) as source:
            if uv_map:
                def photo_ring(ring):
                    return [(uv_map(p)[0] * photo['width'],
                             (1 - uv_map(p)[1]) * photo['height']) for p in ring.coords]
                photo_poly = Polygon(photo_ring(poly.exterior),
                                     [photo_ring(ring) for ring in poly.interiors])
            else:
                photo_poly = poly
            image = _frame_texture(photo, photo_poly, preserve_clear=material.name == 'clear-acetate')
            image.thumbnail((2048, 2048))
            textured = _material("#ffffff", float(material.baseColorFactor[3]) / 255,
                                 material.metallicFactor, material.roughnessFactor, image)
            textured.name = material.name
            mesh.visual = trimesh.visual.TextureVisuals(uv=np.asarray(uv), material=textured)
    else:
        mesh.visual = trimesh.visual.TextureVisuals(material=material)
    mesh.metadata["bevel_mm"] = bevel_mm
    mesh.metadata["bevel_min_mm"] = float(min(np.min(depths) for depths in local_depths))
    mesh.metadata["bevel_median_mm"] = float(np.median(np.concatenate(local_depths)))
    mesh.metadata["bevel_full_fraction"] = float(np.mean(np.concatenate(local_depths) >= bevel_mm * .95))
    mesh.metadata['wall_faces'] = wall_faces
    mesh.metadata['back_faces'] = back_faces
    return name, mesh


def _tube(points_m: np.ndarray, radius_mm: float, material, name: str):
    segments = []
    for a, b in zip(points_m[:-1], points_m[1:]):
        d = b - a
        if np.linalg.norm(d) < 1e-8:
            continue
        segment = trimesh.creation.cylinder(radius=radius_mm / 1000, segment=[a, b], sections=8)
        segments.append(segment)
    if not segments:
        raise NeedsCorrection(f"{name} has no length")
    mesh = trimesh.util.concatenate(segments)
    mesh.visual = trimesh.visual.TextureVisuals(material=material)
    return name, mesh


def _polyline(points: np.ndarray, xy_to_model, radius_mm: float, material, name: str):
    return _tube(np.asarray([xy_to_model(p) for p in points]), radius_mm, material, name)


def _wire_loop(points: np.ndarray, xy_to_model, radius_mm: float, material, name: str):
    """Continuous circular wire swept around an approved closed lens contour."""
    centers = np.asarray([xy_to_model(p) for p in points], dtype=float)
    count = len(centers)
    if count < 3:
        raise NeedsCorrection(f"{name} needs at least three contour points")
    sections = 8
    vertices = []
    for i in range(count):
        tangent = centers[(i + 1) % count] - centers[(i - 1) % count]
        tangent /= np.linalg.norm(tangent)
        normal = np.array([-tangent[1], tangent[0], 0.0])
        normal /= max(np.linalg.norm(normal), 1e-9)
        binormal = np.cross(tangent, normal)
        for j in range(sections):
            angle = j * 2 * math.pi / sections
            vertices.append(centers[i] + radius_mm / 1000 *
                            (math.cos(angle) * normal + math.sin(angle) * binormal))
    faces = []
    for i in range(count):
        nxt = (i + 1) % count
        for j in range(sections):
            k = (j + 1) % sections
            a, b, c, d = i * sections + j, i * sections + k, nxt * sections + k, nxt * sections + j
            faces.extend(([a, b, c], [a, c, d]))
    mesh = trimesh.Trimesh(vertices=np.asarray(vertices), faces=np.asarray(faces), process=False)
    mesh.visual = trimesh.visual.TextureVisuals(material=material)
    if not mesh.is_watertight:
        raise NeedsCorrection(f"{name} wire mesh is open")
    return name, mesh


def _optional_anchor(annotation: dict | None, key: str, photo: dict) -> np.ndarray | None:
    point = (annotation or {}).get("anchors", {}).get(key)
    if point is None:
        return None
    try:
        result = np.array([float(point["x"]), float(point["y"])])
    except (TypeError, KeyError, ValueError):
        raise NeedsCorrection(f"{key} anchor is invalid", {"anchor": key}) from None
    if not np.isfinite(result).all() or np.any(result < 0) or result[0] > photo["width"] or result[1] > photo["height"]:
        raise NeedsCorrection(f"{key} anchor lies outside its photo", {"anchor": key})
    return result


def _measurement(snapshot, name):
    item = snapshot.get("measurements", {}).get(name)
    if item is None:
        raise NeedsCorrection(f"Measure {name} in millimeters before reconstruction", {"measurement": name})
    value = item.get("value") if isinstance(item, dict) else item
    if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise NeedsCorrection(f"{name} must be a positive finite millimeter value", {"measurement": name})
    minimum=10 if name in {'frame_width','temple_length'} else .1 if name=='bridge_width' else 1
    if value<minimum or value>500:
        raise NeedsCorrection(f'{name} is {value} mm. Enter millimeters, not meters (0.2 m = 200 mm), or use approximate draft sizes in Dimensions.',{'measurement':name,'value_mm':value,'units':'millimeters'})
    return float(value)


def _photo_for(photos, photo_id):
    return next((p for p in photos if str(p.get("id")) == str(photo_id)), None)


def _is_validation_photo(photo: dict) -> bool:
    labels = " ".join(str(photo.get(key, "")).lower() for key in ("role", "view", "split", "purpose"))
    return bool(photo.get("validation") or photo.get("is_validation") or photo.get("withheld") or
                photo.get("use_for_reconstruction") is False or
                any(word in labels for word in ("validation", "withheld", "holdout", "test-only")))


def _anchor(annotation: dict, key: str, photo: dict) -> np.ndarray:
    point = annotation.get("anchors", {}).get(key)
    if point is None:
        raise NeedsCorrection(f"Approved top outline needs a {key} anchor", {"part": "top", "anchor": key})
    try:
        result = np.array([float(point["x"]), float(point["y"])])
    except (KeyError, TypeError, ValueError):
        raise NeedsCorrection(f"{key} anchor is invalid", {"part": "top", "anchor": key}) from None
    if not np.isfinite(result).all() or np.any(result < 0) or result[0] > photo["width"] or result[1] > photo["height"]:
        raise NeedsCorrection(f"{key} anchor lies outside the top photo", {"part": "top", "anchor": key})
    return result


def _fit_top_view(annotation: dict, photo: dict, snapshot: dict, measurements: dict) -> tuple[float, float | None, float, dict]:
    """Top image: hinge line is horizontal model X; bridge-to-hinge is rearward."""
    outline = _points(annotation, int(photo["width"]), int(photo["height"]))
    bridge = _anchor(annotation, "bridge", photo)
    left = _anchor(annotation, "hinge-left", photo)
    right = _anchor(annotation, "hinge-right", photo)
    hinge_line = right - left
    hinge_span_px = float(np.linalg.norm(hinge_line))
    if hinge_span_px < 10:
        raise NeedsCorrection("Top hinge anchors are too close to establish scale", {"part": "top"})
    unit_x = hinge_line / hinge_span_px
    unit_rear = np.array([-unit_x[1], unit_x[0]])
    hinge_center = (left + right) / 2
    if (hinge_center - bridge) @ unit_rear < 0:
        unit_rear *= -1
    bridge_to_hinge_px = float((hinge_center - bridge) @ unit_rear)
    if bridge_to_hinge_px < 1:
        raise NeedsCorrection("Top bridge and hinge anchors do not establish front curvature", {"part": "top"})
    scale = measurements["frame_width"] / hinge_span_px
    observed_sag = bridge_to_hinge_px * scale
    if observed_sag > 30:
        raise NeedsCorrection("Top-view curvature exceeds the supported 30 mm range; check anchors and scale", {"part": "top"})
    projected_depth = (outline - bridge) @ unit_rear
    hinge_depth = (hinge_center - bridge) @ unit_rear
    tip_left = annotation.get("anchors", {}).get("tip-left")
    tip_right = annotation.get("anchors", {}).get("tip-right")
    observed_spread = None
    observed_rear_depth = None
    tip_source = None
    if tip_left is not None and tip_right is not None:
        tip_a = _anchor(annotation, "tip-left", photo)
        tip_b = _anchor(annotation, "tip-right", photo)
        tip_span = abs(float((tip_b - tip_a) @ unit_x))
        observed_spread = (hinge_span_px - tip_span) * scale / 2
        observed_rear_depth = (float((tip_a + tip_b) @ unit_rear) / 2 - float(hinge_center @ unit_rear)) * scale
        tip_source = "approved tip anchors"
    elif float(np.max(projected_depth) - hinge_depth) * scale > measurements["temple_length"] * .25:
        rear = outline[projected_depth >= np.quantile(projected_depth, .85)]
        if len(rear) >= 2:
            rear_x = rear @ unit_x
            tip_span = float(np.max(rear_x) - np.min(rear_x))
            observed_spread = (hinge_span_px - tip_span) * scale / 2
            observed_rear_depth = (float(np.max(projected_depth)) - hinge_depth) * scale
            tip_source = "rear extrema of approved top outline"
    if observed_spread is not None and not -10 <= observed_spread <= 40:
        raise NeedsCorrection("Top-view arm spread conflicts with hinge width; check the top outline and tip anchors", {"part": "top"})
    if observed_rear_depth is not None and observed_rear_depth <= 0:
        raise NeedsCorrection("Top-view arm tips must lie behind the hinges", {"part": "top"})

    def solve(name, observed):
        item = snapshot.get("measurements", {}).get(name)
        prior = float(item["value"]) if isinstance(item, dict) and item.get("value") is not None else None
        source = item.get("source", "manual") if isinstance(item, dict) else None
        prior_weight = 1 if source in ("measured", "published") else .15
        if prior is None:
            return observed, 0.0, None
        fitted = float(least_squares(lambda p: [p[0] - observed, (p[0] - prior) * prior_weight],
                                     [observed], bounds=(0, 40)).x[0])
        return fitted, fitted - observed, fitted - prior

    sag, sag_residual, sag_prior_residual = solve("front_curve", observed_sag)
    spread = None
    spread_residual = None
    spread_prior_residual = None
    if observed_spread is not None:
        spread, spread_residual, spread_prior_residual = solve("temple_spread", observed_spread)
    effective_spread = spread if spread is not None else float(snapshot.get("measurements", {}).get("temple_spread", {}).get("value", 8.0))
    nominal_rear_depth = math.sqrt(max(1.0, measurements["temple_length"] ** 2 - effective_spread ** 2 - 6 ** 2))
    if observed_rear_depth is None:
        rear_depth = nominal_rear_depth
    else:
        rear_depth = float(least_squares(
            lambda value: [value[0] - observed_rear_depth,
                           math.sqrt(value[0] ** 2 + effective_spread ** 2 + 6 ** 2) - measurements["temple_length"]],
            [nominal_rear_depth], bounds=(1.0, 300.0)).x[0])
    fitted_arm_length = math.sqrt(rear_depth ** 2 + effective_spread ** 2 + 6 ** 2)
    fit = {"source_photo_id": photo["id"], "method": "top anchor fit in calibrated hinge-line coordinates",
           "observations_mm": {"front_curve": observed_sag, "temple_spread": observed_spread,
                               "arm_rear_projection": observed_rear_depth},
           "residuals_mm": {"front_curve_to_top": sag_residual, "front_curve_to_measurement": sag_prior_residual,
                            "temple_spread_to_top": spread_residual, "temple_spread_to_measurement": spread_prior_residual,
                            "arm_rear_projection_to_top": rear_depth - observed_rear_depth if observed_rear_depth is not None else None,
                            "arm_length_to_measurement": fitted_arm_length - measurements["temple_length"]},
           "temple_spread_evidence": tip_source, "hinge_span_px": hinge_span_px,
           "scale_mm_per_px": scale, "axis_convention": "bridge toward hinges and arm tips is rearward"}
    return sag, spread, rear_depth, fit


def extract_annotations(photo: dict, *, backend: str | None = None) -> dict:
    """Suggest editable pixel-space contours; no suggestion is implicitly approved."""
    if photo.get('view') in {'inner','rear'}:
        from .anatomy import audit_photo
        return {'annotations':[], 'anatomy':audit_photo(photo, []), 'warnings':['Inner/rear photos are checked during generation for registered inner-frame appearance. Only confidently matched visible regions are applied. Use a side-on Inner left or Inner right photo for an individual arm; a rear view is not flattened into front geometry.']}
    path = Path(photo.get("path", ""))
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise NeedsCorrection("Photo cannot be decoded", {"photo_id": photo.get("id")})
    from .image_resolution import prepare_extraction_image
    image, scale_x, scale_y, resolution = prepare_extraction_image(image)
    from .detection_background import prepare_detection_background
    appearance_image = image
    image, background_calibration = prepare_detection_background(image)
    result = _extract_image_annotations(photo, image, scale_x, scale_y)
    side_annotation = next((a for a in result.get('annotations', []) if a.get('part', '').startswith('temple-')), None)
    if side_annotation:
        from .opposite_arm import opposite_arm_evidence
        result['appearance_evidence'] = opposite_arm_evidence(appearance_image, side_annotation, scale_x, scale_y)
    if str(photo.get('view', 'front')).lower() in {'front', 'frontal', 'straight-front', 'front-on'}:
        from .construction_detection import analyze_construction
        from .construction_classifier import suggest_construction
        from .full_rim_recovery import recover_dark_front
        dark_front = recover_dark_front(image, photo, scale_x, scale_y)
        if dark_front is not None:
            result = dark_front
        if not {'front','lens-left','lens-right'} <= {a.get('part') for a in result.get('annotations', [])}:
            from .full_rim_recovery import recover_double_bridge_front
            full_rim = recover_double_bridge_front(image, photo, scale_x, scale_y)
            if full_rim is not None:
                result = full_rim
        specialized = None if dark_front is not None else analyze_construction(image, photo, scale_x, scale_y)
        if specialized is not None:
            result = specialized
        else:
            result['construction'] = suggest_construction(image, result.get('annotations', []), scale_x, scale_y)
        if not {'front','lens-left','lens-right'} <= {a.get('part') for a in result.get('annotations',[])}:
            from .rimless_detection import recover_rimless_front
            rimless = recover_rimless_front(image,photo,scale_x,scale_y)
            if rimless is not None:
                result = rimless
        from .zoom_refinement import refine_front_zoom
        if not {'front','lens-left','lens-right'} <= {a.get('part') for a in result.get('annotations',[])} and not {'bridge-hardware','lens-left','lens-right'} <= {a.get('part') for a in result.get('annotations',[])} and not any(a.get('part')=='shield' for a in result.get('annotations',[])):
            from .product_local import recover_product_local_front
            local = recover_product_local_front(image, photo, scale_x, scale_y)
            if local is not None:
                result = local
                result.setdefault('construction', suggest_construction(image, result['annotations'], scale_x, scale_y))
                # The local detector's geometry owns its own smoothing path.
                dark_front = None
        if not {'front','lens-left','lens-right'} <= {a.get('part') for a in result.get('annotations',[])} and not {'bridge-hardware','lens-left','lens-right'} <= {a.get('part') for a in result.get('annotations',[])}:
            from .gradient_rimless import recover_gradient_rimless
            gradient_rimless = recover_gradient_rimless(appearance_image, photo, scale_x, scale_y)
            if gradient_rimless is not None:
                result = gradient_rimless
                dark_front = None
        result['zoom_refinement'] = refine_front_zoom(result.get('annotations', []), scale_x, scale_y)
        if dark_front is not None:
            from .smooth_curve import smooth_closed_curve
            curve_scale = max(image.shape[:2])/1080*min(scale_x,scale_y)
            outer = smooth_closed_curve(Polygon([(p['x'],p['y']) for p in result['annotations'][0]['points']]),curve_scale)
            result['annotations'][0]['points']=[{'x':float(x),'y':float(y)} for x,y in list(outer.exterior.coords)[:-1]]
            # Remove numerically collinear subdivisions left by raster
            # refinement. Earcut can bridge across these on a thin top bar.
            for ann in result['annotations']:
                poly = Polygon([(p['x'], p['y']) for p in ann['points']])
                if ann['part'].startswith('nose-pad-'):
                    continue
                if ann['part'] != 'front':
                    poly = smooth_closed_curve(poly,curve_scale)
                if ann['part'] != 'front' and outer.boundary.distance(poly) < .5*min(scale_x,scale_y):
                    # A tangent hole is valid in GEOS but yields a nonmanifold
                    # wall. Reserve a subpixel rim only at those contact points.
                    clipped = poly.intersection(outer.buffer(-.5*min(scale_x,scale_y)))
                    if clipped.geom_type == 'Polygon' and clipped.area >= poly.area*.995:
                        poly = clipped
                poly = poly.simplify(1e-6, preserve_topology=True)
                ann['points'] = [{'x':float(x),'y':float(y)} for x,y in list(poly.exterior.coords)[:-1]]
    import os
    selected_backend = backend or os.environ.get('EYEWEAR_DETECTION_BACKEND', 'heuristics')
    if selected_backend == 'cpu-metal' and str(photo.get('view','')).lower() == 'front':
        from .cpu_metal import detect_cpu_metal
        metal = detect_cpu_metal(photo)
        if metal is not None:
            result = metal
        result['cpu_metal_detection'] = {'requested': True, 'applied': metal is not None,
            'method': 'cpu-thin-metal-native-ridges', 'learned_model_active': False}
    if selected_backend in {'hybrid','sam2','sam3','eyewear-trained'}:
        from .anatomy import FRONT_VIEWS, SIDE_VIEWS
        if selected_backend == 'eyewear-trained' and str(photo.get('view', 'front')).lower() in FRONT_VIEWS | SIDE_VIEWS:
            from .eyewear_trained import apply_candidate
            result = apply_candidate(photo, appearance_image, image, result, scale_x, scale_y)
        elif str(photo.get('view', 'front')).lower() in FRONT_VIEWS:
            from .hybrid_detection import apply_hybrid
            result = apply_hybrid(photo, appearance_image, image, result, scale_x, scale_y,
                                  model=selected_backend if selected_backend in {'sam2','sam3'} else 'legacy')
        elif selected_backend in {'sam2','sam3'} and str(photo.get('view','')).lower() in SIDE_VIEWS:
            from .foundation_detection import apply_side
            result=apply_side(photo,appearance_image,result,scale_x,scale_y,model=selected_backend)
            side_annotation=next((a for a in result.get('annotations',[]) if a.get('part','').startswith('temple-')),None)
            if side_annotation:
                from .opposite_arm import opposite_arm_evidence
                result['appearance_evidence']=opposite_arm_evidence(appearance_image,side_annotation,scale_x,scale_y)
        else:
            result['hybrid_detection'] = dict(requested_backend=selected_backend,inference_ran=False,
                proposals_used=False,experimental=True,status='not-applicable',
                reason='Hybrid lens proposals apply to straight front views; this view uses the existing contour detector.')
    if str(photo.get('view','')).lower()=='front' and not any(a.get('part','').startswith('lens-') for a in result.get('annotations',[])):
        kept=[]
        for ann in result.get('annotations',[]):
            poly=Polygon([(p['x'],p['y']) for p in ann['points']])
            photo_width=image.shape[1]*scale_x
            if ann.get('part')=='front' and poly.bounds[2]-poly.bounds[0]<photo_width*.6 and abs(poly.centroid.x/photo_width-.5)>.08:
                result.setdefault('warnings',[]).append('A partial dark region was rejected: it does not establish a complete front frame. No full-frame outline was substituted.')
            else:
                kept.append(ann)
        result['annotations']=kept
    result['resolution'] = resolution
    result['background_calibration'] = background_calibration
    from .anatomy import SIDE_VIEWS
    if str(photo.get('view', 'front')).lower() in SIDE_VIEWS:
        from .outline_finish import finish_side_outlines
        result['outline_finish'] = finish_side_outlines(image, result.get('annotations', []), scale_x, scale_y)
    if str(photo.get('view', 'front')).lower() in {'front', 'frontal', 'straight-front', 'front-on'}:
        from .outline_finish import finish_front_outlines
        result['outline_finish'] = ({'method':'CPU wire group already source-refined','smoothed_parts':[]}
            if result.get('cpu_metal_detection',{}).get('applied') else
            finish_front_outlines(image, result.get('annotations', []), scale_x, scale_y))
        from .nose_support import recover_visible_nose_support
        pads, diagnostics = recover_visible_nose_support(image, photo, result.get('annotations', []), scale_x, scale_y)
        result['nose_support'] = diagnostics
        if pads:
            result['annotations'].extend(pads)
            result.setdefault('warnings', []).append('Additional visible nasal components were traced from stable source pixels. Check that they are nose pads rather than rim hardware; hidden portions and depth remain unmeasured.')
    from .source_zoom import zoom_diagnostics
    result['source_zoom'] = zoom_diagnostics(photo, result.get('annotations', []))
    result['warnings'] = list(dict.fromkeys(resolution.get('warnings', []) + result.get('warnings', [])))
    from .anatomy import attach_evidence
    return attach_evidence(photo, result, image.shape[1]*scale_x, image.shape[0]*scale_y)


def _extract_image_annotations(photo, image, scale_x, scale_y):
    height, width = image.shape[:2]
    if min(width, height) < 80:
        raise NeedsCorrection("Photo is too small for contour extraction", {"photo_id": photo.get("id")})
    mask, segmentation = _foreground_mask(image)
    if str(photo.get('view','')).lower() in {'left','right','left-side','right-side','side-left','side-right','inner-left','inner-right'}:
        from .side_extraction import extract_side, recover_pale_side, recover_opaque_side
        opaque = recover_opaque_side(image,photo,scale_x,scale_y)
        from .branch_side import recover_branch_side
        branch = recover_branch_side(image, photo, scale_x, scale_y)
        if opaque is not None:
            # A seam detector can also lock onto a transparent shaft's printed
            # hardware at another resolution. Do not let it select a shorter
            # far-arm tip when a stable full outer branch is available.
            if branch and branch.get('annotations'):
                old_tip = opaque['annotations'][0]['anchors']['tip']['x']
                new_tip = branch['annotations'][0]['anchors']['tip']['x']
                if abs(old_tip-new_tip) > image.shape[1]*scale_x*.05:
                    return branch
            return opaque
        if branch is not None:
            return branch
        side = extract_side(photo,mask,scale_x,scale_y)
        if side['annotations']:
            return side
        recovered = recover_pale_side(image,photo,scale_x,scale_y)
        if recovered is not None:
            return recovered
        recovered = _faint_side_suggestion(image,photo,scale_x,scale_y)
        if recovered is not None:
            return recovered
        side['warnings'].append('The detector could not confirm a continuous faint arm across contrast levels. Try another clear side photo or review the boundary manually.')
        return side
    role = (str(photo.get('role', '')) + ' ' + str(photo.get('view', ''))).lower()
    fallback_reason = None
    if 'top' not in role and 'side' not in role and 'temple' not in role:
        initial_contours, initial_hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        has_pair = False
        if initial_hierarchy is not None:
            parents = initial_hierarchy[0, :, 3]
            for i, contour in enumerate(initial_contours):
                if parents[i] == -1 and cv2.contourArea(contour) > width * height * .02:
                    if sum(parents[j] == i and cv2.contourArea(c) > width * height * .003
                           for j, c in enumerate(initial_contours)) >= 2:
                        has_pair = True
                        break
        if not has_pair:
            recovered, fallback_info = _transparent_front_mask(image)
            if recovered is not None:
                mask = recovered
                segmentation = {'method': 'stable-faint-front', 'threshold_lab': fallback_info,
                                'review_required': True}
            else:
                fallback_reason = fallback_info
                from .clear_front import recover_clear_front_mask
                neutral, neutral_info = recover_clear_front_mask(image)
                if neutral is not None:
                    mask = neutral
                    segmentation = {'method':'stable-neutral-front', 'threshold_lab':neutral_info,
                                    'review_required':True}
                    fallback_reason = None
    # The old 5px close fused detached contact shadows into the lower rim.
    # Keep the actual contrast boundary, including genuine thin bridge bars.
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    warnings = ["Background/shadow separation is contrast-based, not depth measurement. Check faint, transparent, and reflective edges before approving."]
    annotations = []
    faint_front = segmentation['method'] in {'stable-faint-front', 'stable-neutral-front'}
    if faint_front:
        warnings.append('Faint frame recovered from consistent aperture and edge boundaries; inspect the outer rim and both completed lens edges before approval.')
    if segmentation['method'] == 'stable-neutral-front':
        warnings.append('Colorless-frame outlines are provisional: nose pads, hinges and reflections can merge with the front rim. Correct these boundaries in Advanced before approval if they form spikes or notches; closed outlines alone do not verify the physical frame.')
    if hierarchy is None:
        return {"annotations": [], "warnings": [
            "The detector could not confidently isolate a frame contour. Transparent edges, reflections, or background variation may be involved; try another clear front photo or review the outline manually.",
            *([fallback_reason] if fallback_reason else [])]}
    h = hierarchy[0]
    outer_candidates = [(cv2.contourArea(c), i) for i, c in enumerate(contours)
                        if h[i][3] == -1 and cv2.contourArea(c) > width * height * .02]
    if not outer_candidates:
        return {"annotations": [], "warnings": [
            "The detector could not confidently isolate a frame contour. Transparent edges, reflections, or background variation may be involved; try another clear front photo or review the outline manually.",
            *([fallback_reason] if fallback_reason else [])]}
    _, outer_i = max(outer_candidates)

    def annotation(i, part):
        contour = contours[i]
        completed_invalid_lens = False
        epsilon = max(.6, cv2.arcLength(contour, True) * .0004)
        approx = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
        # Pixel contours can revisit a narrow junction. Simplify suggestions only;
        # never repair an approved merchant polygon behind their back.
        for trial in (.75, 1., 1.5, 2., 2.5, 3., 4., epsilon * 2, epsilon * 3, epsilon * 5):
            if len(approx) >= 3 and Polygon(approx).is_valid:
                break
            approx = cv2.approxPolyDP(contour, trial, True).reshape(-1, 2)
        if part.startswith('lens-') and (len(approx) < 3 or not Polygon(approx).is_valid):
            # A rear arm can touch a rim at a one-pixel neck. OpenCV's hole
            # then doubles back along that neck and is not a simple polygon.
            # Complete from the ORIGINAL contour, not a damaged approximation;
            # never pass invalid geometry through as an approvable lens.
            approx = cv2.convexHull(contour).reshape(-1, 2)
            completed_invalid_lens = True
            warnings.append(f'{part}: completed a boundary across touching background hardware; inspect this inferred lens edge.')
        if len(approx) < 3 or not Polygon(approx).is_valid:
            warnings.append(f"The {part} suggestion has an ambiguous boundary; manually correct it before approval.")
        return {"photo_id": str(photo.get("id", "")), "part": part,
                "points": [{"x": float(x * scale_x), "y": float(y * scale_y)} for x, y in approx],
                "anchors": {}, "approved": False,
                "source": ("convex-occlusion-completion:" if completed_invalid_lens else "") +
                          ("automatic-stable-faint-v1" if faint_front else "automatic-shadow-aware-v4")}

    if fallback_reason:
        warnings.append(fallback_reason)
    if "top" in role:
        annotations.append(annotation(outer_i, "top"))
        warnings.append("Place bridge and left/right hinge anchors on the top contour before approval.")
    elif "side" in role or "temple" in role or str(photo.get("view", "")).lower() in ("left", "right"):
        annotations.append(annotation(outer_i, "temple-left" if "left" in role else "temple-right"))
        warnings.append("Verify the side contour and assign the correct left/right temple before approval.")
    else:
        annotations.append(annotation(outer_i, "front"))
        holes = [(cv2.contourArea(c), i) for i, c in enumerate(contours)
                 if h[i][3] == outer_i and cv2.contourArea(c) > width * height * .003]
        if len(holes) > 2:
            warnings.append("More than two lens-window candidates found; arms or reflections may cross the openings. Review outlines manually.")
        all_holes = sorted(holes, reverse=True)
        holes = all_holes[:2]
        holes.sort(key=lambda pair: cv2.boundingRect(contours[pair[1]])[0])
        for side, (_, i) in zip(("left", "right"), holes):
            annotations.append(annotation(i, f"lens-{side}"))
            candidate = annotations[-1]
            polygon = Polygon([(p['x'],p['y']) for p in candidate['points']])
            if polygon.is_valid and polygon.area/polygon.convex_hull.area < .98:
                # A suggestion, never an invisible repair of approved geometry.
                # Convex completion is appropriate only after merchant review.
                completed = polygon.convex_hull
                if faint_front:
                    outer = Polygon([(p['x'], p['y']) for p in annotations[0]['points']])
                    if outer.is_valid and not outer.contains(completed):
                        # At thumbnail resolution a one-pixel transparent rim
                        # may vanish where a completed aperture meets its outer
                        # edge. Trim only a sub-percent overshoot; a larger
                        # overlap is ambiguous and must still be withheld.
                        inset = max(scale_x, scale_y) * .5
                        clipped = completed.intersection(outer.buffer(-inset))
                        if (clipped.geom_type == 'Polygon' and clipped.is_valid and
                                clipped.area >= completed.area * .99):
                            completed = clipped.simplify(inset * .2, preserve_topology=True)
                            warnings.append(f'{side.capitalize()} lens: completed edge was inset within a very thin outer rim; verify this inferred clearance.')
                if faint_front and (not outer.is_valid or not outer.contains(completed)):
                    annotations.pop()
                    warnings.append(f'{side.capitalize()} lens: completion crosses the thin outer rim; this uncertain outline was withheld.')
                    continue
                candidate['points'] = [{'x':float(x),'y':float(y)} for x,y in list(completed.exterior.coords)[:-1]]
                candidate['source'] = ('convex-occlusion-completion:automatic-stable-faint-v1'
                                       if faint_front else 'convex-occlusion-completion')
                warnings.append(f'{side.capitalize()} lens: suggested a convex boundary across an obstruction. Check the restored edge; concave lens designs require manual correction.')
            hull_area = cv2.contourArea(cv2.convexHull(contours[i]))
            if hull_area and cv2.contourArea(contours[i]) / hull_area < .82:
                warnings.append(f"{side.capitalize()} lens-window candidate is strongly concave; an arm or reflection may intrude.")
            if any(h[j][3] == i and cv2.contourArea(c) > width * height * .0005
                   for j, c in enumerate(contours)):
                warnings.append(f"Foreground object inside {side} lens-window candidate; inspect for an arm or reflection.")
        if len(holes) < 2:
            warnings.append("Two lens openings were not confidently found; draw or correct them manually.")
        elif len(all_holes)>2:
            left_box=cv2.boundingRect(contours[holes[0][1]])
            right_box=cv2.boundingRect(contours[holes[1][1]])
            center_min=left_box[0]+left_box[2]*.7
            center_max=right_box[0]+right_box[2]*.3
            for _,i in all_holes[2:]:
                x,y,w,hole_height=cv2.boundingRect(contours[i])
                if center_min < x+w/2 < center_max:
                    annotations.append(annotation(i,'bridge-opening'))
                    warnings.append('Detected an extra bridge opening. Review its boundary before approval.')
                    break
        warnings.append("Automatic contours are suggestions; approve each outline after checking its pixels.")
    # A displayed set of three labels is not proof that the frame can be
    # triangulated. Withhold invalid/crossing suggestions so the quick workflow
    # cannot invite approval of geometry known to fail at generation.
    safe_annotations = []
    outer_polygon = None
    accepted_holes = []
    for ann in annotations:
        polygon = Polygon([(p['x'],p['y']) for p in ann['points']])
        valid = polygon.is_valid and polygon.area >= 8
        if valid and ann['part'] == 'front':
            outer_polygon = polygon
        elif valid and (ann['part'].startswith('lens-') or ann['part'] == 'bridge-opening'):
            valid = (outer_polygon is not None and outer_polygon.contains(polygon)
                     and all(not polygon.intersects(other) for other in accepted_holes))
            if valid:
                accepted_holes.append(polygon)
        if valid:
            safe_annotations.append(ann)
        else:
            warnings.append(f"{ann['part']}: boundary is not safe to reconstruct; this suggestion was withheld instead of presenting an invalid outline for approval.")
    annotations = safe_annotations
    refinements = {}
    if annotations and annotations[0]['part'] == 'front':
        from .contour_refinement import refine_closed_contour
        from .front_cleanup import clean_lower_rim
        def outline(a):
            return np.array([[p['x'], p['y']] for p in a['points']])
        if faint_front:
            cleaned, applied = clean_lower_rim(Polygon(outline(annotations[0])),
                [Polygon(outline(a)) for a in annotations if a['part'].startswith('lens-')])
            if applied:
                annotations[0]['points'] = [{'x':float(x),'y':float(y)} for x,y in list(cleaned.exterior.coords)[:-1]]
                refinements['lower_rim_cleanup'] = {'applied': True}
        for ann in annotations:
            refined, diagnostic = refine_closed_contour(outline(ann), sample_spacing_px=7)
            old = ann['points']
            ann['points'] = [{'x': float(x), 'y': float(y)} for x, y in refined]
            candidate = Polygon(outline(annotations[0]), [outline(a) for a in annotations[1:]])
            if not candidate.is_valid:
                ann['points'] = old
                diagnostic = {'applied': False, 'reason': 'preserve_frame_and_hole_topology'}
            refinements[ann['part']] = diagnostic
    return {"annotations": annotations, "warnings": warnings, "segmentation": segmentation,
            "contour_refinement": refinements}


def reconstruct(snapshot: dict, output_dir: Path) -> dict:
    started = time.perf_counter()
    category = snapshot.get("category")
    if category not in {"opaque-full-rim", "thin-metal", "transparent-full-rim", "semi-rimless", "rimless", "sports-shield"}:
        raise NeedsCorrection("Choose one of the six supported frame categories")
    measurements = {k: _measurement(snapshot, k) for k in
                    ("frame_width", "lens_width", "lens_height", "bridge_width", "temple_length")}
    if 2 * measurements["lens_width"] + measurements["bridge_width"] > measurements["frame_width"] * 1.02:
        raise NeedsCorrection("Lens widths plus bridge exceed frame width", {"measurements": measurements})
    all_photos = snapshot.get("photos", [])
    photos = [p for p in all_photos if not _is_validation_photo(p)]
    # One active capture per reconstructable view. A new upload must never silently
    # reuse old approved contours, including projects created before this rule.
    def view_group(p):
        return {'inner-left':'left','inner-right':'right'}.get(p.get('view'),p.get('view'))
    latest = {view_group(p): p.get('id') for p in photos if view_group(p) in {'front','left','right','top'}}
    photos = [p for p in photos if view_group(p) not in latest or p.get('id') == latest[view_group(p)]]
    eligible_ids = {str(p.get("id")) for p in photos}
    anns = [a for a in snapshot.get("annotations", []) if a.get("approved") and
            str(a.get("photo_id")) in eligible_ids]
    front_ann = next((a for a in anns if a.get("part") in ("front", "shield") and
                      _photo_for(photos, a.get("photo_id"))), None)
    left_ann = next((a for a in anns if a.get("part") == "lens-left"), None)
    right_ann = next((a for a in anns if a.get("part") == "lens-right"), None)
    if category in ("rimless",) and (left_ann is None or right_ann is None):
        raise NeedsCorrection("Approve both lens outlines for rimless reconstruction",
                              {"missing": [p for p, a in (("lens-left", left_ann), ("lens-right", right_ann)) if a is None]})
    if front_ann is None and category != "rimless":
        raise NeedsCorrection("Approve a front photo outline before reconstruction", {"missing": ["front"]})
    if front_ann and front_ann.get('source') in {'automatic', 'merchant-reviewed:automatic', 'automatic-shadow-aware-v3', 'merchant-reviewed:automatic-shadow-aware-v3'}:
        raise NeedsCorrection('Analyze the current front photo again and approve the shadow-aware outlines. Older automatic outlines can contain photographed shadows; the old asset is preserved.')
    if category in ("opaque-full-rim", "thin-metal", "transparent-full-rim", "semi-rimless") and (left_ann is None or right_ann is None):
        raise NeedsCorrection("Approve both lens opening outlines before reconstruction", {"missing": ["lens-left", "lens-right"]})
    primary = front_ann or left_ann
    front_photo = _photo_for(photos, primary["photo_id"])
    if front_photo is None:
        raise NeedsCorrection("Approved front outline is not linked to an available photo")
    if str(front_photo.get("view", "")).lower() not in ("front", "frontal", "straight-front", "front-on"):
        raise NeedsCorrection("Front and lens outlines require a straight front photo; correct the camera view or capture one",
                              {"photo_id": front_photo.get("id"), "view": front_photo.get("view")})
    width, height = int(front_photo["width"]), int(front_photo["height"])
    if width <= 0 or height <= 0:
        raise NeedsCorrection("Front photo dimensions are invalid")
    front = _points(front_ann, width, height) if front_ann else None
    lenses = {}
    for side, ann in (("left", left_ann), ("right", right_ann)):
        if ann:
            if str(ann.get("photo_id")) != str(primary["photo_id"]):
                raise NeedsCorrection("Front and lens outlines must use the same front photograph")
            lenses[side] = _points(ann, width, height)
            polygon = Polygon(lenses[side])
            if ('extraction' in ann.get('source','') or ann.get('source')=='automatic') and polygon.area/polygon.convex_hull.area < .98:
                raise NeedsCorrection('Old extracted lens outlines may contain photographed arms. Analyze the current front photo again and review the occlusion-corrected boundaries.')
    if front is None:
        all_lens = np.vstack(list(lenses.values()))
        bounds = np.array([all_lens[:, 0].min(), all_lens[:, 0].max()])
    else:
        bounds = np.array([front[:, 0].min(), front[:, 0].max()])
    px_width = bounds[1] - bounds[0]
    if px_width <= 1:
        raise NeedsCorrection("Front outline has no usable width")
    sx = measurements["frame_width"] / px_width
    lens_heights = [np.ptp(p[:, 1]) for p in lenses.values()]
    if lens_heights:
        result = least_squares(lambda a: np.array(lens_heights) * a[0] - measurements["lens_height"],
                               [measurements["lens_height"] / np.mean(lens_heights)], bounds=(1e-6, np.inf))
        sy = float(result.x[0])
    else:
        sy = sx
    front_bridge_anchor = _optional_anchor(front_ann, "bridge", front_photo)
    if front_bridge_anchor is not None:
        center_x = float(front_bridge_anchor[0])
    elif len(lenses) == 2:
        inner_left = float(np.max(lenses["left"][:, 0]))
        inner_right = float(np.min(lenses["right"][:, 0]))
        center_x = (inner_left + inner_right) / 2
    else:
        center_x = float(np.mean(bounds))
    if front_bridge_anchor is not None:
        center_y = float(front_bridge_anchor[1])
    elif len(lenses) == 2:
        center_y = float(np.mean([(np.min(p[:, 1]) + np.max(p[:, 1])) / 2 for p in lenses.values()]))
    elif front is not None:
        center_y = float((front[:, 1].min() + front[:, 1].max()) / 2)
    else:
        center_y = float(np.mean(np.vstack(list(lenses.values()))[:, 1]))
    sag = float(snapshot.get("measurements", {}).get("front_curve", {}).get("value", 4.0))
    sag = max(0.0, min(sag, 30.0))
    spread = float(snapshot.get("measurements", {}).get("temple_spread", {}).get("value", 8.0))
    rear_depth = math.sqrt(max(1.0, measurements["temple_length"] ** 2 - spread ** 2 - 6 ** 2))
    fit = {"source_photo_id": None, "method": "measurement or explicit default",
           "observations_mm": {}, "residuals_mm": {}}
    top_ann = next((a for a in anns if a.get("part") == "top"), None)
    if top_ann:
        top_photo = _photo_for(photos, top_ann.get("photo_id"))
        if "top" not in str(top_photo.get("view", "")).lower() and "overhead" not in str(top_photo.get("view", "")).lower():
            raise NeedsCorrection("Top outline requires a top-view photo", {"photo_id": top_photo.get("id")})
        sag, fitted_spread, rear_depth, fit = _fit_top_view(top_ann, top_photo, snapshot, measurements)
        if fitted_spread is not None:
            spread = fitted_spread

    joint_fit = {"schema_version": 1, "status": "disabled", "applied": False, "views": [],
                 "reason": "Enable experimental multi-view front fitting to test shared curvature."}
    if snapshot.get("reconstruction_settings", {}).get("joint_front_fit"):
        curve_item = snapshot.get("measurements", {}).get("front_curve", {})
        if top_ann or (isinstance(curve_item, dict) and curve_item.get("source") in {"measured", "published", "manual"}):
            joint_fit.update(status="withheld", reason="Approved top geometry or explicit front-curve measurement is retained; experiment will not overwrite it.")
        else:
            from .joint_fit import fit_project_front
            joint_fit = fit_project_front(photos, anns, front_photo, (center_x, center_y),
                                          (sx, sy), measurements["frame_width"], sag)
            if joint_fit.get("applied"):
                sag = float(joint_fit["fitted_front_curve_mm"])
                fit["method"] = "experimental shared front curvature and oblique camera fit"
        fit["joint_front"] = joint_fit

    def xyz(p):
        x = (float(p[0]) - center_x) * sx
        y = (center_y - float(p[1])) * sy
        return np.array([x, y, _curve_z(x, measurements["frame_width"] / 2, sag)]) / 1000

    model_xmin = (float(bounds[0]) - center_x) * sx / 1000
    model_xmax = (float(bounds[1]) - center_x) * sx / 1000

    def inside_front_width(point: np.ndarray, radius_mm: float) -> np.ndarray:
        adjusted = np.asarray(point, dtype=float).copy()
        margin = radius_mm / 1000
        adjusted[0] = np.clip(adjusted[0], model_xmin + margin, model_xmax - margin)
        return adjusted

    default_material = snapshot.get("material") or {}
    color = default_material.get("color", "#252a32")
    opacity = float(default_material.get("opacity", 1.0))
    metalness = float(default_material.get("metalness", .0))
    roughness = float(default_material.get("roughness", .35))
    lens_color = default_material.get("lens_color", "#b5d9e5")
    lens_opacity = float(np.clip(default_material.get("lens_opacity", .16),0,1))
    inferred_lens = None
    if lens_color.lower() == '#b5d9e5' and abs(lens_opacity-.16)<1e-6:
        from .lens_appearance import infer_dark_lens_material
        inferred_lens = infer_dark_lens_material(front_photo,lenses,include_clear=True)
        if inferred_lens:
            lens_color = inferred_lens['lens_color']
            lens_opacity = inferred_lens['lens_opacity']
    if category == "thin-metal":
        metalness = max(.65, metalness)
    frame_mat = _material(color, opacity, metalness, roughness)
    if category == 'transparent-full-rim':
        frame_mat.name = 'clear-acetate'
    lens_mat = _material(lens_color, lens_opacity, .0, .08)
    if inferred_lens:
        # _material already converts every authored/sampled colour once.
        lens_mat.name = 'photo-estimated-lens'
    parts: list[tuple[str, trimesh.Trimesh]] = []
    warnings: list[str] = []
    if joint_fit["status"] != "disabled":
        warnings.append("Experimental multi-view front fit: " + joint_fit["reason"])
    if inferred_lens:
        warnings.append('Lens appearance was estimated from both photo interiors because lens settings were unchanged defaults. Optical transmission is approximate; custom lens settings override this estimate.')
    completed_lenses = [a['part'] for a in anns if a.get('part','').startswith('lens-') and 'occlusion-completion' in a.get('source','')]
    completed_arms = [a['part'] for a in anns if a.get('part','').startswith('temple-') and 'occlusion-completion' in a.get('source','')]
    if completed_arms:
        warnings.append('Arm profiles separate a visible tip branch from overlapping arms: '+', '.join(completed_arms)+'. Ownership and thickness of the shared shaft remain inferred after review; hidden hinge hardware is unmeasured.')
    mirrored_arms = [a['part'] for a in anns if 'mirrored' in a.get('source','')]
    if mirrored_arms:
        warnings.append('Explicit symmetry assumption: '+', '.join(mirrored_arms)+' mirrored from the reviewed opposite arm, not independently observed.')
    if completed_lenses:
        warnings.append('Lens boundaries were completed across photographed obstructions using a reviewed boundary assumption: '+', '.join(completed_lenses)+'. Hidden edges are inferred, not measured.')
    if any(snapshot.get('measurements',{}).get(k,{}).get('source') in {'default','inferred'}
           for k in ('frame_width','lens_width','lens_height','bridge_width','temple_length')):
        warnings.append('Approximate-size draft: one or more physical dimensions are assumed or inferred. Photos alone do not establish real-world scale; do not use this model to judge physical fit.')
    if top_ann and fit.get("temple_spread_evidence") is None:
        warnings.append("Top view does not identify both arm tips; temple spread uses its measurement or default.")
    if top_ann and abs(fit["residuals_mm"]["front_curve_to_top"]) > 3:
        warnings.append("Top-view curvature and supplied front_curve differ; inspect anchors and measurement.")
    if top_ann and fit["residuals_mm"]["temple_spread_to_top"] is not None and abs(fit["residuals_mm"]["temple_spread_to_top"]) > 3:
        warnings.append("Top-view arm spread and supplied temple_spread differ; inspect anchors and measurement.")
    if top_ann and fit["residuals_mm"]["arm_rear_projection_to_top"] is not None and abs(fit["residuals_mm"]["arm_rear_projection_to_top"]) > 10:
        warnings.append("Top-view arm depth and measured temple length disagree; inspect view calibration and dimensions.")
    if top_ann:
        warnings.append("Top-view depth fit assumes a near-orthographic photograph; perspective distortion is not calibrated.")
    thickness_item = snapshot.get("measurements", {}).get("frame_thickness")
    if thickness_item is None:
        thickness = 1.2 if category in {"thin-metal", "semi-rimless", "rimless"} else 4.0
        if category == "transparent-full-rim" and front is not None and lenses:
            # Shape clearance is not depth evidence. It only bounds a draft
            # prior so a thin clear rim is not extruded as a 4 mm slab.
            from shapely.geometry import Point
            boundary = Polygon(front).exterior
            clearances = [boundary.distance(Point(p))*sx for outline in lenses.values()
                          for p in outline if p[1] < np.median(outline[:,1])]
            if clearances:
                thickness = float(np.clip(np.median(clearances), .8, 4.0))
        thickness_source = "default"
        warnings.append(f"Front material depth uses an unverified {thickness:g} mm draft value; measure frame thickness for a calibrated side profile.")
    else:
        thickness = thickness_item.get("value") if isinstance(thickness_item, dict) else thickness_item
        thickness_source = thickness_item.get("source", "manual") if isinstance(thickness_item, dict) else "manual"
        if not isinstance(thickness, (int, float)) or not math.isfinite(thickness) or not .8 <= thickness <= 16:
            raise NeedsCorrection("frame_thickness must be a measured depth from 0.8 to 16 mm", {"measurement": "frame_thickness"})
        thickness = float(thickness)
        if thickness_source in {"default", "inferred"}:
            warnings.append("Front material depth is approximate; measure frame thickness for a calibrated side profile.")
    sports_dual = category == "sports-shield" and front_ann is not None and front_ann.get("part") == "front" and len(lenses) == 2
    if category == "sports-shield" and front_ann.get("part") == "front" and not sports_dual:
        raise NeedsCorrection("Sports frames need either an approved shield outline or a front outline with both lens openings",
                              {"missing": ["shield or lens-left and lens-right"]})
    if category in ("opaque-full-rim", "thin-metal", "transparent-full-rim", "semi-rimless") or sports_dual:
        outer_poly = Polygon(front)
        hole_polys = [Polygon(lenses[side]) for side in ("left", "right")]
        if not all(outer_poly.contains(hole) for hole in hole_polys) or hole_polys[0].intersects(hole_polys[1]):
            raise NeedsCorrection("Lens openings must be separate and fully inside the front outline")
        cutouts = [a for a in anns if a.get('part')=='bridge-opening' and a.get('photo_id')==primary['photo_id']]
        extra_holes = [_points(a,width,height) for a in cutouts]
        if extra_holes and category=='thin-metal':
            raise NeedsCorrection('A bridge opening needs a solid-frame construction; the thin-metal wire path does not support this extra cutout. Verify the frame category.')
        ring = Polygon(front, [list(p) for p in lenses.values()]+[list(p) for p in extra_holes])
        if not ring.is_valid:
            raise NeedsCorrection("Front outline and lens openings produce invalid frame geometry")
        if category == "thin-metal":
            gap_mm = min(outer_poly.exterior.distance(hole.exterior) * sx for hole in hole_polys)
            if thickness_source not in {"default", "inferred"} and thickness / 2 > min(1.15, gap_mm * .35):
                raise NeedsCorrection("Measured frame thickness exceeds the photographed thin-metal rim clearance; check category, contour, or measurement",
                                      {"measurement": "frame_thickness"})
            radius = max(.35, min(thickness / 2, 1.15, gap_mm * .35))
            for side, outline in lenses.items():
                parts.append(_wire_loop(outline, xyz, radius, frame_mat, f"metal_rim_{side}"))
            inner_left = lenses["left"][np.argmax(lenses["left"][:, 0])]
            inner_right = lenses["right"][np.argmin(lenses["right"][:, 0])]
            # Follow the uppermost actual strip of frame between the apertures.
            # A three-point V bridge loses the observed arch in the photograph.
            bridge_profile=[]
            for x in np.linspace(inner_left[0],inner_right[0],33):
                crossing=ring.intersection(LineString([(x,-1),(x,height+1)]))
                segments=[crossing] if crossing.geom_type=='LineString' else [g for g in getattr(crossing,'geoms',[]) if g.geom_type=='LineString']
                if segments:
                    top_strip=min(segments,key=lambda line:line.bounds[1])
                    bridge_profile.append([x,(top_strip.bounds[1]+top_strip.bounds[3])/2])
            if len(bridge_profile)<2:
                raise NeedsCorrection('The front outline does not establish a connected bridge between the lenses.')
            bridge_profile=np.asarray(bridge_profile)
            endpoint_left=lenses['left'][np.argmin(np.linalg.norm(lenses['left']-bridge_profile[0],axis=1))]
            endpoint_right=lenses['right'][np.argmin(np.linalg.norm(lenses['right']-bridge_profile[-1],axis=1))]
            from .component_texture import component_photo
            observed_bridge = next((a for a in anns if a.get('part')=='bridge-hardware' and
                                    a.get('photo_id')==primary['photo_id']), None)
            if observed_bridge:
                profile = Polygon(_points(observed_bridge,width,height))
                parts.append(_triangulated_plate(profile, min(thickness,1.2), xyz,
                    _material('#ffffff',1,metalness,roughness), 'metal_bridge',
                    component_photo(observed_bridge,front_photo,photos,output_dir), texture=True))
            else:
                parts.append(_polyline(np.vstack((endpoint_left,bridge_profile,endpoint_right)), xyz,
                                       radius, frame_mat, "metal_bridge"))
            fit['bridge']={'method':'approved photographed bridge profile; depth inferred' if observed_bridge else 'centerline of uppermost photographed frame strip','source_photo_id':front_photo['id']}
            for side, outline in lenses.items():
                outer_point = outline[np.argmin(outline[:, 0]) if side == "left" else np.argmax(outline[:, 0])]
                anchor = _optional_anchor(front_ann, f"hinge-{side}", front_photo)
                if anchor is None:
                    anchor = front[np.argmin(front[:, 0]) if side == "left" else np.argmax(front[:, 0])]
                parts.append(_tube(np.vstack((xyz(outer_point), inside_front_width(xyz(anchor), radius))),
                                   radius, frame_mat, f"metal_endpiece_{side}"))
        elif category == "semi-rimless":
            # The upper structure follows the approved polygon. An exposed
            # lower optical edge does not prove the presence of a metal wire.
            # Keep the cut just above sampled horizontal lens extrema. Cutting
            # through a tangent point creates zero-width slits in GEOS polygons.
            cutoff = float(np.mean([np.mean(p[:, 1]) for p in lenses.values()])) - 0.5
            from shapely.geometry import box
            ends = [_optional_anchor(front_ann, key, front_photo) for key in
                    ('rim-outer-end-left','rim-end-left','rim-end-right','rim-outer-end-right')]
            if all(point is not None for point in ends) and all(ends[i][0] < ends[i+1][0] for i in range(3)):
                clip = Polygon([(0,0),(width,0),(width,float(ends[-1][1])),
                                *[(float(p[0]),float(p[1])) for p in reversed(ends)],
                                (0,float(ends[0][1]))])
                cropped = ring.intersection(clip)
                fit['semi_rimless_support'] = {'method':'reviewed inner/outer rim-end anchors', 'lower_support':'unverified'}
            else:
                cropped = ring.intersection(box(0, 0, width, cutoff))
                warnings.append('Upper-rim endpoint heights are inferred; add rim-end evidence for a more accurate semi-rimless termination.')
            geoms = [cropped] if isinstance(cropped, Polygon) else list(cropped.geoms)
            for i, shape in enumerate(geoms):
                if isinstance(shape, Polygon) and shape.area > 1:
                    parts.append(_triangulated_plate(shape, thickness, xyz, frame_mat,
                                                     f"upper_frame_{i}", front_photo, texture=True))
            warnings.append('Semi-rimless lower support is unverified: no opaque lower wire was invented. Clear lens edges come from the lens geometry.')
        elif category == "opaque-full-rim":
            parts.append(_beveled_plate(ring, thickness, min(.35, thickness * .12), sx,
                                        xyz, frame_mat, "front_frame", front_photo))
        else:
            parts.append(_triangulated_plate(ring, thickness, xyz, frame_mat, "front_frame", front_photo, texture=True))
    elif category == "sports-shield":
        shield = Polygon(front)
        parts.append(_triangulated_plate(shield, 1.4, xyz, lens_mat, "shield_lens"))
        top = front[front[:, 1] <= np.quantile(front[:, 1], .25)]
        top = top[np.argsort(top[:, 0])]
        if len(top) > 1:
            brow_radius = max(.8, thickness / 3)
            parts.append(_tube(np.asarray([inside_front_width(xyz(p), brow_radius) for p in top]),
                               brow_radius, frame_mat, "shield_brow"))
        warnings.append("Shield lens is modeled as one uniform tint; optical properties are approximate.")
    if category != "sports-shield" or sports_dual:
        for side, outline in lenses.items():
            if category in {'semi-rimless','rimless'}:
                parts.append(_beveled_plate(Polygon(outline), .7, .18, sx, xyz, lens_mat,
                    f'lens_{side}', front_photo, bevel_back=True, texture=False))
            else:
                parts.append(_triangulated_plate(Polygon(outline), .7, xyz, lens_mat, f"lens_{side}"))
    from .component_texture import component_photo
    rimless_mounts = {}
    if category == "rimless":
        from .rimless_mount import visible_mounts
        rimless_mounts = visible_mounts(front_photo,lenses)
        left = lenses["left"]; right = lenses["right"]
        left_inner = left[np.argmax(left[:, 0])]
        right_inner = right[np.argmin(right[:, 0])]
        observed_bridge = next((a for a in anns if a.get('part')=='bridge-hardware' and a.get('photo_id')==primary['photo_id']),None)
        if observed_bridge:
            parts.append(_triangulated_plate(Polygon(_points(observed_bridge,width,height)),1.2,xyz,
                _material('#ffffff',1,metalness,roughness),'bridge',component_photo(observed_bridge,front_photo,photos,output_dir),texture=True))
            warnings.append('Bridge/mount uses the approved photographic silhouette with inferred 1.2 mm depth; hardware thickness and rear structure are unmeasured.')
        elif not any(a.get('part')=='front-support' and a.get('photo_id')==primary['photo_id'] for a in anns):
            parts.append(_tube(np.vstack((xyz(left_inner), xyz(right_inner))), 1.0, frame_mat, "bridge"))
        for side, outline in lenses.items():
            observed = rimless_mounts.get(side)
            if observed:
                parts.append(_triangulated_plate(observed['polygon'],observed['depth_mm'],xyz,
                    _material('#ffffff',1,metalness,roughness),f'endpiece_{side}',front_photo,texture=True))
                warnings.append(f'{side.capitalize()} front mounting profile and colour use visible source pixels; its 1.2 mm depth and hidden hinge remain inferred.')
                continue
            outer_point = outline[np.argmin(outline[:, 0]) if side == "left" else np.argmax(outline[:, 0])]
            endpoint = inside_front_width(xyz(outer_point), 1.2)
            parts.append(_tube(np.vstack((endpoint, endpoint + np.array([0, 0, -.004]))),
                               1.2, frame_mat, f"endpiece_{side}"))
    # Separately reviewed support/hardware silhouettes use their source pixels.
    # They are shallow plates with explicit inferred depth, not hinge mechanics.
    for component in anns:
        part = component.get('part', '')
        if component.get('photo_id') != primary['photo_id'] or part not in {'front-support','hinge-left','hinge-right'}:
            continue
        shape = Polygon(_points(component,width,height))
        name = 'front_support' if part=='front-support' else 'endpiece_observed_'+part.removeprefix('hinge-')
        parts.append(_triangulated_plate(shape,1.2,xyz,
            _material('#ffffff',1,metalness,roughness),name,component_photo(component,front_photo,photos,output_dir),texture=True))
        warnings.append(f'{part}: approved source silhouette and colour; 1.2 mm depth is inferred, hidden construction is unmeasured.')
    pad_anchors = {}
    lens_details = []
    if inferred_lens:
        from .lens_details import visible_lens_details
        pad_shapes = {side: Polygon(_points(a,width,height))
                      for side in ('left','right') for a in anns
                      if a.get('part')==f'nose-pad-{side}' and a.get('photo_id')==primary['photo_id']}
        lens_details = visible_lens_details(front_photo,lenses,pad_shapes)
        if lens_details:
            with Image.open(front_photo['path']) as original:
                detail_texture=original.convert('RGB')
                detail_texture.thumbnail((2048,2048))
            for index,detail in enumerate(lens_details):
                def detail_xyz(point):
                    return xyz(point)+np.array([0,0,.0002])
                name,mesh=_triangulated_plate(detail['polygon'],.02,detail_xyz,
                    _material('#ffffff',1,0,.4),f"lens_detail_{detail['side']}_{index}",front_photo,texture=True)
                mesh.visual.material.baseColorTexture=detail_texture
                mesh.visual.material.name='photographic-lens-detail'
                parts.append((name,mesh))
            warnings.append('Isolated lens printing and visible pad-mount appearance were transferred from this photo. These shallow photographic patches do not recover hidden mounting hardware or optical depth.')
        from .lens_appearance import measured_dark_edge_width, measured_clear_edge_width
        for side,outline in lenses.items():
            edge_width=measured_dark_edge_width(front_photo,outline)
            clear_edge = False
            if edge_width is None and category in {'rimless','semi-rimless'}:
                edge_width=measured_clear_edge_width(front_photo,outline)
                clear_edge = edge_width is not None
            if edge_width is not None:
                boundary=Polygon(outline)
                inset=boundary.buffer(-edge_width/2 if clear_edge else -edge_width)
                if inset.geom_type=='Polygon' and not inset.is_empty:
                    if clear_edge:
                        boundary=boundary.buffer(edge_width/2)
                    edge=Polygon(boundary.exterior.coords,[inset.exterior.coords])
                    def edge_xyz(point):
                        return xyz(point)+np.array([0,0,.00015])
                    edge_material = _material('#ffffff',1,0,.2)
                    if clear_edge:
                        # Faint neutral optical edges are real source colour,
                        # not missing frame pixels to replace with dark metal.
                        edge_material.name = 'clear-acetate'
                    parts.append(_triangulated_plate(edge,.15,edge_xyz,edge_material,
                        f'optical_edge_{side}',front_photo,texture=True))
    if category in ("thin-metal", "rimless") or any(a.get('part') in {'nose-pad-left','nose-pad-right'} for a in anns) or any(
            a and any(k.startswith('nose-pad-') for k in a.get('anchors',{}))
            for a in (front_ann,left_ann,right_ann)):
        pad_material = _material("#d5cec2", .72, .0, .7)
        for side in ("left", "right"):
            key = f"nose-pad-{side}"
            pad_outline = next((a for a in anns if a.get('part')==key and a.get('photo_id')==primary['photo_id']),None)
            if pad_outline is not None:
                shape = Polygon(_points(pad_outline,width,height))
                def pad_xyz(point):
                    return xyz(point)+np.array([0,0,-.005])
                photographed_pad = _material('#ffffff',.85,0,.15)
                photographed_pad.name='clear-acetate'
                parts.append(_triangulated_plate(shape,1.2,pad_xyz,
                    photographed_pad,f'nose_pad_{side}',component_photo(pad_outline,front_photo,photos,output_dir),texture=True))
                pad_anchors[side] = pad_outline['photo_id']
                continue
            annotation = next((a for a in (front_ann, left_ann, right_ann)
                               if a and a.get("anchors", {}).get(key)), None)
            anchor = _optional_anchor(annotation, key, front_photo)
            if anchor is None:
                warnings.append(f"{side.capitalize()} nose pad is not annotated; omitted from the model.")
                continue
            if str(annotation.get("photo_id")) != str(primary["photo_id"]):
                raise NeedsCorrection("Nose pad anchors must use the calibrated front photograph")
            center = xyz(anchor) + np.array([0, 0, -.005])
            pad = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
            pad.vertices = pad.vertices * np.array([.003, .004, .0016]) + center
            actual_pad_material = (_material('#e5e5e5', .45, .0, .15)
                                   if 'automatic-dark-double-bridge' in annotation.get('source','')
                                   else pad_material)
            pad.visual = trimesh.visual.TextureVisuals(material=actual_pad_material)
            parts.append((f"nose_pad_{side}", pad))
            pad_anchors[side] = annotation["photo_id"]
    # Hinges follow front end landmarks, and arms follow approved side profiles when present.
    front_pts = front if front is not None else np.vstack(list(lenses.values()))
    temple_length = measurements["temple_length"]
    temple_radius = max(.65, min(3.0, float(snapshot.get("measurements", {}).get("temple_thickness", {}).get("value", 2.2)) / 2))
    structural_names = ("front_frame", "upper_frame_", "metal_rim_", "metal_endpiece_",
                        "endpiece_", "lower_wire_", "shield_brow", "bridge", "front_support")
    front_structure = [(name, mesh) for name, mesh in parts
                       if name.startswith(structural_names)]

    def attachment_surface(point):
        """Find actual front material below a projected hinge point, excluding lenses."""
        surfaces = []
        for name, mesh in front_structure:
            triangles = mesh.triangles
            first = triangles[:, 0, :2]
            edge_a = triangles[:, 1, :2] - first
            edge_b = triangles[:, 2, :2] - first
            relative = point[:2] - first
            cross = lambda a, b: a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
            denominator = cross(edge_a, edge_b)
            valid = np.abs(denominator) > 1e-14
            u = np.zeros(len(triangles))
            v = np.zeros(len(triangles))
            u[valid] = cross(relative[valid], edge_b[valid]) / denominator[valid]
            v[valid] = cross(edge_a[valid], relative[valid]) / denominator[valid]
            hit = valid & (u >= -1e-5) & (v >= -1e-5) & (u + v <= 1.00001)
            if np.any(hit):
                z = triangles[:, 0, 2] + u * (triangles[:, 1, 2] - triangles[:, 0, 2]) + v * (triangles[:, 2, 2] - triangles[:, 0, 2])
                surfaces.append((name, float(np.min(z[hit]))))
        return min(surfaces, key=lambda surface: surface[1]) if surfaces else None

    for side, x_select in (("left", np.min), ("right", np.max)):
        side_ann = next((a for a in anns if a.get('part')==f'temple-{side}'),None)
        if side_ann is None:
            warnings.append(f'{side.capitalize()} arm omitted: no approved side outline. This is an incomplete reconstruction, not a complete try-on asset.')
            continue
        if side_ann.get('source') in {'merchant-reviewed-side', 'explicit-mirrored-side-assumption'} or 'single-visible-arm-suggestion' in side_ann.get('source', ''):
            raise NeedsCorrection('Analyze the side arm again and approve the smooth single-arm outline. Older automatic traces can contain the overlapping second arm and pixel spikes.')
        extreme_x = x_select(front_pts[:, 0])
        at_end = _optional_anchor(front_ann, f"hinge-{side}", front_photo)
        observed_hinge = next((a for a in anns if a.get('part')==f'hinge-{side}' and a.get('photo_id')==primary['photo_id']),None)
        if at_end is None and observed_hinge:
            at_end = _optional_anchor(observed_hinge,'hinge',front_photo)
            if at_end is None:
                point = Polygon(_points(observed_hinge,width,height)).representative_point()
                at_end = np.array([point.x,point.y])
        explicit_hinge = at_end is not None
        if at_end is None:
            if category == "rimless":
                lens_outline = lenses[side]
                at_end = rimless_mounts[side]['anchor'] if side in rimless_mounts else lens_outline[np.argmin(lens_outline[:, 0]) if side == "left" else np.argmax(lens_outline[:, 0])]
            else:
            # A single extreme vertex may be the top or bottom of an endpiece.
            # Intersect the silhouette just inside its outer edge and use the
            # midpoint of the outer shoulder, where the hinge is attached.
                offset_px = min(2.0 / sx, float(np.ptp(front_pts[:, 0])) * .04)
                shoulder_x = float(extreme_x + (offset_px if side == "left" else -offset_px))
                crossing = Polygon(front_pts).intersection(LineString([(shoulder_x, -1),
                                                                        (shoulder_x, height + 1)]))
                segments = [crossing] if crossing.geom_type == "LineString" else [
                    g for g in getattr(crossing, "geoms", []) if g.geom_type == "LineString"]
                if segments:
                    shoulder = min(segments, key=lambda segment: (segment.bounds[1] + segment.bounds[3]) / 2)
                    at_end = np.array([shoulder_x, (shoulder.bounds[1] + shoulder.bounds[3]) / 2])
                else:
                    nearby = front_pts[np.abs(front_pts[:, 0] - extreme_x) <= offset_px]
                    at_end = np.array([extreme_x, float(np.median(nearby[:, 1]))])
        # The buried leaf/barrel of a hinge is narrower than the arm root.
        # The photo texture already carries visible hardware on the endpiece;
        # a broad exposed cylinder would reproduce it as a second bulky blob.
        hinge_radius = max(.45, min(.9, temple_radius * .6))
        hinge = inside_front_width(xyz(at_end), hinge_radius)
        surface = attachment_surface(hinge)
        if surface is None and not explicit_hinge:
            candidate_y = np.linspace(float(front_pts[:, 1].min()), float(front_pts[:, 1].max()), 101)
            for y in sorted(candidate_y, key=lambda y: abs(y - at_end[1])):
                candidate = np.array([at_end[0], y])
                candidate_hinge = inside_front_width(xyz(candidate), hinge_radius)
                surface = attachment_surface(candidate_hinge)
                if surface is not None:
                    at_end, hinge = candidate, candidate_hinge
                    break
        if surface is None:
            raise NeedsCorrection(f"{side} hinge is outside the front frame material; move its front anchor onto the endpiece")
        side_ann = next((a for a in anns if a.get("part") == f"temple-{side}"), None)
        if side_ann:
            photo = _photo_for(photos, side_ann.get("photo_id"))
            if not photo:
                raise NeedsCorrection(f"Approved {side} temple outline is missing its source photo")
            view = str(photo.get("view", "")).lower()
            if view not in ("left", "right", "left-side", "right-side", "side-left", "side-right", "inner-left", "inner-right"):
                raise NeedsCorrection(f"{side} temple outline requires a side-view photo", {"photo_id": photo.get("id")})
            outline = _points(side_ann, int(photo["width"]), int(photo["height"]))
            span = np.ptp(outline[:, 0])
            if span < 2:
                raise NeedsCorrection(f"{side} temple outline has no usable length")
            side_anchors = side_ann.get("anchors", {})
            if "hinge" in side_anchors and "tip" in side_anchors:
                hinge_px = float(side_anchors["hinge"]["x"])
                tip_px = float(side_anchors["tip"]["x"])
                y_center = float(side_anchors["hinge"]["y"])
                # Contour refinement can move an endpoint by a fraction of a
                # source pixel. Seat only that tiny discrepancy on the existing
                # outline; never expand the contour or rewrite saved anchors.
                from shapely.geometry import Point
                from shapely.ops import nearest_points
                source_poly = Polygon(outline)
                source_root = Point(hinge_px, y_center)
                root_distance = source_poly.distance(source_root)
                if root_distance > .1:
                    raise NeedsCorrection(f"{side} temple hinge anchor does not meet its outline; place it at the center of the arm root",
                        {"photo_id": photo['id'], "part": f"temple-{side}", "anchor": "hinge",
                         "distance_source_px": float(root_distance), "allowed_rounding_px": .1})
                if root_distance > 1e-9:
                    seated = nearest_points(source_poly, source_root)[0]
                    hinge_px, y_center = float(seated.x), float(seated.y)
                    warnings.append(f"{side.capitalize()} temple root seated by {root_distance:.4f} source pixels after contour refinement; saved outline and anchors retained.")
                direction_px = tip_px - hinge_px
                if abs(direction_px) < 2:
                    raise NeedsCorrection(f"{side} temple hinge and tip anchors are too close")
                anchors_used = True
            else:
                tip_on_right = str(photo.get("view", "")).lower().find("right") < 0
                hinge_px = float(np.min(outline[:, 0]) if tip_on_right else np.max(outline[:, 0]))
                direction_px = float(span if tip_on_right else -span)
                y_center = float(np.mean(outline[:, 1]))
                anchors_used = False
            q = (outline[:, 0] - hinge_px) / direction_px
            if np.min(q) < -.15 or np.max(q) > 1.15:
                warnings.append(f"{side.capitalize()} temple contour extends beyond its hinge/tip anchors; verify side calibration.")
            y_scale = temple_length / abs(direction_px)
            projected_height = float(np.ptp(outline[:, 1]) * y_scale)
            if projected_height > 40:
                warnings.append(f"{side.capitalize()} side profile maps to {projected_height:.1f} mm height; verify its photo scale and outline.")
            # The side photograph can be pitched relative to the front view.
            # Remove only the straight shaft's linear image slope; the curved
            # ear hook beyond it remains in the silhouette.
            def shaft_center(lo, hi):
                values = outline[(q >= lo) & (q <= hi), 1]
                if len(values) == 0:
                    values = outline[np.argsort(np.abs(q - (lo + hi) / 2))[:max(2, len(outline) // 30)], 1]
                return float(np.median(values))
            shaft_slope_px = (shaft_center(.55, .68) - shaft_center(.04, .16)) / .515
            poly = Polygon(np.column_stack((q * temple_length,
                                            (y_center - outline[:, 1] + shaft_slope_px * q) * y_scale)))
            if not poly.is_valid:
                raise NeedsCorrection(f"{side} temple profile is invalid")
            from shapely.geometry import Point
            if not poly.buffer(1e-8).covers(Point(0, 0)):
                raise NeedsCorrection(f"{side} temple hinge anchor does not meet its outline; place it at the center of the arm root",
                    {"photo_id": photo['id'], "part": f"temple-{side}", "anchor": "hinge",
                     "distance_profile_mm": float(poly.distance(Point(0, 0)))})
            # Seat the arm against the actual rear of the local endpiece. A
            # wire, shield brow, or rimless lug need not have the draft front
            # depth, even when that default is used elsewhere in the frame.
            root_z = surface[1] + .0005
            attachment_q = 0.0
            attachment_y_mm = 0.0
            frame_attachment = side_anchors.get('frame-attachment')
            if frame_attachment is not None:
                attachment_q = (float(frame_attachment['x'])-hinge_px)/direction_px
                if not -.15 <= attachment_q < 0 or not Polygon(outline).buffer(1).covers(
                        Point(float(frame_attachment['x']),float(frame_attachment['y']))):
                    raise NeedsCorrection(f'{side} endpiece attachment must lie on the visible continuation beyond the arm seam')
                attachment_y_mm = (y_center-float(frame_attachment['y'])+shaft_slope_px*attachment_q)*y_scale
                root_z += attachment_q*rear_depth/1000
                warnings.append(f'{side.capitalize()} visible endpiece is included from the side outline. Its depth assumes a near-side photograph; hidden hinge construction remains unverified.')
            connector_front_z = surface[1] + .0004
            connector_rear_z = min(surface[1] - .001, connector_front_z - .001)
            connector_start = hinge.copy()
            connector_end = hinge.copy()
            connector_start[2] = connector_front_z
            connector_end[2] = connector_rear_z
            connector_material = frame_mat
            if Path(front_photo["path"]).exists():
                with Image.open(front_photo["path"]) as image:
                    pixel_x = int(np.clip(round(at_end[0]), 0, image.width - 1))
                    pixel_y = int(np.clip(round(at_end[1]), 0, image.height - 1))
                    sampled = image.convert("RGB").getpixel((pixel_x, pixel_y))
                connector_material = _material("#%02x%02x%02x" % sampled,
                                               opacity, metalness, roughness)
            parts.append(_tube(np.vstack((connector_start, connector_end)),
                               hinge_radius, connector_material, f"hinge_{side}"))
            inward = 1 if side == "left" else -1
            side_base = hinge.copy()
            side_base[0] -= inward*spread*attachment_q/1000
            side_base[1] -= attachment_y_mm/1000
            def side_map(p, base=side_base):
                return np.array([base[0] + temple_radius / 1000 + inward * spread * (p[0] / temple_length) / 1000,
                                 base[1] + p[1] / 1000,
                                 root_z - (p[0] / temple_length) * rear_depth / 1000])
            def side_uv(p):
                proportion = p[0] / temple_length
                image_x = hinge_px + proportion * direction_px
                image_y = y_center + shaft_slope_px * proportion - p[1] / y_scale
                return [image_x / photo["width"], 1 - image_y / photo["height"]]
            # Keep the approved side contour as the shoulder of the extrusion.
            # A shallow local bevel removes the rectangular slab edge without
            # rounding away the shaft, sleeve, hook, or hinge silhouette.
            temple_part = _beveled_plate(poly, temple_radius * 2,
                                         min(.22, temple_radius * .25), 1.0,
                                         side_map, frame_mat, f"temple_{side}", photo,
                                         extrusion_axis=0, uv_map=side_uv,
                                         bevel_back=True)
            parts.append(temple_part)
            side_hardware = next((a for a in anns if a.get('part')==f'hinge-{side}' and a.get('photo_id')==photo['id']),None)
            if side_hardware:
                pixels=_points(side_hardware,photo['width'],photo['height'])
                t=(pixels[:,0]-hinge_px)/direction_px
                profile=Polygon(np.column_stack((t*temple_length,(y_center-pixels[:,1]+shaft_slope_px*t)*y_scale)))
                parts.append(_triangulated_plate(profile,1.2,side_map,
                    _material('#ffffff',1,metalness,roughness),f'hinge_profile_{side}',photo,
                    texture=True,extrusion_axis=0,uv_map=side_uv))
                warnings.append(f'{side} hinge profile uses its reviewed side silhouette; 1.2 mm cross-section and internal mechanism remain inferred.')
            fit.setdefault("side", {})[side] = {"source_photo_id": photo["id"],
                                                 "calibration": "approved hinge/tip anchors" if anchors_used else "outline extrema inferred as hinge/tip",
                                                 "scale_mm_per_px": y_scale,
                                                 "shaft_pitch_removed_mm": shaft_slope_px * y_scale,
                                                 "edge_bevel_mm": temple_part[1].metadata["bevel_mm"],
                                                 "projected_profile_height_mm": projected_height,
                                                 "projected_outline_length_mm": float(np.ptp(q) * temple_length)}
            fit.setdefault("joints", {})[side] = {
                "attachment": "approved front hinge anchor" if explicit_hinge else "inferred outer front shoulder",
                "front_photo_id": front_photo["id"], "attachment_px": [float(at_end[0]), float(at_end[1])],
                "surface_part": surface[0], "arm_root_contains_joint": attachment_q == 0,
                "visible_endpiece_contains_front_joint": attachment_q < 0,
                "rear_surface_z_mm": surface[1] * 1000,
                "arm_root_z_mm": root_z * 1000,
                "arm_root_overlap_mm": (root_z - surface[1]) * 1000 if attachment_q == 0 else None,
                "front_attachment_overlap_mm": .5,
                "visible_endpiece_depth_mm": -attachment_q*rear_depth,
                "endpiece_depth_measured": False,
                "connector_depth_mm": [float((hinge[2] - connector_front_z) * 1000),
                                       float((hinge[2] - connector_rear_z) * 1000)],
                "connector_appearance": "front-photo color sampled at attachment; hidden behind front surface"}
            if abs(shaft_slope_px * y_scale) > 2:
                warnings.append(f"{side.capitalize()} side-view shaft pitch was leveled; its 3D slope is unverified without a calibrated side view.")
    if not parts or any(not np.isfinite(mesh.vertices).all() for _, mesh in parts):
        raise NeedsCorrection("Generated geometry is empty or non-finite")
    # Pixel Y inversion and different extrusion axes change face winding. A
    # watertight mesh can still have inward caps or inconsistent bevel normals.
    # Repair face orientation only: vertex positions and UV indices stay intact.
    for name, mesh in parts:
        mesh.fix_normals(multibody=True)
        if not mesh.is_winding_consistent or mesh.volume <= 0:
            raise NeedsCorrection(f"{name} has inconsistent surface orientation; review its contour")
    model_width = (bounds[1] - bounds[0]) * sx
    front_geometry = [mesh for name, mesh in parts if not name.startswith("temple_")]
    actual_frame_width = (max(float(mesh.vertices[:, 0].max()) for mesh in front_geometry) -
                          min(float(mesh.vertices[:, 0].min()) for mesh in front_geometry)) * 1000
    # Wire/endpiece construction ends at hinge centers, while published frame width
    # is an outer-edge measurement. Calibrate the assembled horizontal span once,
    # consistently including both arms, instead of treating these as equal spans.
    width_fit_scale = measurements['frame_width'] / actual_frame_width
    if not .90 <= width_fit_scale <= 1.10:
        raise NeedsCorrection('Constructed frame width is inconsistent with the photo anchors; check hinge positions and overall width.',
                              {'measured_mm':measurements['frame_width'],'constructed_mm':actual_frame_width})
    if abs(width_fit_scale-1)>.0001:
        for _, mesh in parts:
            mesh.vertices[:,0] *= width_fit_scale
        fit['assembled_width']={'method':'horizontal physical-scale calibration of all parts','scale':width_fit_scale,'before_mm':actual_frame_width,'target_mm':measurements['frame_width']}
        if abs(width_fit_scale-1)>.01:
            warnings.append(f'Assembly width was calibrated by {(width_fit_scale-1)*100:.1f}% to match the published/measured outer width; inspect hinge and lens spacing.')
    actual_frame_width = (max(float(mesh.vertices[:,0].max()) for mesh in front_geometry)-min(float(mesh.vertices[:,0].min()) for mesh in front_geometry))*1000
    width_error = actual_frame_width - measurements["frame_width"]
    checks = {"finite_vertices": True,
              "consistent_outward_normals": all(mesh.is_winding_consistent and mesh.volume > 0 for _, mesh in parts),
              "frame_width_within_1_percent": bool(abs(width_error) <= measurements["frame_width"] * .01),
              "front_parts_watertight": all(mesh.is_watertight for mesh in front_geometry),
              "approved_front_geometry": front_ann is not None or category == "rimless",
              "geometry_count": len(parts)}
    if not checks["frame_width_within_1_percent"]:
        raise NeedsCorrection("Exported front assembly width differs from measured frame width by more than 1%",
                              {"measured_mm": measurements["frame_width"], "exported_mm": actual_frame_width})
    scene = trimesh.Scene()
    from .render_mesh import prepare_render_mesh
    from .multiview import analyze_multiview
    from .inner_texture import build_registered_inner_texture
    front_evidence = [a for a in anns if str(a.get('photo_id')) == str(front_photo['id'])]
    multiview = analyze_multiview(all_photos, front_photo, front_evidence)
    # Preserve visible rear-arm appearance separately from the approved near
    # arm. It must not be unioned into geometry or silently copied to both UVs.
    from .opposite_arm import opposite_arm_evidence
    from .image_resolution import prepare_extraction_image
    opposite_evidence = []
    for side_ann in anns:
        if not side_ann.get('part', '').startswith('temple-') or 'mirrored' in side_ann.get('source', ''):
            continue
        side_photo = _photo_for(photos, side_ann.get('photo_id'))
        if not side_photo or side_photo.get('view') not in {'left', 'right', 'left-side', 'right-side', 'side-left', 'side-right'}:
            continue
        pixels = cv2.imread(str(side_photo['path']))
        if pixels is None or not {'hinge', 'tip'} <= set(side_ann.get('anchors', {})):
            continue
        pixels, evidence_sx, evidence_sy, _ = prepare_extraction_image(pixels)
        opposite_evidence.extend(opposite_arm_evidence(pixels, side_ann, evidence_sx, evidence_sy))
    multiview['opposite_arm_appearance'] = opposite_evidence
    if any(name == 'front_frame' for name, _ in parts):
        inner_texture, inner_diagnostics = build_registered_inner_texture(front_photo, front_evidence, all_photos, multiview)
    else:
        inner_texture, inner_diagnostics = None, {'reason': 'No supported continuous textured front-frame surface', 'applied': [], 'observed_fraction': 0}
    multiview['inner_surface'] = inner_diagnostics
    multiview['limitations'] = [
        'Registration supports local front-frame appearance, not measured depth or hidden hinge construction.',
        'Inner-frame registration is not applied to non-coplanar temple arms; unseen inner printing remains unknown.',
        'Upsampling improves working resolution but cannot restore missing photographic detail.',
    ]
    reported = {str(row['photo_id']) for row in multiview.get('views', [])}
    for photo in all_photos:
        photo_id = str(photo.get('id'))
        if photo_id in reported:
            continue
        evidence = [a for a in anns if str(a.get('photo_id')) == photo_id]
        used = bool(evidence) and photo_id in eligible_ids
        multiview.setdefault('views', []).append({
            'photo_id': photo_id, 'view': photo.get('view'),
            'status': 'used' if used else 'skipped',
            'usage': 'Approved contours and photographic material' if used else 'Not applied',
            'reason': 'approved_evidence' if used else 'No active approved geometry; held-out, superseded, or unsupported view',
        })
    from .temple_texture import transfer_inner_tip
    for name, mesh in parts:
        cap_texture = inner_texture if name == 'front_frame' else None
        texture_faces = None
        if name in {'temple_left', 'temple_right'}:
            part_name = name.replace('_', '-')
            target_annotation = next((a for a in anns if a.get('part') == part_name), None)
            target_photo = _photo_for(photos, target_annotation.get('photo_id')) if target_annotation else None
            candidate = next((e for e in opposite_evidence if e['part'] == part_name), None)
            if candidate and target_photo and not target_photo.get('view', '').startswith('inner-'):
                source_photo = _photo_for(photos, candidate['photo_id'])
                base = getattr(mesh.visual.material, 'baseColorTexture', None)
                if source_photo and base is not None:
                    cap_texture, registration = transfer_inner_tip(source_photo, candidate, target_photo, target_annotation, base)
                    candidate['registration'] = registration
                    if cap_texture is not None:
                        candidate.update(applied=True, status='used', reason='Visible distal inner-arm pixels applied with approximate tip/width registration; hidden shaft is not recovered.')
                        # Inner face points toward the head: +X on the left,
                        # -X on the right. Never replace the outer face or wall.
                        sign = 1 if name == 'temple_left' else -1
                        texture_faces = np.flatnonzero(mesh.face_normals[:, 0] * sign > .5).tolist()
                        warnings.append(f'{part_name}: visible inner tip transferred from opposite-arm evidence using approximate local registration; inspect logo placement. Hidden inner shaft remains inferred.')
        scene.add_geometry(prepare_render_mesh(mesh, back_texture=cap_texture, texture_faces=texture_faces), node_name=name, geom_name=name)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model.glb"
    scene.export(str(model_path), file_type="glb", include_normals=True)
    if lenses:
        measured_lens_width = np.mean([np.ptp(p[:, 0]) * sx * width_fit_scale for p in lenses.values()])
        if abs(measured_lens_width - measurements["lens_width"]) / measurements["lens_width"] > .1:
            warnings.append(f"Photo lens width maps to {measured_lens_width:.1f} mm, which differs from the supplied {measurements['lens_width']:.1f} mm; verify contour or measurements.")
    provenance = {"front": {"photo_id": primary["photo_id"], "source": primary.get("source", "manual"), "approved": True,
                            "bridge_anchor": front_bridge_anchor is not None,
                            "hinge_anchors": {side: _optional_anchor(front_ann, f"hinge-{side}", front_photo) is not None for side in ("left", "right")}},
                  "top": {"photo_id": top_ann["photo_id"], "source": top_ann.get("source", "manual"), "approved": True} if top_ann else None,
                  "measurements": {k: (snapshot["measurements"][k].get("source", "manual") if isinstance(snapshot["measurements"][k], dict) else "manual") for k in measurements},
                  "front_depth": {"value_mm": thickness, "source": thickness_source},
                  "depth": "experimental oblique multi-view front curvature fit" if joint_fit.get("applied") else "fitted from approved top view" if top_ann else "inferred from front_curve or default 4 mm",
                  "arms": {side: "mirrored — explicit symmetry assumption" if f'temple-{side}' in mirrored_arms else "approved overlap completion — inferred shared shaft" if f'temple-{side}' in completed_arms else "approved side outline" if any(a.get("part") == f"temple-{side}" for a in anns) else "omitted — unverified" for side in ("left", "right")},
                  "nose_pads": {side: ("approved visible silhouette; 1.2 mm inferred depth" if any(a.get('part')==f'nose-pad-{side}' for a in anns)
                                       else "approved front anchor" if side in pad_anchors else "not modeled") for side in ("left", "right")},
                  "lens_material": inferred_lens or {'lens_color':lens_color,'lens_opacity':lens_opacity,'method':'project-settings'},
                  "rimless_mounts": {side:{k:v for k,v in observed.items() if k not in {'polygon','anchor'}}
                                     for side,observed in rimless_mounts.items()},
                  "lens_details": [{k:v for k,v in detail.items() if k!='polygon'} for detail in lens_details],
                  "appearance": "front photo projected onto front solid; approved side photos projected onto corresponding arms; lens tint and surfaces approximated"}
    from .anatomy import audit_project
    anatomy = audit_project(snapshot, anns)
    checks['anatomy_geometry_consistent'] = not anatomy['blocking']
    if anatomy['blocking']:
        warnings.append('Additional anatomy checks found conflicting evidence. Review the anatomy report; this export is not verified for production use.')
    metadata = {"engine_version": ENGINE_VERSION, "anatomy": anatomy, "category": category, "dimensions_mm": {**measurements, "constructed_frame_width": actual_frame_width,
                       "front_landmark_span": model_width, "frame_width_error": width_error,
                       "front_curve": sag, "temple_spread": spread, "arm_rear_depth": rear_depth,
                       "frame_thickness": thickness,
                       "front_bevel": next((mesh.metadata.get("bevel_mm") for name, mesh in parts if name == "front_frame"), None),
                       "front_bevel_median": next((mesh.metadata.get("bevel_median_mm") for name, mesh in parts if name == "front_frame"), None)},
                "checks": checks, "warnings": warnings, "provenance": provenance, "multiview": multiview,
                "reconstruction_status": "incomplete" if any(not any(a.get('part')==f'temple-{side}' for a in anns) for side in ('left','right')) else "draft — review required",
                "bridge_openings": len([a for a in anns if a.get('part')=='bridge-opening']),
                "occlusion_completions": completed_lenses,
                "fit": fit,
                "parts": [name for name, _ in parts], "elapsed_seconds": round(time.perf_counter() - started, 3),
                "coordinate_system": {"units": "meters", "origin": "bridge center", "x": "wearer left", "y": "up", "z": "forward; arms extend negative z"},
                "model_file": "model.glb"}
    if snapshot.get("reconstruction_settings", {}).get("joint_front_fit"):
        from .export_review import validate_export
        # Source images are transient inputs to overlay generation, not metadata paths.
        for view in joint_fit.get("views", []):
            view["_photo"] = _photo_for(photos, view.get("photo_id"))
        joint_fit["_side_photos"] = [p for p in photos if p.get("view") in {"left", "right", "inner-left", "inner-right"}]
        metadata["export_review"] = validate_export(model_path, metadata["parts"], anns,
            front_photo, (center_x, center_y), (sx, sy), width_fit_scale, joint_fit, output_dir)
        joint_fit.pop("_side_photos", None)
        for view in joint_fit.get("views", []):
            view.pop("_photo", None)
        metadata["checks"]["serialized_components_preserved"] = metadata["export_review"]["component_preservation_passed"]
        metadata["checks"]["serialized_geometry_finite"] = metadata["export_review"]["finite_nonempty_geometry"]
        if not metadata["export_review"]["component_preservation_passed"] or not metadata["export_review"]["finite_nonempty_geometry"]:
            raise NeedsCorrection("An approved component did not survive GLB serialization", metadata["export_review"])
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    return metadata
