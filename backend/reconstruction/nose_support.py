"""Visible nasal components on bright studio fronts, independent of frame family.

This follows observed pixels outside established optical boundaries. It does not
invent whole pads hidden behind lenses, assign hardware depth or infer absence.
"""
import cv2
import numpy as np
from shapely.geometry import Polygon, box
from shapely.ops import unary_union


def recover_visible_nose_support(image, photo, annotations, scale_x=1., scale_y=1.):
    parts = {a['part']: a for a in annotations}
    if not {'lens-left', 'lens-right'} <= parts.keys():
        return [], {'method': 'stable-visible-nasal-components', 'reason': 'missing-optical-boundaries'}
    h, w = image.shape[:2]
    border = np.concatenate((image[:5].reshape(-1, 3), image[-5:].reshape(-1, 3),
                             image[:, :5].reshape(-1, 3), image[:, -5:].reshape(-1, 3)))
    background = np.median(border, axis=0)
    if background.min() < 240 or np.percentile(np.abs(border.astype(float)-background), 95) > 6:
        return [], {'method': 'stable-visible-nasal-components', 'reason': 'unsupported-background'}
    lenses = [Polygon([(p['x']/scale_x, p['y']/scale_y) for p in parts[f'lens-{side}']['points']])
              for side in ('left', 'right')]
    if not all(p.is_valid and p.area > 0 for p in lenses) or lenses[0].intersects(lenses[1]):
        return [], {'method': 'stable-visible-nasal-components', 'reason': 'invalid-optical-boundaries'}
    left, right = lenses
    lw = min(p.bounds[2]-p.bounds[0] for p in lenses)
    lh = min(p.bounds[3]-p.bounds[1] for p in lenses)
    center = (left.bounds[2]+right.bounds[0])/2
    if lw < 40 or lh < 30:
        return [], {'method': 'stable-visible-nasal-components', 'reason': 'insufficient-source-detail'}
    region = box(left.bounds[2]-lw*.1, min(p.bounds[1] for p in lenses)+lh*.35,
                 right.bounds[0]+lw*.1, min(p.bounds[3] for p in lenses)-lh*.08)
    # Scale by the product, not blank image margins. Separate the rim-adjacent
    # material from the candidate; no convex-hull filling of background.
    clearance = max(1., lw*.008)
    excluded = unary_union([p.buffer(clearance) for p in lenses])
    if 'front' in parts:
        support = Polygon([(p['x']/scale_x, p['y']/scale_y) for p in parts['front']['points']])
        if support.is_valid:
            excluded = excluded.union(support.difference(unary_union(lenses)).buffer(clearance))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    candidates = {side: [] for side in ('left', 'right')}
    for level in (245, 248, 250):
        mask = np.uint8(gray < level)*255
        radius = max(1, round(lw/400))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((radius*2+1, radius*2+1), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        pieces = []
        for contour in contours:
            if len(contour) < 3 or cv2.contourArea(contour) < 8:
                continue
            shape = Polygon(contour[:, 0, :]).buffer(0).intersection(region).difference(excluded)
            pieces.extend(shape.geoms if hasattr(shape, 'geoms') else [shape])
        for side, lens in zip(('left', 'right'), lenses):
            choices = []
            for p in pieces:
                if p.geom_type != 'Polygon' or not p.is_valid or p.is_empty:
                    continue
                x0, y0, x1, y1 = p.bounds
                if (lens.area*.0025 < p.area < lens.area*.035 and
                        (p.centroid.x < center) == (side == 'left') and
                        lh*.08 < y1-y0 < lh*.45 and lw*.015 < x1-x0 < lw*.2 and
                        p.distance(region.boundary) > clearance and
                        p.convex_hull.area/p.area < 1.65):
                    choices.append(p)
            if len(choices) == 1:
                candidates[side].append(choices[0])
    output, recovered = [], []
    for side, choices in candidates.items():
        if f'nose-pad-{side}' in parts or len(choices) != 3:
            continue
        middle = choices[1]
        if any(middle.intersection(p).area/middle.union(p).area < .7 for p in choices):
            continue
        # Keep concave observed geometry; simplify only below a source pixel.
        shape = middle.simplify(max(.25, lw*.001), preserve_topology=True)
        if not shape.is_valid or shape.geom_type != 'Polygon':
            continue
        output.append(dict(photo_id=str(photo.get('id', '')), part=f'nose-pad-{side}',
                           points=[dict(x=float(x*scale_x), y=float(y*scale_y))
                                   for x, y in shape.exterior.coords[:-1]],
                           anchors={}, approved=False, source='automatic-visible-nasal-component-v1'))
        recovered.append(side)
    return output, dict(method='stable-visible-nasal-components', recovered_sides=recovered,
                        hidden_shape_recovered=False, depth_measured=False)
