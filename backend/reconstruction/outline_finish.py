"""Finish automatic front suggestions in working pixels, before review.

Shadow trimming requires a quiet light backdrop and a stronger observed rim
immediately above a pale neutral extension. Pale rims without this evidence
are left intact. Neither operation changes source photos or saved outlines.
"""
import cv2
import numpy as np
from shapely.geometry import Polygon, Point, box
from shapely.ops import unary_union

from .smooth_curve import smooth_detected_curve


def _automatic_source(source):
    return source.startswith(('automatic', 'single-visible-arm-', 'product-local:',
                              'convex-occlusion-completion:'))


def _mask(poly, shape):
    mask = np.zeros(shape, np.uint8)
    cv2.fillPoly(mask, [np.rint(poly.exterior.coords).astype(np.int32)], 1)
    return mask


def trim_lower_shadow(image, front, lenses):
    info = dict(applied=False, removed_pixels=0)
    h, w = image.shape[:2]
    band = max(1, min(h, w) // 40)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    border = np.concatenate((lab[:band].reshape(-1, 3), lab[-band:].reshape(-1, 3),
                             lab[:, :band].reshape(-1, 3), lab[:, -band:].reshape(-1, 3)))
    background = np.median(border, axis=0)
    if background[0] < 225 or np.percentile(np.linalg.norm(border-background, axis=1), 90) > 5:
        info['reason'] = 'No verified quiet light background'
        return front, info
    contrast = np.linalg.norm(lab-background, axis=2)
    chroma = np.linalg.norm(lab[:, :, 1:]-background[1:], axis=2)
    # Contact shadows can be visibly gray (not just near-white). Demand
    # substantially darker/chromatic material above the neutral extension;
    # an ordinary global foreground threshold also admits these shadows.
    strong = (contrast > 90) | (chroma > 8)
    pale = (contrast < 80) & (chroma < 4)
    mask = _mask(front, (h, w))
    trimmed = mask.copy()
    changed_columns = 0
    material_area = front.area - sum(lens.area for lens in lenses)
    for lens in lenses:
        lens_mask = _mask(lens, (h, w))
        x0, y0, x1, y1 = lens.bounds
        # Avoid mounting points and lateral endpieces. No bridge region is cut.
        margin = (x1-x0) * .08
        reach = max(3, min(18, (y1-y0) * .07))
        for x in range(max(0, int(x0+margin)), min(w, int(x1-margin)+1)):
            lens_rows = np.flatnonzero(lens_mask[:, x])
            if not len(lens_rows) or lens_rows[-1] < y0 + (y1-y0) * .65:
                continue
            bottom = int(lens_rows[-1])
            rows = np.flatnonzero(mask[bottom+1:, x]) + bottom+1
            if not len(rows) or rows[-1]-bottom > max(32, (y1-y0)*.20):
                continue
            end = int(rows[-1])
            source = np.flatnonzero(strong[bottom+1:end+1, x]) + bottom+1
            if len(source):
                edge = int(source[-1])
                # A real supported rim precedes this neutral extension.
                cut = edge+2
            else:
                # A tinted lens can end at a pale transparent rim. Below it,
                # the neutral ground shadow rises and fades again without a
                # second material edge. Retain the lens boundary and up to two
                # transition pixels; never cut a coloured/dark outer rim.
                inside = slice(max(0,bottom-3),bottom+1)
                below = slice(bottom+1,end+1)
                if (np.max(contrast[inside,x]) < 90 and np.max(chroma[inside,x]) < 8):
                    continue
                if (np.max(contrast[below,x]) < 18 or contrast[end,x] > 16 or
                        np.max(chroma[below,x]) > 4):
                    continue
                cut = bottom+1
                while cut <= min(end,bottom+2) and not pale[cut,x]:
                    cut += 1
            if not 2 <= end-cut+1 <= reach:
                continue
            if not np.all(pale[cut:end+1, x]):
                continue
            # `cut` exists for both observed-rim and pale-rim paths; `edge`
            # exists only when a strong material pixel was found above it.
            if not np.all(mask[bottom+1:cut, x]):
                continue
            trimmed[cut:end+1, x] = 0
            changed_columns += 1
    if changed_columns < max(8, w*.01):
        info['reason'] = 'No sustained source-supported lower shadow extension'
        return front, info
    contours, _ = cv2.findContours(trimmed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    if not contours:
        return front, info
    # Trimming an attached ledge may leave isolated, one-pixel shadow flecks.
    # Discard only tiny neutral fragments, never a second material component.
    pixel_scale = max(.5, max(h, w)/1080)
    for fragment in contours[1:]:
        x, y, fw, fh = cv2.boundingRect(fragment)
        fragment_mask = np.zeros((h, w), np.uint8)
        cv2.drawContours(fragment_mask, [fragment], -1, 1, -1)
        pixels = fragment_mask > 0
        # A long, thin ground shadow can become detached when its narrow
        # contact with the photographed rim is cut. Its horizontal length
        # alone is not evidence of a second physical frame component.
        lens_height = min(p.bounds[3]-p.bounds[1] for p in lenses)
        lower_band = max(p.centroid.y for p in lenses) + lens_height*.28
        neutral_ground = (y >= lower_band and fh <= max(8, lens_height*.04) and
                          np.max(contrast[pixels]) <= 70 and np.max(chroma[pixels]) <= 4)
        tiny_fleck = (cv2.contourArea(fragment) <= 9*pixel_scale**2 and
                      fh <= 3*pixel_scale and fw <= 20*pixel_scale and
                      np.all((contrast[pixels] <= 85) & (chroma[pixels] <= 4)))
        if not (neutral_ground or tiny_fleck):
            info['reason'] = 'Detached region could still be real material'
            info['fragment_bounds'] = [int(x),int(y),int(fw),int(fh)]
            info['fragment_max_contrast'] = float(np.max(contrast[pixels]))
            info['fragment_max_chroma'] = float(np.max(chroma[pixels]))
            return front, info
        trimmed[pixels] = 0
    candidate = Polygon(contours[0].reshape(-1, 2))
    if not candidate.is_valid:
        # Raster contours may double back along a one-pixel contact ledge.
        # Remove zero-width excursions only; never join separate components.
        candidate = candidate.buffer(0)
    if candidate.geom_type == 'MultiPolygon':
        pieces = sorted(candidate.geoms,key=lambda p:p.area,reverse=True)
        main = pieces[0]
        lens_height = min(p.bounds[3]-p.bounds[1] for p in lenses)
        lower_band = max(p.centroid.y for p in lenses) + lens_height*.28
        satellites_safe = main.area/candidate.area >= .998
        for satellite in pieces[1:]:
            x0,y0,x1,y1=satellite.bounds
            fragment_mask = _mask(satellite,(h,w))
            pixels = fragment_mask>0
            satellites_safe &= (y0>=lower_band and y1-y0<=max(12,lens_height*.05) and
                                np.max(contrast[pixels])<=70 and np.max(chroma[pixels])<=4)
        if satellites_safe:
            candidate=main
    if candidate.geom_type != 'Polygon' or candidate.is_empty:
        info['reason'] = 'Shadow cut disconnected possible frame material'
        return front, info
    if any(not candidate.contains(lens) for lens in lenses):
        # Raster cuts can touch a subpixel part of the traced lens. Restore a
        # bounded clearance only from the original front, so no new exterior
        # silhouette is invented.
        protected = unary_union([lens.buffer(1.25) for lens in lenses]).intersection(front)
        candidate = candidate.union(protected)
    # Remove short neutral teeth left where a ground shadow still touches the
    # lower arc. Bound the cut by the *observed* lower rim clearance elsewhere
    # on this same lens; do not alter the brow, bridge, or lateral endpieces.
    for lens in lenses:
        x0,y0,x1,y1=lens.bounds
        lw,lh=x1-x0,y1-y0
        lower=[Point(x,y) for x,y in candidate.exterior.coords
               if x0+lw*.1<x<x1-lw*.1 and lens.centroid.y+lh*.2<y<y1-lh*.08]
        if len(lower)<10:
            continue
        distances=[lens.boundary.distance(point) for point in lower]
        clearance=float(np.clip(np.percentile(distances,20),2,12))
        region=box(x0+lw*.08,lens.centroid.y+lh*.2,x1-lw*.08,y1+30)
        bounded=candidate.difference(region.difference(lens.buffer(clearance+.75)))
        if (bounded.geom_type=='Polygon' and bounded.is_valid and bounded.contains(lens)
                and candidate.difference(bounded).area < material_area*.12):
            candidate=bounded
    lower_regions = unary_union([box(
        lens.bounds[0]+(lens.bounds[2]-lens.bounds[0])*.08,
        lens.centroid.y+(lens.bounds[3]-lens.bounds[1])*.2,
        lens.bounds[2]-(lens.bounds[2]-lens.bounds[0])*.08,
        lens.bounds[3]+30) for lens in lenses])
    rounded = candidate.buffer(-2.5*pixel_scale).buffer(2.5*pixel_scale)
    if rounded.geom_type=='Polygon':
        smoothed = candidate.difference(lower_regions).union(rounded.intersection(lower_regions))
        if (smoothed.geom_type=='Polygon' and smoothed.is_valid and
                all(smoothed.contains(lens) for lens in lenses) and
                candidate.symmetric_difference(smoothed).area < material_area*.01 and
                smoothed.difference(front).area < front.area*.00005):
            candidate=smoothed
    removed_area = front.difference(candidate).area
    if (not candidate.is_valid or candidate.area < front.area*.95 or
            removed_area > material_area*.25 or
            any(not candidate.contains(lens) for lens in lenses)):
        info['reason'] = 'Shadow cut failed joint outline checks'
        info['candidate_valid'] = bool(candidate.is_valid)
        info['retained_area_fraction'] = float(candidate.area/front.area)
        info['removed_material_fraction'] = float(removed_area/material_area)
        info['contains_lenses'] = [bool(candidate.contains(lens)) for lens in lenses]
        return front, info
    info.update(applied=True, removed_pixels=int(np.sum(mask-trimmed)),
                reason='Pale neutral extension below a stronger observed rim')
    return candidate, info


def finish_front_outlines(image, annotations, scale_x=1., scale_y=1.):
    """Finish the whole optical group atomically; preserve valid thin rims."""
    selected = [a for a in annotations if a.get('part') in
                {'front', 'lens-left', 'lens-right', 'bridge-opening', 'shield'} and
                not a.get('approved') and _automatic_source(a.get('source', ''))]
    polygons = {a['part']: Polygon([(p['x']/scale_x, p['y']/scale_y)
                                  for p in a['points']]) for a in selected}
    info = dict(smoothed_parts=[], shadow=dict(applied=False, removed_pixels=0))
    if not polygons or any(not p.is_valid or p.is_empty for p in polygons.values()):
        return info
    front = polygons.get('front')
    lenses = [polygons[key] for key in ('lens-left', 'lens-right') if key in polygons]
    if front is not None and len(lenses) == 2 and all(front.contains(p) for p in lenses):
        front, info['shadow'] = trim_lower_shadow(image, front, lenses)
        polygons['front'] = front
    scale = max(image.shape[:2]) / 1080
    smooth = {part: smooth_detected_curve(poly, scale, max_displacement=4.)
              for part, poly in polygons.items()}
    outer = smooth.get('front')
    holes = [p for part, p in smooth.items() if part in {'lens-left', 'lens-right', 'bridge-opening'}]
    if ((outer is not None and any(not outer.contains(p) for p in holes)) or
            ('lens-left' in smooth and 'lens-right' in smooth and
             not smooth['lens-left'].disjoint(smooth['lens-right']))):
        # Keep source-supported shadow removal even if joint smoothing fails.
        smooth = polygons
    for ann in selected:
        poly = smooth[ann['part']]
        if not poly.equals(polygons[ann['part']]):
            info['smoothed_parts'].append(ann['part'])
        ann['points'] = [dict(x=float(x*scale_x), y=float(y*scale_y))
                         for x, y in list(poly.exterior.coords)[:-1]]
    return info


def finish_side_outlines(image, annotations, scale_x=1., scale_y=1.):
    info = dict(smoothed_parts=[])
    for ann in annotations:
        if (not ann.get('part', '').startswith('temple-') or ann.get('approved') or
                not _automatic_source(ann.get('source', ''))):
            continue
        poly = Polygon([(p['x']/scale_x, p['y']/scale_y) for p in ann['points']])
        anchors = [(p['x']/scale_x, p['y']/scale_y) for p in ann.get('anchors', {}).values()]
        clean = smooth_detected_curve(poly, max(image.shape[:2])/1080,
                                      anchors=anchors, max_area_change=.02, max_displacement=4.)
        from shapely.geometry import Point
        # Protected vertices alone do not protect the midpoint of a root cap.
        # Retain the previous contour if smoothing displaces a covered anchor.
        if any(poly.buffer(1e-8).covers(Point(*p)) and
               not clean.buffer(1e-8).covers(Point(*p)) for p in anchors):
            clean = poly
        if not clean.equals(poly):
            ann['points'] = [dict(x=float(x*scale_x), y=float(y*scale_y))
                             for x, y in list(clean.exterior.coords)[:-1]]
            info['smoothed_parts'].append(ann['part'])
    return info
