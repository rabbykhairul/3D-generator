"""Conservative extraction of a paired, dark-top/clear-bottom front.

This deliberately recognizes a narrow photographic case.  It returns no
suggestions when the faint lower edge cannot be distinguished from a shadow,
an arm, or a dark complete rim.
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter
from shapely.geometry import Polygon, box
from shapely.ops import unary_union


def _points(coords, sx, sy):
    return [{"x": float(x * sx), "y": float(y * sy)} for x, y in coords]


def _annotation(photo, part, coords, sx, sy):
    return {"photo_id": str(photo.get("id", "")), "part": part,
            "points": _points(coords, sx, sy), "anchors": {}, "approved": False,
            "source": "automatic-paired-semi-rimless-v1"}


def _upper_path(labels, label, left, right, top, height):
    samples = []
    for x in np.linspace(left, right, 45):
        ix = int(round(x))
        band = labels[max(0, top - 3):min(labels.shape[0], top + int(height * .13)),
                      max(0, ix - 3):min(labels.shape[1], ix + 4)] == label
        ys = np.where(np.any(band, axis=1))[0]
        if len(ys):
            samples.append((x, float(ys[0] + max(0, top - 3))))
    if len(samples) < 35:
        return None
    # The selected dark component may include a side tab.  Within the lens
    # span, its highest pixels trace the metal brow with little arm influence.
    ys = cv2.GaussianBlur(np.asarray([p[1] for p in samples], np.float32)
                           .reshape(-1, 1), (1, 5), 0).ravel()
    return [(float(p[0]), float(y)) for p, y in zip(samples, ys)]


def _lower_path(gray, left, right, outside_y, inside_y, bottom_y, side):
    """Follow light edge within a lens-shaped corridor, excluding arm regions."""
    width = right - left
    center = (left + right) / 2
    samples = []
    contrast = 255.0 - gray.astype(np.float32)
    for x in np.linspace(left, right, 75):
        if side == "left":
            from_outside = (x - left) / (center - left)
            from_inside = (right - x) / (right - center)
        else:
            from_outside = (right - x) / (right - center)
            from_inside = (x - left) / (center - left)
        if from_outside <= 1:
            predicted = bottom_y - (bottom_y - outside_y) * max(0, 1 - from_outside) ** 5
        else:
            predicted = bottom_y - (bottom_y - inside_y) * max(0, 1 - from_inside) ** 3
        ix = int(round(x))
        # A small corridor prevents the stronger dark rear arm and the lower
        # ground shadow from replacing the transparent lens edge.
        radius = max(5, int(width * .045))
        lo = max(0, int(predicted - radius))
        hi = min(gray.shape[0], int(predicted + radius + 1))
        if hi <= lo:
            return None
        line = contrast[lo:hi, max(0, ix - 2):min(gray.shape[1], ix + 3)].mean(axis=1)
        line = cv2.GaussianBlur(line.astype(np.float32).reshape(-1, 1), (1, 5), 0).ravel()
        target = np.arange(lo, hi, dtype=float)
        merit = line - .065 * (target - predicted) ** 2
        best = int(np.argmax(merit))
        samples.append((float(x), float(lo + best), float(line[best])))
    if len(samples) != 75:
        return None
    # Test the broad visible lower arc, away from screws, nose pads and arms.
    core = np.asarray([p[2] for p in samples[15:60]])
    if np.median(core) < 8 or np.mean(core > 7) < .8:
        return None
    # Independent column maxima can briefly jump onto a ground shadow. A
    # short robust arc fit rejects that local excursion without replacing
    # the lower edge by a generic ellipse. Keep the attachment endpoints.
    ys=np.array([p[1] for p in samples])
    smooth=savgol_filter(median_filter(ys,size=11,mode='nearest'),17,3,mode='interp')
    smooth[:3]=ys[:3];smooth[-3:]=ys[-3:]
    if np.max(np.abs(smooth-ys))>width*.03:
        return None
    samples=[(x,float(y),v) for (x,_,v),y in zip(samples,smooth)]
    return samples


def analyze_construction(image, photo, scale_x, scale_y):
    """Return draft annotations and construction evidence, or decline.

    ``image`` is the bounded OpenCV BGR analysis raster; output points use
    source-photo pixels via the caller's two independent coordinate scales.
    """
    if str(photo.get("view", "front")).lower() not in {"front", ""}:
        return None
    if image is None or image.ndim != 3 or image.shape[2] != 3:
        return None
    h, w = image.shape[:2]
    if min(h, w) < 250:
        return None
    # Locate the two dark brows before choosing a search window. Their bounds
    # provide a product coordinate system regardless of canvas padding, crop,
    # or aspect ratio. A crop that removes a brow cannot pass the pair check.
    raw_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dark = (raw_gray < 180).astype(np.uint8)
    count0, labels0, stats0, centers0 = cv2.connectedComponentsWithStats(dark, 8)
    large = sorted((tuple(map(int, stats0[i])) for i in range(1, count0)
                    if stats0[i, 2] >= 50 and stats0[i, 4] >= 100),
                   key=lambda s: s[2], reverse=True)
    if len(large) < 2:
        return None
    first, second = sorted(large[:2], key=lambda s: s[0])
    if not (.72 < first[2]/second[2] < 1.38 and
            abs(first[1]-second[1]) < max(first[2],second[2])*.07 and
            first[0]+first[2] < second[0]+second[2]*.15):
        return None
    span = second[0] + second[2] - first[0]
    if span < 400 or span > min(w,h)*3.2:
        return None
    crop_width = span * 1.082
    crop_x = first[0] - span*.041
    crop_y = min(first[1],second[1]) - crop_width*.43
    factor = 1080 / crop_width
    work = cv2.warpAffine(image,
                          np.array([[factor, 0, -crop_x*factor],
                                    [0, factor, -crop_y*factor]], dtype=np.float32),
                          (1080, 1080), borderValue=(255,255,255),
                          flags=cv2.INTER_AREA if factor < 1 else cv2.INTER_LINEAR)
    H, W = work.shape[:2]
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    corners = np.concatenate((gray[:int(H*.3), :int(W*.2)].ravel(),
                              gray[:int(H*.3), -int(W*.2):].ravel()))
    if np.median(corners) < 247 or np.std(corners) > 8:
        return None
    roi = np.zeros_like(gray, dtype=np.uint8)
    roi[int(H*.36):int(H*.62)] = (gray[int(H*.36):int(H*.62)] < 180).astype(np.uint8)
    count, labels, stats, centers = cv2.connectedComponentsWithStats(roi, 8)
    candidates = []
    for i in range(1, count):
        x, y, cw, ch, area = map(int, stats[i])
        if cw > W*.32 and cw < W*.51 and ch > H*.07 and ch < H*.17 and area > W*H*.002:
            candidates.append((area, i, x, y, cw, ch))
    candidates.sort(reverse=True)
    if len(candidates) < 2:
        return None
    pair = sorted(candidates[:2], key=lambda c: c[2])
    left, right = pair
    if not (left[2] < W*.13 and W*.39 < left[2]+left[4] < W*.49 and
            W*.51 < right[2] < W*.61 and right[2]+right[4] > W*.87 and
            abs(left[3]-right[3]) < H*.025):
        return None
    # The optical lower edge is consistently faint. A complete dark lower
    # rim or a rear arm crossing the lower arc fails this check.
    tops = []
    lowers = []
    rim_ends = []
    outer_rim_ends = []
    rim_widths = []
    endpieces = []
    for candidate, side in ((left, "left"), (right, "right")):
        _, label, bx, by, bw, bh = candidate
        if side == "left":
            lx, rx = bx + .035*W, bx + bw - .025*W
            inner_box = (int(rx-.045*W), int(rx+.01*W))
        else:
            lx, rx = bx + .025*W, bx + bw - .035*W
            inner_box = (int(lx-.01*W), int(lx+.045*W))
        upper = _upper_path(labels, label, lx, rx, by, H)
        if upper is None:
            return None
        near_nose = roi[int(H*.53):int(H*.62), max(0,inner_box[0]):min(W,inner_box[1])]
        yy, xx = np.where(near_nose)
        if len(yy) < 10:
            return None
        inside_y = float(min(H*.61, int(H*.53)+np.percentile(yy, 98)))
        end_rows = yy >= np.percentile(yy, 97)
        rim_ends.append((float(max(0,inner_box[0]) + np.median(xx[end_rows])),
                         float(int(H*.53) + np.median(yy[end_rows]))))
        outside_y = float(by + bh)
        outer_rim_ends.append((lx if side == 'left' else rx, outside_y))
        cx = int(round((lx+rx)/2))
        bottom_band = (255-gray[int(H*.63):int(H*.71), int(cx-.12*(rx-lx)):int(cx+.12*(rx-lx))]).mean(axis=1)
        bottom_band = cv2.GaussianBlur(bottom_band.astype(np.float32).reshape(-1,1),(1,7),0).ravel()
        bottom_y = float(int(H*.63)+int(np.argmax(bottom_band)))
        if bottom_band.max() < 8 or bottom_y < by + H*.19 or bottom_y > by + H*.28:
            return None
        lower = _lower_path(gray, lx, rx, outside_y, inside_y, bottom_y, side)
        if lower is None:
            return None
        if np.median([v for _,_,v in lower[15:60]]) > 55:
            return None
        bottom_pixels = []
        for x, y, _ in lower[15:60]:
            ix, iy = int(round(x)), int(round(y))
            bottom_pixels.append(work[iy, ix].astype(np.float32))
        bottom_pixels = np.asarray(bottom_pixels)
        chroma = np.median(bottom_pixels.max(axis=1)-bottom_pixels.min(axis=1))
        # Rose-gold or brown full lower rims may be pale in grayscale, but
        # their color is stable along the whole arc. A clear edge is neutral.
        if chroma > 15:
            return None
        thickness = []
        for x, y in upper:
            ix, iy = int(round(x)), int(round(y))
            column = labels[max(0,iy-2):min(H,iy+20), ix] == label
            runs = np.where(column)[0]
            if not len(runs):
                return None
            start = int(runs[0]); end = start
            while end+1 < len(column) and column[end+1]:
                end += 1
            thickness.append(float(np.clip(end-start+1, 1, 18)))
        # Local observed width replaces the fixed five-pixel inner offset.
        # A narrow brow must not become a broad band from uniform dilation.
        thickness = median_filter(np.asarray(thickness), size=5, mode='nearest')
        rim_widths.extend(thickness[10:35:3])
        inner = [(x, y+float(t)) for (x,y),t in zip(upper,thickness)]
        tops.append(inner)
        lowers.append(lower)
        # Retain the actual outer tab, not a synthetic offset from the lens.
        # The narrow outside corridor excludes the folded arm inside the lens.
        tab_mask = np.zeros((H,W), np.uint8)
        x0,x1 = (bx, int(lx+3)) if side=='left' else (int(rx-3), bx+bw)
        tab_mask[by:by+bh+1, x0:x1+1] = (labels[by:by+bh+1, x0:x1+1]==label).astype(np.uint8)
        contours,_ = cv2.findContours(tab_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        tab = max(contours, key=cv2.contourArea, default=None)
        if tab is not None and cv2.contourArea(tab)>10:
            shape=Polygon(tab[:,0,:]).buffer(0)
            if shape.geom_type=='Polygon':
                endpieces.append((side, shape.buffer(.5)))
    if abs(max(y for _,y,_ in lowers[0])-max(y for _,y,_ in lowers[1])) > H*.025:
        return None
    lens_polygons = []
    for top, lower in zip(tops, lowers):
        upper_inner = list(top)
        lower_inner = [(x, y-4) for x,y,_ in reversed(lower)]
        polygon = Polygon(upper_inner + lower_inner)
        if not polygon.is_valid or polygon.area < W*H*.025:
            return None
        lens_polygons.append(polygon)
    # The single front outline follows both clear lens edges and the narrow
    # bridge. Its lower perimeter is clear lens edge, not an opaque metal band.
    bridge_left,bridge_right=tops[0][-1][0],tops[1][0][0]
    center_left=int(bridge_left+(bridge_right-bridge_left)*.3)
    center_right=int(bridge_right-(bridge_right-bridge_left)*.3)
    bridge_guess=max(tops[0][-1][1],tops[1][0][1])
    y0=max(0,int(bridge_guess-H*.025));y1=min(H,int(bridge_guess+H*.085))
    profile=np.median(255-gray[y0:y1,center_left:center_right].astype(float),axis=1)
    if not len(profile) or profile.max()<10:
        return None
    peak=int(np.argmax(profile));low=high=peak
    while low>0 and profile[low-1]>profile[peak]*.45:low-=1
    while high+1<len(profile) and profile[high+1]>profile[peak]*.45:high+=1
    bridge=box(bridge_left-4,y0+low-1,bridge_right+4,y0+high+1)
    # Keep the provisional side/lower envelope used to terminate the upper
    # rim, but never let that clearance inflate its photographed top edge.
    envelope = float(np.clip(np.median(rim_widths)/2+4, 5, 12))
    front = unary_union([p.buffer(envelope, join_style=1) for p in lens_polygons] + [bridge]).buffer(0)
    for candidate, inner in zip(pair, tops):
        _,label,bx,by,bw,bh=candidate
        outer=_upper_path(labels,label,inner[0][0],inner[-1][0],by,H)
        if outer is None:
            return None
        roof=Polygon([(outer[0][0],0), (outer[-1][0],0)] + list(reversed(outer)))
        front=front.difference(roof)
    front=unary_union([front]+[shape for _,shape in endpieces]).buffer(0)
    if front.geom_type != "Polygon" or not all(front.contains(p) for p in lens_polygons):
        return None
    def source_coords(coords):
        return [(x/factor+crop_x, y/factor+crop_y) for x,y in coords]
    anns = [_annotation(photo, "front", source_coords(list(front.exterior.coords)[:-1]), scale_x, scale_y)]
    anns[0]["anchors"] = {f"rim-end-{side}": {"x": float((xy[0]/factor+crop_x)*scale_x),
                                                    "y": float((xy[1]/factor+crop_y)*scale_y)}
                            for side, xy in zip(("left", "right"), rim_ends)}
    anns[0]['anchors'].update({f'rim-outer-end-{side}': {
        'x':float((xy[0]/factor+crop_x)*scale_x), 'y':float((xy[1]/factor+crop_y)*scale_y)}
        for side,xy in zip(('left','right'),outer_rim_ends)})
    for side,shape in endpieces:
        point=shape.representative_point()
        anns[0]['anchors'][f'hinge-{side}']={
            'x':float((point.x/factor+crop_x)*scale_x),
            'y':float((point.y/factor+crop_y)*scale_y)}
    anns += [_annotation(photo, f"lens-{side}", source_coords(list(p.exterior.coords)[:-1]), scale_x, scale_y)
             for side,p in zip(("left", "right"),lens_polygons)]
    return {"annotations": anns,
            "warnings": ["Semi-rimless draft: verify both faint lower lens edges against the source pixels before approval; rear arms and ground shadows are excluded.",
                         "The front envelope around each lens and the bridge is inferred from visible edge thickness; review its material boundary.",
                         "The lower lens edge is clear; its support wire cannot be verified from this photo."],
            "segmentation": {"method": "paired-dark-brow-light-lens-edge", "review_required": True},
            "construction": {"category": "semi-rimless", "confidence": "medium",
                             "reason": "Two matching dark upper rims and two faint, lens-shaped lower edges on a light background.",
                             "lower_support": "unverified"}}
