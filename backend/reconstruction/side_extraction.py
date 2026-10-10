"""Conservative, review-only extraction of one visible side-photo arm."""

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter
from shapely.geometry import Polygon


def _runs(column, minimum_fraction=.35):
    rows = np.flatnonzero(column)
    if not len(rows):
        return []
    runs = np.split(rows, np.where(np.diff(rows) > 1)[0] + 1)
    # A one-pixel metal wire can be real at thumbnail size. Suppress only
    # fragments dwarfed by another run in the same column (e.g. a highlight
    # slit in the 24-pixel merged crossing of the reference photo).
    longest = max(map(len, runs))
    return [run for run in runs if len(run) >= max(1, longest * minimum_fraction)]


def _repair_crossing(upper, lower, selected, split, other_below, *, allow_long_cap=True):
    """Bridge a second arm's overlap without treating its pixels as sleeve material."""
    u = np.asarray(upper, dtype=float)
    l = np.asarray(lower, dtype=float)
    widths = l - u
    split_at = np.flatnonzero(split)
    if not len(split_at):
        return u, l
    length = len(u)
    max_gap = max(3, round(length * .055))
    boundary_tolerance = max(.75, float(np.percentile(widths, 20)) * .16)

    # Separated columns on either side of a short merge identify the obscured
    # boundary. Do not flatten a genuine sleeve across a distant gap.
    for left, right in zip(split_at[:-1], split_at[1:]):
        if right - left > max_gap:
            continue
        for index in range(left + 1, right):
            fraction = (index - left) / (right - left)
            predicted = selected[left] * (1 - fraction) + selected[right] * fraction
            if other_below[left] and other_below[right] and l[index] > predicted + boundary_tolerance:
                l[index] = predicted
            elif not other_below[left] and not other_below[right] and u[index] < predicted - boundary_tolerance:
                u[index] = predicted

    first = int(split_at[0])
    if allow_long_cap and first >= max(10, round(length * .10)):
        # An overlapping arm can touch the selected arm long before the first
        # white gap. Cap that rising width at the slim section's own thickness.
        baseline = float(np.percentile(widths[max(0, first - round(length * .286)):
                                            max(1, first - round(length * .08))], 25))
        limit = baseline + max(.75, baseline * .22)
        begin = first
        while begin > max(0, first - round(length * .206)) and widths[begin - 1] > limit:
            begin -= 1
        if first - begin >= max(2, round(length * .014)):
            if other_below[first]:
                l[begin:first] = np.minimum(l[begin:first], u[begin:first] + limit)
            else:
                u[begin:first] = np.maximum(u[begin:first], l[begin:first] - limit)
    return u, l


def _repair_short_lower_overlap(upper, lower, split, other_below):
    """Remove a short lower spur where a separated rear arm joins the shaft.

    The repair is anchored at the last visible gap. It ends as soon as the
    lower boundary returns to that boundary, so a later sleeve widening is
    preserved instead of being forced to a global shaft thickness.
    """
    u, l = upper.copy(), lower.copy()
    split = np.asarray(split, dtype=bool)
    below = np.asarray(other_below, dtype=bool)
    typical = float(np.percentile(l - u, 25))
    jump = max(2.5, typical * .20)
    search = max(12, round(len(l) * .12))
    for anchor in np.flatnonzero(split & below):
        if anchor < 4 or split[anchor - 1]:
            continue
        # Extraction proceeds hinge to tip. A merged crossing lies immediately
        # before the first again-separated column.
        peak = max(0, anchor - search)
        if np.max(l[peak:anchor]) - l[anchor] < jump:
            continue
        # The last boundary near the separated arm's height can already be
        # contaminated by the rear arm. Anchor at the preceding local minimum
        # in thickness instead. Subtracting the upper edge removes the shaft's
        # image slope; the bounded search leaves distant sleeve changes intact.
        widths = l - u
        local = median_filter(widths[peak:anchor], size=5, mode='nearest')
        begin = peak + int(np.argmin(local))
        if anchor - begin < 4 or np.max(l[begin:anchor]) - max(l[begin], l[anchor]) < jump:
            continue
        # The upper edge must continue smoothly through the join. Otherwise
        # this may be a real hook or a different branch entirely.
        if np.max(np.abs(np.diff(u[begin:anchor + 1]))) > max(3., typical * .12):
            continue
        l[begin:anchor + 1] = np.interp(np.arange(begin, anchor + 1),
                                        [begin, anchor], [l[begin], l[anchor]])
    return u, l


def _repair_shaft_before_sleeve(upper, lower, split, other_below):
    """Infer only a short obscured shaft edge before an observed sleeve step.

    Requires a separate lower arm and a sharp upper-edge sleeve transition.
    Never regularizes a single arm or the actual wider sleeve/ear hook.
    """
    u,l=np.asarray(upper,float),np.asarray(lower,float).copy()
    separated=np.flatnonzero(np.asarray(split)&np.asarray(other_below))
    if not len(separated):
        return u,l
    first=int(separated[0]); n=len(u)
    lo=max(4,first-round(n*.26));hi=first-round(n*.06)
    if hi-lo<10:
        return u,l
    delta=u[lo+3:hi+3]-u[lo:hi]
    seam=lo+int(np.argmin(delta))
    edge_jump=max(1.25,float(np.percentile(l-u,25))*.15)
    if np.min(delta)>-edge_jump:
        return u,l
    a=max(0,seam-round(n*.23));b=seam-round(n*.13)
    if b-a<15 or np.any(np.asarray(split)[a:b]):
        return u,l
    xs=np.arange(a,b);fit=np.polyfit(xs,l[a:b],1)
    if np.percentile(np.abs(l[a:b]-np.polyval(fit,xs)),90)>1.5:
        return u,l
    indices=np.arange(b,seam+1);predicted=np.polyval(fit,indices)
    width=predicted-u[indices]
    reference=np.median(l[a:b]-u[a:b])
    if np.any(width<reference*.45) or np.any(width>reference*1.6):
        return u,l
    excess=l[indices]-predicted
    surplus=max(1.25,reference*.2)
    if np.max(excess)>surplus*1.4 and np.count_nonzero(excess>surplus)>len(indices)*.3:
        l[indices]=np.minimum(l[indices],predicted+.5)
    return u,l


def extract_side(photo, mask, scale_x, scale_y, *, start_at_frame_boundary=False,
                 frame_start_fraction=.12, regularize_pale=False,
                 start_x=None, repair_short_overlap=False, upper_branch=False,
                 hinge_on_right=None):
    _, xs = np.where(mask > 0)
    if len(xs) < 12:
        return {'annotations': [], 'warnings': ['No arm silhouette found. Use a plain background or trace the arm in Advanced.']}
    lo, hi = int(xs.min()), int(xs.max())
    span = hi - lo
    if span < 20:
        return {'annotations': [], 'warnings': ['The arm occupies too few pixels; upload a larger side photo.']}
    # The frame is normally the vertically thick end; orient it toward the hinge.
    band = max(2, int(span * .18))
    left = float(np.max(np.sum(mask[:, lo:lo + band] > 0, axis=0)))
    right = float(np.max(np.sum(mask[:, hi - band:hi + 1] > 0, axis=0)))
    reverse = right > left
    if hinge_on_right is not None:
        reverse = bool(hinge_on_right)
    work = np.fliplr(mask) if reverse else mask
    if reverse:
        lo, hi = mask.shape[1] - 1 - hi, mask.shape[1] - 1 - lo
    middle = np.sum(work[:, lo + round(span * .3):lo + round(span * .6)] > 0, axis=0)
    has_frame = hinge_on_right is None and max(left, right) > max(3, float(np.median(middle)) * 2)
    start = lo + max(2, round(span * frame_start_fraction)) if has_frame else lo
    if not start_at_frame_boundary:
        for x in range(lo + max(2, round(span * .12)), lo + round(span * .35)):
            if len(_runs(work[:, x])) > 1:
                start = x
                break
    if start_x is not None:
        start = mask.shape[1] - 1 - int(start_x) if reverse else int(start_x)

    x_values, upper, lower, split, selected_edge, other_below = [], [], [], [], [], []
    previous = None
    missing = 0
    for x in range(start, hi + 1):
        runs = _runs(work[:, x], .08 if upper_branch else .35)
        if not runs:
            missing += 1
            if missing <= 2:
                continue
            break
        # Follow one branch through separated columns, retaining its own edge
        # where a second visible arm touches it.
        run = runs[0] if upper_branch else (runs[-1] if previous is None else min(runs, key=lambda r: abs((r[0] + r[-1]) / 2 - previous)))
        center = float(run[0] + run[-1]) / 2
        if previous is not None and abs(center - previous) > max(4, mask.shape[0] * .04):
            # At thumbnail resolution a one-column gap can separate a thin
            # wire from its neighbor. Skip it and reconnect to the same run.
            missing += 1
            if missing <= 2:
                continue
            break
        if len(run) > mask.shape[0] * .75:
            break
        missing = 0
        index = next(i for i, candidate in enumerate(runs) if candidate is run)
        neighbor_below = index < len(runs) - 1
        neighbor_above = index > 0
        x_values.append(x)
        upper.append(float(run[0]) - .4)
        lower.append(float(run[-1]) + .4)
        split.append(neighbor_below or neighbor_above)
        selected_edge.append(float(run[-1]) + .4 if neighbor_below else float(run[0]) - .4)
        other_below.append(neighbor_below)
        previous = center
    if len(upper) < max(15, span * .45):
        return {'annotations': [], 'warnings': ['Could not separate one continuous arm from the frame and other arm. Use Advanced to trace only one arm, or upload a clearer side photo.']}

    u, l = _repair_crossing(upper, lower, np.asarray(selected_edge), np.asarray(split), np.asarray(other_below), allow_long_cap=not upper_branch)
    if repair_short_overlap:
        u, l = _repair_short_lower_overlap(u, l, split, other_below)
        if upper_branch:
            u,l=_repair_shaft_before_sleeve(u,l,split,other_below)
    # A fixed starting column can still lie inside the front rim in an
    # oblique side photo. Stop at the observed shaft/root transition, not at
    # the bottom of the connected vertical lens rim. Require a stable nearby
    # shaft and a gross thickness excess, so ordinary wider hinge sleeves
    # are not shortened. Coordinates here always run from frame toward tip.
    if upper_branch and len(u) > 40:
        widths = l - u
        sample = widths[round(len(u) * .06):round(len(u) * .14)]
        reference = float(np.median(sample))
        stable = np.percentile(sample, 90) - np.percentile(sample, 10)
        if reference > 2 and stable < reference * .3 and widths[0] > reference * 2.2:
            limit = min(round(len(u) * .12), len(u) - 20)
            for root in range(1, limit):
                if np.all(widths[root:root + 3] <= reference * 1.15):
                    x_values = x_values[root:]
                    u, l = u[root:], l[root:]
                    break
    if regularize_pale:
        widths = l - u
        # At the attachment the metal disappears under the front frame. The
        # merged front is far taller than the arm in adjacent visible columns.
        reference = float(np.median(widths[max(8, round(len(widths) * .04)):
                                           max(18, round(len(widths) * .10))]))
        root = 0
        while (root < min(round(len(widths) * .05), len(widths) - 20) and
               widths[root] > max(12, reference * 1.45)):
            root += 1
        if root:
            x_values = x_values[root:]
            u, l = u[root:], l[root:]
            widths = l - u
        # Interior decorative highlights divide a textured shaft into short
        # apparent runs. Fill only narrow inward cuts; a wider sleeve remains.
        typical = median_filter(widths, size=31, mode='nearest')
        narrow = (widths < typical * .70) & (typical > 5)
        centers = (u + l) / 2
        u[narrow] = centers[narrow] - typical[narrow] / 2
        l[narrow] = centers[narrow] + typical[narrow] / 2
        upper_med = median_filter(u, size=21, mode='nearest')
        lower_med = median_filter(l, size=21, mode='nearest')
        u = np.clip(u, upper_med - 2.5, upper_med + 2.5)
        l = np.clip(l, lower_med - 2.5, lower_med + 2.5)
    # Remove the one-pixel staircase of a binary mask. A short kernel retains
    # the sleeve seam and the natural hook at the tip.
    sigma = max(.65, min(2.0, len(u) * .0023))
    u = gaussian_filter1d(u, sigma)
    l = gaussian_filter1d(l, sigma)
    if np.any(l <= u):
        return {'annotations': [], 'warnings': ['Arm boundary is ambiguous. Correct it in Advanced.']}
    keep = np.unique(np.r_[np.arange(0, len(u), max(1, round(len(u) / 290))), len(u) - 1])

    def point(x, y):
        return {'x': float((mask.shape[1] - 1 - x if reverse else x) * scale_x),
                'y': max(0, float(y * scale_y))}

    coords = [(x_values[i], u[i]) for i in keep] + [(x_values[i], l[i]) for i in keep[::-1]]
    poly = Polygon([(point(x, y)['x'], point(x, y)['y']) for x, y in coords])
    if not poly.is_valid:
        return {'annotations': [], 'warnings': ['Arm boundary is ambiguous. Correct it in Advanced.']}
    poly = poly.simplify(.6 * max(scale_x, scale_y), preserve_topology=True)
    from .contour_refinement import refine_closed_contour
    refined, _ = refine_closed_contour(list(poly.exterior.coords)[:-1],
        max_displacement_px=.85*max(scale_x,scale_y),
        sample_spacing_px=3.*max(scale_x,scale_y),window_px=5.*max(scale_x,scale_y))
    refined=np.array(Polygon(refined).simplify(.4*max(scale_x,scale_y),preserve_topology=True).exterior.coords[:-1])
    # Keep the physical attachment midpoint on the reconstructed boundary:
    # if smoothing moved the root cap, retain the previous valid contour.
    from shapely.geometry import Point
    hinge_point=point(x_values[0], (u[0]+l[0])/2)
    if Polygon(refined).distance(Point(hinge_point['x'],hinge_point['y'])) > 1e-7:
        refined=np.array(list(poly.exterior.coords)[:-1])
    points = [{'x': float(x), 'y': float(y)} for x, y in refined]
    part = 'temple-left' if 'left' in photo.get('view', '') else 'temple-right'
    ann = {'photo_id': str(photo['id']), 'part': part, 'points': points, 'approved': False,
           'source': 'single-visible-arm-smooth-v4',
           'anchors': {'hinge': point(x_values[0], (u[0] + l[0]) / 2),
                       'tip': point(x_values[-1], (u[-1] + l[-1]) / 2)}}
    return {'annotations': [ann], 'warnings': [
        'One visible arm was separated; the frame-side attachment is estimated. Inspect the highlighted silhouette, hinge and tip before approval. Other visible arms are excluded.',
        'Low-resolution side photo: thickness, hinge detail and hidden surfaces remain uncertain.' if min(mask.shape) < 128 else 'Side extrusion approximates cross-section; it is not recovered manufacturing geometry.']}


def recover_pale_side(image, photo, scale_x, scale_y):
    """Recover a textured pale shaft whose bright gaps split each mask column.

    This route is confined to a uniform light catalogue backdrop. A short
    vertical close joins one- or two-pixel highlight slits while retaining the
    actual arm's top, bottom, sleeve step, and bent ear tip. Agreement at three
    nearby contrast levels is required before suggesting an editable outline.
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
        return None
    distance = np.linalg.norm(lab - background, axis=2)
    chroma = np.linalg.norm(lab[:, :, 1:] - background[1:], axis=2)
    candidates = []
    for threshold in (8, 10, 12):
        if threshold < max(6, noise * 2.2):
            continue
        foreground = np.uint8((distance > threshold) &
                              ((chroma > 3) | (distance > 40))) * 255
        closed = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE,
                                  np.ones((5, 1), np.uint8))
        result = extract_side(photo, closed, scale_x, scale_y,
                              start_at_frame_boundary=True,
                              frame_start_fraction=.10,
                              regularize_pale=True)
        if not result['annotations']:
            continue
        ann = result['annotations'][0]
        polygon = Polygon([(p['x'], p['y']) for p in ann['points']])
        if not polygon.is_valid or polygon.area < width * height * scale_x * scale_y * .002:
            continue
        x_length = abs(ann['anchors']['tip']['x'] - ann['anchors']['hinge']['x'])
        if x_length < width * scale_x * .55:
            continue
        candidates.append((result, polygon))
    if len(candidates) != 3:
        return None
    result, polygon = candidates[1]
    if any(polygon.intersection(other).area / polygon.union(other).area < .78
           for _, other in (candidates[0], candidates[2])):
        return None
    # Two visible distal arms can merge into one shaft; an envelope then
    # switches physical branches at the crossing. Leave that view to the
    # branch-following extractor instead of proposing a false continuous arm.
    tip_x = result['annotations'][0]['anchors']['tip']['x'] / scale_x
    hinge_x = result['annotations'][0]['anchors']['hinge']['x'] / scale_x
    sampled = np.linspace(tip_x + (hinge_x - tip_x) * .08,
                          tip_x + (hinge_x - tip_x) * .35, 8)
    foreground = np.uint8((distance > 10) &
                          ((chroma > 3) | (distance > 40))) * 255
    closed = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE,
                              np.ones((5, 1), np.uint8))
    if sum(len(_runs(closed[:, int(round(x))])) > 1 for x in sampled) >= 3:
        return None
    result['annotations'][0]['source'] = 'single-visible-arm-pale-continuous-v1'
    result['warnings'] = [
        'A pale textured side contour was suggested from consistent image boundaries. Inspect the metal shaft, sleeve step, hinge attachment and bent tip before approval.',
        'The arm overlaps the front near the hinge; that hidden attachment and its cross-section remain estimated.']
    result['segmentation'] = {'method': 'stable-pale-side-vertical-gap-recovery',
                              'threshold_lab': 10, 'review_required': True}
    return result


def recover_opaque_side(image, photo, scale_x, scale_y):
    """Recover a dark opaque arm with a cutout mark and a visible hinge seam.

    A mark can divide one shaft into separate binary runs. The strong seam
    across the whole arm gives a physical root before the front, while a
    short lower spur at the rear-arm join is repaired by ``extract_side``.
    This route requires all of those image cues on a quiet light backdrop.
    """
    height, width = image.shape[:2]
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    margin = max(3, min(height, width) // 30)
    border = np.concatenate((lab[:margin].reshape(-1, 3),
                             lab[-margin:].reshape(-1, 3),
                             lab[:, :margin].reshape(-1, 3),
                             lab[:, -margin:].reshape(-1, 3)))
    background = np.median(border, axis=0)
    noise = float(np.percentile(np.linalg.norm(border - background, axis=1), 90))
    if background[0] < 225 or noise >= 7:
        return None
    distance = np.linalg.norm(lab - background, axis=2)
    threshold, _ = cv2.threshold(np.uint8(np.clip(distance, 0, 255)), 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if threshold < max(35, noise * 2.2):
        return None
    mask = np.uint8(distance > threshold) * 255
    ys, xs = np.where(mask > 0)
    if len(xs) < width * height * .01:
        return None
    lo, hi = int(xs.min()), int(xs.max())
    span = hi - lo
    if span < width * .55:
        return None
    band = max(2, round(span * .18))
    left = np.max(np.count_nonzero(mask[:, lo:lo + band], axis=0))
    right = np.max(np.count_nonzero(mask[:, hi - band:hi + 1], axis=0))
    reverse = right > left
    work = cv2.flip(mask, 1) if reverse else mask.copy()
    pixels = cv2.flip(lab, 1) if reverse else lab
    if reverse:
        lo, hi = width - 1 - hi, width - 1 - lo

    # Compare adjacent image columns through the uppermost arm run. A printed
    # mark has partial-height edges; a true joint seam crosses most of the arm.
    candidates = []
    for x in range(lo + round(span * .14), lo + round(span * .31)):
        runs = _runs(work[:, x])
        if not runs:
            continue
        upper = runs[0]
        if len(upper) < max(12, height * .018):
            continue
        edge = np.linalg.norm(pixels[upper, x] - pixels[upper, x - 1], axis=1)
        candidates.append((float(np.median(edge)), x))
    if not candidates:
        return None
    seam_strength, seam = max(candidates)
    if seam_strength < 15:
        return None

    # A bounded patch of split columns on the tip side of the seam identifies
    # an interior mark. Limit the vertical close to that patch: elsewhere a
    # close could join the separate rear arm to the main arm.
    patch_end = min(hi, seam + round(span * .10))
    split_columns = []
    for x in range(seam + 8, patch_end):
        runs = _runs(work[:, x])
        if len(runs) < 2:
            continue
        first, second = runs[:2]
        gap = int(second[0] - first[-1] - 1)
        if len(first) >= 6 and len(second) >= 6 and 1 <= gap <= max(20, height * .023):
            split_columns.append(x)
    if len(split_columns) < max(5, round(span * .007)):
        return None
    patch = work[:, min(split_columns) - 2:max(split_columns) + 3]
    closed = cv2.morphologyEx(patch, cv2.MORPH_CLOSE, np.ones((23, 1), np.uint8))
    work[:, min(split_columns) - 2:max(split_columns) + 3] = closed
    recovered_mask = cv2.flip(work, 1) if reverse else work
    root_x = width - 1 - seam if reverse else seam
    result = extract_side(photo, recovered_mask, scale_x, scale_y,
                          start_x=root_x, repair_short_overlap=True)
    if not result['annotations']:
        return None
    ann = result['annotations'][0]
    polygon = Polygon([(p['x'], p['y']) for p in ann['points']])
    if (not polygon.is_valid or
            abs(ann['anchors']['tip']['x'] - ann['anchors']['hinge']['x']) < span * scale_x * .65):
        return None
    ann['source'] = 'single-visible-arm-opaque-seam-v2'
    result['segmentation'] = {'method': 'opaque-arm-seam-and-mark-recovery',
                              'threshold_lab': float(threshold), 'review_required': True}
    result['warnings'] = [
        'The visible hinge seam and opaque arm boundaries were used to suggest one arm. Inspect the rear-arm crossing, hinge and tip before approval.',
        'The hidden attachment and arm cross-section remain estimated.']
    return result
