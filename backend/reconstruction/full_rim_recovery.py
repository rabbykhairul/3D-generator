"""Recover a full front with two lenses and a separate bridge opening.

The usual hierarchy contour can self-touch at single-pixel diagonal joins.
This helper accepts the topology only when it is stable across nearby color
thresholds and the repaired boundary differs only by tiny detached slivers.
"""

from __future__ import annotations

import cv2
import numpy as np
from shapely.geometry import Polygon


def _largest_valid(contour):
    if len(contour)<3 or cv2.contourArea(contour)<=0:
        return None
    raw = Polygon(contour.reshape(-1, 2))
    repaired = raw.buffer(0)
    if repaired.is_empty:
        return None
    if repaired.geom_type == "MultiPolygon":
        main = max(repaired.geoms, key=lambda p: p.area)
        if main.area < repaired.area * .995:
            return None
        repaired = main
    return repaired if repaired.geom_type == "Polygon" and repaired.is_valid else None


def _candidate(mask, width, height, complete_lenses=True, allow_single=False):
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return None
    parent = hierarchy[0, :, 3]
    outer_options = sorted(((cv2.contourArea(c), i) for i,c in enumerate(contours)
                            if parent[i] == -1), reverse=True)
    if not outer_options:
        return None
    _, outer_index = outer_options[0]
    outer = _largest_valid(contours[outer_index])
    if outer is None:
        return None
    x0,y0,x1,y1 = outer.bounds
    bw,bh = x1-x0,y1-y0
    # Frame proportions belong to the object, not the amount of blank photo
    # margin. Tight landscape crops must not fail the square-catalogue limits.
    if not (.45*width < bw < .99*width and .12*bw < bh < .65*bw and bh < .97*height and
            x0 > 1 and y0 > 1 and x1 < width-1 and y1 < height-1):
        return None
    openings = sorted(((cv2.contourArea(c),i) for i,c in enumerate(contours)
                       if parent[i] == outer_index and cv2.contourArea(c) > outer.area*.012),
                      reverse=True)
    if len(openings) not in ((2,3) if allow_single else (3,)):
        return None
    lens_pairs = sorted(openings[:2], key=lambda ai: cv2.boundingRect(contours[ai[1]])[0])
    bridge_pair = openings[2] if len(openings)==3 else None
    if min(a for a,_ in lens_pairs)/max(a for a,_ in lens_pairs) < .72:
        return None
    lenses = []
    for area, i in lens_pairs:
        observed = _largest_valid(contours[i])
        if observed is None or observed.area < outer.area*.2:
            return None
        completed = observed.convex_hull if complete_lenses else observed
        if completed.area > observed.area*1.12:
            return None
        lenses.append(completed)
    bridge = _largest_valid(contours[bridge_pair[1]]) if bridge_pair else None
    if bridge_pair and (bridge is None or bridge.area < outer.area*.012 or bridge.area > outer.area*.055):
        return None
    # Repair the bridge's few pixel self-touches without inventing a convex
    # nose opening: its observed upper and lower bars remain in the geometry.
    holes = lenses + ([bridge] if bridge is not None else [])
    if not all(outer.contains(p) for p in holes):
        return None
    if any(holes[i].intersects(holes[j]) for i in range(len(holes)) for j in range(i+1,len(holes))):
        return None
    left,right = lenses
    if bridge is not None and not (left.centroid.x < bridge.centroid.x < right.centroid.x and
            bridge.bounds[1] < min(left.centroid.y,right.centroid.y) and
            bridge.bounds[3] < min(left.bounds[3],right.bounds[3]) and
            left.bounds[0] < bridge.bounds[0] < bridge.bounds[2] < right.bounds[2]):
        return None
    frame = Polygon(outer.exterior.coords,[p.exterior.coords for p in holes])
    if not frame.is_valid or frame.area < outer.area*.06:
        return None
    return outer, lenses, bridge


def recover_double_bridge_front(image, photo, scale_x, scale_y):
    """Return editable source-pixel annotations, or ``None`` on ambiguity."""
    if image is None or image.ndim != 3 or image.shape[2] != 3:
        return None
    height,width = image.shape[:2]
    if min(height,width) < 400:
        return None
    lab = cv2.cvtColor(image,cv2.COLOR_BGR2LAB).astype(np.float32)
    edge = max(3,min(width,height)//30)
    border = np.concatenate((lab[:edge].reshape(-1,3),lab[-edge:].reshape(-1,3),
                             lab[:,:edge].reshape(-1,3),lab[:,-edge:].reshape(-1,3)))
    background = np.median(border,axis=0)
    noise = float(np.percentile(np.linalg.norm(border-background,axis=1),90))
    if background[0] < 225 or noise >= 7:
        return None
    distance = np.linalg.norm(lab-background,axis=2)
    chroma = np.linalg.norm(lab[:,:,1:]-background[1:],axis=2)
    candidates = []
    for threshold in (6,8,10,12,16,20,24,30):
        if threshold < max(6,noise*2.2):
            continue
        mask = np.uint8((distance>threshold)&((chroma>3)|(distance>40)))*255
        geometry = _candidate(mask,width,height)
        if geometry is not None:
            boxes = [p.bounds for p in (geometry[0],*geometry[1],geometry[2])]
            candidates.append((threshold,geometry,boxes))
    if len(candidates) < 4:
        return None
    stable = []
    for threshold,geometry,boxes in candidates:
        neighbors = sum(1 for other,_,other_boxes in candidates
                        if other != threshold and abs(other-threshold)<=10 and
                        max(abs(a-b) for box_a,box_b in zip(boxes,other_boxes)
                            for a,b in zip(box_a,box_b)) < max(width,height)*.025)
        if neighbors >= 3:
            stable.append((threshold,geometry))
    if not stable:
        return None
    threshold,(outer,lenses,bridge) = min(stable,key=lambda item:abs(item[0]-16))
    from .front_cleanup import clean_lower_rim
    outer, _ = clean_lower_rim(outer, lenses)
    result = []
    for part,poly,completed in [("front",outer,False),("lens-left",lenses[0],True),
                                ("lens-right",lenses[1],True),("bridge-opening",bridge,False)]:
        # Simplification reduces pixel stair-stepping, but only retain it if
        # topology and all separations survive. The original is the fallback.
        refined = poly.simplify(.75,preserve_topology=True)
        if refined.geom_type == "Polygon" and refined.is_valid and refined.area >= poly.area*.995:
            poly = refined
        coords = list(poly.exterior.coords)[:-1]
        result.append({"photo_id":str(photo.get("id","")),"part":part,
                       "points":[{"x":float(x*scale_x),"y":float(y*scale_y)} for x,y in coords],
                       "anchors":{},"approved":False,
                       "source":("convex-occlusion-completion:" if completed else "")+
                                "automatic-stable-double-bridge-v1"})
    actual = [Polygon([(p['x'],p['y']) for p in ann['points']]) for ann in result]
    if not all(p.is_valid for p in actual) or not all(actual[0].contains(p) for p in actual[1:]) or any(
            actual[i].intersects(actual[j]) for i in range(1,4) for j in range(i+1,4)):
        return None
    return {"annotations":result,
            "warnings":["Double-bridge frame recovered from stable color and aperture boundaries. Inspect the outer rim and all three openings before approval.",
                        "The rear arms hide parts of both lens edges; those short occluded arcs are suggested by convex completion and require review."],
            "segmentation":{"method":"stable-double-bridge","threshold_lab":threshold,"review_required":True}}


def recover_dark_front(image, photo, scale_x, scale_y):
    """Separate filled tinted lenses from light metal; never assume a dark blob is a rim.

    Require two solid, balanced lenses at multiple thresholds, plus an observed
    connected rim. An extra bridge aperture is optional, never invented.
    All coordinates are measured in the image.
    """
    h, w = image.shape[:2]
    if min(h, w) < 80:
        return None
    factor = 1080 / max(h, w)
    work = cv2.resize(image, (round(w*factor), round(h*factor)), interpolation=cv2.INTER_AREA if factor < 1 else cv2.INTER_CUBIC)
    h, w = work.shape[:2]
    lab = cv2.cvtColor(work, cv2.COLOR_BGR2LAB).astype(np.float32)
    border = np.concatenate((lab[:8].reshape(-1,3), lab[-8:].reshape(-1,3),
                             lab[:,:8].reshape(-1,3), lab[:,-8:].reshape(-1,3)))
    bg = np.median(border, axis=0)
    if bg[0] < 240 or np.percentile(np.linalg.norm(border-bg, axis=1),90) > 5:
        return None
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    pairs = []
    for threshold in range(90,191,10):
        dark = np.uint8(gray < threshold)*255
        contours,_ = cv2.findContours(dark,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        large = sorted((c for c in contours if cv2.contourArea(c) > w*h*.04),key=cv2.contourArea,reverse=True)
        if len(large) != 2:
            continue
        pair = sorted(large,key=lambda c:cv2.boundingRect(c)[0])
        polys = [_largest_valid(c) for c in pair]
        if any(p is None for p in polys):
            continue
        left,right = polys
        if min(left.area,right.area)/max(left.area,right.area) < .75 or left.intersects(right):
            continue
        if abs(left.centroid.y-right.centroid.y) > h*.025 or not left.centroid.x < w*.5 < right.centroid.x:
            continue
        occupied = np.zeros_like(dark)
        cv2.drawContours(occupied,pair,-1,255,-1)
        if np.count_nonzero(dark & occupied)/np.count_nonzero(occupied) < .9:
            continue
        pairs.append((pair,polys))
    if len(pairs) < 3:
        return None
    # Find a stable interval, not agreement with every higher threshold. A
    # higher threshold can absorb a grey rim, bridge or endpiece into a lens;
    # that must not invalidate a separately supported lower-contrast interval.
    stable=[]
    for start in range(len(pairs)-2):
        group=pairs[start:start+3]
        if all(a.hausdorff_distance(b)<=6.5
               for _,ps in group for _,qs in group for a,b in zip(ps,qs)):
            stable=group
            break
    if not stable:
        return None
    pair,polys = stable[0]
    distance = np.linalg.norm(lab-bg,axis=2)
    chroma = np.linalg.norm(lab[:,:,1:]-bg[1:],axis=2)
    candidates = []
    for neutral in (False,True):
        candidates=[]
        for threshold in (6,8,10,12):
            mask = np.uint8((distance>threshold)&(True if neutral else ((chroma>3)|(distance>40))))*255
            cv2.drawContours(mask,pair,-1,0,-1)
            geometry = _candidate(mask,w,h,complete_lenses=False,allow_single=True)
            if geometry is not None:
                candidates.append(geometry)
        if len(candidates)>=3:
            break
    if len(candidates) < 3:
        return None
    outer,lenses,bridge = candidates[1]
    if any((g[2] is None)!=(bridge is None) for g in candidates):
        return None
    if any(outer.hausdorff_distance(g[0]) > 15 or
           outer.intersection(g[0]).area / outer.union(g[0]).area < .99 for g in candidates):
        return None
    # Nose pads protrude into the nasal gap but are not the frontal metal rim.
    # Measure rim width on the unobstructed lower arcs, then bound only that
    # gap below the bridge. Endpieces and both bridge bars stay untouched.
    from shapely.geometry import Point, box, LineString
    from shapely.ops import unary_union
    distances = [min(p.exterior.distance(Point(x,y)) for p in lenses)
                 for x,y in outer.exterior.coords
                 if y > max(p.centroid.y for p in lenses)]
    rim = float(np.clip(np.median(distances), 2, 10))
    center_x=(lenses[0].bounds[2]+lenses[1].bounds[0])/2
    bridge_bottom = bridge.bounds[3] if bridge is not None else outer.intersection(LineString([(center_x,0),(center_x,h)])).bounds[3]
    holes=lenses+([bridge] if bridge is not None else [])
    gap = box(lenses[0].centroid.x, bridge_bottom+rim*2,
              lenses[1].centroid.x, max(p.bounds[3] for p in lenses))
    allowed = unary_union([p.buffer(rim*1.3) for p in lenses])
    trimmed = outer.difference(gap.difference(allowed))
    pad_centers = {}
    pad_shapes = {}
    removed = outer.difference(trimmed)
    pieces = list(removed.geoms) if hasattr(removed,'geoms') else [removed]
    for side,lens in zip(('left','right'),lenses):
        options = [p for p in pieces if p.geom_type=='Polygon' and p.area > lens.area*.001
                   and p.area < lens.area*.025 and
                   abs(p.centroid.x-lens.centroid.x) < (lens.bounds[2]-lens.bounds[0])*.7
                   and p.bounds[1] > bridge_bottom]
        if options:
            observed = max(options,key=lambda p:p.area)
            # A long concave strip beside the rim is not a solid pad. Its
            # convex hull can fill a large triangular region of white space.
            if observed.convex_hull.area <= observed.area*1.5:
                pad_shapes[side] = observed.simplify(.7,preserve_topology=True)
                pad_centers[side] = pad_shapes[side].centroid
    # Independently trace dark pads in the nasal gap. This is separate from
    # the rim-trimming residue and preserves observed silhouettes rather than
    # treating a convex completion of missing pixels as hardware evidence.
    pad_observations={side:[] for side in ('left','right')}
    for threshold in (90,110,130):
        pad_mask=np.uint8(gray<threshold)*255
        for lens in lenses:
            cv2.fillPoly(pad_mask,[np.rint(lens.buffer(3).exterior.coords).astype(np.int32)],0)
        contours,_=cv2.findContours(pad_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        for side,lens in zip(('left','right'),lenses):
            options=[]
            for contour in contours:
                shape=_largest_valid(contour)
                if shape is None:
                    continue
                x0,y0,x1,y1=shape.bounds
                if (lens.area*.001<shape.area<lens.area*.025 and
                        y0>bridge_bottom+rim*2 and y1-y0<(lens.bounds[3]-lens.bounds[1])*.45 and
                        lenses[0].centroid.x<x0<x1<lenses[1].centroid.x and
                        (shape.centroid.x<center_x)==(side=='left')):
                    options.append(shape)
            if len(options)==1:
                pad_observations[side].append(options[0])
    for side,observations in pad_observations.items():
        if len(observations)==3 and all(
                observations[1].intersection(p).area/observations[1].union(p).area>.7
                for p in observations):
            pad_shapes[side]=observations[1].simplify(.5,preserve_topology=True)
            pad_centers[side]=pad_shapes[side].centroid
    if trimmed.geom_type == 'Polygon' and trimmed.is_valid and all(trimmed.contains(p) for p in holes):
        outer = trimmed
    if neutral:
        # On pale metal, a white-background shadow can connect to the rim.
        # Estimate its thickness from the lower side arcs, then constrain only
        # the bottom arc. Do not clip endpieces, bridge or detected pad shapes.
        for lens in lenses:
            x0,y0,x1,y1 = lens.bounds
            lower = [(x,y) for x,y in outer.exterior.coords
                     if x0 < x < x1 and y > lens.centroid.y + (y1-y0)*.15]
            if len(lower) < 12:
                continue
            widths = [lens.boundary.distance(Point(x,y)) for x,y in lower]
            thickness = float(np.percentile(widths,25))
            if not 1 < thickness < 10:
                continue
            region = box(x0, y0+(y1-y0)*.65, x1, y1+30)
            bounded = outer.difference(region.difference(lens.buffer(thickness*1.3)))
            if (bounded.geom_type == 'Polygon' and bounded.is_valid and
                    bounded.area > outer.area*.995 and all(bounded.contains(p) for p in holes)):
                outer = bounded
    # Regularize threshold stair steps in vector space before the joint thin
    # rim topology check. Supersampling alone cannot remove raw fallback teeth.
    from scipy.ndimage import gaussian_filter1d
    smoothed = []
    for poly in [outer,*holes]:
        boundary = poly.exterior
        count = max(32,int(boundary.length))
        samples = np.array([boundary.interpolate(t).coords[0] for t in np.linspace(0,boundary.length,count,endpoint=False)])
        candidate = Polygon(gaussian_filter1d(samples,1.5,axis=0,mode='wrap')).simplify(.25)
        smoothed.append(candidate if candidate.is_valid and candidate.boundary.hausdorff_distance(boundary)<2 else poly)
    new_outer = smoothed[0]
    new_holes = [p.intersection(new_outer.buffer(-.6)) for p in smoothed[1:]]
    if all(p.geom_type=='Polygon' and p.is_valid for p in new_holes) and all(
            p.area>=original.area*.995 for p,original in zip(new_holes,holes)):
        outer,lenses,bridge = new_outer,new_holes[:2],new_holes[2] if len(new_holes)>2 else None
    refined = [p.simplify(.65,preserve_topology=True) for p in [outer,*lenses]+([bridge] if bridge is not None else [])]
    if all(refined[0].contains(p) for p in refined[1:]):
        outer,*rest = refined
        lenses,bridge = rest[:2],rest[2] if len(rest)>2 else None
    annotations = []
    for part,poly in [('front',outer),('lens-left',lenses[0]),('lens-right',lenses[1])]+([('bridge-opening',bridge)] if bridge is not None else []):
        annotations.append({'photo_id':str(photo.get('id','')), 'part':part,
                            'points':[{'x':float(x*scale_x/factor),'y':float(y*scale_y/factor)}
                                      for x,y in list(poly.exterior.coords)[:-1]],
                            'anchors':{},'approved':False,'source':'automatic-dark-double-bridge-v1' if bridge is not None else 'automatic-dark-single-bridge-v1'})
    if len(pad_centers)==2 and pad_centers['left'].x < pad_centers['right'].x:
        annotations[0]['anchors'] = {f'nose-pad-{side}':{'x':float(p.x*scale_x/factor),'y':float(p.y*scale_y/factor)}
                                     for side,p in pad_centers.items()}
        for side,pad in pad_shapes.items():
            annotations.append({'photo_id':str(photo.get('id','')),'part':f'nose-pad-{side}',
                'points':[{'x':float(x*scale_x/factor),'y':float(y*scale_y/factor)} for x,y in list(pad.exterior.coords)[:-1]],
                'anchors':{},'approved':False,'source':'automatic-visible-pad-silhouette-v1'})
    return {'annotations':annotations,
            'warnings':['Tinted lenses and the connected rim were traced separately. Review the rim, bridge and any pad suggestions before approval.'],
            'segmentation':{'method':'stable-dark-double-bridge' if bridge is not None else 'stable-dark-single-bridge','review_required':True}}


# Compatibility for existing diagnostic callers.
recover_dark_double_bridge = recover_dark_front
