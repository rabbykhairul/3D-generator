"""Experimental learned lens proposals plus bounded source-pixel refinement.

The model never supplies frame/arm silhouettes, approves geometry or invents a
rimless bridge. Existing complete contours are preserved. Missing full-rim
geometry requires an observed source contour.
"""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
import threading

import cv2
import numpy as np
from shapely.geometry import Polygon, box

ROOT = Path(__file__).resolve().parents[2]
INFERENCE_SLOT = threading.BoundedSemaphore(1)


def infer_lenses(image):
    """Keep PyTorch outside the app environment, with bounded request lifetime."""
    if not INFERENCE_SLOT.acquire(timeout=1):
        raise RuntimeError('The CPU lens worker is busy. Try analysis again shortly.')
    try:
        return _infer_lenses(image)
    finally:
        INFERENCE_SLOT.release()


def _infer_lenses(image):
    python = Path(os.environ.get('EYEWEAR_SEGMENTER_PYTHON', ROOT/'artifacts/segmenter-venv/bin/python'))
    manifest = Path(os.environ.get('EYEWEAR_SEGMENTER_MANIFEST', ROOT/'artifacts/segmenter-evaluation/medium.json'))
    if not python.is_file() or not manifest.is_file():
        raise RuntimeError('The isolated segmenter environment or local checkpoint manifest is missing. Run the segmenter setup first.')
    with tempfile.TemporaryDirectory(prefix='eyewear-lenses-') as directory:
        directory = Path(directory)
        image_path, output = directory/'analysis.png', directory/'lenses.npy'
        if not cv2.imwrite(str(image_path), image):
            raise RuntimeError('Could not prepare the local analysis image')
        started = time.perf_counter()
        run = subprocess.run([str(python), str(ROOT/'scripts/segmenter_infer.py'),
                              '--manifest',str(manifest),'--image',str(image_path),'--output',str(output)],
                             cwd=ROOT, capture_output=True, text=True, timeout=45)
        if run.returncode != 0 or not output.is_file():
            # Do not return local paths or dependency tracebacks through the API.
            raise RuntimeError('CPU lens inference failed; check the isolated environment and local checkpoints.')
        scores = np.load(output, allow_pickle=False)
        diagnostic = json.loads(run.stdout)
        diagnostic['request_seconds'] = round(time.perf_counter()-started,4)
    if (scores.shape != image.shape[:2] or not np.isfinite(scores).all() or
            scores.min() < 0 or scores.max() > 1):
        raise ValueError('The segmenter returned an invalid probability image')
    return scores, diagnostic


def _polygon(contour):
    if len(contour) < 3:
        return None
    points = cv2.approxPolyDP(contour, .7, True).reshape(-1,2)
    polygon = Polygon(points)
    return polygon if polygon.is_valid and polygon.area > 8 else None


def _pair(scores):
    """Reject extra optical instances, clipped shapes and implausible pairings."""
    height,width = scores.shape
    mask = np.uint8(scores >= .5)*255
    contours,_ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    polygons = [_polygon(c) for c in contours if cv2.contourArea(c) >= height*width*.01]
    if len(polygons) != 2 or any(p is None for p in polygons):
        return None
    left,right = sorted(polygons,key=lambda p:p.centroid.x)
    x0,y0,x1,y1 = left.union(right).bounds
    span = x1-x0
    if (not left.disjoint(right) or min(left.area,right.area)/max(left.area,right.area) < .45 or
            not .15 < (y1-y0)/span < .8 or
            abs(left.centroid.y-right.centroid.y) > (y1-y0)*.25 or
            x0 <= 1 or y0 <= 1 or x1 >= width-2 or y1 >= height-2):
        return None
    return left,right


def _sample(image, coordinates):
    return cv2.remap(image, coordinates[...,0].astype(np.float32),
                     coordinates[...,1].astype(np.float32),cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)


def refine_lens(image, proposal, support=None):
    """Try decreasing edge-search bands when a wider band folds the curve.

    High-resolution masks can sit beside mounting hardware or a second bevel
    edge. A wider search may have strong contrast yet invalid topology. Every
    narrower attempt retains the same source-support and geometry checks.
    """
    radius = max(3, min(24, round(max(image.shape[:2])/256*3)))
    attempts = []
    for band in dict.fromkeys((radius, max(3, radius//2), max(3, radius//4), 3)):
        candidate, detail = _refine_lens_at_radius(image, proposal, support, band)
        attempts.append(detail)
        if candidate is not None:
            if len(attempts) > 1:
                detail = dict(detail, wider_search_rejections=attempts[:-1])
            return candidate, detail
    return None, dict(attempts[-1], search_attempts=attempts)


def _refine_lens_at_radius(image, proposal, support, radius):
    """Snap a coarse closed boundary to nearby colour edges without templates.

    A cyclic dynamic program discourages jumping between rim and arm edges.
    Low evidence, large changes or invalid shapes are withheld, not completed
    with a convex hull or a generic lens.
    """
    perimeter = proposal.length
    count = max(32,min(600,int(perimeter/4)))
    points = np.array([proposal.exterior.interpolate(i*perimeter/count).coords[0] for i in range(count)])
    tangent = np.roll(points,-1,axis=0)-np.roll(points,1,axis=0)
    length = np.linalg.norm(tangent,axis=1)
    if np.any(length < 1e-4):
        return None,dict(reason='degenerate-proposal')
    normals = np.stack((-tangent[:,1],tangent[:,0]),axis=1)/length[:,None]
    offsets = np.arange(-radius,radius+1,dtype=float)
    coordinates = points[:,None,:]+normals[:,None,:]*offsets[None,:,None]
    lab = cv2.cvtColor(cv2.GaussianBlur(image,(3,3),.6),cv2.COLOR_BGR2LAB).astype(np.float32)
    inside = _sample(lab,coordinates-normals[:,None,:]*1.5)
    outside = _sample(lab,coordinates+normals[:,None,:]*1.5)
    strength = np.linalg.norm(outside-inside,axis=2)
    costs = -np.minimum(strength,80)/20 + .6*np.abs(offsets)[None,:]/radius
    if support is not None:
        # A full-rim optical boundary must have photographed support on both
        # sides of this sampling band. The high-contrast rim/background edge
        # must not win over the actual inner lens edge.
        support_mask=np.zeros(image.shape[:2],np.float32)
        cv2.fillPoly(support_mask,[np.rint(support.exterior.coords).astype(np.int32)],1)
        valid=(_sample(support_mask,coordinates-normals[:,None,:]*3)>.99)&(
               _sample(support_mask,coordinates+normals[:,None,:]*3)>.99)
        if not np.all(valid.any(axis=1)):
            return None,dict(reason='proposal-outside-observed-support',search_radius_px=radius)
        costs=np.where(valid,costs,1e6)
    transition = .18*np.abs(offsets[:,None]-offsets[None,:])
    best = None
    # Evaluate several starts and score the closing transition too.
    for first in np.argsort(costs[0])[:3]:
        previous = np.full(len(offsets),np.inf)
        previous[first] = costs[0,first]
        parents = np.zeros((count,len(offsets)),dtype=np.int16)
        for i in range(1,count):
            paths = previous[:,None]+transition
            parents[i] = np.argmin(paths,axis=0)
            previous = costs[i]+np.min(paths,axis=0)
        closing = previous+transition[:,first]
        last = int(np.argmin(closing))
        score = closing[last]
        chosen = np.zeros(count,dtype=int)
        chosen[-1] = last
        for i in range(count-1,0,-1):
            chosen[i-1] = parents[i,chosen[i]]
        if best is None or score < best[0]:
            best = score,chosen
    chosen = best[1]
    refined = points+normals*offsets[chosen,None]
    support = float(np.mean(strength[np.arange(count),chosen] >= 8))
    candidate = Polygon(refined).simplify(.6,preserve_topology=True)
    diagnostic = dict(edge_support_fraction=round(support,4),search_radius_px=radius,
                      max_displacement_px=float(np.max(np.abs(offsets[chosen]))))
    if not candidate.is_valid:
        repaired=candidate.buffer(0)
        # Dense foundation-model masks can contain a touching, zero-area pixel
        # filament. Remove only that tiny junction; never convexify a lens or
        # accept a repair that changes appreciable optical area.
        if (repaired.geom_type=='Polygon' and abs(repaired.area-candidate.area)<=2 and
                repaired.hausdorff_distance(candidate)<=radius/2+1):
            candidate=repaired
            diagnostic['pixel_junction_repaired']=True
    if (support < .7 or not candidate.is_valid or
            not .8 <= candidate.area/proposal.area <= 1.2 or
            candidate.intersection(proposal).area/candidate.union(proposal).area < .8):
        return None,dict(diagnostic,reason='insufficient-source-evidence-or-unstable-shape')
    return candidate,dict(diagnostic,reason='source-edge-refinement')


def outer_lens_envelope(image,proposal):
    """Resample the outer optical boundary across internal mask obstructions.

    Retain measured radial shape rather than fitting a circle or convex hull.
    Only narrow inward cuts are regularized. Area/topology and source-edge
    checks still gate the result; appreciable completion is labelled inferred.
    """
    from scipy.ndimage import median_filter
    from .smooth_curve import smooth_closed_curve
    h,w=image.shape[:2]
    center=np.array(proposal.centroid.coords[0])
    from shapely.geometry import Point
    if not proposal.contains(Point(center)):
        return None,dict(reason='optical-center-outside-proposal')
    radius=float(np.max(np.linalg.norm(np.array(proposal.exterior.coords)-center,axis=1)))+2
    count=720
    angles=np.linspace(0,2*np.pi,count,endpoint=False)
    units=np.column_stack((np.cos(angles),np.sin(angles)))
    distances=np.arange(0,radius,.5)
    coordinates=center+units[:,None,:]*distances[None,:,None]
    mask=np.zeros((h,w),np.uint8)
    cv2.fillPoly(mask,[np.rint(proposal.exterior.coords).astype(np.int32)],1)
    samples=cv2.remap(mask,coordinates[:,:,0].astype(np.float32),coordinates[:,:,1].astype(np.float32),
                      cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT)>0
    if not samples.any(axis=1).all():
        return None,dict(reason='incomplete-radial-envelope')
    far=samples.shape[1]-1-np.argmax(samples[:,::-1],axis=1)
    radii=distances[far]
    # A 15-degree neighbourhood closes short mounting/arm intrusions while
    # retaining broad photographed concavities. It is still an assumption.
    radii=np.maximum(radii,median_filter(radii,31,mode='wrap'))
    candidate=Polygon(center+units*radii[:,None]).simplify(.7,preserve_topology=True)
    candidate=smooth_closed_curve(candidate,max(h,w)/1080)
    added=candidate.difference(proposal).area/proposal.area
    removed=proposal.difference(candidate).area/proposal.area
    if not candidate.is_valid or added>.15 or removed>.01:
        return None,dict(reason='excessive-optical-completion',added_area_fraction=round(added,4))
    return candidate,dict(reason='outer-optical-envelope',added_area_fraction=round(added,4),
                          removed_area_fraction=round(removed,4),occlusion_completion=added>.01)


def _annotation(photo,part,polygon,scale_x,scale_y,source):
    return dict(photo_id=str(photo['id']),part=part,approved=False,anchors={},source=source,
                points=[dict(x=float(x*scale_x),y=float(y*scale_y)) for x,y in list(polygon.exterior.coords)[:-1]])


def _faint_source_front(image, lenses):
    """Recover connected pale hardware on a verified quiet light backdrop.

    Lens proposals select source contours; they are never painted into this
    mask or joined by a synthetic bridge. Require agreement across thresholds.
    """
    from .full_rim_recovery import _largest_valid
    h,w = image.shape[:2]
    n=max(1,min(h,w)//40)
    lab=cv2.cvtColor(image,cv2.COLOR_BGR2LAB).astype(np.float32)
    border=np.concatenate((lab[:n].reshape(-1,3),lab[-n:].reshape(-1,3),
                           lab[:,:n].reshape(-1,3),lab[:,-n:].reshape(-1,3)))
    background=np.median(border,axis=0)
    noise=float(np.percentile(np.linalg.norm(border-background,axis=1),90))
    if background[0]<225 or noise>4:
        return None
    distance=np.linalg.norm(lab-background,axis=2)
    candidates=[]
    for threshold in (max(3,noise*3),max(5,noise*3+2),max(8,noise*3+5)):
        mask=np.uint8(distance>threshold)*255
        contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        options=[]
        for contour in contours:
            if cv2.contourArea(contour)<sum(p.area for p in lenses):
                continue
            outer=_largest_valid(contour)
            if outer is None or outer.area>h*w*.8:
                continue
            x0,y0,x1,y1=outer.bounds
            if x0<=1 or y0<=1 or x1>=w-2 or y1>=h-2:
                continue
            # Raster boundaries have fractional-pixel intersections. Permit
            # only a tiny optical-area discrepancy, never a missing lens.
            if any(p.difference(outer.buffer(-.65)).area>p.area*.001 for p in lenses):
                continue
            # Exclude broad contact shadows/foreign background components.
            material=outer.difference(lenses[0].union(lenses[1]))
            nearby=lenses[0].union(lenses[1]).buffer(max(5,(y1-y0)*.08))
            bridge=Polygon([(lenses[0].bounds[2],y0),(lenses[1].bounds[0],y0),
                            (lenses[1].bounds[0],y1),(lenses[0].bounds[2],y1)])
            support_region=nearby.union(bridge)
            for side,lens in enumerate(lenses):
                lx0,ly0,lx1,ly1=lens.bounds
                reach=(lx1-lx0)*.14
                # Endpieces can extend beyond a thin rim. Limit that extra
                # source support to the upper outer mounting neighborhood.
                mount=box(lx0-reach,ly0,lx0+reach,ly0+(ly1-ly0)*.55) if side==0 else box(
                          lx1-reach,ly0,lx1+reach,ly0+(ly1-ly0)*.55)
                support_region=support_region.union(mount)
            if material.difference(support_region).area>material.area*.02:
                continue
            options.append(outer)
        if options:
            candidates.append(min(options,key=lambda p:p.area))
    if len(candidates)<2:
        return None
    stable=[p for p in candidates if sum(p.intersection(q).area/p.union(q).area>.985
                                        for q in candidates)>=2]
    return max(stable,key=lambda p:p.area) if stable else None


def _apply_lens_hybrid(photo,appearance_image,detection_image,result,scale_x,scale_y,*,model='legacy',precomputed=None):
    """Return a reviewable result with explicit inference/use/fallback diagnostics."""
    result = deepcopy(result)
    lens_source=f'automatic-{model}-lens-refined-v1' if model in {'sam2','sam3','eyewear-trained'} else 'automatic-hybrid-lens-refined-v1'
    diagnostics = dict(requested_backend=model if model in {'sam2','sam3','eyewear-trained'} else 'hybrid',
                       proposal_backend={'sam2':'sam2.1-hiera-small','sam3':'sam3','eyewear-trained':'eyewear-components-lraspp'}.get(model,'glasses-segmenter-medium-lenses'),
                       inference_ran=False,proposals_used=False,experimental=True)
    result['hybrid_detection'] = diagnostics
    try:
        if precomputed is not None:
            scores,runtime=precomputed
        elif model=='sam3':
            from .sam3_detection import infer_lenses as infer_concepts
            scores,runtime=infer_concepts(appearance_image)
        elif model=='sam2':
            from .foundation_detection import infer_lenses as infer_foundation
            scores,runtime=infer_foundation(appearance_image)
        else:
            scores, runtime = infer_lenses(appearance_image)
    except (RuntimeError,ValueError,OSError,subprocess.TimeoutExpired,json.JSONDecodeError) as error:
        diagnostics.update(status='unavailable',reason=str(error) if isinstance(error,RuntimeError) else 'Lens inference was unavailable or exceeded its time limit.')
        result.setdefault('warnings',[]).append('Experimental hybrid unavailable: '+diagnostics['reason']+' Existing contour detection was retained.')
        return result
    diagnostics.update(inference_ran=True,runtime=runtime)
    pair = _pair(scores)
    if pair is None:
        diagnostics.update(status='rejected',reason='The model did not establish two usable separate lens proposals.')
        return result
    baseline = {a['part']:a for a in result.get('annotations',[])}
    if {'lens-left','lens-right'} <= baseline.keys() and ({'front'} <= baseline.keys() or 'bridge-hardware' in baseline):
        # A successful proposal does not justify replacing source-refined geometry.
        diagnostics.update(status='retained',reason='Existing complete source contours were retained; model proposals were not substituted for them.')
        return result
    support=None
    if 'front' in baseline:
        candidate_support=Polygon([(p['x']/scale_x,p['y']/scale_y) for p in baseline['front']['points']])
        if candidate_support.is_valid:
            support=candidate_support
    refined, details, lens_sources = [], [], []
    for proposal in pair:
        envelope,envelope_detail=outer_lens_envelope(appearance_image,proposal)
        lens=None
        if envelope is not None and support is not None and envelope_detail['occlusion_completion']:
            lens,detail=refine_lens(appearance_image,envelope,support=support)
            if lens is not None:
                detail=dict(detail,envelope=envelope_detail)
        if lens is None:
            lens,detail=refine_lens(appearance_image,proposal,support=support)
        if lens is None:
            original_detail=detail
            if envelope is not None and (support is not None or not envelope_detail['occlusion_completion']):
                lens,detail=refine_lens(appearance_image,envelope,support=support)
                detail=dict(detail,envelope=envelope_detail,original_refinement=original_detail)
        if lens is None and support is not None and detail.get('reason') == 'proposal-outside-observed-support':
            # A threshold front is an unverified suggestion, not ground truth.
            # Recover independently supported lenses when that front excludes
            # their photographed edges. A new joint source audit below still
            # decides whether any frame contour may be retained.
            support_failure = detail
            lens, detail = refine_lens(appearance_image, proposal)
            if lens is None and envelope is not None and not envelope_detail['occlusion_completion']:
                lens, detail = refine_lens(appearance_image, envelope)
                detail = dict(detail, envelope=envelope_detail)
            detail = dict(detail, rejected_front_support=support_failure)
            diagnostics['baseline_front_support_rejected'] = True
        details.append(detail)
        if lens is None:
            diagnostics.update(status='rejected',reason='Lens proposals could not be refined into valid source-supported boundaries.',refinement=details)
            return result
        refined.append(lens)
        completed=detail.get('envelope',{}).get('occlusion_completion',False)
        lens_sources.append(lens_source.replace('lens-refined','lens-occlusion-completion') if completed else lens_source)
    left,right = refined
    if not left.disjoint(right):
        diagnostics.update(status='rejected',reason='Refined optical boundaries intersect.',refinement=details)
        return result
    def partial(reason):
        # Optical components remain usable even when support is not found.
        # Do not retain a conflicting old outer contour around the new lenses.
        extras=[a for a in result.get('annotations',[]) if a['part'] not in
                {'front','lens-left','lens-right','bridge-opening'}]
        annotations=[_annotation(photo,part,p,scale_x,scale_y,source)
                     for part,p,source in zip(('lens-left','lens-right'),refined,lens_sources)]+extras
        from .anatomy import audit_photo
        if audit_photo(photo,annotations,width=appearance_image.shape[1]*scale_x,
                       height=appearance_image.shape[0]*scale_y)['blocking']:
            diagnostics.update(status='rejected',reason='Optical component audit failed.',refinement=details)
            return result
        from .construction_classifier import suggest_construction
        diagnostics.update(status='partial',proposals_used=True,reason=reason,refinement=details,
                           baseline_warnings=list(result.get('warnings',[])))
        result.update(annotations=annotations,construction=suggest_construction(detection_image,annotations,scale_x,scale_y))
        result['warnings']=[w for w in result.get('warnings',[]) if not any(s in w.lower() for s in
            ('two lens openings were not confidently found','boundary is not safe to reconstruct',
             'suggested a convex boundary','completed a boundary across touching','lens-window candidate',
             'faint frame and two lens openings are not stable'))]
        result['warnings'].append('Both source-refined lens boundaries were retained. Frame support is unresolved; missing rims do not establish rimless construction. These outlines remain unapproved.')
        return result
    outer = None
    if 'front' in baseline:
        outer = Polygon([(p['x']/scale_x,p['y']/scale_y) for p in baseline['front']['points']])
    else:
        from .engine import _foreground_mask
        from .full_rim_recovery import _largest_valid
        mask,_ = _foreground_mask(detection_image)
        contours,hierarchy = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for i,contour in enumerate(contours):
            if hierarchy[0,i,3] != -1:
                continue
            # Reuse the source contour's bounded single-pixel-junction repair;
            # do not convexify the observed front or bridge openings.
            polygon = _largest_valid(contour)
            if polygon is not None and all(polygon.contains(lens) for lens in refined):
                candidates.append(polygon)
        if candidates:
            outer = min(candidates,key=lambda p:p.area)
    if outer is None or not outer.is_valid or not all(outer.contains(p) for p in refined):
        faint=_faint_source_front(detection_image,refined)
        if faint is not None:
            clipped=[p.intersection(faint.buffer(-.65)) for p in refined]
            if all(p.geom_type=='Polygon' and p.is_valid and
                   p.area/q.area>=.999 for p,q in zip(clipped,refined)):
                outer=faint
                refined=clipped
                left,right=refined
                diagnostics['frame_recovery']='lens-guided-source-contrast-consensus'
    # Reject clipping, huge background contours, intersecting openings and tiny
    # clearances. Frame material must be present along the observed outer edge.
    height,width = detection_image.shape[:2]
    if outer is None or not outer.is_valid or not all(outer.contains(p) for p in refined):
        return partial('Both lenses have source boundary evidence; a connected frame or bridge could not be established.')
    x0,y0,x1,y1 = outer.bounds
    frame = Polygon(outer.exterior.coords,[p.exterior.coords for p in refined])
    if (x0<=1 or y0<=1 or x1>=width-2 or y1>=height-2 or
            outer.area>width*height*.8 or not frame.is_valid or frame.area/outer.area < .025 or
            any(outer.boundary.distance(p.boundary)<.5 for p in refined)):
        return partial('Both lenses have source boundary evidence; frame topology, clipping or rim clearance failed.')
    extras = [a for a in result.get('annotations',[]) if a['part'] not in {'front','lens-left','lens-right'}]
    if 'bridge-opening' in baseline:
        opening = Polygon([(p['x']/scale_x,p['y']/scale_y) for p in baseline['bridge-opening']['points']])
        if not opening.is_valid or not outer.contains(opening) or any(opening.intersects(p) for p in refined):
            return partial('Both lenses have source boundary evidence; the bridge opening conflicts with frame recovery.')
    annotations = [_annotation(photo,'front',outer,scale_x,scale_y,'automatic-hybrid-source-front-v1'),
                   _annotation(photo,'lens-left',left,scale_x,scale_y,lens_sources[0]),
                   _annotation(photo,'lens-right',right,scale_x,scale_y,lens_sources[1]),*extras]
    from .anatomy import audit_photo
    if audit_photo(photo,annotations,width=width*scale_x,height=height*scale_y)['blocking']:
        diagnostics.update(status='rejected',reason='The source geometry audit rejected the hybrid candidate.',refinement=details)
        return result
    from .construction_classifier import suggest_construction
    result.update(annotations=annotations,construction=suggest_construction(detection_image,annotations,scale_x,scale_y))
    diagnostics.update(status='applied',proposals_used=True,reason='Refined lens proposals and an observed source front passed geometry checks.',refinement=details)
    # Keep the failed baseline evidence in diagnostics, not as contradictory
    # instructions alongside a new candidate that has passed the geometry audit.
    diagnostics['baseline_warnings'] = list(result.get('warnings',[]))
    superseded = ('could not confidently isolate a frame contour','two lens openings were not confidently found',
                  'the front suggestion has an ambiguous boundary','boundary is not safe to reconstruct',
                  'faint frame and two lens openings are not stable',
                  'suggested a convex boundary','completed a boundary across touching',
                  'lens-window candidate','lens-window candidates')
    result['warnings'] = [warning for warning in result.get('warnings',[]) if
                          not any(phrase in warning.lower() for phrase in superseded)]
    result.setdefault('warnings',[]).append('Experimental pretrained lens proposals were refined against source pixels. Review lenses, frame construction and attachments; hidden geometry and physical fit remain unverified.')
    if any('occlusion-completion' in source for source in lens_sources):
        result['warnings'].append('Lens interiors were completed across photographed obstructions using the source-supported outer boundary. These sections are inferred; check unusual concave designs and mounting points before approval.')
    return result


def apply_hybrid(photo,appearance_image,detection_image,result,scale_x,scale_y,*,model='legacy'):
    result=_apply_lens_hybrid(photo,appearance_image,detection_image,result,scale_x,scale_y,model=model)
    if model=='sam3' and result.get('hybrid_detection',{}).get('inference_ran'):
        from .sam3_components import apply_front_components
        result=apply_front_components(photo,appearance_image,result,scale_x,scale_y)
    return result
