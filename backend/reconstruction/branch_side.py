"""Stable outer-arm extraction on quiet catalogue backgrounds.

Keep the long, upper distal branch through the overlap; do not switch to
the closest run's center when a transparent shaft splits into printed edges.
This is image-conditioned contour inference, not semantic AI segmentation.
"""
import cv2
import numpy as np
from shapely.geometry import Polygon, box
from .side_extraction import extract_side, _runs


def recover_branch_side(image, photo, scale_x, scale_y):
    result = _recover_branch_side(image, photo, scale_x, scale_y)
    if result and result.get('annotations'):
        return result
    height, width = image.shape[:2]
    if min(height,width) >= 320 and max(height,width) != 1080:
        factor = 1080/max(height,width)
        work = cv2.resize(image, (round(width*factor),round(height*factor)),
                          interpolation=cv2.INTER_AREA if factor<1 else cv2.INTER_CUBIC)
        normalized = _recover_branch_side(work, photo, scale_x*width/work.shape[1],
                                          scale_y*height/work.shape[0])
        if normalized and normalized.get('annotations'):
            return normalized
        neutral = _recover_branch_side(work, photo, scale_x*width/work.shape[1],
                                       scale_y*height/work.shape[0], neutral=True)
    else:
        neutral = _recover_branch_side(image, photo, scale_x, scale_y, neutral=True)
    return neutral if neutral and neutral.get('annotations') else result


def _recover_branch_side(image, photo, scale_x, scale_y, neutral=False):
    height, width = image.shape[:2]
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    band = max(3, min(height, width)//30)
    border = np.concatenate([lab[:band].reshape(-1,3), lab[-band:].reshape(-1,3),
                             lab[:,:band].reshape(-1,3), lab[:,-band:].reshape(-1,3)])
    background = np.median(border, axis=0)
    noise = np.percentile(np.linalg.norm(border-background, axis=1),90)
    if background[0] < 225 or noise >= 7:
        return None
    distance = np.linalg.norm(lab-background,axis=2)
    chroma = np.linalg.norm(lab[:,:,1:]-background[1:],axis=2)
    if neutral:
        foreground = distance > 20
        if not foreground.any() or np.median(chroma[foreground]) > 3:
            return None
        distance = cv2.GaussianBlur(distance, (0,0), max(.6,min(height,width)/1080.))
    candidates=[]
    for threshold in ((12,16,20) if neutral else (8,12,16)):
        mask=np.uint8((distance>max(threshold,noise*2.5)) &
                      (True if neutral else ((chroma>3)|(distance>40))))*255
        # Close narrow *vertical* internal highlight gaps, not the horizontal
        # gap between separate arms. Scale with the analysis raster.
        kernel=max(3,round(min(height,width)*(.03 if neutral else .012)))|1
        mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((kernel,1),np.uint8))
        ys,xs=np.where(mask)
        if len(xs)<12:
            return None
        lo,hi=int(xs.min()),int(xs.max())
        span=hi-lo
        if span < width*.55:
            return None
        band=max(2,round(span*.18))
        frame_right=np.max(np.count_nonzero(mask[:,hi-band:hi+1],axis=0)) > np.max(np.count_nonzero(mask[:,lo:lo+band],axis=0))
        # Require two separated distal arms, with the upper branch connected
        # to the outermost tip. This does not assume every photo has this pose.
        samples=np.linspace(lo+span*.23,lo+span*.42,16) if frame_right else np.linspace(hi-span*.42,hi-span*.23,16)
        split_count=sum(len(_runs(mask[:,round(x)]))>=2 for x in samples)
        if split_count<4:
            return None
        result=extract_side(photo,mask,scale_x,scale_y,start_at_frame_boundary=True,
                            frame_start_fraction=.12,upper_branch=True,repair_short_overlap=True)
        recovered_seed = not result['annotations']
        if not result['annotations']:
            # Oblique fronts (including dark lenses with pale metal temples)
            # can cover the usual seed. Try nearby seeds for both chromatic
            # and neutral material; still require the same outer tip and
            # consensus across all three contrast masks.
            for fraction in (.14,.16,.18,.20,.22):
                result=extract_side(photo,mask,scale_x,scale_y,start_at_frame_boundary=True,
                    frame_start_fraction=fraction,upper_branch=True,repair_short_overlap=True)
                if result['annotations']:
                    break
        if not result['annotations']:
            return {'annotations':[], 'warnings':['Two arms overlap, but a continuous outer-arm boundary could not be verified. Do not join different arm tips; correct the branch in Advanced.']}
        ann=result['annotations'][0]
        expected_tip=lo if frame_right else hi
        if abs(ann['anchors']['tip']['x']/scale_x-expected_tip)>max(3,span*.025):
            # This pose does not have the long upper distal branch this route
            # verifies (e.g. the lower branch is the longer visible arm).
            # Leave it to the existing general tracker, not this topology.
            return None
        polygon=Polygon([(p['x'],p['y']) for p in ann['points']])
        if recovered_seed and not neutral:
            original_polygon = polygon
            # Faint edge fragments can leave short inward threshold teeth.
            # Bound regularization by both displacement and area; never fill
            # a missing shaft or merge the separated rear branch.
            radius = span*min(scale_x,scale_y)*.008
            smooth = polygon.buffer(radius).buffer(-radius).simplify(.5*min(scale_x,scale_y))
            if (smooth.geom_type == 'Polygon' and smooth.is_valid and
                    smooth.area < polygon.area*1.035 and
                    smooth.boundary.hausdorff_distance(polygon.boundary) < radius):
                polygon = smooth
                ann['points']=[{'x':float(x),'y':float(y)} for x,y in polygon.exterior.coords[:-1]]
            # Fit only a demonstrably straight upper shaft, never the curved
            # earpiece or hinge. Lower-envelope residuals ignore inward gaps.
            from shapely.geometry import LineString
            tip_x,hinge_x = ann['anchors']['tip']['x'],ann['anchors']['hinge']['x']
            samples = np.linspace(tip_x+(hinge_x-tip_x)*.4, tip_x+(hinge_x-tip_x)*.94,80)
            tops = np.array([polygon.intersection(LineString([(x,0),(x,height*scale_y)])).bounds[1] for x in samples])
            fit = np.polyfit(samples,tops,1)
            residual = tops-np.polyval(fit,samples)
            fit[1] += np.percentile(residual,15)
            error = tops-np.polyval(fit,samples)
            if np.max(np.abs(error)) < radius*1.2 and np.mean(np.abs(error)<2*scale_y) > .6:
                points = np.array(polygon.exterior.coords[:-1])
                selected = ((points[:,0]>=min(samples))&(points[:,0]<=max(samples))&
                            (np.abs(points[:,1]-np.polyval(fit,points[:,0]))<radius*1.5))
                points[selected,1] = np.polyval(fit,points[selected,0])
                straight = Polygon(points).simplify(.3*min(scale_x,scale_y),preserve_topology=True)
                if straight.is_valid and straight.area < polygon.area*1.035:
                    polygon = straight
                    ann['points']=[{'x':float(x),'y':float(y)} for x,y in polygon.exterior.coords[:-1]]
            # Keep exact cap/root coordinates so the attachment anchor remains
            # on its boundary after buffer arithmetic and simplification.
            a,b=sorted((tip_x,hinge_x))
            middle=box(a+(b-a)*.15,0,b-(b-a)*.06,height*scale_y)
            restored=polygon.intersection(middle).union(original_polygon.difference(middle))
            if restored.geom_type=='Polygon' and restored.is_valid:
                polygon=restored
                ann['points']=[{'x':float(x),'y':float(y)} for x,y in polygon.exterior.coords[:-1]]
        if neutral:
            # Close short inward notches from bright reflections without
            # straightening genuine sleeve-width transitions. Keep the tip
            # and hinge coordinates unchanged, and bound area/displacement.
            tip_x, hinge_x = ann['anchors']['tip']['x'], ann['anchors']['hinge']['x']
            a,b = sorted((tip_x,hinge_x))
            radius = (b-a)*.004
            delta = hinge_x-tip_x
            m0,m1 = sorted((tip_x+delta*.18,tip_x+delta*.94))
            middle = box(m0,0,m1,height*scale_y)
            from shapely.geometry import LineString
            from scipy.ndimage import grey_opening
            s0,s1 = sorted((tip_x+delta*.4,tip_x+delta*.94))
            xs = np.linspace(s0,s1,240)
            tops = np.array([polygon.intersection(LineString([(x,0),(x,height*scale_y)])).bounds[1] for x in xs])
            # An opening removes short downward excursions only; a sustained
            # width change or curved earpiece is not fitted to a straight line.
            envelope = grey_opening(tops,size=13,mode='nearest')
            envelope = np.where(tops-envelope<radius*3,envelope,tops)
            # Union a thin sampled strip instead of moving sparse vertices:
            # moving vertices at vertical seams can make a self-intersection.
            strip = Polygon([*zip(xs,envelope),*zip(xs[::-1],(tops+.1*scale_y)[::-1])]).buffer(0)
            repaired = polygon.union(strip)
            if (repaired.geom_type=='Polygon' and repaired.is_valid and repaired.area < polygon.area*1.015 and
                    repaired.boundary.hausdorff_distance(polygon.boundary)<radius*3):
                polygon = repaired
                ann['points']=[{'x':float(x),'y':float(y)} for x,y in polygon.exterior.coords[:-1]]
            closed = polygon.buffer(radius).buffer(-radius)
            regular = closed.intersection(middle).union(polygon.difference(middle))
            if (regular.geom_type=='Polygon' and regular.is_valid and
                    regular.area < polygon.area*1.015 and
                    regular.boundary.hausdorff_distance(polygon.boundary) < radius):
                polygon = regular.simplify(.25*min(scale_x,scale_y),preserve_topology=True)
                ann['points']=[{'x':float(x),'y':float(y)} for x,y in polygon.exterior.coords[:-1]]
            # A clear rounded tip may retain its perimeter but lose a bright
            # interior strip. Complete only the short distal cap, never the
            # curved arm or crossing. Large missing tips remain unverified.
            tip = ann['anchors']['tip']['x']
            hinge = ann['anchors']['hinge']['x']
            reach = abs(hinge-tip)*.12
            region = box(tip-1, 0, tip+reach, height*scale_y) if tip<hinge else box(tip-reach,0,tip+1,height*scale_y)
            cap = polygon.intersection(region)
            # Include disconnected, strong perimeter pixels beside the cap;
            # the tracker may otherwise keep an interior metal strip only.
            yy,xx=np.where(distance>40)
            px,py=xx*scale_x,yy*scale_y
            near=((px>=region.bounds[0])&(px<=region.bounds[2])&
                  (py>=cap.bounds[1]-reach*.15)&(py<=cap.bounds[3]+reach*.15))
            from shapely.geometry import MultiPoint
            observed=MultiPoint(np.column_stack((px[near],py[near])))
            completed = polygon.union(cap.union(observed).convex_hull)
            if completed.geom_type == 'Polygon' and completed.area < polygon.area*1.08:
                polygon = completed
                ann['points']=[{'x':float(x),'y':float(y)} for x,y in polygon.exterior.coords[:-1]]
        candidates.append((result,polygon))
    result,polygon=candidates[1]
    if any(polygon.intersection(other).area/max(polygon.union(other).area,1e-9)<.86 for _,other in candidates):
        return {'annotations':[], 'warnings':['Transparent side boundaries change with contrast. A stable complete arm could not be confirmed; review the outline instead of approving internal highlights.']}
    from .side_endpiece import extend_visible_endpiece
    result['annotations'][0]=extend_visible_endpiece(image,result['annotations'][0],scale_x,scale_y)
    result['annotations'][0]['source']='single-visible-arm-branch-consensus-v1'
    result['warnings']=['One continuous outer arm was traced through the overlap at three contrast levels. Check the hinge and outer tip before approval; the second arm is excluded.',
                        'Transparent material boundaries and hidden cross-section remain approximate.']
    result['segmentation']={'method':'outer-branch-contrast-consensus','review_required':True}
    if 'frame-attachment' in result['annotations'][0]['anchors']:
        result['warnings'].append('The visible coloured endpiece continues beyond the arm seam to a proposed frame attachment. Review both anchors; attachment depth assumes a near-side view, and the hidden hinge mechanism is not recovered.')
    if neutral:
        result['annotations'][0]['source']='single-visible-arm-neutral-consensus-v1'
        result['segmentation']['method']='neutral-outer-branch-contrast-consensus'
    return result
