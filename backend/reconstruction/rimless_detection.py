"""Independent lens-edge recovery for light-background rimless fronts.

Requires paired stable faint closed boundaries and a separate contrasting
central mount. Never interprets missing frame evidence alone as rimless.
"""
import cv2
import numpy as np
from shapely.geometry import Polygon, box
from shapely.ops import unary_union
from .smooth_curve import smooth_closed_curve


def recover_rimless_front(image, photo, scale_x=1., scale_y=1.):
    h,w=image.shape[:2]
    if min(h,w)<80:return None
    factor=1080/max(h,w)
    work=cv2.resize(image,(round(w*factor),round(h*factor)),interpolation=cv2.INTER_AREA if factor<1 else cv2.INTER_CUBIC)
    gray=cv2.cvtColor(work,cv2.COLOR_BGR2GRAY)
    h,w=gray.shape
    border=np.concatenate([gray[:8].ravel(),gray[-8:].ravel(),gray[:,:8].ravel(),gray[:,-8:].ravel()])
    if np.median(border)<248 or np.percentile(255-border,90)>8:return None
    candidates=[]
    candidate_thresholds=[]
    # Downsampling dilutes a one-pixel clear lens edge into its background.
    # First prefer the established contrast levels; retry weaker levels only
    # when those fail. Three stable closed perimeters are still mandatory.
    for threshold in (230,235,240,245,248,250,252):
        if threshold == 248 and len(candidates) >= 3:
            break
        mask=np.uint8(gray<threshold)*255
        mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
        contours,hierarchy=cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is None:continue
        holes=[]
        for i,c in enumerate(contours):
            if hierarchy[0][i][3]<0 or cv2.contourArea(c)<h*w*.025:continue
            p=Polygon(c[:,0,:]).convex_hull
            x0,y0,x1,y1=p.bounds
            if not p.is_valid or not .65<(x1-x0)/(y1-y0)<2.8:continue
            holes.append(p)
        if len(holes)!=2:continue
        left,right=sorted(holes,key=lambda p:p.centroid.x)
        if left.intersects(right) or min(left.area,right.area)/max(left.area,right.area)<.8:continue
        if abs(left.centroid.y-right.centroid.y)>h*.025:continue
        if not left.centroid.x<w/2<right.centroid.x:continue
        candidates.append([left,right])
        candidate_thresholds.append(threshold)
    if len(candidates)<3:return None
    lenses=candidates[len(candidates)//2]
    stability_limit = max(5., factor*2.)
    if any(a.boundary.hausdorff_distance(b.boundary)>stability_limit for pair in candidates for a,b in zip(lenses,pair)):return None
    # Check narrow pale lens edges, rather than a dark full rim or dark brows.
    # Ignore short attachment interruptions; require supported perimeter pixels.
    support_threshold = max(240, max(candidate_thresholds))
    edge_dist=cv2.distanceTransform(np.uint8(gray>=support_threshold),cv2.DIST_L2,5)
    for lens in lenses:
        coords=np.array([lens.exterior.interpolate(t).coords[0] for t in np.linspace(0,lens.length,240,endpoint=False)])
        xy=np.rint(coords).astype(int)
        if np.mean(edge_dist[xy[:,1],xy[:,0]]<max(3.,factor*1.5))<.82:return None
        samples=[]
        for x,y in xy:
            samples.append(np.min(gray[max(0,y-2):y+3,max(0,x-2):x+3]))
        if np.mean(np.array(samples)<170)>.2:return None
    left,right=lenses
    # A contrasting central bridge is positive construction evidence. Use the
    # entire observed connected hardware component, not a generic bridge tube.
    dark=cv2.morphologyEx(np.uint8(gray<150),cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    count,labels,stats,centers=cv2.connectedComponentsWithStats(dark,8)
    center=(left.bounds[2]+right.bounds[0])/2
    gap=right.bounds[0]-left.bounds[2]
    options=[]
    for i in range(1,count):
        x,y,bw,bh,area=stats[i]
        if (x<center<x+bw and gap*.65<bw<gap*2.5 and
                min(left.bounds[1],right.bounds[1])<y<left.centroid.y and
                40<area<min(left.area,right.area)*.1):
            options.append(i)
    if len(options)!=1:return None
    contours,_=cv2.findContours(np.uint8(labels==options[0]),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    raw=max(contours,key=cv2.contourArea)
    hardware=Polygon(raw[:,0,:]).buffer(0)
    if hardware.geom_type!='Polygon' or not hardware.is_valid:return None
    hardware=hardware.simplify(.4,preserve_topology=True)
    # Separate pads by subtracting the independently established lens edges
    # and metal support. Require agreement at three faint-edge thresholds.
    pad_candidates={side:[] for side in ('left','right')}
    lens_width=min(p.bounds[2]-p.bounds[0] for p in lenses)
    lens_height=min(p.bounds[3]-p.bounds[1] for p in lenses)
    region=box(left.bounds[2]-lens_width*.06,hardware.bounds[1]+lens_height*.15,
               right.bounds[0]+lens_width*.06,min(p.bounds[3] for p in lenses)-lens_height*.1)
    excluded=unary_union([p.buffer(3) for p in lenses]+[hardware.buffer(3)])
    for threshold in (235,240,245):
        mask=cv2.morphologyEx(np.uint8(gray<threshold),cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
        contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        if not contours:continue
        silhouette=Polygon(max(contours,key=cv2.contourArea)[:,0,:]).buffer(0)
        remaining=silhouette.intersection(region).difference(excluded)
        pieces=list(remaining.geoms) if hasattr(remaining,'geoms') else [remaining]
        for side,lens in zip(('left','right'),lenses):
            choices=[p for p in pieces if p.geom_type=='Polygon' and
                     lens.area*.005<p.area<lens.area*.04 and
                     (p.centroid.x<center)==(side=='left') and
                     p.distance(region.boundary)>1]
            if len(choices)==1:pad_candidates[side].append(choices[0])
    pads=[]
    for side,choices in pad_candidates.items():
        if len(choices)==3 and all(choices[1].intersection(p).area/choices[1].union(p).area>.75 for p in choices):
            pads.append((f'nose-pad-{side}',smooth_closed_curve(Polygon(choices[1].exterior.coords))))
    anns=[]
    for part,p in [(f'lens-{side}',smooth_closed_curve(lens)) for side,lens in zip(('left','right'),lenses)]+[('bridge-hardware',hardware)]+pads:
        anns.append({'photo_id':str(photo.get('id','')),'part':part,
            'points':[{'x':float(x*scale_x/factor),'y':float(y*scale_y/factor)}for x,y in p.exterior.coords[:-1]],
            'anchors':{},'approved':False,'source':'automatic-rimless-paired-edge-v1'})
    warnings=['Rimless draft: review both lens boundaries, separate bridge/mount silhouette and any pad suggestions. Hidden hardware depth is not measured; untraced pads remain omitted and endpieces remain approximate.']
    missing_pads = sorted({'nose-pad-left','nose-pad-right'}-{part for part,_ in pads})
    if missing_pads:
        warnings.append('Could not verify ' + ', '.join(missing_pads) +
                        '. Review these regions or supply a nasal close-up; no replacement pads were added.')
    return {'annotations':anns,
        'warnings':warnings,
        'segmentation':{'method':'stable-independent-rimless-lenses','review_required':True,
                        'contrast_thresholds':candidate_thresholds,
                        'max_source_boundary_disagreement_px':float(stability_limit/factor)},
        'construction':{'category':'rimless','confidence':'medium','reason':'Two stable pale lens perimeters and a separate contrasting central bridge/mount; no continuous dark rim.'}}
