"""Partial inner-temple texture transfer, with explicit registration assumptions.

Use the separated distal branch, never the overlapping shaft/front. Register
tip-to-tip and local widths along the target centerline. This is a local strip
approximation, not recovered camera calibration or inferred hidden printing.
"""
import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter1d


def transfer_inner_tip(source_photo, evidence, target_photo, annotation, base_texture):
    source = cv2.imread(str(source_photo['path']))
    if not {'hinge','tip'} <= set(annotation.get('anchors',{})):
        return None, {'reason':'target_missing_anchors'}
    if source is None or len(evidence['points']) < 4:
        return None, {'reason': 'missing_source'}
    points = np.array([[p['x'],p['y']] for p in evidence['points']], np.float32)
    direction=evidence.get('root_direction_x',1 if points[:,0].mean()<source_photo['width']/2 else -1)
    start=np.maximum(0,np.floor(points.min(axis=0)-4).astype(int))
    end=np.minimum([source.shape[1],source.shape[0]],np.ceil(points.max(axis=0)+5).astype(int))
    source=source[start[1]:end[1],start[0]:end[0]]
    points-=start
    if source.size==0:
        return None, {'reason':'source_outside_image'}
    # Exclude edge-connected white background while retaining enclosed light
    # printing. Extend nearby object color across sampling seams, not shadow.
    from scipy.ndimage import binary_fill_holes, distance_transform_edt
    observed=np.zeros(source.shape[:2],np.uint8)
    cv2.fillPoly(observed,[np.rint(points).astype(np.int32)],255)
    background=np.median(np.concatenate([source[0],source[-1],source[:,0],source[:,-1]]),axis=0)
    foreground=np.linalg.norm(source.astype(float)-background,axis=2)>25
    safe=binary_fill_holes(foreground)&(observed>0)
    if safe.sum()<30:
        return None, {'reason':'insufficient_foreground_texture'}
    _,nearest=distance_transform_edt(~safe,return_indices=True)
    source=source[nearest[0],nearest[1]]
    center = points.mean(axis=0)
    _, _, axes = np.linalg.svd(points-center, full_matrices=False)
    axis = axes[0]
    # Distal tip is away from the selected arm's hinge (source photo pose).
    source_right = direction > 0
    if (axis[0] > 0) != source_right:
        axis = -axis
    normal = np.array([-axis[1],axis[0]])
    local = np.column_stack(((points-center)@axis,(points-center)@normal))
    offset = local.min(axis=0)-3
    local -= offset
    dimensions = np.ceil(local.max(axis=0)+4).astype(int)
    if dimensions[0] < 30 or dimensions[1] < 5:
        return None, {'reason':'insufficient_visible_surface'}
    mask = np.zeros((dimensions[1],dimensions[0]), np.uint8)
    cv2.fillPoly(mask,[np.rint(local).astype(np.int32)],255)
    xs = np.arange(mask.shape[1])
    upper = np.array([np.flatnonzero(mask[:,x])[0] if mask[:,x].any() else np.nan for x in xs])
    lower = np.array([np.flatnonzero(mask[:,x])[-1] if mask[:,x].any() else np.nan for x in xs])
    valid = np.isfinite(upper)&((lower-upper)>3)
    if valid.sum()<25:
        return None, {'reason':'insufficient_strip'}
    lo,hi = xs[valid][[0,-1]]
    # The last section touches the selected arm: exclude that overlap.
    hi = int(lo+(hi-lo)*.80)
    upper = np.interp(xs,xs[valid],upper[valid]); lower = np.interp(xs,xs[valid],lower[valid])
    source_width = float(np.median((lower-upper)[lo+5:hi]))
    target = np.array(base_texture.convert('RGBA')).copy()
    th,tw = target.shape[:2]
    factors = np.array([tw/target_photo['width'],th/target_photo['height']])
    polygon = np.array([[p['x'],p['y']] for p in annotation['points']])*factors
    target_mask = np.zeros((th,tw),np.uint8)
    cv2.fillPoly(target_mask,[np.rint(polygon).astype(np.int32)],255)
    hinge = annotation['anchors']['hinge']['x']*factors[0]
    tip = annotation['anchors']['tip']['x']*factors[0]
    columns = np.arange(max(0,int(polygon[:,0].min())),min(tw,int(polygon[:,0].max())+1))
    if tip>hinge:
        columns=columns[::-1]
    columns=np.array([x for x in columns if target_mask[:,x].any()])
    if len(columns)<30:
        return None, {'reason':'insufficient_target'}
    top=np.array([np.flatnonzero(target_mask[:,x])[0] for x in columns],float)
    bottom=np.array([np.flatnonzero(target_mask[:,x])[-1] for x in columns],float)
    middle=gaussian_filter1d((top+bottom)/2,2)
    distance=np.r_[0,np.cumsum(np.hypot(np.diff(columns),np.diff(middle)))]
    target_width=float(np.median((bottom-top)[max(3,len(columns)//20):max(10,len(columns)//5)]))
    ratio=source_width/max(target_width,1)
    map_x=np.full((th,tw),-1,np.float32);map_y=map_x.copy()
    weights=np.zeros((th,tw),np.float32)
    for i,x in enumerate(columns):
        station=lo+distance[i]*ratio
        if station>hi:
            break
        k=int(station)
        ys=np.arange(int(top[i]),int(bottom[i])+1)
        across=(ys-top[i])/max(bottom[i]-top[i],1)
        # Keep distal outline orientation; only the inner cap is assigned it.
        lateral=upper[k]+across*(lower[k]-upper[k])
        positions=center+np.outer(np.full(len(ys),station)+offset[0],axis)+np.outer(lateral+offset[1],normal)
        map_x[ys,x]=positions[:,0];map_y[ys,x]=positions[:,1]
        weights[ys,x]=min(1.,max(0.,(hi-station)/max((hi-lo)*.15,1)))
    remapped=cv2.remap(cv2.cvtColor(source,cv2.COLOR_BGR2RGBA),map_x,map_y,cv2.INTER_LINEAR)
    applied=(map_x>=0)&(target_mask>0)
    if applied.sum()<30:
        return None, {'reason':'no_registered_pixels'}
    blend=weights[applied,None]
    target[applied]=np.rint(remapped[applied]*blend+target[applied]*(1-blend)).astype(np.uint8)
    # UV seams sample beside the polygon; extend the registered cap by two
    # pixels rather than bleeding the white photographic background.
    dist,nearest=distance_transform_edt(~applied,return_indices=True)
    bleed=(dist<=2)&~applied
    target[bleed]=target[nearest[0][bleed],nearest[1][bleed]]
    return Image.fromarray(target), {'applied':True,'method':'tip-and-width local strip registration',
        'source_photo_id':source_photo['id'],'target_photo_id':target_photo['id'],
        'observed_fraction':float(applied.sum()/max((target_mask>0).sum(),1)),
        'assumption':'Opposite-arm inner face; local width sets approximate longitudinal scale. Hidden shaft appearance remains the existing inferred finish.'}
