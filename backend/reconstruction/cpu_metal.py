"""CPU thin-metal proposals from lens interiors and source-supported wire paths.

No pretrained inference, SKU templates or image-result cache. Scope: a quiet
light background, two visible lenses and narrow connected wire support.
Clear interiors require independently stable closed source edges. Open or
occluded boundaries without that evidence fall back to other CPU contours.
"""
import cv2
import numpy as np
from PIL import Image
from shapely.geometry import Polygon, LineString
from shapely.ops import unary_union
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter


def _annotation(photo,part,poly,scale,anchors=None):
    return dict(photo_id=str(photo['id']),part=part,
        points=[dict(x=float(x*scale[0]),y=float(y*scale[1])) for x,y in poly.exterior.coords[:-1]],
        anchors=anchors or {},approved=False,source='automatic-cpu-metal-source-v1')


def _pair(gray):
    h,w=gray.shape;candidates=[]
    k=max(3,round(max(h,w)/200));k+=1-k%2
    for level in (140,170,195,220):
        mask=cv2.morphologyEx(np.uint8(gray<level)*255,cv2.MORPH_OPEN,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(k,k)))
        contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        choices=[]
        for c in contours:
            area=cv2.contourArea(c);x,y,cw,ch=cv2.boundingRect(c)
            if len(c)<12 or not .015*h*w<area<.35*h*w or not .55<cw/ch<2.8:continue
            p=Polygon(c[:,0,:]).buffer(0)
            if p.geom_type!='Polygon' or p.area/p.convex_hull.area<.85:continue
            choices.append(p)
        choices=sorted(choices,key=lambda p:p.area,reverse=True)[:4]
        for i,a in enumerate(choices):
            for b in choices[i+1:]:
                left,right=sorted((a,b),key=lambda p:p.centroid.x)
                lw=min(p.bounds[2]-p.bounds[0] for p in (left,right))
                gap=right.bounds[0]-left.bounds[2]
                if (left.disjoint(right) and min(a.area,b.area)/max(a.area,b.area)>.65 and
                    abs(a.centroid.y-b.centroid.y)<lw*.12 and 0<gap<lw*.65 and
                    min(a.bounds[3]-a.bounds[1],b.bounds[3]-b.bounds[1])>30):
                    candidates.append((level,[left,right]));break
    for i,(level,pair) in enumerate(candidates):
        agreeing=[(l,p) for l,p in candidates if l!=level and
                  all(a.boundary.hausdorff_distance(b.boundary)<min(a.bounds[2]-a.bounds[0],b.bounds[2]-b.bounds[0])*.015+1
                      for a,b in zip(pair,p))]
        if len(agreeing)>=1:return pair,[level,*[a[0] for a in agreeing]]
    return None,[]


def _wire_path(image, left, right, upper=False):
    """Column ridge in a bounded inter-lens corridor; broad shadows decline."""
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY).astype(float)
    chroma=image.max(axis=2).astype(float)-image.min(axis=2).astype(float)
    edge=cv2.magnitude(cv2.Sobel(gray.astype(np.float32),cv2.CV_32F,1,0),
                       cv2.Sobel(gray.astype(np.float32),cv2.CV_32F,0,1))
    if upper:
        x0=left.centroid.x;x1=right.centroid.x
        top=min(left.bounds[1],right.bounds[1]);lh=min(p.bounds[3]-p.bounds[1] for p in (left,right))
        lo=max(0,int(top-lh*.12));hi=min(image.shape[0],int(top+lh*.10))
    else:
        x0=left.bounds[2]-2;x1=right.bounds[0]+2
        lo=max(0,int(min(left.centroid.y,right.centroid.y)-min(left.bounds[3]-left.bounds[1],right.bounds[3]-right.bounds[1])*.40))
        hi=min(image.shape[0],int(max(left.centroid.y,right.centroid.y)+8))
    if x1-x0<8 or hi-lo<4:return None,None
    xs=np.linspace(x0,x1,max(32,int(x1-x0)))
    # Trace a continuous ridge through the corridor with bounded vertical
    # transitions. Per-column maxima alone jump onto pads, rim edges or arms.
    excluded=np.zeros(gray.shape,np.uint8)
    for lens in (left,right):
        cv2.fillPoly(excluded,[np.round(lens.exterior.coords).astype(np.int32)],1)
    columns=np.clip(np.round(xs).astype(int),0,image.shape[1]-1)
    merit=(chroma[lo:hi,columns]*2+edge[lo:hi,columns]*.12+
           np.maximum(0,245-gray[lo:hi,columns])*.15)
    merit[excluded[lo:hi,columns]>0]=-1000
    merit=np.minimum(merit,120.)
    step=max(1,round(min(left.bounds[2]-left.bounds[0],right.bounds[2]-right.bounds[0])*.006))
    costs=-merit[:,0];parents=np.zeros(merit.shape,np.int32)
    for col in range(1,len(columns)):
        choices=[]
        for dy in range(-step,step+1):
            shifted=np.full(len(costs),np.inf)
            if dy>=0:shifted[dy:]=costs[:len(costs)-dy]
            else:shifted[:dy]=costs[-dy:]
            choices.append(shifted+dy*dy*.8)
        choices=np.asarray(choices);which=np.argmin(choices,axis=0)
        parents[:,col]=np.arange(len(costs))-(which-step)
        costs=np.min(choices,axis=0)-merit[:,col]
    path=np.zeros(len(columns),np.int32);path[-1]=np.argmin(costs)
    for col in range(len(columns)-1,0,-1):path[col-1]=parents[path[col],col]
    ys=path.astype(float)+lo;supported=[];widths=[]
    for col,(ix,y) in enumerate(zip(columns,ys.astype(int))):
        values=merit[:,col];j=y-lo
        run=np.flatnonzero(values>=max(5,values[j]*.45))
        nearby=run[np.abs(run-j)<=8];width=max(1,len(nearby))
        supported.append(values[j]>8 and (chroma[y,ix]>5 or edge[y,ix]>65) and width<=14)
        widths.append(width)
    if np.mean(supported)<.86:return None,None
    ys=np.asarray(ys,float);window=min(len(ys)//2*2-1,max(15,int(len(ys)*.12)//2*2+1))
    smooth=savgol_filter(median_filter(ys,size=5),window,2)
    smooth=np.clip(smooth,ys-2,ys+2)
    if np.max(np.abs(np.diff(smooth)))>step+1:return None,None
    line=LineString(np.column_stack((xs,smooth)))
    radius=float(np.clip(np.median(widths)/2,1.,4.))
    poly=line.buffer(radius,cap_style=1,join_style=1).simplify(.5,preserve_topology=True)
    if upper:
        # A high bar needs source support where it reaches both rims. It is not
        # inferred merely from similar lens shapes or a central line.
        if max(poly.distance(p) for p in (left,right))>6:return None,None
    elif max(poly.distance(p) for p in (left,right))>4:return None,None
    return poly,{'source_supported_column_fraction':float(np.mean(supported)),
                 'median_visible_width_px':radius*2,'method':'local source ridge; hidden depth unknown'}


def _clear_edge_pair(gray):
    """Recover bright apertures inside narrow closed rims, not dark interiors.

    Keep the observed contour (including concave designs); neither a convex
    hull nor an ellipse may substitute for an interrupted source boundary.
    Nearby contrast levels must agree, and perimeter pixels must have edges.
    """
    h,w=gray.shape
    candidates=[]
    for threshold in (170,195,220,235,245,250):
        mask=np.uint8(gray<threshold)*255
        contours,hierarchy=cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is None:continue
        choices=[]
        for i,c in enumerate(contours):
            if hierarchy[0][i][3]<0 or len(c)<12:continue
            area=cv2.contourArea(c);x,y,cw,ch=cv2.boundingRect(c)
            if not .015*h*w<area<.35*h*w or not .55<cw/ch<2.8:continue
            p=Polygon(c[:,0,:])
            if not p.is_valid:continue
            choices.append(p)
        pairs=[]
        for i,a in enumerate(choices):
            for b in choices[i+1:]:
                left,right=sorted((a,b),key=lambda p:p.centroid.x)
                lw=min(p.bounds[2]-p.bounds[0] for p in (left,right))
                gap=right.bounds[0]-left.bounds[2]
                if (left.disjoint(right) and min(a.area,b.area)/max(a.area,b.area)>.65 and
                    abs(a.centroid.y-b.centroid.y)<lw*.12 and 0<gap<lw*.65 and
                    min(a.bounds[3]-a.bounds[1],b.bounds[3]-b.bounds[1])>30):
                    pairs.append([left,right])
        # More than one plausible pair is unresolved ownership, not a reason
        # to pick the largest unrelated background or arm opening.
        if len(pairs)==1:candidates.append((threshold,pairs[0]))
    for threshold,pair in candidates:
        lw=min(p.bounds[2]-p.bounds[0] for p in pair)
        agreeing=[(level,other) for level,other in candidates if level!=threshold and
                  all(a.boundary.hausdorff_distance(b.boundary)<lw*.012+1 for a,b in zip(pair,other))]
        if len(agreeing)<2:continue
        edge=cv2.Canny(gray,12,40)
        distance=cv2.distanceTransform(np.uint8(edge==0),cv2.DIST_L2,5)
        fractions=[]
        for lens in pair:
            xy=np.rint([lens.exterior.interpolate(t).coords[0] for t in
                        np.linspace(0,lens.length,240,endpoint=False)]).astype(int)
            fractions.append(float(np.mean(distance[xy[:,1],xy[:,0]]<=2)))
        if min(fractions)<.9:continue
        return pair,[threshold,*[a[0] for a in agreeing]]
    return None,[]


def detect_cpu_metal(photo):
    with Image.open(photo['path']) as handle:
        image=handle.convert('RGB');image.thumbnail((2000,2000))
        work=cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR)
    scale=np.array([photo['width']/work.shape[1],photo['height']/work.shape[0]])
    gray=cv2.cvtColor(work,cv2.COLOR_BGR2GRAY)
    border=np.concatenate((gray[:5].ravel(),gray[-5:].ravel(),gray[:,:5].ravel(),gray[:,-5:].ravel()))
    if np.median(border)<240 or np.percentile(255-border,90)>15:return None
    lenses,levels=_pair(gray)
    optical_method='stable-tinted-interiors'
    if lenses is None:
        lenses,levels=_clear_edge_pair(gray)
        optical_method='stable-clear-aperture-edges'
    if lenses is None:return None
    left,right=lenses
    from .smooth_curve import smooth_detected_curve
    lenses=[smooth_detected_curve(p,max(work.shape[:2])/1080,max_displacement=2.) for p in lenses]
    left,right=lenses
    # Inspect outside the lens edge along its outward normals at source scale.
    rim_widths=[];support=[]
    for lens in lenses:
        coords=np.array([lens.exterior.interpolate(t).coords[0] for t in np.linspace(0,lens.length,180,endpoint=False)])
        tangent=np.roll(coords,-1,axis=0)-np.roll(coords,1,axis=0)
        normal=np.column_stack((tangent[:,1],-tangent[:,0]));normal/=np.maximum(np.linalg.norm(normal,axis=1)[:,None],1e-6)
        if np.mean(np.sum(normal*(coords-np.array(lens.centroid.coords[0])),axis=1))<0:normal=-normal
        widths=[];good=[]
        for point,n in zip(coords,normal):
            xy=np.rint(point+n*np.arange(1,25)[:,None]).astype(int)
            xy[:,0]=np.clip(xy[:,0],0,work.shape[1]-1);xy[:,1]=np.clip(xy[:,1],0,work.shape[0]-1)
            rgb=work[xy[:,1],xy[:,0]].astype(float);chroma=rgb.max(axis=1)-rgb.min(axis=1)
            material=((chroma>7)&(rgb.min(axis=1)<245))|(rgb.mean(axis=1)<205)
            indices=np.flatnonzero(material)
            width=int(indices[-1]+1) if len(indices) else 0
            widths.append(width);good.append(0<width<=max(5,(lens.bounds[2]-lens.bounds[0])*.035))
        fraction=float(np.mean(good));support.append(fraction)
        rim_widths.append(float(np.median([v for v,g in zip(widths,good) if g])) if any(good) else 0.)
    if min(support)<.65:return None
    bridge,bridge_info=_wire_path(work,left,right)
    if bridge is None:return None
    bar,bar_info=_wire_path(work,left,right,True)
    rim=max(1.5,float(np.median(rim_widths)))
    hinges=[]
    for side,lens in zip(('left','right'),lenses):
        lw=lens.bounds[2]-lens.bounds[0];lh=lens.bounds[3]-lens.bounds[1]
        edge_x=lens.bounds[0] if side=='left' else lens.bounds[2]
        x0=max(0,int(edge_x-lw*.14 if side=='left' else edge_x-2))
        x1=min(work.shape[1],int(edge_x+2 if side=='left' else edge_x+lw*.14))
        y0=max(0,int(lens.bounds[1]+lh*.12));y1=min(work.shape[0],int(lens.bounds[1]+lh*.48))
        crop=gray[y0:y1,x0:x1]
        mask=cv2.morphologyEx(np.uint8(crop<245)*255,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
        rim_exclusion=np.zeros(mask.shape,np.uint8)
        cv2.fillPoly(rim_exclusion,[np.round(np.asarray(lens.buffer(rim).exterior.coords)-[x0,y0]).astype(np.int32)],1)
        mask[rim_exclusion>0]=0
        contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        choices=[]
        for contour in contours:
            if len(contour)<4:continue
            shape=Polygon(contour[:,0,:]+[x0,y0]).buffer(0)
            if shape.geom_type!='Polygon':continue
            bx,by,ex,ey=shape.bounds
            if (shape.area>lw*lh*.001 and ex-bx>lw*.035 and ey-by<lh*.18 and
                    shape.distance(lens)<rim+4):choices.append(shape)
        if len(choices)==1:hinges.append((side,choices[0].simplify(.5,preserve_topology=True)))
    components=[p.buffer(rim) for p in lenses]+[bridge]
    if bar is not None:components.append(bar)
    components.extend(p.buffer(2.) for _,p in hinges)
    union=unary_union(components).buffer(.5)
    if union.geom_type!='Polygon':return None
    # The envelope is used to establish scale and ownership. Thin-metal meshing
    # uses separate wire loops and observed bridge/bar, not a filled bridge slab.
    outer=Polygon(union.exterior.coords).simplify(.5,preserve_topology=True)
    if not all(outer.contains(p) for p in lenses):return None
    anchors={f'hinge-{side}':dict(x=float(p.centroid.x*scale[0]),y=float(p.centroid.y*scale[1])) for side,p in hinges}
    anns=[_annotation(photo,'front',outer,scale,anchors)]
    anns.extend(_annotation(photo,'lens-'+side,p,scale) for side,p in zip(('left','right'),lenses))
    anns.append(_annotation(photo,'bridge-hardware',bridge,scale))
    if bar is not None:anns.append(_annotation(photo,'front-support',bar,scale))
    anns.extend(_annotation(photo,'hinge-'+side,p,scale) for side,p in hinges)
    # Hinge endpoints need independently observed contours; no generic hinges
    # or nose pads are inserted by this detector.
    return {'annotations':anns,'warnings':['CPU thin-metal draft: review wire boundaries, bridge and any upper bar. Surface connectivity is observed; hidden depth, hinge mechanics and material optics remain inferred.'],
        'construction':{'category':'thin-metal','confidence':'medium','reason':'Two stable optical boundaries, narrow rim evidence and a source-supported connecting bridge.'},
        'segmentation':{'method':'cpu-thin-metal-native-ridges','optical_method':optical_method,'learned_model_active':False,'levels':levels,
            'rim_support_fractions':support,'rim_widths_px':rim_widths,'bridge':bridge_info,'upper_bar':bar_info,
            'no_template_or_result_cache':True,'review_required':True}}
