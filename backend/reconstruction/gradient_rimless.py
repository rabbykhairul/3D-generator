"""CPU gradient optics with positive visible lens-mounted bridge evidence.

Closed edges, quiet exterior bands and intruding mounting leaves are required.
A missing outer frame alone never establishes rimless construction.
"""
import cv2
import numpy as np
from shapely.geometry import Polygon
from .cpu_metal import _wire_path
from .smooth_curve import smooth_detected_curve


def recover_gradient_rimless(image,photo,scale_x=1.,scale_y=1.):
    h,w=image.shape[:2]
    if min(h,w)<160:return None
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    border=np.concatenate((gray[:5].ravel(),gray[-5:].ravel(),gray[:,:5].ravel(),gray[:,-5:].ravel()))
    if np.median(border)<245 or np.percentile(255-border,90)>10:return None
    scale=max(h,w)/1080
    blurred=cv2.GaussianBlur(gray,(3,3),0)
    pairs=[]
    for low,high in ((5,15),(10,25),(12,40),(20,60)):
        edges=cv2.Canny(blurred,low,high)
        edges=cv2.morphologyEx(edges,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
        contours,hierarchy=cv2.findContours(edges,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is None:continue
        candidates=[]
        for i,c in enumerate(contours):
            if hierarchy[0][i][3]<0 or len(c)<12 or not .025*h*w<cv2.contourArea(c)<.3*h*w:continue
            p=Polygon(c[:,0,:])
            if not p.is_valid:
                repaired=p.buffer(0)
                if repaired.geom_type!='Polygon' or abs(repaired.area-p.area)>max(1,p.area*.001):continue
                p=repaired
            x0,y0,x1,y1=p.bounds
            if .6<(x1-x0)/(y1-y0)<2.8:candidates.append(p)
        if len(candidates)!=2:continue
        left,right=sorted(candidates,key=lambda p:p.centroid.x)
        lw=min(p.bounds[2]-p.bounds[0] for p in (left,right))
        if (left.disjoint(right) and min(left.area,right.area)/max(left.area,right.area)>.75 and
            abs(left.centroid.y-right.centroid.y)<lw*.1 and 0<right.bounds[0]-left.bounds[2]<lw*.65):
            pairs.append((low,high,[left,right]))
    if len(pairs)<3:return None
    lenses=pairs[len(pairs)//2][2]
    def agrees(a,b):
        # Attachment junctions can produce short raster excursions. Demand
        # agreement on 98% of both observed perimeters; retain review status.
        for first,second in ((a,b),(b,a)):
            distances=[second.boundary.distance(first.exterior.interpolate(t)) for t in
                       np.linspace(0,first.length,240,endpoint=False)]
            if np.percentile(distances,98)>max(3,scale*4):return False
        return True
    if sum(all(agrees(a,b) for a,b in zip(lenses,pair))
           for _,_,pair in pairs)<3:return None
    # A mounting leaf can interrupt a small part of the optical contour.
    # Accept a convex envelope only with positive edge evidence around it.
    edge_distance=cv2.distanceTransform(np.uint8(cv2.Canny(blurred,10,25)==0),cv2.DIST_L2,5)
    verified=[]
    for p in lenses:
        envelope=p.convex_hull
        xy=np.rint([envelope.exterior.interpolate(t).coords[0] for t in
                    np.linspace(0,envelope.length,400,endpoint=False)]).astype(int)
        if envelope.area>p.area*1.12 or np.mean(edge_distance[xy[:,1],xy[:,0]]<=3*scale)<.975:return None
        verified.append(envelope)
    lenses=verified
    chroma=image.max(2).astype(float)-image.min(2)
    mounts=[];exterior=[]
    bridge,bridge_info=_wire_path(image,*lenses)
    if bridge is None:return None
    for side,lens in zip(('left','right'),lenses):
        x0,y0,x1,y1=lens.bounds;lw=x1-x0;lh=y1-y0
        mask=np.zeros(gray.shape,np.uint8)
        cv2.fillPoly(mask,[np.rint(lens.exterior.coords).astype(np.int32)],1)
        outer=np.zeros(gray.shape,np.uint8);inner=outer.copy()
        cv2.fillPoly(outer,[np.rint(lens.buffer(14*scale).exterior.coords).astype(np.int32)],1)
        cv2.fillPoly(inner,[np.rint(lens.buffer(8*scale).exterior.coords).astype(np.int32)],1)
        band=(outer>0)&(inner==0)
        exterior.append(float(np.mean((gray[band]>245)&(chroma[band]<10))))
        # Positive attachment evidence: a bright neutral horizontal mounting
        # leaf intrudes into a chromatic optical interior at the nasal edge.
        xa,xb=(x1-lw*.16,x1) if side=='left' else (x0,x0+lw*.16)
        ya,yb=bridge.bounds[1]-8*scale,bridge.bounds[3]+8*scale
        region=np.zeros(gray.shape,np.uint8)
        cv2.rectangle(region,(max(0,int(xa)),max(0,int(ya))),
                      (min(w-1,int(xb)),min(h-1,int(yb))),1,-1)
        material=np.uint8((mask>0)&(region>0)&(gray>175)&(chroma<30))*255
        cs,_=cv2.findContours(material,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        options=[]
        for c in cs:
            bx,by,cw,ch=cv2.boundingRect(c)
            if cw>lw*.04 and cw/ch>1.5 and cv2.contourArea(c)>lw*lh*.0004:
                p=Polygon(c[:,0,:])
                if not p.is_valid:
                    repaired=p.buffer(0)
                    pieces=list(repaired.geoms) if hasattr(repaired,'geoms') else [repaired]
                    main=max((piece for piece in pieces if piece.geom_type=='Polygon'),
                             key=lambda piece:piece.area,default=None)
                    if main is None or main.area<repaired.area*.995:continue
                    p=main
                if p.is_valid:options.append(p)
        if len(options)!=1:return None
        mounts.append(options[0])
    if min(exterior)<.85:return None
    annotations=[]
    for side,p in zip(('left','right'),lenses):
        p=smooth_detected_curve(p,scale,max_displacement=1.5)
        annotations.append(dict(photo_id=str(photo['id']),part='lens-'+side,approved=False,anchors={},
            source='automatic-cpu-gradient-rimless-edges-v1',
            points=[dict(x=float(x*scale_x),y=float(y*scale_y)) for x,y in p.exterior.coords[:-1]]))
    # Keep separate observed leaf profiles. Filling their connecting span or
    # hidden mechanics would be an inference, not a measured hardware outline.
    hardware=[('bridge-hardware',bridge)]+[('bridge-mount-'+side,p) for side,p in zip(('left','right'),mounts)]
    for part,p in hardware:
        if part.startswith('bridge-mount'):continue  # evidence only; not an unsupported annotation part
        annotations.append(dict(photo_id=str(photo['id']),part=part,approved=False,anchors={},
            source='automatic-cpu-gradient-rimless-edges-v1',
            points=[dict(x=float(x*scale_x),y=float(y*scale_y)) for x,y in p.exterior.coords[:-1]]))
    return dict(annotations=annotations,
        construction=dict(category='rimless',confidence='medium',reason='Stable optical edges, quiet exterior bands and visible nasal mounting leaves inside both tinted lenses.'),
        warnings=['Gradient rimless CPU draft: review faint lower edges, source bridge and lens-mounted leaves. No outer frame was added. Mounting depth, hidden hardware and untraced components remain unresolved.'],
        segmentation=dict(method='cpu-gradient-rimless-source-edges',review_required=True,learned_model_active=False,
            stable_edge_settings=len(pairs),quiet_exterior_fractions=exterior,bridge=bridge_info,
            visible_nasal_mounts=2,complete_hardware_verified=False))
