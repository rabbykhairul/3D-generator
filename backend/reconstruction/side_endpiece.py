"""Visible chromatic endpiece continuation; not hidden hinge recovery."""
import cv2
import numpy as np
from shapely.geometry import Polygon, LineString
from .side_extraction import _runs


def extend_visible_endpiece(image, annotation, scale_x, scale_y):
    anchors=annotation.get('anchors',{})
    if not {'hinge','tip'} <= anchors.keys():
        return annotation
    root=anchors['hinge'];tip=anchors['tip']
    x=float(root['x']/scale_x);y=float(root['y']/scale_y)
    direction=1 if root['x']>tip['x'] else -1
    span=abs(root['x']-tip['x'])/scale_x
    h,w=image.shape[:2]
    polygon=Polygon([(p['x']/scale_x,p['y']/scale_y) for p in annotation['points']])
    section=polygon.intersection(LineString([(x,0),(x,h)]))
    if section.is_empty:
        return annotation
    top,bottom=section.bounds[1],section.bounds[3]
    height=bottom-top
    if height<3:
        return annotation
    lab=cv2.cvtColor(image,cv2.COLOR_BGR2LAB).astype(float)
    # Neutral clear rims cannot establish this continuation without a separate
    # component detector. Restrict to a coherent visible coloured endpiece.
    chroma=np.linalg.norm(lab[:,:,1:]-128,axis=2)
    mask=(chroma>5)&(lab[:,:,0]<245)
    samples=[(x,top,bottom)];gap=0
    for step in range(1,max(2,round(span*.12))):
        ix=round(x)+direction*step
        if not 0<=ix<w:
            break
        runs=_runs(mask[:,ix],.08)
        runs=[r for r in runs if r[0]<=y+height*.55 and r[-1]>=y-height*.55]
        if not runs:
            gap+=1
            if gap>max(3,round(span*.007)):
                break
            continue
        run=min(runs,key=lambda r:abs((r[0]+r[-1])/2-y))
        if len(run)>height*1.6 or abs((run[0]+run[-1])/2-y)>height*.3:
            break
        gap=0
        samples.append((ix,float(run[0])-.4,float(run[-1])+.4))
    length=abs(samples[-1][0]-x)
    if length<span*.025 or len(samples)<length*.75:
        return annotation
    # Hitting the search cap provides no physical attachment boundary.
    if length>=round(span*.12)-2:
        return annotation
    extension=Polygon([(px,u) for px,u,_ in samples]+[(px,l) for px,_,l in reversed(samples)])
    combined=polygon.union(extension.buffer(.15)).buffer(0)
    if combined.geom_type!='Polygon' or not combined.is_valid:
        return annotation
    end=samples[-1]
    return dict(annotation,points=[{'x':float(px*scale_x),'y':float(py*scale_y)}
                                  for px,py in combined.simplify(.35).exterior.coords[:-1]],
                anchors={**anchors,'frame-attachment':{'x':float(end[0]*scale_x),
                                                     'y':float((end[1]+end[2])/2*scale_y)}})
