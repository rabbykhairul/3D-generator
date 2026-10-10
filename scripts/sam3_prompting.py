"""Automatic visual prompts derived from the detected optical pair."""
import numpy as np


def component_boxes(lens_masks):
    bounds=[]
    for mask in lens_masks:
        y,x=np.where(mask)
        if len(x):bounds.append((int(x.min()),int(y.min()),int(x.max()),int(y.max())))
    if len(bounds)!=2:return []
    left,right=sorted(bounds)
    lw=min(left[2]-left[0],right[2]-right[0]);lh=min(left[3]-left[1],right[3]-right[1])
    if lw<40 or lh<30 or left[2]>=right[0]:return []
    top=min(left[1],right[1]);bottom=max(left[3],right[3])
    boxes=[('bridge',(left[2]-.12*lw,top-.05*lh,right[0]+.12*lw,top+.55*lh))]
    for edge in (left[0],right[2]):
        boxes.append(('hinge',(edge-.18*lw,top+.1*lh,edge+.18*lw,top+.6*lh)))
    return boxes


def normalized_box(bounds,width,height):
    x0,y0,x1,y1=bounds
    x0=max(0.,x0);y0=max(0.,y0);x1=min(width-1.,x1);y1=min(height-1.,y1)
    return [(x0+x1)/(2*width),(y0+y1)/(2*height),(x1-x0)/width,(y1-y0)/height]
