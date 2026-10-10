"""Conservative photographic detail patches, not inferred hardware geometry.

Only isolated dark components in upper outer lens corners or beside a traced
nose pad qualify. Rear arms, broad reflections and connected rim edges do not.
"""
import cv2
import numpy as np
from shapely.geometry import Polygon, Point


def visible_lens_details(photo, lenses, pads):
    image = cv2.imread(str(photo['path']), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return []
    scale = max(image.shape)/1080
    result = []
    for side, points in lenses.items():
        lens = Polygon(points)
        x0,y0,x1,y1 = lens.bounds
        mask = np.zeros(image.shape,np.uint8)
        cv2.fillPoly(mask,[np.rint(points).astype(np.int32)],255)
        radius = max(1,round(2*scale))
        interior = cv2.erode(mask,np.ones((radius*2+1,radius*2+1),np.uint8))
        pixels = image[interior>0]
        if len(pixels)<100:
            continue
        level = float(np.median(pixels))
        if not 25<level<190:
            continue
        dark = np.uint8((image<level*.75)&(interior>0))
        count,labels,stats,centers = cv2.connectedComponentsWithStats(dark,8)
        for index in range(1,count):
            x,y,w,h,area = stats[index]
            cx,cy = centers[index]
            if not max(4,20*scale*scale)<area<lens.area*.012:
                continue
            if w>(x1-x0)*.16 or h>(y1-y0)*.2:
                continue
            outer = cx<x0+(x1-x0)*.25 if side=='left' else cx>x1-(x1-x0)*.25
            printing = outer and y0+(y1-y0)*.06<cy<y0+(y1-y0)*.25
            pad = pads.get(side)
            mounting = (pad is not None and pad.distance(Point(cx,cy))<max(12*scale,(x1-x0)*.08)
                        and pad.bounds[1]-5*scale<cy<pad.bounds[3]+5*scale)
            if not (printing or mounting):
                continue
            contours,_ = cv2.findContours(np.uint8(labels==index),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            contour=max(contours,key=cv2.contourArea)
            if len(contour)<3:
                continue
            shape=Polygon(contour[:,0,:]).convex_hull.buffer(.6*scale).intersection(lens)
            if shape.geom_type!='Polygon' or not shape.is_valid or shape.area<1:
                continue
            result.append({'side':side,'kind':'pad-mount-appearance' if mounting else 'lens-print-appearance',
                           'polygon':shape,'source_photo_id':photo.get('id')})
    return result
