"""Source-pixel mounting profiles next to approved rimless lens boundaries.

The horizontal front leaf can be observed even when the rear arm touches it
in the photograph. This is a shallow extrusion, not recovered hinge mechanics.
"""
import cv2
import numpy as np
from shapely.geometry import Polygon


def visible_mounts(photo, lenses):
    image = cv2.imread(str(photo['path']), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return {}
    result = {}
    for side, points in lenses.items():
        x0,y0,x1,y1 = Polygon(points).bounds
        w,h = x1-x0,y1-y0
        outer = x0 if side=='left' else x1
        lo,hi = (outer-.12*w, outer+.16*w) if side=='left' else (outer-.16*w, outer+.12*w)
        xa,xb = max(0,int(lo)), min(image.shape[1],int(np.ceil(hi)))
        ya,yb = max(0,int(y0+.10*h)), min(image.shape[0],int(y0+.48*h))
        if xb-xa<12 or yb-ya<8:
            continue
        patch = image[ya:yb,xa:xb]
        # Shadows and faint lens borders do not qualify as metal hardware.
        dark = np.uint8(patch<160)
        rows = dark.sum(axis=1)
        peak = int(np.argmax(rows))
        if rows[peak] < .08*w:
            continue
        half = max(4,round(.045*h))
        top,bottom = max(0,peak-half), min(len(rows),peak+half+1)
        strip = np.zeros_like(dark)
        strip[top:bottom] = dark[top:bottom]
        strip = cv2.morphologyEx(strip,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
        contours,_ = cv2.findContours(strip,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for contour in contours:
            x,y,cw,ch = cv2.boundingRect(contour)
            if not .08*w<cw<.29*w or not 2<ch<.13*h or cw/ch<2.2:
                continue
            if not xa+x <= outer <= xa+x+cw:
                continue
            contour = contour[:,0,:].astype(float)+[xa,ya]
            shape = Polygon(contour)
            # OpenCV can return a touching pixel junction as a repeated
            # vertex. Remove that zero-area junction without completing edges.
            if not shape.is_valid:
                repaired = shape.buffer(0)
                if repaired.geom_type=='Polygon' and abs(repaired.area-shape.area)<1:
                    shape = repaired
            if shape.is_valid and shape.area>20 and shape.area/(cw*ch)>.35:
                candidates.append(shape)
        if len(candidates)!=1:
            continue
        shape = candidates[0].buffer(.4)
        # Seat the rear connector under the actual front leaf, rather than
        # whichever lens vertex happens to be furthest to the outside.
        ys = np.where(strip[:,max(0,min(xb-xa-1,round(outer-xa)))]>0)[0]
        if not len(ys):
            continue
        result[side] = {'polygon':shape, 'anchor':np.array([outer,float(np.median(ys)+ya)]),
                        'source_photo_id':photo.get('id'), 'depth_measured':False,
                        'method':'source-contrast-horizontal-mount-profile', 'depth_mm':1.2}
    return result
