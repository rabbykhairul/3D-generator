"""Explainable construction suggestions from detected image evidence.

These are reviewable visual hypotheses, not material identification or a
trained universal classifier. Missing boundaries must never imply rimless.
"""
import cv2
import numpy as np


def suggest_construction(image, annotations, scale_x=1., scale_y=1.):
    def unknown(reason):
        return {'category':None,'confidence':'low','reason':reason,
                'alternatives':['opaque-full-rim','thin-metal','transparent-full-rim',
                                'semi-rimless','rimless','sports-shield']}
    parts={a.get('part'):a for a in annotations}
    if 'shield' in parts:
        return {'category':'sports-shield','confidence':'medium',
                'reason':'A single reviewed shield boundary is available; confirm that it is one continuous lens.',
                'alternatives':['opaque-full-rim']}
    if not all(part in parts for part in ('front','lens-left','lens-right')):
        return unknown('There is not enough boundary evidence to determine construction. Missing rims are not evidence of rimless construction.')
    height,width=image.shape[:2]
    def mask(part):
        points=np.array([[p['x']/scale_x,p['y']/scale_y] for p in parts[part]['points']],np.float32)
        output=np.zeros((height,width),np.uint8)
        cv2.fillPoly(output,[np.round(points).astype(np.int32)],255)
        return output
    outer=mask('front');left=mask('lens-left');right=mask('lens-right')
    solid=(outer>0)&(left==0)&(right==0)
    if 'bridge-opening' in parts: solid &= mask('bridge-opening')==0
    count=np.count_nonzero(solid)
    if count<12: return unknown('The inferred frame is too thin to classify reliably at this resolution.')
    lab=cv2.cvtColor(image,cv2.COLOR_BGR2LAB)
    pixels=lab[solid]
    # Ignore the faintest quarter of pixels, which can be antialiasing or a
    # residual white background band at an inferred silhouette boundary.
    selected=pixels[pixels[:,0]<=np.percentile(pixels[:,0],75)]
    light=float(np.median(selected[:,0]))
    fraction=count/max(np.count_nonzero(outer),1)
    diagnostics={'frame_area_fraction':round(float(fraction),4),'median_frame_lightness':round(light,1)}
    if light>170:
        return {'category':'transparent-full-rim','confidence':'medium',
                'reason':'Two enclosed lens openings and pale frame pixels suggest a translucent full rim. A pale opaque finish can look similar; confirm the material.',
                'alternatives':['opaque-full-rim','thin-metal'],'diagnostics':diagnostics}
    if fraction<.14:
        return {'category':'thin-metal','confidence':'medium',
                'reason':'Both lenses have narrow continuous rims. Thin metal is suggested, but a photo alone cannot distinguish every thin plastic frame.',
                'alternatives':['opaque-full-rim'],'diagnostics':diagnostics}
    return {'category':'opaque-full-rim','confidence':'medium',
            'reason':'Two enclosed lens openings and substantial contrasting rim material suggest an opaque full rim.',
            'alternatives':['transparent-full-rim','thin-metal'],'diagnostics':diagnostics}
