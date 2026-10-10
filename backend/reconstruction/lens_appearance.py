"""Conservative flat lens tint from paired photographed lens interiors.

This estimates appearance on white, not physical optical transmission. It does
not bake photographed rear arms, logos or reflections onto the lens mesh.
"""
import cv2
import numpy as np


def measured_dark_edge_width(photo, points):
    image=cv2.imread(str(photo['path']),cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    mask=np.zeros(image.shape,np.uint8)
    cv2.fillPoly(mask,[np.rint(points).astype(np.int32)],1)
    distance=cv2.distanceTransform(mask,cv2.DIST_L2,5)
    scale=max(image.shape)/1080
    profile=[]
    for n in range(7):
        samples=image[(distance>n*scale)&(distance<=(n+1)*scale)]
        if len(samples)<20:
            return None
        profile.append(float(np.median(samples)))
    dark=[i for i in range(1,5) if profile[i]<min(profile[5:])*.8]
    return (max(dark)+1)*scale if dark else None


def infer_dark_lens_material(photo, outlines, *, include_clear=False):
    image = cv2.imread(str(photo['path']))
    if image is None or len(outlines)!=2:
        return None
    colors = []
    for points in outlines.values():
        mask = np.zeros(image.shape[:2],np.uint8)
        cv2.fillPoly(mask,[np.rint(points).astype(np.int32)],255)
        radius = max(3,round(np.sqrt(np.count_nonzero(mask))*.08))
        mask = cv2.erode(mask,np.ones((radius,radius),np.uint8))
        pixels = image[mask>0].astype(float)
        if len(pixels)<100:
            return None
        light = pixels.mean(axis=1)
        # Trim darker photographed arms and brightest highlights.
        pixels = pixels[(light>=np.percentile(light,35))&(light<=np.percentile(light,80))]
        color = np.median(pixels,axis=0)
        if not 25 < color.mean() <= (255 if include_clear else 190) or np.percentile(np.linalg.norm(pixels-color,axis=1),80)>35:
            return None
        colors.append(color)
    if np.linalg.norm(colors[0]-colors[1])>25:
        return None
    observed = np.mean(colors,axis=0)[::-1]/255
    if observed.mean() > 190/255:
        # Clear lenses are mostly the photographed background. Only a neutral
        # white backdrop supports this estimate; never convert a coloured
        # backdrop into a product tint.
        border = np.concatenate((image[0], image[-1], image[:,0], image[:,-1]))
        backdrop = np.median(border,axis=0)[::-1]/255
        if backdrop.min() < .9 or np.ptp(backdrop) > .04:
            return None
    # Alpha blending happens in linear light, not in photographic sRGB.
    linear = np.where(observed<=.04045,observed/12.92,((observed+.055)/1.055)**2.4)
    # Use the largest neutral transmission that leaves a small nonnegative
    # tint contribution. This is a white-background appearance approximation,
    # not a measurement of optical transmission.
    transmission = float(np.min(linear)*.85)
    opacity = max(.02, 1-transmission)
    tint = np.clip((linear-transmission)/opacity,0,1)
    color = np.where(tint<=.0031308,tint*12.92,1.055*tint**(1/2.4)-.055)
    return {'lens_color':'#'+''.join(f'{round(v*255):02x}' for v in color),
            'lens_opacity':opacity,'method':'paired-interior-linear-white-composite',
            'optical_transmission_measured':False}


def measured_clear_edge_width(photo, points):
    """A narrow optical border needs contrast along most of its perimeter."""
    image=cv2.imread(str(photo['path']),cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    mask=np.zeros(image.shape,np.uint8)
    cv2.fillPoly(mask,[np.rint(points).astype(np.int32)],1)
    distance=cv2.distanceTransform(mask,cv2.DIST_L2,5)
    scale=max(image.shape)/1080
    radius=max(1,round(3*scale))
    kernel=np.ones((radius*2+1,radius*2+1),np.uint8)
    band=cv2.dilate(mask,kernel)-cv2.erode(mask,kernel)
    edge=image[band>0]
    inner=image[(distance>6*scale)&(distance<=10*scale)]
    if len(edge)<100 or len(inner)<100 or np.median(inner)<210:
        return None
    return 2*scale if np.median(inner)-np.median(edge)>6 else None
