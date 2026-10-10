"""Retain separated rear-arm pixels as appearance evidence, never geometry.

The lower branch in this catalogue pose is a candidate inner surface of the
opposite arm. Its foreshortened visible section is not a calibrated UV map.
"""
import cv2
import numpy as np


def clean_appearance_mask(mask, contrast):
    """Reject a weak attached shadow only beside a substantially stronger arm.

    A pale/transparent arm retains its low threshold. Enclosed light printing
    is still handled as texture by the caller, not cut into geometry.
    """
    values=contrast[mask>0]
    if not len(values):
        return mask
    threshold=max(28.,min(110.,float(np.percentile(values,60))*.60))
    strong=np.uint8((mask>0)&(contrast>=threshold))*255
    count,labels,stats,_=cv2.connectedComponentsWithStats(strong)
    if count<2:
        return mask
    component=1+np.argmax(stats[1:,cv2.CC_STAT_AREA])
    selected=np.uint8(labels==component)*255
    # Refuse destructive cleanup of a mixed-material or largely pale arm.
    if np.count_nonzero(selected)<np.count_nonzero(mask)*.75:
        return mask
    # A contact shadow can have a dark one-pixel connection even after the
    # contrast check. Remove only sub-width branches, retaining >=90% of the
    # visible component. Fill printing for this operation, then restore it.
    from scipy.ndimage import binary_fill_holes
    solid=np.uint8(binary_fill_holes(selected>0))*255
    x,y,width,height=cv2.boundingRect(solid)
    thickness=np.count_nonzero(solid)/max(np.hypot(width,height),1)
    radius=max(1,min(3,round(thickness*.18)))
    opened=cv2.morphologyEx(solid,cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(radius*2+1,radius*2+1)))
    if np.count_nonzero(opened)>=np.count_nonzero(solid)*.90:
        selected=cv2.bitwise_and(selected,opened)
    return selected


def opposite_arm_evidence(image, annotation, scale_x=1., scale_y=1.):
    h, w = image.shape[:2]
    points = np.array([[p['x']/scale_x, p['y']/scale_y]
                       for p in annotation['points']], dtype=np.float32)
    selected = np.zeros((h, w), np.uint8)
    cv2.fillPoly(selected, [np.rint(points).astype(np.int32)], 255)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(float)
    border = np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])
    background = np.median(border, axis=0)
    if background[0] < 225 or np.percentile(np.linalg.norm(border-background, axis=1), 90) > 7:
        return []
    contrast = np.linalg.norm(lab-background, axis=2)
    foreground = np.uint8(contrast > 28)*255
    hinge = annotation['anchors']['hinge']['x']/scale_x
    tip = annotation['anchors']['tip']['x']/scale_x
    span = abs(hinge-tip)
    if span < 30:
        return []
    # Ignore front/hinge and the near tip; retain only separated pixels below
    # the selected arm. A scaled gap prevents its antialias fringe appearing
    # as another component. No logo/OCR or sidedness certainty is inferred.
    for x in range(w):
        q = (x-hinge)/(tip-hinge)
        occupied = np.flatnonzero(selected[:, x])
        if q < .25 or q > .92 or not len(occupied):
            foreground[:, x] = 0
        else:
            foreground[:min(h, int(occupied[-1])+max(3, round(span*.008))), x] = 0
    count, labels, stats, _ = cv2.connectedComponentsWithStats(foreground)
    candidates = [i for i in range(1, count)
                  if stats[i, cv2.CC_STAT_WIDTH] > span*.12
                  and stats[i, cv2.CC_STAT_AREA] > span*span*.001]
    if not candidates:
        return []
    label = max(candidates, key=lambda i: stats[i, cv2.CC_STAT_AREA])
    mask = np.uint8(labels == label)*255
    mask = clean_appearance_mask(mask, contrast)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, [contour], -1, 255, -1)
    # Enclosed light regions are possible printing or hardware highlights,
    # not holes in the arm. Retain them in the appearance reference.
    holes = cv2.bitwise_and(filled, cv2.bitwise_not(mask))
    n, _, regions, _ = cv2.connectedComponentsWithStats(holes)
    marks = sum(1 for i in range(1, n) if regions[i, cv2.CC_STAT_AREA] >= max(3, span*span*.00001))
    contour = cv2.approxPolyDP(contour, max(.6, min(1.2, span*.001)), True).reshape(-1, 2)
    from .contour_refinement import refine_closed_contour
    contour, smoothing = refine_closed_contour(contour,
        max_displacement_px=.85, sample_spacing_px=3., window_px=5.)
    return [{'photo_id': annotation['photo_id'],
             'part': 'temple-left' if annotation['part'] == 'temple-right' else 'temple-right',
             'surface': 'candidate-inner', 'status': 'review', 'applied': False,
             'root_direction_x': 1 if hinge > tip else -1,
             'contour_refinement': smoothing,
             'points': [{'x': float(x*scale_x), 'y': float(y*scale_y)} for x, y in contour],
             'possible_mark_regions': marks,
             'reason': 'Visible opposite-arm appearance retained separately. Light details may be printing or reflections. Generation can transfer the visible tip to an existing opposite arm using approximate tip/width registration; inspect its placement.'}]
