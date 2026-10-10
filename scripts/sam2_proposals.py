"""Automatic SAM 2.1 image prompts; no merchant clicks or eyewear training.

SAM masks are general regions, not semantic eyewear part labels. Component
assignment and photographic boundary validation happen in the app process.
"""
import cv2
import numpy as np


def automatic_grid(image):
    h,w=image.shape[:2]
    n=max(1,min(h,w)//40)
    lab=cv2.cvtColor(image,cv2.COLOR_RGB2LAB).astype(np.float32)
    border=np.concatenate((lab[:n].reshape(-1,3),lab[-n:].reshape(-1,3),
                           lab[:,:n].reshape(-1,3),lab[:,-n:].reshape(-1,3)))
    background=np.median(border,axis=0)
    noise=float(np.percentile(np.linalg.norm(border-background,axis=1),90))
    x0,y0,x1,y1=0,0,w,h
    if noise<6 and background[0]>170:
        mask=np.uint8(np.linalg.norm(lab-background,axis=2)>max(12,noise*3))
        count,labels,stats,_=cv2.connectedComponentsWithStats(mask,8)
        regions=[stats[i] for i in range(1,count) if stats[i,4]>h*w*.001]
        if regions:
            x0=min(s[0] for s in regions);y0=min(s[1] for s in regions)
            x1=max(s[0]+s[2] for s in regions);y1=max(s[1]+s[3] for s in regions)
    xs=np.linspace(x0+(x1-x0)*.04,x1-(x1-x0)*.04,12)/w
    ys=np.linspace(y0+(y1-y0)*.04,y1-(y1-y0)*.04,6)/h
    return np.array([(x,y) for y in ys for x in xs],np.float32)


def generate(model, image):
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    generator=SAM2AutomaticMaskGenerator(model,points_per_side=None,
        point_grids=[automatic_grid(image)],points_per_batch=4,
        pred_iou_thresh=.75,stability_score_thresh=.9,crop_n_layers=0,
        min_mask_region_area=0,output_mode='binary_mask',use_m2m=False)
    return generator.generate(image)


def guided_lenses(model, image, probabilities):
    """A coarse locator supplies prompts, never final reconstruction masks."""
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    mask=np.uint8(probabilities>.5)
    h,w=mask.shape
    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    contours=[c for c in contours if cv2.contourArea(c)>.01*h*w]
    if len(contours)!=2:
        return None
    predictor=SAM2ImagePredictor(model)
    predictor.set_image(image)
    regions=[]
    for contour in sorted(contours,key=lambda c:cv2.boundingRect(c)[0]):
        x,y,bw,bh=cv2.boundingRect(contour)
        if x<=1 or y<=1 or x+bw>=w-2 or y+bh>=h-2:
            return None
        proposal=np.zeros((h,w),np.uint8);cv2.fillPoly(proposal,[contour],1)
        distance=cv2.distanceTransform(proposal,cv2.DIST_L2,5)
        cy,cx=np.unravel_index(np.argmax(distance),distance.shape)
        points=np.array([[cx,cy],[x-4,y+bh/2],[x+bw+4,y+bh/2],
                         [x+bw/2,y-4],[x+bw/2,y+bh+4]],np.float32)
        masks,scores,logits=predictor.predict(point_coords=points,point_labels=np.array([1,0,0,0,0]),
            box=np.array([x,y,x+bw,y+bh]),multimask_output=True)
        options=[]
        for candidate,score,logit in zip(masks,scores,logits):
            candidate=np.asarray(candidate,dtype=bool)
            union=np.count_nonzero(candidate|proposal.astype(bool))
            overlap=np.count_nonzero(candidate&(proposal>0))/max(union,1)
            if float(score)>=.75 and overlap>=.65:
                stability=np.count_nonzero(logit>1)/max(np.count_nonzero(logit>-1),1)
                options.append((float(score),candidate,overlap,float(stability)))
        if not options:
            return None
        score,candidate,overlap,stability=max(options,key=lambda r:r[0])
        regions.append(dict(segmentation=candidate,predicted_iou=score,stability_score=stability,
                            locator_overlap=overlap))
    return regions
