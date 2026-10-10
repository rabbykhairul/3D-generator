"""Local SAM 2.1 general-region proposals with automatic optical assignment.

General segmentation is not semantic recognition of eyewear hardware. The
shared source-edge/topology checks still decide which proposals are usable.
"""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
import cv2
import numpy as np
from .hybrid_detection import ROOT, INFERENCE_SLOT, _polygon


def select_lenses(masks, scores, shape):
    """Assign paired optical candidates without SKU names or shape templates."""
    h,w=shape
    candidates=[]
    for mask,score in zip(masks,scores):
        contours,_=cv2.findContours(np.uint8(mask),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour=max(contours,key=cv2.contourArea)
        p=_polygon(contour)
        if p is None:
            from .full_rim_recovery import _largest_valid
            p=_largest_valid(contour)
            if p is None:
                continue
        x0,y0,x1,y1=p.bounds
        bw,bh=x1-x0,y1-y0
        if (x0<=1 or y0<=1 or x1>=w-2 or y1>=h-2 or not .015*h*w<p.area<.32*h*w
                or not .6<bw/max(bh,1)<3 or p.area/(bw*bh)<.5):
            continue
        if any(p.intersection(q).area/p.union(q).area>.85 for q,_ in candidates):
            continue
        candidates.append((p,float(score)))
    options=[]
    for i,(a,sa) in enumerate(candidates):
        for b,sb in candidates[i+1:]:
            left,right=sorted((a,b),key=lambda p:p.centroid.x)
            x0,y0,x1,y1=left.union(right).bounds
            span=x1-x0
            ratio=min(a.area,b.area)/max(a.area,b.area)
            if (not a.disjoint(b) or ratio<.45 or not .15<(y1-y0)/span<.8 or
                    abs(a.centroid.y-b.centroid.y)>(y1-y0)*.25 or
                    left.bounds[2]>=right.bounds[0] or
                    max(a.bounds[2]-a.bounds[0],b.bounds[2]-b.bounds[0])/span>.55):
                continue
            # Prefer a broad optical pair over tiny hardware regions. Model
            # predicted IoU is a mask-quality estimate, not part confidence.
            rank=(a.area+b.area)/(h*w)*2+min(sa,sb)+ratio*.1
            options.append((rank,left,right))
    if not options:
        return np.zeros(shape,np.float32),{'optical_pair_selected':False,'optical_candidates':len(candidates)}
    _,left,right=max(options,key=lambda o:o[0])
    output=np.zeros(shape,np.float32)
    for p in (left,right):
        cv2.fillPoly(output,[np.rint(p.exterior.coords).astype(np.int32)],1)
    return output,{'optical_pair_selected':True,'optical_candidates':len(candidates),
                  'component_assignment':'automatic paired-region geometry; no semantic hinge/pad labels'}


def infer_lenses(image):
    if not INFERENCE_SLOT.acquire(timeout=1):
        raise RuntimeError('The CPU segmentation worker is busy. Try analysis again shortly.')
    try:
        return _infer(image)
    finally:
        INFERENCE_SLOT.release()


def _infer(image, task='lenses'):
    python=Path(os.environ.get('EYEWEAR_SAM2_PYTHON',ROOT/'artifacts/segmenter-venv/bin/python'))
    manifest=Path(os.environ.get('EYEWEAR_SAM2_MANIFEST',ROOT/'artifacts/sam2/manifest.json'))
    if not python.is_file() or not manifest.is_file():
        raise RuntimeError('Local SAM 2.1 runtime or checkpoint is missing. Run scripts/setup_sam2.py first.')
    with tempfile.TemporaryDirectory(prefix='eyewear-sam2-') as temporary:
        directory=Path(temporary);photo=directory/'analysis.png';output=directory/'regions.npz'
        if not cv2.imwrite(str(photo),image):
            raise RuntimeError('Could not prepare the segmentation image')
        started=time.perf_counter()
        try:
            run=subprocess.run([str(python),str(ROOT/'scripts/sam2_infer.py'),'--manifest',str(manifest),
                '--image',str(photo),'--output',str(output),'--task',task],cwd=ROOT,capture_output=True,text=True,timeout=120)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError('SAM 2.1 analysis exceeded the 120-second CPU limit. Retry when no other analysis is running.') from exc
        if run.returncode!=0 or not output.is_file():
            raise RuntimeError('SAM 2.1 CPU inference failed. Check the isolated runtime and checkpoint.')
        runtime=json.loads(run.stdout)
        with np.load(output,allow_pickle=False) as regions:
            shape=tuple(regions['shape'].tolist());packed=regions['masks'];scores=regions['scores']
            if (shape!=image.shape[:2] or packed.dtype!=np.uint8 or packed.ndim!=2 or
                    len(packed)>216 or len(scores)!=len(packed) or not np.isfinite(scores).all() or
                    packed.shape[1]!=(shape[0]*shape[1]+7)//8):
                raise ValueError('SAM returned invalid region data')
            masks=np.unpackbits(packed,axis=1,count=shape[0]*shape[1]).reshape((-1,*shape))
        if task=='side':
            runtime.update(request_seconds=round(time.perf_counter()-started,3))
            return masks,scores,runtime
        probabilities,selection=select_lenses(masks,scores,shape)
        runtime.update(selection,request_seconds=round(time.perf_counter()-started,3))
    return probabilities,runtime


def infer_side(image):
    if not INFERENCE_SLOT.acquire(timeout=1):
        raise RuntimeError('The CPU segmentation worker is busy. Try analysis again shortly.')
    try:
        return _infer(image,task='side')
    finally:
        INFERENCE_SLOT.release()


def select_side(masks, scores, photo, scale_x, scale_y, *, minimum_score=.75):
    """Require a continuous elongated region, never a union of two arms.

    An adjacent upright region locates the frame end. That establishes an
    attachment estimate, not recognition of the hinge's mechanical design.
    """
    from .side_extraction import extract_side
    candidates=[]; upright=[]; branched=[]
    for mask,score in zip(masks,scores):
        contours,_=cv2.findContours(np.uint8(mask),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        if not contours or float(score)<minimum_score:
            continue
        c=max(contours,key=cv2.contourArea);x,y,bw,bh=cv2.boundingRect(c)
        h,w=mask.shape
        if x<=1 or y<=1 or x+bw>=w-2 or y+bh>=h-2:
            continue
        area=cv2.contourArea(c)
        if bh>bw*1.5 and area>h*w*.005:
            upright.append((x+bw/2,y,y+bh))
        if bw<w*.45 or bw/max(bh,1)<3 or not h*w*.002<area<h*w*.18:
            continue
        filled=np.zeros_like(mask,dtype=np.uint8);cv2.fillPoly(filled,[c],1)
        split=0;gaps=0
        for column in filled[:,x:x+bw].T:
            rows=np.flatnonzero(column)
            if not len(rows):
                gaps+=1;continue
            runs=np.split(rows,np.where(np.diff(rows)>1)[0]+1)
            split+=sum(len(r)>=2 for r in runs)>1
        if gaps>2:
            continue
        if split>max(2,bw*.03):
            if split<bw*.65:
                branched.append((bw,float(score),filled,(x,y,x+bw-1,y+bh-1)))
            continue
        candidates.append((bw,float(score),filled,(x,y,x+bw-1,y+bh-1)))
    candidates.sort(key=lambda r:(r[0],r[1]),reverse=True)
    if not candidates:
        completions=[]
        for _,score,mask,bounds in sorted(branched,key=lambda r:(r[0],r[1]),reverse=True):
            direction=_frame_endpoint(bounds,upright)
            if direction is None:
                continue
            result,selection=_separate_overlapping_side(mask,photo,scale_x,scale_y,direction)
            if result:
                selection['mask_predicted_iou']=score
                completions.append((result,selection))
        if completions:
            from shapely.geometry import Polygon
            selected=Polygon([(p['x'],p['y']) for p in completions[0][0]['annotations'][0]['points']])
            for other,_ in completions[1:]:
                polygon=Polygon([(p['x'],p['y']) for p in other['annotations'][0]['points']])
                if selected.intersection(polygon).area/selected.union(polygon).area<.8:
                    return None,{'single_arm_selected':False,'reason':'Competing SAM masks imply different arm paths; no overlap completion was selected.'}
            return completions[0]
        return None,{'single_arm_selected':False,'reason':'No continuous single-arm SAM region passed the geometry checks.'}
    span,score,mask,bounds=candidates[0]
    for other in candidates[1:]:
        overlap=np.count_nonzero(mask&other[2])/max(np.count_nonzero(mask|other[2]),1)
        if other[0]>span*.9 and overlap<.65:
            return None,{'single_arm_selected':False,'reason':'Multiple competing arm regions remain ambiguous.'}
    direction=_frame_endpoint(bounds,upright)
    if direction is None:
        return None,{'single_arm_selected':False,'reason':'An arm region was found, but its frame-side attachment could not be located.'}
    result=extract_side(photo,mask*255,scale_x,scale_y,hinge_on_right=direction)
    if not result['annotations']:
        return None,{'single_arm_selected':False,'reason':'The SAM arm region failed the continuous-profile check.'}
    ann=result['annotations'][0]
    ann['source']='automatic-sam2-side-v1'
    ann['anchor_provenance']={'hinge':'inferred-frame-side-endpoint','tip':'observed-region-endpoint'}
    return result,{'single_arm_selected':True,'mask_predicted_iou':score,
                   'reason':'SAM supplied a continuous arm region; its frame-side attachment is inferred. Hidden hinge hardware is not measured.'}


def _frame_endpoint(bounds,upright):
    x0,y0,x1,y1=bounds
    span=x1-x0
    attachments=[(min(abs(cx-x0),abs(cx-x1)),cx) for cx,uy0,uy1 in upright
                 if uy0<=y1 and uy1>=y0 and min(abs(cx-x0),abs(cx-x1))<span*.12]
    if not attachments:
        return None
    sides={cx>(x0+x1)/2 for _,cx in attachments}
    return sides.pop() if len(sides)==1 else None


def _separate_overlapping_side(mask,photo,scale_x,scale_y,hinge_on_right):
    """Follow both envelopes, accepting only a distinctly longer tip path.

    This is projection-based completion, not a claim that the shared shaft
    belongs exclusively to one arm. Equal-length competing paths are withheld.
    """
    from .side_extraction import extract_side
    from .anatomy import audit_photo
    h,w=mask.shape
    paths=[]
    for flip in (False,True):
        work=np.flipud(mask).copy() if flip else mask
        result=extract_side(photo,work*255,scale_x,scale_y,hinge_on_right=hinge_on_right,
                            start_at_frame_boundary=True,upper_branch=True,repair_short_overlap=True)
        if not result['annotations']:
            continue
        ann=result['annotations'][0]
        if flip:
            for point in ann['points']+list(ann['anchors'].values()):
                point['y']=(h-1)*scale_y-point['y']
        if audit_photo(photo,[ann])['blocking']:
            continue
        length=abs(ann['anchors']['hinge']['x']-ann['anchors']['tip']['x'])/scale_x
        paths.append((length,result))
    if len(paths)!=2:
        return None,{'single_arm_selected':False}
    paths.sort(key=lambda r:r[0],reverse=True)
    length,result=paths[0];other_length,other=paths[1]
    ys,xs=np.where(mask)
    span=int(xs.max()-xs.min())
    if length<span*.95 or other_length>length*.9 or other_length<length*.45:
        return None,{'single_arm_selected':False}
    ann=result['annotations'][0]
    from shapely.geometry import Polygon,Point
    selected=Polygon([(p['x']/scale_x,p['y']/scale_y) for p in ann['points']])
    if not selected.is_valid:
        return None,{'single_arm_selected':False}
    raster=np.zeros_like(mask);cv2.fillPoly(raster,[np.rint(selected.exterior.coords).astype(np.int32)],1)
    # Keep the completion on the model's foreground and exclude the other tip.
    support=np.count_nonzero(raster&mask)/max(np.count_nonzero(raster),1)
    other_tip=other['annotations'][0]['anchors']['tip']
    if support<.95 or selected.distance(Point(other_tip['x']/scale_x,other_tip['y']/scale_y))<max(1,max(h,w)*.003):
        return None,{'single_arm_selected':False}
    ann['source']='automatic-sam2-side-occlusion-completion-v1'
    ann['anchor_provenance']={'hinge':'inferred-frame-side-endpoint','tip':'observed-region-endpoint'}
    ann['evidence']=dict(schema_version=1,state='inferred',method='geometric-anatomy-audit',
        review_required=True,notes=['The longer visible tip path was selected automatically. The overlapping shaft has inferred ownership and thickness.'])
    reason='SAM captured overlapping arms. The longer visible branch was separated; the shared shaft and frame-side attachment are inferred. Hidden hinge hardware is not measured.'
    result['warnings']=[reason,'Check the highlighted arm path before approval. Side extrusion approximates cross-section and hidden depth.']
    return result,dict(single_arm_selected=True,overlap_completion=True,source_mask_support=round(float(support),4),
                       branch_length_ratio=round(other_length/length,4),reason=reason)


def apply_side(photo,image,baseline,scale_x,scale_y,*,model='sam2'):
    diagnostic=dict(requested_backend=model,proposal_backend='sam3' if model=='sam3' else 'sam2.1-hiera-small',
                    inference_ran=False,proposals_used=False,experimental=True,status='unavailable')
    result=dict(baseline);result['warnings']=list(baseline.get('warnings',[]))
    result['hybrid_detection']=diagnostic
    try:
        if model=='sam3':
            from .sam3_detection import infer_side as infer_concept_arm
            masks,scores,runtime=infer_concept_arm(image)
        else:
            masks,scores,runtime=infer_side(image)
        diagnostic.update(inference_ran=True,runtime=runtime)
        # SAM 2 supplies predicted mask IoU; SAM 3's official processor supplies
        # concept probability multiplied by presence probability, filtered at
        # .5. Applying the SAM 2 IoU gate discards valid SAM 3 regions before
        # any continuous-profile/topology check can run.
        candidate,selection=select_side(masks,scores,photo,scale_x,scale_y,
                                       minimum_score=.5 if model=='sam3' else .75)
        if model=='sam3' and 'mask_predicted_iou' in selection:
            selection['concept_presence_score']=selection.pop('mask_predicted_iou')
        if model=='sam3' and candidate:
            for ann in candidate['annotations']:
                ann['source']=ann['source'].replace('automatic-sam2-', 'automatic-sam3-')
        diagnostic.update(selection)
    except (RuntimeError,ValueError,OSError,subprocess.SubprocessError) as exc:
        diagnostic['reason']=str(exc)
        result['warnings'].append('SAM side analysis was unavailable; any displayed outline comes from standard analysis.')
        return result
    from .anatomy import audit_photo
    if baseline.get('annotations') and not audit_photo(photo,baseline['annotations'])['blocking']:
        diagnostic.update(status='retained',reason='SAM side analysis ran; the existing valid arm outline was retained.')
    elif candidate and not audit_photo(photo,candidate['annotations'])['blocking']:
        result['annotations']=candidate['annotations'];result['warnings']=candidate['warnings']
        result['warnings'].append(selection['reason'])
        diagnostic.update(status='applied',proposals_used=True)
    else:
        diagnostic['status']='rejected'
        if candidate:
            diagnostic['reason']='The SAM arm proposal failed the anatomy audit.'
        result['warnings'].append(selection['reason'])
    if model=='sam3':
        from .sam3_components import apply_side_components
        result=apply_side_components(photo,image,result,scale_x,scale_y)
    return result
