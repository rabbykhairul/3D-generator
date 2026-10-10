"""Reviewable component evidence from registered extra catalogue photos.

No view is assumed to share a camera or physical depth. Source silhouettes are
checked in their own photo, then mapped as explicitly inferred front-plane
proposals. Unregistered, ambiguous, withheld and validation photos contribute
no geometry. Existing front components are preserved.
"""
from copy import deepcopy
import cv2
import numpy as np
from .multiview import analyze_multiview, _excluded, _load
from .anatomy import audit_photo, attach_evidence
from .hybrid_detection import apply_hybrid
from .sam3_components import apply_front_components

REFERENCE_VIEWS={'front-left','front-right','front_left','front_right','angled','oblique','three-quarter','inner','rear','back'}
TRANSFER_ROLES={'front-support':'frame','bridge-hardware':'bridge','nose-pad-left':'nose-pad','nose-pad-right':'nose-pad','hinge-left':'hinge','hinge-right':'hinge'}


def combine_reference_components(project,photo,result):
    result=deepcopy(result)
    diagnostic=dict(views=[],accepted_parts=[],method='inferred-front-plane-registration',depth_measured=False)
    result['component_multiview']=diagnostic
    if result.get('anatomy',{}).get('blocking') or not {'lens-left','lens-right'}<={a['part'] for a in result.get('annotations',[])}:
        diagnostic['reason']='Source-checked front optical boundaries are required.'
        return result
    latest={}
    for p in project.get('photos',[]):
        if p.get('view') in REFERENCE_VIEWS and not _excluded(p):latest[p['view']]=p
    if not latest:
        diagnostic['reason']='No eligible extra component photos.'
        return result
    # Feature checks precede expensive segmentation. Provisional contours are
    # used for registration bounds without changing any approval flags.
    registration=analyze_multiview(list(latest.values()),photo,result['annotations'],allow_unapproved=True,component_registration=True)
    target=cv2.imread(str(photo['path']))
    if target is None:return result
    used=0
    for row in registration.get('views',[]):
        info={k:v for k,v in row.items() if k in {'photo_id','view','reason','inliers','median_error_px'}}
        info['parts_added']=[];diagnostic['views'].append(info)
        local=row.get('component_registration')
        fit=row if row.get('reason')=='registered' else local or {}
        if (fit.get('inliers',0)<10 or 'source_to_front_homography' not in fit):
            info['usage']='No component geometry transferred';continue
        if used>=2:
            info['usage']='Two-view CPU budget reached';continue
        source=latest[row['view']]
        image,size=_load(source)
        if image is None:continue
        sx,sy=size[0]/image.shape[1],size[1]/image.shape[0]
        # This is component analysis of the source view, not permission to use
        # an oblique optical outline as the reconstruction's primary front.
        candidate=apply_hybrid(source,image,image,dict(annotations=[],warnings=[]),sx,sy,model='sam3')
        used+=1
        matrix=np.asarray(fit['source_to_front_homography'],float)
        proposals=[]
        source_scores={entry['part']:entry['score'] for entry in candidate.get('component_detection',{}).get('accepted',[])}
        info['source_inference']=candidate.get('hybrid_detection',{}).get('reason')
        for ann in candidate.get('annotations',[]):
            if ann['part'] not in TRANSFER_ROLES or ann['part'] in {a['part'] for a in result['annotations']}:continue
            coords=np.array([[p['x'],p['y']] for p in ann['points']],float)
            homogeneous=np.column_stack((coords,np.ones(len(coords)))) @ matrix.T
            if np.any(np.abs(homogeneous[:,2])<1e-6):continue
            points=homogeneous[:,:2]/homogeneous[:,2:]
            if not np.isfinite(points).all():continue
            from shapely.geometry import Polygon
            if local:
                coverage=Polygon(local['front_coverage_polygon'])
                component=Polygon(points)
                if not component.is_valid or not coverage.buffer(2).covers(component):continue
            proposals.append(dict(role=TRANSFER_ROLES[ann['part']],score=source_scores.get(ann['part'],.5),points=points.tolist(),holes=[],
                registration=dict(source_photo_id=str(source['id']),source_points=ann['points'],
                    source_to_front_homography=matrix.tolist(),method='inferred-front-plane-registration')))
        if not proposals:
            info['usage']='No missing source-checked components';continue
        previous={a['part'] for a in result['annotations']}
        working=deepcopy(result)
        working.setdefault('hybrid_detection',{}).setdefault('runtime',{})['component_proposals']=proposals
        working=apply_front_components(photo,target,working)
        if audit_photo(photo,working['annotations'])['blocking']:
            info['usage']='Mapped components failed the joint anatomy check';continue
        # Retain primary detector diagnostics independently of view transfer.
        component_diagnostic=result.get('component_detection')
        result=working
        if component_diagnostic:result['component_detection']=component_diagnostic
        info['parts_added']=sorted({a['part'] for a in result['annotations']}-previous)
        info['usage']='Inferred registered component proposals' if info['parts_added'] else 'Mapped components rejected or already present'
        diagnostic['accepted_parts'].extend(info['parts_added'])
    result['component_multiview']=diagnostic
    if diagnostic['accepted_parts']:
        result.setdefault('warnings',[]).append('Extra-photo components were mapped using a front-plane approximation. Their source silhouettes are observed; transferred position, depth and occluded geometry are inferred. Review these proposals before approval.')
    return attach_evidence(photo,result,photo['width'],photo['height'])
