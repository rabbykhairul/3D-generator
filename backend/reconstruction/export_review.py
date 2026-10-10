"""Review the serialized GLB, not merely construction meshes.

These are source consistency and component-preservation checks, not a product
accuracy certificate. Front traces are construction inputs, so their IoU is
explicitly labelled non-independent.
"""
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import trimesh



def validate_export(path, expected_parts, annotations, front_photo, center, scales,
                    width_scale, joint_fit, output_dir):
    scene = trimesh.load(path, force='scene', process=False)
    names = set(scene.geometry)
    missing = sorted(set(expected_parts) - names)
    component_checks = []
    for side in ('left','right'):
        for part, geometry in ((f'temple-{side}',f'temple_{side}'),(f'nose-pad-{side}',f'nose_pad_{side}'),
                               (f'lens-{side}',f'lens_{side}')):
            approved=any(a.get('part')==part for a in annotations)
            if approved:
                component_checks.append({'part':part,'export_geometry':geometry,'present':geometry in names})
    finite = all(np.isfinite(m.vertices).all() and len(m.faces)>0 for m in scene.geometry.values())
    report = {'schema_version':1,'status':'source checks only',
              'serialized_geometry_count':len(names),'missing_exported_parts':missing,
              'finite_nonempty_geometry':bool(finite),'approved_components':component_checks,
              'component_preservation_passed':not missing and all(c['present'] for c in component_checks),
              'independent_fidelity_verified':False,
              'limitations':['Front overlap compares to the approved construction outlines, not independent labels.',
                             'Oblique overlays use fitted weak-perspective cameras and show front components only.',
                             'Hidden hardware, physical fit and material realism are not validated.'],
              'images':[]}
    scale=min(1.,1000/max(front_photo['width'],front_photo['height']))
    shape=(max(1,round(front_photo['height']*scale)),max(1,round(front_photo['width']*scale)))
    expected=np.zeros(shape,np.uint8);rendered=expected.copy()
    front_ann=next((a for a in annotations if a.get('photo_id')==front_photo['id'] and a.get('part')=='front'),None)
    # Continuous full rims only: thin wire construction and rimless do not share
    # the outer-frame-minus-openings topology of this comparison.
    if front_ann and 'front_frame' in names:
        cv2.fillPoly(expected,[np.round(np.array([[p['x'],p['y']] for p in front_ann['points']])*scale).astype(np.int32)],255)
        for ann in annotations:
            if ann.get('photo_id')==front_photo['id'] and ann.get('part') in {'lens-left','lens-right','bridge-opening'}:
                cv2.fillPoly(expected,[np.round(np.array([[p['x'],p['y']] for p in ann['points']])*scale).astype(np.int32)],0)
        mesh=scene.geometry['front_frame']
        points=mesh.vertices[:,:2]*1000/np.array([scales[0]*width_scale,-scales[1]])+np.asarray(center)
        for tri in mesh.faces:
            cv2.fillConvexPoly(rendered,np.round(points[tri]*scale).astype(np.int32),255)
        union=np.count_nonzero((expected>0)|(rendered>0))
        report['front_trace_iou']=float(np.count_nonzero((expected>0)&(rendered>0))/max(1,union))
        with Image.open(front_photo['path']) as im:
            image=np.asarray(im.convert('RGB').resize((shape[1],shape[0]))).copy()
        for mask,color in ((expected,(0,170,110)),(rendered,(240,120,30))):
            contours,_=cv2.findContours(mask,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(image,contours,-1,color,1)
        Image.fromarray(image).save(Path(output_dir)/'fit-front.png')
        report['images'].append({'file':'fit-front.png','label':'Front: green approved trace; orange serialized GLB outline (not independent validation).'})
    front_meshes=[m for n,m in scene.geometry.items() if not n.startswith(('temple_','hinge_'))]
    for i,view in enumerate(joint_fit.get('views',[])[:2]):
        photo=view.get('_photo')
        if not photo:continue
        with Image.open(photo['path']) as im:
            im=im.convert('RGB');im.thumbnail((1000,1000));image=np.asarray(im).copy()
        shrink=np.array([image.shape[1]/photo['width'],image.shape[0]/photo['height']])
        camera=view['camera'];params=np.r_[camera['yaw_pitch_roll_radians'],np.log(camera['pixels_per_mm']),camera['translation_px']]
        from scipy.spatial.transform import Rotation
        rotation=Rotation.from_euler('yxz',params[:3]).as_matrix()
        for mesh in front_meshes:
            xyz=mesh.vertices*1000;xyz=xyz.copy();xyz[:,0]/=width_scale
            uv=(xyz@rotation.T)[:,:2]*np.array([1.,-1.])*np.exp(params[3])+params[4:6]
            mask=np.zeros(image.shape[:2],np.uint8)
            for tri in mesh.faces:
                coords=np.round(uv[tri]*shrink).astype(np.int32)
                if np.max(np.abs(coords))<100000:cv2.fillConvexPoly(mask,coords,255)
            contours,_=cv2.findContours(mask,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(image,contours,-1,(240,120,30),1)
        for observed,predicted in zip(view['withheld_observed_px'],view['withheld_predicted_px']):
            a=tuple(np.round(np.array(observed)*shrink).astype(int));b=tuple(np.round(np.array(predicted)*shrink).astype(int))
            cv2.circle(image,a,3,(0,170,110),1);cv2.circle(image,b,2,(190,40,130),-1);cv2.line(image,a,b,(190,40,130),1)
        filename=f'fit-view-{i}.png';Image.fromarray(image).save(Path(output_dir)/filename)
        report['images'].append({'file':filename,'label':f"{photo['view']}: orange exported front; green withheld feature; pink fitted feature. Fit {joint_fit['status']}; cameras are experimental."})
    from .source_zoom import zoom_diagnostics
    zooms=[]
    for photo in [front_photo, *joint_fit.get('_side_photos', [])]:
        photo_annotations=[a for a in annotations if a.get('photo_id')==photo['id']]
        if photo_annotations:
            filename='fit-zoom-front.png' if photo['id']==front_photo['id'] else 'fit-zoom-'+('left' if 'left' in photo['view'] else 'right')+'.png'
            diag=zoom_diagnostics(photo,photo_annotations,Path(output_dir)/filename)
            zooms.append({'photo_id':photo['id'],**diag})
            if (Path(output_dir)/filename).exists():
                report['images'].append({'file':filename,'label':photo['view']+': 2× native-source crops with approved outlines; inspect frame/hinge/tip ownership.'})
    report['zoom_review']=zooms
    return report
