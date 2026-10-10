"""Source-checked SAM 3 component silhouettes, in original photo coordinates.

Concept masks propose part identity. Pixel boundaries and optical relationships
must agree before an unapproved annotation is added; hidden depth is not measured.
"""
from copy import deepcopy
import cv2
import numpy as np
from shapely.geometry import Polygon, Point, box
from shapely.ops import unary_union
from .hybrid_detection import _annotation, _sample, _refine_lens_at_radius


def component_proposals(masks, scores, roles):
    proposals=[]
    for mask, score, role in zip(masks,scores,roles):
        if role not in {'frame','bridge','nose-pad','hinge','lens-mount'} or float(score)<.5:
            continue
        contours,hierarchy=cv2.findContours(np.uint8(mask),cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is None:
            continue
        for i in sorted(range(len(contours)),key=lambda i:cv2.contourArea(contours[i]),reverse=True)[:16]:
            if hierarchy[0,i,3]!=-1 or cv2.contourArea(contours[i])<8:
                continue
            holes=[];child=hierarchy[0,i,2]
            while child!=-1:
                if cv2.contourArea(contours[child])>2:
                    holes.append(cv2.approxPolyDP(contours[child],.4,True)[:,0,:].tolist())
                child=hierarchy[0,child,0]
            points=cv2.approxPolyDP(contours[i],.4,True)[:,0,:].tolist()
            if len(points)<3 or len(points)>2000 or any(len(h)<3 for h in holes):
                continue
            polygon=Polygon(points,holes)
            if not polygon.is_valid:
                fixed=polygon.buffer(0)
                if fixed.geom_type!='Polygon' or abs(fixed.area-polygon.area)>2:
                    continue
                polygon=fixed
            if polygon.is_empty:
                continue
            proposals.append(dict(role=str(role),score=float(score),
                points=[list(p) for p in polygon.exterior.coords[:-1]],
                holes=[[list(p) for p in h.coords[:-1]] for h in polygon.interiors]))
    return proposals


def _poly(annotation,sx,sy):
    return Polygon([(p['x']/sx,p['y']/sy) for p in annotation['points']])


def _edge_support(image,polygon):
    count=max(24,min(400,round(polygon.length/2)))
    pts=np.array([polygon.exterior.interpolate(i*polygon.length/count).coords[0] for i in range(count)])
    tangent=np.roll(pts,-1,axis=0)-np.roll(pts,1,axis=0)
    length=np.linalg.norm(tangent,axis=1)
    if np.any(length<1e-5):return 0.
    normals=np.column_stack((-tangent[:,1],tangent[:,0]))/length[:,None]
    lab=cv2.cvtColor(cv2.GaussianBlur(image,(3,3),.6),cv2.COLOR_BGR2LAB).astype(np.float32)
    strengths=[]
    for offset in (-1.,0.,1.):
        coords=pts+normals*offset
        strengths.append(np.linalg.norm(_sample(lab,coords+normals*1.5)-_sample(lab,coords-normals*1.5),axis=1))
    return float(np.mean(np.max(strengths,axis=0)>=8))


def _checked(image,proposal):
    p=Polygon(proposal['points'],proposal.get('holes',[]))
    h,w=image.shape[:2]
    if not p.is_valid or p.area<8:return None,'invalid-outline'
    x0,y0,x1,y1=p.bounds
    if x0<=1 or y0<=1 or x1>=w-2 or y1>=h-2:return None,'clipped-region'
    # Preserve topology: never fill a meaningful hole in a hardware silhouette.
    if p.interiors and proposal['role']!='frame':return None,'hardware-holes-unsupported'
    exterior=Polygon(p.exterior)
    radius=max(1,min(3,round(max(h,w)/600)))
    refined,detail=_refine_lens_at_radius(image,exterior,None,radius)
    if refined is None:
        if _edge_support(image,exterior)<.7:return None,'weak-source-boundary'
        refined=exterior
    if p.interiors:
        refined=Polygon(refined.exterior,[ring.coords for ring in p.interiors])
        if not refined.is_valid:return None,'refined-hole-conflict'
    return refined,'source-checked'


def _distinct(options):
    selected=[]
    for p,proposal in sorted(options,key=lambda v:v[1]['score'],reverse=True):
        if any(p.intersection(q).area/max(p.union(q).area,1)>.65 for q,_ in selected):continue
        selected.append((p,proposal))
    return selected


def _follows_optical_edge(p, lenses):
    """A long, narrow mask on a lens edge is not separate pad evidence.

    Use the rotated footprint rather than the axis-aligned box: sloping rim
    strips can otherwise look wide enough to pass ordinary pad size checks.
    """
    _,edges,_=cv2.minAreaRect(np.asarray(p.exterior.coords,dtype=np.float32))
    if min(edges)<=0 or max(edges)/min(edges)<5:
        return False
    for lens in lenses:
        width=lens.bounds[2]-lens.bounds[0]
        corridor=lens.boundary.buffer(max(2,width*.022))
        if p.intersection(corridor).area/p.area>.8:
            return True
    return False


def apply_front_components(photo,image,result,sx=1.,sy=1.,*,source_prefix='sam3'):
    result=deepcopy(result)
    display_name='SAM 3' if source_prefix=='sam3' else source_prefix.replace('-', ' ').title()
    runtime=result.get('hybrid_detection',{}).get('runtime',{})
    proposals=runtime.pop('component_proposals',[])
    diagnostics=dict(method=source_prefix+'-concept-and-source-boundaries',accepted=[],rejected=[],depth_measured=False)
    result['component_detection']=diagnostics
    annotations=result.setdefault('annotations',[])
    parts={a['part']:a for a in annotations}
    if not {'lens-left','lens-right'}<=parts.keys():
        diagnostics['reason']='Optical boundaries are required to assign front hardware.'
        return result
    lenses=[_poly(parts['lens-'+side],sx,sy) for side in ('left','right')]
    if any(not p.is_valid for p in lenses):return result
    left,right=lenses;optical=unary_union(lenses)
    lw=min(p.bounds[2]-p.bounds[0] for p in lenses);lh=min(p.bounds[3]-p.bounds[1] for p in lenses)
    top=min(p.bounds[1] for p in lenses);bottom=max(p.bounds[3] for p in lenses)
    center=(left.bounds[2]+right.bounds[0])/2;span=right.bounds[2]-left.bounds[0]
    bridge_box=box(left.bounds[2]-.14*lw,top-.15*lh,right.bounds[0]+.14*lw,top+.65*lh)
    nasal_box=box(left.bounds[2]-.24*lw,top+.1*lh,right.bounds[0]+.24*lw,bottom+.1*lh)
    options={part:[] for part in ('front','front-support','bridge-hardware','nose-pad-left','nose-pad-right','hinge-left','hinge-right','lens-mount-left','lens-mount-right')}
    for proposal in proposals:
        if proposal.get('registration'):
            p=Polygon(proposal['points']);reason='source-checked-in-reference-view'
            if not p.is_valid or not box(1,1,image.shape[1]-2,image.shape[0]-2).contains(p):p=None
        else:
            p,reason=_checked(image,proposal)
        role=proposal['role'];target=None
        if p is None:
            diagnostics['rejected'].append(dict(role=role,reason=reason));continue
        x0,y0,x1,y1=p.bounds;bw=x1-x0;bh=y1-y0
        if role=='frame':
            outer=Polygon(p.exterior)
            if all(outer.contains(lens) for lens in lenses):
                material=outer.difference(optical)
                # The mask itself must identify material around both openings.
                # A whole-glasses mask with lens interiors is not a frame rim.
                if (all(p.intersection(lens.buffer(-max(1,lw*.01))).area<lens.area*.15 for lens in lenses)
                        and material.area>outer.area*.025 and p.intersection(material).area/material.area>.7
                        and all(outer.boundary.distance(lens.boundary)>.5 for lens in lenses)):
                    target='front';p=outer
            else:
                # A connected brow/central support is a component, not an outer
                # frame enclosing the lenses. Do not manufacture missing rims.
                band=box(left.bounds[0]-.1*lw,top-.2*lh,right.bounds[2]+.1*lw,top+.24*lh).union(bridge_box)
                if (not p.interiors and .003< p.area/optical.area <.16 and bw>span*.45
                        and p.intersection(band).area/p.area>.94
                        and all(p.distance(lens)<lw*.04 for lens in lenses)
                        and all(p.intersection(lens.buffer(-max(1,lw*.008))).area<lens.area*.015 for lens in lenses)):
                    target='front-support'
        elif role=='bridge':
            if (not p.interiors and p.intersection(bridge_box).area/p.area>.94 and bw<lw*.65
                    and .0005<p.area/optical.area<.06
                    and all(p.distance(lens)<lw*.06 for lens in lenses)
                    and all(p.intersection(lens.buffer(-max(1,lw*.01))).area<lens.area*.02 for lens in lenses)):
                target='bridge-hardware'
        elif role=='nose-pad':
            if _follows_optical_edge(p,lenses):
                diagnostics['rejected'].append(dict(role=role,reason='rim-strip-is-not-separate-pad-evidence'))
                continue
            if (p.intersection(nasal_box).area/p.area>.9 and .0015<p.area/min(l.area for l in lenses)<.09
                    and lw*.012<bw<lw*.28 and lh*.045<bh<lh*.55 and bh>bw*.65):
                target='nose-pad-left' if p.centroid.x<center else 'nose-pad-right'
        elif role=='hinge':
            for side,lens in zip(('left','right'),lenses):
                edge=lens.bounds[0] if side=='left' else lens.bounds[2]
                region=box(edge-lw*.25,top-lh*.1,edge+lw*.25,top+lh*.75)
                if (p.intersection(region).area/p.area>.94 and .0005<p.area/lens.area<.08
                        and bw<lw*.3 and bh<lh*.45 and p.distance(lens)<lw*.1):
                    target='hinge-'+side
        elif role=='lens-mount':
            for side,lens in zip(('left','right'),lenses):
                edge=lens.bounds[0] if side=='left' else lens.bounds[2]
                region=box(edge-lw*.18,top-lh*.1,edge+lw*.18,top+lh*.7)
                if (p.intersection(region).area/p.area>.94 and
                    .0003<p.area/lens.area<.06 and bw<lw*.26 and bh<lh*.38 and
                    p.distance(lens)<lw*.06):
                    target='lens-mount-'+side
        if target:
            options[target].append((p,proposal))
        else:
            diagnostics['rejected'].append(dict(role=role,reason='component-location-or-material-conflict'))
    # A coarse frame concept can locate peripheral material even when the
    # hinge word finds nothing. Trace a visible mounting leaf from source
    # pixels, and require that the SAM frame mask actually supports it.
    if photo.get('path') and any(p['role']=='frame' for p in proposals):
        from .rimless_mount import visible_mounts
        from shapely.affinity import scale as scale_polygon
        mounts=visible_mounts(photo,{side:np.array([[v['x'],v['y']] for v in parts['lens-'+side]['points']]) for side in ('left','right')})
        frames=[Polygon(p['points'],p.get('holes',[])) for p in proposals if p['role']=='frame' and not p.get('registration')]
        for side,mount in mounts.items():
            profile=scale_polygon(mount['polygon'],xfact=1/sx,yfact=1/sy,origin=(0,0))
            if _edge_support(image,profile)<.7 or not any(p.is_valid and p.intersection(profile).area/profile.area>.8 for p in frames):continue
            options['hinge-'+side].append((profile,dict(score=max(p['score'] for p in proposals if p['role']=='frame'),
                method='frame-supported-visible-mount-profile')))
    from .anatomy import audit_photo
    for part,choices in options.items():
        if part in parts or part=='front-support' and 'front' in parts:continue
        choices=_distinct(choices)
        if len(choices)!=1:
            if choices:diagnostics['rejected'].append(dict(role=part,reason='competing-component-regions'))
            continue
        p,proposal=choices[0]
        source='automatic-'+source_prefix+'-'+part+'-source-v1'
        if proposal.get('registration'):source='automatic-'+source_prefix+'-view-registration-'+part+'-v1'
        ann=_annotation(photo,part,p,sx,sy,source)
        if proposal.get('registration'):ann['registration']=proposal['registration']
        ann['anchors']={}
        if part.startswith('hinge-'):
            point=p.representative_point()
            ann['anchors']['hinge']=dict(x=point.x*sx,y=point.y*sy)
            ann['anchor_provenance']={'hinge':'inferred-front-plane-registration' if proposal.get('registration') else 'observed-sam3-source-hardware'}
        if audit_photo(photo,[*annotations,ann],width=image.shape[1]*sx,height=image.shape[0]*sy)['blocking']:
            diagnostics['rejected'].append(dict(role=part,reason='joint-anatomy-conflict'));continue
        annotations.append(ann);parts[part]=ann
        diagnostics['accepted'].append(dict(part=part,score=proposal['score'],source_boundary_checked=True,method=proposal.get('method','semantic-component-mask')))
    if diagnostics['accepted']:
        if 'front-support' in parts or 'bridge-hardware' in parts:
            result['warnings']=[w.replace('Frame support is unresolved;','Complete outer-frame topology is unresolved;') for w in result.get('warnings',[])]
        result.setdefault('warnings',[]).append(display_name+' component silhouettes passed source and location checks. Review component identity; hardware depth and hidden mechanics remain inferred.')
        diagnostic=result.get('hybrid_detection',{})
        diagnostic['proposals_used']=True
        diagnostic['status']='applied' if 'front' in parts else 'partial'
        diagnostic['reason']='Source-checked '+display_name+' components were added; unresolved parts remain uncertain.'
        from .construction_classifier import suggest_construction
        result['construction']=suggest_construction(image,annotations,sx,sy)
    diagnostics['reason']='Accepted only unambiguous source-supported components.'
    return result


def apply_side_components(photo,image,result,sx=1.,sy=1.,*,source_prefix='sam3'):
    result=deepcopy(result)
    proposals=result.get('hybrid_detection',{}).get('runtime',{}).pop('component_proposals',[])
    diagnostics=dict(method=source_prefix+'-concept-and-source-boundaries',accepted=[],rejected=[],depth_measured=False)
    result['component_detection']=diagnostics
    annotations=result.get('annotations',[])
    arm=next((a for a in annotations if a['part'].startswith('temple-')),None)
    if not arm:return result
    polygon=_poly(arm,sx,sy);root=arm.get('anchors',{}).get('hinge')
    if not root:return result
    point=Point(root['x']/sx,root['y']/sy);span=polygon.bounds[2]-polygon.bounds[0]
    candidates=[]
    for proposal in proposals:
        if proposal['role']!='hinge':continue
        p,reason=_checked(image,proposal)
        if p is None:
            diagnostics['rejected'].append(dict(role='hinge',reason=reason));continue
        if (p.distance(point)>span*.055 or p.area>polygon.area*.4 or
                p.bounds[2]-p.bounds[0]>span*.12 or p.distance(polygon)>span*.015):
            diagnostics['rejected'].append(dict(role='hinge',reason='outside-selected-arm-root'));continue
        candidates.append((p,proposal))
    candidates=_distinct(candidates)
    if len(candidates)==1:
        p,proposal=candidates[0]
        # The observed hardware must meet the selected arm. Place the anchor
        # inside their intersection, not on a hinge from the folded far arm.
        joint=p.intersection(polygon)
        if joint.area<1:return result
        point=joint.representative_point()
        arm['anchors']['hinge']=dict(x=point.x*sx,y=point.y*sy)
        arm.setdefault('anchor_provenance',{})['hinge']='observed-'+source_prefix+'-source-hardware'
        part=arm['part'].replace('temple-','hinge-')
        source='automatic-'+source_prefix+'-'+part+'-source-v1'
        if proposal.get('registration'):source='automatic-'+source_prefix+'-view-registration-'+part+'-v1'
        ann=_annotation(photo,part,p,sx,sy,source)
        if proposal.get('registration'):ann['registration']=proposal['registration'];ann['anchors']={}
        if not any(a['part']==part for a in annotations):annotations.append(ann)
        diagnostics['accepted'].append(dict(part=part,score=proposal['score'],source_boundary_checked=True,method=proposal.get('method','semantic-component-mask')))
    elif candidates:
        diagnostics['rejected'].append(dict(role='hinge',reason='competing-arm-root-hardware'))
    return result
