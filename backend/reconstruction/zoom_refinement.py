"""Bounded two-times contour raster refinement; never fabricate photo detail.

Work in analysis pixels then return source coordinates. A supersampled soft
mask reduces short threshold teeth; topology and displacement guards keep
thin rims, lens separation and physical corners from being erased.
"""
import cv2
import numpy as np
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union


def refine_front_zoom(annotations, scale_x=1., scale_y=1.):
    front=next((a for a in annotations if a.get('part')=='front'),None)
    holes=[a for a in annotations if a.get('part','').startswith('lens-') or a.get('part')=='bridge-opening']
    if front is None or len(holes)<2:
        return {'applied':False,'reason':'missing_frame_or_apertures'}
    factors=np.array([scale_x,scale_y])
    def coords(a):return np.array([[p['x'],p['y']] for p in a['points']])/factors
    originals=[coords(a) for a in [front,*holes]]
    extent=np.ptp(originals[0],axis=0).max()
    # At thumbnail scale there is not enough evidence to allow multi-pixel
    # movement. At catalogue size smooth ~2px teeth, never large notches.
    sigma=max(.35,min(4.5,extent*.0045))
    displacement=max(.8,min(3.5,extent*.0035))
    diagnostics=[]
    for index,ann in enumerate([front,*holes]):
        raw=coords(ann);polygon=Polygon(raw)
        if not polygon.is_valid:
            continue
        origin=np.floor(raw.min(axis=0))-8
        size=np.ceil((raw.max(axis=0)-origin+8)*2).astype(int)
        if max(size)>4160 or min(size)<8:
            continue
        mask=np.zeros((size[1],size[0]),np.uint8)
        cv2.fillPoly(mask,[np.rint((raw-origin)*2).astype(np.int32)],255)
        blurred=cv2.GaussianBlur(mask,(0,0),sigmaX=sigma*2,sigmaY=sigma*2)
        contours,_=cv2.findContours(np.uint8(blurred>=128)*255,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        if len(contours)!=1:
            continue
        points=cv2.approxPolyDP(contours[0],.6,True).reshape(-1,2)/2+origin
        candidate=Polygon(points)
        shift=polygon.boundary.hausdorff_distance(candidate.boundary)
        if candidate.is_valid and shift>displacement:
            # Keep sharp features locally rather than letting one corner
            # prevent improvement of the entire rim. Preserve neighborhoods
            # wherever either boundary would exceed the movement budget.
            protected=[Point(p).buffer(max(6.,sigma*4)) for p in raw
                       if candidate.boundary.distance(Point(p))>displacement*.8]
            protected += [Point(p).buffer(max(6.,sigma*4)) for p in points
                          if polygon.boundary.distance(Point(p))>displacement*.8]
            if protected:
                region=unary_union(protected)
                candidate=candidate.difference(region).union(polygon.intersection(region))
                if candidate.geom_type!='Polygon' or candidate.interiors:
                    diagnostics.append({'part':ann['part'],'applied':False,'reason':'protected_corner_topology'})
                    continue
                points=np.array(candidate.exterior.coords[:-1])
                shift=polygon.boundary.hausdorff_distance(candidate.boundary)
        candidate=candidate.simplify(.55,preserve_topology=True)
        # Geometry unions at protected corners can leave near-collinear
        # machine-precision vertices. Quantize far below a source pixel so
        # triangulation sees stable coordinates instead of numerical slivers.
        points=np.round(np.array(candidate.exterior.coords[:-1]),3)
        candidate=Polygon(points)
        if not candidate.is_valid:
            repaired=candidate.buffer(0)
            if repaired.geom_type=='Polygon' and not repaired.interiors and abs(repaired.area-candidate.area)<.001:
                candidate=repaired
                points=np.array(candidate.exterior.coords[:-1])
        shift=polygon.boundary.hausdorff_distance(candidate.boundary)
        if not candidate.is_valid or shift>displacement or abs(candidate.area-polygon.area)>polygon.area*.015:
            diagnostics.append({'part':ann['part'],'applied':False,'reason':'displacement_guard','max_analysis_pixel_shift':float(shift)})
            continue
        previous=ann['points']
        ann['points']=[{'x':float(x),'y':float(y)} for x,y in points*factors]
        assembled=Polygon(coords(front),[coords(a) for a in holes])
        if not assembled.is_valid:
            ann['points']=previous
            diagnostics.append({'part':ann['part'],'applied':False,'reason':'thin_rim_topology_guard'})
        else:
            diagnostics.append({'part':ann['part'],'applied':True,'max_analysis_pixel_shift':float(shift)})
    return {'applied':any(d['applied'] for d in diagnostics),'zoom':2,
            'method':'supersampled bounded contour regularization','adds_source_detail':False,'parts':diagnostics}
