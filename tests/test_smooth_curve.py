import numpy as np
from shapely.geometry import Polygon
from backend.reconstruction.smooth_curve import smooth_closed_curve


def test_dense_curve_removes_short_teeth_without_moving_silhouette():
    t=np.linspace(0,2*np.pi,240,endpoint=False)
    radius=100+np.sin(t*60)*.8
    raw=Polygon(np.column_stack((200+radius*np.cos(t),200+radius*.8*np.sin(t))))
    smooth=smooth_closed_curve(raw)
    assert smooth.is_valid
    assert smooth.boundary.hausdorff_distance(raw.boundary)<1.75
    assert abs(smooth.area-raw.area)/raw.area<.008
    def roughness(p):
        points=np.array(p.exterior.coords[:-1])
        delta=np.roll(points,-1,axis=0)-points
        unit=delta/np.linalg.norm(delta,axis=1)[:,None]
        return np.percentile(np.linalg.norm(np.roll(unit,-1,axis=0)-unit,axis=1),95)
    assert roughness(smooth)<roughness(raw)*.5
