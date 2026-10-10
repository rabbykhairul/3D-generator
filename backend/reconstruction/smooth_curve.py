"""Bounded, dense curve suggestions; never rewrite approved geometry."""
import numpy as np
from scipy.ndimage import gaussian_filter1d, distance_transform_edt
from shapely.geometry import Polygon


def smooth_closed_curve(poly, scale=1.):
    boundary=poly.exterior
    count=max(24,int(np.ceil(boundary.length/(3*scale))))
    samples=np.array([boundary.interpolate(t).coords[0] for t in np.linspace(0,boundary.length,count,endpoint=False)])
    smooth=gaussian_filter1d(samples,1.25,axis=0,mode='wrap')
    delta=smooth-samples
    length=np.linalg.norm(delta,axis=1)
    smooth=samples+delta*np.minimum(1,1.25*scale/np.maximum(length,1e-12))[:,None]
    candidate=Polygon(smooth)
    if candidate.is_valid and candidate.boundary.hausdorff_distance(boundary)<1.75*scale and abs(candidate.area-poly.area)<poly.area*.008:
        return candidate
    return poly


def smooth_detected_curve(poly, scale=1., *, anchors=(), max_area_change=.008,
                          max_displacement=2.5):
    """Suppress short raster waves while retaining sustained design corners.

    Only automatic suggestions call this. A two-sided tangent over a longer
    arc distinguishes a bridge/endpiece corner from alternating pixel teeth.
    Movement is bounded in image coordinates, not by a fitted ellipse.
    """
    if not poly.is_valid or poly.is_empty:
        return poly
    scale = max(.25, float(scale))
    boundary = poly.exterior
    count = max(32, int(np.ceil(boundary.length / scale)))
    samples = np.array([boundary.interpolate(t).coords[0]
                       for t in np.linspace(0, boundary.length, count, endpoint=False)])
    step = boundary.length / count
    reach = min(count // 8, max(2, round(10 * scale / step)))
    before = samples - np.roll(samples, reach, axis=0)
    after = np.roll(samples, -reach, axis=0) - samples
    cosine = np.sum(before * after, axis=1) / np.maximum(
        np.linalg.norm(before, axis=1) * np.linalg.norm(after, axis=1), 1e-12)
    # Protect a neighborhood around genuine abrupt turns, including bridge
    # notches. Long-radius lens arcs remain eligible for smoothing.
    broad_reach = min(count//8, max(reach+1, round(25*scale/step)))
    broad_before = samples-np.roll(samples, broad_reach, axis=0)
    broad_after = np.roll(samples, -broad_reach, axis=0)-samples
    broad_cosine = np.sum(broad_before*broad_after, axis=1)/np.maximum(
        np.linalg.norm(broad_before, axis=1)*np.linalg.norm(broad_after, axis=1), 1e-12)
    corners = ((cosine < np.cos(np.deg2rad(55))) &
               (broad_cosine < np.cos(np.deg2rad(45))))
    protected = corners.copy()
    for offset in range(1, max(2, round(12 * scale / step))):
        protected |= np.roll(corners, offset) | np.roll(corners, -offset)
    for anchor in anchors:
        protected |= np.linalg.norm(samples-np.asarray(anchor), axis=1) < 4*scale
    smooth = gaussian_filter1d(samples, 6 * scale / step, axis=0, mode='wrap')
    delta = smooth - samples
    length = np.linalg.norm(delta, axis=1)
    delta *= np.minimum(1, max_displacement * scale / np.maximum(length, 1e-12))[:, None]
    # A hard on/off neighborhood around corners creates a new kink where
    # smoothing starts. Ease its influence along the closed curve instead.
    if protected.any():
        distance=distance_transform_edt(np.tile(~protected,3))[count:2*count]*step
        influence=1-np.exp(-.5*(distance/(4*scale))**2)
        delta *= influence[:,None]
    delta[protected] = 0
    candidate = Polygon(samples + delta).simplify(.08 * scale, preserve_topology=True)
    if (candidate.is_valid and
            candidate.boundary.hausdorff_distance(boundary) <= (max_displacement+.25) * scale and
            abs(candidate.area - poly.area) <= poly.area * max_area_change):
        return candidate
    return poly
