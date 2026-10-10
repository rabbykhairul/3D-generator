"""Small-scale contour cleanup, restricted to inferred lower lens-rim arcs.

The filled outer silhouette is filtered, never the thin frame annulus itself.
This prevents erosion from deleting thin metal rims. Brow corners and bridge
are outside the correction region. No saved merchant geometry is modified.
"""
from shapely.geometry import Polygon, box
from shapely.ops import unary_union


def clean_lower_rim(front, lenses):
    if not front.is_valid or len(lenses) != 2 or any(not p.is_valid for p in lenses):
        return front, False
    widths = [p.bounds[2] - p.bounds[0] for p in lenses]
    radius = max(.8, min(4., min(widths) * .01))
    # Rolling-disc opening removes narrow outward spurs, then closing removes
    # adjacent pixel nicks. Lens openings aren't subtracted until AFTER this.
    smooth = front.buffer(-radius).buffer(radius).buffer(radius).buffer(-radius)
    if smooth.geom_type != 'Polygon':
        return front, False
    regions = unary_union([box(p.bounds[0] - radius * 4,
                               p.bounds[1] + (p.bounds[3] - p.bounds[1]) * .40,
                               p.bounds[2] + radius * 4, p.bounds[3] + radius * 4)
                           for p in lenses])
    result = front.difference(regions).union(smooth.intersection(regions))
    if (result.geom_type != 'Polygon' or not result.is_valid or
            any(not result.contains(p) for p in lenses) or
            abs(result.area - front.area) > front.area * .005):
        return front, False
    return result.simplify(.18, preserve_topology=True), True
