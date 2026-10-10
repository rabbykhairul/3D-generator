"""CPU photo-contour eyewear reconstruction.

The engine preserves approved image outlines. Dimensions constrain the mapping,
while unobserved depth and arms are recorded as inferences in the metadata.
"""

from .engine import NeedsCorrection, extract_annotations, reconstruct

__all__ = ["NeedsCorrection", "extract_annotations", "reconstruct"]
