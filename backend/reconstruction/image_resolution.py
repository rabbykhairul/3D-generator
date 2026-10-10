"""Bound contour-extraction memory while preserving source pixel coordinates.

The returned image is only a temporary analysis raster. The uploaded source is
kept at its EXIF-oriented full resolution for review and texture projection.
Interpolation can make tiny shapes easier for contour code to process, but it
cannot recover an edge that is absent from the source pixels.
"""

from __future__ import annotations

import cv2
import numpy as np

MAX_WORKING_SIDE = 2048
MIN_WORKING_SIDE = 160
LOW_DETAIL_SIDE = 500
THUMBNAIL_SIDE = 128


def resolution_diagnostics(width: int, height: int) -> dict:
    """Describe the extraction raster without allocating or modifying pixels."""
    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive")
    long_side, short_side = max(width, height), min(width, height)
    if long_side > MAX_WORKING_SIDE:
        ratio = MAX_WORKING_SIDE / long_side
        mode = "downsampled"
    elif short_side < MIN_WORKING_SIDE and long_side * MIN_WORKING_SIDE / short_side <= MAX_WORKING_SIDE:
        ratio = MIN_WORKING_SIDE / short_side
        mode = "upsampled"
    else:
        ratio = 1.0
        mode = "source"
    working_width = max(1, round(width * ratio))
    working_height = max(1, round(height * ratio))
    # Rounding is independent on each axis, so use measured coordinate scales.
    scale_x = width / working_width
    scale_y = height / working_height
    warnings = []
    if short_side < THUMBNAIL_SIDE:
        warnings.append(
            "Thumbnail accepted for a low-detail draft subject to review: automatic contours may be incomplete. "
            "Use your largest available photo; enlarging this image will not recover missing detail."
        )
    if short_side < LOW_DETAIL_SIDE:
        warnings.append("Low resolution: 500 pixels or more per side is recommended.")
    if mode == "upsampled":
        warnings.append(
            "Extraction uses ordinary interpolation to process this small photo; it does not add source detail. "
            "Review every suggested boundary against the original pixels."
        )
    elif mode == "downsampled":
        warnings.append(
            "Extraction uses a bounded working copy; the full-resolution source remains available for "
            "review and appearance projection. Check fine wires and transparent edges."
        )
    if min(working_width, working_height) < 80:
        warnings.append(
            "This photo is too narrow in one direction for reliable automatic contours at the safe working size. "
            "Review or draw the outline manually."
        )
    return {
        "source_width": width,
        "source_height": height,
        "working_width": working_width,
        "working_height": working_height,
        "mode": mode,
        "scale_x": scale_x,
        "scale_y": scale_y,
        "warnings": warnings,
    }


def prepare_extraction_image(image: np.ndarray) -> tuple[np.ndarray, float, float, dict]:
    """Return a bounded OpenCV image and working-to-source coordinate scales.

    Coordinates in suggested annotations are mapped with the returned scales.
    The source ndarray is never changed, even when interpolation is necessary.
    """
    if not isinstance(image, np.ndarray) or image.ndim not in (2, 3) or image.dtype != np.uint8:
        raise ValueError("Expected a decoded 8-bit OpenCV image")
    height, width = image.shape[:2]
    diagnostics = resolution_diagnostics(width, height)
    size = (diagnostics["working_width"], diagnostics["working_height"])
    if size == (width, height):
        working = image
    else:
        interpolation = cv2.INTER_AREA if diagnostics["mode"] == "downsampled" else cv2.INTER_LINEAR
        working = cv2.resize(image, size, interpolation=interpolation)
    return working, diagnostics["scale_x"], diagnostics["scale_y"], diagnostics
