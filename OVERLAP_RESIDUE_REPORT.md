# Residual opposite-arm sliver

## Confirmed cause

The previous short-overlap repair used the last lower-edge sample near the separated arm's height. That sample was already part of the merged far-arm silhouette. Interpolation removed the large tab but retained a shallow triangular bulge (about 4.5 source pixels at x=455 on the reproduced blue photo). Polygon smoothing made it smooth, not correct.

## Correction

`backend/reconstruction/side_extraction.py` now locates the preceding local thickness minimum in a bounded neighborhood. Thickness is measured relative to the upper contour, avoiding a raw image-height assumption on a sloping shaft. The repair still requires a separated lower neighboring arm, a significant merging spur, and a continuous upper edge. Distant genuine widening, ear-tip curvature, and the root are not globally flattened. New suggestions identify themselves as `single-visible-arm-opaque-seam-v2`.

No product IDs, blue-color constants, or this photo's coordinates were added to the extraction logic. Test coordinates pin the reported regression only. This is a guarded silhouette inference; it cannot prove hidden manufacturing geometry for arbitrary photos.

## Verification

- Preserved before extraction/GLB under `artifacts/overlap-residue/before`.
- Generated after extraction/GLB without changing any saved merchant project.
- Inspected same-crop `outline-comparison.png` and matching-camera `mesh-comparison.png`. The latter uses a neutral clay material on reloaded GLBs to separate geometry from texture. It is not a live AR screenshot.
- Hinge and tip anchors are identical before/after. Wider shaft at x=550 is unchanged; lower boundary near the residual spur moves from y=561.25 to y=556.75 at x=455.
- Added a strict chord-envelope regression that the previous outline fails, plus synthetic tests that preserve genuine widening, absent-overlap cases, and upper-edge corners. Existing mirrored/gray/pale-metal/transparent regressions pass.
- Reloaded components have finite vertices and welded watertight geometry.
- Final Python suite: **131 passed**, with one existing Starlette dependency deprecation warning.

## Applying it

Existing approved outlines deliberately remain unchanged. With the updated backend running, choose **Analyze side arm**, review/approve the new suggestion, then regenerate. Regeneration alone would continue using the old approved contour. No frontend or AR renderer changes were needed for this extraction defect.
