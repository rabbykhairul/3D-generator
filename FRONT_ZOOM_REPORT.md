# Two-times front contour refinement — engine 15

## What was implemented

The uploaded image is not enlarged and saved as a replacement. The detector's suggested front/lens masks are refined in a temporary two-times-resolution raster. Short threshold teeth are softened and converted back to source-image coordinates. This combines subpixel contour placement with bounded smoothing; it does not recover missing photographic information.

`backend/reconstruction/zoom_refinement.py` limits contour movement, preserves local sharp features that would otherwise exceed that limit, and rejects changes that damage frame/lens topology. It retains only sub-millipixel numerical repairs after coordinate stabilization. `engine.py` invokes it for front suggestions; saved annotations remain untouched. The engine version advances to 15.

For the lavender bb reference, the final outer contour moved at most 2.5 analysis/source pixels; lens edges at most 1.0 and 0.91 pixels. The upper-rim zigzags are reduced, not replaced with a generic ellipse or symmetry template. Larger uncertain shape errors remain outside this refinement's scope.

## Verification

- Final full Python regression suite: **178 passed**.

- `tests/test_zoom_refinement.py`: lavender variants at 256/540/1080/2160px, original coordinate bounds, valid contained lenses, and no invented missing apertures.
- Targeted catalog generation, API upload/approval/generation, upload-guide reference and zoom tests: 9 passed.
- Early trial contours exposed near-collinear numerical slivers during mesh triangulation. Stabilizing coordinates and bounded tiny-polygon repair resolved the targeted regressions before delivery.
- `scripts/verify_front_zoom.py`: same-source before/after outlines plus front-only generation, GLB reload and welded closure checks in isolated output. Diagnostic dimensions are assumed, not product measurements.
- Matching enlarged crops: `artifacts/front-zoom/before-brow.png` and `after-brow.png`.
- Matching clay renders: `artifacts/front-zoom/before-clay.png` and `after-clay.png`.

No merchant project, source image or saved approval was changed. No frontend changes were required. Live AR was not tested in this follow-up. Very small images still cannot establish invisible edges, thickness or printed detail; this is not AI super-resolution or a universal automatic segmentation fix.
