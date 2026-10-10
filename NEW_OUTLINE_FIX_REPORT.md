# New-photo outline correction — 2026-09-28

## Exact input, not the old product

Project `ooo` / `CE81F4FF`, front `0614da10abb84db08c7ed2110a9f0993` and right side `071de32f5a964f5796f996d0fc31524c`. Source hashes are recorded in `artifacts/new-outline-fix/source-hashes.json`. No saved project, merchant approval, or model pointer was modified.

## Confirmed causes and changes

1. **Front arm wrongly becoming rim:** the OpenCV lens-hole contour doubled back where the rear arm touched the rim. Its polygon was invalid, so the existing convex-completion branch skipped it. Complete invalid lens suggestions from the original contour's hull, disclose the inferred edge, and validate containment/disjointness before exposing approvable suggestions. The arm is now inside the aperture instead of inside the frame material.
2. **Small lower-rim protrusions:** faint-mask thresholding retained tiny shadow/pad-edge spurs. Bounded rolling-disc filtering operates on the filled outer silhouette only within lower-lens regions, not on the thin frame annulus. It retains the metal rim and leaves brow corners/bridge outside its correction region.
3. **Broken pale side shaft:** global Otsu contrast was 95 for this photo; the previous faint retry at 24 fragmented the patterned metal into narrow internal runs. A guarded retry at nearby lower contrast levels uses vertical gap closing, stable-contour agreement, local continuity and short-notch cleanup. It preserves the sleeve step and curved tip. An initial result included the front endpiece in the root; visual inspection rejected it. The final result stops at the visible shaft before that merged front region.
4. **Apparently unchanged analysis:** the front Analyze button sometimes reused approved automatic contours rather than extracting again. Explicit Analyze now always detects afresh; Show approved outlines is separate. Suggestions remain unapproved and unsaved until an explicit action. Side Adjust now carries the actual suggestion into Advanced instead of discarding it.

## Evidence inspected

- [Front before/after](artifacts/new-outline-fix/front-comparison.png)
- [Right side before/after](artifacts/new-outline-fix/right-comparison.png)
- [Exported/reloaded GLB diagnostic views](artifacts/new-outline-fix/generated-views.png): front, right side, three-quarter; photo and neutral clay.
- [Isolated QA GLB](artifacts/new-outline-fix/generated/model.glb)

Overlays use the exact uploaded photos at matching resolution/crop. GLB renders are CPU diagnostics, not browser or live AR screenshots. Clear lenses are omitted in these diagnostic renders so the rim is visible. Only the photographed right arm is present; an opposite arm was not silently invented or approved.

## Verification

- **101 Python tests passed**, including the new product's real upload → extraction → explicit test-only approval → generation → download → GLB reload in an isolated database.
- Both lens holes are valid, disjoint and contained. Rear-arm sample pixels lie inside lens apertures; tested lower-rim material survives and identified spurs are excluded.
- The right joint belongs to the front surface and lies within its arm root. Reloaded parts have closed welded geometry, consistent winding and finite vertices/normals. Front and arm textures remain embedded; lens material retains blending.
- New side tested in original and horizontally flipped orientations. Previous gray side unchanged. The ambiguous two-arm lavender view is deliberately excluded from this new recovery method rather than replacing it with the wrong merged arm.
- Front/side component event-handler tests pass: Analyze does not overwrite approvals; Show approved preserves them; approval and correction writes are explicit. Upload-guide tests pass.
- Production frontend build passes, with the pre-existing large-bundle warning. Python emits a dependency deprecation warning.

## Limits and use

These fixes address the reported new-photo outline failures; they do not establish exact manufacturing geometry or universal input support. Hidden lens-edge completion and the occluded hinge attachment remain inferred. Existing project size/depth/material calibration is unchanged. In particular, this project is currently set to `transparent-full-rim`; the black-brow/metal product's material classification was not silently changed by an outline fix. Photo colors are retained, but calibrated material/physical depth is a separate limitation.

Native browser startup is blocked by this session's sandbox; live browser/head-motion acceptance was not claimed. Restart the Python backend to load new extraction code, refresh the frontend, then run Analyze on the front and right-side photos. Existing saved outlines do not silently migrate; inspect the fresh suggestions before approving generation.

## Files changed

- `backend/reconstruction/engine.py`: invalid-hole completion, topology gate, lower-rim cleanup integration, pale-side recovery routing.
- `backend/reconstruction/front_cleanup.py`: bounded lower-rim cleanup.
- `backend/reconstruction/side_extraction.py`: guarded pale-shaft continuity and front/root separation.
- `frontend/src/components/QuickDraft.tsx`, `SideDraft.tsx`: real reanalysis, separate saved outlines, retain side correction drafts.
- `tests/test_new_catalog_outlines.py`, `tests/test_side_smoothing.py`, `scripts/draft-component-test.mjs`: regressions.
- `scripts/verify_new_outlines.py`, `scripts/verify-new-side-outline.py`: repeatable diagnostic evidence.
- `NEW_OUTLINE_FIX_PLAN.md`, `PROGRESS.md`: checked progress.
