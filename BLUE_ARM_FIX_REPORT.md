# Blue frame pp: overlapping arm and incomplete hinge

Exact input: project `pp` / `C8E4014E`, right photo `e7304c95abdb452e811de55581bf5ac5`. Original photo hash, extraction, saved project snapshot and exported model are preserved under `artifacts/blue-arm-fix`. The merchant project itself was not changed.

## Confirmed causes

- The binary silhouette contained both arms at their crossing. The existing repair did not remove a short lower-edge spur: at original-photo x=420 the old outline reached y=575 instead of the main arm's approximately y=560.
- The printed white logo divided the dark shaft into separate mask runs. The detector followed its lower run and stopped at x=847, before the visible hinge seam around x=867–869. That became the deep notch in the generated root.
- Texture cleanup separately treated enclosed white printing as background, replacing it with dark neighboring material. This affected appearance, not just extraction.

## Implementation

`side_extraction.py` adds guarded opaque-arm recovery on a quiet light background. A strong full-height photometric seam supports the root. Localized gap filling treats the enclosed mark as solid for geometry; a short crossing repair connects the main lower boundary while retaining the genuine wider shaft farther along. It returns unapproved suggestions and declines when its required evidence is absent. No product ID, blue-color constant or image-coordinate override is used in extraction.

`engine.py` routes suitable photos through that recovery before the generic extractor. Texture sampling preserves small enclosed light markings inside approved solid material while rejecting large apertures and edge-connected background. Engine version 9 marks older exports stale for regeneration without deleting reviews.

`SideDraft.tsx` retains an already approved mirroring choice for the same photo during re-analysis. New photos still default to no mirroring. This avoids accidentally dropping the opposite arm while replacing this project's reviewed right arm.

## Evidence

- [Same-photo outline comparison](artifacts/blue-arm-fix/outline-comparison.png)
- [Exported side-view comparison](artifacts/blue-arm-fix/renders/side-photo.png)
- [Neutral-clay hinge comparison](artifacts/blue-arm-fix/renders/hinge-clay.png)
- [Three-quarter clay comparison](artifacts/blue-arm-fix/renders/threequarter-clay.png)
- [Corrected isolated GLB](artifacts/blue-arm-fix/after/model.glb)

The exported comparisons use identical cameras for before and after. They are CPU diagnostic renders, **not live browser or AR screenshots**; clear lenses are intentionally omitted in those renders. Both arms in the QA copy follow the project's previously approved symmetry assumption, not independent photographic evidence.

## Verified scope

Final verification: **107 Python tests passed** after the texture optimization. Frontend production build and no-browser component event-handler tests passed. Existing dependency deprecation and large-bundle warnings remain non-blocking.

The exact blue photo now produces a continuous shaft through the logo, a root at x=867, and no former short downward tab at the overlap. A mirrored input produces the reflected result. New tests cover the wider shaft, tip, logo and root, plus previous product styles.

The isolated API workflow uploads the product photos, extracts the side, explicitly approves in the test database, generates, downloads and reloads the GLB. It checks both joints, closed welded geometry, winding, finite normals/vertices, embedded arm textures, and retention of over 95% of the source logo's white pixels. The saved merchant project remains unchanged.

Physical depth, the hidden hinge mechanism and facial fit remain estimates; the visible seam is evidence of the arm endpoint, not a recovered mechanical hinge model. This correction is not a claim of universal segmentation or measured physical accuracy. Live AR/head-motion verification was not performed in this session.

## Loading the correction

Restart the Python backend and refresh. In **pp**, use **Analyze side arm** again and approve the new outline before regeneration. Regenerating the old saved outline alone cannot remove the old tab. The existing explicit mirroring checkbox is retained so the same correction can update both arms.

## Changed files

- `backend/reconstruction/side_extraction.py`, `engine.py`, `backend/engine_version.py`
- `frontend/src/components/SideDraft.tsx`
- `tests/test_opaque_side_recovery.py`, `test_blue_arm_export.py`, `test_enclosed_texture_marks.py`
- `scripts/draft-component-test.mjs`, `scripts/verify-blue-arm.py`
- `BLUE_ARM_FIX_PLAN.md`, `PROGRESS.md`
