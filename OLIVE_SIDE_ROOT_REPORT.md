# Side attachment and opposite-arm appearance — engine 12

## Fixed attachment detection

The outer-branch detector began at a fixed fraction of the product width. In the olive side reference this column crossed the connected front rim, including its full vertical extent in the temple profile and placing the hinge at approximately (917,602), rather than at the arm root. `side_extraction.py` now checks for gross root thickness excess against a stable adjacent shaft and trims to the observed transition. It does not trim ordinary, moderately wider hinge sleeves.

On this native 1080px source the corrected anchor is approximately (885,535). The vertical front-rim strip is excluded. Tests cover 540/1080/2160px variants and mirrored orientation, alongside existing side regressions. This is image-relative logic, not a project-ID condition.

## Opposite-arm appearance cross-check

`opposite_arm.py` extracts a separate candidate visible inner-surface region below the selected arm on supported light-background side poses. Enclosed light details remain appearance evidence, not holes or extra geometry. It flags possible markings without claiming OCR, logo identity, or verified material classification.

The extraction response exposes this evidence, `SideDraft.tsx` shows a cropped review preview, and reconstruction metadata retains it under `multiview.opposite_arm_appearance`. `api.ts` types the response. Project/photo changes discard stale previews and pending responses.

**Texture transfer is not implemented.** The region is explicitly marked `applied: false`: a foreshortened partial rear arm does not yet have a verified correspondence with the opposite model surface. This change cross-checks and retains the visible marking, but does not claim it appears on the generated GLB. No copying to both arms occurs. Arbitrary poses and nonuniform backgrounds remain unsupported by this evidence detector.

## Verification

- Full Python suite: 168 passed.
- Draft handler checks and TypeScript/Vite build passed (existing bundle-size warning).
- `scripts/verify_olive_side.py`: disposable source extraction, front + observed right-arm reconstruction, GLB reload and closed-mesh checks; separate opposite-arm evidence retained with `applied: false`.
- `artifacts/olive-side-root/after-outline.png`: orange selected arm and cyan appearance-only opposite region.
- `artifacts/olive-side-root/after-clay.png`: neutral-material reloaded GLB. Other front/depth fidelity limitations remain; this is not a manufacturing-accurate model.
- Before evidence: user-supplied screenshot showing front-rim overrun. No saved projects or approvals were changed. AR/head-motion rendering was not tested in this follow-up.

Files changed: `backend/reconstruction/{side_extraction,opposite_arm,engine}.py`, `backend/engine_version.py`, `frontend/src/api.ts`, `frontend/src/components/SideDraft.tsx`, `tests/test_opposite_arm.py`, `scripts/verify_olive_side.py`, this report and `PROGRESS.md`.
