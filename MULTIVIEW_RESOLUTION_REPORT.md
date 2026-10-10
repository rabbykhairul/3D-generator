# Four-view and resolution improvements — engine 10

## Implemented

The existing front, side, angled, and inner/rear roles are retained; no additional views are required. Saved merchant projects and approvals were not edited.

- Upload originals remain intact after the existing EXIF orientation handling. JPEG/PNG limits remain 20 MB, 40 megapixels, and at least 64 pixels on each axis.
- Extraction now uses an adaptive working raster: small inputs can be interpolated to a 160-pixel short side, while large images use at most a 2048-pixel long side. Coordinates return to the original image space with separate X/Y scale factors. Working dimensions and limitations are reported.
- Front and approved side outlines continue to supply geometry and appearance. One-sided mirroring still requires explicit approval.
- Angled photos undergo feature registration or a weaker color/aperture consistency check. The latter supplies no warp or geometry and is labelled “Checked, not applied.” A consistency check is not proof that two photos depict the same SKU.
- Inner/rear photos can supply a separate inner-front-frame texture when registration is supported. A cropped lens fallback requires neighboring-lens orientation evidence, reverses rear-view screen sides, and limits application to the supported rim region. Geometry, lens transparency, and temple transforms are not changed by this texture operation.
- Back caps/back bevels have independent texture coordinates in an embedded GLB atlas. The front photo stays on the front. Unobserved inside regions receive an explicitly inferred plain material rather than copied outer logos. Source background and approved lens openings are excluded.
- Review now lists actual photo contributions, checks-only, and exclusions. Optional evidence is not silently described as used merely because it was uploaded.

## Verification

- Resolution matrix: blue opaque, pale transparent, and black-brow/pale-metal front and side images at 64, 100, 256, 1080, and 4096 pixels: **21 complete, 2 partial, 7 declined, 0 invalid/error**. “Complete” means required valid outlines exist, not pixel-perfect reconstruction or measured 3D accuracy. The larger derivatives test scale handling; they do not add detail to the original source.
- Transparent front declines at 64/100; its side declines at 64/100/256. Pale-metal front declines at 64 and lacks lens openings at 100/256. Results are deliberately not hidden behind a generic model.
- GLB tests cover embedded separate inner/outer texture tiles, reloaded UV assignments, unchanged bounds, explicit normals, and welded watertight geometry.
- Isolated blue-product generation uses the existing approved QA snapshot, not a saved-project mutation. All seven exported components reload with finite vertices and closed geometry.
- The final cropped rear mapping supplies 31.11% of approved frame-material pixels on the corresponding opposite screen-side rim. Its angled image passes only the weaker color/aperture check; it does not supply depth or a texture warp.
- Matching-camera front/rear/three-quarter CPU diagnostic renders are saved under `artifacts/multiview-resolution/generation/`. Clear lenses are deliberately omitted in these diagnostics; they are not live AR or physically based reference renders.
- The production frontend build passes. Upload guide, evidence-summary rendering, and draft action tests pass.
- Final Python regression suite: **130 passed**, with one existing Starlette dependency deprecation warning. Frontend build retains its existing large-chunk warning.
- Browser material-policy testing could not run: Chrome terminated during launch in this sandbox. Live AR/head movement and browser visual acceptance are **not verified** by the CPU renders or GLB tests.

## Files changed

- `backend/main.py`, `backend/engine_version.py`
- `backend/reconstruction/image_resolution.py`, `engine.py`, `multiview.py`, `inner_texture.py`, `render_mesh.py`
- `frontend/src/components/Capture.tsx`, `PhotoGuide.tsx`, `photoGuide.css`, `EvidenceSummary.tsx`, `evidenceSummary.css`
- `frontend/src/App.tsx`, `frontend/src/types.ts`
- Tests: image resolution/extraction, multiview, render mesh, API messaging; scripts `evaluate_resolution_matrix.py`, `verify_multiview_generation.py`, `photo-guide-test.mjs`, `evidence-summary-test.mjs`.

## Remaining limits

This is still photo-assisted draft reconstruction, not supplier CAD quality. This change does not implement full volumetric multi-view optimization, measured hinge mechanics, or automatic mapping of an oblique inner-arm inscription. Front-frame registration cannot safely texture non-coplanar arms. Continuous textured front-frame meshes support the new back atlas; other construction categories retain their existing material path.

Small, blurred, transparent, occluded, or mismatched photos can still need correction or be declined. Upsampling cannot restore missing information. A larger original does not remove ambiguity from hidden surfaces, physical dimensions, or transparency. No ≥90% fidelity claim has been established.

Regenerate with engine 10 to use these changes. Old GLBs are not rewritten automatically. The backend must be running the updated source; refresh alone cannot update a stale Python process.
