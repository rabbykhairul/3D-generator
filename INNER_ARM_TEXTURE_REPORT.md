# Inner-arm texture implementation — engine 13

## Implemented

The previous version only collected opposite-arm evidence; it never passed those pixels to the GLB texture builder. `temple_texture.py` now registers the visible distal region to the opposite temple using tip position, local width and the target centerline. `engine.py` applies that map only when the opposite temple already has approved geometry. It never creates an arm just to display a logo. An explicit mirrored-shape approval is accepted; the saved olive project already has that approval.

`render_mesh.py` supports selecting the inward-facing cap by normal (+X for left, -X for right), keeping the original outer texture and depth-wall material separate. Raw enclosed white printing is preserved, edge-connected background is excluded, and the transition to the existing unobserved finish is blended. Export metadata reports the source, applied coverage and registration assumption. `SideDraft.tsx` now describes the actual generation behavior.

The tested olive GLB now visibly contains the circular inner-tip marking after export and reload. It does not appear on the outer-face render. The hinge fix remains intact. No merchant project, measurement or approval was edited.

## Sharpness investigation

The reference is 1080px, below the existing 2048px texture limit, so that limit did not downsample this example. Surface maps remain photographic; depth walls intentionally have low-frequency inferred color. Both standard and AR viewers previously left texture anisotropy at its default. `preserveTextureDetail` now requests up to the device-supported 16× anisotropy in both loaders to reduce oblique-angle texture filtering loss while retaining mipmaps. This is not artificial sharpening or a recovery of missing source detail. GPU/live-AR visual improvement has not been measured in this environment.

## Verification and evidence

- Full Python suite: 170 passed; after final source-crop/background sampling changes, 7 targeted texture/evidence tests passed.
- Frontend TypeScript/Vite build passes (existing bundle-size warning).
- Disposable olive reconstruction uses freshly detected contours and the existing explicit mirroring choice, exports and reloads GLB, checks closed meshes, then renders the isolated inner and outer faces.
- `artifacts/olive-side-root/inner-arm-photo.png`: generated inner face with marking.
- `artifacts/olive-side-root/outer-arm-photo.png`: separate outer face without that marking.
- `artifacts/olive-side-root/before-inner-arm-photo.png`: same-camera comparison with the original outer map restored to the inner tile; a reconstructed baseline, not a historical screenshot.
- `scripts/verify_olive_side.py` reproduces these checks without database writes.

## Remaining limits

Registration is a **local tip/width approximation**, not solved perspective calibration. Printing placement/shape may differ from the physical product. Only the observed distal region is transferred; hidden inner-shaft markings remain unknown. It depends on the current supported separated-arm pose and background. No universal arbitrary-image, pixel-perfect, or live-AR fidelity claim is made. Direct approved inner-side photographs remain preferable and are not overwritten by this inferred transfer.

Changed implementation files: `backend/reconstruction/{temple_texture,opposite_arm,render_mesh,engine}.py`, `backend/engine_version.py`, `frontend/src/viewer/materialPolicy.ts`, `frontend/src/components/{ModelViewer,SideDraft}.tsx`, `tests/test_temple_texture.py`, and `scripts/verify_olive_side.py`.
