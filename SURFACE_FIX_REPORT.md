# Surface and temple correction — 2026-09-28

## Confirmed causes

1. The old side mask incorporated pixels of the second arm where it crossed the visible arm near the ear sleeve. Its 447-point outline retained a hanging block, three spikes and pixel stair steps. The new 55-point single-arm profile uses separated-arm evidence around the crossing, scale-aware filtering and conservative smoothing. It keeps the genuine sleeve transition and curved tip. Both arms in this project use the same profile under the pre-existing explicit symmetry assumption.
2. The previous GLB omitted `NORMAL` attributes. The loader therefore used flat shading. Export now includes angle-weighted, crease-aware normals; sharp construction corners remain split while smooth curves share normals.
3. The same photographic edge pixels were stretched across the full extruded depth. New UV seams separate the original photographic caps from a low-frequency, locally sampled color field on unobserved depth walls. No fixed product color is substituted.
4. Texture padding eroded a narrow diagonal arm's sample mask, then extended the nearest surviving pixel globally. This produced repeated Voronoi color bands visible as scallops even on a geometrically straight edge. Only synthetic padding is now filtered; observed interior photo pixels and alpha remain intact.

The original product photos, not the erroneous generated screenshots, are the design evidence. Photo/clay/silhouette modes isolate the physical contour from its texture and lighting.

## Implementation

- `backend/reconstruction/side_extraction.py`: scale-aware single-arm extraction, crossing repair, continuous sleeve/ear-tip profile, thin-wire and thumbnail handling.
- `backend/reconstruction/contour_refinement.py`: subpixel-bounded smoothing of new automatic front suggestions, with construction corners/extrema locked and polygon validation. Existing manual approvals are not silently smoothed.
- `backend/reconstruction/engine.py`: refinement integration, v4 provenance/review gates, explicit normal export, cap/wall face labels and filtered texture padding. Dense-contour triangulation additionally required rejecting zero-area cap triangles, excluding a triangle's own vertices from T-junction splitting, and using consistent coordinate-key rounding at cap/wall seams.
- `backend/reconstruction/render_mesh.py`: crease-aware normals, texture atlas and coincident rendering seams. `_CONSTRUCTION_VERTEX` preserves original topology IDs so export checks can distinguish UV/normal seams from physical holes.
- `backend/store.py`, `frontend/src/App.tsx`, `QuickDraft.tsx`, `SideDraft.tsx`: old automatic traces require reanalysis; retain side algorithm provenance and explicit mirrored-arm choice.
- `frontend/src/components/ModelViewer.tsx`: photographic/material/clay/silhouette inspection and reference-oriented side view. Shared preview/AR material policy and independent lens opacity/tint remain in use.
- Regression tests and scripts cover contour refinement, sleeve extraction, triangulation, exported normals, render seams, atlas sampling, prior assembly/shadow checks and UI regeneration.

## Evidence and verification

Baseline is saved project `aa`, revision 15, job `a4fbcb83ead2497b937bea132c8229e1`. Its GLB and snapshot are preserved under `artifacts/surface-fix/`; the historical previous baseline remains in `historical-1f/`. Candidate renders are explicitly labeled and are not evidence that the saved project was refreshed.

Final project is revision **17**, job `5508434d9ae94369b59d3b24945d0285`, engine 4. Front and right-arm suggestions were reviewed and approved through the actual UI; the existing mirrored-left assumption was retained. The regeneration script asserts photos, dimensions, materials and category are unchanged. Original exports remain available. CPU generation took about 3.4 seconds on this local machine.

The final GLB SHA256 is `5adcb114e0291c0b72bbeef316386fa3f661c4d7a03035046a59d4ac22f08457`, identical to the visually checked final candidate. Close-ups show both arms free of the unsupported block/spikes, a retained small sleeve transition, and substantially reduced photo-only bands. Small photographic variations remain; this is not a texture-free CAD replica.

- [Matched arm close-ups: before/after, photo/clay/silhouette](artifacts/surface-fix/comparison-after-side-closeup.png)
- [Matched frame close-ups: before/after](artifacts/surface-fix/comparison-after-threequarter-closeup.png)
- [Front after](artifacts/surface-fix/after-front-photo.png), [front before](artifacts/surface-fix/baseline-front-photo.png)
- [AR test-portrait screenshot](artifacts/surface-fix/qa-ar-photo.png), [AR test results](artifacts/surface-fix/ar-results.json)
- [Saved-project regeneration checks](artifacts/surface-fix/regeneration-17-result.json), [render capture provenance](artifacts/surface-fix/after-result.json)
- [Actual exported-asset geometry/material checks](artifacts/surface-fix/saved-export-regression.json)

Verification: **76 Python tests pass**, TypeScript/Vite production build passes (existing large-chunk warning remains), five pose-scale assertions and approximate-dimension tests pass. The actual two-arm/inner-view/relabel UI workflow passes. All seven exported/reloaded parts have closed outward-oriented construction topology and explicit finite unit normals (maximum length error below 4.1e-8); rendering UV/normal seams do not introduce physical gaps. Shadow probes remain empty while the real thin lower rim and upper bar remain solid. Both joint roots intersect their hinges; maps are embedded and lens tint/alpha remain unchanged.

Shared material-policy tests load the final GLB and rotate it through three AR-style parent poses without altering relative component transforms or independent materials. Real MediaPipe landmark detection on a public test portrait, simulated-stream tracking/loss/recovery, camera denial, track cleanup and zero AR face uploads pass without browser errors. The forced software renderer measured approximately 8 FPS for the still preview and 2 FPS/1 tracking update per second for the simulated stream under concurrent headless testing: **not** a validated mobile-performance result. No user camera was accessed.

## Remaining limitations

This is photo-guided reconstruction, not ground-truth 3D scanning. The unseen arm is explicitly mirrored, dimensions/depth and hinge internals remain approximate, and unobserved wall colors are inferred from local photographed colors. Camera perspective is only partly constrained by the supplied views. Filtering cannot recover missing detail from thumbnails. Real-camera head motion, calibrated physical fit and mobile performance require device testing; synthetic tracking and parent-transform checks do not establish these.
