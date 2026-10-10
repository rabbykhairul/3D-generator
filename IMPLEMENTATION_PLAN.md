# Eyewear reconstruction demo — implementation plan

Standalone local research app. No files in `sl-new` or existing R&D projects are modified. Goal: guided product photos + measurements → editable photo-derived geometry → GLB → webcam AR. Six categories: opaque full-rim, thin metal, transparent full-rim, semi-rimless, rimless, sports/shield. Guided corrections are allowed; automatic accuracy is evaluated separately.

## Stack and contracts
React/TypeScript/Vite; Three.js/MediaPipe. Python 3.11/FastAPI/Pydantic/OpenCV/Pillow/NumPy/SciPy/Shapely/Trimesh. SQLite and local files; one CPU worker. Node 22. No paid API, server GPU, Shopify integration or public deployment.

All geometry input measurements are millimeters; GLB uses meters, origin at bridge center, +X wearer left, +Y up, +Z forward (arms extend -Z). Front image coordinates are pixels, x right/y down; an explicit mapping converts them to model coordinates. Inputs carry revision numbers; jobs use immutable snapshots. Edits make old artifacts stale. Frontend uses shared contract types reflecting OpenAPI.

## Modules

Current follow-up: [FIDELITY_OCCLUSION_PLAN.md](FIDELITY_OCCLUSION_PLAN.md) tracks exact-product research, local bevel/depth candidates and skull/live-face occlusion. Implementation is not acceptance: numerical fidelity remains below target and new live-browser verification is blocked by the unavailable local server. See [FIDELITY_OCCLUSION_REPORT.md](FIDELITY_OCCLUSION_REPORT.md).
1. M0 Foundation: runtime setup, common schemas, fixture conventions, startup scripts and module ownership.
2. M1 Dataset: public source photographs, original/source URLs, dimensions and missing evidence. Target two products/category; use synthetic fixtures only for numerical tests. Withhold a view when available. No invented dimensions/photos or public republishing.
3. M2 Capture: 12 overview slots (8 horizontal angles, top/bottom, elevated front-left/right), additional detail shots, open-arm and scale guidance. Validate JPEG/PNG bytes, EXIF orientation, 20MB/file, 40MP decoded, 40 photos/project. Warn on blur/duplicates and missing views.
4. M3 Interpretation: extract contours, distinguish lens holes/arms, editable pixel-space outlines/anchors, zoom/undo/redo and persisted corrections. Ambiguous images require correction. Pose/calibration is explicit; no planar flattening of curved objects.
5. M4 Geometry: real approved outlines drive frame geometry; thickness/bevel/curvature and arms come from measured or explicitly inferred constraints. Fit against landmarks/contours using SciPy. Family-specific solid/wire/partial/rimless/shield paths. Never return a generic fallback as reconstruction.
6. M5 Export: named connected parts, materials/texture projection, clear lens windows, explicit appearance approximations, GLB + metadata, finite geometry/dimension/topology checks.
7. M6 Review: orbit/wireframe/parts/measurements, source-vs-render comparison, supported silhouette/landmark metrics, independent capture/geometry/fidelity status, revise/regenerate/download.
8. M7 AR: exported GLB, face pose/smoothing/occlusion/tracking loss, PD/manual calibration, browser-only camera, resource cleanup, photo fallback and quality settings.
9. M8 Integration: end-to-end persistence/retry/staleness/download checks, browser QA, tests and evidence report. Record real fidelity limitations and untested devices.

## Ownership and concurrency
Primary owns service/API/storage/worker, dataset, shared contracts, integration, documentation and progress. Agent A owns reconstruction Python package + numerical tests. Agent B owns merchant UI/editor. Agent C owns viewer/AR. Primary freezes contracts first, then runs three agents concurrently, integrates and delegates fixes. Only primary edits PROGRESS.md.

## Acceptance
- Changing an input contour changes the model. Saved corrections reload deterministically. All six category paths have fixtures/tests.
- Invalid images/polygons, missing scale and conflicting measurements produce actionable errors.
- GLBs load independently, have finite vertices and supplied width within 1%; this is numerical construction, not physical accuracy.
- AR uses actual exported model, handles denial/loss and releases camera.
- Real-product report separates measured/inferred/default/manual values, automatic/corrected results, processing and correction times. Target withheld-view IoU >=0.90 and landmark error <=2% frame width only when reliable references exist; do not fabricate passes. No ground-truth 3D means exact reconstruction unproven.
- Measure CPU time/memory/asset size/FPS where possible; AR target >=24FPS on the tested device, mobile coverage stated.
- Complete startup docs, sample projects and downloadable outputs. Distinguish implemented features, validated categories and remaining research.

## Defaults
Local only. No sign-in for loopback demo. One generation per product revision. Camera frames never sent to backend. No optical prescription simulation or physical-fit certification. Public-photo datasets may be incomplete; missing evidence stays visible.

## Surface and sleeve correction — 2026-09-28

- [x] Preserve the reported GLB and inspect matched photographic, neutral-clay and silhouette views against the original side photograph.
- [x] Replace merged-arm pixel tracing with scale-aware single-arm profiles; preserve the true sleeve step and ear hook, suppress crossing-arm blocks and pixel stairs.
- [x] Refine automatic front contours within a subpixel displacement bound, preserving construction corners and reviewed/manual inputs.
- [x] Export explicit crease-aware normals and separate cap/depth-wall UV sampling; retain original cap photographs and product-specific colors.
- [x] Validate refined cap topology, review the actual candidate and regenerate the saved project through its front/side approval workflow.
- [x] Verify matched close-ups, closed construction geometry after export, prior shadow/joint checks, material loading and synthetic AR; publish evidence and limitations in `SURFACE_FIX_REPORT.md`.

## Assembly, shadow and AR color correction — 2026-09-28

- [x] Preserve the reported project's original export and capture matching baseline front/side/three-quarter and AR views.
- [x] Trace original photos, approved contours, model coordinates, GLB materials and both render paths; do not use broken renders as design references.
- [x] Replace fixed low-contrast background segmentation with a reviewed adaptive light-background split; keep thin dark components, and extend only foreground colors into texture borders.
- [x] Remove forced hinge height; center temple cross-sections on frame shoulders and separate apparent side-photo shaft pitch from the ear-hook profile.
- [x] Share photograph-color rendering between inspection and AR without replacing lens tint/opacity or untextured materials.
- [x] Verify actual regenerated asset, joint overlap, rotated views, GLB reload, and synthetic-stream AR tracking; publish before/after evidence and remaining uncertainty in `ASSEMBLY_FIX_REPORT.md`. Real-camera head turns remain unverified.

## Merchant-first revision (2026-09-27)

The default journey is Photos → Dimensions → Review / AR, not a twelve-view capture assignment. Show front as required, side/angle as helpful, and the remaining views as optional in Advanced. Suggest front contours automatically and allow a single visual approval before generation; reserve tracing and side/top anchors for Advanced correction. Never claim that every uploaded angle influences geometry.

Replacing a fitting front/left/right/top photo supersedes that view's old geometry. Keep historical photos and jobs, but invalidate old approvals and exclude superseded captures at reconstruction time. Hide stale previews; label historical downloads. Test that different source silhouettes produce different exported vertex geometry and that an unusable replacement cannot reuse an old model. Show construction category explicitly; photos do not automatically change category or dimensions.
