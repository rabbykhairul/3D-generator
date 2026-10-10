# Multi-view fidelity and AR occlusion

Scope: standalone demo only. Keep Shopify and the Desktop prototype unchanged. Preserve the current product photos/approvals and historical GLBs. Target project D9CB85EA / aa starts at revision 9, job `2818bd221e9049b1b23a4505a87dd3dd`.

## Reconstruction validation

- [x] Identify the exact product from original photographs; retrieve public same-product views with source URLs. Do not substitute visually similar glasses or a different color/size.
- [x] Preserve revision-9 baseline and compute camera-pinned foreground silhouette diagnostics; record camera/mask limitations and reject folded/open comparisons.
- [ ] Preserve baseline and establish held-out silhouette IoU plus landmark error. Report per-view scores; exclude white background from accuracy. Training-outline agreement is not independent validation.
- [x] Implement local clearance-limited bevels, explicit depth validation and slimmer buried hinge connectors; distinguish measured, inferred and default dimensions. Overall fidelity improvement remains unproven.
- [x] Export offline depth candidates and run geometry/material regressions without modifying the saved project.
- [x] Export engine-7 candidate, inspect matching-angle CPU photo/clay/silhouette renders and run geometry/material regressions. Browser rendering remains separate below.
- [x] Replace global-depth arm placement with actual rear-surface overlap and add conservative two-face temple bevels preserving silhouette bounds and UVs.
- [x] Add repeatable multi-view camera manifests, source hashes, aspect-preserving masks and cropped/folded-view exclusions. No independent accuracy claim.
- [x] Exercise actual reference upload, reviewed-outline restore, worker generation, downloads and GLB/material reload in an isolated temporary API database.
- [ ] Iterate against independently reviewed references. Claim the requested >=90% only if the defined held-out metric reaches it. Full 3D/material accuracy cannot be established by an uncalibrated photo score.

## AR

- [x] Inspect/copy supplied head occluder and register it independently from rigid glasses fit.
- [x] Implement live facial depth mesh and side-head skull depth coverage; visual validation remains open.
- [x] Expose skull alignment, live-face mesh, fitted/detected anchors, glasses wireframe and soft shadow controls, plus depth/head/glasses calibration.
- [x] Persist calibration per model; handle corrupt/missing settings and asset load failures safely.
- [x] Run numerical projection, reset, skull registration, depth-material and storage tests.
- [x] Fix live/skull shadow receivers, exclude lenses from opaque frame-shadow casting, and cancel pending AR startup on stop/switch.
- [ ] Verify depth ordering, controls, tracking loss/recovery, resize/mirroring and resource cleanup; preserve colors and lens transparency.

## Delivery

- [x] Save baseline/candidate silhouette overlays, actual scores and limitations in FIDELITY_OCCLUSION_REPORT.md.
- [x] Capture and inspect matching-angle CPU-rendered before/after GLB evidence; label renderer and clear-lens limitations.
- [ ] Capture browser/live AR evidence. Local server is unavailable; this session cannot bind localhost ports. User reports the app stopped; awaiting Terminal error output or restart.

Depth occlusion hides virtual geometry behind the face/head. It does not remove real glasses or their arms from camera pixels. Live physical-device tracking, hidden hinge internals, exact scale and calibrated material appearance remain separate validation requirements.
