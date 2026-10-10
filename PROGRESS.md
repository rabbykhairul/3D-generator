# Module progress

## Front contour zoom refinement — engine 15

- [x] Add temporary 2× contour-mask refinement with source-coordinate mapping and no claimed new source detail.
- [x] Reduce lavender brow teeth while protecting lens containment and sharp local features.
- [x] Resolve triangulation regressions and verify front-only GLB generation/reload plus matching-crop comparisons.
- [ ] Missing low-resolution evidence and arbitrary-image detection remain unresolved by zoom alone. See FRONT_ZOOM_REPORT.md.

## Rear-arm shadow and contour refinement — engine 14

- [x] Reproduce exact gray-tip shadow inclusion and remove the attached weak spur.
- [x] Add bounded corner-aware smoothing to rear appearance and newly detected temple outlines; preserve root anchors.
- [x] Save matching-crop before/after evidence and regression guards for pale material and real tip pixels.
- [ ] Universal shadow separation and live AR visual verification remain unproven. See REAR_SHADOW_REPORT.md.

## Partial inner-temple texture transfer — engine 13

- [x] Apply observed opposite-arm distal appearance to the correct inner cap; leave outer surface and missing geometry unchanged.
- [x] Preserve enclosed light printing and remove edge-connected background sampling; report approximate tip/width registration and coverage.
- [x] Verify logo in reloaded GLB CPU renders, separate outer map, 170-test full suite and final 7-test texture/evidence regression.
- [x] Configure hardware-supported anisotropic filtering in preview and AR loaders.
- [ ] General perspective calibration, unseen inner-shaft details and GPU/live-camera sharpness verification remain open. See INNER_ARM_TEXTURE_REPORT.md (supersedes engine-12 transfer limitation).

## Olive side attachment / opposite-arm appearance — engine 12

- [x] Remove connected front-rim excess from the detected temple root; verify mirrored and 540–2160px cases.
- [x] Cross-check visible opposite-arm appearance separately from selected-arm geometry and show a cropped evidence preview.
- [x] Preserve appearance evidence in generated metadata, without claiming texture application; 168 Python tests pass and isolated GLB reload verified.
- [ ] Perspective registration and actual inner-arm texture transfer remain unimplemented. See OLIVE_SIDE_ROOT_REPORT.md.

## Clear-frame and automatic construction follow-up — engine 11

- [x] Recover provisional neutral-clear outlines using stable luminance boundaries rather than requiring frame color.
- [x] Add qualified construction suggestions with manual override and save-on-approval behavior.
- [x] Add semi-rimless and double-bridge recovery; omit invented opaque lower wires.
- [x] Verify 163 Python tests, draft UI handler checks and production build; generate/reload isolated front-only diagnostic GLBs.
- [ ] Water-clear visual fidelity remains incomplete: nose-pad spikes and hinge nicks persist despite valid closed geometry. See CONSTRUCTION_CLEAR_FRAME_REPORT.md and its clay renders.
- [ ] Arbitrary merchant-photo reliability and live AR verification remain open; no universal detection claim.

## Transparent shaft / switched arm branch

- [x] Reproduce pink/lavender hybrid-arm results and repair continuous outer-branch selection, transparent shaft envelope, and crossing residue.
- [x] Prevent high-resolution competing detector from selecting the shorter far-arm tip.
- [x] Verify both products at 64–4096px and mirrored views, including explicit declines for unresolved tiny transparent shafts.
- [x] Fix lavender front lens containment at 128/160px; leave unsupported rimless/clipped outlines declined.
- [x] Compare outlines and neutral-material reloaded GLBs without changing merchant approvals; 151 tests and production build pass.
- [ ] Live AR remains unverified. See BRANCH_DETECTION_FIX_REPORT.md for evidence and remaining limits.

## Residual opposite-arm sliver

- [x] Preserve the actual blue-photo detection and generated GLB before changing extraction.
- [x] Replace contaminated repair endpoint with a bounded local thickness minimum, retaining genuine shaft widening and hinge/tip anchors.
- [x] Add a strict residual-bump regression plus guards for absent rear-arm evidence and intentional upper-edge steps; mirrored and other-material regressions pass.
- [x] Visually compare same-crop outlines and same-camera reloaded GLB clay renders (`artifacts/overlap-residue`). Final full suite: 131 passed.
- [x] Version new suggestions as opaque-seam-v2. Saved approvals are not overwritten; re-analyze and approve the side to update an existing model.

## Four-view evidence and resolution — engine 10

- [x] Preserve originals, bound extraction resolution, and map thumbnail/high-resolution contours into original pixels.
- [x] Test 30 blue/transparent/pale-metal cases from 64px through 4K: 21 complete, 2 partial, 7 declined; no invalid polygons/errors.
- [x] Process angle evidence as registration/consistency checks and registered inner evidence as a separate rear-frame texture; correct rear-view side reversal.
- [x] Display actual per-photo use and limitations in Review; update existing upload guidance without extra required photos.
- [x] Verify GLB separate-surface UV/material reload, finite closed geometry, and matching-camera CPU renders in isolated QA output.
- [ ] Live browser/AR visual verification remains unavailable: browser launch fails in the sandbox.
- [ ] Full volumetric reconstruction, exact hidden hinges, and oblique inner-arm text mapping remain unimplemented; no 90% fidelity claim. See MULTIVIEW_RESOLUTION_REPORT.md.

## Blue pp overlapping arm / hinge cutout — 2026-09-28

- [x] Preserve exact pp baseline photo extraction, saved-project snapshot and GLB without changing the merchant project.
- [x] Exclude short rear-arm crossing spur while preserving the genuine wider shaft; use visible seam to recover full logo-area shaft and root.
- [x] Keep small enclosed white printing as texture rather than erasing it; engine 9 invalidates older exports for regeneration.
- [x] Retain an existing explicit mirroring choice on same-photo re-analysis; keep default off for new photos.
- [x] Verify exact blue input, horizontal reflection, prior-product regressions, API → GLB export/reload and embedded printing.
- [x] Inspect matching-camera side/hinge/three-quarter photo and clay comparisons. Tests/build results and links in BLUE_ARM_FIX_REPORT.md.
- [ ] Live AR/head-motion check is not performed; hidden hinge mechanics/depth remain estimated.

## New black-brow / pale-metal photo outline failures — 2026-09-28

- [x] Reproduce using project ooo's NEW photos, retain before overlays, and visually compare final front/side overlays.
- [x] Repair invalid occluded lens contours; prevent known-invalid or intersecting suggestions from reaching quick approval; clean small lower-rim spurs without eroding the thin rim.
- [x] Recover pale side-shaft continuity and sleeve/tip; reject initial front-contaminated root and verify the corrected root. Preserve existing gray/lavender behavior and test flipped orientation.
- [x] Fix front Analyze reuse of old approvals; preserve side suggestions on explicit Advanced correction.
- [x] Pass real new-product isolated API → generation → GLB reload, all 101 Python tests, no-browser component handler tests, upload-guide tests and production build.
- [x] Inspect front/side/three-quarter photo and neutral-clay exported-asset renders. See NEW_OUTLINE_FIX_REPORT.md and NEW_OUTLINE_FIX_PLAN.md.
- [ ] Native browser/live AR acceptance remains unverified due to sandbox browser startup restrictions. No claim of universal image support or exact hidden geometry.

## Merchant photo examples and transparent-front recovery — 2026-09-28

- [x] Engine 8 preserves faint tinted texture pixels within approved geometry; centralized version invalidates old exports without deleting reviews.
- [x] Faint-side fallback supplies an unapproved, explicitly ambiguous overlapping-arm suggestion. Visual review still requires shaft/branch corrections; do not treat it as finished geometry.
- [x] Final checks: 96 Python tests, upload-guide render/action tests and production frontend build pass. No live browser/AR verification this turn.

- [x] Add front/side/three-quarter sample photographs with matching upload actions, capture guidance, required/optional distinctions and explicit mirroring limitations. Example photo redistribution rights remain unverified for retail release.
- [x] Replace misleading clearer-photo failure text with missing-boundary details. Preserve suggestions when opening Advanced correction; never auto-approve them.
- [x] Add guarded faint-tinted-front recovery using stable paired apertures across contrast levels and color evidence, retaining the existing dark-frame shadow filter. Reject ambiguous background/neutral pale cases rather than inventing geometry.
- [x] Verify supplied lavender front now has valid nested front/two-lens suggestions without the observed shadow shelves; it generates a closed front-only GLB after explicit test-only approval. No merchant project was changed.
- [x] Upload-guide server-render/action/asset tests and production build pass. Python regressions cover synthetic pale frames, existing shadow cases and the supplied real guide photo.
- [ ] Broad production image coverage and real merchant completion rates remain unvalidated; see MERCHANT_UPLOAD_PLAN.md. JPEG/PNG guided catalog views are supported, not guaranteed arbitrary-image reconstruction.

## Near arm hidden during live AR — 2026-09-28

- [x] Identify unconditional generic-skull depth masking: the existing Show skull checkbox controlled only debugging, not actual occlusion.
- [x] Keep tracked-face masking on by default; make uncalibrated generic skull masking opt-in. Add independent live-face/skull depth switches and calibration guidance without modifying GLB geometry or forcing arms to render on top.
- [x] Tie shadow visibility to the corresponding depth-mask switch; keep alignment overlays independently available before enabling a mask. Node occlusion tests and production build pass.
- [ ] Verify the reported physical head pose after refresh. This mitigates uncalibrated skull over-occlusion, not a measured fit for every head. Both masks can be temporarily disabled to distinguish masking from absent geometry; far arms will then show through the head.

## Engine 7 and final AR audit — 2026-09-28

- [x] Anchor arms to the actual local structural rear surface; add conservative symmetric arm bevels with bounds/UV/root-overlap regressions.
- [x] Apply optional shadows to tracked facial geometry, clip skull shadow consistently, exclude lens shadow casters, and cancel late AR startup.
- [x] Explain material depth and validate its accepted range before saving; mark older engine exports stale without deleting approvals/history.
- [x] Add source-hashed pinned multi-view evaluation and aspect-preserving reference masks; reject cropped/folded views from full-product acceptance.
- [x] Generate and visually inspect CPU matching-angle photo/clay/silhouette pairs under `artifacts/fidelity/cpu-engine7/`.
- [x] Pass 92 Python tests, including actual reference upload → approved-outline restore → generation → download → GLB reload in a temporary database. Node dimension/AR tests and production build pass.
- [x] Improve dev launcher failure diagnostics; reproduce frontend bind EPERM with correct nonzero supervisor exit. No port occupant was found.
- [ ] Independent >=90% fidelity: latest front/front-left/side diagnostic IoU is 74.22% / 58.95% / 63.68%. These are not physical-accuracy percentages and do not improve over baseline.
- [ ] Actual browser/head-motion visual acceptance: user reports the app stopped; session cannot bind ports. Awaiting restart or Terminal error output. Saved project D9CB85EA remains unchanged; code applies on regeneration.

## Multi-view fidelity and skull occlusion — 2026-09-28

- [x] Identify exact Lenskart Air LA E13069-C6 product and cache five official images with provenance. Exclude folded-arm photo from open-arm geometry acceptance.
- [x] Preserve D9CB85EA revision 9 and original GLB; generate separate depth/bevel candidates without changing saved project data.
- [x] Engine 5: localized clearance-limited bevels, explicit depth validation and slimmer buried connectors; preserve approved current outlines across engine-only upgrades.
- [x] Add registered skull and live facial depth mesh, independent head/glasses fit, debug overlays, wireframe, optional soft shadow and bounded per-model calibration persistence.
- [x] Verify 81 Python tests, Node occlusion/calibration tests and TypeScript/Vite production build. Existing Starlette deprecation and bundle-size warnings remain.
- [x] Record baseline/candidate silhouette overlays, unsuccessful parameter sweeps and reference limitations in FIDELITY_OCCLUSION_REPORT.md. Three gpt-6-sol agents handled geometry, AR and reference diagnostics.
- [ ] Reach fidelity acceptance: pinned front-left IoU is 57.75% baseline versus 57.38% candidate; depth changes alone do not solve the mismatch. No 90% claim.
- [ ] Capture matching-angle WebGL and live AR evidence. Local server is unavailable and sandbox port binding is denied; user must start `.venv/bin/python scripts/dev.py` before visual verification can continue.
- [ ] Resolve physical depth and conflicting 142/145 mm temple specifications; validate real-device head motion and performance. Official width is 134 mm, while the saved project retains its existing 140 mm draft setting.

## Surface and sleeve correction — 2026-09-28

- [x] Separate contour defects from shading using the original photo and textured/clay/silhouette GLB renders.
- [x] Identify merged second-arm pixels in the sleeve transition and per-column mask stair steps; previous export omitted normals and stretched edge pixels over depth walls.
- [x] Add bounded automatic front-curve refinement, explicit crease-aware export normals and independent depth-wall texture coordinates; retain original photographed caps and exact construction vertices.
- [x] Verify scale-aware single-arm extraction preserves the genuine sleeve while excluding blocks/spikes; full-size and 100-pixel reference regressions pass.
- [x] Resolve dense-contour cap triangulation/seam rounding and nearest-color padding bands; preserve observed photo interiors.
- [x] Review new suggestions and regenerate saved project aa revision 17/job `5508434d9ae94369b59d3b24945d0285` through UI front/side approvals, preserving photos, measurements/materials and explicit symmetry choice.
- [x] Verify actual export normals, welded construction closure, prior shadow/joint checks, texture/lens preservation and three parent poses; 76 Python tests, production build, two-arm browser flow and simulated-stream AR pass.
- [x] Publish matched photo/clay/silhouette front/side/three-quarter close-ups under `artifacts/surface-fix/`; see `SURFACE_FIX_REPORT.md`. Physical-camera head motion, exact hidden geometry and mobile performance remain unverified.

## Assembly, shadow and AR color correction — 2026-09-28

- [x] Preserve project `aa` revision 11/job `676144e72de843759e3ee5563c7f31e1` and capture its exported GLB, three preview angles and AR before images.
- [x] Confirm causes: capped/offset hinge coordinates, one-sided arm extrusion, side-photo pitch mapped into 3D; fixed low background threshold and 5-pixel closing included contact shadows; AR applied bright PBR lighting to baked photographic colors while inspection bypassed it.
- [x] Add shadow-aware contrast segmentation without silhouette erosion, and foreground-only texture padding for both front and arms. Keep suggestions unapproved and require reanalysis of legacy automatic front outlines.
- [x] Share photo-color material policy across preview/AR, leaving lens appearance independent. Verify actual GLB textures survive loading.
- [x] Correct shoulder attachment and centered arm extrusion; retain explicit existing mirrored-arm assumption and record unverified photo pitch.
- [x] Regenerate project aa through the actual UI; final revision 12/job `1f8a6e3c9d3c4cd68258ed7ca28a9176` preserves the original photos, approved side evidence, size and lens/material settings.
- [x] Verify actual export shadow probes, retained lower rim/upper bar, joint surfaces/root overlap, seven closed outward-oriented meshes and embedded textures. Repair inconsistent front winding/inward lens normals.
- [x] Fix deficient side-cap triangulation without altering the approved outline; real two-arm/inner-view/relabel workflow and all 60 Python tests pass.
- [x] Capture matching preview/rotated/top and AR before/after evidence; shared material/three-pose checks and MediaPipe portrait/synthetic-stream tests pass. See `ASSEMBLY_FIX_REPORT.md` for scope and limitations.
- [ ] Physical-camera head-turn and calibrated product-fit validation remain outside the available evidence; synthetic tracking/pose checks are not substitutes.

## Millimeter-scale recovery — 2026-09-28

- [x] Diagnose the reported 0.2 mm frame: manual scale was preserved, inferred lenses followed it, and the previous 1 mm bridge floor made the dimensions inconsistent.
- [x] Reject implausibly tiny required dimensions with an explicit meters-versus-millimeters message before generation; retain a backend guard for direct jobs.
- [x] Remove the arbitrary bridge floor. Recompute inferred proportions when the frame scale changes; manual edits now have manual provenance.
- [x] Add explicit Use approximate draft sizes (140 mm) recovery in Dimensions. It replaces five size fields locally, preserves photos/materials/annotations, and requires Save before changing the project. No automatic conversion of user-entered values.
- [x] Unit tests cover the exact bad values, non-mutating recovery and updated scale; Python suite now has 47 tests.

## Right/inner views and skipped arm review — 2026-09-28

- [x] Diagnose the actual saved project: side upload labeled Left, no approved arm geometry; front approval bypassed the arm-review check.
- [x] Route all approval-to-generation paths through pending side reviews, including consecutive left and right reviews. Allow explicit reference-only classification when a side cannot be reconstructed.
- [x] Expose Front, Left, Right, Inner/rear and angled upload slots. Add per-photo view reassignment without reuploading; invalidate old approvals when relabeled.
- [x] Support individual Inner left/right side-on photos as arm geometry inputs. Overall Inner/rear stays comparison-only; newest inner/outer capture per arm is selected consistently.
- [x] Fix extraction when the frame faces right and the two arm roots overlap; visually inspect the actual 1080-pixel side upload.
- [x] Add default Photo colors mode for inspection, avoiding studio lighting and tone mapping on photo-textured surfaces. Exported GLB/AR materials remain lit; physical color calibration is not claimed.
- [x] Browser regression verifies front → left → right → export, actual arm meshes, inner-right relabel/export and photo-color toggling. Current-color screenshots in `artifacts/photo-color-preview.png` and `lit-color-preview.png`.
- [x] 46 Python tests pass. Build and the two-arm browser workflow pass using the actual latest side upload; test-created projects are removed recoverably after the run.

## Project removal and side reconstruction — 2026-09-28

- [x] Add per-project Remove with exact name/ID confirmation. Soft-remove from the sidebar; preserve photos, jobs and models. Removed projects list supports restoration after reload.
- [x] Add short IDs to distinguish duplicate saved-project names. No existing user projects were bulk-removed.
- [x] Extract one visible arm from a side photo, excluding the front frame and second visible arm; supply estimated hinge/tip anchors and require visual approval.
- [x] Add side-review panels directly to Photos, with Advanced correction and approve/regenerate controls.
- [x] Support an explicit, default-off symmetry option to mirror a reviewed arm. Preserve the assumption in export warnings and provenance; never mirror silently.
- [x] Visually inspect the provided 100 × 100 side photograph; extracted lower-arm silhouette follows its pixels. Low-resolution detail and attachment uncertainty remain explicit.
- [x] 41 Python tests pass, including remove/restore persistence, two-arm separation, and reviewed/mirrored exports. Build passes.
- [x] Browser checks pass on the real side thumbnail: review/export one arm, opt-in mirroring, cancel/remove/reload/restore. Generate routes an unreviewed side upload to analysis instead of ignoring it. Test-created projects are soft-removed after verification.
- [ ] General side segmentation across all products/angles and accurate hinge/thickness recovery remain unvalidated. A 100-pixel reference cannot establish exact manufacturing geometry.

## Duplicate-arm fix — 2026-09-28

- [x] Remove procedural arm fallback completely; export only approved side geometry and label front-only models incomplete.
- [x] Suggest convex completion for occluded lens apertures, preserving explicit unapproved/reviewed provenance. This is a shape assumption, not exact recovery of hidden edges.
- [x] Detect, edit and triangulate a central bridge opening as an actual solid-frame cutout.
- [x] Extend frame colors across texture boundaries to reduce white background bleeding onto extruded edges.
- [x] Invalidate prior engine exports without deleting historical artifacts; regenerate old extracted outlines rather than silently reusing faulty traces.
- [x] Verify the supplied olive frame visually in WebGL and numerically: clear lens openings, one bridge cutout, only three exported parts (front and two lenses), no invented arms. Saved as **Olive frame — occlusion fix review**.
- [x] 39 Python tests and production build pass; new tests cover occlusion-completion suggestions, actual mesh cutouts and omission of missing arms. Screenshots: `artifacts/occluded-reference/`.
- [ ] Complete product geometry still requires reviewed side evidence. Frame depth, cross-sections, hidden boundaries and physical size remain approximate.

## Merchant-first revision — 2026-09-27

- [x] Generate now starts front-photo analysis and scrolls to the approval panel when geometry or dimensions are missing, instead of displaying a prerequisite dead-end. Missing front photos still get a specific upload request. Approval and approximate-size consent are not bypassed.

- [x] Accept 100 × 100 reference thumbnails with explicit quality warnings (64-pixel minimum); do not upscale them or claim recovered detail.
- [x] Add opt-in approximate sizing when physical dimensions are unavailable: 140 mm placeholder width/arm length, outline-derived front proportions where possible; preserve provided measurements and label provenance.
- [x] Carry approximate-size warnings through the UI and exported metadata. Unit checks cover inference/preservation; 37 Python tests cover thumbnails and export warnings.

- [x] Default to three familiar photo slots; distinguish required front, helpful side/angle, and optional advanced views.
- [x] Move manual outlines and full capture library into Advanced; add automatic front extraction with visual one-click approval/generation.
- [x] Expose frame construction selection and warn that new photos do not change category or dimensions.
- [x] Prevent superseded photos from supplying approved geometry, including legacy projects; invalidate previous same-view approvals on upload.
- [x] Hide stale 3D previews and label historical downloads explicitly; remount viewer for each new artifact.
- [x] Merchant browser regression passes; 35 Python tests pass, including different uploaded silhouettes producing different GLB output and validation photos not replacing fitting photos.
- [ ] Automatic reliable segmentation of arbitrary merchant photographs remains unproven; dark lenses/reflections may require correction. Side/top geometry still requires advanced reviewed outlines; angled photos are comparison evidence only.

- [x] M0 Foundation and contracts — standalone React/Python app, SQLite, separate CPU worker, revision snapshots and startup script
- [x] M1 Public reference dataset — 7 real products across 6 categories; 14 source images; published dimensions and explicit missing views; 6 synthetic fixtures separately labeled
- [x] M2 Guided capture and uploads — twelve overview slots plus details, validated uploads, duplicate/blur notices and fitting/validation roles
- [x] M3 Guided interpretation and correction editor — suggestions, polygon/brush editing, anchors, zoom, undo/redo and persisted approvals; guided front/side/top constraints, not general camera calibration
- [x] M4 Reconstruction engine — six family paths, approved photo outlines, side/top constraints, explicit inferred geometry; numerical checks pass
- [x] M5 Materials, assembly and export — named GLB parts, texture projection, dimensional and topology checks; browser loading passes
- [x] M6 Model review — orbit, wireframe, parts, dimensions, source comparison, evidence warnings and downloads; calibrated accuracy metrics remain unavailable
- [x] M7 AR functionality — MediaPipe tracking, PD calibration, photo/camera modes, loss/recovery and cleanup pass with a simulated stream; physical-device validation remains open
- [x] M8 Integration delivery — production build, 33 Python tests, browser/editor tests, five pose assertions and AR checks; polling race fixed; report and startup documentation complete

Updated 2026-09-27. Checkboxes cover the listed demo functionality, not every original research target or scientific proof of reconstruction fidelity.

## Original-plan targets still open

- [ ] M1 Dataset: two physical products per category and complete guided capture sets; current seven-product public dataset has incomplete views.
- [ ] M3 Calibration: general multi-camera pose/intrinsic/distortion optimization; current fitting assumes guided near-orthogonal captures.
- [ ] M6 Fidelity: calibrated withheld-view IoU ≥0.90 and landmark error ≤2%; no ground-truth accuracy claim yet.
- [ ] M7 Performance: ≥24 FPS live AR on target mobile devices; headless live tests were below target and mobile tests have not been run.
- [ ] M8 Merchant validation: capture/correction time, completion rate, real-camera motion and diverse faces/products.

## Evidence
- Engine 17: water-clear front detection now separates stable lens topology from faint upper-silhouette evidence, with normalized analysis and multi-threshold boundary agreement. Actual-photo regression checks cover 540/1080/2160 pixels; the existing 4096 and GLB reconstruction checks pass. Before/after overlays are in `artifacts/clear-boundary`; see `CLEAR_BOUNDARY_FOLLOWUP.md`. Saved merchant approvals are unchanged; live AR was not verified in this follow-up.
- Python 3.11 isolated environment installed; frontend dependencies installed; production TypeScript/Vite build passes.
- Public sources cached in `data/references/manifest.json` with URLs, dimensions, view labels and missing-evidence notices.
- `artifacts/public-metal-correction-overlay.png` records the visually reviewed lens correction experiment on a real product photograph.
- Tests verify missing geometry raises an actionable correction request, old snapshots stay immutable, input changes mark exports stale, and validation views cannot drive fitting.
- The final reviewed metal-frame trial exported successfully with photo-derived rims and bridge; inferred arms/depth and omitted nose pads remain explicit.
- Browser, editor, AR and benchmark evidence is saved under `artifacts/`; see `VALIDATION_REPORT.md`. The name `ar-hardware-results.json` denotes default Chrome GPU selection, not verified hardware acceleration.
- Three concurrent **gpt-6-sol** agents implemented reconstruction, merchant UI and viewer/AR; primary implemented service/data integration and final fixes/checks.
- All implementation files are in `eyewear-reconstruction-demo`; the Shopify app was not modified.
