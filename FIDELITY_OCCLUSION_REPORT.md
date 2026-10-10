# Fidelity study and AR occlusion — 2026-09-28

## Status: implemented changes; acceptance target NOT achieved

### Latest implementation pass — engine 7

Implemented a second geometry/AR pass after the initial report. **This is not full acceptance:** the 90% target and live-camera visual verification remain open.

Confirmed additional causes and corrections:

- Arm roots used a global front-depth offset even when local structural parts had different rear surfaces. Roots now overlap the actual local rear surface by 0.5 mm; cross-category joint tests cover this.
- Temples were rectangular extruded plates with no edge bevel. Both faces now receive a bounded local bevel (0.22 mm for this product), while keeping approved shaft/sleeve/hook bounds, root overlap and source-photo UVs. It improves the edge cross-section; it does not prove the unseen manufacturing profile.
- The optional AR shadow receiver did not follow the live facial geometry, and the skull receiver did not share its depth mask's front clipping. Both now do. Lenses no longer cast opaque shadow patches over the eyes.
- Pending AR initialization was not cancelled on stop/switch. Abort handling now disposes late assets/trackers and prevents stale sessions.
- Dimensions now explains the unverified default material depth, distinguishes it from front-view rim width, and validates the same 0.8–16 mm input range as generation.
- The startup supervisor previously exited successfully without identifying which child stopped. It now prints the failing service/exit code and returns failure; it still cannot bypass sandbox port restrictions.

Verification: 92 Python tests, dimension checks, AR occlusion/projection/storage checks, and production compilation pass. The real reference photos also passed an isolated **upload → saved reviewed outlines → worker generation → download → GLB reload** regression, retaining both arms, embedded textures and independent lens tint/opacity. This exercises the API without opening a port; it is not fresh automatic outline detection or browser/camera validation. No saved merchant project was modified.

Matching-angle before/after renders are available as [side neutral clay](artifacts/fidelity/cpu-engine7/side-clay.png), [three-quarter photo colors](artifacts/fidelity/cpu-engine7/threequarter-photo.png), and [hinge close-up](artifacts/fidelity/cpu-engine7/hinge-photo.png). These are CPU renders of exported GLBs with a shared camera. Clear lenses are deliberately omitted, and the renderer is not a PBR or AR reference. The side clay view visibly shows the new arm edge transitions; the close-up still shows unverified 4 mm front depth.

The updated scorer preserves non-square image aspect ratios and pins each baseline-fitted camera for every candidate. Exact source/baseline hashes and transforms are stored in `scripts/fidelity_pinned_cameras.json`; [engine-7 results](artifacts/fidelity/multiview-engine7/report.json) record:

| Complete-product view | Baseline | Engine 7 | Unverified 2 mm trial |
|---|---:|---:|---:|
| Front | 74.42% | 74.22% | 75.33% |
| Front-left | 59.56% | 58.95% | 56.29% |
| Side | 63.70% | 63.68% | 53.80% |

These foreground IoUs **do not demonstrate overall fidelity improvement**. A cropped inner close-up and folded-arm photograph are excluded from full-product acceptance. All available complete views have already been used/seen during fitting; there is still no independent held-out acceptance set or calibrated material ground truth. A thinner frame is not automatically a more accurate model.

Reproduce the latest comparison with a new, non-existing output directory:

```sh
.venv/bin/python scripts/evaluate_fidelity_views.py --candidate engine7=artifacts/fidelity/engine7/model.glb --output artifacts/fidelity/new-review
```

Latest files: `backend/reconstruction/engine.py`, `backend/store.py`; `ModelViewer.tsx`, `Measurements.tsx`, `draftMeasurements.ts`, `viewer/occlusion.ts`; geometry/API/real-product/CPU-render regression tests; `scripts/render_fidelity_cpu.py`, multi-view evaluator and camera manifests; `scripts/dev.py`; and the isolated `fidelity-review.html` workbench, which now includes a GLB material/transform verification button. The workbench's actual browser execution remains pending.

The sections below retain the initial engine-5 experiments for comparison; their scores used an earlier camera-fitting protocol and should not be mixed with the table above.

No 90% physical-accuracy claim is supported. The saved project D9CB85EA / aa has not been overwritten with an unverified calibration trial. Its baseline (revision 9) and generated candidates are preserved under `artifacts/fidelity/`.

Browser/camera verification is **blocked**, not passed: the local app server stopped and this session cannot bind localhost ports (`listen EPERM` for Vite 5188 and API 8018). Production compilation and offline tests are available; they do not establish correct live occlusion or shadow appearance. To resume browser verification, run `.venv/bin/python scripts/dev.py` in this directory and keep it running.

## Exact product and source evidence

The original product is [Lenskart Air LA E13069-C6, Gunmetal Grey](https://www.lenskart.com/lenskart-air-la-e13069-c6-eyeglasses.html). The temple inscription and official listing agree on the model/color. Five official images were retrieved with provenance in `data/references/lenskart-la-e13069-c6/manifest.json`.

Four official images are effectively the already-uploaded photographs (RGB mean absolute difference approximately 0.14–0.28). The fifth has **folded arms**. It must not be compared directly with an open-arm model as a geometric accuracy test. Earlier rear-view diagnostic folders are retained but their reports explicitly mark that comparison ineligible.

The official page publishes frame width **134 mm** and size **53–16–145**. The photographed arm reads **53–16–142, M=134**. The temple-length conflict remains unresolved. Physical front depth and hidden hinge hardware are not measured in these sources. The original saved project still uses its existing 140 mm draft width; no calibration changes were silently applied.

## What the geometry changes do

- Engine version 5 replaces global bevel reduction with local clearance-limited bevels and localized relaxation. The narrow top bridge no longer forces the whole rim/endpiece bevel from 0.35 mm down to 0.022 mm.
- Explicit `frame_thickness` values are validated and honored. Missing depth remains a clearly labeled, unverified 4 mm draft, not an automatically detected measurement.
- Buried hinge connectors are slimmer and maintain overlap with the front and arm. Their internals remain approximate; this is not a reconstructed mechanical hinge.
- The generated GLB keeps photo textures, explicit normals, sharp UV seams, closed construction topology and independent lens materials.
- Export version changes mark old models stale, but a geometry-version upgrade alone no longer forces the merchant to retrace already approved current outlines.

Candidates include unchanged-input 4 mm, explicitly inferred depth trials 1.5/2/2.5/3/4 mm, and a separate inscription-size trial. Every trial carries a snapshot and metadata. A trial's visual score is not a measured dimension.

## Quantitative comparisons and unsuccessful iterations

`scripts/score_heldout_silhouette.py` projects non-lens GLB triangles with a pinhole camera. Foreground IoU excludes the white background; blank background cannot inflate the score. Reference masks are threshold-based approximations, not independently annotated ground truth. Transparent lens centers and sufficiently pale shadows are excluded; reflection/shadow ambiguity remains.

The camera was optimized against the reference and then pinned for initial candidate comparisons. This is an optimistic **2D silhouette diagnostic**, not the percentage of correct 3D geometry, material, design or fit.

| Open front-left view, pinned camera | Foreground IoU |
|---|---:|
| Original model | 57.75% |
| Local-bevel/slimmer-joint model, same 4 mm depth | 57.38% |
| 1.5 mm trial | 51.91% |
| 2 mm trial | 53.84% |
| 2.5 mm trial | 55.40% |
| 3 mm trial | 56.37% |

The small silhouette change is not evidence of overall fidelity improvement. Depth reduction alone did not resolve the mismatch. Additional sweeps of front curve, temple spread, width, length and individual arm positions did not improve both tested open views consistently. Search records are in `calibration-sensitivity.json` and `arm-pose-sensitivity.json`.

The near-side reference also exposed an evaluation limitation: allowing yaw near 86°, instead of capping it at 75°, raised its diagnostic IoU to about 65% with unchanged geometry. Camera calibration matters. The scorer now exposes `--yaw-limit 89` for near-side views. Do not compare scores from different camera-fitting procedures as if they were geometry improvements.

Once the front-left image was used to choose trial parameters it became a **calibration view**, not held-out validation. All same-articulation official views now seen/used cannot establish independent generalization. High pixel-proximity coverage under a 2–3 px tolerance is also not a substitute for 90% exact silhouette IoU or physical accuracy.

Evidence: [front-left baseline overlay](artifacts/fidelity/heldout-baseline/overlay.png), [candidate overlay](artifacts/fidelity/heldout-candidate/overlay.png). Orange is reference-only, blue model-only, green overlap. These are CPU silhouette diagnostics, not screenshots from the live AR viewer.

## AR changes

The supplied `head-occluder.glb` and canonical face topology are copied into the standalone demo; the Desktop prototype is unchanged.

- Live 468-vertex facial depth mesh, with 898 triangles; camera pixels are never replaced by this mesh.
- Registered side/back skull depth coverage, independent of the rigid glasses fit.
- Show skull alignment; show live face occluder; show fitted/detected anchors; show glasses wireframe; optional soft frame shadow.
- Separate head width/height, head offsets, glasses scale/offset/depth, PD and skull-depth controls; H×H/G fit readout.
- Calibration is stored locally **per exported model URL** with bounded values and safe recovery from corrupt/unavailable storage. A new GLB starts a new calibration.
- Face loss/reacquisition resets mesh smoothing. Debug overlays do not write depth. Depth-only occluders do not write visible color. Asynchronous load failures and session cleanup dispose assets and release streams.

The skull registration and soft shadow remain approximations pending browser and real-device inspection. Depth occlusion hides **virtual** arms behind the head; it cannot remove real glasses already present in webcam pixels.

## Files and verification

Geometry: `backend/reconstruction/engine.py`, `backend/store.py`, `frontend/src/App.tsx`, `tests/test_bevel_regression.py`.

AR: `frontend/src/components/ModelViewer.tsx`, `viewer.css`, `frontend/src/viewer/occlusion.ts`, `arFit.ts`, their Node tests, and local model assets. Asset provenance is in `THIRD_PARTY.md`.

Validation: `scripts/build-fidelity-candidates.py`, `score_heldout_silhouette.py`, `preserve-fidelity-baseline.py`, `fidelity-capture.mjs`, `tests/test_fidelity_metrics.py`. `frontend/fidelity-review.html` provides a same-camera before/after/2 mm trial viewer with PNG download when Vite is running; it does not mutate projects.

Production TypeScript/Vite build passes (existing large-chunk warning). Geometry tests, supplied-skull registration, face projection under yaw/pitch/roll and mirrored coordinates, stale-mesh reset, and per-model calibration/storage tests pass. Final Python test count is recorded in `PROGRESS.md`. No new live-camera, browser-depth-order or shadow-render success is claimed.

## Remaining acceptance work

1. Restart the local server; inspect actual GLB loading, occlusion depth ordering, overlay alignment, shadow and loss/recovery in browser.
2. Add reliable per-view camera/hinge-state calibration and independent foreground/landmark ground truth. Do not optimize the same image and then call it held out.
3. Obtain verified front-depth/local endpiece dimensions and resolve the 142/145 mm temple specification conflict.
4. Improve geometry against calibrated views; reserve a new same-articulation view for acceptance. The 90% target is still open.
5. Validate physical head turns, device performance and fit separately. A silhouette number does not establish any of these.
