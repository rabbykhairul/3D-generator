# Validation report

## Outcome

### Latest upload diagnosis and correction — 2026-09-28

The user's latest project had Front, Front left and Left photo labels, but no approved arm outlines. Front approval bypassed the pending-side review gate. All generation entry points now route through current arm-photo reviews before export. A new regression checks front approval → left approval → right approval → GLB, and relabeling Right → Inner right followed by another export. Overall Inner/rear photos are deliberately comparison-only; side-on Inner left/right photos are geometry sources.

The user's actual 1080-pixel side image also exposed a detector failure for right-facing frames with overlapping arm roots. Frame exclusion now handles that case; the resulting outline was visually checked against that upload. Inspection now defaults to unlit photo-texture colors rather than the bright studio-lighting preview. Screenshots `artifacts/photo-color-preview.png` and `lit-color-preview.png` document the difference. This display fix is not calibrated color reconstruction and does not change lighting-dependent GLB/AR materials.

Current checks: **46 Python tests**, production build and two-arm browser workflow pass. The browser regression deliberately places the same supplied side photograph in both side slots to test dispatch and export; it does not establish independent fidelity of both physical arms. Test projects are soft-removed afterward; the user's source project is not relabeled or auto-approved.

### Side workflow and saved projects — 2026-09-28

Current suite: **41 Python tests pass**. Recoverable project removal/restoration preserves revisions, annotations and image bytes. Side extraction now selects one visible arm with estimated hinge/tip anchors; a two-arm regression verifies that the other arm and front frame are excluded. Export tests verify a reviewed arm and an opt-in mirrored counterpart with explicit symmetry provenance. The supplied 100 × 100 side image was visually inspected in `artifacts/side-reference/outline.png`; only its lower arm is selected. Exact hinge location, cross-section and hidden surfaces remain unvalidated.

### Correctness revision — 2026-09-28

The previous implementation duplicated photographed arms inside extruded front geometry and added procedural arms without side evidence. Those were reconstruction defects despite successful workflow tests. Engine version 2 removes fallback arms, marks incomplete reconstruction explicitly, suggests reviewed convex aperture completion and preserves a central bridge cutout. Old engine artifacts are marked stale without deleting them.

The user's olive-frame photograph has been re-extracted, visually reviewed and exported as **Olive frame — occlusion fix review**. WebGL inspection confirms the long invented arms and short duplicated arm segments are absent, and the bridge aperture is open. The export has three parts: front frame and two lenses. `artifacts/occluded-reference/` records source overlays and front/orbit screenshots. **39 Python tests and the production build pass.** Mesh tests verify a real bridge hole and no fallback arms; suggestions remain unapproved until reviewed.

This is an incomplete front-only draft, not a finished faithful 3D product. Convex completion assumes hidden lens boundaries; it can be wrong for concave designs. Color extension reduces background texture bleeding but does not recover measured side materials. The real-photo worker run took about 0.73 seconds and reached 314 MB RSS with texture processing; earlier timing/memory observations below predate this correction.

The standalone demo generates actual GLB geometry from approved photo outlines and dimensions on CPU, and renders that exported model in browser AR. All six frame-family construction paths are implemented. This establishes a working guided reconstruction workflow, **not exact automatic reconstruction of arbitrary eyewear**.

## Verified behavior

Photo-only update: **37 Python tests pass**, including acceptance of 100-pixel thumbnails with quality warnings and persistence of approximate-size warnings in GLB metadata. Approximate-measurement unit checks verify image-proportion inference, default provenance and preservation of supplied values. No physical size is recovered from uncalibrated photos alone. The supplied 1600-pixel front photograph yields three contour suggestions but triggers a multiple-opening warning; its extra bridge aperture and arms visible through the lenses prevent an automatic-exactness claim.

Merchant-first revision: **35 Python tests pass**. New regressions verify that replacing a front upload invalidates previous approvals, legacy approvals cannot select superseded photos, distinct source silhouettes produce distinct GLBs, and held-out photos do not supersede fitting geometry. `artifacts/merchant-results.json` records the simplified browser flow: automatic extraction, preview/approval, actual generation, optional advanced capture, stale-model hiding and an actionable failure on an unusable replacement photo. This does not establish automatic extraction accuracy on arbitrary merchant photos.

- Python coverage includes geometry for six categories, contour sensitivity, physical width checked against exported vertices, closed-solid topology, front/side/top anchors, constrained fitting, validation-photo exclusion, input validation, uploads, brush correction, immutable jobs, persistence and stale artifacts.
- TypeScript checking and Vite production build pass.
- Browser integration checks pass: project creation, actionable missing-evidence rejection, sample import, actual CPU generation, GLB loading in WebGL, wireframe, downloadable GLB bytes, reload and mobile overflow checks.
- Browser editor checks pass: brush correction changes the saved contour, changes invalidate approval, anchors persist, explicit approval persists, invalid measurements are rejected and zero front curvature saves.
- AR checks pass using real MediaPipe inference on a public test portrait, then a simulated camera stream derived from that portrait. Camera denial and tracking loss/recovery are handled; stopping ends media tracks; no face data is uploaded. The user's camera was not accessed.
- Five independent pose-scale assertions confirm yaw compensation prevents double foreshortening and validates PD handling.

Evidence files: `artifacts/browser-results.json`, `editor-results.json`, `ar-results.json`, `ar-hardware-results.json`, `seed-results.json`, `public-metal-result.json`, and corresponding screenshots. The hardware-named file uses default Chrome GPU selection; it does not verify hardware acceleration.

## Real-photo research coverage

Seven public product references were collected with fourteen images. Every frame category is represented, but complete guided capture sets were unavailable.

| Category | References | What has been established |
|---|---|---|
| Opaque full-rim | Zenni 4447121 | Public dimensions and angled photo imported; straight front capture still needed |
| Thin metal | Zenni 137912 | Real front contours visually corrected, generated and exported; angle photo withheld |
| Transparent full-rim | Zenni 4414123, 4440123 | Public front/angled references available; transparent boundary/material fidelity unvalidated |
| Semi-rimless | Zenni 3239016 | Imported public references; complete reconstruction evidence missing |
| Rimless | Zenni 3230911 | Imported public references; complete reconstruction evidence missing |
| Sports/shield | Zenni 1143121 | Dual-lens wraparound reference with front/angle views; full curvature evidence missing |

The metal trial uses actual front image contours with convex aperture corrections to exclude folded arms visible through the lenses. The correction overlay is saved for inspection. Its depth and arms are explicitly inferred; nose pads are omitted without annotated anchors. The 125 mm front assembly width matches the published dimension after a recorded 2.9% horizontal calibration. This is a dimensional construction check, not independent accuracy evidence.

No category has been independently validated against a calibrated multi-view capture or ground-truth 3D model. Therefore the proposed withheld-view IoU ≥0.90 and landmark-error ≤2% targets are **not established**. The interface displays no invented accuracy percentage. Correction time and merchant completion rate have not been benchmarked with merchants.

## Compute observations

On this local development machine, the final warmed CPU construction run for six known-outline fixtures took **0.014–0.082 seconds**, producing **14–61 KB** GLBs. The final corrected metal trial, with finer rims and a photo-derived bridge centerline, took **0.036 seconds**, producing a **66.5 KB** GLB. These timings begin after annotations exist and exclude photo capture, manual correction, reference download and cold dependency initialization. They must not be described as full photo-to-3D generation times.

Observed worker resident memory was approximately **108–125 MB** across runs; the final worker lifetime peak was about **125.5 MB**. This is not a worst-case maximum-resolution upload benchmark. Contour extraction caps its working image at 2048 pixels on the long edge and maps results back to original coordinates.

Headless AR successfully tracks the portrait but runs below the 24 FPS live target. The final low-quality/default-Chrome-selection check sampled **31 render FPS in still-photo mode**, but only **1 render FPS and 1 tracking update/second during the simulated live stream**. These are instantaneous diagnostics, not a standardized hardware benchmark. This does not establish production mobile performance. The UI reports render FPS and tracking updates separately and provides a low-quality mode. Real camera motion, multiple face shapes, occlusion at large angles and mobile hardware remain to be evaluated.

## Implementation limits and next research gate

- The current solver uses front contours, approved side profiles and top-view anchors. Additional oblique/rear photos are review evidence; the app does not yet perform general joint multi-camera bundle adjustment or recover arbitrary camera intrinsics/distortion.
- Photo-derived plates, swept wires, small chamfers and inferred cross-sections approximate manufacturing geometry. Hidden hinges, complex fittings, thickness variation and optical lens behavior are not recovered automatically.
- Transparency, metallic reflections, baked photographic lighting and unobserved textures remain approximations.
- Top/side fitting assumes guided near-orthogonal views. Bad camera angles require recapture or manual correction, not silent perspective flattening.
- The next feasibility gate is a measured physical product set with the twelve-view protocol, withheld calibrated views, merchant correction-time measurement and mobile AR testing. Prototype completion and proof of exact reconstruction are separate outcomes.
