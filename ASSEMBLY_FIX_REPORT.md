# Eyewear assembly, shadow and AR color correction

Date: 2026-09-28. Scope: standalone `eyewear-reconstruction-demo`, not the Shopify app.

## Reference and preserved baseline

The source of truth is project **aa** (`04863342af864ae0a4cabb8854273c49`): its original 1080px front and right-arm product photos, with the inner and angled captures as comparison evidence. The supplied broken-result screenshots were used only to locate the defects.

Baseline revision 11 / job `676144e72de843759e3ee5563c7f31e1` is preserved, including its original GLB. The existing left arm was an **explicitly approved mirrored copy** of the right-arm trace; that assumption was retained, not silently added. Original photos, physical-size settings and lens/material settings were not replaced.

Corrected project: revision **12**, engine **3**, final job **`1f8a6e3c9d3c4cd68258ed7ca28a9176`**. It is already the current model in the saved project; refresh the app and open **aa → Review / AR**.

## Confirmed causes and corrections

### Hinge and temple construction

- The previous code chose an arbitrary extreme outline vertex, then capped its model-space Y at 7 mm. This moved joints below the actual outer shoulder even when the photo implied a higher attachment.
- Side plates extruded in negative X without centering their cross-section on the joint. Side-photo shaft pitch was copied directly into model-space height.
- The placeholder connector protruded in front of the frame and used the generic blue-gray frame material.
- New inferred anchors use the outer shoulder cross-section. Explicit front anchors retain their height. A triangle-level projected intersection check requires real structural material at the attachment, excluding the lenses. Incorrect anchors now request correction rather than creating floating parts.
- Arm roots are centered on their hinge. A profile-space test requires the root to contain the joint; connectors overlap the frame's actual rear surface and the root, sit behind the front, and sample each product's attachment color.
- The straight shaft's apparent image slope is removed while retaining the ear-hook outline and inverse UV mapping. This is recorded as an **inferred leveling assumption**, not calibrated camera recovery.
- Mesh orientation is corrected before export: the previous beveled front was watertight but had inconsistent winding; lens solids faced inward. Export checks now require consistent outward normals and positive volume, without changing positions or UV indices.
- An additional integration check found a valid side contour for which the filtered Delaunay triangulation omitted a 0.2309 mm² patch. The engine now falls back to boundary-constrained Earcut when cap area is deficient and verifies area completeness, instead of modifying the approved outline or exporting an open arm.

### Shadow-derived lower-rim ledge

- The previous 16-unit Lab background cutoff accepted the pale contact shadow. A 5-pixel morphological close then joined it to the lower rim. Extruding that approved outer contour created the ledge; it was not a viewer shadow.
- For uniform light backgrounds with a clear contrast separation, extraction now uses a per-image adaptive split. This reference selects 76 Lab units. Silhouette erosion/opening is not used, preserving thin dark wires and genuine lower-rim material.
- Front and temple textures now extend only foreground colors into UV boundaries, rather than allowing photographic white background or contact-shadow colors to bleed onto the surfaces.
- Old automatic front outlines require fresh analysis and approval. Existing manual outlines are not silently rewritten. Engine-version staleness prevents showing an old export as the current correction.
- The newly detected contour was visually checked against the original photograph and approved through the actual UI. Pixel probes on the original shadow ledge are checked against the old and new exported front meshes; genuine lower-rim and upper-bar probes must remain material.

### Lightened AR frame

- The GLB retained its embedded front and side textures; loading did not lose references or substitute a white fallback. White base-color factors on textured parts were normal texture multiplication, not missing product color.
- Inspection already used an unlit photograph-color mode. AR instead kept lit PBR frame materials under exposure 1.5, bright hemisphere/directional lights and ACES tone mapping. This re-lit colors that already contained photographic lighting, producing the substantially lighter appearance.
- Inspection and AR now share one material policy for photographed surfaces: retain the map, color factor, vertex-color flag, alpha map, opacity, transparency and depth settings; bypass scene relighting and tone mapping for those surfaces.
- Lens tint/transparency and untextured hardware remain independent GLB materials. No gray-specific color multiplier was introduced; regression fixtures include different colored products.
- The GLB still contains PBR materials and embedded textures. **An unrelated external viewer with its own strong lighting may look different**; the matching photograph-color policy is applied by this app after loading. This is photographic appearance preservation, not measured reflectance recovery.

## Evidence

- [Original front contours, including the shadow](artifacts/assembly-fix/before-front-contours.png) · [Corrected contours on the same original photo](artifacts/assembly-fix/after-front-contours.png)
- [Paired preview views](artifacts/assembly-fix/before-after.png)
- [Additional rotated and top comparisons](artifacts/assembly-fix/angles-before-after.png)
- [AR before](artifacts/ar-photo-before.png) · [AR after](artifacts/assembly-fix/after-ar-photo.png)
- [Final synthetic-camera AR capture](artifacts/assembly-fix/after-ar-live.png)
- [Current export/render checks](artifacts/assembly-fix/after-result.json) · [Additional angle checks](artifacts/assembly-fix/angles-result.json) · [Numerical export checks](artifacts/assembly-fix/export-regression.json)

## Verification scope

The actual workflow is original uploaded photos → analysis → reviewed contours → CPU worker → exported GLB → browser reload → inspection/AR. The same product is used for the before/after comparisons, not a replacement fixture model.

Numerical tests cover joint location/overlap and invalid anchors, thin-material preservation, shadow exclusion and foreground texture padding, material retention, all six supported construction branches, staleness and export. Browser checks cover front/side/rotated/top views, GLB/metadata downloads and absence of page/asset-loading errors.

The top-view comparison also exposed a camera-preset singularity: looking exactly along Y with a Y-up camera could roll the image arbitrarily. Presets now clear residual orbit damping and use a tiny fixed top-view offset. Both baseline and corrected assets were recaptured with that same camera policy; camera roll is not counted as a geometry improvement.

The actual GLB is also loaded into Three.js for material and hierarchy tests. Three mirrored/scaled AR-style parent poses preserve all component-relative transforms. AR tests use **real MediaPipe inference on a public portrait** and a **synthetic webcam stream** for permission denial, tracking, loss/recovery, cleanup and no face-image uploads. They do not exercise physical-camera head turns or constitute a mobile performance benchmark.

Final numerical and workflow results:

- **60 Python tests pass**, including joint guards, triangulation fallback, thin components, multiple product colors, all six construction categories and consistent outward surfaces.
- The actual front-analysis/approval/CPU-generation flow regenerated the saved product; exported GLB reloads with seven named parts, embedded front/temple maps and unchanged transparent lens RGBA values.
- `verify-assembly-export.py` reproduces the original defect at photo pixels `(390,733)` and `(210,735)`, confirms those points are no longer geometry, and retains genuine lower-rim `(270,730)` and thin upper-bar `(500,420)` material.
- The separate two-arm/inner-view/relabel browser workflow passes using the real side photo. That workflow copies the same photo to two slots to test dispatch; it is **not independent left/right design evidence**.
- Dimensions/pose scripts and the production TypeScript/Vite build pass. Remaining tool warnings are a dependency deprecation and the existing large browser bundle warning.
- The final material and AR browser reruns used job `1f8a6e3c9d3c4cd68258ed7ca28a9176`. Headless SwiftShader measured roughly 3–6 render FPS in the final simulated-stream run; this does not meet the mobile FPS target and is not a hardware-camera benchmark.

## Files changed

Runtime:

- `backend/reconstruction/engine.py`: segmentation, safe texture sampling, triangulation, joints, normals, validation, provenance and engine version.
- `backend/store.py`: mark older engine exports stale.
- `frontend/src/App.tsx`, `frontend/src/components/QuickDraft.tsx`: route legacy automatic contours through fresh review.
- `frontend/src/components/ModelViewer.tsx`, `frontend/src/viewer/materialPolicy.ts`: consistent preview/AR photographic appearance.
- `requirements.txt`: NetworkX, used by Trimesh's face-orientation repair.

Verification and documentation:

- `tests/test_shadow_regression.py`, `tests/test_joint_regression.py`, `tests/test_reconstruction.py`.
- `frontend/src/viewer/materialPolicy.test.ts`, `scripts/material-policy-test.mjs`.
- `scripts/assembly-visual-test.mjs`, `scripts/inspect-shadow-fix.py`, `scripts/regenerate-reviewed-project.mjs`, `scripts/verify-assembly-export.py`.
- `IMPLEMENTATION_PLAN.md`, `PROGRESS.md`, `README.md`, this report. Existing AR and two-arm flow scripts are rerun as integration checks.

## Remaining reference limitations

- No measured physical scale was supplied: 140 mm frame width/temple length remain explicitly approximate. Do not use this draft to judge fit.
- Only one independent arm profile is reconstructed; the opposite side is the previously approved symmetry assumption. The overall inner image is not a calibrated full inner-surface reconstruction.
- Hinge pin/barrel internals, frame depth, cross-sections, wrap and true shaft angle are not recoverable exactly from these views. Hidden connectors are conservative geometry, not a manufacturing replica.
- Occluded lens edges use the reviewed convex completion. Lens tint/opacity remain the saved approximation; a photograph against a white background does not establish optical transmission or prescription properties.
- Contrast segmentation is not a universal object/shadow classifier. Transparent, pale, strongly reflective, low-contrast or busy-background products may still require manual outline review.
- Real-camera head motion, target-mobile FPS and calibrated comparison to ground-truth 3D remain unverified.
