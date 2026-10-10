# Source-fidelity work — gemini

Target: high-quality commercial eyewear assets on CPU. Fittingbox equivalence is
an evaluation target, not an achieved capability or a promise from passing tests.

## Colour-consistency follow-up — engine 34

- Solid hex/photo-sampled material colours are decoded from sRGB into linear
  glTF base-colour factors exactly once. Automatic dark-lens colours are no
  longer converted a second time; photographic texture bytes remain sRGB.
- `transparent-full-rim` no longer silently caps requested frame opacity at
  0.55. Explicit alpha and lens tint settings remain unchanged.
- Texture padding now replaces only unobserved pixels, preserving source edge
  highlights/dark pixels that were previously removed by erosion. Inferred
  wall textures still use a low-frequency field; unseen pattern is not recovered.
- Photo-colour inspection and AR bypass relighting/tone mapping for solid
  hardware as well as photo-textured parts. Authored optical surfaces remain
  separate; material mode and external PBR viewers may still relight the export.
- CPU diagnostic factor conversion is updated; it is not a physical renderer.

Six colours round-trip through actual GLB exports within six sRGB code values
(8-bit linear-factor quantization). The old grey `#808080` multiplier decoded to
about RGB 188; the corrected multiplier decodes to about RGB 128 without lighting.
Additional tests check preservation of dark source edges/printed colours and
requested opacity at 1.0, 0.8 and 0.4 through transparent-category exports.

`node scripts/color-policy-offline-test.mjs` executes the actual Three.js policy
without a server/GPU and checks textures, colour factors, hardware, alpha and
optical separation. This does not constitute browser screenshot/device colour
validation or establish a measured 30–40% whole-product improvement.
Verification: **313 backend tests passed** (one existing deprecation warning),
the offline Three.js material-policy checks passed, TypeScript/Vite production
build passed (existing bundle-size warning), and `git diff --check` passed.

Restart, refresh and regenerate the model; existing contours do not need
re-analysis for the colour-only changes. Use Photo colors for source comparison.

## Side overlap and endpiece follow-up — engine 33

The latest burgundy semi-rimless side image contains two touching arms. A merged
lower strip on the main shaft survived contrast consensus: consensus did not
establish physical part ownership. The side trace also stopped at the metal seam,
omitting the visible coloured endpiece beyond it.

- A short occluded shaft edge is now inferred from a stable preceding boundary
  only when a separate lower branch and an upper-edge sleeve step are present.
  The genuine wider sleeve and hook are preserved; single-arm and no-step cases
  are not flattened. This is bounded contour inference, not universal recognition.
- A stable coloured endpiece continuation can be retained beyond the seam,
  with a separate reviewable `frame-attachment` anchor. Neutral/transparent or
  ambiguous continuations decline this recovery rather than inventing hardware.
- Generation seats that visible continuation at the front anchor, offsets the
  arm seam behind it, and projects the original side pixels onto the continuation.
  The small buried connector remains small; it is not stretched into a generic
  endpiece. Endpiece depth uses projected side scale, not measured camera pose.
- Eight new tests cover 540/1080/2160-pixel images, mirroring, negative repairs,
  preservation of the sleeve and actual textured/watertight export (UV seams welded).

Isolated diagnostics (no saved project changes):

```sh
.venv/bin/python scripts/verify_source_fidelity.py PROJECT_ID \
  --reanalyze-sides --output artifacts/side-overlap-fidelity
```

The supplied source's shaft cross-sections at x=440–500 are now about 10–12
pixels wide, versus 18–24 before. The sleeve remains about 21 pixels wide. The
visible endpiece continues to a proposed attachment near x=988, with approximately
12 mm projected depth. Source overlay and matching-side-camera CPU render were
visually inspected. Lens/front geometry, transparent materials, hidden joints
and mirrored arm placement are still approximate; commercial equivalence is not
established. Verification: **305 backend tests passed** (one existing
deprecation warning), followed by **22 joint/export tests passed** after the
final joint-provenance clarification. TypeScript/Vite production build passed
(existing bundle-size warning), and `git diff --check` passed.

Restart and refresh, then re-analyze **both front and side** and review/approve
the new suggestions. Regeneration does not replace the old saved side outline.

## Thin-rim follow-up — engine 32

The reported semi-rimless source was classified correctly, but its geometry was
not: a fixed outer-envelope dilation widened the brow, visible outer tabs were
omitted, and the inferred shoulder attachment differed from the tab position.

- Upper boundaries now follow the sampled source brow instead of its inflated
  envelope; inner offsets follow local contiguous dark-pixel widths.
- Visible tab silhouettes are retained and tab-interior attachment candidates
  are proposed. Their exact pivot and rear mechanism remain inferred.
- Unmeasured thin-metal/semi-rimless/rimless front depth uses a 1.2 mm draft
  prior instead of 4 mm. Transparent-full-rim draft depth is bounded by outline
  clearance (0.8–4 mm); visible width is not a depth measurement. Explicit depth
  measurements are never replaced.
- Original source-space brow checks now constrain six cross-sections to less
  than 8 pixels wide. Previously those sections included uniform dilation.
- Existing mirrored/resized nasal-component tests remain passing.

Isolated source overlay and saved-outline/current-outline matching-camera
diagnostics: `artifacts/thin-rim-fidelity/`. The export geometry checks pass;
visual inspection shows narrower upper rims and retained tabs, but does not
verify complete materials, pads, actual arm spread or commercial equivalence.
No merchant project approvals/data were changed. Re-analysis and approval are
needed to replace the saved outlines; regeneration alone retains old contours.
Verification: **297 backend tests passed** (one existing deprecation warning),
TypeScript/Vite production build passed (existing bundle-size warning), and
`git diff --check` passed. New checks cover source-space thin brow widths, tab
attachment candidates, thin transparent draft depth and measured-depth retention.

## First delivery: reproduce and correct demonstrated omissions

- [x] Confirm latest semi-rimless exports omit both visible nose pads.
- [x] Confirm the viewer initially uses an elevated oblique camera, not the source
      front view; apparent upward arms are not by themselves proof of inverted geometry.
- [x] Add conservative source-supported nasal component recovery across frame
      families, without manufacturing generic pads.
- [x] Make initial inspection use a reproducible front camera.
- [x] Generate isolated before/after models and matching-camera CPU diagnostics.
- [x] Verify source-space contours, pad meshes, existing regressions and build.

## Subsequent production work (not completed by this delivery)

- CPU part-model evaluation with licensed checkpoints and an independent labelled
  product set. A foreground model alone is not an eyewear part recognizer.
- Joint multi-view camera/component fitting with visibility/articulation checks.
- Transparent-material fidelity evaluated in the browser's actual PBR renderer.
- Per-product auto-accept gates for component completeness and visual fidelity,
  with capture/correction requests for failures.
- Independent comparison with reference commercial assets on unseen SKUs.

Do not overwrite merchant annotations or approve detected geometry automatically.
New candidates must be reviewed. Preserve prior engine-30/anatomy branch work.

## First-delivery evidence — engine 31

The latest supplied semi-rimless reference previously exported 7 meshes with no
pad meshes. An isolated current extraction/export now has 9 meshes, including
both source-supported visible nasal components. All export geometry checks pass.
Merchant project data and approvals were not changed. The same recovery also
finds genuine separate clear pads in the black-brow/pale-metal reference.

`tests/test_nose_support.py`: 9 tests passed, including 540/1080/2160 pixels,
mirrored variants, removal of pad evidence, preservation of existing pad contours,
and actual GLB generation. `tests/test_new_catalog_outlines.py`: 3 extraction/API/
export tests passed with the newly retained pad meshes. Production build passed.
Final complete backend suite: **294 passed**, with one existing Starlette/httpx
deprecation warning. `git diff --check` passed.

Reproduce an isolated saved-project comparison with:

```sh
.venv/bin/python scripts/verify_source_fidelity.py PROJECT_ID
```

Outputs are under `artifacts/source-fidelity/`: source-space overlay, before/after
GLBs, identical-front-camera CPU comparison and JSON report. The CPU renderer is
not PBR and cannot verify glass optics, refraction or actual browser appearance.

Visual inspection **did not establish full fidelity**: the added nasal polygons
cover stable visible portions, not the full overlapping pad geometry. Clear
lenses still lack source-like edge highlights and retain approximate material.
The front camera corrects a misleading initial viewing angle; it does not recover
true arm spread, fit or depth. No part model was trained/evaluated and no joint
multi-view geometry optimizer was implemented in this first delivery.

## Learned segmentation evaluation status

MobileSAM is an evaluation candidate, not an installed production dependency.
Its official repository documents prompt-based segmentation and CPU use; this
does not establish automatic eyewear part identification or transparent-boundary
accuracy. Repository license: Apache-2.0. Sources:

- https://github.com/ChaoningZhang/MobileSAM
- https://github.com/ChaoningZhang/MobileSAM/blob/master/LICENSE

The current app environment has no Torch/ONNX segmentation runtime. An isolated
temporary evaluation directory was prepared, but repeat checkpoint-download
checks failed DNS resolution for `raw.githubusercontent.com`. No checkpoint was
downloaded, no inference ran, and no model accuracy or CPU latency was measured.
Next evaluation requires available runtime/checkpoint downloads and labelled
unseen images; results must be recorded before deciding on app integration.

To try these changes, restart the backend and refresh the browser, re-analyze the
front photo, inspect the blue candidate contours, then approve/regenerate. Do not
expect regeneration alone to replace older saved outlines.
