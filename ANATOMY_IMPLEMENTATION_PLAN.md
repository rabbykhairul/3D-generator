# CPU eyewear anatomy implementation — gemini

## Objective and boundary

Replace an implicit “a mask is a frame” assumption with an explicit, reviewable
eyewear structure. Keep existing CPU extraction and procedural reconstruction;
do not download model weights, add a GPU requirement, or silently invent hidden
parts. This is a structural foundation and broader candidate recovery, not a
claim of universal image recognition or measured depth from photographs.

## Implementation sequence and progress

- [x] Inspect current extraction, contracts, review UI and export paths.
- [x] Define a versioned anatomy graph with part states, relationships, evidence,
      geometry checks and actionable capture/review guidance.
- [x] Add product-local front recovery for off-centre/large-margin photos using
      the existing evidence-based detectors, with conservative background guards.
- [x] Integrate extraction audits and persist detection evidence without treating
      machine suggestions as merchant approval.
- [x] Add project-level cross-view structure reporting to generated metadata;
      distinguish observed, missing and explicitly mirrored parts.
- [x] Surface anatomy findings in front/side/advanced review and export summary.
- [x] Add regression tests for unfamiliar shapes, translations, invalid geometry,
      negative images, manual corrections and API evidence round trips.
- [x] Run focused tests, complete backend suite and frontend production build.
- [x] Document verified results and remaining production acceptance requirements.

## Acceptance criteria

1. A complete front hypothesis requires two separate optical boundaries (or a
   shield) and appropriate construction evidence; two arbitrary blobs are not
   accepted as a full frame.
2. Invalid/out-of-image polygons, overlapping lenses, incompatible views, missing
   side anchors and unapproved/inferred parts remain explicit in reports.
3. Nose pads are reported separately; absence of detection is not proof of absence.
4. Cross-arm appearance does not become selected-arm geometry. Mirroring remains
   an explicit assumption. Rear references are not flattened into front contours.
5. Coordinates survive product-local recovery in original photo pixels.
6. Reports never call a geometric heuristic a trained semantic recognizer or
   calibrated confidence probability. No automatic approval or invisible hardware.
7. Existing regressions and frontend build pass. Synthetic/new-shape tests are
   engineering checks, not an independent commercial unseen-SKU benchmark.

## Production gate after this implementation

Collect independently labelled, held-out SKUs covering clear, rimless, wire,
acetate, shields, coloured backgrounds, low resolution and overlapping arms.
Measure per-part boundary accuracy, construction correctness, manual-review rate,
colour/texture fidelity, geometry validity and CPU latency. A trained/pretrained
CPU part recognizer remains an optional next stage to evaluate against this graph;
its suitability is not established merely by adding an ONNX runtime dependency.

## Verification log

Completed 2026-10-03 on `gemini`, engine version 30. No dependency/model downloads,
GPU service, Shopify integration or changes to merchant projects were made.

- Final complete backend suite: **285 passed**, 158.20 seconds, one existing
  Starlette/httpx deprecation warning. Baseline was 263 tests; 22 were added.
- Focused final integration suite: **52 passed**; schema and generated-metadata
  assertions also passed in a separate 2-test check.
- `npm run build`: TypeScript and Vite production build passed. Existing large
  JavaScript chunk warning remains; it does not establish mobile performance.
- `git diff --check`: passed.
- `scripts/verify_anatomy.py`: 4 fixture diagnostics returned supported structure,
  no blocking geometry findings. Approximate extraction time on this development
  machine: grey front 2.350 s, translated grey front 2.462 s, rimless front
  0.676 s, rimless side 0.518 s. These are single runs, not a deployment p95.
- Translated-photo overlay visually inspected: original coordinates restored;
  both lens areas, outer support, bridge opening and separate dark pads retained.
- Isolated translated-grey GLB generation/load passed: finite vertices, consistent
  outward normals, watertight front parts, width within 1%, and a consistent
  anatomy graph. Both pad meshes were retained; no unverified arm was added.
  Diagnostic output: `artifacts/anatomy/offset-grey-model/`.
- Browser/live-server acceptance **not verified**: starting an isolated API store
  failed with `operation not permitted` while binding localhost port 8018 in this
  restricted session. No permission change or deployment was attempted. API flows
  were verified with an isolated TestClient store instead.

The implementation checklist is complete; the broader production gate above is
not. Universal automatic recognition, hidden mechanical depth, independent
unseen-SKU accuracy and target-server/mobile acceptance remain unestablished.
See `ANATOMY_ARCHITECTURE.md` for the implemented boundaries and API contract.
