# Stable unknown-product reconstruction: implementation tracker

Scope: core CPU-only photo-to-3D engine, no Shopify work or new deployment.
This roadmap supersedes claims that anatomy checks alone solve recognition.
No universal automation or Fittingbox parity has been demonstrated.

## Delivery order and current status

| Step | Deliverable | Acceptance | Status |
| --- | --- | --- | --- |
| 1 | Saved-upload replay and persistent analysis diagnostics | No project/approval mutation; no image transfer; missing boundaries exposed | Implemented; API/storage tests and local replay verified |
| 2 | Real pretrained part-segmenter evaluation on CPU | Real checkpoint inference, licence review, independent labels, per-family/product accuracy and measured latency | Small/medium inference and latency measured on saved photos; independent accuracy and deployment rights pending |
| 3 | Validated detection backend with instance/attachment decoding | Correct front/arm separation, optical topology, measured boundary error, no guessed hardware approval | Experimental hybrid front recovery integrated; independent accuracy and arm/attachment decoding pending |
| 4 | Joint multi-view camera/geometry fitting | Approved corresponding landmarks + physical scale; front/side reprojection and held-out oblique renders pass | Pending; existing registration/extrusion is not this deliverable |
| 5 | Material/optics reconstruction | Lens tint/transmission, frame colour and thin transparent boundaries validated under fixed rendering conditions | Existing colour fixes retained; full optics validation pending |
| 6 | Independent release gate | Frozen development product exclusions, independently labelled unseen families, render/attachment/material tests | Detection integrity gate implemented; independent dataset and render gate pending |

## What was implemented in this increment

- Extraction responses contain `detection_report`: missing required boundaries,
  actual recognizer, sources, correction action and explicit false fidelity/
  automatic-acceptance flags. Suggestions remain unapproved.
- SQLite retains the latest 100 diagnostic records per project, without photo
  copies or revision changes. `GET /api/projects/{id}/detection-runs` only returns
  records for currently attached photos. Archived projects retain their records.
- Front and arm review panels display the report rather than implying that a
  structure check means learned recognition succeeded.
- `scripts/replay_saved_detection.py` inventories/replays saved fitting uploads.
  It excludes validation/reference photos and does not create ground-truth labels.
- `scripts/evaluate_glasses_segmenter.py` evaluates real local checkpoints on CPU
  in a separate environment. It records source-sized binary masks and optional
  independent-mask IoU/false positives/false negatives. This is NOT a production
  backend, 3D generator or certification gate.
- Detection benchmark checks image hashes, independent label provenance and
  product/image overlap against frozen development manifests.

## Candidate and concrete blockers

Candidate: [Glasses Detector](https://github.com/mantasu/glasses-detector).
Its [documented segmenter](https://mantasu.github.io/glasses-detector/_modules/glasses_detector/segmenter.html)
provides frames, lenses and legs masks. Frames include arms; legs are not separate
left/right instances. It supplies neither hinge keypoints nor nose-pad detection.
Do not select it as a universal solution before measuring catalogue-photo accuracy.
Repository licence is MIT; checkpoint/training-data rights still need review for
the selected artifact and deployment use.

The app environment is Python 3.11; this candidate requires 3.12+. On 2026-10-05,
download access worked and a separate Python 3.12 environment was installed under
ignored `artifacts/segmenter-venv`. Actual small/medium checkpoint inference ran
on CPU over 24 unique saved photos from 12 active projects, without inference
errors. This is development evidence, not an independent accuracy benchmark.
The working application environment was not replaced and no trained detector
was connected to production. See the evaluation instructions and decision below.

## Local replay (no resending photos)

```sh
.venv/bin/python scripts/replay_saved_detection.py --limit 10 --output artifacts/detection-inventory.json
.venv/bin/python scripts/replay_saved_detection.py --project PROJECT_ID --replay --output artifacts/detection-replay.json
```

The resulting manifest is explicitly development/unlabelled. A successful
outline proposal is not a fidelity pass; empty extraction is not absence of a part.
Reports contain local file references: keep them private like the uploaded images.

## Candidate evaluation handoff

Use a separate Python 3.12 environment with CPU PyTorch/torchvision matching the
candidate's installation instructions and `glasses-detector==1.0.4`. This optional
evaluation tool does not add PyTorch to the production requirements.
Obtain trusted checkpoints outside request handling; do not load untrusted pickle
weights. Record each exact SHA256, source and licence review in a manifest:

```json
{
  "candidate": {"checkpoints": {
    "lenses": {"path": "weights/lenses.pth", "size": "medium",
      "sha256": "ACTUAL_SHA256", "source_url": "ACTUAL_ARTIFACT_SOURCE",
      "license_review": "REVIEW_RECORD"}
  }},
  "cases": [{"sku": "held-out-product", "view": "front",
    "path": "photos/front.png", "expected_masks": {"lenses": "labels/lenses.png"}}]
}
```

```sh
/path/to/eval-venv/bin/python scripts/evaluate_glasses_segmenter.py candidate.json --output artifacts/candidate-evaluation --threads 2
```

Mask labels are original-photo-sized white-positive/black-negative images drawn
independently, not masks generated by either engine. Reused development products
cannot demonstrate unknown-product performance. Include thin wire, crystal,
rimless, semi-rimless, opaque rims, shields, low-resolution and crossing-arm cases.

## Release decision

### Pretrained CPU evaluation, 2026-10-05

Actual publisher v1.0.0 frames/lenses/legs checkpoints were evaluated through
`glasses-detector==1.0.4`, PyTorch 2.6.0 and torchvision 0.21.0, with two CPU
threads and the documented 256x256 input. Source-sized binary masks, overlays
and report JSON are in `artifacts/segmenter-evaluation/{small,medium}`.
`comparison/index.html` shows the current contours beside both model variants.

Warm per-mask median/p95 prediction time on this Mac was 80.5/187.1 ms for small
and 62.75/77.45 ms for medium. These include preprocessing and source-size output
interpolation, exclude mask/overlay file writing, and are not full reconstruction
latency or server SLAs. The earlier initial run included an implicit torchvision
ImageNet-backbone download. The evaluator now constructs the backbone with
downloads disabled and loads the complete local checkpoint with strict state
matching and `weights_only=True`. Both variants were rerun with socket connections
blocked; all 72 medium masks were byte-identical to the original run.

Visual inspection found more complete medium lens masks than small on the saved
front examples. Both variants inflate thin frame boundaries; arm masks can label
front rims as arms and merge two visible arms with front material in side photos.
The current contour proposals are narrower on some thin/clear examples. Therefore
neither model is accepted as a direct replacement for final geometry extraction.
The medium lens model is a candidate for coarse lens proposals followed by
source-resolution refinement and topology/attachment validation. That hybrid is
integrated as an experimental recovery option below; unknown-product accuracy
has not been demonstrated.

No independent labels were supplied, so there is no measured IoU or accuracy
claim. This set includes development uploads and no independently held-out
products; missing-view/angled-only merchant onboarding is not validated here.
The publisher MIT repository notice is stored alongside weights; distinct
checkpoint/training-data rights for commercial deployment remain unestablished.

Reproduce without changing the application's `.venv`:

```sh
.venv/bin/python -m pip install --target artifacts/eval-tools uv
artifacts/eval-tools/bin/uv venv --python 3.12 artifacts/segmenter-venv
artifacts/eval-tools/bin/uv pip install --python artifacts/segmenter-venv/bin/python -r requirements.segmenter.txt
.venv/bin/python scripts/prepare_segmenter_evaluation.py
artifacts/segmenter-venv/bin/python scripts/evaluate_glasses_segmenter.py artifacts/segmenter-evaluation/small.json --output artifacts/segmenter-evaluation/small --threads 2
artifacts/segmenter-venv/bin/python scripts/evaluate_glasses_segmenter.py artifacts/segmenter-evaluation/medium.json --output artifacts/segmenter-evaluation/medium --threads 2
.venv/bin/python scripts/compare_segmenter_evaluation.py artifacts/segmenter-evaluation
```

On Linux use uv's `--torch-backend cpu` installation option to avoid CUDA wheels.
Setup downloads code/weights only; saved product photographs remain local.
The inventory reads SQLite in read-only mode, excludes archived/reference photos,
deduplicates image hashes and never treats saved approvals as ground truth.

### Independent gate (not achieved)

The optional hybrid backend is now callable from the app's front-analysis selector,
the extraction API (`?backend=hybrid`) or the dev launcher (`--hybrid`). It runs
only the medium lens model in a separate CPU process, with local checksum-checked
weights, a 45-second subprocess limit and one inference slot per API process.
No model downloads happen during requests and PyTorch is not added to the app
environment. Unavailable inference is explicitly reported and retains standard
contours. Non-front views use existing detection; no arm masks or guessed rimless
bridge are substituted.
The median isolated inference request across these saved fronts was 3.2033 seconds,
including process/model startup. This is distinct from the warm per-mask timings
above and is not a server latency guarantee.

Coarse lens boundaries are refined along nearby source colour edges with a cyclic
continuity constraint in the bounded analysis raster (up to 2048 pixels). Weak
edges, large shape changes, clipping, conflicting openings and invalid topology
are rejected. Valid optical boundaries are retained independently when frame
support is unresolved, with a partial report rather than all components uncertain.
A source-observed front must enclose both lenses for complete frame suggestions;
a quiet bright-background contrast consensus can recover connected faint hardware
selected by the lens pair. This uses unfilled source masks and never creates a
frame or bridge around isolated lens predictions. Existing complete source
contours are retained. New hybrid suggestions remain unapproved and report their
provenance; photos/project revisions/approvals are not changed by extraction.

`scripts/verify_hybrid_detection.py` ran actual inference on 12 saved front photos:
one previously empty standard result produced audited front/lens geometry, and
11 complete results were retained. No inference errors occurred. The newly
recovered front exported a GLB with finite geometry, consistent normals and
watertight front/lens geometry after welding coincident UV/normal seams. This
front-only numerical draft used inferred dimensions; it is not exact physical
fit or product fidelity evidence. Reports/overlays/GLB are in ignored
`artifacts/hybrid-validation/`. Both this comparison and the earlier model
evaluation use development products and cannot establish unseen-product accuracy.

UI smoke testing used a separate API/data directory and actual CPU inference. It
verified the experimental selector, model-use report, clearing stale suggestions
when switching backends, desktop/mobile layout and unchanged saved approvals.

The existing polygon benchmark now additionally requires `development_manifests`
(paths to frozen development manifests), nonempty `required_families`, and each
case's `sha256`, `label_source: independent-human`, `label_reviewer` and `expected`.
Its defaults are 20 distinct products per family, 95% whole-product pass rate,
IoU >= 0.9 and boundary error <= 1% of image size. These are detection targets,
not commercial 3D fidelity targets. A manifest assertion cannot prove third-party
training data is disjoint, detect renamed SKUs or prove labels are independent;
dataset-owner review and near-duplicate checking remain necessary.

Do not promote a model based on averages that hide transparent/thin-frame failures.
Do not generate guessed nose pads, hidden hinges or thickness and call them observed.
If real candidate results fail, acquire/licence a catalogue-domain part detector
or train/fine-tune one; training may use a GPU while deployed inference remains CPU.
Faithful unseen surfaces require additional views, measurements or reviewed/CAD
geometry. No image pipeline can establish absent photographic information.

## Verification for this increment (2026-10-04)

- Full backend suite: **320 passed**, one existing Starlette/httpx deprecation warning.
- Frontend production build: passed; existing bundle-size warning remains.
- Offline Three.js colour-policy test: passed, not a pixel-render/device colour test.
- Local replay: six fitting photos in the three latest projects; one front returned
  no safe outlines, five returned review-required suggestions. No fidelity pass is
  implied and saved annotations/approvals were not changed.
- Running the release gate on that unlabelled development inventory correctly
  exited 2. It cannot certify accuracy without independent labels/coverage.
- Candidate script help/contract tests passed. **Actual checkpoint inference did
  not run**; dependencies, trusted weights and download access are missing.
- `git diff --check`: passed. Live browser/render acceptance not performed.

Private local diagnostic outputs: `artifacts/detection-inventory.json`,
`artifacts/detection-replay.json`, `artifacts/detection-development-gate.json`.
They are development artifacts, not an independent test dataset.
