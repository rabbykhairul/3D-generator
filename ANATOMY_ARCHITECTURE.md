# Eyewear structure, not a million SKU templates

Engine 30 defines a shared anatomy contract around two optical areas or one
shield, their support, bridges/openings/hardware, hinge attachments, individual
temple arms and separate nose-pad evidence. Arbitrary contours remain freeform:
round, rectangular and unfamiliar shapes do not need their own template asset.
Colour does not select a shape template. Existing finish-aware CPU detectors
still provide the candidate contours.

## Processing path

1. Preserve source pixels and prepare the existing scale-aware detection image.
2. Run current view-specific front, rimless, clear-frame or single-arm extraction.
3. If front evidence is incomplete, try a guarded product-local crop on quiet
   bright studio backgrounds. Reuse paired-optics/connected-support detectors.
   Restore all contour/anchor coordinates to original pixels. Never crop a clipped
   object into an apparently complete silhouette or connect two blobs artificially.
4. Audit candidates against the anatomy graph: polygon validity, bounds, duplicate
   parts, optical separation, rim containment, bridge-hole separation, view
   compatibility and connected hinge/tip anchors.
5. Show part states and targeted guidance. Suggestions are unapproved. The review
   UI re-audits current geometry before approval. A manual edit clears obsolete
   detection evidence; backend audits never trust persisted evidence as validation.
6. Use the existing approved-evidence procedural generator and material pipeline.
   Export an aggregate graph from active approved fitting annotations. Existing
   reconstruction checks remain authoritative; the additional graph check is
   recorded as `checks.anatomy_geometry_consistent` with warnings on conflicts.
   It is not an independent 3D metrology or commercial certification result.

## Evidence states

| State | Meaning |
| --- | --- |
| observed | A valid traced silhouette or anchor is present; still inspect it. |
| inferred | Explicit symmetry, occlusion completion, or a connection implied by support. |
| uncertain | Evidence is missing or conflicting; absence is not established. |
| not-applicable | An alternative topology applies, e.g. paired lenses versus a shield. |
| not-observed-in-view | This photo is not suitable evidence for that component. |

States are not confidence probabilities. An integrated nose rest cannot be
distinguished from an invisible separate pad simply because pad extraction failed.
Hinge anchors locate attachment; they do not establish internal mechanical design.
Shaft/bend/tip and inner/outer surfaces are part of the arm ontology, but a contour
alone does not measure their hidden depth or independently segment every region.
Cross-arm appearance candidates remain separate from selected-arm geometry; their
actual application remains governed by the existing surface-registration pipeline.

## API and persistence

- `GET /api/anatomy/schema`: versioned ontology and structural rules.
- `POST /api/projects/{id}/photos/{photo_id}/extract`: existing suggestions plus
  `anatomy`, per-annotation `evidence`, and optional `product_local` diagnostics.
- `POST /api/projects/{id}/photos/{photo_id}/anatomy`: read-only audit of
  `{ "annotations": [...], "category": "thin-metal" }`. Accepts only that photo's
  coordinates. Does not approve, save, change revision or start a job.
- Generated `metadata.json` includes `anatomy` alongside `multiview`. The former
  describes structural evidence; the latter reports photo pixels actually used.

Old annotations without `evidence` remain accepted. Validation images and
unapproved annotations cannot become modeled anatomy. Explicit mirrored arms
remain inferred even after merchant approval. Engine version 30 marks older
exports stale without discarding photos or approvals. Re-analyze to obtain new
candidate recovery; regeneration alone does not replace saved contours.

## What is still not solved

This is **not universal automatic recognition**. A geometric graph can reject
inconsistent candidates but cannot reveal invisible rims, restore low-resolution
detail, identify every material, recover hidden hardware or distinguish all
non-eyewear lookalikes. Product-local recovery currently requires a quiet light
studio background. Unsupported photos still require better capture or correction.

A next recognizer can supply part candidates through this same contract, whether
CPU ONNX inference or another independently evaluated method. No new model weights,
GPU dependency, external AI service or model-training claim was introduced here.
Before a commercial auto-accept workflow, run independently labelled held-out
SKUs through the existing benchmark, measure correction rate and 3D/appearance
fidelity, and validate runtime on the actual production CPU. Synthetic unfamiliar
shapes and transformations are regression tests, not proof of unseen-SKU coverage.

Run `.venv/bin/python scripts/verify_anatomy.py` for isolated front, translated
front and rimless front/side overlays, graph JSON and timing diagnostics under
`artifacts/anatomy/`. It does not edit merchant projects or their approvals.
