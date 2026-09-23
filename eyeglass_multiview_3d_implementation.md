# Eyeglass Multi-View 3D — Implementation Specification and Delivery Tracker

## 1. Purpose

This document turns `eyeglass_multiview_3d_prd.md` into an executable delivery plan. It is the source of truth for scope, interfaces, acceptance criteria, and implementation status.

The service accepts calibrated eyewear photographs, deforms a controlled CAD template, projects source imagery, uses Hunyuan3D-Paint 2.1 for PBR finishing, produces a VTO-ready GLB, and publishes the result to Cloudflare R2.

## 2. Status rules

| Status | Meaning |
|---|---|
| `NOT STARTED` | No implementation exists. |
| `ONGOING` | Implementation is being changed. |
| `CODE COMPLETE` | Code and non-GPU tests pass, but GPU execution is not verified. |
| `DONE` | Acceptance criteria are verified in the intended environment. |
| `BLOCKED` | An external artifact or decision prevents completion. |

GPU-dependent modules cannot become `DONE` until the baked container passes the server GPU test plan. They may be `CODE COMPLETE` locally.

## 3. Delivery tracker

Update this table in the same change that advances a module.

| ID | Module | Status | Completion evidence | Depends on |
|---|---|---|---|---|
| M00 | Requirements and architecture | `DONE` | This specification defines contracts and acceptance criteria. | — |
| M01 | Service foundation and configuration | `DONE` | Typed settings, health checks, lint, strict mypy, and CPU-only tests pass. | M00 |
| M02 | API, request validation, and job lifecycle | `DONE` | State/API tests pass, including durable merchant-scoped create idempotency and idempotent finalization. | M01 |
| M03 | Input ingestion and image quality gates | `DONE` | Multipart, content, bounds, normalization, cleanup, archival, and API tests pass. | M02 |
| M04 | Storage abstraction and Cloudflare R2 | `DONE` | Local/R2 upload, download, promotion, traversal, and fresh-worker restoration contracts pass. | M01 |
| M05 | SAM 2.1 segmentation adapter | `CODE COMPLETE` | Lazy adapter and artifact/mock contracts pass; GPU quality gate pending M16. | M03, M11 |
| M06 | Eyewear landmark adapter | `BLOCKED` | Adapter contract passes; trained `eyewear_pose.pt` has not been supplied. | M03, M11 |
| M07 | Template catalog, selection, and deformation | `BLOCKED` | Catalog/selection/build validation pass; production CAD templates and deformation anchors are absent. | M06 |
| M08 | Multi-view projection and UV merge | `BLOCKED` | Requires the production UV topology plus approved orthographic/camera calibration before projection can be verified. | M07 |
| M09 | Hunyuan3D-Paint 2.1 PBR and composition | `BLOCKED` | Offline adapter/mock contracts pass; source-albedo/lens composer requires M08 UV artifacts before implementation can be verified. | M08, M11 |
| M10 | GLB optimization, VTO metadata, and QA | `DONE` | Pinned gltfpack/Meshopt optimization, semantic preservation, GLB budgets, and metadata tests pass. | M09 |
| M11 | Baked-weight CUDA container | `BLOCKED` | Pinned bake, offline manifest/preflight, template gate, and static checks exist; a full image cannot build until YOLO/CAD artifacts are supplied. | M01 |
| M12 | GHCR CI/CD | `BLOCKED` | Workflow and Docker definition validate; first push requires the repository plus model variables/secrets and M11 artifacts. | M11 |
| M13 | End-to-end orchestration and failure recovery | `BLOCKED` | Mock E2E, durable cross-replica inputs, authenticated dispatch/execution, retry, cancellation, and failure persistence pass; production assembly and crash-resume checkpoints depend on M07–M09/provider selection. | M02–M10 |
| M14 | Merchant web UI and 3D preview | `CODE COMPLETE` | Bundled upload/status/review UI and static/API integration tests pass; browser GPU fixture pending M16. | M02 |
| M15 | Observability, security, and operations | `BLOCKED` | Tracing, metrics, readiness, shared-key auth, archival, cleanup, and runbook tests pass; merchant-scoped identity/authorization and delivery policy remain undecided. | M13 |
| M16 | Server GPU qualification | `BLOCKED` | Runbook is ready; execution awaits M06–M13 inputs and a 24 GB+ NVIDIA deployment. | M05–M15 |

## 4. Locked architecture decisions

### 4.1 Geometry strategy

Production geometry comes from a versioned, symmetric CAD template. Generative shape weights are excluded from the production image unless a future ADR changes this decision. This avoids geometric hallucination and reduces the baked image size.

### 4.2 Texture strategy

Hunyuan3D-Paint 2.1 PBR is used because the desired output includes albedo and metallic/roughness information. Its complete model snapshot, DINOv2 dependency, and RealESRGAN checkpoint must be available offline inside the final image.

The pinned upstream Paint 2.1 pipeline currently uses only the first reference image for diffusion conditioning even when passed a list. Therefore it is not the authoritative multi-view color fusion stage. M08 must deterministically project all calibrated source views into the template UV layout. M09 may generate seam/PBR candidates, but final composition must restore the approved projected albedo and deterministic lens material after painting. This integration remains blocked until a production template exposes its real UV topology.

### 4.3 Model delivery

All runtime model artifacts are baked into `/opt/models`. Runtime network download is forbidden. The container runs with `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, and `DIFFUSERS_OFFLINE=1`. A startup preflight verifies every artifact in `model-manifest.json` before the service reports ready.

The Docker build may access the internet. GitHub Actions builds the immutable image and pushes `latest` and the commit SHA to GHCR. Model revisions are build arguments and must be pinned to commits, never floating `main` in production.

### 4.4 Execution model

Durable job state lives in PostgreSQL with optimistic version checks so status polling and finalization remain correct across serverless replicas. Validated raw and normalized inputs are written to object storage before a job is accepted, and their object keys are persisted with the job. A GPU worker reconstructs its disposable local workspace from those keys, so processing never depends on the HTTP replica's filesystem. The selected provider integration must dispatch a persisted job ID to one pipeline execution per GPU worker; that adapter remains part of blocked M13. GPU concurrency defaults to one. Production scale comes from additional serverless workers, not multiple simultaneous diffusion jobs on one GPU.

### 4.5 Local development

Local development never imports or initializes CUDA models by default. `PIPELINE_BACKEND=mock` provides deterministic contract testing. `PIPELINE_BACKEND=gpu` is accepted only when model preflight and CUDA readiness pass.

## 5. API contract

### 5.1 Create generation job

`POST /v1/jobs` as `multipart/form-data`:

- `merchant_id`: required, 1–64 URL-safe characters.
- `product_sku`: required, 1–96 URL-safe characters.
- `front`, `left`, `right`: required JPEG, PNG, or WebP.
- `back`, `hero`: optional JPEG, PNG, or WebP.
- `frame_width_mm`: required until camera calibration is implemented; range 90–180 mm.
- `Idempotency-Key`: optional merchant-scoped header, 1–128 characters; a replay returns the original job and conflicting parameters return HTTP `409`.

Response: HTTP `202` with a job resource and polling URL.

### 5.2 Read job

`GET /v1/jobs/{job_id}` returns status, current stage, progress, timestamps, sanitized error information, and output URLs when available.

### 5.3 Finalize job

`POST /v1/jobs/{job_id}/finalize` promotes preview objects to final asset state. It is idempotent and only valid after generation succeeds.

### 5.4 Cancel and retry

- `POST /v1/jobs/{job_id}/cancel` is idempotent and is accepted only before uploading begins. A running worker observes the optimistic-version conflict at its next stage boundary and returns the persisted cancellation.
- `POST /v1/jobs/{job_id}/retry` is accepted only for `FAILED` jobs. It increments `attempt`, clears the prior public error, and returns the job to `VALIDATING` while preserving its durable inputs.

### 5.5 Health endpoints

- `GET /health/live`: process is running.
- `GET /health/ready`: configuration, manifest, storage, and—in GPU mode—CUDA/model readiness pass.

## 6. Job state machine

```text
ACCEPTED
  -> VALIDATING
  -> SEGMENTING
  -> LANDMARKING
  -> DEFORMING
  -> PROJECTING
  -> PAINTING
  -> OPTIMIZING
  -> UPLOADING
  -> READY_FOR_REVIEW
  -> FINALIZED
```

Any processing state can transition to `FAILED`. Retrying creates a new attempt under the same job and preserves diagnostic metadata. Cancellation is allowed before `UPLOADING` and transitions to `CANCELLED`.

## 7. Module specifications

### M01 — Foundation and configuration

Deliverables: typed environment configuration, dependency boundaries, JSON logging, application factory, and startup validation. Secrets are read only at runtime. Acceptance: the API imports and health/liveness runs without Torch or a GPU.

### M02 — API and jobs

Deliverables: versioned FastAPI routes, job state models, repository interface, idempotency support, and sanitized errors. Acceptance: invalid transitions are rejected and all public schemas appear in OpenAPI.

### M03 — Input ingestion

Deliverables: streaming upload limits, MIME sniffing, decoded-image verification, minimum dimensions, edge/subject-presence checks, normalized orientation, and isolated per-job workspaces. Never trust filenames supplied by clients.

Initial limits: 15 MB per image, 60 MB per request, 1024 px minimum on the longest edge, 8192 px maximum dimension. EXIF orientation is normalized and metadata is stripped from derived files.

### M04 — Storage

Deliverables: local and R2 implementations behind one interface. Object keys:

```text
assets/{merchant_id}/{product_sku}/{job_id}/
  raw/{view}.{ext}
  preview/frame_model.glb
  preview/thumbnail_hero.webp
  preview/metadata.json
  final/frame_model.glb
  final/thumbnail_hero.webp
  final/metadata.json
```

R2 credentials are runtime secrets and must not enter the image or build cache.

### M05 — Segmentation

Inputs: normalized images. Outputs: binary mask PNGs and contour diagnostics. Model: SAM 2.1 Hiera Large. The adapter lazily loads the model once per worker and must release intermediate tensors after the stage.

Acceptance targets on the golden set: mean mask IoU ≥ 0.97 for frames, no disconnected loss of either temple in ≥ 95% of valid inputs.

### M06 — Landmarks

Inputs: normalized images and masks. Outputs: named 2D points with confidences. Required landmarks include rim extrema, bridge edges, hinge centers, temple endpoints, and nose-pad centers where present.

The production weight is `/opt/models/yolo/eyewear_pose.pt`. Until the labeled dataset and trained artifact exist, this module remains blocked even if its adapter is code-complete.

### M07 — Templates and deformation

Every template contains semantic submeshes (`frame`, `lens_left`, `lens_right`, `temple_left`, `temple_right`, `hardware`), millimetre units, +Y up, +Z forward, origin at bridge center, UVs, deformation handles, and hinge pivots.

Acceptance targets: bilateral rim width difference < 0.25 mm; supplied frame width within ±1 mm; no self-intersections visible in the golden fixture; lens meshes remain separate.

### M08 — Projection

Calibrated cameras project source pixels into template UV space. Visibility uses z-buffering; view weights use normal angle, landmark confidence, sharpness, and mask distance. It produces an albedo draft, coverage mask, and seam map for PBR finishing.

### M09 — PBR painting

Inputs: deformed GLB, preferred reference image, projected texture/coverage artifacts. Output: textured GLB with albedo and metallic-roughness maps. Lens transmission is applied deterministically after painting; generative output cannot change lens opacity or geometry.

The composer must treat generated albedo as a gap-fill candidate only. Pixels covered by calibrated M08 projection win over generated pixels, while generated metallic/roughness may be retained after material QA. The composer must reapply the original semantic lens nodes and transmission material after Hunyuan returns.

Server target: NVIDIA GPU with at least 24 GB VRAM. Default is six views at 512 resolution. Only one paint job executes per worker.

### M10 — Export and QA

The production GLB uses millimetre-derived metre scale, +Y up, separate lens material, no external URIs, and deterministic VTO metadata. Initial budgets: ≤ 100k triangles, ≤ 20 MB GLB, textures ≤ 2048 px for the web asset. Compression policy is Meshopt plus KTX2 where the target renderer supports it.

Automated QA checks parseability, finite transforms, material assignments, triangle count, bounds, symmetry, size, and required metadata.

### M11 — Container and model manifest

Required offline artifacts:

- SAM 2.1 checkpoint and matching YAML configuration.
- Custom eyewear YOLO pose weight.
- Complete `hunyuan3d-paintpbr-v2-1` snapshot.
- `facebook/dinov2-giant` snapshot used by the paint pipeline.
- `RealESRGAN_x4plus.pth`.
- Versioned CAD templates and deformation metadata.
- Hunyuan custom rasterizer and differentiable renderer compiled during build.

The final image contains no Hugging Face token. A build fails if the manifest cannot be produced or required files are absent.

### M12 — CI/CD

Pushes to `main` and manual dispatch build `linux/amd64`, use registry-backed BuildKit cache, and push to GHCR. The large-image runner frees disk and moves Docker data to `/mnt`. CI never runs GPU inference.

### M13 — Orchestration

Each stage writes an immutable result record. Re-entry skips a completed stage only when its input digest and implementation version match. Errors are classified as client input, capacity, transient infrastructure, or internal pipeline failure.

### M14 — UI

Five guided upload slots, client-side validation, accessible progress updates, retry guidance, Three.js or model-viewer preview, lighting presets, and explicit finalization. Front/left/right are mandatory; hero/back are optional in v1.

### M15 — Operations and security

Runtime API authentication, merchant authorization, request IDs, structured timing per stage, GPU memory metrics, bounded workspaces, cleanup, R2 lifecycle policy, dependency scanning, and an incident/runbook document.

### M16 — Server qualification

Run only after code completion. Verify offline startup with outbound networking disabled, CUDA capability, model loading, one golden job, repeated warm jobs, OOM recovery, cold start, GLB browser rendering, R2 upload, and finalization.

## 8. Model manifest contract

`/opt/models/model-manifest.json` records:

```json
{
  "schema_version": 1,
  "models": [
    {
      "name": "example",
      "source": "repository-or-url",
      "revision": "immutable-revision",
      "path": "/opt/models/example",
      "required": true
    }
  ]
}
```

Production should additionally record SHA-256 for every standalone checkpoint. Directory snapshots use the pinned upstream commit plus a generated file inventory.

## 9. Environment contract

| Variable | Required | Purpose |
|---|---:|---|
| `APP_ENV` | no | `development`, `test`, or `production`. |
| `PIPELINE_BACKEND` | no | `mock` locally; `gpu` on the server. |
| `MODEL_ROOT` | no | Defaults to `/opt/models`. |
| `WORK_ROOT` | no | Per-job scratch root. |
| `MAX_UPLOAD_BYTES` | no | Per-file upload limit. |
| `R2_ENDPOINT_URL` | production | Cloudflare R2 S3 endpoint. |
| `R2_ACCESS_KEY_ID` | production | Runtime secret. |
| `R2_SECRET_ACCESS_KEY` | production | Runtime secret. |
| `R2_BUCKET` | production | Asset bucket. |
| `R2_PUBLIC_BASE_URL` | optional | Public/custom-domain URL prefix. |
| `API_KEYS` | production | Comma-separated initial API key set; replace with an identity provider later. |
| `DATABASE_URL` | production | PostgreSQL URL for durable job state. |
| `DISPATCH_URL` | production | Provider acceptance endpoint receiving the durable job ID. |
| `DISPATCH_TOKEN` | production | Bearer credential for the provider acceptance endpoint. |
| `DISPATCH_TIMEOUT_SECONDS` | no | Short control-plane acceptance timeout; defaults to 10 seconds. |
| `WORKER_TOKEN` | production | Bearer credential protecting the GPU execution endpoint. |

## 10. Verification policy

Local/CI verification is deliberately CPU-only:

1. Formatting, linting, typing, and unit tests.
2. API contract tests using in-memory jobs and local storage.
3. Mock pipeline end-to-end test.
4. Dockerfile/model-manifest static checks.
5. No model loading and no CUDA inference.

GPU verification follows M16 and is the only path for GPU modules to reach `DONE`.

Latest local verification: 62 CPU-only tests, Ruff, and strict mypy pass. The merchant UI bundle builds, and Docker BuildKit's static definition check reports no warnings. No model checkpoint was downloaded, imported, or executed on the development machine.

## 11. Open dependencies

1. A commercially acceptable legal review of Tencent Hunyuan and all transitive model licenses. Hunyuan 3D 2.1's current community license excludes the EU, UK, and South Korea, adds hosted-service disclosure and downstream-terms duties, and requires separate approval above its stated one-million-MAU threshold. Production must enforce an approved service territory or use a differently licensed painter.
2. A labeled eyewear landmark dataset and trained `eyewear_pose.pt`.
3. Production CAD templates with semantic submeshes and deformation handles.
4. A standardized capture/calibration method. V1 requires `frame_width_mm` as a known scale.
5. Serverless provider choice, request timeout model, persistent scratch behavior, and maximum image size.
6. Cloudflare R2 account configuration and public/private delivery policy, plus the merchant identity-to-API-credential mapping. The current shared-key gate authenticates callers but does not enforce tenant isolation.
7. Golden image set and approved reference GLBs for quantitative qualification.

### 11.1 Unblock order

| Order | Needed input or decision | Unblocks | Completion signal |
|---:|---|---|---|
| 1 | Trained, licensed `eyewear_pose.pt` plus its immutable repository revision | M06, M11 | Adapter passes the golden landmark set on the server. |
| 2 | Production template catalog, GLBs, semantic nodes, UVs, deformation handles, and hinge pivots | M07, M08, M11 | `scripts/validate_templates.py` passes the real catalog during the image build. |
| 3 | Approved capture/calibration definition and golden input set | M08, M16 | Projection accuracy and coverage can be measured repeatably. |
| 4 | Serverless GPU provider selection and adapter deployment | M12, M13 | The provider-neutral contract in `docs/serverless-dispatch.md` passes against the chosen platform. |
| 5 | GHCR repository settings, `YOLO_MODEL_REPO`, `YOLO_MODEL_REV`, and build-time `HF_TOKEN` if required | M11, M12 | Commit-SHA image is pushed and its digest recorded. |
| 6 | Runtime PostgreSQL, R2, API-key, and delivery-domain secrets | M13, M15, M16 | Production readiness succeeds without embedding any secret in the image. |
| 7 | Approved Hunyuan license territory/use decision, or an alternate painter | M09, M16 | Product/legal owner records an approved model policy. |

Work resumes in this order. A downstream module may remain code-complete internally, but it is not marked `DONE` while one of its acceptance dependencies is unresolved.

## 12. Change discipline

- Advance only one module to `ONGOING` at a time unless an explicit dependency requires otherwise.
- Mark `CODE COMPLETE` only after its local acceptance checks pass.
- Mark GPU modules `DONE` only with a dated server qualification result.
- Record blockers rather than replacing real integrations with silent mocks.
- Pin code, model, and template revisions for every release image.
