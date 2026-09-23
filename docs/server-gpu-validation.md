# Server GPU qualification runbook

This is the M16 acceptance runbook. Do not run it on a developer laptop.

## Prerequisites

- Linux AMD64 NVIDIA worker with at least 24 GB VRAM.
- NVIDIA Container Toolkit configured.
- The GHCR image built from an immutable commit tag, not `latest`.
- Runtime R2 credentials and API key supplied as environment secrets.
- Outbound networking disabled for the actual offline test.
- Approved golden input set containing front/left/right and, where available, back/hero images plus physical frame width.

## 1. Pull and inventory

```bash
docker pull ghcr.io/OWNER/REPOSITORY:COMMIT_SHA
docker image inspect ghcr.io/OWNER/REPOSITORY:COMMIT_SHA
```

Record image digest, compressed/uncompressed size, host GPU model, driver, available VRAM, and timestamp in the qualification report.

## 2. Offline model and CUDA preflight

Run with network disabled. This must finish without attempting a download:

```bash
docker run --rm --gpus all --network none \
  -e APP_ENV=development \
  -e PIPELINE_BACKEND=gpu \
  -e STORAGE_BACKEND=local \
  ghcr.io/OWNER/REPOSITORY:COMMIT_SHA \
  python -m eyewear_vto.preflight
```

Expected: JSON containing all five model families and the CUDA device name. Any Hugging Face connection attempt, missing artifact, or CPU fallback is a failure.

## 3. Production startup

Supply secrets through the serverless provider, never through image build arguments:

```text
APP_ENV=production
PIPELINE_BACKEND=gpu
STORAGE_BACKEND=r2
R2_ENDPOINT_URL=...
R2_ACCESS_KEY_ID=...
R2_SECRET_ACCESS_KEY=...
R2_BUCKET=...
R2_PUBLIC_BASE_URL=...
API_KEYS=...
DATABASE_URL=postgresql+psycopg://...
DISPATCH_URL=...
DISPATCH_TOKEN=...
WORKER_TOKEN=...
```

Start one worker and verify `/health/live` and `/health/ready` before submitting a job. Before this step can run, M13 must add the selected provider's execution entrypoint and wire persisted job IDs to it; job creation is intentionally not an in-process background task.

## 4. Golden job

Submit the fixed golden fixture and record:

- cold-start time;
- each stage duration;
- peak GPU memory;
- total job duration;
- resulting object keys and SHA-256;
- GLB bytes, triangle count, texture dimensions, and material count;
- mask and landmark quality metrics;
- frame width and bilateral symmetry errors;
- screenshots under Studio, Outdoor, and Indoor lighting.

The job must reach `READY_FOR_REVIEW`. Finalization must create the three `final/` objects and be idempotent.

## 5. Warm and recovery tests

1. Run three sequential warm jobs; no model reload or additional download is allowed.
2. Submit an invalid image set; it must fail as a client error without GPU work.
3. Interrupt one job during painting; the worker must return a persisted `FAILED` state and remain healthy.
4. Submit a second job after failure; it must complete without restarting the worker.
5. Confirm the single-GPU concurrency limit prevents simultaneous paint passes.
6. Restart the container and verify no finalized R2 asset is lost.

## 6. Acceptance gates

- No outbound model fetch at startup or inference.
- No OOM on the chosen serverless GPU profile.
- Frame-mask mean IoU ≥ 0.97 on the approved golden set.
- Bilateral rim width difference < 0.25 mm.
- Requested frame width within ±1 mm when calibrated input requirements are met.
- Web GLB ≤ 20 MB and ≤ 100k triangles.
- GLB renders without external resources or browser console errors.
- Lens meshes remain separate and transparent; frame/hardware PBR remains opaque.
- Pixels covered by calibrated multi-view projection retain the source-view albedo after the PBR pass.
- All output and legal-territory requirements have product-owner approval.

Only after a dated report passes every applicable gate should M05, M09, M11, M12, M13, and M16 be advanced to `DONE`.
