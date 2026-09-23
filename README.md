# Eyewear Multi-View 3D Service

Implementation of the hybrid CAD-template and Hunyuan3D-Paint pipeline described in
[`eyeglass_multiview_3d_prd.md`](eyeglass_multiview_3d_prd.md).

Delivery status, contracts, and acceptance criteria live in
[`eyeglass_multiview_3d_implementation.md`](eyeglass_multiview_3d_implementation.md).

Deployment references:

- [`docs/model-artifacts.md`](docs/model-artifacts.md)
- [`docs/ghcr-build.md`](docs/ghcr-build.md)
- [`docs/serverless-dispatch.md`](docs/serverless-dispatch.md)
- [`docs/server-gpu-validation.md`](docs/server-gpu-validation.md)

## Local development

Local mode does not load CUDA or model weights.

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
PIPELINE_BACKEND=mock .venv/bin/uvicorn eyewear_vto.main:app --reload
```

Run CPU-only checks:

```bash
.venv/bin/ruff check .
.venv/bin/mypy src
.venv/bin/pytest
```

## Production container

The `final` Docker target bakes pinned model snapshots into `/opt/models` and enables offline model loading. It is intended for a Linux AMD64 NVIDIA serverless worker with at least 24 GB VRAM.

Never put R2 credentials or Hugging Face tokens in the image. Build-time model access uses a BuildKit secret; runtime secrets are environment variables.
