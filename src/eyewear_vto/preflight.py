from __future__ import annotations

import importlib
import json

from eyewear_vto.config import PipelineBackend, Settings, get_settings
from eyewear_vto.model_manifest import verify_manifest


def run_preflight(settings: Settings | None = None) -> dict[str, object]:
    settings = settings or get_settings()
    result: dict[str, object] = {"pipeline_backend": settings.pipeline_backend.value}
    if settings.pipeline_backend == PipelineBackend.MOCK:
        result["models"] = "skipped"
        return result

    artifacts = verify_manifest(settings.manifest_path, verify_checksums=False)
    torch = importlib.import_module("torch")
    if not torch.cuda.is_available():
        raise RuntimeError("GPU pipeline requested but CUDA is unavailable")
    result["models"] = [artifact.name for artifact in artifacts]
    result["cuda_device"] = torch.cuda.get_device_name(0)
    return result


def main() -> None:
    print(json.dumps(run_preflight(), sort_keys=True))


if __name__ == "__main__":
    main()
