import json
from pathlib import Path

import pytest

from eyewear_vto.config import PipelineBackend, Settings
from eyewear_vto.model_manifest import ModelManifestError, sha256_file, verify_manifest
from eyewear_vto.preflight import run_preflight


def test_manifest_verifies_required_file_and_checksum(tmp_path: Path) -> None:
    weight = tmp_path / "weight.pt"
    weight.write_bytes(b"weight")
    manifest = tmp_path / "model-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "models": [
                    {
                        "name": "weight",
                        "source": "test",
                        "revision": "abc",
                        "path": str(weight),
                        "required": True,
                        "sha256": sha256_file(weight),
                    }
                ],
            }
        )
    )

    artifacts = verify_manifest(manifest, verify_checksums=True)

    assert artifacts[0].name == "weight"


def test_manifest_rejects_missing_artifact(tmp_path: Path) -> None:
    manifest = tmp_path / "model-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "models": [
                    {
                        "name": "missing",
                        "source": "test",
                        "revision": "abc",
                        "path": str(tmp_path / "missing.pt"),
                        "required": True,
                    }
                ],
            }
        )
    )

    with pytest.raises(ModelManifestError, match="missing"):
        verify_manifest(manifest)


def test_mock_preflight_skips_model_and_cuda_checks(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = Settings(pipeline_backend=PipelineBackend.MOCK, _env_file=None)
    monkeypatch.setattr("eyewear_vto.preflight.get_settings", lambda: settings)

    assert run_preflight() == {"pipeline_backend": "mock", "models": "skipped"}
