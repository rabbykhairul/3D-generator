from pathlib import Path

import pytest
from pydantic import ValidationError

from eyewear_vto.config import AppEnvironment, PipelineBackend, Settings, StorageBackend
from eyewear_vto.dispatch import create_dispatcher


def test_defaults_are_safe_for_local_development() -> None:
    settings = Settings(_env_file=None)

    assert settings.pipeline_backend == PipelineBackend.MOCK
    assert settings.storage_backend == StorageBackend.LOCAL
    assert settings.manifest_path == Path("/opt/models/model-manifest.json")


def test_production_rejects_mock_pipeline() -> None:
    with pytest.raises(ValidationError, match="PIPELINE_BACKEND=gpu"):
        Settings(app_env=AppEnvironment.PRODUCTION, _env_file=None)


def test_api_keys_parse_from_comma_separated_string() -> None:
    settings = Settings(api_keys="first, second", _env_file=None)

    assert settings.api_keys == ("first", "second")


def test_dispatch_token_is_required_when_url_is_configured() -> None:
    with pytest.raises(ValueError, match="DISPATCH_TOKEN"):
        create_dispatcher("https://gpu.example/jobs", None, 10)


def test_complete_production_contract_is_accepted() -> None:
    settings = Settings(
        app_env=AppEnvironment.PRODUCTION,
        pipeline_backend=PipelineBackend.GPU,
        storage_backend=StorageBackend.R2,
        r2_endpoint_url="https://account.r2.cloudflarestorage.com",
        r2_access_key_id="access",
        r2_secret_access_key="secret",
        r2_bucket="assets",
        api_keys="api-key",
        database_url="postgresql+psycopg://service:secret@database/service",
        dispatch_url="https://gpu.example/jobs",
        dispatch_token="dispatch-secret",
        worker_token="worker-secret",
        _env_file=None,
    )

    assert settings.app_env == AppEnvironment.PRODUCTION
