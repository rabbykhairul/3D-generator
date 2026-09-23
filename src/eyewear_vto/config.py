from __future__ import annotations

from enum import Enum
from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppEnvironment(str, Enum):
    DEVELOPMENT = "development"
    TEST = "test"
    PRODUCTION = "production"


class PipelineBackend(str, Enum):
    MOCK = "mock"
    GPU = "gpu"


class StorageBackend(str, Enum):
    LOCAL = "local"
    R2 = "r2"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "eyewear-vto-3d"
    app_env: AppEnvironment = AppEnvironment.DEVELOPMENT
    pipeline_backend: PipelineBackend = PipelineBackend.MOCK
    storage_backend: StorageBackend = StorageBackend.LOCAL
    model_root: Path = Path("/opt/models")
    work_root: Path = Path("/tmp/eyewear-vto/work")
    local_storage_root: Path = Path("/tmp/eyewear-vto/storage")
    ui_root: Path = Path("/app/ui")
    model_manifest_path: Path | None = None

    max_upload_bytes: int = Field(default=15 * 1024 * 1024, ge=1024)
    max_request_bytes: int = Field(default=60 * 1024 * 1024, ge=4096)
    min_image_long_edge: int = Field(default=1024, ge=64)
    max_image_dimension: int = Field(default=8192, ge=512)
    gpu_concurrency: int = Field(default=1, ge=1, le=1)
    workspace_max_age_seconds: int = Field(default=24 * 60 * 60, ge=60)
    retain_workspaces: bool = False

    r2_endpoint_url: str | None = None
    r2_access_key_id: str | None = None
    r2_secret_access_key: str | None = None
    r2_bucket: str | None = None
    r2_public_base_url: str | None = None
    api_keys: tuple[str, ...] = ()
    database_url: str | None = None
    dispatch_url: str | None = None
    dispatch_token: str | None = None
    dispatch_timeout_seconds: float = Field(default=10.0, gt=0, le=60)
    worker_token: str | None = None

    @field_validator("api_keys", mode="before")
    @classmethod
    def parse_api_keys(cls, value: object) -> object:
        if isinstance(value, str):
            return tuple(item.strip() for item in value.split(",") if item.strip())
        return value

    @model_validator(mode="after")
    def validate_environment_contract(self) -> Settings:
        if self.max_request_bytes < self.max_upload_bytes * 3:
            raise ValueError("MAX_REQUEST_BYTES must permit at least the three mandatory views")

        if self.app_env == AppEnvironment.PRODUCTION:
            if self.pipeline_backend != PipelineBackend.GPU:
                raise ValueError("Production requires PIPELINE_BACKEND=gpu")
            if self.storage_backend != StorageBackend.R2:
                raise ValueError("Production requires STORAGE_BACKEND=r2")
            required_r2 = {
                "R2_ENDPOINT_URL": self.r2_endpoint_url,
                "R2_ACCESS_KEY_ID": self.r2_access_key_id,
                "R2_SECRET_ACCESS_KEY": self.r2_secret_access_key,
                "R2_BUCKET": self.r2_bucket,
            }
            missing = [name for name, value in required_r2.items() if not value]
            if missing:
                raise ValueError(f"Missing production settings: {', '.join(missing)}")
            if not self.api_keys:
                raise ValueError("Production requires at least one API key")
            if not self.database_url:
                raise ValueError("Production requires DATABASE_URL for durable job state")
            if not self.dispatch_url or not self.dispatch_token:
                raise ValueError("Production requires DISPATCH_URL and DISPATCH_TOKEN")
            if not self.worker_token:
                raise ValueError("Production requires WORKER_TOKEN")

        if self.pipeline_backend == PipelineBackend.GPU and not self.manifest_path.is_absolute():
            raise ValueError("GPU mode requires an absolute model manifest path")
        return self

    @property
    def manifest_path(self) -> Path:
        return self.model_manifest_path or self.model_root / "model-manifest.json"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
