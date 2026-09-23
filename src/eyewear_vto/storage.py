from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

import boto3  # type: ignore[import-untyped]

from eyewear_vto.config import Settings, StorageBackend
from eyewear_vto.ingestion import validate_identifier


@dataclass(frozen=True)
class StoredObject:
    key: str
    url: str


class ObjectStorage(Protocol):
    def put_file(self, source: Path, key: str, content_type: str) -> StoredObject: ...

    def get_file(self, key: str, destination: Path) -> Path: ...

    def copy(self, source_key: str, destination_key: str) -> StoredObject: ...

    def exists(self, key: str) -> bool: ...

    def healthcheck(self) -> None: ...


def object_prefix(merchant_id: str, product_sku: str, job_id: str) -> str:
    validate_identifier(merchant_id, "merchant_id", 64)
    validate_identifier(product_sku, "product_sku", 96)
    validate_identifier(job_id, "job_id", 64)
    return f"assets/{merchant_id}/{product_sku}/{job_id}"


def validate_object_key(key: str) -> str:
    path = PurePosixPath(key)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError("Object key must be a relative normalized path")
    return path.as_posix()


class LocalObjectStorage:
    def __init__(self, root: Path, public_base_url: str = "/assets") -> None:
        self.root = root.resolve()
        self.public_base_url = public_base_url.rstrip("/")
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        clean_key = validate_object_key(key)
        path = (self.root / clean_key).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("Object key escapes storage root")
        return path

    def put_file(self, source: Path, key: str, content_type: str) -> StoredObject:
        del content_type
        destination = self._path(key)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        return StoredObject(key=key, url=f"{self.public_base_url}/{key}")

    def get_file(self, key: str, destination: Path) -> Path:
        source = self._path(key)
        if not source.is_file():
            raise FileNotFoundError(key)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        return destination

    def copy(self, source_key: str, destination_key: str) -> StoredObject:
        source = self._path(source_key)
        if not source.is_file():
            raise FileNotFoundError(source_key)
        return self.put_file(source, destination_key, "application/octet-stream")

    def exists(self, key: str) -> bool:
        return self._path(key).is_file()

    def healthcheck(self) -> None:
        if not self.root.is_dir():
            raise OSError(f"Local storage root is unavailable: {self.root}")


class R2ObjectStorage:
    def __init__(
        self,
        *,
        endpoint_url: str,
        access_key_id: str,
        secret_access_key: str,
        bucket: str,
        public_base_url: str | None = None,
        client: Any = None,
    ) -> None:
        self.bucket = bucket
        self.public_base_url = public_base_url.rstrip("/") if public_base_url else None
        self.client = client or boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
            region_name="auto",
        )

    def _url(self, key: str) -> str:
        if self.public_base_url:
            return f"{self.public_base_url}/{key}"
        return f"s3://{self.bucket}/{key}"

    def put_file(self, source: Path, key: str, content_type: str) -> StoredObject:
        clean_key = validate_object_key(key)
        self.client.upload_file(
            str(source),
            self.bucket,
            clean_key,
            ExtraArgs={"ContentType": content_type},
        )
        return StoredObject(key=clean_key, url=self._url(clean_key))

    def get_file(self, key: str, destination: Path) -> Path:
        clean_key = validate_object_key(key)
        destination.parent.mkdir(parents=True, exist_ok=True)
        self.client.download_file(self.bucket, clean_key, str(destination))
        return destination

    def copy(self, source_key: str, destination_key: str) -> StoredObject:
        clean_source = validate_object_key(source_key)
        clean_destination = validate_object_key(destination_key)
        self.client.copy_object(
            Bucket=self.bucket,
            CopySource={"Bucket": self.bucket, "Key": clean_source},
            Key=clean_destination,
        )
        return StoredObject(key=clean_destination, url=self._url(clean_destination))

    def exists(self, key: str) -> bool:
        clean_key = validate_object_key(key)
        try:
            self.client.head_object(Bucket=self.bucket, Key=clean_key)
        except self.client.exceptions.ClientError as exc:
            response_code = str(exc.response.get("Error", {}).get("Code", ""))
            if response_code in {"404", "NoSuchKey", "NotFound"}:
                return False
            raise
        return True

    def healthcheck(self) -> None:
        self.client.head_bucket(Bucket=self.bucket)


def create_storage(settings: Settings) -> ObjectStorage:
    if settings.storage_backend == StorageBackend.LOCAL:
        return LocalObjectStorage(settings.local_storage_root)
    if not all(
        (
            settings.r2_endpoint_url,
            settings.r2_access_key_id,
            settings.r2_secret_access_key,
            settings.r2_bucket,
        )
    ):
        raise ValueError("R2 storage selected without complete credentials")
    assert settings.r2_endpoint_url is not None
    assert settings.r2_access_key_id is not None
    assert settings.r2_secret_access_key is not None
    assert settings.r2_bucket is not None
    return R2ObjectStorage(
        endpoint_url=settings.r2_endpoint_url,
        access_key_id=settings.r2_access_key_id,
        secret_access_key=settings.r2_secret_access_key,
        bucket=settings.r2_bucket,
        public_base_url=settings.r2_public_base_url,
    )
