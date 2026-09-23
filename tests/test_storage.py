from __future__ import annotations

from pathlib import Path

import pytest

from eyewear_vto.storage import LocalObjectStorage, R2ObjectStorage, object_prefix, validate_object_key


class FakeS3Client:
    def __init__(self) -> None:
        self.upload: tuple[str, str, str, dict[str, str]] | None = None
        self.copied: dict[str, object] | None = None
        self.download: tuple[str, str, str] | None = None
        self.head_bucket_name: str | None = None

    def upload_file(self, source: str, bucket: str, key: str, ExtraArgs: dict[str, str]) -> None:  # noqa: N803
        self.upload = (source, bucket, key, ExtraArgs)

    def copy_object(self, **kwargs: object) -> None:
        self.copied = kwargs

    def download_file(self, bucket: str, key: str, destination: str) -> None:
        self.download = (bucket, key, destination)
        Path(destination).write_bytes(b"downloaded")

    def head_bucket(self, *, Bucket: str) -> None:  # noqa: N803
        self.head_bucket_name = Bucket


def test_local_storage_put_copy_and_exists(tmp_path: Path) -> None:
    source = tmp_path / "frame.glb"
    source.write_bytes(b"glb")
    storage = LocalObjectStorage(tmp_path / "objects")

    preview = storage.put_file(source, "assets/m/s/j/preview/frame.glb", "model/gltf-binary")
    final = storage.copy(preview.key, "assets/m/s/j/final/frame.glb")
    restored = storage.get_file(preview.key, tmp_path / "restored" / "frame.glb")

    assert storage.exists(preview.key)
    assert storage.exists(final.key)
    assert restored.read_bytes() == b"glb"
    assert final.url == "/assets/assets/m/s/j/final/frame.glb"


@pytest.mark.parametrize("key", ["../secret", "/absolute", "safe/../../secret"])
def test_object_key_rejects_path_traversal(key: str) -> None:
    with pytest.raises(ValueError):
        validate_object_key(key)


def test_prefix_validates_all_identifiers() -> None:
    assert object_prefix("merchant-1", "sku.1", "job_1") == "assets/merchant-1/sku.1/job_1"
    with pytest.raises(ValueError):
        object_prefix("../../merchant", "sku", "job")


def test_r2_storage_uses_s3_contract(tmp_path: Path) -> None:
    source = tmp_path / "frame.glb"
    source.write_bytes(b"glb")
    client = FakeS3Client()
    storage = R2ObjectStorage(
        endpoint_url="https://example.r2.cloudflarestorage.com",
        access_key_id="access",
        secret_access_key="secret",
        bucket="bucket",
        public_base_url="https://assets.example.com/",
        client=client,
    )

    result = storage.put_file(source, "assets/m/s/j/preview/frame.glb", "model/gltf-binary")
    restored = storage.get_file(result.key, tmp_path / "restored.glb")
    copied = storage.copy(result.key, "assets/m/s/j/final/frame.glb")

    assert client.upload == (
        str(source),
        "bucket",
        "assets/m/s/j/preview/frame.glb",
        {"ContentType": "model/gltf-binary"},
    )
    assert client.copied == {
        "Bucket": "bucket",
        "CopySource": {"Bucket": "bucket", "Key": "assets/m/s/j/preview/frame.glb"},
        "Key": "assets/m/s/j/final/frame.glb",
    }
    assert client.download == ("bucket", result.key, str(restored))
    assert restored.read_bytes() == b"downloaded"
    assert copied.url == "https://assets.example.com/assets/m/s/j/final/frame.glb"
    storage.healthcheck()
    assert client.head_bucket_name == "bucket"
