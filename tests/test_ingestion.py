from __future__ import annotations

import io
import json

import pytest
from fastapi import UploadFile
from PIL import Image, ImageDraw

from eyewear_vto.config import Settings
from eyewear_vto.ingestion import InputValidationError, ingest_uploads, validate_identifier


def image_bytes(size: tuple[int, int] = (128, 64), image_format: str = "PNG") -> bytes:
    image = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((20, 15, 108, 49), outline="black", width=5)
    output = io.BytesIO()
    image.save(output, format=image_format)
    return output.getvalue()


def upload(name: str, content: bytes, content_type: str = "image/png") -> UploadFile:
    return UploadFile(filename=name, file=io.BytesIO(content), headers={"content-type": content_type})


def settings(tmp_path, **overrides: object) -> Settings:
    values: dict[str, object] = {
        "work_root": tmp_path / "work",
        "local_storage_root": tmp_path / "storage",
        "min_image_long_edge": 64,
        "max_image_dimension": 1024,
        "max_upload_bytes": 1024 * 1024,
        "max_request_bytes": 4 * 1024 * 1024,
        "_env_file": None,
    }
    values.update(overrides)
    return Settings(**values)


@pytest.mark.asyncio
async def test_ingestion_uses_safe_names_and_writes_manifest(tmp_path) -> None:
    payload = image_bytes()
    bundle = await ingest_uploads(
        job_id="job-1",
        merchant_id="merchant-1",
        product_sku="sku-1",
        uploads={
            "front": upload("../../front.png", payload),
            "left": upload("left.png", payload),
            "right": upload("right.png", payload),
        },
        settings=settings(tmp_path),
    )

    assert bundle.images["front"].raw_path.name == "front.png"
    assert bundle.images["front"].normalized_path.name == "front.png"
    manifest = json.loads((bundle.workspace / "input_manifest.json").read_text())
    assert set(manifest["images"]) == {"front", "left", "right"}


@pytest.mark.asyncio
async def test_ingestion_rejects_undecodable_content_and_cleans_workspace(tmp_path) -> None:
    config = settings(tmp_path)
    with pytest.raises(InputValidationError, match="not a valid decodable image"):
        await ingest_uploads(
            job_id="job-2",
            merchant_id="merchant-1",
            product_sku="sku-1",
            uploads={
                "front": upload("front.png", b"not an image"),
                "left": upload("left.png", image_bytes()),
                "right": upload("right.png", image_bytes()),
            },
            settings=config,
        )

    assert not (config.work_root / "job-2").exists()


@pytest.mark.asyncio
async def test_ingestion_rejects_per_file_limit(tmp_path) -> None:
    payload = image_bytes() + (b"padding" * 1024)
    with pytest.raises(InputValidationError, match="per-file limit"):
        await ingest_uploads(
            job_id="job-3",
            merchant_id="merchant-1",
            product_sku="sku-1",
            uploads={
                "front": upload("front.png", payload),
                "left": upload("left.png", payload),
                "right": upload("right.png", payload),
            },
            settings=settings(tmp_path, max_upload_bytes=1024),
        )


def test_identifier_validation_rejects_path_characters() -> None:
    with pytest.raises(InputValidationError):
        validate_identifier("../../merchant", "merchant_id", 64)
