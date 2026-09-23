from __future__ import annotations

import io

from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from eyewear_vto.config import Settings
from eyewear_vto.main import create_app


def image_bytes() -> bytes:
    image = Image.new("RGB", (128, 64), "white")
    draw = ImageDraw.Draw(image)
    draw.ellipse((16, 12, 58, 52), outline="black", width=4)
    draw.ellipse((70, 12, 112, 52), outline="black", width=4)
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def test_create_job_accepts_three_required_views(tmp_path) -> None:
    config = Settings(
        work_root=tmp_path / "work",
        local_storage_root=tmp_path / "storage",
        min_image_long_edge=64,
        _env_file=None,
    )
    payload = image_bytes()
    files = {
        "front": ("front.png", payload, "image/png"),
        "left": ("left.png", payload, "image/png"),
        "right": ("right.png", payload, "image/png"),
    }

    with TestClient(create_app(config)) as client:
        response = client.post(
            "/v1/jobs",
            data={"merchant_id": "merchant-1", "product_sku": "sku-1", "frame_width_mm": "140"},
            files=files,
        )

    assert response.status_code == 202
    assert response.json()["status"] == "validating"
    job = response.json()
    assert set(job["inputs"]["raw"]) == {"front", "left", "right"}
    assert set(job["inputs"]["normalized"]) == {"front", "left", "right"}
    for key in (*job["inputs"]["raw"].values(), *job["inputs"]["normalized"].values()):
        assert (config.local_storage_root / key).is_file()


def test_create_job_rejects_fake_image(tmp_path) -> None:
    config = Settings(
        work_root=tmp_path / "work",
        local_storage_root=tmp_path / "storage",
        min_image_long_edge=64,
        _env_file=None,
    )
    files = {
        "front": ("front.png", b"fake", "image/png"),
        "left": ("left.png", b"fake", "image/png"),
        "right": ("right.png", b"fake", "image/png"),
    }

    with TestClient(create_app(config)) as client:
        response = client.post(
            "/v1/jobs",
            data={"merchant_id": "merchant-1", "product_sku": "sku-1", "frame_width_mm": "140"},
            files=files,
        )

    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "invalid_image"


def test_create_job_replays_idempotency_key_without_duplicate(tmp_path) -> None:
    config = Settings(
        work_root=tmp_path / "work",
        local_storage_root=tmp_path / "storage",
        min_image_long_edge=64,
        _env_file=None,
    )
    payload = image_bytes()

    def files():
        return {
            view: (f"{view}.png", payload, "image/png")
            for view in ("front", "left", "right")
        }

    with TestClient(create_app(config)) as client:
        first = client.post(
            "/v1/jobs",
            headers={"Idempotency-Key": "catalog-import-42"},
            data={"merchant_id": "merchant-1", "product_sku": "sku-1", "frame_width_mm": "140"},
            files=files(),
        )
        replay = client.post(
            "/v1/jobs",
            headers={"Idempotency-Key": "catalog-import-42"},
            data={"merchant_id": "merchant-1", "product_sku": "sku-1", "frame_width_mm": "140"},
            files=files(),
        )
        conflict = client.post(
            "/v1/jobs",
            headers={"Idempotency-Key": "catalog-import-42"},
            data={"merchant_id": "merchant-1", "product_sku": "sku-2", "frame_width_mm": "140"},
            files=files(),
        )

    assert first.status_code == 202
    assert replay.status_code == 202
    assert replay.json()["id"] == first.json()["id"]
    assert conflict.status_code == 409
