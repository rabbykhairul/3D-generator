import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend import store
from backend.main import app
from backend.reconstruction.image_resolution import prepare_extraction_image, resolution_diagnostics


@pytest.mark.parametrize(
    "size,expected_mode",
    [((64, 64), "upsampled"), ((100, 75), "upsampled"),
     ((1080, 1080), "source"), ((4096, 2160), "downsampled"),
     ((8000, 1000), "downsampled")],
)
def test_working_raster_is_bounded_and_maps_back_to_original(size, expected_mode):
    width, height = size
    source = np.zeros((height, width, 3), dtype=np.uint8)
    # A source patch away from the image border verifies that image evidence
    # stays in the same region after preprocessing and coordinate conversion.
    x0, x1 = width // 4, width // 2
    y0, y1 = height // 4, height // 2
    source[y0:y1, x0:x1] = (12, 80, 235)
    working, scale_x, scale_y, diagnostics = prepare_extraction_image(source)
    assert diagnostics["mode"] == expected_mode
    assert working.shape[:2] == (diagnostics["working_height"], diagnostics["working_width"])
    assert max(working.shape[:2]) <= 2048
    assert working.nbytes <= 2048 * 2048 * 3
    assert working.shape[1] * scale_x == width
    assert working.shape[0] * scale_y == height
    assert source[y0, x0].tolist() == [12, 80, 235]
    for source_x, source_y in ((0, 0), (width / 2, height / 2), (width, height)):
        working_x = source_x / scale_x
        working_y = source_y / scale_y
        assert working_x * scale_x == pytest.approx(source_x)
        assert working_y * scale_y == pytest.approx(source_y)
    sample_x = round((x0 + x1) / (2 * scale_x))
    sample_y = round((y0 + y1) / (2 * scale_y))
    assert working[sample_y, sample_x].tolist() == [12, 80, 235]
    assert x0 <= sample_x * scale_x <= x1
    assert y0 <= sample_y * scale_y <= y1


def test_low_resolution_diagnostics_do_not_claim_recovered_detail():
    tiny = resolution_diagnostics(64, 64)
    assert tiny["working_width"] >= 80
    assert tiny["source_width"] == 64
    assert any("will not recover missing detail" in note for note in tiny["warnings"])
    assert any("ordinary interpolation" in note for note in tiny["warnings"])
    assert resolution_diagnostics(1080, 1080)["warnings"] == []


def test_extreme_aspect_ratio_stays_bounded_and_reports_limitation():
    diagnostics = resolution_diagnostics(64, 10000)
    assert diagnostics["working_height"] == 2048
    assert diagnostics["working_width"] < 80
    assert any("too narrow" in note for note in diagnostics["warnings"])


def test_upload_preserves_source_pixels_and_reports_working_size(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA", tmp_path / "data")
    monkeypatch.setenv("EYEWEAR_START_WORKER", "0")
    source = Image.new("RGB", (64, 100), "white")
    source.putpixel((20, 30), (7, 31, 201))
    data = io.BytesIO()
    source.save(data, "PNG")
    with TestClient(app) as client:
        project = client.post("/api/projects", json={"name": "Resolution mapping"}).json()
        response = client.post(
            f"/api/projects/{project['id']}/photos",
            data={"view": "front"},
            files={"file": ("small.png", data.getvalue(), "image/png")},
        )
        assert response.status_code == 200
        photo = response.json()["photos"][0]
        assert photo["width"] == 64 and photo["height"] == 100
        assert photo["resolution"]["mode"] == "upsampled"
        assert photo["resolution"]["working_width"] >= 80
        with Image.open(io.BytesIO(client.get(photo["url"]).content)) as saved:
            assert saved.size == (64, 100)
            assert saved.getpixel((20, 30)) == (7, 31, 201)
