from fastapi.testclient import TestClient

from eyewear_vto.config import Settings
from eyewear_vto.main import create_app


def test_liveness_does_not_require_gpu_or_model_imports(tmp_path) -> None:
    settings = Settings(work_root=tmp_path / "work", local_storage_root=tmp_path / "storage", _env_file=None)

    with TestClient(create_app(settings)) as client:
        response = client.get("/health/live")

    assert response.status_code == 200
    assert response.json()["status"] == "live"


def test_readiness_and_metrics_work_in_mock_mode(tmp_path) -> None:
    settings = Settings(work_root=tmp_path / "work", local_storage_root=tmp_path / "storage", _env_file=None)

    with TestClient(create_app(settings)) as client:
        ready = client.get("/health/ready", headers={"x-request-id": "request-123"})
        metrics = client.get("/metrics")

    assert ready.status_code == 200
    assert ready.json()["status"] == "ready"
    assert ready.headers["x-request-id"] == "request-123"
    assert ready.headers["x-content-type-options"] == "nosniff"
    assert metrics.status_code == 200
    assert "eyewear_http_requests_total" in metrics.text
