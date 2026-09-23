from __future__ import annotations

import io
from pathlib import Path
from uuid import UUID

from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from eyewear_vto.config import Settings
from eyewear_vto.dispatch import DispatchError
from eyewear_vto.jobs import InMemoryJobRepository, Job, JobStatus
from eyewear_vto.main import create_app


class RecordingDispatcher:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.job_ids: list[UUID] = []

    async def dispatch(self, job_id: UUID) -> None:
        self.job_ids.append(job_id)
        if self.fail:
            raise DispatchError("provider unavailable")


class RecordingExecutor:
    def __init__(self, repository: InMemoryJobRepository) -> None:
        self.repository = repository
        self.job_ids: list[UUID] = []

    def run(self, job_id: UUID) -> Job:
        self.job_ids.append(job_id)
        return self.repository.get(job_id)


def upload_files() -> dict[str, tuple[str, bytes, str]]:
    image = Image.new("RGB", (128, 64), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((16, 12, 112, 52), outline="black", width=4)
    output = io.BytesIO()
    image.save(output, format="PNG")
    payload = output.getvalue()
    return {view: (f"{view}.png", payload, "image/png") for view in ("front", "left", "right")}


def settings(tmp_path: Path, *, worker_token: str | None = None) -> Settings:
    return Settings(
        work_root=tmp_path / "work",
        local_storage_root=tmp_path / "storage",
        min_image_long_edge=64,
        worker_token=worker_token,
        _env_file=None,
    )


def test_create_dispatches_persisted_job_id(tmp_path: Path) -> None:
    dispatcher = RecordingDispatcher()
    with TestClient(create_app(settings(tmp_path), dispatcher=dispatcher)) as client:
        response = client.post(
            "/v1/jobs",
            data={"merchant_id": "merchant-1", "product_sku": "sku-1", "frame_width_mm": "140"},
            files=upload_files(),
        )

    assert response.status_code == 202
    assert dispatcher.job_ids == [UUID(response.json()["id"])]


def test_dispatch_failure_is_persisted_as_retryable(tmp_path: Path) -> None:
    dispatcher = RecordingDispatcher(fail=True)
    with TestClient(create_app(settings(tmp_path), dispatcher=dispatcher)) as client:
        response = client.post(
            "/v1/jobs",
            data={"merchant_id": "merchant-1", "product_sku": "sku-1", "frame_width_mm": "140"},
            files=upload_files(),
        )

    assert response.status_code == 202
    assert response.json()["status"] == "failed"
    assert response.json()["error"] == {
        "code": "dispatch_failed",
        "message": "The GPU worker could not be scheduled",
        "retryable": True,
    }


def test_retry_dispatches_the_same_durable_job_again(tmp_path: Path) -> None:
    dispatcher = RecordingDispatcher(fail=True)
    with TestClient(create_app(settings(tmp_path), dispatcher=dispatcher)) as client:
        created = client.post(
            "/v1/jobs",
            data={"merchant_id": "merchant-1", "product_sku": "sku-1", "frame_width_mm": "140"},
            files=upload_files(),
        )
        dispatcher.fail = False
        retried = client.post(f"/v1/jobs/{created.json()['id']}/retry")

    job_id = UUID(created.json()["id"])
    assert retried.status_code == 200
    assert retried.json()["status"] == "validating"
    assert retried.json()["attempt"] == 2
    assert dispatcher.job_ids == [job_id, job_id]


def test_worker_endpoint_requires_token_and_runs_injected_executor(tmp_path: Path) -> None:
    repository = InMemoryJobRepository()
    job = repository.create(
        Job(
            merchant_id="merchant-1",
            product_sku="sku-1",
            frame_width_mm=140,
            status=JobStatus.VALIDATING,
            progress=5,
        )
    )
    executor = RecordingExecutor(repository)
    app = create_app(settings(tmp_path, worker_token="worker-secret"), repository, executor=executor)

    with TestClient(app) as client:
        unauthorized = client.post(f"/v1/worker/jobs/{job.id}/execute")
        executed = client.post(
            f"/v1/worker/jobs/{job.id}/execute",
            headers={"Authorization": "Bearer worker-secret"},
        )

    assert unauthorized.status_code == 401
    assert executed.status_code == 200
    assert executor.job_ids == [job.id]
