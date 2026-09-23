from fastapi.testclient import TestClient

from eyewear_vto.config import Settings
from eyewear_vto.jobs import InMemoryJobRepository, Job, JobError, JobStatus
from eyewear_vto.main import create_app
from eyewear_vto.storage import LocalObjectStorage, object_prefix


def test_get_job_and_finalize_are_idempotent(tmp_path) -> None:
    repository = InMemoryJobRepository()
    job = Job(
        merchant_id="merchant-1",
        product_sku="sku-1",
        frame_width_mm=140,
        status=JobStatus.READY_FOR_REVIEW,
        progress=100,
    )
    repository.create(job)
    settings = Settings(work_root=tmp_path / "work", local_storage_root=tmp_path / "storage", _env_file=None)
    storage = LocalObjectStorage(settings.local_storage_root)
    source = tmp_path / "asset"
    source.write_bytes(b"asset")
    prefix = object_prefix(job.merchant_id, job.product_sku, str(job.id))
    storage.put_file(source, f"{prefix}/preview/frame_model.glb", "model/gltf-binary")
    storage.put_file(source, f"{prefix}/preview/thumbnail_hero.webp", "image/webp")
    storage.put_file(source, f"{prefix}/preview/metadata.json", "application/json")

    with TestClient(create_app(settings, repository, storage)) as client:
        assert client.get(f"/v1/jobs/{job.id}").status_code == 200
        first = client.post(f"/v1/jobs/{job.id}/finalize")
        second = client.post(f"/v1/jobs/{job.id}/finalize")

    assert first.status_code == 200
    assert first.json()["status"] == "finalized"
    assert first.json()["outputs"]["final_glb_url"].endswith("/final/frame_model.glb")
    assert second.status_code == 200


def test_finalize_rejects_incomplete_job(tmp_path) -> None:
    repository = InMemoryJobRepository()
    job = repository.create(Job(merchant_id="merchant-1", product_sku="sku-1", frame_width_mm=140))
    settings = Settings(work_root=tmp_path / "work", local_storage_root=tmp_path / "storage", _env_file=None)

    with TestClient(create_app(settings, repository)) as client:
        response = client.post(f"/v1/jobs/{job.id}/finalize")

    assert response.status_code == 409


def test_unknown_job_returns_404(tmp_path) -> None:
    settings = Settings(work_root=tmp_path / "work", local_storage_root=tmp_path / "storage", _env_file=None)

    with TestClient(create_app(settings)) as client:
        response = client.get("/v1/jobs/4ea3e7fc-6d46-46de-9853-8f506c20216e")

    assert response.status_code == 404


def test_cancel_and_retry_job_lifecycle(tmp_path) -> None:
    repository = InMemoryJobRepository()
    cancellable = repository.create(
        Job(
            merchant_id="merchant-1",
            product_sku="sku-1",
            frame_width_mm=140,
            status=JobStatus.VALIDATING,
            progress=5,
        )
    )
    failed = repository.create(
        Job(
            merchant_id="merchant-1",
            product_sku="sku-2",
            frame_width_mm=140,
            status=JobStatus.FAILED,
            error=JobError(code="pipeline_failed", message="failed"),
        )
    )
    settings = Settings(work_root=tmp_path / "work", local_storage_root=tmp_path / "storage", _env_file=None)

    with TestClient(create_app(settings, repository)) as client:
        cancelled = client.post(f"/v1/jobs/{cancellable.id}/cancel")
        cancelled_again = client.post(f"/v1/jobs/{cancellable.id}/cancel")
        retried = client.post(f"/v1/jobs/{failed.id}/retry")
        retry_conflict = client.post(f"/v1/jobs/{failed.id}/retry")

    assert cancelled.status_code == 200
    assert cancelled.json()["status"] == "cancelled"
    assert cancelled_again.status_code == 200
    assert retried.status_code == 200
    assert retried.json()["status"] == "validating"
    assert retried.json()["attempt"] == 2
    assert retried.json()["error"] is None
    assert retry_conflict.status_code == 409
