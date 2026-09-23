from pathlib import Path

import pytest

from eyewear_vto.jobs import ConcurrentJobUpdate, DuplicateIdempotencyKey, Job, JobStatus
from eyewear_vto.sql_jobs import SqlJobRepository


def repository(tmp_path: Path) -> SqlJobRepository:
    return SqlJobRepository(f"sqlite:///{tmp_path / 'jobs.sqlite'}", create_schema=True)


def test_sql_repository_persists_job_across_instances(tmp_path: Path) -> None:
    first = repository(tmp_path)
    job = first.create(Job(merchant_id="merchant-1", product_sku="sku-1", frame_width_mm=140))

    second = SqlJobRepository(f"sqlite:///{tmp_path / 'jobs.sqlite'}")
    loaded = second.get(job.id)
    loaded.status = JobStatus.VALIDATING
    saved = second.save(loaded)

    assert saved.version == 2
    assert first.get(job.id).status == JobStatus.VALIDATING
    first.healthcheck()


def test_sql_repository_rejects_stale_write(tmp_path: Path) -> None:
    repo = repository(tmp_path)
    job = repo.create(Job(merchant_id="merchant-1", product_sku="sku-1", frame_width_mm=140))
    first_copy = repo.get(job.id)
    stale_copy = repo.get(job.id)
    first_copy.status = JobStatus.VALIDATING
    repo.save(first_copy)

    stale_copy.status = JobStatus.FAILED
    with pytest.raises(ConcurrentJobUpdate):
        repo.save(stale_copy)


def test_sql_repository_enforces_merchant_scoped_idempotency(tmp_path: Path) -> None:
    repo = repository(tmp_path)
    original = repo.create(
        Job(
            merchant_id="merchant-1",
            product_sku="sku-1",
            frame_width_mm=140,
            idempotency_key="request-1",
        )
    )

    assert repo.get_by_idempotency_key("merchant-1", "request-1") == original
    assert repo.get_by_idempotency_key("merchant-2", "request-1") is None
    with pytest.raises(DuplicateIdempotencyKey):
        repo.create(
            Job(
                merchant_id="merchant-1",
                product_sku="sku-2",
                frame_width_mm=145,
                idempotency_key="request-1",
            )
        )
