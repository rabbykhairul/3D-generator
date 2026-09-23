from uuid import uuid4

import pytest

from eyewear_vto.jobs import (
    ConcurrentJobUpdate,
    InMemoryJobRepository,
    InvalidJobTransition,
    Job,
    JobNotFound,
    JobStatus,
    retry_job,
    transition_job,
)


def make_job() -> Job:
    return Job(merchant_id="merchant-1", product_sku="sku-1", frame_width_mm=140)


def test_repository_returns_copies() -> None:
    repository = InMemoryJobRepository()
    created = repository.create(make_job())
    created.status = JobStatus.FAILED

    assert repository.get(created.id).status == JobStatus.ACCEPTED


def test_repository_rejects_stale_copy() -> None:
    repository = InMemoryJobRepository()
    job = repository.create(make_job())
    first = repository.get(job.id)
    stale = repository.get(job.id)
    repository.save(first)

    with pytest.raises(ConcurrentJobUpdate, match=str(job.id)):
        repository.save(stale)


def test_repository_raises_for_unknown_job() -> None:
    with pytest.raises(JobNotFound):
        InMemoryJobRepository().get(uuid4())


def test_state_machine_accepts_only_the_next_pipeline_stage() -> None:
    job = make_job()

    transition_job(job, JobStatus.VALIDATING)

    assert job.progress == 5
    with pytest.raises(InvalidJobTransition):
        transition_job(job, JobStatus.DEFORMING)


def test_job_cannot_be_cancelled_after_upload_begins() -> None:
    job = make_job()
    job.status = JobStatus.UPLOADING

    with pytest.raises(InvalidJobTransition):
        transition_job(job, JobStatus.CANCELLED)


def test_failed_job_retry_clears_error_and_increments_attempt() -> None:
    job = make_job()
    job.status = JobStatus.FAILED
    job.error = {"code": "pipeline_failed", "message": "failed"}  # type: ignore[assignment]

    retry_job(job)

    assert job.status == JobStatus.VALIDATING
    assert job.progress == 5
    assert job.attempt == 2
    assert job.error is None
