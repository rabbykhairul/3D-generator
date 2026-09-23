from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from enum import Enum
from threading import RLock
from typing import Protocol
from uuid import UUID, uuid4

from pydantic import BaseModel, Field
from pydantic.types import JsonValue


class JobStatus(str, Enum):
    ACCEPTED = "accepted"
    VALIDATING = "validating"
    SEGMENTING = "segmenting"
    LANDMARKING = "landmarking"
    DEFORMING = "deforming"
    PROJECTING = "projecting"
    PAINTING = "painting"
    OPTIMIZING = "optimizing"
    UPLOADING = "uploading"
    READY_FOR_REVIEW = "ready_for_review"
    FINALIZED = "finalized"
    FAILED = "failed"
    CANCELLED = "cancelled"


PIPELINE_SEQUENCE = (
    JobStatus.ACCEPTED,
    JobStatus.VALIDATING,
    JobStatus.SEGMENTING,
    JobStatus.LANDMARKING,
    JobStatus.DEFORMING,
    JobStatus.PROJECTING,
    JobStatus.PAINTING,
    JobStatus.OPTIMIZING,
    JobStatus.UPLOADING,
    JobStatus.READY_FOR_REVIEW,
    JobStatus.FINALIZED,
)

PROGRESS_BY_STATUS = {
    JobStatus.ACCEPTED: 0,
    JobStatus.VALIDATING: 5,
    JobStatus.SEGMENTING: 15,
    JobStatus.LANDMARKING: 25,
    JobStatus.DEFORMING: 40,
    JobStatus.PROJECTING: 55,
    JobStatus.PAINTING: 70,
    JobStatus.OPTIMIZING: 85,
    JobStatus.UPLOADING: 95,
    JobStatus.READY_FOR_REVIEW: 100,
    JobStatus.FINALIZED: 100,
    JobStatus.FAILED: 0,
    JobStatus.CANCELLED: 0,
}


class InvalidJobTransition(ValueError):
    pass


class JobNotFound(KeyError):
    pass


class ConcurrentJobUpdate(RuntimeError):
    pass


class DuplicateIdempotencyKey(RuntimeError):
    pass


class JobError(BaseModel):
    code: str
    message: str
    retryable: bool = False


class JobOutputs(BaseModel):
    preview_glb_url: str | None = None
    thumbnail_url: str | None = None
    metadata_url: str | None = None
    final_glb_url: str | None = None


class JobInputs(BaseModel):
    raw: dict[str, str] = Field(default_factory=dict)
    normalized: dict[str, str] = Field(default_factory=dict)


class StageCheckpoint(BaseModel, frozen=True):
    stage: str
    input_digest: str
    implementation_version: str
    attempt: int
    artifacts: dict[str, str] = Field(default_factory=dict)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)
    completed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Job(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    merchant_id: str
    product_sku: str
    frame_width_mm: float
    idempotency_key: str | None = None
    status: JobStatus = JobStatus.ACCEPTED
    progress: int = 0
    attempt: int = 1
    version: int = 1
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    error: JobError | None = None
    inputs: JobInputs = Field(default_factory=JobInputs)
    checkpoints: list[StageCheckpoint] = Field(default_factory=list)
    outputs: JobOutputs = Field(default_factory=JobOutputs)


class JobRepository(Protocol):
    def create(self, job: Job) -> Job: ...

    def get(self, job_id: UUID) -> Job: ...

    def get_by_idempotency_key(self, merchant_id: str, key: str) -> Job | None: ...

    def save(self, job: Job) -> Job: ...

    def healthcheck(self) -> None: ...


class InMemoryJobRepository:
    def __init__(self) -> None:
        self._jobs: dict[UUID, Job] = {}
        self._lock = RLock()

    def create(self, job: Job) -> Job:
        with self._lock:
            if job.id in self._jobs:
                raise ValueError(f"Job {job.id} already exists")
            if job.idempotency_key is not None:
                existing = self.get_by_idempotency_key(job.merchant_id, job.idempotency_key)
                if existing is not None:
                    raise DuplicateIdempotencyKey(job.idempotency_key)
            self._jobs[job.id] = deepcopy(job)
        return deepcopy(job)

    def get(self, job_id: UUID) -> Job:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise JobNotFound(str(job_id))
            return deepcopy(job)

    def get_by_idempotency_key(self, merchant_id: str, key: str) -> Job | None:
        with self._lock:
            for job in self._jobs.values():
                if job.merchant_id == merchant_id and job.idempotency_key == key:
                    return deepcopy(job)
        return None

    def save(self, job: Job) -> Job:
        with self._lock:
            if job.id not in self._jobs:
                raise JobNotFound(str(job.id))
            current = self._jobs[job.id]
            if job.version != current.version:
                raise ConcurrentJobUpdate(str(job.id))
            job.version += 1
            job.updated_at = datetime.now(timezone.utc)
            self._jobs[job.id] = deepcopy(job)
        return deepcopy(job)

    def healthcheck(self) -> None:
        return None


def transition_job(job: Job, target: JobStatus) -> Job:
    if target == JobStatus.FAILED:
        if job.status in {JobStatus.FINALIZED, JobStatus.CANCELLED}:
            raise InvalidJobTransition(f"Cannot fail a {job.status.value} job")
    elif target == JobStatus.CANCELLED:
        if job.status not in set(PIPELINE_SEQUENCE[: PIPELINE_SEQUENCE.index(JobStatus.UPLOADING)]):
            raise InvalidJobTransition(f"Cannot cancel a {job.status.value} job")
    else:
        try:
            current_index = PIPELINE_SEQUENCE.index(job.status)
            target_index = PIPELINE_SEQUENCE.index(target)
        except ValueError as exc:
            raise InvalidJobTransition(f"Cannot transition {job.status.value} to {target.value}") from exc
        if target_index != current_index + 1:
            raise InvalidJobTransition(f"Cannot transition {job.status.value} to {target.value}")

    job.status = target
    job.progress = PROGRESS_BY_STATUS[target]
    job.updated_at = datetime.now(timezone.utc)
    return job


def retry_job(job: Job) -> Job:
    if job.status != JobStatus.FAILED:
        raise InvalidJobTransition(f"Cannot retry a {job.status.value} job")
    job.status = JobStatus.VALIDATING
    job.progress = PROGRESS_BY_STATUS[JobStatus.VALIDATING]
    job.attempt += 1
    job.error = None
    job.updated_at = datetime.now(timezone.utc)
    return job


def resume_interrupted_job(job: Job) -> Job:
    resumable = set(PIPELINE_SEQUENCE[PIPELINE_SEQUENCE.index(JobStatus.SEGMENTING) :]) - {
        JobStatus.READY_FOR_REVIEW,
        JobStatus.FINALIZED,
    }
    if job.status not in resumable:
        raise InvalidJobTransition(f"Cannot resume a {job.status.value} job")
    job.status = JobStatus.VALIDATING
    job.progress = PROGRESS_BY_STATUS[JobStatus.VALIDATING]
    job.attempt += 1
    job.error = None
    job.updated_at = datetime.now(timezone.utc)
    return job
