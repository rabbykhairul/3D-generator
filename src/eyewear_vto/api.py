from __future__ import annotations

import secrets
from pathlib import Path
from typing import Annotated, cast
from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, Request, UploadFile, status

from eyewear_vto.config import AppEnvironment, Settings
from eyewear_vto.dispatch import DispatchError, JobDispatcher, JobExecutor
from eyewear_vto.ingestion import InputValidationError, ingest_uploads
from eyewear_vto.jobs import (
    DuplicateIdempotencyKey,
    InMemoryJobRepository,
    InvalidJobTransition,
    Job,
    JobError,
    JobInputs,
    JobNotFound,
    JobRepository,
    JobStatus,
    retry_job,
    transition_job,
)
from eyewear_vto.storage import ObjectStorage, object_prefix

router = APIRouter(prefix="/v1")


def get_repository(request: Request) -> JobRepository:
    return cast(JobRepository, request.app.state.job_repository)


def get_storage(request: Request) -> ObjectStorage:
    return cast(ObjectStorage, request.app.state.storage)


def get_dispatcher(request: Request) -> JobDispatcher:
    return cast(JobDispatcher, request.app.state.dispatcher)


def authorize(request: Request, x_api_key: Annotated[str | None, Header()] = None) -> None:
    settings = cast(Settings, request.app.state.settings)
    if settings.app_env == AppEnvironment.PRODUCTION and x_api_key not in settings.api_keys:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")


@router.post(
    "/jobs",
    response_model=Job,
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(authorize)],
)
async def create_job(
    request: Request,
    repository: Annotated[JobRepository, Depends(get_repository)],
    storage: Annotated[ObjectStorage, Depends(get_storage)],
    dispatcher: Annotated[JobDispatcher, Depends(get_dispatcher)],
    merchant_id: Annotated[str, Form(min_length=1, max_length=64)],
    product_sku: Annotated[str, Form(min_length=1, max_length=96)],
    frame_width_mm: Annotated[float, Form(ge=90, le=180)],
    front: Annotated[UploadFile, File()],
    left: Annotated[UploadFile, File()],
    right: Annotated[UploadFile, File()],
    back: Annotated[UploadFile | None, File()] = None,
    hero: Annotated[UploadFile | None, File()] = None,
    idempotency_key: Annotated[
        str | None,
        Header(alias="Idempotency-Key", min_length=1, max_length=128),
    ] = None,
) -> Job:
    settings = cast(Settings, request.app.state.settings)
    if idempotency_key is not None:
        existing = repository.get_by_idempotency_key(merchant_id, idempotency_key)
        if existing is not None:
            ensure_idempotent_request_matches(existing, product_sku, frame_width_mm)
            return existing
    job = Job(
        merchant_id=merchant_id,
        product_sku=product_sku,
        frame_width_mm=frame_width_mm,
        idempotency_key=idempotency_key,
    )
    try:
        bundle = await ingest_uploads(
            job_id=str(job.id),
            merchant_id=merchant_id,
            product_sku=product_sku,
            uploads={"front": front, "left": left, "right": right, "back": back, "hero": hero},
            settings=settings,
        )
    except InputValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail={"code": exc.code, "message": str(exc)},
        ) from exc
    prefix = object_prefix(merchant_id, product_sku, str(job.id))
    raw_keys: dict[str, str] = {}
    normalized_keys: dict[str, str] = {}
    for view, image in bundle.images.items():
        raw_key = f"{prefix}/raw/{view}{image.raw_path.suffix.lower()}"
        normalized_key = f"{prefix}/inputs/normalized/{view}.png"
        storage.put_file(image.raw_path, raw_key, image_content_type(image.raw_path))
        storage.put_file(image.normalized_path, normalized_key, "image/png")
        raw_keys[view] = raw_key
        normalized_keys[view] = normalized_key
    job.inputs = JobInputs(raw=raw_keys, normalized=normalized_keys)
    transition_job(job, JobStatus.VALIDATING)
    try:
        created = repository.create(job)
    except DuplicateIdempotencyKey as exc:
        assert idempotency_key is not None
        existing = repository.get_by_idempotency_key(merchant_id, idempotency_key)
        if existing is None:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Idempotency conflict") from exc
        ensure_idempotent_request_matches(existing, product_sku, frame_width_mm)
        return existing
    return await dispatch_job(created, repository, dispatcher)


@router.get("/jobs/{job_id}", response_model=Job, dependencies=[Depends(authorize)])
def get_job(job_id: UUID, repository: Annotated[JobRepository, Depends(get_repository)]) -> Job:
    try:
        return repository.get(job_id)
    except JobNotFound as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found") from exc


@router.post("/jobs/{job_id}/cancel", response_model=Job, dependencies=[Depends(authorize)])
def cancel_job(job_id: UUID, repository: Annotated[JobRepository, Depends(get_repository)]) -> Job:
    try:
        job = repository.get(job_id)
        if job.status == JobStatus.CANCELLED:
            return job
        transition_job(job, JobStatus.CANCELLED)
        return repository.save(job)
    except JobNotFound as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found") from exc
    except InvalidJobTransition as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc


@router.post("/jobs/{job_id}/retry", response_model=Job, dependencies=[Depends(authorize)])
async def retry_failed_job(
    job_id: UUID,
    repository: Annotated[JobRepository, Depends(get_repository)],
    dispatcher: Annotated[JobDispatcher, Depends(get_dispatcher)],
) -> Job:
    try:
        job = repository.get(job_id)
        retry_job(job)
        saved = repository.save(job)
    except JobNotFound as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found") from exc
    except InvalidJobTransition as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    return await dispatch_job(saved, repository, dispatcher)


@router.post("/worker/jobs/{job_id}/execute", response_model=Job)
def execute_job(
    job_id: UUID,
    request: Request,
    repository: Annotated[JobRepository, Depends(get_repository)],
    authorization: Annotated[str | None, Header()] = None,
) -> Job:
    settings = cast(Settings, request.app.state.settings)
    expected = f"Bearer {settings.worker_token}" if settings.worker_token else None
    if expected is None or authorization is None or not secrets.compare_digest(authorization, expected):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid worker token")
    executor = cast(JobExecutor | None, request.app.state.executor)
    if executor is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="GPU executor is unavailable")
    try:
        job = repository.get(job_id)
        if job.status != JobStatus.VALIDATING:
            return job
        return executor.run(job_id)
    except JobNotFound as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found") from exc


@router.post("/jobs/{job_id}/finalize", response_model=Job, dependencies=[Depends(authorize)])
def finalize_job(
    job_id: UUID,
    repository: Annotated[JobRepository, Depends(get_repository)],
    storage: Annotated[ObjectStorage, Depends(get_storage)],
) -> Job:
    try:
        job = repository.get(job_id)
        if job.status == JobStatus.FINALIZED:
            return job
        if job.status != JobStatus.READY_FOR_REVIEW:
            raise InvalidJobTransition(f"Cannot transition {job.status.value} to finalized")
        prefix = object_prefix(job.merchant_id, job.product_sku, str(job.id))
        final_glb = storage.copy(
            f"{prefix}/preview/frame_model.glb",
            f"{prefix}/final/frame_model.glb",
        )
        storage.copy(
            f"{prefix}/preview/thumbnail_hero.webp",
            f"{prefix}/final/thumbnail_hero.webp",
        )
        storage.copy(
            f"{prefix}/preview/metadata.json",
            f"{prefix}/final/metadata.json",
        )
        job.outputs.final_glb_url = final_glb.url
        transition_job(job, JobStatus.FINALIZED)
        return repository.save(job)
    except JobNotFound as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found") from exc
    except InvalidJobTransition as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except (FileNotFoundError, OSError) as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Asset promotion failed") from exc


def default_job_repository() -> InMemoryJobRepository:
    return InMemoryJobRepository()


def image_content_type(path: Path) -> str:
    return {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }.get(path.suffix.lower(), "application/octet-stream")


def ensure_idempotent_request_matches(job: Job, product_sku: str, frame_width_mm: float) -> None:
    if job.product_sku != product_sku or job.frame_width_mm != frame_width_mm:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Idempotency-Key was already used with different job parameters",
        )


async def dispatch_job(job: Job, repository: JobRepository, dispatcher: JobDispatcher) -> Job:
    try:
        await dispatcher.dispatch(job.id)
        return job
    except DispatchError:
        job.error = JobError(
            code="dispatch_failed",
            message="The GPU worker could not be scheduled",
            retryable=True,
        )
        transition_job(job, JobStatus.FAILED)
        return repository.save(job)
