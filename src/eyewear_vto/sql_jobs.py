from __future__ import annotations

from datetime import datetime
from uuid import UUID

from sqlalchemy import (
    DateTime,
    Integer,
    String,
    Table,
    Text,
    UniqueConstraint,
    create_engine,
    delete,
    insert,
    select,
    update,
)
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.schema import Column, MetaData

from eyewear_vto.jobs import ConcurrentJobUpdate, DuplicateIdempotencyKey, Job, JobNotFound

metadata = MetaData()
jobs_table = Table(
    "generation_jobs",
    metadata,
    Column("id", String(36), primary_key=True),
    Column("merchant_id", String(64), nullable=False, index=True),
    Column("product_sku", String(96), nullable=False, index=True),
    Column("idempotency_key", String(128), nullable=True),
    Column("status", String(32), nullable=False, index=True),
    Column("version", Integer, nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False, index=True),
    Column("payload", Text, nullable=False),
    UniqueConstraint("merchant_id", "idempotency_key", name="uq_generation_jobs_merchant_idempotency"),
)


class SqlJobRepository:
    def __init__(self, database_url: str, *, create_schema: bool = False) -> None:
        self.engine: Engine = create_engine(database_url, pool_pre_ping=True)
        if create_schema:
            metadata.create_all(self.engine)

    def create(self, job: Job) -> Job:
        values = self._values(job)
        try:
            with self.engine.begin() as connection:
                connection.execute(insert(jobs_table).values(**values))
        except IntegrityError as exc:
            if job.idempotency_key and self.get_by_idempotency_key(job.merchant_id, job.idempotency_key):
                raise DuplicateIdempotencyKey(job.idempotency_key) from exc
            raise
        return job.model_copy(deep=True)

    def get(self, job_id: UUID) -> Job:
        with self.engine.connect() as connection:
            payload = connection.execute(
                select(jobs_table.c.payload).where(jobs_table.c.id == str(job_id))
            ).scalar_one_or_none()
        if payload is None:
            raise JobNotFound(str(job_id))
        return Job.model_validate_json(payload)

    def get_by_idempotency_key(self, merchant_id: str, key: str) -> Job | None:
        with self.engine.connect() as connection:
            payload = connection.execute(
                select(jobs_table.c.payload).where(
                    jobs_table.c.merchant_id == merchant_id,
                    jobs_table.c.idempotency_key == key,
                )
            ).scalar_one_or_none()
        return Job.model_validate_json(payload) if payload is not None else None

    def save(self, job: Job) -> Job:
        expected_version = job.version
        updated = job.model_copy(deep=True, update={"version": expected_version + 1})
        values = self._values(updated)
        with self.engine.begin() as connection:
            result = connection.execute(
                update(jobs_table)
                .where(jobs_table.c.id == str(job.id), jobs_table.c.version == expected_version)
                .values(**values)
            )
        if result.rowcount != 1:
            with self.engine.connect() as connection:
                exists = connection.execute(
                    select(jobs_table.c.id).where(jobs_table.c.id == str(job.id))
                ).scalar_one_or_none()
            if exists is None:
                raise JobNotFound(str(job.id))
            raise ConcurrentJobUpdate(str(job.id))
        job.version = updated.version
        return updated

    def healthcheck(self) -> None:
        with self.engine.connect() as connection:
            connection.execute(select(1)).scalar_one()

    def delete(self, job_id: UUID) -> None:
        with self.engine.begin() as connection:
            connection.execute(delete(jobs_table).where(jobs_table.c.id == str(job_id)))

    @staticmethod
    def _values(job: Job) -> dict[str, str | int | datetime | None]:
        return {
            "id": str(job.id),
            "merchant_id": job.merchant_id,
            "product_sku": job.product_sku,
            "idempotency_key": job.idempotency_key,
            "status": job.status.value,
            "version": job.version,
            "updated_at": job.updated_at,
            "payload": job.model_dump_json(),
        }
