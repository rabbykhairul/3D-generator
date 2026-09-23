from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol
from uuid import UUID

import httpx

from eyewear_vto.jobs import Job


class DispatchError(RuntimeError):
    pass


class JobDispatcher(Protocol):
    async def dispatch(self, job_id: UUID) -> None: ...


class JobExecutor(Protocol):
    def run(self, job_id: UUID) -> Job: ...


@dataclass(frozen=True)
class NoopJobDispatcher:
    async def dispatch(self, job_id: UUID) -> None:
        del job_id


class HttpJobDispatcher:
    def __init__(self, endpoint: str, token: str, timeout_seconds: float = 10.0) -> None:
        self.endpoint = endpoint
        self.token = token
        self.timeout_seconds = timeout_seconds

    async def dispatch(self, job_id: UUID) -> None:
        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.post(
                    self.endpoint,
                    json={"job_id": str(job_id)},
                    headers={
                        "Authorization": f"Bearer {self.token}",
                        "Idempotency-Key": str(job_id),
                    },
                )
                response.raise_for_status()
        except httpx.HTTPError as exc:
            raise DispatchError(f"GPU job dispatch failed: {exc}") from exc


def create_dispatcher(endpoint: str | None, token: str | None, timeout_seconds: float) -> JobDispatcher:
    if endpoint is None:
        return NoopJobDispatcher()
    if token is None:
        raise ValueError("DISPATCH_TOKEN is required when DISPATCH_URL is configured")
    return HttpJobDispatcher(endpoint, token, timeout_seconds)
