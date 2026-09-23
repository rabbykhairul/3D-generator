from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from time import perf_counter
from uuid import uuid4

import structlog
from fastapi import FastAPI, Request, Response, status
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from structlog.contextvars import bind_contextvars, clear_contextvars

from eyewear_vto import __version__
from eyewear_vto.api import default_job_repository, router
from eyewear_vto.config import AppEnvironment, Settings, get_settings
from eyewear_vto.dispatch import JobDispatcher, JobExecutor, create_dispatcher
from eyewear_vto.jobs import JobRepository
from eyewear_vto.logging import configure_logging
from eyewear_vto.metrics import HTTP_DURATION, HTTP_REQUESTS
from eyewear_vto.operations import cleanup_stale_workspaces
from eyewear_vto.preflight import run_preflight
from eyewear_vto.sql_jobs import SqlJobRepository
from eyewear_vto.storage import ObjectStorage, create_storage


def json_string(value: str) -> str:
    import json

    return json.dumps(value)


def create_job_repository(settings: Settings) -> JobRepository:
    if settings.database_url:
        return SqlJobRepository(
            settings.database_url,
            create_schema=settings.app_env != AppEnvironment.PRODUCTION,
        )
    return default_job_repository()


def create_app(
    settings: Settings | None = None,
    job_repository: JobRepository | None = None,
    storage: ObjectStorage | None = None,
    dispatcher: JobDispatcher | None = None,
    executor: JobExecutor | None = None,
) -> FastAPI:
    resolved_settings = settings or get_settings()
    resolved_repository = job_repository or create_job_repository(resolved_settings)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        configure_logging()
        resolved_settings.work_root.mkdir(parents=True, exist_ok=True)
        cleanup_stale_workspaces(
            resolved_settings.work_root,
            resolved_settings.workspace_max_age_seconds,
        )
        if resolved_settings.storage_backend.value == "local":
            resolved_settings.local_storage_root.mkdir(parents=True, exist_ok=True)
        yield

    application = FastAPI(
        title="Eyewear Multi-View 3D API",
        version=__version__,
        lifespan=lifespan,
    )
    application.state.settings = resolved_settings
    application.state.job_repository = resolved_repository
    application.state.storage = storage or create_storage(resolved_settings)
    application.state.dispatcher = dispatcher or create_dispatcher(
        resolved_settings.dispatch_url,
        resolved_settings.dispatch_token,
        resolved_settings.dispatch_timeout_seconds,
    )
    application.state.executor = executor
    application.include_router(router)

    @application.middleware("http")
    async def request_context(request: Request, call_next):  # type: ignore[no-untyped-def]
        request_id = request.headers.get("x-request-id", "")[:64] or str(uuid4())
        bind_contextvars(request_id=request_id)
        started = perf_counter()
        response: Response
        try:
            response = await call_next(request)
        finally:
            elapsed = perf_counter() - started
            route = request.scope.get("route")
            route_path = getattr(route, "path", request.url.path)
            response_status = locals().get("response")
            status_code = getattr(response_status, "status_code", 500)
            HTTP_REQUESTS.labels(request.method, route_path, str(status_code)).inc()
            HTTP_DURATION.labels(request.method, route_path).observe(elapsed)
            structlog.get_logger().info(
                "http_request",
                method=request.method,
                route=route_path,
                status=status_code,
                duration_seconds=round(elapsed, 6),
            )
            clear_contextvars()
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; img-src 'self' blob: data: https:; connect-src 'self' https:; "
            "worker-src 'self' blob:; object-src 'none'; frame-ancestors 'none'; base-uri 'none'"
        )
        return response

    @application.get("/health/live", tags=["health"])
    async def live() -> dict[str, str]:
        return {"status": "live", "version": __version__}

    @application.get("/health/ready", tags=["health"])
    async def ready() -> Response:
        try:
            application.state.storage.healthcheck()
            application.state.job_repository.healthcheck()
            preflight = run_preflight(resolved_settings)
        except Exception as exc:
            return Response(
                content=f'{{"status":"not_ready","reason":{json_string(str(exc)[:300])}}}',
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                media_type="application/json",
            )
        return Response(
            content=f'{{"status":"ready","pipeline_backend":{json_string(str(preflight["pipeline_backend"]))}}}',
            media_type="application/json",
        )

    @application.get("/metrics", include_in_schema=False)
    async def metrics() -> Response:
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

    if resolved_settings.ui_root.is_dir():

        @application.get("/", include_in_schema=False)
        async def root() -> RedirectResponse:
            return RedirectResponse("/ui/")

        application.mount("/ui", StaticFiles(directory=resolved_settings.ui_root, html=True), name="ui")

    if resolved_settings.storage_backend.value == "local":
        application.mount(
            "/assets",
            StaticFiles(directory=resolved_settings.local_storage_root),
            name="local-assets",
        )

    return application


app = create_app()
