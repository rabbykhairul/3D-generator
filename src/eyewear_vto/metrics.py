from __future__ import annotations

import sys
from collections.abc import Iterator
from contextlib import contextmanager
from time import perf_counter

from prometheus_client import Counter, Gauge, Histogram

HTTP_REQUESTS = Counter(
    "eyewear_http_requests_total",
    "HTTP requests processed by the service.",
    ("method", "route", "status"),
)
HTTP_DURATION = Histogram(
    "eyewear_http_request_duration_seconds",
    "HTTP request latency.",
    ("method", "route"),
)
PIPELINE_STAGE_DURATION = Histogram(
    "eyewear_pipeline_stage_duration_seconds",
    "Pipeline stage execution latency.",
    ("stage",),
)
PIPELINE_JOBS = Counter(
    "eyewear_pipeline_jobs_total",
    "Pipeline terminal job results.",
    ("status",),
)
GPU_MEMORY_ALLOCATED = Gauge(
    "eyewear_gpu_memory_allocated_bytes",
    "Torch CUDA allocated memory after a pipeline stage.",
    ("stage",),
)


@contextmanager
def observe_pipeline_stage(stage: str) -> Iterator[None]:
    started = perf_counter()
    try:
        yield
    finally:
        PIPELINE_STAGE_DURATION.labels(stage).observe(perf_counter() - started)
        torch = sys.modules.get("torch")
        if torch is not None and torch.cuda.is_available():
            GPU_MEMORY_ALLOCATED.labels(stage).set(torch.cuda.memory_allocated())
