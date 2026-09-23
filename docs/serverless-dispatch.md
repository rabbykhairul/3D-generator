# Serverless GPU dispatch contract

The API and GPU worker are separated by a small provider-neutral HTTP contract. This keeps the job database and object-storage lifecycle independent of RunPod, Modal, or another serverless platform.

## Control-plane dispatch

After inputs are validated, archived, and the job is committed to PostgreSQL, the API sends:

```http
POST ${DISPATCH_URL}
Authorization: Bearer ${DISPATCH_TOKEN}
Idempotency-Key: {job_id}
Content-Type: application/json

{"job_id":"00000000-0000-0000-0000-000000000000"}
```

The provider adapter must return a successful 2xx response only after accepting the invocation. It must treat `Idempotency-Key` as an at-least-once delivery key. A timeout or non-2xx response records the job as `FAILED` with `dispatch_failed` and `retryable=true`; `POST /v1/jobs/{job_id}/retry` creates the next attempt and dispatches it again.

`DISPATCH_URL` can be the provider's native job endpoint or a thin adapter. Provider credentials belong in `DISPATCH_TOKEN`; they are runtime secrets and are never Docker build arguments.

## Data-plane execution

The adapter invokes the baked application worker endpoint:

```http
POST /v1/worker/jobs/{job_id}/execute
Authorization: Bearer ${WORKER_TOKEN}
```

The worker loads job state from PostgreSQL and input/checkpoint keys from R2. It never expects the API replica's filesystem. Terminal and review-ready jobs return their current representation without starting another pipeline. Provider redelivery of a job left in a processing state after worker loss creates a new attempt and resumes from its latest valid checkpoint chain.

The application accepts an injected `JobExecutor`; production pipeline assembly remains blocked until the real M07 deformation, M08 projection, and M09 composition components are available. The serverless adapter must not be declared production-ready before that executor is installed.

## Timeouts and retries

- Dispatch timeout defaults to 10 seconds and is configured with `DISPATCH_TIMEOUT_SECONDS`.
- Provider acceptance and GPU execution are separate timeouts. The acceptance request should remain short.
- GPU concurrency is one per container.
- Every completed stage is stored under a content-addressed R2 key and recorded in the PostgreSQL job payload. Mid-stage loss recomputes that stage; completed matching stages are restored.
- The provider adapter must honor the job ID as its idempotency key and must not intentionally deliver the same active invocation concurrently. PostgreSQL optimistic writes prevent stale workers from overwriting newer state at stage boundaries.
- Never expose the worker endpoint without the bearer token and a provider/private-network ingress policy.

## Provider adapter acceptance

Before selecting the provider integration as complete, verify:

1. duplicate dispatches do not start duplicate GPU work;
2. a rejected dispatch becomes a visible retryable job failure;
3. the provider can run the GHCR commit-SHA image with one NVIDIA GPU and at least 24 GB VRAM;
4. runtime PostgreSQL/R2 secrets are injected without entering the image;
5. the maximum provider execution window exceeds the measured golden-job duration;
6. cold-start, cancellation, interruption, and retry behavior are recorded in the M16 report.
