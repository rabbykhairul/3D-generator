from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

from pydantic import JsonValue

from eyewear_vto.jobs import Job, JobRepository, StageCheckpoint
from eyewear_vto.storage import ObjectStorage, object_prefix


def pipeline_input_digest(images: Mapping[str, Path], frame_width_mm: float) -> str:
    digest = hashlib.sha256()
    digest.update(f"frame_width_mm={frame_width_mm:.6f}\n".encode())
    for view, path in sorted(images.items()):
        digest.update(f"view={view}\n".encode())
        with path.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


class CheckpointStore:
    def __init__(
        self,
        *,
        job: Job,
        repository: JobRepository,
        storage: ObjectStorage,
        input_digest: str,
        versions: Mapping[str, str],
    ) -> None:
        self.job = job
        self.repository = repository
        self.storage = storage
        self.input_digest = input_digest
        self.versions = versions
        self.prefix = object_prefix(job.merchant_id, job.product_sku, str(job.id))

    def find(self, stage: str) -> StageCheckpoint | None:
        version = self.versions[stage]
        for checkpoint in reversed(self.job.checkpoints):
            if (
                checkpoint.stage == stage
                and checkpoint.input_digest == self.input_digest
                and checkpoint.implementation_version == version
                and all(self.storage.exists(key) for key in checkpoint.artifacts.values())
            ):
                return checkpoint
        return None

    def save(
        self,
        stage: str,
        artifacts: Mapping[str, Path],
        metadata: Mapping[str, JsonValue] | None = None,
        object_keys: Mapping[str, str] | None = None,
    ) -> StageCheckpoint:
        keys = dict(object_keys or {})
        for name, path in artifacts.items():
            content_hash = file_digest(path)
            key = (
                f"{self.prefix}/checkpoints/{stage}/{self.input_digest}/"
                f"{self.versions[stage]}/{content_hash}/{path.name}"
            )
            stored = self.storage.put_file(path, key, checkpoint_content_type(path))
            keys[name] = stored.key
        checkpoint = StageCheckpoint(
            stage=stage,
            input_digest=self.input_digest,
            implementation_version=self.versions[stage],
            attempt=self.job.attempt,
            artifacts=keys,
            metadata=dict(metadata or {}),
        )
        self.job.checkpoints.append(checkpoint)
        self.repository.save(self.job)
        return checkpoint

    def restore(self, checkpoint: StageCheckpoint, name: str, destination: Path) -> Path:
        try:
            key = checkpoint.artifacts[name]
        except KeyError as exc:
            raise ValueError(f"Checkpoint {checkpoint.stage} has no {name} artifact") from exc
        return self.storage.get_file(key, destination)

    def advance(self, checkpoint: StageCheckpoint) -> str:
        payload = {
            "stage": checkpoint.stage,
            "input_digest": checkpoint.input_digest,
            "implementation_version": checkpoint.implementation_version,
            "artifacts": checkpoint.artifacts,
            "metadata": checkpoint.metadata,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        self.input_digest = hashlib.sha256(encoded).hexdigest()
        return self.input_digest


def checkpoint_content_type(path: Path) -> str:
    return {
        ".glb": "model/gltf-binary",
        ".json": "application/json",
        ".png": "image/png",
        ".webp": "image/webp",
    }.get(path.suffix.lower(), "application/octet-stream")
