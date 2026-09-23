from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ModelManifestError(RuntimeError):
    pass


@dataclass(frozen=True)
class ModelArtifact:
    name: str
    source: str
    revision: str
    path: Path
    required: bool
    sha256: str | None = None


def load_manifest(path: Path) -> tuple[ModelArtifact, ...]:
    if not path.is_file():
        raise ModelManifestError(f"Model manifest does not exist: {path}")
    try:
        payload: Any = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != 1 or not isinstance(payload.get("models"), list):
            raise ValueError("unsupported schema")
        return tuple(
            ModelArtifact(
                name=str(item["name"]),
                source=str(item["source"]),
                revision=str(item["revision"]),
                path=Path(item["path"]),
                required=bool(item.get("required", True)),
                sha256=str(item["sha256"]) if item.get("sha256") else None,
            )
            for item in payload["models"]
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ModelManifestError(f"Invalid model manifest: {path}") from exc


def verify_manifest(path: Path, verify_checksums: bool = False) -> tuple[ModelArtifact, ...]:
    artifacts = load_manifest(path)
    errors: list[str] = []
    for artifact in artifacts:
        if not artifact.required:
            continue
        if not artifact.path.exists():
            errors.append(f"{artifact.name}: missing {artifact.path}")
            continue
        if artifact.path.is_file() and artifact.path.stat().st_size == 0:
            errors.append(f"{artifact.name}: empty {artifact.path}")
            continue
        if verify_checksums and artifact.sha256 and artifact.path.is_file():
            actual = sha256_file(artifact.path)
            if actual != artifact.sha256:
                errors.append(f"{artifact.name}: checksum mismatch")
    if errors:
        raise ModelManifestError("; ".join(errors))
    return artifacts


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()

