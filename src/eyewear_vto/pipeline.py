from __future__ import annotations

import json
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from uuid import UUID

from eyewear_vto.jobs import ConcurrentJobUpdate, Job, JobError, JobOutputs, JobRepository, JobStatus, transition_job
from eyewear_vto.landmarks import LandmarkDetector, LandmarkResult
from eyewear_vto.metrics import PIPELINE_JOBS, observe_pipeline_stage
from eyewear_vto.painting import Painter
from eyewear_vto.segmentation import SegmentationResult, Segmenter
from eyewear_vto.storage import ObjectStorage, object_prefix


@dataclass(frozen=True)
class DeformationResult:
    mesh_path: Path
    template_id: str


@dataclass(frozen=True)
class ProjectionResult:
    mesh_path: Path
    reference_image: Path
    coverage_path: Path


@dataclass(frozen=True)
class ExportResult:
    glb_path: Path
    thumbnail_path: Path
    metadata_path: Path


class TemplateDeformer(Protocol):
    def deform(
        self,
        landmarks: LandmarkResult,
        frame_width_mm: float,
        output_dir: Path,
    ) -> DeformationResult: ...


class TextureProjector(Protocol):
    def project(
        self,
        mesh: DeformationResult,
        images: Mapping[str, Path],
        segmentation: SegmentationResult,
        output_dir: Path,
    ) -> ProjectionResult: ...


class AssetExporter(Protocol):
    def export(
        self,
        painted_glb: Path,
        reference_image: Path,
        output_dir: Path,
        metadata: Mapping[str, object],
    ) -> ExportResult: ...


class GenerationPipeline:
    def __init__(
        self,
        *,
        repository: JobRepository,
        storage: ObjectStorage,
        segmenter: Segmenter,
        landmark_detector: LandmarkDetector,
        deformer: TemplateDeformer,
        projector: TextureProjector,
        painter: Painter,
        exporter: AssetExporter,
        work_root: Path,
        retain_workspaces: bool = False,
    ) -> None:
        self.repository = repository
        self.storage = storage
        self.segmenter = segmenter
        self.landmark_detector = landmark_detector
        self.deformer = deformer
        self.projector = projector
        self.painter = painter
        self.exporter = exporter
        self.work_root = work_root
        self.retain_workspaces = retain_workspaces

    def run(self, job_id: UUID) -> Job:
        job = self.repository.get(job_id)
        try:
            if job.status != JobStatus.VALIDATING:
                raise RuntimeError(f"Job must be validating, got {job.status.value}")
            workspace = self.work_root / str(job.id)
            manifest_path = workspace / "input_manifest.json"
            if not manifest_path.is_file():
                materialize_inputs(job, workspace, self.storage)
            images = load_normalized_inputs(manifest_path)
            raw_images = load_raw_inputs(manifest_path)

            self._advance(job, JobStatus.SEGMENTING)
            with observe_pipeline_stage("segmentation"):
                segmentation = self.segmenter.segment(images, workspace / "segmentation")

            self._advance(job, JobStatus.LANDMARKING)
            with observe_pipeline_stage("landmarking"):
                landmarks = self.landmark_detector.detect(images)

            self._advance(job, JobStatus.DEFORMING)
            with observe_pipeline_stage("deformation"):
                deformation = self.deformer.deform(landmarks, job.frame_width_mm, workspace / "deformation")

            self._advance(job, JobStatus.PROJECTING)
            with observe_pipeline_stage("projection"):
                projection = self.projector.project(
                    deformation,
                    images,
                    segmentation,
                    workspace / "projection",
                )

            self._advance(job, JobStatus.PAINTING)
            with observe_pipeline_stage("painting"):
                painted = self.painter.paint(
                    projection.mesh_path,
                    projection.reference_image,
                    workspace / "painting",
                )

            self._advance(job, JobStatus.OPTIMIZING)
            with observe_pipeline_stage("optimization"):
                exported = self.exporter.export(
                    painted.glb_path,
                    projection.reference_image,
                    workspace / "export",
                    {
                        "merchant_id": job.merchant_id,
                        "product_sku": job.product_sku,
                        "frame_width_mm": job.frame_width_mm,
                        "template_id": deformation.template_id,
                    },
                )

            self._advance(job, JobStatus.UPLOADING)
            prefix = object_prefix(job.merchant_id, job.product_sku, str(job.id))
            glb = self.storage.put_file(
                exported.glb_path,
                f"{prefix}/preview/frame_model.glb",
                "model/gltf-binary",
            )
            thumbnail = self.storage.put_file(
                exported.thumbnail_path,
                f"{prefix}/preview/thumbnail_hero.webp",
                "image/webp",
            )
            metadata = self.storage.put_file(
                exported.metadata_path,
                f"{prefix}/preview/metadata.json",
                "application/json",
            )
            if not job.inputs.raw:
                for view, raw_path in raw_images.items():
                    self.storage.put_file(
                        raw_path,
                        f"{prefix}/raw/{view}{raw_path.suffix.lower()}",
                        raw_content_type(raw_path),
                    )
            job.outputs = JobOutputs(
                preview_glb_url=glb.url,
                thumbnail_url=thumbnail.url,
                metadata_url=metadata.url,
            )

            self._advance(job, JobStatus.READY_FOR_REVIEW)
            PIPELINE_JOBS.labels("ready_for_review").inc()
            if not self.retain_workspaces:
                shutil.rmtree(workspace)
            return job
        except ConcurrentJobUpdate:
            latest = self.repository.get(job_id)
            PIPELINE_JOBS.labels(latest.status.value).inc()
            return latest
        except Exception as exc:
            job.error = JobError(code="pipeline_failed", message=str(exc)[:500], retryable=False)
            if job.status not in {JobStatus.FINALIZED, JobStatus.CANCELLED, JobStatus.FAILED}:
                transition_job(job, JobStatus.FAILED)
            PIPELINE_JOBS.labels("failed").inc()
            return self.repository.save(job)

    def _advance(self, job: Job, status: JobStatus) -> None:
        transition_job(job, status)
        self.repository.save(job)


def load_normalized_inputs(manifest_path: Path) -> dict[str, Path]:
    return load_manifest_paths(manifest_path, "normalized_path")


def load_raw_inputs(manifest_path: Path) -> dict[str, Path]:
    return load_manifest_paths(manifest_path, "raw_path")


def materialize_inputs(job: Job, workspace: Path, storage: ObjectStorage) -> Path:
    if not job.inputs.normalized or not job.inputs.raw:
        raise ValueError("Job has no durable input objects and no local input manifest")
    normalized_dir = workspace / "normalized"
    raw_dir = workspace / "raw"
    images: dict[str, dict[str, str]] = {}
    for view, normalized_key in job.inputs.normalized.items():
        raw_key = job.inputs.raw.get(view)
        if raw_key is None:
            raise ValueError(f"Job has no raw input object for {view}")
        normalized_path = normalized_dir / f"{view}.png"
        raw_suffix = Path(raw_key).suffix.lower()
        raw_path = raw_dir / f"{view}{raw_suffix}"
        storage.get_file(normalized_key, normalized_path)
        storage.get_file(raw_key, raw_path)
        images[view] = {
            "normalized_path": str(normalized_path),
            "raw_path": str(raw_path),
        }
    manifest_path = workspace / "input_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps({"schema_version": 1, "images": images}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return manifest_path


def load_manifest_paths(manifest_path: Path, field: str) -> dict[str, Path]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    images = payload.get("images")
    if not isinstance(images, dict):
        raise ValueError("Input manifest has no images")
    result: dict[str, Path] = {}
    for view, record in images.items():
        if not isinstance(record, dict) or field not in record:
            raise ValueError(f"Input manifest has an invalid {view} record")
        path = Path(record[field])
        if not path.is_file():
            raise FileNotFoundError(path)
        result[str(view)] = path
    return result


def raw_content_type(path: Path) -> str:
    return {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }.get(path.suffix.lower(), "application/octet-stream")
