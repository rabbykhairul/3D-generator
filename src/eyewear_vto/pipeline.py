from __future__ import annotations

import json
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from uuid import UUID

from eyewear_vto.checkpoints import CheckpointStore, pipeline_input_digest
from eyewear_vto.jobs import ConcurrentJobUpdate, Job, JobError, JobOutputs, JobRepository, JobStatus, transition_job
from eyewear_vto.landmarks import Landmark, LandmarkDetector, LandmarkResult, ViewLandmarks
from eyewear_vto.metrics import PIPELINE_JOBS, observe_pipeline_stage
from eyewear_vto.painting import Painter, PaintResult
from eyewear_vto.segmentation import SegmentationResult, Segmenter
from eyewear_vto.storage import ObjectStorage, object_prefix

PIPELINE_IMPLEMENTATION_VERSIONS = {
    "segmentation": "1",
    "landmarks": "1",
    "deformation": "1",
    "projection": "1",
    "painting": "1",
    "export": "1",
    "upload": "1",
}


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
        implementation_versions: Mapping[str, str] | None = None,
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
        self.implementation_versions = dict(implementation_versions or PIPELINE_IMPLEMENTATION_VERSIONS)
        missing_versions = set(PIPELINE_IMPLEMENTATION_VERSIONS) - self.implementation_versions.keys()
        if missing_versions:
            raise ValueError(f"Missing pipeline implementation versions: {', '.join(sorted(missing_versions))}")

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
            input_digest = pipeline_input_digest(images, job.frame_width_mm)
            checkpoints = CheckpointStore(
                job=job,
                repository=self.repository,
                storage=self.storage,
                input_digest=input_digest,
                versions=self.implementation_versions,
            )

            self._advance(job, JobStatus.SEGMENTING)
            segmentation_checkpoint = checkpoints.find("segmentation")
            if segmentation_checkpoint is None:
                with observe_pipeline_stage("segmentation"):
                    segmentation = self.segmenter.segment(images, workspace / "segmentation")
                scores_path = workspace / "segmentation" / "scores.json"
                write_json(scores_path, dict(segmentation.scores))
                artifacts = {f"mask.{view}": path for view, path in segmentation.masks.items()}
                artifacts["scores"] = scores_path
                segmentation_checkpoint = checkpoints.save("segmentation", artifacts)
            else:
                masks = {
                    view: checkpoints.restore(
                        segmentation_checkpoint,
                        f"mask.{view}",
                        workspace / "segmentation" / f"{view}.png",
                    )
                    for view in images
                }
                scores_path = checkpoints.restore(
                    segmentation_checkpoint,
                    "scores",
                    workspace / "segmentation" / "scores.json",
                )
                segmentation = SegmentationResult(masks=masks, scores=load_float_mapping(scores_path))
            checkpoints.advance(segmentation_checkpoint)

            self._advance(job, JobStatus.LANDMARKING)
            landmarks_checkpoint = checkpoints.find("landmarks")
            landmarks_path = workspace / "landmarks" / "landmarks.json"
            if landmarks_checkpoint is None:
                with observe_pipeline_stage("landmarking"):
                    landmarks = self.landmark_detector.detect(images)
                write_landmarks(landmarks_path, landmarks)
                landmarks_checkpoint = checkpoints.save("landmarks", {"landmarks": landmarks_path})
            else:
                checkpoints.restore(landmarks_checkpoint, "landmarks", landmarks_path)
                landmarks = read_landmarks(landmarks_path)
            checkpoints.advance(landmarks_checkpoint)

            self._advance(job, JobStatus.DEFORMING)
            deformation_checkpoint = checkpoints.find("deformation")
            if deformation_checkpoint is None:
                with observe_pipeline_stage("deformation"):
                    deformation = self.deformer.deform(landmarks, job.frame_width_mm, workspace / "deformation")
                deformation_checkpoint = checkpoints.save(
                    "deformation",
                    {"mesh": deformation.mesh_path},
                    {"template_id": deformation.template_id},
                )
            else:
                mesh_path = checkpoints.restore(
                    deformation_checkpoint,
                    "mesh",
                    workspace / "deformation" / "frame.glb",
                )
                deformation = DeformationResult(
                    mesh_path=mesh_path,
                    template_id=checkpoint_string(deformation_checkpoint.metadata, "template_id"),
                )
            checkpoints.advance(deformation_checkpoint)

            self._advance(job, JobStatus.PROJECTING)
            projection_checkpoint = checkpoints.find("projection")
            if projection_checkpoint is None:
                with observe_pipeline_stage("projection"):
                    projection = self.projector.project(
                        deformation,
                        images,
                        segmentation,
                        workspace / "projection",
                    )
                projection_checkpoint = checkpoints.save(
                    "projection",
                    {"mesh": projection.mesh_path, "coverage": projection.coverage_path},
                    {"reference_view": reference_view(images, projection.reference_image)},
                )
            else:
                projection = ProjectionResult(
                    mesh_path=checkpoints.restore(
                        projection_checkpoint,
                        "mesh",
                        workspace / "projection" / "projected.glb",
                    ),
                    reference_image=images[
                        checkpoint_string(projection_checkpoint.metadata, "reference_view")
                    ],
                    coverage_path=checkpoints.restore(
                        projection_checkpoint,
                        "coverage",
                        workspace / "projection" / "coverage.png",
                    ),
                )
            checkpoints.advance(projection_checkpoint)

            self._advance(job, JobStatus.PAINTING)
            painting_checkpoint = checkpoints.find("painting")
            if painting_checkpoint is None:
                with observe_pipeline_stage("painting"):
                    painted = self.painter.paint(
                        projection.mesh_path,
                        projection.reference_image,
                        workspace / "painting",
                    )
                painting_checkpoint = checkpoints.save(
                    "painting",
                    {"mesh": painted.glb_path},
                    {"max_views": painted.max_views, "resolution": painted.resolution},
                )
            else:
                painted = PaintResult(
                    glb_path=checkpoints.restore(
                        painting_checkpoint,
                        "mesh",
                        workspace / "painting" / "textured_mesh.glb",
                    ),
                    max_views=checkpoint_int(painting_checkpoint.metadata, "max_views"),
                    resolution=checkpoint_int(painting_checkpoint.metadata, "resolution"),
                )
            checkpoints.advance(painting_checkpoint)

            self._advance(job, JobStatus.OPTIMIZING)
            export_checkpoint = checkpoints.find("export")
            if export_checkpoint is None:
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
                export_checkpoint = checkpoints.save(
                    "export",
                    {
                        "glb": exported.glb_path,
                        "thumbnail": exported.thumbnail_path,
                        "metadata": exported.metadata_path,
                    },
                )
            else:
                exported = ExportResult(
                    glb_path=checkpoints.restore(
                        export_checkpoint,
                        "glb",
                        workspace / "export" / "frame_model.glb",
                    ),
                    thumbnail_path=checkpoints.restore(
                        export_checkpoint,
                        "thumbnail",
                        workspace / "export" / "thumbnail_hero.webp",
                    ),
                    metadata_path=checkpoints.restore(
                        export_checkpoint,
                        "metadata",
                        workspace / "export" / "metadata.json",
                    ),
                )
            checkpoints.advance(export_checkpoint)

            self._advance(job, JobStatus.UPLOADING)
            prefix = object_prefix(job.merchant_id, job.product_sku, str(job.id))
            upload_checkpoint = checkpoints.find("upload")
            if upload_checkpoint is None:
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
                upload_checkpoint = checkpoints.save(
                    "upload",
                    {},
                    {
                        "preview_glb_url": glb.url,
                        "thumbnail_url": thumbnail.url,
                        "metadata_url": metadata.url,
                    },
                    {
                        "glb": glb.key,
                        "thumbnail": thumbnail.key,
                        "metadata": metadata.key,
                    },
                )
            output_metadata = upload_checkpoint
            job.outputs = JobOutputs(
                preview_glb_url=checkpoint_string(output_metadata.metadata, "preview_glb_url"),
                thumbnail_url=checkpoint_string(output_metadata.metadata, "thumbnail_url"),
                metadata_url=checkpoint_string(output_metadata.metadata, "metadata_url"),
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
            job.error = classify_pipeline_error(exc)
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


def write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def load_float_mapping(path: Path) -> dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Checkpoint score artifact must be an object")
    return {str(key): float(value) for key, value in payload.items()}


def write_landmarks(path: Path, result: LandmarkResult) -> Path:
    payload = {
        "views": {
            view: {
                "detection_confidence": record.detection_confidence,
                "points": [
                    {
                        "name": point.name,
                        "x": point.x,
                        "y": point.y,
                        "confidence": point.confidence,
                    }
                    for point in record.points
                ],
            }
            for view, record in result.views.items()
        }
    }
    return write_json(path, payload)


def read_landmarks(path: Path) -> LandmarkResult:
    payload = json.loads(path.read_text(encoding="utf-8"))
    view_payloads = payload.get("views") if isinstance(payload, dict) else None
    if not isinstance(view_payloads, dict):
        raise ValueError("Landmark checkpoint has no views")
    views: dict[str, ViewLandmarks] = {}
    for view, record in view_payloads.items():
        if not isinstance(record, dict) or not isinstance(record.get("points"), list):
            raise ValueError(f"Landmark checkpoint has an invalid {view} view")
        points = tuple(
            Landmark(
                name=str(point["name"]),
                x=float(point["x"]),
                y=float(point["y"]),
                confidence=float(point["confidence"]),
            )
            for point in record["points"]
            if isinstance(point, dict)
        )
        views[str(view)] = ViewLandmarks(
            view=str(view),
            points=points,
            detection_confidence=float(record["detection_confidence"]),
        )
    return LandmarkResult(views=views)


def reference_view(images: Mapping[str, Path], reference_image: Path) -> str:
    resolved_reference = reference_image.resolve()
    for view, path in images.items():
        if path.resolve() == resolved_reference:
            return view
    raise ValueError("Projection reference image is not one of the normalized inputs")


def checkpoint_string(metadata: Mapping[str, object], key: str) -> str:
    value = metadata.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Checkpoint metadata {key} must be a non-empty string")
    return value


def checkpoint_int(metadata: Mapping[str, object], key: str) -> int:
    value = metadata.get(key)
    if not isinstance(value, int):
        raise ValueError(f"Checkpoint metadata {key} must be an integer")
    return value


def classify_pipeline_error(exc: Exception) -> JobError:
    message = str(exc)[:500]
    normalized = message.lower()
    if isinstance(exc, MemoryError) or "out of memory" in normalized or "cuda oom" in normalized:
        return JobError(code="capacity_exhausted", message=message, retryable=True)
    if isinstance(exc, (ConnectionError, TimeoutError)):
        return JobError(code="infrastructure_transient", message=message, retryable=True)
    return JobError(code="pipeline_failed", message=message, retryable=False)
