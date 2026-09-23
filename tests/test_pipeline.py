from __future__ import annotations

import json
import shutil
from pathlib import Path

from PIL import Image

from eyewear_vto.jobs import InMemoryJobRepository, Job, JobInputs, JobStatus
from eyewear_vto.landmarks import MockLandmarkDetector
from eyewear_vto.painting import MockPainter
from eyewear_vto.pipeline import (
    DeformationResult,
    ExportResult,
    GenerationPipeline,
    ProjectionResult,
)
from eyewear_vto.segmentation import MockSegmenter
from eyewear_vto.storage import LocalObjectStorage


class FakeDeformer:
    def deform(self, landmarks, frame_width_mm: float, output_dir: Path) -> DeformationResult:
        assert landmarks.views
        assert frame_width_mm == 140
        output_dir.mkdir(parents=True, exist_ok=True)
        mesh = output_dir / "frame.glb"
        mesh.write_bytes(b"mock glb")
        return DeformationResult(mesh_path=mesh, template_id="mock")


class FakeProjector:
    def project(self, mesh, images, segmentation, output_dir: Path) -> ProjectionResult:
        assert segmentation.masks
        output_dir.mkdir(parents=True, exist_ok=True)
        projected = output_dir / "projected.glb"
        shutil.copy2(mesh.mesh_path, projected)
        coverage = output_dir / "coverage.png"
        Image.new("L", (8, 8), 255).save(coverage)
        return ProjectionResult(projected, images.get("hero", images["front"]), coverage)


class FakeExporter:
    def export(self, painted_glb: Path, reference_image: Path, output_dir: Path, metadata) -> ExportResult:
        output_dir.mkdir(parents=True, exist_ok=True)
        glb = output_dir / "frame_model.glb"
        shutil.copy2(painted_glb, glb)
        thumbnail = output_dir / "thumbnail_hero.webp"
        with Image.open(reference_image) as image:
            image.save(thumbnail, "WEBP")
        metadata_path = output_dir / "metadata.json"
        metadata_path.write_text(json.dumps({"schema_version": 1, **metadata}))
        return ExportResult(glb, thumbnail, metadata_path)


class FailingSegmenter(MockSegmenter):
    def segment(self, images, output_dir):
        raise RuntimeError("segmentation failed")


def prepare_job(tmp_path: Path) -> tuple[InMemoryJobRepository, Job]:
    repository = InMemoryJobRepository()
    job = Job(
        merchant_id="merchant-1",
        product_sku="sku-1",
        frame_width_mm=140,
        status=JobStatus.VALIDATING,
        progress=5,
    )
    repository.create(job)
    workspace = tmp_path / "work" / str(job.id)
    normalized = workspace / "normalized"
    normalized.mkdir(parents=True)
    image = normalized / "front.png"
    Image.new("RGB", (64, 32), "white").save(image)
    raw = workspace / "raw"
    raw.mkdir()
    raw_image = raw / "front.png"
    shutil.copy2(image, raw_image)
    (workspace / "input_manifest.json").write_text(
        json.dumps({"images": {"front": {"normalized_path": str(image), "raw_path": str(raw_image)}}})
    )
    return repository, job


def build_pipeline(tmp_path: Path, repository: InMemoryJobRepository, segmenter=None) -> GenerationPipeline:
    return GenerationPipeline(
        repository=repository,
        storage=LocalObjectStorage(tmp_path / "storage"),
        segmenter=segmenter or MockSegmenter(),
        landmark_detector=MockLandmarkDetector(),
        deformer=FakeDeformer(),
        projector=FakeProjector(),
        painter=MockPainter(),
        exporter=FakeExporter(),
        work_root=tmp_path / "work",
        retain_workspaces=True,
    )


def test_mock_pipeline_reaches_review_and_uploads_preview(tmp_path: Path) -> None:
    repository, job = prepare_job(tmp_path)

    result = build_pipeline(tmp_path, repository).run(job.id)

    assert result.error is None, result.error
    assert result.status == JobStatus.READY_FOR_REVIEW
    assert result.progress == 100
    assert result.outputs.preview_glb_url is not None
    assert repository.get(job.id).status == JobStatus.READY_FOR_REVIEW


def test_pipeline_persists_sanitized_failure(tmp_path: Path) -> None:
    repository, job = prepare_job(tmp_path)

    result = build_pipeline(tmp_path, repository, FailingSegmenter()).run(job.id)

    assert result.status == JobStatus.FAILED
    assert result.error is not None
    assert result.error.code == "pipeline_failed"
    assert result.error.message == "segmentation failed"


def test_pipeline_restores_durable_inputs_on_a_fresh_worker(tmp_path: Path) -> None:
    repository, job = prepare_job(tmp_path)
    storage = LocalObjectStorage(tmp_path / "storage")
    workspace = tmp_path / "work" / str(job.id)
    raw_key = f"assets/merchant-1/sku-1/{job.id}/raw/front.png"
    normalized_key = f"assets/merchant-1/sku-1/{job.id}/inputs/normalized/front.png"
    storage.put_file(workspace / "raw/front.png", raw_key, "image/png")
    storage.put_file(workspace / "normalized/front.png", normalized_key, "image/png")
    job.inputs = JobInputs(raw={"front": raw_key}, normalized={"front": normalized_key})
    repository.save(job)
    shutil.rmtree(workspace)

    pipeline = GenerationPipeline(
        repository=repository,
        storage=storage,
        segmenter=MockSegmenter(),
        landmark_detector=MockLandmarkDetector(),
        deformer=FakeDeformer(),
        projector=FakeProjector(),
        painter=MockPainter(),
        exporter=FakeExporter(),
        work_root=tmp_path / "work",
        retain_workspaces=True,
    )
    result = pipeline.run(job.id)

    assert result.error is None, result.error
    assert result.status == JobStatus.READY_FOR_REVIEW
    assert (workspace / "input_manifest.json").is_file()
