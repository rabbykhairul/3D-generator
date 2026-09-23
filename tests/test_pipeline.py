from __future__ import annotations

import json
import shutil
from pathlib import Path

from PIL import Image

from eyewear_vto.jobs import InMemoryJobRepository, Job, JobInputs, JobStatus, retry_job
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


class OutOfMemorySegmenter(MockSegmenter):
    def segment(self, images, output_dir):
        raise MemoryError("CUDA out of memory")


class TimeoutSegmenter(MockSegmenter):
    def segment(self, images, output_dir):
        raise TimeoutError("object storage timeout")


class FailingProjector(FakeProjector):
    def project(self, mesh, images, segmentation, output_dir: Path) -> ProjectionResult:
        raise RuntimeError("projection interrupted")


class NeverSegmenter(MockSegmenter):
    def segment(self, images, output_dir):
        raise AssertionError("segmentation should have been restored from its checkpoint")


class NeverLandmarkDetector(MockLandmarkDetector):
    def detect(self, images):
        raise AssertionError("landmarks should have been restored from their checkpoint")


class NeverDeformer(FakeDeformer):
    def deform(self, landmarks, frame_width_mm: float, output_dir: Path) -> DeformationResult:
        raise AssertionError("deformation should have been restored from its checkpoint")


class CountingSegmenter(MockSegmenter):
    def __init__(self) -> None:
        self.calls = 0

    def segment(self, images, output_dir):
        self.calls += 1
        return super().segment(images, output_dir)


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


def make_inputs_durable(tmp_path: Path, repository: InMemoryJobRepository, job: Job) -> LocalObjectStorage:
    storage = LocalObjectStorage(tmp_path / "storage")
    workspace = tmp_path / "work" / str(job.id)
    raw_key = f"assets/merchant-1/sku-1/{job.id}/raw/front.png"
    normalized_key = f"assets/merchant-1/sku-1/{job.id}/inputs/normalized/front.png"
    storage.put_file(workspace / "raw/front.png", raw_key, "image/png")
    storage.put_file(workspace / "normalized/front.png", normalized_key, "image/png")
    job.inputs = JobInputs(raw={"front": raw_key}, normalized={"front": normalized_key})
    repository.save(job)
    return storage


def pipeline_with_components(
    tmp_path: Path,
    repository: InMemoryJobRepository,
    storage: LocalObjectStorage,
    *,
    segmenter=None,
    landmark_detector=None,
    deformer=None,
    projector=None,
    implementation_versions=None,
) -> GenerationPipeline:
    return GenerationPipeline(
        repository=repository,
        storage=storage,
        segmenter=segmenter or MockSegmenter(),
        landmark_detector=landmark_detector or MockLandmarkDetector(),
        deformer=deformer or FakeDeformer(),
        projector=projector or FakeProjector(),
        painter=MockPainter(),
        exporter=FakeExporter(),
        work_root=tmp_path / "work",
        retain_workspaces=True,
        implementation_versions=implementation_versions,
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


def test_pipeline_classifies_capacity_failure_as_retryable(tmp_path: Path) -> None:
    repository, job = prepare_job(tmp_path)

    result = build_pipeline(tmp_path, repository, OutOfMemorySegmenter()).run(job.id)

    assert result.status == JobStatus.FAILED
    assert result.error is not None
    assert result.error.code == "capacity_exhausted"
    assert result.error.retryable is True


def test_pipeline_classifies_transient_infrastructure_failure(tmp_path: Path) -> None:
    repository, job = prepare_job(tmp_path)

    result = build_pipeline(tmp_path, repository, TimeoutSegmenter()).run(job.id)

    assert result.status == JobStatus.FAILED
    assert result.error is not None
    assert result.error.code == "infrastructure_transient"
    assert result.error.retryable is True


def test_pipeline_restores_durable_inputs_on_a_fresh_worker(tmp_path: Path) -> None:
    repository, job = prepare_job(tmp_path)
    storage = make_inputs_durable(tmp_path, repository, job)
    workspace = tmp_path / "work" / str(job.id)
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


def test_retry_restores_completed_stages_on_a_fresh_worker(tmp_path: Path) -> None:
    repository, job = prepare_job(tmp_path)
    storage = make_inputs_durable(tmp_path, repository, job)
    workspace = tmp_path / "work" / str(job.id)
    first = pipeline_with_components(
        tmp_path,
        repository,
        storage,
        projector=FailingProjector(),
    ).run(job.id)

    assert first.status == JobStatus.FAILED
    assert [checkpoint.stage for checkpoint in first.checkpoints] == [
        "segmentation",
        "landmarks",
        "deformation",
    ]
    shutil.rmtree(workspace)
    retried = repository.get(job.id)
    retry_job(retried)
    repository.save(retried)

    result = pipeline_with_components(
        tmp_path,
        repository,
        storage,
        segmenter=NeverSegmenter(),
        landmark_detector=NeverLandmarkDetector(),
        deformer=NeverDeformer(),
    ).run(job.id)

    assert result.error is None, result.error
    assert result.status == JobStatus.READY_FOR_REVIEW
    assert [checkpoint.stage for checkpoint in result.checkpoints] == [
        "segmentation",
        "landmarks",
        "deformation",
        "projection",
        "painting",
        "export",
        "upload",
    ]
    assert {checkpoint.attempt for checkpoint in result.checkpoints[:3]} == {1}
    assert {checkpoint.attempt for checkpoint in result.checkpoints[3:]} == {2}


def test_changed_stage_version_invalidates_that_checkpoint(tmp_path: Path) -> None:
    repository, job = prepare_job(tmp_path)
    storage = make_inputs_durable(tmp_path, repository, job)
    first = pipeline_with_components(
        tmp_path,
        repository,
        storage,
        projector=FailingProjector(),
    ).run(job.id)
    assert first.status == JobStatus.FAILED
    retried = repository.get(job.id)
    retry_job(retried)
    repository.save(retried)
    versions = {
        "segmentation": "2",
        "landmarks": "1",
        "deformation": "1",
        "projection": "1",
        "painting": "1",
        "export": "1",
        "upload": "1",
    }
    segmenter = CountingSegmenter()

    result = pipeline_with_components(
        tmp_path,
        repository,
        storage,
        segmenter=segmenter,
        implementation_versions=versions,
    ).run(job.id)

    assert result.status == JobStatus.READY_FOR_REVIEW
    assert segmenter.calls == 1
    segmentation_versions = [
        checkpoint.implementation_version
        for checkpoint in result.checkpoints
        if checkpoint.stage == "segmentation"
    ]
    assert segmentation_versions == ["1", "2"]
    stage_counts = {
        stage: sum(checkpoint.stage == stage for checkpoint in result.checkpoints)
        for stage in ("segmentation", "landmarks", "deformation", "projection", "painting", "export", "upload")
    }
    assert stage_counts == {
        "segmentation": 2,
        "landmarks": 2,
        "deformation": 2,
        "projection": 1,
        "painting": 1,
        "export": 1,
        "upload": 1,
    }


def test_missing_checkpoint_artifact_recomputes_from_that_stage(tmp_path: Path) -> None:
    repository, job = prepare_job(tmp_path)
    storage = make_inputs_durable(tmp_path, repository, job)
    workspace = tmp_path / "work" / str(job.id)
    first = pipeline_with_components(
        tmp_path,
        repository,
        storage,
        projector=FailingProjector(),
    ).run(job.id)
    segmentation = next(checkpoint for checkpoint in first.checkpoints if checkpoint.stage == "segmentation")
    missing_key = segmentation.artifacts["scores"]
    (storage.root / missing_key).unlink()
    shutil.rmtree(workspace)
    retried = repository.get(job.id)
    retry_job(retried)
    repository.save(retried)
    segmenter = CountingSegmenter()

    result = pipeline_with_components(
        tmp_path,
        repository,
        storage,
        segmenter=segmenter,
    ).run(job.id)

    assert result.status == JobStatus.READY_FOR_REVIEW
    assert segmenter.calls == 1
    assert sum(checkpoint.stage == "segmentation" for checkpoint in result.checkpoints) == 2
