from pathlib import Path

import pytest
from PIL import Image

from eyewear_vto.segmentation import MockSegmenter, ModelArtifactError, Sam2Segmenter


def test_sam_preflight_reports_all_missing_artifacts(tmp_path: Path) -> None:
    segmenter = Sam2Segmenter(tmp_path / "sam.pt", tmp_path / "sam.yaml")

    with pytest.raises(ModelArtifactError, match="sam.pt.*sam.yaml"):
        segmenter.preflight()


def test_sam_preflight_does_not_import_gpu_runtime(tmp_path: Path) -> None:
    checkpoint = tmp_path / "sam.pt"
    config = tmp_path / "sam.yaml"
    checkpoint.write_bytes(b"placeholder")
    config.write_text("model: placeholder")

    Sam2Segmenter(checkpoint, config).preflight()


def test_mock_segmenter_honors_output_contract(tmp_path: Path) -> None:
    image_path = tmp_path / "front.png"
    Image.new("RGB", (32, 16), "white").save(image_path)

    result = MockSegmenter().segment({"front": image_path}, tmp_path / "masks")

    assert result.scores == {"front": 1.0}
    with Image.open(result.masks["front"]) as mask:
        assert mask.mode == "L"
        assert mask.size == (32, 16)
