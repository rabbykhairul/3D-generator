from pathlib import Path

import pytest

from eyewear_vto.painting import HunyuanPbrPainter, MockPainter
from eyewear_vto.segmentation import ModelArtifactError


def test_hunyuan_preflight_requires_all_offline_artifacts(tmp_path: Path) -> None:
    painter = HunyuanPbrPainter(
        hf_cache=tmp_path / "cache",
        realesrgan_checkpoint=tmp_path / "RealESRGAN_x4plus.pth",
    )

    with pytest.raises(ModelArtifactError, match="Hunyuan3D-2.1.*dinov2.*RealESRGAN"):
        painter.preflight()


@pytest.mark.parametrize("views", [5, 13])
def test_hunyuan_rejects_unsupported_view_counts(tmp_path: Path, views: int) -> None:
    with pytest.raises(ValueError, match="max_views"):
        HunyuanPbrPainter(
            hf_cache=tmp_path,
            realesrgan_checkpoint=tmp_path / "realesrgan.pth",
            max_views=views,
        )


def test_mock_painter_copies_mesh_without_gpu(tmp_path: Path) -> None:
    mesh = tmp_path / "mesh.glb"
    reference = tmp_path / "front.png"
    mesh.write_bytes(b"mock glb")
    reference.write_bytes(b"mock image")

    result = MockPainter().paint(mesh, reference, tmp_path / "paint")

    assert result.glb_path.read_bytes() == b"mock glb"
    assert result.resolution == 512
