from __future__ import annotations

import importlib
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from eyewear_vto.segmentation import ModelArtifactError


@dataclass(frozen=True)
class PaintResult:
    glb_path: Path
    max_views: int
    resolution: int


class Painter(Protocol):
    def preflight(self) -> None: ...

    def paint(self, mesh_path: Path, reference_image: Path, output_dir: Path) -> PaintResult: ...


class HunyuanPbrPainter:
    def __init__(
        self,
        *,
        hf_cache: Path,
        realesrgan_checkpoint: Path,
        max_views: int = 6,
        resolution: int = 512,
    ) -> None:
        if max_views < 6 or max_views > 12:
            raise ValueError("Hunyuan PBR max_views must be between 6 and 12")
        if resolution not in {512, 768}:
            raise ValueError("Hunyuan PBR resolution must be 512 or 768")
        self.hf_cache = hf_cache
        self.realesrgan_checkpoint = realesrgan_checkpoint
        self.max_views = max_views
        self.resolution = resolution
        self._pipeline: Any = None

    @property
    def hunyuan_cache(self) -> Path:
        return self.hf_cache / "models--tencent--Hunyuan3D-2.1"

    @property
    def dinov2_cache(self) -> Path:
        return self.hf_cache / "models--facebook--dinov2-giant"

    def preflight(self) -> None:
        required = (self.hunyuan_cache, self.dinov2_cache, self.realesrgan_checkpoint)
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise ModelArtifactError(f"Missing Hunyuan PBR artifacts: {', '.join(missing)}")

    def _load_pipeline(self) -> Any:
        if self._pipeline is not None:
            return self._pipeline
        self.preflight()
        module = importlib.import_module("textureGenPipeline")
        config = module.Hunyuan3DPaintConfig(self.max_views, self.resolution)
        module_file = getattr(module, "__file__", None)
        if module_file is None:
            raise RuntimeError("Cannot resolve the Hunyuan paint module path")
        vendor_paint_root = Path(module_file).resolve().parent
        config.multiview_cfg_path = str(vendor_paint_root / "cfgs" / "hunyuan-paint-pbr.yaml")
        # These repository IDs resolve only from the baked Hugging Face cache because
        # the final image sets HF_HUB_OFFLINE=1.
        config.multiview_pretrained_path = "tencent/Hunyuan3D-2.1"
        config.dino_ckpt_path = "facebook/dinov2-giant"
        config.realesrgan_ckpt_path = str(self.realesrgan_checkpoint)
        self._pipeline = module.Hunyuan3DPaintPipeline(config)
        return self._pipeline

    def paint(self, mesh_path: Path, reference_image: Path, output_dir: Path) -> PaintResult:
        if not mesh_path.is_file():
            raise FileNotFoundError(mesh_path)
        if not reference_image.is_file():
            raise FileNotFoundError(reference_image)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_obj = output_dir / "textured_mesh.obj"
        pipeline = self._load_pipeline()
        pipeline(
            mesh_path=str(mesh_path),
            image_path=str(reference_image),
            output_mesh_path=str(output_obj),
            save_glb=True,
        )
        output_glb = output_obj.with_suffix(".glb")
        if not output_glb.is_file() or output_glb.stat().st_size == 0:
            raise RuntimeError("Hunyuan PBR did not produce the expected GLB")
        return PaintResult(glb_path=output_glb, max_views=self.max_views, resolution=self.resolution)


class MockPainter:
    def preflight(self) -> None:
        return None

    def paint(self, mesh_path: Path, reference_image: Path, output_dir: Path) -> PaintResult:
        if not reference_image.is_file():
            raise FileNotFoundError(reference_image)
        output_dir.mkdir(parents=True, exist_ok=True)
        destination = output_dir / "textured_mesh.glb"
        shutil.copy2(mesh_path, destination)
        return PaintResult(glb_path=destination, max_views=6, resolution=512)
