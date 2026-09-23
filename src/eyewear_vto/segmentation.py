from __future__ import annotations

import importlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from PIL import Image


class ModelArtifactError(RuntimeError):
    pass


@dataclass(frozen=True)
class SegmentationResult:
    masks: Mapping[str, Path]
    scores: Mapping[str, float]


class Segmenter(Protocol):
    def preflight(self) -> None: ...

    def segment(self, images: Mapping[str, Path], output_dir: Path) -> SegmentationResult: ...


class Sam2Segmenter:
    """Lazy SAM 2.1 adapter.

    Heavy dependencies are imported only by ``segment``. Importing the web service on a CPU
    development machine therefore never initializes Torch or CUDA.
    """

    def __init__(self, checkpoint: Path, config: Path, device: str = "cuda") -> None:
        self.checkpoint = checkpoint
        self.config = config
        self.device = device
        self._predictor: Any = None

    def preflight(self) -> None:
        missing = [str(path) for path in (self.checkpoint, self.config) if not path.is_file()]
        if missing:
            raise ModelArtifactError(f"Missing SAM 2 artifacts: {', '.join(missing)}")

    def _load_predictor(self) -> Any:
        if self._predictor is not None:
            return self._predictor
        self.preflight()
        torch = importlib.import_module("torch")
        build_module = importlib.import_module("sam2.build_sam")
        predictor_module = importlib.import_module("sam2.sam2_image_predictor")
        model = build_module.build_sam2(str(self.config), str(self.checkpoint), device=self.device)
        model.eval()
        self._predictor = predictor_module.SAM2ImagePredictor(model)
        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("SAM 2 is configured for CUDA but CUDA is unavailable")
        return self._predictor

    def segment(self, images: Mapping[str, Path], output_dir: Path) -> SegmentationResult:
        np = importlib.import_module("numpy")
        torch = importlib.import_module("torch")
        predictor = self._load_predictor()
        output_dir.mkdir(parents=True, exist_ok=True)
        masks: dict[str, Path] = {}
        scores: dict[str, float] = {}

        for view, image_path in images.items():
            with Image.open(image_path) as source:
                rgb = source.convert("RGB")
                image_array = np.asarray(rgb)
                width, height = rgb.size
            prompt_box = np.array([0.02 * width, 0.02 * height, 0.98 * width, 0.98 * height])
            with torch.inference_mode(), torch.autocast(self.device, dtype=torch.bfloat16):
                predictor.set_image(image_array)
                predicted_masks, predicted_scores, _ = predictor.predict(
                    box=prompt_box,
                    multimask_output=True,
                )
            best_index = int(np.argmax(predicted_scores))
            mask = (predicted_masks[best_index].astype("uint8") * 255)
            mask_path = output_dir / f"{view}.png"
            Image.fromarray(mask, mode="L").save(mask_path)
            masks[view] = mask_path
            scores[view] = float(predicted_scores[best_index])

        return SegmentationResult(masks=masks, scores=scores)


class MockSegmenter:
    def preflight(self) -> None:
        return None

    def segment(self, images: Mapping[str, Path], output_dir: Path) -> SegmentationResult:
        output_dir.mkdir(parents=True, exist_ok=True)
        masks: dict[str, Path] = {}
        scores: dict[str, float] = {}
        for view, image_path in images.items():
            with Image.open(image_path) as source:
                mask = Image.new("L", source.size, 255)
            destination = output_dir / f"{view}.png"
            mask.save(destination)
            masks[view] = destination
            scores[view] = 1.0
        return SegmentationResult(masks=masks, scores=scores)

