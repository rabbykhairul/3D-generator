from __future__ import annotations

import importlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from PIL import Image

from eyewear_vto.segmentation import ModelArtifactError

LANDMARK_NAMES = (
    "rim_left_outer",
    "rim_left_inner",
    "rim_left_top",
    "rim_left_bottom",
    "rim_right_inner",
    "rim_right_outer",
    "rim_right_top",
    "rim_right_bottom",
    "bridge_left",
    "bridge_right",
    "hinge_left",
    "hinge_right",
    "temple_left_end",
    "temple_right_end",
    "nose_pad_left",
    "nose_pad_right",
)


@dataclass(frozen=True)
class Landmark:
    name: str
    x: float
    y: float
    confidence: float


@dataclass(frozen=True)
class ViewLandmarks:
    view: str
    points: tuple[Landmark, ...]
    detection_confidence: float


@dataclass(frozen=True)
class LandmarkResult:
    views: Mapping[str, ViewLandmarks]


class LandmarkDetector(Protocol):
    def preflight(self) -> None: ...

    def detect(self, images: Mapping[str, Path]) -> LandmarkResult: ...


class YoloEyewearPoseDetector:
    def __init__(self, checkpoint: Path, landmark_names: Sequence[str] = LANDMARK_NAMES) -> None:
        self.checkpoint = checkpoint
        self.landmark_names = tuple(landmark_names)
        self._model: Any = None

    def preflight(self) -> None:
        if not self.checkpoint.is_file():
            raise ModelArtifactError(f"Missing eyewear pose checkpoint: {self.checkpoint}")

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model
        self.preflight()
        ultralytics = importlib.import_module("ultralytics")
        self._model = ultralytics.YOLO(str(self.checkpoint), task="pose")
        return self._model

    def detect(self, images: Mapping[str, Path]) -> LandmarkResult:
        model = self._load_model()
        views: dict[str, ViewLandmarks] = {}
        for view, image_path in images.items():
            prediction = model.predict(source=str(image_path), verbose=False)[0]
            if prediction.boxes is None or len(prediction.boxes) == 0 or prediction.keypoints is None:
                raise RuntimeError(f"No eyewear pose detected in {view}")
            confidences = prediction.boxes.conf.detach().cpu().tolist()
            best_index = max(range(len(confidences)), key=confidences.__getitem__)
            xy = prediction.keypoints.xy[best_index].detach().cpu().tolist()
            keypoint_confidence = prediction.keypoints.conf[best_index].detach().cpu().tolist()
            if len(xy) != len(self.landmark_names):
                raise RuntimeError(
                    f"Pose model returned {len(xy)} keypoints; expected {len(self.landmark_names)}"
                )
            points = tuple(
                Landmark(name=name, x=float(point[0]), y=float(point[1]), confidence=float(confidence))
                for name, point, confidence in zip(self.landmark_names, xy, keypoint_confidence, strict=True)
            )
            views[view] = ViewLandmarks(
                view=view,
                points=points,
                detection_confidence=float(confidences[best_index]),
            )
        return LandmarkResult(views=views)


class MockLandmarkDetector:
    def preflight(self) -> None:
        return None

    def detect(self, images: Mapping[str, Path]) -> LandmarkResult:
        views: dict[str, ViewLandmarks] = {}
        for view, image_path in images.items():
            with Image.open(image_path) as image:
                width, height = image.size
            positions = (
                (0.05, 0.5),
                (0.42, 0.5),
                (0.24, 0.28),
                (0.24, 0.72),
                (0.58, 0.5),
                (0.95, 0.5),
                (0.76, 0.28),
                (0.76, 0.72),
                (0.43, 0.5),
                (0.57, 0.5),
                (0.08, 0.5),
                (0.92, 0.5),
                (0.02, 0.55),
                (0.98, 0.55),
                (0.46, 0.62),
                (0.54, 0.62),
            )
            points = tuple(
                Landmark(name=name, x=x * width, y=y * height, confidence=1.0)
                for name, (x, y) in zip(LANDMARK_NAMES, positions, strict=True)
            )
            views[view] = ViewLandmarks(view=view, points=points, detection_confidence=1.0)
        return LandmarkResult(views=views)
