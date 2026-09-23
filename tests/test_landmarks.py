from pathlib import Path

import pytest
from PIL import Image

from eyewear_vto.landmarks import LANDMARK_NAMES, MockLandmarkDetector, YoloEyewearPoseDetector
from eyewear_vto.segmentation import ModelArtifactError


def test_yolo_pose_preflight_requires_custom_weight(tmp_path: Path) -> None:
    with pytest.raises(ModelArtifactError, match="eyewear_pose.pt"):
        YoloEyewearPoseDetector(tmp_path / "eyewear_pose.pt").preflight()


def test_mock_landmarks_use_named_contract(tmp_path: Path) -> None:
    image_path = tmp_path / "front.png"
    Image.new("RGB", (200, 100), "white").save(image_path)

    result = MockLandmarkDetector().detect({"front": image_path})
    front = result.views["front"]

    assert tuple(point.name for point in front.points) == LANDMARK_NAMES
    assert front.points[0].x == 10
    assert front.points[0].y == 50
    assert front.detection_confidence == 1.0
