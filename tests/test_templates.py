import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from eyewear_vto.landmarks import MockLandmarkDetector
from eyewear_vto.templates import TemplateCatalog, TemplateCatalogDocument


def template_record(identifier: str, width: float, height: float) -> dict[str, object]:
    return {
        "id": identifier,
        "version": "1.0.0",
        "family": identifier,
        "mesh": f"{identifier}/frame.glb",
        "deformation_map": f"{identifier}/deformation.json",
        "nominal_frame_width_mm": width,
        "nominal_frame_height_mm": height,
        "nominal_temple_length_mm": 145,
        "semantic_nodes": ["frame", "lens_left", "lens_right", "temple_left", "temple_right", "hardware"],
    }


def test_template_schema_requires_semantic_lens_nodes() -> None:
    record = template_record("round-v1", 140, 48)
    record["semantic_nodes"] = ["frame"]

    with pytest.raises(ValidationError, match="lens_left"):
        TemplateCatalogDocument.model_validate({"schema_version": 1, "templates": [record]})


def test_catalog_selects_closest_front_aspect_ratio(tmp_path: Path) -> None:
    records = [template_record("tall-v1", 120, 60), template_record("wide-v1", 140, 35)]
    for record in records:
        mesh = tmp_path / str(record["mesh"])
        deformation = tmp_path / str(record["deformation_map"])
        mesh.parent.mkdir(parents=True, exist_ok=True)
        mesh.write_bytes(b"glb")
        deformation.write_text("{}")
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(json.dumps({"schema_version": 1, "templates": records}))
    image = tmp_path / "front.png"
    from PIL import Image

    Image.new("RGB", (200, 100), "white").save(image)
    landmarks = MockLandmarkDetector().detect({"front": image})

    selected = TemplateCatalog.load(catalog_path).select(landmarks)

    assert selected.spec.id == "wide-v1"
