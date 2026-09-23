from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, Field, model_validator

from eyewear_vto.landmarks import LandmarkResult, ViewLandmarks

REQUIRED_SEMANTIC_NODES = {
    "frame",
    "lens_left",
    "lens_right",
    "temple_left",
    "temple_right",
    "hardware",
}


class TemplateSpec(BaseModel):
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]+$")
    version: str
    family: str
    mesh: str
    deformation_map: str
    nominal_frame_width_mm: float = Field(gt=0)
    nominal_frame_height_mm: float = Field(gt=0)
    nominal_temple_length_mm: float = Field(gt=0)
    semantic_nodes: tuple[str, ...]

    @model_validator(mode="after")
    def validate_nodes(self) -> TemplateSpec:
        missing = REQUIRED_SEMANTIC_NODES - set(self.semantic_nodes)
        if missing:
            raise ValueError(f"Template is missing semantic nodes: {', '.join(sorted(missing))}")
        return self

    @property
    def aspect_ratio(self) -> float:
        return self.nominal_frame_width_mm / self.nominal_frame_height_mm


class TemplateCatalogDocument(BaseModel):
    schema_version: int = Field(ge=1, le=1)
    templates: tuple[TemplateSpec, ...]


@dataclass(frozen=True)
class ResolvedTemplate:
    spec: TemplateSpec
    mesh_path: Path
    deformation_map_path: Path


class TemplateCatalog:
    def __init__(self, root: Path, document: TemplateCatalogDocument) -> None:
        self.root = root
        self.document = document

    @classmethod
    def load(cls, path: Path) -> TemplateCatalog:
        return cls(path.parent, TemplateCatalogDocument.model_validate_json(path.read_text(encoding="utf-8")))

    def resolve(self, spec: TemplateSpec) -> ResolvedTemplate:
        mesh_path = self._resolve_relative(spec.mesh)
        deformation_path = self._resolve_relative(spec.deformation_map)
        missing = [str(path) for path in (mesh_path, deformation_path) if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Template {spec.id} is incomplete: {', '.join(missing)}")
        return ResolvedTemplate(spec, mesh_path, deformation_path)

    def select(self, landmarks: LandmarkResult) -> ResolvedTemplate:
        front = landmarks.views.get("front")
        if front is None:
            raise ValueError("Front landmarks are required for template selection")
        measured_ratio = front_aspect_ratio(front)
        if not self.document.templates:
            raise ValueError("Template catalog is empty")
        selected = min(
            self.document.templates,
            key=lambda item: abs(math.log(item.aspect_ratio / measured_ratio)),
        )
        return self.resolve(selected)

    def preflight(self) -> tuple[ResolvedTemplate, ...]:
        return tuple(self.resolve(spec) for spec in self.document.templates)

    def _resolve_relative(self, value: str) -> Path:
        candidate = (self.root / value).resolve()
        root = self.root.resolve()
        if not candidate.is_relative_to(root):
            raise ValueError("Template path escapes catalog root")
        return candidate


def front_aspect_ratio(front: ViewLandmarks) -> float:
    points = {point.name: point for point in front.points}
    required = {
        "rim_left_outer",
        "rim_right_outer",
        "rim_left_top",
        "rim_left_bottom",
        "rim_right_top",
        "rim_right_bottom",
    }
    missing = required - points.keys()
    if missing:
        raise ValueError(f"Missing front landmarks: {', '.join(sorted(missing))}")
    width = abs(points["rim_right_outer"].x - points["rim_left_outer"].x)
    left_height = abs(points["rim_left_bottom"].y - points["rim_left_top"].y)
    right_height = abs(points["rim_right_bottom"].y - points["rim_right_top"].y)
    height = (left_height + right_height) / 2
    if width <= 0 or height <= 0:
        raise ValueError("Front landmarks have invalid dimensions")
    return width / height

