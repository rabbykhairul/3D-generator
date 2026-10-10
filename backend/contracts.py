from typing import Literal
import math
from pydantic import BaseModel, Field, field_validator

Category = Literal['opaque-full-rim', 'thin-metal', 'transparent-full-rim', 'semi-rimless', 'rimless', 'sports-shield']

class Measurement(BaseModel):
    value: float = Field(ge=0, le=500, allow_inf_nan=False)
    source: Literal['measured', 'published', 'inferred', 'default', 'manual'] = 'manual'

class Point(BaseModel):
    x: float = Field(allow_inf_nan=False)
    y: float = Field(allow_inf_nan=False)

class DetectionEvidence(BaseModel):
    schema_version: Literal[1] = 1
    state: Literal['observed', 'inferred', 'uncertain'] = 'uncertain'
    method: Literal['geometric-anatomy-audit'] = 'geometric-anatomy-audit'
    review_required: bool = True
    notes: list[str] = Field(default_factory=list, max_length=100)

class ComponentRegistration(BaseModel):
    source_photo_id: str
    source_points: list[Point] = Field(min_length=3,max_length=2000)
    source_to_front_homography: list[list[float]]
    method: Literal['inferred-front-plane-registration'] = 'inferred-front-plane-registration'

    @field_validator('source_to_front_homography')
    @classmethod
    def valid_homography(cls, value):
        if len(value)!=3 or any(len(row)!=3 for row in value) or any(not math.isfinite(v) for row in value for v in row):
            raise ValueError('Registration requires a finite 3x3 homography')
        a,b,c=value
        determinant=a[0]*(b[1]*c[2]-b[2]*c[1])-a[1]*(b[0]*c[2]-b[2]*c[0])+a[2]*(b[0]*c[1]-b[1]*c[0])
        if abs(determinant)<1e-12:raise ValueError('Registration homography is singular')
        return value

class Annotation(BaseModel):
    photo_id: str
    part: str  # front, lens-left, lens-right, temple-left, temple-right, shield, top
    points: list[Point] = Field(default_factory=list)
    anchors: dict[str, Point] = Field(default_factory=dict)
    anchor_provenance: dict[str, str] = Field(default_factory=dict)
    registration: ComponentRegistration | None = None
    approved: bool = False
    source: str = 'manual'
    evidence: DetectionEvidence | None = None

class Material(BaseModel):
    color: str = '#252a32'
    roughness: float = Field(default=0.35, ge=0, le=1)
    metalness: float = Field(default=0, ge=0, le=1)
    opacity: float = Field(default=1, ge=0, le=1)
    lens_color: str = '#b5d9e5'
    lens_opacity: float = Field(default=0.16, ge=0, le=1)

class ProjectCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    category: Category = 'opaque-full-rim'

class PhotoViewPatch(BaseModel):
    view: str

class ReconstructionSettings(BaseModel):
    joint_front_fit: bool = False

class ProjectPatch(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=120)
    category: Category | None = None
    measurements: dict[str, Measurement] | None = None
    material: Material | None = None
    annotations: list[Annotation] | None = None
    notes: str | None = None
    reconstruction_settings: ReconstructionSettings | None = None

class AnatomyReview(BaseModel):
    annotations: list[Annotation] = Field(default_factory=list, max_length=100)
    category: Category | None = None

class MaskStroke(BaseModel):
    mode: Literal['add','erase']
    radius: float = Field(ge=1,le=200,allow_inf_nan=False)
    points: list[Point] = Field(min_length=1,max_length=2000)

class MaskRefine(BaseModel):
    width: int = Field(ge=1,le=40000)
    height: int = Field(ge=1,le=40000)
    points: list[Point] = Field(default_factory=list,max_length=10000)
    strokes: list[MaskStroke] = Field(min_length=1,max_length=100)

# Engine interface: reconstruct(snapshot: dict, output_dir: pathlib.Path) -> dict
# Snapshot is complete Project JSON. Photos include absolute `path` for worker only,
# URL for UI, id/view/role/width/height. Measurements map names to {value,source}.
# Required names: frame_width, lens_width, lens_height, bridge_width, temple_length.
# Optional: frame_thickness, temple_thickness, front_curve, temple_spread.
# Engine writes model.glb + metadata.json. Returns metadata JSON with parts,
# warnings, dimensions_mm, checks, provenance, elapsed_seconds and coordinate_system.
# Engine raises NeedsCorrection(message, details?) for missing/ambiguous geometry.
