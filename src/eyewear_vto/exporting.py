from __future__ import annotations

import json
import struct
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from PIL import Image, ImageOps

from eyewear_vto.pipeline import ExportResult

GLB_MAGIC = b"glTF"
GLB_VERSION = 2
JSON_CHUNK = 0x4E4F534A


class AssetQaError(ValueError):
    pass


@dataclass(frozen=True)
class GlbInspection:
    byte_length: int
    triangle_count: int
    node_names: tuple[str, ...]
    material_names: tuple[str, ...]


class AssetOptimizer(Protocol):
    def optimize(self, source: Path, destination: Path, max_triangles: int) -> None: ...


class GltfpackOptimizer:
    def __init__(self, executable: Path = Path("/usr/local/bin/gltfpack")) -> None:
        self.executable = executable

    def optimize(self, source: Path, destination: Path, max_triangles: int) -> None:
        if not self.executable.is_file():
            raise AssetQaError(f"gltfpack is unavailable: {self.executable}")
        source_inspection = inspect_glb(source, max_bytes=2**63 - 1, max_triangles=2**63 - 1)
        command = [
            str(self.executable),
            "-i",
            str(source),
            "-o",
            str(destination),
            "-cc",
            "-kn",
            "-km",
            "-ke",
        ]
        if source_inspection.triangle_count > max_triangles:
            ratio = max_triangles / source_inspection.triangle_count
            command.extend(("-si", f"{ratio:.8f}"))
        completed = subprocess.run(command, capture_output=True, text=True, check=False)  # noqa: S603
        if completed.returncode != 0:
            message = (completed.stderr or completed.stdout or "gltfpack failed").strip()
            raise AssetQaError(message[-1000:])
        if not destination.is_file() or destination.stat().st_size == 0:
            raise AssetQaError("gltfpack did not produce an output asset")


def inspect_glb(path: Path, *, max_bytes: int = 20 * 1024 * 1024, max_triangles: int = 100_000) -> GlbInspection:
    byte_length = path.stat().st_size
    if byte_length > max_bytes:
        raise AssetQaError(f"GLB exceeds {max_bytes} bytes")
    with path.open("rb") as source:
        header = source.read(12)
        if len(header) != 12:
            raise AssetQaError("GLB header is truncated")
        magic, version, declared_length = struct.unpack("<4sII", header)
        if magic != GLB_MAGIC or version != GLB_VERSION:
            raise AssetQaError("File is not a GLB 2.0 asset")
        if declared_length != byte_length:
            raise AssetQaError("GLB declared length does not match file size")
        chunk_header = source.read(8)
        if len(chunk_header) != 8:
            raise AssetQaError("GLB JSON chunk is missing")
        chunk_length, chunk_type = struct.unpack("<II", chunk_header)
        if chunk_type != JSON_CHUNK:
            raise AssetQaError("First GLB chunk must be JSON")
        try:
            document = json.loads(source.read(chunk_length).decode("utf-8").rstrip(" \t\r\n\x00"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise AssetQaError("GLB JSON chunk is invalid") from exc

    for collection in ("buffers", "images"):
        for item in document.get(collection, []):
            uri = item.get("uri")
            if uri and not str(uri).startswith("data:"):
                raise AssetQaError(f"GLB contains an external {collection[:-1]} URI")

    accessors = document.get("accessors", [])
    triangle_count = 0
    for mesh in document.get("meshes", []):
        for primitive in mesh.get("primitives", []):
            mode = primitive.get("mode", 4)
            if mode != 4:
                continue
            accessor_index = primitive.get("indices")
            if accessor_index is None:
                accessor_index = primitive.get("attributes", {}).get("POSITION")
            if isinstance(accessor_index, int) and accessor_index < len(accessors):
                triangle_count += int(accessors[accessor_index].get("count", 0)) // 3
    if triangle_count > max_triangles:
        raise AssetQaError(f"GLB exceeds {max_triangles} triangles")

    node_names = tuple(str(node.get("name", "")) for node in document.get("nodes", []))
    material_names = tuple(str(material.get("name", "")) for material in document.get("materials", []))
    return GlbInspection(byte_length, triangle_count, node_names, material_names)


class WebAssetExporter:
    def __init__(
        self,
        *,
        max_glb_bytes: int = 20 * 1024 * 1024,
        max_triangles: int = 100_000,
        require_semantic_lenses: bool = True,
        optimizer: AssetOptimizer | None = None,
    ) -> None:
        self.max_glb_bytes = max_glb_bytes
        self.max_triangles = max_triangles
        self.require_semantic_lenses = require_semantic_lenses
        self.optimizer = optimizer or GltfpackOptimizer()

    def export(
        self,
        painted_glb: Path,
        reference_image: Path,
        output_dir: Path,
        metadata: Mapping[str, object],
    ) -> ExportResult:
        output_dir.mkdir(parents=True, exist_ok=True)
        destination = output_dir / "frame_model.glb"
        self.optimizer.optimize(painted_glb, destination, self.max_triangles)
        inspection = inspect_glb(
            destination,
            max_bytes=self.max_glb_bytes,
            max_triangles=self.max_triangles,
        )
        if self.require_semantic_lenses:
            normalized_names = {name.lower() for name in inspection.node_names}
            if not {"lens_left", "lens_right"}.issubset(normalized_names):
                raise AssetQaError("GLB must preserve lens_left and lens_right nodes")

        thumbnail = output_dir / "thumbnail_hero.webp"
        with Image.open(reference_image) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            image.thumbnail((1024, 1024))
            image.save(thumbnail, "WEBP", quality=88, method=6)

        metadata_path = output_dir / "metadata.json"
        payload = {
            "schema_version": 1,
            "units": "meters",
            "up_axis": "+Y",
            "forward_axis": "+Z",
            "glb": {
                "bytes": inspection.byte_length,
                "triangles": inspection.triangle_count,
                "nodes": list(inspection.node_names),
                "materials": list(inspection.material_names),
            },
            **metadata,
        }
        metadata_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return ExportResult(destination, thumbnail, metadata_path)
