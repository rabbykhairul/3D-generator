from __future__ import annotations

import json
import shutil
import struct
import subprocess
from pathlib import Path

import pytest
from PIL import Image

from eyewear_vto.exporting import AssetQaError, GltfpackOptimizer, WebAssetExporter, inspect_glb


class CopyOptimizer:
    def optimize(self, source: Path, destination: Path, max_triangles: int) -> None:
        del max_triangles
        shutil.copy2(source, destination)


def write_glb(path: Path, document: dict[str, object]) -> None:
    encoded = json.dumps(document, separators=(",", ":")).encode("utf-8")
    encoded += b" " * ((4 - len(encoded) % 4) % 4)
    total_length = 12 + 8 + len(encoded)
    path.write_bytes(
        struct.pack("<4sII", b"glTF", 2, total_length)
        + struct.pack("<II", len(encoded), 0x4E4F534A)
        + encoded
    )


def valid_document() -> dict[str, object]:
    return {
        "asset": {"version": "2.0"},
        "accessors": [{"count": 300}],
        "meshes": [{"primitives": [{"indices": 0}]}],
        "nodes": [{"name": "lens_left"}, {"name": "lens_right"}, {"name": "frame"}],
        "materials": [{"name": "frame_material"}, {"name": "lens_material"}],
    }


def test_glb_inspection_reports_budget_values(tmp_path: Path) -> None:
    glb = tmp_path / "frame.glb"
    write_glb(glb, valid_document())

    inspection = inspect_glb(glb)

    assert inspection.triangle_count == 100
    assert "lens_left" in inspection.node_names


def test_glb_rejects_external_resources(tmp_path: Path) -> None:
    glb = tmp_path / "frame.glb"
    document = valid_document()
    document["images"] = [{"uri": "texture.png"}]
    write_glb(glb, document)

    with pytest.raises(AssetQaError, match="external image"):
        inspect_glb(glb)


def test_web_export_writes_thumbnail_and_metadata(tmp_path: Path) -> None:
    glb = tmp_path / "frame.glb"
    write_glb(glb, valid_document())
    reference = tmp_path / "hero.png"
    Image.new("RGB", (1200, 600), "white").save(reference)

    result = WebAssetExporter(optimizer=CopyOptimizer()).export(
        glb,
        reference,
        tmp_path / "output",
        {"frame_width_mm": 140, "template_id": "round-v1"},
    )

    metadata = json.loads(result.metadata_path.read_text())
    assert metadata["units"] == "meters"
    assert metadata["glb"]["triangles"] == 100
    assert result.thumbnail_path.is_file()


def test_gltfpack_optimizer_preserves_semantics_and_applies_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.glb"
    document = valid_document()
    document["accessors"] = [{"count": 600_000}]
    write_glb(source, document)
    executable = tmp_path / "gltfpack"
    executable.write_text("placeholder")
    destination = tmp_path / "optimized.glb"
    captured: list[str] = []

    def fake_run(command, **kwargs):
        del kwargs
        captured.extend(command)
        shutil.copy2(source, destination)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr("eyewear_vto.exporting.subprocess.run", fake_run)

    GltfpackOptimizer(executable).optimize(source, destination, max_triangles=100_000)

    assert captured[0] == str(executable)
    assert {"-cc", "-kn", "-km", "-ke", "-si"}.issubset(captured)
    ratio = float(captured[captured.index("-si") + 1])
    assert ratio == pytest.approx(0.5)
