#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

REQUIRED_NODES = {"frame", "lens_left", "lens_right", "temple_left", "temple_right", "hardware"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    return parser.parse_args()


def resolve(root: Path, value: str) -> Path:
    path = (root / value).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Template path escapes root: {value}")
    return path


def validate_glb(path: Path) -> None:
    with path.open("rb") as source:
        header = source.read(12)
    if len(header) != 12:
        raise ValueError(f"Template GLB is truncated: {path}")
    magic, version, declared_length = struct.unpack("<4sII", header)
    if magic != b"glTF" or version != 2 or declared_length != path.stat().st_size:
        raise ValueError(f"Template is not a valid GLB 2.0 container: {path}")


def main() -> None:
    root = parse_args().root.resolve()
    catalog_path = root / "catalog.json"
    if not catalog_path.is_file():
        raise FileNotFoundError("Production template catalog is missing: /opt/templates/catalog.json")
    document = json.loads(catalog_path.read_text(encoding="utf-8"))
    if document.get("schema_version") != 1 or not document.get("templates"):
        raise ValueError("Template catalog must contain at least one schema-version-1 template")
    for template in document["templates"]:
        missing_nodes = REQUIRED_NODES - set(template.get("semantic_nodes", []))
        if missing_nodes:
            raise ValueError(f"{template.get('id')}: missing semantic nodes {sorted(missing_nodes)}")
        mesh = resolve(root, template["mesh"])
        deformation = resolve(root, template["deformation_map"])
        if not deformation.is_file():
            raise FileNotFoundError(deformation)
        validate_glb(mesh)
        deformation_document = json.loads(deformation.read_text(encoding="utf-8"))
        if deformation_document.get("schema_version") != 1 or not deformation_document.get("anchors"):
            raise ValueError(f"{template.get('id')}: deformation map has no anchors")


if __name__ == "__main__":
    main()
