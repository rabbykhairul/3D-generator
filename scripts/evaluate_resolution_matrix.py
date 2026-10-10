"""Measure contour extraction on temporary resolution derivatives of local photos.

Run from the project root with .venv/bin/python scripts/evaluate_resolution_matrix.py.
Original projects and photo files are read-only. Larger derivatives contain no
additional source detail; their purpose is testing coordinate and memory paths.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
from shapely.geometry import Polygon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend import store  # noqa: E402
from backend.reconstruction import NeedsCorrection, extract_annotations  # noqa: E402

DEFAULT_PROJECTS = {
    "blue": "c8e4014e30044160ace24fe171a57b5e",
    "transparent": "8dd590d016a44d268efcb319d3f86437",
    "gold": "ce81f4ffd03e4b8cac759b32ec5ee650",
}
SIZES = (64, 100, 256, 1080, 4096)


def default_sources() -> list[dict]:
    sources = []
    for family, project_id in DEFAULT_PROJECTS.items():
        project = store.get_project(project_id)
        if project is None:
            continue
        for view in ("front", "right", "left"):
            photo = next((p for p in project["photos"] if p["view"] == view and p["role"] == "fitting"
                          and Path(p["path"]).is_file()), None)
            if photo is not None:
                sources.append({"family": family, "view": view, "path": Path(photo["path"]),
                                "source_photo_id": photo["id"]})
            if view == "front" and photo is None:
                break
            if view in ("right", "left") and photo is not None:
                break
    return sources


def validate_annotations(result: dict, width: int, height: int) -> list[str]:
    issues = []
    polygons = {}
    for annotation in result.get("annotations", []):
        part = annotation.get("part", "unknown")
        points = annotation.get("points", [])
        if len(points) < 3:
            issues.append(f"{part}: fewer than three points")
            continue
        if any(not np.isfinite((p["x"], p["y"])).all() or
               p["x"] < 0 or p["y"] < 0 or p["x"] > width or p["y"] > height for p in points):
            issues.append(f"{part}: point outside source image")
            continue
        polygon = Polygon([(p["x"], p["y"]) for p in points])
        if not polygon.is_valid or polygon.area <= 0:
            issues.append(f"{part}: invalid contour polygon")
        else:
            polygons[part] = polygon
        for name, anchor in annotation.get("anchors", {}).items():
            if not np.isfinite((anchor["x"], anchor["y"])).all() or not (
                    0 <= anchor["x"] <= width and 0 <= anchor["y"] <= height):
                issues.append(f"{part}: {name} anchor outside source image")
        if annotation.get("approved"):
            issues.append(f"{part}: automatic contour was approved")
    if {"front", "lens-left", "lens-right"} <= polygons.keys():
        outer, left, right = (polygons[part] for part in ("front", "lens-left", "lens-right"))
        if not outer.contains(left) or not outer.contains(right):
            issues.append("lens opening outside front contour")
        if left.intersects(right):
            issues.append("lens openings intersect")
    return issues


def missing_required_parts(result: dict, view: str) -> list[str]:
    parts = {a.get("part") for a in result.get("annotations", [])}
    if view == "front":
        required = {"front", "lens-left", "lens-right"}
    elif view in ("left", "right"):
        required = {f"temple-{view}"}
    else:
        required = set()
    return sorted(required - parts)


def derivative(source: np.ndarray, long_side: int) -> np.ndarray:
    height, width = source.shape[:2]
    scale = long_side / max(width, height)
    target = (max(1, round(width * scale)), max(1, round(height * scale)))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    return cv2.resize(source, target, interpolation=interpolation)


def draw_overlay(image: np.ndarray, result: dict, destination: Path) -> None:
    """Draw source-space suggestions on a bounded display preview."""
    height, width = image.shape[:2]
    display_long_side = max(800, min(1600, max(width, height)))
    display_scale = display_long_side / max(width, height)
    display = cv2.resize(
        image,
        (max(1, round(width * display_scale)), max(1, round(height * display_scale))),
        interpolation=cv2.INTER_NEAREST if display_scale > 1 else cv2.INTER_AREA,
    )
    sx, sy = display.shape[1] / width, display.shape[0] / height
    colors = [(0, 90, 240), (60, 190, 0), (200, 20, 170), (0, 170, 210)]
    for index, annotation in enumerate(result.get("annotations", [])):
        points = np.rint([(p["x"] * sx, p["y"] * sy) for p in annotation["points"]]).astype(np.int32)
        if len(points) >= 3:
            cv2.polylines(display, [points], True, colors[index % len(colors)], 2, cv2.LINE_AA)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), display):
        raise OSError(f"Could not write {destination}")


def evaluate_one(source: dict, long_side: int, temporary: Path, output: Path,
                 save_overlay: bool = False) -> dict:
    original = cv2.imread(str(source["path"]), cv2.IMREAD_COLOR)
    if original is None:
        raise OSError(f"Cannot decode {source['path']}")
    image = derivative(original, long_side)
    height, width = image.shape[:2]
    key = f"{source['family']}-{source['view']}-{long_side}"
    path = temporary / f"{key}.png"
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Could not write temporary derivative {path}")
    photo = {"id": key, "path": str(path), "view": source["view"], "role": "fitting",
             "width": width, "height": height}
    started = time.perf_counter()
    try:
        result = extract_annotations(photo)
        issues = validate_annotations(result, width, height)
        count = len(result.get("annotations", []))
        missing = missing_required_parts(result, source["view"])
        status = "invalid" if issues else "declined" if not count else "partial" if missing else "passed"
    except NeedsCorrection as exc:
        result = {"annotations": [], "warnings": [str(exc)]}
        issues = []
        missing = missing_required_parts(result, source["view"])
        count = 0
        status = "declined"
    except Exception as exc:
        result = {"annotations": [], "warnings": []}
        issues = [f"{type(exc).__name__}: {exc}"]
        missing = []
        count = 0
        status = "error"
    elapsed = round(time.perf_counter() - started, 3)
    if save_overlay:
        draw_overlay(image, result, output / f"{key}-overlay.png")
    return {"family": source["family"], "view": source["view"], "target_long_side": long_side,
            "source_photo_id": source.get("source_photo_id"),
            "source_size": [original.shape[1], original.shape[0]],
            "derivative_size": [width, height], "derived_from_source": True,
            "added_source_detail": False, "runtime_seconds": elapsed,
            "status": status, "annotation_count": count,
            "parts": [a.get("part") for a in result.get("annotations", [])],
            "missing_required_parts": missing, "issues": issues, "warnings": result.get("warnings", []),
            "resolution": result.get("resolution")}


def evaluate_matrix(sources: list[dict], output: Path, sizes=SIZES) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    with tempfile.TemporaryDirectory(prefix="eyewear-resolution-") as directory:
        temporary = Path(directory)
        for source in sources:
            for size in sizes:
                row = evaluate_one(source, size, temporary, output,
                                   save_overlay=size in (100, 4096))
                rows.append(row)
                print(f"{source['family']} {source['view']} {size}: {row['status']} "
                      f"{row['annotation_count']} contours in {row['runtime_seconds']}s", flush=True)
    report = {"method": "Source photos resized to requested long side; >source sizes add no detail. "
                        "An automatic contour passes only if every returned polygon is valid and all points/anchors "
                        "are inside derivative source coordinates. A declined extraction is recorded separately.",
              "sizes": list(sizes), "rows": rows,
              "counts": {status: sum(row["status"] == status for row in rows)
                         for status in ("passed", "partial", "declined", "invalid", "error")}}
    (output / "resolution-matrix.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/multiview-resolution")
    args = parser.parse_args()
    sources = default_sources()
    if not sources:
        raise SystemExit("No local source photos found")
    report = evaluate_matrix(sources, args.output)
    print(report["counts"])


if __name__ == "__main__":
    main()
