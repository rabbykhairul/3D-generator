"""Optimistic, reproducible 2-D silhouette check against a reference photo.

Fits a pinhole camera to the evaluation image. The fit makes this a
camera-adjusted diagnostic, never a physical 3-D accuracy percentage.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import trimesh
from scipy.optimize import differential_evolution
from scipy.optimize import minimize


def reference_mask(path: Path, size: int, threshold: int) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Cannot read {path}")
    # Preserve source aspect ratio, including for user-supplied nonsquare photos.
    height, width = image.shape[:2]
    factor = min(size / width, size / height)
    fitted = cv2.resize(image, (max(1, round(width * factor)),
                                max(1, round(height * factor))), interpolation=cv2.INTER_AREA)
    canvas = np.full((size, size, 3), 255, np.uint8)
    left = (size - fitted.shape[1]) // 2
    top = (size - fitted.shape[0]) // 2
    canvas[top:top + fitted.shape[0], left:left + fitted.shape[1]] = fitted
    image = canvas
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Dark frame pixels only; near-white shadows and clear lens centers are out.
    mask = (gray < threshold).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    # Keep substantial components including separated arms, reject compression specks.
    clean = np.zeros_like(mask)
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] >= max(3, size * size // 12000):
            clean[labels == label] = 1
    return clean


def geometry(path: Path) -> list[np.ndarray]:
    scene = trimesh.load(path, force="scene")
    triangles = []
    for node in scene.graph.nodes_geometry:
        matrix, name = scene.graph[node]
        if name.startswith("lens_"):
            continue
        mesh = scene.geometry[name]
        vertices = trimesh.transform_points(mesh.vertices, matrix)
        triangles.append(vertices[mesh.faces])
    return triangles


def projected_vertices(meshes: list[np.ndarray], yaw: float, pitch: float,
                       roll: float, distance: float) -> list[np.ndarray]:
    yr = np.deg2rad(yaw)
    pr = np.deg2rad(pitch)
    cy, sy = np.cos(yr), np.sin(yr)
    cp, sp = np.cos(pr), np.sin(pr)
    rr = np.deg2rad(roll)
    cr, sr = np.cos(rr), np.sin(rr)
    # Model +Y points up, +Z forward; image Y points downward.
    result = []
    for tri in meshes:
        x = cy * tri[..., 0] + sy * tri[..., 2]
        z = -sy * tri[..., 0] + cy * tri[..., 2]
        y = -(cp * tri[..., 1] - sp * z)
        depth = sp * tri[..., 1] + cp * z
        x, y = cr * x - sr * y, sr * x + cr * y
        divisor = np.maximum(distance - depth, 0.02)
        result.append(np.stack((x / divisor, y / divisor), axis=-1))
    return result


def registration(meshes: list[np.ndarray], ref_box: tuple[int, int, int, int],
                 camera: tuple[float, float, float, float]) -> tuple[float, float, float]:
    projected = projected_vertices(meshes, *camera)
    all_points = np.concatenate([part.reshape(-1, 2) for part in projected])
    x0, y0 = all_points.min(axis=0)
    x1, y1 = all_points.max(axis=0)
    rx0, ry0, rx1, ry1 = ref_box
    # Uniform scale from the combined extent. Registration does not deform the mesh.
    scale = min((rx1 - rx0) / max(x1 - x0, 1e-6), (ry1 - ry0) / max(y1 - y0, 1e-6))
    tx = (rx0 + rx1 - scale * (x0 + x1)) / 2
    ty = (ry0 + ry1 - scale * (y0 + y1)) / 2
    return float(scale), float(tx), float(ty)


def render(meshes: list[np.ndarray], camera: tuple[float, float, float, float],
           size: int, transform: tuple[float, float, float]) -> np.ndarray:
    projected = projected_vertices(meshes, *camera)
    scale, tx, ty = transform
    mask = np.zeros((size, size), np.uint8)
    for part in projected:
        pixels = np.rint(part * scale + np.array([tx, ty])).astype(np.int32)
        for tri in pixels:
            cv2.fillConvexPoly(mask, tri, 1)
    return mask


def iou(a: np.ndarray, b: np.ndarray) -> float:
    intersection = np.count_nonzero(a & b)
    union = np.count_nonzero(a | b)
    return intersection / union if union else 0.0


def optimize_transform(meshes: list[np.ndarray], camera: tuple[float, float, float, float],
                       ref: np.ndarray, initial: tuple[float, float, float]) -> tuple[float, float, float]:
    """Bounded 2-D similarity fit; use only for calibration, then pin for all models."""
    size = ref.shape[0]
    scale, tx, ty = initial
    bounds = [(scale * .65, scale * 1.35), (tx - size * .16, tx + size * .16),
              (ty - size * .16, ty + size * .16)]

    def objective(p: np.ndarray) -> float:
        return 1 - iou(render(meshes, camera, size, tuple(map(float, p))), ref)

    result = minimize(objective, np.array(initial), method="Powell", bounds=bounds,
                      options={"maxiter": 35, "maxfev": 160, "xtol": .002, "ftol": .0003})
    fitted = tuple(map(float, result.x))
    return fitted if objective(result.x) < objective(np.array(initial)) else initial


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--threshold", type=int, default=215)
    parser.add_argument("--size", type=int, default=384)
    parser.add_argument("--registration", type=Path, help="Reuse camera and pixel transform from a prior report")
    parser.add_argument("--reference-state", choices=("open", "folded"), default="open",
                        help="Temple articulation in the reference; exported GLB is open")
    parser.add_argument("--yaw-limit", type=float, default=75,
                        help="Camera search extent in degrees; use 89 for a near side-on view")
    parser.add_argument("--optimize-registration", action="store_true",
                        help="Optimize 2-D scale/translation on this calibration reference")
    args = parser.parse_args()
    if not 0 < args.yaw_limit <= 89:
        raise ValueError('yaw-limit must be above 0 and at most 89 degrees')
    ref = reference_mask(args.reference, args.size, args.threshold)
    ys, xs = np.where(ref)
    if len(xs) == 0:
        raise ValueError("Empty reference foreground")
    bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
    meshes = geometry(args.model)
    if args.registration:
        fitted = json.loads(args.registration.read_text())
        if fitted["size"] != args.size:
            raise ValueError("Registration report uses different raster size")
        camera = (fitted["yaw_deg"], fitted["pitch_deg"],
                  fitted["roll_deg"], fitted["camera_distance_m"])
        transform = tuple(fitted["pixel_transform"])
    else:
        def objective(p: np.ndarray) -> float:
            camera_trial = tuple(map(float, p))
            t = registration(meshes, bbox, camera_trial)
            return 1 - iou(render(meshes, camera_trial, args.size, t), ref)
        result = differential_evolution(objective,
                                        [(-args.yaw_limit, args.yaw_limit), (-25, 25), (-12, 12), (0.18, 1.5)],
                                        seed=13069, popsize=7, maxiter=18, polish=False)
        camera = tuple(map(float, result.x))
        transform = registration(meshes, bbox, camera)
        if args.optimize_registration:
            transform = optimize_transform(meshes, camera, ref, transform)
    model_mask = render(meshes, camera, args.size, transform)
    score = iou(ref, model_mask)
    overlay = np.full((args.size, args.size, 3), 255, np.uint8)
    overlay[ref.astype(bool)] = (0, 120, 255)  # reference: orange
    overlay[model_mask.astype(bool)] = (255, 60, 30)  # model: blue
    overlay[(ref & model_mask).astype(bool)] = (45, 180, 45)  # match: green
    args.output.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output / "reference-mask.png"), ref * 255)
    cv2.imwrite(str(args.output / "model-mask.png"), model_mask * 255)
    cv2.imwrite(str(args.output / "overlay.png"), overlay)
    report = {
        "model": str(args.model), "reference": str(args.reference),
        "foreground_iou": round(score, 6),
        "reference_pixels": int(ref.sum()), "model_pixels": int(model_mask.sum()),
        "intersection_pixels": int(np.count_nonzero(ref & model_mask)),
        "yaw_deg": camera[0],
        "pitch_deg": camera[1],
        "roll_deg": camera[2],
        "camera_distance_m": camera[3],
        "pixel_transform": list(transform),
        "registration_source": str(args.registration) if args.registration else "camera optimized on this reference",
        "threshold_gray_less_than": args.threshold,
        "camera_search_yaw_limit": args.yaw_limit if not args.registration else None,
        "size": args.size,
        "model_temple_state": "open",
        "reference_temple_state": args.reference_state,
        "eligible_for_geometry_acceptance": args.reference_state == "open",
        "state_limitation": "Folded reference cannot validate open-temple geometry without matching hinge articulation." if args.reference_state == "folded" else None,
        "registration_method": "reused pinned camera and pixel transform" if args.registration else ("camera fit plus bounded 2-D scale/translation optimization" if args.optimize_registration else "camera fit plus foreground-bounding-box scale/translation"),
        "method": "Pinhole projection; camera and any 2-D registration fitted on the reference are calibration. Source aspect ratio is preserved inside the square raster. Lenses omitted from model; near-white background, clear apertures, and soft shadows excluded from reference.",
        "interpretation": "Optimistic 2-D projected silhouette diagnostic only; neither an independent pose estimate nor a percent of physical 3-D accuracy.",
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
