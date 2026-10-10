"""Pinned-camera silhouette evidence for multiple open-temple studio views.

Each camera is fitted to the baseline model and its own photo, then the exact
camera and pixel transform are reused for every candidate. A withheld *model*
comparison is possible; the photographed poses themselves are calibrated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import differential_evolution

import score_heldout_silhouette as silhouette

ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def overlay(reference: np.ndarray, model: np.ndarray) -> np.ndarray:
    result = np.full((*reference.shape, 3), 255, np.uint8)
    result[reference.astype(bool)] = (0, 120, 255)
    result[model.astype(bool)] = (255, 60, 30)
    result[(reference & model).astype(bool)] = (45, 180, 45)
    return result


def evaluate(manifest: dict, output: Path, manifest_path: Path,
             pinned: dict | None = None, pinned_path: Path | None = None) -> dict:
    size = int(manifest["raster_size"])
    threshold = int(manifest["threshold_gray_less_than"])
    baseline = (ROOT / manifest["baseline"]).resolve()
    if pinned is not None and (pinned["baseline_sha256"] != digest(baseline)
                               or pinned["raster_size"] != size
                               or pinned["threshold_gray_less_than"] != threshold):
        raise ValueError("Pinned cameras were fitted to a different baseline or raster policy")
    models = {"baseline": silhouette.geometry(baseline)}
    model_paths = {"baseline": baseline}
    for item in manifest["candidates"]:
        label = item["name"]
        if label in models or not label.replace("-", "").replace("_", "").isalnum():
            raise ValueError(f"Invalid or duplicate candidate name: {label}")
        path = (ROOT / item["path"]).resolve()
        models[label] = silhouette.geometry(path)
        model_paths[label] = path

    reports = []
    for view in manifest["views"]:
        if view["temple_state"] != "open":
            raise ValueError(f"View {view['id']} is not open-temple articulation")
        if view["role"] not in ("calibration", "withheld"):
            raise ValueError(f"View {view['id']} needs an explicit role")
        source = (ROOT / view["path"]).resolve()
        if "sha256" in view and digest(source) != view["sha256"]:
            raise ValueError(f"Source hash mismatch: {source}")
        reference = silhouette.reference_mask(source, size, threshold)
        ys, xs = np.where(reference)
        if not len(xs):
            raise ValueError(f"Empty foreground: {source}")
        box = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
        yaw_limit = float(view.get("yaw_limit", 89))
        if not 0 < yaw_limit <= 89:
            raise ValueError("yaw_limit must be in (0, 89]")

        def objective(p: np.ndarray) -> float:
            camera = tuple(map(float, p))
            transform = silhouette.registration(models["baseline"], box, camera)
            return 1 - silhouette.iou(reference, silhouette.render(models["baseline"], camera, size, transform))

        if pinned is None:
            fit = differential_evolution(
                objective, [(-yaw_limit, yaw_limit), (-25, 25), (-12, 12), (.18, 1.5)],
                seed=13069 + int(view["index"]), popsize=7, maxiter=18, polish=False,
            )
            camera = tuple(map(float, fit.x))
            box_transform = silhouette.registration(models["baseline"], box, camera)
            transform = silhouette.optimize_transform(models["baseline"], camera, reference, box_transform)
        else:
            entry = pinned["cameras"][view["id"]]
            if entry["source_sha256"] != digest(source):
                raise ValueError(f"Pinned camera source mismatch for {view['id']}")
            camera = tuple(map(float, entry["yaw_pitch_roll_distance"]))
            transform = tuple(map(float, entry["scale_tx_ty"]))
        view_dir = output / view["id"]
        view_dir.mkdir()
        cv2.imwrite(str(view_dir / "reference-mask.png"), reference * 255)
        scores = {}
        for name, meshes in models.items():
            predicted = silhouette.render(meshes, camera, size, transform)
            score = silhouette.iou(reference, predicted)
            scores[name] = {
                "foreground_iou": round(score, 6),
                "intersection_pixels": int(np.count_nonzero(reference & predicted)),
                "model_pixels": int(predicted.sum()),
            }
            cv2.imwrite(str(view_dir / f"{name}-mask.png"), predicted * 255)
            cv2.imwrite(str(view_dir / f"{name}-overlay.png"), overlay(reference, predicted))
        reports.append({
            "id": view["id"], "role": view["role"], "source": str(source.relative_to(ROOT)),
            "source_sha256": digest(source), "temple_state": "open",
            "complete_product_frame": bool(view.get("complete_product_frame", True)),
            "eligible_for_full_product_comparison": bool(view.get("complete_product_frame", True)),
            "independent_of_saved_project": bool(view["independent_of_saved_project"]),
            "previously_used_for_geometry_tuning": bool(view["previously_used_for_geometry_tuning"]),
            "camera_fitted_to_this_reference": True,
            "camera_reused_from_pinned_manifest": pinned is not None,
            "camera_yaw_pitch_roll_distance": list(camera),
            "pixel_transform_scale_tx_ty": list(transform),
            "baseline_bbox_registration_iou": round(1 - objective(np.array(camera)), 6),
            "reference_pixels": int(reference.sum()), "scores": scores,
        })
    return {
        "schema_version": 1, "manifest_sha256": digest(manifest_path),
        "raster_size": size, "aspect_ratio_policy": "letterbox with white, no stretching",
        "threshold_gray_less_than": threshold,
        "model_sha256": {name: digest(path) for name, path in model_paths.items()},
        "pinned_camera_manifest_sha256": digest(pinned_path) if pinned_path else None,
        "camera_policy": "Each reference fitted baseline yaw/pitch/roll/distance, then bounded scale and translation. The resulting view camera and transform are identical for baseline and candidates and reused from a pinned manifest when supplied.",
        "interpretation": "2-D foreground IoU under view-specific fitted cameras; not physical 3-D accuracy or independent pose validation. Withheld labels reserve model selection only. Previously viewed/used photos are not independent product validation. Cropped close-ups are scored only to expose the mismatch and are ineligible for full-product comparison.",
        "views": reports,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "scripts" / "fidelity_view_manifest_v2.json")
    parser.add_argument("--camera-manifest", type=Path, default=ROOT / "scripts" / "fidelity_pinned_cameras.json")
    parser.add_argument("--refit-cameras", action="store_true", help="Fit cameras to baseline anew; calibration only")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate", action="append", default=[], metavar="NAME=GLB",
                        help="Replace manifest candidates for a new engine export; repeat to compare several")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Evidence directory already exists: {args.output}")
    manifest = json.loads(args.manifest.read_text())
    if args.candidate:
        candidates=[]
        for item in args.candidate:
            name, separator, path = item.partition('=')
            if not separator or not name or not path:
                parser.error('--candidate must be NAME=GLB')
            candidates.append({'name': name, 'path': path})
        manifest['candidates']=candidates
    args.output.mkdir(parents=True)
    (args.output / 'evaluated-manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    pinned = None if args.refit_cameras else json.loads(args.camera_manifest.read_text())
    report = evaluate(manifest, args.output, args.manifest, pinned,
                      None if args.refit_cameras else args.camera_manifest)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({view["id"]: {name: score["foreground_iou"] for name, score in view["scores"].items()}
                      for view in report["views"]}, indent=2))


if __name__ == "__main__":
    main()
