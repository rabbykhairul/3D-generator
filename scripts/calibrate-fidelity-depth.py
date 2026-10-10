"""Offline shape sensitivity on existing product views, without project edits.

The score is a projected 2-D silhouette with a camera previously fitted to
view 2. View 2 is also a saved project photo, so this is calibration evidence
and never independent validation or 3-D accuracy.
"""

import copy
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.reconstruction import reconstruct

sys.path.insert(0, str(ROOT / "scripts"))
import score_heldout_silhouette as silhouette  # noqa: E402


def main():
    fidelity = ROOT / "artifacts" / "fidelity"
    snapshot = json.loads((fidelity / "candidate" / "snapshot.json").read_text())
    reference = silhouette.reference_mask(
        ROOT / "data" / "references" / "lenskart-la-e13069-c6" / "official-2.jpg", 384, 215)
    ys, xs = reference.nonzero()
    bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
    report = json.loads((fidelity / "heldout-baseline" / "report.json").read_text())
    camera = (report["yaw_deg"], report["pitch_deg"], report["roll_deg"],
              report["camera_distance_m"])
    trials = []
    # One dimension at a time around the existing draft, then a limited
    # physically plausible combination; published arm length is 142 mm.
    trials.append((4, 8, 140, 140))
    trials.extend((sag, 8, 142, 140) for sag in (0, 2, 4, 6, 8, 10))
    trials.extend((4, spread, 142, 140) for spread in (-5, 0, 4, 8, 12, 18, 24))
    trials.extend((4, 8, 142, width) for width in (132, 136, 144))
    trials.extend((sag, spread, 142, 140)
                  for sag in (2, 6, 10) for spread in (0, 12, 24))
    results = []
    with tempfile.TemporaryDirectory(prefix="eyewear-fidelity-") as temporary:
        for index, (sag, spread, length, width) in enumerate(trials):
            trial = copy.deepcopy(snapshot)
            trial["measurements"].update({
                "front_curve": {"value": sag, "source": "inferred"},
                "temple_spread": {"value": spread, "source": "inferred"},
                "temple_length": {"value": length, "source": "inferred" if length == 142 else "default"},
                "frame_width": {"value": width, "source": "inferred"},
            })
            output = Path(temporary) / str(index)
            reconstruct(trial, output)
            meshes = silhouette.geometry(output / "model.glb")
            transform = silhouette.registration(meshes, bbox, camera)
            rendered = silhouette.render(meshes, camera, 384, transform)
            score = silhouette.iou(reference, rendered)
            record = {"front_curve_mm": sag, "temple_spread_mm": spread,
                      "temple_length_mm": length, "frame_width_mm": width,
                      "view_2_silhouette_iou": round(score, 6)}
            results.append(record)
            print(record, flush=True)
    results.sort(key=lambda result: result["view_2_silhouette_iou"], reverse=True)
    (fidelity / "calibration-sensitivity.json").write_text(json.dumps({
        "reference": "Official view 2 is also a saved project photo; calibration only",
        "camera": list(camera), "registration": "uniform extent fit per model",
        "front_depth_mm": 4, "depth_source": "unverified default",
        "trials": results,
    }, indent=2))


if __name__ == "__main__":
    main()
