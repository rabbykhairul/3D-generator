"""Diagnostic only: test asymmetric open-arm poses against saved product views.

The pose offset bends neither mesh nor temple shape; it changes each arm's
hinge opening angle through a linear lateral tip displacement. The two views
are existing project photos, so scores are calibration values only.
"""

import json
import sys
from pathlib import Path

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import score_heldout_silhouette as silhouette  # noqa: E402


def main():
    path = ROOT / "artifacts" / "fidelity" / "candidate" / "model.glb"
    scene = trimesh.load(path, force="scene")
    original = {}
    for node in scene.graph.nodes_geometry:
        transform, name = scene.graph[node]
        if name.startswith("lens_"):
            continue
        mesh = scene.geometry[name]
        original[name] = trimesh.transform_points(mesh.vertices, transform)[mesh.faces]
    references = {}
    for index, camera in ((2, (-38.2754097284, 14.2610519496, -2.7270278398, .533406795)),
                          (3, (86.03147656, .73816567, -4.35992719, .94494115))):
        image = ROOT / "data" / "references" / "lenskart-la-e13069-c6" / f"official-{index}.jpg"
        mask = silhouette.reference_mask(image, 384, 215)
        ys, xs = mask.nonzero()
        references[index] = (mask, (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())), camera)

    def score(left_delta, right_delta):
        parts = []
        for name, triangles in original.items():
            modified = triangles.copy()
            if name in {"temple_left", "temple_right"}:
                direction = 1 if name == "temple_left" else -1
                delta = left_delta if direction == 1 else right_delta
                z_front = float(modified[..., 2].max())
                q = np.clip((z_front - modified[..., 2]) / .14, 0, 1)
                modified[..., 0] += direction * delta * q / 1000
            parts.append(modified)
        values = {}
        for index, (mask, bbox, camera) in references.items():
            transform = silhouette.registration(parts, bbox, camera)
            rendered = silhouette.render(parts, camera, 384, transform)
            values[f"view_{index}_iou"] = round(float(silhouette.iou(mask, rendered)), 6)
        return values

    results = []
    for side in ("left", "right"):
        for delta in (-20, -10, 0, 10, 20, 30, 40, 50):
            offsets = (delta, 0) if side == "left" else (0, delta)
            result = {"left_tip_lateral_delta_mm": offsets[0],
                      "right_tip_lateral_delta_mm": offsets[1], **score(*offsets)}
            results.append(result)
            print(result, flush=True)
    path = ROOT / "artifacts" / "fidelity" / "arm-pose-sensitivity.json"
    path.write_text(json.dumps({
        "source": "Official views 2 and 3 are also saved project photos; calibration only",
        "method": "Linear lateral displacement of one temple tip with camera and uniform extent registration fixed by each view",
        "limitation": "An articulation diagnostic, not a manufactured arm-shape fit or physical measurement",
        "trials": results,
    }, indent=2))


if __name__ == "__main__":
    main()
