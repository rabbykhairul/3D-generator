"""Build read-only comparison artifacts for the reviewed eyewear project.

The 2 mm variant is a visual calibration trial, not a measured dimension.
Neither variant modifies the saved project or its approved annotations.
"""

import copy
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.reconstruction import reconstruct


PROJECT_ID = "d9cb85ea817b4a7d968e31135814cab4"
DESTINATION = ROOT / "artifacts" / "fidelity"


def main():
    uri = (ROOT / "data" / "studio.sqlite3").as_uri() + "?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        row = connection.execute("SELECT payload FROM projects WHERE id=?", (PROJECT_ID,)).fetchone()
    if row is None:
        raise RuntimeError(f"Project {PROJECT_ID} was not found")
    project = json.loads(row[0])
    if project["revision"] != 9:
        raise RuntimeError("Expected reviewed revision 9; inspect the saved project before building")
    if "frame_thickness" in project["measurements"]:
        raise RuntimeError("Saved project now has a frame depth measurement; the 2 mm trial is obsolete")

    variants = [("candidate", None), ("thin-trial", 2.0)]
    variants.extend((f"depth-{str(value).replace('.', 'p')}", value)
                    for value in (1.5, 2.0, 2.5, 3.0, 4.0))
    variants.append(("published-size", None))
    for label, thickness in variants:
        snapshot = copy.deepcopy(project)
        if label == "published-size":
            snapshot["measurements"].update({
                "lens_width": {"value": 53.0, "source": "published"},
                "bridge_width": {"value": 16.0, "source": "published"},
                "temple_length": {"value": 142.0, "source": "published"},
            })
        if thickness is not None:
            snapshot["measurements"]["frame_thickness"] = {
                "value": thickness, "source": "inferred"}
        output = DESTINATION / label
        metadata = reconstruct(snapshot, output)
        (output / "snapshot.json").write_text(json.dumps(snapshot, indent=2))
        metadata["comparison_label"] = (
            "Published 53-16-142 size; overall width and 4 mm front depth remain unverified"
            if label == "published-size" else
            f"Unverified {thickness:g} mm visual trial; not applied to saved project"
            if thickness is not None else
            "Saved project contours with unverified default 4 mm depth")
        (output / "metadata.json").write_text(json.dumps(metadata, indent=2))
        print(f"{label}: {output / 'model.glb'}; bevel={metadata['dimensions_mm']['front_bevel']:.3f} mm")


if __name__ == "__main__":
    main()
