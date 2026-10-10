"""Fetch public Lenskart LA E13069-C6 product views and record their provenance.

The local cache is for reconstruction research; it is not a redistribution grant.
"""

from __future__ import annotations

import hashlib
import io
import json
import subprocess
import argparse
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PAGE = "https://www.lenskart.com/lenskart-air-la-e13069-c6-eyeglasses.html"
DEST = ROOT / "data/references/lenskart-la-e13069-c6"
SAVED = ROOT / "data/projects/d9cb85ea817b4a7d968e31135814cab4/photos"
IMAGE_BASE = "https://static5.lenskart.com/media/catalog/product/pro/1/thumbnail/1080x1080/9df78eab33525d08d6e5fb8d27136e95//l/i/"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch(url: str) -> bytes:
    return subprocess.check_output(["curl", "-fLsS", "--max-time", "30", url])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fetch", action="store_true", help="Download source images; default processes an existing local cache")
    args = parser.parse_args()
    DEST.mkdir(parents=True, exist_ok=True)
    saved = [(p, np.asarray(Image.open(p).convert("RGB"), dtype=np.int16)) for p in SAVED.glob("*.png")]
    views = []
    for index in (1, 2, 3, 4, 6):
        url = IMAGE_BASE + f"lenskart-air-la-e13069-c6-eyeglasses_210172_({index})_02_11_2023.jpg"
        output = DEST / f"official-{index}.jpg"
        if args.fetch:
            output.write_bytes(fetch(url))
        payload = output.read_bytes()
        with Image.open(io.BytesIO(payload)) as im:
            im.verify()
        pixels = np.asarray(Image.open(io.BytesIO(payload)).convert("RGB"), dtype=np.int16)
        differences = sorted((float(np.abs(pixels - other).mean()), path) for path, other in saved)
        closest_error, closest_path = differences[0]
        views.append({
            "index": index,
            "temple_state": "folded" if index == 6 else "open",
            "eligible_for_open_model_validation": index != 6,
            "path": str(output.relative_to(ROOT)),
            "url": url,
            "sha256": digest(payload),
            "pixel_match_saved_photo": str(closest_path.relative_to(ROOT)) if closest_error < 1.0 else None,
            "closest_saved_mean_absolute_rgb_error": round(closest_error, 4),
        })
    manifest = {
        "product": "Lenskart Air LA E13069-C6",
        "source_page": PAGE,
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "source_evidence": "Saved close-up temple inscription reads LA E13069 C6 and 53-16-142; official page identifies gunmetal full-rim square model.",
        "published_product_details": {
            "product_id": "210172",
            "frame_width_mm": 134,
            "frame_dimensions_text": "53-16-145",
            "weight_g": 17,
            "frame_material": "TR90",
            "temple_material": "Stainless Steel",
            "source": PAGE,
        },
        "inscription_dimensions_text": "53-16-142",
        "limitations": [
            "Product images are uncalibrated studio photographs, not physical dimension ground truth.",
            "A matching official view is not an independent held-out view, even when compressed differently.",
            "Official view 6 has folded temples across the lens openings and cannot validate an open-temple model without articulation matching.",
            "Official page states 53-16-145 while visible temple inscription states 53-16-142; temple length is unresolved.",
            "Public images remain copyrighted by their source; local research cache only.",
        ],
        "views": views,
    }
    (DEST / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
