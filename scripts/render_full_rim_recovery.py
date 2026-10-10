"""Render a local diagnostic overlay for a recovered double-bridge front."""

from pathlib import Path
import argparse
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.reconstruction.full_rim_recovery import recover_double_bridge_front


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("photo", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    image = cv2.imread(str(args.photo))
    if image is None:
        raise SystemExit("Photo cannot be decoded")
    result = recover_double_bridge_front(image, {"view":"front"}, 1, 1)
    if result is None:
        raise SystemExit("No safe double-bridge recovery")
    colors = {"front":(255,30,30),"lens-left":(30,180,20),
              "lens-right":(30,180,20),"bridge-opening":(0,180,255)}
    overlay = image.copy()
    for annotation in result["annotations"]:
        points = np.array([[round(p['x']),round(p['y'])] for p in annotation['points']],np.int32)
        cv2.polylines(overlay,[points],True,colors[annotation['part']],2,cv2.LINE_AA)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    if not cv2.imwrite(str(args.output),overlay):
        raise SystemExit("Could not write overlay")


if __name__ == "__main__":
    main()
