"""Read-only source analysis; writes isolated evidence, never merchant approvals."""
import json
from pathlib import Path

import cv2
import numpy as np

from backend.store import get_project
from backend.reconstruction.engine import extract_annotations


def main():
    project = get_project('b0bc9b703db9463abab0a9a4d30c0b4d')
    photo = next(p for p in project['photos'] if p['view'] == 'front')
    result = extract_annotations(photo)
    folder = Path('artifacts/clear-boundary')
    folder.mkdir(parents=True, exist_ok=True)
    (folder/'after.json').write_text(json.dumps(result, indent=2))
    image = cv2.imread(photo['path'])
    for annotation in result['annotations']:
        points = np.rint([[p['x'], p['y']] for p in annotation['points']]).astype('int32')
        color = (50,160,70) if annotation['part'] == 'front' else (30,120,230)
        cv2.polylines(image, [points], True, color, 2, cv2.LINE_AA)
    cv2.imwrite(str(folder/'after.png'), image)
    side = next(p for p in project['photos'] if p['view'] == 'right')
    side_result = extract_annotations(side)
    (folder/'arm-after.json').write_text(json.dumps(side_result, indent=2))
    side_image = cv2.imread(side['path'])
    for annotation in side_result['annotations']:
        points = np.rint([[p['x'],p['y']] for p in annotation['points']]).astype('int32')
        cv2.polylines(side_image,[points],True,(20,140,240),2,cv2.LINE_AA)
    cv2.imwrite(str(folder/'arm-after.png'),side_image)
    for name in ('before', 'after'):
        source = cv2.imread(str(folder/f'{name}.png'))
        if source is not None:
            cv2.imwrite(str(folder/f'{name}-detail.png'), source[375:650,350:730])


if __name__ == '__main__':
    main()
