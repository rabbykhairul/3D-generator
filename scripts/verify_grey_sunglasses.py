"""Reproduce the supplied grey-rim regression without touching saved projects."""
from pathlib import Path
import sys
import json
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from backend.reconstruction import extract_annotations  # noqa: E402


def main():
    source=ROOT/'tests/fixtures/grey_double_bridge/front.png'
    output=ROOT/'artifacts/grey-sunglasses'
    output.mkdir(parents=True,exist_ok=True)
    result=extract_annotations({'id':'grey-landscape','view':'front','path':str(source)})
    image=cv2.imread(str(source))
    colors={'front':(90,160,40),'bridge-opening':(200,70,180),
            'nose-pad-left':(180,130,20),'nose-pad-right':(180,130,20)}
    for ann in result['annotations']:
        points=np.rint([(p['x'],p['y']) for p in ann['points']]).astype(np.int32)
        cv2.polylines(image,[points],True,colors.get(ann['part'],(20,130,230)),2,cv2.LINE_AA)
    cv2.imwrite(str(output/'front-outline.png'),image)
    (output/'detection.json').write_text(json.dumps(result,indent=2)+'\n')
    print([a['part'] for a in result['annotations']])


if __name__=='__main__':
    main()
