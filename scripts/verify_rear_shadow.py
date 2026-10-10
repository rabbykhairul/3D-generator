"""Read-only gray reference, same-crop before/after masks."""
from pathlib import Path
import json
import cv2
import numpy as np
from backend.store import get_project
from backend.reconstruction.engine import extract_annotations
from backend.reconstruction import opposite_arm

def main():
    project=get_project('934325061be045f0abda9fb2d233aa46')
    photo=next(p for p in project['photos'] if p['view']=='right')
    original=opposite_arm.clean_appearance_mask
    try:
        opposite_arm.clean_appearance_mask=lambda mask,contrast: mask
        before=extract_annotations(photo)
    finally:
        opposite_arm.clean_appearance_mask=original
    after=extract_annotations(photo)
    output=Path('artifacts/rear-shadow');output.mkdir(parents=True,exist_ok=True)
    for name,result in [('before',before),('after',after)]:
        pixels=cv2.imread(photo['path'])
        for e in result['appearance_evidence']:
            p=np.rint([[p['x'],p['y']] for p in e['points']]).astype(np.int32)
            cv2.polylines(pixels,[p],True,(185,145,35),2,cv2.LINE_AA)
        cv2.imwrite(str(output/f'{name}.png'),pixels)
        cv2.imwrite(str(output/f'{name}-tip.png'),cv2.resize(pixels[615:700,175:315],(700,425)))
        (output/f'{name}.json').write_text(json.dumps(result,indent=2))
    print([(n,[(min(p['y'] for p in e['points']),max(p['y'] for p in e['points'])) for e in r['appearance_evidence']]) for n,r in [('before',before),('after',after)]])

if __name__=='__main__':main()
