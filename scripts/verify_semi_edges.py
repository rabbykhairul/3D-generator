"""Isolated original-reference overlays for reported semi/side cases."""
from pathlib import Path
import cv2
import numpy as np
import json
from backend.store import get_project
from backend.reconstruction.engine import extract_annotations

def main():
    folder=Path('artifacts/semi-edge-cases');folder.mkdir(parents=True,exist_ok=True)
    for name,pid,view in [('semi-front','3ca4f944b2214b72988db3bde88e0602','front'),
                          ('semi-side','3ca4f944b2214b72988db3bde88e0602','right'),
                          ('lavender-side','8dd590d016a44d268efcb319d3f86437','right')]:
        project=get_project(pid);photo=next(p for p in project['photos'] if p['view']==view)
        result=extract_annotations(photo)
        image=cv2.imread(photo['path'])
        for a in result['annotations']:
            points=np.rint([[p['x'],p['y']] for p in a['points']]).astype(np.int32)
            cv2.polylines(image,[points],True,(60,160,80) if a['part']=='front' else (30,120,235),1,cv2.LINE_AA)
        cv2.imwrite(str(folder/f'{name}.png'),image)
        if view=='right':
            cv2.imwrite(str(folder/f'{name}-crossing.png'),cv2.resize(image[500:630,290:520],(920,520)))
        (folder/f'{name}.json').write_text(json.dumps(result,indent=2))

if __name__=='__main__':main()
