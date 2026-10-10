"""Isolated user-supplied rimless fixture audit, without database writes."""
from pathlib import Path
import json
import cv2
import numpy as np
from backend.contracts import Material
from backend.reconstruction.engine import extract_annotations,reconstruct


def main():
    out=Path('artifacts/rimless');out.mkdir(parents=True,exist_ok=True)
    photos=[{'id':view,'view':view,'role':'fitting','path':str(Path('tests/fixtures/rimless')/f'{view}.png'),
             'width':1080,'height':1080}for view in ('front','right')]
    annotations=[]
    for photo in photos:
        result=extract_annotations(photo)
        image=cv2.imread(photo['path'])
        for a in result['annotations']:
            color=(220,130,20) if a['part'].startswith('nose-pad') else (30,160,210) if a['part']=='bridge-hardware' else (20,130,230)
            cv2.polylines(image,[np.rint([[p['x'],p['y']]for p in a['points']]).astype(np.int32)],True,color,2,cv2.LINE_AA)
        cv2.imwrite(str(out/f"{photo['view']}-outline.png"),image)
        annotations.extend(result['annotations'])
    for a in annotations:a['approved']=True
    project={'id':'rimless-diagnostic','revision':1,'category':'rimless','photos':photos,'annotations':annotations,
             'material':Material().model_dump(),'measurements':{k:{'value':v,'source':'inferred'}for k,v in
             {'frame_width':140,'lens_width':54,'lens_height':34,'bridge_width':18,'temple_length':140}.items()}}
    metadata=reconstruct(project,out/'draft')
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2))
    print(metadata['checks'])


if __name__=='__main__':main()
