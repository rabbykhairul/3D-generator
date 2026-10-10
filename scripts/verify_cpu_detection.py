"""Generate read-only fixture regression overlays, not an unseen-product audit."""
from pathlib import Path
import json
import sys
import time

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from backend.reconstruction import extract_annotations  # noqa: E402


def main():
    output=ROOT/'artifacts/cpu-detection'
    output.mkdir(parents=True,exist_ok=True)
    rows=[]
    cases=[('front',320,(255,255,255)),('front',1080,(210,210,210)),
           ('front',1080,(225,235,245)),('right',1080,(225,235,245))]
    for view,size,background in cases:
        source=cv2.imread(str(ROOT/f'tests/fixtures/rimless/{view}.png'))
        image=cv2.resize(source,(size,size),interpolation=cv2.INTER_AREA)
        image=np.clip(np.rint(image.astype(float)*np.array(background)/255),0,255).astype(np.uint8)
        name=f'{view}-{size}-'+ '-'.join(map(str,background))
        path=output/f'{name}-input.png'
        cv2.imwrite(str(path),image)
        start=time.perf_counter()
        result=extract_annotations({'id':name,'view':view,'path':str(path)})
        seconds=time.perf_counter()-start
        overlay=image.copy()
        for ann in result['annotations']:
            color=(200,140,30) if ann['part'].startswith('nose-pad') else (30,160,210)
            points=np.rint([(p['x'],p['y']) for p in ann['points']]).astype(np.int32)
            cv2.polylines(overlay,[points],True,color,max(1,size//540),cv2.LINE_AA)
        cv2.imwrite(str(output/f'{name}-outline.png'),overlay)
        rows.append({'case':name,'seconds':round(seconds,3),'parts':[a['part'] for a in result['annotations']],
                     'warnings':result.get('warnings',[]),'background_calibration':result['background_calibration']})
    (output/'report.json').write_text(json.dumps({'scope':'Existing-fixture derivatives, not independent labels or unseen products',
                                                'cases':rows},indent=2)+'\n')
    print(json.dumps(rows,indent=2))


if __name__=='__main__':
    main()
