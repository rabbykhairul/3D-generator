"""Read-only cross-product audit; isolated overlays and JSON, no approvals."""
from pathlib import Path
import json
import cv2
import numpy as np
from backend.store import get_project
from backend.reconstruction.engine import extract_annotations

REFERENCES={
 'silver-single':'8758d7b8ecf343c4b5f321221757954d',
 'gold-double':'d6fec31607d84030849761a204c137db',
 'clear':'b0bc9b703db9463abab0a9a4d30c0b4d',
 'semi-rimless':'3ca4f944b2214b72988db3bde88e0602',
 'lavender':'8dd590d016a44d268efcb319d3f86437',
 'olive':'7def97639b034aa4be2d224429cc9f45',
 'black-gold':'ce81f4ffd03e4b8cac759b32ec5ee650',
}

def main():
    out=Path('artifacts/frame-families');out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for label,identifier in REFERENCES.items():
        project=get_project(identifier)
        if project is None:
            rows.append({'family':label,'status':'reference-unavailable'});continue
        for view in ('front','right'):
            photo=next((p for p in project['photos'] if p['view']==view),None)
            if photo is None:continue
            result=extract_annotations(photo)
            parts={a['part'] for a in result['annotations']}
            complete=({'front','lens-left','lens-right'}<=parts if view=='front' else 'temple-right' in parts)
            row={'family':label,'view':view,'complete':complete,'parts':sorted(parts),'method':result.get('segmentation',{}).get('method')}
            rows.append(row);print(row,flush=True)
            im=cv2.imread(photo['path'])
            for a in result['annotations']:
                color=(220,140,20) if a['part'].startswith('nose-pad') else (60,170,70) if a['part']=='front' else (20,130,230)
                cv2.polylines(im,[np.rint([[p['x'],p['y']]for p in a['points']]).astype('int32')],True,color,1,cv2.LINE_AA)
            cv2.imwrite(str(out/f'{label}-{view}.png'),im)
    (out/'results.json').write_text(json.dumps(rows,indent=2))

if __name__=='__main__':main()
