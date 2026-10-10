"""Isolated source-detail/GLB diagnostic; never changes merchant approvals."""
from copy import deepcopy
from pathlib import Path
import json
import cv2
import numpy as np
import trimesh
from shapely.geometry import Polygon
from backend.store import get_project
from backend.reconstruction.engine import extract_annotations,reconstruct
from backend.reconstruction.lens_details import visible_lens_details


def main():
    draft=deepcopy(get_project('8758d7b8ecf343c4b5f321221757954d'))
    if draft is None:raise RuntimeError('Local silver reference unavailable')
    out=Path('artifacts/silver-appearance');out.mkdir(parents=True,exist_ok=True)
    draft['photos']=[p for p in draft['photos'] if p['view'] in ('front','right')]
    draft['annotations']=[a for p in draft['photos'] for a in extract_annotations(p)['annotations']]
    for a in draft['annotations']:a['approved']=True
    for key,value in {'frame_width':140,'lens_width':54,'lens_height':46,'bridge_width':18,'temple_length':140}.items():
        draft['measurements'][key]={'value':value,'source':'diagnostic-assumption'}
    photo=next(p for p in draft['photos'] if p['view']=='front')
    shapes={a['part']:Polygon([(p['x'],p['y'])for p in a['points']])for a in draft['annotations'] if a['photo_id']==photo['id']}
    details=visible_lens_details(photo,{s:np.array(shapes[f'lens-{s}'].exterior.coords)for s in ('left','right')},
                                {s:shapes[f'nose-pad-{s}']for s in ('left','right')})
    image=cv2.imread(photo['path'])
    for d in details:
        cv2.polylines(image,[np.rint(d['polygon'].exterior.coords).astype(np.int32)],True,(20,130,230),2)
    cv2.imwrite(str(out/'retained-details.png'),image)
    metadata=reconstruct(draft,out/'draft')
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2))
    scene=trimesh.load(out/'draft/model.glb',force='scene',process=False)
    names=[n for n in scene.geometry if n.startswith('lens_detail_')]
    assert len(names)>=3
    assert metadata['checks']['front_parts_watertight']
    from scripts.render_fidelity_cpu import render
    target=scene.bounds.mean(axis=0)
    render(out/'draft/model.glb',target+[0,0,.34],target,'photo',1080,include_lenses=True).save(out/'cpu-front.png')
    print({'detail_parts':names,'checks':metadata['checks']})


if __name__=='__main__':main()
