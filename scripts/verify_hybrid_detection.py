"""Replay front photos through both backends; export only new hybrid candidates.

Development evidence only. Never changes projects, photos, approvals or jobs.
"""
from copy import deepcopy
import argparse
import hashlib
import json
from pathlib import Path
import sys

from PIL import Image,ImageDraw,ImageOps
from shapely.geometry import Polygon
import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from backend import store
from backend.reconstruction import extract_annotations,reconstruct
from backend.reconstruction.detection_report import detection_report
from backend.reconstruction.render_mesh import welded_geometry


def overlay(image,result):
    image=image.copy()
    draw=ImageDraw.Draw(image)
    for annotation in result['annotations']:
        points=[(p['x'],p['y']) for p in annotation['points']]
        color='#00aa55' if annotation['part']=='front' else '#ef7500'
        draw.line(points+points[:1],fill=color,width=max(2,image.width//350))
    return image


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,default=ROOT/'artifacts/segmenter-evaluation/medium.json')
    parser.add_argument('--output',type=Path,default=ROOT/'artifacts/hybrid-validation')
    args=parser.parse_args()
    manifest=json.loads(args.manifest.read_text())
    args.output.mkdir(parents=True,exist_ok=True)
    rows=[]
    for index,case in enumerate(manifest['cases']):
        if case['view']!='front':continue
        path=Path(case['path'])
        if hashlib.sha256(path.read_bytes()).hexdigest()!=case['sha256']:
            raise ValueError('Source photo changed since evaluation inventory')
        with Image.open(path) as source:image=source.convert('RGB')
        photo=dict(id=case['photo_id'],path=str(path),view='front',role='fitting',width=image.width,height=image.height)
        baseline=extract_annotations(photo,backend='heuristics')
        candidate=extract_annotations(photo,backend='hybrid')
        row=dict(name=case['name'],photo_id=photo['id'],
                 baseline=detection_report(photo,baseline),hybrid=detection_report(photo,candidate),
                 diagnostics=candidate['hybrid_detection'])
        (args.output/f'{index:05d}-annotations.json').write_text(json.dumps(candidate,indent=2)+'\n')
        sheet=Image.new('RGB',(1200,450),'white');draw=ImageDraw.Draw(sheet)
        for column,(label,result) in enumerate([('Standard',baseline),('AI assisted',candidate)]):
            panel=ImageOps.contain(overlay(image,result),(600,410))
            sheet.paste(panel,(column*600+(600-panel.width)//2,30+(410-panel.height)//2))
            draw.text((column*600+8,8),label,fill='black')
        row['comparison']=f'{index:05d}-comparison.jpg'
        sheet.save(args.output/row['comparison'],quality=92)
        if candidate['hybrid_detection'].get('proposals_used'):
            project=deepcopy(store.get_project(case['sku']))
            snapshot=dict(project,photos=[photo],annotations=[dict(a,approved=True,source='test-reviewed:'+a['source']) for a in candidate['annotations']])
            # The script exports a labelled numerical draft, not a measurement.
            parts={a['part']:Polygon([(p['x'],p['y']) for p in a['points']]) for a in candidate['annotations']}
            f,l,r=(parts[p].bounds for p in ('front','lens-left','lens-right'))
            scale=140/(f[2]-f[0])
            values=dict(frame_width=140,temple_length=140,lens_width=((l[2]-l[0])+(r[2]-r[0]))/2*scale,
                        lens_height=((l[3]-l[1])+(r[3]-r[1]))/2*scale,bridge_width=(r[0]-l[2])*scale)
            snapshot['measurements']={k:dict(value=v,source='inferred') for k,v in values.items()}
            directory=args.output/f'{index:05d}-draft'
            metadata=reconstruct(snapshot,directory)
            scene=trimesh.load(directory/'model.glb',force='scene')
            row['export']=dict(path=str(directory/'model.glb'),parts=sorted(scene.geometry),
                               watertight_after_welding_render_seams={name:welded_geometry(mesh).is_watertight for name,mesh in scene.geometry.items()},
                               checks=metadata['checks'],physical_scale_verified=False)
        rows.append(row)
        print(json.dumps(dict(name=case['name'],status=row['diagnostics']['status'])),flush=True)
    summary=dict(cases=rows,independent_labels=False,unknown_product_accuracy_verified=False,
                 inference_errors=sum(r['diagnostics']['status']=='unavailable' for r in rows),
                 newly_recovered=sum(r['diagnostics'].get('proposals_used',False) for r in rows),
                 retained=sum(r['diagnostics']['status']=='retained' for r in rows),
                 note='Operational geometry/export checks on development products, not accuracy or fit certification.')
    (args.output/'report.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='cases'},indent=2))


if __name__=='__main__':main()
