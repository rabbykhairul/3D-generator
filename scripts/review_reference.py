"""Prepare visible, explicitly corrected traces for selected public references.

Run without --approve to produce an overlay for visual review. --approve imports
the reviewed traces and generates a model. This is a guided correction experiment,
not an automatic-reconstruction success or a withheld-view accuracy measurement.
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import httpx
import numpy as np
from PIL import Image, ImageDraw
from backend.reconstruction import extract_annotations

ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--approve',action='store_true');args=parser.parse_args()
    manifest=json.loads((ROOT/'data/references/manifest.json').read_text())
    ref=next(x for x in manifest if x['id']=='zenni-137912')
    photo=next(x for x in ref['photos'] if x['view']=='front')
    path=ROOT/photo['path'];image=Image.open(path).convert('RGB')
    suggested=extract_annotations({'path':str(path),'id':'reference','view':'front','width':image.width,'height':image.height})
    annotations=suggested['annotations']
    if len(annotations)<3:raise SystemExit('Automatic extraction did not locate both lens windows; manual tracing needed.')
    # The inspected round reference has folded arms crossing otherwise convex lens
    # apertures. Hull repair removes those inward intrusions; review overlay first.
    for a in annotations:
        if a['part'].startswith('lens-'):
            points=np.asarray([[p['x'],p['y']] for p in a['points']],dtype=np.float32)
            hull=cv2.convexHull(points).reshape(-1,2)
            a['points']=[{'x':float(x),'y':float(y)} for x,y in hull]
        a['source']='research-guided-correction'
        a['approved']=True
        if a['part']=='front':
            a['anchors']={'bridge':{'x':594,'y':228},'hinge-left':{'x':28,'y':227},'hinge-right':{'x':1172,'y':227}}
    draw=ImageDraw.Draw(image)
    for a,color in zip(annotations,['#00bb66','#004cff','#ec6600']):
        points=[(p['x'],p['y']) for p in a['points']]
        draw.line(points+[points[0]],fill=color,width=3)
    out=ROOT/'artifacts';out.mkdir(exist_ok=True)
    image.save(out/'public-metal-correction-overlay.png')
    (out/'public-metal-proposed-annotations.json').write_text(json.dumps(annotations,indent=2))
    print('Review artifacts/public-metal-correction-overlay.png before using --approve.')
    if not args.approve:return
    with httpx.Client(base_url='http://127.0.0.1:8018/api',timeout=60) as client:
        p=client.post('/samples/zenni-137912/import').json()
        front=next(p for p in p['photos'] if p['view']=='front')
        for a in annotations:a['photo_id']=front['id']
        measurement=p['measurements']
        measurement.update(frame_thickness={'value':1.2,'source':'inferred'},temple_thickness={'value':1.6,'source':'inferred'},front_curve={'value':4,'source':'inferred'},temple_spread={'value':6,'source':'inferred'})
        p=client.patch('/projects/'+p['id'],json={'name':'Zenni 137912 — reviewed photo traces','annotations':annotations,'measurements':measurement,'notes':p['notes']+'\nResearch-guided correction: inspected front lens contours repaired to convex boundaries to exclude folded-arm pixels. Depth and arms remain inferred; angle view withheld. No exact reconstruction claim.'}).json()
        job=client.post('/projects/'+p['id']+'/jobs').json()
        deadline=time.time()+60
        while time.time()<deadline:
            job=client.get('/jobs/'+job['id']).json()
            if job['status'] not in {'queued','processing'}:break
            time.sleep(.25)
        result={'project_id':p['id'],'job':job,'source_url':ref['source_url'],'correction':'convex aperture repair after visual review','withheld_accuracy':'not measured: uncalibrated angle photograph, no ground truth mesh'}
        (out/'public-metal-result.json').write_text(json.dumps(result,indent=2))
        print(job['status'],job.get('error',''),p['id'])

if __name__=='__main__':main()
