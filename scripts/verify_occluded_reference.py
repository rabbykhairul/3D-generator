"""Review a real occluded front photo; never auto-approve user project edits."""
import argparse
import json
from pathlib import Path
from PIL import Image, ImageDraw
from backend.reconstruction import extract_annotations, reconstruct

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('photo',type=Path)
    parser.add_argument('--approve',action='store_true')
    args=parser.parse_args()
    image=Image.open(args.photo).convert('RGB')
    photo={'id':'occluded-front','view':'front','role':'fitting','path':str(args.photo.resolve()),'width':image.width,'height':image.height}
    result=extract_annotations(photo)
    out=Path('artifacts/occluded-reference');out.mkdir(parents=True,exist_ok=True)
    draw=ImageDraw.Draw(image)
    for ann in result['annotations']:
        pts=[(p['x'],p['y']) for p in ann['points']]
        draw.line(pts+[pts[0]],fill='magenta' if ann['part']=='bridge-opening' else 'lime' if ann['part']=='front' else 'orange',width=4)
    image.save(out/'outline-review.png')
    (out/'suggestions.json').write_text(json.dumps(result,indent=2))
    print([(a['part'],a['source']) for a in result['annotations']])
    print(result['warnings'])
    if not args.approve:return
    for ann in result['annotations']:
        ann['approved']=True
        ann['source']='research-reviewed:'+ann['source']
    data={'category':'opaque-full-rim','photos':[photo],'annotations':result['annotations'],
          'measurements':{k:{'value':v,'source':'default'} for k,v in {'frame_width':140,'lens_width':54,'lens_height':43,'bridge_width':20,'temple_length':140}.items()},
          'material':{'color':'#55583b'}}
    metadata=reconstruct(data,out)
    print(json.dumps({'parts':metadata['parts'],'bridge_openings':metadata['bridge_openings'],'status':metadata['reconstruction_status'],'checks':metadata['checks']},indent=2))
    (out/'snapshot.json').write_text(json.dumps(data,indent=2))

if __name__=='__main__':main()
