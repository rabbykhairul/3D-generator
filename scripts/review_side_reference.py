import json
import sys
from pathlib import Path
from PIL import Image,ImageDraw
from backend.reconstruction import extract_annotations

path=Path(sys.argv[1])
image=Image.open(path).convert('RGB')
photo={'id':'side','view':'left','path':str(path),'width':image.width,'height':image.height}
result=extract_annotations(photo)
out=Path('artifacts/side-reference');out.mkdir(parents=True,exist_ok=True)
(out/'suggestions.json').write_text(json.dumps(result,indent=2))
draw=ImageDraw.Draw(image)
for a in result['annotations']:
    p=[(v['x'],v['y']) for v in a['points']];draw.line(p+[p[0]],fill='orange',width=1)
    for name,p in a['anchors'].items():draw.ellipse((p['x']-1,p['y']-1,p['x']+1,p['y']+1),fill='red' if name=='hinge' else 'blue')
image.resize((600,600),Image.Resampling.NEAREST).save(out/'outline.png')
print(json.dumps(result))
