"""Fetch only public product photos actually referenced by their product pages.

No guessed image URLs, no customer photos, no credentials, no scraping bypass.
The resulting data is a local research cache, not licensed redistribution.
"""
import concurrent.futures
import html
import io
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import httpx
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'data' / 'references'
SOURCES = [
 ('zenni-4447121','opaque-full-rim','https://www.zennioptical.com/p/acetate-square-eyeglass-frames/44471/4447121'),
 ('zenni-137912','thin-metal','https://www.zennioptical.com/p/titanium-round-glasses/1379/137912'),
 ('zenni-4414123','transparent-full-rim','https://www.zennioptical.com/p/womens-acetate-plastic-round-eyeglass-frames/44141/4414123'),
 ('zenni-4440123','transparent-full-rim','https://ca.zennioptical.com/p/womens-acetate-round-eyeglass-frames/44401/4440123'),
 ('zenni-3239016','semi-rimless','https://www.zennioptical.com/p/mens-halfrim-stainless-steel-rectangle-eyeglass-frames/32390/3239016'),
 ('zenni-3230911','rimless','https://www.zennioptical.com/p/womens-rimless-stainless-steel-eyeglass-frames/32309/3230911'),
 ('zenni-1143121','sports-shield','https://www.zennioptical.com/p/halfrim-plastic-wrap-around-sunglass-frames/11431/1143121'),
]

def collect(source):
    sid, category, url = source
    item = dict(id=sid,category=category,kind='public',source_url=url,photos=[],measurements={},retrieved_at=datetime.now(timezone.utc).isoformat(),limitations=['Public reference photos are uncalibrated; exact physical fidelity is unverified.','Missing views and thickness must be supplied or explicitly inferred.'],description='Public product photos. Review and correct outlines before generating.')
    try:
        with httpx.Client(follow_redirects=True,timeout=35) as client:
            r=client.get(url);r.raise_for_status()
            content=html.unescape(r.text).replace('\\u0026','&').replace('\\/','/')
            title=re.search(r'<title[^>]*>(.*?)</title>',content,re.S)
            item['name']=re.sub('<[^>]+>','',title.group(1)).split('|')[0].strip() if title else sid
            text=html.unescape(re.sub(r'<[^>]+>',' ',content))
            text=re.sub(r'\s+',' ',text)
            labels={'frame_width':r'Frame Width','lens_width':r'Lens Width','lens_height':r'Lens Height','bridge_width':r'Bridge(?: width)?','temple_length':r'Temple Length'}
            for key,label in labels.items():
                match=re.search(label+r'\s*(\d+(?:\.\d+)?)\s*mm',text,re.I)
                if match:item['measurements'][key]={'value':float(match.group(1)),'source':'published'}
            sku=sid.split('-')[-1]
            candidates=list(dict.fromkeys(re.findall(r'https://static\.zennioptical\.com/production/products/[^\s"<>\\]+?\.(?:jpg|png)',content)))
            candidates=[u for u in candidates if '/'+sku+'-' in u]
            folder=DEST/sid;folder.mkdir(parents=True,exist_ok=True)
            for image_url in candidates[:8]:
                label='front' if 'front' in image_url else 'left' if 'side' in image_url else 'front-left' if 'angle' in image_url else 'detail'
                response=client.get(image_url);response.raise_for_status()
                im=Image.open(io.BytesIO(response.content));im.verify()
                path=folder/f'{len(item["photos"]):02d}.jpg'
                path.write_bytes(response.content)
                item['photos'].append(dict(path=str(path.relative_to(ROOT)),view=label,role='validation' if label=='front-left' and any(p['view']=='front' for p in item['photos']) else 'fitting',image_source_url=image_url))
            # Reserve angle independently of discovery order when a front image exists.
            if any(p['view']=='front' for p in item['photos']):
                for p in item['photos']:
                    if p['view']=='front-left':p['role']='validation'
            item['missing_measurements']=[key for key in labels if key not in item['measurements']]
            item['available_views']=[p['view'] for p in item['photos']]
            item['status']='available' if item['photos'] else 'no-product-images-found'
            if 'front' not in item['available_views']:item['limitations'].append('No front view in the retrieved page. Add a front image before reconstruction.')
    except Exception as e:
        item['status']='unavailable';item['error']=f'{type(e).__name__}: {e}'
        item.setdefault('name',sid)
    print(sid,item['status'],len(item['photos']),'photos',item.get('measurements',{}),flush=True)
    return item

if __name__=='__main__':
    DEST.mkdir(parents=True,exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        items=list(pool.map(collect,SOURCES))
    (DEST/'manifest.json').write_text(json.dumps(items,indent=2))
    print('Saved local reference manifest:',DEST/'manifest.json')
