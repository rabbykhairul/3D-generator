"""Native-pixel local outline review: bounded crops, no invented photo detail."""
import cv2
import numpy as np
from PIL import Image, ImageDraw


def regions(annotation):
    points=np.asarray([[p['x'],p['y']] for p in annotation.get('points',[])],float)
    if len(points)<3 or not np.isfinite(points).all():return []
    part=annotation.get('part','')
    if part.startswith('temple-'):
        anchors=annotation.get('anchors',{})
        left=points[np.argmin(points[:,0])];right=points[np.argmax(points[:,0])]
        if 'hinge' not in anchors or 'tip' not in anchors:
            return [(part+' · end A (ownership inferred)',left,points),(part+' · end B (ownership inferred)',right,points)]
        return [(part+' · '+k,np.array([anchors[k]['x'],anchors[k]['y']]),points) for k in ('hinge','tip')]
    if part in {'front','shield'}:
        return [(part+' · left edge',points[np.argmin(points[:,0])],points),
                (part+' · right edge',points[np.argmax(points[:,0])],points)]
    if part.startswith(('hinge-','nose-pad-')):
        return [(part,np.mean(points,axis=0),points)]
    return []


def zoom_diagnostics(photo, annotations, output=None):
    diagnostics=[];tiles=[]
    with Image.open(photo['path']) as image:
        image=image.convert('RGB')
        radius=int(np.clip(max(image.size)*.07,32,160))
        for annotation in annotations:
            for label,center,points in regions(annotation):
                if len(diagnostics)>=12:break
                x0=max(0,int(center[0])-radius);y0=max(0,int(center[1])-radius)
                x1=min(image.width,int(center[0])+radius);y1=min(image.height,int(center[1])+radius)
                if x1-x0<8 or y1-y0<8:continue
                crop=image.crop((x0,y0,x1,y1))
                gray=cv2.cvtColor(np.asarray(crop),cv2.COLOR_RGB2GRAY)
                # Inspect native crop edges, not the whole-image analysis thumbnail.
                edges=cv2.Canny(gray,30,90)
                distances=cv2.distanceTransform(np.uint8(edges==0),cv2.DIST_L2,3)
                poly=points-np.array([x0,y0]);boundary=np.zeros(gray.shape,np.uint8)
                cv2.polylines(boundary,[np.round(poly).astype(np.int32)],True,255,1)
                values=distances[boundary>0]
                support=float(np.mean(values<=2)) if len(values) else 0.
                diagnostic={'part':annotation.get('part'),'region':label,'source_crop_px':[x0,y0,x1,y1],
                    'native_pixels_inspected':True,'display_zoom':2,'adds_source_detail':False,
                    'edge_support_fraction_within_2px':round(support,3),
                    'status':'inspect faint edge/shadow/outline' if support<.5 else 'edge support present; review ownership'}
                diagnostics.append(diagnostic)
                if output:
                    tile=Image.new('RGB',(360,405),'white');draw=ImageDraw.Draw(crop)
                    draw.line([tuple(x) for x in np.vstack((poly,poly[:1]))],fill=(0,165,100),width=1)
                    crop=crop.resize((crop.width*2,crop.height*2),Image.Resampling.NEAREST)
                    crop.thumbnail((348,348))
                    tile.paste(crop,((360-crop.width)//2,35+(348-crop.height)//2))
                    d=ImageDraw.Draw(tile);d.text((8,8),label[:48],fill=(20,40,50));d.text((8,385),f'Native crop / 2x display / edge support {support:.0%}',fill=(65,80,90));tiles.append(tile)
    if output and tiles:
        rows=(len(tiles)+2)//3;sheet=Image.new('RGB',(1080,405*rows),(230,238,240))
        for i,tile in enumerate(tiles):sheet.paste(tile,((i%3)*360,(i//3)*405))
        sheet.save(output)
    return {'method':'native-source local crop edge review','regions':diagnostics,
            'adds_source_detail':False,'outline_changed':False,
            'limitations':['Edges may be shadows or reflections; support is not a recognition score.',
                           'Zoom inspection cannot establish missing or hidden components.']}
