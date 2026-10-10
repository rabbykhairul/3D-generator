"""Read-only GLB audit; extracted inspection artifacts stay outside app assets."""
import io
import json
import struct
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import trimesh


def main():
    path = Path('/Users/dekuddus/Downloads/sku_215926.glb')
    out = Path('artifacts/supplier-glb-audit')
    out.mkdir(parents=True, exist_ok=True)
    payload = path.read_bytes()
    n = struct.unpack_from('<I',payload,12)[0]
    document = json.loads(payload[20:20+n])
    binary = payload[28+n:]
    images = []
    for i, item in enumerate(document.get('images',[])):
        view = document['bufferViews'][item['bufferView']]
        start = view.get('byteOffset',0)
        image = Image.open(io.BytesIO(binary[start:start+view['byteLength']]))
        image.save(out/f'image-{i}.png')
        images.append(dict(index=i,name=item.get('name'),size=image.size,mode=image.mode))
    scene = trimesh.load(path,force='scene',process=False)
    geometry = []
    for name, mesh in scene.geometry.items():
        geometry.append(dict(name=name,vertices=len(mesh.vertices),triangles=len(mesh.faces),
            bounds=mesh.bounds.tolist(),material=getattr(mesh.visual.material,'name',None),
            uv_count=len(mesh.visual.uv) if getattr(mesh.visual,'uv',None) is not None else 0,
            normal_lengths=[float(v) for v in np.percentile(np.linalg.norm(mesh.vertex_normals,axis=1),[0,50,100])]))
    report = dict(source=str(path),asset=document['asset'],materials=document.get('materials'),
                  nodes=document.get('nodes'),textures=document.get('textures'),images=images,
                  scene_bounds=scene.bounds.tolist(),geometry=geometry)
    (out/'inspection.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(dict(images=images,scene_bounds=report['scene_bounds'],geometry=geometry),indent=2))
    # Diagnostic copy only: omit transparent lenses so interior geometry is visible.
    scene.delete_geometry([name for name,mesh in scene.geometry.items()
                           if getattr(mesh.visual.material,'name',None)=='GLTF_Eye_Lens'])
    model = out/'inspection-without-lenses.glb'
    scene.export(model)
    from render_fidelity_cpu import render
    target = scene.bounds.mean(axis=0)
    cameras = [('Front',target+np.array([0,0,.29])),
               ('Rear / inner surfaces',target+np.array([.16,.08,-.25])),
               ('Side / hinge',target+np.array([-.28,.035,0]))]
    sheet = Image.new('RGB',(1500,550),'white')
    draw = ImageDraw.Draw(sheet)
    for col,(name,eye) in enumerate(cameras):
        draw.text((col*500+10,10),name+' - supplier GLB, neutral clay',fill='black')
        draw.text((col*500+10,28),'CPU geometry diagnostic; lenses omitted',fill='black')
        sheet.paste(render(model,eye,target,'clay',500),(col*500,50))
    sheet.save(out/'supplier-geometry.png')


if __name__=='__main__':
    main()
