"""Compare identical approved geometry with old/new clear texture policy."""
from pathlib import Path
import json
from copy import deepcopy
import trimesh
from backend.store import get_project, list_projects
from backend.reconstruction import engine
from scripts.render_fidelity_cpu import render


def main():
    project = next(get_project(p['id']) for p in list_projects()
                   if p['id'].startswith('f94cc9fb'))
    folder=Path('artifacts/clear-material');folder.mkdir(exist_ok=True,parents=True)
    (folder/'project.json').write_text(json.dumps(project,indent=2))
    original=engine._frame_texture
    for label in ('before','after'):
        try:
            if label=='before':
                engine._frame_texture=lambda photo,poly,**kwargs: original(photo,poly,preserve_clear=False)
            engine.reconstruct(deepcopy(project),folder/label)
        finally:
            engine._frame_texture=original
        scene=trimesh.load(folder/label/'model.glb',force='scene',process=False)
        target=scene.bounds.mean(axis=0)
        # Texture diagnostic only; CPU renderer does not simulate transparency.
        render(folder/label/'model.glb',target+[0,0,.38],target,'photo',900).save(folder/f'{label}-texture.png')
        mesh=next(mesh for name,mesh in scene.geometry.items() if name.startswith('front_frame'))
        mat=mesh.visual.material
        mat.baseColorTexture.save(folder/f'{label}-atlas.png')
        print(label,mat.name,mat.alphaMode,mat.baseColorFactor.tolist())


if __name__=='__main__':main()
