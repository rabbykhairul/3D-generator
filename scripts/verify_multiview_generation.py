"""Isolated four-view generation and matching-camera rear evidence; no DB writes."""
import json
import sys
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.reconstruction import reconstruct
from backend.reconstruction.render_mesh import welded_geometry
from render_fidelity_cpu import render


def main():
    baseline = Path('artifacts/blue-arm-fix/after')
    output = Path('artifacts/multiview-resolution/generation')
    output.mkdir(parents=True, exist_ok=True)
    snapshot = json.loads((baseline/'snapshot.json').read_text())
    metadata = reconstruct(snapshot, output)
    scene = trimesh.load(output/'model.glb', force='scene', process=False)
    frame = scene.geometry['front_frame']
    assert frame.metadata['separate_inner_surface']
    assert metadata['multiview']['inner_surface']['observed_fraction'] > 0
    for name, mesh in scene.geometry.items():
        assert welded_geometry(mesh).is_watertight, name
        assert np.isfinite(mesh.vertices).all(), name
    before = baseline/'model.glb'
    target = trimesh.load(before, force='scene').bounds.mean(axis=0)
    for label, delta in [('rear', [0, .025, -.37]), ('rear-threequarter', [.22, .065, -.3]), ('front', [0, 0, .37])]:
        pair = Image.new('RGB', (1200, 660), 'white')
        draw = ImageDraw.Draw(pair)
        draw.text((12, 10), f'BEFORE / {label}', fill='black')
        draw.text((612, 10), f'AFTER / {label}', fill='black')
        draw.text((12, 30), 'Same-camera CPU diagnostic; lenses omitted. Not a live AR verification.', fill='black')
        for i, model in enumerate([before, output/'model.glb']):
            pair.paste(render(model, target+delta, target, 'photo', 600), (600*i, 60))
        pair.save(output/f'{label}-comparison.png')
    print(json.dumps({'checks': metadata['checks'], 'multiview': metadata['multiview']}, indent=2))


if __name__ == '__main__':
    main()
