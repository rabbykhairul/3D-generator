"""One bounded CPU lens inference request in the isolated Python environment.

Uses local checkpoints only. Input is a temporary analysis image, not an upload
destination. No frames/legs masks are used as final geometry.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluate_glasses_segmenter import checked_checkpoint, load_cpu_segmenter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import torch
    torch.set_num_threads(2)
    manifest = json.loads(args.manifest.read_text())
    spec = manifest['candidate']['checkpoints']['lenses']
    if spec['size'] != 'medium':
        raise ValueError('Hybrid extraction requires the evaluated medium lens checkpoint')
    path = checked_checkpoint(spec, args.manifest.parent)
    model = load_cpu_segmenter(spec, path)
    with Image.open(args.image) as source:
        if max(source.size) > 2048 or min(source.size) < 1:
            raise ValueError('Analysis image exceeds the inference size limit')
        image = source.convert('RGB')
    started = time.perf_counter()
    with torch.inference_mode():
        scores = model.predict(image, format='proba', output_size=image.size)
    np.save(args.output, scores.detach().cpu().numpy().astype(np.float32), allow_pickle=False)
    print(json.dumps(dict(seconds=round(time.perf_counter()-started, 4),
                         checkpoint_sha256=spec['sha256'], device='cpu', input_size=[256,256])))


if __name__ == '__main__':
    main()
