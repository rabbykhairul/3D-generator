"""Isolated CPU inference for an explicitly configured eyewear candidate."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.eyewear_dataset import CLASSES
from scripts.train_eyewear_components import make_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--allow-exploratory', action='store_true',
                        help='Inspect an incomplete research checkpoint outside the app')
    args = parser.parse_args()
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(2)
    started = time.perf_counter()
    payload = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if payload.get('classes') != list(CLASSES) or not 256 <= payload.get('image_size', 0) <= 1536:
        raise ValueError('Eyewear checkpoint ontology or size is incompatible')
    if payload.get('exploratory_only') and not args.allow_exploratory:
        raise ValueError('Exploratory checkpoint cannot be used for app proposals')
    model = make_model()
    model.load_state_dict(payload['state_dict'], strict=True)
    model.eval()
    with Image.open(args.image) as source:
        if min(source.size) < 80 or max(source.size) > 2048:
            raise ValueError('Invalid source dimensions')
        original_size = source.size
        pixels = np.asarray(source.convert('RGB').resize((payload['image_size'],) * 2,
            Image.Resampling.BILINEAR), dtype=np.float32).transpose(2, 0, 1) / 255
    with torch.inference_mode():
        logits = model(torch.from_numpy(pixels)[None])['out']
        probabilities = torch.sigmoid(F.interpolate(logits,
            size=(original_size[1], original_size[0]), mode='bilinear', align_corners=False))[0]
    scores = probabilities.cpu().numpy().astype(np.float16)
    for index, name in enumerate(CLASSES):
        if name not in payload.get('trained_classes', CLASSES):
            scores[index].fill(0)
    np.savez_compressed(args.output, scores=scores, classes=np.asarray(CLASSES),
                        shape=np.asarray([original_size[1], original_size[0]]))
    print(json.dumps({'model': 'eyewear-components-lraspp', 'device': 'cpu',
                      'seconds': round(time.perf_counter() - started, 3),
                      'dataset_sha256': payload['dataset_sha256']}))


if __name__ == '__main__':
    main()
