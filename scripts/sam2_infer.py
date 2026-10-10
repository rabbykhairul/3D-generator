"""Bounded local SAM 2.1 Small inference; no inference-time downloads."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--image',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--task',choices=('lenses','side'),default='lenses')
    args=p.parse_args()
    spec=json.loads(args.manifest.read_text())
    checkpoint=(args.manifest.parent/spec['path']).resolve()
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=spec['sha256']:
        raise ValueError('SAM checkpoint checksum mismatch')
    import torch
    from sam2.build_sam import build_sam2
    from scripts.sam2_proposals import generate, guided_lenses
    torch.set_num_threads(2)
    started=time.perf_counter()
    model=build_sam2(spec['config'],str(checkpoint),device='cpu',apply_postprocessing=False)
    with Image.open(args.image) as source:
        if max(source.size)>2048 or min(source.size)<1:
            raise ValueError('Invalid analysis dimensions')
        pixels=np.array(source.convert('RGB'))
    with torch.inference_mode():
        regions=None
        locator_manifest=Path(os.environ.get('EYEWEAR_SEGMENTER_MANIFEST',ROOT/'artifacts/segmenter-evaluation/medium.json'))
        if args.task=='lenses' and locator_manifest.is_file():
            from scripts.evaluate_glasses_segmenter import checked_checkpoint,load_cpu_segmenter
            locator_spec=json.loads(locator_manifest.read_text())['candidate']['checkpoints']['lenses']
            locator=load_cpu_segmenter(locator_spec,checked_checkpoint(locator_spec,locator_manifest.parent))
            probability=locator.predict(Image.fromarray(pixels),format='proba',output_size=(pixels.shape[1],pixels.shape[0])).cpu().numpy()
            regions=guided_lenses(model,pixels,probability)
        guided=regions is not None
        if not guided:
            regions=generate(model,pixels)
    shape=pixels.shape[:2]
    masks=np.stack([np.packbits(r['segmentation'].ravel()) for r in regions]) if regions else np.empty((0,(shape[0]*shape[1]+7)//8),np.uint8)
    np.savez_compressed(args.output,masks=masks,shape=np.array(shape),
                        scores=np.array([r['predicted_iou'] for r in regions]),
                        stability=np.array([r['stability_score'] for r in regions]))
    runtime=dict(model='sam2.1-hiera-small',parameters_million=46,
        seconds=round(time.perf_counter()-started,3),device='cpu',automatic_prompts=10 if guided else 72,
        prompt_source='coarse-lens-locator' if guided else 'automatic-photo-grid',
        candidate_regions=len(regions),checkpoint_sha256=spec['sha256'])
    print(json.dumps(runtime))


if __name__=='__main__':
    main()
