"""Offline Meta SAM 3 concept segmentation on CPU, one image per process."""
import argparse
from contextlib import redirect_stdout
import sys
import json
from pathlib import Path
import time

from sam3_cpu_compat import install_cpu_distance_transform
from setup_sam3 import digest
from sam3_cpu_resources import protect_cpu_worker

PROMPTS = {
    'lenses': [('lens', 'lens'), ('frame', 'glasses frame'),
               ('bridge', 'glasses bridge'), ('nose-pad', 'nose pad'), ('hinge', 'glasses hinge')],
    'side': [('temple', 'glasses arm'), ('hinge', 'glasses hinge'), ('lens', 'lens')],
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--image', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--task', choices=PROMPTS, default='lenses')
    args = parser.parse_args()
    stop_monitor, peak_memory = protect_cpu_worker()
    spec = json.loads(args.manifest.read_text())
    checkpoint = args.manifest.parent/spec['path']
    if spec.get('model') != 'sam3' or not checkpoint.is_file() or digest(checkpoint) != spec['sha256']:
        raise SystemExit('Missing or invalid official SAM 3 checkpoint')
    import importlib.metadata
    package = importlib.metadata.distribution('sam3').locate_file('sam3')
    for relative, checksum in spec['cpu_adaptations'].items():
        if digest(package/relative) != checksum:
            raise SystemExit('SAM 3 CPU adaptation checksum mismatch; rerun setup_sam3.py')
    install_cpu_distance_transform()
    import numpy as np
    from PIL import Image
    import torch
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    started = time.perf_counter()
    with redirect_stdout(sys.stderr):
        model = build_sam3_image_model(device='cpu', checkpoint_path=str(checkpoint),
                                      load_from_HF=False, enable_inst_interactivity=False, compile=False)
    processor = Sam3Processor(model, device='cpu', confidence_threshold=.5)
    photo = Image.open(args.image).convert('RGB')
    w, h = photo.size
    packed, scores, roles = [], [], []
    with torch.inference_mode():
        state = processor.set_image(photo)
        for role, prompt in PROMPTS[args.task]:
            processor.reset_all_prompts(state)
            state = processor.set_text_prompt(prompt=prompt, state=state)
            for index in state['scores'].argsort(descending=True)[:8].tolist():
                mask = state['masks'][index].detach().cpu().numpy().reshape(h, w) > .5
                packed.append(np.packbits(mask.ravel()))
                scores.append(float(state['scores'][index]))
                roles.append(role)
    masks = np.stack(packed) if packed else np.zeros((0, (h*w+7)//8), np.uint8)
    np.savez_compressed(args.output, masks=masks, scores=np.asarray(scores, np.float32),
                        roles=np.asarray(roles, dtype='<U16'), shape=np.asarray([h,w], np.int32))
    print(json.dumps(dict(model='sam3', device='cpu', precision='float32',
                         image_encoder_resolution=1008, prompt_source='automatic-eyewear-text-concepts',
                         prompts=[prompt for _,prompt in PROMPTS[args.task]],
                         candidate_regions=len(scores), seconds=round(time.perf_counter()-started,3),
                         peak_rss_gib=round(peak_memory[0]/1024**3,3),
                         checkpoint_sha256=spec['sha256'])))
    stop_monitor.set()


if __name__ == '__main__':
    main()
