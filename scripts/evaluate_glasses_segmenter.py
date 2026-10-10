"""Isolated CPU candidate evaluation. NOT a production extraction backend.

Requires Python 3.12+, glasses-detector and local trusted checkpoints. Never
downloads weights, auto-approves geometry or changes uploaded projects.
Binary frames includes both arms; legs does not separate left/right instances.
No nose-pad or hinge detector is supplied by this candidate.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import platform

import numpy as np
from PIL import Image, ImageDraw, ImageOps


def checked_checkpoint(spec, directory):
    path = (directory / spec['path']).resolve()
    if not path.is_file():
        raise ValueError('Local checkpoint is missing')
    if not spec.get('license_review') or not spec.get('source_url'):
        raise ValueError('Record checkpoint provenance and license review before loading')
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != spec.get('sha256'):
        raise ValueError('Checkpoint SHA256 mismatch')
    return path


def mask_metrics(prediction, truth):
    if prediction.shape != truth.shape:
        raise ValueError('Label mask must match original photo dimensions')
    union = np.count_nonzero(prediction | truth)
    intersection = np.count_nonzero(prediction & truth)
    return dict(iou=intersection/union if union else 1.,
                false_positive_pixels=int(np.count_nonzero(prediction & ~truth)),
                false_negative_pixels=int(np.count_nonzero(~prediction & truth)))


def mask_overlay(image, mask):
    """Source-sized overlay for inspection, not a geometry accuracy score."""
    pixels = np.asarray(image).copy()
    pixels[mask] = (pixels[mask] * .55 + np.array([255, 50, 40]) * .45).astype(np.uint8)
    return Image.fromarray(pixels)


def load_cpu_segmenter(spec, path):
    """Construct without torchvision's implicit ImageNet-backbone download."""
    import torch
    from glasses_detector import GlassesSegmenter
    from glasses_detector.architectures import TinyBinarySegmenter
    from torchvision.models.segmentation import lraspp_mobilenet_v3_large

    size = spec.get('size', 'medium')
    if size == 'small':
        network = TinyBinarySegmenter()
    elif size == 'medium':
        network = lraspp_mobilenet_v3_large(weights=None, weights_backbone=None, num_classes=1)
    else:
        raise ValueError('This evaluator supports small and medium checkpoints only')
    network.load_state_dict(torch.load(path, map_location='cpu', weights_only=True), strict=True)
    network.eval()
    # v1.0.4's from_model() fails when initialized with its default custom kind.
    # Bootstrap the tiny architecture (no backbone downloads), then replace it
    # with the strictly loaded network while retaining the official transforms.
    segmenter = GlassesSegmenter(size='small', weights=False, device='cpu')
    segmenter.model = network
    segmenter.size = size
    return segmenter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path, help='cases plus candidate.checkpoints by frames/lenses/legs')
    parser.add_argument('--output', type=Path, required=True, help='Separate evaluation directory')
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('--threads must be positive')
    manifest = json.loads(args.manifest.read_text())
    candidate = manifest.get('candidate', {})
    specs = candidate.get('checkpoints', {})
    if not specs or any(k not in {'frames', 'lenses', 'legs'} for k in specs):
        parser.error('Provide local frames, lenses and/or legs checkpoint records')
    try:
        paths = {kind: checked_checkpoint(spec, args.manifest.parent) for kind, spec in specs.items()}
    except (ValueError, KeyError) as exc:
        parser.error(str(exc))
    if sys.version_info < (3, 12):
        parser.error('Use an isolated Python 3.12+ environment; do not replace the app environment')
    try:
        import torch
        from glasses_detector import GlassesSegmenter
    except ImportError:
        parser.error('Candidate dependencies not installed in this evaluation environment')
    torch.set_num_threads(args.threads)
    started = time.perf_counter()
    models = {kind: load_cpu_segmenter(specs[kind], path) for kind, path in paths.items()}
    load_seconds = time.perf_counter() - started
    for model in models.values():
        model.model.eval()
        with torch.inference_mode():
            model.predict(Image.new('RGB', (256, 256), 'white'), format='proba')
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, case in enumerate(manifest['cases']):
        row = dict(sku=case['sku'], name=case.get('name', case['sku']),
                   photo_id=case.get('photo_id'), view=case['view'], parts={})
        try:
            with Image.open(args.manifest.parent / case['path']) as source:
                image = source.convert('RGB')
            if case.get('sha256') and hashlib.sha256((args.manifest.parent / case['path']).read_bytes()).hexdigest() != case['sha256']:
                raise ValueError('Photo SHA256 mismatch')
            panels = [('Source', image)]
            for kind, model in models.items():
                started = time.perf_counter()
                with torch.inference_mode():
                    scores = model.predict(image, format='proba', output_size=image.size)
                prediction_seconds = time.perf_counter()-started
                probabilities = scores.detach().cpu().numpy()
                if (probabilities.shape != (image.height, image.width) or
                        not np.isfinite(probabilities).all() or
                        probabilities.min() < 0 or probabilities.max() > 1):
                    raise ValueError('Invalid model mask dimensions or values')
                mask = probabilities >= .5
                name = f'{index:05d}-{kind}.png'
                Image.fromarray(mask.astype(np.uint8)*255).save(args.output / name)
                stats = dict(mask=name, seconds=round(prediction_seconds, 4), labelled=False,
                             positive_fraction=round(float(mask.mean()), 6))
                panels.append((kind, mask_overlay(image, mask)))
                label = case.get('expected_masks', {}).get(kind)
                if label:
                    with Image.open(args.manifest.parent / label) as source:
                        truth = np.asarray(source.convert('L')) >= 128
                    stats.update(mask_metrics(mask, truth), labelled=True)
                row['parts'][kind] = stats
            sheet = Image.new('RGB', (320*len(panels), 350), 'white')
            draw = ImageDraw.Draw(sheet)
            for column, (label, panel) in enumerate(panels):
                thumbnail = ImageOps.contain(panel, (320, 310))
                sheet.paste(thumbnail, (column*320+(320-thumbnail.width)//2, 30+(310-thumbnail.height)//2))
                draw.text((column*320+8, 8), label, fill='black')
            sheet.save(args.output / f'{index:05d}-comparison.jpg', quality=92)
            row['comparison'] = f'{index:05d}-comparison.jpg'
        except Exception as exc:
            row['error'] = f'{type(exc).__name__}: {exc}'
        rows.append(row)
    timings = [part['seconds'] for row in rows for part in row['parts'].values()]
    report = dict(cpu_only=True, candidate='glasses-detector', checkpoints=specs, cases=rows,
                  environment=dict(python=platform.python_version(), torch=torch.__version__,
                                   threads=args.threads, input_size=[256,256],
                                   model_load_seconds=round(load_seconds, 4)),
                  timing=dict(prediction_median_seconds=float(np.median(timings)) if timings else None,
                              prediction_p95_seconds=float(np.percentile(timings,95)) if timings else None),
                  release_gate_passed=False, production_connected=False,
                  limits=['Binary mask evaluation only; no attachment positions or part instances.',
                          'No commercial reconstruction fidelity claim. No nose-pad/hinge model.',
                          'Independent product-level holdout and render testing still required.'])
    (args.output / 'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(cases=len(rows), errors=sum('error' in r for r in rows),
                          report=str(args.output / 'report.json')), indent=2))
    return 2 if any('error' in r for r in rows) else 0


if __name__ == '__main__':
    raise SystemExit(main())
