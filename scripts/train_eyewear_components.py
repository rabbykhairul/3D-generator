"""Train a multi-label eyewear segmenter from reviewed product-split masks.

Unknown classes are excluded from loss. Training makes a candidate checkpoint,
never a production approval or a 3D-fidelity claim. Run in segmenter-venv.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.eyewear_dataset import CLASSES, audit, load_manifest


def checked_items(manifest, split):
    items = [item for item in manifest['items'] if item.get('reviewed') and item['split'] == split]
    for item in items:
        if not Path(item['image']).is_file():
            raise ValueError(f"Missing source photo for {item['product_id']}")
        if set(item.get('labels', {})) & set(item.get('absent', [])):
            raise ValueError(f"Contradictory present/absent label for {item['product_id']}")
        for path in item.get('labels', {}).values():
            if not Path(path).is_file():
                raise ValueError(f"Missing reviewed mask for {item['product_id']}")
    return items


def load_example(item, size):
    """Return RGB, per-class targets and per-class validity at fixed resolution."""
    with Image.open(item['image']) as source:
        original_size = source.size
        image = np.asarray(source.convert('RGB').resize((size, size), Image.Resampling.BILINEAR),
                           dtype=np.float32).transpose(2, 0, 1) / 255
    targets = np.zeros((len(CLASSES), size, size), np.float32)
    known = np.zeros((len(CLASSES), 1, 1), np.float32)
    for index, name in enumerate(CLASSES):
        path = item.get('labels', {}).get(name)
        if path:
            with Image.open(path) as source:
                if source.size != original_size:
                    raise ValueError(f"Mask dimensions differ from photo for {item['product_id']}:{name}")
                targets[index] = np.asarray(source.convert('L').resize(
                    (size, size), Image.Resampling.NEAREST)) >= 128
            known[index] = 1
        elif name in item.get('absent', []):
            known[index] = 1
    return image, targets, known


def make_model():
    from torchvision.models.segmentation import lraspp_mobilenet_v3_large
    return lraspp_mobilenet_v3_large(weights=None, weights_backbone=None,
                                    num_classes=len(CLASSES))


def evaluate(model, items, size, device):
    import torch
    import torch.nn.functional as F
    # Intersection, union, known photos, known-absent photos, absent FP pixels.
    counts = {name: [0, 0, 0, 0, 0] for name in CLASSES}
    model.eval()
    with torch.inference_mode():
        for item in items:
            image, target, known = load_example(item, size)
            logits = model(torch.from_numpy(image)[None].to(device))['out']
            predicted = (torch.sigmoid(F.interpolate(logits, size=(size, size),
                mode='bilinear', align_corners=False))[0].cpu().numpy() >= .5)
            for index, name in enumerate(CLASSES):
                if not known[index, 0, 0]:
                    continue
                truth = target[index] >= .5
                counts[name][0] += int(np.count_nonzero(predicted[index] & truth))
                counts[name][1] += int(np.count_nonzero(predicted[index] | truth))
                counts[name][2] += 1
                if name in item.get('absent', []):
                    counts[name][3] += 1
                    counts[name][4] += int(np.count_nonzero(predicted[index]))
    return {name: {'iou': round(hit / union, 4) if union else None,
                   'known_photos': known, 'known_absent_photos': absent,
                   'absent_false_positive_pixels': false_positive}
            for name, (hit, union, known, absent, false_positive) in counts.items()}


def train(dataset, output, epochs, size, device_name, backbone_checkpoint=None,
          exploratory=False):
    manifest = load_manifest(dataset)
    review = audit(manifest)
    if not exploratory and not review['ready_for_training']:
        raise ValueError('Dataset is not ready: ' + json.dumps(review))
    if exploratory and (review['split_conflicts'] or any(
            review['products_by_split'][split] == 0 for split in ('train', 'val', 'test'))):
        raise ValueError('Exploratory training requires conflict-free product holdouts')
    import torch
    import torch.nn.functional as F
    if size < 256 or size > 1536 or epochs < 1:
        raise ValueError('Use image size 256–1536 and at least one epoch')
    items = {split: checked_items(manifest, split) for split in ('train', 'val', 'test')}
    trained_classes = [name for name in CLASSES if any(
        name in item.get('labels', {}) or name in item.get('absent', [])
        for item in items['train'])]
    device = torch.device(device_name)
    torch.set_num_threads(2)
    torch.manual_seed(7)
    model = make_model().to(device)
    backbone = None
    if backbone_checkpoint:
        path = Path(backbone_checkpoint)
        pretrained = torch.load(path, map_location='cpu', weights_only=True)
        current = model.state_dict()
        compatible = {key: value for key, value in pretrained.items()
                      if key.startswith('backbone.') and key in current and
                      value.shape == current[key].shape}
        if len(compatible) < 100:
            raise ValueError('Checkpoint does not contain a compatible pretrained backbone')
        current.update(compatible)
        model.load_state_dict(current, strict=True)
        backbone = {'path': str(path.resolve()),
                    'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                    'loaded_tensors': len(compatible)}
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    best, history = float('-inf'), []
    output.mkdir(parents=True, exist_ok=True)
    manifest_hash = hashlib.sha256((dataset / 'manifest.json').read_bytes()).hexdigest()
    for epoch in range(epochs):
        model.train()
        losses = []
        for item in items['train']:
            image, target, known = load_example(item, size)
            x = torch.from_numpy(image)[None].to(device)
            y = torch.from_numpy(target)[None].to(device)
            valid = torch.from_numpy(known)[None].to(device)
            if not valid.any():
                continue
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)['out']
            logits = F.interpolate(logits, size=(size, size), mode='bilinear', align_corners=False)
            loss = (F.binary_cross_entropy_with_logits(logits, y, reduction='none') * valid).sum() / (
                valid.sum() * size * size)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        val = evaluate(model, items['val'], size, device)
        observed = [value['iou'] for value in val.values() if value['iou'] is not None]
        score = float(np.mean(observed)) if observed else float('-inf')
        history.append({'epoch': epoch + 1, 'train_loss': round(float(np.mean(losses)), 5),
                        'val_mean_iou': round(score, 4) if observed else None})
        if score > best:
            best = score
            torch.save({'state_dict': model.cpu().state_dict(), 'classes': list(CLASSES),
                        'image_size': size, 'dataset_sha256': manifest_hash,
                        'trained_classes': trained_classes,
                        'exploratory_only': exploratory}, output / 'candidate.pt')
            model.to(device)
    selected = torch.load(output / 'candidate.pt', map_location=device, weights_only=True)
    model.load_state_dict(selected['state_dict'], strict=True)
    training_diagnostics = evaluate(model, items['train'], size, device)
    test = evaluate(model, items['test'], size, device)
    report = {'dataset': str((dataset / 'manifest.json').resolve()), 'dataset_sha256': manifest_hash,
              'counts': review, 'device': device_name, 'epochs': epochs, 'image_size': size,
              'backbone': backbone, 'history': history, 'training_diagnostics': training_diagnostics,
              'test': test, 'release_gate_passed': False,
              'exploratory_only': exploratory, 'trained_classes': trained_classes,
              'reason': 'Component IoU alone does not validate rimless topology, hidden hardware or GLB fidelity.'}
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=ROOT / 'artifacts/eyewear-training')
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/eyewear-training/run-1')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--size', type=int, default=768)
    parser.add_argument('--device', choices=('cpu', 'mps', 'cuda'), default='cpu')
    parser.add_argument('--exploratory', action='store_true',
                        help='Train an isolated research checkpoint on incomplete labels and holdouts')
    parser.add_argument('--backbone-checkpoint', type=Path,
                        help='Optional local compatible pretrained LRASPP backbone; review its commercial rights')
    args = parser.parse_args()
    report = train(args.dataset, args.output, args.epochs, args.size,
                   args.device, args.backbone_checkpoint, args.exploratory)
    print(json.dumps({'candidate': str(args.output / 'candidate.pt'),
                      'test': report['test'], 'release_gate_passed': False}, indent=2))


if __name__ == '__main__':
    main()
