"""Small CPU adaptations for the pinned Meta SAM 3 image implementation."""
import hashlib
from pathlib import Path
import sys
import types


def patch_installation(package):
    replacements = {
        'model/position_encoding.py': ('device="cuda")', 'device="cpu")'),
        'model/decoder.py': ('feat_size, feat_size, device="cuda"', 'feat_size, feat_size, device="cpu"'),
        'model/geometry_encoders.py': (
            'scale = scale.pin_memory().to(device=boxes_xyxy.device, non_blocking=True)',
            'scale = scale.to(device=boxes_xyxy.device)'),
        'model_builder.py': ('torch.load(f, map_location="cpu", weights_only=True)',
                             'torch.load(checkpoint_path, map_location="cpu", weights_only=True, mmap=True)'),
        'perflib/fused.py': ('    self = linear.bias.detach()',
            '    if mat1.device.type == "cpu":\n'
            '        y = torch.nn.functional.linear(mat1, linear.weight, linear.bias)\n'
            '        if activation in [torch.nn.functional.relu, torch.nn.ReLU]:\n'
            '            return torch.nn.functional.relu(y)\n'
            '        if activation in [torch.nn.functional.gelu, torch.nn.GELU]:\n'
            '            return torch.nn.functional.gelu(y)\n'
            '        raise ValueError(f"Unexpected activation {activation}")\n'
            '    self = linear.bias.detach()'),
    }
    provenance = {}
    for relative, (old, new) in replacements.items():
        path = Path(package)/relative
        content = path.read_text()
        if new in content:
            pass
        elif old in content:
            if content.count(old) != (3 if relative == 'model_builder.py' else 1):
                raise RuntimeError(f'Unexpected SAM 3 source layout: {relative}')
            content = content.replace(old, new)
            path.write_text(content)
        elif new not in content:
            raise RuntimeError(f'CPU adaptation no longer matches {relative}')
        provenance[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return provenance


def install_cpu_distance_transform():
    """Avoid an unconditional optional CUDA/Triton tracker import on macOS.

    This is a real CPU distance transform, not a model or segmentation stub.
    Image-only concept inference does not use the tracker point-selection EDT.
    """
    import numpy as np
    import torch
    from scipy.ndimage import distance_transform_edt

    def edt(data):
        if data.ndim != 3 or data.device.type != 'cpu':
            raise ValueError('CPU EDT expects a BxHxW CPU tensor')
        arrays = data.detach().numpy()
        result = np.stack([distance_transform_edt(a) for a in arrays]).astype(np.float32)
        return torch.from_numpy(result)

    module = types.ModuleType('sam3.model.edt')
    module.edt_triton = edt
    sys.modules['sam3.model.edt'] = module
