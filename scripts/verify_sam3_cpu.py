"""Check installed CPU adaptations without loading the full SAM 3 checkpoint."""
import importlib.metadata
import json
from pathlib import Path
import torch
from sam3_cpu_compat import install_cpu_distance_transform, patch_installation

package=importlib.metadata.distribution('sam3').locate_file('sam3')
before=patch_installation(package)
assert patch_installation(package)==before, 'CPU setup must be idempotent'
manifest=Path(__file__).resolve().parents[1]/'artifacts/sam3/manifest.json'
if manifest.exists():
    assert json.loads(manifest.read_text())['cpu_adaptations']==before
install_cpu_distance_transform()
from sam3.model.vitdet import Mlp
from sam3.model.edt import edt_triton
from sam3.model.position_encoding import PositionEmbeddingSine
model=Mlp(8,16).eval()
x=torch.randn(2,5,8)
with torch.inference_mode():
    actual=model(x)
    expected=model.fc2(torch.nn.functional.gelu(model.fc1(x)))
assert actual.device.type=='cpu' and actual.dtype==torch.float32
assert torch.allclose(actual,expected)
mask=torch.zeros((1,5,5),dtype=torch.bool);mask[:,1:4,1:4]=True
assert edt_triton(mask)[0,2,2]==2
assert len(PositionEmbeddingSine(256,normalize=True,precompute_resolution=1008).cache)==8
print('CPU adaptations are idempotent; float32 MLP, distance transform, caches and manifest pass.')
