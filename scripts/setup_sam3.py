"""Prepare official gated SAM 3 weights for offline CPU image inference.

Authenticate locally with artifacts/sam3-venv/bin/hf auth login after receiving
access to facebook/sam3. No tokens, photos or project records are printed.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import GatedRepoError
from sam3_cpu_compat import patch_installation

ROOT = Path(__file__).resolve().parents[1]
CODE_REVISION = '2345a4ad109ac29c569da749c91d84f10dc08c40'
WEIGHTS_REVISION = '3c879f39826c281e95690f02c7821c4de09afae7'


def digest(path):
    sha = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b''):
            sha.update(chunk)
    return sha.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'artifacts/sam3')
    parser.add_argument('--compat-only', action='store_true')
    args = parser.parse_args()
    directory = args.output.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    package = importlib.metadata.distribution('sam3').locate_file('sam3')
    patches = patch_installation(package)
    (directory/'cpu-adaptations.json').write_text(json.dumps(patches, indent=2)+'\n')
    if args.compat_only:
        print('Prepared SAM 3 CPU adaptations; pretrained weights were not loaded.')
        return
    try:
        info = HfApi().model_info('facebook/sam3', revision=WEIGHTS_REVISION, files_metadata=True)
        weight = next(f for f in info.siblings if f.rfilename == 'sam3.pt')
        expected = weight.lfs.sha256
        downloaded = Path(hf_hub_download('facebook/sam3', 'sam3.pt', revision=WEIGHTS_REVISION))
    except GatedRepoError:
        raise SystemExit('Hugging Face denied access to the official SAM 3 weights. '
                         'Open https://huggingface.co/facebook/sam3 in the account used for local login, '
                         'accept the conditions and request access. After approval, ensure your token '
                         'allows reading this gated repository. Login alone does not grant model access. '
                         'Rerun this script when repository access is granted.')
    if digest(downloaded) != expected:
        raise SystemExit('Official SAM 3 checkpoint checksum mismatch')
    checkpoint = directory/'weights/sam3.pt'
    checkpoint.parent.mkdir(exist_ok=True)
    if not checkpoint.exists() or digest(checkpoint) != expected:
        shutil.copyfile(downloaded, checkpoint)
    license_path = hf_hub_download('facebook/sam3', 'LICENSE', revision=WEIGHTS_REVISION)
    shutil.copyfile(license_path, directory/'LICENSE')
    spec = dict(model='sam3', path='weights/sam3.pt', sha256=expected,
                checkpoint_revision=WEIGHTS_REVISION, code_revision=CODE_REVISION,
                source_url=f'https://huggingface.co/facebook/sam3/blob/{WEIGHTS_REVISION}/sam3.pt',
                license='SAM License', cpu_adaptations=patches)
    (directory/'manifest.json').write_text(json.dumps(spec, indent=2)+'\n')
    print('Prepared official SAM 3 checkpoint and CPU provenance manifest.')


if __name__ == '__main__':
    main()
