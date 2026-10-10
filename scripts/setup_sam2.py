"""Prepare publisher SAM 2.1 Small weights for local offline inference.

Install requirements.sam2.txt in the isolated environment first. This script
does not upload images, train a model or change project/approval records.
"""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
REVISION='2b90b9f5ceec907a1c18123530e92e794ad901a4'
URL='https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt'
SHA256='6d1aa6f30de5c92224f8172114de081d104bbd23dd9dc5c58996f0cad5dc4d38'


def digest(path):
    result=hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b''):
            result.update(chunk)
    return result.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'artifacts/sam2')
    args=parser.parse_args();directory=args.output.resolve()
    checkpoint=directory/'weights/sam2.1_hiera_small.pt'
    checkpoint.parent.mkdir(parents=True,exist_ok=True)
    if not checkpoint.exists():
        temporary=checkpoint.with_suffix('.download')
        try:
            with urllib.request.urlopen(URL,timeout=60) as response,temporary.open('wb') as handle:
                while chunk:=response.read(1024*1024):
                    handle.write(chunk)
            if digest(temporary)!=SHA256:
                raise ValueError('Publisher checkpoint checksum mismatch')
            temporary.replace(checkpoint)
        finally:
            temporary.unlink(missing_ok=True)
    if digest(checkpoint)!=SHA256:
        raise ValueError('Existing SAM checkpoint checksum mismatch')
    license_path=directory/'LICENSE'
    if not license_path.is_file():
        url=f'https://raw.githubusercontent.com/facebookresearch/sam2/{REVISION}/LICENSE'
        with urllib.request.urlopen(url,timeout=30) as response:
            license_path.write_bytes(response.read())
    spec=dict(model='sam2.1-hiera-small',config='configs/sam2.1/sam2.1_hiera_s.yaml',
        path=str(checkpoint.relative_to(directory)),sha256=SHA256,source_url=URL,
        code_revision=REVISION,license='Apache-2.0')
    (directory/'manifest.json').write_text(json.dumps(spec,indent=2)+'\n')
    print('Prepared local SAM 2.1 Small checkpoint and provenance manifest.')


if __name__=='__main__':
    main()
