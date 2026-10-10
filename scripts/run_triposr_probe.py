"""Run the official TripoSR on one eyewear photo as an isolated GLB baseline.

Requires an unmodified checkout of VAST-AI-Research/TripoSR and the official
stabilityai/TripoSR config.yaml + model.ckpt. This is not the app generator.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPO = ROOT / 'artifacts/triposr-src'
DEFAULT_WEIGHTS = ROOT / 'artifacts/triposr-weights'
DEFAULT_PYTHON = ROOT / 'artifacts/triposr-venv/bin/python'
DEFAULT_OUTPUT = ROOT / 'artifacts/triposr-baseline'


def default_photo():
    manifest = ROOT / 'artifacts/eyewear-training/lenskart-research/collection.json'
    if not manifest.is_file():
        return None
    for product in json.loads(manifest.read_text()).get('products', []):
        if 'vc-e13785' in product.get('page_url', '') and product.get('photos'):
            return Path(product['photos'][0]['path'])
    return None


def preflight(args):
    paths = {'photo': args.image, 'official_runner': args.triposr_dir / 'run.py',
             'model_config': args.weights / 'config.yaml',
             'model_checkpoint': args.weights / 'model.ckpt',
             'python': args.python}
    missing = [name for name, path in paths.items() if path is None or not path.is_file()]
    return {'paths': {name: str(path) if path else None for name, path in paths.items()},
            'missing': missing, 'device': 'cpu', 'mesh_resolution': args.mc_resolution}


def mesh_report(path):
    import trimesh
    scene = trimesh.load(path, force='scene', process=False)
    meshes = list(scene.geometry.values())
    if not meshes or not any(len(mesh.faces) for mesh in meshes):
        raise ValueError('TripoSR exported an empty mesh')
    return {'mesh_count': len(meshes),
            'vertices': sum(len(mesh.vertices) for mesh in meshes),
            'faces': sum(len(mesh.faces) for mesh in meshes),
            'watertight_meshes': sum(bool(mesh.is_watertight) for mesh in meshes),
            'file_bytes': path.stat().st_size,
            'note': 'Mesh validity is not eyewear shape, component or fitting accuracy.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', type=Path, default=default_photo())
    parser.add_argument('--triposr-dir', type=Path, default=DEFAULT_REPO)
    parser.add_argument('--weights', type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument('--python', type=Path, default=DEFAULT_PYTHON)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--mc-resolution', type=int, default=128,
                        help='CPU-friendly exploratory grid; official default is 256')
    args = parser.parse_args()
    if not 32 <= args.mc_resolution <= 256:
        parser.error('Mesh resolution must be 32–256')
    state = preflight(args)
    args.output.mkdir(parents=True, exist_ok=True)
    report_path = args.output / 'report.json'
    if state['missing']:
        report_path.write_text(json.dumps({'status': 'blocked', **state}, indent=2) + '\n')
        print(json.dumps({'status': 'blocked', 'missing': state['missing'],
                          'report': str(report_path)}, indent=2))
        return 2
    image = args.image.resolve()
    command = [str(args.python.resolve()), str((args.triposr_dir / 'run.py').resolve()),
               str(image), '--pretrained-model-name-or-path', str(args.weights.resolve()),
               '--device', 'cpu', '--chunk-size', '1024', '--mc-resolution',
               str(args.mc_resolution), '--model-save-format', 'glb',
               '--output-dir', str(args.output.resolve())]
    start = time.monotonic()
    with (args.output / 'runner.log').open('w') as log:
        result = subprocess.run(command, cwd=args.triposr_dir.resolve(), stdout=log,
                                stderr=subprocess.STDOUT, check=False)
    elapsed = round(time.monotonic() - start, 2)
    mesh = args.output / '0/mesh.glb'
    report = {'status': 'generated' if result.returncode == 0 and mesh.is_file() else 'failed',
              'source_image': str(image), 'elapsed_seconds': elapsed,
              'return_code': result.returncode, 'glb': str(mesh) if mesh.is_file() else None,
              'runner_log': str(args.output / 'runner.log'),
              'provenance': 'official TripoSR single-image CPU baseline',
              'fidelity_validated': False}
    if report['status'] == 'generated':
        try:
            report['mesh'] = mesh_report(mesh)
        except Exception as error:
            report.update(status='failed-validation', mesh_error=f'{type(error).__name__}: {error}')
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    return 0 if report['status'] == 'generated' else 1


if __name__ == '__main__':
    sys.exit(main())
