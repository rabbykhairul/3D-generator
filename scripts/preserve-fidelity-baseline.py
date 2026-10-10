"""Preserve the current project/asset without modifying the database."""
import json
import shutil
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend import store

project = store.get_project('d9cb85ea817b4a7d968e31135814cab4')
out = Path('artifacts/fidelity')
out.mkdir(parents=True, exist_ok=True)
snapshot = out / 'baseline-project.json'
if snapshot.exists():
    raise SystemExit('Baseline already preserved; refusing to overwrite')
snapshot.write_text(json.dumps(project, indent=2))
artifact = store.DATA / 'artifacts' / project['artifact']['job_id']
shutil.copy2(artifact / 'model.glb', out / 'baseline-model.glb')
shutil.copy2(artifact / 'metadata.json', out / 'baseline-metadata.json')
print('Preserved revision', project['revision'], project['artifact']['job_id'])
