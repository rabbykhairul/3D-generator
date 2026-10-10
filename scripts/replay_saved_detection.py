"""Replay saved fitting photos locally without altering projects or approvals.

Output is DEVELOPMENT evidence, never independent ground truth. Images remain
in the existing store and are not copied or sent to an external service.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend import store
from backend.reconstruction import extract_annotations, NeedsCorrection
from backend.reconstruction.detection_report import detection_report
from backend.reconstruction.anatomy import FRONT_VIEWS, SIDE_VIEWS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', action='append', help='Limit to project ID (repeatable)')
    parser.add_argument('--limit', type=int, default=10, help='Latest projects; 0 means all')
    parser.add_argument('--replay', action='store_true', help='Run current heuristic extraction')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.limit < 0:
        parser.error('--limit cannot be negative')
    if not (store.DATA / 'studio.sqlite3').exists():
        parser.error('No existing project store. Set EYEWEAR_DATA_DIR if needed.')
    projects = [p for p in store.list_projects() if not p.get('archived_at')
                and (not args.project or p['id'] in args.project)]
    if args.limit:
        projects = projects[:args.limit]
    cases, runs = [], []
    for project in projects:
        for photo in project['photos']:
            if photo.get('role') != 'fitting' or photo['view'] not in FRONT_VIEWS | SIDE_VIEWS:
                continue
            path = Path(photo['path']).resolve()
            case = dict(sku=project['id'], photo_id=photo['id'], family=project['category'],
                        family_label_source='project-selection-not-independent', view=photo['view'],
                        path=str(path), expected={})
            if path.is_file():
                case['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
            cases.append(case)
            if args.replay:
                start = time.perf_counter()
                try:
                    result = extract_annotations(photo)
                except NeedsCorrection as exc:
                    result = dict(annotations=[], warnings=[str(exc)])
                except Exception as exc:
                    runs.append(dict(photo_id=photo['id'], project_id=project['id'],
                                     status='error', error=f'{type(exc).__name__}: {exc}'))
                    continue
                runs.append(dict(project_id=project['id'], seconds=round(time.perf_counter()-start, 4),
                                 **detection_report(photo, result)))
    output = dict(split='development', label_provenance='unlabelled-saved-uploads',
                  independent_labels=False, release_gate_passed=False, cases=cases, runs=runs,
                  note='Saved merchant approvals and engine suggestions are not independent benchmark labels.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2)+'\n')
    counts = {s: sum(r['status'] == s for r in runs) for s in sorted({r['status'] for r in runs})}
    print(json.dumps(dict(photos=len(cases), replay_statuses=counts, output=str(args.output)), indent=2))


if __name__ == '__main__':
    main()
