"""CPU-only labelled-product detection benchmark; never alters project data.

See CPU_PRODUCTION_READINESS.md for the manifest and release-gate contract.
Exit 2 with --gate when any family misses the requested quality targets.
"""
from __future__ import annotations

import argparse
import hashlib
from collections import defaultdict
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from shapely.geometry import Polygon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.reconstruction import extract_annotations  # noqa: E402


def holdout_failures(manifest, directory):
    """Detect product/image leakage against explicitly frozen development sets.

    This checks supplied provenance, not an unknowable third-party training set.
    SKU aliases/near-duplicate photographs still require dataset-owner review.
    """
    failures = []
    references = manifest.get('development_manifests', [])
    if not references:
        failures.append('Frozen development manifests are required for product-disjoint evaluation')
    seen_skus, seen_hashes = set(), set()
    for reference in references:
        try:
            path = (directory / reference).resolve()
            development = json.loads(path.read_text())
            for case in development['cases']:
                seen_skus.add(case['sku'])
                image = (path.parent / case['path']).resolve()
                seen_hashes.add(hashlib.sha256(image.read_bytes()).hexdigest())
        except (OSError, KeyError, ValueError) as exc:
            failures.append(f'Cannot verify development manifest: {type(exc).__name__}')
    if not manifest.get('required_families'):
        failures.append('Declare required construction families; an empty coverage gate is not valid')
    for case in manifest.get('cases', []):
        if case.get('label_source') != 'independent-human' or not case.get('label_reviewer'):
            failures.append(f"{case.get('sku', '?')}: independent label provenance is missing")
        if case.get('sku') in seen_skus:
            failures.append(f"{case['sku']}: product appears in development data")
        try:
            digest = hashlib.sha256((directory / case['path']).resolve().read_bytes()).hexdigest()
            if digest != case.get('sha256'):
                failures.append(f"{case['sku']}: source image hash missing or changed")
            if digest in seen_hashes:
                failures.append(f"{case['sku']}: source image appears in development data")
        except (OSError, KeyError):
            failures.append(f"{case.get('sku', '?')}: source image cannot be verified")
    return failures


def polygon(points):
    return Polygon([(p['x'],p['y']) if isinstance(p,dict) else p for p in points])


def assess(result, expected, width, height, min_iou=.9, max_boundary_fraction=.01):
    """Compare independent labels with source-space suggestions, not themselves."""
    predicted = {}
    issues = []
    for ann in result.get('annotations',[]):
        part=ann['part']
        shape=polygon(ann['points'])
        if part in predicted:
            issues.append(f'{part}: duplicate part')
        if shape.is_empty or not shape.is_valid or shape.area<=0:
            issues.append(f'{part}: invalid geometry')
            continue
        x0,y0,x1,y1=shape.bounds
        if x0<0 or y0<0 or x1>width or y1>height:
            issues.append(f'{part}: outside source image')
        predicted[part]=shape
    lenses=[predicted[p] for p in ('lens-left','lens-right') if p in predicted]
    if len(lenses)==2 and lenses[0].intersects(lenses[1]):
        issues.append('Lens boundaries overlap')
    if 'front' in predicted:
        holes=[p for name,p in predicted.items() if name.startswith('lens-') or name=='bridge-opening']
        if holes and not Polygon(predicted['front'].exterior.coords,
                                 [p.exterior.coords for p in holes]).is_valid:
            issues.append('Frame/aperture topology is invalid')
    measurements={}
    for part,points in expected.items():
        truth=polygon(points)
        if not truth.is_valid or truth.is_empty or truth.area<=0:
            raise ValueError(f'Invalid ground-truth polygon: {part}')
        x0,y0,x1,y1=truth.bounds
        if x0<0 or y0<0 or x1>width or y1>height:
            raise ValueError(f'Ground truth outside source image: {part}')
        if part not in predicted:
            issues.append(f'{part}: missing')
            measurements[part]={'detected':False}
            continue
        found=predicted[part]
        iou=truth.intersection(found).area/truth.union(found).area
        error=truth.boundary.hausdorff_distance(found.boundary)/max(width,height)
        measurements[part]={'detected':True,'iou':round(iou,5),
                            'boundary_error_fraction':round(error,5)}
        if iou<min_iou or error>max_boundary_fraction:
            issues.append(f'{part}: boundary differs from independent label')
    return {'passed': bool(expected) and not issues, 'labelled':bool(expected),
            'issues':issues,'parts':measurements}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--min-iou',type=float,default=.9)
    parser.add_argument('--max-boundary-fraction',type=float,default=.01)
    parser.add_argument('--min-family-pass-rate',type=float,default=.95)
    parser.add_argument('--min-products-per-family',type=int,default=20)
    parser.add_argument('--max-seconds',type=float,default=None)
    parser.add_argument('--gate',action='store_true')
    args=parser.parse_args()
    if not 0<args.min_iou<=1 or not 0<args.min_family_pass_rate<=1 or args.max_boundary_fraction<=0:
        parser.error('Quality thresholds must be positive, with rates at most 1')
    if args.max_seconds is not None and args.max_seconds<=0:
        parser.error('--max-seconds must be positive')
    if args.min_products_per_family<1:
        parser.error('--min-products-per-family must be positive')
    manifest=json.loads(args.manifest.read_text())
    cases=manifest['cases']
    if not cases:
        parser.error('Manifest must contain cases')
    rows=[]
    for case in cases:
        path=(args.manifest.parent/Path(case['path'])).resolve()
        started=time.perf_counter()
        row={'sku':case['sku'],'family':case['family'],'view':case['view'],'path':str(path)}
        try:
            image=cv2.imread(str(path))
            if image is None:
                raise ValueError('Cannot decode benchmark image')
            height,width=image.shape[:2]
            result=extract_annotations({'id':'benchmark','view':case['view'],'path':str(path)})
            row.update(assess(result,case.get('expected',{}),width,height,
                              args.min_iou,args.max_boundary_fraction))
            row['warnings']=result.get('warnings',[])
            row['source_size']=[width,height]
            row['background_calibration']=result.get('background_calibration',{})
        except Exception as exc:
            row.update(passed=False,labelled=bool(case.get('expected')),
                       issues=[f'{type(exc).__name__}: {exc}'],parts={})
        row['seconds']=round(time.perf_counter()-started,4)
        rows.append(row)
    grouped=defaultdict(list)
    for row in rows:
        grouped[row['family']].append(row)
    families={family:{'cases':len(group),'products':len({r['sku'] for r in group}),
                      'pass_rate':sum(r['passed'] for r in group)/len(group),
                      'product_pass_rate':sum(all(r['passed'] for r in group if r['sku']==sku)
                                              for sku in {r['sku'] for r in group})/len({r['sku'] for r in group}),
                      'p95_seconds':float(np.percentile([r['seconds'] for r in group],95))}
              for family,group in grouped.items()}
    failures=holdout_failures(manifest,args.manifest.parent)
    if any(not r['labelled'] for r in rows):
        failures.append('Unlabelled cases cannot establish detection accuracy')
    if manifest.get('split')!='unseen-products':
        failures.append('Manifest must declare an independently maintained unseen-products split')
    for family in manifest.get('required_families',[]):
        if family not in families:
            failures.append(f'{family}: no benchmark cases')
    for family,stats in families.items():
        if stats['products']<args.min_products_per_family:
            failures.append(f'{family}: too few distinct products')
        if stats['product_pass_rate']<args.min_family_pass_rate:
            failures.append(f'{family}: product pass rate below target')
        if args.max_seconds is not None and stats['p95_seconds']>args.max_seconds:
            failures.append(f'{family}: CPU latency above target')
    report={'scope':'Detection only; not a material, hidden-depth or rendered-model fidelity audit',
            'cpu_only':True,'split':manifest.get('split'),'release_gate_passed':not failures,
            'gate_failures':failures,'targets':{'min_iou':args.min_iou,
            'max_boundary_fraction':args.max_boundary_fraction,
            'min_family_pass_rate':args.min_family_pass_rate,
            'min_products_per_family':args.min_products_per_family,'max_seconds':args.max_seconds},
            'families':families,'cases':rows}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'families':families,'gate_failures':failures,'report':str(args.output)},indent=2))
    return 2 if args.gate and failures else 0


if __name__=='__main__':
    raise SystemExit(main())
