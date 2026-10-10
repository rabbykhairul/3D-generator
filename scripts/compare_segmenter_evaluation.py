"""Render current contour proposals beside pretrained masks, without approvals."""
import argparse
import html
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.reconstruction import extract_annotations
from backend.reconstruction.detection_report import detection_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    directory = args.directory.resolve()
    manifest = json.loads((directory/'medium.json').read_text())
    small = json.loads((directory/'small/report.json').read_text())
    medium = json.loads((directory/'medium/report.json').read_text())
    output = directory/'comparison'
    output.mkdir(exist_ok=True)
    rows, cards = [], []
    for i, case in enumerate(manifest['cases']):
        with Image.open(case['path']) as source:
            image = source.convert('RGB')
        photo = dict(id=case['photo_id'], path=case['path'], view=case['view'],
                     role='fitting', width=image.width, height=image.height)
        started = time.perf_counter()
        result = extract_annotations(photo)
        seconds = time.perf_counter()-started
        report = detection_report(photo, result)
        overlay = image.copy()
        draw = ImageDraw.Draw(overlay)
        for annotation in result['annotations']:
            points = [(p['x'],p['y']) for p in annotation['points']]
            color = '#00ad49' if annotation['part'] == 'front' else '#ef7500'
            draw.line(points + points[:1], fill=color, width=max(2, image.width//400))
        baseline = Image.new('RGB', (1280, 350), 'white')
        draw = ImageDraw.Draw(baseline)
        for column, (label, panel) in enumerate([('Source',image),('Current contours',overlay)]):
            thumb = ImageOps.contain(panel,(640,310))
            baseline.paste(thumb,(column*640+(640-thumb.width)//2,30+(310-thumb.height)//2))
            draw.text((column*640+8,8),label,fill='black')
        sheet = Image.new('RGB',(1280,1090),'white')
        sheet.paste(baseline,(0,0))
        draw = ImageDraw.Draw(sheet)
        for offset, label, row, folder in [(370,'Small pretrained',small['cases'][i],'small'),
                                            (740,'Medium pretrained',medium['cases'][i],'medium')]:
            draw.text((8,offset-18),label,fill='black')
            if row.get('comparison'):
                with Image.open(directory/folder/row['comparison']) as panel:
                    sheet.paste(panel,(0,offset))
        name = f'{i:05d}.jpg'
        sheet.save(output/name,quality=92)
        rows.append(dict(name=case['name'],sku=case['sku'],view=case['view'],comparison=name,
                         baseline_seconds=round(seconds,4),baseline=report))
        cards.append(f'<article><h2>{html.escape(case["name"])} — {html.escape(case["view"])}</h2>'
                     f'<p>Current: {html.escape(report["status"])}. Masks are proposals, not verified geometry.</p>'
                     f'<img loading="lazy" src="{name}" alt="Current contours and small/medium pretrained masks"></article>')
    summary = dict(cases=rows, independent_labels=False, release_gate_passed=False,
                   baseline_median_seconds=float(np.median([r['baseline_seconds'] for r in rows])) if rows else None)
    (output/'report.json').write_text(json.dumps(summary,indent=2)+'\n')
    (output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Eyewear segmentation evaluation</title>'
        '<style>body{font:16px system-ui;margin:32px;max-width:1280px}img{width:100%}article{margin:40px 0;border-top:1px solid #ccc}</style>'
        '<h1>CPU eyewear segmentation comparison</h1><p>Unlabelled saved development photos. '
        'Red overlays show predicted pixels; green/orange lines show current contours. '
        'These examples do not establish accuracy on unseen products.</p>'+''.join(cards))
    print(json.dumps(dict(cases=len(rows),review=str(output/'index.html')),indent=2))


if __name__ == '__main__':
    main()
