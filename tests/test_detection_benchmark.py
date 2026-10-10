from scripts.benchmark_detection import assess
import json
from pathlib import Path
import subprocess
import sys
import pytest


def test_release_quality_uses_independent_boundary_labels():
    points=[[10,10],[30,10],[30,30],[10,30]]
    ann={'part':'lens-left','points':points}
    assert assess({'annotations':[ann]},{'lens-left':points},100,100)['passed']
    shifted=[[x+5,y] for x,y in points]
    assert not assess({'annotations':[ann]},{'lens-left':shifted},100,100)['passed']
    assert not assess({'annotations':[]},{'lens-left':points},100,100)['passed']
    assert not assess({'annotations':[ann]}, {},100,100)['passed']


def test_invalid_or_duplicate_predictions_never_pass():
    points=[[10,10],[30,10],[30,30],[10,30]]
    ann={'part':'lens-left','points':points}
    assert not assess({'annotations':[ann,ann]},{'lens-left':points},100,100)['passed']
    crossed={'part':'lens-left','points':[[10,10],[30,30],[10,30],[30,10]]}
    assert not assess({'annotations':[crossed]},{'lens-left':points},100,100)['passed']
    with pytest.raises(ValueError):
        assess({'annotations':[ann]},{'lens-left':[[-1,10],[30,10],[30,30]]},100,100)


def test_overlapping_lenses_cannot_pass_even_when_individual_labels_match():
    points=[[10,10],[30,10],[30,30],[10,30]]
    anns=[{'part':part,'points':points} for part in ('lens-left','lens-right')]
    result=assess({'annotations':anns},{a['part']:points for a in anns},100,100)
    assert not result['passed']
    assert 'Lens boundaries overlap' in result['issues']


def test_cli_does_not_certify_an_unlabelled_reference(tmp_path):
    root=Path(__file__).resolve().parents[1]
    manifest=tmp_path/'manifest.json'
    manifest.write_text(json.dumps({'split':'development','cases':[{
        'sku':'rimless-reference','family':'rimless','view':'front',
        'path':str(root/'tests/fixtures/rimless/front.png'),'expected':{}}]}))
    output=tmp_path/'report.json'
    run=subprocess.run([sys.executable,str(root/'scripts/benchmark_detection.py'),
                        str(manifest),'--output',str(output),'--gate'],
                       capture_output=True,text=True,timeout=30)
    assert run.returncode==2,run.stderr
    report=json.loads(output.read_text())
    assert not report['release_gate_passed']
    assert report['families']['rimless']['pass_rate']==0
    assert report['cases'][0]['seconds']>0
    assert any('Unlabelled' in reason for reason in report['gate_failures'])
