from copy import deepcopy
import numpy as np
import pytest
from PIL import Image
from shapely.geometry import Polygon,Point,box
from test_reconstruction import snapshot,rectangle
from backend.reconstruction.engine import reconstruct,NeedsCorrection
from backend.reconstruction.outline_finish import finish_side_outlines


def arm_project(tmp_path,root_offset):
    data=snapshot(tmp_path,'opaque-full-rim')
    path=tmp_path/'side.png';Image.new('RGB',(400,160),'white').save(path)
    data['photos'].append(dict(id='left',view='left',role='fitting',width=400,height=160,path=str(path)))
    data['annotations'].append(dict(photo_id='left',part='temple-left',approved=True,
        source='merchant-reviewed-side:single-visible-arm-opaque-seam-v2',points=rectangle(10,60,300,68),
        anchors={'hinge':dict(x=10-root_offset,y=64),'tip':dict(x=300,y=64)}))
    return data

@pytest.mark.parametrize('offset',[0,.0274,.099])
def test_subpixel_root_seats_without_mutating_approved_data(tmp_path,offset):
    data=arm_project(tmp_path,offset);before=deepcopy(data)
    result=reconstruct(data,tmp_path/'model')
    assert data==before
    assert 'temple_left' in result['parts']
    import trimesh
    scene=trimesh.load(tmp_path/'model/model.glb',force='scene',process=False)
    assert all(np.isfinite(m.vertices).all() for m in scene.geometry.values())
    if offset:assert any('root seated' in w for w in result['warnings'])


def test_genuinely_misplaced_root_still_rejected_with_useful_details(tmp_path):
    with pytest.raises(NeedsCorrection) as exc:
        reconstruct(arm_project(tmp_path,2),tmp_path/'bad')
    assert 'hinge anchor does not meet' in str(exc.value)
    assert exc.value.details['part']=='temple-left'
    assert exc.value.details['distance_source_px']==pytest.approx(2)


def test_side_smoothing_cannot_move_root_cap_away_from_anchor(monkeypatch):
    annotation=dict(part='temple-left',approved=False,source='single-visible-arm-test',
        points=rectangle(10,60,300,68),anchors={'hinge':dict(x=10,y=64),'tip':dict(x=300,y=64)})
    before=deepcopy(annotation)
    monkeypatch.setattr('backend.reconstruction.outline_finish.smooth_detected_curve',lambda *a,**k:box(10.03,60,299.97,68))
    finish_side_outlines(np.full((160,400,3),255,np.uint8),[annotation])
    assert annotation==before
    assert Polygon([(p['x'],p['y']) for p in annotation['points']]).covers(Point(10,64))
