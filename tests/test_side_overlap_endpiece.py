from pathlib import Path
import cv2
import numpy as np
import pytest
from shapely.geometry import Polygon,LineString
from backend.reconstruction import extract_annotations,reconstruct
from backend.reconstruction.side_extraction import _repair_shaft_before_sleeve
from backend.contracts import Material

FIXTURES=Path(__file__).parent/'fixtures/semi_rimless'


@pytest.mark.parametrize('size',[540,1080,2160])
@pytest.mark.parametrize('mirror',[False,True])
def test_rear_strip_excluded_but_sleeve_and_endpiece_preserved(tmp_path,size,mirror):
    image=cv2.resize(cv2.imread(str(FIXTURES/'side.png')),(size,size))
    if mirror: image=cv2.flip(image,1)
    path=tmp_path/'side.png';cv2.imwrite(str(path),image)
    ann=extract_annotations(dict(id='side',view='right',path=str(path)))['annotations'][0]
    scale=1080/size
    def xy(x,y):return (1080-scale-x*scale if mirror else x*scale,y*scale)
    poly=Polygon([xy(p['x'],p['y']) for p in ann['points']])
    def section(x):return poly.intersection(LineString([(x,0),(x,1080)])).length
    for x in (440,460,480,500):assert 7<section(x)<15
    assert 18<section(300)<26  # Wider genuine sleeve is retained.
    assert 22<section(960)<38  # Visible endpiece, not a missing/generic connector.
    assert 'frame-attachment' in ann['anchors']


def test_single_arm_or_no_sleeve_step_is_not_flattened():
    u=np.full(500,20.);l=np.full(500,30.)
    u[330:]=16.;l[270:380]+=6
    split=np.zeros(500,bool);below=np.ones(500,bool)
    _,same=_repair_shaft_before_sleeve(u,l,split,below)
    assert np.array_equal(same,l)
    split[380:]=True
    _,same=_repair_shaft_before_sleeve(np.full(500,20.),l,split,below)
    assert np.array_equal(same,l)


def test_visible_endpiece_survives_export_without_long_generic_hinge(tmp_path):
    import trimesh
    photos=[dict(id=view,view=view,role='fitting',path=str(FIXTURES/f'{name}.png'),width=1080,height=1080)
            for view,name in [('front','front'),('right','side')]]
    annotations=[]
    for photo in photos:annotations.extend(dict(a,approved=True) for a in extract_annotations(photo)['annotations'])
    project=dict(category='semi-rimless',photos=photos,annotations=annotations,material=Material().model_dump(),
                 measurements={k:dict(value=v,source='inferred') for k,v in
                 dict(frame_width=140,lens_width=58,lens_height=40,bridge_width=22,temple_length=140).items()})
    meta=reconstruct(project,tmp_path/'export')
    joint=meta['fit']['joints']['right']
    assert 8<joint['visible_endpiece_depth_mm']<15
    assert joint['endpiece_depth_measured'] is False
    geometry=trimesh.load(tmp_path/'export/model.glb',force='scene').geometry
    assert np.ptp(geometry['hinge_right'].vertices[:,2])<.003
    arm=geometry['temple_right'].copy()
    arm.merge_vertices(merge_tex=True,merge_norm=True)  # GLB splits UV/normal seams.
    assert arm.is_watertight
    assert meta['checks']['anatomy_geometry_consistent']
