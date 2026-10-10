from pathlib import Path

import cv2
import pytest
from shapely.geometry import Polygon, Point

from backend.reconstruction.engine import extract_annotations

REFERENCE=Path(__file__).parent/'fixtures/grey_double_bridge/front.png'


@pytest.mark.parametrize('width',[324,648,1298,2596])
@pytest.mark.parametrize('mirror',[False,True])
def test_grey_rim_dark_lenses_in_landscape_crop(tmp_path,width,mirror):
    original=cv2.imread(str(REFERENCE))
    height=round(original.shape[0]*width/original.shape[1])
    image=cv2.resize(original,(width,height),interpolation=cv2.INTER_AREA if width<1298 else cv2.INTER_CUBIC)
    if mirror:
        image=cv2.flip(image,1)
    path=tmp_path/'front.png'
    cv2.imwrite(str(path),image)
    result=extract_annotations({'id':'grey-unseen','view':'front','path':str(path)})
    shapes={a['part']:Polygon([(p['x']*1298/width,p['y']*618/height) for p in a['points']])
            for a in result['annotations']}
    assert {'front','lens-left','lens-right','bridge-opening','nose-pad-left','nose-pad-right'}<=shapes.keys()
    assert all(p.is_valid for p in shapes.values())
    assert Polygon(shapes['front'].exterior.coords,
                   [shapes[p].exterior.coords for p in ('lens-left','lens-right','bridge-opening')]).is_valid
    assert shapes['lens-left'].contains(Point(300,300))
    assert shapes['lens-right'].contains(Point(950,300))
    assert not shapes['front'].contains(Point(650,540)), 'Ground shadow is not frame geometry'
    assert not shapes['lens-left'].intersects(shapes['lens-right'])
    for pad in ('nose-pad-left','nose-pad-right'):
        assert shapes[pad].area<shapes['lens-left'].area*.04
    for shape in shapes.values():
        assert all(-1e-6<=x<=1298 and -1e-6<=y<=618 for x,y in shape.exterior.coords)
    assert not any(a['approved'] for a in result['annotations'])


def test_blank_margin_changes_do_not_change_the_object(tmp_path):
    original=cv2.imread(str(REFERENCE))
    baseline=extract_annotations({'id':'crop','view':'front','path':str(REFERENCE)})
    padded=cv2.copyMakeBorder(original,340,340,0,0,cv2.BORDER_CONSTANT,value=(254,254,254))
    path=tmp_path/'padded.png'
    cv2.imwrite(str(path),padded)
    result=extract_annotations({'id':'padded','view':'front','path':str(path)})
    expected={a['part']:Polygon([(p['x'],p['y']) for p in a['points']]) for a in baseline['annotations']}
    for ann in result['annotations']:
        shape=Polygon([(p['x'],p['y']-340) for p in ann['points']])
        truth=expected[ann['part']]
        assert shape.intersection(truth).area/shape.union(truth).area>.98
    assert {a['part'] for a in result['annotations']}==set(expected)


def test_detected_grey_frame_generates_closed_geometry_without_invented_arms(tmp_path):
    from backend.contracts import Material
    from backend.reconstruction.engine import reconstruct
    import trimesh
    photo={'id':'front','view':'front','role':'fitting','path':str(REFERENCE),
           'width':1298,'height':618}
    result=extract_annotations(photo)
    annotations=result['annotations']
    for ann in annotations:
        ann['approved']=True  # Isolated diagnostic only; no saved user project.
    project={'id':'grey-fixture','revision':1,'category':result['construction']['category'],'photos':[photo],
             'annotations':annotations,'material':Material().model_dump(),
             'measurements':{k:{'value':v,'source':'diagnostic-assumption'} for k,v in
                 {'frame_width':140,'lens_width':54,'lens_height':42,
                  'bridge_width':18,'temple_length':140}.items()}}
    metadata=reconstruct(project,tmp_path/'model')
    assert metadata['checks']['anatomy_geometry_consistent']
    assert metadata['anatomy']['schema_version']==1
    nodes={n['id']:n for n in metadata['anatomy']['nodes']}
    assert nodes['nose-pad-left']['state']=='observed'
    assert nodes['temple-left']['state']=='uncertain'
    assert metadata['checks']['front_parts_watertight']
    scene=trimesh.load(tmp_path/'model/model.glb',force='scene',process=False)
    assert {'front_frame','lens_left','lens_right','nose_pad_left','nose_pad_right'}<=set(scene.geometry)
    assert not any(name.startswith('temple') for name in scene.geometry)
