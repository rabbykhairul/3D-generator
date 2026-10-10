from copy import deepcopy
import cv2
import numpy as np
import pytest
from shapely.geometry import Polygon,Point
from backend.store import get_project
from backend.reconstruction.engine import extract_annotations,reconstruct
from backend.reconstruction.side_extraction import _runs


def reference():
    project=get_project('8758d7b8ecf343c4b5f321221757954d')
    if project is None:
        pytest.skip('Local silver sunglasses reference unavailable')
    return project


@pytest.mark.parametrize('size',[540,1080,2160])
@pytest.mark.parametrize('mirror',[False,True])
def test_single_bridge_and_narrow_arm(tmp_path,size,mirror):
    project=reference()
    for view in ('front','right'):
        photo=next(p for p in project['photos'] if p['view']==view)
        im=cv2.resize(cv2.imread(photo['path']),(size,size),interpolation=cv2.INTER_AREA if size<1080 else cv2.INTER_CUBIC)
        if mirror:im=cv2.flip(im,1)
        path=tmp_path/f'{view}.png';cv2.imwrite(str(path),im)
        result=extract_annotations({'id':view,'view':'left' if view=='right' and mirror else view,'path':str(path)})
        shapes={a['part']:Polygon([(p['x']*1080/size,p['y']*1080/size)for p in a['points']])for a in result['annotations']}
        assert all(p.is_valid for p in shapes.values())
        if view=='front':
            assert {'front','lens-left','lens-right'}<=shapes.keys()
            assert 'bridge-opening' not in shapes
            assert shapes['front'].bounds[0]<100 and shapes['front'].bounds[2]>980
            assert all(shapes['front'].contains(shapes[k]) for k in ('lens-left','lens-right'))
            for key in ('lens-left','lens-right'):
                lens=shapes[key];x0,y0,x1,y1=lens.bounds
                lower=[Point(x,y) for x,y in shapes['front'].exterior.coords
                       if x0<x<x1 and y>y0+(y1-y0)*.7]
                assert lower and max(lens.boundary.distance(p) for p in lower)<12
        else:
            assert len(shapes)==1
            shape=next(iter(shapes.values()))
            def point(x,y):return Point(1080-1080/size-x if mirror else x,y)
            assert shape.contains(point(600,540))
            assert shape.contains(point(663,537)), 'Reflection notch must not cut the shaft'
            assert not shape.contains(point(320,670))
            assert not shape.contains(point(950,640))
            assert abs(result['annotations'][0]['anchors']['tip']['x']*1080/size-(1038 if mirror else 41))<20


def test_narrow_run_survives_wider_overlapping_branch():
    column=np.zeros(200,np.uint8);column[20:30]=1;column[90:150]=1
    assert len(_runs(column,.08))==2


def test_off_center_dark_blob_is_not_a_complete_frame(tmp_path):
    image=np.full((540,1080,3),255,np.uint8)
    cv2.ellipse(image,(270,270),(180,150),0,0,360,(80,80,80),-1)
    path=tmp_path/'partial.png';cv2.imwrite(str(path),image)
    result=extract_annotations({'id':'partial','view':'front','path':str(path)})
    assert not any(a['part']=='front' for a in result['annotations'])


def test_single_bridge_reconstructs_with_arm(tmp_path):
    draft=deepcopy(reference())
    draft['photos']=[p for p in draft['photos'] if p['view'] in ('front','right')]
    draft['annotations']=[a for p in draft['photos'] for a in extract_annotations(p)['annotations']]
    for a in draft['annotations']:a['approved']=True
    for key,value in {'frame_width':140,'lens_width':54,'lens_height':46,'bridge_width':18,'temple_length':140}.items():
        draft['measurements'][key]={'value':value,'source':'diagnostic-assumption'}
    result=reconstruct(draft,tmp_path/'model')
    assert result['checks']['front_parts_watertight']
    assert 'temple_right' in result['parts']
    assert result['bridge_openings']==0
