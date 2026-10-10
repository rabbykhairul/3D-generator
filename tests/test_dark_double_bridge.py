from copy import deepcopy
import cv2
import numpy as np
import pytest
from shapely.geometry import Polygon, Point

from backend.store import get_project
from backend.reconstruction.engine import extract_annotations, reconstruct
from backend.reconstruction.full_rim_recovery import recover_dark_double_bridge


@pytest.mark.parametrize('size', [540,1080,2160])
def test_dark_aviator_keeps_both_lenses_and_bridge(tmp_path, size):
    project = get_project('d6fec31607d84030849761a204c137db')
    if project is None:
        pytest.skip('Local sunglasses reference unavailable')
    photo = next(p for p in project['photos'] if p['view']=='front')
    source = cv2.imread(photo['path'])
    path = tmp_path/'front.png'
    cv2.imwrite(str(path),cv2.resize(source,(size,size),interpolation=cv2.INTER_AREA if size<1080 else cv2.INTER_CUBIC))
    result = extract_annotations({'id':'test','path':str(path),'view':'front'})
    assert result['segmentation']['method']=='stable-dark-double-bridge'
    assert {'nose-pad-left','nose-pad-right'} <= {a['part'] for a in result['annotations']}
    shapes = {a['part']:Polygon([(p['x']*1080/size,p['y']*1080/size) for p in a['points']]) for a in result['annotations'] if not a['part'].startswith('nose-pad-')}
    assert set(shapes)=={'front','lens-left','lens-right','bridge-opening'}
    assert all(p.is_valid for p in shapes.values())
    assert all(shapes['front'].contains(shapes[k]) for k in shapes if k!='front')
    assert all(shapes['front'].boundary.distance(shapes[k]) > 0 for k in shapes if k!='front')
    assert shapes['lens-left'].contains(Point(260,560))
    assert shapes['lens-right'].contains(Point(800,560))
    assert shapes['bridge-opening'].contains(Point(540,440))
    assert not shapes['front'].contains(Point(490,570))  # pad, not rim
    assert not shapes['front'].contains(Point(275,743))  # ground shadow
    assert not any(a['approved'] for a in result['annotations'])


def test_dark_blobs_without_measured_rims_are_not_glasses():
    image = np.full((1080,1080,3),255,np.uint8)
    for center in [(280,540),(800,540)]:
        cv2.ellipse(image,center,(190,160),0,0,360,(80,80,80),-1)
    assert recover_dark_double_bridge(image,{},1,1) is None


@pytest.mark.parametrize('custom',[False,True])
def test_dark_front_generates_closed_beveled_mesh(tmp_path,custom):
    project = get_project('d6fec31607d84030849761a204c137db')
    if project is None:
        pytest.skip('Local sunglasses reference unavailable')
    draft = deepcopy(project)
    photo = next(p for p in draft['photos'] if p['view']=='front')
    draft['photos'] = [photo]
    draft['annotations'] = extract_annotations(photo)['annotations']
    if custom:
        draft['material'].update(lens_color='#123456',lens_opacity=.95)
    for a in draft['annotations']:
        a['approved'] = True
    for name,value in {'frame_width':140,'lens_width':54,'lens_height':46,'bridge_width':18,'temple_length':140}.items():
        draft['measurements'][name] = {'value':value,'source':'diagnostic-assumption'}
    result = reconstruct(draft,tmp_path/'draft')
    assert result['checks']['front_parts_watertight']
    assert result['checks']['consistent_outward_normals']
    import trimesh
    scene = trimesh.load(tmp_path/'draft/model.glb',force='scene',process=False)
    assert {'nose_pad_left','nose_pad_right'} <= set(scene.geometry)
    for side in ('left','right'):
        material = scene.geometry[f'lens_{side}'].visual.material
        expected_alpha=round(result['provenance']['lens_material']['lens_opacity']*255)
        assert abs(int(material.baseColorFactor[3])-expected_alpha)<=1
        if custom:
            from backend.reconstruction.engine import _color
            assert material.baseColorFactor[:3].tolist()==_color('#123456')[:3].tolist()
        else:
            assert material.name=='photo-estimated-lens'
            assert max(material.baseColorFactor[:3]) < 40
    assert result['provenance']['lens_material']['method']==('project-settings' if custom else 'paired-interior-linear-white-composite')


@pytest.mark.parametrize('size',[540,1080,2160])
@pytest.mark.parametrize('mirrored',[False,True])
def test_pale_sunglass_arm_excludes_rear_branch(tmp_path,size,mirrored):
    project = get_project('d6fec31607d84030849761a204c137db')
    if project is None:
        pytest.skip('Local sunglasses reference unavailable')
    photo = next(p for p in project['photos'] if p['view']=='right')
    image = cv2.resize(cv2.imread(photo['path']),(size,size))
    if mirrored:
        image = cv2.flip(image,1)
    path = tmp_path/'side.png'
    cv2.imwrite(str(path),image)
    result = extract_annotations({'id':'side','path':str(path),'view':'left' if mirrored else 'right'})
    assert len(result['annotations'])==1
    ann = result['annotations'][0]
    poly = Polygon([(p['x']*1080/size,p['y']*1080/size) for p in ann['points']])
    def point(x,y):
        return Point(1080-1080/size-x if mirrored else x,y)
    assert poly.is_valid
    assert poly.contains(point(600,530))
    assert poly.contains(point(60,711))
    assert not poly.contains(point(300,650))  # rear tip
    assert not poly.contains(point(950,600))  # dark lens
    assert not poly.contains(point(80,745))  # ground shadow
