import io
import numpy as np
import pytest
import trimesh
from PIL import Image
from shapely.geometry import Polygon
from backend.reconstruction.engine import _color,_material,_frame_texture


@pytest.mark.parametrize('color',['#808080','#123456','#b5d9e5','#bc526a','#ffffff','#000000'])
def test_srgb_material_roundtrips_through_glb_without_brightening(color):
    mesh=trimesh.creation.box()
    mesh.visual=trimesh.visual.TextureVisuals(material=_material(color,.7,0,.35))
    scene=trimesh.load(io.BytesIO(mesh.export(file_type='glb')),file_type='glb',force='scene')
    material=next(iter(scene.geometry.values())).visual.material
    linear=material.baseColorFactor[:3]/255
    displayed=np.where(linear<=.0031308,linear*12.92,1.055*linear**(1/2.4)-.055)*255
    original=np.array([int(color[i:i+2],16) for i in (1,3,5)])
    assert np.max(np.abs(displayed-original))<=6  # 8-bit linear-factor quantization.
    assert abs(material.baseColorFactor[3]/255-.7)<1/255


def test_padding_preserves_observed_dark_edges_and_printed_colors(tmp_path):
    pixels=np.full((80,100,3),255,np.uint8)
    pixels[15:65,20:80]=[140,45,65]
    pixels[15:18,20:80]=[35,15,20]
    pixels[32:40,40:55]=[210,190,40]
    path=tmp_path/'photo.png';Image.fromarray(pixels).save(path)
    texture=np.asarray(_frame_texture(dict(path=str(path),width=100,height=80),
                                     Polygon([(20,15),(79,15),(79,64),(20,64)])))
    assert np.array_equal(texture[16,45,:3],pixels[16,45])
    assert np.array_equal(texture[35,48,:3],pixels[35,48])
    assert np.array_equal(texture[50,60,:3],pixels[50,60])


def test_transparent_category_does_not_override_requested_frame_opacity(tmp_path):
    from test_joint_regression import _snapshot
    from backend.reconstruction import reconstruct
    for opacity in (1.,.8,.4):
        project=_snapshot(tmp_path)
        project['category']='transparent-full-rim'
        project['material']=dict(color='#808080',opacity=opacity)
        folder=tmp_path/f'model-{opacity}'
        reconstruct(project,folder)
        material=trimesh.load(folder/'model.glb',force='scene').geometry['front_frame'].visual.material
        assert abs(material.baseColorFactor[3]/255-opacity)<=1/255
