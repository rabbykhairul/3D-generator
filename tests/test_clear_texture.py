import numpy as np
from PIL import Image
from shapely.geometry import Polygon
from backend.reconstruction.engine import _frame_texture, _beveled_plate, _material


def test_clear_interior_is_not_replaced_by_dark_hardware(tmp_path):
    pixels=np.full((100,200,3),255,np.uint8)
    pixels[20:80,20:180]=[244,245,246]
    pixels[20:80,20:30]=[25,25,25]
    pixels[20:25,20:180]=[65,65,65]
    path=tmp_path/'clear.png';Image.fromarray(pixels).save(path)
    photo={'path':str(path),'width':200,'height':100}
    poly=Polygon([(20,20),(180,20),(180,80),(20,80)])
    clear=np.asarray(_frame_texture(photo,poly,preserve_clear=True))
    opaque=np.asarray(_frame_texture(photo,poly))
    assert np.array_equal(clear[40:60,60:160,:3],pixels[40:60,60:160])
    assert opaque[50,100,:3].mean()<150 # reproduces previous dark propagation
    assert clear[50,25,:3].mean()<40 # genuine hardware is preserved
    assert np.all(clear[:,:,3]==255) # material alpha controls transparency


def test_clear_material_and_alpha_survive_textured_bevel(tmp_path):
    path=tmp_path/'photo.png';Image.new('RGB',(100,100),'white').save(path)
    material=_material('#ffffff',.55,0,.35);material.name='clear-acetate'
    _,mesh=_beveled_plate(Polygon([(10,10),(90,10),(90,90),(10,90)]),
        2,.2,1,lambda p:[p[0]/1000,p[1]/1000,0],material,'frame',
        {'path':str(path),'width':100,'height':100})
    assert mesh.visual.material.name=='clear-acetate'
    assert mesh.visual.material.alphaMode=='BLEND'
    assert mesh.visual.material.baseColorFactor[3]==140


def test_thin_rim_does_not_erase_dark_edge_samples(tmp_path):
    pixels=np.full((100,100,3),255,np.uint8)
    pixels[20:24,20:81]=[210,190,175]
    pixels[20,20:81]=[35,35,35]
    path=tmp_path/'rim.png';Image.fromarray(pixels).save(path)
    result=np.asarray(_frame_texture({'path':str(path),'width':100,'height':100},
        Polygon([(20,20),(80,20),(80,23),(20,23)])))
    assert result[20,50,:3].mean()<50
