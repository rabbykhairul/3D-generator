import cv2
import numpy as np
from PIL import Image
from backend.reconstruction.temple_texture import transfer_inner_tip
from backend.reconstruction.render_mesh import prepare_render_mesh
import trimesh


def test_partial_transfer_retains_print_without_replacing_whole_arm(tmp_path):
    source=np.full((100,220,3),255,np.uint8)
    cv2.rectangle(source,(20,60),(95,75),(25,40,30),-1)
    cv2.circle(source,(34,67),4,(255,255,255),-1)
    path=tmp_path/'source.png';cv2.imwrite(str(path),source)
    evidence={'points':[{'x':20,'y':60},{'x':95,'y':60},{'x':95,'y':75},{'x':20,'y':75}]}
    annotation={'points':[{'x':10,'y':40},{'x':190,'y':40},{'x':190,'y':55},{'x':10,'y':55}],
                'anchors':{'hinge':{'x':190,'y':47},'tip':{'x':10,'y':47}}}
    photo={'id':'source','path':str(path),'width':220,'height':100}
    base=Image.new('RGBA',(220,100),(20,30,20,255))
    transferred,info=transfer_inner_tip(photo,evidence,photo,annotation,base)
    assert info['applied'] and .1<info['observed_fraction']<.5
    result=np.asarray(transferred)
    assert np.count_nonzero(np.all(result[40:56,10:60,:3]>220,axis=2))>10
    assert np.array_equal(result[40:56,150:180],np.asarray(base)[40:56,150:180])
    assert np.asarray(base)[47,24,0]==20  # source outer texture not mutated


def test_inner_cap_texture_does_not_replace_outer_cap():
    mesh=trimesh.creation.box()
    mesh.visual=trimesh.visual.TextureVisuals(uv=np.full((len(mesh.vertices),2),.5),
        material=trimesh.visual.material.PBRMaterial(baseColorTexture=Image.new('RGBA',(8,8),'red')))
    mesh.metadata['wall_faces']=np.flatnonzero(abs(mesh.face_normals[:,0])<.5).tolist()
    inner=np.flatnonzero(mesh.face_normals[:,0]>.5).tolist()
    render=prepare_render_mesh(mesh,back_texture=Image.new('RGBA',(8,8),'blue'),texture_faces=inner)
    for i,face in enumerate(render.faces):
        tile=int(np.floor(render.visual.uv[face[0],0]*3))
        assert tile==(1 if i in inner else 2 if i in mesh.metadata['wall_faces'] else 0)
