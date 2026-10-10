import cv2
import numpy as np
from backend.reconstruction.rimless_mount import visible_mounts
from backend.reconstruction.lens_appearance import infer_dark_lens_material, measured_clear_edge_width
from backend.reconstruction.multiview import _front_bounds


def test_mount_preserves_offset_profile_and_does_not_copy_rear_arm(tmp_path):
    image=np.full((400,800,3),255,np.uint8)
    lenses={'left':np.array([[100,100],[350,100],[350,300],[100,300]]),
            'right':np.array([[450,100],[700,100],[700,300],[450,300]])}
    # A deliberately asymmetric visible mounting leaf, with a rear arm
    # crossing behind the lens. It must not become a peg or an entire arm.
    cv2.rectangle(image,(76,152),(132,163),(65,65,65),-1)
    cv2.rectangle(image,(668,145),(724,156),(65,65,65),-1)
    cv2.line(image,(135,132),(210,255),(35,35,35),12)
    path=tmp_path/'mount.png';cv2.imwrite(str(path),image)
    mounts=visible_mounts({'id':'front','path':str(path)},lenses)
    assert set(mounts)=={'left','right'}
    assert 152<mounts['left']['anchor'][1]<164
    assert mounts['left']['polygon'].bounds[2]<150
    assert mounts['left']['polygon'].bounds[3]<175
    assert mounts['right']['anchor'][1]<mounts['left']['anchor'][1]
    assert not mounts['left']['depth_measured']
    # A white image and a broad vertical rear-arm silhouette supply no mount.
    image[:]=255;cv2.line(image,(100,115),(100,270),(35,35,35),15)
    cv2.imwrite(str(path),image)
    assert visible_mounts({'path':str(path)},lenses)=={}


def test_clear_lenses_do_not_inherit_blue_defaults_or_background_tint(tmp_path):
    image=np.full((220,400,3),255,np.uint8)
    outlines={'left':np.array([[30,30],[180,30],[180,180],[30,180]]),
              'right':np.array([[220,30],[370,30],[370,180],[220,180]])}
    for points in outlines.values():
        cv2.polylines(image,[points],True,(185,185,185),2)
        cv2.line(image,tuple(points[0]+20),tuple(points[2]-20),(15,15,15),12)
    path=tmp_path/'clear.png';cv2.imwrite(str(path),image)
    result=infer_dark_lens_material({'path':str(path)},outlines,include_clear=True)
    assert result['lens_color']=='#ffffff'
    assert result['lens_opacity']<.2
    # Scale the example to catalogue resolution so the measured narrow border
    # covers sufficient pixels; broad rear-arm obstruction does not tint it.
    large=cv2.resize(image,(2000,1100));cv2.imwrite(str(path),large)
    assert measured_clear_edge_width({'path':str(path)},outlines['left']*5) is not None
    large[:]=(255,220,220);cv2.imwrite(str(path),large)
    assert infer_dark_lens_material({'path':str(path)}, {k:v*5 for k,v in outlines.items()},include_clear=True) is None


def test_rimless_multiview_requires_both_approved_lenses():
    annotations=[{'part':part,'approved':True,'points':[{'x':x,'y':y} for x,y in points]}
        for part,points in [('lens-left',[(80,100),(250,100),(250,250),(80,250)]),
                            ('lens-right',[(350,100),(520,100),(520,250),(350,250)])]]
    assert _front_bounds(annotations,600,400)==(80,100,441,151)
    annotations[1]['approved']=False
    assert _front_bounds(annotations,600,400) is None
