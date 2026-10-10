import cv2
import numpy as np
from backend.reconstruction.lens_appearance import infer_dark_lens_material, measured_dark_edge_width


def test_lens_sampling_ignores_rear_arms_and_rejects_clear_lenses(tmp_path):
    outlines = {'left':np.array([[30,30],[180,30],[180,180],[30,180]]),
                'right':np.array([[220,30],[370,30],[370,180],[220,180]])}
    image = np.full((220,400,3),255,np.uint8)
    path=tmp_path/'lens.png'
    cv2.imwrite(str(path),image)
    assert infer_dark_lens_material({'path':str(path)},outlines) is None
    for points in outlines.values():
        cv2.fillPoly(image,[points],(110,120,125))
        cv2.line(image,tuple(points[0]+20),tuple(points[2]-20),(15,15,15),14)
    cv2.imwrite(str(path),image)
    result=infer_dark_lens_material({'path':str(path)},outlines)
    assert result and 0<result['lens_opacity']<1
    rgb=np.array([int(result['lens_color'][i:i+2],16) for i in (1,3,5)])/255
    linear=np.where(rgb<=.04045,rgb/12.92,((rgb+.055)/1.055)**2.4)
    composed=linear*result['lens_opacity']+1-result['lens_opacity']
    displayed=np.where(composed<=.0031308,composed*12.92,1.055*composed**(1/2.4)-.055)*255
    assert np.max(np.abs(displayed-[125,120,110]))<2


def test_dark_optical_border_must_have_measured_contrast(tmp_path):
    image=np.full((1080,1080),255,np.uint8)
    outline=np.array([[200,300],[800,300],[800,750],[200,750]])
    cv2.fillPoly(image,[outline],110)
    path=tmp_path/'edge.png';cv2.imwrite(str(path),image)
    assert measured_dark_edge_width({'path':str(path)},outline) is None
    cv2.polylines(image,[outline],True,35,4)
    cv2.imwrite(str(path),image)
    assert 2 <= measured_dark_edge_width({'path':str(path)},outline) <= 5
