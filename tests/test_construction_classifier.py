import cv2
import numpy as np
from backend.reconstruction.construction_classifier import suggest_construction


def annotation(part, points):
    return {'part':part,'points':[{'x':x,'y':y} for x,y in points]}


def test_missing_boundaries_are_not_classified_as_rimless():
    result=suggest_construction(np.full((200,400,3),255,np.uint8),[])
    assert result['category'] is None and result['confidence']=='low'


def test_full_rim_suggestions_explain_appearance_uncertainty():
    outer=[(30,40),(370,40),(370,160),(30,160)]
    left=[(50,60),(180,60),(180,140),(50,140)]
    right=[(220,60),(350,60),(350,140),(220,140)]
    annotations=[annotation('front',outer),annotation('lens-left',left),annotation('lens-right',right)]
    for color,expected in [(40,'opaque-full-rim'),(205,'transparent-full-rim')]:
        image=np.full((200,400,3),255,np.uint8)
        cv2.fillPoly(image,[np.asarray(outer)],(color,color,color))
        for hole in [left,right]: cv2.fillPoly(image,[np.asarray(hole)],(255,255,255))
        result=suggest_construction(image,annotations)
        assert result['category']==expected
        assert result['confidence']=='medium' and result['alternatives']
