"""CPU wire detection: varied sources, conservative rejection, no AI routing."""
from pathlib import Path
import cv2
import numpy as np
import pytest
from PIL import Image, ImageDraw
from shapely.geometry import Polygon
from backend.reconstruction.cpu_metal import detect_cpu_metal
from backend.reconstruction.engine import extract_annotations
from backend.reconstruction.detection_report import detection_report


def photo(tmp_path,color='#d2b978',width=800,rim=6,bridge=True,bar=True):
    image=Image.new('RGB',(800,500),'white');draw=ImageDraw.Draw(image)
    for bounds in [(80,160,350,420),(450,160,720,420)]:
        draw.ellipse((bounds[0]-rim,bounds[1]-rim,bounds[2]+rim,bounds[3]+rim),fill=color)
        draw.ellipse(bounds,fill='#657779')
    if bridge:draw.line([(347,285),(365,266),(400,256),(435,266),(453,285)],fill=color,width=5)
    if bar:draw.line([(215,156),(585,156)],fill=color,width=5)
    image=image.resize((width,round(width*500/800)),Image.Resampling.LANCZOS)
    path=tmp_path/'source.png';image.save(path)
    return dict(id='front',view='front',path=str(path),width=image.width,height=image.height)


@pytest.mark.parametrize('color,width',[('#d2b978',800),('#cccccc',800),('#d2b978',1200)])
def test_wire_paths_and_lens_coordinates_follow_source(tmp_path,color,width):
    p=photo(tmp_path,color,width);result=detect_cpu_metal(p)
    assert result is not None
    by_part={a['part']:a for a in result['annotations']}
    assert {'front','lens-left','lens-right','bridge-hardware','front-support'}<=set(by_part)
    for a in by_part.values():
        assert a['approved'] is False
        poly=Polygon([(pt['x'],pt['y']) for pt in a['points']])
        assert poly.is_valid and poly.area>0
        assert 0<=poly.bounds[0]<poly.bounds[2]<p['width']
        assert 0<=poly.bounds[1]<poly.bounds[3]<p['height']
    assert result['segmentation']['bridge']['source_supported_column_fraction']>.86
    assert result['segmentation']['learned_model_active'] is False
    left=Polygon([(pt['x'],pt['y']) for pt in by_part['lens-left']['points']])
    assert left.bounds[0]==pytest.approx(80*width/800,abs=3)


@pytest.mark.parametrize('rim,bridge',[(0,False),(0,True),(22,True),(6,False)])
def test_missing_or_broad_support_is_not_fabricated(tmp_path,rim,bridge):
    assert detect_cpu_metal(photo(tmp_path,rim=rim,bridge=bridge,bar=False)) is None


@pytest.mark.parametrize('kind',['blank','rectangles','shadow'])
def test_unrelated_background_is_rejected(tmp_path,kind):
    p=photo(tmp_path);image=Image.new('RGB',(800,500),'white');draw=ImageDraw.Draw(image)
    if kind=='rectangles':
        draw.rectangle((80,160,350,420),fill='#657779');draw.rectangle((450,160,720,420),fill='#657779')
    if kind=='shadow':
        image=Image.open(p['path']);arr=np.asarray(image).copy()
        cv2.ellipse(arr,(400,440),(300,18),0,0,360,(225,225,225),-1)
        image=Image.fromarray(arr)
        p['path']=str(tmp_path/'shadow.png');image.save(p['path'])
        result=detect_cpu_metal(p)
        assert result is not None
        outer=next(a for a in result['annotations'] if a['part']=='front')
        assert max(pt['y'] for pt in outer['points'])<440
        return
    image.save(p['path']);assert detect_cpu_metal(p) is None


def test_explicit_cpu_selection_ignores_sam_default_and_reports_truthfully(tmp_path,monkeypatch):
    p=photo(tmp_path);monkeypatch.setenv('EYEWEAR_DETECTION_BACKEND','sam3')
    monkeypatch.setattr('backend.reconstruction.hybrid_detection.apply_hybrid',lambda *a,**k:pytest.fail('CPU selected: AI must not run'))
    result=extract_annotations(p,backend='cpu-metal');report=detection_report(p,result)
    assert result['cpu_metal_detection']['applied']
    assert report['recognizer']['backend']=='cpu-thin-metal'
    assert not report['recognizer']['learned_model_active']
    assert 'no AI inference' in report['recognizer']['reason']
    Image.new('RGB',(800,500),'white').save(p['path'])
    result=extract_annotations(p,backend='cpu-metal');report=detection_report(p,result)
    assert not result['cpu_metal_detection']['applied']
    assert not report['recognizer']['experimental']
    assert 'standard CPU contours retained' in report['recognizer']['reason']


def test_approved_source_bridge_is_preserved_in_serialized_export(tmp_path):
    import trimesh
    from test_reconstruction import snapshot,rectangle
    from backend.reconstruction.engine import reconstruct
    from backend.reconstruction.render_mesh import welded_geometry
    data=snapshot(tmp_path,'thin-metal')
    data['annotations'].append(dict(photo_id='front',part='bridge-hardware',approved=True,
        source='automatic-cpu-metal-source-v1',points=rectangle(185,61,215,66)))
    result=reconstruct(data,tmp_path/'export')
    assert result['fit']['bridge']['method']=='approved photographed bridge profile; depth inferred'
    mesh=trimesh.load(tmp_path/'export/model.glb',force='scene',process=False).geometry['metal_bridge']
    assert np.isfinite(mesh.vertices).all() and welded_geometry(mesh).is_watertight
    assert mesh.extents[1]*1000==pytest.approx(2,abs=.05)

from test_api import client

def test_cpu_backend_is_available_through_api_without_saving_approvals(client,tmp_path):
    p=photo(tmp_path)
    project=client.post('/api/projects',json={'name':'CPU pilot','category':'thin-metal'}).json()
    url=f"/api/projects/{project['id']}"
    uploaded=client.post(url+'/photos',data={'view':'front','role':'fitting'},
        files={'file':('source.png',Path(p['path']).read_bytes(),'image/png')})
    assert uploaded.status_code==200
    current=uploaded.json();front=current['photos'][0]
    response=client.post(url+'/photos/'+front['id']+'/extract?backend=cpu-metal')
    assert response.status_code==200
    result=response.json()
    assert result['cpu_metal_detection']['applied']
    assert result['detection_report']['recognizer']['backend']=='cpu-thin-metal'
    assert result['detection_report']['engine_version']>=45
    assert all(not a['approved'] for a in result['annotations'])
    after=client.get(url).json()
    assert after['annotations']==current['annotations']
    assert after['revision']==current['revision']

@pytest.mark.parametrize('color,reflection',[('#d2b978',False),('#cccccc',False),('#d2b978',True)])
def test_clear_lenses_use_source_edges_without_requiring_tint(tmp_path,color,reflection):
    p=photo(tmp_path,color=color)
    image=Image.open(p['path']).convert('RGB');pixels=np.asarray(image).copy()
    pixels[np.all(pixels==[101,119,121],axis=2)]=255
    if reflection:cv2.ellipse(pixels,(225,260),(60,20),-20,0,360,(232,232,232),-1)
    Image.fromarray(pixels).save(p['path'])
    result=detect_cpu_metal(p)
    assert result is not None
    assert result['segmentation']['optical_method']=='stable-clear-aperture-edges'
    assert {'front','lens-left','lens-right','bridge-hardware'}<={a['part'] for a in result['annotations']}


def test_broken_clear_rim_is_not_closed_with_an_invented_edge(tmp_path):
    p=photo(tmp_path);pixels=np.asarray(Image.open(p['path']).convert('RGB')).copy()
    pixels[np.all(pixels==[101,119,121],axis=2)]=255
    pixels[330:375,65:105]=255
    Image.fromarray(pixels).save(p['path'])
    assert detect_cpu_metal(p) is None
