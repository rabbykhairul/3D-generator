"""The merchant guide's real pale frame must stay a usable reviewed draft."""
from pathlib import Path

import trimesh
from shapely.geometry import Point, Polygon

from backend.reconstruction import extract_annotations, reconstruct


def test_real_guide_front_recovers_clean_reviewable_geometry(tmp_path):
    path = Path(__file__).resolve().parents[1] / 'frontend/public/upload-guide/front.png'
    photo = dict(id='guide-front', path=str(path), view='front', role='fitting', width=1080, height=1080)
    result = extract_annotations(photo)
    parts = {a['part']: a for a in result['annotations']}
    assert {'front','lens-left','lens-right'} <= parts.keys()
    assert result['segmentation']['method'] == 'stable-faint-front'
    assert all(not a['approved'] for a in result['annotations'])
    polygons = {name: Polygon([(p['x'],p['y']) for p in a['points']]) for name,a in parts.items()}
    front = polygons['front']
    left, right = polygons['lens-left'], polygons['lens-right']
    assert all(poly.is_valid for poly in polygons.values())
    assert front.contains(left) and front.contains(right) and not left.intersects(right)
    # These were the detached/attached shadow shelves in the first failed fallback.
    for point in [(200,733),(250,735),(800,736),(850,734)]:
        assert not front.covers(Point(*point))
    # Explicit test-only approval; the real API/UI never approves suggestions automatically.
    scale = 140 / (front.bounds[2]-front.bounds[0])
    measurements = {
        'frame_width': {'value':140,'source':'default'},
        'temple_length': {'value':140,'source':'default'},
        'lens_width': {'value':((left.bounds[2]-left.bounds[0])+(right.bounds[2]-right.bounds[0]))/2*scale,'source':'inferred'},
        'lens_height': {'value':((left.bounds[3]-left.bounds[1])+(right.bounds[3]-right.bounds[1]))/2*scale,'source':'inferred'},
        'bridge_width': {'value':(right.bounds[0]-left.bounds[2])*scale,'source':'inferred'},
    }
    snapshot = {'category':'transparent-full-rim','photos':[photo], 'measurements':measurements,
                'annotations':[dict(a,approved=True,source='merchant-reviewed:'+a['source']) for a in result['annotations']],
                'material':{'color':'#cec4da','opacity':.4,'lens_opacity':.1}}
    metadata = reconstruct(snapshot,tmp_path/'reviewed-draft')
    assert metadata['checks']['front_parts_watertight']
    assert metadata['checks']['consistent_outward_normals']
    scene = trimesh.load(tmp_path/'reviewed-draft/model.glb',force='scene')
    assert set(scene.geometry) == {'front_frame','lens_left','lens_right'}, 'No invented arms from a front-only photo'
