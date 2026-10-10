"""Verify the reported gray frame through an exported, reloaded GLB.

``--candidate`` reads the offline snapshot and metadata, without touching the
saved project's annotations or artifact pointer.
"""
import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np
import trimesh
from shapely.geometry import Point, Polygon

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend import store
from backend.reconstruction.render_mesh import welded_geometry


def projected_material(project, metadata, scene, pixel):
    annotations = {a['part']: a for a in project['annotations']}
    points = lambda part: np.array([[p['x'], p['y']] for p in annotations[part]['points']])
    front, left, right = [points(part) for part in ['front', 'lens-left', 'lens-right']]
    heights = np.array([np.ptp(left[:, 1]), np.ptp(right[:, 1])])
    sx = project['measurements']['frame_width']['value'] / np.ptp(front[:, 0])
    sy = project['measurements']['lens_height']['value'] * heights.sum() / (heights @ heights)
    cx = (left[:, 0].max() + right[:, 0].min()) / 2
    cy = np.mean([(a[:, 1].min() + a[:, 1].max()) / 2 for a in [left, right]])
    x_scale = metadata['fit'].get('assembled_width', {}).get('scale', 1)
    point = Point((pixel[0] - cx) * sx * x_scale / 1000, (cy - pixel[1]) * sy / 1000)
    return any(Polygon(triangle[:, :2]).covers(point) for triangle in scene.geometry['front_frame'].triangles)


def glb_manifest(path):
    payload = path.read_bytes()
    assert payload[:4] == b'glTF', 'Expected binary glTF export'
    length = struct.unpack_from('<I', payload, 12)[0]
    return json.loads(payload[20:20 + length])


def verify(candidate_dir=None):
    assembly = Path('artifacts/assembly-fix')
    before = json.loads((assembly / 'baseline-project.json').read_text())
    old_scene = trimesh.load(assembly / 'baseline-model.glb', force='scene')
    if candidate_dir is None:
        after = store.get_project(before['id'])
        metadata = after['artifact']['metadata']
        glb = store.DATA / 'artifacts' / after['artifact']['job_id'] / 'model.glb'
        result_path = Path('artifacts/surface-fix/saved-export-regression.json')
        source = after['artifact']['job_id']
    else:
        candidate_dir = Path(candidate_dir)
        after = json.loads((candidate_dir / 'snapshot.json').read_text())
        metadata = json.loads((candidate_dir / 'metadata.json').read_text())
        glb = candidate_dir / 'model.glb'
        result_path = candidate_dir.parent / 'candidate-export-regression.json'
        source = str(candidate_dir)
    new_scene = trimesh.load(glb, force='scene')
    manifest = glb_manifest(glb)

    # Original-photo shadow ledges must not become physical lower-rim material.
    shadow_points = [(390, 733), (210, 735)]
    for pixel in shadow_points:
        assert projected_material(before, before['artifact']['metadata'], old_scene, pixel), 'Baseline must reproduce shadow bug'
        assert not projected_material(after, metadata, new_scene, pixel), 'Shadow must not become solid geometry'
    assert projected_material(after, metadata, new_scene, (270, 730)), 'Preserve actual thin lower rim'
    assert projected_material(after, metadata, new_scene, (500, 420)), 'Preserve actual thin upper bridge bar'

    parts = new_scene.geometry
    assert set(parts) == {'front_frame', 'lens_left', 'lens_right', 'hinge_left', 'hinge_right',
                          'temple_left', 'temple_right'}
    assert metadata['checks']['front_parts_watertight']
    assert len(manifest['meshes']) == len(parts)
    for item in manifest['meshes']:
        for primitive in item['primitives']:
            assert 'NORMAL' in primitive['attributes'], 'GLB must carry explicit normals'
            assert '_CONSTRUCTION_VERTEX' in primitive['attributes'], 'GLB must carry seam weld IDs'
    assert manifest.get('images') and all('bufferView' in image and image.get('mimeType', '').startswith('image/')
                                          for image in manifest['images']), 'Textures must be embedded in GLB'

    max_normal_error = 0.0
    for name, mesh in parts.items():
        welded = welded_geometry(mesh)
        assert welded.is_watertight and welded.is_winding_consistent and welded.volume > 0, name
        assert np.isfinite(mesh.vertices).all() and np.isfinite(mesh.vertex_normals).all(), name
        error = float(np.max(np.abs(np.linalg.norm(mesh.vertex_normals, axis=1) - 1)))
        max_normal_error = max(max_normal_error, error)
        assert error < 1e-5, f'{name} has non-unit normals'
        assert np.isfinite(mesh.face_normals).all(), name
        if name in ('front_frame', 'temple_left', 'temple_right'):
            texture = mesh.visual.material.baseColorTexture
            assert texture is not None and np.isfinite(mesh.visual.uv).all(), name
            atlas = np.asarray(texture)
            assert atlas.shape[1] == 2 * atlas.shape[0], f'{name} is missing its photo/depth-wall atlas'
            assert (mesh.visual.uv[:, 0] < .5).any() and (mesh.visual.uv[:, 0] > .5).any(), name
        if name.startswith('lens_'):
            assert mesh.visual.material.alphaMode == 'BLEND'
            assert np.array_equal(mesh.visual.material.baseColorFactor,
                                  old_scene.geometry[name].visual.material.baseColorFactor)
            assert 0 < mesh.visual.material.baseColorFactor[3] < 255

    # The previous accepted model uses the same photos but older contour masks.
    # Compare only the photographic cap half; revised padding near the new
    # boundary may differ, while the inferred wall half has no old counterpart.
    accepted = trimesh.load(assembly / 'after-model.glb', force='scene')
    cap_color_difference = {}
    for name in ('front_frame', 'temple_left', 'temple_right'):
        previous = np.asarray(accepted.geometry[name].visual.material.baseColorTexture)[:, :, :3].astype(np.int16)
        atlas = np.asarray(parts[name].visual.material.baseColorTexture)
        assert atlas.shape[0] == previous.shape[0] and atlas.shape[1] == 2 * previous.shape[1]
        current = atlas[:, :previous.shape[1], :3].astype(np.int16)
        difference = np.abs(current - previous)
        median, p95 = float(np.median(difference)), float(np.percentile(difference, 95))
        assert median < 10 and p95 < 50, f'{name} photographic cap changed unexpectedly'
        cap_color_difference[name] = {'median_channel_error': median, 'p95_channel_error': p95}

    for side in ('left', 'right'):
        joint = metadata['fit']['joints'][side]
        assert joint['surface_part'] == 'front_frame' and joint['arm_root_contains_joint']
        hinge = parts[f'hinge_{side}'].bounds
        arm = parts[f'temple_{side}'].bounds
        assert hinge[0, 2] < arm[1, 2] < hinge[1, 2] + .001
        assert np.all(np.minimum(hinge[1], arm[1]) - np.maximum(hinge[0], arm[0]) > 0)

    result = dict(passed=True, source=source, glb=str(glb), shadow_pixels_removed=shadow_points,
                  lower_rim_and_upper_bar_retained=True, all_seven_parts_closed_outward_oriented=True,
                  explicit_unit_normals=True, max_normal_length_error=max_normal_error,
                  construction_seams_welded=True, embedded_photo_and_wall_atlases=True,
                  photographic_cap_color_vs_accepted=cap_color_difference,
                  lens_tint_and_alpha_unchanged=True, joint_surfaces_and_root_overlap=True,
                  limitations=['Checks use the supplied gray-frame photograph, not ground-truth 3D.',
                               'Actual hinge internals and exact dimensions are not established by these images.'])
    result_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', nargs='?', const='artifacts/surface-fix/candidate',
                        help='Use an offline candidate directory instead of the saved project artifact')
    args = parser.parse_args()
    verify(args.candidate)
