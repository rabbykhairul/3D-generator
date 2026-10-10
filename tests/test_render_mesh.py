import numpy as np
import json
import struct
import trimesh
from PIL import Image

from backend.reconstruction.render_mesh import prepare_render_mesh, welded_geometry


def test_glb_exports_explicit_normals_and_preserves_construction_seams(tmp_path):
    source = trimesh.creation.cylinder(radius=.02, height=.005, sections=64)
    source.visual = trimesh.visual.TextureVisuals(material=trimesh.visual.material.PBRMaterial())
    mesh = prepare_render_mesh(source)
    path = tmp_path / 'normals.glb'
    trimesh.Scene(mesh).export(path, include_normals=True)
    payload = path.read_bytes()
    length = struct.unpack_from('<I', payload, 12)[0]
    header = json.loads(payload[20:20+length])
    for item in header['meshes']:
        for primitive in item['primitives']:
            assert 'NORMAL' in primitive['attributes']
            assert '_CONSTRUCTION_VERTEX' in primitive['attributes']
    loaded = next(iter(trimesh.load(path, force='scene').geometry.values()))
    assert np.allclose(np.linalg.norm(loaded.vertex_normals, axis=1), 1)
    assert welded_geometry(loaded).is_watertight
    assert np.allclose(loaded.bounds, source.bounds)


def test_sharp_corners_have_independent_normals_without_moving_vertices():
    original = trimesh.creation.box(extents=(.02, .03, .04))
    original.visual = trimesh.visual.TextureVisuals(material=trimesh.visual.material.PBRMaterial())
    mesh = prepare_render_mesh(original)
    assert len(mesh.vertices) == 24  # three normal groups per physical corner
    assert len(welded_geometry(mesh).vertices) == 8
    assert welded_geometry(mesh).is_watertight
    assert np.allclose(mesh.bounds, original.bounds)
    assert np.isclose(mesh.volume, original.volume)
    assert np.allclose(np.abs(mesh.vertex_normals).max(axis=1), 1)


def test_round_wall_normals_are_smooth_but_cap_rim_stays_sharp():
    source = trimesh.creation.cylinder(radius=.02, height=.005, sections=64)
    source.visual = trimesh.visual.TextureVisuals(material=trimesh.visual.material.PBRMaterial())
    mesh = prepare_render_mesh(source)
    side = np.abs(mesh.vertex_normals[:, 2]) < .1
    radial = mesh.vertices[side, :2]
    radial /= np.linalg.norm(radial, axis=1)[:, None]
    assert np.allclose(mesh.vertex_normals[side, :2], radial, atol=.003)
    assert welded_geometry(mesh).is_watertight


def test_wall_texture_atlas_keeps_cap_photo_and_suppresses_pixel_bands():
    source = trimesh.creation.box()
    pixels = np.zeros((32, 32, 4), dtype=np.uint8)
    pixels[:] = [80, 100, 50, 255]
    pixels[:, ::2, :3] = [130, 150, 90]
    source.visual = trimesh.visual.TextureVisuals(uv=np.tile([.3, .5], (8, 1)),
        material=trimesh.visual.material.PBRMaterial(baseColorTexture=Image.fromarray(pixels)))
    source.metadata['wall_faces'] = [0, 1]
    mesh = prepare_render_mesh(source)
    atlas = np.asarray(mesh.visual.material.baseColorTexture)
    assert np.array_equal(atlas[:, :32], pixels)
    assert atlas[:, 32:, 0].std() < pixels[:, :, 0].std() * .1
    assert set(np.round(mesh.visual.uv[:, 0], 2)) == {.15, .65}
    assert welded_geometry(mesh).is_watertight


def test_separate_inner_outer_surfaces_survive_glb_reload(tmp_path):
    source = trimesh.creation.box()
    source.visual = trimesh.visual.TextureVisuals(uv=np.tile([.3, .5], (8, 1)),
        material=trimesh.visual.material.PBRMaterial(baseColorTexture=Image.new('RGBA', (32, 32), (30, 60, 90, 255))))
    source.metadata.update(wall_faces=[0, 1], back_faces=[2, 3])
    mesh = prepare_render_mesh(source, back_texture=Image.new('RGB', (32, 32), (100, 40, 20)))
    path = tmp_path / 'separate-surfaces.glb'
    trimesh.Scene(mesh).export(path, include_normals=True)
    loaded = next(iter(trimesh.load(path, force='scene').geometry.values()))
    atlas = np.asarray(loaded.visual.material.baseColorTexture)
    assert atlas.shape[1] == 96
    assert np.all(atlas[:, :32, :3] == [30, 60, 90])
    assert np.all(atlas[:, 32:64, :3] == [100, 40, 20])
    for face_id, expected in [(0, 2.3/3), (2, 1.3/3), (4, .3/3)]:
        assert np.allclose(loaded.visual.uv[loaded.faces[face_id], 0], expected)
    assert welded_geometry(loaded).is_watertight
    assert np.allclose(loaded.bounds, source.bounds)
