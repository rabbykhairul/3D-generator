"""Render seams and explicit crease-aware normals, without moving geometry.

Closed construction meshes stay welded for validation. GLB shading needs separate
vertices where a hard normal or texture UV changes. Those coincident seams are
not physical holes; topology checks on exported geometry must weld them first.
"""
import math
import cv2
import numpy as np
from PIL import Image
import trimesh


def prepare_render_mesh(mesh, crease_degrees=55, back_texture=None, texture_faces=None):
    normals = mesh.face_normals
    faces = mesh.faces
    vertices = mesh.vertices
    adjacency = [[] for _ in vertices]
    incident = [[] for _ in vertices]
    for face_id, face in enumerate(faces):
        for vertex in face:
            incident[vertex].append(face_id)
    threshold = math.cos(math.radians(crease_degrees))
    for (a, b), edge in zip(mesh.face_adjacency, mesh.face_adjacency_edges):
        if np.dot(normals[a], normals[b]) >= threshold:
            for vertex in edge:
                adjacency[vertex].append((int(a), int(b)))

    corner_groups = {}
    group_normals = {}
    for vertex_id, face_ids in enumerate(incident):
        parent = {f: f for f in face_ids}
        def find(f):
            while parent[f] != f:
                parent[f] = parent[parent[f]]
                f = parent[f]
            return f
        for a, b in adjacency[vertex_id]:
            parent[find(a)] = find(b)
        sums = {}
        for face_id in face_ids:
            group = find(face_id)
            face = faces[face_id]
            others = vertices[face[face != vertex_id]] - vertices[vertex_id]
            cosine = np.dot(others[0], others[1]) / max(np.linalg.norm(others[0])*np.linalg.norm(others[1]), 1e-20)
            angle = math.acos(float(np.clip(cosine, -1, 1)))
            sums[group] = sums.get(group, np.zeros(3)) + normals[face_id] * angle
            corner_groups[vertex_id, face_id] = group
        for group, normal in sums.items():
            group_normals[vertex_id, group] = normal / max(np.linalg.norm(normal), 1e-20)

    visual = mesh.visual
    original_uv = getattr(visual, 'uv', None)
    material = visual.material.copy()
    wall_faces = set(mesh.metadata.get('wall_faces', []))
    back_faces = set(mesh.metadata.get('back_faces', []) if texture_faces is None else texture_faces)
    separate_back = back_texture is not None and bool(back_faces)
    tiles = 3 if separate_back else 2
    atlas = original_uv is not None and bool(wall_faces) and material.baseColorTexture is not None
    if atlas:
        # Caps retain the exact photographic map. Unobserved depth walls receive
        # only a low-frequency local color field, not stretched edge highlights,
        # antialias noise or tiny printed marks. This is an inferred side finish.
        photo = np.array(material.baseColorTexture.convert('RGBA'))
        sigma = max(2., max(photo.shape[:2]) / 120.)
        smooth = cv2.GaussianBlur(photo, (0, 0), sigmaX=sigma, sigmaY=sigma)
        smooth[:, :, 3] = photo[:, :, 3]
        images = (photo, np.asarray(back_texture.convert('RGBA').resize((photo.shape[1], photo.shape[0]))), smooth) if separate_back else (photo, smooth)
        material.baseColorTexture = Image.fromarray(np.concatenate(images, axis=1))
    output_vertices, output_normals, output_uv, output_faces, source_vertices = [], [], [], [], []
    lookup = {}
    for face_id, face in enumerate(faces):
        wall = atlas and face_id in wall_faces
        tile = (tiles - 1) if wall else (1 if atlas and separate_back and face_id in back_faces else 0)
        output_face = []
        for vertex_id in face:
            group = corner_groups[vertex_id, face_id]
            key = (int(vertex_id), group, tile)
            if key not in lookup:
                lookup[key] = len(output_vertices)
                output_vertices.append(vertices[vertex_id])
                source_vertices.append(vertex_id)
                output_normals.append(group_normals[vertex_id, group])
                if original_uv is not None:
                    u, v = original_uv[vertex_id]
                    output_uv.append([(u + tile) / tiles if atlas else u, v])
            output_face.append(lookup[key])
        output_faces.append(output_face)
    result = trimesh.Trimesh(vertices=output_vertices, faces=output_faces,
                             vertex_normals=output_normals, process=False)
    result.visual = trimesh.visual.TextureVisuals(uv=output_uv if original_uv is not None else None, material=material)
    result.vertex_attributes['_CONSTRUCTION_VERTEX'] = np.asarray(source_vertices, dtype=np.float32)
    result.metadata = {key: value for key, value in mesh.metadata.items() if key not in {'wall_faces', 'back_faces'}}
    result.metadata['separate_inner_surface'] = bool(atlas and separate_back)
    result.metadata.update(render_crease_degrees=crease_degrees,
                           geometric_vertices=len(vertices), render_vertices=len(result.vertices),
                           depth_wall_appearance='local low-frequency photographic color' if atlas else 'material')
    return result


def welded_geometry(mesh):
    """Remove only coincident render seams for geometric closure checks."""
    original = mesh.vertex_attributes.get('_CONSTRUCTION_VERTEX')
    if original is not None:
        _, first, inverse = np.unique(np.asarray(original).ravel(), return_index=True, return_inverse=True)
        if not np.allclose(mesh.vertices, mesh.vertices[first][inverse], atol=1e-10, rtol=0):
            raise ValueError('A render seam moved away from its construction vertex')
        return trimesh.Trimesh(vertices=mesh.vertices[first], faces=inverse[mesh.faces], process=False)
    result = mesh.copy()
    result.merge_vertices(merge_tex=True, merge_norm=True)
    return result
