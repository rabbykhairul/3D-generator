"""Deterministic GLB evidence renders without a browser/GPU/server.

This is a geometry/embedded-texture diagnostic, NOT an AR screenshot or a PBR
reference renderer. Both models share one camera; no per-model shape alignment.
Clear lenses are omitted deliberately so the actual rim outline is visible.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import trimesh


def unit(vector):
    vector = np.asarray(vector, dtype=float)
    return vector / max(np.linalg.norm(vector), 1e-12)


def camera_basis(eye, target):
    forward = unit(np.asarray(target) - eye)
    right = unit(np.cross(forward, [0, 1, 0]))
    up = unit(np.cross(right, forward))
    return np.array([right, up, forward])


def sample_texture(texture, uv):
    """Bilinear GLTF repeat sampling, with image rows opposite to UV V."""
    pixels = np.asarray(texture.convert('RGB')) if isinstance(texture, Image.Image) else texture
    height, width = pixels.shape[:2]
    xy = (uv % 1) * [width, height] - .5
    xy[..., 1] = height - 1 - xy[..., 1]
    low = np.floor(xy).astype(int)
    frac = xy - low
    x0, y0 = low[..., 0] % width, low[..., 1] % height
    x1, y1 = (x0 + 1) % width, (y0 + 1) % height
    fx, fy = frac[..., 0, None], frac[..., 1, None]
    return (pixels[y0, x0] * (1-fx) + pixels[y0, x1] * fx) * (1-fy) + (
        pixels[y1, x0] * (1-fx) + pixels[y1, x1] * fx) * fy


def raster_triangle(image, depth_buffer, screen, depth, colors=None, uv=None,
                    texture=None, normals=None, surface='clay', alpha=1., linear_blend=False):
    """Z-buffer with perspective-correct attributes, including smooth normals."""
    height, width = depth_buffer.shape
    lo = np.maximum(np.floor(screen.min(axis=0)).astype(int), [0, 0])
    hi = np.minimum(np.ceil(screen.max(axis=0)).astype(int), [width-1, height-1])
    if np.any(hi < lo) or np.any(depth <= .0001):
        return
    a, b, c = screen
    determinant = (b[1]-c[1])*(a[0]-c[0])+(c[0]-b[0])*(a[1]-c[1])
    if abs(determinant) < 1e-9:
        return
    yy, xx = np.mgrid[lo[1]:hi[1]+1, lo[0]:hi[0]+1]
    x, y = xx+.5, yy+.5
    w0 = ((b[1]-c[1])*(x-c[0])+(c[0]-b[0])*(y-c[1]))/determinant
    w1 = ((c[1]-a[1])*(x-c[0])+(a[0]-c[0])*(y-c[1]))/determinant
    weights = np.stack((w0, w1, 1-w0-w1), axis=-1)
    inverse_depth = (weights / depth).sum(axis=-1)
    z = 1 / np.maximum(inverse_depth, 1e-12)
    visible = (weights.min(axis=-1) >= -1e-8) & (z < depth_buffer[yy, xx])
    if not visible.any():
        return
    bary = weights[visible] / depth / inverse_depth[visible, None]
    if surface == 'silhouette':
        rgb = np.full((len(bary), 3), 30.)
    elif surface == 'photo' and texture is not None and uv is not None:
        # Deliberately unlit photograph colors, like the app's photo policy.
        rgb = sample_texture(texture, bary @ uv)
    elif surface == 'photo':
        rgb = np.broadcast_to(colors if colors is not None else [120, 120, 120], (len(bary), 3))
    else:
        normal = bary @ normals
        normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)
        illumination = .3 + .7 * np.clip(normal @ unit([-.4, .6, 1]), 0, 1)
        rgb = illumination[:, None] * [158, 164, 161]
    behind = image[yy[visible],xx[visible]]
    if linear_blend:
        def decode(value):
            value=value/255
            return np.where(value<=.04045,value/12.92,((value+.055)/1.055)**2.4)
        blended=decode(rgb)*alpha+decode(behind)*(1-alpha)
        rgb=np.where(blended<=.0031308,blended*12.92,1.055*blended**(1/2.4)-.055)*255
    else:
        rgb = rgb*alpha + behind*(1-alpha)
    image[yy[visible], xx[visible]] = np.clip(rgb, 0, 255).astype(np.uint8)
    depth_buffer[yy[visible], xx[visible]] = z[visible]


def render(model, eye, target, surface='clay', size=720, include_lenses=False):
    scene = trimesh.load(model, force='scene', process=False)
    image = np.full((size, size, 3), [244, 246, 245], dtype=np.uint8)
    depth_buffer = np.full((size, size), np.inf)
    basis = camera_basis(eye, target)
    focal = size / (2 * np.tan(np.deg2rad(32)/2))
    for node in sorted(scene.graph.nodes_geometry,key=lambda n:(str(n).startswith('lens_'),str(n))):
        transform, name = scene.graph[node]
        lens = name.startswith('lens_')
        if lens and not include_lenses:
            continue
        mesh = scene.geometry[name]
        vertices = trimesh.transform_points(mesh.vertices, transform)
        coords = (vertices - eye) @ basis.T
        screen = coords[:, :2] / np.maximum(coords[:, 2, None], .0001) * [focal, -focal] + size/2
        normals = mesh.vertex_normals @ np.linalg.inv(transform[:3, :3]) @ basis.T
        # Camera coordinate +Z points into scene; lighting assumes +Z at camera.
        normals[:, 2] *= -1
        material = getattr(mesh.visual, 'material', None)
        texture_image = getattr(material, 'baseColorTexture', None)
        texture = np.asarray(texture_image.convert('RGB')) if texture_image is not None else None
        color = getattr(material, 'baseColorFactor', None)
        color = np.asarray(color[:3], dtype=float) if color is not None else None
        if color is not None:
            linear=color/255
            color=np.where(linear<=.0031308,linear*12.92,1.055*linear**(1/2.4)-.055)*255
        uv = getattr(mesh.visual, 'uv', None)
        face_normals = mesh.face_normals @ np.linalg.inv(transform[:3,:3]) @ basis.T
        for index,face in enumerate(mesh.faces):
            if lens and face_normals[index,2] >= 0:
                continue
            raster_triangle(image, depth_buffer, screen[face], coords[face, 2],
                            color, uv[face] if uv is not None else None,
                            texture, normals[face], surface,
                            float(material.baseColorFactor[3])/255 if lens else 1.,
                            lens)
    return Image.fromarray(image)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('before', type=Path)
    parser.add_argument('after', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--size', type=int, default=720)
    args = parser.parse_args()
    if not 64 <= args.size <= 2048:
        parser.error('size must be between 64 and 2048')
    args.output.mkdir(parents=True, exist_ok=True)
    # Frame the combined baseline extent, not a hardcoded bridge-depth guess.
    # Reuse this exact target for candidates so all full-product views stay fair.
    target = trimesh.load(args.before, force='scene').bounds.mean(axis=0)
    cameras = {
        'front': (target + [0, 0, .37], target),
        'side': (target + [-.37, .012, .015], target),
        'threequarter': (target + [-.24, .065, .3], target),
        'hinge': (np.array([-.15, .035, .10]), np.array([-.065, .010, -.008])),
    }
    for angle, (eye, look_at) in cameras.items():
        for surface in ('photo', 'clay', 'silhouette'):
            pair = Image.new('RGB', (args.size*2, args.size+65), 'white')
            draw = ImageDraw.Draw(pair)
            draw.text((12, 8), f'BEFORE - {angle} / {surface}', fill='black')
            draw.text((args.size+12, 8), f'AFTER - {angle} / {surface}', fill='black')
            draw.text((12, 29), 'CPU GLB diagnostic - same camera; clear lenses omitted; NOT a browser/AR screenshot', fill='black')
            for i, model in enumerate((args.before, args.after)):
                pair.paste(render(model, eye, look_at, surface, args.size), (i*args.size, 65))
            pair.save(args.output / f'{angle}-{surface}.png')
            print(f'{angle}-{surface}', flush=True)
    (args.output / 'render-settings.json').write_text(json.dumps({
        'before': str(args.before), 'after': str(args.after), 'size': args.size,
        'camera_fov_degrees': 32,
        'cameras': {k: {'eye': e.tolist(), 'target': t.tolist()} for k, (e, t) in cameras.items()},
        'limitations': 'CPU geometry/embedded color diagnostic, no PBR lighting or real AR. Clear lenses omitted. Not an accuracy score.',
    }, indent=2))


if __name__ == '__main__':
    main()
