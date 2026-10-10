"""Texture a registered component from its own source photograph."""
import hashlib
from pathlib import Path
import cv2
import numpy as np


def component_photo(annotation,front,photos,output_dir):
    registration=annotation.get('registration')
    if not registration:return front
    source=next((p for p in photos if str(p.get('id'))==registration['source_photo_id']),None)
    if source is None:raise ValueError('Registered component source photo is unavailable')
    from .engine import _is_validation_photo
    if _is_validation_photo(source):raise ValueError('Validation photo cannot texture a fitting component')
    image=cv2.imread(str(source['path']))
    if image is None:raise ValueError('Registered component source photo cannot be decoded')
    matrix=np.asarray(registration['source_to_front_homography'],float)
    scale=min(1.,2048/max(front['width'],front['height']))
    size=(max(1,round(front['width']*scale)),max(1,round(front['height']*scale)))
    matrix=np.diag([scale,scale,1.]) @ matrix
    key=hashlib.sha256((str(source['id'])+str(registration['source_to_front_homography'])).encode()).hexdigest()[:16]
    path=Path(output_dir)/('component-texture-'+key+'.png')
    path.parent.mkdir(parents=True,exist_ok=True)
    if not path.exists():
        warped=cv2.warpPerspective(image,matrix,size,borderMode=cv2.BORDER_CONSTANT,borderValue=(255,255,255))
        if not cv2.imwrite(str(path),warped):raise ValueError('Could not write registered component texture')
    return {**front,'path':str(path)}
