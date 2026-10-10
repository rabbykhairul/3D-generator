"""Translation/margin normalization before existing CPU front detectors.

No synthetic contour, template substitution or semantic model: the cropped view
must still pass a detector's paired optics/connected support evidence checks.
Only quiet bright studio borders qualify. Source coordinates are restored exactly.
"""
import cv2
import numpy as np


def recover_product_local_front(image, photo, scale_x, scale_y):
    h, w = image.shape[:2]
    strip = max(2, min(h, w) // 60)
    border = np.concatenate((image[:strip].reshape(-1, 3), image[-strip:].reshape(-1, 3),
                             image[:, :strip].reshape(-1, 3), image[:, -strip:].reshape(-1, 3)))
    background = np.median(border, axis=0)
    distance = np.max(np.abs(border.astype(float) - background), axis=1)
    if background.min() < 225 or np.percentile(distance, 95) > 5:
        return None
    deviation = np.max(np.abs(image.astype(np.float32) - background), axis=2)
    mask = np.uint8(deviation > max(12, float(np.percentile(distance, 95)) * 3)) * 255
    # Ignore isolated dust, retain detached optical/hardware components. Do not
    # close/morph a bridge into existence or discard pale material by cropping
    # tightly to its dark internal wire.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    qualifying = [i for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] >= max(16, h*w*.0001)]
    if not qualifying:
        return None
    xs = [int(stats[i, cv2.CC_STAT_LEFT]) for i in qualifying]
    ys = [int(stats[i, cv2.CC_STAT_TOP]) for i in qualifying]
    x0, y0 = min(xs), min(ys)
    x1 = max(int(stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH]) for i in qualifying)
    y1 = max(int(stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT]) for i in qualifying)
    bw, bh = x1-x0, y1-y0
    if bw < 80 or bh < 30 or not .1*bw < bh < .8*bw:
        return None
    # Never treat a clipped object as a complete silhouette.
    if x0 <= strip or y0 <= strip or x1 >= w-strip or y1 >= h-strip:
        return None
    margin = max(12, round(bw * .12))
    x0, y0 = max(0, x0-margin), max(0, y0-margin)
    x1, y1 = min(w, x1+margin), min(h, y1+margin)
    if (x1-x0)*(y1-y0) >= w*h*.85:
        return None
    crop = image[y0:y1, x0:x1]
    from .full_rim_recovery import recover_dark_front, recover_double_bridge_front
    from .rimless_detection import recover_rimless_front
    for detector in (recover_dark_front, recover_double_bridge_front, recover_rimless_front):
        candidate = detector(crop, photo, scale_x, scale_y)
        if candidate is None:
            continue
        parts = {a['part'] for a in candidate['annotations']}
        if not {'lens-left', 'lens-right'} <= parts or not parts & {'front', 'bridge-hardware'}:
            continue
        for annotation in candidate['annotations']:
            for point in [*annotation['points'], *annotation.get('anchors', {}).values()]:
                point['x'] += x0 * scale_x
                point['y'] += y0 * scale_y
            annotation['source'] = 'product-local:' + annotation.get('source', 'automatic')
        candidate['product_local'] = dict(method='quiet-studio-product-crop',
                                         bounds_source_px=[x0*scale_x, y0*scale_y, x1*scale_x, y1*scale_y],
                                         detector=detector.__name__)
        candidate.setdefault('warnings', []).append('Product-local recovery reduced blank margins. Review boundaries in the original photo; no additional photographic detail was created.')
        return candidate
    return None
