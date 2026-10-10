"""Conservative flat-background calibration for CPU contour detection only.

Never save these pixels or use them as photographic material. A quiet, bright
border is positive evidence for calibration; textured/dark scenes are left
unchanged, rather than forced into a catalogue silhouette.
"""
import numpy as np


def prepare_detection_background(image):
    height, width = image.shape[:2]
    band = max(1, min(height, width) // 40)
    edges = [image[:band].reshape(-1, 3), image[-band:].reshape(-1, 3),
             image[:, :band].reshape(-1, 3), image[:, -band:].reshape(-1, 3)]
    border = np.concatenate(edges).astype(np.float32)
    background = np.median(border, axis=0)
    noise = float(np.percentile(np.linalg.norm(border-background, axis=1), 90))
    agreement = float(max(np.linalg.norm(np.median(edge, axis=0)-background) for edge in edges))
    info = {"applied": False, "source_pixels_changed": False,
            "background_bgr": background.tolist(), "border_noise": round(noise, 3),
            "edge_disagreement": round(agreement, 3)}
    if background.min() < 170 or noise > 6 or agreement > 6:
        info["reason"] = "Background is not a verified uniform light surface"
        return image, info
    if background.min() >= 248:
        info["reason"] = "Background already near white"
        return image, info
    # Per-channel gain handles exposure and a mild background colour cast.
    # Multiplication preserves relative contrast; subtraction would whiten
    # faint edges and change their contrast with exposure.
    work = np.clip(np.rint(image.astype(np.float32)*(255/background)), 0, 255).astype(np.uint8)
    info.update(applied=True, method="verified-flat-background-gain",
                reason="Uniform light border calibrated on a temporary detection copy")
    return work, info
