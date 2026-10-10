# Rear-arm shadow and contour refinement — engine 14

## Confirmed cause

The rear-arm appearance detector used one background-distance threshold. In the gray reference (`934325061be045f0abda9fb2d233aa46`), its contact shadow passed that threshold and remained connected to the tip. Polygon simplification preserved this erroneous branch; smoothing alone could not remove it.

## Changes

- `opposite_arm.py`: component-relative contrast cleanup excludes weak attached pixels beside a stronger arm. A bounded opening removes narrow residual contact-shadow connections only when at least 90% of the filled component remains. Pale components keep the low detection threshold; enclosed light details remain available for photographic texture transfer.
- The rear appearance contour now uses the existing bounded, corner-aware contour refinement, with at most 0.85 analysis-pixel displacement. It does not replace the observed arm with a generic straight shape.
- `side_extraction.py`: apply the same bounded refinement to newly suggested temple geometry, simplify redundant collinear samples, and reject refinements that move the hinge outside its root. Saved approvals are untouched.
- `engine_version.py`: 14, so newly generated assets use this processing.

## Evidence

Final full Python regression suite: **173 passed**. The isolated olive GLB reloaded with closed geometry and retained the inner-tip logo. No frontend source changed in this follow-up.

`scripts/verify_rear_shadow.py` compares the same gray source and crop, with cleanup disabled for the before condition:

- `artifacts/rear-shadow/before-tip.png`: horizontal shadow spur included.
- `artifacts/rear-shadow/after-tip.png`: spur excluded; actual black tip retained.
- `before.json` / `after.json`: original-coordinate suggestions and smoothing diagnostics.

Regression coverage includes the exact gray shadow point, mirrored orientation, synthetic attached shadow, retained pale material, preserved enclosed printing, and existing olive 540–2160px tests. `scripts/verify_olive_side.py` separately verifies reloaded GLB topology and inner/outer texture separation after cleanup.

## Limits

This is bounded image-conditioned cleanup, not universal semantic shadow segmentation. Very dark shadows indistinguishable from material, strong occlusions and uncertain transparent boundaries remain difficult. Genuine bends, sleeve transitions and corners must remain; the system does not force the whole arm into straight lines. Existing models require new analysis/review and generation to change saved geometry. No saved merchant project was modified and live AR was not visually tested in this follow-up.
