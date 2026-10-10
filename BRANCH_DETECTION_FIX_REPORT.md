# Transparent frame / hybrid-arm detection fixes

## Confirmed defects

1. On the pink and lavender uploads, the previous side detector began from an internal shaft run, followed the nearest run center, and continued into the far arm. Its result combined two physically different arms. A previous test even pinned the incorrect lavender far-tip coordinate; it now requires the actual outer tip.
2. Transparent shaft highlights and printing split the threshold mask into small runs, which became spikes, notches and missing shaft sections.
3. At 4096px, resizing changed which detector won: an opaque-seam fallback could preempt the corrected branch and select the far tip again.
4. For the lavender front at 128/160px, completing an occluded lens slightly exceeded the extremely thin outer mask. A containment guard discarded that lens, even though almost all its completed area was supported.

## Implementation

- `backend/reconstruction/branch_side.py`: a new guarded catalogue-photo route verifies the long upper distal branch, fills narrow vertical highlight gaps, retains one outer branch through separation/overlap, and requires polygon agreement at three contrast levels. It is not a trained segmentation model, nor a claim to handle every pose. Unsupported branch arrangements return to the general route; unstable recognized boundaries are withheld.
- `backend/reconstruction/side_extraction.py`: explicit upper-branch selection can be requested without changing the default tracking route. Existing local crossing cleanup is applied to remove the remaining opposite-arm tab.
- `backend/reconstruction/engine.py`: route selection prevents an older seam detector from overriding a stable outer branch with a different, shorter tip. Front completion clips a tiny overshoot only for faint fronts with less than 1% area loss; uncertain larger errors remain rejected.
- Regression tests in `test_branch_side.py`, `test_side_smoothing.py`, `test_reconstruction.py`, and `test_resolution_extraction.py`. No product ID, source-image coordinate or specific color was added to production detection rules.
- `scripts/verify_branch_side.py` reproduces legacy/current contours, generates both models in isolated output folders, reloads them, checks welded closure, and renders neutral-material comparisons.

## Results and evidence

- Final Python suite: **151 passed**. Frontend production build passed. Existing warnings remain for a Starlette dependency deprecation and the frontend bundle size.

- Pink side: correct complete outer tip and shaft from 64 through 4096px in the tested matrix.
- Lavender side: correct complete outer branch at 256/512/1080/2160/4096px. 64/100/128px are still declined rather than producing a hybrid arm. The pixel derivatives test resolution handling, not newly acquired high-resolution information.
- Horizontal reflection retains the selected physical branch for both products.
- Existing blue opaque, gray, pale-metal and synthetic arm regressions remain part of the full suite.
- Lavender front: 128/160px changed from missing a lens to valid front plus both lens openings. Full-size PINK and lavender front photos already succeeded; they were not incorrectly claimed as newly fixed. The front audit found a rimless-wire photo and a clipped old photo still cannot safely supply a full-rim outline.
- Same-photo comparisons: `artifacts/branch-side-fix/pink/outline-comparison.png` and `artifacts/branch-side-fix/lavender/outline-comparison.png`.
- Reloaded neutral-material meshes: the corresponding `mesh-comparison.png` files. These are same-camera CPU diagnostics, not live browser/AR screenshots.
- Front evidence: `artifacts/transparent-front-followup/`.

## Applying / limitations

Saved merchant projects, categories and approvals were not edited. Run the updated backend, re-analyze the current front/side photos as applicable, approve the corrected suggestions, and regenerate. Merely regenerating keeps old approved geometry.

The detector remains constrained photo-assisted reconstruction. Arbitrary backgrounds, hidden physical boundaries, rimless wires and extremely small transparent arms are not solved universally. Depth, hinge mechanics and transparency remain approximations. No 90% fidelity or universal-resolution accuracy claim is made. Live AR is not reverified by this change.
