# Reported semi-rimless and crossing fixes — engine 16

## Implemented

- Side branch-consensus extraction no longer applies the broad pre-crossing width cap. That cap could remove a genuine thicker sleeve. Bounded local overlap repair remains active.
- Short lower-edge overlap repair detects smaller rear-arm fragments (scale-relative threshold with a 2.5 analysis-pixel floor, previously 5).
- Semi-rimless lower lens evidence is locally regularized with a median filter and bounded smooth fit. This removes the reported short shadow detour without replacing the lens with a generic ellipse. Large deviations still reject the suggestion.
- Semi-rimless bridge height/thickness now comes from central image evidence, not the height of the brow endpoints. The burgundy reference no longer produces the high rectangular bridge.
- Semi/rimless lenses now have front/back bevels on their physical 0.7 mm cut edges. Their existing transparent tint/opacity is preserved. No photograph, arm, or ground shadow is mapped onto the lens.

These are shared pipeline changes, not project-ID conditions. Saved merchant outlines and approvals were not modified.

## Verification

- Existing suite: 178 passed, with one existing Starlette deprecation warning.
- Three added actual-reference regressions: passed (genuine sleeve coverage, small lavender crossing spur, gold bridge location).
- API upload → suggestion → approval → generation → GLB reload test passes. Both lenses remain watertight, outward-wound, transparent and untextured; bevel normals survive export.
- Visually inspected refreshed source overlays in `artifacts/semi-edge-cases`: `semi-front.png`, `semi-side-crossing.png`, `lavender-side-crossing.png`.
- Semi-rimless detection tests cover 540/1080/2160 inputs and translated/padded/cropped layouts. This is not a guarantee for arbitrary photographs or unreadable edges.

## Remaining verification / limitations

Clear lens geometry was already present in the reported generation; faint appearance is not equivalent to a missing mesh. Beveled geometry has been corrected and export checked, but live preview/AR visibility has not been visually verified in this run. The preview at localhost:5188 was unreachable. Do not label the clear-lens appearance issue fully resolved on geometry checks alone.

The CPU fidelity renderer deliberately omits lenses and cannot establish their transparency or visibility. No CPU render is offered as proof of a lens appearance fix. Nose-pad construction, exact rim thickness, and perfect arbitrary-image recovery are not certified by these changes.

Re-analysis is required to obtain new suggestions; existing approved outlines are deliberately not silently replaced.
