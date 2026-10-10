# Engine 27: independent rimless construction

The supplied rimless front previously produced no contours. A shared fallback
now verifies two independent faint lens boundaries at multiple contrast levels,
with measured perimeter support, pair balance and a separate central mount.
It does not require a connected outer frame or infer rimless merely because
full-rim detection failed. No project IDs or product coordinates are used.

The bridge/mount contour is traced separately and exposed in Advanced as
`bridge-hardware`. Approved geometry replaces the generic rimless bridge tube.
Lens and bridge masks are separated from nearby protrusions to identify nose-pad
suggestions; each pad requires agreement across three thresholds. The nose-pad
and hardware silhouettes retain photographic texture. Depth is inferred, not
measured. Unknown pads remain omitted; no universal pad shape is substituted.

Separate approved pad polygons are now honored across all six construction
categories even when no pad anchor is present. Previously some categories
silently ignored these polygons. Six additional category-specific generation
tests cover asymmetric pad shapes, photographic maps and closed geometry.

Front and side fixtures supplied by the user are retained under
`tests/fixtures/rimless/`. Tests cover 540/1080/2160 pixels, mirror reversal,
both lenses/pads, bridge topology, shadow exclusion and exported closed meshes.
`scripts/verify_rimless.py` creates an isolated GLB and front/side overlays under
`artifacts/rimless/`, without changing saved projects or approvals.

The existing full-rim, clear-frame, semi-rimless and dark-lens paths remain
in place. This extends coverage; it is not universal image understanding.
The new route assumes a quiet light background and visible paired boundaries.
Different colors, invisible edges, severe occlusion, alternate hardware and
perspective views can still require correction. Endpieces, hidden mounts,
cross-sections and depth are not fully reconstructed from these photos.

Use Analyze on the front, review the rimless suggestion and all component
outlines, analyze/review the side, then regenerate. Old GLBs remain unchanged.

Validation: 229-test full suite passed. After adding the all-category pad fix,
45 focused reconstruction/rimless tests passed, including its six new category
cases. Frontend production build passed. All 14 previous family audit photos
still returned required parts; the supplied rimless front/side add another two.
Front/side overlays were visually inspected and the isolated rimless GLB passed
closed-front and consistent-normal checks. Live-browser/AR rendering and remote
deployment were not verified in this turn.
