# Shared detection recovery — engine 25

The silver single-bridge sunglasses exposed two assumptions: dark-lens recovery
required a third bridge opening, and arm tracking discarded a thin upper run
when the overlapping rear arm was much wider. Recovery now accepts stable
single- or double-bridge topology and retains narrow upper-branch evidence.
Neutral metal has its own contrast fallback; no product IDs are in the detector.

Lower shadow protrusions are bounded against measured lens/rim spacing only on
the neutral dark-lens route. Short inward shaft notches are regularized with
bounded displacement and area, preserving sleeve transitions, tip and hinge.
A lone off-centre dark blob without lens evidence is no longer returned as a
complete front. Merchant approvals and saved artifacts are not overwritten.

`scripts/verify_frame_families.py` produces isolated overlays and JSON under
`artifacts/frame-families/`. Its seven references cover silver single-bridge,
gold double-bridge, clear, semi-rimless, lavender, olive and black/gold frames.
All fourteen front/side photos return the required parts. This is a coverage
check, not a pixel-perfect accuracy metric. The silver reference has additional
540/1080/2160-pixel and mirrored regressions, including rear-arm exclusion,
lower-shadow rejection, bridge topology and closed front generation.

Final validation: `python -m pytest -q` — 218 passed (one existing
Starlette/httpx deprecation warning). All fourteen audit photos still return
their required parts after the final changes. Silver front/arm, clear front/arm
and gold front overlays were inspected visually; live browser rendering was
not verified in this session.

Limits: these are catalogue photos on quiet light backgrounds. Upsampling does
not restore invisible edges; hidden hinge mechanics and exact optical properties
remain inferred. No universal-photo or live-browser-render guarantee is made.
Reanalyze front and side, review the new suggestions, then regenerate to use the
new geometry. A running backend without automatic reload needs restarting.
