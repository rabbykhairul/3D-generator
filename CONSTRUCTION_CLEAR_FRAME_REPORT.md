# Construction and clear-frame follow-up — engine 11

## Outcome

Detection is improved, not universally solved. The exact water-clear `ppp` upload now yields provisional outer-frame and two-lens suggestions at native 1080px and a derived 4096px size. A neutral-material render reveals remaining nose-pad protrusions and hinge notches. Passing polygon/mesh checks must not be interpreted as visual fidelity. These contours still need correction before approval.

## Confirmed causes and changes

- Neutral transparent rims were excluded by the existing color-based fallback. `backend/reconstruction/clear_front.py` adds a nearly uniform-background, LAB-distance fallback requiring persistent paired apertures across several thresholds. Bounded topology repair rejects substantial changes, clipping and unstable outlines. It does not recover information absent from tiny images.
- Tiny self-touching contours caused final validity rejection. The neutral and double-bridge fallbacks repair only bounded invalidity and preserve lens containment. `full_rim_recovery.py` preserves the extra bridge opening rather than filling it.
- `construction_detection.py` recognizes the tested semi-rimless upper-rim/lower-lens pattern using product-relative coordinates. `construction_classifier.py` provides qualified material/construction suggestions, not a universal material classifier.
- `engine.py` integrates these paths, explicitly warns about neutral-frame hardware contamination, and constructs semi-rimless upper material using reviewed inner/outer termination anchors. It no longer invents opaque lower support wires. Semi-rimless endpieces and hidden depth remain approximate.
- `QuickDraft.tsx`, `Capture.tsx`, and `api.ts` expose the construction suggestion with manual override. Category and outlines are saved together on approval; stale analysis responses are ignored. Analysis does not mutate saved projects.
- `backend/engine_version.py` advances to 11.

## Verification

- Full Python suite: **163 passed** (2026-09-29).
- `node scripts/draft-component-test.mjs`: passed.
- Frontend TypeScript/Vite production build: passed; existing bundle-size warning remains.
- Isolated API test covers upload, extraction, explicit approval/category update, generation and GLB checks for the semi-rimless reference.
- `scripts/verify_construction_fronts.py` generated and reloaded disposable front-only GLBs for semi-rimless, water-clear and double-bridge references. All numerical checks passed. Assumed diagnostic dimensions are not product measurements. No merchant approvals or projects were changed.
- Synthetic blank, shadow, cropped and noisy clear-frame negatives are declined. The real clear reference at 64px is declined; tested lower-resolution clear variants through 512px remain unsupported. Upscaling is not evidence of newly recovered detail.

## Visual evidence and limits

See `artifacts/construction-followup/{semi,water-clear,double-bridge}/outline.png`, `front-clay.png`, `threequarter-photo.png` and `verification.json`. The clear-frame clay rendering documents failure as well as progress: hardware contamination still extrudes into the front. Tested smoothing either retained it or damaged the thin rim/lens containment, so no aggressive cosmetic repair was applied.

This work does not establish clean automatic reconstruction of arbitrary transparent, rimless, clipped, reflective or low-resolution photos. The three assets are front-only diagnostics, not completed wearable assemblies. Live browser/AR appearance, head motion, and real-device fit were not verified in this follow-up. No pixel-accuracy percentage is claimed.
