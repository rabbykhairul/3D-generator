# Merchant upload guidance and pale-frame recovery

Scope: standalone demo. No Shopify changes, no edits to saved merchant projects, no paid inference service, and no automatic approval of detected geometry.

- [x] Show real front, side and three-quarter examples directly in Photos, with upload actions for the corresponding slots.
- [x] Explain required versus optional evidence, explicit arm mirroring, photo size/background guidance, transparent rims and cropped/obstructed inputs.
- [x] Replace the generic “clearer photo” failure with missing-boundary details and acknowledge detection can fail on clear catalog images.
- [x] Carry suggested outlines into Advanced correction on request instead of discarding them and making the merchant start over. Keep their approval status unchanged.
- [x] Recover the provided transparent front with stable, photo-derived frame and lens suggestions, without including lower-rim shadows. The fallback requires a uniform light background, tint/contrast evidence and stable paired openings; it does not handle every transparent frame.
- [x] Verify pale-frame regressions, existing shadow protection, upload-guide actions and production compilation. The supplied front also reconstructs successfully after explicit test-only approval; all suggestions stay unapproved in the real app.
- [x] Inspect `artifacts/transparent-detection/provided-front-overlay.png` against the original. First fallback was rejected because it included shadow shelves; chroma-supported filtering removes those shelves. Faint edge uncertainty still requires merchant review.
- [ ] Establish production coverage on a diverse merchant dataset: opaque/clear/metal/sunglass products, backgrounds, compression, cropping, obstruction and lighting. A single working sample does not establish commercial readiness.

## Verification update

- [x] Add a conservative faint-side retry. The supplied side photo returns an unapproved suggestion with an overlapping-arms warning. Visual inspection still shows irregular shaft boundaries and uncertain branch continuity: this is a correction starting point, not an accepted finished arm.
- [x] Preserve pale chromatic pixels inside reviewed texture boundaries instead of replacing them with darker high-contrast edge colors. Engine 8 marks older exports stale without discarding saved approvals.
- [x] Run the full Python suite: 96 passed. Upload-guide render/action tests and frontend production compilation passed. Live browser interaction and live AR were not verified in this turn.

## Honest acceptance policy

The product should accept common supplier photos, try suitable extraction methods, retain useful partial results and explain the smallest necessary correction. It should not promise exact reconstruction from every image. Missing or occluded physical details, unknown absolute size and transparent material appearance remain uncertain even when an image is high resolution.

The current demo supports JPEG/PNG and guided views. It does not automatically recover perspective-calibrated 3D from arbitrary lifestyle or worn-product photos. Retail release still requires licensed example photography, broader quality evaluation, merchant completion/time measurements and device testing.
