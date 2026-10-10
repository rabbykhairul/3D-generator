# Four-view evidence and resolution improvements

Scope: standalone demo only; no saved merchant approvals or projects changed.

- [x] Preserve uploaded originals and report bounded extraction working resolution.
- [x] Adapt extraction for accepted thumbnail inputs and large photos, mapping contours back to source pixels.
- [x] Explain front, side, angle, and inner capture roles without requiring more views.
- [x] Add conservative original-pixel registration with cropped rear-view support.
- [x] Apply registered inner-frame appearance separately from outer surfaces; retain uncertainty for hidden regions.
- [x] Report actual use, unmatched photos, and limitations in generated metadata and Review.
- [x] Verify low-to-high-resolution extraction across multiple products.
- [x] Verify generation, GLB reload, separate surface mapping, and regression tests (130 passed).
- [x] Save visual evidence and final limitations report.
- [ ] Live browser/AR visual acceptance (browser launch blocked in this environment).

Not a promise of exact reconstruction: upsampling does not recover missing detail; rear-frame registration does not determine arm depth or hidden hinge construction. Unknown geometry must remain explicitly inferred. Live camera/browser verification may require the user-run development server.
