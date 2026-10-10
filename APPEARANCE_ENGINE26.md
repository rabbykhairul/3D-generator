# Engine 26 — source-detail preservation and deployment preparation

Changes apply to inferred dark-lens models, not just one project ID:

- Lens tint/alpha compensation is computed in linear light, matching rendering,
  instead of compensating source sRGB values before conversion. Custom lens
  settings remain unchanged. Transmission remains an appearance estimate.
- Inferred transparent lenses no longer write opaque depth in the shared viewer
  policy. This avoids depth-buffer rejection of transparent components behind
  them; it does not solve all object-ordering or optical-rendering limitations.
- Small isolated dark details in upper outer lens corners are retained from the
  source photograph. Traced pad silhouettes support conservative recovery of
  nearby visible mounting appearance. Large rear-arm components are excluded.
- These are shallow photographic patches, **not measured 3D mounting brackets**.
  Low-contrast printing, text connected to the rim, broad reflections, unseen
  inner arm surfaces and hidden hinge construction remain unresolved.
- Existing metallic rim geometry/materials are not replaced by invented chrome
  highlights. Exact metal reflections and fully matched rear-arm geometry have
  not been established from this comparison.

`scripts/verify_silver_appearance.py` generates an isolated engine-26 GLB, source
detail overlay and CPU front render in `artifacts/silver-appearance/`. Three
retained patches correspond to the visible lens mark and both pad mounts.
The diagnostic includes only the independently supplied right-side arm, not a
fabricated matching left arm. Merchant approvals/models are untouched.

The CPU diagnostic now blends estimated lenses in linear light. It is not a
browser screenshot, optical simulator, or proof of live AR appearance. Reanalyze
if contours are stale, review them, and regenerate to create an engine-26 GLB.

See `DEPLOYMENT.md` for password-protected Docker/Caddy deployment, persistent
data, backups, rollback and the existing-proxy alternative. This is a shared
authenticated workspace, not an isolated-account multi-merchant service.

Validation: full Python suite 221 passed; final detail-filter change rechecked
with 14 focused tests passing. `npm run build`, the actual viewer-policy Node
test and `docker compose config --quiet` passed. Docker daemon unavailable;
container build/runtime, Caddy certificate issuance and remote browser QA are
not verified. No remote deployment was performed.
