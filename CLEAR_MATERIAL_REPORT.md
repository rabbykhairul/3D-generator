# Clear-frame texture correction — engine 19

Confirmed on an isolated copy of the latest clear-frame project (`f94cc9fb…`),
which was already categorized as transparent-full-rim. The material category
was not the cause for that project.

`_frame_texture` discarded neutral pale pixels inside the reviewed material
polygon. Nearest-valid-color padding then spread dark hardware and edge colors
across clear acetate, creating the wide gray/black bands. This was photographic
texture corruption, not evidence of metal or a geometry defect.

Clear-acetate materials now retain all source pixels within their approved
material polygon; only outside padding is synthesized. Genuine hinge details
remain intact. The existing opaque-material cleanup is unchanged. Material
identity and BLEND alpha survive textured bevel generation and GLB export.

`scripts/verify_clear_material.py` generates before/after models from the same
saved geometry without modifying the project. `artifacts/clear-material/` holds
both GLBs, texture atlases and matching-camera CPU texture diagnostics. Those
diagnostics reproduce/remove the reported bands, but do NOT simulate alpha,
refraction or lens optics. Both exported frames have alphaMode BLEND and base
color alpha 140/255, retaining the current clear-frame opacity policy.

Added tests check preservation of pale interior pixels and dark hardware,
unchanged opaque cleanup, material identity, and transparent bevel material.

Live preview verification was attempted on an isolated Vite server. Chrome
aborted before opening the page; computer-use reported no available browser.
Thus no browser or AR visual-fidelity claim is made. The separate browser
capture script is retained for a capable environment. No merchant approvals
or saved models were overwritten. Existing assets require regeneration.
