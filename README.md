# Frame Lab — local eyewear reconstruction demo

Independent research app: product photos and measurements → editable outlines → CPU-generated GLB → browser inspection and AR. No Shopify integration, cloud GPU or paid generation API.

Engine 34 corrects sRGB-to-linear conversion for solid GLB material factors,
removes category-forced frame opacity, preserves observed edge pixels during
texture padding, and keeps solid hardware unlit in photo-color inspection/AR.
User opacity/tint settings are retained; authored optical surfaces and material
mode are not promises of exact photographic colour. Export factors have 8-bit
quantization, unseen walls remain inferred and blurred, and live browser colour
acceptance is still required. Restart and regenerate old exports for this change.

Engine 33 removes a merged rear-arm strip before an observed sleeve transition,
and retains a stable visible coloured endpiece beyond the side-view arm seam.
The endpiece is textured from its source outline and seated at the front anchor;
its projected depth still assumes a near-side photograph. It does not recover
hidden hinge mechanics or arbitrary neutral/transparent endpieces. Re-analyze
the side photo and review the added frame-attachment anchor before approving.

Engine 32 removes fixed-margin inflation above detected semi-rimless brows,
uses locally observed brow widths, retains visible outer tabs and proposes
attachment anchors inside those tabs. Unmeasured thin/semi-rimless fronts use a
1.2 mm draft depth instead of 4 mm; thin transparent full rims use a bounded
outline-clearance prior. These are **not measured depth or hinge mechanisms**.
Measured front depth is preserved. Re-analyze and review the front photo before
regenerating: existing approved outlines are deliberately not overwritten.

Engine 31 adds source-supported visible nasal-component recovery beyond the
rimless/dark-frame detector paths, and starts the inspection viewer at a straight
front angle. Re-analyze the front photo to obtain new suggestions; regenerate
only after reviewing them. Detected components can be partial and their depth
remains inferred. This is not a Fittingbox-equivalence claim. See
[source-fidelity progress](FIDELITY_IMPLEMENTATION_PROGRESS.md) for verified work
and the remaining recognizer, geometry-fitting and material-validation work.

Engine 30 adds a versioned eyewear anatomy graph, source-space geometry audits,
review guidance and product-local recovery for off-centre studio photographs.
It preserves existing procedural CPU generation rather than matching individual
SKU templates. See [implementation and verification plan](ANATOMY_IMPLEMENTATION_PLAN.md)
and [anatomy architecture](ANATOMY_ARCHITECTURE.md). Restart the backend, refresh the
page, then re-analyze and regenerate; old approvals are preserved, not auto-replaced.

Engine 29 included detection-only calibration for verified flat light backgrounds,
scale-aware faint rimless-edge recovery, stable dark-lens intervals for grey rims
in landscape crops, independent dark-pad silhouettes, and labelled benchmark gates.
See [CPU production-readiness work](CPU_PRODUCTION_READINESS.md) for verified
scope, benchmark usage and remaining release requirements. This remains a
reviewable reconstruction workflow, not universal automatic reconstruction.

For the unknown-product reliability work, see the tracked
[reconstruction roadmap](RECONSTRUCTION_ROADMAP.md). Saved-photo replay and
detection diagnostics are implemented; the isolated pretrained CPU evaluator
has installed dependencies and recorded checkpoints, but still needs independent labels.
Standard detection remains rule-based. An experimental AI-assisted front-analysis
option now uses medium pretrained lens proposals with source-edge refinement and
geometry checks; it retains complete standard contours and rejects unsupported
candidates. Valid lens boundaries remain available as partial suggestions when
frame recovery fails. On a verified quiet light background, the lens pair can
guide a search for connected faint source rims/bridge; agreement across contrast
thresholds is required, and disconnected rims are never joined synthetically.
Small and medium pretrained
eyewear segmenters have now run on 24 unique saved photos in a separate CPU
environment. Lens proposals are promising, but frame/arm errors prevent direct
replacement of geometry extraction. See the [evaluation and reproduction instructions](RECONSTRUCTION_ROADMAP.md#pretrained-cpu-evaluation-2026-10-05)
and local `artifacts/segmenter-evaluation/comparison/index.html` for comparisons.

Engine 45 defaults to **CPU · thin metal + contours (no AI)** for front analysis and
**CPU · source contours (no AI)** for sides. SAM 3 remains an optional selection; see setup below.
SAM 2.1 remains selectable. It uses
Meta's pinned **SAM 2.1 Hiera Small** checkpoint (46M parameters) at its native
image-encoder resolution. The existing 3.22M lens model provides coarse automatic
box/positive/negative prompts; SAM produces the detailed masks. If the locator
cannot supply usable prompts, SAM uses an automatic image grid. No merchant clicks,
custom training or image uploads are required. General masks are assigned optical
roles by geometric checks, not a trained hinge/nose-pad recognizer. Source boundary
and topology audits still apply, and complete existing contours are preserved.
The earlier model is available as **Legacy lens model · experimental**.

Automatic front outlines receive a common finishing pass after source and AI
extraction: short raster ripples are smoothed with bounded pixel displacement,
while sharp bridge/endpiece corners and joint rim topology are retained. On a
verified uniform light background, pale neutral lower extensions can be removed
when stronger observed rim pixels establish the edge above them. Ambiguous pale
material remains unchanged. This affects fresh suggestions only; reanalyze and
review before regenerating a model from older outlines.
This pass also handles product-local and occlusion-completion fallback sources.
Side-arm suggestions receive bounded smoothing with hinge/tip neighborhoods
protected; saved approvals and attachment anchors are not rewritten.

Lens refinement retries narrower source-edge search bands when a wider band
produces invalid geometry near bevels or mounting hardware. An incomplete
threshold front cannot veto independently source-supported optical contours:
those lenses still pass the same evidence/topology checks, and a conflicting
front is omitted unless joint source support is established. Recovering lenses
does not establish a complete brow, bridge or frame construction.

SAM setup, using the existing optional Python 3.12 environment:

```sh
artifacts/eval-tools/bin/uv pip install --python artifacts/segmenter-venv/bin/python -r requirements.segmenter.txt setuptools==84.0.0
SAM2_BUILD_CUDA=0 artifacts/eval-tools/bin/uv pip install --python artifacts/segmenter-venv/bin/python --no-build-isolation -r requirements.sam2.txt
.venv/bin/python scripts/setup_sam2.py
```

SAM weights and Apache-2.0 notice are stored in ignored `artifacts/sam2/`; the
manifest pins the publisher URL, code revision and checksum. Runtime downloads
are disabled. `EYEWEAR_SAM2_PYTHON` and `EYEWEAR_SAM2_MANIFEST` can override paths.
Use `scripts/dev.py --sam2` or `?backend=sam2` for API-default/per-request selection.
CPU inference is serialized and bounded to 120 seconds. On four saved front
photos, the pipeline recovered tinted/opaque suggestions and retained existing
clear/rimless contours when SAM refinements were unsuitable. This is a development
smoke check with no independent labels, not evidence of universal accuracy.

The selectable SAM 2.1 side-arm analysis It uses an automatic image
grid, accepts continuous elongated regions with an identifiable frame-side
endpoint, and retains valid standard contours. When SAM merges overlapping arms,
both envelope paths are evaluated. Only a distinctly longer, continuous branch
supported by the mask and excluding the other tip is suggested. Its shared shaft
and attachment remain explicitly inferred. Equal-length or unresolved competing
paths are withheld; the model does not recognize the hinge's mechanical design.
Photo anatomy reports describe that photo only. The opposite arm is not evidence
from a single side view. This is automatic projection-based draft geometry, not
complete semantic eyewear recognition or recovery of hidden manufacturing detail.

Lens proposals describe visible pixels, which can exclude folded arms seen
through the lens. The optical-window pass follows the outer radial boundary,
regularizes narrow inward cuts, and refines against source edges inside the
observed front. It does not fit a circle or use a convex hull; broad concavities
remain. Completion is bounded to 15% added area and remains explicitly inferred
when it exceeds 1%. Missing frame support, weak source edges or excessive shape
changes cannot justify a completed optical window. A source-support sampling
band prevents the outer frame/background edge being accepted as the lens edge.

Refresh
the page after updating and click **Analyze current front photo**. You can select
**Standard analysis** explicitly for comparison. It uses the isolated environment and
local checkpoints prepared during evaluation; no model downloads occur during
analysis. The report distinguishes used proposals, retained standard contours,
rejected candidates and unavailable inference. Suggestions still require approval.
Use `.venv/bin/python scripts/dev.py --hybrid` to make this the server default,
or `?backend=hybrid` on the photo extraction API for one request. Configure optional
`EYEWEAR_SEGMENTER_PYTHON` and `EYEWEAR_SEGMENTER_MANIFEST` paths for another machine.
The default environment/checkpoints live under ignored `artifacts/`, so a clean
checkout needs the evaluation setup before using this option. Side/angled-only
and missing-view reconstruction are not solved by this lens model.
The isolated process loads its model for each analysis: the measured median
inference request was about 3.2 seconds on this Mac, including startup, rather
than the warm model-only timings in the evaluation.

## Start

Dependencies are installed in this checkout. From this directory:

```sh
.venv/bin/python scripts/dev.py
```

Open **http://127.0.0.1:5188**. API documentation is at **http://127.0.0.1:8018/docs**. Ctrl-C stops the frontend, API and its CPU worker. Restart after changing Python reconstruction code; frontend changes refresh automatically.

For a new installation, use Python 3.11 and Node 22 LTS:

```sh
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.lock.txt
npm ci
.venv/bin/python scripts/dev.py
```

The tested Python dependency snapshot is in `requirements.lock.txt`; the frontend uses `package-lock.json`. The existing local MediaPipe model and WASM runtime are bundled under `frontend/public/` so AR needs no runtime CDN. The current browser test uses installed Google Chrome.

## Try the demo

Photos now opens with real front, side and angled upload examples. The automatic front detector has a guarded fallback for pale tinted frames on uniform light backgrounds, including the supplied lavender example; its suggestions still require review. Missing-boundary messages no longer assume a poor photo, and opening Advanced retains detected suggestions without approving them. Restart the Python backend after this extraction update, refresh Photos and click **Analyze current front photo** again. See [merchant upload plan](MERCHANT_UPLOAD_PLAN.md) for tested scope and remaining production work.

Latest fidelity/occlusion work is documented in [FIDELITY_OCCLUSION_REPORT.md](FIDELITY_OCCLUSION_REPORT.md). Engine 7 adds local front/temple bevels, slimmer buried connectors and arm roots attached to the actual local rear surface; explicit frame depth is honored, but missing depth still uses an unverified 4 mm default. Existing current approved outlines can be reused when regenerating an older engine export. The requested 90% fidelity target has **not** been achieved. Matching-angle CPU render evidence and a real-product API generation/reload regression are available; browser/live-camera acceptance remains pending.

AR now includes independent skull/head and glasses calibration, skull/live-face/anchor overlays, wireframe and an optional shadow. Settings persist per exported model URL. These additions pass offline tests but still need browser and physical-device verification. Depth masking hides virtual arms behind the head; it does not erase real glasses in camera images. With the server running, `/fidelity-review.html` opens the isolated matching-camera baseline/candidate comparison without editing your projects.

Each sidebar row is a saved project. Click its **×** to remove it after confirming the name/short ID. This is recoverable: **Removed projects / Restore** brings back photos, annotations and models. No files are permanently deleted by this action.

For missing arms, upload a Left or Right side photo and scroll to **Reconstruct the … arm** in Photos. Click **Analyze side arm**, inspect the orange outline and hinge/tip markers, then **Approve arm & regenerate**. Advanced correction remains available. Only one visible arm is selected, even if the photo shows both. With one usable side view, you may explicitly choose **Also mirror …** to assume symmetry for the unseen arm; it is off by default and labeled as an assumption in metadata. An existing independently reviewed opposite arm cannot be overwritten by mirroring.

Front approval now proceeds through every pending side review before submitting generation. Use the **Photo view** dropdown under an uploaded photo to correct Left/Right labels without uploading again. Overall **Inner / rear** photos are comparison references, not automatic sources of hidden surfaces. **Inner left/right** accepts a side-on view of an individual arm for geometry; the newest outer/inner capture for that arm is used, rather than merging incompatible viewpoints. If extraction is unusable, you can explicitly keep that photo as reference only and omit the arm.

**Surface display → Photo colors (no relighting)** is the default in 3D inspection and is also used by AR. It displays photo-textured surfaces without studio relighting or filmic tone mapping, while preserving independent lens and hardware materials. Choose material, neutral clay or silhouette to distinguish texture/shading defects from geometry. The exported GLB retains its embedded textures and PBR materials, so unrelated external viewers can still relight it differently. This preserves photographic appearance; it does not recover calibrated reflectance.

Version 4 requires review of older automatic front/arm traces. New side suggestions exclude short overlapping-arm crossings and pixel spikes; automatic front curves receive bounded refinement. Explicit smooth normals and separate photographic-cap/depth-wall texture sampling improve close-up surfaces. Existing manual geometry is not silently smoothed. See [surface and sleeve correction report](SURFACE_FIX_REPORT.md) for implementation, matched-angle evidence and remaining reconstruction limits.

For older automatic front traces, **Regenerate model** now requests fresh shadow-aware analysis and review. Check that the green boundary follows actual frame material, not a contact shadow beneath the rims. Pale/transparent frames may need manual corrections; no classifier can be assumed reliable for every background. The original exports remain available. See [assembly/shadow/AR correction report](ASSEMBLY_FIX_REPORT.md) for the current gray-frame example and before/after evidence.

For the current real-photo example, select **Olive frame — occlusion fix review** and open **Review / AR**. The lens obstructions are removed and bridge opening preserved. Unverified arms are omitted; this is explicitly an incomplete front-only model, not proof of exact product fidelity. Older engine exports require regeneration.

1. Import a sample from the sidebar. **Public** samples contain real retailer photos and published dimensions. **Synthetic** samples contain known drawn outlines solely for numerical/workflow tests.
2. In Photos, start with an existing straight front product photo. Side and angled photos are helpful, not required. Extra views and validation roles are under Advanced. Confirm frame construction; a different product also needs its own dimensions.
3. Click **Analyze current front photo**, inspect the detected frame/lens boundaries and **Approve outlines & generate draft**. If detection fails, upload a clearer photo or open Advanced correction. Extraction never auto-approves. Side/top outlines and anchors remain an advanced workflow; angled images are review evidence only.
4. In Dimensions, supply the five required values in millimeters, optional depth measurements and material settings. Label each value measured, published, manual, inferred or default. Save.
5. Generate, inspect the GLB, compare it with fitting/withheld photos, or download GLB and metadata. Edits hide the outdated preview; historical downloads stay explicitly labeled. The latest fitting upload per front/left/right/top view supplies geometry; superseded approved outlines cannot silently drive the next model.
6. Open Webcam AR. Start the camera or choose a local face photo. PD and manual controls calibrate approximate placement. Camera/photo frames remain in the browser.

### Only photos, no physical measurements?

If a generation error shows values like `frame_width: 0.2`, the field has been entered in millimeters, not meters. Do not multiply all fields blindly—some may already be millimeters. In Dimensions, either enter verified millimeter values or click **Use approximate draft sizes (140 mm)**, inspect the replacements and **Save dimensions & appearance**. This explicit recovery recomputes consistent front proportions without changing photos, outlines or materials. It does not determine your real product size.

After analyzing the front photo, select **Use approximate size for this draft**. Missing overall width and temple length use 140 mm placeholders, not claimed product measurements. Lens/bridge proportions are inferred from outlines when available. Existing positive measurements are preserved. Both the interface and exported metadata warn that physical size and fit are unverified; replace estimates with product specifications later.

100 × 100 thumbnails are accepted with quality warnings; the minimum upload size is 64 pixels per side. Use the largest straight-on photo as Front and keep small alternate front images under Advanced detail/validation so they do not supersede it. Enlarging thumbnails cannot recover detail. Occluded lens outlines may receive an unapproved convex-completion suggestion; review it carefully, especially for intentionally concave lenses. A central `bridge-opening` annotation becomes a real cutout in solid-frame construction. Arbitrary extra openings are not automatically recovered. Missing side geometry is omitted, never replaced by invented arms.

### Frame categories and annotations

| Category | Required front annotations | Construction |
|---|---|---|
| Opaque / transparent full-rim | `front`, `lens-left`, `lens-right` | Front solid with lens holes, curvature and thickness |
| Thin metal | `front`, both lens outlines | Swept metal rims and bridge |
| Semi-rimless | `front`, both lens outlines | Upper rim and fine lower support |
| Rimless | Both lens outlines | Separate lenses, bridge and endpieces |
| Sports/shield | `shield` for one continuous shield; `front` + both lenses for dual lenses | Curved shield or dual-lens wrap geometry |

Part left/right refers to the left/right side of the source front image. Output +X is the wearer's left. All front/lens boundaries must come from the same straight front fitting image. An angled image cannot serve as a straight front reconstruction input.

For `temple-left` / `temple-right`, trace the arm on its side photo and mark `hinge` at the center of its root and `tip` at its far end. Front hinge anchors must be on real outer frame material, not inside a lens opening. Without an explicit front anchor, the outer shoulder attachment is inferred. Connectors sit behind the frame and arm cross-sections are centered on the joint. Apparent straight-shaft pitch is leveled conservatively and recorded as unverified; a nonstandard intentional shaft angle needs calibrated evidence. For `top`, trace the top silhouette and mark `bridge`, `hinge-left` and `hinge-right`; `tip-left`/`tip-right` improve spread constraints. The top view provides weak-perspective curvature/spread constraints, not a full calibrated multi-camera solution. If side/top evidence is missing, arm/depth inference is explicitly recorded.

`front_curve` is front-to-hinge wrap depth. `temple_spread` is the inward displacement of each arm from its hinge, not total tip-to-tip width. Lens tint is a visual approximation, not prescription optics.

## References and evidence

Public reference cache: `data/references/manifest.json`. Each entry retains source-page/image URLs, published measurements, available views and limitations. Fetch again with:

```sh
.venv/bin/python scripts/fetch_references.py
```

While the app is running, import examples and generate numerical fixtures:

```sh
.venv/bin/python scripts/seed_demo.py
```

For the real metal-frame correction experiment:

```sh
.venv/bin/python -m scripts.review_reference
# Inspect artifacts/public-metal-correction-overlay.png first.
.venv/bin/python -m scripts.review_reference --approve
```

That experiment repairs lens aperture contours after visual review; it is **guided correction**, not an automatic success. Hidden depth remains inferred. Current exports omit unverified arms (older exports included procedural arms). The angled photo is withheld, but an uncalibrated product image cannot establish ground-truth 3D accuracy.

## Checks

```sh
.venv/bin/python -m pytest tests -q
npm run build
# Keep scripts/dev.py running for browser checks:
npm run test:browser
node scripts/editor-test.mjs
node scripts/pose-test.mjs
# Requires the local test portrait described in THIRD_PARTY.md:
node scripts/ar-test.mjs
```

See `PROGRESS.md` for module checkmarks and `VALIDATION_REPORT.md` for evidence and remaining research. Screenshots and generated benchmark results are in `artifacts/`.

## Data and service boundaries

- `data/studio.sqlite3` stores projects and jobs. `data/projects/` holds source images; `data/artifacts/` holds versioned GLB/metadata outputs.
- Jobs snapshot the exact revision. A single locked worker processes the queue. Interrupted jobs become retryable failures on restart.
- Removing a photo removes it from the current project; its bytes remain for old immutable job snapshots.
- This is a loopback-only single-user demo. It deliberately has no merchant accounts or production tenant isolation. Do not bind it to a public interface.
- Real product photos are source-owner content cached for local research, not a redistributable catalog. See `THIRD_PARTY.md`.

## What is not proven

The demo implements geometric construction and guided editing. It does not prove fully automatic exact reconstruction across arbitrary frames, solve general camera calibration from arbitrary images, recover hidden manufacturing geometry, or certify physical fit. Transparent materials, reflections, asymmetric components and deeply curved frames still need better captures and visual review. Collection of many angles does not imply all of them are jointly optimized: front outlines, approved side profiles and top anchors drive the current geometry; remaining angles support review.

### SAM 3 on local CPU

SAM 3 uses automatic text concepts for lenses and temple arms in an isolated
Python 3.12 environment. The image encoder runs at 1008 pixels, on CPU in float32
with two Torch threads. Frame, bridge, nose-pad and hinge concepts now feed
source-edge, optical-location and topology checks. Accepted components become
separate, unapproved outlines and can generate textured geometry after review.
A visible brow/support piece is represented separately from a complete outer
frame; a filled lens mask cannot establish frame material. Ambiguous or unsupported
hardware remains uncertain. Component thickness and hidden mechanics are inferred.

Extra views can supply missing components only after source validation and a
supported image registration. Transferred geometry retains its source photo and
is explicitly marked inferred. At most two eligible extra views are analyzed,
sequentially; held-out validation photos are excluded. This remains experimental
and does not establish universal eyewear recognition or physical fit accuracy.

Install the runtime without changing the SAM 2 environment:

```sh
artifacts/eval-tools/bin/uv venv artifacts/sam3-venv --python 3.12
artifacts/eval-tools/bin/uv pip install --python artifacts/sam3-venv/bin/python -r requirements.sam3.txt
```

Request access and accept the publisher terms at https://huggingface.co/facebook/sam3,
then authenticate on this machine and prepare the checkpoint:

```sh
artifacts/sam3-venv/bin/hf auth login
artifacts/sam3-venv/bin/python scripts/setup_sam3.py
.venv/bin/python scripts/dev.py --sam3
```

The official checkpoint is installed on this development machine. Native CPU
image inference completed on one saved front photo in approximately 100 seconds
including startup/checkpoint verification, with about 2.9 GiB peak resident process
memory sampled each second. The app health endpoint remained responsive. The
initial compound prompt `eyeglass lens` missed the lens pair on that photo.
A prompt comparison found `lens` produced two high-scoring masks; replaying these
actual CPU masks through extraction produced two source-checked lens outlines.
The complete outer frame remained unresolved. The default now uses `lens`,
with the same source-edge and topology validators; this is not a universal
eyewear detection or reconstruction accuracy result.
A second saved round-frame photo passed live API extraction with front and both
lens outlines in 118 seconds, with 2.83 GiB peak resident worker memory. Neither
test changed saved annotations or generated artifacts. Results and overlays are
in ignored `artifacts/sam3/first-cpu-test/`. Missing weights produce an explicit
unavailable report without claiming AI ran.

Engine 42's live CPU extraction of the saved `kjjk` front photo took about
121 seconds and 2.89 GiB peak resident worker memory. Both lenses, a connected
front support and the right nose pad passed checks. The left pad and hinges
remained unresolved; the extra views did not provide safe registered geometry.
An isolated, temporarily approved test snapshot exported closed, textured support
and pad meshes without changing the merchant's saved annotations or approvals.
Diagnostics and isolated exports are in ignored `artifacts/sam3/components42/`.

Engine 43 rejects elongated pad proposals that follow an optical boundary,
including the two inner-rim strips proposed as pads on the saved `;;` photo.
Curve finishing now fades smoothing around protected corners rather than
switching it on abruptly. An isolated export without those false pad pieces
passes the geometry checks; saved merchant annotations are unchanged.
A magnified crop of that product's exposed hinge returned no masks for `hinge`,
`glasses hinge` or `metal hinge`. Hinge recognition remains unresolved; the
pipeline must not treat the inferred joint as recovered hardware.

Setup pins official code and checkpoint revisions, verifies the checkpoint SHA256,
and records hashes of CPU adaptations: CPU coordinate caches, memory-mapped
checkpoint loading, float32 linear/GELU on CPU, unpinned CPU box scaling,
and a SciPy distance transform replacing an optional CUDA tracker dependency. Inference makes no downloads and sends no product photos
to external services. CPU analysis may take minutes and uses substantially more
RAM than SAM 2 Small. The worker uses reduced scheduling priority and stops if
resident process memory exceeds 5 GiB (or 62.5% of RAM on smaller machines).
This limit does not guarantee that other applications will remain responsive.
Only one local segmentation process runs at a time, with a 900-second timeout
(configurable with `EYEWEAR_SAM3_TIMEOUT`, maximum 1800).

Verify installed CPU adaptations without loading the full checkpoint:

```sh
artifacts/sam3-venv/bin/python scripts/verify_sam3_cpu.py
```

## Engine 44: opt-in multi-view front fitting experiment

This is a bounded prototype, not a complete faithful photo-to-3D system. Existing
outline approvals and dimension requirements remain. It jointly optimizes a
shared parabolic front wrap and weak-perspective cameras using distinctive
front-material details in front-left/front-right fitting photos. The straight
front fixes XY and scale. It does not optimize arm pose, frame/lens thickness,
nose-pad depth or hidden hinge mechanisms, and adds no new AI model or GPU requirement.

To test:

1. Restart the API and worker with `.venv/bin/python scripts/dev.py` after stopping
   the existing development session; refresh the browser.
2. Open a project with approved straight-front and side outlines and dimensions.
   Add at least one **Front left / Front right** fitting photo; both are preferable.
3. Enable **Multi-view front fitting · experiment**, then Generate / Regenerate.
4. Inspect **Fit result**, serialized GLB checks and clickable zoom/overlay images.
   If the fit says **withheld**, the original front curvature was retained.

Explicit measured/published/manual front curvature and approved top-view geometry
are never overwritten. Blank or unrelated views, insufficient broad matches,
weak curvature sensitivity, poor withheld-feature residuals or lack of improvement
withhold the fit. Withheld features come from the same photos; they are not an
independent product benchmark. Source-outline overlap is a construction consistency
check, not accuracy. Zoom checks inspect native source crops and report edge support;
they do not modify approved outlines or infer missing details.

The experiment reviews the serialized GLB for finite geometry and preserved
approved components. Close-ups cover frame ends, observed hardware/pads and arm
hinge/tip regions. Existing automatic boundary refinement still runs separately.
A ready-made model can remain wrong despite passing these checks.


## Engine 45: CPU thin-metal pilot

Front and Advanced analysis explicitly select `cpu-metal`; SAM 3 is not loaded
even if the server was started with `--sam3`. Run `.venv/bin/python scripts/dev.py`
without that flag to keep the default analysis on CPU contours. Restart
the API and worker after Python changes and refresh the browser.

The CPU pilot uses stable tinted-lens interiors, narrow source-supported rims
and a continuous source ridge for the bridge. It can retain a separately observed
upper bar and visible hinge silhouettes. Native source zoom checks remain available.
Approved bridge outlines now build a photographed bridge profile; hidden depth
still uses an estimate. All proposals remain unapproved until reviewed.

Scope: two visible tinted lenses on a quiet light background. Clear/open lens
interiors, severe occlusion and unsupported wire paths fall back to standard CPU
contours. This is not a universal detector or a demonstrated 90% fidelity pass.
Side contour analysis is unchanged. The aviator pilot export includes both pads,
rims, bridge, upper bar and arms, with inferred dimensions; no saved project
annotations or approvals were changed by that local test.

See [local CPU evidence](reports/cpu-metal-preview/index.html).

## Engine 46: clear lens edges and subpixel side roots

The CPU metal path now also checks closed optical apertures across contrast
levels, with source-edge support, when tinted-interior separation fails. This
supports visible clear lenses inside narrow metal rims; it does not reconstruct
missing edges or establish universal performance on clear/rimless products.

Generation seats a hinge anchor at most **0.1 source pixels** outside an approved
side contour on that existing contour, without changing saved data or expanding
the mesh outline. Larger errors still require correction and now include useful
photo/part/distance details. Side smoothing retains a covered root/tip anchor.
The saved `lll` project regenerated successfully in an isolated export:
[temple root fix evidence](reports/hinge-root-fix/index.html).

## Engine 47: transparent shadows and gradient rimless optics

The front finisher now removes long neutral ground shadows below a supported
lower rim, including narrow contact points that previously left horizontal
ledges in the proposal. The cut is limited to quiet light backgrounds, lower
lens arcs, source contrast and the observed rim clearance. Inspect transparent
material at each edge before approval. The saved `fvgb` front was checked with
Legacy lens proposals, which remain optional; automatic CPU contour detection
alone did not identify its complete front.

A separate CPU route proposes paired gradient rimless lens edges and a visible
bridge when several edge settings agree, the exterior is quiet and nasal
mounting leaves are visible. It adds no outer frame and does not infer missing
nose pads or arms. The saved `ppp` front now yields two lenses and bridge; an
isolated, assumed-size GLB confirms they serialize, but remains incomplete.
See [source overlays](reports/outline-fix-preview/index.html).

The eyewear-specific dataset audit, guarded training entry point, and
non-default candidate inference adapter are documented in
[docs/eyewear-training.md](docs/eyewear-training.md). The current approved
photos lack hinge and rimless-mount labels, so no trained model is active.
