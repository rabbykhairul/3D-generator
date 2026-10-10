# Eyewear component training pilot

This pilot trains a **seven-channel visible-component segmenter** for lenses,
frame material, bridge, nose pads, hinges, temple arms, and lens-mounted
attachment hardware. The frame channel can be empty for a genuinely rimless
product. A missing mask is **unknown**, never evidence that a part is absent.
Only a reviewed `absent` entry means confirmed absence.

The model is a component proposal source, not a photo-to-GLB model. The app
still checks masks against source pixels and topology. It does not infer
hidden depth, accurate thickness, material, or fit from these labels.

## Available data and current block

The pilot began with nine reviewed photos from five products. One rimless front
photo has a visually checked lens mask and a confirmed absent frame; its bridge,
mounts, hinges, nose pads, and temples remain unknown. Twenty-four local
product photos and 120 Lenskart research photos were staged. Four research
photos now have source-reviewed labels; the remaining photos are unreviewed.
The research crawl collected 16 product pages; five other page records failed
(four HTTP 404s and one DNS error). The 120 files decoded successfully, but
most product identities, views, and component masks have not been visually
verified. Color variants have distinct product IDs while sharing their model's
train/validation/test split. Photos of the four
previously approved products were matched to their downloaded copies and
grouped under the same product IDs to prevent train/test leakage.

The exploratory split still has very few reviewed products. It has no reviewed
bridge, hinge, or lens-mount mask, and
no rimless validation/test product. It is insufficient to benchmark the
requested recognizer. An attempt to fetch public datasets from this environment
failed because DNS resolution is unavailable. Commercial training rights for
the downloaded merchant photos also require verification.

Useful external starting points include the
[Glasses Detector dataset collection](https://github.com/mantasu/glasses-detector)
(frame, lens, arm and shadow masks) and
[STRAT-glasses](https://github.com/bcmi/Accessory-Try-On-Dataset-STRAT)
(2D full-glasses masks). These do not supply verified hinge, nose-pad,
lens-drill mount and multi-view 3D labels for the full goal. Check the rights
of each underlying dataset and any pretrained backbone before commercial use.

## Prepare the dataset

```bash
.venv/bin/python scripts/eyewear_dataset.py export-approved
.venv/bin/python scripts/eyewear_dataset.py audit
```

For a local research collection of public Lenskart product views, the
JavaScript collector traverses the site's product sitemaps, visits discovered
product pages, captures Chrome image responses, and ranks image URLs by SKU
or product-name overlap. It saves the response bytes and a resumable manifest
with source URLs and hashes. Start with a small limit to inspect the results:

```bash
node scripts/collect-lenskart-playwright.mjs \
  --sitemap-url 'https://www.lenskart.com/sitemap.xml' \
  --max-products 20 --max-images 8
.venv/bin/python scripts/eyewear_dataset.py import-research-collection \
  --collection artifacts/eyewear-training/lenskart-research/collection.json
```

Omit `--max-products`, or use `--max-products 0`, to visit every product URL
discovered from the sitemap. `--product-url` and `--listing-url` can provide
additional starting pages. The manifest distinguishes image bytes captured
from Chrome's network responses from fallback downloads of URLs present in
the product page. Sitemap coverage, page loading, and SKU/name matching may
still miss products or photos, so audit the manifest before using it as a
dataset. The collector does not assign component masks or view labels.

All collected photos remain **unreviewed and unlabeled**. Their view names,
product grouping and reuse rights must be checked before annotation or a
commercial training run. The collector does not bypass access controls. The
Playwright collector uses the project's existing `@playwright/test` dependency;
`--cdp-url` can connect to an already running debugging browser. In this
sandbox, fresh Chrome/Chromium processes exit before a page opens; the 120
research photos came from the user's local browser run. The plain HTTP
collector's GET request fails DNS here.

The manifest is under `artifacts/eyewear-training/manifest.json`, outside
version control.

For automatic triage without a manual review session:

```bash
.venv/bin/python scripts/auto_triage_eyewear_research.py
```

This CPU pass generated 21 new unapproved drafts across the 120 research
photos (nine possible fronts and 12 side-arm traces), found 11 pre-existing
drafts, withheld 83 unsuitable or unresolved photos, and left four approved
photos plus one skipped photo untouched. The report is at
`artifacts/eyewear-training/auto-triage-report.json`. It created **zero**
training labels. The side branch name does not independently verify anatomical
left/right orientation. Its thin-wire outer contours cannot be turned into
frame masks by filling the polygon. No automatic bridge, hinge or lens-mount
labels were established. Self-training on these proposals would reproduce the
current detector's mistakes, not validate or solve them.

For the staged Lenskart photos, open the local review queue:

```bash
.venv/bin/python scripts/review_eyewear_collection.py
```

Open `http://127.0.0.1:5193/`. Choose the actual front/left/right view or
skip a lifestyle or unsuitable image. Generate a CPU outline draft only after
choosing the view. Zoom into transparent rims, lens edges, hinges and shadows;
edit polygons and choose the frame construction. **Save draft** preserves work
without entering training. **Approve masks** validates the polygons and saves
source-sized PNG masks to the dataset manifest. Unknown or hidden components
remain unlabeled. Rimless requires an explicit absent-frame confirmation.
This queue is local-only and does not contact an external annotation service.
Its HTTP port could not be opened from this execution sandbox; the mask and
dataset functions were tested directly.

Three pilot front photos, including opaque and transparent full rims, have
source-reviewed frame and lens masks. One side view has a source-reviewed temple
mask. An angled product photo was skipped after its front-view draft proved
unsuitable.
An isolated five-epoch CPU pass confirmed that the new approved masks flow into
the trainer, but the unchanged one-product test scores are still only lens
0.7045, frame 0.1794 and temple 0.3503 IoU. The pilot does not supply held-out
hinge, bridge or lens-mount labels and is not ready for a full training run.

To add a public mask collection after downloading it:

```bash
.venv/bin/python scripts/eyewear_dataset.py import-public \
  --images /path/to/images --masks /path/to/masks \
  --class-name lens --source-name source-v1 \
  --license-note 'Recorded dataset terms and attribution'
```

Imported items remain **unreviewed**. Inspect the source and mask, assign the
same `product-id` to all views of one physical pair, and mark the actual
construction. For rimless products, use `--absent frame` only after visual
verification. Public near-duplicates must be grouped under one product ID so
they cannot leak across train/validation/test. Example:

```bash
.venv/bin/python scripts/eyewear_dataset.py review-item \
  --source-name source-v1 --photo-id image-001 --product-id physical-001 \
  --view front --construction rimless --absent frame
```

For COCO polygon-segmentation archives, map the dataset's category names
explicitly. For example, after extracting a rights-cleared archive:

```bash
.venv/bin/python scripts/eyewear_dataset.py import-coco \
  --images /path/to/archive/train \
  --annotations /path/to/archive/train/_annotations.coco.json \
  --category-map 'glasses lens:lens' \
  --source-name example-v1 \
  --license-note 'Record the dataset-specific terms and attribution here'
```

This importer rejects path escapes and dimension mismatches, skips COCO RLE
annotations, and leaves all imported photos unreviewed. Check the skipped
count and masks before review. A repository's open-source **code** license
does not establish permission for every underlying image dataset it lists.
Do not map a whole-glasses silhouette to `frame`, or a frame shadow to any
eyewear component.

For the missing tiny hardware classes, add reviewed source-sized PNG masks
through the manifest or an annotation tool. Source images and masks must have
the same dimensions. Mark unavailable classes as unknown, especially behind
transparent lenses. Capture front, left and right views of varied products;
measure a subset of actual pairs for later 3D validation. Manufacturer CAD
is helpful but not required to label visible components.

In Advanced, `lens-mount-left` and `lens-mount-right` can mark visible outer
attachments where a rimless lens meets its arm. These outlines become training
labels; they do not claim the hidden hinge shape or add geometry to a GLB.

## Train and evaluate

`audit` refuses the training run until all seven classes and rimless examples
appear in **each** product-disjoint train/validation/test split.
This is a **minimum completeness check**, not evidence of accuracy.

```bash
artifacts/segmenter-venv/bin/python scripts/train_eyewear_components.py \
  --epochs 20 --size 768 --device cpu \
  --backbone-checkpoint artifacts/segmenter-evaluation/weights/segmentation_frames_lraspp_mobilenet_v3_large.pth
```

The optional local backbone was obtained for earlier research evaluation;
its checkpoint and training-data redistribution rights require separate
review before commercial deployment. Without that option the network trains
from random initialization, which is unlikely to generalize from a small
pilot. The run writes `candidate.pt` and `report.json` under
`artifacts/eyewear-training/run-1`. The report includes per-class test IoU;
it deliberately sets `release_gate_passed` to false. Review overlays,
failure families, product-level leakage, source photos, and generated GLBs
against measured held-out pairs before deployment.

The app has a **non-default experimental API backend** named
`eyewear-trained`. It loads a candidate only when `EYEWEAR_TRAINED_MODEL`
points to a local checkpoint. It is not shown in the merchant UI before a
model is trained and validated. It can propose source-checked front components
and side arms; it does not auto-approve them. Never infer that a rimless frame
exists from a missing outer contour alone.

### Exploratory CPU run (8 October 2026)

An isolated five-epoch, 256 px run used the existing local backbone and the
product-disjoint pilot split:

```bash
artifacts/segmenter-venv/bin/python scripts/train_eyewear_components.py \
  --exploratory --epochs 5 --size 256 --device cpu \
  --output artifacts/eyewear-training/exploratory-1 \
  --backbone-checkpoint artifacts/segmenter-evaluation/weights/segmentation_frames_lraspp_mobilenet_v3_large.pth
```

It completed in about 18 seconds. On **one** held-out product, lens IoU was
0.7045, frame IoU 0.1794, and temple IoU 0.3503. There is no held-out score
for bridge, nose pad, hinge, or lens mount. On the training rimless photo the
model drew false nose-pad regions and almost no temple; its confirmed-absent
frame still had 232 false-positive pixels at 256 px evaluation size. The
single labeled training nose-pad photo reached only 0.0167 IoU. This visual
check is not a generalization test. These numbers are not an accuracy claim. The
checkpoint is explicitly marked `exploratory_only` and normal app inference
rejects it. The full report is at
`artifacts/eyewear-training/exploratory-1/report.json`.

Next, obtain rights-cleared component masks, with multiple **different**
rimless products and all seven components represented across product-disjoint
train, validation, and test sets. Review small hardware at source resolution
before training again. The held-out score and overlays, rather than a training
loss alone, determine whether the model can become an app proposal source.
