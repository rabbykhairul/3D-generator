# Third-party assets and sources

- Optional SAM inference uses Meta's [SAM 2.1](https://github.com/facebookresearch/sam2) Hiera Small checkpoint and code revision `2b90b9f5ceec907a1c18123530e92e794ad901a4`. The publisher licenses its code and checkpoints under Apache-2.0. The notice, publisher URL and pinned SHA256 are retained in local `artifacts/sam2/`; `scripts/setup_sam2.py` prepares them without uploading photographs. The existing glasses-detector model supplies coarse prompts only in this mode; its separate provenance remains below.

- Optional CPU evaluation uses `glasses-detector==1.0.4` from [mantasu/glasses-detector](https://github.com/mantasu/glasses-detector), and the publisher's v1.0.0 small/medium frames, lenses and legs checkpoint assets. The repository MIT notice is retained in local `artifacts/segmenter-evaluation/PUBLISHER_LICENSE.txt`; exact checkpoint URLs and SHA256 values are in the generated manifests. These are local research assets, not bundled production models. Separate checkpoint/training-data commercial redistribution rights have not been established by this evaluation.

- `frontend/public/upload-guide/{front,side,angle}.png` are the three catalog photographs supplied by the user in this conversation, copied unchanged for the standalone local demo's upload guidance. They are examples, not proof of automatic-detection coverage. Redistribution rights are not established; replace with owned/licensed photography before public release.

- Three.js, React, Vite and Python library licenses remain with their packages. Dependency versions are locked locally.
- MediaPipe Face Landmarker task and WASM files were copied from the existing sibling AR prototype, which uses the same `@mediapipe/tasks-vision` version. Source documentation: https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker/web_js . Review upstream model/package licensing before redistribution.
- Retailer product photos are downloaded from product pages and their referenced image URLs only. URLs, retrieval timestamps and dimensions are retained in `data/references/manifest.json`. They are local research evidence, not a licensed app catalog. No customer photos are collected.
- AR test portrait: Grace Hopper, Matplotlib sample data, retrieved from https://raw.githubusercontent.com/matplotlib/matplotlib/v3.10.0/lib/matplotlib/mpl-data/sample_data/grace_hopper.jpg . Stored in ignored local `artifacts/` for browser testing only. The portrait is test evidence and never uploaded by the AR viewer.
- Synthetic fixtures are generated numerical outlines in `backend/samples.py` and are expressly labeled synthetic throughout the interface.
- The user-supplied `head-occluder.glb` and the prototype's `canonical_face_model.obj` were copied from `/Users/dekuddus/Desktop/try-on-glass-prototype/public/models/` into this demo's `frontend/public/models/` for local AR testing. The source prototype was not modified. The canonical topology is used with MediaPipe landmarks; the skull is an approximate registered shape, not a scan of the wearer. Asset redistribution rights have not been established here and must be reviewed before shipping.
- Same-product Lenskart Air LA E13069-C6 reference photos are local fidelity research evidence, with retrieval/source records in `data/references/`. Neither downloaded images nor private project textures are licensed by this implementation for public redistribution.

## Meta SAM 3 (optional CPU runtime)

Official source: https://github.com/facebookresearch/sam3, revision
`2345a4ad109ac29c569da749c91d84f10dc08c40`. Official gated checkpoint:
https://huggingface.co/facebook/sam3, revision
`3c879f39826c281e95690f02c7821c4de09afae7`. SAM License terms apply;
setup saves the publisher license alongside the checksum manifest. This is
not the Apache-2.0 SAM 2 license. Weights are not bundled with the app.
Local CPU adaptations and their hashes are recorded by `scripts/setup_sam3.py`.
