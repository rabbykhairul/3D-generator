# Baked model artifact contract

Runtime model downloads are prohibited. The final image contains:

| Artifact | Pinned source | Runtime location |
|---|---|---|
| Hunyuan3D Paint PBR 2.1 | `tencent/Hunyuan3D-2.1` | `/opt/models/huggingface/hub/models--tencent--Hunyuan3D-2.1` |
| DINOv2 Giant | `facebook/dinov2-giant` | `/opt/models/huggingface/hub/models--facebook--dinov2-giant` |
| SAM 2.1 Hiera Large | Meta checkpoint URL | `/opt/models/sam2/sam2.1_hiera_large.pt` |
| Eyewear YOLO pose | project-owned Hugging Face repository | `/opt/models/yolo/eyewear_pose.pt` |
| RealESRGAN x4plus | upstream release | `/opt/models/realesrgan/RealESRGAN_x4plus.pth` |

`/opt/models/model-manifest.json` is created during the build. Standalone weights include SHA-256 values. Hugging Face models use pinned snapshot commits and baked `refs/main` entries so upstream code resolves them while `HF_HUB_OFFLINE=1` is active.

The custom YOLO repository must contain `eyewear_pose.pt`. Its GitHub Actions variables are documented in `docs/ghcr-build.md`.

Model cache directories are copied as separate OCI layers so changing the YOLO model does not invalidate the large Hunyuan layer and the registry never receives the entire model collection as one layer.

Do not add model binaries to this Git repository. They are fetched only in the `model-bake` Docker stage and cached in GHCR BuildKit layers.
