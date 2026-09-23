#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import urllib.request
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

SAM_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"
REALESRGAN_URL = (
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", type=Path, default=Path("/opt/models"))
    parser.add_argument("--hunyuan-revision", required=True)
    parser.add_argument("--dinov2-revision", required=True)
    parser.add_argument("--yolo-repo", required=True)
    parser.add_argument("--yolo-revision", required=True)
    parser.add_argument("--yolo-filename", default="eyewear_pose.pt")
    parser.add_argument("--sam-config", type=Path, required=True)
    return parser.parse_args()


def download_url(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    with urllib.request.urlopen(url) as response, temporary.open("wb") as output:  # noqa: S310
        shutil.copyfileobj(response, output)
    temporary.replace(destination)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def set_main_ref(cache_root: Path, repo_id: str, revision: str) -> None:
    repository = cache_root / f"models--{repo_id.replace('/', '--')}"
    refs = repository / "refs"
    refs.mkdir(parents=True, exist_ok=True)
    (refs / "main").write_text(revision, encoding="utf-8")


def main() -> None:
    args = parse_args()
    token = os.environ.get("HF_TOKEN") or None
    root: Path = args.model_root
    hf_cache = root / "huggingface" / "hub"
    root.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id="tencent/Hunyuan3D-2.1",
        revision=args.hunyuan_revision,
        allow_patterns=["hunyuan3d-paintpbr-v2-1/*"],
        cache_dir=hf_cache,
        token=token,
    )
    set_main_ref(hf_cache, "tencent/Hunyuan3D-2.1", args.hunyuan_revision)

    snapshot_download(
        repo_id="facebook/dinov2-giant",
        revision=args.dinov2_revision,
        cache_dir=hf_cache,
        token=token,
    )
    set_main_ref(hf_cache, "facebook/dinov2-giant", args.dinov2_revision)

    yolo_cached = Path(
        hf_hub_download(
            repo_id=args.yolo_repo,
            filename=args.yolo_filename,
            revision=args.yolo_revision,
            cache_dir=hf_cache,
            token=token,
        )
    )
    yolo_path = root / "yolo" / "eyewear_pose.pt"
    yolo_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(yolo_cached, yolo_path)

    sam_checkpoint = root / "sam2" / "sam2.1_hiera_large.pt"
    sam_config = root / "sam2" / "sam2.1_hiera_l.yaml"
    download_url(SAM_URL, sam_checkpoint)
    sam_config.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.sam_config, sam_config)

    realesrgan = root / "realesrgan" / "RealESRGAN_x4plus.pth"
    download_url(REALESRGAN_URL, realesrgan)

    manifest = {
        "schema_version": 1,
        "models": [
            {
                "name": "hunyuan3d-paintpbr-v2-1",
                "source": "tencent/Hunyuan3D-2.1",
                "revision": args.hunyuan_revision,
                "path": str(hf_cache / "models--tencent--Hunyuan3D-2.1"),
                "required": True,
            },
            {
                "name": "dinov2-giant",
                "source": "facebook/dinov2-giant",
                "revision": args.dinov2_revision,
                "path": str(hf_cache / "models--facebook--dinov2-giant"),
                "required": True,
            },
            {
                "name": "sam2.1-hiera-large",
                "source": SAM_URL,
                "revision": "092824",
                "path": str(sam_checkpoint),
                "sha256": sha256_file(sam_checkpoint),
                "required": True,
            },
            {
                "name": "eyewear-pose",
                "source": args.yolo_repo,
                "revision": args.yolo_revision,
                "path": str(yolo_path),
                "sha256": sha256_file(yolo_path),
                "required": True,
            },
            {
                "name": "realesrgan-x4plus",
                "source": REALESRGAN_URL,
                "revision": "v0.1.0",
                "path": str(realesrgan),
                "sha256": sha256_file(realesrgan),
                "required": True,
            },
            {
                "name": "eyewear-template-catalog",
                "source": "project-build-context",
                "revision": "image-commit",
                "path": "/opt/templates/catalog.json",
                "required": True,
            },
        ],
    }
    (root / "model-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
