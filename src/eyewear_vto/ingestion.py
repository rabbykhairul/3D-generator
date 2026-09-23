from __future__ import annotations

import hashlib
import json
import re
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from fastapi import UploadFile
from PIL import Image, ImageFilter, ImageOps, UnidentifiedImageError

from eyewear_vto.config import Settings

MANDATORY_VIEWS = ("front", "left", "right")
OPTIONAL_VIEWS = ("back", "hero")
ALL_VIEWS = MANDATORY_VIEWS + OPTIONAL_VIEWS
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}
FORMAT_EXTENSIONS = {"JPEG": "jpg", "PNG": "png", "WEBP": "webp"}
SAFE_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class InputValidationError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


@dataclass(frozen=True)
class ValidatedImage:
    view: str
    raw_path: Path
    normalized_path: Path
    sha256: str
    width: int
    height: int
    source_format: str


@dataclass(frozen=True)
class InputBundle:
    workspace: Path
    images: Mapping[str, ValidatedImage]


def validate_identifier(value: str, field: str, max_length: int) -> str:
    if not value or len(value) > max_length or SAFE_IDENTIFIER.fullmatch(value) is None:
        raise InputValidationError(
            "invalid_identifier",
            f"{field} must be 1-{max_length} URL-safe characters",
        )
    return value


async def ingest_uploads(
    *,
    job_id: str,
    merchant_id: str,
    product_sku: str,
    uploads: Mapping[str, UploadFile | None],
    settings: Settings,
) -> InputBundle:
    validate_identifier(merchant_id, "merchant_id", 64)
    validate_identifier(product_sku, "product_sku", 96)
    missing = [view for view in MANDATORY_VIEWS if uploads.get(view) is None]
    if missing:
        raise InputValidationError("missing_view", f"Missing mandatory views: {', '.join(missing)}")

    workspace = settings.work_root / job_id
    raw_dir = workspace / "raw"
    normalized_dir = workspace / "normalized"
    if workspace.exists():
        raise InputValidationError("workspace_exists", "A workspace already exists for this job")
    raw_dir.mkdir(parents=True, exist_ok=False)
    normalized_dir.mkdir(parents=True, exist_ok=False)

    images: dict[str, ValidatedImage] = {}
    request_bytes = 0
    try:
        for view in ALL_VIEWS:
            upload = uploads.get(view)
            if upload is None:
                continue
            if upload.content_type not in ALLOWED_CONTENT_TYPES:
                raise InputValidationError("unsupported_media_type", f"{view} must be JPEG, PNG, or WebP")

            temporary_path = raw_dir / f"{view}.upload"
            digest = hashlib.sha256()
            file_bytes = 0
            with temporary_path.open("xb") as output:
                while chunk := await upload.read(1024 * 1024):
                    file_bytes += len(chunk)
                    request_bytes += len(chunk)
                    if file_bytes > settings.max_upload_bytes:
                        raise InputValidationError("file_too_large", f"{view} exceeds the per-file limit")
                    if request_bytes > settings.max_request_bytes:
                        raise InputValidationError("request_too_large", "Combined uploads exceed the request limit")
                    digest.update(chunk)
                    output.write(chunk)
            await upload.close()

            validated = validate_and_normalize_image(
                view=view,
                temporary_path=temporary_path,
                normalized_dir=normalized_dir,
                settings=settings,
                sha256=digest.hexdigest(),
            )
            images[view] = validated

        manifest_path = workspace / "input_manifest.json"
        manifest = {
            "schema_version": 1,
            "merchant_id": merchant_id,
            "product_sku": product_sku,
            "images": {
                view: {
                    "raw_path": str(image.raw_path),
                    "normalized_path": str(image.normalized_path),
                    "sha256": image.sha256,
                    "width": image.width,
                    "height": image.height,
                    "source_format": image.source_format,
                }
                for view, image in images.items()
            },
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        return InputBundle(workspace=workspace, images=images)
    except Exception:
        shutil.rmtree(workspace, ignore_errors=True)
        raise


def validate_and_normalize_image(
    *,
    view: str,
    temporary_path: Path,
    normalized_dir: Path,
    settings: Settings,
    sha256: str,
) -> ValidatedImage:
    try:
        with Image.open(temporary_path) as candidate:
            source_format = candidate.format
            candidate.verify()
        if source_format not in FORMAT_EXTENSIONS:
            raise InputValidationError("unsupported_image", f"{view} has unsupported encoded image data")

        extension = FORMAT_EXTENSIONS[source_format]
        raw_path = temporary_path.with_suffix(f".{extension}")
        temporary_path.rename(raw_path)

        with Image.open(raw_path) as source:
            image = ImageOps.exif_transpose(source)
            image.load()
            width, height = image.size
            if max(width, height) < settings.min_image_long_edge:
                raise InputValidationError("image_too_small", f"{view} is below the minimum resolution")
            if max(width, height) > settings.max_image_dimension:
                raise InputValidationError("image_too_large", f"{view} exceeds the maximum dimensions")
            if not has_visible_edges(image):
                raise InputValidationError("no_subject_edges", f"{view} does not contain enough visible structure")

            normalized_path = normalized_dir / f"{view}.png"
            normalized = image.convert("RGBA" if "A" in image.getbands() else "RGB")
            normalized.save(normalized_path, format="PNG", optimize=True)
    except InputValidationError:
        raise
    except (Image.DecompressionBombError, UnidentifiedImageError, OSError) as exc:
        raise InputValidationError("invalid_image", f"{view} is not a valid decodable image") from exc

    return ValidatedImage(
        view=view,
        raw_path=raw_path,
        normalized_path=normalized_path,
        sha256=sha256,
        width=width,
        height=height,
        source_format=source_format,
    )


def has_visible_edges(image: Image.Image) -> bool:
    thumbnail = ImageOps.grayscale(image)
    thumbnail.thumbnail((256, 256))
    edges = thumbnail.filter(ImageFilter.FIND_EDGES)
    histogram = edges.histogram()
    pixel_count = max(1, edges.width * edges.height)
    strong_edge_pixels = sum(histogram[32:])
    return strong_edge_pixels / pixel_count >= 0.002

