# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Small client and media helpers shared by the Cosmos3 NIM examples."""

import base64
import binascii
import json
from pathlib import Path

import requests

_MIME_TYPES = {
    ".jpeg": "image/jpeg",
    ".jpg": "image/jpeg",
    ".mp4": "video/mp4",
    ".png": "image/png",
    ".webp": "image/webp",
}


def require_runtime(
    nim_url: str,
    *,
    expected_runtime: str,
    expected_endpoint: str,
) -> dict:
    """Fail before inference when ``nim_url`` targets the wrong runtime."""
    response = requests.get(f"{nim_url}/v1/metadata", timeout=30)
    response.raise_for_status()
    metadata = response.json()
    if not isinstance(metadata, dict):
        raise TypeError("/v1/metadata must return a JSON object")

    actual_runtime = metadata.get("model_type")
    actual_endpoint = metadata.get("inference_endpoint")
    if actual_runtime != expected_runtime or actual_endpoint != expected_endpoint:
        raise RuntimeError(
            f"Expected the {expected_runtime!r} runtime at {nim_url}, but "
            f"/v1/metadata reported model_type={actual_runtime!r} and "
            f"inference_endpoint={actual_endpoint!r}. Start the correct runtime "
            "or update NIM_URL."
        )
    return metadata


def require_generator_profile(
    nim_url: str,
    *,
    allowed_variants: tuple[str, ...],
) -> dict:
    """Require a selected Generator profile compatible with an example."""
    metadata = require_runtime(
        nim_url,
        expected_runtime="generator",
        expected_endpoint="/v1/infer",
    )

    selected_profile_id = metadata.get("selectedModelProfileId")
    if not isinstance(selected_profile_id, str) or not selected_profile_id:
        raise RuntimeError(
            "/v1/metadata did not report a non-empty selectedModelProfileId. "
            "Confirm the selected profile before sending an inference request."
        )

    model_variant = metadata.get("model_variant")
    if model_variant not in allowed_variants:
        expected = ", ".join(repr(variant) for variant in allowed_variants)
        raise RuntimeError(
            f"The selected Generator profile uses model_variant={model_variant!r}; "
            f"this example requires one of: {expected}. Start a compatible "
            "Generator model or choose its matching example."
        )
    return metadata


def compact_json_file(path: Path) -> str:
    """Load a JSON asset and return the compact string expected by Generator."""
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"JSON file does not exist: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"))


def media_to_data_url(path: Path) -> str:
    """Read a supported local image or video as a base64 data URL."""
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Media file does not exist: {path}")
    try:
        mime_type = _MIME_TYPES[path.suffix.lower()]
    except KeyError as exc:
        supported = ", ".join(sorted(_MIME_TYPES))
        raise ValueError(f"Unsupported media type; use one of: {supported}") from exc
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def decode_video(encoded_video: str) -> bytes:
    """Decode a raw base64 video or base64 video data URL."""
    if not isinstance(encoded_video, str):
        raise TypeError("b64_video must be a string")
    if encoded_video.startswith("data:"):
        header, separator, encoded_video = encoded_video.partition(",")
        if not separator or ";base64" not in header:
            raise ValueError("Malformed base64 video data URL")
    try:
        video = base64.b64decode(encoded_video, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("b64_video is not valid base64") from exc
    if not video:
        raise ValueError("b64_video decoded to an empty video")
    return video


def decode_image(encoded_image: str) -> bytes:
    """Decode a raw base64 image or base64 image data URL."""
    if not isinstance(encoded_image, str):
        raise TypeError("b64_image must be a string")
    if encoded_image.startswith("data:"):
        header, separator, encoded_image = encoded_image.partition(",")
        if not separator or ";base64" not in header:
            raise ValueError("Malformed base64 image data URL")
    try:
        image = base64.b64decode(encoded_image, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("b64_image is not valid base64") from exc
    if not image:
        raise ValueError("b64_image decoded to an empty image")
    return image
