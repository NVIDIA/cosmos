# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Small media helpers shared by the Cosmos3 NIM examples."""

import base64
import binascii
from pathlib import Path

_MIME_TYPES = {
    ".jpeg": "image/jpeg",
    ".jpg": "image/jpeg",
    ".mp4": "video/mp4",
    ".png": "image/png",
    ".webp": "image/webp",
}


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
