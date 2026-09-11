# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Small client and media helpers shared by the Cosmos3 NIM examples."""

import base64
import binascii
import copy
import json
import os
import time
from pathlib import Path

import requests

_MIME_TYPES = {
    ".jpeg": "image/jpeg",
    ".jpg": "image/jpeg",
    ".mp4": "video/mp4",
    ".png": "image/png",
    ".webp": "image/webp",
}

DEFAULT_NIM_URL = "http://localhost:8000"


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


def prompt_or_json(value: str) -> str:
    """Return ``value`` verbatim, or the compact JSON string of an existing ``.json`` file.

    CLI ``--prompt`` flags accept either a raw prompt string or the path to a
    prompt asset. A value that names an existing ``.json`` file is loaded and
    compacted (matching ``compact_json_file``); anything else is used as-is.
    """
    candidate = Path(value)
    if candidate.suffix.lower() == ".json" and candidate.is_file():
        return compact_json_file(candidate)
    return value


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


# ---------------------------------------------------------------------------
# Additive shared helpers (opt-in; existing scripts can adopt them gradually)
# ---------------------------------------------------------------------------


def get_default_nim_url() -> str:
    """Return the NIM base URL from ``NIM_URL`` or the default, without a trailing slash."""
    return os.environ.get("NIM_URL", DEFAULT_NIM_URL).rstrip("/")


#: HTTP status codes that indicate a transient failure worth retrying.
_RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}


def nim_infer(
    nim_url: str,
    *,
    request: dict,
    timeout: int = 1800,
    retries: int = 2,
    base_delay: float = 2.0,
) -> dict:
    """POST ``request`` to ``/v1/infer`` and return the JSON object response.

    Transient failures — connection errors and retryable HTTP status codes
    (408, 429, 5xx) — are retried with exponential backoff (``base_delay * 2**n``).
    Non-retryable failures (4xx other than 408/429, malformed responses) fail fast.
    ``retries=0`` disables retrying.
    """
    attempt = 0
    while True:
        try:
            response = requests.post(f"{nim_url}/v1/infer", json=request, timeout=timeout)
        except requests.exceptions.ConnectionError:
            if attempt < retries:
                attempt += 1
                time.sleep(base_delay * (2 ** (attempt - 1)))
                continue
            raise
        if response.status_code in _RETRYABLE_STATUS and attempt < retries:
            attempt += 1
            time.sleep(base_delay * (2 ** (attempt - 1)))
            continue
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise TypeError("/v1/infer must return a JSON object")
        return payload


def default_output_dir(script_file: str | Path) -> Path:
    """Return the ``<examples>/outputs`` directory that holds a script's outputs."""
    return Path(script_file).resolve().parent / "outputs"


def write_media_output(
    output_dir: Path,
    *,
    name: str,
    response_payload: dict,
    key: str,
    unique: bool = False,
) -> Path:
    """Decode a base64 media field, write it under ``output_dir``, and print the path.

    With ``unique=True``, an existing ``name`` is left untouched and a
    ``-1``/``-2``/... suffix is appended so prior runs are never clobbered.
    """
    if key not in response_payload:
        raise ValueError(f"response payload did not contain {key!r}")
    decoder = decode_video if key == "b64_video" else decode_image
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / name
    if unique:
        path = _unique_path(path)
    path.write_bytes(decoder(response_payload[key]))
    kind = key.removeprefix("b64_")
    print(f"Saved {kind} to {path}")
    return path


def _unique_path(path: Path) -> Path:
    """Return ``path``, or a ``stem-1.ext`` variant if ``path`` already exists."""
    if not path.exists():
        return path
    counter = 1
    while True:
        candidate = path.with_name(f"{path.stem}-{counter}{path.suffix}")
        if not candidate.exists():
            return candidate
        counter += 1


def build_generator_infer(base: dict, **overrides) -> dict:
    """Return a deep copy of ``base`` merged with ``overrides``.

    The copy lets callers ``del`` keys (e.g. guidance/flow-shift for 4-step
    profiles) without mutating the shared base template.
    """
    request = copy.deepcopy(base)
    request.update(overrides)
    return request
