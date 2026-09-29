# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Single source of truth for the Cosmos3 NIM example request builders.

Each ``build_<script>_request()`` reuses the asset constants already defined in
the example scripts themselves, so a request built here is byte-for-byte
identical to what the script sends. This module imports **no** HTTP code at
module import time — only the scripts' request dicts and asset constants — so
it is safe to import from tests or the batch runner without a live NIM.

The example scripts remain fully runnable on their own; this module exists so
a request can be inspected (and the planned ``--check`` path validated) without
posting to ``/v1/infer``.
"""

from __future__ import annotations

import importlib

import settings


def _import_script(module_name: str) -> object:
    """Import an example script module by name (never by local ``__file__``)."""
    return importlib.import_module(module_name)


def _path_of(module: object, attr: str):
    """Resolve a module attribute that holds a ``Path`` (or iterable of paths)."""
    value = getattr(module, attr)
    if isinstance(value, str):
        return value  # URL constant, not a local path
    return value


def _flow(name: str) -> dict:
    """Return the shared generation defaults for ``name`` (with env overrides)."""
    return settings.flow_defaults(name)


# --- Per-script builders ----------------------------------------------------
#
# Each builder mirrors the exact dict the corresponding script sends. It is
# defined here (not by calling a script function) because the scripts build
# their requests inline in ``main()``. Reading the shared constants keeps the
# two in lockstep: if a script's canonical asset changes, this changes with it.


def build_t2v_request() -> dict:
    t2v = _import_script("t2v")
    from common import compact_json_file

    return {
        "model_mode": "text2video",
        "prompt": compact_json_file(_path_of(t2v, "PROMPT")),
        "negative_prompt": compact_json_file(_path_of(t2v, "NEGATIVE_PROMPT")),
        **_flow("text2video"),
    }


def build_t2i_request() -> dict:
    t2i = _import_script("t2i")
    from common import compact_json_file

    return {
        "model_mode": "text2image",
        "prompt": compact_json_file(_path_of(t2i, "PROMPT")),
        "negative_prompt": "",
        **_flow("text2image"),
    }


def build_i2v_request() -> dict:
    i2v = _import_script("i2v")
    from common import compact_json_file, media_to_data_url

    return {
        "model_mode": "image2video",
        "prompt": compact_json_file(_path_of(i2v, "PROMPT")),
        "negative_prompt": compact_json_file(_path_of(i2v, "NEGATIVE_PROMPT")),
        "input_reference": media_to_data_url(_path_of(i2v, "IMAGE")),
        **_flow("image2video"),
    }


def build_i2v_4step_request() -> dict:
    i2v4 = _import_script("i2v_4step")
    from common import compact_json_file, media_to_data_url

    # The 4-step profile owns num_inference_steps / guidance_scale / flow_shift,
    # which the "image2video-4step" flow omits.
    return {
        "model_mode": "image2video",
        "prompt": compact_json_file(_path_of(i2v4, "PROMPT")),
        "negative_prompt": compact_json_file(_path_of(i2v4, "NEGATIVE_PROMPT")),
        "input_reference": media_to_data_url(_path_of(i2v4, "IMAGE")),
        **_flow("image2video-4step"),
    }


def build_t2i_4step_request() -> dict:
    t2i4 = _import_script("t2i_4step")
    from common import compact_json_file

    return {
        "model_mode": "text2image",
        "prompt": compact_json_file(_path_of(t2i4, "PROMPT")),
        "negative_prompt": "",
        **_flow("text2image-4step"),
    }


def build_v2v_request() -> dict:
    v2v = _import_script("v2v")
    from common import media_to_data_url

    return {
        "model_mode": "video2video",
        "prompt": (
            "A red sports car drives through a dramatic landscape with realistic "
            "motion, stable geometry, and cinematic lighting."
        ),
        "input_reference": media_to_data_url(_path_of(v2v, "VIDEO")),
        "condition_frame_indexes_vision": [0, 1],
        "condition_video_keep": "first",
        **_flow("video2video"),
    }


def build_transfer_request(case: str = "precomputed_edge") -> dict:
    transfer = _import_script("transfer")
    return transfer.build_request(case)


def build_action_request(case: str = "av_forward") -> dict:
    action = _import_script("action")
    return action.build_request(case)


# --- Registry ---------------------------------------------------------------


def request(registry_id: str, case: str | None = None) -> dict:
    """Return the request dict for ``registry_id``.

    ``case`` is passed through for scripts that expose a per-case CLI
    (``transfer``, ``action``); it is ignored for single-case scripts.
    """
    builders = {
        "t2v": build_t2v_request,
        "t2i": build_t2i_request,
        "i2v": build_i2v_request,
        "i2v_4step": build_i2v_4step_request,
        "t2i_4step": build_t2i_4step_request,
        "v2v": build_v2v_request,
        "transfer": build_transfer_request,
        "action": build_action_request,
    }
    if case is not None and registry_id in {"transfer", "action"}:
        return builders[registry_id](case)
    return builders[registry_id]()


GENERATOR_IDS = ("t2v", "t2i", "i2v", "i2v_4step", "t2i_4step", "v2v", "transfer", "action")

#: Assets each builder touches, so ``--check`` can pre-flight them offline.
#: Keys are registry ids; values are module attribute names.
_CHECK_ASSETS: dict[str, tuple[str, ...]] = {
    "t2v": ("PROMPT", "NEGATIVE_PROMPT"),
    "t2i": ("PROMPT",),
    "i2v": ("IMAGE", "PROMPT", "NEGATIVE_PROMPT"),
    "i2v_4step": ("IMAGE", "PROMPT", "NEGATIVE_PROMPT"),
    "t2i_4step": ("PROMPT",),
    "v2v": ("VIDEO",),
    "transfer": ("NEGATIVE_PROMPT", "DERIVED_VIDEO"),
    "action": ("ACTION_ROOT", "INVERSE_VIDEO_URL"),
}


def check_assets(registry_id: str) -> list[str]:
    """Return the list of missing asset paths for ``registry_id`` (empty if OK)."""
    module = _import_script(registry_id)
    missing: list[str] = []
    for attr in _CHECK_ASSETS.get(registry_id, ()):
        value = getattr(module, attr)
        if isinstance(value, str):
            continue  # URL constant — not a local asset
        if hasattr(value, "is_dir"):
            if not value.exists():
                missing.append(str(value))
            continue
        if not value.exists():
            missing.append(str(value))
    return missing


def script_path(registry_id: str) -> str:
    """Return the absolute path of the example script for ``registry_id``."""
    module = _import_script(registry_id)
    path = getattr(module, "__file__", None)
    if not path:
        raise RuntimeError(f"{registry_id} did not define __file__")
    return path