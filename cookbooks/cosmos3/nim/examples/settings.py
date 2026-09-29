# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Single source of truth for Cosmos3 NIM example generation defaults.

Every Generator script and request builder previously hard-coded the same
`seed` / `resolution` / `num_frames` / `fps` / `num_inference_steps` /
`guidance_scale` / `flow_shift` literals. These consolidated defaults live
here so that:

- the example scripts and `request_builders` stay in lockstep,
- the batch runner can override values without touching each script,
- drift (two files disagreeing on a default) becomes impossible.

Overrides enter through the ``COSMOS3_OVERRIDES`` environment variable (a JSON
object the batch runner exports from its ``--override KEY=VALUE`` flag). When
the variable is unset, defaults are used verbatim, so existing behaviour is
unchanged.
"""

from __future__ import annotations

import json
import os

#: Canonical defaults per generation flow. ``None`` means "the profile owns this
#: value and the request must not carry it" (the ``*_4step`` profiles).
FLOW_DEFAULTS: dict[str, dict] = {
    "text2video": {
        "resolution": "720_16_9",
        "num_frames": 189,
        "fps": 24.0,
        "num_inference_steps": 35,
        "guidance_scale": 6.0,
        "flow_shift": 10.0,
        "seed": 0,
    },
    "text2image": {
        "resolution": "720_1_1",
        "num_frames": 1,
        "fps": 24.0,
        "num_inference_steps": 50,
        "guidance_scale": 4.0,
        "flow_shift": 3.0,
        "seed": 0,
    },
    "image2video": {
        "resolution": "720",
        "num_frames": 189,
        "fps": 24.0,
        "num_inference_steps": 35,
        "guidance_scale": 6.0,
        "flow_shift": 10.0,
        "seed": 0,
    },
    "video2video": {
        "resolution": "720",
        "num_frames": 93,
        "fps": 24.0,
        "num_inference_steps": 35,
        "guidance_scale": 6.0,
        "flow_shift": 10.0,
        "seed": 0,
    },
    # The 4-step profiles own num_inference_steps / guidance_scale / flow_shift.
    "image2video-4step": {
        "resolution": "720",
        "num_frames": 189,
        "fps": 24.0,
        "seed": 0,
    },
    "text2image-4step": {
        "resolution": "720_1_1",
        "num_frames": 1,
        "fps": 24.0,
        "seed": 0,
    },
}

OVERRIDES_ENV = "COSMOS3_OVERRIDES"


def apply_overrides(defaults: dict, overrides: dict) -> dict:
    """Return ``defaults`` with ``overrides`` applied (types coerced to ``defaults``).

    Unknown keys are ignored (a typo cannot inject a stray request field).
    Values are coerced to the type of the corresponding default so that
    ``--override num_frames=93`` from a string CLI flag still produces an int.
    """
    if not overrides:
        return dict(defaults)
    out = dict(defaults)
    for key, value in overrides.items():
        if key not in defaults or defaults[key] is None:
            continue
        if isinstance(defaults[key], bool):
            out[key] = bool(value)
        elif isinstance(defaults[key], int):
            out[key] = int(value)
        elif isinstance(defaults[key], float):
            out[key] = float(value)
        else:
            out[key] = value
    return out


def overrides_from_env() -> dict:
    """Read ``COSMOS3_OVERRIDES`` (a JSON object) or return ``{}`` when unset."""
    raw = os.environ.get(OVERRIDES_ENV)
    if not raw:
        return {}
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"{OVERRIDES_ENV} must be a JSON object")
    return value


def flow_defaults(flow: str) -> dict:
    """Return the defaults for ``flow``, with any environment overrides applied."""
    if flow not in FLOW_DEFAULTS:
        raise KeyError(f"unknown generation flow: {flow!r}")
    return apply_overrides(FLOW_DEFAULTS[flow], overrides_from_env())