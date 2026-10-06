# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Offline tests for the shared generation defaults in ``settings.py``."""

from __future__ import annotations

import json

import pytest

import settings


def test_flow_defaults_known_flows():
    for flow in (
        "text2video",
        "text2image",
        "image2video",
        "video2video",
        "image2video-4step",
        "text2image-4step",
    ):
        assert settings.flow_defaults(flow)["seed"] == 0


def test_flow_defaults_unknown_flow():
    with pytest.raises(KeyError):
        settings.flow_defaults("not-a-flow")


def test_4step_omits_profile_owned_keys():
    for flow in ("image2video-4step", "text2image-4step"):
        d = settings.flow_defaults(flow)
        assert "num_inference_steps" not in d
        assert "guidance_scale" not in d
        assert "flow_shift" not in d


def test_apply_overrides_coerces_types():
    defaults = {"num_frames": 189, "fps": 24.0}
    out = settings.apply_overrides(defaults, {"num_frames": "93", "fps": "30"})
    assert out["num_frames"] == 93
    assert isinstance(out["num_frames"], int)
    assert out["fps"] == 30.0
    assert isinstance(out["fps"], float)


def test_apply_overrides_ignores_unknown():
    defaults = {"seed": 0}
    out = settings.apply_overrides(defaults, {"not-a-key": 1})
    assert out == {"seed": 0}


def test_overrides_from_env(monkeypatch):
    monkeypatch.setenv("COSMOS3_OVERRIDES", json.dumps({"num_frames": 99}))
    assert settings.flow_defaults("text2video")["num_frames"] == 99


def test_overrides_env_empty_by_default(monkeypatch):
    monkeypatch.delenv("COSMOS3_OVERRIDES", raising=False)
    assert settings.overrides_from_env() == {}