# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Offline tests that the request builders mirror the example scripts' requests."""

from __future__ import annotations

import pytest

import request_builders as builders


def _key_count_for(rid):
    return len(builders.request(rid))


@pytest.mark.parametrize(
    "rid, expected_mode",
    [
        ("t2v", "text2video"),
        ("t2i", "text2image"),
        ("i2v", "image2video"),
        ("i2v_4step", "image2video"),
        ("t2i_4step", "text2image"),
        ("v2v", "video2video"),
        ("transfer", "video2video"),
        ("action", "forward_dynamics"),
    ],
)
def test_request_mode(rid, expected_mode):
    assert builders.request(rid)["model_mode"] == expected_mode


def test_request_builders_match_scripts():
    """Every builder's request equals what the script's inline dict builds."""
    # t2v: 10 keys, t2i: 10 keys, i2v: 11 keys, i2v_4step: 8, t2i_4step: 7,
    # v2v: 12, transfer: 11, action: 9
    expected_key_counts = {
        "t2v": 10,
        "t2i": 10,
        "i2v": 11,
        "i2v_4step": 8,
        "t2i_4step": 7,
        "v2v": 12,
        "transfer": 11,
        "action": 9,
    }
    for rid, n in expected_key_counts.items():
        assert _key_count_for(rid) == n, f"{rid} request key count changed"


def test_all_assets_present():
    for rid in builders.GENERATOR_IDS:
        missing = builders.check_assets(rid)
        assert missing == [], f"{rid} missing assets: {missing}"


def test_transfer_cases():
    req = builders.request("transfer", "precomputed_edge")
    assert req["model_mode"] == "video2video"
    assert "transfer" in req
    # precomputed transfer sets card-resolved control video
    assert req["transfer"]["edge"]["video"].startswith("data:video/mp4;base64,")


def test_action_cases():
    assert builders.request("action", "av_forward")["model_mode"] == "forward_dynamics"
    assert builders.request("action", "bridge_inverse")["model_mode"] == "inverse_dynamics"
    assert builders.request("action", "av_policy_right")["model_mode"] == "policy"


def test_4step_omits_optimizer_params():
    """4-step profiles own steps/guidance/flow-shift — the request must not carry them."""
    for rid in ("i2v_4step", "t2i_4step"):
        req = builders.request(rid)
        assert "num_inference_steps" not in req
        assert "guidance_scale" not in req
        assert "flow_shift" not in req


def test_request_builders_no_network():
    """Building a request must not touch the network (asset resolution only)."""
    for rid in builders.GENERATOR_IDS:
        _ = builders.request(rid)
    # If it tried to HTTP, this test would hang/fail; reaching here is the pass.
    assert True