# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Run one canonical Cosmos3 Action request through the Generator runtime."""

import argparse
import json
import math
import os
from pathlib import Path

import requests
from common import decode_video, media_to_data_url, require_generator_profile

NIM_URL = os.environ.get("NIM_URL", "http://localhost:8000").rstrip("/")
COSMOS3_ROOT = Path(__file__).resolve().parents[2]
ACTION_ROOT = COSMOS3_ROOT / "generator" / "action" / "assets"
OUTPUTS = Path(__file__).parent / "outputs"
INVERSE_VIDEO_URL = (
    "https://github.com/nvidia-cosmos/cosmos-dependencies/raw/"
    "2b17a2413bd86b2cf9b03823637108851e4ddf2d/inputs/action/"
    "bridge_20260501_0.mp4"
)
DOMAIN_IDS = {
    "av": 1,
    "umi": 6,
    "bridge_orig_lerobot": 7,
    "droid_lerobot": 8,
}
CANONICAL_CASES = (
    "av_forward",
    "av_left",
    "av_right",
    "umi_forward",
    "av_inverse_0",
    "av_inverse_1",
    "bridge_inverse",
    "av_policy_left",
    "av_policy_right",
)
CASE_ALIASES = {
    "forward_dynamics": "av_forward",
    "inverse_dynamics": "bridge_inverse",
    "policy": "av_policy_right",
    "av_policy": "av_policy_right",
}
AV_POLICY_PROMPTS = {
    "left": (
        "You are an autonomous vehicle planning system. Turn left onto the road "
        "and continue driving in the leftmost legal lane."
    ),
    "right": (
        "You are an autonomous vehicle planning system. Turn right onto the road "
        "and continue driving in the rightmost lane."
    ),
}
AV_POLICY_EXPECTATIONS = {
    "av_policy_left": (
        "the ego vehicle turns left onto the roadway and continues in that direction"
    ),
    "av_policy_right": (
        "the ego vehicle turns right onto the roadway and continues in that direction"
    ),
}


def load_trajectory(path: Path, *, steps: int, width: int) -> list[list[float]]:
    """Load and validate one finite numeric trajectory chunk."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or len(value) < steps:
        raise ValueError(f"{path} must contain at least {steps} action rows")

    trajectory = []
    for index, row in enumerate(value):
        if not isinstance(row, list) or len(row) != width:
            raise ValueError(f"{path} row {index} must contain {width} values")
        if any(
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(item)
            for item in row
        ):
            raise ValueError(f"{path} row {index} must contain finite numbers")
        trajectory.append([float(item) for item in row])
    return trajectory[:steps]


def av_forward_request(direction: str) -> dict:
    trajectory = load_trajectory(
        ACTION_ROOT / "actions" / f"av_traj_{direction}.json",
        steps=60,
        width=9,
    )
    return {
        "model_mode": "forward_dynamics",
        "prompt": "You are an autonomous vehicle planning system.",
        "input_reference": media_to_data_url(ACTION_ROOT / "images" / "av_0.jpg"),
        "action_params": {
            "domain_name": "av",
            "action_chunk_size": 60,
            "action": trajectory,
            "raw_action_dim": 9,
            "action_space": "joint_pos",
            "image_size": "480",
            "action_fps": 10.0,
        },
        "fps": 10.0,
        "num_inference_steps": 30,
        "guidance_scale": 1.0,
        "flow_shift": 10.0,
        "seed": 0,
    }


def umi_forward_request() -> dict:
    trajectory = load_trajectory(
        ACTION_ROOT / "actions" / "umi.json",
        steps=16,
        width=10,
    )
    return {
        "model_mode": "forward_dynamics",
        "prompt": "mouse arrangement",
        "input_reference": media_to_data_url(ACTION_ROOT / "images" / "umi.png"),
        "action_params": {
            "domain_name": "umi",
            "action_chunk_size": 16,
            "action": trajectory,
            "raw_action_dim": 10,
            "action_space": "joint_pos",
            "image_size": "256",
            "action_fps": 20.0,
        },
        "fps": 20.0,
        "num_inference_steps": 30,
        "guidance_scale": 1.0,
        "flow_shift": 10.0,
        "seed": 0,
    }


def av_inverse_request(index: int) -> dict:
    return {
        "model_mode": "inverse_dynamics",
        "prompt": "You are an autonomous vehicle planning system.",
        "input_reference": media_to_data_url(
            ACTION_ROOT / "videos" / f"av_{index}.mp4"
        ),
        "action_params": {
            "domain_name": "av",
            "action_chunk_size": 60,
            "raw_action_dim": 9,
            "action_space": "joint_pos",
            "image_size": "480",
            "action_fps": 10.0,
        },
        "fps": 10.0,
        "num_inference_steps": 30,
        "guidance_scale": 1.0,
        "flow_shift": 10.0,
        "seed": 0,
    }


def bridge_inverse_request() -> dict:
    return {
        "model_mode": "inverse_dynamics",
        "prompt": "Put the pot to the left of the purple item.",
        "input_reference": INVERSE_VIDEO_URL,
        "action_params": {
            "domain_name": "bridge_orig_lerobot",
            "action_chunk_size": 16,
            "raw_action_dim": 10,
            "action_space": "joint_pos",
            "image_size": "480",
            "action_fps": 5.0,
        },
        "fps": 5.0,
        "num_inference_steps": 30,
        "guidance_scale": 1.0,
        "flow_shift": 10.0,
        "seed": 0,
    }


def av_policy_request(direction: str) -> dict:
    try:
        prompt = AV_POLICY_PROMPTS[direction]
    except KeyError as exc:
        raise ValueError(f"Unknown AV policy direction: {direction}") from exc
    return {
        "model_mode": "policy",
        "prompt": prompt,
        "input_reference": media_to_data_url(ACTION_ROOT / "images" / "av_0.jpg"),
        "action_params": {
            "domain_name": "av",
            "action_chunk_size": 60,
            "raw_action_dim": 9,
            "action_space": "joint_pos",
            "image_size": "480",
            "action_fps": 10.0,
        },
        "fps": 10.0,
        "num_inference_steps": 30,
        "guidance_scale": 1.0,
        "flow_shift": 10.0,
        "seed": 0,
    }


def build_request(case: str) -> dict:
    """Build one request while preserving the legacy CLI names as aliases."""
    case = CASE_ALIASES.get(case, case)
    if case.startswith("av_") and case.removeprefix("av_") in {
        "forward",
        "left",
        "right",
    }:
        return av_forward_request(case.removeprefix("av_"))
    if case == "umi_forward":
        return umi_forward_request()
    if case.startswith("av_inverse_"):
        return av_inverse_request(int(case.removeprefix("av_inverse_")))
    if case == "bridge_inverse":
        return bridge_inverse_request()
    if case.startswith("av_policy_"):
        return av_policy_request(case.removeprefix("av_policy_"))
    raise ValueError(f"Unknown Action case: {case}")


def validate_action_output(action: object, request: dict) -> None:
    """Validate predicted action metadata, shape, and finite numeric values."""
    mode = request["model_mode"]
    if mode == "forward_dynamics":
        if action is not None:
            raise ValueError("Forward-dynamics response must not predict an action")
        return
    if not isinstance(action, dict):
        raise ValueError(f"{mode} response must contain an action object")

    params = request["action_params"]
    expected_shape = [params["action_chunk_size"], params["raw_action_dim"]]
    shape = action.get("shape")
    if (
        not isinstance(shape, list)
        or len(shape) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) for item in shape)
    ):
        raise ValueError("Action shape must contain two integer dimensions")
    if shape != expected_shape:
        raise ValueError(f"Expected action shape {expected_shape}, received {shape}")

    data = action.get("data")
    if not isinstance(data, list) or len(data) != shape[0]:
        raise ValueError("Action data row count does not match action shape")
    for index, row in enumerate(data):
        if not isinstance(row, list) or len(row) != shape[1]:
            raise ValueError(f"Action data row {index} does not match action shape")
        if any(
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(item)
            for item in row
        ):
            raise ValueError(f"Action data row {index} must contain finite numbers")

    expected_domain = DOMAIN_IDS[params["domain_name"]]
    expected_metadata = {
        "dtype": "float32",
        "raw_action_dim": expected_shape[1],
        "action_mode": mode,
        "domain_id": expected_domain,
    }
    for field, expected in expected_metadata.items():
        if action.get(field) != expected:
            raise ValueError(
                f"Expected action {field}={expected!r}, received {action.get(field)!r}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case",
        choices=CANONICAL_CASES + tuple(CASE_ALIASES),
        default="av_forward",
    )
    selected_case = parser.parse_args().case
    case = CASE_ALIASES.get(selected_case, selected_case)
    allowed_variants = (
        ("nano",) if case.startswith("av_policy_") else ("nano", "super")
    )
    require_generator_profile(
        NIM_URL,
        allowed_variants=allowed_variants,
    )
    request = build_request(case)
    expected_behavior = AV_POLICY_EXPECTATIONS.get(case)
    if expected_behavior:
        print(f"Expected qualitative behavior: {expected_behavior}.")

    response = requests.post(f"{NIM_URL}/v1/infer", json=request, timeout=1800)
    response.raise_for_status()
    result = response.json()
    if not isinstance(result, dict):
        raise TypeError("Action response must be a JSON object")

    action = result.get("action")
    validate_action_output(action, request)
    if not result.get("b64_video"):
        raise ValueError("General Action response did not contain rollout video")

    OUTPUTS.mkdir(exist_ok=True)
    video_path = OUTPUTS / f"action_{case}.mp4"
    video_path.write_bytes(decode_video(result["b64_video"]))
    print(f"Saved video to {video_path}")

    if action is not None:
        action_path = OUTPUTS / f"action_{case}.json"
        action_path.write_text(
            json.dumps(action, indent=2) + "\n", encoding="utf-8"
        )
        print(f"Saved validated action to {action_path}")

    if expected_behavior:
        print(
            "Structural response validation passed. Review the saved video for "
            "directional task compliance."
        )


if __name__ == "__main__":
    main()
