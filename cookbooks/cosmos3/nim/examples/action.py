# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Run one Cosmos3 action request: forward dynamics, policy, or inverse dynamics."""

import argparse
import json
import os
from pathlib import Path

import requests
from common import decode_video, media_to_data_url

NIM_URL = os.environ.get("NIM_URL", "http://localhost:8000").rstrip("/")
COSMOS3_ROOT = Path(__file__).resolve().parents[2]
ACTION_ROOT = COSMOS3_ROOT / "generator" / "action" / "assets"
OUTPUTS = Path(__file__).parent / "outputs"
INVERSE_VIDEO_URL = (
    "https://github.com/nvidia-cosmos/cosmos-dependencies/raw/"
    "2b17a2413bd86b2cf9b03823637108851e4ddf2d/inputs/action/"
    "bridge_20260501_0.mp4"
)


def build_request(case: str) -> dict:
    if case == "inverse_dynamics":
        return {
            "prompt": "Put the pot to the left of the purple item.",
            "video": INVERSE_VIDEO_URL,
            "action_params": {
                "mode": "inverse_dynamics",
                "domain_name": "bridge_orig_lerobot",
                "action_chunk_size": 16,
                "raw_action_dim": 10,
                "action_space": "joint_pos",
                "image_size": "480",
                "action_fps": 5.0,
            },
            "num_output_frames": 17,
            "fps": 5.0,
            "steps": 30,
            "guidance_scale": 1.0,
            "flow_shift": 10.0,
            "seed": 0,
        }

    request = {
        "prompt": "You are an autonomous vehicle planning system.",
        "image": media_to_data_url(ACTION_ROOT / "images" / "av_0.jpg"),
        "action_params": {
            "mode": case,
            "domain_name": "av",
            "action_chunk_size": 60,
            "raw_action_dim": 9,
            "action_space": "joint_pos",
            "image_size": "480",
            "action_fps": 10.0,
        },
        "num_output_frames": 61,
        "fps": 10.0,
        "steps": 30,
        "guidance_scale": 1.0,
        "flow_shift": 10.0,
        "seed": 0,
    }
    if case == "forward_dynamics":
        trajectory_path = ACTION_ROOT / "actions" / "av_traj_forward.json"
        request["action_params"]["action"] = json.loads(
            trajectory_path.read_text(encoding="utf-8")
        )
    return request


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case",
        choices=("forward_dynamics", "policy", "inverse_dynamics"),
        default="forward_dynamics",
    )
    case = parser.parse_args().case

    response = requests.post(
        f"{NIM_URL}/v1/infer", json=build_request(case), timeout=1800
    )
    response.raise_for_status()
    result = response.json()

    OUTPUTS.mkdir(exist_ok=True)
    video_path = OUTPUTS / f"action_{case}.mp4"
    video_path.write_bytes(decode_video(result["b64_video"]))
    print(f"Saved video to {video_path}")

    if result.get("action") is not None:
        action_path = OUTPUTS / f"action_{case}.json"
        action_path.write_text(
            json.dumps(result["action"], indent=2) + "\n", encoding="utf-8"
        )
        print(f"Saved predicted action to {action_path}")


if __name__ == "__main__":
    main()
