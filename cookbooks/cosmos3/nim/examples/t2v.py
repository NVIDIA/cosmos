# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Generate a video from a text prompt with the Generator runtime."""

import argparse
import os
from pathlib import Path

import requests
from common import compact_json_file, decode_video, require_generator_profile

NIM_URL = os.environ.get("NIM_URL", "http://localhost:8000").rstrip("/")
COSMOS3_ROOT = Path(__file__).resolve().parents[2]
ASSETS = COSMOS3_ROOT / "generator" / "audiovisual" / "assets"
PROMPT = ASSETS / "prompts" / "text2video" / "robot_kitchen.json"
NEGATIVE_PROMPT = ASSETS / "negative_prompts" / "text2video" / "neg_prompt.json"
OUTPUT = Path(__file__).parent / "outputs" / "t2v_robot_kitchen.mp4"


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    require_generator_profile(
        NIM_URL,
        allowed_variants=("nano", "super"),
    )

    request = {
        "model_mode": "text2video",
        "prompt": compact_json_file(PROMPT),
        "negative_prompt": compact_json_file(NEGATIVE_PROMPT),
        "resolution": "720_16_9",
        "num_frames": 189,
        "fps": 24.0,
        "num_inference_steps": 35,
        "guidance_scale": 6.0,
        "flow_shift": 10.0,
        "seed": 0,
    }

    response = requests.post(f"{NIM_URL}/v1/infer", json=request, timeout=1800)
    response.raise_for_status()

    OUTPUT.parent.mkdir(exist_ok=True)
    OUTPUT.write_bytes(decode_video(response.json()["b64_video"]))
    print(f"Saved video to {OUTPUT}")


if __name__ == "__main__":
    main()
