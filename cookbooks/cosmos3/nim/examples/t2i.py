# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Generate an image from a text prompt with the Generator runtime."""

import argparse
import os
from pathlib import Path

import requests
from common import compact_json_file, decode_image, require_generator_profile

NIM_URL = os.environ.get("NIM_URL", "http://localhost:8000").rstrip("/")
COSMOS3_ROOT = Path(__file__).resolve().parents[2]
PROMPT = (
    COSMOS3_ROOT
    / "generator"
    / "audiovisual"
    / "assets"
    / "prompts"
    / "text2image"
    / "robot_draping.json"
)
OUTPUT = Path(__file__).parent / "outputs" / "t2i_robot_draping.jpg"


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    require_generator_profile(
        NIM_URL,
        allowed_variants=("nano", "super", "super-t2i"),
    )

    request = {
        "model_mode": "text2image",
        "prompt": compact_json_file(PROMPT),
        "negative_prompt": "",
        "resolution": "720_1_1",
        "num_frames": 1,
        "fps": 24.0,
        "num_inference_steps": 50,
        "guidance_scale": 4.0,
        "flow_shift": 3.0,
        "seed": 0,
    }

    response = requests.post(f"{NIM_URL}/v1/infer", json=request, timeout=1800)
    response.raise_for_status()

    OUTPUT.parent.mkdir(exist_ok=True)
    OUTPUT.write_bytes(decode_image(response.json()["b64_image"]))
    print(f"Saved image to {OUTPUT}")


if __name__ == "__main__":
    main()
