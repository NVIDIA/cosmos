# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Generate an image with a selected super-t2i-4step profile."""

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
OUTPUT = Path(__file__).parent / "outputs" / "t2i_robot_draping_4step.jpg"


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    require_generator_profile(
        NIM_URL,
        allowed_variants=("super-t2i-4step",),
    )

    # Start the NIM with NIM_MODEL_VARIANT=super-t2i-4step. The profile owns
    # num_inference_steps, guidance_scale, and flow_shift, so this request
    # omits all three.
    request = {
        "model_mode": "text2image",
        "prompt": compact_json_file(PROMPT),
        "negative_prompt": "",
        "resolution": "720_1_1",
        "num_frames": 1,
        "fps": 24.0,
        "seed": 0,
    }

    response = requests.post(f"{NIM_URL}/v1/infer", json=request, timeout=1800)
    response.raise_for_status()

    OUTPUT.parent.mkdir(exist_ok=True)
    OUTPUT.write_bytes(decode_image(response.json()["b64_image"]))
    print(f"Saved image to {OUTPUT}")


if __name__ == "__main__":
    main()
