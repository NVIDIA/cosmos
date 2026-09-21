# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Generate an image-conditioned video with a super-i2v-4step profile."""

import argparse
import os
from pathlib import Path

import requests
from common import (
    compact_json_file,
    decode_video,
    media_to_data_url,
    require_generator_profile,
)

NIM_URL = os.environ.get("NIM_URL", "http://localhost:8000").rstrip("/")
COSMOS3_ROOT = Path(__file__).resolve().parents[2]
ASSETS = COSMOS3_ROOT / "generator" / "audiovisual" / "assets"
IMAGE = ASSETS / "images" / "image2video" / "car_driving.jpg"
PROMPT = ASSETS / "prompts" / "image2video" / "car_driving.json"
NEGATIVE_PROMPT = ASSETS / "negative_prompts" / "image2video" / "neg_prompt.json"
OUTPUT = Path(__file__).parent / "outputs" / "i2v_car_driving_4step.mp4"


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    require_generator_profile(
        NIM_URL,
        allowed_variants=("super-i2v-4step",),
    )

    # Start the NIM with NIM_MODEL_VARIANT=super-i2v-4step. The profile owns
    # num_inference_steps, guidance_scale, and flow_shift, so this request
    # omits all three.
    request = {
        "model_mode": "image2video",
        "prompt": compact_json_file(PROMPT),
        "negative_prompt": compact_json_file(NEGATIVE_PROMPT),
        "input_reference": media_to_data_url(IMAGE),
        "resolution": "720",
        "num_frames": 189,
        "fps": 24.0,
        "seed": 0,
    }

    response = requests.post(f"{NIM_URL}/v1/infer", json=request, timeout=1800)
    response.raise_for_status()

    OUTPUT.parent.mkdir(exist_ok=True)
    OUTPUT.write_bytes(decode_video(response.json()["b64_video"]))
    print(f"Saved video to {OUTPUT}")


if __name__ == "__main__":
    main()
