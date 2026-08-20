# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Generate a video conditioned on an existing cookbook video."""

import argparse
import os
from pathlib import Path

import requests
from common import decode_video, media_to_data_url, require_generator_profile

NIM_URL = os.environ.get("NIM_URL", "http://localhost:8000").rstrip("/")
COSMOS3_ROOT = Path(__file__).resolve().parents[2]
VIDEO = (
    COSMOS3_ROOT
    / "generator"
    / "audiovisual"
    / "assets"
    / "videos"
    / "car_driving_plain.mp4"
)
OUTPUT = Path(__file__).parent / "outputs" / "v2v.mp4"


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    require_generator_profile(
        NIM_URL,
        allowed_variants=("nano", "super"),
    )

    request = {
        "model_mode": "video2video",
        "prompt": (
            "A red sports car drives through a dramatic landscape with realistic "
            "motion, stable geometry, and cinematic lighting."
        ),
        "input_reference": media_to_data_url(VIDEO),
        "condition_frame_indexes_vision": [0, 1],
        "condition_video_keep": "first",
        "resolution": "720",
        "num_frames": 93,
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
