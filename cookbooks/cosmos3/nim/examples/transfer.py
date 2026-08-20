# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Run one Cosmos3 transfer request using a precomputed or derived control."""

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
TRANSFER_ROOT = COSMOS3_ROOT / "generator" / "transfer" / "assets"
NEGATIVE_PROMPT = TRANSFER_ROOT / "negative_prompt.json"
DERIVED_VIDEO = (
    COSMOS3_ROOT
    / "generator"
    / "audiovisual"
    / "assets"
    / "videos"
    / "car_driving_plain.mp4"
)
OUTPUTS = Path(__file__).parent / "outputs"
PRECOMPUTED_HINTS = ("edge", "blur", "depth", "seg", "wsm")
CASES = tuple(f"precomputed_{hint}" for hint in PRECOMPUTED_HINTS) + (
    "derived_edge",
    "derived_blur",
)


def precomputed_request(hint: str) -> dict:
    prompt_path = TRANSFER_ROOT / hint / "prompt.json"
    control_path = TRANSFER_ROOT / hint / f"control_{hint}.mp4"
    return {
        "model_mode": "video2video",
        "prompt": compact_json_file(prompt_path),
        "negative_prompt": compact_json_file(NEGATIVE_PROMPT),
        "transfer": {
            hint: {"video": media_to_data_url(control_path)},
            "control_guidance": (
                3.0 if hint == "wsm" else 2.0 if hint == "seg" else 1.5
            ),
            "num_conditional_frames": 1,
            "num_first_chunk_conditional_frames": 0,
            "num_video_frames_per_chunk": 101 if hint == "wsm" else 121,
        },
        "resolution": "720_4_3" if hint == "blur" else "720_16_9",
        "num_frames": 101 if hint == "wsm" else 121,
        "fps": 10.0 if hint == "wsm" else 30.0,
        "num_inference_steps": 50,
        "guidance_scale": 1.0 if hint == "wsm" else 3.0,
        "flow_shift": 10.0,
        "seed": 2026,
    }


def derived_request(hint: str) -> dict:
    preset = (
        {"preset_edge_threshold": "medium"}
        if hint == "edge"
        else {"preset_blur_strength": "medium"}
    )
    return {
        "model_mode": "video2video",
        "prompt": (
            "A red sports car drives through a dramatic landscape with stable "
            "geometry, realistic motion, and cinematic lighting."
        ),
        "negative_prompt": compact_json_file(NEGATIVE_PROMPT),
        "input_reference": media_to_data_url(DERIVED_VIDEO),
        "transfer": {
            hint: preset,
            "control_guidance": 1.5,
            "num_conditional_frames": 1,
            "num_first_chunk_conditional_frames": 0,
            "num_video_frames_per_chunk": 121,
        },
        "resolution": "720_16_9",
        "num_frames": 121,
        "fps": 30.0,
        "num_inference_steps": 50,
        "guidance_scale": 3.0,
        "flow_shift": 10.0,
        "seed": 2026,
    }


def build_request(case: str) -> dict:
    if case.startswith("precomputed_"):
        return precomputed_request(case.removeprefix("precomputed_"))
    return derived_request(case.removeprefix("derived_"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=CASES, default="precomputed_edge")
    case = parser.parse_args().case
    require_generator_profile(
        NIM_URL,
        allowed_variants=("nano", "super"),
    )
    request = build_request(case)

    response = requests.post(f"{NIM_URL}/v1/infer", json=request, timeout=3600)
    response.raise_for_status()

    OUTPUTS.mkdir(exist_ok=True)
    output = OUTPUTS / f"transfer_{case}.mp4"
    output.write_bytes(decode_video(response.json()["b64_video"]))
    print(f"Saved video to {output}")


if __name__ == "__main__":
    main()
