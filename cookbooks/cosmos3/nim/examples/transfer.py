# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Run one Cosmos3 transfer request using a precomputed or derived control."""

import argparse
import os
from pathlib import Path

import requests
from common import (
    compact_json_file,
    media_to_data_url,
    nim_infer,
    require_generator_profile,
    write_media_output,
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
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output MP4 path (default: outputs/transfer_<case>.mp4).",
    )
    parser.add_argument(
        "--unique",
        action="store_true",
        help="Avoid overwriting an existing output file (append a suffix).",
    )
    args = parser.parse_args()
    case = args.case
    require_generator_profile(
        NIM_URL,
        allowed_variants=("nano", "super"),
    )
    request = build_request(case)

    payload = nim_infer(NIM_URL, request=request, timeout=3600)
    output = args.output or OUTPUTS / f"transfer_{case}.mp4"
    write_media_output(
        output.parent,
        name=output.name,
        response_payload=payload,
        key="b64_video",
        unique=args.unique,
    )


if __name__ == "__main__":
    main()
