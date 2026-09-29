# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Generate a video conditioned on an existing cookbook video."""

import argparse
from pathlib import Path

import settings
from common import (
    get_default_nim_url,
    media_to_data_url,
    nim_infer,
    require_generator_profile,
    write_media_output,
)

NIM_URL = get_default_nim_url()
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
    flow = settings.flow_defaults("video2video")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT, help="Output MP4 path.")
    parser.add_argument(
        "--seed",
        type=int,
        default=flow["seed"],
        help="Random seed (default: the shared video2video default).",
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=VIDEO,
        help="Conditioning video path (default: the cookbook car_driving_plain video).",
    )
    parser.add_argument(
        "--prompt",
        default=(
            "A red sports car drives through a dramatic landscape with realistic "
            "motion, stable geometry, and cinematic lighting."
        ),
        help="Text prompt (default: the cookbook car-driving prompt).",
    )
    parser.add_argument(
        "--unique",
        action="store_true",
        help="Avoid overwriting an existing output file (append a suffix).",
    )
    args = parser.parse_args()

    require_generator_profile(
        NIM_URL,
        allowed_variants=("nano", "super"),
    )

    request = {
        "model_mode": "video2video",
        "prompt": args.prompt,
        "input_reference": media_to_data_url(args.video),
        "condition_frame_indexes_vision": [0, 1],
        "condition_video_keep": "first",
        **flow,
        "seed": args.seed,
    }

    payload = nim_infer(NIM_URL, request=request)
    write_media_output(
        args.output.parent,
        name=args.output.name,
        response_payload=payload,
        key="b64_video",
        unique=args.unique,
    )


if __name__ == "__main__":
    main()