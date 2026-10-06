# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Generate a video conditioned on an existing cookbook image."""

import argparse
from pathlib import Path

import settings
from common import (
    compact_json_file,
    get_default_nim_url,
    media_to_data_url,
    nim_infer,
    prompt_or_json,
    require_generator_profile,
    write_media_output,
)

NIM_URL = get_default_nim_url()
COSMOS3_ROOT = Path(__file__).resolve().parents[2]
ASSETS = COSMOS3_ROOT / "generator" / "audiovisual" / "assets"
IMAGE = ASSETS / "images" / "image2video" / "car_driving.jpg"
PROMPT = ASSETS / "prompts" / "image2video" / "car_driving.json"
NEGATIVE_PROMPT = ASSETS / "negative_prompts" / "image2video" / "neg_prompt.json"
OUTPUT = Path(__file__).parent / "outputs" / "i2v_car_driving.mp4"


def main() -> None:
    flow = settings.flow_defaults("image2video")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT, help="Output MP4 path.")
    parser.add_argument(
        "--seed",
        type=int,
        default=flow["seed"],
        help="Random seed (default: the shared image2video default).",
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=IMAGE,
        help="Conditioning image path (default: the cookbook car_driving image).",
    )
    parser.add_argument(
        "--prompt",
        default=compact_json_file(PROMPT),
        help="Text prompt or path to a .json prompt asset (default: the cookbook prompt).",
    )
    parser.add_argument(
        "--negative-prompt",
        default=compact_json_file(NEGATIVE_PROMPT),
        help="Negative prompt text (default: the cookbook negation).",
    )
    parser.add_argument(
        "--unique",
        action="store_true",
        help="Avoid overwriting an existing output file (append a suffix).",
    )
    args = parser.parse_args()

    require_generator_profile(
        NIM_URL,
        allowed_variants=("nano", "super", "super-i2v"),
    )

    request = {
        "model_mode": "image2video",
        "prompt": prompt_or_json(args.prompt),
        "negative_prompt": prompt_or_json(args.negative_prompt),
        "input_reference": media_to_data_url(args.image),
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