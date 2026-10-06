# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Generate an image from a text prompt with the Generator runtime."""

import argparse
from pathlib import Path

import settings
from common import (
    compact_json_file,
    get_default_nim_url,
    nim_infer,
    prompt_or_json,
    require_generator_profile,
    write_media_output,
)

NIM_URL = get_default_nim_url()
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
    flow = settings.flow_defaults("text2image")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT, help="Output JPG path.")
    parser.add_argument(
        "--seed",
        type=int,
        default=flow["seed"],
        help="Random seed (default: the shared text2image default).",
    )
    parser.add_argument(
        "--prompt",
        default=compact_json_file(PROMPT),
        help="Text prompt or path to a .json prompt asset (default: the cookbook prompt).",
    )
    parser.add_argument(
        "--negative-prompt",
        default="",
        help="Negative prompt text (default: empty).",
    )
    parser.add_argument(
        "--unique",
        action="store_true",
        help="Avoid overwriting an existing output file (append a suffix).",
    )
    args = parser.parse_args()

    require_generator_profile(
        NIM_URL,
        allowed_variants=("nano", "super", "super-t2i"),
    )

    request = {
        "model_mode": "text2image",
        "prompt": prompt_or_json(args.prompt),
        "negative_prompt": prompt_or_json(args.negative_prompt),
        **flow,
        "seed": args.seed,
    }

    payload = nim_infer(NIM_URL, request=request)
    write_media_output(
        args.output.parent,
        name=args.output.name,
        response_payload=payload,
        key="b64_image",
        unique=args.unique,
    )


if __name__ == "__main__":
    main()