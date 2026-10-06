# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Generate an image with a selected super-t2i-4step profile."""

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
OUTPUT = Path(__file__).parent / "outputs" / "t2i_robot_draping_4step.jpg"


def main() -> None:
    flow = settings.flow_defaults("text2image-4step")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT, help="Output JPG path.")
    parser.add_argument(
        "--seed",
        type=int,
        default=flow["seed"],
        help="Random seed (default: the shared text2image-4step default).",
    )
    parser.add_argument(
        "--prompt",
        default=compact_json_file(PROMPT),
        help="Text prompt or path to a .json prompt asset (default: the cookbook prompt).",
    )
    parser.add_argument(
        "--unique",
        action="store_true",
        help="Avoid overwriting an existing output file (append a suffix).",
    )
    args = parser.parse_args()

    require_generator_profile(
        NIM_URL,
        allowed_variants=("super-t2i-4step",),
    )

    # Start the NIM with NIM_MODEL_VARIANT=super-t2i-4step. The profile owns
    # num_inference_steps, guidance_scale, and flow_shift; the shared
    # "text2image-4step" flow omits all three.
    request = {
        "model_mode": "text2image",
        "prompt": prompt_or_json(args.prompt),
        "negative_prompt": "",
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