#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Load a Cosmos3 diffusers pipeline from a converted checkpoint and run inference.

CUDA_VISIBLE_DEVICES=0 python inference_cosmos3.py \
    --pipeline-path converted/cosmos3-nano-pipeline \
    --input inputs/omni/i2v.json

The pipeline must have been produced by convert_cosmos3_to_diffusers.py
with --save-pipeline.
"""

import argparse
import json
import pathlib

import torch
from diffusers_cosmos3 import Cosmos3OmniDiffusersPipeline
from diffusers_cosmos3.pipeline import save_img_or_video


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pipeline-path",
        default="converted/cosmos3-nano-pipeline",
        help="Path to directory saved by cosmos_convert_cosmos_to_diffusers.py --save-pipeline.",
    )
    parser.add_argument(
        "--input",
        default="inputs/omni/i2v.json",
        help="Path to JSON input file with 'prompt' and optional 'vision_path'.",
    )
    parser.add_argument("--output", default=".", help="Directory to save generated video/image files.")
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--num-frames", type=int, default=189)
    args = parser.parse_args()

    pipeline_path = pathlib.Path(args.pipeline_path)
    print(f"Loading pipeline from {pipeline_path} …")
    pipeline = Cosmos3OmniDiffusersPipeline.from_pretrained(
        str(pipeline_path),
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    print("Pipeline loaded successfully.")

    # --- Load JSON input ---
    input_path = pathlib.Path(args.input)
    print(f"Loading input from {input_path} …")
    with open(input_path) as f:
        input_data = json.load(f)
    prompt = input_data["prompt"]
    vision_path = input_data.get("vision_path", None)

    output_dir = pathlib.Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    result_vision = pipeline(
        prompt=prompt,
        image=vision_path,
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        output_type="latent",
    )

    result_frames = pipeline.decode_latents(result_vision)
    for i, frames in enumerate(result_frames):
        save_path = str(output_dir / f"sample-{i}")
        save_img_or_video(frames, save_path)
        print(f"Saved: {save_path}.mp4")


if __name__ == "__main__":
    main()
