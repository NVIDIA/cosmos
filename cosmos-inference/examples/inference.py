# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Inference example."""

from cosmos3.common.init import init_output_dir, init_script

init_script()

from pathlib import Path

import safetensors.torch
import torch

from cosmos3.args import DEFAULT_CHECKPOINT, OmniSampleOverrides
from cosmos3.inference import get_sample_data
from cosmos3.model import Cosmos3OmniModel
from cosmos3._src.imaginaire.utils import log
from cosmos3._src.imaginaire.visualize.video import save_img_or_video
from cosmos3._src.vfm.configs.base.defaults.model_config import ParallelismConfig


def inference():
    name = "inference"
    output_dir = Path(f"outputs/{name}").absolute()
    init_output_dir(output_dir)

    log.info("Loading model...")
    checkpoint_path = DEFAULT_CHECKPOINT.download()
    model = Cosmos3OmniModel.from_pretrained_dcp(
        Path(checkpoint_path),
        parallelism_config=ParallelismConfig(
            use_torch_compile=True,
        ),
    ).model

    # Create batch
    sample_args = OmniSampleOverrides(
        name=name,
        output_dir=output_dir,
        prompt="A medium shot of a modern robotics research laboratory with white walls and a gray floor. A robotic arm with a metallic finish is mounted on a clean white workbench, its gripper positioned above a row of small colored objects. A laptop and neatly arranged tools sit beside the robot. A large monitor on the wall behind displays a software interface. The scene is brightly lit by overhead fluorescent lights.",
        num_frames=1,
    ).build_sample(model_config=model.config)
    data_batch = get_sample_data(sample_args, model)

    # Generate samples
    log.info("Generating samples...")
    outputs = model.generate_samples_from_batch(data_batch, seed=[0])

    # Decode
    def decode_vision(vision_latent: torch.Tensor) -> torch.Tensor:
        vision = model.decode(vision_latent)  # Decode to pixel space
        return (1.0 + vision.clamp(-1, 1)) / 2  # [0, 1]

    outputs["vision"] = [decode_vision(vision) for vision in outputs.pop("vision")]
    outputs = {k: torch.cat(v, dim=0) for k, v in outputs.items()}

    # Save outputs
    log.info("Saving outputs...")
    safetensors.torch.save_file(outputs, output_dir / "outputs.safetensors")
    save_img_or_video(outputs["vision"][0], str(output_dir / "vision"), fps=data_batch["fps"][0].item())
    log.success(f"Saved outputs to {output_dir}")


def main():
    inference()


if __name__ == "__main__":
    main()
