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

"""Inference example using the pipeline."""

from cosmos3.common.init import init_output_dir, init_script

init_script()

from pathlib import Path

from cosmos3.args import DEFAULT_CHECKPOINT_NAME, OmniSampleOverrides, OmniSetupOverrides
from cosmos3.inference import OmniInference, get_sample_data
from cosmos3._src.imaginaire.utils import log


def inference_pipeline():
    name = "inference_pipeline"
    output_dir = Path(f"outputs/{name}").absolute()
    init_output_dir(output_dir)

    log.info("Loading model...")
    setup_args = OmniSetupOverrides(
        checkpoint_path=DEFAULT_CHECKPOINT_NAME,
        output_dir=output_dir,
    ).build_setup()
    pipe = OmniInference.create(setup_args)

    sample_args = OmniSampleOverrides(
        name=name,
        output_dir=output_dir,
        prompt="the quick brown fox is happily jumping over the fence.",
        num_frames=1,
    ).build_sample(model_config=pipe.model_config)
    data_batch = get_sample_data(sample_args, model=pipe.model)

    log.info("Generating samples...")
    pipe.generate_batch([sample_args], data_batch)


def main():
    inference_pipeline()


if __name__ == "__main__":
    main()
