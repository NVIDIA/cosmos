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

"""Convert a Hugging Face model to a DCP checkpoint."""

from cosmos3.common.init import init_script

init_script(
    env={
        "COSMOS_DEVICE": "cpu",
    }
)

import json
import shutil
from typing import Annotated

import pydantic
import torch.distributed.checkpoint as dcp
import tyro
from torch.distributed.checkpoint.filesystem import FileSystemWriter
from torch.distributed.checkpoint.state_dict import get_model_state_dict

from cosmos3.args import OmniSetupOverrides
from cosmos3.common.args import CheckpointOverrides, ResolvedPath
from cosmos3.common.config import fix_config_dict
from cosmos3.model import Cosmos3OmniModel
from cosmos3._src.vfm.checkpointer.dcp import CustomSavePlanner


class Args(pydantic.BaseModel):
    checkpoint: CheckpointOverrides
    """Hugging Face checkpoint."""
    output_path: Annotated[ResolvedPath, tyro.conf.arg(aliases=("-o",))]
    """Output DCP checkpoint directory."""


def convert_model_to_dcp(args: Args):
    print("Loading model...")
    checkpoint_config = args.checkpoint.build_checkpoint(checkpoints=OmniSetupOverrides.CHECKPOINTS)
    hf_path = checkpoint_config.download_checkpoint()
    model_dict = json.loads((hf_path / "config.json").read_text())["model"]
    model_dict = fix_config_dict(model_dict)
    hf_model = Cosmos3OmniModel.from_pretrained_dcp(hf_path)

    print("Saving model...")
    storage_writer = FileSystemWriter(args.output_path / "model")
    dcp.save(
        state_dict=get_model_state_dict(hf_model.model), storage_writer=storage_writer, planner=CustomSavePlanner()
    )
    shutil.copy(hf_path / "checkpoint.json", args.output_path / "checkpoint.json")
    shutil.copy(hf_path / "config.json", args.output_path / "model/config.json")

    print(f"Saved checkpoint to {args.output_path}")


def main():
    args = tyro.cli(Args, description=__doc__, config=(tyro.conf.OmitArgPrefixes,))
    convert_model_to_dcp(args)


if __name__ == "__main__":
    main()
