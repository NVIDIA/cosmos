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

"""Convert DCP checkpoint to Hugging Face model."""

from cosmos3.common.init import init_script

init_script(
    env={
        "COSMOS_DEVICE": "cpu",
        "COSMOS_TRAINING": "1",
    }
)

import json
from pathlib import Path
from typing import Annotated, Any, Callable

import attrs
import safetensors.torch
import torch.distributed.checkpoint as dcp
import tyro
from torch.distributed.checkpoint.filesystem import FileSystemReader
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict

from cosmos3.common.args import (
    CheckpointOverrides,
    ParallelismOverrides,
    ResolvedPath,
    tyro_cli,
)
from cosmos3.common.checkpoints import register_checkpoints
from cosmos3.common.config import serialize_config_dict
from cosmos3.common.init import is_rank0
from cosmos3.model import Cosmos3OmniConfig, Cosmos3OmniModel
from cosmos3._src.imaginaire.checkpointer.s3_filesystem import S3StorageReader
from cosmos3._src.imaginaire.lazy_config.registry import convert_target_to_string
from cosmos3._src.imaginaire.utils import log
from cosmos3._src.imaginaire.utils.checkpoint_db import CheckpointConfig, sanitize_uri
from cosmos3._src.vfm.checkpointer.dcp import CustomLoadPlanner
from cosmos3._src.vfm.configs.base.defaults.model_config import OmniMoTModelConfig
from cosmos3._src.vfm.models.omni_mot_model import OmniMoTModel


def _coerce_to_base_model(model_dict: dict[str, Any]) -> None:
    """For distillation training configs, rewrite the target to the base
    OmniMoTModel so the exported checkpoint only contains the student network."""
    target = model_dict.get("_target_", "")
    if "OmniMoTModel" in target:
        return

    log.info(f"Overriding model target from {target} to OmniMoTModel for export")
    model_dict["_target_"] = convert_target_to_string(OmniMoTModel)

    config = model_dict["config"]
    base_field_names = {f.name for f in attrs.fields(OmniMoTModelConfig)}
    extra_keys = [k for k in config if k not in base_field_names and not k.startswith("_")]
    for k in extra_keys:
        del config[k]

    metadata = config.get("_metadata", {})
    metadata["object_type"] = convert_target_to_string(OmniMoTModelConfig)
    config["_metadata"] = metadata


class Args(ParallelismOverrides):
    checkpoint: CheckpointOverrides = CheckpointOverrides.model_construct()
    output_dir: Annotated[ResolvedPath, tyro.conf.arg(aliases=("-o",))]
    """Output model directory."""
    config_only: bool = False
    """If True, only export config."""
    vit: bool = False
    """If True, export ViT weights."""


def _load_safetensor_weights(model_dir: Path, predicate: Callable[[str], bool]) -> dict:
    """Load weights from a safetensors file."""
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            weight_map = json.load(f)["weight_map"]
        shards = {v for k, v in weight_map.items() if predicate(k)}
        vision_weights = {}
        for shard in shards:
            tensors = safetensors.torch.load_file(model_dir / shard)
            vision_weights.update({k: v for k, v in tensors.items() if predicate(k)})
    else:
        tensors = safetensors.torch.load_file(model_dir / "model.safetensors")
        vision_weights = {k: v for k, v in tensors.items() if predicate(k)}
    return vision_weights


def export_model(args: Args):
    register_checkpoints()
    checkpoint_args = args.checkpoint.build_checkpoint(checkpoints={})
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    log.info("Loading config...")
    model_dict = checkpoint_args.load_model_config_dict()
    model_dict["config"]["ema"]["enabled"] = False

    # Download VLM checkpoint
    if args.vit:
        vlm_checkpoint_path = model_dict["config"]["vlm_config"]["checkpoint_path"]
        vlm_checkpoint_path = sanitize_uri(vlm_checkpoint_path)
        checkpoint: CheckpointConfig | None = CheckpointConfig.maybe_from_uri(vlm_checkpoint_path)
        if checkpoint is None:
            raise ValueError(f"Invalid checkpoint path: {vlm_checkpoint_path}")
        vlm_checkpoint_path = checkpoint.hf.download()
    else:
        vlm_checkpoint_path = None

    # Load model
    log.info("Loading model...")
    _coerce_to_base_model(model_dict)
    hf_config = Cosmos3OmniConfig(model=model_dict)
    hf_config.save_pretrained(args.output_dir)
    hf_model = Cosmos3OmniModel(hf_config)

    # Save model
    log.info("Saving model...")
    if not args.config_only:
        # Load checkpoint
        if checkpoint_args.checkpoint_path.startswith("s3://"):
            storage_reader = S3StorageReader(
                credential_path=checkpoint_args.credential_path,
                path=checkpoint_args.checkpoint_path,
            )
        else:
            storage_reader = FileSystemReader(checkpoint_args.checkpoint_path)
        state_dict = get_model_state_dict(hf_model.model)
        dcp.load(
            state_dict=state_dict,
            storage_reader=storage_reader,
            planner=CustomLoadPlanner(
                load_ema_to_reg=checkpoint_args.use_ema_weights,
            ),
        )
        state_dict = get_model_state_dict(
            hf_model,
            options=StateDictOptions(
                full_state_dict=True,
                cpu_offload=True,
            ),
        )
        if not is_rank0():
            return

        # Load ViT from VLM checkpoint
        if args.vit:
            assert vlm_checkpoint_path is not None
            vit_state_dict = _load_safetensor_weights(
                Path(vlm_checkpoint_path), lambda x: x.startswith("model.visual.")
            )
            assert vit_state_dict, "No vision weights found"
            state_dict.update(vit_state_dict)

        # Save checkpoint
        hf_model.save_pretrained(
            args.output_dir,
            state_dict=state_dict,
        )

    # Re-write 'config.json' to apply replacements.
    hf_config_file = args.output_dir / "config.json"
    hf_config_json = json.loads(hf_config_file.read_text())
    serialize_config_dict(hf_config_json, hf_config_file)

    # Write 'checkpoint.json' last to indicate that the model is complete.
    serialize_config_dict(checkpoint_args.model_dump(mode="json"), args.output_dir / "checkpoint.json")

    print(f"Saved model to {args.output_dir}")


def main():
    args = tyro_cli(Args, description=__doc__, config=(tyro.conf.OmitArgPrefixes,))
    export_model(args)


if __name__ == "__main__":
    main()
