# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""
Checkpoint utilities for VFM models.
Contains utilities for loading model checkpoints with key remapping.
"""

import time

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.metadata import STATE_DICT_TYPE, Metadata

from cosmos.checkpoint.s3_filesystem import S3StorageReader
from cosmos.utils import log


class RenameLoadPlanner(dcp.DefaultLoadPlanner):
    """
    RenameLoadPlanner that does two renaming operations during pretrained model load.
    1. Renames model parameters from "model." to "model.language_model." if the core model is a VLM.
    2. Renames model parameters from "_orig_mod." to "." since the pretrained model does not have torch compiled modules.
    """

    def set_up_planner(
        self,
        state_dict: STATE_DICT_TYPE,
        metadata: Metadata | None = None,
        is_coordinator: bool = False,
    ) -> None:
        state_dict = remove_orig_mode_from_state_dict(state_dict)

        # Renames model parameters from "model." to "model.language_model." if the core model is a VLM.
        # This is necessary because the checkpoint is saved with "model.language_model." for a VLM.
        # If the core model is an LLM, this renaming is not necessary.
        has_language_model = any("model.language_model." in key for key in metadata.state_dict_metadata)
        if has_language_model:
            state_dict = insert_language_model_in_state_dict(state_dict)

        # Perform more error checking to ensure the checkpoint is valid. If the keys are missing,
        # then the missing keys should be from the generation pathway. All keys from the
        # understanding pathway must be present in the checkpoint. Additionally, for 2B and 4B
        # dense Qwen VLMs, the `lm_head.weight` key is not present in the checkpoint. For these
        # models, the input embedding and generation layer share the same params due to
        # `tie_word_embeddings` being set to True in the configs. For the 0.6B LLM, 8B and 32B dense
        # VLMs, and the 30B and 235B MoE VLMs, the `lm_head.weight` key is present in the
        # checkpoint.
        for key in state_dict:
            if key not in metadata.state_dict_metadata and "_moe_gen" not in key and "lm_head.weight" not in key:
                raise ValueError(f"Missing key: {key} in checkpoint.")

        # If the keys are unexpected, then the unexpected keys should be from the visual part of the
        # VLM. All keys in the language part should be used by the Cosmos3 model.
        for key in metadata.state_dict_metadata:
            if key not in state_dict and "model.visual." not in key:
                raise ValueError(f"Unexpected key: {key} in checkpoint.")

        super().set_up_planner(
            state_dict=state_dict,
            metadata=metadata,
            is_coordinator=is_coordinator,
        )


def remove_orig_mode_from_state_dict(state_dict: STATE_DICT_TYPE) -> STATE_DICT_TYPE:
    """Renames model parameters from "_orig_mod." to "." since the pretrained model does not have torch compiled modules."""
    return {key.replace("_orig_mod.", ""): value for key, value in state_dict.items()}


def insert_language_model_in_state_dict(state_dict: STATE_DICT_TYPE) -> STATE_DICT_TYPE:
    """Renames model parameters from "model." to "model.language_model." if the core model is a VLM."""
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("model."):
            new_state_dict[key.replace("model.", "model.language_model.")] = value
        else:
            new_state_dict[key] = value
    return new_state_dict


def load_language_model(
    model: torch.nn.Module, checkpoint_path: str, credential_path: str, enable_gcs_patch_in_boto3: bool = False
) -> None:
    """
    Universal language model loading function using DCP format.
    Handles key remapping for "model.language_model." -> "model." by default.

    Args:
        model: The language model to load weights into
        checkpoint_path: Path to checkpoint (must end with .safetensors or dcp)
        credential_path: Path to S3 credentials
        enable_gcs_patch_in_boto3: Whether to enable GCS patch in boto3 for DCP loading from GCS
    """
    start_time = time.time()
    if not checkpoint_path.strip("/").endswith("dcp"):
        raise ValueError(f"Checkpoint path {checkpoint_path} must end with dcp")

    log.info(f"Loading language model weights in DCP format from: {checkpoint_path}")
    state_dict = model.state_dict()

    # Choose storage reader
    if checkpoint_path.startswith("s3://"):
        storage_reader = S3StorageReader(
            credential_path=credential_path,
            path=checkpoint_path,
            enable_gcs_patch_in_boto3=enable_gcs_patch_in_boto3,
        )
    else:
        storage_reader = dcp.FileSystemReader(checkpoint_path)

    # Load checkpoint
    dcp.load(
        state_dict=state_dict,
        storage_reader=storage_reader,
        planner=RenameLoadPlanner(allow_partial_load=False),
    )

    log.info(f"Successfully loaded language model from {checkpoint_path}")
    log.info(f"Time taken to load language model: {time.time() - start_time} seconds")
