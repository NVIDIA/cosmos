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

import os
from dataclasses import MISSING, field
from typing import Union

import attrs
import torch

from cosmos3._src.imaginaire.config import make_freezable


def skip_ui_field(*, default=MISSING, default_factory=MISSING, **kwargs):
    metadata = kwargs.pop("metadata", {})
    metadata["skip_ui"] = True
    if default_factory is not MISSING:
        return field(default_factory=default_factory, metadata=metadata, **kwargs)
    elif default is not MISSING:
        return field(default=default, metadata=metadata, **kwargs)
    else:
        raise ValueError("Must provide either default or default_factory.")


@make_freezable
@attrs.define(slots=False)
class TrainPolicyConfig:
    mini_batch: int = 1
    type: str = "sft"


@make_freezable
@attrs.define(slots=False)
class FP8:
    enable_fp8: bool = False


@make_freezable
@attrs.define(slots=False)
class TrainConfig:
    # Whether to use async tensor parallelism
    async_tp_enabled: bool = False
    # Whether to use torch.compile
    compile: bool = False

    # The data type for parameters and activations
    param_dtype: str = "bfloat16"
    master_dtype: str = "float32"

    # The data type for reduction in FSDP
    fsdp_reduce_dtype: str = "float32"

    # Whether to offload the model to CPU if using FSDP
    fsdp_offload: bool = False

    # Reshard the param after forward pass in FSDP
    fsdp_reshard_after_forward: str = "default"

    # The batch size for training per iteration in one replica, this is the local batch size for each gradient accumulation step
    train_batch_per_replica: int = 1

    # The interval of train step for synchronizing weights between replicas.
    sync_weight_interval: int = 1

    # Train policy
    train_policy: TrainPolicyConfig = TrainPolicyConfig()
    fp8: FP8 = FP8()
    deterministic: bool = True

    def __post_init__(self):
        self.ckpt.__post_init__()
        if self.async_tp_enabled and not self.compile:
            raise ValueError("Async tensor parallelism requires torch.compile to be enabled")

    def key_values(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

    @property
    def param_torch_dtype(self):
        return {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[self.param_dtype]

    @property
    def master_torch_dtype(self):
        return {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[self.master_dtype]

    @property
    def fsdp_reduce_torch_dtype(self):
        return {"float32": torch.float32}[self.fsdp_reduce_dtype]


@make_freezable
@attrs.define(slots=False)
class ParallelismConfig:
    # Number of initial replicas to be created
    n_init_replicas: int = 1
    # Tensor parallelism size
    tp_size: int = 1
    # Context parallelism size
    cp_size: int = 1
    # Expert parallelism size
    ep_size: int = 1
    # Data Parallelism size in sharded mode
    dp_shard_size: int = -1
    # Pipeline parallelism size
    pp_size: int = 1
    # Pipeline parallelism dynamic shape
    pp_dynamic_shape: bool = False
    # Pipeline parallelism micro batch size
    pp_micro_batch_size: int = 1
    # Data Parallelism size in replicate mode
    dp_replicate_size: int = 1
    # The method to rotate kv shards during context parallelism
    cp_rotate_method: str = "allgather"

    @property
    def world_size(self):
        world_size = os.environ.get("WORLD_SIZE", 1)
        return int(world_size)

    @property
    def local_world_size(self):
        local_world_size = os.environ.get("LOCAL_WORLD_SIZE", 1)
        return int(local_world_size)


# Why we does not make this freezable?
# Because we need to path the cache model dir as model_name_or_path to the cosmos-rl model to use the
# model weights downloaded from s3. If cosmos-rl support reading model from s3 directly, we can make it freezable.
@attrs.define(slots=False)
class PolicyConfig:
    # Parallelism configuration
    parallelism: ParallelismConfig = ParallelismConfig()
    # The model name or path, compatible with huggingface model name or local path
    model_name_or_path: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    # The maximum length for training, longer than this will be ignored for training stability
    model_max_length: int = 16000
    # Whether to use gradient checkpointing
    model_gradient_checkpointing: bool = True
    # The maximum length for video tokens, only applied to qwen model
    qwen_max_video_token_length: int = 8000
    param_torch_dtype: str = "bfloat16"

    # Pretrain weights (Optional)
    pretrain_weights_path_vlm: str = ""
    pretrain_weights_path_llm: str = ""
    pretrain_weights_path_vit: str = ""
    pretrain_weights_cred: str = "credentials/s3_training.secret"

    # Extra model config
    lora: Union[str, None] = None
    enable_liger_kernel: bool = False
    trainable_map: Union[str, None] = None
    monkey_patch_for_text_only_data: bool = False
