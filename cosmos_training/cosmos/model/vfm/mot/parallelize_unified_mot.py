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

import torch
import torch.nn as nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper as ptd_checkpoint_wrapper,
)
from torch.distributed.fsdp import fully_shard

from configs.base.defaults.model_config import ParallelismConfig
from cosmos.utils.vfm.parallelism import ParallelDims


def apply_ac(model: nn.Module):
    """Apply activation checkpointing to the model."""

    for layer_id, block in model.model.layers.named_children():
        block = ptd_checkpoint_wrapper(block, preserve_rng_state=True)
        model.model.layers.register_module(layer_id, block)


def apply_compile(model: nn.Module, config: ParallelismConfig):
    """
    Apply torch.compile to each TransformerBlock, which makes compilation efficient due to
    repeated structure. Alternatively one can compile the whole model (after applying DP).
    """
    compile_options = {}
    if config.max_autotune_pointwise:
        compile_options["max_autotune_pointwise"] = True
    if config.coordinate_descent_tuning:
        compile_options["coordinate_descent_tuning"] = True

    for layer_id, block in model.model.layers.named_children():
        block = torch.compile(
            block,
            fullgraph=True,
            dynamic=config.compile_dynamic,
            mode="reduce-overhead" if config.use_cuda_graphs else None,
            options=compile_options or None,
        )
        model.model.layers.register_module(layer_id, block)


def apply_fsdp(
    model: nn.Module,
    parallel_dims: ParallelDims,
):
    """
    Apply data parallelism (via FSDP2) to the model.

    Args:
        model (nn.Module): The model to apply data parallelism to.
        parallel_dims (ParallelDims): The device mesh to use for data parallelism and expert parallel.
    """
    for _, block in model.model.layers.named_children():
        fully_shard(block, mesh=parallel_dims.dp_mesh)


def parallelize_unified_mot(
    model: nn.Module,
    parallel_dims: ParallelDims | None,
    config: ParallelismConfig,
) -> nn.Module:
    """Optimize the model using FSDP, activation checkpointing, and torch.compile.

    FSDP reduces memory usage by sharding the model parameters across multiple GPUs.
    Activation checkpointing reduces memory usage by selectively checkpointing only
    the outputs of each layer. Torch.compile compiles the model for faster training.
    """
    if config.use_activation_checkpointing:
        apply_ac(model)
    if config.use_torch_compile:
        apply_compile(model, config)
    if parallel_dims is not None and parallel_dims.dp_enabled:
        apply_fsdp(model, parallel_dims)
    return model
