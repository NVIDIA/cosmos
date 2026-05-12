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

from __future__ import annotations

from contextlib import contextmanager
from functools import partial

import torch
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp._runtime_utils import (
    _post_forward,
    _post_forward_reshard,
    _pre_forward,
    _pre_forward_unshard,
    _root_pre_forward,
)
from torch.distributed.utils import _p_assert

from cosmos3._src.imaginaire.utils import distributed, log


def apply_fsdp_checkpointing(model, list_block_cls):
    """apply activation checkpointing to model
    returns None as model is updated directly
    """
    log.critical("--> applying fdsp activation checkpointing...")
    non_reentrant_wrapper = partial(
        checkpoint_wrapper,
        # offload_to_cpu=False,
        checkpoint_impl=CheckpointImpl.NO_REENTRANT,
    )

    def check_fn(submodule):
        result = False
        for block_cls in list_block_cls:
            if isinstance(submodule, block_cls):
                result = True
                break
        return result

    apply_activation_checkpointing(model, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=check_fn)


@contextmanager
def possible_fsdp_scope(
    model: torch.nn.Module,
):
    enabled = isinstance(model, FSDP)
    if enabled:
        assert not torch.is_grad_enabled(), "FSDP context should be entered with grad disabled"
        handle = model._handle
        args, kwargs = [0], dict(dummy=0)
        with torch.autograd.profiler.record_function("FullyShardedDataParallel.possible_fsdp_scope"):
            args, kwargs = _root_pre_forward(model, model, args, kwargs)
            unused = None
            args, kwargs = _pre_forward(
                model,
                handle,
                _pre_forward_unshard,
                model._fsdp_wrapped_module,
                args,
                kwargs,
            )
            if handle:
                _p_assert(
                    handle.flat_param.device == model.compute_device,
                    "Expected `FlatParameter` to be on the compute device "
                    f"{model.compute_device} but got {handle.flat_param.device}",
                )
    try:
        yield None
    finally:
        if enabled:
            output = {"output": 1}
            _post_forward(model, handle, _post_forward_reshard, model, unused, output)


def hsdp_device_mesh(replica_group_size=None, sharding_group_size=None, device=None):
    """
     Initializes a device mesh for use with Hybrid Sharding strategy in FSDP (HSDP) training.

    This function requires explicit sizes for replica and sharding groups to accommodate models
    whose GPU fit is unknown, providing flexibility in distributed training setups.

    Args:
        replica_group_size (int): The size of each replica group. Must be provided to ensure
            the model fits within the available resources.
        sharding_group_size (int): The size of each sharding group that the model can fit. Must be provided to
            ensure the correct distribution of model parameters.
        device (str, optional): The device to use (e.g., "cuda:0"). If None, defaults to "cuda"
            with the local rank as the device index.

    Returns:
        A device mesh object compatible with FSDP.

    Raises:
        ValueError: If replica_group_size or sharding_group_size are not provided, or if the
            world size is not evenly divisible by the sharding group size.
        RuntimeError: If a valid device mesh cannot be created.

    Usage:
        If your model fits on 4 GPUS, and you have 3 nodes of 8 GPUs, then:
        Sharding_Group_Size = 4
        Replica_Groups_Size = (24 total gpus, 4 per sharding group) = 6 Replica Groups
        >>> device_mesh = initialize_device_mesh(replica_group_size, sharding_group_size)
        >>> sharded_model = FSDP(model, device_mesh=device_mesh, ...)
    """

    # world_size = int(os.getenv("WORLD_SIZE", "1"))
    world_size = distributed.get_world_size()
    if sharding_group_size is None:
        sharding_group_size = min(world_size, 8)
    sharding_group_size = min(sharding_group_size, world_size)
    if replica_group_size is None:
        replica_group_size = world_size // sharding_group_size

    device = device or "cuda"

    if world_size % sharding_group_size != 0:
        raise ValueError(
            f"World size {world_size} is not evenly divisible by sharding group size {sharding_group_size}."
        )

    if (world_size // sharding_group_size) % replica_group_size != 0:
        raise ValueError(
            f"The calculated number of replica groups is not evenly divisible by "
            f"replica_group_size {replica_group_size}."
        )

    device_mesh = init_device_mesh(
        device, (replica_group_size, sharding_group_size), mesh_dim_names=("replicate", "shard")
    )
    if device_mesh is None:
        raise RuntimeError("Failed to create a valid device mesh.")

    log.critical(
        f"Device mesh initialized with replica group size {replica_group_size} and sharding group size {sharding_group_size}"
    )

    return device_mesh
