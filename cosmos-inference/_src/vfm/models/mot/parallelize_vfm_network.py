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

from typing import Callable

import torch
from torch.distributed.fsdp import fully_shard
from torch.nn.attention.flex_attention import BlockMask

from cosmos3._src.vfm.configs.base.defaults.model_config import ParallelismConfig
from cosmos3._src.vfm.datasets.sequence_packing import (
    FactoredSequencePack,
    JointSequencePack,
)
from cosmos3._src.vfm.models.mot.attention import SplitInfo, dispatch_attention
from cosmos3._src.vfm.models.mot.context_parallel_utils import context_parallel_attention
from cosmos3._src.vfm.models.mot.parallelize_unified_mot import parallelize_unified_mot
from cosmos3._src.vfm.models.utils.memory import KVToStore, MemoryValue
from cosmos3._src.vfm.utils.parallelism import ParallelDims


class ContextParallelDispatch(torch.nn.Module):
    """CP-aware wrapper for the installed attention dispatch function.

    Installed on ``PackedAttentionMoT.dispatch_attention_fn`` when context
    parallelism is enabled, replacing whatever dispatch function was there
    previously.  The call signature of :meth:`forward` matches
    ``dispatch_attention`` so the two are interchangeable.

    All paths delegate to :func:`context_parallel_attention`, which wraps
    the inner ``wrapped_dispatch`` with Ulysses-style all-to-all
    communication.  This includes the AR frame 1+ gen-only path — the inner
    dispatch routes to ``attention_AR_gen_only`` which operates on the
    head-sharded tensors produced by the all-to-all.

    All cache writes flow through the ``MemoryState`` interface; neither this
    class nor the CP attention functions write to the cache directly.
    """

    def __init__(
        self,
        cp_mesh,
        wrapped_dispatch: Callable = dispatch_attention,
    ):
        super().__init__()
        self.cp_mesh = cp_mesh
        self.wrapped_dispatch = wrapped_dispatch

    def forward(
        self,
        packed_query_states: FactoredSequencePack | JointSequencePack,
        packed_key_states: FactoredSequencePack | JointSequencePack,
        packed_value_states: FactoredSequencePack | JointSequencePack,
        attention_mask: BlockMask | SplitInfo,
        natten_metadata: dict | None = None,
        memory_value: MemoryValue | None = None,
    ) -> tuple[FactoredSequencePack | JointSequencePack, KVToStore | None]:
        if memory_value is not None and not memory_value.supports_context_parallel_attention:
            raise ValueError("Context-parallel doesn't work when training with a KV-cache.")

        return context_parallel_attention(
            self.cp_mesh,
            packed_query_states,
            packed_key_states,
            packed_value_states,
            attention_mask,
            attention_function=self.wrapped_dispatch,
            natten_metadata=natten_metadata,
            memory_value=memory_value,
        )


def apply_compile(model: torch.nn.Module, config: ParallelismConfig):
    """Apply torch.compile to the VFM encode/decode heads.

    The MoT-side ``compile_dynamic`` knob on ``ParallelismConfig`` intentionally
    does **not** propagate here.  The VFM encode/decode paths have no graph
    breaks and their input shapes are stable across a prompt, so we always
    trace them as a single dynamic graph (``fullgraph=True, dynamic=True``).
    This keeps AR inference (which sets ``compile_dynamic=False`` on MoT for
    shape-specialized kernels) from accidentally regressing the VFM compile.
    """

    inductor_options = {}
    if config.max_autotune_pointwise:
        inductor_options["max_autotune_pointwise"] = True
    if config.coordinate_descent_tuning:
        inductor_options["coordinate_descent_tuning"] = True

    compile_options = {
        "fullgraph": True,
        "dynamic": True,
        "mode": "reduce-overhead" if config.use_cuda_graphs else None,
        "options": inductor_options or None,
    }

    model._encode_text = torch.compile(model._encode_text, **compile_options)
    model._encode_vision = torch.compile(model._encode_vision, **compile_options)
    model._encode_action = torch.compile(model._encode_action, **compile_options)
    model._decode_vision = torch.compile(model._decode_vision, **compile_options)
    model._decode_action = torch.compile(model._decode_action, **compile_options)
    return model


def context_parallel_unified_mot(
    model: torch.nn.Module,
    parallel_dims: ParallelDims | None,
) -> torch.nn.Module:
    for i in range(len(model.model.layers)):
        attn = model.model.layers[i].self_attn
        cp_dispatch = ContextParallelDispatch(
            parallel_dims.cp_mesh,
            wrapped_dispatch=attn.dispatch_attention_fn,
        )
        attn.dispatch_attention_fn = cp_dispatch
        attn.cp_mesh = parallel_dims.cp_mesh

    return model


def parallelize_vfm_network(
    model: torch.nn.Module,
    parallel_dims: ParallelDims | None,
    config: ParallelismConfig,
) -> torch.nn.Module:
    """Optimize the model using FSDP, CP, activation checkpointing, and torch.compile.

    FSDP reduces memory usage by sharding the model parameters across multiple GPUs.
    Activation checkpointing reduces memory usage by selectively checkpointing only
    the outputs of each layer. Torch.compile compiles the model for faster training.
    """
    if parallel_dims is not None and parallel_dims.cp_enabled:
        model.parallel_dims = parallel_dims
        model.language_model = context_parallel_unified_mot(
            model.language_model,
            parallel_dims=parallel_dims,
        )

    model.language_model = parallelize_unified_mot(
        model.language_model,
        parallel_dims=parallel_dims,
        config=config,
    )

    if config.use_torch_compile and config.compiled_region == "all":
        model = apply_compile(model, config)

    if parallel_dims is not None and parallel_dims.dp_enabled:
        # Collect parameters to ignore during FSDP wrapping
        ignored_params = set()
        if model.latent_pos_embed is not None:
            ignored_params.update(model.latent_pos_embed.parameters())

        model = fully_shard(
            module=model,
            mesh=parallel_dims.dp_mesh,
            ignored_params=ignored_params,
        )

    return model
