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
"""FSDP2 wrapping for Cosmos3 VLM ``HFModel`` instances.

Hosts the single VLM-specific ``parallelize`` entry point used by
``vlm_model.VLMModel._init_vlm``.  Lives under ``projects/cosmos3/vfm/models/``
so the FSDP wrapping concern sits next to the model class it operates on
(mirroring the layout of ``models/mot/parallelize_unified_mot.py`` for the
MoT path).

Pure parallelism plumbing — :class:`~projects.cosmos3.vfm.utils.parallelism.ParallelDims`
and its meshes — stays in ``vfm/utils/parallelism.py``.
"""

import torch
from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy, fully_shard

from cosmos3._src.imaginaire.utils import log
from cosmos3._src.vfm.models.hf_model import HFModel
from cosmos3._src.vfm.utils.parallelism import ParallelDims


def parallelize(
    model: HFModel,
    parallel_dims: ParallelDims,
    train_config,
) -> None:
    """Apply FSDP2 to an HFModel in-place.

    Uses torch.distributed.fsdp.fully_shard (FSDP2).  Each transformer block is
    sharded individually for fine-grained memory savings; the outer model is then
    wrapped to cover remaining parameters (embeddings, layer norms, lm_head).

    Supported architectures:
    - Language models: ``inner.model.layers`` (standard HF LLM structure)
    - Vision-language models: additionally ``inner.visual.blocks`` (Qwen3-VL)

    No-op when FSDP is not needed (single-GPU or replicate-only).

    Args:
        model:         HFModel instance (``_model`` attribute must be on meta or CPU device).
        parallel_dims: ParallelDims with meshes already built via
                       :meth:`ParallelDims.build_meshes`.
        train_config:  Train sub-config — provides ``param_torch_dtype`` and
                       ``fsdp_offload``.  Pass the TRAIN sub-config object,
                       not the full config.
    """
    if not parallel_dims.dp_shard_enabled:
        # No shard axis: dp_shard <= 1.  FSDP2 (fully_shard) has nothing to do.
        # For replicate-only (dp_replicate > 1, dp_shard == 1), use DDP outside
        # this function.
        log.info("parallelize: dp_shard <= 1 — skipping FSDP2 wrapping")
        return

    mp_policy = MixedPrecisionPolicy(
        param_dtype=train_config.param_torch_dtype,
        reduce_dtype=torch.float32,
    )

    # 2-D (dp_replicate × dp_shard) mesh for HSDP, or 1-D dp_shard sub-mesh
    # for pure FSDP. In the overlay design cp does NOT fold into the FSDP
    # shard axis; cp/cfgp are handled by separate meshes.
    if parallel_dims.dp_replicate_enabled:
        fsdp_mesh = parallel_dims.dp_mesh
    else:
        fsdp_mesh = parallel_dims.dp_shard_mesh
    fsdp_kwargs = {"mesh": fsdp_mesh, "mp_policy": mp_policy}

    inner = model._model

    no_split_names = set(getattr(inner, "_no_split_modules", []))
    wrapped = 0
    for module in reversed(list(inner.modules())):
        if type(module).__name__ in no_split_names:
            fully_shard(module, **fsdp_kwargs)
            wrapped += 1
    log.info(f"Wrapped {wrapped} sub-modules.")

    # Wrap the full inner model to cover remaining parameters
    # (embed_tokens, final layer norm, lm_head, visual projector stem, etc.)
    cpu_offload_policy = None
    if getattr(train_config, "fsdp_offload", False):
        cpu_offload_policy = CPUOffloadPolicy()

    fully_shard(inner, offload_policy=cpu_offload_policy, **fsdp_kwargs)
    log.info("parallelize: FSDP2 applied to HFModel._model")
