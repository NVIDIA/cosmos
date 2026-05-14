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

"""Distributed safetensors loading and HF→Cosmos3 weight conversion.

Layered API
-----------

Three layers of functionality, lowest first:

1. **Multi-rank checkpoint I/O** — :class:`MultiRankCheckpointLoader` distributes
   safetensors file reads across the FSDP ``dp_shard`` ranks and then
   broadcasts each tensor to every rank.  It is checkpoint-format-agnostic:
   it just yields ``(name, tensor)`` pairs from the raw HF state dict.

2. **Name / weight conversion** — Per-family converters translate raw HF
   parameter names (and optionally shard the tensor along FSDP / EP axes)
   into the Cosmos3 VFM layout:

   - :func:`convert_weight_from_qwen3_hf`         — Qwen3 VL / LLM (dense + MoE).
   - :func:`convert_weight_from_nemotron_vl_hf`   — Nemotron-3 Dense VL (hybrid 56-block layout).
   - :func:`convert_weight_from_nemotron_llm_hf`  — Nemotron-3 pure LLM.

   For the generic VLM path, :func:`_make_name_converter` consumes the model's
   ``_checkpoint_conversion_mapping`` (transformers v4) or falls back to
   suffix-lookup against the model's own state dict (transformers v5).

3. **High-level loaders** — Composing the above:

   - :func:`load_language_model` — loads HF text-tower weights into the MoT
     language model.  Auto-detects the checkpoint format
     (:func:`detect_vlm_checkpoint_format`).
   - :func:`load_vlm_model` — generic loader for HF VLM checkpoints into an
     FSDP-wrapped ``HFModel``; honors a skip-pattern overlay and the
     ``fsdp_offload`` mode.

Borrowed from cosmos_rl's ``MultiRankWeightLoader`` (renamed to
``MultiRankCheckpointLoader`` here) with modifications for loading from
S3 / GCS and support for Cosmos3 VFM models.
https://github.com/nvidia-cosmos/cosmos-rl/blob/main/cosmos_rl/utils/multi_rank_weight_loader.py
"""

import os
import re
import time
from collections.abc import Callable, Iterator

import torch
import torch.distributed as dist
from safetensors.torch import load as load_safetensors
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor

from cosmos.utils.flags import INTERNAL
from cosmos.utils import log
from cosmos.utils.easy_io import easy_io
from cosmos.utils.vfm.parallelism import ParallelDims

# Prefixes stripped when matching checkpoint keys to model state-dict keys.
# Order matters: longest first.  For each model key, the longest matching
# prefix is stripped (yielding the most specific tail) before we record it
# in the lookup table.  The trailing empty string acts as a default that
# leaves keys without any known prefix unchanged.
# Ref: cosmos-rl cosmos_rl/policy/model/hf_models/__init__.py:465-472.
_VLM_KEY_PREFIXES: tuple[str, ...] = (
    "model.language_model.model.",
    "model.language_model.",
    "language_model.model.",
    "language_model.",
    "model.",
    "",
)

_HF_URI_PREFIX = "hf://"


def _looks_like_hf_repo_id(checkpoint_path: str) -> bool:
    """Return True for unambiguous bare Hugging Face repo IDs.

    Explicit ``hf://`` paths are handled separately.  For bare paths, require the
    common ``namespace/repo`` shape so local relative paths such as ``ckpt`` are
    not silently treated as Hub repos.
    """
    if os.path.exists(os.path.expanduser(checkpoint_path)):
        return False
    if checkpoint_path.startswith(("/", "./", "../", "~")):
        return False
    if "://" in checkpoint_path:
        return False
    return re.fullmatch(r"[\w.-]+/[\w.-]+", checkpoint_path) is not None


def _download_hf_checkpoint(checkpoint_path: str) -> str:
    """Download safetensors from Hugging Face Hub and return the local snapshot path."""
    from huggingface_hub import snapshot_download

    repo_id = checkpoint_path.removeprefix(_HF_URI_PREFIX)
    hf_home = os.environ.get("HF_HOME")
    cache_dir = os.path.join(hf_home, "hub") if hf_home else None
    token = os.environ.get("HF_TOKEN")
    log.info(f"Resolving Hugging Face checkpoint: {repo_id}", rank0_only=False)
    local_path = snapshot_download(
        repo_id=repo_id,
        token=token,
        cache_dir=cache_dir,
        allow_patterns=["*.safetensors", "*.safetensors.index.json"],
    )
    log.info(f"Resolved Hugging Face checkpoint {repo_id} to {local_path}", rank0_only=False)
    return local_path


def _is_hf_checkpoint_candidate(checkpoint_path: str) -> bool:
    return checkpoint_path.startswith(_HF_URI_PREFIX) or _looks_like_hf_repo_id(checkpoint_path)


def _make_backend_args(checkpoint_path: str, credential_path: str | None) -> dict[str, str | None] | None:
    if checkpoint_path.startswith("s3://"):
        return {
            "backend": "s3",
            "s3_credential_path": credential_path,
        }
    return None


def _list_safetensors_files(
    checkpoint_path: str,
    backend_args: dict[str, str | None] | None,
) -> list[str]:
    return list(
        easy_io.list_dir_or_file(
            checkpoint_path,
            list_dir=False,
            list_file=True,
            suffix="safetensors",
            recursive=False,
            backend_args=backend_args,
        )
    )


def _get_local_rank_and_size(device_mesh: DeviceMesh) -> tuple[int, int]:
    """Get the local rank and size of a device mesh.

    Args:
        device_mesh: The device mesh to get the attributes from.

    Returns:
        A tuple of (local rank, size).
    """
    return device_mesh.get_local_rank(), device_mesh.size()


def _shard_tensor_on_fsdp_mesh(
    tensor: torch.Tensor,
    parallel_dims: ParallelDims | None,
) -> torch.Tensor:
    """Slice ``tensor`` along dim 0 according to the FSDP ``dp_shard`` mesh.

    Returns the rank-local shard when ``dp_shard`` is enabled, otherwise the
    full tensor (made contiguous).  Requires that ``tensor.shape[0]`` is
    divisible by ``dp_shard_size`` — this is a hard requirement of the even-
    split semantics; uneven splits should go through :func:`_shard_first_dim`.

    Args:
        tensor: The tensor to shard.
        parallel_dims: Parallel dims object, or None for single-rank.

    Returns:
        Contiguous rank-local shard (or full tensor if dp_shard is disabled).
    """
    if parallel_dims is None or not parallel_dims.dp_shard_enabled:
        return tensor.contiguous()

    fsdp_rank, fsdp_size = _get_local_rank_and_size(parallel_dims.dp_shard_mesh)
    if tensor.shape[0] % fsdp_size != 0:
        raise ValueError(f"Shard shape {tensor.shape} is not divisible by dp_shard_size {fsdp_size} on dim 0")
    shard = tensor.chunk(chunks=fsdp_size, dim=0)[fsdp_rank]
    return shard.contiguous()


def _get_dp_shard_mesh(parallel_dims: ParallelDims | None) -> DeviceMesh | None:
    """Get the dp_shard mesh from the parallel dimensions.

    Args:
        parallel_dims: The parallel dimensions to use for the conversion.

    Returns:
        The dp_shard mesh, or None if dp_shard is not enabled.
    """
    if parallel_dims is not None and parallel_dims.dp_shard_enabled:
        return parallel_dims.dp_shard_mesh
    else:
        return None


def _build_model_key_by_tail(state_dict: dict) -> dict[str, str]:
    """Build a ``tail → model_key`` lookup table for suffix-based key matching.

    For each model key, strip the longest matching prefix in
    ``_VLM_KEY_PREFIXES`` and record ``tail -> model_key``.  The longest
    prefix yields the shortest, most specific tail.  The trailing empty
    prefix in ``_VLM_KEY_PREFIXES`` ensures keys with no known prefix map
    to themselves as their own tail.
    """
    table: dict[str, str] = {}
    for model_key in state_dict:
        for pfx in _VLM_KEY_PREFIXES:
            if model_key.startswith(pfx):
                tail = model_key[len(pfx) :]
                if tail and tail not in table:
                    table[tail] = model_key
                    break
    return table


def _is_moe_vlm(model: torch.nn.Module) -> bool:
    """Detect whether an HF VLM is a Mixture-of-Experts model.

    MoE VLMs (Qwen3-VL-30B-A3B, Qwen3-VL-235B-A22B) need replicated-gate +
    FSDP-fused-expert shard rules that load_vlm_model does NOT yet implement.
    Callers use this to raise NotImplementedError before sharding.

    Detection sources (any one is sufficient):
    - ``model.config.text_config.num_experts`` (if present and non-None)
    - ``model.config.text_config.num_local_experts`` (if present and non-None)
    - Same attributes on ``model.config`` directly (text-only fallback)
    - Any state-dict key containing ``.mlp.experts.``
    """
    text_cfg = getattr(model.config, "text_config", None) or model.config
    for attr in ("num_experts", "num_local_experts"):
        value = getattr(text_cfg, attr, None)
        if value is not None and value != 0:
            return True
    for name in model.state_dict().keys():
        if ".mlp.experts." in name:
            return True
    return False


def _make_name_converter(
    state_dict: dict,
    hf_conv_map: dict[str, str] | None,
) -> Callable[[str], str]:
    """Return a callable that maps checkpoint keys to model keys.

    Two strategies, matching cosmos-rl's flow:
    1. If ``hf_conv_map`` is non-empty (transformers v4 pre-computed pattern
       mapping), apply each pattern/replacement as a regex substitution and
       return on the first match (no further fallback).
    2. Otherwise (transformers v5 or no map), use a direct-match against the
       model's state dict, then a longest-prefix-stripped suffix lookup
       through ``_VLM_KEY_PREFIXES``.  Names that match nothing are returned
       unchanged (the caller is responsible for filtering / raising).
    """
    model_key_by_tail = _build_model_key_by_tail(state_dict)

    def convert(name: str) -> str:
        if hf_conv_map:
            for pattern, replacement in hf_conv_map.items():
                if re.search(pattern, name):
                    return re.sub(pattern, replacement, name)
            return name
        if name in state_dict:
            return name
        for pfx in _VLM_KEY_PREFIXES:
            if name.startswith(pfx):
                tail = name[len(pfx) :]
                if tail and tail in model_key_by_tail:
                    return model_key_by_tail[tail]
        return name

    return convert


class MultiRankCheckpointLoader:
    """Multi-rank loader for model weights stored as safetensors files.

    Files in the checkpoint directory are statically partitioned across the
    ranks of the ``dp_shard`` sub-mesh by ``file_idx % world_size``.  Each
    rank reads its assigned files locally and the per-tensor data is later
    broadcast (via :meth:`broadcast_tensor`) so every rank ends up with the
    full tensor before sharding.

    When constructed with ``dp_shard_mesh=None`` the loader degrades to a
    single-rank fallback: ``world_size = 1``, every rank reads every file,
    and broadcasts are no-ops.

    Renamed from cosmos-rl's ``MultiRankWeightLoader`` and extended to load
    from S3 / GCS via easy_io and to support Cosmos3 VFM models.
    https://github.com/nvidia-cosmos/cosmos-rl/blob/main/cosmos_rl/utils/multi_rank_weight_loader.py
    """

    # Mapping from dtype to integer for broadcasting
    DTYPE_TO_INT = {
        torch.float32: 0,
        torch.float16: 1,
        torch.bfloat16: 2,
        torch.int64: 3,
        torch.int32: 4,
        torch.int8: 5,
        torch.uint8: 6,
        torch.float8_e4m3fn: 7,
        torch.float8_e5m2: 8,
    }
    # Mapping from integer to dtype for broadcasting
    INT_TO_DTYPE = {v: k for k, v in DTYPE_TO_INT.items()}

    def __init__(self, dp_shard_mesh: DeviceMesh | None):
        """Initialize the multi-rank weight loader.

        Args:
            dp_shard_mesh: 1-D ``dp_shard`` mesh, or None if dp_shard is not
                enabled.  Callers should obtain this via
                :func:`_get_dp_shard_mesh` so the ``parallel_dims is None`` and
                ``dp_shard <= 1`` cases collapse to the single-rank fallback.
        """
        if dp_shard_mesh is None:
            self.group = None
            self.rank = 0
            self.world_size = 1
        else:
            self.group = dp_shard_mesh.get_group()
            self.rank = dp_shard_mesh.get_local_rank()
            self.world_size = dp_shard_mesh.size()

    def load_files_parallel(
        self,
        checkpoint_path: str,
        credential_path: str | None,
        loading_device: torch.device,
    ) -> tuple[
        dict[str, torch.Tensor],
        dict[str, tuple[list, int]],
        set[str],
    ]:
        """
        Load safetensors files in parallel across ranks.

        Args:
            checkpoint_path: Path to the model directory. Local paths and S3
                URIs are tried first; if no safetensors are found, explicit
                ``hf://org/model`` Hub URIs and bare ``org/model`` repo IDs
                fall back to Hugging Face.
            credential_path: Path to the credential file for S3/GCS.
            loading_device: Device to load tensors on.

        Returns:
            Tuple of (rank_tensors, rank_tensor_metadata, weights_of_ckpt_names):
            - rank_tensors: Dict mapping tensor names to tensors loaded by this rank.
            - rank_tensor_metadata: Dict mapping tensor names to (shape, dtype_int) tuples.
            - weights_of_ckpt_names: Set of all tensor names found by this rank.
        """
        rank_tensors = {}  # {tensor_name: tensor_data} for this rank
        rank_tensor_metadata = {}  # {tensor_name: (shape, dtype)} for this rank
        weights_of_ckpt_names = set()

        backend_args = _make_backend_args(checkpoint_path, credential_path)

        log.info(f"Loading safetensors files from: {checkpoint_path}", rank0_only=False)
        log.info(f"Credential path: {credential_path}", rank0_only=False)
        list_error: Exception | None = None
        if checkpoint_path.startswith(_HF_URI_PREFIX):
            safetensors_files: list[str] = []
        else:
            try:
                safetensors_files = _list_safetensors_files(checkpoint_path, backend_args)
            except Exception as exc:
                if not _is_hf_checkpoint_candidate(checkpoint_path):
                    raise
                list_error = exc
                safetensors_files = []

        if not safetensors_files:
            if _is_hf_checkpoint_candidate(checkpoint_path):
                original_checkpoint_path = checkpoint_path
                # Multi-node: serialize through global rank 0 so we don't race on the
                # shared HF cache. snapshot_download's per-blob locks are unreliable on
                # NFS/lustre under concurrent access; without this gate the snapshot
                # dir can end up with only config.json and the listing below fails.
                if dist.is_initialized() and dist.get_world_size() > 1:
                    if dist.get_rank() == 0:
                        checkpoint_path = _download_hf_checkpoint(checkpoint_path)
                    dist.barrier()
                    if dist.get_rank() != 0:
                        checkpoint_path = _download_hf_checkpoint(checkpoint_path)
                    dist.barrier()
                else:
                    checkpoint_path = _download_hf_checkpoint(checkpoint_path)
                backend_args = None
                log.info(
                    "No local/S3 safetensors found; falling back to Hugging Face checkpoint "
                    f"{original_checkpoint_path} -> {checkpoint_path}",
                    rank0_only=False,
                )
                safetensors_files = _list_safetensors_files(checkpoint_path, backend_args)
            elif list_error is not None:
                raise list_error

        if not safetensors_files:
            raise FileNotFoundError(f"No .safetensors files found in checkpoint path: {checkpoint_path}")

        for file_idx, file_path in enumerate(safetensors_files):
            file_rank = file_idx % self.world_size
            if self.rank == file_rank:
                log.info(f"Loading safetensors file: {file_path}", rank0_only=False)
                full_path = easy_io.join_path(checkpoint_path, file_path, backend_args=backend_args)
                # Download the file
                weights_data = easy_io.get(full_path, backend_args=backend_args)
                state_dict = load_safetensors(weights_data)
                for name, tensor in state_dict.items():
                    # Names are stored RAW here; per-checkpoint name
                    # conversion (see _make_name_converter / the
                    # convert_weight_from_*_hf functions) is applied later
                    # by the caller after broadcast.
                    weights_of_ckpt_names.add(name)
                    rank_tensors[name] = tensor.to(device=loading_device)
                    rank_tensor_metadata[name] = (
                        list(tensor.shape),
                        self.DTYPE_TO_INT.get(tensor.dtype, 0),
                    )

        return rank_tensors, rank_tensor_metadata, weights_of_ckpt_names

    def gather_tensor_names_and_build_mapping(
        self, weights_of_ckpt_names: set[str], rank_tensors: dict[str, torch.Tensor]
    ) -> tuple[set[str], dict[str, int]]:
        """
        Gather all tensor names from all ranks and build a tensor-to-rank mapping.

        Args:
            weights_of_ckpt_names: Set of tensor names found by this rank.
            rank_tensors: Dict of tensors loaded by this rank.

        Returns:
            Tuple of (all_tensor_names, tensor_to_rank_map):
            - all_tensor_names: Set of all tensor names across all ranks.
            - tensor_to_rank_map: Dict mapping tensor names to the rank that loaded them.
        """
        if self.world_size > 1:
            # all_gather_object requires output list to be pre-initialized with world_size
            all_tensor_names_lists: list[list[str] | None] = [None] * self.world_size
            dist.all_gather_object(all_tensor_names_lists, list(weights_of_ckpt_names), group=self.group)
            # Flatten the list and create a set
            all_tensor_names = set()
            for names_list in all_tensor_names_lists:
                if names_list is not None:
                    all_tensor_names.update(names_list)

            # Build tensor-to-rank mapping: gather which rank has which tensors
            # Create a dict mapping tensor_name -> rank for this rank
            local_tensor_to_rank = {name: self.rank for name in rank_tensors.keys()}
            all_tensor_to_rank_dicts: list[dict[str, int] | None] = [None] * self.world_size
            dist.all_gather_object(all_tensor_to_rank_dicts, local_tensor_to_rank, group=self.group)

            # Merge all dicts into a global ``tensor_name -> rank`` map.
            # Duplicates aren't expected (each tensor lives in exactly one
            # file, which is owned by exactly one rank), but if they do
            # occur the lowest rank wins.
            tensor_to_rank_map = {}
            for rank_idx, tensor_dict in enumerate(all_tensor_to_rank_dicts):
                if tensor_dict is not None:
                    for tensor_name in tensor_dict:
                        if tensor_name not in tensor_to_rank_map:
                            tensor_to_rank_map[tensor_name] = rank_idx
        else:
            all_tensor_names = weights_of_ckpt_names
            tensor_to_rank_map = {name: 0 for name in rank_tensors.keys()}

        return all_tensor_names, tensor_to_rank_map

    def broadcast_tensor(
        self,
        name: str,
        tensor_rank: int,
        rank_tensors: dict[str, torch.Tensor],
        rank_tensor_metadata: dict[str, tuple[list, int]],
        device: torch.device,
    ) -> torch.Tensor:
        """
        Broadcast a tensor from the rank that has it to all ranks.

        Args:
            name: Name of the tensor to broadcast.
            tensor_rank: Rank that has the tensor.
            rank_tensors: Dict of tensors loaded by this rank.
            rank_tensor_metadata: Dict of tensor metadata (shape, dtype) for this rank.
            device: Device to create tensors on.

        Returns:
            The broadcasted tensor (same on all ranks).
        """
        # Get tensor from the rank that has it
        if self.rank == tensor_rank:
            ckpt_tensor = rank_tensors[name]
            tensor_shape, tensor_dtype_int = rank_tensor_metadata[name]
            # Move tensor from CPU to GPU if needed (tensors are loaded to CPU to avoid OOM)
            ckpt_tensor = ckpt_tensor.to(device=device)
        else:
            ckpt_tensor = None
            tensor_shape = []
            tensor_dtype_int = 0

        # Broadcast tensor metadata (shape, dtype) from the rank that has it
        if self.world_size > 1:
            # Ensure all ranks participate in broadcast
            if self.rank == tensor_rank:
                shape_len = len(tensor_shape)
                shape_len_tensor = torch.tensor([shape_len], dtype=torch.long, device=device)
                shape_tensor = torch.tensor(tensor_shape, dtype=torch.long, device=device)
                dtype_int_tensor = torch.tensor([tensor_dtype_int], dtype=torch.long, device=device)
            else:
                shape_len_tensor = torch.zeros(1, dtype=torch.long, device=device)
                shape_tensor = None  # Will be created after knowing shape_len
                dtype_int_tensor = torch.zeros(1, dtype=torch.long, device=device)

            # Broadcast shape length first
            dist.broadcast(shape_len_tensor, group=self.group, group_src=tensor_rank)
            shape_len = shape_len_tensor.item()

            # Create shape_tensor with correct size for all ranks
            if self.rank != tensor_rank:
                shape_tensor = torch.zeros(shape_len, dtype=torch.long, device=device)

            # Broadcast shape values
            dist.broadcast(shape_tensor, group=self.group, group_src=tensor_rank)

            # Broadcast dtype
            dist.broadcast(dtype_int_tensor, group=self.group, group_src=tensor_rank)

            if self.rank != tensor_rank:
                tensor_shape = shape_tensor.cpu().tolist()
                tensor_dtype = self.INT_TO_DTYPE.get(dtype_int_tensor.item(), torch.float32)
                ckpt_tensor = torch.empty(tensor_shape, dtype=tensor_dtype, device=device)

            # Broadcast the actual tensor data
            dist.broadcast(ckpt_tensor, group=self.group, group_src=tensor_rank)

        # Ensure ckpt_tensor is not None
        if ckpt_tensor is None:
            raise ValueError(
                f"Failed to get tensor {name} on rank {self.rank}. "
                f"tensor_rank={tensor_rank}, world_size={self.world_size}, "
                f"group={self.group}"
            )

        return ckpt_tensor

    def iterate_tensors(
        self,
        all_tensor_names: set[str],
        tensor_to_rank_map: dict[str, int],
        rank_tensors: dict[str, torch.Tensor],
        rank_tensor_metadata: dict[str, tuple[list, int]],
        device: torch.device,
    ) -> Iterator[tuple[str, torch.Tensor]]:
        """
        Iterate over all tensors, broadcasting them as needed.

        Args:
            all_tensor_names: Set of all tensor names across all ranks.
            tensor_to_rank_map: Dict mapping tensor names to the rank that loaded them.
            rank_tensors: Dict of tensors loaded by this rank.
            rank_tensor_metadata: Dict of tensor metadata (shape, dtype) for this rank.
            device: Device to create tensors on.

        Yields:
            Tuple of (tensor_name, tensor) for each tensor.
        """
        for name in sorted(all_tensor_names):
            tensor_rank = tensor_to_rank_map.get(name)
            if tensor_rank is None:
                continue

            tensor = self.broadcast_tensor(name, tensor_rank, rank_tensors, rank_tensor_metadata, device)
            yield name, tensor


def convert_weight_from_qwen3_hf(
    tensor: torch.Tensor,
    name: str,
    parallel_dims: ParallelDims | None,
) -> tuple[str | None, torch.Tensor | None]:
    """Map Qwen3 VL / LLM HF weights to the Cosmos3 VFM layout and shard them.

    Steps:

    1. Strip the ``model.language_model.`` prefix (so keys from the VL
       checkpoint variant collapse onto the LLM key namespace).
    2. Classify the resulting ``dest_name`` against two allowlists:

       - **used_patterns** — embeddings, norms, attention/MLP projections,
         fused ``mlp.experts.{gate_up_proj,down_proj}``, and
         ``mlp.gate.weight``.  Matching keys are kept.
       - **discarded_patterns** — currently just ``model.visual.*`` (the
         vision tower is loaded separately).  Matching keys are dropped
         and the function returns ``(None, None)``.

       A key that matches neither raises :class:`ValueError`.
    3. Shard kept tensors along dim 0 on the FSDP ``dp_shard`` mesh via
       :func:`_shard_tensor_on_fsdp_mesh`.  Expert parallelism is **not**
       handled here — fused expert tensors flow through the same FSDP
       sharding as the rest, which is correct only for ``ep == 1``; the
       moe-mesh sharding path will need to be added back when EP support
       lands.

    Args:
        tensor: Raw HF tensor.
        name: HF parameter name (with ``model.`` / ``model.language_model.``
            prefix as it appears in the safetensors checkpoint).
        parallel_dims: Parallel dims; ``None`` skips sharding.

    Returns:
        Tuple ``(dest_name, sharded_tensor)`` in the Cosmos3 layout, or
        ``(None, None)`` if the tensor is intentionally discarded.
    """
    dest_name = name.replace("model.language_model.", "model.")

    used_patterns = [
        r"^lm_head\.weight$",
        r"^model\.embed_tokens\.weight$",
        r"^model\.norm\.weight$",
        r"^model\.layers\.(\d+)\.(input_layernorm|post_attention_layernorm)\.weight$",
        r"^model\.layers\.(\d+)\.self_attn\.(q_norm|k_norm|v_norm)\.weight$",
        r"^model\.layers\.(\d+)\.self_attn\.(q_proj|k_proj|v_proj|o_proj)\.weight$",
        r"^model\.layers\.(\d+)\.mlp\.(gate_proj|up_proj|down_proj)\.weight$",
        r"^model\.layers\.(\d+)\.mlp\.experts\.(gate_up_proj|down_proj)$",
        r"^model\.layers\.(\d+)\.mlp\.gate\.weight$",
    ]

    discarded_patterns = [
        r"^model\.visual\.",
    ]

    def _is_used_pattern(dest_name: str) -> bool:
        for used_pattern in used_patterns:
            if re.search(used_pattern, dest_name) is not None:
                return True

        for discarded_pattern in discarded_patterns:
            if re.search(discarded_pattern, dest_name) is not None:
                return False

        raise ValueError(f"Unexpected weight found in checkpoint: {dest_name}")

    if _is_used_pattern(dest_name):
        return dest_name, _shard_tensor_on_fsdp_mesh(tensor, parallel_dims)

    return None, None


def convert_weight_from_nemotron_vl_hf(
    tensor: torch.Tensor,
    name: str,
    parallel_dims: ParallelDims | None,
) -> tuple[str | None, torch.Tensor | None]:
    """Map Nemotron VLM HF keys (56 hybrid blocks) to Cosmos3 VFM MoT keys (28 paired layers).

    The Nemotron 3 Dense VL checkpoint (NVIDIA-Nemotron-3-Dense-VL-2B-BF16-Alignment)
    uses a hybrid layout with 56 alternating attention and MLP blocks, where:

        - Even-indexed blocks (0, 2, 4, ...) contain attention (``mixer.q/k/v/o_proj``)
        - Odd-indexed blocks  (1, 3, 5, ...) contain MLP      (``mixer.up/down_proj``)
        - Each block has a ``norm.weight`` (pre-attention or post-attention layer norm)

    The MoT model uses a standard layout with 28 paired layers, each containing both
    attention and MLP sub-modules.

    Weight mapping (HF → MoT)::

        model.visual.*, model.projector.*, model.multi_modal_projector.*
            → skipped (vision weights, loaded separately)

        model.lm_head.weight / lm_head.weight → lm_head.weight
        model.language_model.embeddings.weight → model.embed_tokens.weight
        model.language_model.norm_f.weight     → model.norm.weight

        model.language_model.layers.{2i}.norm.weight
            → model.layers.{i}.input_layernorm.weight
        model.language_model.layers.{2i+1}.norm.weight
            → model.layers.{i}.post_attention_layernorm.weight

        model.language_model.layers.{2i}.mixer.{q,k,v,o}_proj.weight
            → model.layers.{i}.self_attn.{q,k,v,o}_proj.weight

        model.language_model.layers.{2i+1}.mixer.{up,down}_proj.weight
            → model.layers.{i}.mlp.{up,down}_proj.weight
    """
    if name.startswith("model.visual.") or name.startswith("model.projector."):
        return None, None
    if name.startswith("model.multi_modal_projector."):
        return None, None

    dest_name: str | None = None
    if name == "lm_head.weight" or name == "model.lm_head.weight":
        dest_name = "lm_head.weight"
    elif name == "model.language_model.embeddings.weight":
        dest_name = "model.embed_tokens.weight"
    elif name == "model.language_model.norm_f.weight":
        dest_name = "model.norm.weight"
    else:
        # Layer norm: even idx → pre-attention (input_layernorm), odd idx → post-attention
        m = re.match(r"model\.language_model\.layers\.(\d+)\.norm\.weight", name)
        if m is not None:
            idx = int(m.group(1))
            paired = idx // 2
            if idx % 2 == 0:
                dest_name = f"model.layers.{paired}.input_layernorm.weight"
            else:
                dest_name = f"model.layers.{paired}.post_attention_layernorm.weight"
        else:
            # Attention projections: must be at even indices
            m = re.match(
                r"model\.language_model\.layers\.(\d+)\.mixer\.(q_proj|k_proj|v_proj|o_proj)\.weight",
                name,
            )
            if m is not None:
                idx = int(m.group(1))
                if idx % 2 != 0:
                    raise ValueError(f"Expected attention block at even layer index, got {name}")
                paired = idx // 2
                dest_name = f"model.layers.{paired}.self_attn.{m.group(2)}.weight"
            else:
                # MLP projections: must be at odd indices
                m = re.match(
                    r"model\.language_model\.layers\.(\d+)\.mixer\.(up_proj|down_proj)\.weight",
                    name,
                )
                if m is not None:
                    idx = int(m.group(1))
                    if idx % 2 != 1:
                        raise ValueError(f"Expected MLP block at odd layer index, got {name}")
                    paired = idx // 2
                    dest_name = f"model.layers.{paired}.mlp.{m.group(2)}.weight"

    if dest_name is None:
        raise ValueError(f"Unexpected Nemotron checkpoint tensor: {name}")

    return dest_name, _shard_tensor_on_fsdp_mesh(tensor, parallel_dims)


def convert_weight_from_nemotron_llm_hf(
    tensor: torch.Tensor,
    name: str,
    parallel_dims: ParallelDims | None,
) -> tuple[str | None, torch.Tensor | None]:
    """Map Nemotron pure-LLM HF keys (CosmosNemotronForCausalLM) to MoT language model keys.

    The Nemotron 3 LLM checkpoint (NVIDIA-Nemotron-3-2B-BF16) uses a standard
    decoder-only layout with 28 layers, each containing attention and MLP. The key
    names are already close to the MoT model's expected layout, so most keys pass
    through with minimal renaming.

    Weight mapping (HF → MoT)::

        model.embeddings.weight → model.embed_tokens.weight
        lm_head.weight          → lm_head.weight
        model.norm.weight       → model.norm.weight

        model.layers.{i}.input_layernorm.weight          → (unchanged)
        model.layers.{i}.post_attention_layernorm.weight  → (unchanged)
        model.layers.{i}.self_attn.{q,k,v,o}_proj.weight → (unchanged)
        model.layers.{i}.mlp.{up,down}_proj.weight        → (unchanged)
    """
    if name == "model.embeddings.weight":
        dest_name = "model.embed_tokens.weight"
    elif name in ("lm_head.weight", "model.lm_head.weight"):
        dest_name = "lm_head.weight"
    elif name == "model.norm.weight":
        dest_name = "model.norm.weight"
    elif re.match(r"model\.layers\.\d+\.(input_layernorm|post_attention_layernorm)\.weight", name):
        dest_name = name
    elif re.match(r"model\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)\.weight", name):
        dest_name = name
    elif re.match(r"model\.layers\.\d+\.mlp\.(up_proj|down_proj)\.weight", name):
        dest_name = name
    else:
        raise ValueError(f"Unexpected Nemotron LLM checkpoint tensor: {name}")

    return dest_name, _shard_tensor_on_fsdp_mesh(tensor, parallel_dims)


def _shard_first_dim(tensor: torch.Tensor, world_size: int, rank: int) -> torch.Tensor:
    """Slice a tensor along dim 0 for FSDP sharding.

    Matches cosmos-rl weight_converter.py:71-79 semantics: even splits use
    tensor_split; uneven splits use ceil-divide with the last rank getting
    the remainder (may be smaller than average).  This layout must match
    FSDP2's local_view shape per rank — caller asserts shape equality.
    """
    tensor = tensor.contiguous()
    row_size = tensor.shape[0]
    if world_size == 1:
        return tensor
    if row_size % world_size == 0:
        return tensor.tensor_split(world_size, dim=0)[rank].contiguous()
    avg = (row_size + world_size - 1) // world_size
    start = rank * avg
    end = min(start + avg, row_size)
    return tensor[start:end].contiguous()


def detect_vlm_checkpoint_format(all_tensor_names: set[str]) -> str:
    """Detect the checkpoint family from its tensor key set.

    Detection rules (first match wins):

    - ``"nemotron_3_dense_vl"`` — any key shaped like
      ``model.language_model.layers.*.mixer.q_proj.*``.  This is the hybrid
      56-block layout where attention and MLP live in alternating blocks
      under ``mixer.``.
    - ``"nemotron_3_llm"`` — checkpoints that expose
      ``model.embeddings.weight`` (Nemotron's pure LLM key for the input
      embedding; Qwen3 uses ``model.embed_tokens.weight``).
    - ``"qwen3"`` — default; covers Qwen3 VL and Qwen3 LLM (dense and MoE).

    The resulting tag is consumed by :func:`load_language_model` to dispatch
    to the matching ``convert_weight_from_*_hf`` converter.
    """
    for n in all_tensor_names:
        if "model.language_model.layers." in n and ".mixer.q_proj" in n:
            return "nemotron_3_dense_vl"
    if "model.embeddings.weight" in all_tensor_names:
        return "nemotron_3_llm"
    return "qwen3"


def load_language_model(
    model: torch.nn.Module,
    checkpoint_path: str,
    credential_path: str | None,
    parallel_dims: ParallelDims | None,
    checkpoint_format: str | None = None,
) -> set[str]:
    """
    Universal language model loading function using SafeTensors (.safetensors) format.
    Handles key remapping for "model.language_model." -> "model." by default.

    Args:
        model: The language model to load weights into.
        checkpoint_path: Path to checkpoint containing .safetensors files.
        credential_path: Path to S3 credentials
        parallel_dims: ParallelDims object to use for parallel loading.
            If None, the loading is done in a single rank.
        checkpoint_format: ``"qwen3"``, ``"nemotron_3_dense_vl"``, ``"nemotron_3_llm"``, or None to auto-detect.

    Returns:
        Set of model state-dict keys successfully loaded from the checkpoint.
    """
    if not INTERNAL:
        from cosmos.utils.checkpoint_db import download_checkpoint, sanitize_uri

        checkpoint_path = download_checkpoint(sanitize_uri(checkpoint_path))

    start_time = time.time()
    log.info(f"load_language_model: loading weights from {checkpoint_path}")

    lm_state_dict = {}
    for name, tensor in model.state_dict().items():
        # Remove the original module (torch compiled module) and checkpoint wrapped module prefixes.
        final_name = name.replace("_orig_mod.", "").replace("_checkpoint_wrapped_module.", "")
        lm_state_dict[final_name] = tensor

    # Initialize multi-rank weight loader
    loader = MultiRankCheckpointLoader(_get_dp_shard_mesh(parallel_dims))

    # Step 1: Load files in parallel
    rank_tensors, rank_tensor_metadata, weights_of_ckpt_names = loader.load_files_parallel(
        checkpoint_path=checkpoint_path,
        credential_path=credential_path,
        loading_device="cpu",
    )

    # Step 2: Gather tensor names and build mapping
    all_tensor_names, tensor_to_rank_map = loader.gather_tensor_names_and_build_mapping(
        weights_of_ckpt_names, rank_tensors
    )

    resolved_format = checkpoint_format or detect_vlm_checkpoint_format(all_tensor_names)
    log.info(f"Language model checkpoint format: {resolved_format}", rank0_only=False)

    # Step 3: Process each tensor
    keys_loaded = set()
    for name, tensor in loader.iterate_tensors(
        all_tensor_names,
        tensor_to_rank_map,
        rank_tensors,
        rank_tensor_metadata,
        device="cuda",
    ):
        if resolved_format == "nemotron_3_dense_vl":
            dest_name, dest_weight = convert_weight_from_nemotron_vl_hf(
                tensor=tensor,
                name=name,
                parallel_dims=parallel_dims,
            )
        elif resolved_format == "nemotron_3_llm":
            dest_name, dest_weight = convert_weight_from_nemotron_llm_hf(
                tensor=tensor,
                name=name,
                parallel_dims=parallel_dims,
            )
        elif resolved_format == "qwen3":
            dest_name, dest_weight = convert_weight_from_qwen3_hf(
                tensor=tensor,
                name=name,
                parallel_dims=parallel_dims,
            )
        else:
            raise ValueError(f"Unexpected checkpoint format: {resolved_format}")

        if dest_name is None:
            # This is due to the visual weights of VLM models.
            continue

        # If the weight is not found in the language model's state dict, then the weight is
        # unexpected. The unexpected weights should be from the visual part of the VLM (already
        # handled by the previous check). All weights in the language part should be used by
        # the Cosmos3 VFM.
        if dest_name not in lm_state_dict:
            raise ValueError(
                f"Unexpected weight found in checkpoint: {name}, "
                f"language model's corresponding weight {dest_name} not found."
            )

        target_tensor = lm_state_dict[dest_name]
        is_dist_tensor = isinstance(target_tensor, DTensor)
        local_view = target_tensor.to_local() if is_dist_tensor else target_tensor

        if dest_weight.device != local_view.device:
            dest_weight = dest_weight.to(local_view.device)

        assert local_view.shape == dest_weight.shape, (
            f"Shape mismatch: {local_view.shape} != {dest_weight.shape} "
            f"for {dest_name} with original shape {target_tensor.shape}"
        )
        with torch.no_grad():
            local_view.data.copy_(dest_weight)

        keys_loaded.add(dest_name)

    # Perform more error checking to ensure the checkpoint is valid. If the keys are missing,
    # then the missing keys should be from the generation pathway. All keys from the
    # understanding pathway must be present in the checkpoint. Additionally, for 2B and 4B
    # dense Qwen VLMs, the `lm_head.weight` key is not present in the checkpoint. For these
    # models, the input embedding and generation layer share the same params due to
    # `tie_word_embeddings` being set to True in the configs. For the 0.6B LLM, 8B and 32B dense
    # VLMs, and the 30B and 235B MoE VLMs, the `lm_head.weight` key is present in the
    # checkpoint.
    keys_missing = set(lm_state_dict.keys()) - keys_loaded
    tie = getattr(model.config, "tie_word_embeddings", False)
    real_keys_missing = {k for k in keys_missing if not ("_moe_gen" in k or (tie and "lm_head.weight" in k))}
    if real_keys_missing:
        raise ValueError(
            f"load_language_model: {len(real_keys_missing)} required model "
            f"parameter(s) not found in checkpoint '{checkpoint_path}'. "
            f"First up to 10: {sorted(real_keys_missing)[:10]}"
        )

    log.info(
        f"load_language_model: successfully loaded {len(keys_loaded)} tensors "
        f"from {checkpoint_path} in {time.time() - start_time:.1f}s"
    )
    return keys_loaded


def load_vlm_model(
    model: torch.nn.Module,
    checkpoint_path: str,
    credential_path: str | None,
    parallel_dims: ParallelDims | None,
    tensor_names_to_skip: list[str] | None = None,
    extra_skip_patterns: list[str] | None = None,
) -> set[str]:
    """Load a HF VLM checkpoint (safetensors) into an FSDP-wrapped HFModel.

    Local paths and S3 URIs are tried first; if no safetensors are found,
    explicit ``hf://org/model`` Hub URIs and bare ``org/model`` repo IDs fall
    back to Hugging Face.

    Both ``tensor_names_to_skip`` and ``extra_skip_patterns`` are lists of
    regex patterns applied to the RESOLVED model key (post-name_converter).
    Phase-5 skips any model key matched by either list; Phase-6's
    completeness check tolerates missing model keys matched by either
    list.  The two kwargs are semantically identical — separate names let
    call sites distinguish "model-type fixed skips" (from
    ``_tensor_names_to_skip_for``) from "overlay-specific skips" (from
    ``VLMModel._init_vlm`` for the pretrain_weights_path_llm overlay).

    Cosmos-rl-style universal loader — no per-family hand-coded key mapping.
    Resolves the FSDP shard sub-group via :func:`_get_dp_shard_mesh`, which
    reads ``parallel_dims.dp_shard_mesh`` (the 1-D ``dp_shard`` sub-mesh
    populated by ``ParallelDims.build_meshes()``).  ``cp`` and ``cfgp`` live
    in their own overlay meshes and do NOT participate in checkpoint sharding.

    Preconditions:
    - ``parallelize()`` has been called on the HFModel (parameters are DTensors).
    - ``HFModel.tie_embeddings()`` has been called before this function so that
      tied ``lm_head.weight`` / ``embed_tokens.weight`` share DTensor storage.
    - When ``parallel_dims`` is provided AND ``parallel_dims.dp_shard > 1``,
      ``parallel_dims.build_meshes()`` MUST have been called by the caller.
      Otherwise ``dp_shard_mesh`` returns None and the loader silently falls
      back to single-rank loading — every rank reads every file and slices
      locally, which is correct for ``dp_shard <= 1`` but a silent perf /
      correctness regression for FSDP runs.  Pass ``parallel_dims=None``
      explicitly for the single-process / unit-test fallback.

    Raises:
        NotImplementedError: for MoE VLMs (not yet supported — see spec §2.2).
        ValueError: when the checkpoint is missing a required model parameter.

    Returns:
        Set of model state-dict keys successfully loaded from the checkpoint.
    """
    start_time = time.time()
    log.info(f"Loading VLM weights in safetensors format from: {checkpoint_path}")

    # Phase 1: canonical model state dict with compile/FSDP wrapper prefixes stripped.
    vlm_state_dict = {
        name.replace("_orig_mod.", "").replace("_checkpoint_wrapped_module.", ""): tensor
        for name, tensor in model.state_dict().items()
    }

    # Phase 2+3: suffix-lookup table + name converter.
    hf_conv_map = getattr(model, "_checkpoint_conversion_mapping", None)
    name_converter = _make_name_converter(
        vlm_state_dict,
        hf_conv_map=hf_conv_map if hf_conv_map else None,
    )

    # Phase 4: MoE precheck — fail early rather than silently mis-shard.
    if _is_moe_vlm(model):
        raise NotImplementedError(
            "load_vlm_model does not yet support MoE VLMs "
            "(e.g. Qwen3-VL-30B-A3B, Qwen3-VL-235B-A22B). Expected follow-up MR "
            "ports cosmos-rl's is_moe_mlp_fused_into_dp_shard / replicated-gate "
            "handling. Use a dense VLM checkpoint (2B, 4B, 8B, 32B) until then."
        )

    # Detect fsdp_offload mode by inspecting a sample parameter's device.  In
    # offload mode, the FSDP-materialized local_views live on CPU; routing the
    # loader's distributed broadcast through CUDA would materialize the full
    # checkpoint tensor on GPU transiently (defeats the point of offload).  Use
    # a single-rank fallback in that case: every rank reads every file on CPU,
    # slices locally, no broadcast.  I/O-redundant but memory-safe, matching
    # the pre-MR _load_vlm_weights behavior under offload.
    sample_target = next(iter(vlm_state_dict.values())) if vlm_state_dict else None
    sample_local = sample_target.to_local() if isinstance(sample_target, DTensor) else sample_target
    offload_mode = sample_local is not None and sample_local.device.type == "cpu"

    # Pick the loader's group: single-rank (no broadcast) in offload mode to
    # keep memory off-GPU; the dp_shard sub-mesh otherwise.
    loader = MultiRankCheckpointLoader(_get_dp_shard_mesh(parallel_dims) if not offload_mode else None)
    rank_tensors, rank_tensor_meta, ckpt_names = loader.load_files_parallel(
        checkpoint_path=checkpoint_path,
        credential_path=credential_path if credential_path else "",
        loading_device="cpu",
    )
    all_tensor_names, tensor_to_rank = loader.gather_tensor_names_and_build_mapping(
        ckpt_names,
        rank_tensors,
    )

    # Phase 5: per-tensor copy.  Skip patterns match the MODEL key (post-
    # name_converter), not the raw ckpt key — this matches cosmos-rl's
    # semantics and avoids fragility with prefix variations.  The two lists
    # are concatenated; they share Phase-5 skip + Phase-6 tolerance
    # semantics.
    _all_skip_patterns = (tensor_names_to_skip or []) + (extra_skip_patterns or [])
    skip_patterns = [re.compile(p) for p in _all_skip_patterns]
    keys_loaded: set[str] = set()
    skipped_model_keys: set[str] = set()

    # Broadcast/iterate device: CUDA (NCCL) unless we're in offload mode, in
    # which case everything stays on CPU.  In offload mode the loader group
    # is single-rank, so iterate_tensors doesn't actually broadcast — device
    # just controls where the tensor is yielded.
    if offload_mode or not torch.cuda.is_available():
        target_device = "cpu"
    else:
        target_device = "cuda"

    # Resolve the shard axis for the FSDP slicing.  Even in offload mode we
    # still need the real (shard_rank, shard_size) from parallel_dims so each
    # rank takes its own FSDP slice — we just route the LOAD/BROADCAST through
    # world_size=1 (single-rank fallback) to avoid the GPU spike.
    dp_shard_mesh = _get_dp_shard_mesh(parallel_dims)
    if dp_shard_mesh is not None:
        shard_rank = dp_shard_mesh.get_local_rank()
        shard_size = dp_shard_mesh.size()
    else:
        shard_rank = 0
        shard_size = 1

    for ckpt_name, tensor in loader.iterate_tensors(
        all_tensor_names,
        tensor_to_rank,
        rank_tensors,
        rank_tensor_meta,
        device=target_device,
    ):
        dest_name = name_converter(ckpt_name)

        if any(p.fullmatch(dest_name) for p in skip_patterns):
            skipped_model_keys.add(dest_name)
            continue

        if dest_name not in vlm_state_dict:
            continue  # extra checkpoint key — ignore

        target = vlm_state_dict[dest_name]
        is_dtensor = isinstance(target, DTensor)
        local_view = target.to_local() if is_dtensor else target

        # Slice using the REAL FSDP shard_rank/shard_size derived from
        # parallel_dims.dp_shard_mesh, NOT loader.rank/world_size.  In
        # offload mode those two differ: the loader runs single-rank
        # (world_size=1) but FSDP still shards across N ranks.
        shard = _shard_first_dim(tensor, shard_size, shard_rank)
        if shard.device != local_view.device:
            shard = shard.to(local_view.device)

        if shard.shape != local_view.shape:
            raise ValueError(
                f"Shape mismatch for {dest_name}: local_view={tuple(local_view.shape)}, shard={tuple(shard.shape)}"
            )
        with torch.no_grad():
            local_view.data.copy_(shard)
        keys_loaded.add(dest_name)

    # Phase 6: completeness check with tied-embedding AND skip-list tolerance.
    missing = set(vlm_state_dict) - keys_loaded - skipped_model_keys

    # Also tolerate missing model keys that match a skip pattern directly —
    # handles the case where the ckpt doesn't contain the key at all, so the
    # Phase 5 loop never saw it and skipped_model_keys didn't accumulate it.
    missing = {k for k in missing if not any(p.fullmatch(k) for p in skip_patterns)}
    tie = getattr(model.config, "tie_word_embeddings", False)
    real_missing = {k for k in missing if not (tie and "lm_head.weight" in k)}
    if real_missing:
        raise ValueError(
            f"load_vlm_model: {len(real_missing)} required model parameter(s) "
            f"not found in checkpoint '{checkpoint_path}'. First up to 10: "
            f"{sorted(real_missing)[:10]}"
        )
    log.info(
        f"load_vlm_model: loaded {len(keys_loaded)} tensors from {checkpoint_path} in {time.time() - start_time:.1f}s"
    )
    return keys_loaded
