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
"""Minimal HFModel for the vfm/ unified VLM training path.

Responsibilities:
  - ``__init__``: meta-init the underlying HF model via the appropriate
    AutoClass (``AutoModelForImageTextToText`` / ``AutoModel`` /
    ``AutoModelForCausalLM`` — see ``HFModel`` for selection rules);
    no weights are loaded.
  - ``apply_gradient_checkpointing``: wraps HF's standard
    ``gradient_checkpointing_enable`` API.
  - ``tie_embeddings``: re-establishes the input/output embedding tie after
    FSDP wrapping + meta-materialization.
  - ``load_weights``: dispatches to ``load_vlm_model`` (VLM) or
    ``load_language_model`` (LLM) from ``safetensors_loader.py`` based on
    ``vision_config``; returns the set of checkpoint keys that were loaded.
  - ``forward``: pass-through returning logits.

FSDP wrapping lives in ``vfm/models/parallelize_vlm.py::parallelize()``,
NOT here.
"""

import torch
import torch.nn as nn
from accelerate import init_on_device
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoModelForImageTextToText

from cosmos3._src.imaginaire.utils import log
from cosmos3._src.vfm.models.utils.safetensors_loader import load_language_model, load_vlm_model
from cosmos3._src.vfm.utils.parallelism import ParallelDims


def _tensor_names_to_skip_for(model_type: str) -> list[str]:
    """Per-model-type tensor-name regex skip list for load_vlm_model.

    Mirrors the upstream HF-model ``tensor_names_to_skip`` property from the
    legacy VLM policy registry.  Patterns match the **resolved model key**
    (post-name_converter).  Patterns are used twice inside load_vlm_model:
    (1) the per-tensor loop skips copying matched model keys, (2) the
    completeness check tolerates matched model keys that are absent from
    the checkpoint.

    Registered VLMs (see
    cosmos3/_src/vfm/configs/base/vlm/defaults/vlm_policy.py):
    - Qwen3-VL dense (2B/4B/8B/32B): no skips needed.
    - NemotronH_Nano_VL_V2 (nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16):
      RADIO backbone buffers — initialized by the module, not from ckpt.
    """
    table: dict[str, list[str]] = {
        "NemotronH_Nano_VL_V2": [
            r"vision_model\.radio_model\.summary_idxs",
            r"vision_model\.radio_model\.input_conditioner\.norm_mean",
            r"vision_model\.radio_model\.input_conditioner\.norm_std",
        ],
    }
    return table.get(model_type, [])


class HFModel(nn.Module):
    """Minimal HF model wrapper for the vfm/ unified VLM training path.

    Loads any HF causal LM or VL model on the meta device (no GPU memory)
    via the appropriate AutoClass — see selection rules below. Weights are NOT
    loaded in ``__init__``. Call :meth:`load_weights` after FSDP wrapping +
    explicit meta-tensor materialization so each rank only fills its own shard.

    AutoClass selection (by vision_config presence + ``auto_map``):
    - VLM with standard transformers registration (e.g. Qwen3-VL)
      → ``AutoModelForImageTextToText``.  Returns the full conditional-generation
      class (e.g. ``Qwen3VLForConditionalGeneration``), which exposes ``.logits``
      on forward output.  ``AutoModelForCausalLM`` raises ``ValueError`` for VLM
      configs (``Qwen3VLConfig`` is not registered for that auto class), so it
      cannot be used here.
    - VLM with custom ``auto_map`` (e.g. NemotronVL): the registered entry maps
      the full causal-LM class through ``AutoModel`` rather than
      ``AutoModelForImageTextToText`` — use ``AutoModel`` for this case only.
    - LLM (no ``vision_config``) → ``AutoModelForCausalLM``.  Standard causal LM
      with ``.logits``.

    Do NOT use ``AutoModel`` for the standard VLM path — it returns the backbone
    only (e.g. ``Qwen3VLModel``), which does NOT have ``.logits``.

    FSDP / TP wrapping is applied externally by ``parallelize()`` in
    ``vfm/models/parallelize_vlm.py``.
    """

    def __init__(
        self,
        model_name_or_path: str,
        dtype: torch.dtype = torch.bfloat16,
        attn_implementation: str = "flash_attention_2",
        trust_remote_code: bool = True,
    ):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        hf_config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)
        self.hf_config = hf_config

        # AutoClass selection by model type:
        # - Standard VLM (Qwen3-VL, etc.): AutoModelForImageTextToText returns the full causal
        #   LM with .logits (Qwen3VLForConditionalGeneration, etc.).
        # - Custom VLM with auto_map (e.g. NemotronVL): AutoModelForImageTextToText is not
        #   registered; use AutoModel instead which maps to the full causal LM via auto_map.
        # - LLM (no vision_config): AutoModelForCausalLM → standard causal LM with .logits.
        is_vlm = getattr(hf_config, "vision_config", None) is not None
        auto_map = getattr(hf_config, "auto_map", None) or {}
        if is_vlm:
            if "AutoModelForImageTextToText" in auto_map or not auto_map:
                # Standard VLM or no auto_map (rely on registered transformers type)
                model_cls = AutoModelForImageTextToText
            else:
                # Custom VLM: use AutoModel which maps to the full causal-LM class via auto_map
                model_cls = AutoModel
        else:
            model_cls = AutoModelForCausalLM

        # Meta init: allocates no GPU memory. FSDP2's ``fully_shard`` does NOT
        # auto-materialize meta tensors; the caller (see ``vlm_model._init_vlm``)
        # must explicitly materialize via ``_apply(empty_like, ...)`` between
        # ``parallelize()`` and ``load_weights()``.
        with init_on_device("meta", include_buffers=False):
            self._model = model_cls.from_config(
                hf_config,
                attn_implementation=attn_implementation,
                torch_dtype=dtype,
                trust_remote_code=trust_remote_code,
            )
        log.info(f"HFModel: {hf_config.model_type} ({'VLM' if is_vlm else 'LLM'}), dtype={dtype}")

        # Normalize floating-point *parameter* dtypes to ``dtype``. HF's
        # ``from_config`` installs ``torch.set_default_dtype(dtype)`` around the
        # init, but some HF submodules (and vendored remote-code classes) read
        # ``config.torch_dtype`` directly or build tensors with an explicit
        # ``dtype=`` kwarg, so their params can end up in the checkpoint's dtype
        # (typically bf16) while the rest of the model is in ``dtype``. FSDP2's
        # ``_init_mp_dtypes`` then asserts "uniform original parameter dtype …
        # {bf16, fp32}". Normalize on meta (no GPU memory) so all FSDP units see
        # a single original dtype. Buffers are left alone — ``inv_freq`` etc.
        # must stay fp32 (enforced by e.g. qwen3_vl.py's inv_freq assertion).
        n_cast = 0
        with torch.no_grad():
            for p in self._model.parameters(recurse=True):
                if p.is_floating_point() and p.dtype != dtype:
                    p.data = p.data.to(dtype)
                    n_cast += 1
        if n_cast:
            log.info(f"HFModel: normalized {n_cast} param(s) to {dtype} post-from_config")

        # Patch Qwen3-VL forward for text-only batches (no pixel_values / image_grid_thw).
        # Required to avoid errors when a batch contains only text: every FSDP rank must
        # call visual() each step for all-gather sync; the patch runs a lightweight dummy
        # image and slices the output to [0:0] so it contributes no features.
        # Must happen BEFORE parallelize() so FSDP captures the patched forward.
        if hf_config.model_type == "qwen3_vl" and hasattr(self._model, "model"):
            from cosmos3._src.vfm.utils.monkey_patch import patch_qwen3_vl_forward

            patch_qwen3_vl_forward(self._model.model)
            log.info("HFModel: applied patch_qwen3_vl_forward for text-only batch support")

    def apply_gradient_checkpointing(self) -> None:
        """Enable gradient checkpointing via HF's standard API."""
        self._model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        log.info("HFModel: gradient checkpointing enabled")

    def tie_embeddings(self) -> None:
        """Tie output embedding weight to input embedding, matching post_to_empty_hook behavior.

        Must be called AFTER ``parallelize()`` and AFTER the explicit
        meta-tensor materialization step (FSDP2 does not auto-materialize —
        see ``vlm_model._init_vlm`` step e), and BEFORE ``load_weights()`` so
        the tied pointer survives weight loading.

        Two strategies, matching the HF API split:
        1. ``get_output_embeddings()`` path — standard for most HF models.
        2. ``_tied_weights_keys`` fallback — some VLMs (notably
           ``Qwen3VLForConditionalGeneration``) define ``lm_head`` and
           ``_tied_weights_keys = ["lm_head.weight"]`` but do NOT override
           ``get_output_embeddings``.  For those, walk the dotted key to the
           owning module and assign its ``.weight`` directly.  See spec §8.3.

        Reference: the legacy VLM HF-model tie_embeddings implementation.
        """
        if not getattr(self.hf_config, "tie_word_embeddings", False):
            return
        input_embeddings = self._model.get_input_embeddings()
        if input_embeddings is None:
            return
        output_embeddings = self._model.get_output_embeddings()
        if output_embeddings is not None:
            output_embeddings.weight = input_embeddings.weight
            log.info("HFModel: tied input/output embeddings via get_output_embeddings")
            return
        # Fallback: HF models that use _tied_weights_keys instead of
        # overriding get_output_embeddings (e.g. Qwen3VLForConditionalGeneration
        # defines _tied_weights_keys = ["lm_head.weight"] but returns None
        # from the default get_output_embeddings).  Walk the dotted key to
        # the owning module and assign the Parameter directly.
        tied_keys = getattr(self._model, "_tied_weights_keys", None) or ()
        if not tied_keys:
            return
        for key in tied_keys:
            parts = key.split(".")
            *mod_path, attr = parts
            target = self._model
            for name in mod_path:
                target = getattr(target, name, None)
                if target is None:
                    log.warning(
                        f"HFModel.tie_embeddings: could not resolve path {key!r} on "
                        f"{type(self._model).__name__}; skipping tie (weights will "
                        f"remain untied for this key)."
                    )
                    break
            else:
                setattr(target, attr, input_embeddings.weight)
                log.info(f"HFModel: tied {key} via _tied_weights_keys fallback")

    def load_weights(
        self,
        checkpoint_path: str,
        credential_path: str | None = None,
        parallel_dims: ParallelDims | None = None,
        extra_skip_patterns: list[str] | None = None,
    ) -> set[str]:
        r"""Load weights from a HF model directory (safetensors format).

        Dispatches on model type:
        - VLM (vision_config present): ``load_vlm_model`` (universal
          suffix-lookup loader inherited from the legacy VLM path; MoE VLMs
          explicitly blocked — see spec §2.2).
        - LLM (no vision_config): ``load_language_model`` — handles VFM-specific
          per-family key remapping for Qwen3 / Nemotron (unchanged from today).

        Must be called AFTER ``parallelize()`` so parameters are DTensors with
        CUDA local views.  For tied-embedding models, ``tie_embeddings()`` must
        be called between ``parallelize()`` and this function.

        Args:
            checkpoint_path: Path to a directory containing .safetensors files.
                Local paths and S3 URIs are tried first; if no safetensors are
                found, explicit ``hf://org/model`` Hub URIs and bare
                ``org/model`` repo IDs fall back to Hugging Face.
            credential_path: S3 credential file, or None for local/HF.
            parallel_dims: ``ParallelDims`` instance (from
                ``projects.cosmos3.vfm.utils.parallelism``).  The loader uses
                it via :func:`~projects.cosmos3.vfm.models.utils.safetensors_loader._get_dp_shard_mesh`
                to obtain the 1-D ``dp_shard`` sub-mesh (or None when
                ``dp_shard <= 1``) for striping checkpoint reads across
                FSDP shard ranks.  When non-None, the caller MUST have
                called ``parallel_dims.build_meshes()`` first — neither
                this method nor ``load_vlm_model`` re-checks this.  Pass
                ``parallel_dims=None`` for the single-rank fallback used
                by single-process / non-distributed runs.
            extra_skip_patterns: Optional list of regex patterns appended to
                ``tensor_names_to_skip`` inside ``load_vlm_model``.  Use when
                overlaying an LLM-only checkpoint onto a VLM model (e.g. swapping
                the language tower while preserving visual + projector params)
                — pass patterns like ``r"model\.visual\."`` so those keys are
                skipped during the overlay.  Only takes effect on the VLM
                dispatch path; ignored when the model is a pure LLM (no
                ``vision_config``).

        Returns:
            Set of model state-dict keys that were loaded from the checkpoint.
        """
        is_vlm = getattr(self.hf_config, "vision_config", None) is not None
        if is_vlm:
            keys_loaded = load_vlm_model(
                model=self._model,
                checkpoint_path=checkpoint_path,
                credential_path=credential_path,
                parallel_dims=parallel_dims,
                tensor_names_to_skip=_tensor_names_to_skip_for(self.hf_config.model_type),
                extra_skip_patterns=extra_skip_patterns,
            )
        else:
            keys_loaded = load_language_model(
                model=self._model,
                checkpoint_path=checkpoint_path,
                credential_path=credential_path if credential_path else "",
                parallel_dims=parallel_dims,
            )
        log.info(f"HFModel: weights loaded from {checkpoint_path} ({len(keys_loaded)} keys)")
        return keys_loaded

    # Keys added by the VLM collate_fn (vlm/datasets/collate_fn.py) that are NOT valid
    # HF model forward arguments. These must be stripped before calling self._model.forward().
    # A blocklist (not a whitelist) is used so that legitimate kwargs passed via the model's
    # **kwargs — e.g. second_per_grid_ts for Qwen3-VL temporal encoding, output_router_logits
    # for MoE load-balancing — are forwarded correctly even when not named in the signature.
    _COLLATE_NON_MODEL_KEYS: frozenset[str] = frozenset(
        {
            "token_mask",
            "pad_token_id",
            "ignore_index",
            "collated",
            "raw_image",
            "raw_video",
            # image_sizes is collected by collate_fn but is NOT a Qwen3-VL forward arg
            # (Qwen3-VL uses image_grid_thw instead). Strip it so strict HF signatures
            # don't reject it.
            # a future Phase extends to those, remove this entry.
            "image_sizes",
        }
    )

    def forward(self, **kwargs) -> torch.Tensor:
        """Pass-through forward. Returns logits (B, T, V).

        Strips collate-added non-model keys (see ``_COLLATE_NON_MODEL_KEYS``:
        token_mask, pad_token_id, ignore_index, collated, raw_image, raw_video,
        image_sizes) before forwarding. Forces use_cache=False for training.
        All remaining keys (including ``**kwargs`` pass-throughs such as
        second_per_grid_ts) are forwarded unchanged.

        For nemotron_vl: attention_mask is also dropped. NemotronVLModel.get_rope_index
        strips padding positions when attention_mask is present, returning position_ids
        shorter than inputs_embeds (padded_len). With right-padding + causal attention,
        valid tokens never attend to padding tokens regardless, so dropping attention_mask
        is equivalent and avoids the shape mismatch.
        """
        filtered = {k: v for k, v in kwargs.items() if k not in self._COLLATE_NON_MODEL_KEYS}
        if self.hf_config.model_type == "nemotron_vl":
            filtered.pop("attention_mask", None)
        filtered["use_cache"] = False
        out = self._model(**filtered)
        return out.logits
