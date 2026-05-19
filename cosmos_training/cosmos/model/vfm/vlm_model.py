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
"""VLMModel: config-instantiable ImaginaireModel for VLM training.

Config usage (in vfm/configs/base/vlm/config.py):
    config.model = LazyCall(VLMModel)(
        policy=config.policy, checkpoint=config.checkpoint, train=config.train
    )

Phase 0 — bootstrap via the legacy VLM init path, ParallelDims, and async_safe_ce.
Phase 1 — ParallelDims switches to vfm/utils/parallelism.py.
Phase 2 — legacy init replaced by direct HFModel path (_init_vlm); async_safe_ce
           replaced by vfm/algorithm/loss/cross_entropy.py::cross_entropy_loss.
Phase 3 — init_flash_attn_meta ported to vfm/utils/flash_attn.py;
           config unified under vfm/configs/base/vlm/config.py.
"""

import os
import re
from collections.abc import Callable
from functools import partial

import torch
import torch.nn as nn

from cosmos.utils.lazy_config import instantiate
from cosmos.model._base import ImaginaireModel
from cosmos.utils import log
from cosmos.model.vfm.algorithm.loss.cross_entropy import cross_entropy_loss
from configs.base.defaults.model_config import VLMModelConfig
from cosmos.model.vfm.hf_model import HFModel
from cosmos.model.vfm.parallelize_vlm import parallelize
from cosmos.utils.vfm.parallelism import ParallelDims
from cosmos.utils.vlm.constant import IGNORE_INDEX
from cosmos.utils.vlm.create_position_ids import get_position_ids

# Model-type dispatch sets. Using hf_config.model_type (stable HF-defined string)
# rather than backbone.model_name avoids the brittleness of substring-matching a local
# filesystem path that VLMModel._init_vlm has already rewritten (see _init_vlm: the
# downloader returns a local cache path, so the configured model name is lost).
#
# ``qwen3_vl_moe`` is listed here as forward-compat — MoE dispatch in every family
# helper below is already wired for the 30B-A3B / 235B-A22B variants. End-to-end
# training still fails earlier at load_vlm_model's MoE precheck
# (safetensors_loader.py _is_moe_vlm / NotImplementedError) because sharded MoE
# weight loading is unimplemented; see spec §2.2. Removing ``qwen3_vl_moe`` here
# would regress the family helpers the moment MoE load support lands.
_QWEN_VL_TYPES = {"qwen2_5_vl", "qwen3_vl", "qwen3_vl_moe"}
# InternVL variants register both "internvl" and "internvl_chat" as model_type
# in the upstream InternVL HF policy registry.
_INTERNVL_TYPES = {"internvl", "internvl_chat"}


def _get_overlay_config(model_type: str) -> tuple[list[str], Callable[[str], bool]]:
    """Return ``(skip_patterns, is_lm_key)`` for the backbone.pretrained_weights overlay.

    ``skip_patterns`` are regex patterns for resolved model keys that are expected to
    be absent from the LLM overlay checkpoint (visual encoder + projector); they are
    passed as ``extra_skip_patterns`` to ``load_vlm_model`` so Phase-6 completeness
    check tolerates them. Every OTHER missing model key still raises.

    ``is_lm_key`` is a predicate that decides whether a key returned in ``keys_loaded``
    counts as a "language-model parameter" for VLMModel's post-overlay sanity check.
    Implemented as the inverse of ``skip_patterns`` — a loaded key counts as an LM key
    iff it does NOT match any of the visual/projector skip regexes. This mirrors
    exactly what ``load_vlm_model``'s Phase-5 skip logic does, so the two checks can
    never disagree under HF state-dict layout variations (e.g. ``model.model.*``
    vs. ``model.language_model.*``).

    Family-specific because non-LM params differ across VLM families (projectors may
    live outside ``model.visual.*``). Raises ``NotImplementedError`` for unsupported
    families — safer than silently mis-skipping. Add a new entry when onboarding a
    new VLM family.

    MoE note: ``qwen3_vl_moe`` is accepted here but end-to-end MoE training still
    fails earlier at ``load_vlm_model``'s MoE precheck (see module docstring on
    ``_QWEN_VL_TYPES``).
    """
    if model_type in _QWEN_VL_TYPES:
        # Qwen2.5-VL / Qwen3-VL dense + MoE: the visual encoder AND merger/projector
        # both live under a ``visual.*`` subtree (merger is a submodule of visual —
        # see Qwen3VLForConditionalGeneration / Qwen2_5_VLForConditionalGeneration).
        # Every non-visual resolved key counts as an LM key (language_model layers,
        # norm, embed_tokens, top-level lm_head).
        #
        # The ``(?:model\.)*`` prefix makes both the loader-side Phase-5 skip AND
        # the VLMModel-side LM predicate tolerate three layouts uniformly:
        #
        #   1. Bare  (Qwen2.5-VL official HF class)          — ``visual.merger.*``,
        #      ``model.embed_tokens.*`` / ``lm_head.weight``.  See
        #      projects/cosmos3/vlm/scripts/convert_qwenvl_ckpt.py:101-118 which
        #      inspects ``state_dict()`` for keys starting with ``visual.merger``.
        #   2. One wrapper (Qwen3-VL official HF class)      — ``model.visual.*``,
        #      ``model.language_model.*`` / ``lm_head.weight``.
        #   3. Two+ wrappers (HFModel-shim-wrapped callers)  — ``model.model.visual
        #      .*`` etc., e.g. hf_model_test.py::test_vlm_load_hf_native_keys:644.
        #
        # A narrower regex (e.g. requiring a leading ``model.``) would either
        # reject valid Qwen2.5 visual keys in Phase-6 completeness OR misclassify
        # wrapper-layout visual keys as LM keys in the post-overlay safety check.
        skip_patterns = [r"^(?:model\.)*visual\..*"]
        compiled_skips = [re.compile(p) for p in skip_patterns]
        return (
            skip_patterns,
            lambda k: not any(r.match(k) for r in compiled_skips),
        )
    # Nemotron / InternVL / etc: projectors live outside ``model.visual.*``
    # (e.g. ``model.multi_modal_projector.*``, ``model.projector.*``), and lm_head
    # may be nested (``model.lm_head.weight``). The Qwen-shaped skip list would fail
    # Phase-6 completeness on those families; the Qwen-shaped predicate would misreport
    # a successful overlay as "0 language-model parameters". Fail loudly rather than
    # silently.
    raise NotImplementedError(
        f"VLMModel: backbone.pretrained_weights overlay not yet supported for "
        f"model_type={model_type!r}. Supported types: {sorted(_QWEN_VL_TYPES)}. "
        f"Add a new entry in _get_overlay_config() when onboarding a new VLM family "
        f"(see docs/superpowers/specs/2026-04-20-vlm-pretrain-weights-path-llm-design.md §7)."
    )


def _get_vision_encoder_modules(model: nn.Module, model_type: str) -> list:
    if model_type in _QWEN_VL_TYPES:

        # which returns only [patch_embed, blocks]. Qwen3-VL adds a learnable `pos_embed`
        # (nn.Embedding — see qwen3_vl.py Qwen3VLVisionModel); leaving it trainable while
        # freezing the rest of the vision encoder contradicts the intent of
        # freeze_vision_encoder=True. `hasattr` gate preserves Qwen2.5-VL compatibility
        # (no pos_embed there).
        mods = [model.visual.patch_embed, model.visual.blocks]
        if hasattr(model.visual, "pos_embed"):
            mods.append(model.visual.pos_embed)
        return mods
    elif model_type in _INTERNVL_TYPES:
        return [model.vision_model]
    raise ValueError(f"freeze_vision_encoder not supported for model_type={model_type!r}")


def _get_mm_projector_modules(model: nn.Module, model_type: str) -> list:
    if model_type == "qwen2_5_vl":
        return [model.visual.merger]
    elif model_type in {"qwen3_vl", "qwen3_vl_moe"}:
        mods = [model.visual.merger]
        if hasattr(model.visual, "deepstack_merger_list"):
            mods.append(model.visual.deepstack_merger_list)
        return mods
    elif model_type in _INTERNVL_TYPES:
        # Legacy InternVL helper used `model.model.model.multi_modal_projector`
        # because it operated on a wrapped HFModel (ImaginaireModel -> HFModel ->
        # raw HF InternVL).  We receive the raw HF model directly
        # (hf_model._model), so drop the two wrapper hops.  Best-effort until L1
        # GPU validation on a real InternVL3_5 checkpoint.
        return [model.model.multi_modal_projector]
    raise ValueError(f"freeze_mm_projector not supported for model_type={model_type!r}")


def _get_llm_modules(model: nn.Module, model_type: str) -> list:
    if model_type in _QWEN_VL_TYPES:
        # model.language_model is a @property on Qwen3VLForConditionalGeneration /
        # Qwen2_5_VLForConditionalGeneration that delegates to self.model.language_model
        # — avoids accidentally freezing `visual` which also lives inside self.model.
        # model.lm_head is a top-level submodule on the conditional-generation class.
        return [model.language_model, model.lm_head]
    elif model_type in _INTERNVL_TYPES:
        # Legacy InternVL helper returned `[model.language_model, model.model.lm_head]`
        # for the wrapped HFModel.  Same raw-HF adjustment as mm_projector above:
        # the raw HF InternVL class exposes `.language_model` at the top level but
        # its `lm_head` lives one level deeper under `.model`.  Best-effort until
        # L1 validation.
        return [model.language_model, model.model.lm_head]
    raise ValueError(f"freeze_llm not supported for model_type={model_type!r}")


def _apply_freeze_config(model: nn.Module, model_type: str, cfg) -> int:
    """Apply freeze config in-place. Returns trainable parameter-tensor count.

    ``cfg`` is duck-typed: accepts a ``VLMFreezeConfig`` instance or a
    ``DictConfig`` (LazyCall-backed) where ``__attrs_post_init__`` may not
    have fired. The mutual-exclusivity check below mirrors the attrs
    validator so both paths fail loudly before any parameter is frozen.
    """
    trainable_params = getattr(cfg, "trainable_params", None)
    frozen_params = getattr(cfg, "frozen_params", None)

    if trainable_params is not None and frozen_params is not None:
        raise ValueError("VLMFreezeConfig: set at most one of trainable_params or frozen_params, not both.")

    # Step 1 — legacy named flags via module-probing
    if cfg.freeze_vision_encoder:
        for m in _get_vision_encoder_modules(model, model_type):
            for p in m.parameters():
                p.requires_grad = False

    if cfg.freeze_mm_projector:
        for m in _get_mm_projector_modules(model, model_type):
            for p in m.parameters():
                p.requires_grad = False

    if cfg.freeze_llm:
        for m in _get_llm_modules(model, model_type):
            for p in m.parameters():
                p.requires_grad = False

    # Step 2 — regex override (mutually exclusive; already validated above).
    #
    # `remove_duplicate=False` is required for tied weights. Qwen3 configs set
    # `tie_word_embeddings=True`, so `hf_model.tie_embeddings()` makes
    # `lm_head.weight` and `model.embed_tokens.weight` the same tensor. The default
    # `named_parameters()` dedups by tensor id and keeps only the first traversed
    # name (`model.embed_tokens.weight`); a regex aimed at `lm_head` would silently
    # match nothing and user intent would be lost. Iterating with duplicates
    # preserves both names so either can trigger a match.
    if trainable_params is not None:
        # OR-semantics across tied names: first freeze everything, then unfreeze
        # any tensor whose *any* registered name matches. Cannot write
        # `requires_grad = any(...)` directly because a second visit could flip
        # True back to False on the same shared tensor.
        for p in model.parameters():
            p.requires_grad = False
        for param_name, p in model.named_parameters(remove_duplicate=False):
            if any(re.search(pat, param_name) for pat in trainable_params):
                p.requires_grad = True
    elif frozen_params is not None:
        for param_name, p in model.named_parameters(remove_duplicate=False):
            if any(re.search(pat, param_name) for pat in frozen_params):
                p.requires_grad = False

    n = sum(p.requires_grad for p in model.parameters())
    if not any([cfg.freeze_vision_encoder, cfg.freeze_mm_projector, cfg.freeze_llm, trainable_params, frozen_params]):
        log.warning("freeze config: no freeze mechanism set — all parameters are trainable (full fine-tune)")
    assert n > 0, "freeze config left 0 trainable parameters — check patterns"
    return n


class VLMModel(ImaginaireModel):
    """Config-instantiable ImaginaireModel for VLM training.

    Args:
        config:     VLMModelConfig (policy, train, ema).
        checkpoint: root CheckpointConfig (load_path, load_from_object_store).
    """

    def __init__(self, config: VLMModelConfig, checkpoint):
        super().__init__()
        from cosmos.utils.vfm.flash_attn import init_flash_attn_meta

        self.config = config
        # Expose model.precision so LowPrecisionCallback can read it (mirrors OmniMoTModel).
        self.precision = getattr(torch, config.policy.parallelism.precision)
        init_flash_attn_meta(config.train.deterministic)
        self._init_vlm(config.policy, checkpoint, config.train)

        # Apply freeze before the optimizer is built — ``build_optimizer`` reads
        # ``requires_grad`` off ``named_parameters``.
        n_trainable = _apply_freeze_config(self.model._model, self.hf_config.model_type, self.config.freeze)
        log.info(
            f"freeze config applied (model_type={self.hf_config.model_type}): {n_trainable} trainable parameter tensors"
        )

        dp_group = None
        cp_group = None
        if self.parallel_dims is not None:
            if self.parallel_dims.dp_shard_enabled:
                dp_group = self.parallel_dims.dp_shard_mesh.get_group()
            if self.parallel_dims.cp_enabled:
                cp_group = self.parallel_dims.cp_mesh.get_group()

        self._loss_fn = partial(
            cross_entropy_loss,
            loss_scaling_factor=1.0,
            dp_group=dp_group,
            cp_group=cp_group,
            ignore_index=IGNORE_INDEX,
        )

    def _init_vlm(self, policy, checkpoint, train) -> None:
        """Initialize VLM without the legacy ModelRegistry (Phase 2+).

        Sequence (ordering is critical — do not reorder):
          a. Download HF weights from S3 to local cache.
          b. Meta-init HFModel (params on meta, buffers on CPU via include_buffers=False;
          c. Build ParallelDims + device mesh.
          d. Apply FSDP2 via parallelize() — meta tensors are NOT auto-materialized.
          e. Explicitly materialize meta tensors; move CPU buffers to CUDA.
          f. Tie output embedding → input embedding if tie_word_embeddings=True.
          g. Load pretrain weights into sharded CUDA tensors.
          h. Apply gradient checkpointing if configured.
        """
        from cosmos.utils.vlm.pretrained_models_downloader import (
            maybe_download_hf_model_from_s3,
        )

        load_pretrain_weights = checkpoint.load_path == ""
        log.info(f"checkpoint.load_path: {checkpoint.load_path!r} | load_pretrain_weights: {load_pretrain_weights}")

        # ── a. Download HF model files (config + tokenizer; weights only if no ckpt) ──
        local_path = maybe_download_hf_model_from_s3(
            policy.backbone.model_name,
            checkpoint.load_from_object_store.credentials,
            checkpoint.load_from_object_store.bucket,
            include_model_weights=load_pretrain_weights,
        )
        # local_path is exposed below as self.model_name_or_path; the (frozen) policy
        # config is not mutated.

        # ── b. Meta-init HFModel ──
        hf_model = HFModel(
            model_name_or_path=local_path,
            dtype=train.master_torch_dtype,
            attn_implementation="flash_attention_2",
        )

        # ── b.1. Early family-gate for backbone.pretrained_weights ──
        # Fail-fast on unsupported VLM families BEFORE any expensive work
        # (parallelize, materialize, base-weight load, overlay download).
        # ``hf_config.model_type`` is populated by HFModel's meta-init; no
        # weights touched yet. Empty backbone_path == no overlay, matching
        # the later overlay guard at step g.2.
        if policy.backbone.pretrained_weights.backbone_path:
            _get_overlay_config(hf_model.hf_config.model_type)

        # ── c. Build ParallelDims + device mesh ──
        # Overlay-mesh design (see vfm/utils/parallelism.py): cp/cfgp do NOT
        # consume FSDP rank slots, so dp_replicate * dp_shard == world_size
        # alone. The VLM HFModel doesn't have a CP-aware attention path.
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        _dp_replicate = policy.parallelism.data_parallel_replicate_degree
        # Single-process run: force dp_replicate=1 so ParallelDims doesn't
        # auto-infer it to world_size (which would equal 1 anyway, but guards
        # against environments where WORLD_SIZE is unset/inconsistent).
        if not torch.distributed.is_initialized():
            _dp_replicate = 1

        parallel_dims = ParallelDims(
            world_size=world_size,
            dp_shard=policy.parallelism.data_parallel_shard_degree,
            dp_replicate=_dp_replicate,
            cp=policy.parallelism.context_parallel_shard_degree,
            enable_inference_mode=False,
        )

        # VLM does not currently support cp or cfgp. CP needs a CP-aware
        # attention path (see ``vfm/models/mot/context_parallel_utils.py``) that
        # is not wired into the VLM HFModel; CFGP is inference-only.
        assert parallel_dims.cp == 1, f"VLM does not support CP (got cp={parallel_dims.cp})"
        assert parallel_dims.cfgp == 1, f"VLM does not support CFGP (got cfgp={parallel_dims.cfgp})"

        if torch.distributed.is_initialized():
            parallel_dims.build_meshes(device_type="cuda")

        # Replicate-only (DDP) is not implemented in Phase 2's parallelize().
        # Raise early rather than running with no gradient synchronization and
        # silently producing wrong training results.
        if parallel_dims.dp_replicate_enabled and not parallel_dims.dp_shard_enabled:
            raise NotImplementedError(
                "VLMModel Phase 2 does not support replicate-only DDP "
                "(dp_replicate > 1, dp_shard == 1). "
                "Use dp_shard > 1 for FSDP2. DDP support is planned for Phase 3."
            )

        # ── d. Apply FSDP2 ──
        if torch.distributed.is_initialized():
            parallelize(
                hf_model,
                parallel_dims,
                policy.parallelism.precision,
                train.fsdp_offload,
            )

        # ── e. Materialize meta tensors ──
        # FSDP2 fully_shard does NOT auto-materialize meta tensors. We must
        # explicitly allocate empty tensors so that load_weights() can copy
        # into them via local_view.data.copy_().
        # - Normal (no offload): materialize to CUDA directly.
        # - CPUOffloadPolicy: FSDP2 keeps params on CPU between fwd/bwd and
        #   moves them to GPU during forward. Materialize to CPU here so that
        #   load_weights() can write into a real (non-meta) CPU tensor; FSDP
        #   will move them to GPU on first forward.
        # Reference: vlm/train.py:188-192 (same guard; offload handled by post_to_empty_hook).
        if train.fsdp_offload:
            hf_model._model._apply(
                lambda t: torch.empty_like(t, device="cpu") if t.device.type == "meta" else t,
                recurse=True,
            )
        else:
            hf_model._model._apply(
                lambda t: torch.empty_like(t, device="cuda") if t.device.type == "meta" else t.to("cuda"),
                recurse=True,
            )

        # ── f. Tie embeddings (replaces the legacy post_to_empty_hook) ──
        hf_model.tie_embeddings()

        # ── g. Load pretrain weights ──
        if load_pretrain_weights:
            hf_model.load_weights(
                checkpoint_path=local_path,
                credential_path=None,  # local path after download
                parallel_dims=parallel_dims if torch.distributed.is_initialized() else None,
            )

            # ── g.2. Optional LLM overlay (backbone.pretrained_weights) ──
            # Overlay the language tower with a separate LLM checkpoint.
            # Visual + projector params are preserved from the VLM load above
            # (skipped via extra_skip_patterns).  The existing name converter
            # in load_vlm_model tail-matches raw LLM keys into
            # model.language_model.*, so no temp-dir remap is needed.
            # Mirrors legacy vlm/train.py:221-233 semantics.
            llm_path = policy.backbone.pretrained_weights.backbone_path
            if llm_path:
                overlay_skip_patterns, is_lm_key = _get_overlay_config(hf_model.hf_config.model_type)
                llm_local_path = maybe_download_hf_model_from_s3(
                    llm_path,
                    checkpoint.load_from_object_store.credentials,
                    checkpoint.load_from_object_store.bucket,
                    include_model_weights=True,
                    require_s3_exists=True,
                )
                keys_loaded = hf_model.load_weights(
                    checkpoint_path=llm_local_path,
                    credential_path=None,
                    parallel_dims=parallel_dims if torch.distributed.is_initialized() else None,
                    extra_skip_patterns=overlay_skip_patterns,
                )
                lm_loaded = {k for k in keys_loaded if is_lm_key(k)}
                if not lm_loaded:
                    raise RuntimeError(
                        f"VLMModel overlay: loaded 0 language-model parameters from "
                        f"{llm_path!r} (local path: {llm_local_path!r}). The LLM "
                        "checkpoint did not match any language_model.* key in the "
                        "VLM; check model-family / layer-count compatibility."
                    )
                log.info(f"VLMModel: overlaid {len(lm_loaded)} language-model params from {llm_path}")

        # ── i. Gradient checkpointing ──
        if policy.parallelism.use_activation_checkpointing:
            hf_model.apply_gradient_checkpointing()

        self.model = hf_model
        self.parallel_dims = parallel_dims
        self.model_name_or_path = local_path
        self.hf_config = hf_model.hf_config

    def on_train_start(self, memory_format) -> None:
        """Called by trainer after model.to("cuda"). No device move needed here."""

    def on_after_backward(self, iteration: int = 0) -> None:
        """No-op — FSDP handles gradient synchronization internally."""

    def init_optimizer_scheduler(self, optimizer_config, scheduler_config):
        """Build optimizer + scheduler from hydra-instantiated configs.

        Freeze was applied in ``__init__``; ``build_optimizer`` reads
        ``requires_grad`` off ``named_parameters``.
        """
        optimizer_config.model = self.model
        optimizer = instantiate(optimizer_config)

        scheduler_config.optimizer = optimizer
        scheduler = instantiate(scheduler_config)

        return optimizer, scheduler

    def training_step(self, data: dict, iteration: int) -> tuple[dict, torch.Tensor]:
        """position_ids → forward → CE loss."""
        position_ids = get_position_ids(
            self.hf_config,
            input_ids=data["input_ids"],
            image_grid_thw=data.get("image_grid_thw"),
            video_grid_thw=data.get("video_grid_thw"),
            attention_mask=data.get("attention_mask"),
        )
        if position_ids is not None:
            data["position_ids"] = position_ids

        labels = data.pop("labels")
        data.pop("attention_mask", None)
        logits = self.model(**data)
        loss = self._loss_fn(logits, labels)

        # loss_avg: DP-averaged loss for logging (matches cosmos-rl ReduceOp.AVG).
        # Does not affect the backward scalar. Pick the same 1-D sub-mesh the
        # legacy single-mesh ``ParallelDims.dp_mesh`` returned — dp_shard if
        # sharding is on, else dp_replicate — so the reduction group is
        # byte-identical to pre-merge behavior.
        loss_avg = loss.detach().clone()
        pd = getattr(self, "parallel_dims", None)
        dp_mesh = pd.dp_mesh if pd is not None else None
        if torch.distributed.is_initialized() and dp_mesh is not None:
            sub_dim = "dp_shard" if pd.dp_shard_enabled else "dp_replicate"
            torch.distributed.all_reduce(
                loss_avg, op=torch.distributed.ReduceOp.AVG, group=dp_mesh[sub_dim].get_group()
            )
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            log.info(f"train/loss_avg: {loss_avg.item():.5f} (iteration {iteration})")

        return {"loss": loss, "loss_avg": loss_avg, "labels": labels}, loss

    @torch.no_grad()
    def validation_step(self, data: dict, iteration: int) -> tuple[dict, torch.Tensor]:
        """Required: VLM experiments enable validation by default (pre_exp01x.py:607).
        ImaginaireTrainer.validate() calls this — must not raise NotImplementedError."""
        position_ids = get_position_ids(
            self.hf_config,
            input_ids=data["input_ids"],
            image_grid_thw=data.get("image_grid_thw"),
            video_grid_thw=data.get("video_grid_thw"),
            attention_mask=data.get("attention_mask"),
        )
        if position_ids is not None:
            data["position_ids"] = position_ids

        labels = data.pop("labels")
        data.pop("attention_mask", None)
        logits = self.model(**data)
        loss = self._loss_fn(logits, labels)
        return {"loss": loss, "labels": labels}, loss
