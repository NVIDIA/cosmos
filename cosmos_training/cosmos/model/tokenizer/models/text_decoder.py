# -----------------------------------------------------------------------------
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# -----------------------------------------------------------------------------

"""Text decoder modules for image-to-text generation in the tokenizer.

Provides:
    - ImagePositionEmbeddings: Learnable 2D spatial position embeddings
    - SpatialPatchMerger: 2x2 spatial-to-channel merger for image token reduction
    - TextDecoderWrapper: causal LM wrapper with vision token injection

Architecture flow (mid-training / ITG path):
    x_no_proj [N, encoder_dim]
        -> SpatialPatchMerger (2x2 concat + MLP) -> [N/4, decoder_hidden]
        -> ImagePositionEmbeddings (add 2D spatial pos) -> [N/4, decoder_hidden]
        -> inject into text token embeddings at image_patch_indices
        -> causal LM -> logits [B, S, vocab_size]
        -> CrossEntropy ITG loss (ignore_index=-100 masks vision/padding tokens)

Label alignment (next-token prediction):
    input_ids and labels have the same length (no pre-shift).
    Vision token targets are -100, so loss only backprops through caption tokens.
    The shift (logits[:-1] vs labels[1:]) happens in the loss function.
"""

import os
import re
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger as logging

from cosmos.model.tokenizer.utils.hf import (
    load_auto_tokenizer_from_cache,
    prepare_nemotron_tokenizer_snapshot,
    resolve_hf_snapshot_path,
)

QWEN3_PAD_TOKEN_ID = 151643
QWEN3_IM_START_TOKEN_ID = 151644
QWEN3_IM_END_TOKEN_ID = 151645
QWEN3_VISION_START_TOKEN_ID = 151652
QWEN3_VISION_END_TOKEN_ID = 151653
QWEN3_VISION_PAD_TOKEN_ID = 151655
QWEN3_THINK_START_TOKEN_ID = 151667
QWEN3_THINK_END_TOKEN_ID = 151668
NEMOTRON_2B_PAD_TOKEN_ID = 0
NEMOTRON_2B_EOS_TOKEN_ID = 11
NEMOTRON_2B_VISION_START_TOKEN_ID = 20
NEMOTRON_2B_VISION_END_TOKEN_ID = 21
NEMOTRON_2B_VISION_PAD_TOKEN_ID = 22
NEMOTRON_2B_IM_START_TOKEN = "<|im_start|>"
NEMOTRON_2B_IM_END_TOKEN = "<|im_end|>"
NEMOTRON_2B_THINK_START_TOKEN = "<think>"
NEMOTRON_2B_THINK_END_TOKEN = "</think>"
TEXT_DECODER_ATTN_IMPLEMENTATION_ENV = "TOKENIZER_TEXT_DECODER_ATTN_IMPLEMENTATION"


@dataclass(frozen=True)
class TextDecoderFamilySpec:
    """Static configuration for one supported text decoder family."""

    family: str
    default_model_name: str
    pad_token_id: int
    eos_token_ids: tuple[int, ...]
    vision_start_token_id: int
    vision_end_token_id: int
    vision_pad_token_id: int
    suppress_token_ids: tuple[int, ...] = ()
    trust_remote_code: bool = True
    supports_inputs_embeds_forward: bool = True
    supports_inputs_embeds_generate: bool = True
    supports_cache_position: bool = True


QWEN3_SPEC = TextDecoderFamilySpec(
    family="qwen3",
    default_model_name="Qwen/Qwen3-0.6B",
    pad_token_id=QWEN3_PAD_TOKEN_ID,
    eos_token_ids=(QWEN3_IM_END_TOKEN_ID,),
    vision_start_token_id=QWEN3_VISION_START_TOKEN_ID,
    vision_end_token_id=QWEN3_VISION_END_TOKEN_ID,
    vision_pad_token_id=QWEN3_VISION_PAD_TOKEN_ID,
    suppress_token_ids=(QWEN3_THINK_START_TOKEN_ID,),
)

NEMOTRON_2B_SPEC = TextDecoderFamilySpec(
    family="nemotron_2b",
    default_model_name="nvidia/NVIDIA-Nemotron-3-2B-BF16",
    pad_token_id=NEMOTRON_2B_PAD_TOKEN_ID,
    eos_token_ids=(NEMOTRON_2B_EOS_TOKEN_ID,),
    vision_start_token_id=NEMOTRON_2B_VISION_START_TOKEN_ID,
    vision_end_token_id=NEMOTRON_2B_VISION_END_TOKEN_ID,
    vision_pad_token_id=NEMOTRON_2B_VISION_PAD_TOKEN_ID,
    trust_remote_code=True,
    supports_inputs_embeds_forward=False,
    supports_inputs_embeds_generate=False,
    supports_cache_position=False,
)

TEXT_DECODER_FAMILY_SPECS: dict[str, TextDecoderFamilySpec] = {
    QWEN3_SPEC.family: QWEN3_SPEC,
    NEMOTRON_2B_SPEC.family: NEMOTRON_2B_SPEC,
}


def _resolve_attn_implementation(attn_implementation: str) -> str:
    """Resolve the attention backend, allowing an env override for eval flows."""
    env_override = os.environ.get(TEXT_DECODER_ATTN_IMPLEMENTATION_ENV)
    if env_override:
        return env_override
    return attn_implementation


def _is_flash_attention_error(exc: Exception) -> bool:
    """Return whether one model-load failure is caused by FlashAttention selection."""
    error_text = str(exc).lower()
    return "flash attention" in error_text or "flash_attention" in error_text


def _repair_nemotron_rotary_buffers(module: nn.Module) -> int:
    """Reset Nemotron RoPE inv_freq buffers after low-memory HF loading.

    Nemotron's remote-code rotary module registers ``inv_freq`` as a
    non-persistent buffer. With ``low_cpu_mem_usage=True`` those buffers are not
    restored from the checkpoint and can contain uninitialized values.
    """
    repaired_count = 0
    for child_module in module.modules():
        if (
            not hasattr(child_module, "inv_freq")
            or not hasattr(child_module, "dim")
            or not hasattr(child_module, "base")
        ):
            continue
        old_inv_freq = child_module.inv_freq
        if isinstance(old_inv_freq, torch.Tensor) and old_inv_freq.device.type != "meta":
            device = old_inv_freq.device
        else:
            device = torch.device("cpu")
        dim = int(child_module.dim)
        base = float(child_module.base)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim))  # [D/2]
        child_module.register_buffer("inv_freq", inv_freq, persistent=False)
        repaired_count += 1
    return repaired_count


def _keep_only_first_image_placeholder(
    text: str,
    image_placeholder: str = "<image>",
) -> str:
    """Keep only the first image placeholder occurrence in a prompt string."""
    first_index = text.find(image_placeholder)
    if first_index == -1:
        return text

    before_first = text[:first_index]
    after_first = text[first_index + len(image_placeholder) :]
    escaped_placeholder = re.escape(image_placeholder)
    after_first_cleaned = re.sub(escaped_placeholder + r"\s*", "", after_first)
    return before_first + image_placeholder + after_first_cleaned


def infer_text_decoder_family(model_name: str) -> str:
    """Infer the decoder family from a HuggingFace model name."""
    model_name_lower = model_name.lower()
    if "nemotron" in model_name_lower:
        return NEMOTRON_2B_SPEC.family
    return QWEN3_SPEC.family


def _get_required_token_id(tokenizer: Any, token: str) -> int:
    """Resolve one tokenizer token name into an ID and fail fast if missing."""
    token_id = tokenizer.convert_tokens_to_ids(token)
    if token_id is None:
        raise ValueError(f"Tokenizer does not define required token {token!r}.")
    unk_token_id = getattr(tokenizer, "unk_token_id", None)
    if unk_token_id is not None and int(token_id) == int(unk_token_id):
        token_string = getattr(tokenizer, "unk_token", None)
        raise ValueError(f"Tokenizer maps required token {token!r} to unknown token {token_string!r}.")
    return int(token_id)


def get_text_decoder_family_spec(
    family: str | None = None,
    model_name: str | None = None,
) -> TextDecoderFamilySpec:
    """Resolve one supported text decoder family specification."""
    resolved_family = family or infer_text_decoder_family(model_name or QWEN3_SPEC.default_model_name)
    if resolved_family not in TEXT_DECODER_FAMILY_SPECS:
        raise ValueError(f"Unsupported text decoder family: {resolved_family}")
    return TEXT_DECODER_FAMILY_SPECS[resolved_family]


class ImagePositionEmbeddings(nn.Module):
    """Learnable position embeddings added per coordinate dimension.

    For the text decoder, we use coord_dim=2 for (H, W) after spatial merging.
    Each dimension has its own embedding table; embeddings are summed and added to features.

    Args:
        hidden_size: Feature dimension (must match input features).
        max_position: Maximum coordinate value in any dimension.
        coord_dim: Number of coordinate dimensions (default 2 for H, W).
    """

    def __init__(self, hidden_size: int, max_position: int = 128, coord_dim: int = 2):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_position = max_position
        self.coord_dim = coord_dim

        self.position_embeddings = nn.ModuleList([nn.Embedding(max_position, hidden_size) for _ in range(coord_dim)])
        for emb in self.position_embeddings:
            nn.init.normal_(emb.weight, mean=0.0, std=0.02)

    def forward(self, features: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """Add position embeddings to features.

        Args:
            features: [N, hidden_size] feature tensor.
            coords: [N, coord_dim] integer coordinates, each in [0, max_position).

        Returns:
            [N, hidden_size] features with position embeddings added.
        """
        pos_emb = torch.zeros_like(features)
        for dim in range(self.coord_dim):
            positions = coords[:, dim].long().clamp(0, self.max_position - 1)
            pos_emb = pos_emb + self.position_embeddings[dim](positions)
        return features + pos_emb


class SpatialPatchMerger(nn.Module):
    """Qwen3VL-style spatial patch merger for reducing visual tokens before LLM.

    Merges spatial patches by:
    1. Grouping tokens into merge_size x merge_size windows by (H, W) coordinates
    2. Concatenating features within each window (preserves all information)
    3. Projecting through a 2-layer MLP to the LLM hidden size

    Reduces token count by merge_size^2 (e.g., 4x for 2x2 merging).

    Design notes vs reference qwen_text_decoder.py:
    - Vectorized scatter instead of Python for-loop for placing features into windows.
      The reference iterates per-token which is O(N) Python ops; we use advanced indexing.
    - Properly segments by (batch/segment, T) to support video frames.

    Args:
        input_hidden_size: Encoder output dim (e.g., 1152 for SigLIP2 SO400M).
        out_hidden_size: LLM hidden dim (e.g., 896 for Qwen3-0.6B, read from model config).
        spatial_merge_size: Window size (default 2 for 2x2 merging).
    """

    def __init__(
        self,
        input_hidden_size: int = 1152,
        out_hidden_size: int = 896,
        spatial_merge_size: int = 2,
    ):
        super().__init__()
        self.input_hidden_size = input_hidden_size
        self.out_hidden_size = out_hidden_size
        self.spatial_merge_size = spatial_merge_size
        self.merged_hidden_size = input_hidden_size * (spatial_merge_size**2)

        # Pre-merge LayerNorm on input features
        self.norm = nn.LayerNorm(input_hidden_size, eps=1e-6)
        # 2-layer MLP: concat_dim -> concat_dim (GELU) -> out_hidden_size
        self.linear_fc1 = nn.Linear(self.merged_hidden_size, self.merged_hidden_size)
        self.act_fn = nn.GELU()
        self.linear_fc2 = nn.Linear(self.merged_hidden_size, out_hidden_size)

    def forward(
        self,
        feats: torch.Tensor,
        coords: torch.Tensor,
        layout: list[slice] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, list[slice] | None]:
        """Merge spatial patches and project to LLM hidden size.

        Args:
            feats: [N, input_hidden_size] token features.
            coords: [N, 4] as (T, H, W, Z) or [N, 5] as (batch/seg, T, H, W, Z).
            layout: Optional batch layout slices from SparseTensor.

        Returns:
            merged_feats: [N', out_hidden_size] where N' ~ N / merge_size^2.
            merged_coords: [N', coord_dims] with scaled H, W.
            new_layout: Updated layout slices.
        """
        if feats.shape[0] == 0:
            empty = torch.zeros(0, self.out_hidden_size, device=feats.device, dtype=feats.dtype)
            return empty, coords, layout

        merge_size = self.spatial_merge_size
        device = feats.device
        orig_dtype = feats.dtype

        # LayerNorm in float32 for numerical stability, cast back
        feats = F.layer_norm(
            feats.float(),
            self.norm.normalized_shape,
            self.norm.weight.float() if self.norm.weight is not None else None,
            self.norm.bias.float() if self.norm.bias is not None else None,
            self.norm.eps,
        ).to(orig_dtype)

        coords_int = coords.long()
        has_segment_dim = coords_int.shape[1] == 5

        if has_segment_dim:
            seg_col, t_col, h_col, w_col, z_col = 0, 1, 2, 3, 4
        else:
            # 4-dim: T serves as segment indicator
            seg_col, t_col, h_col, w_col, z_col = 0, 0, 1, 2, 3

        # Iterate over unique segment values. For images (T=0 always), this
        # gives one group per batch element. For video, different T frames
        # within the same segment share the same seg_col value (the batch idx),
        # so they are merged together spatially — which is correct since spatial
        # merging operates on (H, W) within a frame, and all frames of one image
        # in a packed sample share the same segment_id.
        segment_indices = coords_int[:, seg_col].unique(sorted=True)

        merged_feats_list = []
        merged_coords_list = []
        new_layout = [] if layout is not None else None
        current_offset = 0

        for seg_idx in segment_indices:
            seg_mask = coords_int[:, seg_col] == seg_idx
            seg_feats = feats[seg_mask]
            seg_coords = coords_int[seg_mask]

            h_coords = seg_coords[:, h_col]
            w_coords = seg_coords[:, w_col]

            # Compute merged position and local position within window
            merged_h = h_coords // merge_size
            merged_w = w_coords // merge_size
            local_h = h_coords % merge_size
            local_w = w_coords % merge_size
            local_idx = local_h * merge_size + local_w  # 0..merge_size^2-1

            # Include T in merge keys so patches from different frames are never merged.
            t_coords = seg_coords[:, t_col]
            max_merged_w = (w_coords.max().item() // merge_size) + 1
            max_merged_h = (h_coords.max().item() // merge_size) + 1
            merge_keys = t_coords * (max_merged_h * max_merged_w) + merged_h * max_merged_w + merged_w

            unique_keys, inverse_indices = merge_keys.unique(sorted=True, return_inverse=True)
            num_merged = unique_keys.shape[0]

            # Allocate [num_merged, merge_size^2, input_hidden_size]
            merged_seg_feats = torch.zeros(
                num_merged,
                merge_size * merge_size,
                self.input_hidden_size,
                device=device,
                dtype=orig_dtype,
            )
            # Vectorized scatter: O(1) GPU ops instead of O(N) Python loop.
            # Each token at (inverse_indices[i], local_idx[i]) gets seg_feats[i].
            merged_seg_feats[inverse_indices, local_idx] = seg_feats

            # Flatten: [num_merged, merge_size^2 * input_hidden_size]
            merged_seg_feats = merged_seg_feats.view(num_merged, self.merged_hidden_size)
            merged_feats_list.append(merged_seg_feats)

            # Build merged coordinates (recover T, H, W from flattened key)
            hw_size = max_merged_h * max_merged_w
            merged_t_coords = unique_keys // hw_size
            merged_h_coords = (unique_keys % hw_size) // max_merged_w
            merged_w_coords = (unique_keys % hw_size) % max_merged_w
            z_coord = seg_coords[0, z_col]

            if has_segment_dim:
                merged_seg_coords = torch.stack(
                    [
                        seg_idx.expand(num_merged),
                        merged_t_coords,
                        merged_h_coords,
                        merged_w_coords,
                        z_coord.expand(num_merged),
                    ],
                    dim=1,
                )
            else:
                merged_seg_coords = torch.stack(
                    [
                        merged_t_coords,
                        merged_h_coords,
                        merged_w_coords,
                        z_coord.expand(num_merged),
                    ],
                    dim=1,
                )

            merged_coords_list.append(merged_seg_coords)

            if new_layout is not None:
                new_layout.append(slice(current_offset, current_offset + num_merged))
                current_offset += num_merged

        merged_feats = torch.cat(merged_feats_list, dim=0)
        merged_coords = torch.cat(merged_coords_list, dim=0)

        # MLP projection: merged_hidden_size -> out_hidden_size
        merged_feats = self.linear_fc2(self.act_fn(self.linear_fc1(merged_feats)))

        return merged_feats, merged_coords, new_layout


class TextDecoderWrapper(nn.Module):
    """Family-aware causal LM wrapper with vision token injection for image captioning.

    The wrapper keeps the Qwen3 path intact while adding a Nemotron-specific
    adapter for models that do not expose ``inputs_embeds`` on the standard
    decoder forward/generation API.
    """

    def __init__(
        self,
        model_name: str | None = None,
        image_hidden_size: int = 1152,
        spatial_pool_size: int = 2,
        gradient_checkpointing: bool = True,
        dtype: torch.dtype | None = None,
        attn_implementation: str = "flash_attention_2",
        family_spec: TextDecoderFamilySpec | None = None,
    ):
        super().__init__()

        if dtype is None:
            dtype = torch.bfloat16

        from transformers import AutoModelForCausalLM

        self.spec = family_spec or get_text_decoder_family_spec(model_name=model_name)
        self._model_name: str = model_name or self.spec.default_model_name
        resolved_attn_implementation = _resolve_attn_implementation(attn_implementation)
        logging.info(f"Loading text decoder: {self._model_name}")
        logging.info(f"Using text decoder attention backend: {resolved_attn_implementation}")
        hf_cache_dir = os.environ.get("HF_HOME")
        local_files_only = hf_cache_dir is not None
        local_snapshot = resolve_hf_snapshot_path(self._model_name, hf_cache_dir, required_files=("config.json",))
        model_source = local_snapshot or self._model_name
        model_cache_dir = None if local_snapshot is not None else hf_cache_dir
        model_local_files_only = True if local_snapshot is not None else local_files_only

        try:
            self.text_decoder = AutoModelForCausalLM.from_pretrained(
                model_source,
                dtype=dtype,
                attn_implementation=resolved_attn_implementation,
                device_map=None,
                low_cpu_mem_usage=True,
                trust_remote_code=self.spec.trust_remote_code,
                cache_dir=model_cache_dir,
                local_files_only=model_local_files_only,
            )
        except Exception as e:
            if _is_flash_attention_error(e):
                logging.warning("Flash attention unavailable, falling back to eager")
                self.text_decoder = AutoModelForCausalLM.from_pretrained(
                    model_source,
                    dtype=dtype,
                    attn_implementation="eager",
                    device_map=None,
                    low_cpu_mem_usage=True,
                    trust_remote_code=self.spec.trust_remote_code,
                    cache_dir=model_cache_dir,
                    local_files_only=model_local_files_only,
                )
            else:
                raise

        if self.spec.family == NEMOTRON_2B_SPEC.family:
            repaired_rotary_buffers = _repair_nemotron_rotary_buffers(self.text_decoder)
            if repaired_rotary_buffers == 0:
                logging.warning(
                    "No Nemotron rotary embedding buffers were reinitialized; "
                    "the remote-code rotary module may have changed its inv_freq/dim/base attributes."
                )
            else:
                logging.info(f"Reinitialized {repaired_rotary_buffers} Nemotron rotary embedding buffers")

        self.lm_config = self.text_decoder.config
        self.lm_config.pad_token_id = self.spec.pad_token_id
        if getattr(self.lm_config, "eos_token_id", None) is None:
            if len(self.spec.eos_token_ids) == 1:
                self.lm_config.eos_token_id = self.spec.eos_token_ids[0]
            else:
                self.lm_config.eos_token_id = list(self.spec.eos_token_ids)
        self.vision_token_id = self.spec.vision_pad_token_id
        self.image_hidden_size = image_hidden_size
        self.spatial_pool_size = spatial_pool_size
        self._caption_tokenizer: Any | None = None
        hidden_size = self.lm_config.hidden_size

        # Spatial merging or simple projection
        if spatial_pool_size > 1:
            self.spatial_merger = SpatialPatchMerger(
                input_hidden_size=image_hidden_size,
                out_hidden_size=hidden_size,
                spatial_merge_size=spatial_pool_size,
            )
            self.image_proj = None
            logging.info(
                f"SpatialPatchMerger enabled: {image_hidden_size} -> {hidden_size}, "
                f"merge_size={spatial_pool_size} ({spatial_pool_size**2}x token reduction)"
            )
        else:
            self.spatial_merger = None
            self.image_proj = nn.Sequential(
                nn.Linear(image_hidden_size, image_hidden_size * 2),
                nn.GELU(),
                nn.Linear(image_hidden_size * 2, hidden_size),
            )

        # 2D position embeddings for merged image features
        self.image_pos_embed = ImagePositionEmbeddings(
            hidden_size=hidden_size,
            max_position=128,  # supports up to ~2048px / patch_size=16
            coord_dim=2,  # (H, W) only
        )

        # Disable KV cache for training forwards. Generation paths opt back in.
        self.text_decoder.config.use_cache = False

        self._manual_gradient_checkpointing_enabled = False
        gradient_checkpointing_enabled = False
        if gradient_checkpointing:
            try:
                self.text_decoder.gradient_checkpointing_enable()
                gradient_checkpointing_enabled = True
            except ValueError as exc:
                if "does not support gradient checkpointing" in str(exc).lower():
                    if not self.spec.supports_inputs_embeds_forward:
                        self._manual_gradient_checkpointing_enabled = True
                        gradient_checkpointing_enabled = True
                        logging.warning(
                            f"Text decoder {self.text_decoder.__class__.__name__} does not support native "
                            "gradient checkpointing; using manual activation checkpointing in the decoder "
                            "layer loop"
                        )
                    else:
                        logging.warning(
                            f"Text decoder {self.text_decoder.__class__.__name__} does not support "
                            "gradient checkpointing; continuing without it"
                        )
                else:
                    raise
        elif hasattr(self.text_decoder, "gradient_checkpointing_disable"):
            self.text_decoder.gradient_checkpointing_disable()

        logging.info(
            f"TextDecoderWrapper ready: family={self.spec.family}, hidden_size={hidden_size}, "
            f"layers={len(self.text_decoder.model.layers)}, "
            f"gradient_checkpointing={gradient_checkpointing_enabled}, use_cache=False"
        )

    def _get_eos_token_ids(self) -> tuple[int, ...]:
        """Resolve all EOS token IDs used for generation and bookkeeping."""
        eos_token_id = getattr(self.lm_config, "eos_token_id", None)
        if isinstance(eos_token_id, list):
            return tuple(int(token_id) for token_id in eos_token_id)
        if eos_token_id is None:
            return self.spec.eos_token_ids
        return (int(eos_token_id),)

    def _ensure_caption_tokenizer(self) -> Any:
        """Lazily load the tokenizer used for generation-time decoding."""
        if self._caption_tokenizer is None:
            hf_cache_dir = os.environ.get("HF_HOME")
            if self.spec.family == NEMOTRON_2B_SPEC.family:
                from transformers import AutoTokenizer

                patched_snapshot = prepare_nemotron_tokenizer_snapshot(self._model_name, hf_cache_dir)
                if patched_snapshot is not None:
                    self._caption_tokenizer = AutoTokenizer.from_pretrained(
                        patched_snapshot,
                        local_files_only=True,
                        trust_remote_code=self.spec.trust_remote_code,
                    )
                else:
                    self._caption_tokenizer = load_auto_tokenizer_from_cache(
                        self._model_name,
                        hf_cache_dir,
                        trust_remote_code=self.spec.trust_remote_code,
                    )
            else:
                self._caption_tokenizer = load_auto_tokenizer_from_cache(
                    self._model_name,
                    hf_cache_dir,
                    trust_remote_code=self.spec.trust_remote_code,
                )
        return self._caption_tokenizer

    def _embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Embed token IDs using the model's input embedding module."""
        return self.text_decoder.get_input_embeddings()(input_ids)

    def _forward_from_embeddings(
        self,
        text_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        cache_position: torch.Tensor | None = None,
        use_cache: bool = False,
        past_key_values: Any | None = None,
    ) -> SimpleNamespace:
        """Run the LM from caller-provided token embeddings."""
        if self.spec.supports_inputs_embeds_forward:
            model_kwargs: dict[str, Any] = {
                "inputs_embeds": text_embeds,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "use_cache": use_cache,
                "return_dict": True,
            }
            if past_key_values is not None:
                model_kwargs["past_key_values"] = past_key_values
            if self.spec.supports_cache_position and cache_position is not None:
                model_kwargs["cache_position"] = cache_position
            return self.text_decoder.model(**model_kwargs)

        hidden_states = text_embeds
        next_decoder_cache = () if use_cache else None
        hidden_padding_mask = None
        if (
            attention_mask is not None
            and attention_mask.dim() == 2
            and attention_mask.shape[0] == hidden_states.shape[0]
            and attention_mask.shape[1] == hidden_states.shape[1]
        ):
            # The manual Nemotron path batches unequal packed segments. Keep
            # padded rows finite between layers so masked SDPA outputs cannot
            # poison later key/value projections.
            hidden_padding_mask = attention_mask.to(device=hidden_states.device, dtype=torch.bool)  # [B,T]
        use_manual_gradient_checkpointing = (
            self._manual_gradient_checkpointing_enabled
            and self.training
            and not use_cache
            and hidden_states.requires_grad
        )
        for idx, decoder_layer in enumerate(self.text_decoder.model.layers):
            layer_past = past_key_values[idx] if past_key_values is not None else None
            if use_manual_gradient_checkpointing:

                def _layer_forward(
                    layer_hidden_states: torch.Tensor,
                    layer_module: nn.Module = decoder_layer,
                ) -> torch.Tensor:
                    layer_outputs = layer_module(
                        layer_hidden_states,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_value=None,
                        use_cache=False,
                    )
                    return layer_outputs[0]

                hidden_states = torch.utils.checkpoint.checkpoint(
                    _layer_forward,
                    hidden_states,
                    use_reentrant=False,
                )
                if hidden_padding_mask is not None:
                    hidden_states = hidden_states.masked_fill(~hidden_padding_mask[:, :, None], 0)  # [B,T,D]
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=layer_past,
                    use_cache=use_cache,
                )
                hidden_states = layer_outputs[0]
                if hidden_padding_mask is not None:
                    hidden_states = hidden_states.masked_fill(~hidden_padding_mask[:, :, None], 0)  # [B,T,D]
                if use_cache:
                    next_decoder_cache += (layer_outputs[1],)

        hidden_states = self.text_decoder.model.norm(hidden_states)
        return SimpleNamespace(last_hidden_state=hidden_states, past_key_values=next_decoder_cache)

    def _lm_head(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Project decoder hidden states to token logits."""
        return self.text_decoder.lm_head(hidden_states)

    def _sample_next_token(
        self,
        logits: torch.Tensor,
        *,
        do_sample: bool,
        temperature: float,
        top_p: float,
        top_k: int,
        suppress_token_ids: tuple[int, ...],
    ) -> torch.Tensor:
        """Sample or greedily select the next token from one-step logits."""
        next_token_logits = logits[:, -1, :].float()
        if suppress_token_ids:
            next_token_logits[:, list(suppress_token_ids)] = torch.finfo(next_token_logits.dtype).min

        if not do_sample:
            return torch.argmax(next_token_logits, dim=-1)

        safe_temperature = max(float(temperature), 1.0e-5)
        next_token_logits = next_token_logits / safe_temperature

        if top_k > 0:
            top_values, _ = torch.topk(next_token_logits, k=min(top_k, next_token_logits.shape[-1]), dim=-1)
            kth_values = top_values[:, -1].unsqueeze(-1)
            next_token_logits = next_token_logits.masked_fill(next_token_logits < kth_values, float("-inf"))

        if 0.0 < top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True, dim=-1)
            sorted_probs = torch.softmax(sorted_logits, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            sorted_mask = cumulative_probs > top_p
            sorted_mask[:, 1:] = sorted_mask[:, :-1].clone()
            sorted_mask[:, 0] = False
            next_token_logits.scatter_(
                1,
                sorted_indices,
                next_token_logits.gather(1, sorted_indices).masked_fill(sorted_mask, float("-inf")),
            )

        next_token_probs = torch.softmax(next_token_logits, dim=-1)
        return torch.multinomial(next_token_probs, num_samples=1).squeeze(-1)

    def _build_vqa_user_turn(
        self,
        tokenizer: Any,
        question: str,
        num_image_tokens: int,
    ) -> tuple[list[int], int]:
        """Build one multimodal user turn and return the image-pad start offset."""

        def _encode(text: str) -> list[int]:
            return tokenizer.encode(text, add_special_tokens=False)

        image_placeholder = "<image>"
        normalized_question = _keep_only_first_image_placeholder(str(question).strip(), image_placeholder)
        if self.spec.family == QWEN3_SPEC.family:
            user_prefix = [QWEN3_IM_START_TOKEN_ID] + _encode("user\n")

            if image_placeholder in normalized_question:
                placeholder_index = normalized_question.find(image_placeholder)
                before_text = normalized_question[:placeholder_index]
                after_text = normalized_question[placeholder_index + len(image_placeholder) :]
                before_ids = _encode(before_text)
                after_ids = _encode(after_text)
                image_pad_offset = len(user_prefix) + len(before_ids) + 1
                user_turn = (
                    user_prefix
                    + before_ids
                    + [QWEN3_VISION_START_TOKEN_ID]
                    + [QWEN3_VISION_PAD_TOKEN_ID] * num_image_tokens
                    + [QWEN3_VISION_END_TOKEN_ID]
                    + after_ids
                    + [QWEN3_IM_END_TOKEN_ID]
                    + _encode("\n")
                )
                return user_turn, image_pad_offset

            image_pad_offset = len(user_prefix) + 1
            user_turn = (
                user_prefix
                + [QWEN3_VISION_START_TOKEN_ID]
                + [QWEN3_VISION_PAD_TOKEN_ID] * num_image_tokens
                + [QWEN3_VISION_END_TOKEN_ID]
                + _encode("\n")
                + _encode(normalized_question)
                + [QWEN3_IM_END_TOKEN_ID]
                + _encode("\n")
            )
            return user_turn, image_pad_offset

        if self.spec.family == NEMOTRON_2B_SPEC.family:
            im_start_id = _get_required_token_id(tokenizer, NEMOTRON_2B_IM_START_TOKEN)
            im_end_id = _get_required_token_id(tokenizer, NEMOTRON_2B_IM_END_TOKEN)
            user_prefix = [im_start_id] + _encode("user\n")

            if image_placeholder in normalized_question:
                placeholder_index = normalized_question.find(image_placeholder)
                before_text = normalized_question[:placeholder_index]
                after_text = normalized_question[placeholder_index + len(image_placeholder) :]
                before_ids = _encode(before_text)
                after_ids = _encode(after_text)
                image_pad_offset = len(user_prefix) + len(before_ids) + 1
                user_turn = (
                    user_prefix
                    + before_ids
                    + [self.spec.vision_start_token_id]
                    + [self.spec.vision_pad_token_id] * num_image_tokens
                    + [self.spec.vision_end_token_id]
                    + after_ids
                    + [im_end_id]
                    + _encode("\n")
                )
                return user_turn, image_pad_offset

            image_pad_offset = len(user_prefix) + 1
            user_turn = (
                user_prefix
                + [self.spec.vision_start_token_id]
                + [self.spec.vision_pad_token_id] * num_image_tokens
                + [self.spec.vision_end_token_id]
                + _encode("\n")
                + _encode(normalized_question)
                + [im_end_id]
                + _encode("\n")
            )
            return user_turn, image_pad_offset

        raise NotImplementedError(
            f"VQA prompt construction is not implemented for text decoder family {self.spec.family!r}."
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        image_feats_tensor: torch.Tensor,
        image_coords: torch.Tensor,
        image_patch_indices: torch.Tensor,
        image_layout: list[slice] | None = None,
        segment_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, int]:
        """Forward pass: inject image features into text and run causal LM.

        When segment_ids are provided (packed sequences), the packed [1, S] sequence
        is split into per-segment batch elements [num_segments, max_seg_len].
        HuggingFace flash attention with unpadding then produces cu_seqlens per batch
        element, giving segment-isolated causal attention.
        Position IDs reset per segment for correct RoPE.

        Args:
            input_ids: [B, S] text token IDs (with <|image_pad|> as placeholders).
            image_feats_tensor: [N, encoder_dim] raw encoder output features.
            image_coords: [N, 4 or 5] spatial coordinates.
            image_patch_indices: [N_pooled] flat indices into [B*S] sequence where
                merged image features should be inserted. Padded entries are -1.
            image_layout: Optional batch layout slices for the image features.
            segment_ids: [B, S] segment IDs for packed sequences. Values >= 0 indicate
                valid segments, -1 indicates padding. When provided, enables
                segment-isolated attention and per-segment position IDs.

        Returns:
            lm_logits: [B, S, vocab_size] next-token prediction logits.
            num_pooled_tokens: Number of image tokens after spatial merging.
        """
        image_features = image_feats_tensor
        current_coords = image_coords

        if len(image_features) > 0:
            # Spatial merging + MLP projection
            if self.spatial_merger is not None:
                image_features, current_coords, _ = self.spatial_merger(
                    feats=image_features,
                    coords=current_coords,
                    layout=image_layout,
                )
            elif self.image_proj is not None:
                image_features = self.image_proj(image_features)

            # Extract (H, W) for 2D position embeddings
            if current_coords.shape[1] == 5:
                coords_2d = current_coords[:, 2:4]  # (seg, T, H, W, Z) -> (H, W)
            else:
                coords_2d = current_coords[:, 1:3]  # (T, H, W, Z) -> (H, W)

            image_features = self.image_pos_embed(features=image_features, coords=coords_2d)

        num_pooled_tokens = len(image_features)

        # Embed text tokens
        text_embeds = self._embed_input_ids(input_ids)  # [B, S, d]
        B, S, d = text_embeds.shape

        # Zero out vision placeholder positions, then insert real image features
        vision_mask = (input_ids != self.vision_token_id).to(dtype=text_embeds.dtype)
        text_embeds = text_embeds * vision_mask[:, :, None]

        if num_pooled_tokens > 0 and len(image_patch_indices) > 0:
            valid_indices = image_patch_indices[image_patch_indices >= 0]
            max_idx = B * S - 1
            valid_indices = valid_indices[valid_indices <= max_idx]
            num_to_insert = min(len(valid_indices), num_pooled_tokens)

            if num_to_insert > 0:
                text_embeds_flat = text_embeds.reshape(B * S, d)
                insert_idx = valid_indices[:num_to_insert].to(device=text_embeds.device, dtype=torch.long)
                text_embeds_flat[insert_idx] = image_features[:num_to_insert].to(text_embeds.dtype)
                text_embeds = text_embeds_flat.reshape(B, S, d)

        # Segment-aware forward pass for packed sequences
        if segment_ids is not None:
            lm_logits = self._forward_packed(text_embeds, input_ids, segment_ids)
        else:
            lm_logits = self._forward_standard(text_embeds, input_ids=input_ids)

        return lm_logits, num_pooled_tokens

    def _forward_standard(
        self,
        text_embeds: torch.Tensor,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Standard forward pass without segment isolation."""
        effective_text_embeds = text_embeds
        original_seq_len = text_embeds.shape[1]
        effective_seq_len = original_seq_len
        attention_mask = None
        if input_ids is not None:
            pad_token_id = self.lm_config.pad_token_id
            if pad_token_id is None:
                attention_mask = torch.ones_like(input_ids, dtype=torch.long)
            else:
                attention_mask = (input_ids != pad_token_id).to(dtype=torch.long)
                valid_lengths = attention_mask.sum(dim=1)
                if valid_lengths.numel() > 0:
                    # Caption/VQA batches are right padded. Trim the common trailing
                    # padding tail so non-packed Nemotron runs do not forward the
                    # entire fixed max_seq_len block through the decoder.
                    effective_seq_len = max(int(valid_lengths.max().item()), 1)
                    if effective_seq_len < original_seq_len:
                        effective_text_embeds = text_embeds[:, :effective_seq_len, :]
                        attention_mask = attention_mask[:, :effective_seq_len]

        cache_position = torch.arange(effective_seq_len, device=text_embeds.device)
        position_ids = cache_position.unsqueeze(0)
        outputs = self._forward_from_embeddings(
            text_embeds=effective_text_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache_position=cache_position,
            use_cache=False,
        )
        lm_logits = self._lm_head(outputs.last_hidden_state)
        if effective_seq_len == original_seq_len:
            return lm_logits

        pad_len = original_seq_len - effective_seq_len
        padded_logits = lm_logits.new_zeros(lm_logits.shape[0], pad_len, lm_logits.shape[-1])
        return torch.cat([lm_logits, padded_logits], dim=1)

    def _forward_packed(
        self,
        text_embeds: torch.Tensor,
        input_ids: torch.Tensor,
        segment_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Segment-isolated forward pass for packed sequences.

        Splits packed [B, S] into per-segment batch [num_segments, max_seg_len].
        Flash attention with unpadding naturally isolates batch elements via cu_seqlens.
        Position IDs reset per segment for correct RoPE.

        Args:
            text_embeds: [B, S, d] embeddings with image features already injected.
            input_ids: [B, S] original token IDs (for reconstructing output shape).
            segment_ids: [B, S] segment IDs (>=0 for valid, -1 for padding).

        Returns:
            lm_logits: [B, S, vocab_size] reassembled in original packed layout.
        """
        B, S, d = text_embeds.shape
        device = text_embeds.device
        dtype = text_embeds.dtype

        # Process each batch element (typically B=1 for packed sequences)
        all_logits = []
        for b in range(B):
            seg_ids = segment_ids[b]  # [S]
            embeds = text_embeds[b]  # [S, d]

            # Find valid segments (>= 0)
            valid_indices = (seg_ids >= 0).nonzero(as_tuple=True)[0]
            if valid_indices.numel() == 0:
                # All padding — return zeros
                V = self.lm_config.vocab_size
                all_logits.append(torch.zeros(S, V, device=device, dtype=dtype))
                continue

            valid_seg_ids = seg_ids.index_select(0, valid_indices)
            valid_embeds = embeds.index_select(0, valid_indices)

            # Packed collate emits each segment as one contiguous run, so
            # unique_consecutive gives the per-segment lengths without
            # scalarizing each segment length back to Python.
            segment_values, seg_lengths = torch.unique_consecutive(valid_seg_ids, return_counts=True)
            if __debug__:
                unique_segment_count = int(valid_seg_ids.unique().numel())
                assert segment_values.numel() == unique_segment_count, (
                    "Segment IDs must be contiguous within packed text decoder inputs"
                )
            num_segments = seg_lengths.numel()
            max_seg_len = int(seg_lengths.max().item())

            segment_rows = torch.repeat_interleave(
                torch.arange(num_segments, device=device, dtype=torch.long),
                seg_lengths,
            )
            segment_starts = torch.cumsum(seg_lengths, dim=0) - seg_lengths
            segment_positions = torch.arange(valid_indices.numel(), device=device, dtype=torch.long)
            segment_positions = segment_positions - torch.repeat_interleave(segment_starts, seg_lengths)

            # Build per-segment batch: [num_segments, max_seg_len, d]
            batch_embeds = torch.zeros(num_segments, max_seg_len, d, device=device, dtype=dtype)
            batch_embeds[segment_rows, segment_positions] = valid_embeds

            position_template = torch.arange(max_seg_len, device=device, dtype=torch.long)
            batch_mask = (position_template.unsqueeze(0) < seg_lengths.unsqueeze(1)).to(dtype=torch.long)
            batch_pos = position_template.unsqueeze(0).expand(num_segments, -1) * batch_mask

            # Forward through the text decoder with one batch element per segment,
            # which isolates attention across packed examples.
            cache_position = position_template
            outputs = self._forward_from_embeddings(
                text_embeds=batch_embeds,
                attention_mask=batch_mask,
                position_ids=batch_pos,
                cache_position=cache_position,
                use_cache=False,
            )
            seg_logits = self._lm_head(outputs.last_hidden_state)
            # seg_logits: [num_segments, max_seg_len, vocab_size]

            # Reassemble into original packed layout [S, vocab_size]
            V = seg_logits.shape[-1]
            packed_logits = torch.zeros(S, V, device=device, dtype=seg_logits.dtype)
            packed_logits[valid_indices] = seg_logits[segment_rows, segment_positions]

            all_logits.append(packed_logits)

        # Guard: empty batch (all samples skipped by collate) -> empty logits tensor.
        if len(all_logits) == 0:
            V = self.lm_config.vocab_size
            return torch.zeros(0, S, V, device=device, dtype=dtype)
        # Stack batch: [B, S, vocab_size]
        return torch.stack(all_logits, dim=0)

    def _decode_generation_result(
        self,
        generated_ids: torch.Tensor,
        input_len: int,
        tokenizer: Any,
        max_new_tokens: int,
        eos_token_ids: tuple[int, ...],
    ) -> tuple[str, dict[str, bool | int]]:
        """Decode generated token IDs and expose basic generation metadata."""
        generated_only = generated_ids[0, input_len:]
        generated_token_count = int(generated_only.shape[0])
        finished_with_eos = generated_token_count > 0 and int(generated_only[-1].item()) in eos_token_ids
        truncated = generated_token_count >= max_new_tokens and not finished_with_eos
        text = tokenizer.decode(generated_only, skip_special_tokens=True)
        metadata: dict[str, bool | int] = {
            "generated_tokens": generated_token_count,
            "finished_with_eos": finished_with_eos,
            "truncated": truncated,
        }
        return text, metadata

    def _generate_from_prefix_embeddings(
        self,
        *,
        input_ids: torch.Tensor,
        text_embeds: torch.Tensor,
        max_new_tokens: int,
        do_sample: bool,
        temperature: float,
        eos_token_ids: tuple[int, ...],
    ) -> torch.Tensor:
        """Autoregressively generate from a caller-provided embedding prefix."""
        if self.spec.supports_inputs_embeds_generate:
            eos_token_id: int | list[int] = eos_token_ids[0] if len(eos_token_ids) == 1 else list(eos_token_ids)
            generate_kwargs = dict(
                input_ids=input_ids,
                inputs_embeds=text_embeds,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                use_cache=True,
                pad_token_id=self.spec.pad_token_id,
                eos_token_id=eos_token_id,
            )
            if self.spec.suppress_token_ids:
                generate_kwargs["suppress_tokens"] = list(self.spec.suppress_token_ids)
            if do_sample:
                generate_kwargs["temperature"] = temperature
                generate_kwargs["top_p"] = 0.8
                generate_kwargs["top_k"] = 20
            else:
                generate_kwargs["top_p"] = 1.0
                generate_kwargs["top_k"] = 0
                generate_kwargs["temperature"] = 1.0
            return self.text_decoder.generate(**generate_kwargs)

        if input_ids.shape[0] != 1:
            raise ValueError("Manual caption generation for the Nemotron text decoder only supports batch size 1.")

        prefix_len = text_embeds.shape[1]
        prefix_position_ids = torch.arange(prefix_len, device=text_embeds.device, dtype=torch.long).unsqueeze(0)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        outputs = self._forward_from_embeddings(
            text_embeds=text_embeds,
            attention_mask=attention_mask,
            position_ids=prefix_position_ids,
            use_cache=True,
        )
        logits = self._lm_head(outputs.last_hidden_state)
        past_key_values = outputs.past_key_values
        generated_tokens: list[torch.Tensor] = []
        eos_token_tensor = torch.tensor(eos_token_ids, device=input_ids.device, dtype=input_ids.dtype)

        for step_idx in range(max_new_tokens):
            next_token = self._sample_next_token(
                logits,
                do_sample=do_sample,
                temperature=temperature,
                top_p=0.8 if do_sample else 1.0,
                top_k=20 if do_sample else 0,
                suppress_token_ids=self.spec.suppress_token_ids,
            )
            generated_tokens.append(next_token.unsqueeze(1))

            if bool(torch.isin(next_token, eos_token_tensor).all()):
                break
            if step_idx == max_new_tokens - 1:
                break

            total_len = input_ids.shape[1] + len(generated_tokens)
            attention_mask = torch.ones((input_ids.shape[0], total_len), dtype=torch.long, device=input_ids.device)
            position_ids = torch.full(
                (input_ids.shape[0], 1),
                fill_value=total_len - 1,
                device=input_ids.device,
                dtype=torch.long,
            )
            next_token_embeds = self._embed_input_ids(next_token.unsqueeze(1))
            outputs = self._forward_from_embeddings(
                text_embeds=next_token_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=True,
            )
            logits = self._lm_head(outputs.last_hidden_state)
            past_key_values = outputs.past_key_values

        if len(generated_tokens) == 0:
            return input_ids

        return torch.cat([input_ids, torch.cat(generated_tokens, dim=1)], dim=1)

    @torch.no_grad()
    def generate_caption(
        self,
        image_feats_tensor: torch.Tensor,
        image_coords: torch.Tensor,
        max_new_tokens: int = 256,
        do_sample: bool = True,
        temperature: float = 0.7,
        return_metadata: bool = False,
    ) -> str | tuple[str, dict[str, bool | int]]:
        """Generate a caption from a single image's encoder features.

        Args:
            image_feats_tensor: [N, encoder_dim] features for ONE image.
            image_coords: [N, 4 or 5] coordinates.
            max_new_tokens: Maximum tokens to generate.
            do_sample: Whether to sample (False = greedy for deterministic vis).
            temperature: Sampling temperature if do_sample=True.
            return_metadata: Whether to also return generation metadata.

        Returns:
            Generated caption string, or ``(caption, metadata)`` when requested.
        """
        # Spatial merge + position embeddings
        if self.spatial_merger is not None:
            image_features, coords, _ = self.spatial_merger(image_feats_tensor, image_coords)
        elif self.image_proj is not None:
            image_features = self.image_proj(image_feats_tensor)
            coords = image_coords
        else:
            image_features = image_feats_tensor
            coords = image_coords

        if coords.shape[1] == 5:
            coords_2d = coords[:, 2:4]
        else:
            coords_2d = coords[:, 1:3]
        image_features = self.image_pos_embed(image_features, coords_2d)

        num_image_tokens = len(image_features)

        # Build input: [<|vision_start|>, <|image_pad|>×N, <|vision_end|>]
        device = image_features.device
        input_ids = torch.tensor(
            [self.spec.vision_start_token_id]
            + [self.spec.vision_pad_token_id] * num_image_tokens
            + [self.spec.vision_end_token_id],
            dtype=torch.long,
            device=device,
        ).unsqueeze(0)

        # Embed and inject image features
        text_embeds = self._embed_input_ids(input_ids)
        vision_mask = (input_ids != self.spec.vision_pad_token_id).to(dtype=text_embeds.dtype)
        text_embeds = text_embeds * vision_mask[:, :, None]
        text_embeds[0, 1 : 1 + num_image_tokens] = image_features.to(text_embeds.dtype)

        input_len = input_ids.shape[1]
        eos_token_ids = self._get_eos_token_ids()
        generated_ids = self._generate_from_prefix_embeddings(
            input_ids=input_ids,
            text_embeds=text_embeds,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            eos_token_ids=eos_token_ids,
        )

        tokenizer = self._ensure_caption_tokenizer()
        caption, metadata = self._decode_generation_result(
            generated_ids=generated_ids,
            input_len=input_len,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
            eos_token_ids=eos_token_ids,
        )
        if return_metadata:
            return caption, metadata
        return caption

    @torch.no_grad()
    def generate_answer(
        self,
        image_feats_tensor: torch.Tensor,
        image_coords: torch.Tensor,
        question: str,
        max_new_tokens: int = 512,
        do_sample: bool = True,
        temperature: float = 0.7,
        return_metadata: bool = False,
    ) -> str | tuple[str, dict[str, bool | int]]:
        """Generate an answer to a question about an image.

        Qwen3 uses its native chat template. Nemotron uses its own native
        chat template with ``<|im_start|>``, ``<|im_end|>``, and
        ``</think>`` in no-thinking mode.

        Args:
            image_feats_tensor: [N, encoder_dim] features for ONE image.
            image_coords: [N, 4 or 5] coordinates.
            question: Question text about the image.
            max_new_tokens: Maximum tokens to generate.
            do_sample: Whether to sample.
            temperature: Sampling temperature if do_sample=True.
            return_metadata: Whether to also return generation metadata.

        Returns:
            Generated answer string, or ``(answer, metadata)`` when requested.
        """
        # Spatial merge + position embeddings (same as generate_caption)
        if self.spatial_merger is not None:
            image_features, coords, _ = self.spatial_merger(image_feats_tensor, image_coords)
        elif self.image_proj is not None:
            image_features = self.image_proj(image_feats_tensor)
            coords = image_coords
        else:
            image_features = image_feats_tensor
            coords = image_coords

        if coords.shape[1] == 5:
            coords_2d = coords[:, 2:4]
        else:
            coords_2d = coords[:, 1:3]
        image_features = self.image_pos_embed(image_features, coords_2d)

        num_image_tokens = len(image_features)
        device = image_features.device

        # Lazy-load tokenizer
        tok = self._ensure_caption_tokenizer()

        def _encode(s):
            return tok.encode(s, add_special_tokens=False)

        if self.spec.family == QWEN3_SPEC.family:
            system_turn = (
                [QWEN3_IM_START_TOKEN_ID]
                + _encode("system\nYou are a helpful assistant.")
                + [QWEN3_IM_END_TOKEN_ID]
                + _encode("\n")
            )
            user_turn, user_image_pad_offset = self._build_vqa_user_turn(
                tokenizer=tok,
                question=question,
                num_image_tokens=num_image_tokens,
            )
            # Empty think block signals Qwen3 non-thinking mode
            no_think = [QWEN3_THINK_START_TOKEN_ID] + _encode("\n\n") + [QWEN3_THINK_END_TOKEN_ID] + _encode("\n\n")
            asst_prefix = [QWEN3_IM_START_TOKEN_ID] + _encode("assistant\n") + no_think
            eos_token_ids = self._get_eos_token_ids()
        elif self.spec.family == NEMOTRON_2B_SPEC.family:
            im_start_id = _get_required_token_id(tok, NEMOTRON_2B_IM_START_TOKEN)
            im_end_id = _get_required_token_id(tok, NEMOTRON_2B_IM_END_TOKEN)
            end_think_id = _get_required_token_id(tok, NEMOTRON_2B_THINK_END_TOKEN)
            system_turn = [im_start_id] + _encode("system\nYou are a helpful assistant.") + [im_end_id] + _encode("\n")
            user_turn, user_image_pad_offset = self._build_vqa_user_turn(
                tokenizer=tok,
                question=question,
                num_image_tokens=num_image_tokens,
            )
            asst_prefix = [im_start_id] + _encode("assistant\n") + [end_think_id] + _encode("\n")
            eos_token_ids = self._get_eos_token_ids()
        else:
            raise NotImplementedError(
                f"VQA generation is not implemented for text decoder family {self.spec.family!r}."
            )

        prompt_ids = system_turn + user_turn + asst_prefix
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

        # Embed and inject image features
        text_embeds = self._embed_input_ids(input_ids)
        vision_mask = (input_ids != self.spec.vision_pad_token_id).to(dtype=text_embeds.dtype)
        text_embeds = text_embeds * vision_mask[:, :, None]

        # Image features start inside the multimodal user turn at the first image_pad token.
        ip_start = len(system_turn) + user_image_pad_offset
        text_embeds[0, ip_start : ip_start + num_image_tokens] = image_features.to(text_embeds.dtype)

        # Generate
        # Qwen3 best practice: do NOT use greedy decoding — causes repetitions.
        # Nemotron reuses the same defaults for consistency with captioning.
        input_len = input_ids.shape[1]
        generated_ids = self._generate_from_prefix_embeddings(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            text_embeds=text_embeds,
            temperature=temperature,
            eos_token_ids=eos_token_ids,
        )

        answer, metadata = self._decode_generation_result(
            generated_ids=generated_ids,
            input_len=input_len,
            tokenizer=tok,
            max_new_tokens=max_new_tokens,
            eos_token_ids=eos_token_ids,
        )
        if return_metadata:
            return answer, metadata
        return answer
