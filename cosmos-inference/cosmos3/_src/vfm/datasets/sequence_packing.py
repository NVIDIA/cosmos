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

"""Functions for implementing sequence packing with flexible attention modes.

This module provides utilities for packing text and image sequences together
with support for different attention patterns (causal, full, noise).

Key Components:
---------------
1. Attention Mask Creation:
   - create_sparse_mask(): Creates sparse masks for flex attention
   - prepare_attention_mask_per_sample(): Creates dense attention masks

2. Position ID Generation:
   - get_flattened_position_ids_extrapolate(): Extrapolation-based position encoding
   - get_flattened_position_ids_interpolate(): Interpolation-based position encoding

3. Tokenizer Setup:
   - add_special_tokens(): Adds image boundary tokens to tokenizer

4. Sequence Packing:
   - pack_input_sequence(): Main function for packing text and image sequences
   - Helper functions: _pack_text_tokens(), _pack_image_tokens(), _finalize_packed_data()

Sequence Format:
---------------
Each sample consists of alternating text and image sections:
  [text_tokens] <eos> <vision_start> [image_tokens] <vision_end> ...

Attention Modes:
---------------
- 'causal': Standard causal/autoregressive attention for text
- 'full': Bidirectional attention for images
- 'noise': Special mode for noise conditioning
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import torch
from torch.nn.attention.flex_attention import and_masks, or_masks

from cosmos3._src.imaginaire.attention.checks import check_valid_tuple_or_element
from cosmos3._src.imaginaire.attention.varlen import generate_multi_dim_varlen_parameters
from cosmos3._src.imaginaire.utils import log
from cosmos3._src.vfm.models.mot.unified_3dmrope_utils import (
    get_3d_mrope_ids_text_tokens,
    get_3d_mrope_ids_vae_tokens,
)
from cosmos3._src.vfm.models.utils.data_and_condition import GenerationDataClean
from cosmos3._src.vfm.tokenizers.tokenization_qwen2 import Qwen2Tokenizer

MAX_CAUSAL_LEN_IMAGE_BATCH = 0
MAX_FULL_LEN_IMAGE_BATCH = 0
MAX_CAUSAL_LEN_VIDEO_BATCH = 0
MAX_FULL_LEN_VIDEO_BATCH = 0


# ============================================================================
# Attention mask creation
# ============================================================================


def create_sparse_mask(document_lens, split_lens, attn_modes, device):
    """Create a sparse attention mask combining multiple attention patterns.

    Args:
        document_lens: List of document lengths
        split_lens: List of split lengths within documents
        attn_modes: List of attention modes ('causal', 'full', 'noise') for each split
        device: Device to place tensors on

    Returns:
        Combined mask using flex attention API
    """

    # Build sequence ID tensors for tracking full/noise attention regions
    full_and_noise_seq_ids = []
    noise_seq_ids = []

    for seq_idx, (length, attn_mode) in enumerate(zip(split_lens, attn_modes)):
        # Assign sequence ID for full/noise regions, -1 for causal regions
        seq_id = seq_idx if attn_mode in ["full", "noise"] else -1
        full_and_noise_seq_ids.extend([seq_id] * length)

        # Assign sequence ID only for noise regions
        noise_seq_id = seq_idx if attn_mode == "noise" else -1
        noise_seq_ids.extend([noise_seq_id] * length)

    full_and_noise_seq_id = torch.tensor(full_and_noise_seq_ids, device=device)  # [seq_len]
    noise_seq_id = torch.tensor(noise_seq_ids, device=device)  # [seq_len]
    document_id = torch.cat([torch.full((l,), i) for i, l in enumerate(document_lens, start=1)]).to(device)  # [seq_len]

    # Define component mask functions
    def causal_mask(b, h, q_idx, kv_idx):
        """Standard causal attention: query can only attend to prior keys."""
        return q_idx >= kv_idx

    def full_and_noise_mask(b, h, q_idx, kv_idx):
        """Allow attention within same full/noise sequence."""
        return (full_and_noise_seq_id[q_idx] == full_and_noise_seq_id[kv_idx]) & (full_and_noise_seq_id[q_idx] >= 0)

    def remove_noise_mask(b, h, q_idx, kv_idx):
        """Prevent attending to noise tokens from different sequences."""
        return ~((noise_seq_id[kv_idx] >= 0) & (noise_seq_id[q_idx] != noise_seq_id[kv_idx]))

    def sample_mask(b, h, q_idx, kv_idx):
        """Ensure attention stays within same document/sample."""
        return document_id[q_idx] == document_id[kv_idx]

    # Combine all masks: (causal OR full_and_noise) AND remove_noise AND sample
    return and_masks(or_masks(causal_mask, full_and_noise_mask), remove_noise_mask, sample_mask)


def prepare_attention_mask_per_sample(split_lens, attn_modes, device="cpu"):
    """Prepare dense attention mask for a single sample with multiple splits.

    Args:
        split_lens: List of integers indicating length of each split within the sample
        attn_modes: List of attention modes for each split ('causal', 'full', or 'noise')
        device: Device to place the attention mask tensor on

    Returns:
        Attention mask tensor of shape (sample_len, sample_len) with -inf for masked positions
    """
    sample_len = sum(split_lens)
    attention_mask = torch.zeros((sample_len, sample_len), dtype=torch.bool, device=device)  # [sample_len,sample_len]

    # First pass: Set up basic attention patterns for each split
    current_pos = 0
    for split_len, attn_mode in zip(split_lens, attn_modes):
        assert attn_mode in ["causal", "full", "noise"], f"Invalid attention mode: {attn_mode}"

        split_start = current_pos
        split_end = current_pos + split_len

        if attn_mode == "causal":
            # Causal: lower triangular within split + full attention to previous splits
            attention_mask[split_start:split_end, split_start:split_end] = torch.ones(
                (split_len, split_len), device=device
            ).tril()  # [split_len,split_len]
            attention_mask[split_start:split_end, :split_start] = 1
        else:  # "full" or "noise"
            # Full attention within split and to previous splits
            attention_mask[split_start:split_end, split_start:split_end] = torch.ones(
                (split_len, split_len), device=device
            )  # [split_len,split_len]
            attention_mask[split_start:split_end, :split_start] = 1

        current_pos += split_len

    # Second pass: Handle noise mode - mask out noise columns except within same split
    current_pos = 0
    for split_len, attn_mode in zip(split_lens, attn_modes):
        if attn_mode == "noise":
            split_start = current_pos
            split_end = current_pos + split_len

            # Zero out the entire column for noise tokens
            attention_mask[:, split_start:split_end] = 0
            # But allow self-attention within the noise split
            attention_mask[split_start:split_end, split_start:split_end] = 1

        current_pos += split_len

    # Convert boolean mask to float with -inf for masked positions
    attention_mask = torch.zeros_like(attention_mask, dtype=torch.float).masked_fill_(
        ~attention_mask, float("-inf")
    )  # [sample_len,sample_len]

    return attention_mask


# ============================================================================
# Tokenizer utilities
# ============================================================================


def add_special_tokens(tokenizer):
    """Add image-related special tokens to tokenizer if not already present.

    Args:
        tokenizer: Tokenizer to add special tokens to

    Returns:
        Tuple of (modified tokenizer, dict of new token IDs)
    """
    # Collect existing special tokens
    existing_special_tokens = []
    for key, value in tokenizer.special_tokens_map.items():
        if isinstance(value, str):
            existing_special_tokens.append(value)
        elif isinstance(value, list):
            existing_special_tokens.extend(value)

    # Define image boundary tokens to add if missing
    tokens_to_add = []
    if "<|vision_start|>" not in existing_special_tokens:
        tokens_to_add.append("<|vision_start|>")
    if "<|vision_end|>" not in existing_special_tokens:
        tokens_to_add.append("<|vision_end|>")

    # Add new tokens to tokenizer vocabulary
    if tokens_to_add:
        tokenizer.add_tokens(tokens_to_add)

    # Get token IDs for image boundary tokens
    new_token_ids = {
        "start_of_generation": tokenizer.convert_tokens_to_ids("<|vision_start|>"),
        "end_of_generation": tokenizer.convert_tokens_to_ids("<|vision_end|>"),
    }

    return tokenizer, new_token_ids


# ============================================================================
# Data structures
# ============================================================================


@dataclass
class ModalityData:
    """Unified container for a single generation modality's data.

    This dataclass serves dual purposes:
    1. During packing: Acts as a builder, accumulating data in lists
    2. After finalize(): Holds finalized tensors ready for model consumption

    Attributes:
        sequence_indexes: Indices in the packed sequence where this modality's tokens appear.
            List during building, Tensor after finalize().
        timesteps: Diffusion timesteps for each noised token.
            List during building, Tensor after finalize().
        mse_loss_indexes: Indices where MSE loss should be computed (noised tokens only).
            List during building, Tensor after finalize().
        token_shapes: Shape metadata for each sample's tokens.
            For vision: list of (T, H, W) tuples.
            For action: list of (T,) tuples.
        tokens: The actual latent tokens. List during build, Tensor after finalize().
        condition_mask: Mask indicating clean frames (1=clean, 0=noised). Only after finalize().
        noisy_frame_indexes: Indices of noised frames. Constructed from condition_mask during
            sequence packing to reduce GPU->CPU synchronization later. Only after finalize().
        domain_id: Domain ID for multi-domain training. Only after finalize(). NOTE: only used for action modality.
        raw_action_dim: Raw action dimension. Only after finalize(). NOTE: only used for action modality.
    """

    # Core tracking (list during build, tensor after finalize)
    sequence_indexes: list[int] | torch.Tensor = field(default_factory=list)
    timesteps: list[float] | torch.Tensor = field(default_factory=list)
    mse_loss_indexes: list[int] | torch.Tensor = field(default_factory=list)
    # list[tuple[int,int,int]] for vision, list[tuple[int]] for action, list[tuple[int,int,int]] for sound
    token_shapes: list = field(default_factory=list)

    # Populated during finalization (from GenerationDataClean / noise path)
    tokens: list[torch.Tensor] = field(default_factory=list)
    condition_mask: list[torch.Tensor] = field(default_factory=list)
    noisy_frame_indexes: list[torch.Tensor] = field(default_factory=list)
    domain_id: list[torch.Tensor] = field(default_factory=list)
    raw_action_dim: list[torch.Tensor | None] | None = field(default_factory=list)

    def to_cuda(self) -> None:
        """Move all tensor fields to CUDA in-place."""
        if isinstance(self.sequence_indexes, torch.Tensor):
            self.sequence_indexes = self.sequence_indexes.cuda()
        if isinstance(self.timesteps, torch.Tensor):
            self.timesteps = self.timesteps.cuda()
        if isinstance(self.mse_loss_indexes, torch.Tensor):
            self.mse_loss_indexes = self.mse_loss_indexes.cuda()
        self.tokens = [token.cuda() for token in self.tokens]
        self.condition_mask = [cm.cuda() for cm in self.condition_mask]
        self.noisy_frame_indexes = [ni.cuda() for ni in self.noisy_frame_indexes]
        self.domain_id = [d.cuda() for d in self.domain_id]
        # raw_action_dim is optional (e.g., when action-channel masking is disabled).
        if self.raw_action_dim is not None:
            self.raw_action_dim = [d.cuda() if d is not None else None for d in self.raw_action_dim]


@dataclass
class PackedSequence:
    """Unified sequence container - works as builder during packing and final output.

    This dataclass replaces the old SequenceStatus + PackedSequence pattern:
    - Build phase: Accumulate data using lists, modalities use ModalityData builders
    - After finalize(): Ready for model consumption with tensors

    Attributes:
        # Sequence structure
        sample_lens: Length of each sample in the packed sequence.
        split_lens: Length of each split (text/vision/action sections).
        attn_modes: Attention mode for each split ('causal', 'full').
        is_image_batch: Whether this batch contains images (vs videos).
        sequence_length: Total length of packed sequence. Computed during finalize().

        # Build-time tracking (not used after finalize)
        curr: Current position in the packed sequence during building.

        # Text modality (list during build, tensor after finalize)
        text_ids: All text token IDs (including special tokens).
        text_indexes: Indices where text tokens appear in sequence.
        position_ids: RoPE position IDs for all tokens.

        # Loss computation - Cross Entropy (text)
        label_ids: Label IDs for cross-entropy loss.
        ce_loss_indexes: Indices for computing cross-entropy loss.
        ce_loss_weights: Weights for cross-entropy loss.

        # Generation modalities - named fields for type safety
        vision: Vision modality data (images/videos). None if no vision in batch.
        action: Action modality data (robotics). None if no actions in batch.
        sound: Sound modality data (audio). None if no sound in batch.
    """

    # Sequence structure
    sample_lens: list[int] = field(default_factory=list)
    split_lens: list[int] = field(default_factory=list)
    attn_modes: list[str] = field(default_factory=list)
    is_image_batch: bool = False
    sequence_length: int = 0

    # Build-time tracking (used during packing, not after finalize)
    curr: int = 0

    # Text modality (list during build, tensor after finalize)
    text_ids: list[int] | torch.Tensor = field(default_factory=list)
    text_indexes: list[int] | torch.Tensor = field(default_factory=list)
    position_ids: list[int] | torch.Tensor = field(default_factory=list)

    # Loss computation - Cross Entropy (text)
    label_ids: list[int] | torch.Tensor | None = field(default_factory=list)
    ce_loss_indexes: list[int] | torch.Tensor | None = field(default_factory=list)
    ce_loss_weights: list[float] | torch.Tensor | None = field(default_factory=list)

    # Build-time mRoPE tracking (used during packing, not after finalize)
    # When _use_mrope=True, position_ids accumulates (3, N) tensors instead of ints,
    # and finalize() produces a (3, total_seq_len) tensor instead of (total_seq_len,).
    _use_mrope: bool = False
    # Running temporal index for mRoPE position ID generation within a single sample.
    # Reset to 0 at the start of each sample, then advanced by text and vision helpers
    # as segments are packed. Action reuses the pre-vision snapshot (parallel temporal
    # range) without advancing it. Float when FPS modulation is enabled.
    # E.g. offset=0 -> text(4 tokens) -> offset=4 -> vision(3 frames) -> offset=7.
    _mrope_temporal_offset: int | float = 0
    _mrope_reset_spatial: bool = True

    # Temporal causal: whether supertoken 0's action slot contains null tokens.
    # True for all training calls and AR frame 0; False for AR frame N>0 (real actions).
    # Used by three_way_attention to zero out V for null action tokens (inline when attention_meta.null_action_supertokens=True).
    null_action_supertokens: bool = False

    # Temporal causal: number of action tokens prefixing each vision supertoken.
    # Equals temporal_compression_factor when actions are packed inline; 0 when
    # action_gen=False or for non-temporal-causal layouts. Single source of truth
    # for downstream attention/KV-cache code (per-supertoken layout is
    # num_action_tokens_per_supertoken + H_p * W_p).
    num_action_tokens_per_supertoken: int = 0

    # Generation modalities - NAMED FIELDS for type safety
    vision: ModalityData | None = None
    action: ModalityData | None = None
    sound: ModalityData | None = None

    def finalize(
        self,
        gen_data_clean: GenerationDataClean,
    ) -> "PackedSequence":
        """Convert all lists to tensors and compute derived values.

        Args:
            gen_data_clean: GenerationDataClean for metadata (e.g., action domain IDs).

        Returns:
            New PackedSequence instance with tensors instead of lists.
        """
        # Compute sequence length
        sequence_length = sum(self.sample_lens)
        sample_lens = self.sample_lens.copy()
        split_lens = self.split_lens.copy()
        attn_modes = self.attn_modes.copy()

        # Prepare loss-related tensors (cross-entropy)
        label_ids: torch.Tensor | None = None
        ce_loss_indexes: torch.Tensor | None = None
        ce_loss_weights: torch.Tensor | None = None
        if self.label_ids and len(self.label_ids) > 0:
            label_ids = torch.tensor(self.label_ids)  # [N_ce_tokens]
            ce_loss_indexes = torch.tensor(self.ce_loss_indexes)  # [N_ce_tokens]
            ce_loss_weights = torch.tensor(self.ce_loss_weights)  # [N_ce_tokens]

        # The condition_mask and noisy_frame_indexes are kept as lists to support variable shapes.

        # Finalize vision modality
        vision: ModalityData | None = None
        if self.vision is not None and len(self.vision.sequence_indexes) > 0:
            vision = ModalityData(
                sequence_indexes=torch.tensor(self.vision.sequence_indexes, dtype=torch.long),  # [N_vision_tokens]
                timesteps=torch.tensor(self.vision.timesteps),  # [N_vision_noisy_tokens]
                mse_loss_indexes=torch.tensor(
                    self.vision.mse_loss_indexes, dtype=torch.long
                ),  # [N_vision_noisy_tokens]
                token_shapes=list(self.vision.token_shapes),
                tokens=self.vision.tokens,
                condition_mask=list(self.vision.condition_mask),
                noisy_frame_indexes=list(self.vision.noisy_frame_indexes),
            )

        # Finalize action modality
        action: ModalityData | None = None
        if self.action is not None and len(self.action.sequence_indexes) > 0:
            action = ModalityData(
                sequence_indexes=torch.tensor(self.action.sequence_indexes, dtype=torch.long),  # [N_action_tokens]
                timesteps=torch.tensor(self.action.timesteps),  # [N_action_noisy_tokens]
                mse_loss_indexes=torch.tensor(
                    self.action.mse_loss_indexes, dtype=torch.long
                ),  # [N_action_noisy_tokens]
                token_shapes=list(self.action.token_shapes),
                tokens=self.action.tokens,
                condition_mask=list(self.action.condition_mask),  # Keep as list to support variable shapes
                noisy_frame_indexes=list(self.action.noisy_frame_indexes),
                domain_id=(
                    gen_data_clean.action_domain_id
                    if gen_data_clean.action_domain_id is not None
                    else [torch.zeros(1, dtype=torch.long)] * len(self.action.token_shapes)
                ),
                raw_action_dim=gen_data_clean.raw_action_dim,
            )

        # Finalize sound modality (placeholder for future)
        sound: ModalityData | None = None
        if self.sound is not None and len(self.sound.sequence_indexes) > 0:
            sound = ModalityData(
                sequence_indexes=torch.tensor(self.sound.sequence_indexes, dtype=torch.long),  # [N_sound_tokens]
                timesteps=torch.tensor(self.sound.timesteps),  # [N_sound_noisy_tokens]
                mse_loss_indexes=torch.tensor(self.sound.mse_loss_indexes, dtype=torch.long),  # [N_sound_noisy_tokens]
                token_shapes=list(self.sound.token_shapes),
                tokens=self.sound.tokens,
                condition_mask=list(self.sound.condition_mask),
                noisy_frame_indexes=list(self.sound.noisy_frame_indexes),
            )

        # Finalize position IDs: 3D mRoPE (3, seq_len) or 1D RoPE (seq_len,)
        if self._use_mrope and len(self.position_ids) > 0 and isinstance(self.position_ids[0], torch.Tensor):
            mrope_tensors: list[torch.Tensor] = self.position_ids  # type: ignore[assignment]
            position_ids = torch.cat(mrope_tensors, dim=1)  # [3,actual_seq_len]
        else:  # Original 1D RoPE from Bagel, where all the media tokens share the same 1D position ID
            position_ids = torch.tensor(self.position_ids)  # [seq_len]

        return PackedSequence(
            # Sequence structure
            sequence_length=sequence_length,
            sample_lens=sample_lens,
            split_lens=split_lens,
            attn_modes=attn_modes,
            is_image_batch=gen_data_clean.is_image_batch,
            # Text modality (converted to tensors)
            text_ids=torch.tensor(self.text_ids, dtype=torch.long),  # [N_text_tokens]
            text_indexes=torch.tensor(self.text_indexes, dtype=torch.long),  # [N_text_tokens]
            position_ids=position_ids,  # [seq_len] or [3,seq_len]
            # Loss computation - Cross Entropy
            label_ids=label_ids,
            ce_loss_indexes=ce_loss_indexes,
            ce_loss_weights=ce_loss_weights,
            # Generation modalities
            vision=vision,
            action=action,
            sound=sound,
            # Temporal causal
            null_action_supertokens=self.null_action_supertokens,
            num_action_tokens_per_supertoken=self.num_action_tokens_per_supertoken,
        )

    def to_cuda(self) -> None:
        """Move all tensor fields to CUDA in-place."""
        if isinstance(self.text_ids, torch.Tensor):
            self.text_ids = self.text_ids.cuda()
        if isinstance(self.text_indexes, torch.Tensor):
            self.text_indexes = self.text_indexes.cuda()
        if isinstance(self.position_ids, torch.Tensor):
            self.position_ids = self.position_ids.cuda()
        if isinstance(self.label_ids, torch.Tensor):
            self.label_ids = self.label_ids.cuda()
        if isinstance(self.ce_loss_indexes, torch.Tensor):
            self.ce_loss_indexes = self.ce_loss_indexes.cuda()
        if isinstance(self.ce_loss_weights, torch.Tensor):
            self.ce_loss_weights = self.ce_loss_weights.cuda()
        if self.vision is not None:
            self.vision.to_cuda()
        if self.action is not None:
            self.action.to_cuda()
        if self.sound is not None:
            self.sound.to_cuda()


@dataclass
class SequencePlan:
    """Plan describing which modalities are present in a sample.

    This dataclass tracks the presence of different modalities (text, vision, action)
    and their conditioning configurations for a dataset sample. Unlike SequencePlan
    which holds the actual tensor data, this class provides a lightweight summary
    of what modalities exist and how they should be conditioned.

    Attributes:
        has_text: Whether text/caption tokens are present for this sample.
            Used for text-conditioned generation (e.g., text-to-image/video).
        has_vision: Whether vision input (image or video latents) is present.
            Defaults to False.
        condition_frame_indexes_vision: Indexes of latent vision frames that are clean/conditioning.
            [] means all frames are noised/supervised.
            All frames specified means all frames are clean (no MSE supervision).
            For multi-item samples (e.g. image editing where each sample has multiple
            separately-encoded images), this applies to each vision item individually.
            The number of items per sample is tracked by
            ``GenerationDataClean.num_vision_items_per_sample``.
        has_action: Whether action input is present for robotics/embodied AI tasks.
            Defaults to False.
        condition_frame_indexes_action: Indexes of action steps that are clean/conditioning.
            [] means all steps are noised/supervised.
            All steps specified means all steps are clean (no MSE supervision).
    """

    # -- understanding (text conditioning) --
    has_text: bool

    # -- vision modality --
    has_vision: bool = False
    condition_frame_indexes_vision: list[int] = field(default_factory=list)
    # If True, all vision items in this sample share the same temporal mRoPE grid
    # (controlnet-style transfer: target frame i is spatio-temporally aligned with
    # control frame i). Each item gets the same temporal_offset; spatial reset
    # behavior is unchanged. Requires num_vision_items_per_sample > 1, equal latent_t,
    # and equal fps across items. Default False preserves single-clip and
    # image-editing semantics where items represent distinct time states.
    share_vision_temporal_positions: bool = False

    # -- action modality --
    has_action: bool = False
    condition_frame_indexes_action: list[int] = field(default_factory=list)
    action_start_frame_offset: int = 1

    # -- sound modality --
    has_sound: bool = False
    condition_frame_indexes_sound: list[int] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "has_text": self.has_text,
            "has_vision": self.has_vision,
            "has_action": self.has_action,
            "has_sound": self.has_sound,
            "condition_frame_indexes_vision": self.condition_frame_indexes_vision,
            "condition_frame_indexes_action": self.condition_frame_indexes_action,
            "condition_frame_indexes_sound": self.condition_frame_indexes_sound,
            "share_vision_temporal_positions": self.share_vision_temporal_positions,
        }


# ============================================================================
# Helper functions for packing sequences
# ============================================================================


def compute_text_split_length(
    num_caption_tokens: int,
    special_tokens: Dict[str, int],
    has_generation: bool = True,
) -> int:
    """Compute the total text split length without mutating any state.

    This is the number of token positions occupied by the text split in a
    packed sequence: caption tokens + optional BOS + EOS + optional BOV.

    Args:
        num_caption_tokens: Number of raw caption token IDs (before special tokens).
        special_tokens: Dictionary of special token IDs (checked for ``"bos_token_id"``).
        has_generation: Whether a start-of-generation (BOV) token follows text.

    Returns:
        Total text split length (positions consumed in the packed sequence).
    """
    n = num_caption_tokens
    if "bos_token_id" in special_tokens:
        n += 1
    n += 1  # EOS
    if has_generation:
        n += 1  # start-of-generation / BOV
    return n


def _pack_text_tokens(
    packed_seq: PackedSequence,
    text_ids: List[int],
    special_tokens: Dict[str, int],
    curr_rope_id: int,
    has_generation: bool,
    use_float_positions: bool = False,
) -> Tuple[int, int, int]:
    """Pack text tokens into the sequence.

    Args:
        packed_seq: PackedSequence instance to accumulate data into.
        text_ids: List of text token IDs (integers).
        special_tokens: Dictionary of special token IDs.
        curr_rope_id: Current RoPE position ID.
        has_generation: Whether there's media/action after text.
        use_float_positions: If True, generate float position IDs for 3D mRoPE
            (for consistency with FPS-modulated vision tokens).

    Returns:
        Tuple of (updated curr_rope_id, split_length, sample_length).
    """
    # Ensure we're in build mode (fields are lists, not tensors)
    assert isinstance(packed_seq.text_ids, list), "PackedSequence must be in build mode"
    assert isinstance(packed_seq.text_indexes, list)
    assert isinstance(packed_seq.position_ids, list)
    assert isinstance(packed_seq.label_ids, list)
    assert isinstance(packed_seq.ce_loss_indexes, list)
    assert isinstance(packed_seq.ce_loss_weights, list)

    curr = packed_seq.curr

    # Prepend BOS token if available
    if "bos_token_id" in special_tokens:
        shifted_text_ids = [special_tokens["bos_token_id"]] + text_ids
    else:
        shifted_text_ids = text_ids

    split_len = 0

    # Add text tokens to sequence
    packed_seq.text_ids.extend(shifted_text_ids)
    packed_seq.text_indexes.extend(range(curr, curr + len(shifted_text_ids)))

    # Configure loss computation for text tokens
    packed_seq.ce_loss_indexes.extend(range(curr, curr + len(shifted_text_ids)))
    packed_seq.ce_loss_weights.extend([1.0] * len(shifted_text_ids))
    packed_seq.label_ids.extend(text_ids[1:] + [special_tokens["eos_token_id"]])

    curr += len(shifted_text_ids)
    split_len += len(shifted_text_ids)

    # Add EOS token
    packed_seq.text_ids.append(special_tokens["eos_token_id"])
    packed_seq.text_indexes.append(curr)
    curr += 1
    split_len += 1

    # Add start-of-generation token, but only if there's media/action present.
    if has_generation:
        packed_seq.text_ids.append(special_tokens["start_of_generation"])
        packed_seq.text_indexes.append(curr)
        curr += 1
        split_len += 1

    # Sanity check -- compute_text_split_length() is called elsewhere.
    assert split_len == compute_text_split_length(len(text_ids), special_tokens, has_generation)

    # Update position IDs and attention mode for text split
    if packed_seq._use_mrope:
        text_mrope_ids, packed_seq._mrope_temporal_offset = get_3d_mrope_ids_text_tokens(
            num_tokens=split_len,
            temporal_offset=packed_seq._mrope_temporal_offset,
            use_float_positions=use_float_positions,
        )  # text_mrope_ids: [3,split_len]
        packed_seq.position_ids.append(text_mrope_ids)
    else:
        packed_seq.position_ids.extend(range(curr_rope_id, curr_rope_id + split_len))
    packed_seq.attn_modes.append("causal")
    packed_seq.split_lens.append(split_len)

    packed_seq.curr = curr
    return curr_rope_id + split_len, split_len, split_len


def _pack_vision_tokens(
    packed_seq: PackedSequence,
    input_vision_tokens: torch.Tensor,
    condition_frame_indexes_vision: list[int],
    input_timestep: float | torch.Tensor,
    curr_rope_id: int,
    latent_patch_size: int = 1,
    vision_fps: float | None = None,
    enable_fps_modulation: bool = False,
    base_fps: float = 24.0,
    temporal_compression_factor: int = 4,
) -> int:
    """Pack vision tokens into the sequence.

    Args:
        packed_seq: PackedSequence instance to accumulate data into.
        input_vision_tokens: Vision latent tokens (C, T, H, W).
        condition_frame_indexes_vision: Indexes of conditioning frames.
        input_timestep: Diffusion timestep. Either a float (teacher_forcing/none — all frames
            share the same sigma) or a Tensor(T_max,) (diffusion_forcing — per-frame sigma;
            indexed as input_timestep[frame_idx] for each noisy frame).
        curr_rope_id: Current RoPE position ID.
        latent_patch_size: Patch size for latent patchification.
        vision_fps: Frames per second of the video. Used when enable_fps_modulation=True.
        enable_fps_modulation: If True, scale temporal position IDs based on video FPS.
        base_fps: Base FPS for normalization (default 24.0).
        temporal_compression_factor: VAE temporal compression factor (default 4).
    Returns:
        Vision split length.
    """
    # Ensure we're in build mode
    assert isinstance(packed_seq.position_ids, list), "PackedSequence must be in build mode"

    curr = packed_seq.curr
    vision_split_len = 0

    # Initialize vision modality if not present.
    if packed_seq.vision is None:
        packed_seq.vision = ModalityData()

    # Ensure vision modality is in build mode
    assert isinstance(packed_seq.vision.sequence_indexes, list)
    assert isinstance(packed_seq.vision.mse_loss_indexes, list)
    assert isinstance(packed_seq.vision.timesteps, list)
    assert isinstance(packed_seq.vision.tokens, list)

    # Compute position IDs for image patches
    _, _, latent_t, latent_h, latent_w = input_vision_tokens.shape
    if latent_patch_size < 1:
        raise ValueError(f"latent_patch_size must be >= 1, got {latent_patch_size}")
    # Use ceil to support latent dims not divisible by patch size (padding handled in network)
    patch_h = math.ceil(latent_h / latent_patch_size)
    patch_w = math.ceil(latent_w / latent_patch_size)
    packed_seq.vision.token_shapes.append((latent_t, patch_h, patch_w))
    packed_seq.vision.tokens.append(input_vision_tokens)

    # Add image token indexes and loss information
    num_vision_tokens = latent_t * patch_h * patch_w
    packed_seq.vision.sequence_indexes.extend(range(curr, curr + num_vision_tokens))

    # Supervise vision tokens based on conditioning frames
    condition_set = {idx for idx in condition_frame_indexes_vision if 0 <= idx < latent_t}
    assert isinstance(packed_seq.vision.condition_mask, list)

    vision_condition_mask = torch.zeros(
        (latent_t, 1, 1), device=input_vision_tokens.device, dtype=input_vision_tokens.dtype
    )  # [T,1,1]
    for frame_idx in condition_set:
        vision_condition_mask[frame_idx, 0, 0] = 1.0
    packed_seq.vision.condition_mask.append(vision_condition_mask)

    vision_noisy_frame_indexes = torch.tensor(
        [idx for idx in range(latent_t) if idx not in condition_set],
        device=input_vision_tokens.device,
        dtype=torch.long,
    )  # [N_noisy_frames]
    assert isinstance(packed_seq.vision.noisy_frame_indexes, list)
    packed_seq.vision.noisy_frame_indexes.append(vision_noisy_frame_indexes)

    frame_token_stride = patch_h * patch_w
    for frame_idx in range(latent_t):
        if frame_idx in condition_set:
            continue
        frame_start = curr + frame_idx * frame_token_stride
        frame_end = frame_start + frame_token_stride
        packed_seq.vision.mse_loss_indexes.extend(range(frame_start, frame_end))
        if isinstance(input_timestep, torch.Tensor):
            frame_ts = input_timestep[frame_idx].item()
        else:
            frame_ts = input_timestep
        packed_seq.vision.timesteps.extend([frame_ts] * frame_token_stride)

    curr += num_vision_tokens
    vision_split_len += num_vision_tokens

    # Update position IDs for image split
    if packed_seq._use_mrope:
        # Determine FPS for this vision segment (None disables FPS modulation)
        effective_fps = vision_fps if enable_fps_modulation else None

        vision_mrope_ids, packed_seq._mrope_temporal_offset = get_3d_mrope_ids_vae_tokens(
            grid_t=latent_t,
            grid_h=patch_h,
            grid_w=patch_w,
            temporal_offset=packed_seq._mrope_temporal_offset,
            reset_spatial_indices=packed_seq._mrope_reset_spatial,
            fps=effective_fps,
            base_fps=base_fps,
            temporal_compression_factor=temporal_compression_factor,
        )  # vision_mrope_ids: [3,N_vision_tokens]
        packed_seq.position_ids.append(vision_mrope_ids)
    else:
        # All image tokens share the same RoPE position ID
        packed_seq.position_ids.extend([curr_rope_id] * vision_split_len)

    packed_seq.curr = curr
    return vision_split_len


def _pack_action_tokens(
    packed_seq: PackedSequence,
    input_action_tokens: torch.Tensor,
    condition_frame_indexes_action: list[int],
    input_timestep: float,
    curr_rope_id: int,
    action_temporal_offset: int | float = 0,
    enable_fps_modulation: bool = False,
    base_fps: float = 24.0,
    action_fps: float | None = None,
    base_temporal_compression_factor: int | None = None,
    action_start_frame_offset: int = 1,
) -> int:
    """Pack action tokens into the sequence.

    Args:
        packed_seq: PackedSequence instance to accumulate data into.
        input_action_tokens: Action latent tokens (T, D).
        condition_frame_indexes_action: Indexes of conditioning action steps.
        input_timestep: Diffusion timestep.
        curr_rope_id: Current RoPE position ID.
        action_temporal_offset: Temporal offset for action mRoPE IDs (typically
            the vision start offset so action aligns temporally with vision).
        enable_fps_modulation: If True, scale temporal position IDs based on FPS.
        base_fps: Base FPS for normalization (default 24.0).
        action_fps: Frames per second of the action data. Used when enable_fps_modulation=True.
        base_temporal_compression_factor: Base temporal compression factor for FPS scaling.
            Should be set to the vision temporal compression factor (e.g. 4) so that action
            tokens advance at frame rate (4x finer) relative to vision latent frames.
            Only affects behavior when FPS modulation is enabled.
        action_start_frame_offset: Frame offset for aligning action[0] with the
            corresponding vision frame. Default 1 aligns action[0] with vision frame 1.
    Returns:
        Number of action tokens added.
    """
    # Ensure we're in build mode
    assert isinstance(packed_seq.position_ids, list), "PackedSequence must be in build mode"

    curr = packed_seq.curr
    action_split_len = input_action_tokens.shape[0]

    # Initialize action modality if not present
    if packed_seq.action is None:
        packed_seq.action = ModalityData()

    # Ensure action modality is in build mode
    assert isinstance(packed_seq.action.sequence_indexes, list)
    assert isinstance(packed_seq.action.mse_loss_indexes, list)
    assert isinstance(packed_seq.action.timesteps, list)
    assert isinstance(packed_seq.action.tokens, list)

    # Add token indexes and loss information
    action_indexes = list(range(curr, curr + action_split_len))
    packed_seq.action.sequence_indexes.extend(action_indexes)
    packed_seq.action.token_shapes.append((action_split_len,))
    packed_seq.action.tokens.append(input_action_tokens)


    condition_set = {idx for idx in condition_frame_indexes_action if 0 <= idx < action_split_len}
    assert isinstance(packed_seq.action.condition_mask, list)

    action_condition_mask = torch.zeros(
        (action_split_len, 1), device=input_action_tokens.device, dtype=input_action_tokens.dtype
    )  # [T_action,1]
    for frame_idx in condition_set:
        action_condition_mask[frame_idx, 0] = 1.0
    packed_seq.action.condition_mask.append(action_condition_mask)

    action_noisy_frame_indexes = torch.tensor(
        [idx for idx in range(action_split_len) if idx not in condition_set],
        device=input_action_tokens.device,
        dtype=torch.long,
    )  # [N_noisy_action_frames]
    assert isinstance(packed_seq.action.noisy_frame_indexes, list)
    packed_seq.action.noisy_frame_indexes.append(action_noisy_frame_indexes)

    frame_token_stride = 1  # Action has 1 token per frame (no spatial dimension)
    for frame_idx in range(action_split_len):
        if frame_idx in condition_set:
            continue
        frame_start = curr + frame_idx * frame_token_stride
        frame_end = frame_start + frame_token_stride
        packed_seq.action.mse_loss_indexes.extend(range(frame_start, frame_end))
        packed_seq.action.timesteps.extend([input_timestep] * frame_token_stride)

    # Update RoPE position IDs for action tokens.
    if packed_seq._use_mrope:
        # 3D mRoPE: action tokens use a 1x1 spatial grid with start_frame_offset=1
        # so action[0] (null token) aligns with vision frame 1, not frame 0.
        effective_fps = action_fps if enable_fps_modulation else None

        action_mrope_ids, _ = get_3d_mrope_ids_vae_tokens(
            grid_t=action_split_len,
            grid_h=1,
            grid_w=1,
            temporal_offset=action_temporal_offset,
            reset_spatial_indices=packed_seq._mrope_reset_spatial,
            fps=effective_fps,
            base_fps=base_fps,
            temporal_compression_factor=1,  # Action is at frame rate (no temporal compression)
            base_temporal_compression_factor=base_temporal_compression_factor,
            start_frame_offset=action_start_frame_offset,  # Align action[0] with vision frame action_start_frame_offset
        )  # action_mrope_ids: [3,N_action_tokens]
        packed_seq.position_ids.append(action_mrope_ids)
        # Note: we don't update _mrope_temporal_offset here because action tokens
        # share the temporal space with vision tokens (they run in parallel).
    else:
        # All action tokens share the SAME RoPE position as vision tokens (see docs/sequence_packing.md).
        packed_seq.position_ids.extend([curr_rope_id] * action_split_len)

    packed_seq.curr = curr + action_split_len
    return action_split_len


def _pack_sound_tokens(
    packed_seq: PackedSequence,
    input_sound_tokens: torch.Tensor,
    condition_frame_indexes_sound: list[int],
    input_timestep: float,
    curr_rope_id: int,
    sound_temporal_offset: int | float = 0,
    enable_fps_modulation: bool = False,
    base_fps: float = 24.0,
    sound_fps: float | None = None,
) -> int:
    """Pack sound/audio tokens into the sequence.

    Sound latents have shape [C, T] where C is channels and T is temporal frames.
    Sound tokens are added to the unified generation split to maintain FactoredSequencePack's
    2-split invariant (causal + full).

    Args:
        packed_seq: PackedSequence instance to accumulate data into.
        input_sound_tokens: Sound latent tokens (C, T).
        condition_frame_indexes_sound: Indexes of conditioning frames.
            [] means all frames are noised/supervised.
            All frames specified means all frames are clean (no MSE supervision).
        input_timestep: Diffusion timestep.
        curr_rope_id: Current RoPE position ID.
        sound_temporal_offset: Temporal offset for m-RoPE position IDs (aligned with vision start).
        enable_fps_modulation: If True, scale temporal positions by FPS ratio.
        base_fps: Base FPS for normalization (default 24.0).
        sound_fps: Sound latent FPS (e.g., 25.0). Used for FPS-aware m-RoPE positions.

    Returns:
        Number of sound tokens added.
    """
    # Ensure we're in build mode
    assert isinstance(packed_seq.position_ids, list), "PackedSequence must be in build mode"

    curr = packed_seq.curr

    # Sound latent shape: [C, T] → T tokens
    _, sound_split_len = input_sound_tokens.shape

    # Initialize sound modality if not present
    if packed_seq.sound is None:
        packed_seq.sound = ModalityData()

    # Ensure sound modality is in build mode
    assert isinstance(packed_seq.sound.sequence_indexes, list)
    assert isinstance(packed_seq.sound.mse_loss_indexes, list)
    assert isinstance(packed_seq.sound.timesteps, list)
    assert isinstance(packed_seq.sound.tokens, list)

    # Add token indexes - sound uses (T, 1, 1) shape for compatibility with 3D RoPE
    packed_seq.sound.token_shapes.append((sound_split_len, 1, 1))
    packed_seq.sound.sequence_indexes.extend(range(curr, curr + sound_split_len))
    packed_seq.sound.tokens.append(input_sound_tokens)

    # Supervise sound tokens based on conditioning frames
    condition_set = {idx for idx in condition_frame_indexes_sound if 0 <= idx < sound_split_len}
    assert isinstance(packed_seq.sound.condition_mask, list)

    # Condition mask: shape (T, 1) — 1 = clean/conditioning, 0 = noised/supervised
    sound_condition_mask = torch.zeros(
        (sound_split_len, 1), device=input_sound_tokens.device, dtype=input_sound_tokens.dtype
    )  # [T_sound,1]
    for frame_idx in condition_set:
        sound_condition_mask[frame_idx, 0] = 1.0
    packed_seq.sound.condition_mask.append(sound_condition_mask)

    sound_noisy_frame_indexes = torch.tensor(
        [idx for idx in range(sound_split_len) if idx not in condition_set],
        device=input_sound_tokens.device,
        dtype=torch.long,
    )  # [N_noisy_sound_frames]
    assert isinstance(packed_seq.sound.noisy_frame_indexes, list)
    packed_seq.sound.noisy_frame_indexes.append(sound_noisy_frame_indexes)

    # Add to MSE loss indexes and timesteps for non-conditioning frames
    for frame_idx in range(sound_split_len):
        if frame_idx in condition_set:
            continue
        # Sound has 1 token per frame (no spatial dimension)
        frame_start = curr + frame_idx
        frame_end = frame_start + 1
        packed_seq.sound.mse_loss_indexes.extend(range(frame_start, frame_end))
        packed_seq.sound.timesteps.extend([input_timestep])

    # Update RoPE position IDs for sound tokens.
    if packed_seq._use_mrope:
        # 3D mRoPE: sound tokens use a 1x1 spatial grid, aligned with vision temporal positions.
        # sound[0] aligns with vision frame 0 (start_frame_offset=0, unlike action which offsets by 1).
        effective_fps = sound_fps if enable_fps_modulation else None

        sound_mrope_ids, _ = get_3d_mrope_ids_vae_tokens(
            grid_t=sound_split_len,
            grid_h=1,
            grid_w=1,
            temporal_offset=sound_temporal_offset,
            reset_spatial_indices=packed_seq._mrope_reset_spatial,
            fps=effective_fps,
            base_fps=base_fps,
            temporal_compression_factor=1,  # Sound latent is already at sound_latent_fps (no further compression)
            start_frame_offset=0,  # Sound[0] aligns with vision frame 0
        )  # sound_mrope_ids: [3,N_sound_tokens]
        packed_seq.position_ids.append(sound_mrope_ids)
        # Note: we don't update _mrope_temporal_offset here because sound tokens
        # share the temporal space with vision tokens (they run in parallel).
    else:
        # All sound tokens share the SAME RoPE position as vision/action tokens (unified generation split).
        packed_seq.position_ids.extend([curr_rope_id] * sound_split_len)

    packed_seq.curr = curr + sound_split_len
    return sound_split_len


def _pack_supertokens_temporal_causal(
    packed_seq: "PackedSequence",
    input_vision_tokens: torch.Tensor,
    input_action_tokens: torch.Tensor | None,
    condition_frame_indexes_vision: list[int],
    input_timestep: float | torch.Tensor,
    curr_rope_id: int,
    latent_patch_size: int,
    temporal_compression_factor: int,
    action_dim: int,
    vision_fps: float | None = None,
    action_fps: float | None = None,
    enable_fps_modulation: bool = False,
    base_fps: float = 24.0,
    pack_action_tokens: bool = True,
) -> tuple[int, bool]:
    """Pack vision and (optionally) action tokens in supertoken order for temporal causal attention.

    Buffer layout per frame:
        pack_action_tokens=True:  [action_t (tcf), vision_t (H*W)]  — supertoken size tcf + H*W
        pack_action_tokens=False: [vision_t (H*W)]                  — supertoken size H*W

    Use ``pack_action_tokens=False`` when ``config.action_gen=False``; the resulting
    ``num_action_tokens_per_supertoken=0`` is stamped on the pack and read by the
    attention builder so NATTEN metadata stays in sync automatically.

    mRoPE layout (with actions, unified_3d_mrope only):
        - Null actions (frame 0):          all tcf tokens at ``temporal_offset``.
        - Real training actions (frames 1..T-1): ``start_frame_offset=1`` so the
          last action in group i co-locates with vision frame i.
        - AR real actions (single supertoken): ``start_frame_offset=0``.
        - Interleaved per frame as cat([action_ids, vision_ids]).

    ``input_timestep`` is float (TF/none) or Tensor(T_max,) (DF, per-frame sigma).
    Conditioning frames are excluded from mse_loss_indexes either way.

    Returns (total_split_len, null_action_flag); null_action_flag is False when
    pack_action_tokens=False.
    """
    assert isinstance(packed_seq.position_ids, list), "PackedSequence must be in build mode"

    _, _, latent_t, latent_h, latent_w = input_vision_tokens.shape
    patch_h = math.ceil(latent_h / latent_patch_size)
    patch_w = math.ceil(latent_w / latent_patch_size)
    tcf = temporal_compression_factor
    patches_per_frame = patch_h * patch_w
    supertoken_len = tcf + patches_per_frame if pack_action_tokens else patches_per_frame  # S

    # Initialize modalities if needed
    if packed_seq.vision is None:
        packed_seq.vision = ModalityData()
    if pack_action_tokens and packed_seq.action is None:
        packed_seq.action = ModalityData()

    assert isinstance(packed_seq.vision.sequence_indexes, list)
    assert isinstance(packed_seq.vision.mse_loss_indexes, list)
    assert isinstance(packed_seq.vision.timesteps, list)
    assert isinstance(packed_seq.vision.tokens, list)
    assert isinstance(packed_seq.vision.condition_mask, list)
    if pack_action_tokens:
        assert isinstance(packed_seq.action.sequence_indexes, list)
        assert isinstance(packed_seq.action.mse_loss_indexes, list)
        assert isinstance(packed_seq.action.timesteps, list)
        assert isinstance(packed_seq.action.tokens, list)
        assert isinstance(packed_seq.action.condition_mask, list)

    device = input_vision_tokens.device
    dtype = input_vision_tokens.dtype

    null_action_flag: bool
    if pack_action_tokens:
        # Build all_action_tokens: shape (latent_t * tcf, action_dim)
        #
        # Cases:
        #   1. Training with conditioning frame (latent_t > 1, real_actions < latent_t*tcf):
        #      Prepend tcf null tokens for frame 0, then real actions for frames 1..T-1.
        #   2. KV-cache continuation (latent_t > 1, real_actions == latent_t*tcf): all supertokens
        #      carry real actions (no conditioning frame in-segment).
        #   3. AR frame N>0 (latent_t == 1, action provided): real actions, no null prefix.
        #   4. AR frame 0 / image2video (action is None): all null tokens.
        if input_action_tokens is not None:
            # input_action_tokens shape: (1, T*tcf, D) or (T*tcf, D) for training; (tcf, D) for AR frame N>0
            if input_action_tokens.dim() == 3:
                real_actions = input_action_tokens.squeeze(0)  # [T*tcf,action_dim] or [N,action_dim]
            else:
                real_actions = input_action_tokens  # [N,action_dim]
            null_tokens = torch.zeros(tcf, action_dim, device=device, dtype=real_actions.dtype)  # [tcf,action_dim]
            if latent_t == 1:
                # AR frame N>0: single supertoken with real actions, no null prefix
                all_action_tokens = real_actions  # [tcf,action_dim]
                null_action_flag = False
            elif real_actions.shape[0] == latent_t * tcf:
                # All frames have real actions (e.g. KV-cache continuation segments)
                all_action_tokens = real_actions
                null_action_flag = False
            else:
                # Conditioning frame present: null for supertoken 0, real for 1..T-1
                all_action_tokens = torch.cat([null_tokens, real_actions], dim=0)  # [T*tcf,action_dim]
                null_action_flag = True
        else:
            # AR frame 0 or image2video: all action tokens are null
            all_action_tokens = torch.zeros(
                latent_t * tcf, action_dim, device=device, dtype=dtype
            )  # [T*tcf,action_dim]
            null_action_flag = True
    else:
        # pack_action_tokens=False: action tokens must not be supplied.
        assert input_action_tokens is None, (
            "pack_action_tokens=False requires input_action_tokens=None; got a non-None tensor."
        )
        null_action_flag = False

    # Record vision token shapes and tokens
    packed_seq.vision.token_shapes.append((latent_t, patch_h, patch_w))
    packed_seq.vision.tokens.append(input_vision_tokens)

    # Vision conditioning mask: (T, 1, 1)
    condition_set_vision = {idx for idx in condition_frame_indexes_vision if 0 <= idx < latent_t}
    vision_condition_mask = torch.zeros((latent_t, 1, 1), device=device, dtype=dtype)  # [T,1,1]
    for fidx in condition_set_vision:
        vision_condition_mask[fidx, 0, 0] = 1.0
    packed_seq.vision.condition_mask.append(vision_condition_mask)

    vision_noisy_frame_indexes = torch.tensor(
        [idx for idx in range(latent_t) if idx not in condition_set_vision],
        device=device,
        dtype=torch.long,
    )  # [N_noisy_frames]
    packed_seq.vision.noisy_frame_indexes.append(vision_noisy_frame_indexes)

    if pack_action_tokens:
        # Action token shapes: latent_t * tcf total (including null tokens)
        packed_seq.action.token_shapes.append((latent_t * tcf,))
        packed_seq.action.tokens.append(all_action_tokens)

        # Action conditioning mask: all action tokens are conditioning (not supervised)
        # Null tokens are always conditioning; real actions are conditioning too (they are inputs)
        action_condition_mask = torch.ones((latent_t * tcf, 1), device=device, dtype=dtype)  # [T*tcf,1]
        packed_seq.action.condition_mask.append(action_condition_mask)

    # Pack in interleaved supertoken order: [action_t, vision_t] for each frame t
    # (or just [vision_t] per frame when pack_action_tokens=False)
    curr = packed_seq.curr
    total_split_len = 0

    # mRoPE: snapshot offset before this sample, compute IDs
    if packed_seq._use_mrope:
        temporal_offset = packed_seq._mrope_temporal_offset
        effective_vision_fps = vision_fps if enable_fps_modulation else None

        # AR frame N>=1 with action_gen=True (latent_t==1 and real actions supplied):
        # shift both vision and action by start_frame_offset=1 so the last action in
        # the group co-locates with vision frame N, mirroring training's layout.
        # All other cases (training latent_t>1, AR action_gen=False, AR frame 0 null)
        # keep start_frame_offset=0. The caller in pack_input_sequence_autoregressive
        # seeds temporal_offset accordingly (N-1 frames back when this shift applies).
        ar_with_real_actions = latent_t == 1 and pack_action_tokens and input_action_tokens is not None
        vision_sfo = 1 if ar_with_real_actions else 0

        vision_ids_flat, new_offset = get_3d_mrope_ids_vae_tokens(
            grid_t=latent_t,
            grid_h=patch_h,
            grid_w=patch_w,
            temporal_offset=temporal_offset,
            reset_spatial_indices=packed_seq._mrope_reset_spatial,
            fps=effective_vision_fps,
            base_fps=base_fps,
            temporal_compression_factor=tcf,
            start_frame_offset=vision_sfo,
        )  # vision_ids_flat: [3,T*patch_h*patch_w]

        if pack_action_tokens:
            effective_action_fps = action_fps if enable_fps_modulation else None

            # Action IDs: null for frame 0 (all tcf tokens share temporal_offset,
            # co-located with vision frame 0), real for frames 1..T-1.
            # Real tokens (training and AR) use start_frame_offset=1 so the last
            # action in a group co-locates with vision frame i.
            fps_active = effective_action_fps is not None
            t_dtype = torch.float32 if fps_active else torch.long
            t_offset = float(temporal_offset) if fps_active else int(temporal_offset)
            null_t = torch.full((tcf,), t_offset, dtype=t_dtype)  # [tcf]
            null_hw = torch.zeros(tcf, dtype=t_dtype)  # [tcf]
            null_ids = torch.stack([null_t, null_hw, null_hw])  # [3,tcf]

            def _real_action_ids(n_frames: int, start_frame_offset: int) -> torch.Tensor:
                flat, _ = get_3d_mrope_ids_vae_tokens(
                    grid_t=n_frames * tcf,
                    grid_h=1,
                    grid_w=1,
                    temporal_offset=temporal_offset,
                    reset_spatial_indices=packed_seq._mrope_reset_spatial,
                    fps=effective_action_fps,
                    base_fps=base_fps,
                    temporal_compression_factor=1,
                    base_temporal_compression_factor=tcf,
                    start_frame_offset=start_frame_offset,
                )
                return flat.reshape(3, n_frames, tcf)  # [3,n_frames,tcf]

            if latent_t > 1 and input_action_tokens is not None:
                if real_actions.shape[0] == latent_t * tcf:
                    # KV continuation: real action in every supertoken (including frame 0)
                    action_ids_3d = _real_action_ids(latent_t, start_frame_offset=0)
                else:
                    # Training with conditioning frame: supertoken 0 = null, 1..T-1 = real
                    null_ids_3d = null_ids.reshape(3, 1, tcf)  # [3,1,tcf]
                    real_ids_3d = _real_action_ids(latent_t - 1, start_frame_offset=1)  # [3,T-1,tcf]
                    action_ids_3d = torch.cat([null_ids_3d, real_ids_3d], dim=1)  # [3,T,tcf]
            elif latent_t > 1:
                # No action tensor (all-null layout): same ID structure as training w/ conditioning frame.
                null_ids_3d = null_ids.reshape(3, 1, tcf)  # [3,1,tcf]
                real_ids_3d = _real_action_ids(latent_t - 1, start_frame_offset=1)  # [3,T-1,tcf]
                action_ids_3d = torch.cat([null_ids_3d, real_ids_3d], dim=1)  # [3,T,tcf]
            elif input_action_tokens is None:
                # AR frame 0 / image2video: only null
                action_ids_3d = null_ids.reshape(3, 1, tcf)  # [3,1,tcf]
            else:
                # AR frame N>=1: single supertoken with real actions. start_frame_offset=1
                # matches training (last action co-locates with vision frame N); caller
                # seeds temporal_offset to (N-1) frame-strides back to compensate.
                action_ids_3d = _real_action_ids(1, start_frame_offset=1)  # [3,1,tcf]

            # (3, T*H*W) → (3, T, H*W)
            vision_ids_3d = vision_ids_flat.reshape(3, latent_t, patches_per_frame)  # [3,T,patch_h*patch_w]

            # Interleave per frame: (3, T, tcf+H*W) → (3, T*S)
            interleaved_ids = torch.cat([action_ids_3d, vision_ids_3d], dim=2).reshape(
                3, latent_t * supertoken_len
            )  # [3,T*S]
            packed_seq.position_ids.append(interleaved_ids)
        else:
            # No action tokens: just vision IDs, already in (3, T*H*W) order.
            packed_seq.position_ids.append(vision_ids_flat)

        packed_seq._mrope_temporal_offset = new_offset

    for frame_t in range(latent_t):
        if pack_action_tokens:
            # Pack action tokens for this frame (indexes only; tokens already stored in packed_seq.action.tokens)
            action_indexes = list(range(curr, curr + tcf))
            packed_seq.action.sequence_indexes.extend(action_indexes)
            # Action tokens are never in MSE loss (always conditioning)
            curr += tcf
            total_split_len += tcf

            if not packed_seq._use_mrope:
                packed_seq.position_ids.extend([curr_rope_id] * tcf)

        # Pack vision tokens for this frame
        frame_indexes = list(range(curr, curr + patches_per_frame))
        packed_seq.vision.sequence_indexes.extend(frame_indexes)
        curr += patches_per_frame
        total_split_len += patches_per_frame

        if not packed_seq._use_mrope:
            packed_seq.position_ids.extend([curr_rope_id] * patches_per_frame)

        # Vision MSE loss: supervise non-conditioning frames
        if frame_t not in condition_set_vision:
            packed_seq.vision.mse_loss_indexes.extend(frame_indexes)
            frame_ts = input_timestep[frame_t].item() if isinstance(input_timestep, torch.Tensor) else input_timestep
            packed_seq.vision.timesteps.extend([frame_ts] * patches_per_frame)

    packed_seq.curr = curr
    return total_split_len, null_action_flag


# ============================================================================
# Main packing function
# ============================================================================


def pack_input_sequence(
    sequence_plans: list[SequencePlan],
    input_text_indexes: list[list[int]],
    gen_data_clean: GenerationDataClean,
    input_timesteps: torch.Tensor,
    special_tokens: dict[str, int],
    max_num_tokens: int | None = None,
    latent_patch_size: int = 1,
    skip_text_tokens: bool = False,
    include_end_of_generation_token: bool = False,
    position_embedding_type: str = "3d_rope",
    unified_3d_mrope_reset_spatial_ids: bool = True,
    unified_3d_mrope_temporal_modality_margin: int = 0,
    enable_fps_modulation: bool = False,
    base_fps: float = 24.0,
    temporal_compression_factor: int = 4,
    video_temporal_causal: bool = False,
    action_dim: int = 32,
    initial_mrope_temporal_offset: int | float = 0,
) -> PackedSequence:
    """
    Pack a sequence of input strings and VAE latents into a packed tensor format.
    Uses SequencePlan to determine which modalities are present for each sample,
    and maintains separate indices for text, vision, action, and sound to handle variable modality presence.

    Args:
        sequence_plans: List of SequencePlan items describing which modalities are present.
        input_text_indexes: List of text token ID sequences (only for samples where has_text=True).
        gen_data_clean: GenerationDataClean containing vision, action, and sound tensors.
            - x0_tokens_vision: Vision tensors for samples where has_vision=True
            - x0_tokens_action: Action tensors for samples where has_action=True
            - x0_tokens_sound: Sound tensors (list of [C, T]) for samples where has_sound=True
        input_timesteps: Diffusion timesteps for each sample. Shape (B,) or (B, 1) for
            teacher_forcing/none (all frames share the same sigma), or (B, T_max) for
            diffusion_forcing (per-frame independent sigma). Entries are extracted per
            sample as a float (numel==1) or Tensor(T_max,) for per-frame indexing.
        special_tokens: Dictionary containing special token IDs (eos_token_id, start_of_generation, end_of_generation)
        max_num_tokens: Maximum number of tokens in the packed sequence
        latent_patch_size: Patch size used by the network to pack latents
        skip_text_tokens: If True, skip packing text tokens
        include_end_of_generation_token: If True, append end-of-generation token
        position_embedding_type: Position embedding type for vision tokens:
            - "3d_rope": Additive 3D RoPE embeddings + 1D position IDs for attention
            - "flattened_sin_cos": Additive flattened sin/cos embeddings + 1D position IDs
            - "unified_3d_mrope": No additive embedding + 3D position IDs for Qwen3VL-style mRoPE
        unified_3d_mrope_reset_spatial_ids: If True (default), spatial (H, W) indices
            start from 0 for each vision segment. If False, spatial indices are offset
            by the temporal offset (Qwen2VL-style). Only used when position_embedding_type="unified_3d_mrope".
        enable_fps_modulation: If True, scale temporal position IDs based on video FPS
            to reflect real time. Requires fps_vision in gen_data_clean.
            Uses the same flag as diffusion_expert_config.enable_fps_modulation.
        base_fps: Base FPS for normalization (default 24.0).
            Uses the same value as diffusion_expert_config.base_fps.
        temporal_compression_factor: VAE temporal compression factor (default 4).
            Obtained from the VAE tokenizer at runtime.
    Returns:
        PackedSequence containing all packed tensors and metadata. See PackedSequence for field details.
    """
    del max_num_tokens

    assert special_tokens is not None, "Special tokens must be provided"
    assert isinstance(input_timesteps, torch.Tensor), "input_timesteps must be a tensor"
    if input_timesteps.is_cuda:
        raise ValueError("input_timesteps must be on CPU, not CUDA")
    if isinstance(input_text_indexes, torch.Tensor):
        raise ValueError("input_text_tokens must be a list, not a tensor")

    # Initialize packed sequence (acts as builder during packing)
    packed_seq = PackedSequence()

    # Configure 3D mRoPE on the builder (enabled when position_embedding_type is unified_3d_mrope)
    packed_seq._use_mrope = position_embedding_type == "unified_3d_mrope"
    packed_seq._mrope_reset_spatial = unified_3d_mrope_reset_spatial_ids

    # Maintain separate indices for each modality
    idx_text = 0
    idx_vision = 0
    idx_action = 0
    idx_sound = 0
    null_action_flags: list[bool] = []  # collected from TC path; asserted consistent after the loop

    # Validate: all samples must have text (causal split is always required for two-way attention).
    # CFG dropout only drops text *content*, not the structural text split.
    if not skip_text_tokens:
        for plan in sequence_plans:
            assert plan.has_text, "All sequence plans must have has_text=True when skip_text_tokens=False"

    # Pack each sample based on its sequence plan
    for sample_idx, sequence_plan in enumerate(sequence_plans):
        curr_rope_id = 0
        sample_len = 0

        # mRoPE temporal offset resets per sample.
        # initial_mrope_temporal_offset is non-zero only for AR inference (frame N seeds at N*tcf).
        packed_seq._mrope_temporal_offset = initial_mrope_temporal_offset

        _ts = input_timesteps[sample_idx]
        input_timestep = _ts.item() if _ts.numel() == 1 else _ts  # float (TF) or Tensor(T_max,) (DF)

        # Pack text tokens if has_text=True and not skipped
        if sequence_plan.has_text and not skip_text_tokens:
            text_ids = input_text_indexes[idx_text]
            idx_text += 1

            has_generation_for_sample = sequence_plan.has_vision or sequence_plan.has_action or sequence_plan.has_sound
            curr_rope_id, _, text_sample_len = _pack_text_tokens(
                packed_seq,
                text_ids,
                special_tokens,
                curr_rope_id,
                has_generation=has_generation_for_sample,
                use_float_positions=enable_fps_modulation,
            )
            sample_len += text_sample_len

            # End of text modality, add an offset as the boundary between text and vision.
            packed_seq._mrope_temporal_offset += unified_3d_mrope_temporal_modality_margin

        # Save temporal offset before vision for action tokens (action uses same offset as vision start)
        vision_start_temporal_offset = packed_seq._mrope_temporal_offset

        # Pack vision (and optionally action) tokens
        if video_temporal_causal and sequence_plan.has_vision:
            # Temporal causal path: when sequence_plan.has_action=True, interleaved supertokens
            # [action_t, vision_t]; when False, supertokens are just vision patches.
            assert position_embedding_type == "unified_3d_mrope", (
                "video_temporal_causal=True requires position_embedding_type='unified_3d_mrope'"
            )
            input_vision_tokens = gen_data_clean.x0_tokens_vision[idx_vision]
            idx_vision += 1

            vision_fps = None
            if (
                enable_fps_modulation
                and gen_data_clean.fps_vision is not None
                and idx_vision - 1 < len(gen_data_clean.fps_vision)
            ):
                vision_fps = float(gen_data_clean.fps_vision[idx_vision - 1].item())

            input_action_tokens_tc: torch.Tensor | None = None
            action_fps_tc: float | None = None
            if sequence_plan.has_action:
                input_action_tokens_tc = gen_data_clean.x0_tokens_action[idx_action]
                if (
                    enable_fps_modulation
                    and gen_data_clean.fps_action is not None
                    and idx_action < len(gen_data_clean.fps_action)
                ):
                    action_fps_tc = float(gen_data_clean.fps_action[idx_action].item())
                idx_action += 1

            supertoken_split_len, null_flag = _pack_supertokens_temporal_causal(
                packed_seq=packed_seq,
                input_vision_tokens=input_vision_tokens,
                input_action_tokens=input_action_tokens_tc,
                condition_frame_indexes_vision=sequence_plan.condition_frame_indexes_vision,
                input_timestep=input_timestep,
                curr_rope_id=curr_rope_id,
                latent_patch_size=latent_patch_size,
                temporal_compression_factor=temporal_compression_factor,
                action_dim=action_dim,
                vision_fps=vision_fps,
                action_fps=action_fps_tc,
                enable_fps_modulation=enable_fps_modulation,
                base_fps=base_fps,
                pack_action_tokens=sequence_plan.has_action,
            )
            null_action_flags.append(null_flag)
            # We assume all samples in a batch share the same has_action layout, so
            # stamp the supertoken layout constant directly here. This is the
            # single source of truth read by downstream attention / KV-cache
            # code (no recomputation in the network).
            packed_seq.num_action_tokens_per_supertoken = temporal_compression_factor if sequence_plan.has_action else 0
            sample_len += supertoken_split_len
            vision_split_len = supertoken_split_len
            action_split_len = 0  # Already absorbed into supertoken_split_len

        else:
            # Standard path: vision and action packed separately
            if sequence_plan.has_vision:
                # Determine how many vision items this sample owns.
                # For multi-item samples (e.g. image editing), num_vision_items_per_sample
                # records [2, 2, ...]; for standard T2I/T2V it is None (1 item per sample).
                num_vis = (
                    gen_data_clean.num_vision_items_per_sample[sample_idx]
                    if gen_data_clean.num_vision_items_per_sample is not None
                    else 1
                )

                vision_split_len = 0
                # Controlnet-style transfer: when set, all vision items share the same
                # temporal mRoPE grid. We snapshot the offset before the loop and
                # rewind to it before each item, so every item produces identical
                # temporal IDs. Each _pack_vision_tokens call still advances the
                # offset by latent_t internally; in shared-grid mode the post-loop
                # offset equals snapshot + latent_t (single-clip semantics for
                # downstream EOV / next-modality tokens).
                shared_grid = sequence_plan.share_vision_temporal_positions and num_vis > 1
                items_temporal_offset_snapshot = packed_seq._mrope_temporal_offset
                shared_latent_t: int | None = None
                shared_patch_h: int | None = None
                shared_patch_w: int | None = None
                # FPS is recorded per-sample (shape [B]); for multi-item samples
                # (transfer / image-edit) every vision item in this sample shares
                # the same conditioning FPS, so we read by sample_idx, not by the
                # flat idx_vision counter (which would alias to a neighbor sample's
                # fps and corrupt RoPE FPS modulation).
                sample_vision_fps: float | None = None
                if (
                    enable_fps_modulation
                    and gen_data_clean.fps_vision is not None
                    and sample_idx < len(gen_data_clean.fps_vision)
                ):
                    sample_vision_fps = float(gen_data_clean.fps_vision[sample_idx].item())

                for item_idx in range(num_vis):
                    input_vision_tokens = gen_data_clean.x0_tokens_vision[idx_vision]
                    vision_fps = sample_vision_fps
                    idx_vision += 1

                    # Determine conditioning for this vision item.
                    # For multi-item mode: all items except the last are fully conditioned
                    # (all frames are clean); the last item uses the SequencePlan's
                    # condition_frame_indexes_vision (typically [] = fully generated).
                    if num_vis > 1 and item_idx < num_vis - 1:
                        # Conditioning item (e.g. source image): mark all frames as clean
                        latent_t = input_vision_tokens.shape[2]
                        item_condition_frames = list(range(latent_t))
                    else:
                        # Generation item (single-item mode or last item in multi-item)
                        item_condition_frames = sequence_plan.condition_frame_indexes_vision

                    if shared_grid:
                        item_latent_t = input_vision_tokens.shape[2]
                        item_latent_h = input_vision_tokens.shape[3]
                        item_latent_w = input_vision_tokens.shape[4]
                        if shared_latent_t is None:
                            shared_latent_t = item_latent_t
                            shared_patch_h = item_latent_h
                            shared_patch_w = item_latent_w
                        else:
                            assert item_latent_t == shared_latent_t, (
                                f"share_vision_temporal_positions requires equal latent_t across items, "
                                f"got item {item_idx} latent_t={item_latent_t} vs first={shared_latent_t}"
                            )
                            assert item_latent_h == shared_patch_h and item_latent_w == shared_patch_w, (
                                f"share_vision_temporal_positions requires equal spatial grid across items, "
                                f"got item {item_idx} (H,W)=({item_latent_h},{item_latent_w}) "
                                f"vs first=({shared_patch_h},{shared_patch_w})"
                            )
                        # Rewind so this item starts at the same temporal offset as item 0.
                        packed_seq._mrope_temporal_offset = items_temporal_offset_snapshot

                    item_split_len = _pack_vision_tokens(
                        packed_seq=packed_seq,
                        input_vision_tokens=input_vision_tokens,
                        condition_frame_indexes_vision=item_condition_frames,
                        input_timestep=input_timestep,
                        curr_rope_id=curr_rope_id,
                        latent_patch_size=latent_patch_size,
                        vision_fps=vision_fps,
                        enable_fps_modulation=enable_fps_modulation,
                        base_fps=base_fps,
                        temporal_compression_factor=temporal_compression_factor,
                    )
                    vision_split_len += item_split_len
                sample_len += vision_split_len

            else:
                vision_split_len = 0

            # Pack action tokens if has_action=True
            if sequence_plan.has_action:
                input_action_tokens = gen_data_clean.x0_tokens_action[idx_action]

                # Get FPS for action (action may have its own FPS independent of vision)
                action_fps: float | None = None
                if (
                    enable_fps_modulation
                    and gen_data_clean.fps_action is not None
                    and idx_action < len(gen_data_clean.fps_action)
                ):
                    action_fps = float(gen_data_clean.fps_action[idx_action].item())

                idx_action += 1

                action_split_len = _pack_action_tokens(
                    packed_seq=packed_seq,
                    input_action_tokens=input_action_tokens,
                    condition_frame_indexes_action=sequence_plan.condition_frame_indexes_action,
                    input_timestep=input_timestep,
                    curr_rope_id=curr_rope_id,
                    action_temporal_offset=vision_start_temporal_offset,
                    enable_fps_modulation=enable_fps_modulation,
                    base_fps=base_fps,
                    action_fps=action_fps,
                    base_temporal_compression_factor=temporal_compression_factor,
                    action_start_frame_offset=sequence_plan.action_start_frame_offset,
                )
                sample_len += action_split_len
            else:
                action_split_len = 0

        # Pack sound tokens if has_sound=True
        if sequence_plan.has_sound:
            input_sound_tokens = gen_data_clean.x0_tokens_sound[idx_sound]

            # Get FPS for sound (from gen_data_clean, like vision and action)
            sound_fps: float | None = None
            if (
                enable_fps_modulation
                and gen_data_clean.fps_sound is not None
                and idx_sound < len(gen_data_clean.fps_sound)
            ):
                sound_fps = float(gen_data_clean.fps_sound[idx_sound].item())

            idx_sound += 1

            sound_split_len = _pack_sound_tokens(
                packed_seq=packed_seq,
                input_sound_tokens=input_sound_tokens,
                condition_frame_indexes_sound=sequence_plan.condition_frame_indexes_sound,
                input_timestep=input_timestep,
                curr_rope_id=curr_rope_id,
                sound_temporal_offset=vision_start_temporal_offset,
                enable_fps_modulation=enable_fps_modulation,
                base_fps=base_fps,
                sound_fps=sound_fps,
            )
            sample_len += sound_split_len
        else:
            sound_split_len = 0

        # Add end-of-generation token if needed
        eov_len = 0
        has_any_generation = sequence_plan.has_vision or sequence_plan.has_action or sequence_plan.has_sound
        if include_end_of_generation_token and has_any_generation:
            # Type narrowing: we're in build mode, fields are lists
            assert isinstance(packed_seq.text_ids, list)
            assert isinstance(packed_seq.text_indexes, list)
            assert isinstance(packed_seq.position_ids, list)

            packed_seq.text_ids.append(special_tokens["end_of_generation"])
            packed_seq.text_indexes.append(packed_seq.curr)

            # EOV position IDs: 3D mRoPE or 1D RoPE
            if packed_seq._use_mrope:
                # Use float dtype when FPS modulation is enabled for consistency
                eov_dtype = torch.float32 if enable_fps_modulation else torch.long
                eov_mrope_ids = torch.full((3, 1), packed_seq._mrope_temporal_offset, dtype=eov_dtype)  # [3,1]
                packed_seq.position_ids.append(eov_mrope_ids)  # type: ignore[arg-type]
                packed_seq._mrope_temporal_offset += 1
            else:
                packed_seq.position_ids.append(curr_rope_id)  # type: ignore[arg-type]

            packed_seq.curr += 1
            eov_len = 1
            sample_len += 1

        combined_split_len = vision_split_len + action_split_len + sound_split_len + eov_len
        packed_seq.attn_modes.append("full")
        packed_seq.split_lens.append(combined_split_len)
        packed_seq.sample_lens.append(sample_len)

    # Assert consistent null_action_supertokens across all TC samples, then set once
    if null_action_flags:
        assert len(set(null_action_flags)) == 1, (
            f"Inconsistent null_action_supertokens across samples: {null_action_flags}. "
            "All samples in a batch must have the same structure (all training or all AR inference)."
        )
        packed_seq.null_action_supertokens = null_action_flags[0]

    # Finalize and return packed data
    return packed_seq.finalize(
        gen_data_clean=gen_data_clean,
    )


# ============================================================================
# SequencePack:Operations on packed sequences
# ============================================================================

"""
SequencePack is a dictionary-based container for packed sequences.
We provide two implementations:

JointSequencePack:    Stores all sub-sequences for all-sequences in a single tensor.
                      It is more flexible but is less performant. In this implementation, understanding tokens
                      can be placed in either causal or full-attention sub-sequences.
FactoredSequencePack:
                      Stores causal/undersanding and full/generation sub-sequences as separate tensors.
                      It is less flexible but is more performant. In this implementation, understanding tokens
                      must be on the causal sub-sequence, and generation tokens must be in the full-attention sub-sequence.

NOTES:
 - We are aiming to deprecate and remove JointSequencePack; keeping it available for backwards compatibility at the moment.
 - The reason we're implementing them via dict instead of python classes is to make torch.compile + activation checkpointing to work.

is_sharded (bool):
    This flag indicates whether the sequence pack contains global data or a local shard for Context Parallelism (CP).
    - When True, tensors represent only the local slice (Global_Length / CP_World_Size).
    - Padding and reconstruction logic is skipped in `from_joint`.
    - Operations requiring global context (e.g., `get_all_seq`, position ID reconstruction) are not allowed when is_sharded is True.
"""


# "Fake" types for readability; everything is plain dict at runtime.
FactoredSequencePack = dict[str, Any]
JointSequencePack = dict[str, Any]
SequencePack = FactoredSequencePack | JointSequencePack

# ------------------------------------
# SequencePack: internal helpers
# ------------------------------------


def _find_non_causal_text_token_idx(
    attn_modes: List[str], split_lens: List[int], und_token_indexes: List[int]
) -> List[int]:
    """
    Find the indexes of the "und" tokens that are under the "full" mode.
    This are indices into the full_only_seq.
    """
    # Return indexes *into* full_only_seq, not into the original packed sequence.
    # The order within full_only_seq is the concatenation of each "full" split in order.
    out = []
    full_offset = 0
    packed_idx = 0
    und_token_set = set(und_token_indexes)
    for attn_mode, split_len in zip(attn_modes, split_lens):
        if attn_mode == "full":
            split_indices = range(packed_idx, packed_idx + split_len)
            # For this "full" split, find the und tokens within this split, mapped local to full_only_seq offset
            for local_idx, split_idx in enumerate(split_indices):
                if split_idx in und_token_set:
                    out.append(full_offset + local_idx)
            full_offset += split_len
        packed_idx += split_len
    return out


def _compute_mode_indices_and_offsets(
    split_lens: torch.Tensor | List[int], attn_modes: List[str], mode: str, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute indices from a joint tensor that are in the given mode.
    """
    indices = []
    offsets = [0]
    next_offset = 0
    start = 0

    if isinstance(split_lens, torch.Tensor):
        split_lens = split_lens.tolist()

    for i, (split_len, attn_mode) in enumerate(zip(split_lens, attn_modes)):
        if attn_mode == mode:
            indices.extend(range(start, start + split_len))
            next_offset += split_len
            offsets.append(next_offset)
        start += split_len
    return torch.tensor(indices, dtype=torch.int32, device=device), torch.tensor(  # [N_mode_tokens], [N_mode_splits+1]
        offsets, dtype=torch.int32, device=device
    )


# Pad causal_seq and full_only_seq to have length 2048 if not already at that size
def _pad_to_N(N, x: torch.Tensor) -> torch.Tensor:
    assert x.shape[0] <= N
    padded = x.new_zeros((N, *x.shape[1:]))
    padded[: x.shape[0]] = x
    return padded


def _round_up_to_N(n: int, cp_world_size: int = 1, pad_for_cuda_graphs: bool = False) -> int:
    if pad_for_cuda_graphs:
        # Reduce recompilations / CUDA graph re-captures by bucketing lengths.
        # <= 2K: 128,  <= 4K: 256,  <= 8K: 512,  <= 16K: 1024,  > 16K: 2048
        if n <= 2048:
            alignment = 128
        elif n <= 4096:
            alignment = 256
        elif n <= 8192:
            alignment = 512
        elif n <= 16384:
            alignment = 1024
        else:
            alignment = 2048
        n = ((n + alignment - 1) // alignment) * alignment

    # ensure it's divisible by cp_world_size
    if cp_world_size > 1:
        remainder = n % cp_world_size
        if remainder != 0:
            n += cp_world_size - remainder

    return n


def _pad(
    causal_seq: torch.Tensor, full_only_seq: torch.Tensor, max_causal_len: int, max_full_len: int
) -> tuple[torch.Tensor, torch.Tensor]:
    causal_seq = _pad_to_N(max_causal_len, causal_seq)
    full_only_seq = _pad_to_N(max_full_len, full_only_seq)
    return causal_seq, full_only_seq


def _ensure_core_metadata(pack: SequencePack) -> None:
    required = [
        "sample_offsets",
        "max_sample_len",
        "max_causal_len",
        "max_full_len",
        "_causal_indices",
        "_full_indices",
        "_causal_seq_offsets",
        "_full_only_seq_offsets",
        "is_sharded",
    ]
    for key in required:
        if key not in pack:
            raise KeyError(f"Missing required pack field: {key}")


def _init_sequence_pack(
    sample_lens: List[int],
    split_lens: List[int],
    attn_modes: List[str],
    device: torch.device,
) -> dict[str, Any]:
    _max_sample_len = max(sample_lens)
    _max_causal_len = max((split_lens[i] for i in range(len(split_lens)) if attn_modes[i] == "causal"), default=0)
    _max_full_len = max((split_lens[i] for i in range(len(split_lens)) if attn_modes[i] == "full"), default=0)

    sample_lens_cu = torch.tensor([0] + sample_lens, device=device, dtype=torch.int32)  # [N_samples+1]
    _sample_offsets = torch.cumsum(sample_lens_cu, dim=0, dtype=torch.int32)  # [N_samples+1]

    _causal_indices, _causal_seq_offsets = _compute_mode_indices_and_offsets(split_lens, attn_modes, "causal", device)
    _full_indices, _full_only_seq_offsets = _compute_mode_indices_and_offsets(split_lens, attn_modes, "full", device)

    return dict(
        sample_offsets=_sample_offsets,
        max_sample_len=_max_sample_len,
        max_causal_len=_max_causal_len,
        max_full_len=_max_full_len,
        _causal_indices=_causal_indices,
        _full_indices=_full_indices,
        _causal_seq_offsets=_causal_seq_offsets,
        _full_only_seq_offsets=_full_only_seq_offsets,
        _num_causal_tokens=len(_causal_indices),
        _num_full_tokens=len(_full_indices),
        split_lens=split_lens,
        attn_modes=attn_modes,
    )


# ------------------------------------
# SequencePack constructors
# ------------------------------------


def _round_up_for_cuda_graphs_or_cp(
    causal_seq: torch.Tensor,
    full_only_seq: torch.Tensor,
    need_causal: int,
    need_full: int,
    is_image_batch: bool,
    pad_for_cuda_graphs: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad causal/full sequences to the required lengths, growing global bounds for CUDA graphs."""
    if pad_for_cuda_graphs:
        global \
            MAX_CAUSAL_LEN_IMAGE_BATCH, \
            MAX_FULL_LEN_IMAGE_BATCH, \
            MAX_CAUSAL_LEN_VIDEO_BATCH, \
            MAX_FULL_LEN_VIDEO_BATCH
        if is_image_batch:
            if need_causal > MAX_CAUSAL_LEN_IMAGE_BATCH:
                MAX_CAUSAL_LEN_IMAGE_BATCH = need_causal
                log.info(f"Growing MAX_CAUSAL_LEN_IMAGE_BATCH to {MAX_CAUSAL_LEN_IMAGE_BATCH}", rank0_only=False)
            if need_full > MAX_FULL_LEN_IMAGE_BATCH:
                MAX_FULL_LEN_IMAGE_BATCH = need_full
                log.info(f"Growing MAX_FULL_LEN_IMAGE_BATCH to {MAX_FULL_LEN_IMAGE_BATCH}", rank0_only=False)
            causal_seq, full_only_seq = _pad(
                causal_seq,
                full_only_seq,
                max_causal_len=MAX_CAUSAL_LEN_IMAGE_BATCH,
                max_full_len=MAX_FULL_LEN_IMAGE_BATCH,
            )
        else:
            if need_causal > MAX_CAUSAL_LEN_VIDEO_BATCH:
                MAX_CAUSAL_LEN_VIDEO_BATCH = need_causal
                log.info(f"Growing MAX_CAUSAL_LEN_VIDEO_BATCH to {MAX_CAUSAL_LEN_VIDEO_BATCH}", rank0_only=False)
            if need_full > MAX_FULL_LEN_VIDEO_BATCH:
                MAX_FULL_LEN_VIDEO_BATCH = need_full
                log.info(f"Growing MAX_FULL_LEN_VIDEO_BATCH to {MAX_FULL_LEN_VIDEO_BATCH}", rank0_only=False)
            causal_seq, full_only_seq = _pad(
                causal_seq,
                full_only_seq,
                max_causal_len=MAX_CAUSAL_LEN_VIDEO_BATCH,
                max_full_len=MAX_FULL_LEN_VIDEO_BATCH,
            )
    elif need_causal != int(causal_seq.shape[0]) or need_full != int(full_only_seq.shape[0]):
        causal_seq, full_only_seq = _pad(causal_seq, full_only_seq, need_causal, need_full)
    return causal_seq, full_only_seq


def factored_from_joint_sequence(
    packed_sequence: torch.Tensor,
    attn_modes: List[str],
    split_lens: List[int],
    sample_lens: List[int],
    packed_und_token_indexes: torch.Tensor,
    packed_gen_token_indexes: torch.Tensor,
    is_image_batch: bool = False,
    cp_world_size: int = 1,
    pad_for_cuda_graphs: bool = False,
) -> FactoredSequencePack:
    """
    Create a factored sequence pack from a packed sequence and metadata.
    NOTE: Some arguments seem redundant because they in principle support more flexible sequence setups.
          This constructor checks that the required invariants for FactoredSequencePack are satisfied.
    NOTE: This constructor checks that there are no "und" tokens under "full" mode, and no "gen" tokens under "causal" mode,
          since this is a requirement for FactoredSequencePack.
    Args:
        packed_sequence (torch.Tensor): Tensor containing all tokens in the batch of sequences.
        attn_modes (List[str]): List of attention modes. Must be alternating ["causal", "full", ... "causal", "full"]
        split_lens (List[int]): Length of each subsequence. len(split_lens) == len(attn_modes)
        sample_lens (List[int]): Length of each sequence. len(sample_lens) == number of samples.
        packed_und_token_indexes (torch.Tensor): The indexes of the understanding tokens in the packed sequence.
        packed_gen_token_indexes (torch.Tensor): The indexes of the generating tokens in the packed sequence.
    """
    del packed_gen_token_indexes

    non_causal_text_idxs = _find_non_causal_text_token_idx(attn_modes, split_lens, packed_und_token_indexes.tolist())
    assert len(non_causal_text_idxs) == 0, "non_causal_text_idxs should be empty"

    assert sum(sample_lens) == packed_sequence.shape[0], (
        "sum(sample_lens) must be equal to the length of the packed sequence"
    )

    meta = _init_sequence_pack(sample_lens, split_lens, attn_modes, packed_sequence.device)
    causal_seq = packed_sequence[meta["_causal_indices"]]  # [N_causal_tokens,D]
    full_only_seq = packed_sequence[meta["_full_indices"]]  # [N_full_tokens,D]

    need_causal = _round_up_to_N(int(causal_seq.shape[0]), cp_world_size, pad_for_cuda_graphs)
    need_full = _round_up_to_N(int(full_only_seq.shape[0]), cp_world_size, pad_for_cuda_graphs)

    causal_seq, full_only_seq = _round_up_for_cuda_graphs_or_cp(
        causal_seq,
        full_only_seq,
        need_causal,
        need_full,
        is_image_batch,
        pad_for_cuda_graphs,
    )

    pack: FactoredSequencePack = {
        **meta,
        "max_num_tokens": sum(sample_lens),
        "causal_seq": causal_seq,
        "full_only_seq": full_only_seq,
        "is_sharded": False,
    }
    return pack


def _validate_single_dim_params(params: Mapping, layer_idx: int, num_dims: int | None) -> dict:
    """
    Helper function to validate NATTEN parameters for a dimensionality profile.

    Args:
        params (Mapping): parameter dict with window_size/window_size_float and other params
        layer_idx (int): layer index for error messages
        num_dims (int | None): 1, 2, 3, or None (for single-profile format)

    Returns:
        dict: validated parameter dict with proper types
    """
    if not isinstance(params, Mapping):
        dim_str = f" ({num_dims}-D)" if num_dims else ""
        raise ValueError(f"Parameters for layer {layer_idx}{dim_str} must be a dict or None, got {params=}.")

    is_causal = False if "is_causal" not in params else params["is_causal"]

    if "window_size_float" in params:
        window_size_float = params["window_size_float"]
        if (
            not isinstance(window_size_float, Sequence)
            or len(window_size_float) not in [1, 2, 3]
            or any(not isinstance(x, float) for x in window_size_float)
        ):
            raise ValueError(f"'window_size_float' must be a float tuple of size 1, 2, or 3, got {window_size_float=}")
        window_size_float = tuple(k for k in window_size_float)

        num_dims = len(window_size_float)

        def check_stride_dilation(x):
            if isinstance(x, float):
                if 0.0 <= x <= 1.0:
                    return tuple(x for _ in range(num_dims))
            elif (
                isinstance(x, Sequence)
                and len(x) == num_dims
                and all(isinstance(y, float) and 0.0 <= y <= 1.0 for y in x)
            ):
                return tuple(y for y in x)
            else:
                raise ValueError(f"Invalid natten float parameter: {x=}")

        stride_float = 0.0 if "stride_float" not in params else params["stride_float"]
        dilation_float = 0.0 if "dilation_float" not in params else params["dilation_float"]

        stride_float = check_stride_dilation(stride_float)
        dilation_float = check_stride_dilation(dilation_float)
        is_causal = check_valid_tuple_or_element(
            is_causal, num_dims=num_dims, typename=bool, raise_error=True, param_name="is_causal"
        )

        if any(x in params for x in ["window_size", "stride", "dilation"]):
            raise ValueError(
                f"Please either use _float parameters, or integer ones, and not mix the two. Got {params=}."
            )

        return {
            "window_size_float": window_size_float,
            "stride_float": stride_float,
            "dilation_float": dilation_float,
            "is_causal": is_causal,
        }

    elif "window_size" in params:
        window_size = params["window_size"]
        num_dims = len(window_size)

        stride = 1 if "stride" not in params else params["stride"]
        dilation = 1 if "dilation" not in params else params["dilation"]

        if any("_float" in x for x in params.keys()):
            raise ValueError(
                f"Please either use _float parameters, or integer ones, and not mix the two. Got {params=}."
            )

        window_size = check_valid_tuple_or_element(
            window_size, num_dims=num_dims, typename=int, raise_error=True, param_name="window_size"
        )
        stride = check_valid_tuple_or_element(
            stride, num_dims=num_dims, typename=int, raise_error=True, param_name="stride"
        )
        dilation = check_valid_tuple_or_element(
            dilation, num_dims=num_dims, typename=int, raise_error=True, param_name="dilation"
        )
        is_causal = check_valid_tuple_or_element(
            is_causal, num_dims=num_dims, typename=bool, raise_error=True, param_name="is_causal"
        )

        return {"window_size": window_size, "stride": stride, "dilation": dilation, "is_causal": is_causal}
    else:
        raise ValueError(
            "Sparse parameters for a layer must have key 'window_size' or 'window_size_float', "
            f"got {params=} in layer index {layer_idx}."
        )


def verify_natten_parameter_list(
    natten_parameter_list: list | None,
    num_layers: int,
) -> list | None:
    """
    Converts list of NATTEN parameters into expected types, and assigns defaults to unset
    parameters.
    This needs to be done separately during model initialization, and not forward pass.
    There are no torch operations in this function.

    Args:
        natten_parameter_list (list | None): list of NATTEN parameters. Must be either None, or a
            list of mappings, one for each layer. Each list element must be either None,
            representing no sparsity / masking (full dense attention), or a mapping of NATTEN
            parameters.

            Parameters can be specified directly with integer or float format:
                - 'window_size_float' (required), 'stride_float', 'dilation_float'
                - 'window_size' (required), 'stride', 'dilation'

            Or, parameters can be specified for multiple dimensionality profiles in case of
            mixed-training (i.e. image and video training) using keys "1d", "2d", "3d":
                - Each key maps to either None (dense attention) or a parameter dict

            Integer and float parameters cannot be used together in the same layer!
            Additionally, you can specify 'is_causal'.

            Examples:
            ```
            # 50 percent sparsity along each dimension in a 2-D token layout
            {'window_size_float': (0.5, 0.5)}  # valid

            # 50 percent sparsity along each dimension in a 2-D token layout
            # Maximum dilation along first dimension, no dilation along second dimension
            {'window_size_float': (0.5, 0.5), 'dilation_float': (1.0, 0.0)}  # valid

            # Fixed window size of 8x8, dilation of 2x1.

            {'window_size': (8, 8), 'dilation': (2, 1)}  # valid

            # Multi-profile: different parameters for 2D (images) and 3D (videos)
            {
                "2d": {"window_size_float": (0.5, 0.5)},
                "3d": {"window_size_float": (1.0, 0.5, 0.5)}
            }  # valid

            # Multi-profile: 2D uses dense attention, 3D uses sparse
            {
                "2d": None,
                "3d": {"window_size_float": (1.0, 0.5, 0.5)}
            }  # valid

            # Invalid:
            {'window_size_float': (0.5, 0.5), 'dilation': (2, 1)}
            ```

        num_layers (int): number of layers in the model. Just used to verify list length.

    Returns:
        output_parameter_list (list | None): verified and type-checked NATTEN parameters, or None if
            no parameters passed.
    """

    if natten_parameter_list is not None:
        parameter_list_out = []
        if not isinstance(natten_parameter_list, Sequence):
            raise ValueError(f"Argument 'natten_parameter_list' must be a list or None, got {natten_parameter_list=}.")

        if len(natten_parameter_list) != num_layers:
            raise ValueError(
                "Number of elements in 'natten_parameter_list' must match number of layers "
                f"in the model, got {num_layers=}, {len(natten_parameter_list)=}."
            )

        for i, layer_parameters in enumerate(natten_parameter_list):
            if layer_parameters is None:
                log.debug(f"Layer {i} will use DENSE attention.")
                parameter_list_out.append(None)
                continue

            if not isinstance(layer_parameters, Mapping):
                raise ValueError(
                    f"Sparse parameters for a layer must be a dict or None, got {layer_parameters=} in layer index {i}."
                )

            # Detect format: multi-profile if has keys "1d", "2d", or "3d"
            dim_keys = {"1d", "2d", "3d"}
            has_dim_keys = any(k in layer_parameters for k in dim_keys)

            if has_dim_keys:
                # Multi-profile format: validate each explicitly defined dimensionality profile
                validated_multi_profile = {}
                for dim_str, dim_int in [("1d", 1), ("2d", 2), ("3d", 3)]:
                    if dim_str in layer_parameters:
                        dim_params = layer_parameters[dim_str]
                        if dim_params is None:
                            validated_multi_profile[dim_int] = None
                        else:
                            validated_multi_profile[dim_int] = _validate_single_dim_params(dim_params, i, dim_int)
            else:
                # Single-profile format: validate and convert to multi-profile format
                # Infer dimensionality from parameter tuple length
                validated_params = _validate_single_dim_params(layer_parameters, i, None)
                if "window_size_float" in validated_params:
                    num_dims = len(validated_params["window_size_float"])
                else:  # "window_size"
                    num_dims = len(validated_params["window_size"])
                validated_multi_profile = {num_dims: validated_params}

            # If all explicitly defined profiles are None, treat as fully dense layer
            if all(v is None for v in validated_multi_profile.values()):
                log.debug(f"Layer {i} will use DENSE attention (all profiles None).")
                parameter_list_out.append(None)
            else:
                parameter_list_out.append(validated_multi_profile)
                log.info(f"Layer {i} NATTEN parameters: {validated_multi_profile}")

        return parameter_list_out

    return None


def generate_natten_metadata(
    token_shapes: list[tuple[int, int, int]],
    head_dim: int,
    num_layers: int,
    device: torch.device,
    dtype: torch.dtype,
    requires_grad: bool,
    natten_parameter_list: list | None = None,
) -> list | None:
    """
    Generates list of metadata required by Variable-Sized (variable-length) operations in NATTEN.
    Required when training with three_way attention and NATTEN (multi-dimensional / sparse
    attention).

    Args:
        token_shapes (list[tuple]): list of integer tuples corresponding to the
            post-tokenization/patchify token layout shapes in the packed sequence. Must strictly be
            integer tuples with the same profile (all 1D, 2D, or 3D). 1s will be automatically
            stripped (i.e. [(1, 8, 8), (1, 16, 16)] is interpreted as [(8, 8), (16, 16)]).

        head_dim (int): Attention head dimension (used to select NATTEN kernel configurations).

        num_layers (int): number of layers in the model. Just used to verify list length.

        device (torch.device): PyTorch device for offset tensors (should match QKV device).

        dtype (torch.dtype): Expected QKV dtype.

        requires_grad (bool): Determines whether backprop is expected, and sets up metadata for
            backward pass as well.

        natten_parameter_list (list | None): list of NATTEN parameters. Must be either None, or a
            list of mappings, one for each layer. Each list element must be either None,
            representing no sparsity / masking (full dense attention), or a mapping of NATTEN
            parameters in either integer or float format:
                - 'window_size_float' (required), 'stride_float', 'dilation_float'
                - 'window_size' (required), 'stride', 'dilation'

            Integer and float parameters cannot be used together in the same layer!
            Additionally, you can specify 'is_causal'.

            Examples:
            ```
            # 50 percent sparsity along each dimension in a 2-D token layout
            {'window_size_float': (0.5, 0.5)}  # valid

            # 50 percent sparsity along each dimension in a 2-D token layout
            # Maximum dilation along first dimension, no dilation along second dimension
            {'window_size_float': (0.5, 0.5), 'dilation_float': (1.0, 0.0)}  # valid

            # Fixed window size of 8x8, dilation of 2x1.

            {'window_size': (8, 8), 'dilation': (2, 1)}  # valid

            # Invalid:
            {'window_size_float': (0.5, 0.5), 'dilation': (2, 1)}
            ```

    Returns:
        natten_metadata_list (list | None): list of NATTEN varlen metadata, or Nones (dense layers).
            Each non-None element will be a dictionary containing final parameters, and varlen
            metadata (offset and size tensors, max lengths).
            NOTE: to avoid excessive recompilations in torch.compile, we must carefully index into
            this list during model.forward, and ideally using the iteration counter from the loop
            over layers (nn.ModuleList).
    """


    if token_shapes is None or len(token_shapes) < 1:
        raise ValueError("'token_shapes' is required for 'three_way' attention.")

    natten_metadata = None

    if natten_parameter_list is not None:
        natten_metadata = []
        if not isinstance(natten_parameter_list, list):
            raise ValueError(f"Argument 'natten_parameter_list' must be a list or None, got {natten_parameter_list=}.")

        if len(natten_parameter_list) != num_layers:
            raise ValueError(
                "Number of elements in 'natten_parameter_list' must match number of layers "
                f"in the model, got {num_layers=}, {len(natten_parameter_list)=}."
            )

        # We need to filter out 1s from shapes
        def filter_shape(shape: tuple) -> tuple:
            return tuple(x for x in shape if x > 1)

        # Infer token layout rank (dimensionality)
        num_dims = max([len(filter_shape(token_shape)) for token_shape in token_shapes])

        # Single pass: check if all layers support this dimensionality and if any need processing
        needs_processing = False
        for i, layer_parameters in enumerate(natten_parameter_list):
            if layer_parameters is None:
                continue

            # Fail fast if this dimensionality is not defined
            if num_dims not in layer_parameters:
                raise ValueError(
                    f"Layer {i}: batch has {num_dims}D data but parameters are not defined for {num_dims}D. "
                    f"Defined dimensionalities: {sorted(layer_parameters.keys())}"
                )

            # Check if this layer needs processing for this dimensionality
            if layer_parameters[num_dims] is not None:
                needs_processing = True

        # Early exit if all layers are dense for this dimensionality profile
        if not needs_processing:
            log.debug(f"All layers use DENSE attention for {num_dims}D data.")
            return None

        # We actually need to process, so validate and filter all shapes
        token_layout_list = []
        for shape in token_shapes:
            assert isinstance(shape, tuple)
            shape_filtered = filter_shape(shape)
            assert len(shape_filtered) == num_dims, (
                f"All data in batch must have same dimensionality, got {num_dims}D and {len(shape_filtered)}D"
            )
            token_layout_list.append(shape_filtered)

        log.debug(f"Batch dimensionality: {num_dims}D, token_layout_list={token_layout_list}")

        for i, layer_parameters in enumerate(natten_parameter_list):
            if layer_parameters is None:
                natten_metadata.append(None)
                continue

            # Get parameters for this dimensionality (already validated above)
            dim_params = layer_parameters[num_dims]

            if dim_params is None:
                # Dense attention for this dimensionality
                natten_metadata.append(None)
                continue

            # Use dim_params (parameters for this specific dimensionality)
            window_size_list = []
            stride_list = []
            dilation_list = []

            if "window_size_float" in dim_params:
                window_size_float = dim_params["window_size_float"]
                stride_float = dim_params["stride_float"]
                dilation_float = dim_params["dilation_float"]

                for token_layout in token_layout_list:
                    window_size_ = tuple(
                        min(x, max(2, int(k * float(x)))) for k, x in zip(window_size_float, token_layout)
                    )
                    stride_ = tuple(min(k, max(1, int(s * float(k)))) for s, k in zip(stride_float, window_size_))
                    max_dilation = tuple(x // k for k, x in zip(window_size_, token_layout))
                    dilation_ = tuple(min(m, max(1, int(d * float(m)))) for d, m in zip(dilation_float, max_dilation))

                    window_size_list.append(window_size_)
                    stride_list.append(stride_)
                    dilation_list.append(dilation_)

                assert len(window_size_list) == len(stride_list) == len(dilation_list) == len(token_layout_list)

                log.debug(f"Layer {i}: {window_size_list=}")
                log.debug(f"Layer {i}: {stride_list=}")
                log.debug(f"Layer {i}: {dilation_list=}")

            elif "window_size" in dim_params:
                window_size = dim_params["window_size"]
                stride = dim_params["stride"]
                dilation = dim_params["dilation"]

                window_size_list = [window_size for _ in range(len(token_layout_list))]
                stride_list = [stride for _ in range(len(token_layout_list))]
                dilation_list = [dilation for _ in range(len(token_layout_list))]
            else:
                raise ValueError(
                    "Sparse parameters for a layer must have key 'window_size' or 'window_size_float', "
                    f"got {dim_params=} in layer index {i}."
                )

            is_causal = dim_params["is_causal"]

            # Create varlen metadata for natten varlen/varsized ops

            # full size, that's why constant window sizes aren't allowed.

            natten_metadata.append(
                generate_multi_dim_varlen_parameters(
                    token_layout_list=token_layout_list,
                    head_dim=head_dim,
                    device=device,
                    dtype=dtype,
                    requires_grad=requires_grad,
                    #
                    window_size_list=window_size_list,
                    stride_list=stride_list,
                    dilation_list=dilation_list,
                    #
                    is_causal=is_causal,
                )
            )

    return natten_metadata


def generate_temporal_causal_natten_metadata(
    vision_token_shapes: list[tuple[int, int, int]],
    num_action_tokens_per_supertoken: int,
    num_layers: int,
    head_dim: int,
    device: torch.device,
    dtype: torch.dtype,
    requires_grad: bool,
) -> list:
    """Generate per-layer varlen metadata for temporal causal attention on supertokens.

    Each sample's generation tokens are laid out as T_i supertokens of size
    S_i = num_action_tokens_per_supertoken + H_i*W_i. Metadata encodes
    is_causal=(True, False): causal across T, full within S. All layers share
    the same metadata (full window, no spatial sparsity).

    Unlike generate_natten_metadata, this function does not apply filter_shape — (T, S) layouts
    are passed directly even when T=1. NATTEN handles T=1 causal masking correctly (trivially
    full attention within S).

    Args:
        vision_token_shapes: List of (T, H, W) per sample.
        num_action_tokens_per_supertoken: Number of action tokens prefixing each
            supertoken (0 when actions are not packed inline).
        num_layers: Number of transformer layers.
        head_dim: Attention head dimension.
        device: Target device.
        dtype: Target dtype.
        requires_grad: Whether metadata tensors require gradient.

    Returns:
        List of length num_layers, each element the same NATTEN varlen metadata dict.
    """
    # T=1: NATTEN requires kernel_size >= 2 and kernel_size <= token_layout, which are mutually
    # exclusive when T=1. Fall back to full dense attention (None) — a single supertoken trivially
    # attends to only itself, so temporal causality is already satisfied.
    # Mixed T=1/T>1 batches are rejected: NATTEN can't mask T=1 samples, and falling back to dense
    # attention for the whole batch would break temporal causality for the T>1 samples.
    # Ensure min_frames >= 5 in the dataloader so that T_latent = 1 + (N-1)//tcf >= 2 always.
    has_short = any(t < 2 for t, h, w in vision_token_shapes)
    if has_short:
        if not all(t < 2 for t, h, w in vision_token_shapes):
            raise ValueError(
                "Mixed T=1 and T>1 samples in causal training batch: NATTEN cannot apply "
                "causal masking when any sample has T=1 (kernel_size constraint), and falling "
                "back to dense attention would break temporal causality for T>1 samples. "
                "Ensure all samples have T_latent >= 2 (set min_frames >= 5 in the dataloader)."
            )
        return [None] * num_layers
    token_layout_list = [(t, num_action_tokens_per_supertoken + h * w) for t, h, w in vision_token_shapes]
    metadata = generate_multi_dim_varlen_parameters(
        token_layout_list=token_layout_list,
        head_dim=head_dim,
        device=device,
        dtype=dtype,
        requires_grad=requires_grad,
        is_causal=(True, False),
    )
    return [metadata] * num_layers


def joint_from_joint_sequence(
    packed_sequence: torch.Tensor,
    attn_modes: List[str],
    split_lens: List[int],
    sample_lens: List[int],
    packed_und_token_indexes: torch.Tensor,
    packed_gen_token_indexes: torch.Tensor,
    is_image_batch: bool = False,
    cp_world_size: int = 1,
    pad_for_cuda_graphs: bool = False,
) -> JointSequencePack:
    f"""
    Create a JointSequencePack from a packed sequence and metadata.
    This is in order to support the legacy joint flex-attention implementation.
    Differently from FactoredSequencePack, it has less strict requirements on the packed sequence.

    Args:
        packed_sequence (torch.Tensor): Tensor containing all tokens in the batch of sequences.
        attn_modes (List[str]): List of attention modes. Supports any sequence of {"causal", "full", "noise"}
        split_lens (List[int]): Length of each subsequence. len(split_lens) == len(attn_modes)
        sample_lens (List[int]): Length of each sequence. In this mode, sequences may have different number of splits,
                                 as opposed to FactoredSequencePack where each sequence has exactly two splits..
        packed_und_token_indexes (torch.Tensor): The indexes of the understanding tokens in the packed sequence.
        packed_gen_token_indexes (torch.Tensor): The indexes of the generating tokens in the packed sequence.
    """
    assert sum(sample_lens) == packed_sequence.shape[0], (
        "sum(sample_lens) must be equal to the length of the packed sequence"
    )
    meta = _init_sequence_pack(sample_lens, split_lens, attn_modes, packed_sequence.device)
    pack: JointSequencePack = {
        **meta,
        "max_num_tokens": sum(sample_lens),
        "packed_sequence": packed_sequence,
        "packed_und_token_indexes": packed_und_token_indexes,
        "packed_gen_token_indexes": packed_gen_token_indexes,
        "is_sharded": False,
    }
    return pack


def zeros_like(orig: FactoredSequencePack | JointSequencePack, shape: Tuple[int, ...] | torch.Size | None = None):
    """
    Create a new sequence pack with the same metadata as the original, but with all tokens set to zero.
    Args:
        orig (FactoredSequencePack | JointSequencePack): The original sequence pack to copy metadata from.
        shape (Tuple[int, ...] | torch.Size | None): The shape of the new sequence pack. If None, the shape will be the same as the original.
    """
    _ensure_core_metadata(orig)
    if "packed_sequence" in orig:
        if shape is None:
            shape_ = orig["packed_sequence"].shape
        else:
            assert len(shape) >= 1 and shape[0] == -1
            shape_ = (orig["packed_sequence"].shape[0],) + tuple(shape)[1:]
        packed_sequence = torch.zeros(
            shape_, device=orig["packed_sequence"].device, dtype=orig["packed_sequence"].dtype
        )  # [seq_len,D]
        return from_joint(packed_sequence, orig)
    else:
        if shape is None:
            shape_causal = orig["causal_seq"].shape
            shape_full = orig["full_only_seq"].shape
        else:
            assert len(shape) >= 1 and shape[0] == -1
            shape_causal = (orig["causal_seq"].shape[0],) + tuple(shape)[1:]
            shape_full = (orig["full_only_seq"].shape[0],) + tuple(shape)[1:]
        causal_seq = torch.zeros(
            shape_causal, device=orig["causal_seq"].device, dtype=orig["causal_seq"].dtype
        )  # [N_causal_tokens,D]
        full_only_seq = torch.zeros(
            shape_full, device=orig["full_only_seq"].device, dtype=orig["full_only_seq"].dtype
        )  # [N_full_tokens,D]
        return from_mode_splits(causal_seq, full_only_seq, orig)


def from_joint(packed_sequence: torch.Tensor, metadata_source: FactoredSequencePack | JointSequencePack):
    """
    Create a new sequence pack from a packed sequence and another sequence pack with the same metadata.
    Args:
        packed_sequence (torch.Tensor): Tensor containing all tokens in the batch of sequences.
        metadata_source (FactoredSequencePack | JointSequencePack): The metadata source to copy from.
    """
    _ensure_core_metadata(metadata_source)
    if "packed_sequence" in metadata_source:
        out = dict(metadata_source)
        out["packed_sequence"] = packed_sequence
        return out
    else:
        if metadata_source["is_sharded"]:
            # Use sharded sequences as is when is_sharded is True (used in Context Parallel)
            causal_seq = packed_sequence[: len(metadata_source["causal_seq"])]  # [N_causal_tokens,D]
            full_only_seq = packed_sequence[len(metadata_source["causal_seq"]) :]  # [N_full_tokens,D]
        else:
            causal_seq = packed_sequence[metadata_source["_causal_indices"]]  # [N_causal_tokens,D]
            full_only_seq = packed_sequence[metadata_source["_full_indices"]]  # [N_full_tokens,D]
            causal_seq, full_only_seq = _pad(
                causal_seq,
                full_only_seq,
                max_causal_len=metadata_source["causal_seq"].shape[0],
                max_full_len=metadata_source["full_only_seq"].shape[0],
            )

        return from_mode_splits(causal_seq, full_only_seq, metadata_source)


def from_mode_splits(
    causal_seq: torch.Tensor,
    full_only_seq: torch.Tensor,
    orig: FactoredSequencePack | JointSequencePack,
    is_sharded: bool | None = None,
):
    """
    Create a new sequence pack from two mode splits.
    Args:
        causal_seq (torch.Tensor): The causal sequence.
        full_only_seq (torch.Tensor): The full-only sequence.
        orig (FactoredSequencePack | JointSequencePack): The metadata source to copy from.
        is_sharded (bool | None): If True, create a local pack for context parallel.
                                  If None, inherits from orig.
    """
    _ensure_core_metadata(orig)
    if is_sharded is None:
        is_sharded = orig.get("is_sharded", False)

    if "packed_sequence" in orig:
        all_len = int(orig["_causal_indices"].shape[0] + orig["_full_indices"].shape[0])
        packed_sequence = causal_seq.new_zeros((all_len, *causal_seq.shape[1:]))  # [seq_len,D]
        packed_sequence[orig["_causal_indices"]] = causal_seq
        packed_sequence[orig["_full_indices"]] = full_only_seq
        return from_joint(packed_sequence, orig)
    else:
        out = dict(orig)
        out["causal_seq"] = causal_seq
        out["full_only_seq"] = full_only_seq
        out["is_sharded"] = is_sharded
        return out


def from_und_gen_splits(und_seq: torch.Tensor, gen_seq: torch.Tensor, orig: FactoredSequencePack | JointSequencePack):
    """
    Create a new sequence pack from two und/gen splits.
    Args:
        und_seq (torch.Tensor): The understanding sequence.
        gen_seq (torch.Tensor): The generating sequence.
        orig (FactoredSequencePack | JointSequencePack): The metadata source to copy from.
    """
    # If we have a joint pack (single packed_sequence), place by und/gen indexes.
    if "packed_sequence" in orig and "packed_und_token_indexes" in orig and "packed_gen_token_indexes" in orig:
        all_len = int(und_seq.shape[0] + gen_seq.shape[0])
        packed_sequence = und_seq.new_zeros((all_len, *und_seq.shape[1:]))  # [seq_len,D]
        packed_sequence[orig["packed_und_token_indexes"]] = und_seq
        packed_sequence[orig["packed_gen_token_indexes"]] = gen_seq
        return from_joint(packed_sequence, orig)
    # Otherwise, treat und/gen as mode splits (und == causal; gen == full).
    return from_mode_splits(und_seq, gen_seq, orig)


# ------------------------------------
# Getters and setters for SequencePack
# ------------------------------------
def get_und_seq(pack: SequencePack) -> torch.Tensor:
    """
    Get all understanding tokens in a sequence pack in a single tensor.

    Args:
        pack (FactoredSequencePack | JointSequencePack): The sequence pack to get the understanding sequence from.
    Returns:
        torch.Tensor: All understanding tokens concatenated over all sequences in the batch.
    """
    if "causal_seq" in pack:
        return pack["causal_seq"]
    if "packed_sequence" in pack and "packed_und_token_indexes" in pack:
        return pack["packed_sequence"][pack["packed_und_token_indexes"]]
    raise KeyError("Cannot derive und_seq from provided pack")


def set_und_seq(pack: SequencePack, value: torch.Tensor) -> None:
    """
    Override the understanding tokens in a sequence pack.
    The order of tokens passed in must correspond to the order of tokens returned by get_und_seq.

    Args:
        pack (FactoredSequencePack | JointSequencePack): The sequence pack to set the understanding sequence in.
        value (torch.Tensor): The understanding sequence to set.
    """
    if "packed_sequence" in pack and "packed_und_token_indexes" in pack:
        pack["packed_sequence"][pack["packed_und_token_indexes"]] = value
    elif "causal_seq" in pack:
        pack["causal_seq"] = value
    else:
        raise KeyError("Cannot set und_seq from provided pack")


def get_gen_seq(pack: SequencePack) -> torch.Tensor:
    """
    Get all generating tokens in a sequence pack in a single tensor.
    Args:
        pack (FactoredSequencePack | JointSequencePack): The sequence pack to get the generating sequence from.
    Returns:
        torch.Tensor: All generating tokens concatenated over all sequences in the batch.
    """
    if "full_only_seq" in pack:
        return pack["full_only_seq"]
    if "packed_sequence" in pack and "packed_gen_token_indexes" in pack:
        return pack["packed_sequence"][pack["packed_gen_token_indexes"]]
    raise KeyError("Cannot derive gen_seq from provided pack")


def set_gen_seq(pack: SequencePack, value: torch.Tensor) -> None:
    """
    Override the generating tokens in a sequence pack.
    The order of tokens passed in must correspond to the order of tokens returned by get_gen_seq.
    Args:
        pack (FactoredSequencePack | JointSequencePack): The sequence pack to set the generating sequence in.
        value (torch.Tensor): The generating sequence to set.
    """
    if "packed_sequence" in pack and "packed_gen_token_indexes" in pack:
        pack["packed_sequence"][pack["packed_gen_token_indexes"]] = value
    elif "full_only_seq" in pack:
        pack["full_only_seq"] = value
    else:
        raise KeyError("Cannot set gen_seq from provided pack")


def get_all_seq(pack: SequencePack) -> torch.Tensor:
    """
    Get all tokens in a sequence pack in a single tensor.
    Args:
        pack (FactoredSequencePack | JointSequencePack): The sequence pack to get the all sequence from.
    Returns:
        torch.Tensor: All tokens concatenated over all sequences in the batch.
    """
    if "all_seq" in pack:
        return pack["all_seq"]
    if "packed_sequence" in pack:
        return pack["packed_sequence"]
    if "causal_seq" in pack and "full_only_seq" in pack:
        _ensure_core_metadata(pack)
        if pack["is_sharded"]:
            assert False, "get_all_seq is not supported in context parallel sharded mode"
        else:
            out = pack["causal_seq"].new_zeros(
                int(pack["_causal_indices"].shape[0] + pack["_full_indices"].shape[0]), *pack["causal_seq"].shape[1:]
            )  # [seq_len,D]
            if pack["causal_seq"].shape[0] > 0:
                out[pack["_causal_indices"]] = pack["causal_seq"][: pack["_causal_indices"].shape[0]]
            if pack["full_only_seq"].shape[0] > 0:
                out[pack["_full_indices"]] = pack["full_only_seq"][: pack["_full_indices"].shape[0]]
        return out
    raise KeyError("Cannot derive all_seq from provided pack")


def set_all_seq(pack: SequencePack, value: torch.Tensor) -> None:
    """
    Override the all tokens in a sequence pack.
    The order of tokens passed in must correspond to the order of tokens returned by get_all_seq.
    Args:
        pack (FactoredSequencePack | JointSequencePack): The sequence pack to set the all sequence in.
        value (torch.Tensor): The all sequence to set.
    """
    if "packed_sequence" in pack:
        pack["packed_sequence"] = value
    elif "causal_seq" in pack and "full_only_seq" in pack:
        _ensure_core_metadata(pack)
        pack["causal_seq"][: pack["_causal_indices"].shape[0]] = value[pack["_causal_indices"]]
        pack["full_only_seq"][: pack["_full_indices"].shape[0]] = value[pack["_full_indices"]]
    else:
        pack["all_seq"] = value


def get_causal_seq(pack: SequencePack) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Get the causal sequence and its offsets in a sequence pack.
    Args:
        pack (FactoredSequencePack | JointSequencePack): The sequence pack to get the causal sequence from.
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The concatenated causal sub-sequences and the starting offset for each sub-sequence.
    """
    _ensure_core_metadata(pack)
    if "causal_seq" in pack:
        return pack["causal_seq"], pack["_causal_seq_offsets"]
    assert "packed_sequence" in pack
    return pack["packed_sequence"][pack["_causal_indices"]], pack["_causal_seq_offsets"]


def get_full_only_seq(pack: SequencePack) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Get the full-only sequence and its offsets in a sequence pack.
    Args:
        pack (FactoredSequencePack | JointSequencePack): The sequence pack to get the full-only sequence from.
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The concatenated full-only sub-sequences and the starting offset for each sub-sequence.
    """
    _ensure_core_metadata(pack)
    if "full_only_seq" in pack:
        return pack["full_only_seq"], pack["_full_only_seq_offsets"]
    assert "packed_sequence" in pack
    return pack["packed_sequence"][pack["_full_indices"]], pack["_full_only_seq_offsets"]


def get_device_and_dtype(pack: SequencePack) -> Tuple[torch.device, torch.dtype]:
    """
    Get the device and dtype of a sequence pack.
    Args:
        pack (FactoredSequencePack | JointSequencePack): The sequence pack to get the device and dtype from.
    Returns:
        Tuple[torch.device, torch.dtype]: The device and dtype of the sequence pack.
    """
    if "packed_sequence" in pack:
        return pack["packed_sequence"].device, pack["packed_sequence"].dtype
    if "causal_seq" in pack and "full_only_seq" in pack:
        return pack["causal_seq"].device, pack["causal_seq"].dtype
    raise KeyError("Cannot derive device and dtype from provided pack")


def build_sequence_plans_from_data_batch(
    data_batch: dict,
    input_video_key,
    input_image_key: str,
) -> list[SequencePlan]:
    """Build or retrieve sequence plans from a data batch dictionary.

    This function extracts sequence plans from the data batch if they exist,
    otherwise creates default SequencePlan objects for each sample
    in the batch.

    Args:
        data_batch: Dictionary containing the data batch from the dataloader.
            Expected keys include 'video' or other tensors to determine batch size.
            If 'sequence_plan' key exists, those plans are returned directly.

    Returns:
        List of SequencePlan objects, one per sample in the batch.
    """

    # For new modalities, please generate the sequence_plan in the dataset class!!!!

    # If sequence_plan already exists in data_batch, return it
    if "sequence_plan" in data_batch:
        return data_batch["sequence_plan"]

    assert "action" not in data_batch or data_batch["action"] is None, "Action data SHOULD have sequence_plans!"
    assert "sound" not in data_batch or data_batch["sound"] is None, "Sound data SHOULD have sequence_plans!"


    # Determine batch size from available tensors
    batch_size = 0
    for key in [input_video_key, input_image_key]:
        if key in data_batch:
            val = data_batch[key]
            if isinstance(val, torch.Tensor):
                batch_size = val.shape[0]
                break
            elif isinstance(val, list):
                batch_size = len(val)
                break

    if batch_size == 0:
        raise ValueError(
            f"Cannot determine batch size from data_batch. Expected {input_video_key}, {input_image_key}, or similar key."
        )

    # Build default SequencePlan objects
    return [
        SequencePlan(
            has_text=True,  # Has text prompt!
            has_vision=True,
            condition_frame_indexes_vision=[],  # No conditioning frames!
        )
        for _ in range(batch_size)
    ]


# ============================================================================
# Demo/Test function
# ============================================================================


def main():
    """Demonstrate sequence packing with sample text and images."""
    # Initialize tokenizer and add special tokens
    tokenizer = Qwen2Tokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    tokenizer, _ = add_special_tokens(tokenizer)

    # Define special tokens (Note: Qwen models don't have bos_token_id)
    special_tokens = {
        "eos_token_id": tokenizer.eos_token_id,
        "start_of_generation": tokenizer.convert_tokens_to_ids("<|vision_start|>"),
        "end_of_generation": tokenizer.convert_tokens_to_ids("<|vision_end|>"),
    }

    # Sample text inputs
    input_strings = ["Hello world", "How are you?", "I am fine"]

    # Tokenize input strings
    input_text_tokens = [tokenizer.encode(text, add_special_tokens=False) for text in input_strings]

    # Create sample images (in practice, these would be VAE latents)
    input_images = torch.stack([torch.randn(3, 1, 64, 64) for _ in range(3)])  # [B, C, T, H, W] format

    # Diffusion timesteps for each image
    input_timesteps = torch.tensor([0.0, 0.5, 0.9])

    # Create GenerationDataClean for images
    gen_data_clean_images = GenerationDataClean(
        batch_size=3,
        is_image_batch=True,
        raw_state_vision=input_images,
        x0_tokens_vision=torch.randn(3, 16, 8, 8),  # dummy tokenized latents
        raw_state_action=None,
    )

    # Create SequencePlan for each sample (all have text and vision)
    sequence_plans = [
        SequencePlan(
            has_text=True,
            has_vision=True,
            has_action=False,
            condition_frame_indexes_vision=[],
            condition_frame_indexes_action=[],
        )
        for _ in range(3)
    ]

    # Pack sequences
    packed_data = pack_input_sequence(
        sequence_plans=sequence_plans,
        input_text_indexes=input_text_tokens,
        gen_data_clean=gen_data_clean_images,
        input_timesteps=input_timesteps,
        special_tokens=special_tokens,
        include_end_of_generation_token=True,
    )

    # Display results (after finalize, fields are tensors)
    print(f"Packed sequence length: {packed_data.sequence_length}")
    assert isinstance(packed_data.text_ids, torch.Tensor)
    print(f"Packed text IDs shape: {packed_data.text_ids.shape}")
    if packed_data.vision:
        assert isinstance(packed_data.vision.sequence_indexes, torch.Tensor)
        print(f"VAE token indexes shape: {packed_data.vision.sequence_indexes.shape}")
    print(f"Packed position_ids: {packed_data.position_ids}")

    ##################
    ## Video data
    input_videos = torch.stack([torch.randn(3, 5, 64, 64) for _ in range(2)])  # [B, C, T, H, W] format

    # Diffusion timesteps for each video
    input_timesteps_video = torch.tensor([0.5, 0.9])

    # Create GenerationDataClean for videos
    gen_data_clean_videos = GenerationDataClean(
        batch_size=2,
        is_image_batch=False,
        raw_state_vision=input_videos,
        x0_tokens_vision=torch.randn(2, 16, 2, 8, 8),  # dummy tokenized latents
        raw_state_action=None,
    )

    # Create SequencePlan for video samples
    sequence_plans_video = [
        SequencePlan(
            has_text=True,
            has_vision=True,
            has_action=False,
            condition_frame_indexes_vision=[],
            condition_frame_indexes_action=[],
        )
        for _ in range(2)
    ]

    # Pack sequences
    packed_data = pack_input_sequence(
        sequence_plans=sequence_plans_video,
        input_text_indexes=input_text_tokens[0:2],
        gen_data_clean=gen_data_clean_videos,
        input_timesteps=input_timesteps_video,
        special_tokens=special_tokens,
        include_end_of_generation_token=True,
    )

    # Display results (after finalize, fields are tensors)
    print(f"Packed sequence length: {packed_data.sequence_length}")
    assert isinstance(packed_data.text_ids, torch.Tensor)
    print(f"Packed text IDs shape: {packed_data.text_ids.shape}")
    if packed_data.vision:
        assert isinstance(packed_data.vision.sequence_indexes, torch.Tensor)
        print(f"VAE token indexes shape: {packed_data.vision.sequence_indexes.shape}")
    print(f"Packed position_ids: {packed_data.position_ids}")


def get_und_position_ids(position_ids: torch.Tensor, meta: dict[str, Any]) -> torch.Tensor:
    """
    Get the understanding position ids in a sequence pack.
    Args:
        position_ids (torch.Tensor): The position ids. Shape (seq_len,) for 1D RoPE
            or (3, seq_len) for 3D mRoPE.
        meta (dict[str, Any]): The metadata.
    Returns:
        torch.Tensor: The understanding position ids.
    """
    assert not meta["is_sharded"], "get_und_position_ids is not supported in context parallel sharded mode"
    if position_ids.dim() == 2:
        # 3D mRoPE: position_ids is (3, seq_len)
        return position_ids[:, meta["_causal_indices"]]  # [3,N_causal_tokens]
    return position_ids[meta["_causal_indices"]]  # [N_causal_tokens]


def get_gen_position_ids(position_ids: torch.Tensor, meta: dict[str, Any]) -> torch.Tensor:
    """
    Get the generating position ids in a sequence pack.
    Args:
        position_ids (torch.Tensor): The position ids. Shape (seq_len,) for 1D RoPE
            or (3, seq_len) for 3D mRoPE.
        meta (dict[str, Any]): The metadata.
    Returns:
        torch.Tensor: The generating position ids.
    """
    assert not meta["is_sharded"], "get_gen_position_ids is not supported in context parallel sharded mode"
    if position_ids.dim() == 2:
        # 3D mRoPE: position_ids is (3, seq_len)
        return position_ids[:, meta["_full_indices"]]  # [3,N_full_tokens]
    return position_ids[meta["_full_indices"]]  # [N_full_tokens]


if __name__ == "__main__":
    main()
