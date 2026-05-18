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

"""Dense wrapper runtime for frozen tokenizer encode/decode."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from cosmos.model.tokenizer.models.dense_backends import (
    DenseResolvedBackend,
    DenseRuntimeBackend,
    resolve_dense_backend,
    run_batched_block_stack,
    run_varlen_block_stack,
)
from cosmos.model.tokenizer.models.modules.transformer.blocks import LearnedPositionEmbedder
from cosmos.model.tokenizer.models.sparse_autoencoder import AutoencoderKL, SparseTransformerBase


@dataclass(frozen=True)
class DenseTemporalChunkSpec:
    """Temporal chunk configuration for the dense runtime."""

    raw_frames: int
    patch_frames: int


@dataclass(frozen=True)
class DenseGridMetadata:
    """Precomputed dense-grid metadata shared across chunk executions."""

    batch_size: int
    temporal_patches: int
    height_patches: int
    width_patches: int
    learned_pe: torch.Tensor | None
    rope_freqs_cis: torch.Tensor | None
    cu_seqlens: torch.Tensor
    q_seqlen: list[int]
    max_seq_len: int


DenseGridMetadataKey = tuple[str, int, int, int, int, str, str]


class DenseDiagonalGaussianDistribution:
    """Diagonal Gaussian posterior for dense channels-last latent tensors."""

    def __init__(self, parameters: torch.Tensor, deterministic: bool = False) -> None:
        """Initialize the dense posterior from `[mean, logvar]` moments."""
        if parameters.ndim not in (4, 5):
            raise ValueError(
                "DenseDiagonalGaussianDistribution expects 4D/5D channels-last moments, "
                f"got shape {tuple(parameters.shape)}."
            )
        self.original_dtype = parameters.dtype
        self.parameters = parameters.to(torch.float32)
        self.mean, self.logvar = torch.chunk(self.parameters, 2, dim=-1)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)

        if self.deterministic:
            self.var = self.std = torch.zeros_like(
                self.mean,
                device=self.parameters.device,
                dtype=self.parameters.dtype,
            )

    def sample(self) -> torch.Tensor:
        """Sample a dense channels-last latent tensor."""
        sample = torch.randn_like(self.mean)
        return (self.mean + self.std * sample).to(self.original_dtype)

    def kl(self, other: "DenseDiagonalGaussianDistribution" | None = None) -> torch.Tensor:
        """Compute KL divergence per latent token, matching sparse scaling."""
        reduce_dims = (-1,)
        if self.deterministic:
            num_tokens = math.prod(self.mean.shape[:-1])
            return torch.zeros(num_tokens, device=self.parameters.device, dtype=self.parameters.dtype)
        if other is None:
            kl = 0.5 * torch.sum(torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar, dim=reduce_dims)
        else:
            kl = 0.5 * torch.sum(
                torch.pow(self.mean - other.mean, 2) / other.var
                + self.var / other.var
                - 1.0
                - self.logvar
                + other.logvar,
                dim=reduce_dims,
            )
        return kl.reshape(-1)


class DenseAutoencoderRuntime(nn.Module):
    """Dense frozen-runtime wrapper around an existing sparse autoencoder.

    The wrapper intentionally holds the original ``AutoencoderKL`` as a single
    registered submodule and only exposes compile-friendly dense orchestration.
    Backend math is added incrementally in follow-up changes.
    """

    autoencoder: AutoencoderKL
    backend: DenseRuntimeBackend
    _metadata_cache: dict[DenseGridMetadataKey, DenseGridMetadata]

    def __init__(
        self,
        autoencoder: AutoencoderKL,
        backend: DenseRuntimeBackend = "auto",
    ) -> None:
        """Initialize the dense runtime wrapper."""
        super().__init__()
        self.autoencoder = autoencoder
        self.backend = backend
        self._metadata_cache = {}
        self.cg_compiled = False

    @classmethod
    def from_autoencoder(
        cls,
        autoencoder: AutoencoderKL,
        backend: DenseRuntimeBackend = "auto",
    ) -> "DenseAutoencoderRuntime":
        """Build a dense runtime from a supported sparse autoencoder."""
        cls._validate_autoencoder(autoencoder)
        return cls(autoencoder=autoencoder, backend=backend)

    @staticmethod
    def _validate_autoencoder(autoencoder: AutoencoderKL) -> None:
        """Validate that the sparse autoencoder fits the dense-runtime V1 scope."""
        if not hasattr(autoencoder, "decoder"):
            raise ValueError("Dense runtime V1 requires use_decoder=True.")

        encoder = autoencoder.encoder
        decoder = autoencoder.decoder

        if encoder.concat_latent is not None:
            raise ValueError("Dense runtime V1 does not support concat_latent.")
        if autoencoder.use_dual_latent:
            raise ValueError("Dense runtime V1 does not support dual latent.")
        if decoder.multiscale is not None or decoder.multiscale_outputs is not None:
            raise ValueError("Dense runtime V1 does not support decoder multiscale outputs.")
        if any(getattr(block, "multiscale", None) is not None for block in encoder.blocks):
            raise ValueError("Dense runtime V1 does not support encoder multiscale blocks.")
        if any(getattr(block, "multiscale", None) is not None for block in decoder.blocks):
            raise ValueError("Dense runtime V1 does not support decoder multiscale blocks.")
        if encoder.pe_mode not in {"joint", "learned"}:
            raise ValueError(f"Dense runtime V1 currently requires encoder learned/joint PE, got {encoder.pe_mode}.")
        if decoder.pe_mode not in {"joint", "learned"}:
            raise ValueError(f"Dense runtime V1 currently requires decoder learned/joint PE, got {decoder.pe_mode}.")

    @property
    def patch_size(self) -> tuple[int, int, int]:
        """Return the tokenizer patch size."""
        patch_size = self.autoencoder.patch_size
        return int(patch_size[0]), int(patch_size[1]), int(patch_size[2])

    @property
    def patch_volume(self) -> int:
        """Return the tokenizer patch volume."""
        return math.prod(self.patch_size)

    @property
    def encoder_chunk_spec(self) -> DenseTemporalChunkSpec:
        """Return the fixed encoder chunk configuration used in eval mode."""
        raw_frames = int(self.autoencoder.num_sample_frames_batch_size)
        patch_frames = self._raw_frames_to_patch_frames(raw_frames)
        return DenseTemporalChunkSpec(raw_frames=raw_frames, patch_frames=patch_frames)

    @property
    def decoder_window_spec(self) -> DenseTemporalChunkSpec:
        """Return the decoder inference window configuration."""
        raw_frames = int(self.autoencoder.inference_num_sample_frames_batch_size)
        patch_frames = self._raw_frames_to_patch_frames(raw_frames)
        return DenseTemporalChunkSpec(raw_frames=raw_frames, patch_frames=patch_frames)

    @property
    def decoder_stride_spec(self) -> DenseTemporalChunkSpec:
        """Return the decoder inference stride configuration."""
        raw_frames = int(self.autoencoder.inference_num_sample_frames_stride)
        patch_frames = self._raw_frames_to_patch_frames(raw_frames)
        return DenseTemporalChunkSpec(raw_frames=raw_frames, patch_frames=patch_frames)

    @property
    def decoder_cache_spec(self) -> DenseTemporalChunkSpec:
        """Return the decoder inference cache configuration."""
        raw_frames = int(self.autoencoder.inference_kv_cache_size)
        patch_frames = self._raw_frames_to_patch_frames(raw_frames)
        return DenseTemporalChunkSpec(raw_frames=raw_frames, patch_frames=patch_frames)

    def resolve_backend(self, use_compile: bool = False) -> DenseResolvedBackend:
        """Resolve the backend for the current execution mode."""
        return resolve_dense_backend(self.backend, use_compile=use_compile)

    def clear_metadata_cache(self) -> None:
        """Drop cached dense-grid metadata."""
        self._metadata_cache.clear()

    def encode(self, dense_video: torch.Tensor, sample_posterior: bool = False) -> torch.Tensor:
        """Encode a dense video tensor into latent moments or posterior samples."""
        moments = self.encode_moments(dense_video)
        if not sample_posterior:
            return moments
        return self._sample_dense_posterior(moments)

    def encode_moments(
        self,
        video: torch.Tensor,
        chunk_raw_frames: int | None = None,
    ) -> torch.Tensor:
        """Encode a dense video tensor into `[B, T_p, H_p, W_p, 2C]` latent moments."""
        if video.ndim != 5:
            raise ValueError(f"Dense runtime expects 5D video tensor, got {video.ndim}D")
        if video.shape[4] != 3:
            raise ValueError(f"Dense runtime expects video tensor with 3 channels, got {video.shape[4]}")

        batch_size, raw_frames, height, width, _ = video.shape
        patch_time, patch_height, patch_width = self.patch_size
        if raw_frames % patch_time != 0:
            raise ValueError(
                f"Dense runtime requires frame count divisible by patch_size[0]={patch_time}, got {raw_frames}."
            )
        if height % patch_height != 0 or width % patch_width != 0:
            raise ValueError(
                "Dense runtime requires spatial dimensions divisible by patch size "
                f"{(patch_height, patch_width)}, got {(height, width)}."
            )

        del batch_size
        if chunk_raw_frames is None:
            chunk_raw_frames = self.encoder_chunk_spec.raw_frames
        if chunk_raw_frames <= 0:
            raise ValueError(f"chunk_raw_frames must be positive, got {chunk_raw_frames}.")
        if chunk_raw_frames % patch_time != 0:
            raise ValueError(
                f"chunk_raw_frames must be divisible by patch_size[0]={patch_time}, got {chunk_raw_frames}."
            )
        encoded_chunks: list[torch.Tensor] = []
        for start_frame in range(0, raw_frames, chunk_raw_frames):
            end_frame = min(start_frame + chunk_raw_frames, raw_frames)
            video_chunk = video[:, start_frame:end_frame]
            encoded_chunk = self._encode_video_chunk(video_chunk)
            encoded_chunks.append(encoded_chunk)
        return torch.cat(encoded_chunks, dim=1)

    def decode(
        self,
        dense_latent: torch.Tensor,
        chunk_raw_frames: int | None = None,
    ) -> torch.Tensor:
        """Decode a dense latent grid into a dense channels-last video tensor."""
        if self.decoder_cache_spec.patch_frames != 0:
            raise NotImplementedError("Dense runtime decoder V1 does not support KV cache.")

        latent = self._canonicalize_dense_latent(dense_latent)
        temporal_patches = latent.shape[1]
        if chunk_raw_frames is None:
            chunk_patch_frames = self.decoder_window_spec.patch_frames
        else:
            if chunk_raw_frames <= 0:
                raise ValueError(f"chunk_raw_frames must be positive, got {chunk_raw_frames}.")
            if chunk_raw_frames % self.patch_size[0] != 0:
                raise ValueError(
                    f"chunk_raw_frames must be divisible by patch_size[0]={self.patch_size[0]}, got {chunk_raw_frames}."
                )
            chunk_patch_frames = chunk_raw_frames // self.patch_size[0]
        decoded_chunks: list[torch.Tensor] = []
        for start_patch in range(0, temporal_patches, chunk_patch_frames):
            end_patch = min(start_patch + chunk_patch_frames, temporal_patches)
            latent_chunk = latent[:, start_patch:end_patch]
            decoded_chunks.append(self._decode_latent_chunk(latent_chunk))
        return torch.cat(decoded_chunks, dim=1)

    def _metadata_cache_key(
        self,
        module_name: str,
        batch_size: int,
        temporal_patches: int,
        height_patches: int,
        width_patches: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> DenseGridMetadataKey:
        """Build a stable metadata-cache key for one dense grid shape."""
        return (
            module_name,
            int(batch_size),
            int(temporal_patches),
            int(height_patches),
            int(width_patches),
            str(device),
            str(dtype),
        )

    def _raw_frames_to_patch_frames(self, raw_frames: int) -> int:
        """Convert raw video frames into temporal patch steps."""
        patch_time = self.patch_size[0]
        if raw_frames % patch_time != 0:
            raise ValueError(
                f"Dense runtime requires raw frame counts divisible by patch_size[0]={patch_time}, got {raw_frames}."
            )
        return raw_frames // patch_time

    def _canonicalize_dense_latent(self, dense_latent: torch.Tensor) -> torch.Tensor:
        """Normalize latent tensors to channels-last `[B, T_p, H_p, W_p, C]` format."""
        expected_channels = self.autoencoder.latent_channels
        patch_time = self.patch_size[0]
        if dense_latent.ndim == 5:
            channels_last_match = dense_latent.shape[-1] == expected_channels
            channels_first_match = dense_latent.shape[1] == expected_channels
            if channels_last_match and channels_first_match:
                raise ValueError(
                    "Dense runtime cannot infer 5D latent layout when both the channel-last and "
                    "channel-first dimensions match the expected channel count "
                    f"{expected_channels}; got shape {tuple(dense_latent.shape)}."
                )
            if channels_last_match:
                latent = dense_latent
            elif channels_first_match:
                latent = rearrange(dense_latent, "b c t h w -> b t h w c")
            else:
                raise ValueError(
                    "Dense runtime expects 5D latents in `[B, T, H, W, C]` or `[B, C, T, H, W]` format, "
                    f"got shape {tuple(dense_latent.shape)} with expected channels={expected_channels}."
                )
        elif dense_latent.ndim == 4:
            if patch_time != 1:
                raise ValueError(
                    "Dense runtime image latents are only supported when patch_size[0] == 1, "
                    f"got patch_size[0]={patch_time}."
                )
            channels_last_match = dense_latent.shape[-1] == expected_channels
            channels_first_match = dense_latent.shape[1] == expected_channels
            if channels_last_match and channels_first_match:
                raise ValueError(
                    "Dense runtime cannot infer 4D latent layout when both the channel-last and "
                    "channel-first dimensions match the expected channel count "
                    f"{expected_channels}; got shape {tuple(dense_latent.shape)}."
                )
            if channels_last_match:
                latent = dense_latent.unsqueeze(1)
            elif channels_first_match:
                latent = rearrange(dense_latent, "b c h w -> b 1 h w c")
            else:
                raise ValueError(
                    "Dense runtime expects 4D latents in `[B, H, W, C]` or `[B, C, H, W]` format, "
                    f"got shape {tuple(dense_latent.shape)} with expected channels={expected_channels}."
                )
        else:
            raise ValueError(
                "Dense runtime expects latent inputs with 4 or 5 dimensions, "
                f"got tensor with shape {tuple(dense_latent.shape)}."
            )
        return latent.contiguous()

    def _encode_video_chunk(self, dense_video_chunk: torch.Tensor) -> torch.Tensor:
        """Encode one dense video chunk into projected latent moments."""
        batch_size, raw_frames, height, width, _ = dense_video_chunk.shape
        patch_time, patch_height, patch_width = self.patch_size
        temporal_patches = raw_frames // patch_time
        height_patches = height // patch_height
        width_patches = width // patch_width

        patch_feats = self._patchify_dense_video(dense_video_chunk)
        metadata = self._get_or_build_grid_metadata(
            module_name="encoder",
            module=self.autoencoder.encoder,
            batch_size=batch_size,
            temporal_patches=temporal_patches,
            height_patches=height_patches,
            width_patches=width_patches,
            device=patch_feats.device,
            dtype=self.autoencoder.encoder.input_layer.weight.dtype,
        )
        moments = self._encode_chunk_core(
            patch_feats,
            learned_pe=metadata.learned_pe,
            rope_freqs_cis=metadata.rope_freqs_cis,
            q_seqlen=metadata.q_seqlen,
            cu_seqlens_q=metadata.cu_seqlens,
            max_q_seqlen=metadata.max_seq_len,
        )
        if self.cg_compiled:
            moments = moments.clone()
        return moments.reshape(batch_size, temporal_patches, height_patches, width_patches, -1)

    def _encode_chunk_core(
        self,
        patch_feats: torch.Tensor,
        learned_pe: torch.Tensor | None,
        rope_freqs_cis: torch.Tensor | None,
        q_seqlen: list[int] | None = None,
        cu_seqlens_q: torch.Tensor | None = None,
        max_q_seqlen: int | None = None,
    ) -> torch.Tensor:
        """Encode one dense `[B, S, patch_dim]` chunk into projected latent moments."""
        encoder = self.autoencoder.encoder
        input_dtype = encoder.input_layer.weight.dtype
        if patch_feats.dtype != input_dtype:
            patch_feats = patch_feats.to(input_dtype)
        feats = F.linear(patch_feats, encoder.input_layer.weight, encoder.input_layer.bias)
        if learned_pe is not None:
            feats = feats + learned_pe

        block_param = next(encoder.blocks.parameters(), None)
        block_dtype = block_param.dtype if block_param is not None else feats.dtype
        if feats.dtype != block_dtype:
            feats = feats.to(block_dtype)

        feats = self._run_block_stack(
            blocks=encoder.blocks,
            feats=feats,
            q_seqlen=q_seqlen,
            cu_seqlens_q=cu_seqlens_q,
            max_q_seqlen=max_q_seqlen,
            rope_freqs_cis=rope_freqs_cis,
        )
        feats = encoder.post_layernorm(feats)
        return F.linear(feats, self.autoencoder.proj.weight, self.autoencoder.proj.bias)

    def _patchify_dense_video(self, dense_video: torch.Tensor) -> torch.Tensor:
        """Patchify a dense channels-last video chunk into `[B, S, patch_dim]`."""
        batch_size, raw_frames, height, width, channels = dense_video.shape
        patch_time, patch_height, patch_width = self.patch_size
        temporal_patches = raw_frames // patch_time
        height_patches = height // patch_height
        width_patches = width // patch_width
        return rearrange(
            dense_video,
            "b (nt pt) (nh ph) (nw pw) c -> b (nt nh nw) (pt ph pw c)",
            b=batch_size,
            nt=temporal_patches,
            nh=height_patches,
            nw=width_patches,
            pt=patch_time,
            ph=patch_height,
            pw=patch_width,
            c=channels,
        )

    def _decode_latent_chunk(self, dense_latent_chunk: torch.Tensor) -> torch.Tensor:
        """Decode one dense latent chunk into a dense channels-last video chunk."""
        batch_size, temporal_patches, height_patches, width_patches, _ = dense_latent_chunk.shape
        feats = dense_latent_chunk.reshape(batch_size, temporal_patches * height_patches * width_patches, -1)
        metadata = self._get_or_build_grid_metadata(
            module_name="decoder",
            module=self.autoencoder.decoder,
            batch_size=batch_size,
            temporal_patches=temporal_patches,
            height_patches=height_patches,
            width_patches=width_patches,
            device=feats.device,
            dtype=self.autoencoder.decoder.input_layer.weight.dtype,
        )
        patch_feats = self._decode_chunk_core(
            feats,
            learned_pe=metadata.learned_pe,
            rope_freqs_cis=metadata.rope_freqs_cis,
            q_seqlen=metadata.q_seqlen,
            cu_seqlens_q=metadata.cu_seqlens,
            max_q_seqlen=metadata.max_seq_len,
        )
        return self._unpatchify_dense_video_chunk(
            patch_feats,
            batch_size=batch_size,
            temporal_patches=temporal_patches,
            height_patches=height_patches,
            width_patches=width_patches,
        )

    def _decode_chunk_core(
        self,
        feats: torch.Tensor,
        learned_pe: torch.Tensor | None,
        rope_freqs_cis: torch.Tensor | None,
        q_seqlen: list[int] | None = None,
        cu_seqlens_q: torch.Tensor | None = None,
        max_q_seqlen: int | None = None,
    ) -> torch.Tensor:
        """Decode one dense `[B, S, latent_dim]` chunk into patch-space features."""
        decoder = self.autoencoder.decoder
        input_dtype = decoder.input_layer.weight.dtype
        if feats.dtype != input_dtype:
            feats = feats.to(input_dtype)

        feats = F.linear(feats, decoder.input_layer.weight, decoder.input_layer.bias)
        if learned_pe is not None:
            feats = feats + learned_pe

        block_param = next(decoder.blocks.parameters(), None)
        block_dtype = block_param.dtype if block_param is not None else feats.dtype
        if feats.dtype != block_dtype:
            feats = feats.to(block_dtype)

        feats = self._run_block_stack(
            blocks=decoder.blocks,
            feats=feats,
            q_seqlen=q_seqlen,
            cu_seqlens_q=cu_seqlens_q,
            max_q_seqlen=max_q_seqlen,
            rope_freqs_cis=rope_freqs_cis,
        )
        feats = decoder.out_norm(feats)
        return F.linear(feats, decoder.out_layer.weight, decoder.out_layer.bias)

    def _unpatchify_dense_video_chunk(
        self,
        patch_feats: torch.Tensor,
        batch_size: int,
        temporal_patches: int,
        height_patches: int,
        width_patches: int,
    ) -> torch.Tensor:
        """Unpatchify dense decoder outputs into channels-last video chunks."""
        patch_time, patch_height, patch_width = self.patch_size
        patch_volume = patch_time * patch_height * patch_width
        if self.autoencoder.out_channels % patch_volume != 0:
            raise ValueError(
                f"Autoencoder out_channels={self.autoencoder.out_channels} is not divisible by patch volume {patch_volume}."
            )
        output_channels = self.autoencoder.out_channels // patch_volume
        return rearrange(
            patch_feats,
            "b (nt nh nw) (pt ph pw c) -> b (nt pt) (nh ph) (nw pw) c",
            b=batch_size,
            nt=temporal_patches,
            nh=height_patches,
            nw=width_patches,
            pt=patch_time,
            ph=patch_height,
            pw=patch_width,
            c=output_channels,
        )

    def _run_block_stack(
        self,
        blocks: nn.ModuleList,
        feats: torch.Tensor,
        q_seqlen: list[int] | None,
        cu_seqlens_q: torch.Tensor | None,
        max_q_seqlen: int | None,
        rope_freqs_cis: torch.Tensor | None,
    ) -> torch.Tensor:
        """Run one backend-specific transformer block stack over `[B, S, D]` features."""
        backend = self.resolve_backend(use_compile=torch.compiler.is_compiling())
        if backend == "varlen":
            if q_seqlen is None or cu_seqlens_q is None or max_q_seqlen is None:
                raise ValueError("Varlen dense backend requires q_seqlen, cu_seqlens_q, and max_q_seqlen metadata.")
            return run_varlen_block_stack(
                blocks,
                feats,
                q_seqlen=q_seqlen,
                cu_seqlens_q=cu_seqlens_q,
                max_q_seqlen=max_q_seqlen,
                q_freqs_cis=rope_freqs_cis,
            )
        if backend == "batched":
            return run_batched_block_stack(
                blocks,
                feats,
                q_freqs_cis=rope_freqs_cis,
            )
        raise ValueError(f"Unsupported dense runtime backend: {backend}")

    def _get_or_build_grid_metadata(
        self,
        module_name: str,
        module: SparseTransformerBase,
        batch_size: int,
        temporal_patches: int,
        height_patches: int,
        width_patches: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> DenseGridMetadata:
        """Fetch or create dense-grid metadata for one uniform chunk shape."""
        key = self._metadata_cache_key(
            module_name,
            batch_size,
            temporal_patches,
            height_patches,
            width_patches,
            device,
            dtype,
        )
        cached = self._metadata_cache.get(key)
        if cached is not None:
            return cached

        metadata = self._build_grid_metadata(
            module=module,
            batch_size=batch_size,
            temporal_patches=temporal_patches,
            height_patches=height_patches,
            width_patches=width_patches,
            device=device,
        )
        self._metadata_cache[key] = metadata
        return metadata

    def _build_grid_metadata(
        self,
        module: SparseTransformerBase,
        batch_size: int,
        temporal_patches: int,
        height_patches: int,
        width_patches: int,
        device: torch.device,
    ) -> DenseGridMetadata:
        """Precompute dense-grid metadata for one uniform chunk shape."""
        seq_len = temporal_patches * height_patches * width_patches
        q_seqlen = [seq_len] * batch_size
        cu_seqlens = torch.arange(
            0,
            (batch_size + 1) * seq_len,
            seq_len,
            dtype=torch.int32,
            device=device,
        )
        learned_pe = self._build_learned_position_embeddings(
            module,
            temporal_patches=temporal_patches,
            height_patches=height_patches,
            width_patches=width_patches,
            device=device,
        )
        rope_freqs_cis = self._build_rope_freqs_cis(
            module,
            batch_size=batch_size,
            temporal_patches=temporal_patches,
            height_patches=height_patches,
            width_patches=width_patches,
            device=device,
        )
        return DenseGridMetadata(
            batch_size=batch_size,
            temporal_patches=temporal_patches,
            height_patches=height_patches,
            width_patches=width_patches,
            learned_pe=learned_pe,
            rope_freqs_cis=rope_freqs_cis,
            cu_seqlens=cu_seqlens,
            q_seqlen=q_seqlen,
            max_seq_len=seq_len,
        )

    def _build_learned_position_embeddings(
        self,
        module: SparseTransformerBase,
        temporal_patches: int,
        height_patches: int,
        width_patches: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        """Build broadcastable learned spatial embeddings for one uniform chunk."""
        if module.pe_mode not in {"joint", "learned"}:
            return None
        if not isinstance(module.pos_embedder, LearnedPositionEmbedder):
            raise ValueError(
                "Dense runtime V1 expects LearnedPositionEmbedder for learned/joint PE, "
                f"got {type(module.pos_embedder).__name__}."
            )

        pos_embedder = module.pos_embedder
        positional_embeddings = pos_embedder.position_embedding.weight.reshape(
            pos_embedder.position_embedding_size,
            pos_embedder.position_embedding_size,
            -1,
        )
        spatial_embeddings = pos_embedder._get_interpolated_position_embedding(
            positional_embeddings,
            target_height=height_patches,
            target_width=width_patches,
            target_device=device,
        ).to(dtype=positional_embeddings.dtype)
        spatial_flat = spatial_embeddings.reshape(height_patches * width_patches, -1)
        temporal_flat = spatial_flat.repeat(temporal_patches, 1)
        return temporal_flat.unsqueeze(0)

    def _build_rope_freqs_cis(
        self,
        module: SparseTransformerBase,
        batch_size: int,
        temporal_patches: int,
        height_patches: int,
        width_patches: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        """Build RoPE frequencies for one regular dense patch grid."""
        blocks_with_rope = [block for block in module.blocks if getattr(block.attn, "use_rope", False)]
        if not blocks_with_rope:
            return None

        rope_configs = {
            (
                block.attn.rope.head_dim,
                block.attn.rope.pos_cls_token,
            )
            for block in blocks_with_rope
        }
        if len(rope_configs) != 1:
            raise ValueError("Dense runtime V1 requires uniform RoPE configuration across blocks.")

        positions = self._build_regular_patch_positions(
            temporal_patches=temporal_patches,
            height_patches=height_patches,
            width_patches=width_patches,
            device=device,
        )
        if batch_size > 1:
            positions = positions.unsqueeze(0).expand(batch_size, -1, -1).reshape(batch_size * positions.shape[0], -1)
        return blocks_with_rope[0].attn.rope.compute_freqs_cis(positions, has_special_tokens=False)

    def _build_regular_patch_positions(
        self,
        temporal_patches: int,
        height_patches: int,
        width_patches: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Build regular `[S, 4]` patch coordinates in `(t, h, w, z)` order."""
        return (
            torch.stack(
                torch.meshgrid(
                    torch.arange(temporal_patches, device=device),
                    torch.arange(height_patches, device=device),
                    torch.arange(width_patches, device=device),
                    torch.arange(1, device=device),
                    indexing="ij",
                ),
                dim=-1,
            )
            .reshape(-1, 4)
            .to(dtype=torch.int32)
        )

    def _sample_dense_posterior(self, moments: torch.Tensor) -> torch.Tensor:
        """Sample the dense latent posterior from `[mean, logvar]` moments."""
        original_dtype = moments.dtype
        mean, logvar = torch.chunk(moments.to(torch.float32), 2, dim=-1)
        std = torch.exp(0.5 * logvar)
        sample = mean + std * torch.randn_like(std)
        return sample.to(original_dtype)
