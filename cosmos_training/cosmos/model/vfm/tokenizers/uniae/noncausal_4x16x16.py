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

"""UniAE S1 tokenizer wrapper for diffusion training (4x16x16 compression).

Wraps the UniAE sparse autoencoder with DenseAutoencoderRuntime (SDPA compiled)
to provide a VideoTokenizerInterface compatible with diffusion model training.

Usage:
    from cosmos.model.vfm.tokenizers.uniae.noncausal_4x16x16 import UniAEVAE

    vae = UniAEVAE(
        vae_pth="s3://nv-00-10206-checkpoint/pretrained/tokenizers/video/cosmos/...",
        object_store_credential_path_pretrained="credentials/gcp_checkpoint.secret",
    )
    latents = vae.encode(video)   # [B, 3, T, H, W] -> [B, 48, T//4, H//16, W//16]
    recon = vae.decode(latents)   # [B, 48, T//4, H//16, W//16] -> [B, 3, T, H, W]
"""

import torch

from cosmos.utils import log
from cosmos.utils.distributed import get_rank, sync_model_states
from cosmos.utils.easy_io import easy_io
from cosmos.model.tokenizer.models.dense_runtime import DenseAutoencoderRuntime
from cosmos.model.tokenizer.models.sparse_autoencoder import AutoencoderKL
from cosmos.model.vfm.tokenizers.interface import VideoTokenizerInterface

# S1 architecture config (avoids importing configs/base which pulls in loss deps)
_S1_ARCH = dict(
    patch_size=(4, 16, 16),
    in_channels=3072,
    out_channels=3072,
    encoder_model_channels=1152,
    encoder_num_blocks=27,
    encoder_num_heads=16,
    encoder_mlp_channels=4304,
    encoder_attn_mode="full",
    encoder_window_size=None,
    encoder_pe_mode="joint",
    encoder_qk_rms_norm=False,
    encoder_use_bias=True,
    encoder_use_rms_norm=False,
    decoder_model_channels=1152,
    decoder_num_blocks=27,
    decoder_num_heads=16,
    decoder_mlp_channels=4304,
    decoder_attn_mode="full",
    decoder_window_size=None,
    decoder_pe_mode="joint",
    decoder_qk_rms_norm=True,
    decoder_use_bias=False,
    decoder_use_rms_norm=True,
    use_decoder=True,
    quantizer_type="rq",
    quantizer_codebook_size=65536,
    quantizer_num_codebooks=1,
    quantizer_chunk_size=1,
    use_vf_loss=False,
    freeze_encoder=False,
    pretrained_model_name="google/siglip2-so400m-patch16-naflex",
    concat_latent=None,
    random_num_sample_frames_batch_sizes=[8, 12, 16, 20, 24],
    inference_num_sample_frames_batch_size=16,
    inference_num_sample_frames_stride=16,
    inference_kv_cache_size=0,
    use_quantizer=False,
    use_dual_latent=False,
    use_text_alignment=True,
    use_post_text_alignment=True,
)


class UniAEVAE:
    """UniAE S1 VAE wrapper for diffusion training.

    Loads the UniAE sparse autoencoder checkpoint, wraps it with
    DenseAutoencoderRuntime (SDPA backend for compile-friendly inference),
    and provides encode/decode in the standard [B, C, T, H, W] format.
    """

    def __init__(
        self,
        z_dim: int = 48,
        vae_pth: str = "",
        object_store_credential_path_pretrained: str = "",
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
        backend: str = "sdpa",
    ):
        self.dtype = dtype
        self.device = device
        self.z_dim = z_dim
        self._spatial_compression_factor = 16
        self._temporal_compression_factor = 4

        # make compatible with meta device
        autoencoder = AutoencoderKL(
            **_S1_ARCH,
            latent_channels=z_dim,
            quantizer_feature_dim=z_dim,
        )

        autoencoder.eval()
        autoencoder.to(device=device, dtype=dtype)

        # Load checkpoint
        if vae_pth:
            self._load_checkpoint(autoencoder, vae_pth, object_store_credential_path_pretrained, device)

        # Wrap with dense runtime for fast inference
        self.dense_runtime = DenseAutoencoderRuntime.from_autoencoder(autoencoder, backend=backend)
        self.dense_runtime.eval()

        # Freeze all parameters
        for param in self.dense_runtime.parameters():
            param.requires_grad = False

        log.info(
            f"UniAE S1 loaded: {self.count_param() / 1e6:.1f}M params, "
            f"backend={backend}, dtype={dtype}, device={device}"
        )

    def _load_checkpoint(self, model, pretrained_path, credential_path, device):
        """Load checkpoint from local path or S3."""
        if get_rank() == 0:
            if pretrained_path.startswith("s3://"):
                backend_args = {
                    "backend": "s3",
                    "s3_credential_path": credential_path,
                }
            else:
                backend_args = None

            ckpt = easy_io.load(
                pretrained_path,
                backend_args=backend_args,
                map_location=device,
            )

            # Handle different checkpoint formats
            if isinstance(ckpt, dict):
                if "model" in ckpt:
                    state_dict = ckpt["model"]
                elif "state_dict" in ckpt:
                    state_dict = ckpt["state_dict"]
                elif "network" in ckpt:
                    state_dict = ckpt["network"]
                else:
                    state_dict = ckpt
            else:
                state_dict = ckpt

            # Strip common prefixes
            cleaned = {}
            for k, v in state_dict.items():
                for prefix in ["network.", "module.", "model."]:
                    if k.startswith(prefix):
                        k = k[len(prefix) :]
                cleaned[k] = v

            missing, unexpected = model.load_state_dict(cleaned, strict=False)
            if missing:
                log.warning(f"Missing keys: {len(missing)} (e.g., {missing[:3]})")
            if unexpected:
                log.warning(f"Unexpected keys: {len(unexpected)} (e.g., {unexpected[:3]})")
            log.info(f"Loaded checkpoint from {pretrained_path}")
        else:
            model.to_empty(device=device)

        sync_model_states(model)

    def count_param(self) -> int:
        return sum(p.numel() for p in self.dense_runtime.parameters())

    @torch.inference_mode()
    def encode(self, video: torch.Tensor) -> torch.Tensor:
        """Encode image or video to latent space.

        For images (T=1 or 4D input), the input is repeated to 4 frames
        since the non-causal tokenizer requires a full temporal patch.

        Args:
            video: [B, 3, T, H, W] or [B, 3, H, W] (image) in range [-1, 1]

        Returns:
            latent: [B, z_dim, T//4, H//16, W//16]
                    For single-image input, T//4 = 1.
        """
        # Handle image input: [B, C, H, W] -> [B, C, 4, H, W]
        is_image = video.ndim == 4
        if is_image:
            video = video.unsqueeze(2)
            video = torch.nn.functional.pad(
                video, (0, 0, 0, 0, 0, self._temporal_compression_factor - 1), mode="constant", value=0.0
            )

        B, C, T, H, W = video.shape
        tc = self._temporal_compression_factor

        # For non-causal tokenizer, repeat last frame to fill last temporal patch
        if T % tc != 0:
            pad_t = tc - T % tc
            last_frame = video[:, :, -1:].expand(-1, -1, pad_t, -1, -1)
            video = torch.cat([video, last_frame], dim=2)
            T = T + pad_t

        # Convert to channels-last [B, T, H, W, C] for dense runtime
        video_cl = video.permute(0, 2, 3, 4, 1).contiguous().to(dtype=self.dtype)

        # Encode: returns [B, T_p, H_p, W_p, 2*z_dim] moments
        moments = self.dense_runtime.encode(video_cl, sample_posterior=False)

        # Take mean (first half of channels) for deterministic encoding
        mean, logvar = moments.chunk(2, dim=-1)

        # Convert to [B, z_dim, T_p, H_p, W_p]
        return mean.permute(0, 4, 1, 2, 3).contiguous()

    @torch.inference_mode()
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent to image or video.

        Args:
            latent: [B, z_dim, T_p, H_p, W_p]

        Returns:
            video: [B, 3, T, H, W] in range [-1, 1]
        """
        # Convert to channels-last [B, T_p, H_p, W_p, z_dim]
        latent_cl = latent.permute(0, 2, 3, 4, 1).contiguous().to(dtype=self.dtype)

        # Decode: returns [B, T, H, W, C] channels-last
        decoded = self.dense_runtime.decode(latent_cl)

        # Convert to [B, C, T, H, W] and clamp
        video = decoded.permute(0, 4, 1, 2, 3).contiguous()
        return video.clamp(-1, 1).float()

    def get_latent_num_frames(self, num_pixel_frames: int) -> int:
        return num_pixel_frames // self._temporal_compression_factor

    def get_pixel_num_frames(self, num_latent_frames: int) -> int:
        return num_latent_frames * self._temporal_compression_factor


class UniAEVAEInterface(VideoTokenizerInterface):
    """Full VideoTokenizerInterface wrapper for diffusion training config integration."""

    def __init__(
        self,
        bucket_name: str = "",
        object_store_credential_path_pretrained: str = "",
        vae_path: str = "",
        encode_chunk_frames: int = 16,
        spatial_compression_factor: int = 16,
        temporal_compression_factor: int = 4,
    ):
        super().__init__(object_store_credential_path_pretrained)
        self._spatial_compression_factor = spatial_compression_factor
        self._temporal_compression_factor = temporal_compression_factor
        self.encode_chunk_frames = encode_chunk_frames
        self.use_streaming_encode = False

        vae_full_path = vae_path
        if bucket_name and not vae_path.startswith("s3://"):
            vae_full_path = f"s3://{bucket_name}/{vae_path}"

        self.vae = UniAEVAE(
            vae_pth=vae_full_path,
            object_store_credential_path_pretrained=object_store_credential_path_pretrained,
        )

    def reset_dtype(self):
        pass

    def encode(self, state: torch.Tensor) -> torch.Tensor:
        return self.vae.encode(state)

    def compile_encode_for_cudagraphs(
        self,
        *,
        mode: str = "reduce-overhead",
        fullgraph: bool = False,
        dynamic: bool = False,
        backend: str = "inductor",
    ) -> None:
        """Compile the encode function for CUDA graphs."""
        compile_kwargs = dict(mode=mode, fullgraph=fullgraph, dynamic=dynamic, backend=backend)
        if backend == "cudagraphs":
            compile_kwargs.pop("mode", None)
        if backend == "cudagraphs" or compile_kwargs.get("mode", None) == "reduce-overhead":
            self.vae.dense_runtime.cg_compiled = True

        self.vae.dense_runtime._encode_chunk_core = torch.compile(
            self.vae.dense_runtime._encode_chunk_core, **compile_kwargs
        )

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.vae.decode(latent)

    def get_latent_num_frames(self, num_pixel_frames: int) -> int:
        return self.vae.get_latent_num_frames(num_pixel_frames)

    def get_pixel_num_frames(self, num_latent_frames: int) -> int:
        return self.vae.get_pixel_num_frames(num_latent_frames)

    @property
    def spatial_compression_factor(self):
        return self._spatial_compression_factor

    @property
    def temporal_compression_factor(self):
        return self._temporal_compression_factor

    @property
    def spatial_resolution(self):
        return 512

    @property
    def pixel_chunk_duration(self):
        return self.encode_chunk_frames

    @property
    def latent_chunk_duration(self):
        return self.encode_chunk_frames // self._temporal_compression_factor

    @property
    def latent_ch(self) -> int:
        return 48

    @property
    def is_causal(self):
        return False
