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

"""
AVAE (Audio Variational AutoEncoder) Tokenizer for Imaginaire4
ported from https://gitlab-master.nvidia.com/ADLR/bigvgan
commit hash: 80fbd8cfecb1867cc864e6d4fe0a474d8403a474
"""

import torch

from cosmos.utils.easy_io import easy_io
from cosmos.model.vfm.tokenizers.audio.avae_utils.env import AttrDict
from cosmos.model.vfm.tokenizers.audio.avae_utils.models import load_generator
from cosmos.model.vfm.tokenizers.interface import AudioTokenizerInterface


def _load_avae_model(
    pretrained_path: str | None = None,
    config_path: str | None = None,
    sample_rate: int = 44100,
    audio_channels: int = 2,
    io_channels: int = 64,
    hop_size: int = 2048,
    object_store_credential_path_pretrained: str = "",
    device: str = "cuda",
) -> tuple:
    """
    Load AVAE model with default configuration.

    Args:
        pretrained_path: Path to checkpoint (S3 or local)
        config_path: Path to config JSON/YAML (S3 or local). If None, uses default config.
        sample_rate: Audio sample rate (Hz)
        audio_channels: Number of audio channels
        io_channels: Number of latent channels
        hop_size: Temporal downsampling factor
        object_store_credential_path_pretrained: S3 credentials path
        device: Device to load model on

    Returns:
        Tuple of (model, config_dict) where config_dict contains the loaded config
    """
    config_dict = None
    # Try to load config from file (checkpoint path with .json extension), otherwise use default
    config_path_derived = None
    if pretrained_path:
        # Derive config path by replacing checkpoint extension with .json
        for ext in [".ckpt", ".pth", ".pt"]:
            if pretrained_path.endswith(ext):
                config_path_derived = pretrained_path.replace(ext, ".json")
                break

    # Use explicit config_path if provided, otherwise use derived path
    config_path_to_use = config_path if (config_path is not None and config_path != "") else config_path_derived

    if config_path_to_use:
        try:
            if config_path_to_use.startswith("s3://"):
                backend_args = {
                    "backend": "s3",
                    "s3_credential_path": object_store_credential_path_pretrained,
                    "path_mapping": None,
                }
                config_dict = easy_io.load(config_path_to_use, backend_args=backend_args)
            else:
                config_dict = easy_io.load(config_path_to_use)
            config = AttrDict(config_dict)
            print(f"Loaded AVAE config from {config_path_to_use}")
        except Exception as e:
            print(f"Could not load config from {config_path_to_use}: {e}, using default config")
            config_path_to_use = None
            config_dict = None

    if not config_path_to_use:
        # Build default configuration matching avae_latent64_2048x_44kstereo_eqvaemp3_finetune
        config = AttrDict(
            {
                "model_type": "autoencoder_v2",
                "sampling_rate": sample_rate,
                "stereo": audio_channels == 2,
                "use_wav_as_input": True,
                "normalize_volume": True,
                "hop_size": hop_size,
                # Input channels
                "input_channels": 1,
                # Encoder (SpecConvNeXt)
                "enc_type": "spec_convnext",
                "enc_dim": 192,
                "enc_intermediate_dim": 768,
                "enc_num_layers": 12,
                "enc_num_blocks": 2,
                "enc_n_fft": 64,
                "enc_hop_length": 16,
                "enc_latent_dim": 128,
                "enc_c_mults": [1, 2, 4],
                "enc_strides": [4, 4, 8],
                "enc_identity_init": False,
                "enc_use_snake": True,
                # Decoder (Oobleck)
                "dec_type": "oobleck",
                "dec_dim": 320,
                "dec_c_mults": [1, 2, 4, 8, 16],
                "dec_strides": [2, 4, 4, 8, 8],
                "dec_use_snake": True,
                "dec_final_tanh": False,
                "dec_out_channels": audio_channels,
                "dec_anti_aliasing": False,
                "dec_use_nearest_upsample": False,
                "dec_use_tanh_at_final": False,
                # Bottleneck (VAE)
                "bottleneck_type": "vae",
                "bottleneck": AttrDict({"type": "vae"}),
                # Common
                "activation": "snakebeta",
                "snake_logscale": True,
                "anti_aliasing": False,
                "use_cuda_kernel": False,
                "causal": False,
                "padding_mode": "zeros",
                # Vocoder
                "vocoder_input_dim": io_channels,
            }
        )

    # Create model directly on device (don't use meta device)

    # and SpectrogramConvNeXtEncoder. After loading the checkpoint, we must call
    # remove_weight_norm() which requires materialized tensors (not meta tensors).
    # Therefore, we create the model directly on the target device instead of using
    # the meta device optimization pattern.
    model = load_generator(config.model_type, config, device)

    # Load checkpoint if provided
    if pretrained_path is not None and pretrained_path != "":
        if pretrained_path.startswith("s3://"):
            backend_args = {
                "backend": "s3",
                "s3_credential_path": object_store_credential_path_pretrained,
                "path_mapping": None,
            }
            checkpoint = easy_io.load(pretrained_path, backend_args=backend_args, map_location=device)
        else:
            checkpoint = torch.load(pretrained_path, map_location=device)

        # Determine the correct key for state dict
        if "generator" in checkpoint.keys():
            state_dict = checkpoint["generator"]
        elif "state_dict" in checkpoint.keys():
            state_dict = checkpoint["state_dict"]
        else:
            raise RuntimeError(f"No valid key found in checkpoint.keys(): {checkpoint.keys()}")

        # Load state dict (strict=False like original implementation)
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded AVAE checkpoint from {pretrained_path}")

    return model, config_dict


class AVAEModel:
    """
    AVAE model wrapper for audio tokenization.

    This class handles model loading, encoding, and decoding operations.
    """

    def __init__(
        self,
        vae_pth: str = "",
        config_path: str = "",
        object_store_credential_path_pretrained: str = "",
        sample_rate: int = 44100,
        audio_channels: int = 2,
        io_channels: int = 64,
        hop_size: int = 2048,
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
        normalize_volume: bool = True,
    ):
        self.dtype = dtype
        self.device = device
        self.sample_rate = sample_rate
        self.audio_channels = audio_channels
        self.io_channels = io_channels
        self.hop_size = hop_size
        self.normalize_volume = normalize_volume

        # Load model and config
        self.model, self.config_dict = _load_avae_model(
            pretrained_path=vae_pth,
            config_path=config_path,
            sample_rate=sample_rate,
            audio_channels=audio_channels,
            io_channels=io_channels,
            hop_size=hop_size,
            object_store_credential_path_pretrained=object_store_credential_path_pretrained,
            device=device,
        )

        # Set to eval mode and freeze
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        # Remove weight norm (must be done before dtype conversion)
        if hasattr(self.model, "remove_weight_norm"):
            self.model.remove_weight_norm()

        # Convert to target dtype
        self.model = self.model.to(dtype=dtype)

    def count_param(self) -> int:
        """Count model parameters."""
        return sum(p.numel() for p in self.model.parameters())

    @torch.no_grad()
    def encode(self, audio: torch.Tensor, force_pad: bool = False) -> torch.Tensor:
        """
        Encode audio to latent representation.

        Args:
            audio: Audio tensor of shape [B,C,T_audio]
            force_pad: Whether to force padding to hop_size alignment

        Returns:
            Latent tensor of shape [B,io_channels,T_latent]
        """
        in_dtype = audio.dtype
        x = audio.clone().to(self.device)  # [B,C,T_audio]

        # Normalize volume (in original dtype for precision)
        if self.normalize_volume:
            x = x / (x.abs().max() + 1e-5) * 0.95  # [B,C,T_audio]

        # Padding logic
        # Note: AVAEModel is always in eval mode, so we always pad unless explicitly disabled
        # This matches the intended behavior where models in eval mode should pad for inference
        if force_pad or not self.model.training:
            x_len = x.shape[-1]
            pad_amount = (self.hop_size - (x_len % self.hop_size)) % self.hop_size
            if pad_amount > 0:
                x = torch.nn.functional.pad(x, (0, pad_amount), mode="constant", value=0)  # [B,C,T_audio_padded]

        # Convert to model dtype AFTER normalization and padding (matches avae.py behavior)
        x = x.to(self.dtype)  # [B,C,T_audio_padded]

        # Encode
        enc_return_dict = self.model.encode(x)
        if isinstance(enc_return_dict, dict) and "latent" in enc_return_dict:
            latent = enc_return_dict["latent"]  # [B,io_channels,T_latent]
        else:
            latent = enc_return_dict  # [B,io_channels,T_latent]

        return latent.to(in_dtype)  # [B,io_channels,T_latent]

    @torch.no_grad()
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representation to audio.

        Args:
            latent: Latent tensor of shape [B,io_channels,T_latent]

        Returns:
            Audio tensor of shape [B,C,T_audio]
        """
        in_dtype = latent.dtype
        # Convert to device first, then to model dtype (matches avae.py behavior)
        z = latent.to(self.device)  # [B,io_channels,T_latent]
        z = z.to(self.dtype)  # [B,io_channels,T_latent]

        # Decode
        if hasattr(self.model, "decode"):
            dec_return_dict = self.model.decode(z)
            audio_out = dec_return_dict["decoder_out"]  # [B,C,T_audio]
        else:
            audio_out = self.model.decoder(z)  # [B,C,T_audio]

        # Clamp to valid audio range
        audio_out = torch.clamp(audio_out, min=-1.0, max=1.0)  # [B,C,T_audio]

        return audio_out.to(in_dtype)  # [B,C,T_audio]


class AVAEInterface(AudioTokenizerInterface):
    """
    AVAE Interface for audio tokenization.

    Implements AudioTokenizerInterface for integration with the imaginaire4 framework.

    Supports two latent normalization methods:
    1. Tanh companding (normalization_type="tanh"): Soft-clips outliers using tanh function.
       Good for heavy-tailed distributions. Used in exp303.
    2. Mean-std normalization (normalization_type="mean_std"): Classic z-score normalization.
       Legacy method used in exp301.

    Normalization behavior:
    - normalization_type="none": No normalization applied (default)
    - normalization_type="tanh":
        - Encode: latent = tanh(raw_latent / tanh_input_scale) * tanh_output_scale
        - Decode: raw_latent = atanh(clamp(latent / tanh_output_scale)) * tanh_input_scale
    - normalization_type="mean_std":
        - Encode: latent = (raw_latent - latent_mean) / latent_std
        - Decode: raw_latent = latent * latent_std + latent_mean

    Args:
        bucket_name: S3 bucket name for pretrained weights
        object_store_credential_path_pretrained: Path to S3 credentials
        avae_path: Path to AVAE checkpoint within bucket
        avae_config_path: Path to AVAE config JSON/YAML within bucket (optional, uses default if empty)
        sample_rate: Audio sample rate in Hz
        audio_channels: Number of audio channels (1=mono, 2=stereo)
        io_channels: Number of latent channels
        hop_size: Temporal compression factor
        normalize_latents: Legacy flag - if True and normalization_type not set, uses "tanh".
        normalization_type: Type of normalization ("none", "tanh", "mean_std"). Default: "none".
        tanh_input_scale: Scale factor for tanh input (latent / scale before tanh).
                         Controls saturation point. Recommended: 1.0-1.5.
        tanh_output_scale: Scale factor for tanh output (tanh * scale).
                          Controls output range for rectified flow. Recommended: 3.0-3.5.
        tanh_clamp: Clamp value for atanh during decode (prevents inf). Default: 0.995.
        latent_mean: Mean value for mean-std normalization. Required if normalization_type="mean_std".
        latent_std: Std value for mean-std normalization. Required if normalization_type="mean_std".
    """

    def __init__(
        self,
        bucket_name: str = "",
        object_store_credential_path_pretrained: str = "",
        avae_path: str = "",
        avae_config_path: str = "",
        sample_rate: int = 44100,
        audio_channels: int = 2,
        io_channels: int = 64,
        hop_size: int = 2048,
        normalize_latents: bool = False,
        normalization_type: str = "none",
        tanh_input_scale: float = 1.5,
        tanh_output_scale: float = 3.5,
        tanh_clamp: float = 0.995,
        latent_mean: float | list[float] | None = None,
        latent_std: float | list[float] | None = None,
    ):
        super().__init__(object_store_credential_path_pretrained=object_store_credential_path_pretrained)

        # Construct full S3 paths
        vae_path_full = f"s3://{bucket_name}/{avae_path}" if bucket_name and avae_path else avae_path
        config_path_full = (
            f"s3://{bucket_name}/{avae_config_path}" if bucket_name and avae_config_path else avae_config_path
        )

        # Initialize model wrapper
        self.model = AVAEModel(
            vae_pth=vae_path_full,
            config_path=config_path_full,
            object_store_credential_path_pretrained=object_store_credential_path_pretrained,
            sample_rate=sample_rate,
            audio_channels=audio_channels,
            io_channels=io_channels,
            hop_size=hop_size,
            dtype=torch.bfloat16,
            device="cuda",
        )

        self._sample_rate = sample_rate
        self._audio_channels = audio_channels
        self._io_channels = io_channels
        self._hop_size = hop_size

        # Determine normalization type
        # Handle legacy normalize_latents flag for backward compatibility
        if normalization_type == "none" and normalize_latents:
            # Legacy behavior: normalize_latents=True without explicit type means tanh
            normalization_type = "tanh"

        self._normalization_type = normalization_type
        self._normalize_latents = normalization_type != "none"

        # Tanh companding parameters
        self._tanh_input_scale = tanh_input_scale
        self._tanh_output_scale = tanh_output_scale
        self._tanh_clamp = tanh_clamp

        # Mean-std normalization parameters
        # Load from config file if not explicitly provided and using mean_std normalization
        if normalization_type == "mean_std":
            config_dict = self.model.config_dict
            if latent_mean is None:
                if config_dict is not None and "latent_mean" in config_dict:
                    latent_mean = config_dict["latent_mean"]
                    print("[AVAEInterface] Loaded latent_mean from config file")
                else:
                    raise ValueError(
                        "normalization_type='mean_std' requires latent_mean. "
                        "Either provide it explicitly or ensure it's in the AVAE config JSON file."
                    )
            if latent_std is None:
                if config_dict is not None and "latent_std" in config_dict:
                    latent_std = config_dict["latent_std"]
                    print("[AVAEInterface] Loaded latent_std from config file")
                else:
                    raise ValueError(
                        "normalization_type='mean_std' requires latent_std. "
                        "Either provide it explicitly or ensure it's in the AVAE config JSON file."
                    )

        # Convert to tensors for per-channel normalization support
        if latent_mean is not None:
            if isinstance(latent_mean, (list, tuple)):
                self._latent_mean = torch.tensor(latent_mean, dtype=torch.float32, device="cuda")  # [io_channels]
                # Reshape for broadcasting: [C] -> [1, C, 1]
                self._latent_mean = self._latent_mean.view(1, -1, 1)  # [1,io_channels,1]
            else:
                self._latent_mean = float(latent_mean)
        else:
            self._latent_mean = None

        if latent_std is not None:
            if isinstance(latent_std, (list, tuple)):
                self._latent_std = torch.tensor(latent_std, dtype=torch.float32, device="cuda")  # [io_channels]
                # Reshape for broadcasting: [C] -> [1, C, 1]
                self._latent_std = self._latent_std.view(1, -1, 1)  # [1,io_channels,1]
            else:
                self._latent_std = float(latent_std)
        else:
            self._latent_std = None

        # Log normalization settings
        if normalization_type == "tanh":
            print(
                f"[AVAEInterface] Tanh companding enabled: "
                f"input_scale={tanh_input_scale}, output_scale={tanh_output_scale}, "
                f"clamp={tanh_clamp}"
            )
        elif normalization_type == "mean_std":
            mean_info = (
                f"tensor shape {self._latent_mean.shape}"
                if isinstance(self._latent_mean, torch.Tensor)
                else self._latent_mean
            )
            std_info = (
                f"tensor shape {self._latent_std.shape}"
                if isinstance(self._latent_std, torch.Tensor)
                else self._latent_std
            )
            print(f"[AVAEInterface] Mean-std normalization enabled: mean={mean_info}, std={std_info}")
        else:
            print("[AVAEInterface] No latent normalization applied")

    @property
    def dtype(self) -> torch.dtype:
        """Current dtype of the model."""
        return self.model.dtype

    def reset_dtype(self) -> None:
        """Reset the dtype of the model."""
        self.model.model = self.model.model.to(self.model.dtype)

    def encode(self, audio: torch.Tensor, force_pad: bool = False) -> torch.Tensor:
        """Encode audio to latent representation.

        Args:
            audio: [B,C,T_audio]

        Returns:
            [B,io_channels,T_latent]

        Normalization depends on normalization_type:
        - "none": No normalization
        - "tanh": latent = tanh(raw_latent / tanh_input_scale) * tanh_output_scale
        - "mean_std": latent = (raw_latent - latent_mean) / latent_std
        """
        latent = self.model.encode(audio, force_pad=force_pad)  # [B,io_channels,T_latent]

        # Apply normalization based on type
        if self._normalization_type == "tanh":
            in_dtype = latent.dtype
            latent = latent.float()  # [B,io_channels,T_latent]
            latent = torch.tanh(latent / self._tanh_input_scale) * self._tanh_output_scale  # [B,io_channels,T_latent]
            latent = latent.to(in_dtype)  # [B,io_channels,T_latent]
        elif self._normalization_type == "mean_std":
            in_dtype = latent.dtype
            latent = latent.float()  # [B,io_channels,T_latent]
            # Handle both scalar and tensor (per-channel) mean/std
            mean = self._latent_mean
            std = self._latent_std
            if isinstance(mean, torch.Tensor):
                mean = mean.to(latent.device)  # [1,io_channels,1]
            if isinstance(std, torch.Tensor):
                std = std.to(latent.device)  # [1,io_channels,1]
            latent = (latent - mean) / std  # [B,io_channels,T_latent]
            latent = latent.to(in_dtype)  # [B,io_channels,T_latent]

        return latent  # [B,io_channels,T_latent]

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to audio.

        Args:
            latent: [B,io_channels,T_latent]

        Returns:
            [B,C,T_audio]

        Denormalization depends on normalization_type:
        - "none": No denormalization
        - "tanh": raw_latent = atanh(clamp(latent / tanh_output_scale)) * tanh_input_scale
        - "mean_std": raw_latent = latent * latent_std + latent_mean
        """
        # Apply denormalization based on type
        if self._normalization_type == "tanh":
            in_dtype = latent.dtype
            latent = latent.float()  # [B,io_channels,T_latent]
            # Clamp to valid atanh range to avoid inf
            latent = torch.clamp(
                latent / self._tanh_output_scale,
                -self._tanh_clamp,
                self._tanh_clamp,
            )  # [B,io_channels,T_latent]
            latent = torch.atanh(latent) * self._tanh_input_scale  # [B,io_channels,T_latent]
            latent = latent.to(in_dtype)  # [B,io_channels,T_latent]
        elif self._normalization_type == "mean_std":
            in_dtype = latent.dtype
            latent = latent.float()  # [B,io_channels,T_latent]
            # Handle both scalar and tensor (per-channel) mean/std
            mean = self._latent_mean
            std = self._latent_std
            if isinstance(mean, torch.Tensor):
                mean = mean.to(latent.device)  # [1,io_channels,1]
            if isinstance(std, torch.Tensor):
                std = std.to(latent.device)  # [1,io_channels,1]
            latent = latent * std + mean  # [B,io_channels,T_latent]
            latent = latent.to(in_dtype)  # [B,io_channels,T_latent]

        return self.model.decode(latent)  # [B,C,T_audio]

    def get_latent_num_samples(self, num_audio_samples: int) -> int:
        """Calculate number of latent samples from audio samples."""
        return num_audio_samples // self.temporal_compression_factor

    def get_audio_num_samples(self, num_latent_samples: int) -> int:
        """Calculate number of audio samples from latent samples."""
        return num_latent_samples * self.temporal_compression_factor

    @property
    def temporal_compression_factor(self) -> int:
        """Temporal compression factor (hop size)."""
        return self._hop_size

    @property
    def sample_rate(self) -> int:
        """Audio sample rate in Hz."""
        return self._sample_rate

    @property
    def audio_channels(self) -> int:
        """Number of audio channels."""
        return self._audio_channels

    @property
    def latent_ch(self) -> int:
        """Number of latent channels."""
        return self._io_channels

    @property
    def name(self) -> str:
        """Name of the tokenizer."""
        return "avae_tokenizer"

    def count_param(self) -> int:
        """Count the number of parameters in the model."""
        return self.model.count_param()
