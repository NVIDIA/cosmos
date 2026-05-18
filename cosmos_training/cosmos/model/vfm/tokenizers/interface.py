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

import os
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Optional

import torch

from cosmos.utils.env_parsers.cred_env_parser import CRED_ENVS


class VideoTokenizerInterface(ABC):
    def __init__(self, object_store_credential_path_pretrained: Optional[str] = None):
        assert object_store_credential_path_pretrained is None or isinstance(
            object_store_credential_path_pretrained, str
        )
        if object_store_credential_path_pretrained is None:
            self.backend_args = None
        elif os.path.exists(object_store_credential_path_pretrained) or CRED_ENVS.APP_ENV in ["prod", "dev", "stg"]:
            self.backend_args = {
                "backend": "s3",
                "path_mapping": None,
                "s3_credential_path": object_store_credential_path_pretrained,
            }
        else:
            raise FileNotFoundError(
                f"Invalid object_store_credential_path_pretrained: {object_store_credential_path_pretrained} and APP_ENV is not prod/dev/stg"
            )

    @abstractmethod
    def reset_dtype(self):
        """
        Reset the dtype of the model to the dtype its weights were trained with or quantized to.
        """
        pass

    @abstractmethod
    def encode(self, state: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def get_latent_num_frames(self, num_pixel_frames: int) -> int:
        pass

    @abstractmethod
    def get_pixel_num_frames(self, num_latent_frames: int) -> int:
        pass

    @property
    @abstractmethod
    def spatial_compression_factor(self) -> int:
        pass

    @property
    @abstractmethod
    def temporal_compression_factor(self) -> int:
        pass

    @property
    @abstractmethod
    def spatial_resolution(self) -> int:
        pass

    @property
    @abstractmethod
    def pixel_chunk_duration(self):
        pass

    @property
    @abstractmethod
    def latent_chunk_duration(self):
        pass

    @property
    @abstractmethod
    def latent_ch(self) -> int:
        pass

    def compile_encode(
        self,
        warmup_resolutions: Sequence[str],
        output_dir: str,
        aspect_ratio: str | None = None,
    ) -> None:
        """AOT-compile the tokenizer for the given resolutions.

        Subclasses that support AOT compilation should override this method.
        The default raises ``NotImplementedError``.

        Args:
            warmup_resolutions: Resolution keys to compile for.
            output_dir: Root directory where compiled artifacts are stored
                (typically ``config.job.path_local``).
            aspect_ratio: If given, only compile this single aspect ratio.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support compilation")

    @property
    def is_chunk_overlap(self):
        return False

    @property
    def is_causal(self):
        return True


class AudioTokenizerInterface(ABC):
    """Abstract interface for audio tokenizers."""

    def __init__(self, object_store_credential_path_pretrained: Optional[str] = None):
        assert object_store_credential_path_pretrained is None or isinstance(
            object_store_credential_path_pretrained, str
        )
        if object_store_credential_path_pretrained is None:
            self.backend_args = None
        elif os.path.exists(object_store_credential_path_pretrained) or CRED_ENVS.APP_ENV in ["prod", "dev", "stg"]:
            self.backend_args = {
                "backend": "s3",
                "path_mapping": None,
                "s3_credential_path": object_store_credential_path_pretrained,
            }
        else:
            raise FileNotFoundError(
                f"Invalid object_store_credential_path_pretrained: {object_store_credential_path_pretrained} and APP_ENV is not prod/dev/stg"
            )

    @abstractmethod
    def reset_dtype(self):
        """
        Reset the dtype of the model to the dtype its weights were trained with or quantized to.
        """
        pass

    @abstractmethod
    def encode(self, audio: torch.Tensor, force_pad: bool = False) -> torch.Tensor:
        """
        Encode audio waveform to latent representation.

        Args:
            audio: Input audio tensor of shape [B, C, T] where:
                   B = batch size, C = audio channels, T = time samples
            force_pad: Whether to force padding to match compression factor

        Returns:
            Latent tensor of shape [B, latent_ch, T']
        """
        pass

    @abstractmethod
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representation to audio waveform.

        Args:
            latent: Latent tensor of shape [B, latent_ch, T']

        Returns:
            Audio tensor of shape [B, C, T]
        """
        pass

    @abstractmethod
    def get_latent_num_samples(self, num_audio_samples: int) -> int:
        """
        Calculate the number of latent time samples from audio samples.

        Args:
            num_audio_samples: Number of audio samples

        Returns:
            Number of latent time samples
        """
        pass

    @abstractmethod
    def get_audio_num_samples(self, num_latent_samples: int) -> int:
        """
        Calculate the number of audio samples from latent samples.

        Args:
            num_latent_samples: Number of latent time samples

        Returns:
            Number of audio samples
        """
        pass

    @property
    @abstractmethod
    def temporal_compression_factor(self) -> int:
        """
        Temporal compression factor (downsampling ratio).
        audio_samples = latent_samples * temporal_compression_factor
        """
        pass

    @property
    @abstractmethod
    def sample_rate(self) -> int:
        """Audio sample rate in Hz."""
        pass

    @property
    @abstractmethod
    def audio_channels(self) -> int:
        """Number of audio channels (e.g., 1 for mono, 2 for stereo)."""
        pass

    @property
    @abstractmethod
    def latent_ch(self) -> int:
        """Number of latent channels."""
        pass

    @property
    def is_causal(self) -> bool:
        """Whether the model is causal (for streaming)."""
        return False
