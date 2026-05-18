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

import torch

from cosmos.utils import log
from cosmos.utils.distributed import get_rank, sync_model_states
from cosmos.utils.easy_io import easy_io
from cosmos.model.vfm.tokenizers.dc_ae.dc_ae_v import (
    DCAEV,
    DCAEVConfig,
    dc_ae_v_f32t4_encoder_causal_decoder_chunk_causal_4,
)
from cosmos.model.vfm.tokenizers.interface import VideoTokenizerInterface

DEFAULT_MODEL_NAME = "dcae4x32x32_c64_t120_256p_fps_all_encoder_causal_decoder_chunk_causal_4_nogan_cosmos_pad_7_v0.1"


class DCAE4x32x32Interface(VideoTokenizerInterface):
    def __init__(
        self,
        bucket_name: str = "",
        object_store_credential_path_pretrained: str = "",
        vae_path: str = "",
        chunk_duration: int = 81,
        model_name: str = DEFAULT_MODEL_NAME,
        spatial_compression_factor: int = 32,
        temporal_compression_factor: int = 4,
        encode_chunk_frames: int = 128,  # Placeholder
        encode_bucket_multiple: int = 2,  # Placeholder
        device: str = "cuda",
        compilable: bool = True,
    ):
        vae_path_full = f"s3://{bucket_name}/{vae_path}"
        self._spatial_compression_factor = spatial_compression_factor
        self._temporal_compression_factor = temporal_compression_factor
        self.chunk_duration = chunk_duration

        # Build config (without pretrained_path so DCAEV doesn't try to load itself).
        cfg: DCAEVConfig = dc_ae_v_f32t4_encoder_causal_decoder_chunk_causal_4(model_name, pretrained_path=None)
        cfg.compilable = compilable

        # Instantiate model on meta device to avoid double allocation.
        with torch.device("meta"):
            self.model = DCAEV(cfg)

        # Load checkpoint from S3 on rank 0 only, then broadcast.
        if get_rank() == 0:
            backend_args = {
                "backend": "s3",
                "s3_credential_path": object_store_credential_path_pretrained,
            }
            checkpoint = easy_io.load(vae_path_full, backend_args=backend_args, map_location=device)
            log.info(f"loading {vae_path_full}")

            self.model.load_state_dict(checkpoint["model_state_dict"], assign=True)
        else:
            self.model.to_empty(device=device)

        self.model.eval().requires_grad_(False)
        self.model.to(dtype=torch.bfloat16)

        sync_model_states(self.model)
        self.is_compiled = False
        self.use_streaming_encode = False

    def compile_encode_for_cudagraphs(
        self,
        *,
        mode: str = "reduce-overhead",
        fullgraph: bool = False,
        dynamic: bool = False,
        backend: str = "inductor",
    ) -> None:

        self.model.encoder = self.model.encoder.to(memory_format=torch.channels_last_3d)
        self.model.encoder = torch.compile(self.model.encoder, fullgraph=True, mode=mode)
        self.is_compiled = True

    @property
    def dtype(self):
        return self.model.dtype

    def reset_dtype(self):
        pass

    @torch.inference_mode()
    def encode(self, state: torch.Tensor) -> torch.Tensor:
        in_dtype = state.dtype
        tcf = self._temporal_compression_factor
        # Add padding to the sequence length to make it divisible by
        # the temporal compression factor after num_pad_frames padding.
        seq_len = state.shape[2] + self.model.cfg.num_pad_frames
        if seq_len % tcf != 0:
            raise ValueError(f"Sequence length {seq_len} is not divisible by temporal compression factor {tcf}")
        return self.model.encode(state.to(torch.bfloat16)).to(in_dtype)

    @torch.inference_mode()
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        in_dtype = latent.dtype
        return self.model.decode(latent.to(torch.bfloat16)).to(in_dtype)

    def get_latent_num_frames(self, num_pixel_frames: int) -> int:
        return (num_pixel_frames + self.model.cfg.num_pad_frames) // self._temporal_compression_factor

    def get_pixel_num_frames(self, num_latent_frames: int) -> int:
        return num_latent_frames * self._temporal_compression_factor - self.model.cfg.num_pad_frames

    @property
    def spatial_compression_factor(self):
        return self._spatial_compression_factor

    @property
    def temporal_compression_factor(self):
        return self._temporal_compression_factor

    @property
    def pixel_chunk_duration(self):
        return self.chunk_duration

    @property
    def latent_chunk_duration(self):
        return self.get_latent_num_frames(self.chunk_duration)

    @property
    def latent_ch(self):
        return self.model.cfg.latent_channels

    @property
    def spatial_resolution(self):
        return 512

    @property
    def name(self):
        return "dc_ae_4x32x32_tokenizer"
