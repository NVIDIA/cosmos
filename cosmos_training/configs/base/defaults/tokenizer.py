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

from hydra.core.config_store import ConfigStore

from cosmos.utils.lazy_config import PLACEHOLDER, LazyDict
from cosmos.utils.lazy_config import LazyCall as L
from cosmos.model.vfm.tokenizers.audio.avae import AVAEInterface
from cosmos.model.vfm.tokenizers.dc_ae.dc_ae_4x32x32 import DCAE4x32x32Interface
from cosmos.model.vfm.tokenizers.flux_vae_8x8 import FluxVAEInterface
from cosmos.model.vfm.tokenizers.uniae.noncausal_4x16x16 import UniAEVAEInterface
from cosmos.model.vfm.tokenizers.wan2pt1_vae_4x8x8 import Wan2pt1VAEInterface
from cosmos.model.vfm.tokenizers.wan2pt2_vae_4x16x16 import Wan2pt2VAEInterface

PRETRAINED_TOKENIZER_WAN2PT1_VAE_PTH = "pretrained/tokenizers/video/wan2pt1/Wan2.1_VAE.pth"
PRETRAINED_TOKENIZER_WAN2PT2_VAE_PTH = "pretrained/tokenizers/video/wan2pt2/Wan2.2_VAE.pth"
PRETRAINED_TOKENIZER_FLUX_VAE_PTH = "pretrained/tokenizers/image/flux/ae.safetensors"

# UniAE checkpoint paths
PRETRAINED_TOKENIZER_UNIAE_4X16X16_C48_T8TO24_64TO512P_FPS_ALL_ENCODER_NONCAUSAL_DECODER_NONCAUSAL_NOGAN_BEST_S1_VAE_PTH = "pretrained/tokenizers/video/cosmos/uniae4x16x16_c48_t8to24_64to512p_fps_all_encoder_noncausal_decoder_noncausal_nogan_best_s1.pt"

# DCAE checkpoint paths
PRETRAINED_TOKENIZER_DCAE_PTH = "pretrained/tokenizers/video/cosmos/dc-ae-v-1.0-f32t4c64-cosmos-encoder-causal-decoder-chunk-causal-4-frame-120-pad-7-no-gan.pt"
PRETRAINED_TOKENIZER_DCAE_4X32X32_C64_T120_256P_FPS_ALL_ENCODER_CAUSAL_DECODER_CHUNKCAUSAL4_NOGAN_COSMOS_PAD_7_V0PT2_PTH = "pretrained/tokenizers/video/cosmos/dcae4x32x32_c64_t120_256p_fps_all_encoder_causal_decoder_chunk_causal_4_nogan_cosmos_pad_7_v0.2.pt"

# AVAE (Audio VAE) checkpoint paths
PRETRAINED_TOKENIZER_AVAE_PTH = "pretrained/tokenizers/audio/avae/model_unwrap.ckpt"
PRETRAINED_TOKENIZER_AVAE_44K_NONCAUSAL = "pretrained/tokenizers/audio/avae/avae_44k_noncausal_21hz_64ch.ckpt"
PRETRAINED_TOKENIZER_AVAE_44K_CAUSAL = "pretrained/tokenizers/audio/avae/avae_44k_causal_21hz_64ch.ckpt"
PRETRAINED_TOKENIZER_AVAE_48K_25HZ = "pretrained/tokenizers/audio/avae/avae_48k_noncausal_25hz_64ch.ckpt"
PRETRAINED_TOKENIZER_AVAE_48K_6HZ = "pretrained/tokenizers/audio/avae/avae_48k_noncausal_6hz_64ch.ckpt"


# Flux tokenizer config
FluxVAEConfig: LazyDict = L(FluxVAEInterface)(
    # This is the flux image tokenizer.
    # We use it for bagel inference.
    # We do not use it for Cosmos3.
    bucket_name=PLACEHOLDER,
    object_store_credential_path_pretrained=PLACEHOLDER,
    vae_path=PRETRAINED_TOKENIZER_FLUX_VAE_PTH,
    chunk_duration=1,
    spatial_compression_factor=8,
    temporal_compression_factor=1,
)

Wan2pt1VAEConfig: LazyDict = L(Wan2pt1VAEInterface)(
    # 4x8x8 tokenizer
    bucket_name=PLACEHOLDER,
    object_store_credential_path_pretrained=PLACEHOLDER,
    vae_path=PRETRAINED_TOKENIZER_WAN2PT1_VAE_PTH,
    spatial_compression_factor=8,
    temporal_compression_factor=4,
)

Wan2pt2VAEConfig: LazyDict = L(Wan2pt2VAEInterface)(
    bucket_name=PLACEHOLDER,
    object_store_credential_path_pretrained=PLACEHOLDER,
    vae_path=PRETRAINED_TOKENIZER_WAN2PT2_VAE_PTH,
    spatial_compression_factor=16,
    temporal_compression_factor=4,
)

DCAE4x32x32Config: LazyDict = L(DCAE4x32x32Interface)(
    bucket_name=PLACEHOLDER,
    object_store_credential_path_pretrained=PLACEHOLDER,
    vae_path=PRETRAINED_TOKENIZER_DCAE_PTH,
    spatial_compression_factor=32,
    temporal_compression_factor=4,
)

DCAE4x32x32C64T120_256pFpsAllEncoderCausalDecoderChunkCausal4NoganCosmosPad7V0pt2Config: LazyDict = L(
    DCAE4x32x32Interface
)(
    bucket_name=PLACEHOLDER,
    object_store_credential_path_pretrained=PLACEHOLDER,
    vae_path=PRETRAINED_TOKENIZER_DCAE_4X32X32_C64_T120_256P_FPS_ALL_ENCODER_CAUSAL_DECODER_CHUNKCAUSAL4_NOGAN_COSMOS_PAD_7_V0PT2_PTH,
    model_name="dcae4x32x32_c64_t120_256p_fps_all_encoder_causal_decoder_chunk_causal_4_nogan_cosmos_pad_7_v0.2",
    spatial_compression_factor=32,
    temporal_compression_factor=4,
)

UniAE4x16x16C48T8to24_64to512pFpsAllEncoderNoncausalDecoderNoncausalNoganBestS1Config: LazyDict = L(UniAEVAEInterface)(
    bucket_name=PLACEHOLDER,
    object_store_credential_path_pretrained=PLACEHOLDER,
    vae_path=PRETRAINED_TOKENIZER_UNIAE_4X16X16_C48_T8TO24_64TO512P_FPS_ALL_ENCODER_NONCAUSAL_DECODER_NONCAUSAL_NOGAN_BEST_S1_VAE_PTH,
    spatial_compression_factor=16,
    temporal_compression_factor=4,
)

# =============================================================================
# AVAE (Audio VAE) Tokenizer Configs
# =============================================================================

# Legacy config with tanh companding (non-commercial use only)
# Latent rate: 44100 / 2048 = 21.53Hz
AVAETokenizerConfig: LazyDict = L(AVAEInterface)(
    bucket_name=PLACEHOLDER,
    object_store_credential_path_pretrained=PLACEHOLDER,
    avae_path=PRETRAINED_TOKENIZER_AVAE_PTH,
    sample_rate=44100,
    audio_channels=2,
    io_channels=64,
    hop_size=2048,
    normalization_type="tanh",
    tanh_input_scale=1.0,
    tanh_output_scale=3.0,
)


# 44.1kHz Non-causal (PRIMARY - used for V2A/T2A training)
# Latent rate: 44100 / 2048 = 21.53Hz
AVAE_44k_NoncausalConfig: LazyDict = L(AVAEInterface)(
    bucket_name=PLACEHOLDER,
    object_store_credential_path_pretrained=PLACEHOLDER,
    avae_path=PRETRAINED_TOKENIZER_AVAE_44K_NONCAUSAL,
    sample_rate=44100,
    audio_channels=2,
    io_channels=64,
    hop_size=2048,
    normalize_latents=True,
    tanh_input_scale=1.5,
    tanh_output_scale=3.5,
)

# 48kHz 25Hz (higher quality audio)
# Latent rate: 48000 / 1920 = 25Hz
AVAE_48k_25hzConfig: LazyDict = L(AVAEInterface)(
    bucket_name=PLACEHOLDER,
    object_store_credential_path_pretrained=PLACEHOLDER,
    avae_path=PRETRAINED_TOKENIZER_AVAE_48K_25HZ,
    sample_rate=48000,
    audio_channels=2,
    io_channels=64,
    hop_size=1920,
    normalize_latents=True,
    tanh_input_scale=1.5,
    tanh_output_scale=3.5,
)


def register_tokenizer():
    cs = ConfigStore.instance()

    # Wan2pt1 and Wan2pt2 tokenizers
    cs.store(group="tokenizer", package="model.config.tokenizer", name="wan2pt1_tokenizer", node=Wan2pt1VAEConfig)
    cs.store(group="tokenizer", package="model.config.tokenizer", name="wan2pt2_tokenizer", node=Wan2pt2VAEConfig)
    # UniAE tokenizer
    cs.store(
        group="tokenizer",
        package="model.config.tokenizer",
        name="uniae_4x16x16_c48_t8to24_64to512p_fps_all_encoder_noncausal_decoder_noncausal_nogan_best_s1_tokenizer",
        node=UniAE4x16x16C48T8to24_64to512pFpsAllEncoderNoncausalDecoderNoncausalNoganBestS1Config,
    )
    # Flux tokenizer
    cs.store(group="tokenizer", package="model.config.tokenizer", name="flux_tokenizer", node=FluxVAEConfig)
    # DC AE 4x32x32 tokenizer
    cs.store(
        group="tokenizer",
        package="model.config.tokenizer",
        name="dc_ae_4x32x32_tokenizer",
        node=DCAE4x32x32Config,
    )
    cs.store(
        group="tokenizer",
        package="model.config.tokenizer",
        name="dc_ae_4x32x32_c64_t120_256p_fps_all_encoder_causal_decoder_chunk_causal_4_nogan_cosmos_pad_7_v0.2_tokenizer",
        node=DCAE4x32x32C64T120_256pFpsAllEncoderCausalDecoderChunkCausal4NoganCosmosPad7V0pt2Config,
    )


def register_sound_tokenizer():
    """Register sound tokenizers in Hydra ConfigStore under model.config.sound_tokenizer."""
    cs = ConfigStore.instance()
    cs.store(
        group="sound_tokenizer", package="model.config.sound_tokenizer", name="avae_48k_25hz", node=AVAE_48k_25hzConfig
    )
    cs.store(
        group="sound_tokenizer",
        package="model.config.sound_tokenizer",
        name="avae_44k_noncausal",
        node=AVAE_44k_NoncausalConfig,
    )
    cs.store(
        group="sound_tokenizer", package="model.config.sound_tokenizer", name="avae_tokenizer", node=AVAETokenizerConfig
    )
