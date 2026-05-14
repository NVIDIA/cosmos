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



Wan2pt2VAEConfig: LazyDict = L(Wan2pt2VAEInterface)(
    bucket_name=PLACEHOLDER,
    object_store_credential_path_pretrained=PLACEHOLDER,
    vae_path=PRETRAINED_TOKENIZER_WAN2PT2_VAE_PTH,
    spatial_compression_factor=16,
    temporal_compression_factor=4,
)



def register_tokenizer():
    cs = ConfigStore.instance()

    # Wan2pt2 tokenizer (the only one the cosmos release needs)
    cs.store(group="tokenizer", package="model.config.tokenizer", name="wan2pt2_tokenizer", node=Wan2pt2VAEConfig)


def register_sound_tokenizer():
    """Register sound tokenizers in Hydra ConfigStore under model.config.sound_tokenizer."""
    cs = ConfigStore.instance()
