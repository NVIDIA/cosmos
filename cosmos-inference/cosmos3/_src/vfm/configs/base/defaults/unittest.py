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
import attrs

# from cosmos3._src.vfm.configs.base.defaults.cluster import GCPIADGB200Config

# We are hardcoding the unittest assets in this file.

# CLUSTER_CONFIG = GCPIADGB200Config

# add codeowner for cosmos3/_src/vfm/tokenizers


@attrs.define(slots=False)
class SwfitStackPDXrConfig:
    """
    Config for the cluster specific information.
    Everything cluster specific should be here.
    """

    object_store_bucket_data: str
    object_store_credential_data: str


UNITTEST_CONFIG = SwfitStackPDXrConfig(
    object_store_bucket_data="unittest",
    object_store_credential_data="credentials/pdx_dir.secret",
)

TOKENIZER_RECONSTRUCTION_VIDEO_PATH = "tokenizer/video/panda70m_test_0000039_00000.mp4"
AVAE_RECONSTRUCTION_AUDIO_PATH = "tokenizer/audio/test_audio.wav"
