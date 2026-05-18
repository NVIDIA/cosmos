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

from cosmos.utils.env_parsers.env_parser import EnvParser
from cosmos.utils.validator import Bool, Int, String


class InferenceEnvParser(EnvParser):
    MODEL_MODULE = String(default=None)
    MODEL_CLASS = String(default=None)
    TORCH_HOME = String(default="/config/models/checkpoints")
    TRT_ENABLED = Bool(default=False)
    PORT = Int(default=8000)
    CP_SIZE = Int(default=1)
    TP_SIZE = Int(default=1)
    FSDP_ENABLED = Bool(default=False)
    CUSTOMIZATION_TYPE = String(default="")
    NIM_DEPLOYMENT = Bool(default=False)
    RUNAI_DEPLOYMENT = Bool(default=False)
    BLUR_CUDA = Bool(default=False)
    RESIZE_CUDA = Bool(default=False)


INFERENCE_ENVS = InferenceEnvParser()

if __name__ == "__main__":
    INFERENCE_ENVS.to_json("env_params.json")

    b64 = INFERENCE_ENVS.to_b64()

    env_params_restored = InferenceEnvParser(b64)

    print(env_params_restored)
