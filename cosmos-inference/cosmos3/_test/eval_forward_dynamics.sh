# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

BASE_CHECKPOINT_NAME=Cosmos3-Nano
CONFIG_FILE="cosmos3/configs/experiment/action_fdm_sft_nano.yaml"
export DATASET_PATH=$(uvx hf@latest download --repo-type dataset nvidia/bridge_lerobot_v3 --revision b887e193b141f2fe5b6e3d567577aa51c475693b --quiet)

torchrun $TORCHRUN_ARGS -m cosmos3.scripts.eval \
    -o $OUTPUT_DIR/inference \
    --checkpoint-path $BASE_CHECKPOINT_NAME \
    --dataset.config-file $CONFIG_FILE \
    --dataset.num-samples 1 \
    $INFERENCE_ARGS
