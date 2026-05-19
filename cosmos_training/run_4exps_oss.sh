#!/usr/bin/env bash
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

# Sequential 4-experiment OSS smoke runner.
# Each launch_*.sh already has PYTHONHASHSEED=42 and --deterministic baked in.
#
# Usage:
#   srun --overlap --jobid <JOB_ID> --container-name yangyangt_dev \
#        bash -s < /lustre/.../cosmos_training/run_4exps_oss.sh

set -uo pipefail

WORKDIR="/lustre/fsw/portfolios/cosmos/users/yangyangt/Cosmos-prerelease/cosmos_training"

for tag in \
    "launch_mixed_modality_sft_8b" \
    "launch_vlm_llava_ov" \
    "launch_action_libero" \
    "launch_t2w_sft_local_datapacker"
do
    echo ""
    echo ">>> $(date '+%H:%M:%S') OSS SMOKE: $tag"
    bash "$WORKDIR/$tag.sh"
    echo ">>> $(date '+%H:%M:%S') OSS Done $tag (exit ${PIPESTATUS[0]})"
done

echo ""
echo ">>> $(date '+%H:%M:%S') ALL 4 OSS SMOKES COMPLETE"
echo "Logs at: /lustre/fsw/portfolios/cosmos/users/yangyangt/Cosmos-prerelease/training_output/logs/"
