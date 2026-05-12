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

subdirectory="Cosmos3-Test-DCP/cosmos3_vfm/t2w_mot_0p6b_qwen3_vl_runs/t2w_mot_dryrun_exp200_001_qwen3_vl_0p6b_480res_qwen3_captions_mrope_v2/checkpoints/iter_000079500/model"
CHECKPOINT_PATH=$(uvx hf@$HF_VERSION download \
    --repo-type model nvidia/Cosmos-Experimental \
    --revision cb49fa71add22a060936f2391d1eb4e58a49358a \
    --include "$subdirectory/*" \
    --quiet)/$subdirectory

# Use temporary directory, since output is large.
CUDA_VISIBLE_DEVICES= python -m cosmos3.scripts.export_model \
    -o $TMP_DIR/model \
    --checkpoint-path $CHECKPOINT_PATH
