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

# Cosmos3-Test
subdirectory="5d561d7d-080f-45cb-a455-920d444e40cc"
CHECKPOINT_PATH=$(uvx hf@$HF_VERSION download \
    --repo-type model nvidia/Cosmos3-Experimental \
    --revision 844eb561ec6a8d6a917aec463464cdd594d5e965 \
    --include "$subdirectory/*" \
    --quiet)/$subdirectory

# HF -> DCP
# Use temporary directory, since output is large.
CUDA_VISIBLE_DEVICES= python -m cosmos3.scripts.convert_model_to_dcp \
    -o $TMP_DIR/checkpoint \
    --checkpoint-path $CHECKPOINT_PATH

# DCP -> HF
# Use temporary directory, since output is large.
CUDA_VISIBLE_DEVICES= python -m cosmos3.scripts.export_model \
    -o $TMP_DIR/model \
    --checkpoint-path $TMP_DIR/checkpoint/model \
    --config-file $TMP_DIR/checkpoint/model/config.json \
    --no-use-ema-weights

# HF Inference
torchrun $TORCHRUN_ARGS -m cosmos3.scripts.inference \
    -i "$INPUT_DIR/omni/t2i.json" \
    -o $OUTPUT_DIR/inference \
    --checkpoint-path $TMP_DIR/model \
    $INFERENCE_ARGS
