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
CONFIG_FILE="cosmos3/configs/experiment/mixed_modality_sft_nano.yaml"
export DATASET_PATH=$(uvx hf@latest download --repo-type dataset nvidia/bridge-v2-subset-synthetic-captions --revision 46468e12ac0dd36901e9e3240d4fc7620942b5d7 --quiet)/sft_dataset_bridge

# HF -> DCP
# Use temporary directory, since output is large.
python -m cosmos3.scripts.convert_model_to_dcp \
    --checkpoint-path $BASE_CHECKPOINT_NAME \
    -o $TMP_DIR/checkpoint_base

# Train
torchrun $TORCHRUN_ARGS -m cosmos3.scripts.train \
    -o $OUTPUT_DIR/train \
    --config-file $CONFIG_FILE \
    $TRAIN_ARGS \
    --config-overrides \
    "checkpoint.load_path=$TMP_DIR/checkpoint_base" \
    $TRAIN_OVERRIDES

CHECKPOINT_ITER=$(cat $OUTPUT_DIR/train/job/checkpoints/latest_checkpoint.txt)
CHECKPOINT_PATH=$OUTPUT_DIR/train/job/checkpoints/$CHECKPOINT_ITER

# DCP -> HF
# Use temporary directory, since output is large.
python -m cosmos3.scripts.export_model \
    -o $TMP_DIR/model \
    --checkpoint-path $CHECKPOINT_PATH \
    --config-file $OUTPUT_DIR/train/config.yaml

# Exported model inference is already tested in 'scripts/_test/convert_model_to_dcp.sh'
