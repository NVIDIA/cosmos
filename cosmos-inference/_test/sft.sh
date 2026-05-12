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

DATASET_PATH=$(uvx hf@latest download --repo-type dataset nvidia/bridge-v2-subset-synthetic-captions --include "sft_dataset_bridge/*" --quiet)/sft_dataset_bridge

# HF -> DCP
# Use temporary directory, since output is large.
python -m cosmos3.scripts.convert_model_to_dcp \
    --checkpoint-path Cosmos3-Nano \
    -o $TMP_DIR/checkpoint_base

# Train
torchrun $TORCHRUN_ARGS -m cosmos3.scripts.train \
    -o $OUTPUT_DIR/train \
    --config-file cosmos3/configs/experiment/mixed_modality_sft_8b.yaml \
    $TRAIN_ARGS \
    --config-overrides \
    "checkpoint.load_path=$TMP_DIR/checkpoint_base" \
    "dataloader_train.dataloader.datasets.video.dataset.jsonl_paths=$DATASET_PATH/train/video_dataset_file.jsonl" \
    $TRAIN_OVERRIDES

CHECKPOINT_ITER=$(cat $OUTPUT_DIR/train/job/checkpoints/latest_checkpoint.txt)
CHECKPOINT_PATH=$OUTPUT_DIR/train/job/checkpoints/$CHECKPOINT_ITER

# Inference
torchrun $TORCHRUN_ARGS -m cosmos3.scripts.inference \
    --parallelism-preset=latency \
    -i $DATASET_PATH/val/inference_prompt/episode_049683_clip000.json \
    -o $OUTPUT_DIR/inference \
    --checkpoint-path $CHECKPOINT_PATH \
    --config-file $OUTPUT_DIR/train/config.yaml \
    $INFERENCE_ARGS
