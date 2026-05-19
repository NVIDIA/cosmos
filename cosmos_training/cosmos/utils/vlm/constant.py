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

# PyTorch CrossEntropyLoss treats targets equal to -100 as positions to
# exclude from the loss (its upstream default for ignore_index). Used
# wherever label tensors are assembled or CE losses are computed.
IGNORE_INDEX: int = -100

# Per-processor keys that downstream augmentors / collators pass through
# from the HuggingFace processor output into the model batch.
PROCESSOR_KEYS_TO_ADD_QWEN: list[str] = [
    "input_ids",
    "attention_mask",
    "pixel_values",
    "pixel_values_videos",
    "image_grid_thw",
    "video_grid_thw",
    "second_per_grid_ts",
]
PROCESSOR_KEYS_TO_ADD_EAGLE: list[str] = ["input_ids", "attention_mask", "pixel_values", "image_sizes"]

PROCESSOR_KEYS_TO_ADD: list[str] = list(set(PROCESSOR_KEYS_TO_ADD_QWEN + PROCESSOR_KEYS_TO_ADD_EAGLE))
