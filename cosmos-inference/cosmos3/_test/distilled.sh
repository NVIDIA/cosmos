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

COSMOS_TRAINING=1 torchrun $TORCHRUN_ARGS -m cosmos3.scripts.inference \
    -i "$INPUT_DIR/interactive/*.json" \
    -o $OUTPUT_DIR/inference \
    --checkpoint-path eae30a62-7633-466c-976f-47f5a90c843f \
    $INFERENCE_ARGS
