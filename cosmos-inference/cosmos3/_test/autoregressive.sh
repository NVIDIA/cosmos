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

torchrun $TORCHRUN_ARGS -m cosmos3.scripts.inference \
    -i "$INPUT_DIR/autoregressive/*_test.json" \
    -o $OUTPUT_DIR/inference \
    --checkpoint-path 9d47ff88-cb43-4bf0-8f9b-3d74e2bd9a67 \
    --use_cuda_graphs \
    $INFERENCE_ARGS
