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
"""Flash attention initialization for the vfm/ unified VLM training path.

This module replaces `cosmos_rl.policy.kernel.modeling_utils.init_flash_attn_meta`
in VLMModel.__init__, removing the last cosmos_rl import from cosmos3/_src/vfm/.

Why a stub for the FlashAttnMeta singleton part?
  HFModel uses HuggingFace's flash_attention_2 implementation
  (attn_implementation="flash_attention_2" in AutoModel.from_config), which calls
  flash_attn_func directly via the transformers.modeling_flash_attention_utils path.
  This is entirely independent of the cosmos_rl FlashAttnMeta singleton.
  Initializing that singleton has no effect on VLM training: the GPU Phase 2 smoke
  test (Qwen3-VL-8B-Instruct, 4-GPU FSDP2, 10 iters, loss 1.15→1.11) confirmed
  correct gradient flow without the singleton.

The deterministic flag IS honored here: it sets the standard PyTorch and CuDNN
determinism knobs, which is equivalent to what the old VLM trainer's
configure_training() did. VLMModel.__init__ calls this before _init_vlm(), so
determinism is configured before any compute happens.
"""

import os


def init_flash_attn_meta(deterministic: bool = False) -> None:
    """Initialize flash attention for the HFModel (VLM) training path.

    Replaces the cosmos_rl FlashAttnMeta singleton initialization (which is only
    relevant to the cosmos_rl model path, not HFModel). Also applies determinism
    settings when deterministic=True, matching what the old VLM trainer's
    configure_training() did.

    Args:
        deterministic: If True, enables PyTorch and CuDNN deterministic modes.
            HF flash_attention_2 respects torch.backends.cudnn.deterministic
            and torch.use_deterministic_algorithms().
    """
    if deterministic:
        import torch

        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)
        # Required for deterministic CuBLAS on CUDA >= 10.2
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
