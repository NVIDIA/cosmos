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
"""Hydra config registrations for VLM optimizer + LR scheduler.

Uses the shared ``build_optimizer`` / ``build_lr_scheduler`` from
``vfm/utils/optimizer.py``. Freeze configuration lives on the model node
(``VLMModelConfig.freeze``, type ``VLMFreezeConfig``) and is consumed by
``vlm_model._apply_freeze_config`` at model construction time.
"""

from cosmos.utils.lazy_config import PLACEHOLDER
from cosmos.utils.lazy_config import LazyCall as L
from cosmos.utils.config_helper import ConfigStore
from cosmos.utils.vfm.optimizer import build_lr_scheduler, build_optimizer

# VLM SFT recipe: vision_encoder backbone trained at 0.1x base LR; everything
# else at 1.0x. Substring-match style; see ``_filter_params_grouped``.
# Only entries with multiplier != 1.0 need to appear.
_VLM_LR_MULTIPLIERS: dict[str, float] = {"vision_encoder": 0.1}


def register_optimizer() -> None:
    cs = ConfigStore.instance()
    cs.store(
        group="optimizer",
        package="optimizer",
        name="fusedadamw",
        node=L(build_optimizer)(
            model=PLACEHOLDER,
            optimizer_type="FusedAdam",
            lr=2e-6,
            weight_decay=0.1,
            betas=(0.9, 0.95),
            fused=True,
            keys_to_select=[],
            lr_multipliers=_VLM_LR_MULTIPLIERS,
        ),
    )
    cs.store(
        group="optimizer",
        package="optimizer",
        name="adamw",
        node=L(build_optimizer)(
            model=PLACEHOLDER,
            optimizer_type="AdamW",
            lr=2e-6,
            weight_decay=0.1,
            betas=(0.9, 0.95),
            fused=True,
            keys_to_select=[],
            lr_multipliers=_VLM_LR_MULTIPLIERS,
        ),
    )


def register_scheduler() -> None:
    # f_start / f_min are ratios against the optimizer's base ``lr``:
    #     effective_init_lr = lr * f_start
    #     effective_end_lr  = lr * f_min
    # Update these together with ``lr`` if you want absolute LR endpoints to stay fixed.
    cs = ConfigStore.instance()
    cs.store(
        group="scheduler",
        package="scheduler",
        name="warmup_cosine_lr",
        node=L(build_lr_scheduler)(
            optimizer=PLACEHOLDER,
            lr_scheduler_type="LambdaCosine",
            warm_up_steps=[1000],
            cycle_lengths=["${trainer.max_iter}"],
            f_start=[0.05],
            f_max=[1.0],
            f_min=[0.5],
        ),
    )
