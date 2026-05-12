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

"""
Copied from https://gitlab-master.nvidia.com/dir/imaginaire4/-/blob/d0921eb675d1251e73c4b19acdd78e6ad936ae3b/projects/cosmos/reason2/configs/base/defaults/optimizer.py without changes
"""

from cosmos3._src.imaginaire.configs.lr_scheduler import LambdaLinearSchedulerConfig
from cosmos3._src.imaginaire.functional.lr_scheduler import LambdaWarmUpCosineScheduler
from cosmos3._src.imaginaire.lazy_config import PLACEHOLDER
from cosmos3._src.imaginaire.lazy_config import LazyCall as L
from cosmos3._src.imaginaire.utils.config_helper import ConfigStore
from cosmos3._src.vfm.utils.optimizer import build_optimizer

optimizer_kwargs = dict(
    # Learning rate for the optimizer.
    lr=1e-4,
    # Weight decay for the optimizer.
    weight_decay=0.1,
    # Beta1 and beta2 for the optimizer.
    betas=[0.9, 0.99],
    # Epsilon for the optimizer.
    eps=1e-8,
    # Whether to use fuse updates to all parameters.
    fused=True,
    # Keys to select for the optimizer.
    keys_to_select=[],
    # Per-key LR multipliers. Maps parameter name patterns to LR multipliers.
    # E.g. {"sound2llm": 5.0, "llm2sound": 5.0} gives those params 5x the base LR.
    lr_multipliers={},
    # Whether to disable weight decay for one-dimensional params such as norm weights and biases.
    # Default is False to preserve historical optimizer behavior.
    disable_weight_decay_for_1d_params=False,
)


def register_optimizer():
    cs = ConfigStore.instance()
    cs.store(
        group="optimizer",
        package="optimizer",
        name="fusedadamw",
        node=L(build_optimizer)(
            model=PLACEHOLDER,
            optimizer_type="FusedAdam",
            **optimizer_kwargs,
        ),
    )
    cs.store(
        group="optimizer",
        package="optimizer",
        name="adamw",
        node=L(build_optimizer)(
            model=PLACEHOLDER,
            optimizer_type="AdamW",
            **optimizer_kwargs,
        ),
    )


def register_scheduler():
    cs = ConfigStore.instance()
    cs.store(group="scheduler", package="scheduler", name="lambdalinear", node=LambdaLinearSchedulerConfig)
    # Cosine scheduler that works with any optimizer (including fusedadamw)
    cs.store(
        group="scheduler",
        package="scheduler",
        name="lambdacosine",
        node=L(LambdaWarmUpCosineScheduler)(
            warm_up_steps=[2000],
            f_min=[0.0],
            f_max=[1.0],
            f_start=[0.0],
            cycle_lengths=[100000],
            verbosity_interval=0,
        ),
    )
