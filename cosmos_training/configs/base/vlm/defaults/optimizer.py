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

from cosmos.utils.lazy_config import PLACEHOLDER
from cosmos.utils.lazy_config import LazyCall as L
from cosmos.utils.config_helper import ConfigStore
from cosmos.utils.vfm.vlm.optimizer import OptimizerConfig, build_lr_schedulers, build_optimizers


def register_optimizer():
    cs = ConfigStore.instance()
    cs.store(
        group="optimizer",
        package="optimizer",
        name="fusedadamw",
        node=L(build_optimizers)(
            model_parts=PLACEHOLDER,
            model_part_names=PLACEHOLDER,
            config=L(OptimizerConfig)(
                name="FusedAdam",
            ),
        ),
    )
    cs.store(
        group="optimizer",
        package="optimizer",
        name="adamw",
        node=L(build_optimizers)(
            model_parts=PLACEHOLDER,
            model_part_names=PLACEHOLDER,
            config=L(OptimizerConfig)(
                name="AdamW",
            ),
        ),
    )


def register_scheduler():
    cs = ConfigStore.instance()
    cs.store(
        group="scheduler",
        package="scheduler",
        name="warmup_cosine_lr",
        node=L(build_lr_schedulers)(
            optimizers=PLACEHOLDER,
            name="warmup_cosine_lr",
            warmup_iters=1000,
            lr_decay_iters="${trainer.max_iter}",
            lr="${optimizer.config.lr}",
            init_lr="${optimizer.config.init_lr}",
            end_lr="${optimizer.config.end_lr}",
        ),
    )
