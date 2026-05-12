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

import attrs
import hydra

from cosmos3.args import DEFAULT_CHECKPOINT
from cosmos3.common.config import structure_config
from cosmos3.model import Cosmos3OmniConfig
from cosmos3._src.vfm.configs.base.defaults.model_config import ParallelismConfig


def test_config():
    parallelism = ParallelismConfig(
        data_parallel_shard_degree=2,
        context_parallel_shard_degree=2,
        cfg_parallel_shard_degree=2,
        use_torch_compile=True,
        use_cuda_graphs=True,
    )
    parallelism_kwargs = attrs.asdict(parallelism)
    checkpoint_path = DEFAULT_CHECKPOINT.download()
    config = Cosmos3OmniConfig.from_pretrained(checkpoint_path, parallelism=parallelism_kwargs)
    assert (
        hydra.utils.instantiate(structure_config(config.model["config"]["parallelism"], ParallelismConfig))
        == parallelism
    )
