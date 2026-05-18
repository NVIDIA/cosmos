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

from typing import Union

import torch.nn as nn

from cosmos.model._base import ImaginaireModel
from cosmos.utils import distributed, log
from cosmos.utils.callback import Callback
from cosmos.utils.count_params import count_params
from cosmos.utils.distributed import rank0_only
from cosmos.utils.easy_io import easy_io


class ParamCount(Callback):
    def __init__(
        self,
        save_s3: bool = False,
    ):
        self.save_s3 = save_s3
        self.name = self.__class__.__name__

    @rank0_only
    def on_train_start(self, model: Union[ImaginaireModel, list[nn.Module]], iteration: int = 0) -> None:
        if isinstance(model, list):
            num_param = sum([count_params(m) for m in model])
        else:
            num_param = count_params(model)

        log.info(f"Total number of parameters on current rank: {num_param}", rank0_only=False)
        info = {
            "num_parameters": num_param,
        }

        if self.save_s3:
            rank = distributed.get_rank()
            easy_io.dump(info, f"s3://rundir/{self.name}_{rank}.yaml")
