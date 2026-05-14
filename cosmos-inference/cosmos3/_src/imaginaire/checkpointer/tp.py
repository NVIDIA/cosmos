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

from cosmos3._src.imaginaire.checkpointer.ddp import Checkpointer as DDPCheckpointer
from cosmos3._src.imaginaire.model import ImaginaireModel


class Checkpointer(DDPCheckpointer):
    """
    Checkpointer class for Tensor Parallelism (TP) in distributed training.

    This implementation supports the combination of Tensor Parallelism (TP) and Data Parallel Processing (DDP), with optional Context Parallelism (CP).

    Note:
    - Fully Sharded Data Parallelism (FSDP) is not supported by this checkpointer.
    - In principle, this implementation is also compatible with Pipeline Parallelism (PP) and Expert Parallelism (EP), which are other forms of model parallelism. However, PP and EP have not been tested yet.
    """

    def add_type_postfix_to_checkpoint_path(self, key: str, checkpoint_path: str, model: ImaginaireModel) -> str:
        """
        Overwrite the `add_type_postfix_to_checkpoint_path` function of the base class (DDP checkpointer)
        to append the TP-rank postfix to the checkpoint path.
        """
        checkpoint_path = super().add_type_postfix_to_checkpoint_path(key, checkpoint_path, model)
        if key == "trainer":
            return checkpoint_path
        else:
            checkpoint_path = checkpoint_path.replace(".pt", f"_mp_{self.mp_rank}.pt")

        return checkpoint_path
