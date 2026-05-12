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

import gc

from cosmos3._src.imaginaire.callbacks.every_n import EveryN
from cosmos3._src.imaginaire.utils import log


class ManualGarbageCollection(EveryN):
    """
    Disable auto gc and manually trigger garbage collection every N iterations
    It is super useful for large scale training to reduce gpu sync time!
    Can reach 50% speedup.

    It is important to note that this callback only disables gc in main process and have auto gc enabled in subprocesses.

    We start disable gc after warm_up iterations to avoid disabling gc in subprocesses, such as dataloader, which can cause OOM
    """

    def __init__(self, *args, warm_up: int = 5, gc_level: int = 1, **kwargs):
        kwargs["barrier_after_run"] = False
        super().__init__(*args, **kwargs)

        self.counter = 0
        self.warm = warm_up
        self.gc_level = gc_level

    def every_n_impl(self, trainer, model, data_batch, output_batch, loss, iteration):
        del trainer, model, data_batch, output_batch, loss
        self.counter += 1
        if self.counter < self.warm:
            return
        if self.counter == self.warm:
            gc.disable()
            log.critical("Garbage collection disabled")

        gc.collect(self.gc_level)
