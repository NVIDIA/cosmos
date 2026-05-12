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

from dataclasses import dataclass


@dataclass
class BenchmarkTimes:
    """
    Class used to store times computed during tokenizer benchmarking.
    All times are in seconds.
    """

    model_invocation: float = 0.0
    # Model's invocation time + overhead
    total: float = 0.0

    @property
    def overhead(self) -> float:
        return self.total - self.model_invocation

    def __repr__(self) -> str:
        return f"BenchmarkTimes(model_invocation={self.model_invocation}, overhead={self.overhead}, total={self.total})"
