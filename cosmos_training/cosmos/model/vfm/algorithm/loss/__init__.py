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
"""Loss functions used by VFM (rectified flow) and VLM (next-token CE) training paths."""

__all__: list[str] = []

from cosmos.model.vfm.algorithm.loss.cross_entropy import cross_entropy_loss

__all__ += ["cross_entropy_loss"]

from cosmos.model.vfm.algorithm.loss.load_balancing import compute_load_balancing_loss

__all__ += ["compute_load_balancing_loss"]

from cosmos.model.vfm.algorithm.loss.time_weight import TrainTimeWeight

__all__ += ["TrainTimeWeight"]

from cosmos.model.vfm.algorithm.loss.flow_matching import compute_flow_matching_loss

__all__ += ["compute_flow_matching_loss"]
