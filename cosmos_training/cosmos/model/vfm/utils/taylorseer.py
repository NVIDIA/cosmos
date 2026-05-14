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

# Adapted from https://github.com/Shenyi-Z/TaylorSeer/blob/main/TaylorSeers-xDiT/taylorseer_flux/taylorseer_utils/__init__.py

import math
from typing import Dict

import torch


def derivative_approximation(cache_dic: Dict, current: Dict, feature: torch.Tensor):
    """
    Compute derivative approximation.

    :param cache_dic: Cache dictionary
    :param current: Information of the current step
    """
    difference_distance = current["activated_steps"][-1] - current["activated_steps"][-2]

    updated_taylor_factors = {}
    updated_taylor_factors[0] = feature

    for i in range(cache_dic["max_order"]):
        if (
            cache_dic["cache"][-1][current["stream"]][current["layer"]][current["module"]].get(i, None) is not None
        ) and (current["step"] > cache_dic["first_enhance"] - 2):
            updated_taylor_factors[i + 1] = (
                updated_taylor_factors[i]
                - cache_dic["cache"][-1][current["stream"]][current["layer"]][current["module"]][i]
            ) / difference_distance
        else:
            break

    cache_dic["cache"][-1][current["stream"]][current["layer"]][current["module"]] = updated_taylor_factors


def taylor_formula(cache_dic: Dict, current: Dict) -> torch.Tensor:
    """
    Compute Taylor expansion error.

    :param cache_dic: Cache dictionary
    :param current: Information of the current step
    """
    x = current["step"] - current["activated_steps"][-1]
    # x = current['t'] - current['activated_times'][-1]
    output = 0

    for i in range(len(cache_dic["cache"][-1][current["stream"]][current["layer"]][current["module"]])):
        output += (
            (1 / math.factorial(i))
            * cache_dic["cache"][-1][current["stream"]][current["layer"]][current["module"]][i]
            * (x**i)
        )

    return output


def taylor_cache_init(cache_dic: Dict, current: Dict):
    """
    Initialize Taylor cache and allocate storage for different-order derivatives in the Taylor cache.

    :param cache_dic: Cache dictionary
    :param current: Information of the current step
    """
    if (current["step"] == 0) and (cache_dic["taylor_cache"]):
        cache_dic["cache"][-1][current["stream"]][current["layer"]][current["module"]] = {}


# Copied from https://github.com/Shenyi-Z/TaylorSeer/blob/main/TaylorSeers-xDiT/taylorseer_flux/cache_functions/force_scheduler.py


def force_scheduler(cache_dic, current):
    if cache_dic["fresh_ratio"] == 0:
        # FORA
        linear_step_weight = 0.0
    else:
        # TokenCache
        linear_step_weight = 0.0
    step_factor = torch.tensor(1 - linear_step_weight + 2 * linear_step_weight * current["step"] / current["num_steps"])
    threshold = torch.round(cache_dic["fresh_threshold"] / step_factor)

    # no force constrain for sensitive steps, cause the performance is good enough.
    # you may have a try.

    cache_dic["cal_threshold"] = threshold
    # return threshold


# Copied from https://github.com/Shenyi-Z/TaylorSeer/blob/main/TaylorSeers-xDiT/taylorseer_flux/cache_functions/cal_type.py


def cal_type(cache_dic, current):
    """
    Determine calculation type for this step
    """
    if (cache_dic["fresh_ratio"] == 0.0) and (not cache_dic["taylor_cache"]):
        # FORA:Uniform
        first_step = current["step"] == 0
    else:
        # ToCa: First enhanced
        first_step = current["step"] < cache_dic["first_enhance"]

    if not first_step:
        fresh_interval = cache_dic["cal_threshold"]
    else:
        fresh_interval = cache_dic["fresh_threshold"]

    if (first_step) or (cache_dic["cache_counter"] == fresh_interval - 1):
        current["type"] = "full"
        cache_dic["cache_counter"] = 0
        current["activated_steps"].append(current["step"])
        force_scheduler(cache_dic, current)

    elif cache_dic["taylor_cache"]:
        cache_dic["cache_counter"] += 1
        current["type"] = "Taylor"

    elif cache_dic["cache_counter"] % 2 == 1:  # 0: ToCa-Aggresive-ToCa, 1: Aggresive-ToCa-Aggresive
        cache_dic["cache_counter"] += 1
        current["type"] = "ToCa"
    # 'cache_noise' 'ToCa' 'FORA'
    elif cache_dic["Delta-DiT"]:
        cache_dic["cache_counter"] += 1
        current["type"] = "Delta-Cache"
    else:
        cache_dic["cache_counter"] += 1
        current["type"] = "ToCa"


# Modified from https://github.com/Shenyi-Z/TaylorSeer/blob/main/TaylorSeers-xDiT/taylorseer_flux/cache_functions/cache_init.py


def cache_init(self, num_steps: int):
    """
    Initialization for cache.
    """
    cache_dic = {}
    cache = {}
    cache_index = {}
    cache[-1] = {}
    cache_index[-1] = {}
    cache_index["layer_index"] = {}
    cache[-1]["layers_stream"] = {}
    cache_dic["cache_counter"] = 0

    for j in range(len(self.language_model.model.layers)):
        cache[-1]["layers_stream"][j] = {}
        cache_index[-1][j] = {}

    cache_dic["Delta-DiT"] = False
    cache_dic["cache_type"] = "random"
    cache_dic["cache_index"] = cache_index
    cache_dic["cache"] = cache
    cache_dic["fresh_ratio_schedule"] = "ToCa"
    cache_dic["fresh_ratio"] = 0.0
    cache_dic["fresh_threshold"] = 3
    cache_dic["soft_fresh_weight"] = 0.0
    cache_dic["taylor_cache"] = True
    cache_dic["max_order"] = 6
    cache_dic["first_enhance"] = 5

    current = {}
    current["activated_steps"] = [0]
    current["step"] = 0
    current["num_steps"] = num_steps

    return cache_dic, current
