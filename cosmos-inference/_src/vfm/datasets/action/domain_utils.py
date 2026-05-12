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

"""Domain ID helpers for cross-embodiment action datasets."""

EMBODIMENT_TO_DOMAIN_ID: dict[str, int] = {
    "no_action": 0,
    "av": 1,
    "camera_pose": 2,
    "hand_pose": 3,
    "pusht": 4,
    "libero": 5,
    "umi": 6,
    "bridge_orig_lerobot": 7,
    "droid_lerobot": 8,
    "robomind-franka": 8,  # Both Droid and RoboMIND-Franka are using robotiq and franka
    "embodiment_b": 9,
    "robomind-franka-dual": 12,
    "robomind-ur": 13,
    "agibotworld": 15,
    "embodiment_c_gripper": 15,
    "embodiment_c_gripper_ext": 15,
    "fractal": 20,
}


def get_domain_id(embodiment_type: str) -> int:
    """Get the domain ID for a given embodiment type."""
    key = embodiment_type.lower().strip()
    if key not in EMBODIMENT_TO_DOMAIN_ID:
        raise KeyError(
            f"Unknown embodiment type: {embodiment_type!r}. "
            f"Available embodiments: {sorted(EMBODIMENT_TO_DOMAIN_ID.keys())}"
        )
    return EMBODIMENT_TO_DOMAIN_ID[key]
