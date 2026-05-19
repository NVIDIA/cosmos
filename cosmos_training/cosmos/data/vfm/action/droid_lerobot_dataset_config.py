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

_INSTITUTIONS = [
    "AUTOLab",
    "CLVR",
    "GuptaLab",
    "ILIAD",
    "IPRL",
    "IRIS",
    "PennPAL",
    "RAD",
    "RAIL",
    "REAL",
    "RPL",
    "TRI",
    "WEIRD",
]

LEROBOT_ROOTS = {
    "droid_lerobot_20260115_no_noops": None,
    "droid_plus_lerobot_320x180_20260406_sharded": [f"success/{x}" for x in _INSTITUTIONS]
    + [f"failure/{x}" for x in _INSTITUTIONS],
    "droid_plus_lerobot_320x180_20260406": ["success", "failure"],
    "droid_plus_lerobot_640x360_20260412_sharded": [f"success/{x}" for x in _INSTITUTIONS]
    + [f"failure/{x}" for x in _INSTITUTIONS],
    "droid_plus_lerobot_640x360_20260412": ["success", "failure"],
}

IMAGE_FEATURES = {
    "droid_lerobot_20260115_no_noops": {
        "wrist": "observation.images.wrist_image_left",
        "left": "observation.images.exterior_image_1_left",
        "right": "observation.images.exterior_image_2_left",
    },
    "droid_plus_lerobot_320x180_20260406_sharded": {
        "wrist": "observation.image.wrist_image_left",
        "left": "observation.image.exterior_image_1_left",
        "right": "observation.image.exterior_image_2_left",
    },
    "droid_plus_lerobot_320x180_20260406": {
        "wrist": "observation.image.wrist_image_left",
        "left": "observation.image.exterior_image_1_left",
        "right": "observation.image.exterior_image_2_left",
    },
    "droid_plus_lerobot_640x360_20260412_sharded": {
        "wrist": "observation.image.wrist_image_left",
        "left": "observation.image.exterior_image_1_left",
        "right": "observation.image.exterior_image_2_left",
    },
    "droid_plus_lerobot_640x360_20260412": {
        "wrist": "observation.image.wrist_image_left",
        "left": "observation.image.exterior_image_1_left",
        "right": "observation.image.exterior_image_2_left",
    },
}

STATE_FEATURES = {
    "droid_lerobot_20260115_no_noops": "observation.state",
    "droid_plus_lerobot_320x180_20260406_sharded": "observation.state.cartesian_position",
    "droid_plus_lerobot_320x180_20260406": "observation.state.cartesian_position",
    "droid_plus_lerobot_640x360_20260412_sharded": "observation.state.cartesian_position",
    "droid_plus_lerobot_640x360_20260412": "observation.state.cartesian_position",
}

ACTION_FEATURES = {
    "droid_lerobot_20260115_no_noops": "action",
    "droid_plus_lerobot_320x180_20260406_sharded": "action.gripper_position",
    "droid_plus_lerobot_320x180_20260406": "action.gripper_position",
    "droid_plus_lerobot_640x360_20260412_sharded": "action.gripper_position",
    "droid_plus_lerobot_640x360_20260412": "action.gripper_position",
}

IS_FLAT_ACTION = {
    "droid_lerobot_20260115_no_noops": True,
    "droid_plus_lerobot_320x180_20260406_sharded": False,
    "droid_plus_lerobot_320x180_20260406": False,
    "droid_plus_lerobot_640x360_20260412_sharded": False,
    "droid_plus_lerobot_640x360_20260412": False,
}

HAS_MULTI_LANGUAGE_ANNOTATIONS = {
    "droid_lerobot_20260115_no_noops": False,
    "droid_plus_lerobot_320x180_20260406_sharded": True,
    "droid_plus_lerobot_320x180_20260406": True,
    "droid_plus_lerobot_640x360_20260412_sharded": True,
    "droid_plus_lerobot_640x360_20260412": True,
}

IS_GRIPPER_ACTION_FLIPPED = {
    "droid_lerobot_20260115_no_noops": False,
    "droid_plus_lerobot_320x180_20260406_sharded": True,
    "droid_plus_lerobot_320x180_20260406": True,
    "droid_plus_lerobot_640x360_20260412_sharded": True,
    "droid_plus_lerobot_640x360_20260412": True,
}

_JOINT_ACTION_FEATURE = "action.joint_position"
_JOINT_STATE_FEATURE = "observation.state.joint_positions"
_GRIPPER_STATE_FEATURE = "observation.state.gripper_position"
