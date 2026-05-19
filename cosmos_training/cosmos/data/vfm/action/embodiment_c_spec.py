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

"""Shared Embodiment C metadata used by datasets and visualizers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

AgibotGearKind = Literal["gripper"]

AGIBOT_GEAR_VIDEO_KEY = "observation.images.top_head"
AGIBOT_GEAR_URDF_FILENAME = "G1_omnipicker_calibrated.urdf"
AGIBOT_GEAR_ROBOT_NAME = "embodiment_c"
AGIBOT_GEAR_GRIPPER_NORMALIZER_EMBODIMENT_TYPE = "embodiment_c_gripper"
AGIBOT_GEAR_ARM_STATE_SLICE = slice(0, 14)
AGIBOT_GEAR_STATE_HEAD_YAW_IDX = 16
AGIBOT_GEAR_STATE_HEAD_PITCH_IDX = 17
AGIBOT_GEAR_STATE_WAIST_PITCH_IDX = 18
AGIBOT_GEAR_STATE_WAIST_LIFT_IDX = 19
AGIBOT_GEAR_HEAD_PITCH_JOINT_NAME = "idx04_head_pitch_joint"

# -- Ext layout constants (94-dim state) -------------------------------------
# The ext split stores joints at different offsets from the standard layout.
AGIBOT_GEAR_EXT_ARM_STATE_SLICE = slice(54, 68)
AGIBOT_GEAR_EXT_STATE_HEAD_YAW_IDX = 82
AGIBOT_GEAR_EXT_STATE_HEAD_PITCH_IDX = 83
AGIBOT_GEAR_EXT_STATE_WAIST_PITCH_IDX = 84
AGIBOT_GEAR_EXT_STATE_WAIST_LIFT_IDX = 85
AGIBOT_GEAR_EXT_STATE_ROBOT_POSITION_SLICE = slice(86, 89)
AGIBOT_GEAR_EXT_STATE_ROBOT_ORIENTATION_SLICE = slice(89, 93)
AGIBOT_GEAR_EXT_STATE_LEFT_HAND_SLICE = slice(0, 1)
AGIBOT_GEAR_EXT_STATE_RIGHT_HAND_SLICE = slice(1, 2)
AGIBOT_GEAR_ARM_BASE_LINK_NAME = "arm_base_link"
AGIBOT_GEAR_HEAD_LINK_NAME = "head_pitch_link"
AGIBOT_GEAR_HEAD_CAMERA_LINK_NAME = "head_camera_link"
AGIBOT_GEAR_LEFT_EE_LINK_NAME = "gripper_l_base_link"
AGIBOT_GEAR_RIGHT_EE_LINK_NAME = "gripper_r_base_link"
AGIBOT_GEAR_LEFT_GRIPPER_CENTER_LINK_NAME = "gripper_l_center_link"
AGIBOT_GEAR_RIGHT_GRIPPER_CENTER_LINK_NAME = "gripper_r_center_link"
AGIBOT_GEAR_ARM_JOINT_NAMES_LEFT = tuple(f"idx{4 + i:02d}_left_arm_joint{i}" for i in range(1, 8))
AGIBOT_GEAR_ARM_JOINT_NAMES_RIGHT = tuple(f"idx{11 + i:02d}_right_arm_joint{i}" for i in range(1, 8))
AGIBOT_GEAR_WAIST_LIFT_JOINT_NAME = "idx01_waist_lift_joint"
AGIBOT_GEAR_WAIST_PITCH_JOINT_NAME = "idx02_waist_pitch_joint"
AGIBOT_GEAR_HEAD_YAW_JOINT_NAME = "idx03_head_yaw_joint"
AGIBOT_GEAR_GRIPPER_OPEN_ANGLE_RAD = math.pi / 4.0
AGIBOT_GEAR_GRIPPER_OPEN_ACTUATOR_DEG = 120.0
AGIBOT_GEAR_LEFT_GRIPPER_JOINT_MIMICS = (
    ("idx31_gripper_l_inner_joint1", 1.0, 0.0),
    ("idx32_gripper_l_inner_joint3", 0.1, 0.0),
    ("idx33_gripper_l_inner_joint4", 0.25, 0.0),
    ("idx39_gripper_l_inner_joint0", -0.7, 0.0),
    ("idx41_gripper_l_outer_joint1", -1.0, 0.0),
    ("idx42_gripper_l_outer_joint3", 0.1, 0.0),
    ("idx43_gripper_l_outer_joint4", -0.25, 0.0),
    ("idx49_gripper_l_outer_joint0", 0.7, 0.0),
)
AGIBOT_GEAR_RIGHT_GRIPPER_JOINT_MIMICS = (
    ("idx71_gripper_r_inner_joint1", 1.0, 0.0),
    ("idx72_gripper_r_inner_joint3", 0.1, 0.0),
    ("idx73_gripper_r_inner_joint4", 0.25, 0.0),
    ("idx79_gripper_r_inner_joint0", -0.7, 0.0),
    ("idx81_gripper_r_outer_joint1", -1.0, 0.0),
    ("idx82_gripper_r_outer_joint3", 0.1, 0.0),
    ("idx83_gripper_r_outer_joint4", -0.25, 0.0),
    ("idx89_gripper_r_outer_joint0", 0.7, 0.0),
)


@dataclass(frozen=True)
class AgibotGearKindSpec:
    """Layout metadata shared across all embodiments of one hand kind."""

    kind: AgibotGearKind
    state_hand_slice: slice


@dataclass(frozen=True)
class AgibotGearEmbodimentSpec:
    """Per-embodiment metadata shared by training and visualization code."""

    embodiment_type: str
    kind: AgibotGearKind
    action_dim: int


AGIBOT_GEAR_KIND_SPECS: dict[AgibotGearKind, AgibotGearKindSpec] = {
    "gripper": AgibotGearKindSpec(
        kind="gripper",
        state_hand_slice=slice(14, 16),
    ),
}

AGIBOT_GEAR_EMBODIMENT_SPECS: dict[str, AgibotGearEmbodimentSpec] = {
    "embodiment_c_gripper": AgibotGearEmbodimentSpec(
        embodiment_type="embodiment_c_gripper",
        kind="gripper",
        action_dim=29,  # FK output: head(9)+right(9)+gripper(1)+left(9)+gripper(1)
    ),
    "embodiment_c_gripper_ext": AgibotGearEmbodimentSpec(
        embodiment_type="embodiment_c_gripper_ext",
        kind="gripper",
        action_dim=29,  # FK output: head(9)+right(9)+gripper(1)+left(9)+gripper(1)
    ),
}


def get_embodiment_c_embodiment_spec(embodiment_type: str) -> AgibotGearEmbodimentSpec:
    """Return the registered spec for one AgiBot embodiment."""

    try:
        return AGIBOT_GEAR_EMBODIMENT_SPECS[embodiment_type]
    except KeyError as exc:
        raise ValueError(
            f"Unknown Embodiment C embodiment_type={embodiment_type!r}. "
            f"Expected one of {sorted(AGIBOT_GEAR_EMBODIMENT_SPECS)}."
        ) from exc


def get_embodiment_c_kind_spec(embodiment_type: str | AgibotGearKind) -> AgibotGearKindSpec:
    """Resolve an embodiment type or kind to its shared layout metadata."""

    kind = embodiment_type if embodiment_type in AGIBOT_GEAR_KIND_SPECS else get_embodiment_c_kind(embodiment_type)
    return AGIBOT_GEAR_KIND_SPECS[kind]


def get_embodiment_c_kind(embodiment_type: str) -> AgibotGearKind:
    """Return the hand kind used by one AgiBot embodiment."""

    return get_embodiment_c_embodiment_spec(embodiment_type).kind


def get_embodiment_c_action_dim(embodiment_type: str) -> int:
    """Return the action dimension for one AgiBot embodiment."""

    return get_embodiment_c_embodiment_spec(embodiment_type).action_dim


def get_embodiment_c_urdf_path() -> Path:
    """Return the committed Embodiment C G1 omnipicker URDF path."""

    return Path(__file__).resolve().parent / "urdf_visualizer" / AGIBOT_GEAR_URDF_FILENAME
