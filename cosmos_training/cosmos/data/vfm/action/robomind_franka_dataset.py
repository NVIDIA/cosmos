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

"""RoboMIND Franka datasets for single-arm and dual-arm embodiments."""

from __future__ import annotations

import math
import os
from typing import Any, cast

import numpy as np
import torch
import torch.nn.functional as F

from cosmos.data.vfm.action.cosmos3_action_lerobot import (
    ActionNormalization,
    ActionSpec,
    BaseActionLeRobotDataset,
    Gripper,
    Pos,
    Rot,
    build_action_spec,
)
from cosmos.data.vfm.action.pose_utils import (
    PoseConvention,
    build_abs_pose_from_components,
    pose_abs_to_rel,
)
from cosmos.data.vfm.action.robomind_dataset_config import (
    ACTION_FEATURES,
    ALL_CAMERA_KEYS,
    LEROBOT_ROOTS,
    OBSERVATION_FEATURES,
)
from cosmos.data.vfm.action.viewpoint_utils import Viewpoint

_ROBOMIND_FRANKA_TO_OPENCV: np.ndarray = np.array(
    [
        [0.0, -1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)

FrankaInitialPose = torch.Tensor | tuple[torch.Tensor, torch.Tensor]


class RoboMINDFrankaDataset(BaseActionLeRobotDataset):
    """RoboMIND dataset for Franka single-arm and dual-arm embodiments."""

    SUPPORTED_EMBODIMENTS: tuple[str, str] = ("robomind-franka", "robomind-franka-dual")

    # RoboMIND-Franka has ~3x faster motion than the typical teleoperation
    # datasets (bridge / DROID / fractal). Empirically (see
    # ``debug/idle_test/recommend_thresholds_norm.txt``) the slowest 1 % of
    # motion sits at ~22 mm/s for single-arm and ~15 mm/s for dual-arm.
    #
    # **Dual-arm caveat**: the dual-arm dataset frequently has one arm's
    # state recorded as a near-zero stutter throughout a chunk (data quality
    # issue — only one arm is actually being teleoperated). Because the POS
    # branch uses the combined L2 across both arms, the threshold then
    # effectively becomes a per-arm threshold for whichever arm is active.
    # We compensate by tightening dual-arm to the global default (5 mm/s,
    # 1.5°/s) so a single arm doing a slow approach (~1mm/f at 10 Hz) is no
    # longer classified as idle.
    #
    # Class defaults below match single-arm. Dual-arm overrides at instance
    # construction (see ``__init__``).
    _IDLE_EPS_T_SINGLE: float = 22e-3
    _IDLE_EPS_R_SINGLE: float = math.radians(3.0)
    _IDLE_EPS_T_DUAL: float = 5e-3  # = base default; tight enough
    _IDLE_EPS_R_DUAL: float = math.radians(1.5)  # for "single-arm-slow" cases
    idle_eps_t_per_sec: float = _IDLE_EPS_T_SINGLE
    idle_eps_r_per_sec: float = _IDLE_EPS_R_SINGLE

    def __init__(
        self,
        root: str = "/lustre/fsw/portfolios/cosmos/projects/cosmos_base_training/cosmos3_action_datasets/RoboMIND_20251228",
        fps: float = 10.0,
        chunk_length: int = 16,
        split_seed: int = 42,
        split_val_ratio: float = 0.05,
        split: str = "train",
        mode: str = "policy",
        embodiment_type: str = "robomind-franka",
        pose_convention: str = "backward_framewise",
        action_normalization: ActionNormalization | None = None,
        viewpoint: Viewpoint = "concat_view",
        enable_fast_init: bool = False,
    ) -> None:
        if embodiment_type not in self.SUPPORTED_EMBODIMENTS:
            raise ValueError(
                f"RoboMINDFrankaDataset only supports {self.SUPPORTED_EMBODIMENTS}, "
                f"got embodiment_type={embodiment_type!r}"
            )

        super().__init__(
            fps=fps,
            chunk_length=chunk_length,
            split_seed=split_seed,
            split_val_ratio=split_val_ratio,
            split=split,
            mode=mode,
            embodiment_type=embodiment_type,
            viewpoint=viewpoint,
            pose_convention=pose_convention,
            rotation_format="rot6d",
            action_normalization=action_normalization,
            tolerance_s=1e-4,
            enable_fast_init=enable_fast_init,
        )

        self._to_opencv: np.ndarray = _ROBOMIND_FRANKA_TO_OPENCV[:3, :3]
        self._is_concat_view: bool = viewpoint == "concat_view"

        # Per-embodiment idle thresholds (instance-level override of the
        # class default which matches single-arm). Dual-arm tightens both
        # eps_t and eps_r to reflect its smaller per-frame motion tail.
        if embodiment_type == "robomind-franka-dual":
            self.idle_eps_t_per_sec = self._IDLE_EPS_T_DUAL
            self.idle_eps_r_per_sec = self._IDLE_EPS_R_DUAL

        embodiment_key = embodiment_type.removeprefix("robomind-")
        lerobot_roots = LEROBOT_ROOTS[embodiment_key]
        observation_features = list(OBSERVATION_FEATURES[embodiment_key])
        action_features = ACTION_FEATURES[embodiment_key]

        if self._is_concat_view and embodiment_key in ALL_CAMERA_KEYS:
            for cam_key in ALL_CAMERA_KEYS[embodiment_key]:
                if cam_key not in observation_features:
                    observation_features.append(cam_key)

        self._all_shard_roots = [os.path.join(root, shard_root) for shard_root in lerobot_roots]
        self._delta_timestamps = {
            **{key: [i * self._dt for i in range(0, self._chunk_length + 1)] for key in observation_features},
            **{key: [i * self._dt for i in range(0, self._chunk_length)] for key in action_features},
        }

    def _build_relative_poses(
        self,
        positions: torch.Tensor | np.ndarray,
        euler_xyz: torch.Tensor | np.ndarray,
    ) -> tuple[np.ndarray, torch.Tensor]:
        poses_abs = build_abs_pose_from_components(positions, euler_xyz, "euler_xyz")
        poses_abs[:, :3, :3] = poses_abs[:, :3, :3] @ self._to_opencv
        initial_pose = torch.from_numpy(poses_abs[0].copy()).float()
        pose_convention = cast(PoseConvention, self._pose_convention)
        poses_rel = cast(
            np.ndarray, pose_abs_to_rel(poses_abs, rotation_format="rot6d", pose_convention=pose_convention)
        )
        return poses_rel, initial_pose

    def _build_action(self, sample: dict[str, Any]) -> tuple[torch.Tensor, FrankaInitialPose]:
        state = sample["observation.states.end_effector"]
        gripper = sample["actions.joint_position"]

        if self._embodiment_type == "robomind-franka":
            poses_rel, initial_pose = self._build_relative_poses(state[:, 0:3], state[:, 3:6])
            action = torch.cat(
                [
                    torch.from_numpy(poses_rel).float(),
                    1.0 - gripper[:, [7]],
                ],
                dim=-1,
            )  # [T, 10]
            return action, initial_pose

        poses_rel_left, initial_pose_left = self._build_relative_poses(state[:, 0:3], state[:, 3:6])
        poses_rel_right, initial_pose_right = self._build_relative_poses(state[:, 6:9], state[:, 9:12])
        action = torch.cat(
            [
                torch.from_numpy(poses_rel_left).float(),
                1.0 - gripper[:, [7]],
                torch.from_numpy(poses_rel_right).float(),
                1.0 - gripper[:, [15]],
            ],
            dim=-1,
        )  # [T, 20]
        return action, (initial_pose_left, initial_pose_right)

    def _compose_multi_view_franka(self, sample: dict[str, Any]) -> torch.Tensor:  # returns [T,C,H',W']
        top_or_front_key = (
            "observation.images.camera_top"
            if self._embodiment_type == "robomind-franka"
            else "observation.images.camera_front"
        )
        top_or_front = sample[top_or_front_key]  # [T,C,H,W]
        left = sample["observation.images.camera_left"]  # [T,C,H,W]
        right = sample["observation.images.camera_right"]  # [T,C,H,W]

        _, _, height_ref, width_ref = top_or_front.shape
        half_height, half_width = height_ref // 2, width_ref // 2

        left = F.interpolate(
            left, size=(half_height, half_width), mode="bilinear", align_corners=False
        )  # [T,C,H/2,W/2]
        right = F.interpolate(
            right, size=(half_height, half_width), mode="bilinear", align_corners=False
        )  # [T,C,H/2,W/2]
        bottom = torch.cat([left, right], dim=-1)  # [T,C,H/2,W]

        composite = torch.cat([top_or_front, bottom], dim=-2)  # [T,C,3H/2,W]
        return composite  # [T,C,3H/2,W]

    def _build_action_spec(self) -> ActionSpec:
        """RoboMIND Franka: 10D single-arm or 20D dual-arm.

        Single (``robomind-franka``):
            ``[Pos, Rot6d, Gripper]``  (10D)

        Dual (``robomind-franka-dual``):
            ``[L_Pos, L_Rot6d, L_Gripper, R_Pos, R_Rot6d, R_Gripper]``  (20D)
        """
        if self._embodiment_type == "robomind-franka":
            return build_action_spec(Pos(), Rot("rot6d"), Gripper())
        # dual arm
        return build_action_spec(
            Pos(prefix="left"),
            Rot("rot6d", prefix="left"),
            Gripper(prefix="left"),
            Pos(prefix="right"),
            Rot("rot6d", prefix="right"),
            Gripper(prefix="right"),
        )

    def __getitem__(self, idx: int) -> dict[str, Any]:
        mode, _, _, sample = self._fetch_sample(idx)
        ai_caption = sample["task"]

        if self._skip_video_loading:
            video = None
            additional_view_description = None
        elif self._is_concat_view:
            video = self._compose_multi_view_franka(sample)
            additional_view_description = (
                "The top row shows third-person perspective view looking towards the robot from the front. "
                "The bottom-left video shows the third-person perspective view looking at the scene from the left side. "
                "The bottom-right video shows the third-person perspective view looking at the scene from the right side."
            )
        elif self._embodiment_type == "robomind-franka":
            video = sample["observation.images.camera_top"]  # [T,C,H,W]
            additional_view_description = None
        elif self._embodiment_type == "robomind-franka-dual":
            video = sample["observation.images.camera_front"]  # [T,C,H,W]
            additional_view_description = None
        else:
            raise ValueError(f"Unknown embodiment: {self._embodiment_type}")

        action, initial_pose = self._build_action(sample)

        extras: dict[str, Any] = {}
        if isinstance(initial_pose, tuple):
            extras["initial_pose"] = initial_pose[0]
            extras["initial_pose_right"] = initial_pose[1]
        else:
            extras["initial_pose"] = initial_pose

        if additional_view_description is not None:
            extras["additional_view_description"] = additional_view_description

        return self._build_result(
            mode=mode,
            video=video,
            action=action,
            ai_caption=ai_caption,
            **extras,
        )

    @property
    def action_dim(self) -> int:
        if self._embodiment_type == "robomind-franka":
            return 10
        if self._embodiment_type == "robomind-franka-dual":
            return 20
        raise ValueError(f"Unknown embodiment: {self._embodiment_type}")
