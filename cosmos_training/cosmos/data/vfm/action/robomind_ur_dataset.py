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

"""RoboMIND UR dataset for single-arm UR5e embodiment."""

from __future__ import annotations

import os
from typing import Any, cast

import numpy as np
import torch

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
    pose_abs_to_rel,
)
from cosmos.data.vfm.action.robomind_dataset_config import (
    ACTION_FEATURES,
    LEROBOT_ROOTS,
    OBSERVATION_FEATURES,
)
from cosmos.data.vfm.action.viewpoint_utils import Viewpoint

# UR EE frame → OpenCV convention rotation (3×3, post-multiplied).
# Identity: attachment_site (quat="-1 1 0 0" in ur5e_robotiq_2f85.xml) already
# satisfies OpenCV convention (z = approach)
_ROBOMIND_UR_TO_OPENCV: np.ndarray = np.eye(3, dtype=np.float32)

_UR5E_ARM_JOINTS = 6  # shoulder_pan … wrist_3
_UR5E_EE_SITE = "attachment_site"  # flange site in ur5e_robotiq_2f85.xml


class RoboMINDURDataset(BaseActionLeRobotDataset):
    """RoboMIND dataset for UR embodiment.

    Franka variants live in ``robomind_franka_dataset.py``.

    Action format: 10D ``[pos_delta(3) | rot6d_delta(6) | gripper(1)]``
    derived from MuJoCo FK of ``actions.joint_position`` (6 arm joints →
    ``attachment_site`` SE(3) pose).  ``observation.states.end_effector`` is
    NOT used because it is recorded incorrectly (constant) in ~89 % of UR
    episodes; ``actions.joint_position`` is valid for 100 % of episodes.

    The sample also contains ``joint_configs`` — absolute joint angles
    ``(T, 7)`` from ``actions.joint_position[1:T+1]`` — for FK-based robot
    mesh animation in the viewer.
    """

    SUPPORTED_EMBODIMENTS: tuple[str] = ("robomind-ur",)

    def __init__(
        self,
        root: str = "/lustre/fsw/portfolios/cosmos/projects/cosmos_base_training/cosmos3_action_datasets/RoboMIND_20251228",
        fps: float = 10.0,
        chunk_length: int = 16,
        split_seed: int = 42,
        split_val_ratio: float = 0.05,
        split: str = "train",
        mode: str = "policy",
        embodiment_type: str = "robomind-ur",
        pose_convention: str = "backward_framewise",
        action_normalization: ActionNormalization | None = None,
        viewpoint: Viewpoint = "third_person_view",
        enable_fast_init: bool = False,
    ) -> None:
        if embodiment_type not in self.SUPPORTED_EMBODIMENTS:
            raise ValueError(
                f"RoboMINDURDataset only supports {self.SUPPORTED_EMBODIMENTS}; "
                "use RoboMINDFrankaDataset for Franka variants. "
                f"Got embodiment_type={embodiment_type!r}."
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

        self._to_opencv: np.ndarray = _ROBOMIND_UR_TO_OPENCV

        embodiment_key = embodiment_type.removeprefix("robomind-")
        lerobot_roots = LEROBOT_ROOTS[embodiment_key]
        observation_features = list(OBSERVATION_FEATURES[embodiment_key])
        action_features = ACTION_FEATURES[embodiment_key]

        self._all_shard_roots = [os.path.join(root, x) for x in lerobot_roots]

        # T+1 joint positions: frame 0 is the initial state; frames 1..T are
        # the targets after each action step (used for both FK EE poses and mesh).
        _extended = frozenset({"actions.joint_position"})
        self._delta_timestamps = {
            **{k: [i * self._dt for i in range(0, self._chunk_length + 1)] for k in observation_features},
            **{
                k: [i * self._dt for i in range(0, self._chunk_length + (1 if k in _extended else 0))]
                for k in action_features
            },
        }

        # MuJoCo model for FK — loaded once per dataset instance.
        # We derive EE poses from FK on actions.joint_position rather than
        # observation.states.end_effector because the latter is recorded
        # incorrectly (frozen constant) in ~89 % of UR episodes across all 77
        # task shards (6,474 / 7,251 episodes).  actions.joint_position is
        # valid for 100 % of episodes and is the only reliable EE source.
        self._mj_model, self._mj_data, self._ee_site_id = self._init_mujoco()

    @staticmethod
    def _init_mujoco():
        """Load UR5e+Robotiq MuJoCo model (kinematics-only) and locate the EE site.

        Strips all geoms and mesh/texture/material assets from the MJCF via
        ``MjSpec`` before compile, so the model loads without any mesh files
        on disk. FK only needs the kinematic tree (bodies, joints, sites,
        inertials), so ``mj_forward`` + ``site_xpos``/``site_xmat`` still
        produce identical EE poses. Uses the committed XML directly to skip
        the Menagerie mesh download in ``get_mjcf_path``.
        """
        from pathlib import Path

        import mujoco

        mjcf_path = str(Path(__file__).parent / "urdf_visualizer" / "ur5e_robotiq_2f85.xml")
        spec = mujoco.MjSpec.from_file(mjcf_path)

        # Drop all geoms recursively — FK never touches them.
        def _strip_geoms(body):
            for g in list(body.geoms):
                spec.delete(g)
            for child in body.bodies:
                _strip_geoms(child)

        _strip_geoms(spec.worldbody)

        # Drop asset entries that reference external files.
        for m in list(spec.meshes):
            spec.delete(m)
        for t in list(spec.textures):
            spec.delete(t)
        for mat in list(spec.materials):
            spec.delete(mat)

        mj_model = spec.compile()
        mj_data = mujoco.MjData(mj_model)
        ee_site_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, _UR5E_EE_SITE)
        if ee_site_id < 0:
            raise RuntimeError(f"EE site '{_UR5E_EE_SITE}' not found in {mjcf_path}")
        return mj_model, mj_data, ee_site_id

    def _fk_ee_poses(self, arm_q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Run MuJoCo FK for T+1 arm configs → EE site positions and rotations.

        Args:
            arm_q: ``(T+1, 6)`` arm joint angles in radians.

        Returns:
            ``(positions (T+1, 3), rotations (T+1, 3, 3))`` in MuJoCo world frame.
        """
        import mujoco

        T1 = len(arm_q)
        positions = np.empty((T1, 3), dtype=np.float32)
        rotations = np.empty((T1, 3, 3), dtype=np.float32)
        for t in range(T1):
            self._mj_data.qpos[:_UR5E_ARM_JOINTS] = arm_q[t]
            mujoco.mj_forward(self._mj_model, self._mj_data)
            positions[t] = self._mj_data.site_xpos[self._ee_site_id]
            rotations[t] = self._mj_data.site_xmat[self._ee_site_id].reshape(3, 3)
        return positions, rotations

    def _build_action(self, sample: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build 10D UR action from FK EE poses derived from joint positions.

        Returns ``(action_10d, initial_pose, joint_configs)`` where:
        - ``action_10d`` is ``(T, 10)``: ``[pos_delta(3) | rot6d_delta(6) | gripper(1)]``
        - ``initial_pose`` is ``(4, 4)`` float32 tensor (FK EE pose at frame 0)
        - ``joint_configs`` is ``(T, 7)`` float32 — frames 1..T of joint positions
          (arm joints + raw gripper) for FK mesh animation in the viewer.
        """
        q = sample["actions.joint_position"]  # [T+1, 7]: 6 arm joints + 1 gripper
        q_np = q.numpy().astype(np.float32) if isinstance(q, torch.Tensor) else np.asarray(q, dtype=np.float32)
        T = len(q_np) - 1

        # FK EE trajectory: T+1 absolute poses from arm joints via MuJoCo
        fk_pos, fk_rot = self._fk_ee_poses(q_np[:, :_UR5E_ARM_JOINTS])

        poses_abs = np.tile(np.eye(4, dtype=np.float32), (T + 1, 1, 1))
        poses_abs[:, :3, 3] = fk_pos
        poses_abs[:, :3, :3] = fk_rot @ self._to_opencv

        initial_pose = torch.from_numpy(poses_abs[0].copy()).float()
        pose_convention = cast(PoseConvention, self._pose_convention)
        poses_rel = cast(
            np.ndarray, pose_abs_to_rel(poses_abs, rotation_format="rot6d", pose_convention=pose_convention)
        )

        # Raw UR gripper: 0=open, 1=closed (maps directly to Robotiq ctrl: raw*255,
        # where 0=open, 255=closed).  Invert so action uses 0=closed, 1=open
        # joint_configs keeps the raw value; FK mesh uses raw * 255 → Robotiq ctrl.
        gripper = torch.from_numpy(1.0 - q_np[:T, 6:7])
        action = torch.cat([torch.from_numpy(poses_rel).float(), gripper.float()], dim=-1)  # [T, 10]

        # Mesh animation: frames 1..T of joint position (post-action states)
        joint_configs = q_np[1 : 1 + T].copy()  # [T, 7]

        return action, initial_pose, torch.from_numpy(joint_configs)

    def _build_action_spec(self) -> ActionSpec:
        """RoboMIND UR: 10D = ``[Pos, Rot6d, Gripper]``."""
        return build_action_spec(Pos(), Rot("rot6d"), Gripper())

    def __getitem__(self, idx: int) -> dict[str, Any]:
        mode, _, _, sample = self._fetch_sample(idx)

        ai_caption = sample["task"]
        if self._skip_video_loading:
            video = None
        else:
            video = sample["observation.images.camera_top"]  # [T,C,H,W]

        action, initial_pose, joint_configs = self._build_action(sample)

        return self._build_result(
            mode=mode,
            video=video,
            action=action,
            ai_caption=ai_caption,
            initial_pose=initial_pose,
            joint_configs=joint_configs,
        )

    @property
    def action_dim(self) -> int:
        return 10  # 9D SE(3) EE deltas + 1 gripper
