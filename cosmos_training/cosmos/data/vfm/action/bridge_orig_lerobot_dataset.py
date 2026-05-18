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

# * https://github.com/2toinf/X-VLA/blob/30090f81cf91b15da73af234ce2b098fe20590f8/datasets/domain_handler/simulations.py#L70-L93
# * https://github.com/2toinf/X-VLA/issues/11
# * https://github.com/2toinf/X-VLA/issues/33
# * https://github.com/2toinf/X-VLA/issues/67
#

# uses identity stats (q01=-1, q99=1) on the 6D rotation dims 3..8, while
# ``"quantile_rot"`` uses the raw stats and normalizes those columns too.

from typing import Any

import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

from cosmos.utils import log
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
    convert_rotation,
    pose_abs_to_rel,
)
from cosmos.data.vfm.action.viewpoint_utils import Viewpoint

# Bridge rotation decomposition:
#   1) _DEFAULT_ROTATION: raw bridge state → kinematics (MJCF/URDF) frame.
#      The WidowX controller records ``R_state = R_fk @ DEFAULT_ROTATION.T``,
#      so ``R_fk = R_state @ DEFAULT_ROTATION``.
#   2) _TCP_TO_FLANGE: re-reference from ee_gripper_link to gripper_link
#      (pure translation in kinematics frame). See block below.
#   3) _KIN_TO_OPENCV: kinematics → OpenCV convention (for training/vis).
#      The viewer undoes this before IK to recover the kinematic frame.
_DEFAULT_ROTATION = np.array(
    [[0, 0, 1], [0, 1, 0], [-1, 0, 0]],
    dtype=np.float32,
)
_BRIDGE_TO_OPENCV = np.array(
    [[0, 0, 1], [-1, 0, 0], [0, -1, 0]],
    dtype=np.float32,
)

# ---------------------------------------------------------------------------
# TCP → flange (gripper body) offset
# ---------------------------------------------------------------------------
# The bridge dataset records EE poses at ``ee_gripper_link`` — the Interbotix
# SDK's end-effector reference, 93.6 mm past the wrist rotate body
# (``gripper_link``), roughly at the grasp center between the finger pads.
# For action learning we re-reference poses to the *wrist rotate body*
# (``gripper_link``) because:
#   1. It is the last actuated link — its pose is fully determined by joint
#      angles, with no dependence on finger opening.
#   2. The ~10 cm offset reduces the lever-arm effect of small rotation
#      errors on position accuracy.
#   3. Consistent with Google Robot, where we also target the gripper body.
#
# The constant below is the SE(3) transform from ``ee_gripper_link`` to
# ``gripper_link``, computed from the SimplerEnv URDF via pinocchio FK at the
# neutral configuration:
#   T = oMf[ee_gripper_link]⁻¹ · oMf[gripper_link]
# It is pure translation (identity rotation) — the two frames share the
# same orientation by construction (connected via fixed joints with no
# rotational offset).
#
# Source URDF: https://github.com/simpler-env/ManiSkill2_real2sim
#   → mani_skill2_real2sim/assets/descriptions/widowx_description/
#

# so the translation is expressed in the kinematic (MJCF) frame.
# fmt: off
_TCP_TO_FLANGE = np.array([
    [+1.0000000000, +0.0000000000, +0.0000000000, -0.0935750000],
    [+0.0000000000, +1.0000000000, +0.0000000000, +0.0000000000],
    [+0.0000000000, +0.0000000000, +1.0000000000, +0.0000000000],
    [+0.0000000000, +0.0000000000, +0.0000000000, +1.0000000000],
], dtype=np.float32)
# fmt: on


class BridgeOrigLeRobotDataset(BaseActionLeRobotDataset):
    """ """

    def __init__(
        self,
        root: str = "/lustre/fsw/portfolios/cosmos/projects/cosmos_base_training/cosmos3_action_datasets/bridge_raw",
        fps: float = 5.0,
        chunk_length: int = 16,
        split_seed: int = 42,
        split_val_ratio: float = 0.05,
        split: str = "train",
        mode: str = "policy",
        pose_convention: PoseConvention = "backward_framewise",
        action_normalization: ActionNormalization | None = None,
        viewpoint: Viewpoint = "ego_view",
        enable_fast_init: bool = False,
    ) -> None:
        """ """
        super().__init__(
            fps=fps,
            chunk_length=chunk_length,
            split_seed=split_seed,
            split_val_ratio=split_val_ratio,
            split=split,
            mode=mode,
            embodiment_type="bridge_orig_lerobot",
            viewpoint=viewpoint,
            pose_convention=pose_convention,
            rotation_format="rot6d",
            action_normalization=action_normalization,
            tolerance_s=1e-4,
            enable_fast_init=enable_fast_init,
        )
        # _to_opencv is the kinematics→OpenCV part only.
        # The viewer undoes this before IK → recovers kinematic frame directly.
        self._to_opencv = _BRIDGE_TO_OPENCV

        self._all_shard_roots = [root]

        self._delta_timestamps = {
            "observation.images.image_0": [i * self._dt for i in range(0, self._chunk_length + 1)],
            "observation.state": [i * self._dt for i in range(0, self._chunk_length + 1)],
            "action": [i * self._dt for i in range(0, self._chunk_length)],
        }

    # ------------------------------------------------------------------
    # Action computation
    # ------------------------------------------------------------------

    def _compute_absolute_action(self, sample: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        """Absolute action from state + gripper from action.

        EEF xyz+rotation come from observation.state[1:]; gripper from action[:, 6].

        Returns:
            (action_tensor, initial_pose) — initial_pose is the first-frame
            absolute EE pose (4×4, in the corrected OpenCV frame).
        """
        state = sample["observation.state"][1:]  # [T, 8]
        poses_abs = build_abs_pose_from_components(
            state[:, 0:3],
            state[:, 3:6],
            "euler_xyz",
        )

        # 1. Raw → kinematics: apply DEFAULT_ROTATION
        # 2. TCP → flange: shift from ee_gripper_link to gripper_link
        # 3. Kinematics → OpenCV convention (rotation only)
        poses_abs[:, :3, :3] = poses_abs[:, :3, :3] @ _DEFAULT_ROTATION.astype(poses_abs.dtype)
        poses_abs = poses_abs @ _TCP_TO_FLANGE.astype(poses_abs.dtype)
        poses_abs[:, :3, :3] = poses_abs[:, :3, :3] @ self._to_opencv.astype(poses_abs.dtype)

        initial_pose = torch.from_numpy(poses_abs[0].copy()).float()

        translation = torch.from_numpy(poses_abs[:, :3, 3]).float()
        rotation_matrix = torch.from_numpy(poses_abs[:, :3, :3]).float()
        rotation = convert_rotation(rotation_matrix, input_format="matrix", output_format="rot6d").float()

        pose = torch.cat([translation, rotation], dim=-1)  # [T, 9]
        return torch.cat([pose, sample["action"][:, [6]]], dim=-1), initial_pose  # [T, 10]

    def _compute_backward_framewise_action(self, sample: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        """Body-frame (ego-frame) delta: ``T_curr^{-1} @ T_next``.

        Matches Camera/AV ``backward_framewise`` convention. Translation is in
        the current frame's local coordinate system; rotation is
        ``R_curr^{-1} @ R_next``.

        Returns:
            (action_tensor, initial_pose) — initial_pose is the first-frame
            absolute EE pose (4×4, in the corrected OpenCV frame).
        """
        states = sample["observation.state"]  # (chunk_length + 1, 8)
        poses_abs = build_abs_pose_from_components(
            states[:, 0:3],
            states[:, 3:6],
            "euler_xyz",
        )

        # 1. Raw → kinematics: apply DEFAULT_ROTATION
        # 2. TCP → flange: shift from ee_gripper_link to gripper_link
        # 3. Kinematics → OpenCV convention (rotation only)
        poses_abs[:, :3, :3] = poses_abs[:, :3, :3] @ _DEFAULT_ROTATION.astype(poses_abs.dtype)
        poses_abs = poses_abs @ _TCP_TO_FLANGE.astype(poses_abs.dtype)
        poses_abs[:, :3, :3] = poses_abs[:, :3, :3] @ self._to_opencv.astype(poses_abs.dtype)

        initial_pose = torch.from_numpy(poses_abs[0].copy()).float()

        poses_rel = pose_abs_to_rel(
            poses_abs=poses_abs,
            rotation_format="rot6d",
            pose_convention="backward_framewise",
        )
        poses_rel_tensor = torch.from_numpy(poses_rel).float()

        return torch.cat([poses_rel_tensor, sample["action"][:, [6]]], dim=-1), initial_pose

    # ------------------------------------------------------------------
    # Normalization is handled by BaseActionLeRobotDataset.
    # Stats are loaded from:
    #   cosmos/data/vfm/action/normalizers/
    #       bridge_orig_lerobot_<pose_convention>_<rotation_format>.json
    # Regenerate via ``compute_action_stats.py`` + ``debug/stats_all.sh``.
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Episode filtering
    # ------------------------------------------------------------------
    def _filter_valid_episodes(self, meta: LeRobotDatasetMetadata, episode_ids: list[int]) -> list[int]:
        """Drop episodes whose ``tasks`` metadata is empty/whitespace.

        Narrower than the offline
        ``projects/cosmos3/vfm/datasets/action/filter_bridge_dataset.py``
        (which also flags gibberish/question/non-English/patterns via
        ``classify_task``).
        """
        kept: list[int] = []
        dropped = 0
        for ep_id in episode_ids:
            ep = meta.episodes[ep_id]
            tasks = ep.get("tasks", [])
            if isinstance(tasks, str):
                tasks = [tasks]
            has_prompt = any(t and str(t).strip() for t in (tasks or []))
            if has_prompt:
                kept.append(ep_id)
            else:
                dropped += 1
        if dropped:
            log.info(f"BridgeOrigLeRobotDataset: dropped {dropped} / {len(episode_ids)} episodes with empty prompt")
        return kept

    # ------------------------------------------------------------------
    # __getitem__
    # ------------------------------------------------------------------

    def _build_action_spec(self) -> ActionSpec:
        """Bridge: 10D = ``[Pos, Rot6d, Gripper]``."""
        return build_action_spec(Pos(), Rot("rot6d"), Gripper())

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """ """
        mode, _, _, sample = self._fetch_sample(idx)

        ai_caption = sample["task"]

        video = sample["observation.images.image_0"]  # [T,C,H,W]
        if self._pose_convention == "absolute":
            action, initial_pose = self._compute_absolute_action(sample)
        elif self._pose_convention == "backward_framewise":
            action, initial_pose = self._compute_backward_framewise_action(sample)
        else:
            raise ValueError(f"Unknown pose_convention: {self._pose_convention}")

        return self._build_result(
            mode=mode, video=video, action=action, ai_caption=ai_caption, initial_pose=initial_pose
        )

    @property
    def action_dim(self) -> int:
        """Action dimensionality: position(3) + 6D rotation(6) + gripper(1) = 10."""
        return 10
