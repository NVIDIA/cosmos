# Fractal (fractal20220817_data) — Google Robot RT-1 dataset
# LeRobot v2.0 format from IPEC-COMMUNITY/fractal20220817_data_lerobot
#
# Robot: google_robot
# 87,212 episodes, 3,786,400 frames, 599 tasks, fps=3
# state: [x, y, z, rx, ry, rz, rw, gripper]  (8D, quaternion)
# action: [x, y, z, roll, pitch, yaw, gripper] (7D, delta)
# video:  observation.images.image (256×320)

from typing import Any, cast

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
    pose_abs_to_rel,
)
from cosmos.data.vfm.action.viewpoint_utils import Viewpoint

_VALID_POSE_CONVENTIONS = ("backward_anchored", "backward_framewise")
# These episodes contain base motion, which breaks the fixed-base Google Robot
# action assumption used by training and the viewer.
_SKIPPED_EPISODE_IDS: frozenset[int] = frozenset({29, 189, 382})

# Google Robot raw EE frame has x/y axes rotated ~90° around z compared to
# OpenCV convention.  Rz(-90°) as a right-multiply corrects this:
#   new_x = -old_y (rightward), new_y = old_x (downward), z unchanged (approach).
_GOOGLE_ROBOT_TO_OPENCV = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=np.float32)

# ---------------------------------------------------------------------------
# TCP → flange (gripper body) offset
# ---------------------------------------------------------------------------
# The fractal dataset records EE poses at ``link_gripper_tcp`` — a calibrated
# tool-center-point 164 mm past the gripper body (``link_gripper``), roughly
# at the fingertip.  For action learning we re-reference poses to the
# *gripper body* (``link_gripper``) because:
#   1. It is the last actuated link — its pose is fully determined by joint
#      angles, whereas the TCP has a tiny calibration-dependent tilt (~0.25°).
#   2. Gripper body is a more natural frame for grasping tasks: the position
#      is at the wrist, not at the fragile fingertip.
#   3. The ~10 cm offset reduces the lever-arm effect of small rotation
#      errors on position accuracy.
#
# The constant below is the SE(3) transform from ``link_gripper_tcp`` to
# ``link_gripper``, computed from the SimplerEnv URDF via pinocchio FK at the
# neutral configuration:
#   T = oMf[link_gripper_tcp]⁻¹ · oMf[link_gripper]
#
# Source URDF: https://github.com/simpler-env/ManiSkill2_real2sim
#   → mani_skill2_real2sim/assets/descriptions/google_robot_description/
# fmt: off
_TCP_TO_FLANGE = np.array([
    [+0.9999897671, -0.0008686425, +0.0044397163, -0.0050618476],
    [+0.0008745501, +0.9999987346, -0.0013288658, -0.0016717725],
    [-0.0044385564, +0.0013327349, +0.9999892615, -0.1635144743],
    [+0.0000000000, +0.0000000000, +0.0000000000, +1.0000000000],
], dtype=np.float32)
# fmt: on


class FractalLeRobotDataset(BaseActionLeRobotDataset):
    """Action wrapper for the Fractal (Google RT-1) dataset in LeRobot format."""

    def __init__(
        self,
        root: str = "/lustre/fsw/portfolios/cosmos/projects/cosmos_base_training/cosmos3_action_datasets/fractal20220817_data_no_noops",
        fps: float = 3.0,
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
        """Initialize FractalLeRobotDataset.

        Args:
            root: Path to the local LeRobot dataset root.
            fps: Frames per second of the dataset.
            chunk_length: Number of action frames per sample.
            split_seed: Seed for deterministic train/val splitting.
            split_val_ratio: Fraction of episodes held out for validation.
            split: One of "train", "val", or "full".
            mode: Training mode — "policy", "forward_dynamics",
                "inverse_dynamics", "image2video", or "joint".
            pose_convention: Relative-pose convention used to encode SE(3)
                actions. Supports ``"backward_framewise"`` and
                ``"backward_anchored"``. Set to ``None`` to disable action
                construction outside image-to-video mode.
            action_normalization: Optional bundled-stats normalization
                (``"quantile"`` / ``"quantile_rot"`` / ``"meanstd"`` / ``"minmax"``);
                ``None`` returns raw actions.
            viewpoint: Camera viewpoint type for this dataset.
        """
        super().__init__(
            fps=fps,
            chunk_length=chunk_length,
            split_seed=split_seed,
            split_val_ratio=split_val_ratio,
            split=split,
            mode=mode,
            embodiment_type="fractal",
            viewpoint=viewpoint,
            pose_convention=pose_convention,
            rotation_format="rot6d",
            action_normalization=action_normalization,
            tolerance_s=1e-4,
            enable_fast_init=enable_fast_init,
        )

        self._to_opencv = _GOOGLE_ROBOT_TO_OPENCV
        self._all_shard_roots = [root]

        self._delta_timestamps = {
            "observation.images.image": [i * self._dt for i in range(0, self._chunk_length + 1)],
            "observation.state": [i * self._dt for i in range(0, self._chunk_length + 1)],
            "action": [i * self._dt for i in range(0, self._chunk_length)],
        }

    def _filter_valid_episodes(self, meta: LeRobotDatasetMetadata, episode_ids: list[int]) -> list[int]:
        """Drop known-bad raw Fractal episode IDs before index spans are built."""
        kept = [ep_id for ep_id in episode_ids if ep_id not in _SKIPPED_EPISODE_IDS]
        dropped = len(episode_ids) - len(kept)
        if dropped:
            log.info(
                f"FractalLeRobotDataset: dropped {dropped} / {len(episode_ids)} "
                f"episodes from skip list {sorted(_SKIPPED_EPISODE_IDS)} "
                f"(total_episodes={meta.total_episodes})"
            )
        return kept

    def _build_action_spec(self) -> ActionSpec:
        """Fractal: 10D = ``[Pos(3), Rot6d(6), Gripper(1)]``."""
        return build_action_spec(Pos(dim=3), Rot("rot6d"), Gripper())

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Return a single training sample."""
        mode, _, _, sample = self._fetch_sample(idx)

        ai_caption = sample["task"]

        video = sample["observation.images.image"]  # [T,C,H,W]

        # State layout: [x, y, z, rx, ry, rz, rw, gripper]  (T+1 frames)
        # Quaternion order from dataset: (rx, ry, rz, rw) matches scipy's (x, y, z, w).
        state = sample["observation.state"]  # [T+1, 8]
        poses_abs = build_abs_pose_from_components(
            state[:, 0:3],
            state[:, 3:7],
            "quat_xyzw",
        )
        # 1. TCP → flange: shift from link_gripper_tcp to link_gripper
        poses_abs = poses_abs @ _TCP_TO_FLANGE
        # 2. Kinematics → OpenCV convention (rotation only)
        poses_abs[:, :3, :3] = poses_abs[:, :3, :3] @ self._to_opencv
        initial_pose = torch.from_numpy(poses_abs[0].copy()).float()
        poses_rel = pose_abs_to_rel(
            poses_abs,
            rotation_format="rot6d",
            pose_convention=cast(PoseConvention, self._pose_convention),
        )
        action = torch.cat(
            [
                torch.from_numpy(poses_rel).float(),  # SE3 relative pose (rot6d)
                sample["action"][:, [6]],  # gripper (1D)
            ],
            dim=-1,
        )  # [T, 10]
        return self._build_result(
            mode=mode, video=video, action=action, ai_caption=ai_caption, initial_pose=initial_pose
        )

    @property
    def action_dim(self) -> int:
        """Action dimensionality: position(3) + 6D rotation(6) + gripper(1) = 10."""
        return 10
