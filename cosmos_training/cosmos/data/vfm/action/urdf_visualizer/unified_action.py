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

"""Canonical 57D action representation with explicit input formats.

57D layout::

    [ego(9) | R_wrist(9) | R_fingers(15) | L_wrist(9) | L_fingers(15)]

Each 9D SE(3) slot is ``[pos(3) + rot6d(6)]``.
Each finger slot is 3D (position in wrist-local frame), 5 fingers × 3D = 15D.

Any supported action format is converted to ``UnifiedAction(action_57d, mask)``
before the viewer processes it. The mask explicitly declares which slots are valid.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import torch

from cosmos.utils import log
from cosmos.data.vfm.action.embodiment_c_fk import build_viewer_batch
from cosmos.data.vfm.action.embodiment_c_spec import get_embodiment_c_embodiment_spec
from cosmos.data.vfm.action.pose_utils import convert_rotation

FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")
ALL_FINGERS = (True, True, True, True, True)
NO_FINGERS = (False, False, False, False, False)


class ActionFormat(str, Enum):
    """Explicit raw action layouts supported by the viewer pipeline."""

    EGO_9D = "9d"
    SINGLE_ARM_10D = "10d"
    DUAL_ARM_20D = "20d"
    UNIFIED_57D = "57d"

    @property
    def expected_dim(self) -> int:
        """Return the exact trailing dimension required by this format."""
        return {
            ActionFormat.EGO_9D: 9,
            ActionFormat.SINGLE_ARM_10D: 10,
            ActionFormat.DUAL_ARM_20D: 20,
            ActionFormat.UNIFIED_57D: 57,
        }[self]


# ─── Data Structures ─────────────────────────────────────────────────────────


@dataclass
class Action57DMask:
    """Per-component validity for the 57D layout.

    ``right_fingers`` / ``left_fingers`` are tuples of 5 bools
    (thumb, index, middle, ring, pinky) — supports any combination from
    2-finger grippers to full 5-finger hands.
    """

    ego: bool = False
    right_wrist: bool = False
    right_fingers: tuple[bool, ...] = NO_FINGERS
    left_wrist: bool = False
    left_fingers: tuple[bool, ...] = NO_FINGERS


@dataclass
class UnifiedAction:
    """Canonical 57D action for the viewer pipeline.

    ``action`` is always shape ``(T, 57)`` with invalid slots zero-padded.
    ``gripper_right`` / ``gripper_left`` carry auxiliary scalar gripper data
    for embodiments that don't map to finger positions (V-shape visualisation).
    """

    action: np.ndarray  # (T, 57)
    mask: Action57DMask
    gripper_right: np.ndarray | None = None  # (T,) scalar 0-1
    gripper_left: np.ndarray | None = None  # (T,) scalar 0-1


@dataclass
class SceneState:
    """Render-ready world-space geometry reconstructed from ``UnifiedAction``.

    Contract:
    - all SE(3) trajectories live in one shared ``scene_world`` frame
    - fingertip positions are world-space if present
    - gripper signals are scalar open/close values sampled at ``T+1`` frames
    """

    mask: Action57DMask = field(default_factory=Action57DMask)
    # Absolute SE(3) trajectories — (T+1, 4, 4)
    ego_poses: np.ndarray | None = None
    right_poses: np.ndarray | None = None
    left_poses: np.ndarray | None = None
    # World-space fingertip positions — (T+1, 5, 3)
    right_fingers: np.ndarray | None = None
    left_fingers: np.ndarray | None = None
    # Scalar gripper — (T+1,)
    gripper_right: np.ndarray | None = None
    gripper_left: np.ndarray | None = None
    # Metadata
    video: np.ndarray | None = None  # (T+1, H, W, 3) uint8
    action_raw: np.ndarray | None = None  # canonical 57D action tensor for display
    T: int = 0
    # FK mesh animation: raw (T, nq) joint configs populated by datasets that
    # perform EE conversion internally (e.g. robomind-ur). When set, the renderer
    # uses these for FK mesh animation instead of running IK on right_poses.
    joint_configs: np.ndarray | None = None
    # AgiBot source-state FK mesh animation. When set, the renderer uses these
    # native URDF link poses directly instead of re-solving IK from pose targets.
    agibot_link_poses: dict[str, np.ndarray] | None = None


# ─── Converters ───────────────────────────────────────────────────────────────


def to_unified_from_57d(action: np.ndarray) -> UnifiedAction:
    """57D hand_pose → passthrough, all 5 slots valid."""
    return UnifiedAction(
        action=action.astype(np.float32),
        mask=Action57DMask(
            ego=True,
            right_wrist=True,
            right_fingers=ALL_FINGERS,
            left_wrist=True,
            left_fingers=ALL_FINGERS,
        ),
    )


def to_unified_from_10d(action: np.ndarray) -> UnifiedAction:
    """10D single arm ``[pos(3)+rot6d(6)+grip(1)]`` → right wrist + gripper."""
    T = action.shape[0]
    a = np.zeros((T, 57), dtype=np.float32)  # [T,57]
    a[:, 9:18] = action[:, :9]
    return UnifiedAction(
        action=a,
        mask=Action57DMask(right_wrist=True),
        gripper_right=action[:, 9].astype(np.float32),
    )


def to_unified_from_20d(action: np.ndarray) -> UnifiedAction:
    """20D dual arm ``[left(10) | right(10)]`` → both wrists + both grippers.

    Data layout: ``[L_pos(3) + L_rot6d(6) + L_grip(1) | R_pos(3) + R_rot6d(6) + R_grip(1)]``.
    Maps left arm → left wrist slot [33:42], right arm → right wrist slot [9:18].
    """
    T = action.shape[0]
    a = np.zeros((T, 57), dtype=np.float32)  # [T,57]
    a[:, 33:42] = action[:, :9]  # left arm → left wrist slot [33:42]
    a[:, 9:18] = action[:, 10:19]  # right arm → right wrist slot [9:18]
    return UnifiedAction(
        action=a,
        mask=Action57DMask(right_wrist=True, left_wrist=True),
        gripper_right=action[:, 19].astype(np.float32),  # right arm gripper
        gripper_left=action[:, 9].astype(np.float32),  # left arm gripper
    )


def to_unified_from_9d(action: np.ndarray) -> UnifiedAction:
    """9D camera/AV ``[pos(3)+rot6d(6)]`` → ego only."""
    T = action.shape[0]
    a = np.zeros((T, 57), dtype=np.float32)  # [T,57]
    a[:, 0:9] = action[:, :9]
    return UnifiedAction(
        action=a,
        mask=Action57DMask(ego=True),
    )


def _validate_action_shape(action: np.ndarray, action_format: ActionFormat) -> None:
    """Raise when a raw action tensor does not match its declared format."""
    if action.ndim != 2:
        raise ValueError(f"Expected a rank-2 action array, got shape {action.shape}")
    actual_dim = int(action.shape[-1])
    expected_dim = action_format.expected_dim
    if actual_dim != expected_dim:
        raise ValueError(f"Action format {action_format.value} expects trailing dim {expected_dim}, got {actual_dim}")


def to_unified(action: np.ndarray, action_format: ActionFormat) -> UnifiedAction:
    """Convert one explicit raw action format into ``UnifiedAction``."""
    _validate_action_shape(action, action_format)
    if action_format is ActionFormat.UNIFIED_57D:
        return to_unified_from_57d(action)
    if action_format is ActionFormat.DUAL_ARM_20D:
        return to_unified_from_20d(action)
    if action_format is ActionFormat.EGO_9D:
        return to_unified_from_9d(action)
    if action_format is ActionFormat.SINGLE_ARM_10D:
        return to_unified_from_10d(action)
    raise ValueError(f"Unsupported action format: {action_format}")


def _poses_to_pose9d(poses: np.ndarray) -> np.ndarray:
    """Convert ``(T,4,4)`` absolute transforms to ``(T,9)`` pos+rot6d rows."""

    positions = poses[:, :3, 3].astype(np.float32)
    rotations = convert_rotation(poses[:, :3, :3], input_format="matrix", output_format="rot6d")
    rotations = np.asarray(rotations, dtype=np.float32)
    return np.concatenate([positions, rotations], axis=-1).astype(np.float32, copy=False)


def to_unified_from_embodiment_c(
    sample: dict[str, Any],
    *,
    embodiment_type: str,
) -> UnifiedAction:
    """Convert one AgiBot sample to the canonical viewer representation."""

    viewer_batch = build_viewer_batch(sample, embodiment_type)
    embodiment_spec = get_embodiment_c_embodiment_spec(embodiment_type)
    T = int(viewer_batch.head_camera_poses.shape[0])

    action = np.zeros((T, 57), dtype=np.float32)
    action[:, 0:9] = _poses_to_pose9d(viewer_batch.head_camera_poses)
    action[:, 9:18] = _poses_to_pose9d(viewer_batch.right_wrist_poses)
    action[:, 33:42] = _poses_to_pose9d(viewer_batch.left_wrist_poses)

    return UnifiedAction(
        action=action,
        mask=Action57DMask(
            ego=True,
            right_wrist=True,
            left_wrist=True,
        ),
        gripper_right=viewer_batch.right_gripper if embodiment_spec.kind == "gripper" else None,
        gripper_left=viewer_batch.left_gripper if embodiment_spec.kind == "gripper" else None,
    )


def to_unified_from_embodiment_c_fk_action(action: np.ndarray, kind: str = "gripper") -> UnifiedAction:
    """Embodiment C FK-pose action → unified action.

    Relative input layout (gripper, 29D):
        ``[head_delta(9), right_delta(9), right_gripper(1), left_delta(9), left_gripper(1)]``

    These inputs already store ``rot6d`` SE(3) blocks and are copied directly
    into the unified slots.
    """
    T = action.shape[0]
    a = np.zeros((T, 57), dtype=np.float32)

    if kind == "gripper" and action.shape[1] == 29:
        a[:, 0:9] = action[:, 0:9]
        a[:, 9:18] = action[:, 9:18]
        a[:, 33:42] = action[:, 19:28]
        return UnifiedAction(
            action=a,
            mask=Action57DMask(ego=True, right_wrist=True, left_wrist=True),
            gripper_right=action[:, 18].astype(np.float32),
            gripper_left=action[:, 28].astype(np.float32),
        )

    if kind == "gripper":
        raise ValueError(f"Unsupported AgiBot gripper action dim {action.shape[1]}; expected 29.")
    raise ValueError(f"Unsupported AgiBot kind {kind!r}; expected 'gripper'.")


# ─── Trajectory Reconstruction ────────────────────────────────────────────────


def _pos_rot6d_to_mat(se3: np.ndarray) -> np.ndarray:
    """Convert ``(N, 9)`` pos+rot6d to ``(N, 4, 4)`` SE(3) matrices."""
    N = se3.shape[0]
    pos = se3[:, :3]
    r6 = se3[:, 3:9]

    col0 = r6[:, :3].copy()
    col0_norm = np.linalg.norm(col0, axis=-1, keepdims=True) + 1e-8
    col0 = col0 / col0_norm

    col1 = r6[:, 3:6] - np.sum(r6[:, 3:6] * col0, axis=-1, keepdims=True) * col0
    col1_norm = np.linalg.norm(col1, axis=-1, keepdims=True) + 1e-8
    col1 = col1 / col1_norm

    col2 = np.cross(col0, col1)

    mats = np.tile(np.eye(4, dtype=np.float32), (N, 1, 1))
    mats[:, :3, 0] = col0
    mats[:, :3, 1] = col1
    mats[:, :3, 2] = col2
    mats[:, :3, 3] = pos
    return mats


def _chain_se3(
    deltas: np.ndarray,
    initial_pose: np.ndarray | None = None,
    pose_convention: str = "backward_framewise",
) -> np.ndarray:
    """Chain ``(T, 9)`` relative deltas into ``(T+1, 4, 4)`` absolute poses.

    For ``backward_framewise``: ``P_{t+1} = P_t @ delta_t``.
    For ``absolute``: each row is already an absolute pose (no chaining).
    """
    T = deltas.shape[0]
    delta_mats = _pos_rot6d_to_mat(deltas)

    if initial_pose is None:
        initial_pose = np.eye(4, dtype=np.float32)
    else:
        initial_pose = initial_pose.astype(np.float32)

    poses = np.empty((T + 1, 4, 4), dtype=np.float32)
    poses[0] = initial_pose

    if pose_convention == "absolute":
        poses[1:] = delta_mats
    else:
        for t in range(T):
            poses[t + 1] = poses[t] @ delta_mats[t]

    return poses


def _extract_fingers(raw: np.ndarray) -> np.ndarray:
    """``(T, 15)`` → ``(T+1, 5, 3)`` with first frame duplicated."""
    T = raw.shape[0]
    fingers = raw.reshape(T, 5, 3).astype(np.float32)  # [T,5,3]
    return np.concatenate([fingers[:1], fingers], axis=0)


def _to_numpy_float32(value: object) -> np.ndarray:
    """Convert a tensor-like value to a float32 NumPy array."""

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().astype(np.float32)
    return np.asarray(value, dtype=np.float32)


def _quat_xyzw_to_rotmat(q: np.ndarray) -> np.ndarray:
    """Convert ``(N, 4)`` xyzw quaternions to ``(N, 3, 3)`` rotation matrices."""
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.zeros((len(q), 3, 3), dtype=np.float32)
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - z * w)
    R[:, 0, 2] = 2 * (x * z + y * w)
    R[:, 1, 0] = 2 * (x * y + z * w)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - x * w)
    R[:, 2, 0] = 2 * (x * z - y * w)
    R[:, 2, 1] = 2 * (y * z + x * w)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def _build_absolute_from_overlay(sample: dict) -> dict[str, np.ndarray] | None:
    """Build absolute world-frame poses from HandPoseDataset overlay data.

    Returns None if overlay keys are missing.
    """
    raw_cam_pos = sample.get("raw_cam_position")
    if raw_cam_pos is None:
        return None

    cam_pos = _to_numpy_float32(raw_cam_pos)  # [T+1,3]
    cam_rot_q = _to_numpy_float32(sample["raw_cam_rotation"])  # [T+1,4]
    right_3d = _to_numpy_float32(sample["raw_cam_right_3d"])  # [T+1,63]
    left_3d = _to_numpy_float32(sample["raw_cam_left_3d"])  # [T+1,63]
    right_rot = _to_numpy_float32(sample["raw_cam_right_rot"])  # [T+1,84]
    left_rot = _to_numpy_float32(sample["raw_cam_left_rot"])  # [T+1,84]

    T1 = cam_pos.shape[0]
    FTIP = [4, 8, 12, 16, 20]

    # Camera c2w (world frame)
    cam_c2w = np.tile(np.eye(4, dtype=np.float32), (T1, 1, 1))  # [T+1,4,4]
    cam_c2w[:, :3, 3] = cam_pos
    cam_c2w[:, :3, :3] = _quat_xyzw_to_rotmat(cam_rot_q)

    def _wrist_world(pos_63, rot_84):
        wrist_pos = pos_63[:, :3]
        wrist_q = rot_84.reshape(T1, 21, 4)[:, 0]
        wrist_cam = np.tile(np.eye(4, dtype=np.float32), (T1, 1, 1))  # [T+1,4,4]
        wrist_cam[:, :3, 3] = wrist_pos
        wrist_cam[:, :3, :3] = _quat_xyzw_to_rotmat(wrist_q)
        return cam_c2w @ wrist_cam

    def _fingers_world(pos_63):
        joints = pos_63.reshape(T1, 21, 3)[:, FTIP]
        R = cam_c2w[:, :3, :3]
        t = cam_c2w[:, :3, 3]
        return np.einsum("tij,tfj->tfi", R, joints) + t[:, None, :]  # [T+1,5,3]

    return {
        "ego_poses": cam_c2w,
        "right_wrist_poses": _wrist_world(right_3d, right_rot),
        "left_wrist_poses": _wrist_world(left_3d, left_rot),
        "right_fingers": _fingers_world(right_3d),
        "left_fingers": _fingers_world(left_3d),
    }


# ─── Scene State Builder ─────────────────────────────────────────────────────


def build_scene_state(
    unified: UnifiedAction,
    initial_pose: np.ndarray | None = None,
    initial_pose_right: np.ndarray | None = None,
    initial_pose_left: np.ndarray | None = None,
    right_base_pose: np.ndarray | None = None,
    left_base_pose: np.ndarray | None = None,
    pose_convention: str = "backward_framewise",
    sample: dict | None = None,
) -> SceneState:
    """Reconstruct a canonical world-space ``SceneState`` from ``UnifiedAction``.

    Chains SE(3) deltas for valid mask slots. If ``sample`` contains overlay
    data (HandPoseDataset raw camera/joint fields), overrides with absolute
    world-frame poses.

    Args:
        unified: Canonical 57D action with mask.
        initial_pose: Default initial pose for all slots.
        initial_pose_right: Override for right wrist (dual arm).
        initial_pose_left: Override for left wrist (dual arm).
        right_base_pose: Right-arm base pose that maps arm-local trajectories into ``scene_world``.
        left_base_pose: Left-arm base pose that maps arm-local trajectories into ``scene_world``.
        pose_convention: Pose convention for SE(3) chaining.
        sample: Raw dataset sample (for overlay data).
    """

    def _apply_pose_base(poses: np.ndarray | None, base_pose: np.ndarray | None) -> np.ndarray | None:
        if poses is None or base_pose is None:
            return poses
        return np.einsum("ij,njk->nik", base_pose, poses).astype(np.float32)  # [T+1,4,4]

    def _fingers_local_to_world(
        fingers_local: np.ndarray | None,
        wrist_poses_world: np.ndarray | None,
    ) -> np.ndarray | None:
        if fingers_local is None:
            return None
        if wrist_poses_world is None:
            raise ValueError("Finger trajectories require matching wrist poses to build world-space SceneState")
        wrist_rot = wrist_poses_world[:, :3, :3].astype(np.float32)  # [T+1,3,3]
        wrist_pos = wrist_poses_world[:, :3, 3].astype(np.float32)  # [T+1,3]
        return np.einsum("tij,tfj->tfi", wrist_rot, fingers_local) + wrist_pos[:, None, :]  # [T+1,5,3]

    mask = unified.mask
    action = unified.action
    state = SceneState(mask=mask)

    ip_default = initial_pose if initial_pose is not None else np.eye(4, dtype=np.float32)
    ip_right = initial_pose_right if initial_pose_right is not None else ip_default
    ip_left = initial_pose_left if initial_pose_left is not None else ip_default

    if mask.ego:
        state.ego_poses = _chain_se3(action[:, 0:9], ip_default, pose_convention)
    if mask.right_wrist:
        state.right_poses = _chain_se3(action[:, 9:18], ip_right, pose_convention)
    if any(mask.right_fingers):
        state.right_fingers = _extract_fingers(action[:, 18:33])
    if mask.left_wrist:
        state.left_poses = _chain_se3(action[:, 33:42], ip_left, pose_convention)
    if any(mask.left_fingers):
        state.left_fingers = _extract_fingers(action[:, 42:57])

    if unified.gripper_right is not None:
        g = unified.gripper_right
        state.gripper_right = np.concatenate([[g[0]], g]).astype(np.float32, copy=False)  # [T+1]
    if unified.gripper_left is not None:
        g = unified.gripper_left
        state.gripper_left = np.concatenate([[g[0]], g]).astype(np.float32, copy=False)  # [T+1]

    abs_data = _build_absolute_from_overlay(sample) if sample is not None else None
    if abs_data is not None:
        state.ego_poses = abs_data["ego_poses"]
        state.right_poses = abs_data["right_wrist_poses"]
        state.left_poses = abs_data["left_wrist_poses"]
        state.right_fingers = abs_data["right_fingers"]
        state.left_fingers = abs_data["left_fingers"]
        log.info(
            f"Overlay absolute mode | ego range: "
            f"[{abs_data['ego_poses'][:, :3, 3].min():.3f}, "
            f"{abs_data['ego_poses'][:, :3, 3].max():.3f}] | "
            f"R wrist[0]: {abs_data['right_wrist_poses'][0, :3, 3]}"
        )
    else:
        state.right_poses = _apply_pose_base(state.right_poses, right_base_pose)
        state.left_poses = _apply_pose_base(state.left_poses, left_base_pose)
        state.right_fingers = _fingers_local_to_world(state.right_fingers, state.right_poses)
        state.left_fingers = _fingers_local_to_world(state.left_fingers, state.left_poses)

    raw_agibot_link_poses = sample.get("agibot_link_poses") if sample is not None else None
    if isinstance(raw_agibot_link_poses, dict):
        state.agibot_link_poses = {
            str(link_name): _to_numpy_float32(link_poses) for link_name, link_poses in raw_agibot_link_poses.items()
        }  # {name:[T+1,4,4]}

    state.action_raw = unified.action.astype(np.float32)
    state.T = action.shape[0]
    return state


# ─── Video Extraction ─────────────────────────────────────────────────────────


def get_video_from_sample(sample: dict) -> np.ndarray | None:
    """Extract video frames from a dataset sample.

    Returns ``(T+1, H, W, 3)`` uint8 array, or None.
    """
    video = sample.get("video")
    if video is None:
        return None
    if isinstance(video, torch.Tensor):
        video = video.numpy()

    if video.ndim == 4:
        C, T_dim, H, W = video.shape
        if C in (1, 3) and T_dim > 3:
            video = np.transpose(video, (1, 2, 3, 0))
        elif video.shape[1] in (1, 3) and T_dim <= 3:
            video = np.transpose(video, (0, 2, 3, 1))

    if video.dtype in (np.float32, np.float64):
        video = np.clip(video * 255, 0, 255).astype(np.uint8)

    if video.ndim == 4 and video.shape[-1] == 1:
        video = np.repeat(video, 3, axis=-1)

    return video
