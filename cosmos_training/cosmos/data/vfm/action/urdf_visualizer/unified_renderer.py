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

"""Mask-driven renderer for ``SceneState`` — the unified 57D viewer backend.

Owns all viser scene handles. Draws only what the mask declares valid.
No action-format branching — everything goes through ``SceneState``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from cosmos.utils import log
from cosmos.data.vfm.action.embodiment_c_fk import DEFAULT_GRIPPER_LENGTH_M
from cosmos.data.vfm.action.embodiment_c_spec import get_embodiment_c_kind
from cosmos.data.vfm.action.urdf_visualizer.ik_solver import (
    compute_agibot_link_poses_batch_from_configs,
    solve_agibot_trajectory_ik,
)
from cosmos.data.vfm.action.urdf_visualizer.robot_scene_model import RobotSceneModel
from cosmos.data.vfm.action.urdf_visualizer.unified_action import FINGER_NAMES, SceneState
from cosmos.data.vfm.action.urdf_visualizer.urdf_loader import (
    get_embodiment_c_collision_geometries,
)


def _agibot_gripper_tip_positions(
    pos: np.ndarray,
    rot: np.ndarray,
    opening: float,
    max_finger_width: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Approximate AgiBot gripper tip sphere positions from an OpenCV wrist pose."""

    half_w = float(opening) * max_finger_width / 2.0
    gripper_center = pos + DEFAULT_GRIPPER_LENGTH_M * rot[:, 2]  # [3]
    tip_l = gripper_center + rot @ np.array([half_w, 0.0, 0.0], dtype=np.float32)  # [3]
    tip_r = gripper_center + rot @ np.array([-half_w, 0.0, 0.0], dtype=np.float32)  # [3]
    return tip_l.astype(np.float32, copy=False), tip_r.astype(np.float32, copy=False)


def _get_entry_agibot_embodiment_type(entry: Any) -> str:
    """Return the AgiBot robot embodiment declared by a viewer dataset entry."""

    robot_embodiment_type = getattr(entry, "robot_embodiment_type", None)
    if isinstance(robot_embodiment_type, str) and robot_embodiment_type:
        return robot_embodiment_type

    dataset_kwargs = getattr(entry, "dataset_kwargs", {})
    if isinstance(dataset_kwargs, dict):
        embodiment_type = dataset_kwargs.get("embodiment_type", "")
        if isinstance(embodiment_type, str):
            return embodiment_type

    return ""


def _get_wrist_to_opencv(to_opencv: object, wrist_name: str) -> np.ndarray | None:
    """Resolve a native-to-OpenCV wrist rotation from renderer dataset metadata."""

    if isinstance(to_opencv, dict):
        rotation = to_opencv.get(wrist_name)
        if isinstance(rotation, np.ndarray):
            return rotation.astype(np.float32, copy=False)  # [3,3]
        return None
    if isinstance(to_opencv, np.ndarray):
        return to_opencv.astype(np.float32, copy=False)  # [3,3]
    return None


def _undo_wrist_to_opencv_for_ik(
    poses: np.ndarray | None,
    to_opencv: object,
    wrist_name: str,
) -> np.ndarray | None:
    """Convert viewer/OpenCV wrist rotations back to native frame for IK."""

    if poses is None:
        return None
    wrist_to_opencv = _get_wrist_to_opencv(to_opencv, wrist_name)
    if wrist_to_opencv is None or np.allclose(wrist_to_opencv, np.eye(3, dtype=np.float32)):
        return poses
    native_poses = poses.astype(np.float32, copy=True)  # [T,4,4]
    native_poses[:, :3, :3] = native_poses[:, :3, :3] @ wrist_to_opencv.T[None]  # [T,3,3]
    return native_poses


class UnifiedRenderer:
    """Mask-driven 3D renderer for the unified 57D action viewer."""

    # ── Palette ──
    TIP_RADIUS = 0.0042
    FRUSTUM_SCALE = 0.10
    EGO_AXIS_LENGTH = 0.04
    EGO_AXIS_RADIUS = 0.002
    EGO_TRAJ_LENGTH_REFERENCE = 0.55
    EGO_TRAJ_SCALE_MIN = 0.35
    EGO_TRAJ_SCALE_MAX = 1.80
    EGO_TRAJ_LINE_WIDTH = 1.0
    EE_TRAJ_LINE_WIDTH = 1.5
    HAND_AXIS_LENGTH = 0.05
    HAND_AXIS_RADIUS = 0.002
    ROBOT_BODY_AXIS_LENGTH = 0.03
    ROBOT_BODY_AXIS_RADIUS = 0.0012
    ROBOT_SITE_AXIS_LENGTH = 0.022
    ROBOT_SITE_AXIS_RADIUS = 0.0009
    COLOR_EGO = (52, 152, 219)  # blue
    COLOR_EGO_TOP = (231, 76, 60)  # red
    COLOR_RIGHT = (243, 156, 18)  # orange
    COLOR_LEFT = (155, 89, 182)  # purple
    FINGER_COLORS = [
        (231, 76, 60),
        (241, 196, 15),
        (46, 204, 113),
        (52, 152, 219),
        (155, 89, 182),
    ]

    @staticmethod
    def _soften_color(color: tuple[int, int, int], mix: float = 0.35) -> tuple[int, int, int]:
        """Blend a color toward white for less visually dominant trajectories."""
        base = np.asarray(color, dtype=np.float32)
        soft = base * (1.0 - mix) + 255.0 * mix
        rounded = soft.round()
        return int(rounded[0]), int(rounded[1]), int(rounded[2])

    def __init__(self, server):
        import viser.transforms as vtf

        self.server = server
        self.vtf = vtf
        self.state: SceneState | None = None
        self._entry: Any = None
        self._axis_scale = 1.0
        self._ego_frustum_scale_base = self.FRUSTUM_SCALE
        self._ego_axis_length_base = self.EGO_AXIS_LENGTH
        self._ego_axis_radius_base = self.EGO_AXIS_RADIUS
        self._ego_frustum_fov = np.deg2rad(60.0)
        self._ego_frustum_aspect = 4 / 3

        # ── Ego (blue) ──
        self.ego_frame = server.scene.add_frame(
            "/ego/frame",
            axes_length=self.EGO_AXIS_LENGTH,
            axes_radius=self.EGO_AXIS_RADIUS,
        )
        self.ego_frustum = server.scene.add_line_segments(
            "/ego/frustum",
            points=self._make_ego_frustum_wireframe_points(self._ego_frustum_fov, self._ego_frustum_aspect),
            colors=np.array(self.COLOR_EGO, dtype=np.uint8),
            scale=self.FRUSTUM_SCALE,
            line_width=2.0,
            wxyz=(1.0, 0.0, 0.0, 0.0),
            position=(0.0, 0.0, 0.0),
        )
        self.ego_frustum_up = server.scene.add_line_segments(
            "/ego/frustum_up",
            points=self._make_ego_frustum_up_points(self._ego_frustum_fov, self._ego_frustum_aspect),
            colors=np.array(self.COLOR_EGO_TOP, dtype=np.uint8),
            line_width=3.0,
            scale=self.FRUSTUM_SCALE,
            wxyz=(1.0, 0.0, 0.0, 0.0),
            position=(0.0, 0.0, 0.0),
        )
        self.ego_traj = server.scene.add_spline_catmull_rom(
            "/ego/traj",
            positions=np.zeros([2, 3], dtype=np.float32),
            color=self._soften_color(self.COLOR_EGO),
            line_width=self.EGO_TRAJ_LINE_WIDTH,
        )

        # ── Right effector (green) ──
        self.right_frame = server.scene.add_frame(
            "/right/frame",
            axes_length=self.HAND_AXIS_LENGTH,
            axes_radius=self.HAND_AXIS_RADIUS,
        )
        self.right_traj = server.scene.add_spline_catmull_rom(
            "/right/traj",
            positions=np.zeros([2, 3], dtype=np.float32),
            color=self._soften_color(self.COLOR_RIGHT),
            line_width=self.EE_TRAJ_LINE_WIDTH,
        )
        self.right_ee = server.scene.add_point_cloud(
            "/right/point",
            points=np.zeros([1, 3], dtype=np.float32),
            colors=np.array([self.COLOR_RIGHT], dtype=np.uint8),
            point_size=0.015,
            point_shape="circle",
        )
        self.right_fingers = [
            server.scene.add_icosphere(
                f"/right/finger_{FINGER_NAMES[i]}",
                radius=self.TIP_RADIUS,
                color=self.FINGER_COLORS[i],
                position=(0.0, 0.0, 0.0),
            )
            for i in range(5)
        ]
        self.right_gripper_tips = [
            server.scene.add_icosphere(
                f"/right/gripper_tip_{side}",
                radius=self.TIP_RADIUS,
                color=self.FINGER_COLORS[i],
                position=(0.0, 0.0, 0.0),
            )
            for i, side in enumerate(("thumb", "index"))
        ]

        # ── Left effector (red) ──
        self.left_frame = server.scene.add_frame(
            "/left/frame",
            axes_length=self.HAND_AXIS_LENGTH,
            axes_radius=self.HAND_AXIS_RADIUS,
        )
        self.left_traj = server.scene.add_spline_catmull_rom(
            "/left/traj",
            positions=np.zeros([2, 3], dtype=np.float32),
            color=self._soften_color(self.COLOR_LEFT),
            line_width=self.EE_TRAJ_LINE_WIDTH,
        )
        self.left_ee = server.scene.add_point_cloud(
            "/left/point",
            points=np.zeros([1, 3], dtype=np.float32),
            colors=np.array([self.COLOR_LEFT], dtype=np.uint8),
            point_size=0.015,
            point_shape="circle",
        )
        self.left_fingers = [
            server.scene.add_icosphere(
                f"/left/finger_{FINGER_NAMES[i]}",
                radius=self.TIP_RADIUS,
                color=self.FINGER_COLORS[i],
                position=(0.0, 0.0, 0.0),
            )
            for i in range(5)
        ]
        self.left_gripper_tips = [
            server.scene.add_icosphere(
                f"/left/gripper_tip_{side}",
                radius=self.TIP_RADIUS,
                color=self.FINGER_COLORS[i],
                position=(0.0, 0.0, 0.0),
            )
            for i, side in enumerate(("thumb", "index"))
        ]

        # ── IK robot meshes ──
        self.robot_right: list = []
        self.robot_left: list = []
        self._robot_frame_handles_right: dict[str, Any] = {}
        self._robot_frame_handles_left: dict[str, Any] = {}
        self._current_robot: Any | None = None
        self._robot_scene_model: RobotSceneModel | None = None
        self._ik_right: list | None = None
        self._ik_left: list | None = None
        self._robot_frames_right: dict[str, list[tuple[np.ndarray, np.ndarray]]] | None = None
        self._robot_frames_left: dict[str, list[tuple[np.ndarray, np.ndarray]]] | None = None
        self._robot_link_names: list[str] = []
        self._robot_local_transforms: list[np.ndarray] = []

        # ── Video panel (set by viewer.py) ──
        self._cam_handle: Any | None = None

        self.hide_all()

    # ─── Per-Episode ──────────────────────────────────────────────────────────

    def load(
        self,
        state: SceneState,
        entry: Any,
        to_opencv: np.ndarray | dict[str, np.ndarray] | None = None,
    ):
        """Load a new episode. Rebuild trajectories, robot meshes, and IK.

        Args:
            state: Reconstructed ``SceneState`` with absolute poses.
            entry: ``DatasetEntry`` with robot_name, max_finger_width, etc.
            to_opencv: Optional native-to-OpenCV rotation. AgiBot can provide
                side-specific ``{"left_wrist": R, "right_wrist": R}`` values.
        """
        self.state = state
        self._entry = entry
        self._to_opencv = to_opencv if to_opencv is not None else np.eye(3, dtype=np.float32)
        self._ego_frustum_fov = np.deg2rad(entry.camera_fov_deg)
        self._ego_frustum_aspect = float(entry.camera_aspect)
        self.ego_frustum.points = self._make_ego_frustum_wireframe_points(
            self._ego_frustum_fov,
            self._ego_frustum_aspect,
        )
        self.ego_frustum_up.points = self._make_ego_frustum_up_points(
            self._ego_frustum_fov,
            self._ego_frustum_aspect,
        )
        self._update_ego_visual_scale(state.ego_poses if state.mask.ego else None)
        self.update_axis_scale(self._axis_scale)
        self.hide_all()

        # ── Robot meshes + IK from canonical world-space SceneState ──
        self._ik_right = None
        self._ik_left = None
        self._robot_frames_right = None
        self._robot_frames_left = None
        if entry.robot_name:
            self._load_robot_and_ik(state, entry)

        # Rebuild trajectory splines from canonical world-space SceneState.
        if state.mask.ego and state.ego_poses is not None:
            self._rebuild_traj(self.ego_traj, state.ego_poses, self.COLOR_EGO)
        if state.mask.right_wrist and state.right_poses is not None:
            self._rebuild_traj(self.right_traj, state.right_poses, self.COLOR_RIGHT)
        if state.mask.left_wrist and state.left_poses is not None:
            self._rebuild_traj(self.left_traj, state.left_poses, self.COLOR_LEFT)

    def set_video_panel(self, panel_handle: Any | None) -> None:
        """Attach the optional GUI image panel used for episode video."""
        self._cam_handle = panel_handle

    # ─── Per-Frame ────────────────────────────────────────────────────────────

    def update(self, t: int, show: dict):
        """Update all scene elements for time step ``t``.

        Args:
            t: Frame index (0-based).
            show: Visibility flags: ``frames``, ``traj``, ``fingertips``, ``ego``, ``robot``.
        """
        state = self.state
        if state is None:
            return
        mask = state.mask

        # ── Ego ──
        self._update_ego(t, state.ego_poses, mask.ego and show.get("ego", False), show)

        # ── Right effector ──
        self._update_effector(
            t,
            state.right_poses,
            mask.right_wrist,
            self.right_frame,
            self.right_ee,
            self.right_traj,
            show,
        )
        self._update_fingers(
            t,
            state.right_fingers,
            mask.right_fingers,
            self.right_fingers,
            show,
        )
        self._update_gripper(
            t,
            state.right_poses,
            state.gripper_right,
            mask.right_wrist,
            mask.right_fingers,
            self.right_gripper_tips,
            show,
        )

        # ── Left effector ──
        self._update_effector(
            t,
            state.left_poses,
            mask.left_wrist,
            self.left_frame,
            self.left_ee,
            self.left_traj,
            show,
        )
        self._update_fingers(
            t,
            state.left_fingers,
            mask.left_fingers,
            self.left_fingers,
            show,
        )
        self._update_gripper(
            t,
            state.left_poses,
            state.gripper_left,
            mask.left_wrist,
            mask.left_fingers,
            self.left_gripper_tips,
            show,
        )

        # ── IK robot meshes ──
        self._update_robot(t, show)

        # ── Video panel ──
        if self._cam_handle is not None and state.video is not None and t < len(state.video):
            self._cam_handle.image = state.video[t]

    # ─── Action Text ──────────────────────────────────────────────────────────

    def format_action_text(self, t: int) -> str:
        """Return a formatted string showing 57D action values at step ``t``.

        Always shows the full 57D layout. Validity indicator (✓/·) in front of
        each component based on the mask.
        """
        state = self.state
        if state is None or state.action_raw is None:
            return ""
        if t == 0:
            return "*t=0: anchor pose (identity)*"
        if (t - 1) >= len(state.action_raw):
            return ""

        a = state.action_raw[t - 1]  # always 57D (zero-padded)
        mask = state.mask

        def _fmt(v):
            return " ".join(f"{x:+.4f}" for x in v)

        def _v(active):
            return "✓" if active else "·"

        gr = a[18:33].reshape(5, 3)
        gl = a[42:57].reshape(5, 3)

        # Gripper auxiliary values (not in 57D vector)
        grip_r_str = ""
        grip_l_str = ""
        if state.gripper_right is not None and t < len(state.gripper_right):
            grip_r_str = f"  ✓ gripper          {state.gripper_right[t]:+.4f}"
        if state.gripper_left is not None and t < len(state.gripper_left):
            grip_l_str = f"  ✓ gripper          {state.gripper_left[t]:+.4f}"

        parts = [
            f"step {t - 1} → {t}  (57D)",
            "═" * 36,
            f"{_v(mask.ego)} Ego   pos [0:3]    {_fmt(a[0:3])}",
            f"  {' ' * 1}  rot [3:9]    {_fmt(a[3:9])}",
            "",
            f"{_v(mask.right_wrist)} R wrist pos [9:12]   {_fmt(a[9:12])}",
            f"  {' ' * 1}    rot [12:18]  {_fmt(a[12:18])}",
            f"  R fingers [18:33]",
        ]
        for i, name in enumerate(FINGER_NAMES):
            parts.append(f"  {_v(mask.right_fingers[i])} {name:7s} {_fmt(gr[i])}")
        if grip_r_str:
            parts.append(grip_r_str)

        parts += [
            "",
            f"{_v(mask.left_wrist)} L wrist pos [33:36]  {_fmt(a[33:36])}",
            f"  {' ' * 1}    rot [36:42]  {_fmt(a[36:42])}",
            f"  L fingers [42:57]",
        ]
        for i, name in enumerate(FINGER_NAMES):
            parts.append(f"  {_v(mask.left_fingers[i])} {name:7s} {_fmt(gl[i])}")
        if grip_l_str:
            parts.append(grip_l_str)

        return str("```\n" + "\n".join(parts) + "\n```")

    # ─── Private: Effector ────────────────────────────────────────────────────

    @staticmethod
    def _make_ego_frustum_wireframe_points(fov: float, aspect: float) -> np.ndarray:
        """Build wireframe segments for the ego camera frustum."""
        half_height = float(np.tan(fov / 2.0))
        half_width = float(aspect) * half_height
        top_left = np.array([-half_width, -half_height, 1.0], dtype=np.float32)
        top_right = np.array([half_width, -half_height, 1.0], dtype=np.float32)
        bottom_right = np.array([half_width, half_height, 1.0], dtype=np.float32)
        bottom_left = np.array([-half_width, half_height, 1.0], dtype=np.float32)
        origin = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        return np.array(
            [
                [origin, top_left],
                [origin, top_right],
                [origin, bottom_right],
                [origin, bottom_left],
                [top_left, top_right],
                [top_right, bottom_right],
                [bottom_right, bottom_left],
                [bottom_left, top_left],
            ],
            dtype=np.float32,
        )

    @staticmethod
    def _make_ego_frustum_up_points(fov: float, aspect: float) -> np.ndarray:
        """Build a red segment that marks the frustum's far-edge upright tick."""
        half_height = float(np.tan(fov / 2.0))
        _ = aspect
        top_y = -half_height
        return np.array(
            [[[0.0, top_y, 1.0], [0.0, top_y * 1.18, 1.0]]],
            dtype=np.float32,
        )

    def _update_ego(self, t: int, poses: np.ndarray | None, active: bool, show: dict) -> None:
        if active and poses is not None and t < len(poses):
            pos = poses[t, :3, 3]
            rot = poses[t, :3, :3]
            wxyz = self.vtf.SO3.from_matrix(rot).wxyz
            self.ego_frame.position = pos
            self.ego_frame.wxyz = wxyz
            self.ego_frame.visible = show.get("frames", True)
            self.ego_frustum.position = pos
            self.ego_frustum.wxyz = wxyz
            self.ego_frustum.visible = True
            self.ego_frustum_up.position = pos
            self.ego_frustum_up.wxyz = wxyz
            self.ego_frustum_up.visible = True
            self.ego_traj.visible = show.get("traj", True)
        else:
            self.ego_frame.visible = False
            self.ego_frustum.visible = False
            self.ego_frustum_up.visible = False
            self.ego_traj.visible = False

    def _update_effector(self, t, poses, active, frame, ee, traj, show):
        if active and poses is not None and t < len(poses):
            pos = poses[t, :3, 3]
            rot = poses[t, :3, :3]
            frame.position = pos
            frame.wxyz = self.vtf.SO3.from_matrix(rot).wxyz
            frame.visible = show.get("frames", True)
            ee.points = pos[None]
            ee.visible = True
            traj.visible = show.get("traj", True)
        else:
            frame.visible = False
            ee.visible = False
            traj.visible = False

    # ─── Private: Fingers ─────────────────────────────────────────────────────

    def _update_fingers(self, t, fingers, finger_mask, handles, show):
        if fingers is None or t >= len(fingers):
            for h in handles:
                h.visible = False
            return
        if not show.get("fingertips", True):
            for h in handles:
                h.visible = False
            return

        g = fingers[t]  # (5, 3)
        for fi, h in enumerate(handles):
            if finger_mask[fi]:
                h.position = g[fi].astype(np.float32)
                h.visible = True
            else:
                h.visible = False

    # ─── Private: Gripper ─────────────────────────────────────────────────────

    def _update_gripper(self, t, poses, gripper, wrist_active, finger_mask, handle, show):
        has_fingers = any(finger_mask)
        if not wrist_active or has_fingers or gripper is None or poses is None or t >= len(gripper) or t >= len(poses):
            for tip_handle in handle:
                tip_handle.visible = False
            return
        if not show.get("fingertips", True):
            for tip_handle in handle:
                tip_handle.visible = False
            return
        pos = poses[t, :3, 3].astype(np.float32)
        rot = poses[t, :3, :3].astype(np.float32)
        g = float(gripper[t])
        mfw = getattr(self._entry, "max_finger_width", 0.05)
        embodiment_type = _get_entry_agibot_embodiment_type(self._entry)
        if (
            getattr(self._entry, "robot_name", "") == "embodiment_c"
            and embodiment_type
            and get_embodiment_c_kind(embodiment_type) == "gripper"
        ):
            tip_l, tip_r = _agibot_gripper_tip_positions(pos, rot, g, float(mfw))
        else:
            half_w = g * mfw / 2.0
            finger_len = 0.06
            tip_l = pos + rot @ np.array([half_w, 0, finger_len], dtype=np.float32)
            tip_r = pos + rot @ np.array([-half_w, 0, finger_len], dtype=np.float32)
        for tip_handle, tip in zip(handle, (tip_l, tip_r), strict=True):
            tip_handle.position = tip
            tip_handle.visible = True

    # ─── Private: Trajectory ──────────────────────────────────────────────────

    def _rebuild_traj(self, traj_handle, poses, color):
        """Rebuild a trajectory spline from absolute poses."""
        positions = poses[:, :3, 3].astype(np.float32)
        if len(positions) < 2:
            traj_handle.visible = False
            return
        line_width = self.EGO_TRAJ_LINE_WIDTH if traj_handle is self.ego_traj else self.EE_TRAJ_LINE_WIDTH
        # Remove and recreate to avoid stale color issues in viser
        name = traj_handle.name if hasattr(traj_handle, "name") else "/tmp/traj"
        traj_handle.remove()
        new_handle = self.server.scene.add_spline_catmull_rom(
            name,
            positions=positions,
            color=self._soften_color(color),
            line_width=line_width,
        )
        # Update the reference — need to figure out which attribute to update
        if traj_handle is self.ego_traj:
            self.ego_traj = new_handle
        elif traj_handle is self.right_traj:
            self.right_traj = new_handle
        elif traj_handle is self.left_traj:
            self.left_traj = new_handle

    @staticmethod
    def _trajectory_length(poses: np.ndarray | None) -> float:
        """Compute the total path length of a pose trajectory."""
        if poses is None or len(poses) < 2:
            return 0.0
        positions = poses[:, :3, 3].astype(np.float32)
        deltas = np.diff(positions, axis=0)
        return float(np.linalg.norm(deltas, axis=1).sum())

    def _update_ego_visual_scale(self, poses: np.ndarray | None) -> None:
        """Scale the ego camera frame/frustum from the episode trajectory length."""
        traj_length = self._trajectory_length(poses)
        if traj_length <= 0.0:
            traj_scale = 1.0
        else:
            traj_scale = float(
                np.clip(
                    traj_length / self.EGO_TRAJ_LENGTH_REFERENCE,
                    self.EGO_TRAJ_SCALE_MIN,
                    self.EGO_TRAJ_SCALE_MAX,
                )
            )
        self._ego_frustum_scale_base = self.FRUSTUM_SCALE * traj_scale
        self._ego_axis_length_base = self.EGO_AXIS_LENGTH * traj_scale
        self._ego_axis_radius_base = self.EGO_AXIS_RADIUS * traj_scale

    # ─── Private: IK Robot ────────────────────────────────────────────────────

    @classmethod
    def _robot_frame_dims(cls, frame_key: str) -> tuple[float, float]:
        """Return axis length/radius for one robot debug frame."""
        if frame_key.startswith("site:"):
            return cls.ROBOT_SITE_AXIS_LENGTH, cls.ROBOT_SITE_AXIS_RADIUS
        return cls.ROBOT_BODY_AXIS_LENGTH, cls.ROBOT_BODY_AXIS_RADIUS

    @staticmethod
    def _clear_robot_frame_handles(handles: dict[str, Any]) -> None:
        """Remove all robot debug frame handles in one arm."""
        for handle in handles.values():
            handle.remove()
        handles.clear()

    def _robot_frame_selector_key(self, arm: str, frame_key: str) -> str:
        """Return the GUI selector key for one robot debug frame."""
        if self._robot_frames_left is None and arm == "right":
            return frame_key
        return f"{arm}/{frame_key}"

    def get_robot_frame_selectors(self) -> list[tuple[str, str]]:
        """Return selector keys and checkbox labels for available robot frames."""
        selectors = []
        for arm, frames in [("right", self._robot_frames_right), ("left", self._robot_frames_left)]:
            if frames is None:
                continue
            for frame_key in sorted(frames):
                selector_key = self._robot_frame_selector_key(arm, frame_key)
                if self._robot_frames_left is None and arm == "right":
                    label = frame_key
                else:
                    label = selector_key
                selectors.append((selector_key, label))
        return selectors

    def _rebuild_robot_frame_handles(
        self,
        arm: str,
        frames: dict[str, list[tuple[np.ndarray, np.ndarray]]] | None,
    ) -> None:
        """Recreate robot debug frame handles for one arm."""
        handles = self._robot_frame_handles_right if arm == "right" else self._robot_frame_handles_left
        self._clear_robot_frame_handles(handles)
        if frames is None:
            return
        for frame_key in sorted(frames):
            kind, name = frame_key.split(":", 1)
            axes_length, axes_radius = self._robot_frame_dims(frame_key)
            handles[frame_key] = self.server.scene.add_frame(
                f"/robot_{arm}_frames/{kind}/{name}",
                axes_length=axes_length * self._axis_scale,
                axes_radius=axes_radius * self._axis_scale,
            )
        log.info(f"Loaded {len(handles)} robot debug frames for {arm} arm")

    def _update_robot_debug_frames(self, t: int, show: dict) -> None:
        """Update body/site coordinate frame overlays for both arms."""
        filters = show.get("robot_frame_filters", {})
        for arm, handles, frames in [
            ("right", self._robot_frame_handles_right, self._robot_frames_right),
            ("left", self._robot_frame_handles_left, self._robot_frames_left),
        ]:
            for frame_key, handle in handles.items():
                poses = frames.get(frame_key) if frames is not None else None
                selector_key = self._robot_frame_selector_key(arm, frame_key)
                frame_enabled = bool(filters.get(selector_key, False))
                if poses is not None and t < len(poses) and frame_enabled:
                    pos, rot = poses[t]
                    handle.position = pos
                    handle.wxyz = self.vtf.SO3.from_matrix(rot).wxyz
                    handle.visible = True
                else:
                    handle.visible = False

    def _set_agibot_mesh_animation_from_link_poses(self, link_poses: dict[str, np.ndarray], num_steps: int) -> None:
        """Populate AgiBot mesh transforms from absolute URDF link poses."""

        animated_transforms: list[list[tuple[np.ndarray, np.ndarray]]] = []
        for step_idx in range(num_steps):
            frame_transforms: list[tuple[np.ndarray, np.ndarray]] = []
            for link_name, local_transform in zip(
                self._robot_link_names,
                self._robot_local_transforms,
                strict=True,
            ):
                world_transform = link_poses[link_name][step_idx] @ local_transform  # [4,4]
                frame_transforms.append(
                    (
                        world_transform[:3, 3].astype(np.float32, copy=False),  # [3]
                        world_transform[:3, :3].astype(np.float32, copy=False),  # [3,3]
                    )
                )
            animated_transforms.append(frame_transforms)

        self._ik_right = animated_transforms
        self._ik_left = None
        self._robot_frames_right = None
        self._robot_frames_left = None

    def _load_robot_and_ik(self, state: SceneState, entry: Any):
        """Load robot meshes and solve IK for the episode."""
        if entry.robot_name == "embodiment_c":
            embodiment_type = _get_entry_agibot_embodiment_type(entry)
            if not embodiment_type:
                log.warning("AgiBot entry missing embodiment_type — skipping robot animation")
                return

            robot_key = (entry.robot_name, embodiment_type, "urdf_collision")
            if self._current_robot != robot_key:
                for h in self.robot_right + self.robot_left:
                    h.remove()
                self.robot_right = []
                self.robot_left = []
                self._clear_robot_frame_handles(self._robot_frame_handles_right)
                self._clear_robot_frame_handles(self._robot_frame_handles_left)
                self._robot_link_names = []
                self._robot_local_transforms = []

                geometries = get_embodiment_c_collision_geometries()
                for geometry in geometries:
                    handle = self.server.scene.add_mesh_trimesh(
                        f"/robot_right/{geometry.name}",
                        mesh=geometry.mesh,
                    )
                    handle.visible = False
                    self.robot_right.append(handle)
                    self._robot_link_names.append(geometry.link_name)
                    self._robot_local_transforms.append(geometry.local_transform)

                self._current_robot = robot_key
                log.info(f"Loaded {len(geometries)} AgiBot URDF collision geometries")

            if state.agibot_link_poses:
                missing_links = [
                    link_name for link_name in self._robot_link_names if link_name not in state.agibot_link_poses
                ]
                if not missing_links:
                    num_steps = min(
                        int(state.agibot_link_poses[link_name].shape[0]) for link_name in self._robot_link_names
                    )
                    if num_steps > 0:
                        self._set_agibot_mesh_animation_from_link_poses(state.agibot_link_poses, num_steps)
                        return
                log.warning(
                    f"AgiBot direct FK link poses missing {len(missing_links)} robot links; falling back to IK."
                )

            if state.right_poses is None and state.left_poses is None and state.ego_poses is None:
                log.warning("AgiBot viewer has no head/left/right pose targets — skipping robot animation")
                return

            left_wrist_poses_ik = _undo_wrist_to_opencv_for_ik(
                state.left_poses,
                self._to_opencv,
                "left_wrist",
            )  # [T,4,4] | None
            right_wrist_poses_ik = _undo_wrist_to_opencv_for_ik(
                state.right_poses,
                self._to_opencv,
                "right_wrist",
            )  # [T,4,4] | None

            joint_configs = solve_agibot_trajectory_ik(
                head_camera_poses=state.ego_poses,
                left_wrist_poses=left_wrist_poses_ik,
                right_wrist_poses=right_wrist_poses_ik,
                left_gripper_openings=state.gripper_left,
                right_gripper_openings=state.gripper_right,
            )
            if joint_configs is None:
                log.warning("AgiBot IK failed — skipping robot animation")
                return

            link_poses = compute_agibot_link_poses_batch_from_configs(
                joint_configs,
                self._robot_link_names,
            )
            self._set_agibot_mesh_animation_from_link_poses(link_poses, int(joint_configs.shape[0]))
            return

        # Determine robot config key (single vs dual)
        is_dual = state.mask.left_wrist and entry.dual_base_left is not None
        robot_key = (entry.robot_name, "dual") if is_dual else entry.robot_name

        if self._robot_scene_model is None or self._robot_scene_model.robot_name != entry.robot_name:
            try:
                self._robot_scene_model = RobotSceneModel(entry.robot_name)
            except Exception as e:
                log.warning(f"RobotSceneModel unavailable for {entry.robot_name}: {e}")
                self._robot_scene_model = None
                return

        # Only reload meshes if robot changed
        if self._current_robot != robot_key:
            for h in self.robot_right + self.robot_left:
                h.remove()
            self.robot_right = []
            self.robot_left = []
            self._clear_robot_frame_handles(self._robot_frame_handles_right)
            self._clear_robot_frame_handles(self._robot_frame_handles_left)

            if self._robot_scene_model is None:
                self._current_robot = robot_key
                return

            right_meshes = self._robot_scene_model.get_home_meshes(entry.dual_base_right if is_dual else None)
            for name, mesh, transform in right_meshes:
                h = self.server.scene.add_mesh_trimesh(
                    f"/robot_right/{name}",
                    mesh=mesh,
                )
                h.position = transform[:3, 3]
                h.wxyz = self.vtf.SO3.from_matrix(transform[:3, :3]).wxyz
                self.robot_right.append(h)

            if is_dual:
                left_meshes = self._robot_scene_model.get_home_meshes(entry.dual_base_left)
                for name, mesh, transform in left_meshes:
                    h = self.server.scene.add_mesh_trimesh(
                        f"/robot_left/{name}",
                        mesh=mesh,
                    )
                    h.position = transform[:3, 3]
                    h.wxyz = self.vtf.SO3.from_matrix(transform[:3, :3]).wxyz
                    self.robot_left.append(h)

            self._current_robot = robot_key
            log.info(f"Loaded {len(right_meshes)} meshes for {entry.robot_name}")

        if self._robot_scene_model is None:
            return

        # Joint-position datasets (e.g. robomind-ur): bypass IK, use FK directly
        if state.joint_configs is not None:
            from cosmos.data.vfm.action.urdf_visualizer.ik_solver import compute_mujoco_geom_transforms
            from cosmos.data.vfm.action.urdf_visualizer.robot_scene_model import get_mjcf_path

            try:
                mjcf_path = get_mjcf_path(entry.robot_name)
                transforms, _, _fk_ee_poses, robot_frames = compute_mujoco_geom_transforms(
                    mjcf_path, state.joint_configs
                )
                self._ik_right = transforms
                self._robot_frames_right = robot_frames
                self._rebuild_robot_frame_handles("right", robot_frames)
                log.info(f"FK geom transforms computed for {len(transforms)} frames ({entry.robot_name})")
            except Exception as e:
                log.warning(f"FK failed for {entry.robot_name}: {e}")
                import traceback

                traceback.print_exc()
            return

        # Right arm IK
        if state.right_poses is not None:
            try:
                right_result = self._robot_scene_model.solve_visual_trajectory(
                    state.right_poses,
                    gripper_openings=state.gripper_right,
                    to_opencv=self._to_opencv,
                    base_pose=entry.dual_base_right if is_dual else None,
                )
                if right_result is not None:
                    self._ik_right = right_result.mesh_transforms
                    self._robot_frames_right = right_result.named_frames
                    self._rebuild_robot_frame_handles("right", self._robot_frames_right)
                else:
                    self._ik_right = None
                    self._robot_frames_right = None
                    self._rebuild_robot_frame_handles("right", None)
            except Exception as e:
                log.warning(f"IK failed (right): {e}")
                self._ik_right = None
                self._robot_frames_right = None
                self._rebuild_robot_frame_handles("right", None)
        else:
            self._ik_right = None
            self._robot_frames_right = None
            self._rebuild_robot_frame_handles("right", None)

        # Left arm IK (dual only)
        if is_dual and state.left_poses is not None:
            try:
                left_result = self._robot_scene_model.solve_visual_trajectory(
                    state.left_poses,
                    gripper_openings=state.gripper_left,
                    to_opencv=self._to_opencv,
                    base_pose=entry.dual_base_left,
                )
                if left_result is not None:
                    self._ik_left = left_result.mesh_transforms
                    self._robot_frames_left = left_result.named_frames
                    self._rebuild_robot_frame_handles("left", self._robot_frames_left)
                else:
                    self._ik_left = None
                    self._robot_frames_left = None
                    self._rebuild_robot_frame_handles("left", None)
            except Exception as e:
                log.warning(f"IK failed (left): {e}")
                self._ik_left = None
                self._robot_frames_left = None
                self._rebuild_robot_frame_handles("left", None)
        else:
            self._ik_left = None
            self._robot_frames_left = None
            self._rebuild_robot_frame_handles("left", None)

    def _update_robot(self, t: int, show: dict):
        vis = show.get("robot", True)
        for handles, ik in [(self.robot_right, self._ik_right), (self.robot_left, self._ik_left)]:
            for idx, h in enumerate(handles):
                if vis and ik is not None and t < len(ik) and idx < len(ik[t]):
                    p, m = ik[t][idx]
                    h.position = p
                    h.wxyz = self.vtf.SO3.from_matrix(m).wxyz
                    h.visible = True
                else:
                    h.visible = False
        self._update_robot_debug_frames(t, show)

    # ─── Visibility ───────────────────────────────────────────────────────────

    def hide_all(self):
        """Hide every scene element."""
        for attr in [
            self.ego_frame,
            self.ego_frustum,
            self.ego_frustum_up,
            self.ego_traj,
            self.right_frame,
            self.right_ee,
            self.right_traj,
            self.left_frame,
            self.left_ee,
            self.left_traj,
        ]:
            attr.visible = False
        for h in self.right_fingers + self.left_fingers:
            h.visible = False
        for h in self.right_gripper_tips + self.left_gripper_tips:
            h.visible = False
        for h in self.robot_right + self.robot_left:
            h.visible = False
        for handle in list(self._robot_frame_handles_right.values()) + list(self._robot_frame_handles_left.values()):
            handle.visible = False

    def update_axis_scale(self, scale: float):
        """Update coordinate frame axis size and effector point size."""
        self._axis_scale = scale
        s = scale
        self.ego_frame.axes_length = self._ego_axis_length_base * s
        self.ego_frame.axes_radius = self._ego_axis_radius_base * s
        self.ego_frustum.scale = self._ego_frustum_scale_base * s
        self.ego_frustum_up.scale = self._ego_frustum_scale_base * s
        for frame in (self.right_frame, self.left_frame):
            frame.axes_length = self.HAND_AXIS_LENGTH * s
            frame.axes_radius = self.HAND_AXIS_RADIUS * s
        for ee in (self.right_ee, self.left_ee):
            ee.point_size = 0.015 * s
        for handles in (self._robot_frame_handles_right, self._robot_frame_handles_left):
            for frame_key, handle in handles.items():
                axes_length, axes_radius = self._robot_frame_dims(frame_key)
                handle.axes_length = axes_length * s
                handle.axes_radius = axes_radius * s
