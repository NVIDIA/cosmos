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

"""Public robot-scene abstraction for the action viewer.

`RobotSceneModel` is the viewer-facing contract for robot assets and
kinematics. It wraps the lower-level MuJoCo / Pinocchio helpers so callers do
not need to coordinate mesh loading, IK, frame extraction, or world-alignment
corrections themselves.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from cosmos.data.vfm.action.urdf_visualizer import urdf_loader

MeshSpec = tuple[str, object, np.ndarray]
FramePose = tuple[np.ndarray, np.ndarray]
FrameSeries = dict[str, list[FramePose]]


@dataclass(frozen=True)
class RobotTrajectory:
    """Render-ready robot geometry for one solved trajectory."""

    mesh_transforms: list[list[FramePose]]
    named_frames: FrameSeries


def get_robot_config(robot_name: str) -> dict[str, Any]:
    """Return the static config for one supported robot."""
    cfg = urdf_loader.ROBOT_CONFIGS.get(robot_name)
    if cfg is None:
        raise ValueError(f"Unknown robot: {robot_name}. Available: {list(urdf_loader.ROBOT_CONFIGS)}")
    return cfg


def get_urdf_path(robot_name: str) -> str | None:
    """Return the preferred URDF path for one robot, if available."""
    return urdf_loader.get_urdf_path(robot_name)


def get_mjcf_path(robot_name: str) -> str:
    """Return the canonical MJCF path for one robot."""
    return urdf_loader.get_mjcf_path(robot_name)


def get_mujoco_to_pinocchio_world_transform(model: Any, data: Any, robot_name: str | None = None) -> np.ndarray:
    """Map MuJoCo world poses into the root-free Pinocchio world."""
    return urdf_loader.get_mujoco_to_pinocchio_world_transform(model, data, robot_name)


def get_visual_geom_ids(model: Any) -> list[int]:
    """Return the MuJoCo visual geom order used by the viewer."""
    return urdf_loader._get_visual_geom_ids(model)


def get_ee_frame_candidates(robot_name: str | None = None) -> list[str]:
    """Return the ordered end-effector frame candidates for one robot."""
    if robot_name is None:
        return list(urdf_loader._EE_FRAME_CANDIDATES)
    cfg = urdf_loader.ROBOT_CONFIGS.get(robot_name, {})
    ee_frame = cfg.get("ee_frame")
    if ee_frame is None:
        return list(urdf_loader._EE_FRAME_CANDIDATES)
    return [str(ee_frame), *urdf_loader._EE_FRAME_CANDIDATES]


def get_robot_loaders() -> dict[str, Callable[[], tuple[list[MeshSpec], np.ndarray]]]:
    """Return the low-level robot mesh loader registry."""
    return urdf_loader.get_robot_loaders()


def resolve_robot_name_from_mjcf(mjcf_path: str) -> str | None:
    """Infer a configured robot name from an MJCF filename."""
    filename = os.path.basename(mjcf_path)
    for robot_name, cfg in urdf_loader.ROBOT_CONFIGS.items():
        if filename == cfg.get("mjcf"):
            return robot_name
    return None


def _copy_mesh_specs(meshes: list[MeshSpec]) -> list[MeshSpec]:
    """Copy mesh transforms while reusing immutable mesh geometry."""
    return [(name, mesh, transform.copy().astype(np.float32)) for name, mesh, transform in meshes]


def _transform_mesh_specs(meshes: list[MeshSpec], base_pose: np.ndarray | None) -> list[MeshSpec]:
    """Apply one rigid transform to a list of world-space mesh poses."""
    copied = _copy_mesh_specs(meshes)
    if base_pose is None:
        return copied
    transformed: list[MeshSpec] = []
    for name, mesh, transform in copied:
        transformed.append((name, mesh, (base_pose @ transform).astype(np.float32)))  # [4,4]
    return transformed


def _remove_pose_base(poses_world: np.ndarray, base_pose: np.ndarray | None) -> np.ndarray:
    """Map world-space wrist poses back into one arm-local base frame."""
    if base_pose is None:
        return poses_world.astype(np.float32)
    base_inv = np.linalg.inv(base_pose).astype(np.float32)  # [4,4]
    return np.einsum("ij,njk->nik", base_inv, poses_world).astype(np.float32)  # [T,4,4]


def _apply_base_to_mesh_transforms(
    transforms: list[list[FramePose]],
    base_pose: np.ndarray | None,
) -> list[list[FramePose]]:
    """Apply one rigid base pose to per-geom world transforms."""
    if base_pose is None:
        return transforms
    base_rot = base_pose[:3, :3].astype(np.float32)  # [3,3]
    base_pos = base_pose[:3, 3].astype(np.float32)  # [3]
    transformed: list[list[FramePose]] = []
    for frame_transforms in transforms:
        transformed_frame: list[FramePose] = []
        for pos, rot in frame_transforms:
            transformed_frame.append((base_rot @ pos + base_pos, base_rot @ rot))
        transformed.append(transformed_frame)
    return transformed


def _apply_base_to_named_frames(
    frames: FrameSeries,
    base_pose: np.ndarray | None,
) -> FrameSeries:
    """Apply one rigid base pose to named body/site frames."""
    if base_pose is None:
        return frames
    base_rot = base_pose[:3, :3].astype(np.float32)  # [3,3]
    base_pos = base_pose[:3, 3].astype(np.float32)  # [3]
    transformed: FrameSeries = {}
    for frame_key, poses in frames.items():
        transformed[frame_key] = [(base_rot @ pos + base_pos, base_rot @ rot) for pos, rot in poses]
    return transformed


class RobotSceneModel:
    """Single public abstraction for robot meshes, IK/FK, and debug frames."""

    def __init__(self, robot_name: str) -> None:
        self.robot_name = robot_name
        self._config = get_robot_config(robot_name)
        self._mjcf_path = get_mjcf_path(robot_name)
        self._home_meshes: list[MeshSpec] | None = None

    @property
    def mjcf_path(self) -> str:
        """Return the underlying MJCF path for this robot."""
        return self._mjcf_path

    @property
    def ee_frame_name(self) -> str:
        """Return the canonical IK / debug frame name for this robot."""
        return str(self._config.get("ee_frame", "ee_frame"))

    def get_home_meshes(self, base_pose: np.ndarray | None = None) -> list[MeshSpec]:
        """Return home-pose meshes in the requested world frame."""
        if self._home_meshes is None:
            loaders = get_robot_loaders()
            loader = loaders.get(self.robot_name)
            if loader is None:
                raise ValueError(f"No robot loader registered for {self.robot_name}")
            meshes, _ = loader()
            self._home_meshes = _copy_mesh_specs(meshes)
        return _transform_mesh_specs(self._home_meshes, base_pose)

    def solve_visual_trajectory(
        self,
        wrist_poses_world: np.ndarray | None,
        gripper_openings: np.ndarray | None = None,
        to_opencv: np.ndarray | None = None,
        base_pose: np.ndarray | None = None,
    ) -> RobotTrajectory | None:
        """Solve IK for world-space wrist poses and return render-ready robot state."""
        if wrist_poses_world is None or len(wrist_poses_world) < 2:
            return None

        local_wrist_poses = _remove_pose_base(wrist_poses_world, base_pose)  # [T,4,4]
        target_positions = local_wrist_poses[:, :3, 3].astype(np.float32)  # [T,3]
        target_rotations = local_wrist_poses[:, :3, :3].astype(np.float32)  # [T,3,3]
        if to_opencv is not None and not np.allclose(to_opencv, np.eye(3, dtype=np.float32)):
            target_rotations = target_rotations @ to_opencv.T[None]  # [T,3,3]

        from cosmos.data.vfm.action.urdf_visualizer.ik_solver import (
            compute_fk_ee_poses,
            compute_mujoco_geom_transforms,
            solve_trajectory_ik,
        )

        joint_configs = solve_trajectory_ik(
            self.mjcf_path,
            target_positions,
            gripper_openings=gripper_openings,
            world_ee_orientations=target_rotations,
            robot_name=self.robot_name,
        )
        if joint_configs is None:
            return None

        fk_pos, fk_rot = compute_fk_ee_poses(self.mjcf_path, joint_configs, robot_name=self.robot_name)
        mesh_transforms, _, _, named_frames = compute_mujoco_geom_transforms(self.mjcf_path, joint_configs)
        public_frames: FrameSeries = {} if named_frames is None else dict(named_frames)
        public_frames[f"ik:{self.ee_frame_name}"] = list(zip(fk_pos, fk_rot, strict=True))

        mesh_transforms = _apply_base_to_mesh_transforms(mesh_transforms, base_pose)
        public_frames = _apply_base_to_named_frames(public_frames, base_pose)
        return RobotTrajectory(mesh_transforms=mesh_transforms, named_frames=public_frames)
