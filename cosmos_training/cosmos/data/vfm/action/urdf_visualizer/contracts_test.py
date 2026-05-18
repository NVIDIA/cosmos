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

import numpy as np
import pytest

from cosmos.data.vfm.action.urdf_visualizer import ik_solver, robot_scene_model, unified_renderer
from cosmos.data.vfm.action.urdf_visualizer.robot_scene_model import RobotSceneModel
from cosmos.data.vfm.action.urdf_visualizer.unified_action import (
    ALL_FINGERS,
    Action57DMask,
    ActionFormat,
    UnifiedAction,
    build_scene_state,
    to_unified,
)
from cosmos.data.vfm.action.urdf_visualizer.unified_renderer import _agibot_gripper_tip_positions


@pytest.mark.L0
@pytest.mark.parametrize(
    ("action_format", "action", "expected_mask"),
    [
        (
            ActionFormat.EGO_9D,
            np.zeros((2, 9), dtype=np.float32),
            Action57DMask(ego=True),
        ),
        (
            ActionFormat.SINGLE_ARM_10D,
            np.zeros((2, 10), dtype=np.float32),
            Action57DMask(right_wrist=True),
        ),
        (
            ActionFormat.DUAL_ARM_20D,
            np.zeros((2, 20), dtype=np.float32),
            Action57DMask(right_wrist=True, left_wrist=True),
        ),
        (
            ActionFormat.UNIFIED_57D,
            np.zeros((2, 57), dtype=np.float32),
            Action57DMask(
                ego=True,
                right_wrist=True,
                right_fingers=ALL_FINGERS,
                left_wrist=True,
                left_fingers=ALL_FINGERS,
            ),
        ),
    ],
)
def test_to_unified_uses_explicit_action_format(
    action_format: ActionFormat,
    action: np.ndarray,
    expected_mask: Action57DMask,
) -> None:
    """Explicit action formats should decode to one canonical 57D layout."""
    unified = to_unified(action, action_format=action_format)

    assert unified.action.shape == (2, 57)
    assert unified.mask == expected_mask


@pytest.mark.L0
def test_to_unified_rejects_shape_mismatches() -> None:
    """Declared raw action formats should fail fast on incompatible tensors."""
    with pytest.raises(ValueError, match="expects trailing dim 10"):
        to_unified(np.zeros((2, 9), dtype=np.float32), action_format=ActionFormat.SINGLE_ARM_10D)


@pytest.mark.L0
def test_build_scene_state_outputs_world_space_contract() -> None:
    """SceneState should be fully canonicalized into world-space during construction."""
    action = np.zeros((1, 57), dtype=np.float32)
    action[0, 9:18] = np.array([0.1, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)
    action[0, 18:21] = np.array([0.0, 0.0, 0.05], dtype=np.float32)
    action[0, 33:42] = np.array([0.2, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)
    unified = UnifiedAction(
        action=action,
        mask=Action57DMask(
            right_wrist=True,
            right_fingers=ALL_FINGERS,
            left_wrist=True,
        ),
    )

    right_base = np.eye(4, dtype=np.float32)
    right_base[1, 3] = -0.3
    left_base = np.eye(4, dtype=np.float32)
    left_base[1, 3] = 0.3

    state = build_scene_state(
        unified,
        right_base_pose=right_base,
        left_base_pose=left_base,
    )

    np.testing.assert_allclose(state.right_poses[1, :3, 3], np.array([0.1, -0.3, 0.0], dtype=np.float32))
    np.testing.assert_allclose(state.left_poses[1, :3, 3], np.array([0.2, 0.3, 0.0], dtype=np.float32))
    np.testing.assert_allclose(state.right_fingers[0, 0], np.array([0.0, -0.3, 0.05], dtype=np.float32))
    np.testing.assert_allclose(state.right_fingers[1, 0], np.array([0.1, -0.3, 0.05], dtype=np.float32))


@pytest.mark.L0
def test_robot_scene_model_caches_home_meshes_and_applies_base_pose(monkeypatch: pytest.MonkeyPatch) -> None:
    """RobotSceneModel should own home mesh loading and world-frame placement."""
    calls = {"count": 0}
    mesh_transform = np.eye(4, dtype=np.float32)
    mesh_transform[2, 3] = 0.2

    monkeypatch.setattr(robot_scene_model, "get_robot_config", lambda _: {"ee_frame": "tool"})
    monkeypatch.setattr(robot_scene_model, "get_mjcf_path", lambda _: "/tmp/fake.xml")

    def _loader() -> tuple[list[tuple[str, object, np.ndarray]], np.ndarray]:
        calls["count"] += 1
        return [("geom", object(), mesh_transform)], np.eye(4, dtype=np.float32)

    monkeypatch.setattr(robot_scene_model, "get_robot_loaders", lambda: {"fake_robot": _loader})

    model = RobotSceneModel("fake_robot")
    meshes_local = model.get_home_meshes()
    base_pose = np.eye(4, dtype=np.float32)
    base_pose[0, 3] = 1.5
    meshes_world = model.get_home_meshes(base_pose=base_pose)

    assert calls["count"] == 1
    np.testing.assert_allclose(meshes_local[0][2][:3, 3], np.array([0.0, 0.0, 0.2], dtype=np.float32))
    np.testing.assert_allclose(meshes_world[0][2][:3, 3], np.array([1.5, 0.0, 0.2], dtype=np.float32))


@pytest.mark.L0
def test_robot_scene_model_reapplies_base_pose_after_solving(monkeypatch: pytest.MonkeyPatch) -> None:
    """RobotSceneModel should solve in arm-local space but return world-space geometry."""
    captured: dict[str, np.ndarray] = {}

    monkeypatch.setattr(robot_scene_model, "get_robot_config", lambda _: {"ee_frame": "tool"})
    monkeypatch.setattr(robot_scene_model, "get_mjcf_path", lambda _: "/tmp/fake.xml")

    def _fake_solve(
        _mjcf_path: str,
        world_ee_positions: np.ndarray,
        gripper_openings: np.ndarray | None = None,
        world_ee_orientations: np.ndarray | None = None,
        robot_name: str | None = None,
    ) -> np.ndarray:
        captured["positions"] = world_ee_positions.copy()
        captured["orientations"] = world_ee_orientations.copy()
        captured["grippers"] = (
            gripper_openings.copy() if gripper_openings is not None else np.array([], dtype=np.float32)
        )
        assert robot_name == "fake_robot"
        return np.zeros((2, 1), dtype=np.float32)

    def _fake_fk(
        _mjcf_path: str,
        _joint_configs: np.ndarray,
        robot_name: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        assert robot_name == "fake_robot"
        positions = np.array([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]], dtype=np.float32)
        rotations = np.tile(np.eye(3, dtype=np.float32), (2, 1, 1))
        return positions, rotations

    def _fake_geom(
        _mjcf_path: str,
        _joint_configs: np.ndarray,
    ) -> tuple[
        list[list[tuple[np.ndarray, np.ndarray]]],
        None,
        None,
        dict[str, list[tuple[np.ndarray, np.ndarray]]],
    ]:
        transforms = [
            [(np.array([0.0, 0.0, 0.0], dtype=np.float32), np.eye(3, dtype=np.float32))],
            [(np.array([0.2, 0.0, 0.0], dtype=np.float32), np.eye(3, dtype=np.float32))],
        ]
        frames = {
            "body:tool": [
                (np.array([0.0, 0.0, 0.0], dtype=np.float32), np.eye(3, dtype=np.float32)),
                (np.array([0.2, 0.0, 0.0], dtype=np.float32), np.eye(3, dtype=np.float32)),
            ]
        }
        return transforms, None, None, frames

    monkeypatch.setattr(ik_solver, "solve_trajectory_ik", _fake_solve)
    monkeypatch.setattr(ik_solver, "compute_fk_ee_poses", _fake_fk)
    monkeypatch.setattr(ik_solver, "compute_mujoco_geom_transforms", _fake_geom)

    model = RobotSceneModel("fake_robot")
    wrist_poses_world = np.tile(np.eye(4, dtype=np.float32), (2, 1, 1))
    wrist_poses_world[:, 0, 3] = np.array([1.0, 1.2], dtype=np.float32)
    base_pose = np.eye(4, dtype=np.float32)
    base_pose[0, 3] = 1.0
    grippers = np.array([0.0, 0.25], dtype=np.float32)

    result = model.solve_visual_trajectory(
        wrist_poses_world,
        gripper_openings=grippers,
        base_pose=base_pose,
    )

    assert result is not None
    np.testing.assert_allclose(
        captured["positions"][:, 0],
        np.array([0.0, 0.2], dtype=np.float32),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        result.mesh_transforms[1][0][0],
        np.array([1.2, 0.0, 0.0], dtype=np.float32),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        result.named_frames["ik:tool"][1][0],
        np.array([1.2, 0.0, 0.0], dtype=np.float32),
        atol=1e-6,
    )


@pytest.mark.L0
def test_agibot_gripper_tip_spheres_open_along_opencv_x_axis() -> None:
    """AgiBot synthetic tip spheres should open horizontally in the corrected wrist frame."""

    pos = np.array([0.1, 0.2, 0.3], dtype=np.float32)  # [3]
    rot = np.eye(3, dtype=np.float32)  # [3,3]

    tip_l, tip_r = _agibot_gripper_tip_positions(
        pos=pos,
        rot=rot,
        opening=1.0,
        max_finger_width=0.12,
    )

    expected_center = pos + np.array([0.0, 0.0, 0.14308], dtype=np.float32)  # [3]
    np.testing.assert_allclose(tip_l, expected_center + np.array([0.06, 0.0, 0.0], dtype=np.float32), atol=1e-6)
    np.testing.assert_allclose(tip_r, expected_center + np.array([-0.06, 0.0, 0.0], dtype=np.float32), atol=1e-6)


@pytest.mark.L0
def test_agibot_renderer_uses_direct_link_poses_before_ik(monkeypatch: pytest.MonkeyPatch) -> None:
    """Dataset FK link poses should drive AgiBot meshes directly when present."""

    class _Scene:
        def __init__(self) -> None:
            self.handles: list[object] = []

        def add_mesh_trimesh(self, *_args, **_kwargs):
            class _Handle:
                visible = True

                def remove(self) -> None:
                    self.visible = False

            handle = _Handle()
            self.handles.append(handle)
            return handle

    class _Server:
        def __init__(self) -> None:
            self.scene = _Scene()

    class _Entry:
        robot_name = "embodiment_c"
        robot_embodiment_type = "embodiment_c_gripper"
        dataset_kwargs: dict[str, str] = {}

    geometry = type(
        "Geometry",
        (),
        {
            "name": "moving_link_geometry_0",
            "link_name": "moving_link",
            "mesh": object(),
            "local_transform": np.eye(4, dtype=np.float32),
        },
    )()
    monkeypatch.setattr(unified_renderer, "get_embodiment_c_collision_geometries", lambda: [geometry])

    def _fail_ik(*_args, **_kwargs):
        raise AssertionError("IK should not run when direct AgiBot FK link poses are present")

    monkeypatch.setattr(unified_renderer, "solve_agibot_trajectory_ik", _fail_ik)

    poses = np.tile(np.eye(4, dtype=np.float32), (2, 1, 1))  # [T+1,4,4]
    poses[:, 0, 3] = np.array([0.1, 0.4], dtype=np.float32)  # [T+1]
    state = build_scene_state(
        UnifiedAction(np.zeros((1, 57), dtype=np.float32), Action57DMask()),
        sample={"agibot_link_poses": {"moving_link": poses}},
    )

    renderer = object.__new__(unified_renderer.UnifiedRenderer)
    renderer.server = _Server()
    renderer.robot_right = []
    renderer.robot_left = []
    renderer._robot_frame_handles_right = {}
    renderer._robot_frame_handles_left = {}
    renderer._robot_link_names = []
    renderer._robot_local_transforms = []
    renderer._current_robot = None

    renderer._load_robot_and_ik(state, _Entry())

    assert renderer._ik_left is None
    assert len(renderer._ik_right) == 2
    np.testing.assert_allclose(renderer._ik_right[1][0][0], np.array([0.4, 0.0, 0.0], dtype=np.float32))
