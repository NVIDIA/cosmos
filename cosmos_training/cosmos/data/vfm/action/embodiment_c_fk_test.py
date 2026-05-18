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

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

from cosmos.data.vfm.action.embodiment_c_dataset import EmbodimentCGripperDataset
from cosmos.data.vfm.action.embodiment_c_fk import (
    AGIBOT_GEAR_GRIPPER_TO_OPENCV_BY_WRIST,
    build_viewer_batch,
)
from cosmos.data.vfm.action.embodiment_c_spec import (
    AGIBOT_GEAR_EXT_STATE_ROBOT_ORIENTATION_SLICE,
    AGIBOT_GEAR_EXT_STATE_ROBOT_POSITION_SLICE,
    AGIBOT_GEAR_GRIPPER_OPEN_ANGLE_RAD,
    AGIBOT_GEAR_HEAD_CAMERA_LINK_NAME,
    AGIBOT_GEAR_LEFT_EE_LINK_NAME,
    AGIBOT_GEAR_LEFT_GRIPPER_CENTER_LINK_NAME,
    AGIBOT_GEAR_RIGHT_EE_LINK_NAME,
    get_embodiment_c_urdf_path,
)
from cosmos.data.vfm.action.agibotworld_beta_dataset import AgiBotWorldBetaDataset
from cosmos.data.vfm.action.domain_utils import get_domain_id
from cosmos.data.vfm.action.urdf_visualizer.viewer import DATASETS

_REPO_ROOT = Path(__file__).resolve().parents[5]
_LOCAL_GRIPPER_SAMPLE_ROOT = _REPO_ROOT / "dev" / "embodiment_c_samples" / "gripper" / "task_sample_gripper"
_REQUIRES_MUJOCO = pytest.mark.skipif(
    importlib.util.find_spec("mujoco") is None,
    reason="requires mujoco until CI docker images include it",
)


def _make_sample(state_dim: int) -> dict[str, torch.Tensor]:
    state = torch.zeros((3, state_dim), dtype=torch.float32)  # [T+1,S]
    return {
        "observation.state": state,
    }


def _make_beta_robot_pose(num_steps: int) -> dict[str, torch.Tensor]:
    robot_quat = torch.zeros((num_steps, 4), dtype=torch.float32)  # [T,4]
    robot_quat[:, 3] = 1.0
    return {
        "observation.states.robot.position": torch.zeros((num_steps, 3), dtype=torch.float32),  # [T,3]
        "observation.states.robot.orientation": robot_quat,  # [T,4]
    }


def test_get_embodiment_c_urdf_path_exists() -> None:
    assert get_embodiment_c_urdf_path().name == "G1_omnipicker_calibrated.urdf"
    assert get_embodiment_c_urdf_path().parent.name == "urdf_visualizer"
    assert get_embodiment_c_urdf_path().is_file()


@_REQUIRES_MUJOCO
def test_fk_uses_calibrated_head_camera_link() -> None:
    import numpy as np

    from cosmos.data.vfm.action.embodiment_c_fk import compute_fk_transforms, compute_link_poses_batch

    state = np.zeros(32, dtype=np.float32)
    fk = compute_fk_transforms(state, "embodiment_c_gripper")
    link_poses = compute_link_poses_batch(state[None, :], "embodiment_c_gripper")

    np.testing.assert_allclose(fk["head_camera"], link_poses[AGIBOT_GEAR_HEAD_CAMERA_LINK_NAME][0], atol=1e-6)


@_REQUIRES_MUJOCO
def test_fk_uses_gripper_base_not_center_link() -> None:
    import numpy as np

    from cosmos.data.vfm.action.embodiment_c_fk import compute_fk_transforms, compute_link_poses_batch

    state = np.zeros(32, dtype=np.float32)
    fk = compute_fk_transforms(state, "embodiment_c_gripper")
    link_poses = compute_link_poses_batch(state[None, :], "embodiment_c_gripper")

    np.testing.assert_allclose(fk["left_wrist"], link_poses[AGIBOT_GEAR_LEFT_EE_LINK_NAME][0], atol=1e-6)
    np.testing.assert_allclose(fk["right_wrist"], link_poses[AGIBOT_GEAR_RIGHT_EE_LINK_NAME][0], atol=1e-6)
    assert not np.allclose(
        fk["left_wrist"][:3, 3],
        link_poses[AGIBOT_GEAR_LEFT_GRIPPER_CENTER_LINK_NAME][0, :3, 3],
        atol=1e-6,
    )


@_REQUIRES_MUJOCO
@pytest.mark.L0
def test_dataset_gripper_poses_are_aligned_to_opencv_convention() -> None:
    import numpy as np

    from cosmos.data.vfm.action.embodiment_c_fk import compute_fk_transforms, compute_link_poses_batch

    states = np.zeros((3, 32), dtype=np.float32)  # [T+1,S]
    dataset = EmbodimentCGripperDataset(
        root=["unused"],
        chunk_length=2,
        skip_video_loading=True,
        return_agibot_link_poses=True,
    )
    action, extras = dataset._build_fk_action({"observation.state": torch.from_numpy(states)})
    fk = compute_fk_transforms(states[0], "embodiment_c_gripper")
    link_poses = compute_link_poses_batch(states[:1], "embodiment_c_gripper")
    to_opencv = AGIBOT_GEAR_GRIPPER_TO_OPENCV_BY_WRIST

    assert action.shape == (2, 29)
    assert "agibot_link_poses" in extras
    assert extras["agibot_link_poses"][AGIBOT_GEAR_LEFT_EE_LINK_NAME].shape == (3, 4, 4)
    np.testing.assert_allclose(
        extras["initial_pose_left"].numpy()[:3, :3],
        fk["left_wrist"][:3, :3] @ to_opencv["left_wrist"],
        atol=1e-6,
    )
    np.testing.assert_allclose(
        extras["initial_pose_right"].numpy()[:3, :3],
        fk["right_wrist"][:3, :3] @ to_opencv["right_wrist"],
        atol=1e-6,
    )
    np.testing.assert_allclose(fk["left_wrist"], link_poses[AGIBOT_GEAR_LEFT_EE_LINK_NAME][0], atol=1e-6)
    np.testing.assert_allclose(fk["right_wrist"], link_poses[AGIBOT_GEAR_RIGHT_EE_LINK_NAME][0], atol=1e-6)
    np.testing.assert_allclose(
        extras["initial_pose_left"].numpy()[:3, 3],
        link_poses[AGIBOT_GEAR_LEFT_EE_LINK_NAME][0, :3, 3],
        atol=1e-6,
    )
    np.testing.assert_allclose(
        extras["initial_pose_right"].numpy()[:3, 3],
        link_poses[AGIBOT_GEAR_RIGHT_EE_LINK_NAME][0, :3, 3],
        atol=1e-6,
    )


@_REQUIRES_MUJOCO
@pytest.mark.L0
def test_dataset_does_not_return_agibot_link_poses_by_default() -> None:
    import numpy as np

    states = np.zeros((3, 32), dtype=np.float32)  # [T+1,S]
    dataset = EmbodimentCGripperDataset(root=["unused"], chunk_length=2, skip_video_loading=True)

    _, extras = dataset._build_fk_action({"observation.state": torch.from_numpy(states)})

    assert "agibot_link_poses" not in extras


@pytest.mark.L0
def test_agibot_gripper_to_opencv_composes_extra_180deg_z_rotation() -> None:
    import numpy as np

    expected_left = np.asarray(
        [
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    expected_right = np.asarray(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    np.testing.assert_allclose(AGIBOT_GEAR_GRIPPER_TO_OPENCV_BY_WRIST["left_wrist"], expected_left, atol=1e-6)
    np.testing.assert_allclose(AGIBOT_GEAR_GRIPPER_TO_OPENCV_BY_WRIST["right_wrist"], expected_right, atol=1e-6)


def test_renderer_undoes_agibot_to_opencv_for_ik() -> None:
    import numpy as np

    from cosmos.data.vfm.action.urdf_visualizer.unified_renderer import _undo_wrist_to_opencv_for_ik

    to_opencv = AGIBOT_GEAR_GRIPPER_TO_OPENCV_BY_WRIST
    native_poses = np.tile(np.eye(4, dtype=np.float32), (2, 1, 1))  # [T,4,4]
    native_poses[:, :3, 3] = np.asarray([[0.1, 0.0, 0.2], [0.2, 0.0, 0.2]], dtype=np.float32)  # [T,3]
    viewer_poses = native_poses.copy()  # [T,4,4]
    viewer_poses[:, :3, :3] = viewer_poses[:, :3, :3] @ to_opencv["right_wrist"]  # [T,3,3]

    ik_poses = _undo_wrist_to_opencv_for_ik(viewer_poses, to_opencv, "right_wrist")

    assert ik_poses is not None
    np.testing.assert_allclose(ik_poses, native_poses, atol=1e-6)


def test_gripper_open_fraction_conversions() -> None:
    import numpy as np

    from cosmos.data.vfm.action.embodiment_c_fk import convert_gripper_state_to_open_fraction

    open_jitter_degrees = np.array([0.0, 0.21733333], dtype=np.float32)
    np.testing.assert_allclose(
        convert_gripper_state_to_open_fraction(open_jitter_degrees),
        [1.0, 0.9981889],
        atol=1e-6,
    )
    np.testing.assert_allclose(
        convert_gripper_state_to_open_fraction(np.array([0.0], dtype=np.float32)),
        [1.0],
        atol=1e-6,
    )

    radians = np.array([-AGIBOT_GEAR_GRIPPER_OPEN_ANGLE_RAD, -AGIBOT_GEAR_GRIPPER_OPEN_ANGLE_RAD / 2.0, 0.0])
    np.testing.assert_allclose(convert_gripper_state_to_open_fraction(radians), [1.0, 0.5, 0.0], atol=1e-6)

    actuator_degrees = np.array([0.0, 60.0, 120.0], dtype=np.float32)
    np.testing.assert_allclose(convert_gripper_state_to_open_fraction(actuator_degrees), [1.0, 0.5, 0.0], atol=1e-6)


@pytest.mark.L0
def test_gripper_open_fraction_clips_small_actuator_overshoot() -> None:
    import numpy as np

    from cosmos.data.vfm.action.embodiment_c_fk import convert_gripper_state_to_open_fraction

    actuator_degrees = np.array([122.1857], dtype=np.float32)  # [1]
    np.testing.assert_allclose(convert_gripper_state_to_open_fraction(actuator_degrees), [0.0], atol=1e-6)


@_REQUIRES_MUJOCO
def test_link_poses_apply_observed_gripper_state() -> None:
    import numpy as np

    from cosmos.data.vfm.action.embodiment_c_fk import compute_link_poses_batch

    states = np.zeros((2, 32), dtype=np.float32)  # [T,S]
    states[1, 14] = 120.0
    link_poses = compute_link_poses_batch(states, "embodiment_c_gripper")

    open_pose = link_poses["gripper_l_inner_link1"][0]  # [4,4]
    closed_pose = link_poses["gripper_l_inner_link1"][1]  # [4,4]
    assert not np.allclose(open_pose, closed_pose)


@_REQUIRES_MUJOCO
def test_standard_body_head_layout_uses_state19_as_waist_lift() -> None:
    import numpy as np

    from cosmos.data.vfm.action.embodiment_c_fk import compute_fk_transforms_batch

    states = np.zeros((2, 32), dtype=np.float32)  # [T,S]
    states[:, 17] = 0.25  # head pitch
    states[:, 18] = 0.5  # waist pitch
    states[:, 19] = 0.1  # waist lift
    states[1, 19] = 0.2

    fk = compute_fk_transforms_batch(states, "embodiment_c_gripper")

    for key in ("head_camera", "right_wrist", "left_wrist"):
        np.testing.assert_allclose(fk[key][1, 2, 3] - fk[key][0, 2, 3], 0.1, atol=1e-6)


@_REQUIRES_MUJOCO
def test_build_viewer_batch_gripper_shapes() -> None:
    sample = _make_sample(state_dim=32)

    batch = build_viewer_batch(sample, "embodiment_c_gripper")

    assert batch.head_camera_poses.shape == (2, 4, 4)
    assert batch.right_wrist_poses.shape == (2, 4, 4)
    assert batch.left_wrist_poses.shape == (2, 4, 4)
    assert batch.right_gripper is not None
    assert batch.left_gripper is not None
    assert batch.right_gripper.shape == (2,)
    assert batch.left_gripper.shape == (2,)


@_REQUIRES_MUJOCO
@pytest.mark.skipif(not _LOCAL_GRIPPER_SAMPLE_ROOT.is_dir(), reason="local Embodiment C sample data is unavailable")
def test_embodiment_c_dataset_loads_local_gripper_sample_without_video() -> None:
    dataset = EmbodimentCGripperDataset(
        root=[str(_LOCAL_GRIPPER_SAMPLE_ROOT)],
        fps=30.0,
        split="full",
        mode="policy",
        skip_video_loading=True,
    )
    dataset._register_sources()

    sample = dataset[0]

    assert len(dataset) == 184
    assert sample["action"].shape == (16, 29)
    assert sample["__episode_id__"] == 0
    assert sample["__task_root__"] == str(_LOCAL_GRIPPER_SAMPLE_ROOT)


@_REQUIRES_MUJOCO
def test_agibotworld_beta_builds_fk_action_for_viewer() -> None:
    dataset = AgiBotWorldBetaDataset(
        root=[],
        chunk_length=2,
        split="full",
        mode="policy",
    )
    sample = {
        "observation.states.effector.position": torch.zeros((3, 2), dtype=torch.float32),
        "observation.states.joint.position": torch.zeros((3, 14), dtype=torch.float32),
        "observation.states.head.position": torch.zeros((3, 2), dtype=torch.float32),
        "observation.states.waist.position": torch.zeros((3, 2), dtype=torch.float32),
        **_make_beta_robot_pose(3),
    }

    action, extras = dataset._build_fk_action(sample)

    assert action.shape == (2, 29)
    assert extras["initial_pose"].shape == (4, 4)
    assert extras["initial_pose_right"].shape == (4, 4)
    assert extras["initial_pose_left"].shape == (4, 4)


@_REQUIRES_MUJOCO
def test_agibotworld_beta_link_poses_include_observed_base_motion() -> None:
    import numpy as np

    dataset = AgiBotWorldBetaDataset(
        root=[],
        chunk_length=2,
        split="full",
        mode="policy",
        return_agibot_link_poses=True,
    )
    robot_pose = _make_beta_robot_pose(3)
    robot_pose["observation.states.robot.position"] = torch.tensor(
        [[0.0, 0.0, 0.0], [0.4, -0.2, 0.0], [0.7, -0.1, 0.0]],
        dtype=torch.float32,
    )  # [T+1,3]
    sample = {
        "observation.states.effector.position": torch.zeros((3, 2), dtype=torch.float32),
        "observation.states.joint.position": torch.zeros((3, 14), dtype=torch.float32),
        "observation.states.head.position": torch.zeros((3, 2), dtype=torch.float32),
        "observation.states.waist.position": torch.zeros((3, 2), dtype=torch.float32),
        **robot_pose,
    }

    _, extras = dataset._build_fk_action(sample)

    assert "agibot_link_poses" in extras
    head_poses = extras["agibot_link_poses"][AGIBOT_GEAR_HEAD_CAMERA_LINK_NAME].numpy()  # [T+1,4,4]
    np.testing.assert_allclose(head_poses[1, :3, 3] - head_poses[0, :3, 3], [0.4, -0.2, 0.0], atol=1e-6)
    np.testing.assert_allclose(head_poses[2, :3, 3] - head_poses[0, :3, 3], [0.7, -0.1, 0.0], atol=1e-6)


@_REQUIRES_MUJOCO
@pytest.mark.L0
def test_agibotworld_beta_keeps_debug_detail_out_of_ai_caption(monkeypatch: pytest.MonkeyPatch) -> None:
    dataset = AgiBotWorldBetaDataset(
        root=[],
        chunk_length=2,
        split="full",
        mode="policy",
    )
    video = torch.zeros((3, 3, 4, 4), dtype=torch.float32)  # [T+1,C,H,W]
    wrist_video = torch.zeros((3, 3, 2, 2), dtype=torch.float32)  # [T+1,C,H,W]
    sample = {
        "task": "Open the wardrobe and hang the clothes. | Debug detail for viewer only.",
        "observation.images.head": video,
        "observation.images.hand_left": wrist_video,
        "observation.images.hand_right": wrist_video,
        "observation.states.effector.position": torch.zeros((3, 2), dtype=torch.float32),  # [T+1,2]
        "observation.states.joint.position": torch.zeros((3, 14), dtype=torch.float32),  # [T+1,14]
        "observation.states.head.position": torch.zeros((3, 2), dtype=torch.float32),  # [T+1,2]
        "observation.states.waist.position": torch.zeros((3, 2), dtype=torch.float32),  # [T+1,2]
        **_make_beta_robot_pose(3),
    }

    def _fake_fetch_sample(_idx: int) -> tuple[str, int, int, dict[str, object]]:
        return "policy", 0, 0, sample

    monkeypatch.setattr(dataset, "_fetch_sample", _fake_fetch_sample)

    result = dataset[0]

    assert result["ai_caption"] == "Open the wardrobe and hang the clothes."
    assert result["debug_caption"] == "Debug detail for viewer only."


def test_agibotworld_beta_viewer_entry_declares_agibot_robot_embodiment() -> None:
    assert DATASETS["agibotworld_beta"].robot_embodiment_type == "embodiment_c_gripper"
    assert DATASETS["agibotworld_beta"].dataset_kwargs["return_agibot_link_poses"] is True


def test_embodiment_c_gripper_ext_registered_for_viewer_load() -> None:
    assert DATASETS["embodiment_c_gripper_ext"].robot_embodiment_type == "embodiment_c_gripper_ext"
    assert get_domain_id("embodiment_c_gripper_ext") == get_domain_id("embodiment_c_gripper")


def test_agibot_viewer_entries_do_not_force_debug_roots() -> None:
    for name in ("embodiment_c_gripper", "embodiment_c_gripper_ext", "agibotworld_beta"):
        kwargs = DATASETS[name].dataset_kwargs
        assert "root" not in kwargs
        assert "max_loaded_datasets" not in kwargs


@_REQUIRES_MUJOCO
def test_build_viewer_batch_gripper_ext_shapes() -> None:
    """Ext: 94-dim state → same viewer batch shape as standard gripper."""
    sample = _make_sample(state_dim=94)

    batch = build_viewer_batch(sample, "embodiment_c_gripper_ext")

    assert batch.head_camera_poses.shape == (2, 4, 4)
    assert batch.right_wrist_poses.shape == (2, 4, 4)
    assert batch.left_wrist_poses.shape == (2, 4, 4)
    assert batch.right_gripper is not None
    assert batch.left_gripper is not None
    assert batch.right_gripper.shape == (2,)
    assert batch.left_gripper.shape == (2,)


@_REQUIRES_MUJOCO
def test_ext_fk_uses_correct_state_indices() -> None:
    """Verify ext FK reads arm joints from state[54:68], not state[0:14]."""
    import numpy as np

    from cosmos.data.vfm.action.embodiment_c_fk import compute_fk_transforms

    # Build a 94-dim state where arm joints at [54:68] differ from [0:14].
    state_ext = np.zeros(94, dtype=np.float32)
    # Put distinctive values in the correct ext arm joint positions.
    state_ext[54:61] = [0.1, -0.2, 0.3, -0.1, 0.2, 0.5, -0.3]  # left arm
    state_ext[61:68] = [-0.1, 0.2, -0.3, 0.1, -0.2, -0.5, 0.3]  # right arm
    state_ext[82] = 0.0  # head yaw
    state_ext[83] = 0.3  # head pitch
    state_ext[84] = 0.5  # waist pitch
    state_ext[85] = 0.35  # waist lift

    # Build an equivalent 32-dim standard state with the same joint values.
    state_std = np.zeros(32, dtype=np.float32)
    state_std[0:7] = state_ext[54:61]  # left arm
    state_std[7:14] = state_ext[61:68]  # right arm
    state_std[16] = 0.0  # head yaw
    state_std[17] = 0.3  # head pitch
    state_std[18] = 0.5  # waist pitch
    state_std[19] = 0.35  # waist lift

    fk_ext = compute_fk_transforms(state_ext, "embodiment_c_gripper_ext")
    fk_std = compute_fk_transforms(state_std, "embodiment_c_gripper")

    # Ext and standard FK should produce identical transforms.
    for key in ("head_camera", "right_wrist", "left_wrist"):
        np.testing.assert_allclose(fk_ext[key], fk_std[key], atol=1e-6, err_msg=f"FK mismatch for {key}")


@_REQUIRES_MUJOCO
def test_ext_fk_applies_robot_base_motion_to_batch_poses() -> None:
    """Ext FK folds state/robot pose into all head and wrist trajectories."""
    import numpy as np

    from cosmos.data.vfm.action.embodiment_c_fk import compute_fk_transforms_batch

    states = np.zeros((2, 94), dtype=np.float32)  # [T,S]
    states[:, AGIBOT_GEAR_EXT_STATE_ROBOT_ORIENTATION_SLICE] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    states[1, AGIBOT_GEAR_EXT_STATE_ROBOT_POSITION_SLICE] = np.array([0.4, -0.2, 0.0], dtype=np.float32)

    fk = compute_fk_transforms_batch(states, "embodiment_c_gripper_ext")

    for key in ("head_camera", "right_wrist", "left_wrist"):
        np.testing.assert_allclose(fk[key][1, :3, 3] - fk[key][0, :3, 3], [0.4, -0.2, 0.0], atol=1e-6)
