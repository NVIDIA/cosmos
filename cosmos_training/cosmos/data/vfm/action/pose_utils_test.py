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
import torch
from scipy.spatial.transform import Rotation as R

from cosmos.data.vfm.action.pose_utils import (
    _normalize_rotation_matrices,
    _to_numpy_float32,
    build_abs_pose_from_components,
    convert_rotation,
    pose_abs_to_rel,
    pose_rel_to_abs,
)


def _make_example_poses_abs() -> np.ndarray:
    xyz = np.array(
        [
            [1.0, -0.5, 0.25],
            [1.5, 0.0, 0.75],
            [2.0, 0.5, 1.5],
            [2.5, 1.0, 2.0],
        ],
        dtype=np.float32,
    )
    euler = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.1, -0.2, 0.3],
            [0.2, 0.15, -0.1],
            [-0.25, 0.05, 0.4],
        ],
        dtype=np.float32,
    )
    return build_abs_pose_from_components(xyz, euler, "euler_xyz")


@pytest.mark.L0
def test_to_numpy_float32_raises_on_requires_grad_tensor() -> None:
    """Tensor inputs with gradients must be explicitly detached by callers."""
    x = torch.randn(2, 3, requires_grad=True)
    with pytest.raises(ValueError, match="non-differentiable"):
        _to_numpy_float32(x)


@pytest.mark.L0
def test_build_abs_pose_from_components_supports_quat_wxyz() -> None:
    """AV-style wxyz quaternions should produce the same matrices as xyzw."""
    xyz = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]], dtype=np.float32)
    quat_xyzw = np.array(
        [
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4)],
        ],
        dtype=np.float32,
    )
    quat_wxyz = quat_xyzw[:, [3, 0, 1, 2]]

    poses_xyzw = build_abs_pose_from_components(xyz, quat_xyzw, "quat_xyzw")
    poses_wxyz = build_abs_pose_from_components(xyz, quat_wxyz, "quat_wxyz")

    np.testing.assert_allclose(poses_xyzw, poses_wxyz, atol=1e-6)


@pytest.mark.L0
def test_build_abs_pose_from_components_matches_manual_euler_conversion() -> None:
    """Euler component helper should match the previous matrix-building pattern."""
    xyz = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 2.0, 0.0],
        ],
        dtype=np.float32,
    )
    euler = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, np.pi / 2],
            [0.0, np.pi / 4, np.pi / 2],
        ],
        dtype=np.float32,
    )

    poses_abs = build_abs_pose_from_components(xyz, euler, "euler_xyz")
    manual_poses_abs = np.tile(np.eye(4, dtype=np.float32), (xyz.shape[0], 1, 1))
    manual_poses_abs[:, :3, :3] = R.from_euler("xyz", euler).as_matrix()
    manual_poses_abs[:, :3, 3] = xyz

    actual = pose_abs_to_rel(poses_abs, rotation_format="rot6d", pose_convention="backward_framewise")
    expected = pose_abs_to_rel(manual_poses_abs, rotation_format="rot6d", pose_convention="backward_framewise")

    np.testing.assert_allclose(actual, expected, atol=1e-6)


@pytest.mark.L0
def test_build_abs_pose_from_components_applies_translation_scale() -> None:
    """Explicit translation scaling should be applied before building pose matrices."""
    xyz = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [1.5, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    euler = np.zeros((3, 3), dtype=np.float32)

    poses_abs = build_abs_pose_from_components(xyz, euler, "euler_xyz", translation_scale=2.0)

    np.testing.assert_allclose(poses_abs[:, :3, 3], xyz / 2.0, atol=1e-6)


@pytest.mark.L0
def test_pose_abs_to_rel_rotation_formats_follow_centralized_conventions() -> None:
    """Relative-pose conversion should emit the canonical rot6d and euler_xyz blocks."""
    poses_abs = np.tile(np.eye(4, dtype=np.float32), (3, 1, 1))
    euler = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, np.pi / 4, np.pi / 2],
        ],
        dtype=np.float32,
    )
    matrices_np = R.from_euler("xyz", euler).as_matrix().astype(np.float32)
    poses_abs[1:, :3, :3] = matrices_np

    rel_6d = pose_abs_to_rel(poses_abs, rotation_format="rot6d", pose_convention="backward_anchored")
    expected_rot6d = matrices_np[:, :, :2].transpose(0, 2, 1).reshape(2, 6)
    np.testing.assert_allclose(rel_6d[:, 3:], expected_rot6d, atol=1e-6)

    rel_3d = pose_abs_to_rel(poses_abs, rotation_format="euler_xyz", pose_convention="backward_anchored")
    expected_rot3d = R.from_matrix(matrices_np).as_euler("xyz", degrees=False)
    np.testing.assert_allclose(rel_3d[:, 3:], expected_rot3d, atol=1e-6)


@pytest.mark.L0
def test_convert_rotation_rot6d_to_matrix_uses_column_based_action_convention() -> None:
    """rot6d roundtrip should preserve matrices under the centralized column-based convention."""
    euler = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, np.pi / 4, np.pi / 2],
        ],
        dtype=np.float32,
    )
    matrices_np = R.from_euler("xyz", euler).as_matrix().astype(np.float32)
    rot6d = matrices_np[:, :, :2].transpose(0, 2, 1).reshape(2, 6)

    reconstructed = convert_rotation(rot6d, input_format="rot6d", output_format="matrix")

    np.testing.assert_allclose(reconstructed, matrices_np, atol=1e-6)


@pytest.mark.L0
def test_normalize_rotation_matrices_batched_matches_reference_loop() -> None:
    """Batched SVD normalization should match the previous per-matrix loop behavior."""
    rng = np.random.default_rng(42)
    matrices = rng.normal(size=(32, 3, 3)).astype(np.float32)

    # New batched implementation.
    actual = _normalize_rotation_matrices(matrices)

    # Reference: previous loop implementation.
    expected_list: list[np.ndarray] = []
    for rot_mat in matrices:
        U, _, Vt = np.linalg.svd(rot_mat)
        normalized = U @ Vt
        if np.linalg.det(normalized) < 0:
            U[:, -1] *= -1
            normalized = U @ Vt
        expected_list.append(normalized.astype(np.float32))
    expected = np.stack(expected_list, axis=0)

    np.testing.assert_allclose(actual, expected, atol=1e-6)
    np.testing.assert_allclose(np.linalg.det(actual), np.ones(actual.shape[0], dtype=np.float32), atol=1e-5)


@pytest.mark.L0
@pytest.mark.parametrize("rotation_format", ["rot9d", "rot6d", "quat_xyzw", "euler_xyz", "axisangle"])
@pytest.mark.parametrize(
    "pose_convention",
    ["backward_anchored", "backward_framewise"],
)
def test_pose_abs_to_rel_roundtrips_through_pose_rel_to_abs(
    rotation_format: str,
    pose_convention: str,
) -> None:
    """Relative pose encoding should invert back to the original absolute poses."""
    poses_abs = _make_example_poses_abs()

    poses_rel = pose_abs_to_rel(
        poses_abs,
        rotation_format=rotation_format,
        pose_convention=pose_convention,
        translation_scale=2.5,
    )
    reconstructed = pose_rel_to_abs(
        poses_rel,
        rotation_format=rotation_format,
        pose_convention=pose_convention,
        initial_pose=poses_abs[0],
        translation_scale=2.5,
    )

    np.testing.assert_allclose(reconstructed, poses_abs, atol=1e-5)
