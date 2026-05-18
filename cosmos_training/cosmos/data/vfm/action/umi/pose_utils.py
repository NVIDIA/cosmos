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

from typing import Any

import numpy as np
import numpy.typing as npt
from numba import njit


@njit(cache=True)
def qmult(q1: npt.NDArray[Any], q2: npt.NDArray[Any]) -> npt.NDArray[Any]:
    q = np.array(
        [
            q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2] - q1[3] * q2[3],
            q1[0] * q2[1] + q1[1] * q2[0] + q1[2] * q2[3] - q1[3] * q2[2],
            q1[0] * q2[2] - q1[1] * q2[3] + q1[2] * q2[0] + q1[3] * q2[1],
            q1[0] * q2[3] + q1[1] * q2[2] - q1[2] * q2[1] + q1[3] * q2[0],
        ]
    )

    return q


@njit(cache=True)
def qconjugate(q: npt.NDArray[Any]) -> npt.NDArray[Any]:
    return np.array([q[0], -q[1], -q[2], -q[3]])


@njit(cache=True)
def get_absolute_pose(
    init_pose_xyz_wxyz: npt.NDArray[Any],
    relative_pose_xyz_wxyz: npt.NDArray[Any],
):
    """The new pose is in the same frame of reference as the initial pose"""
    new_pose_xyz_wxyz = np.zeros(7, init_pose_xyz_wxyz.dtype)
    relative_pos_in_init_frame_as_quat_wxyz = np.zeros(4, init_pose_xyz_wxyz.dtype)
    relative_pos_in_init_frame_as_quat_wxyz[1:] = relative_pose_xyz_wxyz[:3]
    init_rot_qinv = qconjugate(init_pose_xyz_wxyz[3:])
    relative_pos_in_world_frame_as_quat_wxyz = qmult(
        qmult(init_pose_xyz_wxyz[3:], relative_pos_in_init_frame_as_quat_wxyz),
        init_rot_qinv,
    )
    new_pose_xyz_wxyz[:3] = init_pose_xyz_wxyz[:3] + relative_pos_in_world_frame_as_quat_wxyz[1:]
    quat = qmult(init_pose_xyz_wxyz[3:], relative_pose_xyz_wxyz[3:])
    if quat[0] < 0:
        quat = -quat
    new_pose_xyz_wxyz[3:] = quat
    return new_pose_xyz_wxyz


@njit(cache=True)
def get_relative_pose(
    new_pose_xyz_wxyz: npt.NDArray[Any],
    init_pose_xyz_wxyz: npt.NDArray[Any],
):
    """The two poses are in the same frame of reference"""
    relative_pose_xyz_wxyz = np.zeros(7, new_pose_xyz_wxyz.dtype)
    relative_pos_in_world_frame_as_quat_wxyz = np.zeros(4, new_pose_xyz_wxyz.dtype)
    relative_pos_in_world_frame_as_quat_wxyz[1:] = new_pose_xyz_wxyz[:3] - init_pose_xyz_wxyz[:3]
    init_rot_qinv = qconjugate(init_pose_xyz_wxyz[3:])
    relative_pose_xyz_wxyz[:3] = qmult(
        qmult(init_rot_qinv, relative_pos_in_world_frame_as_quat_wxyz),
        init_pose_xyz_wxyz[3:],
    )[1:]
    quat = qmult(init_rot_qinv, new_pose_xyz_wxyz[3:])
    if quat[0] < 0:
        quat = -quat
    relative_pose_xyz_wxyz[3:] = quat
    return relative_pose_xyz_wxyz


@njit(cache=True)
def invert_pose(pose_xyz_wxyz: npt.NDArray[Any]) -> npt.NDArray[Any]:
    qinv = qconjugate(pose_xyz_wxyz[3:])
    pos_quat_wxyz = np.zeros(4, pose_xyz_wxyz.dtype)
    pos_quat_wxyz[1:] = pose_xyz_wxyz[:3]
    rotated_pos = qmult(
        qmult(qinv, pos_quat_wxyz),
        pose_xyz_wxyz[3:],
    )
    inverted_pose = np.zeros(7, pose_xyz_wxyz.dtype)
    inverted_pose[:3] = -rotated_pos[1:]
    if qinv[0] < 0:
        qinv = -qinv
    inverted_pose[3:] = qinv
    return inverted_pose


@njit(cache=True)
def quat_wxyz_to_rot_6d(quat_wxyz: npt.NDArray[Any]) -> npt.NDArray[Any]:
    """
    Convert a quaternion to a 6D representation: the first two rows of the corresponding rotation matrix.
    https://arxiv.org/pdf/1812.07035
    quat_wxyz: (4, )
    return: (6, )
    """
    assert quat_wxyz.shape == (4,)
    w, x, y, z = quat_wxyz[0], quat_wxyz[1], quat_wxyz[2], quat_wxyz[3]

    R = np.array(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
            [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
            [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
        ]
    )

    rot_6d = np.zeros(6)
    rot_6d[:3] = R[0, :]
    rot_6d[3:] = R[1, :]

    return rot_6d


@njit(cache=True)
def rot_6d_to_quat_wxyz(rot_6d: npt.NDArray[Any]) -> npt.NDArray[Any]:
    """
    Convert a 6D representation to a quaternion.
    https://arxiv.org/pdf/1812.07035
    rot_6d: (6, )
    return: (4, )
    """

    assert rot_6d.shape == (6,)
    a1, a2 = rot_6d[:3], rot_6d[3:]
    b1 = a1 / np.linalg.norm(a1)
    b2 = a2 - np.dot(b1, a2) * b1
    b2 = b2 / np.linalg.norm(b2)
    b3 = np.cross(b1, b2)

    m = np.zeros((3, 3))
    m[0, :] = b1
    m[1, :] = b2
    m[2, :] = b3

    trace = np.trace(m)

    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m[2, 1] - m[1, 2]) * s
        y = (m[0, 2] - m[2, 0]) * s
        z = (m[1, 0] - m[0, 1]) * s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s

    return np.array([w, x, y, z])


# @njit(cache=True)
def quat_wxyz_to_rot_6d_batch(quat_wxyz: npt.NDArray[Any]) -> npt.NDArray[Any]:
    """
    input (..., 4)
    output (..., 6)
    """
    assert quat_wxyz.shape[-1] == 4
    input_shape = quat_wxyz.shape[:-1]
    quat_wxyz = quat_wxyz.copy().reshape(-1, 4)
    rot_6d = np.zeros((quat_wxyz.shape[0], 6))
    for i in range(quat_wxyz.shape[0]):
        rot_6d[i] = quat_wxyz_to_rot_6d(quat_wxyz[i])
    return rot_6d.reshape(*input_shape, 6)


# @njit(cache=True)
def rot_6d_to_quat_wxyz_batch(rot_6d: npt.NDArray[Any]) -> npt.NDArray[Any]:
    """
    input (..., 6)
    output (..., 4)
    """
    assert rot_6d.shape[-1] == 6, f"rot_6d.shape: {rot_6d.shape}"
    input_shape = rot_6d.shape[:-1]
    rot_6d = rot_6d.copy().reshape(-1, 6)
    quat_wxyz = np.zeros((rot_6d.shape[0], 4))
    for i in range(rot_6d.shape[0]):
        quat_wxyz[i] = rot_6d_to_quat_wxyz(rot_6d[i])
    return quat_wxyz.reshape(*input_shape, 4)


def convert_batch_to_10d(eef_xyz_wxyz: npt.NDArray[np.float32], gripper_width: npt.NDArray[np.float32]):
    """
    eef_xyz_wxyz: (batch_size, obs_history_len, 7)
    gripper_width: (batch_size, obs_history_len, 1)

    return:
        robot0_10d: (batch_size, obs_history_len, 10)
    """

    assert eef_xyz_wxyz.shape[0:2] == gripper_width.shape[0:2]

    batch_size, obs_history_len = eef_xyz_wxyz.shape[0:2]

    pose10d = np.zeros((batch_size, obs_history_len, 10))
    pose10d[:, :, :3] = eef_xyz_wxyz[:, :, :3]
    pose10d[:, :, 3:9] = quat_wxyz_to_rot_6d_batch(eef_xyz_wxyz[:, :, 3:])
    pose10d[:, :, 9] = gripper_width[:, :, 0]

    return pose10d


def convert_10d_to_batch(pose10d: npt.NDArray[np.float32]):
    """
    pose10d: (batch_size, obs_history_len, 10)

    return:
        eef_xyz_wxyz: (batch_size, obs_history_len, 7)
        gripper_width: (batch_size, obs_history_len, 1)
    """
    batch_size, obs_history_len = pose10d.shape[0:2]
    eef_xyz_wxyz = np.zeros((batch_size, obs_history_len, 7), dtype=np.float32)
    eef_xyz_wxyz[:, :, :3] = pose10d[:, :, :3]
    eef_xyz_wxyz[:, :, 3:] = rot_6d_to_quat_wxyz_batch(pose10d[:, :, 3:9])
    gripper_width = pose10d[:, :, 9:]

    return eef_xyz_wxyz, gripper_width


# def pos_axang_to_mat(position: npt.NDArray[Any], axis_angle: npt.NDArray[Any]) -> npt.NDArray[Any]:
#     """
#     Convert a position and axis-angle to a 4x4 matrix.
#     position: (3, ) or (N, 3)
#     axis_angle: (3, ) or (N, 3)
#     return: (4, 4) or (N, 4, 4)
#     """
#     pass

# def mat_to_pose10d(mat: npt.NDArray[Any]) -> npt.NDArray[Any]:
#     """
#     Convert a 4x4 matrix to a 10D pose.
#     mat: (4, 4) or (N, 4, 4)
#     return: (10, ) or (N, 10)
#     """
#     if len(mat.shape) == 3:
#         pose10d = np.zeros((mat.shape[0], 10))
#         pose10d[:, :3] = mat[:, :3, 3]
#         pose10d[:, 3:] = quat_wxyz_to_rot_6d(mat[:, :3, :3])
#     else:
#         pose10d = np.zeros(10)
#         pose10d[:3] = mat[:3, 3]
#         pose10d[3:] = quat_wxyz_to_rot_6d(mat[:3, :3])


if __name__ == "__main__":
    pose_1 = np.random.rand(7)
    pose_1[3:] /= np.linalg.norm(pose_1[3:])
    pose_2 = np.random.rand(7)
    pose_2[3:] /= np.linalg.norm(pose_2[3:])

    pose_2_relative_to_pose_1 = get_relative_pose(pose_2, pose_1)
    absolute_pose_2 = get_absolute_pose(pose_1, pose_2_relative_to_pose_1)
    assert np.allclose(pose_2, absolute_pose_2)
    pose_1_relative_to_pose_2 = get_relative_pose(pose_1, pose_2)
    absolute_pose_1 = get_absolute_pose(pose_2, pose_1_relative_to_pose_2)
    assert np.allclose(pose_1, absolute_pose_1)
    inverted_pose_2_relative_to_pose_1 = invert_pose(pose_2_relative_to_pose_1)
    assert np.allclose(inverted_pose_2_relative_to_pose_1, pose_1_relative_to_pose_2)

    assert np.allclose(pose_1, invert_pose(invert_pose(pose_1)))
    assert np.allclose(pose_2, invert_pose(invert_pose(pose_2)))

    rot_6d = quat_wxyz_to_rot_6d(pose_1[3:])
    quat_wxyz = rot_6d_to_quat_wxyz(rot_6d)
    assert np.allclose(quat_wxyz, pose_1[3:])

    rot_6d = quat_wxyz_to_rot_6d(pose_2[3:])
    quat_wxyz = rot_6d_to_quat_wxyz(rot_6d)
    assert np.allclose(quat_wxyz, pose_2[3:])

    print("Test passed")
