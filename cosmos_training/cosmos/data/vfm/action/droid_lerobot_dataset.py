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

import json
import os
import random
from typing import Any, cast

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R

from cosmos.utils import log
from cosmos.data.vfm.action.cosmos3_action_lerobot import (
    ActionNormalization,
    ActionSpec,
    BaseActionLeRobotDataset,
    Gripper,
    Joint,
    Pos,
    Rot,
    build_action_spec,
    build_episode_spans,
    split_episode_ids,
)
from cosmos.data.vfm.action.droid_lerobot_dataset_config import (
    _GRIPPER_STATE_FEATURE,
    _JOINT_ACTION_FEATURE,
    _JOINT_STATE_FEATURE,
    ACTION_FEATURES,
    HAS_MULTI_LANGUAGE_ANNOTATIONS,
    IMAGE_FEATURES,
    IS_FLAT_ACTION,
    IS_GRIPPER_ACTION_FLIPPED,
    LEROBOT_ROOTS,
    STATE_FEATURES,
)
from cosmos.data.vfm.action.pose_utils import (
    PoseConvention,
    build_abs_pose_from_components,
    convert_rotation,
    pose_abs_to_rel,
)
from cosmos.data.vfm.action.viewpoint_utils import Viewpoint

_FILTER_DICT_PATH = "/lustre/fsw/portfolios/cosmos/projects/cosmos_base_training/users/ychao/datasets/raw/KarlP-droid/keep_ranges_1_0_1.json"

# 90-degree clockwise rotation about the Z axis (in local frame), converting
# DROID Franka panda_link8 orientation to the OpenCV camera convention.
_DROID_TO_OPENCV: np.ndarray = np.array(
    [
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)


class DROIDLeRobotDataset(BaseActionLeRobotDataset):
    """ """

    def __init__(
        self,
        root: str = "/lustre/fsw/portfolios/cosmos/projects/cosmos_base_training/cosmos3_action_datasets/droid_plus_lerobot_640x360_20260412",
        fps: float = 15.0,
        chunk_length: int = 16,
        split_seed: int = 42,
        split_val_ratio: float = 0.03,
        split: str = "train",
        mode: str = "policy",
        pose_convention: PoseConvention = "backward_framewise",
        action_normalization: ActionNormalization | None = None,
        tolerance_s=2e-4,
        viewpoint: Viewpoint = "concat_view",
        use_success_only: bool = False,
        video_mode: str | None = None,
        action_space: str = "midtrain",
        use_state: bool = False,
        use_filter_dict: bool = False,
        enable_fast_init: bool = False,
        max_num_history_actions: int = 0,
    ) -> None:
        """ """
        super().__init__(
            fps=fps,
            chunk_length=chunk_length,
            split_seed=split_seed,
            split_val_ratio=split_val_ratio,
            split=split,
            mode=mode,
            embodiment_type="droid_lerobot",
            viewpoint=viewpoint,
            pose_convention=pose_convention,
            rotation_format="rot6d",
            action_normalization=action_normalization,
            tolerance_s=tolerance_s,
            enable_fast_init=enable_fast_init,
        )
        self._use_success_only = use_success_only
        self._video_mode = video_mode
        self._action_space = action_space
        self._use_state = use_state
        self._use_filter_dict = use_filter_dict
        self._max_num_history_actions = max_num_history_actions
        if max_num_history_actions > 0 and action_space not in ("midtrain", "joint_pos"):
            raise ValueError(
                f"max_num_history_actions is only supported with action_space='midtrain' or 'joint_pos', got {action_space!r}"
            )

        self._is_val_temp_seg = split == "val_temp_seg"
        self._to_opencv = _DROID_TO_OPENCV

        version = os.path.basename(root)
        try:
            lerobot_roots = LEROBOT_ROOTS[version]
            self._image_features = IMAGE_FEATURES[version]
            self._state_features = STATE_FEATURES[version]
            self._action_features = ACTION_FEATURES[version]
            self._is_flat_action = IS_FLAT_ACTION[version]
            self._has_multi_language_annotations = HAS_MULTI_LANGUAGE_ANNOTATIONS[version]
            self._is_gripper_action_flipped = IS_GRIPPER_ACTION_FLIPPED[version]
        except KeyError as e:
            raise ValueError(f"Unknown version: {version!r}. Supported: {list(LEROBOT_ROOTS.keys())}") from e

        if self._use_success_only and lerobot_roots:
            lerobot_roots = [x for x in lerobot_roots if x.split("/", 1)[0] == "success"]

        self._all_shard_roots = [os.path.join(root, x) for x in lerobot_roots] if lerobot_roots else [root]

        observation_ts = [i * self._dt for i in range(0, self._chunk_length + 1)]
        action_ts = [i * self._dt for i in range(0, self._chunk_length)]
        if self._max_num_history_actions > 0 and self._action_space in ("midtrain", "joint_pos"):
            observation_ts_ext = [i * self._dt for i in range(-self._max_num_history_actions, self._chunk_length + 1)]
            action_ts_ext = [i * self._dt for i in range(-self._max_num_history_actions, self._chunk_length)]
        else:
            observation_ts_ext = observation_ts
            action_ts_ext = action_ts
        self._delta_timestamps: dict[str, list[float]] = {
            self._state_features: observation_ts_ext,
            self._action_features: action_ts_ext,
        }
        if self._viewpoint in ("wrist_view", "concat_view"):
            self._delta_timestamps[self._image_features["wrist"]] = observation_ts
        if self._viewpoint in ("third_person_view", "concat_view"):
            self._delta_timestamps[self._image_features["left"]] = observation_ts
            self._delta_timestamps[self._image_features["right"]] = observation_ts
        if self._action_space == "joint_pos":
            self._delta_timestamps[_JOINT_ACTION_FEATURE] = action_ts
            if self._use_state or self._max_num_history_actions > 0:
                self._delta_timestamps[_JOINT_STATE_FEATURE] = observation_ts_ext
                self._delta_timestamps[_GRIPPER_STATE_FEATURE] = observation_ts_ext
        if self._use_state and self._action_space != "joint_pos":
            self._delta_timestamps[_GRIPPER_STATE_FEATURE] = observation_ts

        if self._use_filter_dict:
            with open(_FILTER_DICT_PATH) as f:
                self._filter_dict = json.load(f)

    def _append_index_records(self, *, meta, ds_idx: int, dataset_label: str | None = None) -> None:
        """ """
        if not self._use_filter_dict:
            super()._append_index_records(meta=meta, ds_idx=ds_idx, dataset_label=dataset_label)
            return

        episode_ids = split_episode_ids(
            total_episodes=meta.total_episodes,
            seed=self._split_seed,
            val_ratio=self._split_val_ratio,
            split=self._split,
        )
        episode_spans, _, sample_count = build_episode_spans(
            meta.episodes, episode_ids, self._chunk_length, sample_stride=self._sample_stride
        )

        class_name = self.__class__.__name__
        label = f" [{dataset_label}]"

        log.info(f"{class_name}{label}: split={self._split}, num episodes={len(episode_ids)}")

        filtered_count = 0
        for episode_id, sample_start, valid_len in episode_spans:
            ep_id_str = meta.episodes[episode_id]["episode_id"]
            episode_key = f"gs://xembodiment_data/r2d2/r2d2-data-full/{ep_id_str}/recordings/MP4--gs://xembodiment_data/r2d2/r2d2-data-full/{ep_id_str}/trajectory.h5"
            ranges = self._filter_dict.get(episode_key)
            if ranges is None:
                continue
            for s, e in ranges:
                sub_start = max(s, 0)
                sub_end = min(e - self._chunk_length, valid_len)
                sub_valid_len = max(0, sub_end - sub_start)
                if sub_valid_len > 0:
                    self._episode_records.append((ds_idx, sample_start + sub_start, sub_valid_len, episode_id))
                    self._num_valid_indices += sub_valid_len
                    self._episode_cum_ends.append(self._num_valid_indices)
                    filtered_count += sub_valid_len

        if sample_count > 0:
            log.info(
                f"{class_name}{label}: kept {filtered_count} / {sample_count} ({100.0 * filtered_count / sample_count:.2f} %) samples"
            )

    def _register_sources(self, indices: list[int] | None = None) -> None:
        """ """
        super()._register_sources(indices)
        if self._is_val_temp_seg:
            self._apply_temp_seg_filter()

    def _apply_temp_seg_filter(self) -> None:
        """Replace index records with one high-scoring segment per episode.

        A segment is interesting if either:
        - The gripper action changes significantly (open/close transition), or
        - The gripper is closed and the end-effector position is moving.
        Among qualifying segments the one with the highest score is kept.
        """
        ds = self._get_dataset(0)
        chunk_size = self._chunk_length + 1
        gripper_change_threshold = 0.5
        ee_movement_threshold = 0.01

        new_records: list[tuple[int, int, int, int]] = []
        num_episodes = len(self._episode_records)

        for ds_idx, sample_start, valid_len, episode_id in self._episode_records:
            end = sample_start + valid_len + self._chunk_length
            num_candidates = valid_len
            if num_candidates <= 0:
                continue

            episode_data = ds.hf_dataset[sample_start:end]
            actions = torch.tensor(np.array(episode_data[self._action_features]))  # [N,action_dim]
            states = torch.tensor(np.array(episode_data[self._state_features]))  # [N,state_dim]

            gripper_action = actions[:, 6] if self._is_flat_action else actions  # [N]
            ee_pos = states[:, :3]  # [N,3]
            ee_disp = (ee_pos[1:] - ee_pos[:-1]).norm(dim=-1)  # [N-1]

            ee_disp_windows = ee_disp.unfold(0, self._chunk_length, 1)  # [num_candidates,chunk_length]
            gripper_windows = gripper_action.unfold(0, chunk_size, 1)  # [num_candidates,chunk_size]

            gripper_range = gripper_windows.max(dim=1).values - gripper_windows.min(dim=1).values  # [num_candidates]
            total_ee_movement = ee_disp_windows.sum(dim=1)  # [num_candidates]
            gripper_closed_ratio = (gripper_windows < 0.5).float().mean(dim=1)  # [num_candidates]

            has_gripper_change = gripper_range > gripper_change_threshold
            gripper_closed = gripper_closed_ratio > 0.5
            has_ee_movement = total_ee_movement > ee_movement_threshold

            scores = torch.zeros(num_candidates)  # [num_candidates]
            scores[has_gripper_change] = 0.5 + gripper_range[has_gripper_change] + total_ee_movement[has_gripper_change]

            closed_and_moving = gripper_closed & ~has_gripper_change & has_ee_movement
            scores[closed_and_moving] = 1.0 + total_ee_movement[closed_and_moving]

            if scores.max().item() > 0:
                best_offset = int(scores.argmax().item())
                new_records.append((ds_idx, sample_start + best_offset, 1, episode_id))

        self._episode_records = new_records
        self._num_valid_indices = len(new_records)
        self._episode_cum_ends = list(range(1, len(new_records) + 1))

        log.info(f"DROIDLeRobotDataset: val_temp_seg kept {len(new_records)} segments from {num_episodes} episodes")

    def _compose_multi_view(self, sample: dict[str, Any]) -> torch.Tensor:
        """Compose wrist, left, and right views into a single frame.

        Layout (per frame):
            ┌──────────────┐
            │    wrist     │   (H, W)
            ├───────┬──────┤
            │ left  │ right│   (H/2, W/2) each
            └───────┴──────┘

        Left and right exterior cameras are downscaled by 2x so that they
        tile to the same width as the wrist view. The output height is 3H/2.

        Returns:
            Composited raw video tensor in ``(T,C,H_out,W)`` float format.
        """
        wrist = sample[self._image_features["wrist"]]  # [T,C,H,W]
        left = sample[self._image_features["left"]]  # [T,C,H_l,W_l]
        right = sample[self._image_features["right"]]  # [T,C,H_r,W_r]

        _, _, h_w, w_w = wrist.shape
        half_h, half_w = h_w // 2, w_w // 2

        left = F.interpolate(left, size=(half_h, half_w), mode="bilinear", align_corners=False)  # [T,C,H/2,W/2]
        right = F.interpolate(right, size=(half_h, half_w), mode="bilinear", align_corners=False)  # [T,C,H/2,W/2]
        bottom = torch.cat([left, right], dim=-1)  # [T,C,H/2,W]

        composite = torch.cat([wrist, bottom], dim=-2)  # [T,C,3H/2,W]
        return composite  # [T,C,3H/2,W]

    def _build_action_spec(self) -> ActionSpec:
        """DROID: 10D ``[Pos, Rot6d, Gripper]`` for ``ee_pose``,
        8D ``[Joint(7), Gripper]`` for ``joint_pos``.
        """
        if self._action_space == "joint_pos":
            return build_action_spec(Joint(n=7, label="joint"), Gripper())
        return build_action_spec(Pos(), Rot("rot6d"), Gripper())

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """ """
        mode, _, _, sample = self._fetch_sample(idx)

        if self._has_multi_language_annotations:
            tasks = sample["task"].split(" | ")
            ai_caption = random.choice(tasks)
        else:
            ai_caption = sample["task"]

        if self._skip_video_loading:
            video = None
        elif self._video_mode is None:
            if self._viewpoint == "concat_view":
                video = self._compose_multi_view(sample)
            else:
                video = sample[self._image_features["wrist"]]  # [T,C,H,W]
        else:
            if self._video_mode == "wrist":
                video = sample[self._image_features["wrist"]]
            if self._video_mode in ("rand_exterior", "wrist_rand_exterior"):
                exterior_key = random.choice([self._image_features["left"], self._image_features["right"]])
                if self._video_mode == "rand_exterior":
                    video = sample[exterior_key]
                else:
                    video = torch.cat([sample[self._image_features["wrist"]], sample[exterior_key]], dim=2)
            if self._video_mode in ("wrist_left_exterior", "wrist_both_exterior"):
                wrist = sample[self._image_features["wrist"]]
                half_h, half_w = wrist.shape[2] // 2, wrist.shape[3] // 2
                left = F.interpolate(
                    sample[self._image_features["left"]], size=(half_h, half_w), mode="bilinear", align_corners=False
                )
                if self._video_mode == "wrist_left_exterior":
                    right = torch.zeros_like(left)
                if self._video_mode == "wrist_both_exterior":
                    right = F.interpolate(
                        sample[self._image_features["right"]],
                        size=(half_h, half_w),
                        mode="bilinear",
                        align_corners=False,
                    )
                video = torch.cat([wrist, torch.cat([left, right], dim=-1)], dim=-2)

        extras: dict[str, Any] = {}

        if self._action_space == "midtrain":
            pose_convention = cast(PoseConvention, self._pose_convention)
            state = sample[self._state_features]  # [T+1, state_dim] or [H+T+1, state_dim]
            poses_abs = build_abs_pose_from_components(state[:, 0:3], state[:, 3:6], "euler_xyz")
            poses_abs[:, :3, :3] = poses_abs[:, :3, :3] @ self._to_opencv
            initial_pose = torch.from_numpy(poses_abs[-self._chunk_length - 1].copy()).float()
            poses_rel = pose_abs_to_rel(poses_abs, rotation_format="rot6d", pose_convention=pose_convention)
            gripper = (
                sample[self._action_features][:, [6]]
                if self._is_flat_action
                else sample[self._action_features].unsqueeze(-1)
            )
            if self._is_gripper_action_flipped:
                gripper = 1.0 - gripper
            action = torch.from_numpy(
                np.concatenate([poses_rel[-self._chunk_length :], gripper[-self._chunk_length :]], axis=-1)
            ).float()  # [T,10]
            extras["initial_pose"] = initial_pose
            if self._max_num_history_actions > 0:
                _, _, _, frame_offset = self._resolve_index(int(idx))
                num_available = min(self._max_num_history_actions, frame_offset * self._sample_stride)
                actual_h = num_available
                # with 0.5 probability, randomly sample the number of history frames
                if random.random() < 0.5:
                    actual_h = random.randint(0, num_available)
                if actual_h > 0:
                    hist_action_raw = torch.from_numpy(
                        np.concatenate(
                            [
                                poses_rel[-self._chunk_length - actual_h : -self._chunk_length],
                                gripper[-self._chunk_length - actual_h : -self._chunk_length],
                            ],
                            axis=-1,
                        )
                    ).float()
                    extras["history_action"] = self._normalize_action(hist_action_raw)
            if self._use_state:
                initial_gripper = sample[_GRIPPER_STATE_FEATURE][0].unsqueeze(-1)
                if self._is_gripper_action_flipped:
                    initial_gripper = 1.0 - initial_gripper
                initial_rot6d = convert_rotation(
                    poses_abs[-self._chunk_length - 1, :3, :3], input_format="matrix", output_format="rot6d"
                )
                initial_state = torch.from_numpy(
                    np.concatenate((poses_abs[-self._chunk_length - 1, :3, 3], initial_rot6d, initial_gripper), axis=-1)
                ).float()
                action = torch.cat([initial_state.unsqueeze(0), action], dim=0)
        if self._action_space == "ee_pose_delta":
            state = sample[self._state_features]
            pose = np.tile(np.eye(4), (state.shape[0], 1, 1))
            pose[:, :3, :3] = R.from_euler("xyz", state[:, 3:6]).as_matrix()
            pose[:, :3, 3] = state[:, 0:3]
            pose_delta = np.linalg.inv(pose[0]) @ pose[1:]
            gripper = sample[self._action_features].unsqueeze(-1)
            if self._is_gripper_action_flipped:
                gripper = 1.0 - gripper
            action = torch.from_numpy(
                np.concatenate((pose_delta[:, :3, 3], pose_delta[:, :3, 0], pose_delta[:, :3, 1], gripper), axis=-1)
            ).float()
            if self._use_state:
                initial_gripper = sample[_GRIPPER_STATE_FEATURE][0].unsqueeze(-1)
                if self._is_gripper_action_flipped:
                    initial_gripper = 1.0 - initial_gripper
                initial_state = torch.from_numpy(
                    np.concatenate((pose[0, :3, 3], pose[0, :3, 0], pose[0, :3, 1], initial_gripper), axis=-1)
                ).float()
                action = torch.cat([initial_state.unsqueeze(0), action], dim=0)
        if self._action_space == "joint_pos":
            gripper = sample[self._action_features][-self._chunk_length :].unsqueeze(-1)
            if self._is_gripper_action_flipped:
                gripper = 1.0 - gripper
            action = torch.cat((sample[_JOINT_ACTION_FEATURE], gripper), dim=-1).float()
            if self._max_num_history_actions > 0:
                _, _, _, frame_offset = self._resolve_index(int(idx))
                num_available = min(self._max_num_history_actions, frame_offset * self._sample_stride)
                actual_h = num_available
                if random.random() < 0.5:
                    actual_h = random.randint(0, num_available)
                if actual_h > 0:
                    hist_joint = sample[_JOINT_STATE_FEATURE][
                        -self._chunk_length - 1 - actual_h : -self._chunk_length - 1
                    ]
                    hist_gripper = sample[_GRIPPER_STATE_FEATURE][
                        -self._chunk_length - 1 - actual_h : -self._chunk_length - 1
                    ].unsqueeze(-1)
                    if self._is_gripper_action_flipped:
                        hist_gripper = 1.0 - hist_gripper
                    hist_action_raw = torch.cat((hist_joint, hist_gripper), dim=-1).float()
                    extras["history_action"] = self._normalize_action(hist_action_raw)
            if self._use_state:
                initial_gripper = sample[_GRIPPER_STATE_FEATURE][-self._chunk_length - 1].unsqueeze(-1)
                if self._is_gripper_action_flipped:
                    initial_gripper = 1.0 - initial_gripper
                initial_state = torch.cat(
                    (sample[_JOINT_STATE_FEATURE][-self._chunk_length - 1], initial_gripper), dim=-1
                ).float()
                action = torch.cat([initial_state.unsqueeze(0), action], dim=0)

        if self._viewpoint == "concat_view" and self._video_mode in (
            None,
            "wrist_left_exterior",
            "wrist_both_exterior",
        ):
            extras["additional_view_description"] = (
                "The top row is from the wrist-mounted camera. "
                "The bottom row contains two horizontally concatenated third-person perspective views of the scene from opposite sides, with the robot visible."
            )

        return self._build_result(
            mode=mode,
            video=video,
            action=action,
            ai_caption=ai_caption,
            **extras,
        )

    @property
    def action_dim(self) -> int:
        """ """
        return 8 if self._action_space == "joint_pos" else 10
