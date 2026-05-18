# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Hand-pose manipulation dataset.

A bimanual human hand dataset in LeRobot v3 format with 21-keypoint hand pose
annotations (positions + per-joint quaternion rotations) in camera space,
along with per-frame camera pose (position + quaternion rotation), task/subtask
labels and RGB video.

Action layout: ``[camera, R_wrist, R_fingers, L_wrist, L_fingers]`` — camera
pose followed by right-hand then left-hand components.

Action space (``pose_convention``)
-------------------------------
Both action spaces share the same three-stage computation:

  1. Compute **absolute** SE(3) poses for camera and both wrists.
  2. Compute **finger positions** in the per-frame wrist coordinate frame.
  3. Convert absolute camera/wrist poses to **relative** representations
     (anchored or frame-wise).

Layout (both modes): ``[camera, R_wrist, R_fingers, L_wrist, L_fingers]``

  ``backward_anchored``
    Camera and wrist poses anchored to frame 0:
    ``P_{0}^{-1} @ P_{t}`` for camera and each wrist.
    Fingers are positions in the current wrist frame.

  ``backward_framewise``
    Frame-wise SE(3) deltas:
    ``P_{t-1}^{-1} @ P_{t}`` for camera and each wrist.
    Fingers are positions in the current wrist frame.

Action dimensions
~~~~~~~~~~~~~~~~~
Both modes have the same dimensionality:
  - Camera: ``3 + rot_dim``
  - Per hand (×2): wrist ``(3 + rot_dim)`` + fingers ``(N_finger × 3)``
  - Total: ``(3 + rot_dim) + 2 × ((3 + rot_dim) + N_finger × 3)``

Example with ``rot6d`` rotation, ``wrist_plus_finger_tips`` (5 fingertips):
  - Camera: 3 + 6 = 9D
  - Per hand: wrist (3 + 6 = 9D) + fingers (5 × 3 = 15D) = 24D
  - Total: 9 + 24 + 24 = **57D**

Rotation format (``rotation_format``)
--------------------------------------
Applied uniformly to both hand joint rotations and camera ego-motion rotation:
  - ``rot9d``: flattened 3x3 rotation matrix (default, converted from quaternions)
  - ``rot6d``: first 2 columns of the rotation matrix (continuous, Zhou et al. CVPR 2019)
  - ``euler_xyz``: Euler ``xyz`` angles in radians
"""

import os
import random
from bisect import bisect_right
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch

from cosmos.utils import log
from cosmos.data.vfm.action.cosmos3_action_lerobot import (
    ActionNormalization,
    BaseActionLeRobotDataset,
    _parallel_map,
    build_episode_spans,
    split_episode_ids,
)
from cosmos.data.vfm.action.hand_pose_dataset_config import (
    CAM_POSITION_KEY,
    CAM_ROTATION_KEY,
    FINGERTIP_JOINT_IDXS,
    HAND_LEFT_POSITION_KEY,
    HAND_LEFT_ROTATION_KEY,
    HAND_POSE_DATASETS,
    HAND_RIGHT_POSITION_KEY,
    HAND_RIGHT_ROTATION_KEY,
    NO_ACTION_SKIP_LABEL_PREFIXES,
    NO_ACTION_SKIP_LABEL_SUBSTRINGS,
    NO_ACTION_SKIP_LABELS,
    NUM_JOINTS,
    QUAT_DIM_PER_JOINT,
    ROTATION_FORMAT_DIM,
    WRIST_FRAME_ALIGN_EMBODIMENT_A,
    WRIST_JOINT_IDX,
)
from cosmos.data.vfm.action.pose_utils import (
    RotationConvention,
    build_abs_pose_from_components,
    pose_abs_to_rel,
)
from cosmos.data.vfm.action.viewpoint_utils import Viewpoint


class HandPoseDataset(BaseActionLeRobotDataset):
    """Hand-pose manipulation dataset backed by LeRobot v3.

    Each sample returns a video chunk and the corresponding hand-pose action
    representation.  Uses deferred source registration via
    ``BaseActionLeRobotDataset``.
    """

    def __init__(
        self,
        root: str | list[str] = HAND_POSE_DATASETS["embodiment_a_feb08_500hr"],
        fps: float = 15.0,
        chunk_length: int = 16,
        split_seed: int = 42,
        split_val_ratio: float = 0.005,
        split: str = "train",
        mode: str = "policy",
        embodiment_type: str = "hand_pose",
        video_key: str = "observation.images.main",
        keypoint_option: Literal[
            "wrist_only", "wrist_plus_fingers", "wrist_plus_finger_tips"
        ] = "wrist_plus_finger_tips",
        rotation_format: RotationConvention = "rot6d",
        pose_convention: Literal[
            "backward_anchored",
            "backward_framewise",
        ] = "backward_framewise",
        action_normalization: ActionNormalization | None = None,
        intra_episode_val_ratio: float = 0.0,
        tolerance_s: float = 2e-4,
        drop_unannotated_edge_frames: bool = True,
        unannotated_pos_l1_threshold: float = 1e-6,
        max_item_retries: int = 16,
        return_overlay_data: bool = False,
        max_episodes: int | None = None,
        episode_ids: list[int] | None = None,
        load_subtasks: bool = False,
        snap_to_subtask: bool = False,
        skip_no_action: bool = False,
        max_subtasks_per_episode: int | None = None,
        viewpoint: Viewpoint = "ego_view",
        enable_fast_init: bool = False,
    ) -> None:
        super().__init__(
            fps=fps,
            chunk_length=chunk_length,
            split_seed=split_seed,
            split_val_ratio=split_val_ratio,
            split=split,
            mode=mode,
            embodiment_type=embodiment_type,
            viewpoint=viewpoint,
            pose_convention=pose_convention,
            rotation_format=rotation_format,
            action_normalization=action_normalization,
            tolerance_s=tolerance_s,
            enable_fast_init=enable_fast_init,
        )
        if max_episodes is not None and episode_ids is not None:
            raise ValueError("Cannot specify both max_episodes and episode_ids.")
        self._max_episodes = max_episodes
        self._episode_ids = episode_ids

        self._video_key = video_key
        self._load_subtasks = load_subtasks or skip_no_action or snap_to_subtask
        self._snap_to_subtask = snap_to_subtask
        self._skip_no_action = skip_no_action
        self._max_subtasks_per_episode = max_subtasks_per_episode
        self._subtask_transitions: dict[tuple[int, int], list[tuple[int, int]]] = {}
        self._per_shard_orig_subtask_names: dict[int, dict[int, str]] = {}
        self._ds_idx_to_shard_idx: list[int] | None = None
        self._orig_subtask_index: dict[int, np.ndarray] = {}
        self._cached_subtask_col: dict[int, np.ndarray] = {}
        self._orig_subtask_transitions: dict[tuple[int, int], list[tuple[int, int]]] = {}
        self._subtask_start_indices: dict[tuple[int, int], list[tuple[int, int]]] = {}
        self._keypoint_option = keypoint_option
        self._rotation_format = rotation_format
        self._pose_convention = pose_convention
        self._intra_episode_val_ratio = intra_episode_val_ratio
        self._drop_unannotated_edge_frames = drop_unannotated_edge_frames
        self._unannotated_pos_l1_threshold = unannotated_pos_l1_threshold
        self._max_item_retries = max_item_retries
        self._warned_tolerance_once = False
        self._warned_unannotated_once = False
        self._warned_nan_action_once = False
        self._warned_build_action_error_once = False
        self._logged_decode_failures: set[tuple[int, int, int]] = set()
        self._return_overlay_data = return_overlay_data

        # Normalize root: config may pass OmegaConf ListConfig for list of shards; ensure list of str.
        if isinstance(root, str):
            _shard_roots = [root]
        else:
            _shard_roots = [str(p) for p in root]

        self._root = _shard_roots[0]
        self._all_shard_roots = _shard_roots

        # Hand pose keys (camera-space positions + rotations).
        self._position_keys = [HAND_LEFT_POSITION_KEY, HAND_RIGHT_POSITION_KEY]
        self._rotation_keys = [HAND_LEFT_ROTATION_KEY, HAND_RIGHT_ROTATION_KEY]

        if keypoint_option not in {"wrist_only", "wrist_plus_fingers", "wrist_plus_finger_tips"}:
            raise ValueError(f"Unsupported keypoint_option: {keypoint_option!r}")
        if not (0.0 <= intra_episode_val_ratio < 1.0):
            raise ValueError(f"intra_episode_val_ratio must be in [0, 1). Got: {intra_episode_val_ratio}")

        if keypoint_option == "wrist_plus_finger_tips":
            self._position_joint_indices: tuple[int, ...] = (WRIST_JOINT_IDX, *FINGERTIP_JOINT_IDXS)
        elif keypoint_option == "wrist_plus_fingers":
            self._position_joint_indices = tuple(range(NUM_JOINTS))
        else:
            self._position_joint_indices = (WRIST_JOINT_IDX,)

        # Compute raw action dimension.
        # Layout: [camera, R_wrist, R_fingers, L_wrist, L_fingers]
        # Camera: 3 (pos) + rot_dim.  Per hand: 3 (wrist pos) + rot_dim + (N_finger × 3).
        rot_dim_per_joint = ROTATION_FORMAT_DIM[rotation_format]
        num_pos_joints = len(self._position_joint_indices)
        num_finger_joints = num_pos_joints - 1  # exclude wrist
        per_hand_dim = 3 + rot_dim_per_joint + num_finger_joints * 3
        self._raw_action_dim = (3 + rot_dim_per_joint) + 2 * per_hand_dim

        # Build delta_timestamps: T+1 video frames, T+1 hand states.
        ts = [i * self._dt for i in range(self._chunk_length + 1)]
        self._delta_timestamps = {
            self._video_key: list(ts),
        }
        for key in (
            HAND_LEFT_POSITION_KEY,
            HAND_RIGHT_POSITION_KEY,
            HAND_LEFT_ROTATION_KEY,
            HAND_RIGHT_ROTATION_KEY,
        ):
            self._delta_timestamps[key] = list(ts)

        # Always load camera pose (both action spaces include camera).
        self._delta_timestamps[CAM_POSITION_KEY] = list(ts)
        self._delta_timestamps[CAM_ROTATION_KEY] = list(ts)

        self._episode_intrinsics: dict[int, np.ndarray] = {}
        if return_overlay_data:
            self._load_episode_intrinsics(self._all_shard_roots[0])

        # Load task/subtask label mappings for captions.
        self._task_names: dict[int, str] = {}
        self._per_shard_subtask_names: dict[int, dict[int, str]] = {}
        self._load_task_labels()

        _POSE_CONVENTION_DESC = {
            "backward_anchored": (
                "Camera + wrist poses anchored to frame 0, fingers in wrist frame. "
                "Layout: [camera, R_wrist, R_fingers, L_wrist, L_fingers]"
            ),
            "backward_framewise": (
                "Frame-wise camera + wrist deltas, fingers in wrist frame. "
                "Layout: [camera, R_wrist, R_fingers, L_wrist, L_fingers]"
            ),
        }
        log.info(
            f"HandPoseDataset configuration:\n"
            f"  root            = {root}\n"
            f"  split           = {split}\n"
            f"  mode            = {mode}\n"
            f"  pose_convention    = {pose_convention}\n"
            f"    -> {_POSE_CONVENTION_DESC.get(pose_convention, 'unknown')}\n"
            f"  intra_episode_val_ratio = {intra_episode_val_ratio}\n"
            f"  position_keys   = {self._position_keys}\n"
            f"  rotation_keys   = {self._rotation_keys}\n"
            f"  keypoint_option = {keypoint_option}\n"
            f"  selected_joint_count = {num_pos_joints} ({num_finger_joints} finger joints)\n"
            f"  rotation_format = {rotation_format} ({rot_dim_per_joint}D per joint)\n"
            f"  raw_action_dim  = {self._raw_action_dim} "
            f"(cam {3 + rot_dim_per_joint}D + 2 × hand {per_hand_dim}D)\n"
            f"  chunk_length    = {chunk_length} (video frames: {chunk_length + 1})\n"
            f"  fps             = {fps}\n"
            f"  tolerance_s     = {tolerance_s}\n"
            f"  drop_unannotated_edge_frames = {drop_unannotated_edge_frames}\n"
            f"  unannotated_pos_l1_threshold = {unannotated_pos_l1_threshold}\n"
            f"  max_item_retries = {max_item_retries}\n"
            f"  max_episodes    = {max_episodes}\n"
            f"  episode_ids     = {episode_ids}\n"
            f"  snap_to_subtask = {snap_to_subtask}\n"
            f"  skip_no_action  = {skip_no_action}\n"
            f"  max_subtasks_per_episode = {self._max_subtasks_per_episode}\n"
            f"  domain_id       = {self._domain_id} ({embodiment_type})"
        )

    # -------------------------------------------------------------------------
    # Per-episode subtask transition cache
    # -------------------------------------------------------------------------

    def _get_subtask_transitions(self, ds_idx: int, ep_idx: int) -> list[tuple[int, int]]:
        """Return sorted ``(row_start, subtask_index)`` transitions for an episode.

        Built lazily on first access by slicing the ``subtask_index`` column
        from the LeRobot HF dataset (cheap Arrow column access).
        """
        key = (ds_idx, ep_idx)
        cached = self._subtask_transitions.get(key)
        if cached is not None:
            return cached

        ds = self._get_dataset(ds_idx)
        ep = ds.meta.episodes[ep_idx]
        ep_start: int = ep["dataset_from_index"]
        ep_end: int = ep["dataset_to_index"]

        if ds._absolute_to_relative_idx is not None:
            rel_indices = [ds._absolute_to_relative_idx[i] for i in range(ep_start, ep_end)]
            subtask_col = ds.hf_dataset["subtask_index"][rel_indices]
        else:
            subtask_col = ds.hf_dataset["subtask_index"][ep_start:ep_end]

        transitions: list[tuple[int, int]] = []
        prev_si = None
        for offset, si_val in enumerate(subtask_col):
            si = int(si_val) if not isinstance(si_val, int) else si_val
            if si != prev_si:
                transitions.append((ep_start + offset, si))
                prev_si = si

        self._subtask_transitions[key] = transitions
        return transitions

    # -------------------------------------------------------------------------
    # Original (non-augcap) subtask data for "No action" filtering
    # -------------------------------------------------------------------------

    def _load_orig_subtask_index(self, ds_idx: int) -> np.ndarray:
        """Lazily load the original subtask_index column for a dataset shard.

        Reads parquet files from the shard's ``data/`` directory and extracts
        the ``subtask_index`` column as a flat numpy array.
        """
        if ds_idx in self._orig_subtask_index:
            return self._orig_subtask_index[ds_idx]

        build_args = self._dataset_build_args[ds_idx]
        if build_args is None:
            self._orig_subtask_index[ds_idx] = np.array([], dtype=np.int64)
            return self._orig_subtask_index[ds_idx]

        shard_root: str | None = build_args["root"]
        if shard_root is None:
            self._orig_subtask_index[ds_idx] = np.array([], dtype=np.int64)
            return self._orig_subtask_index[ds_idx]
        data_dir = Path(shard_root) / "data"
        parquet_files = sorted(data_dir.rglob("*.parquet"))
        dfs = [pd.read_parquet(str(f), columns=["subtask_index"]) for f in parquet_files]
        if dfs:
            arr = pd.concat(dfs, ignore_index=True)["subtask_index"].to_numpy(dtype=np.int64)
        else:
            arr = np.array([], dtype=np.int64)

        self._orig_subtask_index[ds_idx] = arr
        log.info(f"HandPoseDataset: loaded {len(arr)} original subtask indices for ds_idx={ds_idx}")
        return arr

    def _get_orig_subtask_transitions(self, ds_idx: int, ep_idx: int) -> list[tuple[int, int]]:
        """Return sorted ``(row_start, subtask_index)`` transitions from the original data.

        Reads subtask indices from the shard root's ``data/`` parquets using
        the same episode boundaries as the loaded dataset.
        """
        key = (ds_idx, ep_idx)
        cached = self._orig_subtask_transitions.get(key)
        if cached is not None:
            return cached

        orig_indices = self._load_orig_subtask_index(ds_idx)
        if len(orig_indices) == 0:
            self._orig_subtask_transitions[key] = []
            return []

        ds = self._get_dataset(ds_idx)
        ep = ds.meta.episodes[ep_idx]
        ep_start: int = ep["dataset_from_index"]
        ep_end: int = ep["dataset_to_index"]

        subtask_col = orig_indices[ep_start:ep_end]

        transitions: list[tuple[int, int]] = []
        prev_si = None
        for offset, si_val in enumerate(subtask_col):
            si = int(si_val)
            if si != prev_si:
                transitions.append((ep_start + offset, si))
                prev_si = si

        self._orig_subtask_transitions[key] = transitions
        return transitions

    def _is_no_action(self, subtask_index: int, ds_idx: int = 0) -> bool:
        """Return ``True`` if the original subtask name indicates an idle segment.

        Matches the name (normalized: ``_`` → space, stripped, lowercased)
        against three rule sets from ``hand_pose_dataset_config``:
        exact ``NO_ACTION_SKIP_LABELS``, ``NO_ACTION_SKIP_LABEL_PREFIXES``, or
        ``NO_ACTION_SKIP_LABEL_SUBSTRINGS``.
        """
        name = self._get_orig_subtask_names_for_ds(ds_idx).get(subtask_index, "")
        normalized = name.replace("_", " ").strip().lower()
        if normalized in NO_ACTION_SKIP_LABELS:
            return True
        if NO_ACTION_SKIP_LABEL_PREFIXES and normalized.startswith(NO_ACTION_SKIP_LABEL_PREFIXES):
            return True
        if any(sub in normalized for sub in NO_ACTION_SKIP_LABEL_SUBSTRINGS):
            return True
        return False

    def _skip_no_action_subtask(self, ds_idx: int, row_idx: int, ep_idx: int) -> int | None:
        """Advance ``row_idx`` past any original "No action" subtask.

        Returns:
            The (possibly advanced) ``row_idx``, or ``None`` if every
            remaining subtask in the episode is "No action".
        """
        orig_transitions = self._get_orig_subtask_transitions(ds_idx, ep_idx)
        if not orig_transitions:
            return row_idx

        row_starts = [t[0] for t in orig_transitions]
        ti = bisect_right(row_starts, row_idx) - 1
        if ti < 0:
            return row_idx

        _, orig_si = orig_transitions[ti]
        if not self._is_no_action(orig_si, ds_idx):
            return row_idx

        for next_ti in range(ti + 1, len(orig_transitions)):
            _, next_si = orig_transitions[next_ti]
            if not self._is_no_action(next_si, ds_idx):
                return orig_transitions[next_ti][0]

        return None

    # -------------------------------------------------------------------------
    # Build-time subtask-level reindexing for uniform subtask sampling
    # -------------------------------------------------------------------------

    def _transitions(self, col: np.ndarray, ep_start: int) -> list[tuple[int, int]]:
        """``(row_start, value)`` pairs at each change point (first row always emitted).

        Vectorized when ``enable_fast_init=True``, else the original Python
        ``enumerate`` loop.
        """
        if self._enable_fast_init:
            if len(col) == 0:
                return []
            mask = np.concatenate(([True], col[1:] != col[:-1]))
            idx = np.flatnonzero(mask)
            return list(zip((ep_start + idx).tolist(), col[idx].astype(np.int64).tolist()))
        transitions: list[tuple[int, int]] = []
        prev: int | None = None
        for offset, val in enumerate(col):
            v = int(val)
            if v != prev:
                transitions.append((ep_start + offset, v))
                prev = v
        return transitions

    def _rebuild_snap_indices(self, ds_idx: int, meta: Any, records_before: int) -> None:
        """Replace frame-level indices with subtask-level indices.

        When ``snap_to_subtask`` is active, the default frame-level indexing
        biases sampling toward longer subtasks.  This method replaces the
        episode records with subtask-level records so that each subtask gets
        exactly one index, yielding uniform sampling over subtasks.

        When ``skip_no_action`` is also active, subtasks whose original label
        starts with "No action" are excluded at build time.
        """
        build_args = self._dataset_build_args[ds_idx]
        if build_args is None:
            return
        shard_root: str | None = build_args["root"]
        if shard_root is None:
            return

        if self._enable_fast_init:
            subtask_col = self._cached_subtask_col[ds_idx]
        else:
            data_dir = Path(shard_root) / "data"
            parquet_files = sorted(data_dir.rglob("*.parquet"))
            dfs = [pd.read_parquet(str(f), columns=["subtask_index"]) for f in parquet_files]
            if not dfs:
                return
            subtask_col = pd.concat(dfs, ignore_index=True)["subtask_index"].to_numpy(dtype=np.int64)
        if len(subtask_col) == 0:
            return

        orig_subtask_col: np.ndarray | None = None
        if self._skip_no_action:
            orig_subtask_col = self._load_orig_subtask_index(ds_idx)

        fps_ratio = round(meta.fps / self._fps)

        ep_from = list(meta.episodes["dataset_from_index"])
        ep_to = list(meta.episodes["dataset_to_index"])

        new_records: list[tuple[int, int, int, int]] = []
        total_subtasks = 0
        total_skipped_na = 0
        total_skipped_short = 0

        for rec in self._episode_records[records_before:]:
            rec_ds_idx, sample_start, _old_valid_len, episode_id = rec
            assert rec_ds_idx == ds_idx

            ep_start: int = ep_from[episode_id]
            ep_end: int = ep_to[episode_id]

            transitions = self._transitions(subtask_col[ep_start:ep_end], ep_start)

            orig_trans: list[tuple[int, int]] | None = None
            orig_row_starts: list[int] | None = None
            if orig_subtask_col is not None and len(orig_subtask_col) > 0:
                orig_trans = self._transitions(orig_subtask_col[ep_start:ep_end], ep_start)
                orig_row_starts = [t[0] for t in orig_trans]

            subtask_starts: list[tuple[int, int]] = []
            for i, (row_start, _si) in enumerate(transitions):
                if i + 1 < len(transitions):
                    native_len = transitions[i + 1][0] - row_start
                else:
                    native_len = ep_end - row_start

                if self._get_snapped_video_frame_count(native_len, fps_ratio) == 0:
                    total_skipped_short += 1
                    continue

                if self._skip_no_action and orig_trans is not None and orig_row_starts is not None:
                    ti = bisect_right(orig_row_starts, row_start) - 1
                    if ti >= 0:
                        _, orig_si = orig_trans[ti]
                        if self._is_no_action(orig_si, ds_idx):
                            total_skipped_na += 1
                            continue

                subtask_starts.append((row_start, native_len))

            if self._max_subtasks_per_episode is not None:
                subtask_starts = subtask_starts[: self._max_subtasks_per_episode]

            self._subtask_start_indices[(ds_idx, episode_id)] = subtask_starts

            # Pre-populate transition caches so DataLoader workers inherit
            # them read-only via COW instead of rebuilding per-worker.
            self._subtask_transitions[(ds_idx, episode_id)] = transitions
            if orig_trans is not None:
                self._orig_subtask_transitions[(ds_idx, episode_id)] = orig_trans

            num_subtasks = len(subtask_starts)
            total_subtasks += num_subtasks
            if num_subtasks > 0:
                new_records.append((ds_idx, sample_start, num_subtasks, episode_id))

        self._episode_records = self._episode_records[:records_before] + new_records
        self._episode_cum_ends = self._episode_cum_ends[:records_before]
        self._num_valid_indices = self._episode_cum_ends[-1] if records_before > 0 else 0
        for rec in new_records:
            self._num_valid_indices += rec[2]
            self._episode_cum_ends.append(self._num_valid_indices)

        cap_info = ""
        if self._max_subtasks_per_episode is not None:
            cap_info = f", capped at first {self._max_subtasks_per_episode}/ep"
        log.info(
            f"HandPoseDataset: snap_to_subtask reindex — "
            f"{total_subtasks} subtask starts across {len(new_records)} episodes{cap_info} "
            f"(skipped {total_skipped_na} no-action, {total_skipped_short} too-short)"
        )

    # -------------------------------------------------------------------------
    # Episode filtering (override base class without modifying it)
    # -------------------------------------------------------------------------

    def _append_index_records(self, *, meta: Any, ds_idx: int, dataset_label: str | None = None) -> None:
        """Override to filter episodes after deterministic split selection.

        Supports two mutually exclusive modes:
        - ``max_episodes``: keep the first N episodes **globally** across all
          shards (not per-shard).  This method is called once per shard, so we
          track how many episodes have already been kept via
          ``len(self._episode_records)`` before and after the base-class call.
        - ``episode_ids``: keep only episodes whose dataset-level episode
          index appears in the given list.
        """
        records_before = len(self._episode_records)
        if self._intra_episode_val_ratio > 0.0 and self._split in {"train", "val"}:
            episode_ids = split_episode_ids(
                total_episodes=meta.total_episodes,
                seed=self._split_seed,
                val_ratio=self._split_val_ratio,
                split="train",
            )
            episode_spans, _, sample_count = build_episode_spans(
                episodes=meta.episodes,
                episode_ids=episode_ids,
                chunk_length=self._chunk_length,
            )

            # Prevent train/val leakage caused by overlapping chunk windows.
            # A gap of chunk_length sample-starts ensures no frame overlap between splits.
            non_overlap_gap = self._chunk_length
            valid_count = 0
            for episode_id, sample_start, valid_len in episode_spans:
                if valid_len <= 0:
                    continue
                num_val = int(round(valid_len * self._intra_episode_val_ratio))
                val_start_offset = max(0, valid_len - num_val)  # take tail for validation

                if self._split == "train":
                    split_start = sample_start
                    split_len = max(0, val_start_offset - non_overlap_gap)
                else:
                    split_start = sample_start + val_start_offset
                    split_len = max(0, valid_len - val_start_offset)

                if split_len <= 0:
                    continue

                self._episode_records.append((ds_idx, split_start, split_len, episode_id))
                self._num_valid_indices += split_len
                self._episode_cum_ends.append(self._num_valid_indices)
                valid_count += split_len

            class_name = self.__class__.__name__
            label = f" [{dataset_label}]" if dataset_label else ""
            log.info(
                f"{class_name}{label}: intra-episode split enabled "
                f"(ratio={self._intra_episode_val_ratio:.3f}, split={self._split}, gap={non_overlap_gap})"
            )
            if sample_count > 0:
                log.info(
                    f"{class_name}{label}: kept {valid_count} / {sample_count} "
                    f"({100 * valid_count / sample_count:.2f} %) samples"
                )
        else:
            super()._append_index_records(meta=meta, ds_idx=ds_idx, dataset_label=dataset_label)

        new_records = self._episode_records[records_before:]
        if not new_records:
            return

        if self._episode_ids is not None:
            keep_set = set(self._episode_ids)
            kept: list[tuple[int, int, int, int]] = []
            removed_frames = 0
            for rec in new_records:
                if rec[3] in keep_set:
                    kept.append(rec)
                else:
                    removed_frames += rec[2]
            self._episode_records = self._episode_records[:records_before] + kept
            self._num_valid_indices -= removed_frames
            self._episode_cum_ends = self._episode_cum_ends[:records_before]
            running = self._episode_cum_ends[-1] if records_before > 0 else 0
            for rec in kept:
                running += rec[2]
                self._episode_cum_ends.append(running)
            kept_ids = [rec[3] for rec in kept]
            log.info(
                f"HandPoseDataset: episode_ids filter — "
                f"kept {len(kept)}/{len(new_records)} episodes "
                f"({self._num_valid_indices} valid indices), "
                f"retained episode IDs: {kept_ids}"
            )
        elif self._max_episodes is not None:
            global_total = len(self._episode_records)
            remaining_budget = self._max_episodes - records_before
            if remaining_budget <= 0:
                self._episode_records = self._episode_records[:records_before]
                self._episode_cum_ends = self._episode_cum_ends[:records_before]
                removed_frames = sum(rec[2] for rec in new_records)
                self._num_valid_indices -= removed_frames
                log.info(
                    f"HandPoseDataset: max_episodes={self._max_episodes} — "
                    f"global budget exhausted, dropped all {len(new_records)} episodes "
                    f"from shard ds_idx={ds_idx}"
                )
            elif len(new_records) > remaining_budget:
                keep = records_before + remaining_budget
                removed = self._episode_records[keep:]
                self._episode_records = self._episode_records[:keep]
                self._episode_cum_ends = self._episode_cum_ends[:keep]
                removed_frames = sum(rec[2] for rec in removed)
                self._num_valid_indices -= removed_frames
                if keep > 0:
                    self._episode_cum_ends[-1] = self._num_valid_indices

                retained = self._episode_records[records_before:keep]
                retained_ids = [rec[3] for rec in retained]
                retained_frames = sum(rec[2] for rec in retained)
                log.info(
                    f"HandPoseDataset: max_episodes={self._max_episodes} — "
                    f"shard ds_idx={ds_idx}: kept {remaining_budget}/{len(new_records)} episodes, "
                    f"removed {removed_frames} frames, "
                    f"retained {retained_frames} valid indices, "
                    f"episode IDs: {retained_ids}, "
                    f"global total: {len(self._episode_records)} episodes"
                )
            else:
                retained_ids = [rec[3] for rec in new_records]
                log.info(
                    f"HandPoseDataset: max_episodes={self._max_episodes} — "
                    f"shard ds_idx={ds_idx}: kept all {len(new_records)} episodes "
                    f"(global total: {len(self._episode_records)}/{self._max_episodes}), "
                    f"episode IDs: {retained_ids}"
                )
        else:
            all_ids = [rec[3] for rec in new_records]
            log.info(
                f"HandPoseDataset: using all {len(new_records)} episodes ({self._num_valid_indices} valid indices)"
            )

        if self._snap_to_subtask:
            self._rebuild_snap_indices(ds_idx=ds_idx, meta=meta, records_before=records_before)

    # -------------------------------------------------------------------------
    # Action building
    # -------------------------------------------------------------------------

    def _build_action(self, sample: dict[str, Any]) -> torch.Tensor:
        """Build the action tensor from a LeRobot sample.

        Both modes share the same three-stage pipeline:
          1. Compute absolute SE(3) poses: camera (already world-frame),
             wrists (converted from per-frame camera space to world space
             via ``P_world = P_c2w @ P_cam``).
          2. Compute finger positions in per-frame wrist frame (valid in
             camera space since both are in the same frame at each timestep).
          3. Convert world-frame absolute poses to relative (anchored or
             frame-wise) for camera and wrists.

        Layout: ``[camera, R_wrist, R_fingers, L_wrist, L_fingers]``

        Returns:
            Action tensor of shape ``(T, raw_action_dim)``.
        """
        pose_convention: Literal["backward_anchored", "backward_framewise"] = self._pose_convention  # type: ignore[assignment]

        def _to_np(t: torch.Tensor) -> np.ndarray:
            return t.detach().cpu().numpy()

        # -- Stage 1: absolute SE(3) poses ------------------------------------
        # Camera pose (world frame): c2w transforms
        cam_pos = _to_np(sample[CAM_POSITION_KEY])  # (T+1, 3)
        cam_rot_q = _to_np(sample[CAM_ROTATION_KEY])  # (T+1, 4)
        cam_c2w = build_abs_pose_from_components(cam_pos, cam_rot_q, "quat_xyzw")  # (T+1, 4, 4)

        # Wrist poses in camera frame (right first, then left)
        right_pos_all = _to_np(sample[self._position_keys[1]])  # (T+1, 63) — hand_right_cam
        right_rot_all = _to_np(sample[self._rotation_keys[1]])  # (T+1, 84)
        left_pos_all = _to_np(sample[self._position_keys[0]])  # (T+1, 63) — hand_left_cam
        left_rot_all = _to_np(sample[self._rotation_keys[0]])  # (T+1, 84)

        right_wrist_pos, right_wrist_quat = self._extract_wrist_pose_components(right_pos_all, right_rot_all)
        left_wrist_pos, left_wrist_quat = self._extract_wrist_pose_components(left_pos_all, left_rot_all)

        # Wrist poses in camera frame, aligned to the unified cross-domain convention.
        right_wrist_cam = build_abs_pose_from_components(right_wrist_pos, right_wrist_quat, "quat_xyzw")
        left_wrist_cam = build_abs_pose_from_components(left_wrist_pos, left_wrist_quat, "quat_xyzw")

        if "embodiment_a" in self._root.lower():
            right_wrist_cam = right_wrist_cam @ WRIST_FRAME_ALIGN_EMBODIMENT_A
            left_wrist_cam = left_wrist_cam @ WRIST_FRAME_ALIGN_EMBODIMENT_A

        # Wrist poses in camera frame → world frame: P_world = P_c2w @ P_cam
        right_wrist_world = cam_c2w @ right_wrist_cam  # (T+1, 4, 4)
        left_wrist_world = cam_c2w @ left_wrist_cam  # (T+1, 4, 4)

        # -- Stage 2: finger positions in per-frame wrist frame ---------------
        # (Correct as-is: both finger positions and wrist pose are in the same
        # camera frame at each timestep, so the transform to wrist-local is valid.
        # The alignment rotation only changes the wrist-local axes; finger positions
        # in camera space are unchanged, so their wrist-local coordinates rotate
        # correspondingly to match the unified convention.)
        right_fingers = self._build_fingers_in_wrist_frame(right_pos_all, right_wrist_cam)
        left_fingers = self._build_fingers_in_wrist_frame(left_pos_all, left_wrist_cam)

        # -- Stage 3: convert world-frame absolute → relative poses -----------
        cam_rel = pose_abs_to_rel(cam_c2w, rotation_format=self._rotation_format, pose_convention=pose_convention)
        right_wrist_rel = pose_abs_to_rel(
            right_wrist_world,
            rotation_format=self._rotation_format,
            pose_convention=pose_convention,
        )
        left_wrist_rel = pose_abs_to_rel(
            left_wrist_world,
            rotation_format=self._rotation_format,
            pose_convention=pose_convention,
        )

        # -- Assemble: [camera, R_wrist, R_fingers, L_wrist, L_fingers] -------
        return torch.from_numpy(
            np.concatenate([cam_rel, right_wrist_rel, right_fingers, left_wrist_rel, left_fingers], axis=-1)
        ).float()

    # -- Shared action helpers ------------------------------------------------

    def _extract_wrist_pose_components(
        self,
        pos_data: np.ndarray,
        rot_quat: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract wrist translation and wrist quaternion from full-hand arrays."""
        wrist_pos = pos_data[:, :3]
        wrist_quat = rot_quat.reshape(pos_data.shape[0], NUM_JOINTS, QUAT_DIM_PER_JOINT)[:, WRIST_JOINT_IDX]
        return wrist_pos, wrist_quat

    def _build_fingers_in_wrist_frame(
        self,
        pos_data: np.ndarray,
        wrist_poses_abs: np.ndarray,
    ) -> np.ndarray:
        """Express selected non-wrist keypoints in the per-frame wrist coordinate frame."""
        future_pos = pos_data[1:].astype(np.float32, copy=False)
        T = future_pos.shape[0]

        selected_non_wrist = [j for j in self._position_joint_indices if j != WRIST_JOINT_IDX]
        if not selected_non_wrist:
            return np.empty((T, 0), dtype=np.float32)

        pos_3d = future_pos.reshape(T, NUM_JOINTS, 3)
        finger_pos = pos_3d[:, selected_non_wrist, :]
        finger_pos_h = np.concatenate(
            [finger_pos, np.ones((*finger_pos.shape[:-1], 1), dtype=np.float32)],
            axis=-1,
        )
        wrist_inv = np.linalg.inv(wrist_poses_abs[1:])
        finger_pos_wrist = np.einsum("tij,tnj->tni", wrist_inv, finger_pos_h)[..., :3]
        return finger_pos_wrist.reshape(T, -1)

    def _get_snapped_video_frame_count(self, subtask_native_len: int, fps_ratio: int) -> int:
        """Return tokenizer-compatible video frame count for a snapped subtask.

        The tokenizer keeps one conditioning frame and consumes the remaining
        temporal dimension in groups of four, so snapped clips must be ``1 + 4N``
        frames long.
        """
        capped_video_frames = min(subtask_native_len // fps_ratio, self._chunk_length + 1)
        if capped_video_frames < 5:
            return 0
        return 1 + 4 * ((capped_video_frames - 1) // 4)

    # -------------------------------------------------------------------------
    # Caption helpers
    # -------------------------------------------------------------------------

    def _load_task_labels(self) -> None:
        """Load task and subtask name mappings from meta parquet files.

        Subtask indices are **per-shard** (each shard's indices start at 0 and
        the name lists differ across shards), so we must load and store subtask
        names independently for every shard to avoid cross-shard caption
        mismatch.
        """
        if self._enable_fast_init:
            self._load_task_labels_fast()
            return

        tasks_path = os.path.join(self._all_shard_roots[0], "meta", "tasks.parquet")
        if os.path.exists(tasks_path):
            tasks_df = pd.read_parquet(tasks_path)
            for _, row in tasks_df.iterrows():
                self._task_names[int(row["task_index"])] = str(row.name)
            log.info(f"HandPoseDataset: loaded {len(self._task_names)} task labels")

        if self._load_subtasks:
            for shard_idx, shard_root in enumerate(self._all_shard_roots):
                shard_names: dict[int, str] = {}
                plain_path = os.path.join(shard_root, "meta", "subtasks.parquet")

                if os.path.exists(plain_path):
                    df = pd.read_parquet(plain_path)
                    for _, row in df.iterrows():
                        shard_names[int(row["subtask_index"])] = str(row.name)
                    log.info(f"HandPoseDataset: shard {shard_idx}: loaded {len(shard_names)} subtask labels")

                self._per_shard_subtask_names[shard_idx] = shard_names

        if self._skip_no_action:
            for shard_idx, shard_root in enumerate(self._all_shard_roots):
                orig_subtasks_path = os.path.join(shard_root, "meta", "subtasks.parquet")
                if os.path.exists(orig_subtasks_path):
                    shard_names = {}
                    orig_df = pd.read_parquet(orig_subtasks_path)
                    for _, row in orig_df.iterrows():
                        shard_names[int(row["subtask_index"])] = str(row.name)
                    self._per_shard_orig_subtask_names[shard_idx] = shard_names
                    log.info(
                        f"HandPoseDataset: shard {shard_idx}: loaded {len(shard_names)} "
                        f"original subtask labels (from {shard_root})"
                    )

    def _load_task_labels_fast(self) -> None:
        """Parallel version of ``_load_task_labels`` — bit-exact output."""
        tasks_path = os.path.join(self._all_shard_roots[0], "meta", "tasks.parquet")
        if os.path.exists(tasks_path):
            tasks_df = pd.read_parquet(tasks_path)
            for _, row in tasks_df.iterrows():
                self._task_names[int(row["task_index"])] = str(row.name)
            log.info(f"HandPoseDataset: loaded {len(self._task_names)} task labels")

        if not (self._load_subtasks or self._skip_no_action):
            return

        def _read(shard_root: str) -> dict[int, str] | None:
            path = os.path.join(shard_root, "meta", "subtasks.parquet")
            if not os.path.exists(path):
                return None
            df = pd.read_parquet(path)
            idx_arr = df["subtask_index"].to_numpy()
            name_arr = df.index.to_numpy()
            return dict(zip(idx_arr.astype(np.int64).tolist(), name_arr.astype(str).tolist()))

        roots = self._all_shard_roots
        all_names = _parallel_map(
            _read,
            roots,
            max_workers=max(1, min(self._fast_init_max_workers, len(roots))),
            label="HandPoseDataset: _load_task_labels",
        )
        # Match the serial version's set-membership: _per_shard_subtask_names
        # is always set (``{}`` when missing); _per_shard_orig_subtask_names
        # only when the file exists.
        if self._load_subtasks:
            for shard_idx, names in enumerate(all_names):
                self._per_shard_subtask_names[shard_idx] = names if names is not None else {}
        if self._skip_no_action:
            for shard_idx, names in enumerate(all_names):
                if names is not None:
                    self._per_shard_orig_subtask_names[shard_idx] = names

    def _register_sources(self, indices: list[int] | None = None) -> None:
        """Register shard sources + HandPose-specific ``subtask_index`` prefetch.

        ``indices`` is the subset of ``_all_shard_roots`` assigned to this
        caller (e.g. one slice per DataLoader worker under
        ``shard_across_workers=True``).  ``_ds_idx_to_shard_idx`` maps the
        local ``ds_idx`` back to the global shard index used to key
        ``_per_shard_subtask_names``, so caption lookups keep working.

        The base class owns the generic fast-init (parallel
        ``LeRobotDatasetMetadata`` prefetch + serial ``_register_source``
        append loop).  When ``enable_fast_init=True`` *and* snap/skip
        flags are on, this override additionally prefetches the
        ``subtask_index`` parquet column for each assigned shard in a
        thread pool and caches it into ``_cached_subtask_col`` /
        ``_orig_subtask_index``, so ``_rebuild_snap_indices`` and
        ``_load_orig_subtask_index`` hit the cache instead of re-scanning
        ``data/*.parquet``.
        """
        if indices is None:
            indices = list(range(len(self._all_shard_roots)))
        self._ds_idx_to_shard_idx = list(indices)
        if not indices:
            return

        # Snapshot before ``super()._register_sources`` appends to
        # ``_datasets``; each new ds_idx is ``base_ds_idx + offset``.
        base_ds_idx = len(self._datasets)
        roots = [self._all_shard_roots[i] for i in indices]

        super()._register_sources(indices)

        if not (self._enable_fast_init and (self._snap_to_subtask or self._skip_no_action)):
            return

        import pyarrow.dataset as pa_ds

        def _read_subtask_col(root: str) -> np.ndarray:
            data_dir = Path(root) / "data"
            if not data_dir.exists():
                return np.array([], dtype=np.int64)
            # Explicit ``*.parquet`` glob avoids picking up partial-write
            # residues like ``file-000.parquet<digits>``.
            parquet_files = sorted(str(f) for f in data_dir.rglob("*.parquet"))
            if not parquet_files:
                return np.array([], dtype=np.int64)
            # ``use_threads=False`` avoids oversubscription; the outer
            # pool already saturates lustre at the configured workers.
            table = pa_ds.dataset(parquet_files, format="parquet").to_table(
                columns=["subtask_index"], use_threads=False
            )
            return table.column("subtask_index").to_numpy(zero_copy_only=False).astype(np.int64, copy=False)

        workers = max(1, min(self._fast_init_max_workers, len(roots)))
        subtask_cols = _parallel_map(
            _read_subtask_col,
            roots,
            max_workers=workers,
            label="HandPoseDataset: subtask_index prefetch",
        )
        for offset, col in enumerate(subtask_cols):
            self._cached_subtask_col[base_ds_idx + offset] = col
            self._orig_subtask_index[base_ds_idx + offset] = col

    def _resolve_shard_idx(self, ds_idx: int) -> int:
        """Map a worker-local ``ds_idx`` to the global shard index."""
        if self._ds_idx_to_shard_idx is not None:
            return self._ds_idx_to_shard_idx[ds_idx]
        return ds_idx

    def _get_subtask_names_for_ds(self, ds_idx: int) -> dict[int, str]:
        """Return the subtask name mapping for the shard that owns *ds_idx*."""
        return self._per_shard_subtask_names.get(self._resolve_shard_idx(ds_idx), {})

    def _get_orig_subtask_names_for_ds(self, ds_idx: int) -> dict[int, str]:
        """Return the original subtask name mapping for the shard that owns *ds_idx*."""
        return self._per_shard_orig_subtask_names.get(self._resolve_shard_idx(ds_idx), {})

    def _load_episode_intrinsics(self, root: str) -> None:
        """Load per-episode camera intrinsics from dataset metadata."""
        from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

        meta = LeRobotDatasetMetadata(repo_id="local", root=root)
        eps = meta.episodes
        if "camera_intrinsics" not in eps.column_names:
            log.warning(
                "HandPoseDataset: 'camera_intrinsics' not found in metadata; skeleton overlay will be unavailable."
            )
            return
        for ep in eps:
            self._episode_intrinsics[int(ep["episode_index"])] = np.asarray(ep["camera_intrinsics"], dtype=np.float32)
        log.info(f"HandPoseDataset: loaded intrinsics for {len(self._episode_intrinsics)} episodes")

    def _get_chunk_caption(
        self,
        ds_idx: int,
        row_idx: int,
        ep_idx: int,
        sample: dict[str, Any],
        effective_chunk_length: int | None = None,
    ) -> str:
        """Build a caption covering all subtasks that overlap the chunk window.

        When ``_load_subtasks`` is enabled, finds every subtask whose frame
        range intersects ``[row_idx, row_idx + fps_ratio * chunk_length]`` and
        concatenates their descriptions.  Falls back to anchor-only task name
        otherwise.

        Args:
            effective_chunk_length: If provided, overrides ``self._chunk_length``
                for computing the chunk window (used by ``snap_to_subtask`` to
                restrict the caption to the snapped subtask).
        """
        task_name: str | None = None
        task_idx = sample.get("task_index")
        if task_idx is not None:
            ti = int(task_idx) if not isinstance(task_idx, int) else task_idx
            task_name = self._task_names.get(ti)

        if not self._load_subtasks:
            if task_name:
                return task_name.replace("_", " ")
            return "Human hand manipulation task"

        transitions = self._get_subtask_transitions(ds_idx, ep_idx)
        if not transitions:
            if task_name:
                return task_name.replace("_", " ")
            return "Human hand manipulation task"

        ds = self._get_dataset(ds_idx)
        fps_ratio = round(ds.meta.fps / self._fps)
        cl = effective_chunk_length if effective_chunk_length is not None else self._chunk_length
        chunk_end_row = row_idx + fps_ratio * cl

        row_starts = [t[0] for t in transitions]
        first_ti = max(0, bisect_right(row_starts, row_idx) - 1)

        subtask_parts: list[str] = []
        seen: set[int] = set()
        for ti in range(first_ti, len(transitions)):
            seg_start, si = transitions[ti]
            if seg_start >= chunk_end_row:
                break
            if si in seen:
                continue
            seen.add(si)
            name = self._get_subtask_names_for_ds(ds_idx).get(si)
            if name:
                subtask_parts.append(name.replace("_", " "))

        if subtask_parts:
            return " Then, ".join(subtask_parts)
        if task_name:
            return task_name.replace("_", " ")
        return "Human hand manipulation task"

    # -------------------------------------------------------------------------
    # Data quality filters
    # -------------------------------------------------------------------------

    def _has_unannotated_frames(self, sample: dict[str, Any]) -> bool:
        """Return True if any frame in the sampled chunk appears unannotated.

        A frame is considered unannotated when both hands' position vectors are
        effectively all-zero under ``self._unannotated_pos_l1_threshold``.
        """
        left = sample[self._position_keys[0]]  # (T+1, 63)
        right = sample[self._position_keys[1]]  # (T+1, 63)
        left_l1 = left.abs().sum(dim=-1)
        right_l1 = right.abs().sum(dim=-1)
        missing_mask = (left_l1 <= self._unannotated_pos_l1_threshold) & (
            right_l1 <= self._unannotated_pos_l1_threshold
        )
        return bool(missing_mask.any())

    def _has_nan_action(self, action: torch.Tensor) -> bool:
        """Return True if the action tensor contains any NaN or Inf values."""
        return bool(torch.isnan(action).any() or torch.isinf(action).any())

    # -------------------------------------------------------------------------
    # Dataset interface
    # -------------------------------------------------------------------------

    def _snap_to_subtask_bounds(self, ds_idx: int, row_idx: int, ep_idx: int) -> tuple[int, int | None]:
        """Snap ``row_idx`` to the start of its subtask and return its native-fps duration.

        Returns:
            (snapped_row_idx, subtask_native_len) where subtask_native_len is
            the number of native-fps rows in the subtask, or ``None`` if
            transitions are unavailable.
        """
        transitions = self._get_subtask_transitions(ds_idx, ep_idx)
        if not transitions:
            return row_idx, None
        row_starts = [t[0] for t in transitions]
        ti = bisect_right(row_starts, row_idx) - 1
        if ti < 0:
            return row_idx, None
        snapped = row_starts[ti]
        if ti + 1 < len(transitions):
            subtask_len = row_starts[ti + 1] - snapped
        else:
            ds = self._get_dataset(ds_idx)
            ep_end: int = ds.meta.episodes[ep_idx]["dataset_to_index"]
            subtask_len = ep_end - snapped
        return snapped, subtask_len

    def _choose_mode(self) -> str:
        """Resolve mode with biased sampling toward forward_dynamics in joint mode.
        Overrides the uniform sampling in BaseActionLeRobotDataset.
        """
        if self._mode == "joint":
            return random.choices(
                ("forward_dynamics", "inverse_dynamics", "policy"),
                weights=(0.8, 0.1, 0.1),
                k=1,
            )[0]
        return self._mode

    def __getitem__(self, idx: int) -> dict[str, Any]:
        mode = self._choose_mode()
        sample: dict[str, Any] | None = None
        action: torch.Tensor | None = None
        episode_idx: int = -1
        final_ds_idx: int = -1
        final_row_idx: int = -1
        effective_len: int | None = None
        last_error: Exception | None = None
        current_idx = idx
        ds_idx = -1
        row_idx = -1
        frame_offset = -1

        for _attempt in range(self._max_item_retries + 1):
            try:
                effective_len = None
                ds_idx, row_idx, ep_idx, frame_offset = self._resolve_index(current_idx)
                if self._snap_to_subtask:
                    subtask_info = self._subtask_start_indices.get((ds_idx, ep_idx))
                    if subtask_info is not None and len(subtask_info) > 0:
                        si = frame_offset
                        if si < len(subtask_info):
                            row_idx, subtask_native_len = subtask_info[si]
                            ds = self._get_dataset(ds_idx)
                            fps_ratio = round(ds.meta.fps / self._fps)
                            snapped_video_frames = self._get_snapped_video_frame_count(subtask_native_len, fps_ratio)
                            if snapped_video_frames == 0:
                                last_error = RuntimeError("Snapped subtask is too short for tokenizer; resampling.")
                                current_idx = random.randrange(len(self))
                                continue
                            effective_len = snapped_video_frames - 1
                        else:
                            last_error = RuntimeError("Invalid subtask index; resampling.")
                            current_idx = random.randrange(len(self))
                            continue
                    else:
                        last_error = RuntimeError("Invalid subtask index; resampling.")
                        current_idx = random.randrange(len(self))
                        continue
                elif self._skip_no_action:
                    skipped_row = self._skip_no_action_subtask(ds_idx, row_idx, ep_idx)
                    if skipped_row is None:
                        last_error = RuntimeError("All remaining original subtasks are 'No action'; resampling.")
                        current_idx = random.randrange(len(self))
                        continue
                    row_idx = skipped_row
                candidate = self._get_dataset(ds_idx)[row_idx]
            except RuntimeError as error:
                if self._is_video_decode_error(error):
                    last_error = error
                    self._log_bad_video_decode(
                        ds_idx=ds_idx,
                        row_idx=row_idx,
                        ep_idx=ep_idx,
                        frame_offset=frame_offset,
                        idx=idx,
                        attempt=_attempt,
                        error=error,
                    )
                    current_idx = random.randrange(len(self))
                    continue
                raise
            except (AssertionError, IndexError) as error:
                if isinstance(error, AssertionError) and "violate the tolerance" not in str(error):
                    raise
                last_error = error
                if not self._warned_tolerance_once:
                    log.warning(
                        f"HandPoseDataset: encountered timestamp-tolerance mismatch; resampling index. Failed episode: {ep_idx}, row_idx: {row_idx}"
                    )
                    self._warned_tolerance_once = True
                current_idx = random.randrange(len(self))
                continue

            if self._drop_unannotated_edge_frames and self._has_unannotated_frames(candidate):
                last_error = RuntimeError("Chunk contains all-zero hand annotation frame(s).")
                if not self._warned_unannotated_once:
                    log.warning("HandPoseDataset: detected zero-filled annotation frames; resampling index.")
                    self._warned_unannotated_once = True
                current_idx = random.randrange(len(self))
                continue

            try:
                candidate_action = self._build_action(candidate)
            except (ValueError, RuntimeError) as error:
                last_error = error
                if not self._warned_build_action_error_once:
                    log.warning(f"HandPoseDataset: _build_action failed at index {current_idx} ({error}); resampling.")
                    self._warned_build_action_error_once = True
                current_idx = random.randrange(len(self))
                continue
            if self._has_nan_action(candidate_action):
                last_error = RuntimeError("Action contains NaN/Inf values.")
                if not self._warned_nan_action_once:
                    log.warning(f"HandPoseDataset: NaN/Inf in action at index {current_idx}; resampling.")
                    self._warned_nan_action_once = True
                current_idx = random.randrange(len(self))
                continue

            sample = candidate
            action = candidate_action
            episode_idx = ep_idx
            final_ds_idx = ds_idx
            final_row_idx = row_idx
            break

        if sample is None or action is None:
            raise RuntimeError(
                "HandPoseDataset failed to sample a valid chunk after retries "
                f"(max_item_retries={self._max_item_retries})."
            ) from last_error

        ai_caption = self._get_chunk_caption(
            final_ds_idx,
            final_row_idx,
            episode_idx,
            sample,
            effective_chunk_length=effective_len,
        )
        if self._skip_video_loading:
            raw_video = None
            video = None
        else:
            # [T+1,C,H,W] float from LeRobot; _build_result converts to uint8 [C,T+1,H,W].
            raw_video = sample[self._video_key]
            video = raw_video

        if effective_len is not None and effective_len < self._chunk_length:
            if video is not None:
                video = video[: effective_len + 1]  # [T+1,C,H,W]
            action = action[:effective_len]

        extras: dict[str, Any] = {
            "__episode_id__": episode_idx,
            "__row_idx__": final_row_idx,
            "__dataset_root__": str(self._get_dataset(final_ds_idx).root),
            "__index__": idx,
        }
        if self._return_overlay_data and not self._skip_video_loading:
            extras["raw_cam_left_3d"] = sample[HAND_LEFT_POSITION_KEY]  # (T+1, 63)
            extras["raw_cam_right_3d"] = sample[HAND_RIGHT_POSITION_KEY]  # (T+1, 63)
            extras["raw_cam_left_rot"] = sample[HAND_LEFT_ROTATION_KEY]  # (T+1, 84)
            extras["raw_cam_right_rot"] = sample[HAND_RIGHT_ROTATION_KEY]  # (T+1, 84)
            extras["raw_cam_position"] = sample[CAM_POSITION_KEY]  # (T+1, 3)
            extras["raw_cam_rotation"] = sample[CAM_ROTATION_KEY]  # (T+1, 4)
            extras["orig_video_hw"] = torch.tensor([raw_video.shape[-2], raw_video.shape[-1]], dtype=torch.long)
            intrinsics = self._episode_intrinsics.get(episode_idx)
            if intrinsics is not None:
                extras["camera_intrinsics"] = torch.from_numpy(intrinsics)

        return self._build_result(mode=mode, video=video, action=action, ai_caption=ai_caption, **extras)

    @staticmethod
    def _is_video_decode_error(error: RuntimeError) -> bool:
        msg = str(error)
        common_err = "Requested next frame while there are no more frames left to decode"
        return common_err in msg

    def _log_bad_video_decode(
        self,
        *,
        ds_idx: int,
        row_idx: int,
        ep_idx: int,
        frame_offset: int,
        idx: int,
        attempt: int,
        error: RuntimeError,
    ) -> None:
        """Log decode failures with enough identifiers to locate bad videos later."""
        key = (ds_idx, ep_idx, row_idx)
        if key in self._logged_decode_failures:
            return
        self._logged_decode_failures.add(key)

        ds = self._get_dataset(ds_idx)
        dataset_root = str(getattr(ds, "root", "unknown"))
        repo_id = getattr(getattr(ds, "meta", None), "repo_id", "unknown")
        log.critical(
            "HandPoseDataset video decode failure detected. "
            f"idx={idx}, attempt={attempt}, split={self._split}, mode={self._mode}, "
            f"dataset_idx={ds_idx}, dataset_root={dataset_root}, repo_id={repo_id}, "
            f"episode_id={ep_idx}, row_idx={row_idx}, frame_offset={frame_offset}, "
            f"video_key={self._video_key}, chunk_length={self._chunk_length}, fps={self._fps}, "
            f"error={error!r}"
        )

    @property
    def action_dim(self) -> int:
        """Raw action dimensionality before padding."""
        return self._raw_action_dim
