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

"""LIBERO dataset for training from local storage, supporting multiple dataset roots."""

import random
from pathlib import Path
from typing import Literal

import torch
import torchvision.transforms.functional as F
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from torch.utils.data import Dataset

from cosmos.utils import log
from cosmos.data.vfm.action.action_normalization import (
    load_action_stats,
    normalize_action,
)
from cosmos.data.vfm.action.action_spec import (
    Gripper,
    Pos,
    Rot,
    build_action_spec,
)
from cosmos.data.vfm.action.domain_utils import get_domain_id
from cosmos.data.vfm.action.libero_pose_utils import (
    libero_action_dim,
    libero_rotation_format,
)
from cosmos.data.vfm.action.pose_utils import (
    compute_idle_frames,
    convert_rotation,
)

LIBERO_ROOTS: list[str] = [
    "/lustre/fsw/portfolios/dir/projects/dir_cosmos_base_lustre/maxzhaoshuol/dataset/libero_10_no_noops_1.0.0_lerobot_aligned",
    "/lustre/fsw/portfolios/dir/projects/dir_cosmos_base_lustre/maxzhaoshuol/dataset/libero_90_no_noops_lerobot_shuffled",
    "/lustre/fsw/portfolios/dir/projects/dir_cosmos_base_lustre/maxzhaoshuol/dataset/libero_object_no_noops_1.0.0_lerobot_aligned",
    "/lustre/fsw/portfolios/dir/projects/dir_cosmos_base_lustre/maxzhaoshuol/dataset/libero_spatial_no_noops_1.0.0_lerobot",
    "/lustre/fsw/portfolios/dir/projects/dir_cosmos_base_lustre/maxzhaoshuol/dataset/libero_goal_no_noops_1.0.0_lerobot",
]


class LIBERODataset(Dataset):
    """
    A Dataset wrapper for LeRobot LIBERO dataset(s) designed for training from local storage.

    This dataset:
    - Loads data from local storage using LeRobotDataset
    - Supports multiple dataset roots that are concatenated into one dataset
    - Supports configurable camera modes (image, wrist_image, or concat_view)
    - Filters episodes for train/val split
    - Filters frames at episode boundaries (to avoid padding issues with delta timestamps)
    - Uses task descriptions from meta/tasks.parquet for ai_caption
    """

    _NORMALIZERS_DIR = Path(__file__).parent / "normalizers"

    def __init__(
        self,
        repo_id: str | list[str] = "lerobot/libero_90",
        root: str | list[str] | None = LIBERO_ROOTS,
        image_size: int = 256,
        chunk_length: int = 16,  # must be divisible by 4
        fps: int = 10,  # IMPORTANT! LIBERO is at 20fps. If using frame_wise_relative in policy mode, we have to match the fps.
        mode: str = "policy",
        video_backend: str | None = "torchcodec",
        download_videos: bool = False,
        force_cache_sync: bool = False,
        tolerance_s: float = 1e-4,
        split: str = "train",
        val_ratio: float = 0.01,
        seed: int = 0,
        # Camera configuration
        camera_mode: str = "image",  # 'image', 'wrist_image', or 'concat_view'
        # Action configuration
        action_space: str = "frame_wise_relative",  # "absolute" or "relative" or "frame_wise_relative"
        # rotation_space
        rotation_space: Literal["9d", "6d", "3d"] = "3d",
        # Native simulator frame or shared OpenCV-style EE frame used by midtraining.
        pose_coordinate_frame: Literal["native", "opencv"] = "native",
        # domain-aware configuration
        embodiment_type: str = "libero",
        action_normalization: Literal["quantile", "quantile_rot", "meanstd", "minmax"] | None = None,
        action_stats_path: str | None = None,
        skip_video_loading: bool = False,
    ):
        super().__init__()
        self._embodiment_type = embodiment_type
        self.domain_id = get_domain_id(embodiment_type)
        self.image_size = image_size
        self.chunk_length = chunk_length
        assert self.chunk_length % 4 == 0, "chunk_length must be divisible by 4"
        self.fps = fps
        self.mode = mode
        self.split = split.lower().strip()
        self.val_ratio = val_ratio
        self.seed = seed
        self.camera_mode = camera_mode.lower().strip()
        self.action_space = action_space
        self.action_normalization = action_normalization
        self.rotation_space = rotation_space.lower().strip()
        self.pose_coordinate_frame = pose_coordinate_frame
        self._pose_convention = self.action_space
        self._rotation_format = libero_rotation_format(self.rotation_space)
        # When True, skip video decoding entirely: drop image keys from
        # delta_timestamps so LeRobot never touches the mp4, and return
        # ``video=None`` in __getitem__. Must be set at construction time
        # because LeRobotDataset is eagerly built in __init__.
        self._skip_video_loading = bool(skip_video_loading)

        # Load action normalization stats. ``action_min`` / ``action_range`` are
        # retained for older LIBERO eval code that knows how to invert a
        # range-style [-1, 1] normalization.
        self._norm_stats: dict[str, torch.Tensor] | None = None
        self.action_min: torch.Tensor | None = None
        self.action_max: torch.Tensor | None = None
        self.action_range: torch.Tensor | None = None
        if self.action_normalization is not None:
            stats_path = self._resolve_action_stats_path(action_stats_path)
            stats_key = "global_raw" if self.action_normalization == "quantile_rot" else "global"
            raw_stats = load_action_stats(str(stats_path), stats_key=stats_key)
            self._norm_stats = {}
            for key, value in raw_stats.items():
                self._norm_stats[key] = torch.from_numpy(value).float()  # [D]
            self._set_range_denormalization_stats()
            log.info(
                f"Loaded LIBERO action stats from {stats_path} with action_normalization={self.action_normalization}"
            )

        # Validate camera mode
        if self.camera_mode not in {"image", "wrist_image", "concat_view"}:
            raise ValueError(f"Unsupported camera_mode={camera_mode!r}. Use 'image', 'wrist_image', or 'concat_view'.")

        # Validate split
        if self.split not in {"train", "val", "valid", "validation", "eval", "test", "full"}:
            raise ValueError(f"Unsupported {split=}. Use train/val/full.")

        # Build delta timestamps based on camera mode
        dt = 1.0 / self.fps

        if self.fps != 20:
            log.warning(
                f"LIBERO is at 20fps. If using frame_wise_relative for policy mode training, we have to match the fps. fps={self.fps}"
            )

        # Determine which image keys to use
        if self.camera_mode == "image":
            self.image_keys = ["observation.images.image"]
        elif self.camera_mode == "wrist_image":
            self.image_keys = ["observation.images.wrist_image"]
        else:  # concat_view
            self.image_keys = ["observation.images.image", "observation.images.wrist_image"]

        # Build delta_timestamps for all keys (same convention as PushT: 0 to chunk_length)
        self.delta_timestamps: dict[str, list[float]] = {}
        if not self._skip_video_loading:
            for key in self.image_keys:
                self.delta_timestamps[key] = [i * dt for i in range(0, chunk_length + 1)]
        self.delta_timestamps["observation.state"] = [i * dt for i in range(0, chunk_length + 1)]
        self.delta_timestamps["action"] = [i * dt for i in range(0, chunk_length + 1)]

        # Normalize repo_id and root to lists
        repo_id_list: list[str] = [repo_id] if isinstance(repo_id, str) else list(repo_id)
        root_list: list[str | None]
        if root is None:
            root_list = [None for _ in repo_id_list]
        elif isinstance(root, str):
            root_list = [root]
        else:
            root_list = [r for r in root]

        if len(repo_id_list) != len(root_list):
            raise ValueError(
                f"Length mismatch: repo_id has {len(repo_id_list)} items, root has {len(root_list)} items."
            )

        # Load all datasets
        self.datasets: list[LeRobotDataset] = []
        self.tasks_dfs: list = []  # Store tasks DataFrames for each dataset
        for rid, r in zip(repo_id_list, root_list):
            dataset = LeRobotDataset(
                repo_id=rid,
                root=r,
                delta_timestamps=self.delta_timestamps,  # type: ignore
                tolerance_s=tolerance_s,
                force_cache_sync=force_cache_sync,
                download_videos=download_videos,
                video_backend=video_backend,
                episodes=None,  # Load full dataset, filter later
            )
            self.datasets.append(dataset)
            self.tasks_dfs.append(dataset.meta.tasks)

        # Build index mapping: list of (dataset_idx, local_idx) for valid frames
        self.index_map: list[tuple[int, int, int]] = []  # (dataset_idx, local_idx, episode_idx)
        self._episode_boundaries: list[dict[int, tuple[int, int]]] = []
        self._episode_splits: list[tuple[set[int], set[int]]] = []

        total_episodes = 0
        total_frames = 0
        for ds_idx, dataset in enumerate(self.datasets):
            # Compute episode splits for this dataset
            train_eps, val_eps = self._compute_episode_splits_for_dataset(dataset)
            self._episode_splits.append((train_eps, val_eps))

            # Get episodes for current split
            split_episodes = self._get_split_episodes_for_dataset(ds_idx)

            # Build episode boundaries
            boundaries = self._build_episode_boundaries_for_dataset(dataset)
            self._episode_boundaries.append(boundaries)

            # Filter indices
            indices = self._filter_indices_for_dataset(ds_idx, dataset, split_episodes, boundaries)
            self.index_map.extend(indices)

            total_episodes += dataset.num_episodes
            total_frames += len(dataset)

        log.info(
            f"Loaded LIBERO dataset with {len(repo_id_list)} source(s) split={self.split!r} "
            f"camera_mode={self.camera_mode!r} "
            f"total_episodes={total_episodes} "
            f"total_frames={total_frames} "
            f"valid_indices={len(self.index_map)}"
        )

    def _compute_episode_splits_for_dataset(self, dataset: LeRobotDataset) -> tuple[set[int], set[int]]:
        """Compute train/val episode splits deterministically for a single dataset."""
        total_episodes = int(dataset.meta.total_episodes)

        if not (0.0 < self.val_ratio < 1.0):
            raise ValueError(f"{self.val_ratio=} must be in (0, 1).")

        n_val = max(1, int(round(total_episodes * self.val_ratio)))
        # val_eps = set(range(n_val))
        # train_eps = set(range(n_val, total_episodes))

        # Yihuai: Randomly select validation episodes instead of the first n_val episodes (otherwise task will be repeated)
        rng = random.Random(self.seed)  # To ensure validation episodes are the same on all ranks
        val_eps = set(rng.sample(range(total_episodes), n_val))
        train_eps = set(range(total_episodes)) - val_eps

        log.info(f"train_eps={train_eps}, val_eps={val_eps}")

        return train_eps, val_eps

    def _get_split_episodes_for_dataset(self, ds_idx: int) -> set[int]:
        """Get the episode set for the current split for a specific dataset."""
        train_eps, val_eps = self._episode_splits[ds_idx]
        if self.split in {"val", "valid", "validation", "eval", "test"}:
            return val_eps
        elif self.split == "train":
            return train_eps
        else:  # full
            return train_eps | val_eps

    def _build_episode_boundaries_for_dataset(self, dataset: LeRobotDataset) -> dict[int, tuple[int, int]]:
        """Build a dict of episode_index -> (start_frame, end_frame) for a single dataset."""
        boundaries: dict[int, tuple[int, int]] = {}
        for ep in dataset.meta.episodes:
            ep_idx = int(ep["episode_index"])  # type: ignore[index]
            start = int(ep["dataset_from_index"])  # type: ignore[index]
            end = int(ep["dataset_to_index"])  # type: ignore[index]
            boundaries[ep_idx] = (start, end)
        return boundaries

    def _filter_indices_for_dataset(
        self,
        ds_idx: int,
        dataset: LeRobotDataset,
        split_episodes: set[int],
        boundaries: dict[int, tuple[int, int]],
    ) -> list[tuple[int, int, int]]:
        """Filter valid indices for a single dataset, returning (dataset_idx, local_idx, episode_idx)."""
        index_map: list[tuple[int, int, int]] = []
        all_meta = list(dataset.meta.episodes)

        for ep_idx in split_episodes:
            if ep_idx >= len(all_meta):
                continue
            ep = all_meta[ep_idx]

            ep_start = int(ep["dataset_from_index"])  # type: ignore[index]
            ep_end = int(ep["dataset_to_index"])  # type: ignore[index]

            # Valid range: [start, end - chunk_length - 1] inclusive
            # We drop chunk_length frames at end to ensure we can query up to delta=chunk_length.
            start = ep_start
            end = ep_end - self.chunk_length - 1

            if end >= start:
                for local_idx in range(start, end + 1):
                    index_map.append((ds_idx, local_idx, ep_idx))

        return index_map

    def __len__(self) -> int:
        return len(self.index_map)

    def _get_task_description(self, ds_idx: int, item: dict) -> str:
        """Get task description for the current item from meta/tasks.parquet.

        The tasks.parquet has task descriptions as the DataFrame index (row labels)
        and task_index as an integer column. We look up by task_index and return
        the corresponding index name (the actual task description string).
        """
        task_idx = item.get("task_index")
        if task_idx is not None:
            if isinstance(task_idx, torch.Tensor):
                task_idx = task_idx.item()
            task_idx = int(task_idx)
            tasks_df = self.tasks_dfs[ds_idx]
            if task_idx in tasks_df["task_index"].values:
                row = tasks_df[tasks_df["task_index"] == task_idx].iloc[0]
                # The task description is the index name (row label), not a column value
                return str(row.name)
        raise ValueError(f"Task index {task_idx} not found in tasks.parquet for dataset {ds_idx}")

    def _compute_anchored_actions(
        self,
        state_raw: torch.Tensor,
        action_raw: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute anchored relative actions (batched).

        Converts frame-wise relative actions to anchored relative actions where each
        action[t] represents the target pose (after applying action[t] to state[t])
        expressed in state 0's local coordinate frame.

        Mathematical formulation:
        1. Compute target in world frame (LIBERO convention):
           - p_{t+1} = p_t + delta_p[t]  (position addition in world frame)
           - R_{t+1} = R_delta[t] @ R_t  (rotation composition, delta first)
        2. Compute anchored (left-multiply by T_0^{-1}):
           - anchored_pos[t] = R_0^T @ (p_{t+1} - p_0)
           - anchored_rot[t] = R_0^T @ R_{t+1}

        Args:
            state_raw: State tensor of shape (T+1, 8): [x, y, z, ax, ay, az, grip1, grip2]
                where (ax, ay, az) is axis-angle rotation.
            action_raw: Action tensor of shape (T+1, 7): [dx, dy, dz, dax, day, daz, grip]
                where (dax, day, daz) is axis-angle rotation delta.

        Returns:
            anchored_translation: (T, 3) - position in state_0's local frame
            anchored_rotation_9d: (T, 9) - rotation relative to state_0 as flattened 3x3 matrix
            gripper: (T, 1) - original gripper commands (unchanged)
        """
        # Extract positions and rotations from states
        p_states = state_raw[:, :3]  # [T+1,3]
        rotvec_states = state_raw[:, 3:6]  # [T+1,3] - axis-angle

        # Extract deltas from actions (use first T actions)
        delta_p = action_raw[:-1, :3]  # [T,3]
        delta_rotvec = action_raw[:-1, 3:6]  # [T,3] - axis-angle delta
        gripper = action_raw[:-1, 6:7]  # [T,1]

        # Convert all axis-angle to rotation matrices (batched)
        R_states = convert_rotation(rotvec_states, input_format="axisangle", output_format="matrix")  # [T+1,3,3]
        R_deltas = convert_rotation(delta_rotvec, input_format="axisangle", output_format="matrix")  # [T,3,3]

        # Initial pose (state 0)
        p_0 = p_states[0]  # [3]
        R_0 = R_states[0]  # [3,3]
        R_0_T = R_0.T  # [3,3] - transpose for inverse rotation

        # Current states for t = 0..T-1
        p_t = p_states[:-1]  # [T,3]
        R_t = R_states[:-1]  # [T,3,3]

        # Step 1: Compute target poses in world frame (LIBERO convention)
        # p_target = p_t + delta_p
        p_target = p_t + delta_p  # [T,3]

        # R_target = R_delta @ R_t (batched matrix multiply)
        R_target = torch.bmm(R_deltas, R_t)  # [T,3,3]

        # Step 2: Compute anchored (in state_0's local frame)
        # anchored_p = R_0^T @ (p_target - p_0)
        displacement = p_target - p_0  # [T,3]
        anchored_p = (R_0_T @ displacement.T).T  # [T,3]

        # anchored_R = R_0^T @ R_target (batched)
        R_0_T_expanded = R_0_T.unsqueeze(0).expand(R_target.shape[0], -1, -1)  # [T,3,3]
        anchored_R = torch.bmm(R_0_T_expanded, R_target)  # [T,3,3]

        return anchored_p, anchored_R, gripper

    def _convert_rotation_to_repr(self, rotation_matrix: torch.Tensor) -> torch.Tensor:
        """Convert rotation matrix to the desired representation.

        Args:
            rotation_matrix: Rotation matrices of shape (T, 3, 3).

        Returns:
            Rotation in the configured ``rotation_space`` format.
        """
        return convert_rotation(rotation_matrix, "matrix", libero_rotation_format(self.rotation_space))

    def _normalizer_filename(self) -> str:
        rotation_suffix = {
            "3d": "3d",
            "6d": "rot6d",
            "9d": "rot9d",
        }.get(self.rotation_space)
        if rotation_suffix is None:
            raise ValueError(f"Unsupported rotation_space={self.rotation_space!r}.")
        action_space = self.action_space.replace("-", "_")
        return f"{self._embodiment_type}_{action_space}_{rotation_suffix}.json"

    def _resolve_action_stats_path(self, action_stats_path: str | None) -> Path:
        if action_stats_path is None:
            stats_path = self._NORMALIZERS_DIR / self._normalizer_filename()
            if stats_path.exists():
                return stats_path
            raise FileNotFoundError(
                f"Could not find bundled LIBERO action stats at {stats_path}. "
                "Pass action_stats_path explicitly or regenerate stats with compute_action_stats.py."
            )

        stats_path = Path(action_stats_path)
        if stats_path.is_absolute():
            if stats_path.exists():
                return stats_path
            raise FileNotFoundError(f"Could not find action_stats_path={action_stats_path!r}.")

        module_dir = Path(__file__).resolve().parent
        candidates: list[Path] = []
        for parent in module_dir.parents:
            candidates.append(parent / stats_path)
        candidates.append(self._NORMALIZERS_DIR / stats_path.name)
        candidates.append(module_dir / stats_path.name)
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(
            f"Could not resolve action_stats_path={action_stats_path!r}; tried: {[str(c) for c in candidates]}"
        )

    def _set_range_denormalization_stats(self) -> None:
        if self._norm_stats is None:
            return

        if self.action_normalization == "minmax":
            lo_key, hi_key = "min", "max"
        elif self.action_normalization in ("quantile", "quantile_rot"):
            lo_key, hi_key = "q01", "q99"
        else:
            return

        if lo_key not in self._norm_stats or hi_key not in self._norm_stats:
            raise ValueError(
                f"Action stats for {self.action_normalization!r} normalization require "
                f"{lo_key!r} and {hi_key!r} entries."
            )
        self.action_min = self._norm_stats[lo_key]  # [D]
        self.action_max = self._norm_stats[hi_key]  # [D]
        action_range = self.action_max - self.action_min  # [D]
        self.action_range = torch.clamp(action_range, min=1e-6)  # [D]

    def __getitem__(self, idx: int, _retry_count: int = 0) -> dict[str, torch.Tensor | str]:
        """Get a single item from the dataset."""
        max_retries = 10
        ds_idx, local_idx, ep_idx = self.index_map[idx]
        dataset = self.datasets[ds_idx]
        try:
            item = dataset[local_idx]
        except Exception as e:
            log.warning(
                f"Error loading item (retry {_retry_count}/{max_retries}): idx={idx}, ds_idx={ds_idx}, "
                f"local_idx={local_idx}, ep_idx={ep_idx}, repo_id={dataset.meta.repo_id}, error={e}"
            )
            if _retry_count >= max_retries:
                raise RuntimeError(f"Failed to load data after {max_retries} retries") from e
            new_idx = random.randint(0, len(self) - 1)
            return self.__getitem__(new_idx, _retry_count + 1)

        if self.mode == "joint":
            mode = random.choice(["forward_dynamics", "inverse_dynamics", "policy", "image2video"])
        else:
            mode = self.mode

        # Get task description for ai_caption
        task_description = self._get_task_description(ds_idx, item)

        # Process video based on camera mode (skipped entirely when
        # skip_video_loading=True; image keys are also absent from
        # delta_timestamps so LeRobot never decoded them).
        video: torch.Tensor | None
        if self._skip_video_loading:
            video = None
        else:
            if self.camera_mode == "concat_view":
                # Load both cameras and concatenate horizontally
                video_1: torch.Tensor = item["observation.images.image"]
                video_2: torch.Tensor = item["observation.images.wrist_image"]

                # Resize each if needed
                if video_1.shape[-1] != self.image_size or video_1.shape[-2] != self.image_size:
                    video_1 = F.resize(video_1, [self.image_size, self.image_size])
                if video_2.shape[-1] != self.image_size or video_2.shape[-2] != self.image_size:
                    video_2 = F.resize(video_2, [self.image_size, self.image_size])

                # Concatenate along width dimension (last dim for TCHW)
                video_tchw = torch.cat([video_1, video_2], dim=-1)  # (T, C, H, W*2)
            else:
                # Single camera mode
                image_key = self.image_keys[0]
                video_tchw = item[image_key]

                # Resize if needed
                if video_tchw.shape[-1] != self.image_size or video_tchw.shape[-2] != self.image_size:
                    video_tchw = F.resize(video_tchw, [self.image_size, self.image_size])

            # Convert to uint8 and transpose to (C, T, H, W)
            video = (video_tchw * 255).clamp(0, 255).to(torch.uint8).permute(1, 0, 2, 3)

        # Action (raw): LIBERO actions are 7D (6 DoF + gripper)
        action_raw: torch.Tensor = item["action"]
        # State (raw): LIBERO state is 8D (6 DoF + 2 gripper states)
        state_raw: torch.Tensor = item["observation.state"]

        # Action: (T+1, D) -> (T, D)
        # Take all but last action
        # LIBERO action format: [x, y, z, ax, ay, az, gripper] (7D) where (ax,ay,az) is axis-angle

        if self.action_space == "relative":
            # Compute anchored relative actions
            # Returns: translation (T, 3), rotation_matrix (T, 3, 3), gripper (T, 1)
            translation, rotation_matrix, gripper = self._compute_anchored_actions(state_raw, action_raw.clone())
        elif self.action_space == "frame_wise_relative":
            action = action_raw[:-1].clone()  # [T,7]
            translation = action[:, :3]  # [T,3]
            rotation_rotvec = action[:, 3:6]  # [T,3]
            gripper = action[:, 6:]  # [T,1]
            rotation_matrix = convert_rotation(
                rotation_rotvec, input_format="axisangle", output_format="matrix"
            )  # [T,3,3]
        else:
            raise ValueError(f"Unsupported action space: {self.action_space}")

        rotation = self._convert_rotation_to_repr(rotation_matrix)  # [T,rot_dim]
        action = torch.cat([translation, rotation, gripper], dim=-1)  # [T,action_dim]

        # Compute idle_frames from the raw (un-normalized) action, only when the
        # action layout has correct per-frame idle semantics (frame_wise_relative
        # ⇔ backward_framewise). The other action_spaces ("relative",
        # "absolute") encode per-frame motion differently and would not give
        # meaningful idle counts under the same threshold check.
        idle_frames: torch.Tensor | None = None
        if self.action_space == "frame_wise_relative":
            try:
                spec = build_action_spec(Pos(), Rot(libero_rotation_format(self.rotation_space)), Gripper())
                n = compute_idle_frames(action, spec)
                idle_frames = torch.tensor(n, dtype=torch.long)
            except (ValueError, TypeError):
                idle_frames = None

        if self.action_normalization is not None and self._norm_stats is not None and self.action_min is not None:
            if action.shape[-1] != self.action_min.shape[0]:
                raise ValueError(
                    f"Action dimension {action.shape[-1]} does not match stats dimension "
                    f"{self.action_min.shape[0]}. Recompute stats for the current "
                    f"rotation_space={self.rotation_space!r} and action_space={self.action_space!r}."
                )
            method = "quantile" if self.action_normalization == "quantile_rot" else self.action_normalization
            action = normalize_action(action, method, self._norm_stats)  # [T,D]

        # Index
        key = torch.tensor([local_idx], dtype=torch.long)

        if self.camera_mode == "image":
            viewpoint = "third_person_view"
        elif self.camera_mode == "wrist_image":
            viewpoint = "wrist_view"
        else:
            viewpoint = "concat_view"

        result: dict[str, torch.Tensor | str] = {
            "source_repo_id": dataset.meta.repo_id,
            "video": video,
            "action": action,
            "action_raw": action_raw,
            "conditioning_fps": torch.tensor(self.fps, dtype=torch.long),
            "prompt": task_description,
            "ai_caption": task_description,
            "mode": mode,
            "state": state_raw,
            "action_space": self.action_space,
            "rotation_space": self.rotation_space,
            "pose_coordinate_frame": self.pose_coordinate_frame,
            "__key__": key,
            "domain_id": torch.tensor(self.domain_id, dtype=torch.long),
            "viewpoint": viewpoint,
        }
        if idle_frames is not None:
            result["idle_frames"] = idle_frames

        if self.camera_mode == "concat_view" and not self._skip_video_loading:
            result["additional_view_description"] = (
                "The left half shows the third-person view; the right half shows the wrist-mounted camera."
            )

        return result

    @property
    def action_dim(self) -> int:
        return libero_action_dim(self.rotation_space)
