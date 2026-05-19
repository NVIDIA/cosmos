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

import copy
import json
import multiprocessing
import os
import re
import sys
from typing import Any, Literal, Optional, Union, cast

import numpy as np
import numpy.typing as npt
import torch
import tqdm
import zarr
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import Dataset

from cosmos.utils import log
from cosmos.data.vfm.action.umi.data_classes import (
    DataMeta,
    SourceDataMeta,
    construct_data_meta,
    construct_source_data_meta,
)
from cosmos.data.vfm.action.umi.data_utils import aggregate_batch
from cosmos.data.vfm.action.umi.imagecodecs_numcodecs import register_codecs
from cosmos.data.vfm.action.umi.normalizer import FixedNormalizer
from cosmos.data.vfm.action.umi.transforms import BaseTransforms

_ACTION_DIR = os.path.dirname(os.path.abspath(__file__))
_BUNDLED_NORMALIZERS_DIR = os.path.join(_ACTION_DIR, "normalizers")


def _resolve_normalizer_path(normalizer_dir: str, dataset_name: str) -> str | None:
    """Resolve normalizer JSON path from bundled files shipped with the repo."""
    if normalizer_dir.endswith(".json"):
        filename = os.path.basename(normalizer_dir)
    else:
        filename = f"{dataset_name}_normalizer.json"
    path = os.path.join(_BUNDLED_NORMALIZERS_DIR, filename)
    return path if os.path.exists(path) else None


register_codecs()
import random

import hydra

from cosmos.data.vfm.action.domain_utils import get_domain_id
from cosmos.data.vfm.action.pose_utils import build_abs_pose_from_components, pose_abs_to_rel


class BaseDataset(Dataset[dict[str, torch.Tensor]]):
    """
    Base class for all datasets.
    """

    def __init__(
        self,
        root: str,
        # The folder that contains all the zarr stores
        # data path should be: root/name/episode_data.zarr or root/name.zarr
        name: str,
        robot_num: int,
        # compressed_dir: str,  # The folder that contains the lz4 compressed data (compressed_dir/name.lz4)
        # If dataset is not found in root, the program will extract the data from compressed_dir
        include_episode_num: int,
        include_episode_indices: list[int],
        used_episode_ratio: float,
        index_pool_size_per_episode: int,
        history_padding_length: int,
        future_padding_length: int,
        seed: int,
        source_data_meta: Union[dict[str, dict[str, Any]], DictConfig],
        output_data_meta: Union[dict[str, dict[str, Any]], DictConfig],
        starting_percentile_max: float,
        starting_percentile_min: float,
        apply_image_augmentation_in_cpu: bool,
        use_relative_pose: bool,
        relative_pose_mode: Literal["frame_wise", "anchored"],
        use_relative_gripper_width: bool,
        normalizer_sample_num: int,
        normalizer_dir: str,
        repeat_dataset_num: int,
        mode: str,
        image_size: int,
        use_grasp_state_repr: bool,
        eef_z_offset: float,
        use_eef_pose: bool,
        **unused_kwargs,
    ):
        log.info(f"BaseDataset unused_kwargs: {unused_kwargs}")

        if not use_grasp_state_repr:
            output_data_meta.pop("grasp_states")

        if "eef_poses" not in output_data_meta:
            raise ValueError("eef_poses is not in output_data_meta")

        self.use_eef_pose: bool = use_eef_pose

        if name.endswith(".zarr"):
            name = name.replace(".zarr", "")

        if "/" in name:
            # HACK: For re-organized UMI datasets
            assert len(name.split("/")) == 2, f"name {name} should be in the format of 'project_name/dataset_name'"
            data_project_name = name.split("/")[0]
            name = name.split("/")[1]
            root = os.path.join(root, data_project_name)
            # compressed_dir = os.path.join(compressed_dir, data_project_name)
            assert normalizer_dir == "" or normalizer_dir.endswith(".json"), (
                f"normalizer_dir {normalizer_dir} should be a json file or empty"
            )

        # os.makedirs(root, exist_ok=True)

        assert isinstance(robot_num, int) and robot_num >= 1, (
            f"robot_num must be an integer greater than 0, but got {robot_num}."
        )
        self.robot_num: int = robot_num

        # if not torch.cuda.is_available() or torch.cuda.current_device() == 0:
        #     if os.path.exists(os.path.join(root, name, "episode_data.zarr")):
        #         zarr_path = os.path.join(root, name, "episode_data.zarr")
        #     elif os.path.exists(os.path.join(root, name + ".zarr")):
        #         zarr_path = os.path.join(root, name + ".zarr")
        #     elif os.path.exists(os.path.join(root, name)):
        #         zarr_path = os.path.join(root, name)
        #     else:
        #         print(f"Dataset {name} not found in {root}")
        #         print(
        #             f"Checked paths: {os.path.join(root, name, 'episode_data.zarr')}, {os.path.join(root, name + '.zarr')}, {os.path.join(root, name)}"
        #         )

        #         lz4_path = os.path.join(compressed_dir, name + ".tar.lz4")
        #         lz4_zarr_path = os.path.join(compressed_dir, name + ".zarr.tar.lz4")
        #         if os.path.exists(lz4_path):
        #             print(f"Extracting dataset {name} from {compressed_dir} to {root}")
        #             subprocess.run(
        #                 [f"lz4 -d -c {lz4_path} | tar xf - -C {root}"],
        #                 cwd=root,
        #                 shell=True,
        #                 check=True,
        #             )
        #         elif os.path.exists(lz4_zarr_path):
        #             print(f"Extracting dataset {name} from {compressed_dir} to {root}")
        #             subprocess.run(f"mkdir -p {root}/{name}.zarr", shell=True)
        #             subprocess.run(
        #                 [f"lz4 -d -c {lz4_zarr_path} | tar xf - -C {root}/{name}.zarr --strip-components=1"],
        #                 cwd=root,
        #                 shell=True,
        #                 check=True,
        #             )
        #         else:
        #             # For raw UMI datasets
        #             zip_path = os.path.join(compressed_dir, name + ".zarr.zip")
        #             if os.path.exists(zip_path):
        #                 print(f"Extracting dataset {name} from {compressed_dir} to {root}")
        #                 subprocess.run(
        #                     [f"unzip {zip_path} -d {root}"],
        #                     cwd=root,
        #                     shell=True,
        #                     check=True,
        #                 )
        #                 print(f"Dataset {name} extracted to {root}")
        #             else:
        #                 raise FileNotFoundError(f"Dataset {name} not found in {root} or {compressed_dir}")
        # if dist.is_available() and dist.is_initialized():
        #     dist.barrier()
        if os.path.exists(os.path.join(root, name, "episode_data.zarr")):
            zarr_path = os.path.join(root, name, "episode_data.zarr")
        elif os.path.exists(os.path.join(root, name + ".zarr")):
            zarr_path = os.path.join(root, name + ".zarr")
        elif os.path.exists(os.path.join(root, name)):
            zarr_path = os.path.join(root, name)
        else:
            raise FileNotFoundError(f"Dataset {name} not found in {root}")

        self.zarr_path: str = zarr_path
        self.name: str = name
        log.info(f"Dataset name: {self.name}")

        self.include_episode_num: int = include_episode_num
        self.include_episode_indices: list[int] = include_episode_indices
        self.used_episode_ratio: float = used_episode_ratio
        self.index_pool_size_per_episode: int = index_pool_size_per_episode
        self.history_padding_length: int = history_padding_length
        self.future_padding_length: int = future_padding_length
        self.seed: int = seed
        self.rng: np.random.Generator = np.random.default_rng(seed)
        self.starting_percentile_max: float = starting_percentile_max
        self.starting_percentile_min: float = starting_percentile_min
        self.apply_image_augmentation_in_cpu: bool = apply_image_augmentation_in_cpu
        self.use_relative_pose: bool = use_relative_pose
        assert relative_pose_mode in ["frame_wise", "anchored"], (
            f"relative_pose_mode must be one of frame_wise or anchored, but got {relative_pose_mode}"
        )
        self.relative_pose_mode: Literal["frame_wise", "anchored"] = relative_pose_mode
        self.use_relative_gripper_width: bool = use_relative_gripper_width
        self.mode: str = mode
        self.image_size: int = image_size
        self.domain_id: int = get_domain_id("umi")

        if os.path.exists(f"{self.zarr_path}/../sim_config.yaml"):
            log.info(f"sim_config.yaml found in {self.zarr_path}/../")
            sim_config_dict = OmegaConf.load(f"{self.zarr_path}/../sim_config.yaml")
            self.sim_config_str = OmegaConf.to_yaml(sim_config_dict, resolve=False)
        else:
            self.sim_config_str = ""

        log.info(f"{self.zarr_path=}")
        zarr_store = zarr.open(self.zarr_path, mode="r")
        assert isinstance(zarr_store, zarr.Group), f"zarr store {self.zarr_path} is not a group."
        self.zarr_store: zarr.Group = zarr_store

        assert len(source_data_meta) > 0, "source_data_meta is empty."
        self.source_data_meta: dict[str, SourceDataMeta] = construct_source_data_meta(source_data_meta)

        assert len(output_data_meta) > 0, "output_data_meta is empty."
        self.output_data_meta: dict[str, DataMeta] = construct_data_meta(output_data_meta)

        self.max_history_length: int = max(
            0,
            -min(entry_meta.include_indices[0] for entry_meta in self.source_data_meta.values()),
        )
        self.max_future_length: int = max(
            0,
            max(entry_meta.include_indices[-1] for entry_meta in self.source_data_meta.values()),
        )
        if self.history_padding_length > self.max_history_length:
            raise ValueError(
                f"history_padding_length {self.history_padding_length} is larger than max_history_length {self.max_history_length}. This may cause ambiguity in the data."
            )

        self.normalizer: Optional[FixedNormalizer] = None
        self.normalizer_dir: str = normalizer_dir
        if normalizer_dir != "":
            normalizer_path = _resolve_normalizer_path(normalizer_dir, self.name)
            if normalizer_path is not None:
                log.info(f"Loading normalizer from {normalizer_path}.")
                self.normalizer = FixedNormalizer(self.output_data_meta)
                with open(normalizer_path) as f:
                    self.normalizer.from_dict(json.load(f))
                self.normalizer.to(torch.device("cpu"))
                log.info(f"Normalizer loaded from {normalizer_path}.")
            else:
                raise FileNotFoundError(f"No normalizer found for normalizer_dir={normalizer_dir}.")
        else:
            log.info(f"normalizer_dir is empty.")
        self.normalizer_sample_num: int = normalizer_sample_num

        self.store_episode_num: int
        self.used_episode_indices: list[int]
        self.used_episode_num: int

        self.index_pool: list[tuple[int, int]] = []
        """
        index_pool has self.store_episode_num * self.used_episode_ratio * self.index_pool_size_per_episode items.
        Each item contains a tuple of (episode_idx, index), where index means the 0 index of this trajectory in an episode.
        """

        self.episode_frame_nums: dict[int, int] = {}
        self.episode_valid_indices_min: dict[int, int] = {}
        self.episode_valid_indices_max: dict[int, int] = {}  # Exclusive

        self.transforms: BaseTransforms = BaseTransforms(self.output_data_meta, self.apply_image_augmentation_in_cpu)

        self.repeat_dataset_num: float = repeat_dataset_num
        self.use_grasp_state_repr: bool = use_grasp_state_repr
        self.eef_z_offset: float = eef_z_offset

    def _check_data_validity(self):
        raise NotImplementedError("This method should be implemented in subclasses.")

    def _get_single_traj_data(self, episode_idx: int, traj_idx: int, output_entry_names: list[str] | None = None):
        raise NotImplementedError("This method should be implemented in subclasses.")

    def _create_index_pool(self):
        self.index_pool = []
        for episode_idx in self.used_episode_indices:
            valid_idx_min = self.episode_valid_indices_min[episode_idx]
            valid_idx_max = self.episode_valid_indices_max[episode_idx]
            # valid_idx_min <= sample_idx < valid_idx_max

            zero_idx_max = valid_idx_min + int(
                (valid_idx_max - valid_idx_min) * self.starting_percentile_max
            )  # Exclusive
            zero_idx_min = valid_idx_min + int(
                (valid_idx_max - valid_idx_min) * self.starting_percentile_min
            )  # Inclusive

            if self.index_pool_size_per_episode == -1:
                index_pool_size = zero_idx_max - zero_idx_min
            else:
                assert self.index_pool_size_per_episode > 0, (
                    f"index_pool_size_per_episode must be positive or -1, but got {self.index_pool_size_per_episode}."
                )
                index_pool_size = self.index_pool_size_per_episode

            if index_pool_size >= zero_idx_max - zero_idx_min:
                indices = np.arange(zero_idx_min, zero_idx_max)
                random_indices = self.rng.choice(
                    range(zero_idx_min, zero_idx_max),
                    size=index_pool_size - (zero_idx_max - zero_idx_min),
                    replace=True,
                )
                indices = np.concatenate([indices, random_indices])
            else:
                indices = self.rng.choice(
                    range(zero_idx_min, zero_idx_max),
                    size=index_pool_size,
                    replace=False,
                )
            indices = np.sort(indices)
            for index in indices:
                self.index_pool.append((episode_idx, index))

    def _update_episode_indices(self):
        if len(self.include_episode_indices) > 0:
            log.info(f"Dataset {self.name}: Using specified episode indices: {self.include_episode_indices}.")
            self.include_episode_num = len(self.include_episode_indices)
            for episode_idx in self.include_episode_indices:
                assert episode_idx < self.store_episode_num, (
                    f"episode_idx {episode_idx} is out of range. Max is {self.store_episode_num}."
                )
        else:
            if self.include_episode_num > 0:
                assert self.include_episode_num <= self.store_episode_num, (
                    f"include_episode_num {self.include_episode_num} is greater than the number of episodes {self.store_episode_num}."
                )
                self.include_episode_indices = self.rng.choice(
                    self.store_episode_num, size=self.include_episode_num, replace=False
                ).tolist()
                log.info(
                    f"Dataset {self.name}: Using {self.include_episode_num} episodes from {self.store_episode_num} episodes: {self.include_episode_indices}"
                )
            elif self.include_episode_num == -1:
                self.include_episode_num = self.store_episode_num
                self.include_episode_indices = list(range(self.include_episode_num))
                log.info(
                    f"Dataset {self.name}: Using all {self.include_episode_num} episodes from {self.store_episode_num}"
                )
            else:
                raise ValueError(
                    f"include_episode_num {self.include_episode_num} is invalid. Must be -1 or a positive integer."
                )

        self.include_episode_indices = sorted(self.include_episode_indices)

        self.used_episode_indices = cast(
            list[int],
            self.rng.choice(
                self.include_episode_indices,
                size=int(self.include_episode_num * self.used_episode_ratio),
                replace=False,
            ).tolist(),
        )
        self.used_episode_indices = sorted(self.used_episode_indices)
        self.used_episode_num = len(self.used_episode_indices)

    def repeat_dataset(self, repeat_num: float | None = None):
        if repeat_num is None:
            repeat_num = self.repeat_dataset_num
        else:
            self.repeat_dataset_num = repeat_num
        index_pool_size = len(self.index_pool)
        repeated_size = int(index_pool_size * repeat_num)
        repeated_indices = self.rng.choice(
            range(index_pool_size),
            size=repeated_size,
            replace=True,
        )
        self.index_pool = [self.index_pool[i] for i in repeated_indices]

    def split_unused_episodes(
        self,
        remaining_ratio: float = 1.0,
        other_used_episode_indices: Optional[list[int]] = None,
    ):
        """
        Split unused episodes from the included episodes.
        """
        log.info(
            f"Splitting unused data with remaining ratio {remaining_ratio} and other used episode ids {other_used_episode_indices}."
        )
        unused_dataset = copy.deepcopy(self)
        unused_dataset.rng = np.random.default_rng(unused_dataset.seed)
        if other_used_episode_indices is None:
            other_used_episode_indices = []
        unused_episode_indices = [
            episode_idx
            for episode_idx in self.include_episode_indices
            if episode_idx not in self.used_episode_indices and episode_idx not in other_used_episode_indices
        ]
        unused_dataset.used_episode_indices = cast(
            list[int],
            self.rng.choice(
                unused_episode_indices,
                size=int(len(unused_episode_indices) * remaining_ratio),
                replace=False,
            ).tolist(),
        )
        unused_dataset.used_episode_indices = sorted(unused_dataset.used_episode_indices)
        unused_dataset.used_episode_ratio = len(unused_dataset.used_episode_indices) / len(
            unused_dataset.include_episode_indices
        )
        unused_dataset._check_data_validity()
        unused_dataset._create_index_pool()
        assert len(unused_dataset) >= 1, (
            f"Splitted dataset {unused_dataset.name} has no data. Please check the used_data_ratio and the overall dataset size"
        )
        return unused_dataset

    def sample_data(
        self,
        output_entry_names: list[str],
        sample_num: int,
        augment_data: bool,
        normalize_data: bool,
        sampled_indices: npt.NDArray[np.int64] | None = None,
    ) -> dict[str, torch.Tensor]:
        raise NotImplementedError("This method should be implemented in subclasses.")

    def calc_stats_worker_single_process(self, start_idx: int, end_idx: int, entry_names: list[str]):
        data: dict[str, torch.Tensor] = self.sample_data(
            sample_num=end_idx - start_idx,
            output_entry_names=entry_names,
            augment_data=False,
            normalize_data=False,
            sampled_indices=np.arange(start_idx, end_idx),
        )
        min_vals: dict[str, torch.Tensor] = {}
        max_vals: dict[str, torch.Tensor] = {}
        for entry_name in entry_names:
            # data[entry_name]: (batch_size, traj_length, ...)
            min_vals[entry_name] = torch.min(torch.min(data[entry_name], dim=0).values, dim=0).values
            max_vals[entry_name] = torch.max(torch.max(data[entry_name], dim=0).values, dim=0).values
        return min_vals, max_vals

    def calc_stats(self, entry_names: list[str], process_num: int = 0):
        # assert process_num > 0
        if process_num == 0:
            process_num = min(20, multiprocessing.cpu_count() - 2)
        log.info(f"Calculating stats with {process_num} processes.")

        index_pool_size = len(self.index_pool)
        indices_per_process = index_pool_size // process_num
        start_indices = np.arange(0, index_pool_size, indices_per_process)[:process_num]
        end_indices = start_indices + indices_per_process
        end_indices[-1] = index_pool_size

        with multiprocessing.Pool(process_num) as pool:
            results = pool.starmap(
                self.calc_stats_worker_single_process, zip(start_indices, end_indices, [entry_names] * process_num)
            )

        stats: dict[str, dict[str, torch.Tensor]] = {}

        for result in results:
            for entry_name in entry_names:
                if entry_name not in stats:
                    stats[entry_name] = {
                        "min": result[0][entry_name],
                        "max": result[1][entry_name],
                    }
                else:
                    stats[entry_name]["min"] = torch.min(stats[entry_name]["min"], result[0][entry_name])
                    stats[entry_name]["max"] = torch.max(stats[entry_name]["max"], result[1][entry_name])

        return stats

    def fit_normalizer(self) -> FixedNormalizer:
        log.info(f"Fitting normalizer for {self.name}.")
        self.normalizer = FixedNormalizer(self.output_data_meta)

        normalize_entries = [
            entry_meta.name for entry_meta in self.output_data_meta.values() if entry_meta.normalizer != "identity"
        ]

        stats = self.calc_stats(
            entry_names=normalize_entries,
        )
        log.info(f"Stats: {stats}")

        self.normalizer.from_dict(stats)

        self.normalizer.to(torch.device("cpu"))

        normalizer_state_dict = self.normalizer.as_dict("list")

        if self.normalizer_dir != "":
            os.makedirs(self.normalizer_dir, exist_ok=True)
            normalizer_path = os.path.join(self.normalizer_dir, f"{self.name}_normalizer.json")
            with open(normalizer_path, "w") as f:
                json.dump(normalizer_state_dict, f)
            log.info(f"Normalizer dict saved to {normalizer_path}.")

        return self.normalizer

    def process_image_data(self, data: npt.NDArray[Any]) -> npt.NDArray[np.float32]:
        if (
            data.shape[-1] <= 4
        ):  # (..., H, W, C) where the color dimension is usually a small number (1 (grayscale), 3 (RGB), or 4 (RGBD))
            dims = len(data.shape)
            data = data.transpose((*range(dims - 3), -1, -3, -2))  # (..., C, H, W)

        if data.dtype == np.uint8:
            return (data / 255.0).astype(np.float32)

        return data

    def __len__(self) -> int:
        return len(self.index_pool)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        raise NotImplementedError("This method should be implemented in subclasses.")


class AggregatedDataset(BaseDataset):
    """
    Dataset loader for the iPhUMI dataset.
    Example structure:
    /
    ├── data
    │   ├── camera0_main_rgb (N, image_size, image_size, 3) uint8
    │   ├── camera0_ultrawide_rgb (N/6, image_size, image_size, 3) uint8
    │   ├── robot0_demo_end_pose (N, 6) float64
    │   ├── robot0_demo_start_pose (N, 6) float64
    │   ├── robot0_eef_pos (N, 3) float32
    │   ├── robot0_eef_rot_axis_angle (N, 3) float32
    │   └── robot0_gripper_width (N, 1) float32
    └── meta
        └── episode_ends (K,) int64

    """

    def __init__(
        self,
        down_sample_steps: int,
        **kwargs,
    ):
        self.down_sample_steps: int = down_sample_steps
        kwargs["history_padding_length"] = kwargs["history_padding_length"] * down_sample_steps
        kwargs["future_padding_length"] = kwargs["future_padding_length"] * down_sample_steps
        for meta in kwargs["source_data_meta"].values():
            meta["include_indices"] = [i * down_sample_steps for i in meta["include_indices"]]

        super().__init__(**kwargs)

        data_store = self.zarr_store["data"]
        assert isinstance(data_store, zarr.Group)

        self.data_store: zarr.Group = data_store
        self.data_store_keys: list[str] = list(data_store.keys())

        self.episode_ends: npt.NDArray[np.int64] = np.array(self.zarr_store["meta"]["episode_ends"])
        self.store_episode_num: int = len(self.episode_ends)

        self._update_episode_indices()

        self.episode_starts: npt.NDArray[np.int64] = np.zeros_like(self.episode_ends)

        for i, end in enumerate(self.episode_ends):
            if i == 0:
                self.episode_starts[i] = 0
            else:
                self.episode_starts[i] = self.episode_ends[i - 1]

            self.episode_frame_nums[i] = end - self.episode_starts[i]
            self.episode_valid_indices_min[i] = self.max_history_length - self.history_padding_length
            self.episode_valid_indices_max[i] = (
                self.episode_frame_nums[i] + self.future_padding_length - self.max_future_length
            )

        self._create_index_pool()

        log.info(
            f"Dataset: {self.name}, store_episode_num: {self.store_episode_num}, include_episode_num: {self.include_episode_num}, used_episode_num: {self.used_episode_num}"
        )

    def _check_data_validity(self):
        # Not implemented yet. Will skip checking for now.
        pass

    def _process_source_data(self, data_dict: dict[str, npt.NDArray[Any]]) -> dict[str, npt.NDArray[Any]]:
        # Override this function to process the source data
        raise NotImplementedError("This function should be overridden by the subclass.")

    def _get_single_traj_data(
        self,
        episode_idx: int,
        traj_idx: int,
        output_entry_names: list[str] | None = None,
    ):
        episode_length = self.episode_frame_nums[episode_idx]
        start_idx = self.episode_starts[episode_idx]

        source_data_dict: dict[str, Any] = {}

        if output_entry_names is not None and len(output_entry_names) > 0:
            source_entry_names = [
                self.output_data_meta[target_entry_name].source_entry_names for target_entry_name in output_entry_names
            ]
            source_entry_names = [item for sublist in source_entry_names for item in sublist]
        else:
            source_entry_names = None

        for entry_meta in self.source_data_meta.values():
            if (
                source_entry_names is not None
                and len(source_entry_names) > 0
                and entry_meta.name not in source_entry_names
            ):
                # Skip the entries that are not needed to make data loading faster
                continue

            if entry_meta.name not in self.data_store_keys:
                continue
            indices = [traj_idx + i for i in entry_meta.include_indices]
            # Crop the indices to the valid range. Will introduce padding if the indices are out of range.
            indices = [(0 if i < 0 else episode_length - 1 if i >= episode_length else i) for i in indices]
            global_indices = [start_idx + i for i in indices]
            if entry_meta.name.endswith("ultrawide_rgb"):
                global_indices = self.zarr_store["meta"][f"upsample_index_{entry_meta.name}"][global_indices]

            source_data_dict[entry_meta.name] = np.array(self.data_store[entry_meta.name][global_indices])

        processed_data_dict = self._process_source_data(source_data_dict)

        torch_data_dict: dict[str, Any] = {}
        torch_data_dict["episode_idx"] = torch.tensor([episode_idx])  # [1]
        torch_data_dict["traj_idx"] = torch.tensor([traj_idx])  # [1]

        for entry_meta in self.output_data_meta.values():
            if (
                output_entry_names is not None
                and len(output_entry_names) > 0
                and entry_meta.name not in output_entry_names
            ):
                # Skip the entries that are not needed to make data loading faster
                continue

            assert entry_meta.name in processed_data_dict
            processed_data = processed_data_dict[entry_meta.name]
            if isinstance(processed_data, np.ndarray):
                if entry_meta.data_type == "image":
                    processed_data = self.process_image_data(processed_data)  # -> (..., C, H, W), float32
                processed_data = torch.from_numpy(processed_data)

            torch_data_dict[entry_meta.name] = processed_data

        # Pass through initial_pose (non-normalized route only).
        if "initial_pose" in processed_data_dict:
            torch_data_dict["initial_pose"] = torch.from_numpy(processed_data_dict["initial_pose"])

        return torch_data_dict

    def __getitem__(self, idx: int):
        """
        output_data_dict:
            video: (C, T, H, W) uint8
            camera_poses: (T, 9) # This is actually the TCP pose: the middle point of the gripper tip
            eef_poses: (T, 9) (xyz, rot6d), is defined self.eef_z_offset behind the gripper tip
            eef_commands: (T, 1)
            t5_text_embeddings: (512, 1024)
            t5_text_mask: (512,) int64
            fps: int
            __key__: int

        if self.use_grasp_state_repr:
            grasp_states: (T, 6) (left-finger-xyz, right-finger-xyz) wrt eef_poses
        """
        episode_idx, traj_idx = self.index_pool[idx]

        # skip data if corrupted
        while True:
            try:
                torch_data_dict = self._get_single_traj_data(episode_idx, traj_idx)
                break
            except Exception:
                import traceback

                traceback.print_exc()
                log.warning(
                    f"UMI Dataset WARNING: Corrupted data found at episode_idx: {episode_idx}, traj_idx: {traj_idx}. Randomly selecting a new data.",
                )
                random_idx = self.rng.choice(len(self.index_pool))
                episode_idx, traj_idx = self.index_pool[random_idx]

        # Pop initial_pose before transforms (transforms crash on unknown keys).
        initial_pose = torch_data_dict.pop("initial_pose", None)

        torch_data_dict = self.transforms.apply(torch_data_dict)

        if self.normalizer is not None:
            torch_data_dict = self.normalizer.normalize(torch_data_dict)

        # Re-add initial_pose (non-normalized route only, for viewer).
        if initial_pose is not None:
            torch_data_dict["initial_pose"] = initial_pose

        for entry_meta in self.output_data_meta.values():
            assert torch_data_dict[entry_meta.name].shape == (
                entry_meta.length,
                *entry_meta.shape,
            ), (
                f"entry_meta: {entry_meta.name}, torch_data_dict[entry_meta.name].shape: {torch_data_dict[entry_meta.name].shape}, entry_meta.length: {entry_meta.length}, entry_meta.shape: {entry_meta.shape}"
            )

        # resize to image_size x image_size if needed
        if torch_data_dict["video"].shape[-2:] != (self.image_size, self.image_size):
            torch_data_dict["video"] = torch.nn.functional.interpolate(
                torch_data_dict["video"], size=(self.image_size, self.image_size), mode="bilinear"
            )
        torch_data_dict["video"] = (torch_data_dict["video"] * 255.0).to(torch.uint8)

        if torch_data_dict["video"].shape[1] == 3:  # [T,C,H,W]
            torch_data_dict["video"] = torch_data_dict["video"].permute(1, 0, 2, 3)  # [C,T,H,W]

        torch_data_dict["__key__"] = torch.tensor([idx], dtype=torch.long)  # [1]
        torch_data_dict["conditioning_fps"] = torch.tensor(60 / self.down_sample_steps, dtype=torch.long)  # scalar

        if self.mode == "joint":
            mode = random.choice(["forward_dynamics", "inverse_dynamics", "policy"])
        else:
            mode = self.mode
        torch_data_dict["mode"] = mode

        _name = re.sub(r"(_v?\d+)+$", "", self.name)
        torch_data_dict["ai_caption"] = _name.replace("_", " ")
        torch_data_dict["domain_id"] = torch.tensor([self.domain_id], dtype=torch.long)  # [1]
        torch_data_dict["viewpoint"] = "wrist_view"

        if not self.use_grasp_state_repr:
            # Use eef_poses or camera_poses for the action based on self.use_eef_pose
            action = torch.cat(
                [
                    torch_data_dict["eef_poses"] if self.use_eef_pose else torch_data_dict["camera_poses"],
                    torch_data_dict["eef_commands"],
                ],
                dim=1,
            ).to(torch.float32)  # [T,camera_pose_dim+1]
        else:
            action = torch.cat(
                [
                    torch_data_dict["eef_poses"] if self.use_eef_pose else torch_data_dict["camera_poses"],
                    torch_data_dict["grasp_states"],
                ],
                dim=1,
            ).to(torch.float32)  # [T,eef_pose_dim+grasp_state_dim]

        torch_data_dict["action"] = action

        return torch_data_dict

    def sample_data(
        self,
        output_entry_names: list[str],
        sample_num: int,
        augment_data: bool,
        normalize_data: bool,
        sampled_indices: npt.NDArray[np.int64] | None = None,
    ) -> dict[str, torch.Tensor]:
        if sample_num == -1:
            sample_num = len(self.index_pool)

        if sampled_indices is None:
            sampled_indices = self.rng.choice(
                len(self.index_pool), min(sample_num, len(self.index_pool)), replace=False
            )
        else:
            assert len(sampled_indices) == sample_num, (
                f"sampled_indices should be of length {sample_num}, but got {len(sampled_indices)}"
            )

        samples = []
        log.info(f"Sampling {sample_num} data from {len(self.index_pool)} trajectories.")
        for idx in tqdm.tqdm(sampled_indices):
            episode_idx, traj_idx = self.index_pool[idx]
            samples.append(self._get_single_traj_data(episode_idx, traj_idx, output_entry_names))

        all_samples_data_dict: dict[str, torch.Tensor] = aggregate_batch(samples, aggregate_fn=torch.stack)

        return all_samples_data_dict


class MultiTaskDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        name: str,
        sub_dataset_target: str,
        dataset_configs: dict[str, dict[str, Any]],
        eager_load: bool = False,
        **base_config: dict[str, Any],
    ):
        """
        name: if select certain sub-datasets, should be the name of multiple sub-datasets connected by ","
        root, normalizer_dir should be the same for all datasets
        dataset_configs: {
            "dataset_name_1": {
                "sample_ratio": 1.0,
                "override_config_1": "value_1",
                "override_config_2": "value_2",
                ...
            },
            ...
        }
        """

        if isinstance(dataset_configs, DictConfig):
            dataset_configs = cast(dict[str, dict[str, Any]], OmegaConf.to_container(dataset_configs))
        if ";" in name:
            name = name.replace(";", ",")
        if "," in name:
            dataset_names = name.split(",")
            dataset_configs = {name: dataset_configs[name] for name in dataset_names}

        log.info(f"dataset_configs: {dataset_configs}")
        self.dataset_configs: dict[str, dict[str, Any]] = dataset_configs
        assert len(self.dataset_configs) >= 1, "At least one dataset is required"

        if isinstance(base_config, DictConfig):
            base_config = cast(dict[str, Any], OmegaConf.to_container(base_config))
        self.base_config: dict[str, Any] = base_config

        self.sample_ratios: dict[str, float] = {
            name: config.pop("sample_ratio") for name, config in self.dataset_configs.items()
        }


        self._all_shard_roots = list(self.dataset_configs.keys())

        self.datasets: dict[str, BaseDataset] = {}
        self.index_pool: list[tuple[str, int]] = []

        self.sub_dataset_target = sub_dataset_target

        if eager_load:
            self._register_sources()

    def _register_sources(self, indices: list[int] | None = None):
        if indices is None:
            indices = list(range(len(self._all_shard_roots)))
        log.info(f"Registering UMI Multi-Task datasets: {indices}")

        for idx in indices:
            base_name = self._all_shard_roots[idx]
            dataset_config = self.dataset_configs[base_name]
            num_shards = dataset_config.get("num_shards", 0)
            shard_config = {k: v for k, v in dataset_config.items() if k != "num_shards"}
            if num_shards > 0:
                for i in range(num_shards):
                    shard_name = f"{base_name}_v{i}"
                    log.info(f"Initializing dataset shard: {shard_name}")
                    config = copy.deepcopy(self.base_config)
                    config.update(copy.deepcopy(shard_config))
                    config["name"] = shard_name
                    config["_target_"] = self.sub_dataset_target
                    self.datasets[shard_name] = hydra.utils.instantiate(config)
            else:
                log.info(f"Initializing dataset: {base_name}")
                config = copy.deepcopy(self.base_config)
                config.update(copy.deepcopy(shard_config))
                config["name"] = base_name
                config["_target_"] = self.sub_dataset_target
                self.datasets[base_name] = hydra.utils.instantiate(config)

        self._create_index_pool()

    @property
    def normalizer(self) -> FixedNormalizer | None:
        return next(iter(self.datasets.values())).normalizer

    @property
    def action_dim(self) -> int:
        """ """
        return 10

    def _create_index_pool(self):
        self.index_pool = []
        for dataset_name, dataset in self.datasets.items():
            self.index_pool.extend((dataset_name, i) for i in range(len(dataset)))

    def __len__(self):
        return len(self.index_pool)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        dataset_name, data_idx = self.index_pool[idx]
        data = self.datasets[dataset_name][data_idx]
        dataset_index = list(self.datasets.keys()).index(dataset_name)
        data["dataset_index"] = torch.tensor([dataset_index])  # [1]
        data["data_index"] = torch.tensor([data_idx])  # [1]
        return data

    def split_unused_episodes(
        self,
        remaining_ratio: float = 1.0,
        other_used_episode_indices: list[int] | None = None,
    ):
        unused_dataset = copy.deepcopy(self)
        unused_dataset.index_pool = []
        unused_dataset.datasets = {}

        for dataset_name, dataset in self.datasets.items():
            unused_dataset.datasets[dataset_name] = dataset.split_unused_episodes(
                remaining_ratio, other_used_episode_indices
            )
        unused_dataset._create_index_pool()

        return unused_dataset

    def repeat_dataset(self):
        for dataset_name, dataset in self.datasets.items():
            dataset.repeat_dataset()
        self._create_index_pool()


def _process_source_data(
    self: AggregatedDataset, data_dict: dict[str, npt.NDArray[Any]]
) -> dict[str, npt.NDArray[Any]]:
    """
    Will calculate the following data:
        relative poses
        poses wrt episode start
    This step does not include normalization and data augmentation
    Input data_dict:
        camera0_main_rgb: (..., H, W, 3) uint8
        camera0_ultrawide_rgb: (..., H, W, 3) uint8
        robot0_demo_start_pose: (1, 6) float64 (optional)
        robot0_eef_pos: (..., 3) float32
        robot0_eef_rot_axis_angle: (..., 3) float32
        robot0_gripper_width: (..., 1) float32

    Output data_dict:
        video: (C, T, H, W) uint8
        camera_poses: (T, 9) # 9dim: [pos_3d, rot_6d], rot_6d is the first two rows of the rotation matrix
        eef_poses: (T, 9) (xyz, rot6d), is defined self.eef_z_offset behind the gripper tip
        eef_commands: (T, 1)
        t5_text_embeddings: (512, 1024)
        t5_text_mask: (512,) int64
        fps: int
        __key__: int

    if self.use_grasp_state_repr:
        grasp_states: (T, 6) (left-finger-xyz, right-finger-xyz) wrt eef_poses

    """

    processed_data_dict: dict[str, npt.NDArray[Any]] = {}

    pose_indices = self.source_data_meta["robot0_eef_pos"].include_indices
    assert pose_indices == self.source_data_meta["robot0_eef_rot_axis_angle"].include_indices, (
        f"robot0_eef_pos and robot0_eef_rot_axis_angle must be aligned"
    )

    gripper_width_indices = self.source_data_meta["robot0_gripper_width"].include_indices

    ## Use absolute gripper width
    if self.use_relative_gripper_width:
        assert len(pose_indices) == len(gripper_width_indices), f"{len(pose_indices)=}, {len(gripper_width_indices)=}"
    else:
        assert len(pose_indices) == len(gripper_width_indices) + 1, (
            f"{len(pose_indices)=}, {len(gripper_width_indices)=}"
        )

    zero_idx = pose_indices.index(0)

    assert self.robot_num == 1, "Currently only support one robot"

    for i in range(self.robot_num):
        assert len(pose_indices) == self.output_data_meta[f"camera_poses"].length + 1, (
            f"{len(pose_indices)=}, {self.output_data_meta[f'camera_poses'].length=}"
        )


        camera_poses_meta = self.output_data_meta[f"camera_poses"]
        camera_poses = np.zeros((camera_poses_meta.length, *camera_poses_meta.shape), dtype=np.float32)

        eef_commands_meta = self.output_data_meta[f"eef_commands"]
        eef_commands = np.zeros((eef_commands_meta.length, *eef_commands_meta.shape), dtype=np.float32)

        video_meta = self.output_data_meta[f"video"]
        if video_meta.source_entry_names[0] in data_dict:
            video = data_dict[video_meta.source_entry_names[0]]
            processed_data_dict["video"] = video

        if f"robot{i}_eef_pos" in data_dict and f"robot{i}_eef_rot_axis_angle" in data_dict:
            pose_mat = build_abs_pose_from_components(
                data_dict[f"robot{i}_eef_pos"],
                data_dict[f"robot{i}_eef_rot_axis_angle"],
                "axisangle",
            )
            assert self.use_relative_pose, "Currently only support relative pose"
            pose_convention = "backward_framewise" if self.relative_pose_mode == "frame_wise" else "backward_anchored"

            pose = pose_abs_to_rel(pose_mat, rotation_format="rot6d", pose_convention=pose_convention)
            assert len(pose) == camera_poses_meta.length, (
                f"pose should be one step longer (zero index is excluded) than camera_poses_meta.length, but got {len(pose)} and {camera_poses_meta.length}"
            )
            processed_data_dict[f"camera_poses"] = (
                pose  # The camera poses here are actually TCP pose (the middle point of the gripper tip)
            )

            # Stash the absolute first-frame pose for viewer trajectory reconstruction.
            # Only useful on the non-normalized route (normalizer_dir="").
            if self.normalizer is None:
                processed_data_dict["initial_pose"] = pose_mat[zero_idx].astype(np.float32)

            offset_eef_pose_mat = pose_mat.copy()
            offset_eef_pose_mat[:, :3, 3] += (
                offset_eef_pose_mat[:, :3, 2] * self.eef_z_offset
            )  # Apply a z offset in the local coordinate frame instead of the global frame
            offset_eef_pose = pose_abs_to_rel(
                offset_eef_pose_mat, rotation_format="rot6d", pose_convention=pose_convention
            )
            processed_data_dict[f"eef_poses"] = offset_eef_pose

        if f"robot{i}_gripper_width" in data_dict:
            ## Use absolute gripper width
            # eef_commands[:] = data_dict[f"robot{i}_gripper_width"]
            # processed_data_dict[f"eef_commands"] = eef_commands

            assert not self.use_relative_gripper_width, "Currently only support absolute gripper width"
            processed_gripper_width = data_dict[f"robot{i}_gripper_width"]

            eef_commands[:] = processed_gripper_width
            processed_data_dict[f"eef_commands"] = eef_commands

            if self.use_grasp_state_repr:
                processed_data_dict[f"grasp_states"] = np.zeros((eef_commands_meta.length, 6), dtype=np.float32)
                processed_data_dict[f"grasp_states"][:, 2] = -self.eef_z_offset
                processed_data_dict[f"grasp_states"][:, 5] = -self.eef_z_offset
                processed_data_dict[f"grasp_states"][:, :1] = eef_commands / 2
                processed_data_dict[f"grasp_states"][:, 3:4] = -eef_commands / 2

        # if f"robot{i}_demo_start_pose" in data_dict and f"robot{i}_eef_rot_axis_angle_wrt_start" in self.output_data_meta:
        #     # Calculate relative poses wrt episode start
        #     try:
        #         wrt_start_entry_meta = self.output_data_meta[
        #             f"robot{i}_eef_rot_axis_angle_wrt_start"
        #         ]
        #         assert (
        #             data_dict[f"robot{i}_demo_start_pose"].shape[0] == 1
        #         ), "robot0_demo_start_pose must be (1, 6)"
        #         # HACK: add noise to episode start pose. Copied from the original UMI codebase.
        #         start_pose: npt.NDArray[np.float64] = data_dict[
        #             f"robot{i}_demo_start_pose"
        #         ][0]
        #         start_pose += self.rng.normal(
        #             scale=[0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
        #             size=start_pose.shape,
        #         )
        #         start_pose_mat = pose_to_mat(start_pose)
        #         rel_pose_mat = convert_pose_mat_rep(
        #             pose_mat,
        #             base_pose_mat=start_pose_mat,
        #             pose_rep="relative",
        #             backward=False,
        #         )
        #     except ValueError:
        #         # No wrt_start_entry_meta, so no relative poses wrt episode start
        #         # print(f"No wrt_start_entry_meta for robot{i}")
        #         pass

    return processed_data_dict


class iPhUMISingleTrajDataset(AggregatedDataset):
    def __init__(self, align_to_ultrawide: bool, **kwargs):
        """
        If align_to_ultrawide is True, the dataset will be aligned to the ultrawide camera. (will only use 1/6 of the data, but will guarantee the main camera is aligned to the ultrawide camera)
        """
        if "camera0_ultrawide_rgb" in kwargs["source_data_meta"]:
            # assert kwargs["down_sample_steps"] == 6, "down_sample_steps must be 6 for iPhUMI dataset if using ultrawide camera. This is because the ultrawide camera is captured at 10Hz while others are at 60Hz."
            assert len(kwargs["source_data_meta"]["camera0_ultrawide_rgb"].include_indices) == 1, (
                "camera0_ultrawide_rgb should not include history, because it is captured at 10Hz"
            )
        self.align_to_ultrawide: bool = align_to_ultrawide

        super().__init__(**kwargs)
        if self.align_to_ultrawide:
            assert self.source_data_meta["camera0_ultrawide_rgb"].include_indices == [0], (
                "camera0_ultrawide_rgb should only include the current frame"
            )

    def _create_index_pool(self):
        super()._create_index_pool()
        if self.align_to_ultrawide:
            new_index_pool: list[tuple[int, int]] = []

            ultrawide_indices: npt.NDArray[np.int32] = np.array(
                self.zarr_store["meta"]["upsample_index_camera0_ultrawide_rgb"], dtype=np.int32
            )
            regular_indices: npt.NDArray[np.int32] = np.array(
                self.zarr_store["meta"]["downsample_index_camera0_ultrawide_rgb"], dtype=np.int32
            )
            # Only use the data that is aligned to the ultrawide camera
            for episode_idx, traj_idx in self.index_pool:
                start_idx = self.episode_starts[episode_idx]
                global_idx = start_idx + traj_idx
                if regular_indices[ultrawide_indices[global_idx]] == global_idx:
                    new_index_pool.append((episode_idx, traj_idx))

            log.info(
                f"Aligning to ultrawide camera, index pool size changed from {len(self.index_pool)} to {len(new_index_pool)}"
            )
            self.index_pool = new_index_pool


iPhUMISingleTrajDataset._process_source_data = _process_source_data


class UMISingleTrajDataset(AggregatedDataset):
    pass


UMISingleTrajDataset._process_source_data = _process_source_data


def get_umi_dataset(
    dataset_name: str,
    dataset_type: str,
    is_val: bool,
    mode: str = "policy",
    image_size: int = 256,
    root: str = "/lustre/fsw/portfolios/cosmos/projects/cosmos_base_training/cosmos3_action_datasets/umi/",
    normalizer_dir: str | None = None,
    **kwargs,
):
    # assert len(kwargs) == 0, f"get_umi_dataset does not accept any kwargs, but got {kwargs}"
    log.info(f"get_umi_dataset override kwargs: {kwargs=}")
    try:
        OmegaConf.register_new_resolver("eval", eval)
    except Exception:
        pass
    if dataset_type == "umi":
        cfg = OmegaConf.load("cosmos/data/vfm/action/umi_dataset.yaml")
    elif dataset_type == "umi_multi_task":
        if hydra.core.global_hydra.GlobalHydra().is_initialized():
            cfg = hydra.compose(config_name="umi_dataset_multi_task")
        else:
            with hydra.initialize("."):
                cfg = hydra.compose(config_name="umi_dataset_multi_task")
    else:
        raise ValueError(f"Invalid dataset type: {dataset_type}. Available dataset types: umi, umi_multi_task")

    cfg.update(kwargs)

    cfg.name = dataset_name
    cfg.mode = mode
    if root is not None:
        cfg.root = root
    if normalizer_dir is not None:
        cfg.normalizer_dir = normalizer_dir
    assert mode in ["policy", "joint", "forward_dynamics", "inverse_dynamics", "image2video"], (
        f"Invalid mode: {mode}. Available modes: policy, joint, forward_dynamics, inverse_dynamics, video"
    )
    cfg.image_size = image_size
    if is_val:
        cfg.output_data_meta = cfg.val_output_data_meta
        cfg.eager_load = True
    cfg.val_output_data_meta = None  # type: ignore
    OmegaConf.resolve(cfg)
    dataset = hydra.utils.instantiate(cfg)
    if is_val:
        return dataset.split_unused_episodes()
    else:
        return dataset


if __name__ == "__main__":
    # OmegaConf.register_new_resolver("eval", eval)
    # cfg = OmegaConf.load("cosmos/data/vfm/action/umi_dataset.yaml")
    # os.environ["HYDRA_FULL_ERROR"] = "1"
    args = sys.argv[1:]
    dataset_type = args[0]
    dataset_name = args[1]

    dataset: BaseDataset = get_umi_dataset(
        dataset_name, dataset_type, is_val=False, use_grasp_state_repr=False, use_eef_pose=False
    )

    # dataset.fit_normalizer()

    def save_np_array_as_video(rollout_images: list[np.ndarray] | torch.Tensor, video_path: str, fps: int = 10):
        """
        rollout_images: (T, H, W, C) or (T, C, H, W)
        Saves an MP4 replay of an episode.
        """

        if isinstance(rollout_images, torch.Tensor):
            assert rollout_images.ndim == 4, f"Rollout images must be a 4D tensor, but got {rollout_images.ndim}"
            if rollout_images.shape[1] == 3:
                rollout_images = rollout_images.permute(0, 2, 3, 1)  # [T,H,W,C]
            rollout_images = rollout_images.cpu().numpy()

        import imageio

        video_writer = imageio.get_writer(video_path, fps=fps)
        for img in rollout_images:
            video_writer.append_data(img)
        video_writer.close()
        print(f"Saved rollout MP4 at path {video_path}")

    print(dataset)
    ld = len(dataset)
    iss = [0, ld // 4, ld // 2, ld // 4 * 3, -1]
    for i in iss:
        # for i in range(60, 70):
        data = dataset[i]

        # print(data["video"].shape)  # [C,T,H,W]
        # video_thwc = data["video"].permute(1, 2, 3, 0)  # [T,H,W,C]
        # save_np_array_as_video(video_thwc, f"video_{i}.mp4")
        # print(data["camera_poses"])
        # for k, v in data.items():
        #    print(k, v.shape)
        print("camera_poses", data["camera_poses"])
        print("eef_commands", data["eef_commands"])
        print("eef_poses", data["eef_poses"])
        if dataset.use_grasp_state_repr:
            print("grasp_states", data["grasp_states"])
        print("action", data["action"])
