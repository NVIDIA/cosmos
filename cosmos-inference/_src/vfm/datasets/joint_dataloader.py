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

from collections import deque
from dataclasses import dataclass
from typing import Any, ClassVar, Dict, Union

import numpy as np
import torch
import webdataset
from torch.utils.data.dataloader import default_collate

from cosmos3._src.imaginaire.lazy_config import instantiate
from cosmos3._src.imaginaire.utils import log

_TIMING_KEYS = {"_sample_time", "_aug_time", "_pre_aug_time", "_aug_step_times"}
_BATCH_TIMING_KEYS = {
    "_worker_batch_time",
    "_worker_aug_time",
    "_worker_io_time",
    "_worker_aug_step_times",
    "_worker_id",
}


def custom_collate_fn(batch):
    """
    Collate function that works like default_collate for all keys other than "text_token_ids", "images", and "video".
    For "text_token_ids", "images", and "video" it simply returns them in a list, instead of stacking them as a tensor.
    """
    list_collate_keys = {
        "text_token_ids",
        "images",
        "video",
        "action",
        "domain_id",
        "sequence_plan",
        "sound",
        "raw_action_dim",
        "image_size",
    }

    # Handle the case where the batch is already a dictionary (e.g. column-wise batching)
    if isinstance(batch, dict):
        return {key: (value if key in list_collate_keys else default_collate(value)) for key, value in batch.items()}

    # Handle standard list of samples
    elem = batch[0]
    if isinstance(elem, dict):

        result = {}
        for key in elem:
            if key in _TIMING_KEYS:
                continue
            if key in list_collate_keys:
                result[key] = [d[key] for d in batch]
            else:
                result[key] = default_collate([d[key] for d in batch])
        result.update(_aggregate_worker_timing(batch))
        return result
    else:
        return default_collate(batch)


def _aggregate_worker_timing(samples: list[dict]) -> dict:
    """Extract per-sample timing keys, aggregate into per-batch scalars."""
    info: dict[str, float | int] = {}
    if "_sample_time" in samples[0]:
        info["_worker_batch_time"] = sum(s.get("_sample_time", 0.0) for s in samples)
    if "_aug_time" in samples[0]:
        aug_total = sum(s.get("_aug_time", 0.0) for s in samples)
        info["_worker_aug_time"] = aug_total
        if "_worker_batch_time" in info:
            info["_worker_io_time"] = info["_worker_batch_time"] - aug_total
    if "_aug_step_times" in samples[0]:
        agg: dict[str, float] = {}
        for s in samples:
            for step_name, t in s.get("_aug_step_times", {}).items():
                agg[step_name] = agg.get(step_name, 0.0) + t
        info["_worker_aug_step_times"] = agg
    worker_info = torch.utils.data.get_worker_info()
    info["_worker_id"] = worker_info.id if worker_info is not None else 0
    return info


@dataclass
class _PackingMetrics:
    """Per-batch packing statistics collected during the packing loop.

    Also serves as the single source of truth for packing-related metric names
    via ``STATS_SPEC``, which the dataloading monitor callback consumes to
    drive accumulation and logging.
    """

    current_sequence_length: int = 0
    num_samples: int = 0
    dropped_count: int = 0
    from_buffer: int = 0
    from_workers: int = 0

    STATS_SPEC: ClassVar[list[tuple[str, str, str]]] = [
        # (batch_key, wandb_suffix, aggregation_type)
        ("_num_tokens", "token_fraction", "scalar"),
        ("_num_samples", "samples_per_batch", "list"),
        ("_from_buffer", "from_buffer", "list"),
        ("_from_workers", "from_workers", "list"),
        ("_buffer_size", "buffer_size", "list"),
        ("_dropped_count", "dropped", "scalar"),
    ]

    def attach_to(self, output_batch: dict, buffer_size: int) -> None:
        """Write packing statistics into the output batch dict."""
        output_batch["_num_tokens"] = self.current_sequence_length
        output_batch["_num_samples"] = self.num_samples
        output_batch["_from_buffer"] = self.from_buffer
        output_batch["_from_workers"] = self.from_workers
        output_batch["_buffer_size"] = buffer_size
        output_batch["_dropped_count"] = self.dropped_count


class JointDataLoader(webdataset.WebLoader):
    r"""
    A joint dataloader that supports loading both images and videos.
    """

    def __init__(
        self,
        dataloaders: Dict[str, Dict[str, Union[torch.utils.data.DataLoader, webdataset.WebLoader, int]]],
        tokenizer_spatial_compression_factor: int,
        tokenizer_temporal_compression_factor: int,
        patch_spatial: int,
        max_sequence_length: int | None,
        max_samples_per_batch: int | None,
        sound_latent_fps: float = 0,
        audio_sample_rate: int = 48000,
    ):
        """
        Initialize the JointDataLoader with multiple datasets.

        The effective mini-batch size can be controlled with either max_sequence_length or
        max_samples_per_batch. To use max_sequence_length, max_samples_per_batch needs to be None.
        Vice versa, to use max_samples_per_batch, max_sequence_length needs to be None.
        max_sequence_length and max_samples_per_batch cannot both be None simultaneously.

        Args:
            dataloaders: key - dataset_name; value - {"dataloader": dataloader, "ratio": data_ratio}
            tokenizer_spatial_compression_factor: The spatial compression factor of the tokenizer.
            tokenizer_temporal_compression_factor: The temporal compression factor of the tokenizer.
            patch_spatial: Spatial pathification factor.
            max_samples_per_batch: Max number of samples per packed batch (alternative to max_sequence_length).
            sound_latent_fps: Sound tokenizer latent rate in Hz (e.g. 25). If 0, sound tokens are not counted.
            audio_sample_rate: Audio sample rate in Hz (e.g. 48000). Used with sound_latent_fps to estimate
                sound token count.

        Example:
            joint_loader = IterativeJointDataLoader(
                dataloaders{
                    "image_data": {
                        "dataloader": webdataset.WebLoader(...),
                        "ratio": 4,
                    },
                    "video_data": {
                        "dataloader": torch.utils.data.DataLoader(...),
                        "ratio": 1,
                    },
                }
            )
        """
        self.dataloader_list, self.dataset_name_list, self.data_ratios = [], [], []
        self.tokenizer_spatial_compression_factor = tokenizer_spatial_compression_factor
        self.tokenizer_temporal_compression_factor = tokenizer_temporal_compression_factor
        self.patch_spatial = patch_spatial
        self.max_sequence_length = max_sequence_length
        self.max_samples_per_batch = max_samples_per_batch
        self.sound_latent_fps = sound_latent_fps
        self.audio_sample_rate = audio_sample_rate

        assert (self.max_sequence_length is None) != (self.max_samples_per_batch is None), (
            "Exactly one of max_sequence_length or max_samples_per_batch must be None, but not both."
        )

        for dataset_name, dataloader_data in dataloaders.items():
            assert set(dataloader_data.keys()) == {"dataloader", "ratio"}, f"Invalid config: {dataloader_data}"
            if dataloader_data["ratio"] <= 0:
                continue
            self.dataset_name_list.append(dataset_name)
            self.dataloader_list.append(instantiate(dataloader_data["dataloader"], collate_fn=custom_collate_fn))
            self.data_ratios.append(dataloader_data["ratio"])

        self.global_id = 0
        self.ratio_sum = sum(self.data_ratios)

        total = self.ratio_sum if self.ratio_sum > 0 else 1.0
        lines = [f"JointDataLoader: {len(self.dataset_name_list)} streams"]
        for name, ratio in zip(self.dataset_name_list, self.data_ratios):
            lines.append(f"  {name}: ratio={ratio:.4g} ({ratio / total:.1%})")
        log.info("\n".join(lines))

        self.data_len = 0
        self.dataloaders = [iter(dataloader) for dataloader in self.dataloader_list]
        self.buffers = [deque() for _ in range(len(self.dataloader_list))]
        for data in self.dataloader_list:
            self.data_len += len(data)

    def _compute_num_tokens_per_sample(self, data_batch: dict) -> int:
        """
        This function computes the number of tokens per sample in the data batch.
        This includes text + vision generation tokens + action tokens.

        Args:
            data_batch (dict): The data batch containing the text tokens.

        Returns:
            int: The number of tokens per sample.
        """

        # The token sequence we have is
        # <text tokens> <eos> <vision_start> <image tokens> <vision_end> [<action tokens>]
        # The spatial dimension of image tokens is compressed by
        # vae spatial downsampling factor + pathification
        # The temporal dimension of image tokens is compressed by
        # vae temporal downsampling factor
        # Action tokens have 1 token per time step (no spatial dimension)

        text_token_ids = data_batch["text_token_ids"]
        if isinstance(text_token_ids, list):
            num_text_tokens = text_token_ids[0].shape[0]
        else:
            num_text_tokens = text_token_ids.shape[1]

        num_tokens = num_text_tokens + 1

        # Vision part
        is_image_batch = "images" in data_batch
        input_images_or_videos = data_batch["images" if is_image_batch else "video"]

        # iterate over all the media in the batch
        for media in input_images_or_videos if isinstance(input_images_or_videos, list) else [input_images_or_videos]:
            if is_image_batch:
                _, H, W = media.shape
                T = 1
            else:
                _, T, H, W = media.shape

            vae_spatial_downsample = self.tokenizer_spatial_compression_factor * self.patch_spatial
            vae_temporal_downsample = self.tokenizer_temporal_compression_factor

            latent_h_shape = H // vae_spatial_downsample
            latent_w_shape = W // vae_spatial_downsample
            latent_t_shape = 1 + (T - 1) // vae_temporal_downsample

            num_vision_tokens = latent_h_shape * latent_w_shape * latent_t_shape + 2
            num_tokens += num_vision_tokens

        # Action part: each action time step is 1 token.
        # Action tensor shape is (T_action, D) per sample; stored as a single-element list.
        if "action" in data_batch:
            list_of_actions = data_batch["action"]
            for action in list_of_actions:
                # skip None actions
                if action is None:
                    continue
                num_action_tokens = action.shape[0]
                num_tokens += num_action_tokens

        # Sound part — estimate sound tokens from audio waveform length
        if self.sound_latent_fps > 0 and "sound" in data_batch:
            sound_data = data_batch["sound"]
            if isinstance(sound_data, list) and len(sound_data) > 0:
                first_sound = sound_data[0]
                # Unwrap nested list if needed
                if isinstance(first_sound, list):
                    first_sound = first_sound[0]
                if first_sound is not None and isinstance(first_sound, torch.Tensor):
                    num_audio_samples = first_sound.shape[-1]
                    audio_duration = num_audio_samples / self.audio_sample_rate
                    num_sound_tokens = int(audio_duration * self.sound_latent_fps)
                    num_tokens += num_sound_tokens

        return num_tokens

    # Keys whose value per sample is a list of tensors to be flattened into one list in the batch
    _FLATTEN_LIST_KEYS = {"image_size"}

    def _update_output_batch(self, output_batch: dict, output: dict):
        for key, value in output.items():
            if key in _BATCH_TIMING_KEYS:
                if key not in output_batch:
                    output_batch[key] = value
            elif key in self._FLATTEN_LIST_KEYS and isinstance(value, list):
                if key not in output_batch:
                    output_batch[key] = value
                else:
                    output_batch[key].extend(value)
            elif key not in output_batch:
                output_batch[key] = [value]
            else:
                output_batch[key].append(value)

    def __len__(self) -> int:
        return self.data_len

    # Keys where each sample may hold multiple tensors (e.g. multiple video
    # clips in a packed sequence).  Kept as single-element lists per sample
    # via v[i:i+1] so that _update_output_batch yields list[list[Tensor]].
    _MULTI_ITEM_KEYS = {"text_token_ids", "images", "video", "action", "sound"}

    def _get_next_sample(self, index_id: int) -> dict:
        """Pop the next single-sample dict from the buffer for the given dataloader.

        If the buffer is empty, fetches the next collated batch from the inner
        dataloader and splits it into individual samples.

        Splitting rules:
            - Multi-item list values (keys in ``_MULTI_ITEM_KEYS``): sliced
              via ``v[i:i+1]`` to yield a single-element list ``[tensor]``.
              A packed sequence can contain multiple items per key.
            - Per-sequence metadata list values (all other list keys, e.g.
              ``sequence_plan``, ``domain_id``): direct-indexed via ``v[i]``
              to yield the bare element.
            - Tensor values ``(B, ...)``: sliced to ``(1, ...)`` via
              ``v[i : i + 1]`` to preserve the batch dimension.

        After ``_update_output_batch`` accumulates samples, the packed output
        batch has the following shapes:
            - Multi-item keys (``text_token_ids``, ``video``, ``images``,
              ``action``): ``list[list[Tensor]]`` — each inner list has one
              element from one sub-sample.
            - Per-sequence metadata keys (``sequence_plan``, ``domain_id``,
              ``dataset_name``, etc.): ``list[element]`` — flat list.
            - Tensor-origin keys: ``list[Tensor(1, ...)]``.

        Args:
            index_id: Index of the dataloader to fetch from.

        Returns:
            A single-sample dictionary.
        """
        buffer = self.buffers[index_id]
        if not buffer:
            try:
                batch = next(self.dataloaders[index_id])
            except StopIteration:
                raise

            is_image_batch = "images" in batch
            input_images_or_videos = batch["images" if is_image_batch else "video"]
            batch_size = len(input_images_or_videos)

            for i in range(batch_size):
                sample = {}
                for k, v in batch.items():
                    if k in _BATCH_TIMING_KEYS:
                        sample[k] = v
                    elif isinstance(v, list) and k in self._MULTI_ITEM_KEYS:
                        # For multi-item keys (images, video, etc.), the collated
                        # value is a list with one element per sample.  If the element
                        # is itself a list (e.g. image editing: [src, tgt]), use v[i]
                        # directly to avoid wrapping it in a redundant single-element
                        # list.  Otherwise keep the v[i:i+1] slice so that
                        # _update_output_batch produces list[list[Tensor]].
                        elem = v[i]
                        if isinstance(elem, list):
                            sample[k] = elem
                        else:
                            sample[k] = v[i : i + 1]
                    elif isinstance(v, list):
                        sample[k] = v[i]
                    else:
                        sample[k] = v[i : i + 1]
                buffer.append(sample)

        return buffer.popleft()

    def set_start_iteration(self, iteration: int):
        self.global_id = iteration

    def __iter__(self):
        raise NotImplementedError("__iter__ function is not implemented yet")


class IterativeJointDataLoader(JointDataLoader):
    r"""
    An iterative joint dataloader that supports loading multiple modalities.

    The behavior depends on the ``seed`` parameter:

    - **seed is not None** (Default):
      The modality is randomly selected at each iteration based on the probability distribution
      derived from the ratios. The random state is seeded with ``seed + global_id``, ensuring
      that all ranks select the same modality at the same iteration (assuming synchronized global_id).
      This prevents load imbalance due to mixed resolutions across ranks.

    - **seed is None**:
      The modality selection follows a deterministic round-robin pattern based on the ratios.
      For example, with 2 modalities (image and video) and ratio 2:1:
        - Iterations 0, 1: all ranks process images
        - Iteration 2: all ranks process videos
        - ... and so on.
      This also ensures all ranks process the same modality at the same iteration.
    """

    def __init__(
        self,
        dataloaders: Dict[str, Dict[str, Union[torch.utils.data.DataLoader, webdataset.WebLoader, int]]],
        tokenizer_spatial_compression_factor: int,
        tokenizer_temporal_compression_factor: int,
        patch_spatial: int,
        max_sequence_length: int | None = None,
        max_samples_per_batch: int | None = None,
        sound_latent_fps: float = 0,
        audio_sample_rate: int = 48000,
        seed: int | None = 42,
    ):
        super().__init__(
            dataloaders,
            tokenizer_spatial_compression_factor,
            tokenizer_temporal_compression_factor,
            patch_spatial,
            max_sequence_length,
            max_samples_per_batch,
            sound_latent_fps=sound_latent_fps,
            audio_sample_rate=audio_sample_rate,
        )
        self.seed = seed
        # Calculate probabilities for random sampling
        total_ratio = sum(self.data_ratios)
        self.data_probs = np.array([ratio / total_ratio for ratio in self.data_ratios])

    def __iter__(self):
        while True:
            if self.seed is not None:
                rng = np.random.RandomState(self.seed + self.global_id)
                index_id = rng.choice(len(self.dataloader_list), p=self.data_probs)
            else:
                data_id = self.global_id % self.ratio_sum
                index_id = self._get_dataloader_index(data_id)

            metrics = _PackingMetrics()
            output_batch = dict()
            skipped_samples = deque()
            lookahead_limit = 10
            lookahead_count = 0

            while True:
                # Check max samples limit first
                if self.max_samples_per_batch is not None and metrics.num_samples >= self.max_samples_per_batch:
                    break

                # If we have started packing and tried lookahead_limit times to find a fitting sample but failed, stop.
                if len(output_batch) > 0 and lookahead_count >= lookahead_limit:
                    break

                had_buffer = len(self.buffers[index_id]) > 0
                try:
                    output = self._get_next_sample(index_id)
                except StopIteration:
                    break  # No more data in this dataloader

                if had_buffer:
                    metrics.from_buffer += 1
                else:
                    metrics.from_workers += 1

                num_tokens_in_current_sample = self._compute_num_tokens_per_sample(output)

                if (
                    self.max_sequence_length is not None
                    and metrics.current_sequence_length + num_tokens_in_current_sample >= self.max_sequence_length
                ):
                    if len(output_batch) == 0:
                        # This case happens when current_sequence_length = 0 and num_tokens_in_current_sample > self.max_sequence_length
                        # In this case, we should simply discard the current sample and get the next sample.
                        log.info(
                            f"Discarding oversized sample with {num_tokens_in_current_sample} tokens. Max sequence length: {self.max_sequence_length}",
                            rank0_only=False,
                        )
                        metrics.dropped_count += 1
                        continue

                    # current_sequence_length > 0 and selected sample is too large to fit in the remaining space.
                    # Instead of stopping immediately (creating large padding), we buffer this large sample
                    # and try to find a smaller one that fits in the remaining space.
                    skipped_samples.append(output)
                    lookahead_count += 1
                    continue

                metrics.current_sequence_length += num_tokens_in_current_sample
                metrics.num_samples += 1
                output["dataset_name"] = self.dataset_name_list[index_id]
                self._update_output_batch(output_batch, output)

            # Add back skipped samples to the buffer for the next batch.
            # appendleft puts item at HEAD. So we insert S3, then S2, then S1.
            for sample in reversed(skipped_samples):
                self.buffers[index_id].appendleft(sample)

            if len(output_batch) == 0:
                return

            metrics.attach_to(output_batch, buffer_size=len(self.buffers[index_id]))
            self.global_id += 1
            yield output_batch

    def _get_dataloader_index(self, data_id):
        """Maps global id to the corresponding dataloader index based on ratio."""
        for i, r in enumerate(self.data_ratios):
            if data_id < r:
                return i
            data_id -= r
        raise ValueError("Invalid data_id")


class RankPartitionedDataLoader:
    """Assigns each rank to exactly one dataset based on ratios.

    For N GPUs with datasets having ratios r_1:r_2:...:r_k, the first
    N * r_1 / sum(r) ranks are assigned dataset 1, the next N * r_2 / sum(r)
    ranks are assigned dataset 2, etc.  Each rank instantiates a single
    PyTorch DataLoader for its assigned dataset.

    The sharding information (``shard_world_size`` and ``shard_rank``) is set
    on each dataset so that it shards data only across ranks that share the
    same dataset, rather than across the full world.

    Example:
        With 128 GPUs and datasets ``{"video": {"dataset": ..., "ratio": 3},
        "image": {"dataset": ..., "ratio": 1}}``:

        - Ranks   0-95  -> video  (shard_world_size=96, shard_rank=0..95)
        - Ranks  96-127 -> image  (shard_world_size=32, shard_rank=0..31)
    """

    def __init__(
        self,
        datasets: dict[str, dict[str, Any]],
        **dataloader_kwargs: Any,
    ):
        """
        Args:
            datasets: Mapping of dataset name to config dict with keys:

                - ``"dataset"`` (required): a lazy config or dataset instance.
                - ``"ratio"`` (required): positive int weight.
                - ``"dataloader_kwargs"`` (optional): dict of keyword arguments
                  that override the top-level ``**dataloader_kwargs`` for this
                  dataset only (e.g. different ``num_workers`` or ``batch_size``).

            **dataloader_kwargs: Default kwargs forwarded to
                ``torch.utils.data.DataLoader``. ``collate_fn`` defaults to
                ``custom_collate_fn`` if not given.
        """
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        log.info(f"RankPartitionedDataLoader: world_size: {world_size} and rank: {rank}", rank0_only=False)

        _VALID_KEYS = {"dataset", "ratio", "dataloader_kwargs"}
        names: list[str] = []
        dataset_configs: list[Any] = []
        ratios: list[int] = []
        per_dataset_kwargs: list[dict[str, Any]] = []
        for name, cfg in datasets.items():
            extra = set(cfg.keys()) - _VALID_KEYS
            assert not extra, f"Dataset {name!r}: unexpected keys {extra}. Allowed: {_VALID_KEYS}"
            if cfg["ratio"] <= 0:
                log.warning(
                    f"RankPartitionedDataLoader: Skipping dataset {name} with ratio {cfg['ratio']}", rank0_only=False
                )
                continue
            names.append(name)
            dataset_configs.append(cfg["dataset"])
            ratios.append(cfg["ratio"])
            per_dataset_kwargs.append(cfg.get("dataloader_kwargs", {}))

        assert len(names) > 0, "No datasets with positive ratios provided."
        assert world_size >= len(names), (
            f"world_size ({world_size}) must be >= number of datasets ({len(names)}) "
            f"so each dataset gets at least one rank."
        )

        total_ratio = sum(ratios)
        ideal = [r / total_ratio * world_size for r in ratios]
        allocations = [max(1, int(q)) for q in ideal]
        remaining = world_size - sum(allocations)
        if remaining > 0:
            remainders = sorted(range(len(ratios)), key=lambda i: ideal[i] - allocations[i], reverse=True)
            for j in range(remaining):
                allocations[remainders[j]] += 1
        elif remaining < 0:
            deficit = -remaining
            while deficit > 0:
                best = max(
                    (i for i in range(len(allocations)) if allocations[i] > 1),
                    key=lambda i: (allocations[i] - ideal[i], allocations[i]),
                )
                allocations[best] -= 1
                deficit -= 1

        expected_ratios = [r / total_ratio for r in ratios]
        actual_ratios = [a / world_size for a in allocations]
        lines = [f"RankPartitionedDataLoader allocation ({world_size} GPUs):"]
        start = 0
        for i, (name, alloc) in enumerate(zip(names, allocations)):
            end = start + alloc - 1
            lines.append(
                f"  {name} (ratio {ratios[i]}): ranks {start}-{end} ({alloc} GPUs) "
                f"| expected {expected_ratios[i]:.2%}, actual {actual_ratios[i]:.2%}"
            )
            start += alloc
        log.info("\n".join(lines), rank0_only=False)

        cumulative = 0
        my_dataset_idx = -1
        for i, alloc in enumerate(allocations):
            if rank < cumulative + alloc:
                my_dataset_idx = i
                break
            cumulative += alloc
        assert my_dataset_idx >= 0

        shard_rank = rank - cumulative
        shard_world_size = allocations[my_dataset_idx]

        dataset: Any = instantiate(dataset_configs[my_dataset_idx])
        dataset.shard_world_size = shard_world_size
        dataset.shard_rank = shard_rank
        dataset.shard_id = my_dataset_idx

        merged_kwargs = {**dataloader_kwargs, **per_dataset_kwargs[my_dataset_idx]}
        merged_kwargs.setdefault("collate_fn", custom_collate_fn)
        self.dataloader = torch.utils.data.DataLoader(dataset, **merged_kwargs)
        self.dataset_name = names[my_dataset_idx]
        self.dataset = dataset

    def __iter__(self):
        return iter(self.dataloader)

    def __len__(self) -> int:
        return len(self.dataloader)


class PackingDataLoader(JointDataLoader):
    """Packs multiple samples from a single dataloader into token-budget-constrained batches.

    Unlike the other ``JointDataLoader`` subclasses which manage multiple
    dataloaders with configurable ratios, this class wraps a single dataloader
    and greedily packs consecutive samples until the token budget
    (``max_sequence_length``) or sample count limit (``max_samples_per_batch``)
    is reached.
    """

    def __init__(
        self,
        dataloader: torch.utils.data.DataLoader | webdataset.WebLoader,
        tokenizer_spatial_compression_factor: int,
        tokenizer_temporal_compression_factor: int,
        patch_spatial: int,
        max_sequence_length: int | None = None,
        max_samples_per_batch: int | None = None,
        sound_latent_fps: float = 0,
        audio_sample_rate: int = 48000,
        dataset_name: str = "default",
    ):
        """
        Args:
            dataloader: A single dataloader (or lazy config) to draw samples from.
            tokenizer_spatial_compression_factor: Spatial compression factor of the tokenizer.
            tokenizer_temporal_compression_factor: Temporal compression factor of the tokenizer.
            patch_spatial: Spatial patchification factor.
            max_sequence_length: Max total tokens per packed batch. Mutually exclusive with
                ``max_samples_per_batch``.
            max_samples_per_batch: Max number of samples per packed batch. Mutually exclusive
                with ``max_sequence_length``.
            sound_latent_fps: Sound tokenizer latent rate in Hz. If 0, sound tokens are not counted.
            audio_sample_rate: Audio sample rate in Hz.
            dataset_name: Name tag attached to every sample in the output batch.
        """
        wrapped = {dataset_name: {"dataloader": dataloader, "ratio": 1}}
        super().__init__(
            dataloaders=wrapped,
            tokenizer_spatial_compression_factor=tokenizer_spatial_compression_factor,
            tokenizer_temporal_compression_factor=tokenizer_temporal_compression_factor,
            patch_spatial=patch_spatial,
            max_sequence_length=max_sequence_length,
            max_samples_per_batch=max_samples_per_batch,
            sound_latent_fps=sound_latent_fps,
            audio_sample_rate=audio_sample_rate,
        )

    def __iter__(self):
        inner = self.dataloader_list[0]
        ds_name = getattr(inner, "dataset_name", self.dataset_name_list[0])

        while True:
            current_sequence_length = 0
            num_samples = 0
            output_batch: dict = {}

            skipped_samples: deque = deque()
            lookahead_limit = 10
            lookahead_count = 0

            while True:
                if self.max_samples_per_batch is not None and num_samples >= self.max_samples_per_batch:
                    break

                if len(output_batch) > 0 and lookahead_count >= lookahead_limit:
                    break

                try:
                    output = self._get_next_sample(0)
                except StopIteration:
                    break

                num_tokens_in_current_sample = self._compute_num_tokens_per_sample(output)

                if (
                    self.max_sequence_length is not None
                    and current_sequence_length + num_tokens_in_current_sample >= self.max_sequence_length
                ):
                    if len(output_batch) == 0:
                        # This case happens when current_sequence_length = 0 and num_tokens_in_current_sample > self.max_sequence_length
                        # In this case, we should simply discard the current sample and get the next sample.
                        log.error(
                            f"PackingDataLoader: Discarding oversized sample with {num_tokens_in_current_sample} tokens. Max sequence length: {self.max_sequence_length}",
                            rank0_only=False,
                        )
                        continue

                    skipped_samples.append(output)
                    lookahead_count += 1
                    continue

                current_sequence_length += num_tokens_in_current_sample
                num_samples += 1
                output["dataset_name"] = ds_name
                self._update_output_batch(output_batch, output)

            for sample in reversed(skipped_samples):
                self.buffers[0].appendleft(sample)

            if len(output_batch) == 0:
                return

            self.global_id += 1
            yield output_batch


class RandomJointDataLoader(JointDataLoader):
    r"""
    A random joint dataloader that supports loading multiple modalities with stochastic sampling.

    In this dataloader, the modality is randomly selected at each iteration based on the
    probability distribution derived from the ratios. Each rank independently samples a
    modality, so different ranks may process different modalities at the same iteration.

    For example, with 2 modalities (image and video) and ratio 2:1:
        - Each iteration has 66.7% probability of selecting images
        - Each iteration has 33.3% probability of selecting videos
        - The selection is independent across iterations and ranks

    Note: Unlike IterativeJointDataLoader, this does not guarantee synchronized modality
    selection across ranks.
    """

    def __init__(
        self,
        dataloaders: Dict[str, Dict[str, Union[torch.utils.data.DataLoader, webdataset.WebLoader, int]]],
        tokenizer_spatial_compression_factor: int,
        tokenizer_temporal_compression_factor: int,
        patch_spatial: int,
        max_sequence_length: int | None = None,
        max_samples_per_batch: int | None = None,
        sound_latent_fps: float = 0,
        audio_sample_rate: int = 48000,
    ):
        super().__init__(
            dataloaders,
            tokenizer_spatial_compression_factor,
            tokenizer_temporal_compression_factor,
            patch_spatial,
            max_sequence_length,
            max_samples_per_batch,
            sound_latent_fps=sound_latent_fps,
            audio_sample_rate=audio_sample_rate,
        )

        # Convert data ratios to probabilities
        self.data_ratios = np.array([ratio / sum(self.data_ratios) for ratio in self.data_ratios])

    def __iter__(self):
        while True:
            index_id = np.random.choice(len(self.dataloader_list), p=self.data_ratios)

            metrics = _PackingMetrics()
            output_batch = dict()
            skipped_samples = deque()
            lookahead_limit = 10
            lookahead_count = 0

            while True:
                # Check max samples limit first
                if self.max_samples_per_batch is not None and metrics.num_samples >= self.max_samples_per_batch:
                    break

                # If we have started packing and tried lookahead_limit times to find a fitting sample but failed, stop.
                if len(output_batch) > 0 and lookahead_count >= lookahead_limit:
                    break

                had_buffer = len(self.buffers[index_id]) > 0
                try:
                    output = self._get_next_sample(index_id)
                except StopIteration:
                    break  # No more data in this dataloader

                if had_buffer:
                    metrics.from_buffer += 1
                else:
                    metrics.from_workers += 1

                num_tokens_in_current_sample = self._compute_num_tokens_per_sample(output)

                if (
                    self.max_sequence_length is not None
                    and metrics.current_sequence_length + num_tokens_in_current_sample >= self.max_sequence_length
                ):
                    if len(output_batch) == 0:
                        # This case happens when current_sequence_length = 0 and num_tokens_in_current_sample > self.max_sequence_length
                        # In this case, we should simply discard the current sample and get the next sample.
                        log.info(
                            f"Discarding oversized sample with {num_tokens_in_current_sample} tokens. Max sequence length: {self.max_sequence_length}",
                            rank0_only=False,
                        )
                        metrics.dropped_count += 1
                        continue

                    # current_sequence_length > 0 and selected sample is too large to fit in the remaining space.
                    # Instead of stopping immediately (creating large padding), we buffer this large sample
                    # and try to find a smaller one that fits in the remaining space.
                    skipped_samples.append(output)
                    lookahead_count += 1
                    continue

                metrics.current_sequence_length += num_tokens_in_current_sample
                metrics.num_samples += 1
                output["dataset_name"] = self.dataset_name_list[index_id]
                self._update_output_batch(output_batch, output)

            # Add back skipped samples to the buffer for the next batch.
            # appendleft puts item at HEAD. So we insert S3, then S2, then S1.
            for sample in reversed(skipped_samples):
                self.buffers[index_id].appendleft(sample)

            if len(output_batch) == 0:
                return

            metrics.attach_to(output_batch, buffer_size=len(self.buffers[index_id]))
            yield output_batch
