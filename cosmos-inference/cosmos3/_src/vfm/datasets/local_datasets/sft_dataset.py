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

# SFT dataset loader — reads video metadata + captions from a JSONL file on S3.
import gzip
import hashlib
import io
import json
import os
import random
import tempfile
from pathlib import Path
from typing import Any, Optional

import boto3
import numpy as np
import torch

from cosmos3._src.imaginaire.flags import INTERNAL
from cosmos3._src.imaginaire.lazy_config import instantiate as lazy_instantiate
from cosmos3._src.imaginaire.utils import log
from cosmos3._src.vfm.datasets.local_datasets.helper import (
    client_config,
    download_from_s3,
    ffmpeg_decode_video,
    get_aspect_ratio,
    get_video_metadata,
    parse_s3_url,
)
from cosmos3._src.vfm.datasets.sequence_packing import SequencePlan, add_special_tokens
from cosmos3._src.vfm.datasets.utils import VIDEO_RES_SIZE_INFO
from cosmos3._src.vfm.models.vlm.qwen3_vl.utils import tokenize_caption

_MAX_NUM_TOKENS = 1024
_DURATION_TEMPLATE = "The video is {duration:.1f} seconds long and is of {fps:.0f} FPS."
_RESOLUTION_TEMPLATE = "This video is of {height}x{width} resolution."

# Caption types available in the SFT JSONL.
# Format: {model}_{style}
#   model: qwen3_235b | qwen3_32b | qwen3p5_397b
#   style: short | temporal | descriptive | dense
CAPTION_TYPES_AND_WEIGHTS: dict[str, float] = {
    # short: 10% total
    "qwen3_235b_short": 0.1,
    "qwen3_32b_short": 0.1,
    "qwen3p5_397b_short": 0.1,
    # descriptive: 20% total
    "qwen3_235b_descriptive": 0.2,
    "qwen3_32b_descriptive": 0.2,
    "qwen3p5_397b_descriptive": 0.2,
    # dense: 70% total
    "qwen3_235b_dense": 0.7,
    "qwen3_32b_dense": 0.7,
    "qwen3p5_397b_dense": 0.7,
    # temporal: 0% total
    "qwen3_235b_temporal": 0.0,
    "qwen3_32b_temporal": 0.0,
    "qwen3p5_397b_temporal": 0.0,
}
CAPTION_TYPES = list(CAPTION_TYPES_AND_WEIGHTS.keys())
CAPTION_WEIGHTS = list(CAPTION_TYPES_AND_WEIGHTS.values())


class SFTDataset(torch.utils.data.IterableDataset):
    """Dataset for loading SFT video clips with captions from JSONL metadata on S3."""

    def __init__(
        self,
        metadata: list[dict],
        num_video_frames: int,
        resolution: str,
        s3_credentials: dict,
        temporal_interval_mode: str = "entire_chunk",
        frame_selection_mode: str = "center",
        tokenizer_config: Optional[Any] = None,
        cfg_dropout_rate: float = 0.0,
        use_system_prompt: bool = False,
        append_duration_fps_timestamps: bool = True,
        append_resolution_info: bool = True,
        cfg_dropout_keep_metadata: bool = False,
        caption_suffix: str = "",
        conditioning_fps: float = 24,
        conditioning_fps_noise_std: float = 0.0,
        conditioning_config: dict[int, float] | None = None,
        temporal_compression_factor: int = 4,
    ):
        assert temporal_interval_mode in ("force_one", "max_30fps", "entire_chunk"), (
            f"Unknown temporal_interval_mode={temporal_interval_mode!r}"
        )
        assert frame_selection_mode in ("center", "first", "random"), (
            f"Unknown frame_selection_mode={frame_selection_mode!r}"
        )
        assert temporal_compression_factor >= 1, "temporal_compression_factor must be >= 1"
        self.metadata = metadata
        self.num_video_frames = num_video_frames
        self.resolution = resolution
        self.s3_credentials = s3_credentials
        self.temporal_interval_mode = temporal_interval_mode
        self.frame_selection_mode = frame_selection_mode
        self.tokenizer_config = tokenizer_config
        self.cfg_dropout_rate = cfg_dropout_rate
        self.use_system_prompt = use_system_prompt
        self.append_duration_fps_timestamps = append_duration_fps_timestamps
        self.append_resolution_info = append_resolution_info
        self.cfg_dropout_keep_metadata = cfg_dropout_keep_metadata
        self.caption_suffix = caption_suffix.strip()
        self.conditioning_fps = conditioning_fps
        self.conditioning_fps_noise_std = conditioning_fps_noise_std

        self.temporal_compression_factor = temporal_compression_factor
        self.conditioning_config: dict[int, float] | None = None
        if conditioning_config is not None:
            total_prob = sum(conditioning_config.values())
            assert total_prob > 0, "conditioning_config probabilities must sum to a positive number"
            self.conditioning_config = {k: v / total_prob for k, v in conditioning_config.items()}
            log.info(f"Conditioning config: {self.conditioning_config}")
        # They will be set by the RankPartitionedDataLoader
        self.shard_world_size = None
        self.shard_rank = None
        self.shard_id = 0
        self.is_initialized = False
        self.output_sizes = VIDEO_RES_SIZE_INFO[resolution]

        _vlm_proc = lazy_instantiate(self.tokenizer_config)
        self.vlm_tokenizer = _vlm_proc.tokenizer
        self.vlm_tokenizer, _ = add_special_tokens(self.vlm_tokenizer)

    def __len__(self):
        return len(self.metadata)

    def _tokenize_caption(self, caption: str) -> tuple[list[int], str]:
        text_ids = tokenize_caption(
            caption,
            self.vlm_tokenizer,
            is_video=True,
            use_system_prompt=self.use_system_prompt,
        )
        if len(text_ids) > _MAX_NUM_TOKENS:
            log.warning(f"Text ids are too long, truncating: {len(text_ids)} > {_MAX_NUM_TOKENS}")
        text_ids = text_ids[:_MAX_NUM_TOKENS]
        return text_ids, caption

    def process_one_sample(self, metadata: dict) -> dict | None:
        """Process a single SFT sample: download, decode, and prepare for training.

        A random t2w_window is picked from the video's list of windows each time.
        """
        windows = metadata["t2w_windows"]
        win_idx = random.randrange(len(windows))
        t2w_window = windows[win_idx]
        window_start = t2w_window["start_frame"]
        window_end = t2w_window["end_frame"]

        # Compute output resolution
        input_w, input_h = metadata["width"], metadata["height"]
        target_w, target_h = self.output_sizes[metadata["aspect_ratio"]]
        resize_ratio = max(target_w / input_w, target_h / input_h)
        resize_h, resize_w = (round(input_h * resize_ratio), round(input_w * resize_ratio))
        crop_y, crop_x = (round((resize_h - target_h) / 2), round((resize_w - target_w) / 2))

        video_bytes = download_from_s3(self.s3_client, metadata["vision_path"])
        if video_bytes is None:
            log.warning(f"Failed to download video from S3: {metadata['vision_path']}")
            return None

        # Decode all frames to (T, H, W, 3)
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as tmp_input:
            tmp_input.write(video_bytes)
            tmp_input.flush()
            input_video_path = tmp_input.name
            video_info = get_video_metadata(input_video_path)
            original_fps = video_info["fps"]
            total_frames = video_info["total_frames"]

            # Constrain to the t2w window
            actual_end = min(window_end, total_frames - 1)
            frames_in_window = actual_end - window_start + 1

            if self.num_video_frames == -1:
                # Native chunk mode: use start/end/interval directly from the window
                temporal_interval = t2w_window["temporal_interval"]
                start_frame = window_start
                end_frame = actual_end
            else:
                if frames_in_window < self.num_video_frames:
                    log.warning(
                        f"Not enough frames in window: {metadata['uuid']}, "
                        f"frames_in_window: {frames_in_window}, required: {self.num_video_frames}"
                    )
                    return None

                # Compute temporal interval
                if self.temporal_interval_mode == "force_one":
                    temporal_interval = 1
                elif self.temporal_interval_mode == "max_30fps":
                    temporal_interval = max(1, int(original_fps / 30.0))
                elif self.temporal_interval_mode == "entire_chunk":
                    temporal_interval = frames_in_window // self.num_video_frames
                    temporal_interval = max(1, temporal_interval)
                else:
                    raise ValueError(f"Unknown temporal_interval_mode: {self.temporal_interval_mode}")

                num_frames_before_downsample = (self.num_video_frames - 1) * temporal_interval + 1
                if self.frame_selection_mode == "first":
                    start_frame = window_start
                elif self.frame_selection_mode == "center":
                    start_frame = window_start + (frames_in_window - num_frames_before_downsample) // 2
                elif self.frame_selection_mode == "random":
                    max_offset = frames_in_window - num_frames_before_downsample
                    start_frame = window_start + random.randint(0, max(0, max_offset))
                else:
                    raise ValueError(f"Unknown frame_selection_mode: {self.frame_selection_mode}")
                end_frame = start_frame + num_frames_before_downsample - 1

            fps = original_fps / temporal_interval

            video_chunk = []
            for idx, frame in enumerate(
                ffmpeg_decode_video(input_video_path, scale_hw=(resize_h, resize_w), num_threads=2)
            ):
                if idx < start_frame:
                    continue
                elif idx <= end_frame:
                    if (idx - start_frame) % temporal_interval == 0:
                        video_chunk.append(frame)
                else:
                    break

        if not video_chunk:
            log.warning(
                f"No frames decoded for sample: {metadata['uuid']} "
                f"(start={start_frame}, end={end_frame}, path={metadata['vision_path']})"
            )
            return None

        video_chunk = np.stack(video_chunk, axis=0)  # [T,H,W,3]

        # Truncate temporally to temporal_compression_factor * N + 1
        target_t = (video_chunk.shape[0] - 1) // self.temporal_compression_factor * self.temporal_compression_factor + 1

        # Apply spatial center crop and temporal truncation
        video_chunk = video_chunk[:target_t, crop_y : crop_y + target_h, crop_x : crop_x + target_w]  # [T,H,W,3]

        # THWC -> CTHW
        video_chunk = np.transpose(video_chunk, (3, 0, 1, 2))  # [3,T,H,W]
        video = torch.from_numpy(np.ascontiguousarray(video_chunk)).to(torch.uint8)  # [3,T,H,W]
        padding_mask = torch.zeros((1, target_h, target_w), dtype=torch.float32)
        # image_size: [target_h, target_w, orig_h, orig_w] in pixel space, for the model to crop the video
        image_size = torch.tensor([target_h, target_w, target_h, target_w], dtype=torch.float32)

        available_types = [ct for ct in CAPTION_TYPES if ct in t2w_window]
        if "qwen3_32b_rewrite-dense" in t2w_window:
            caption_key = "qwen3_32b_rewrite-dense"
        elif "caption" in t2w_window:
            caption_key = "caption"
        elif available_types:
            available_weights = [CAPTION_TYPES_AND_WEIGHTS[ct] for ct in available_types]
            caption_key = random.choices(available_types, weights=available_weights, k=1)[0]
        else:
            log.warning(
                f"No known caption key found in t2w_window for sample {metadata['uuid']}. "
                f"Keys: {list(t2w_window)}. Skipping sample."
            )
            return None
        caption = t2w_window[caption_key]
        caption = caption.strip().rstrip(".") + "."

        num_decoded_frames = video.shape[1]
        cond_fps = fps if self.conditioning_fps < 0 else self.conditioning_fps
        if self.conditioning_fps_noise_std > 0:
            noise_factor = np.exp(np.random.randn() * self.conditioning_fps_noise_std)
            cond_fps = cond_fps * noise_factor

        if self.caption_suffix:
            caption = (caption + " " + self.caption_suffix).strip()

        # CFG dropout: when cfg_dropout_keep_metadata is True, dropout fires
        # before appending resolution/duration/FPS so that metadata text is
        # preserved even under unconditional guidance.
        if self.cfg_dropout_keep_metadata and self.cfg_dropout_rate > 0:
            if random.random() < self.cfg_dropout_rate:
                caption = ""

        if self.append_duration_fps_timestamps:
            duration = num_decoded_frames / cond_fps
            suffix = _DURATION_TEMPLATE.format(duration=duration, fps=cond_fps)
            caption = caption + " " + suffix
        if self.append_resolution_info:
            suffix = _RESOLUTION_TEMPLATE.format(height=target_h, width=target_w)
            caption = caption + " " + suffix
        caption = caption.strip()

        if not self.cfg_dropout_keep_metadata and self.cfg_dropout_rate > 0:
            if random.random() < self.cfg_dropout_rate:
                caption = ""
        text_ids, caption = self._tokenize_caption(caption)

        ret = dict(
            __key__=f"{metadata['uuid']}_w{win_idx}",
            __url__=metadata["vision_path"],
            fps=original_fps,
            n_orig_video_frames=total_frames,
            chunk_index=win_idx,
            frame_start=start_frame,
            frame_end=end_frame,
            num_frames=video.shape[1],
            video=video,
            num_multiplier=temporal_interval,
            conditioning_fps=cond_fps,
            padding_mask=padding_mask,
            image_size=image_size,
            ai_caption=caption,
            sampled_caption_style=caption_key,
            text_token_ids=torch.tensor(text_ids),
        )

        if self.conditioning_config is not None:
            num_frames_pixel = video.shape[1]
            t_latent = 1 + (num_frames_pixel - 1) // self.temporal_compression_factor
            frames_options = list(self.conditioning_config.keys())
            weights = list(self.conditioning_config.values())
            num_cond = random.choices(frames_options, weights=weights, k=1)[0]
            num_cond = min(num_cond, t_latent - 1)
            ret["sequence_plan"] = SequencePlan(
                has_text=True,
                has_vision=True,
                condition_frame_indexes_vision=list(range(num_cond)),
            )

        return ret

    def __iter__(self):
        assert not self.is_initialized, "Dataset can only be initialized once."
        assert len(self.metadata) > 0, "Did not find any data."

        self.s3_client = boto3.client(
            "s3",
            **self.s3_credentials,
            config=client_config,
        )
        # Ranks of the same pp/tp/cp group will have the same dp rank and thus share the same group id.
        # zhao: Cosmos3 does not support TP/SP/CP
        if self.shard_world_size is not None:
            train_world_size = self.shard_world_size
            train_rank = self.shard_rank
            log.info(f"Using shard_world_size: {train_world_size} and shard_rank: {train_rank}", rank0_only=False)
        else:
            train_world_size = torch.distributed.get_world_size()
            train_rank = torch.distributed.get_rank()
        train_dp_rank = train_rank
        train_num_dp_groups = train_world_size
        train_dp_group_size = 1

        # Get data worker rank. Each trainer have multiple dataloaders
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            worker_rank = worker_info.id
            total_data_ranks = worker_info.num_workers * train_num_dp_groups
            data_rank = worker_rank + train_dp_rank * worker_info.num_workers
            seed = worker_info.seed
        else:
            log.warning("No data worker info found. Using default worker rank and number of workers.", rank0_only=False)
            total_data_ranks = train_num_dp_groups
            data_rank = train_dp_rank
            seed = 42

        log.info(
            f"train_world_size: {train_world_size}; "
            f"train_rank: {train_rank}; "
            f"train_dp_rank: {train_dp_rank}; "
            f"train_num_dp_groups: {train_num_dp_groups}; "
            f"train_dp_group_size: {train_dp_group_size}; "
            f"worker_info: {worker_info}; "
            f"total_data_ranks: {total_data_ranks}; "
            f"data_rank: {data_rank}; "
            f"seed: {seed}"
            f"shard_id: {self.shard_id}; "
            f"shard_world_size: {self.shard_world_size}; "
            f"shard_rank: {self.shard_rank}",
            rank0_only=False,
        )

        # Make sure len(self.metadata) is divisible by self.num_groups
        multiplier = max(1, total_data_ranks * 50 // len(self.metadata))
        log.info(f"Dataset multiplier: {multiplier}", rank0_only=False)
        self.metadata = self.metadata * multiplier  # reduce bias caused by sharding
        num_pad = total_data_ranks - len(self.metadata) % total_data_ranks
        self.metadata = self.metadata + self.metadata[:num_pad]
        # Deterministic shuffle based on the sha256 hash of uuid
        # Note that the repeated samples are grouped together.
        # Split list to keep only the data for this rank
        if True:  # This gives more diversity
            random.Random(self.shard_id).shuffle(self.metadata)
            log.info(f"Shuffled metadata for shard {self.shard_id}", rank0_only=False)
            self.metadata = self.metadata[data_rank::total_data_ranks]
        else:
            # Keep the repeated samples together to aid cache hits.
            self.metadata.sort(key=lambda x: hashlib.sha256(x["vision_path"].encode("utf-8")).hexdigest())
            # Equally chunk the list (guaranteed to be divisible by total_data_ranks)
            chunk_size = len(self.metadata) // total_data_ranks
            start = data_rank * chunk_size
            end = (data_rank + 1) * chunk_size
            log.info(
                f"DRank {data_rank} has got a chunk {start}-{end} from {len(self.metadata)} data.", rank0_only=False
            )
            self.metadata = self.metadata[start:end]
        num_unique_vision_paths = len(set(metadata["vision_path"] for metadata in self.metadata))
        log.info(
            f"DRank {data_rank} has {len(self.metadata)} data with {num_unique_vision_paths} unique vision_paths.",
            rank0_only=False,
        )

        self.is_initialized = True

        # Make sure the data within a DRank is identical
        rng = random.Random(data_rank + self.shard_id * 12345)
        while True:
            rng.shuffle(self.metadata)
            for metadata in self.metadata:
                sample = self.process_one_sample(metadata)
                if sample is None:
                    log.warning(f"Failed to process sample {metadata['uuid']}, skipping...")
                    continue
                yield sample


def _flatten_metadata_by_window(metadata_list: list[dict]) -> list[dict]:
    """Expand metadata so each entry maps to exactly one t2w_window.

    Each output dict is a shallow copy of the original whose ``t2w_windows``
    list contains a single window.  The ``uuid`` is suffixed with ``_w{idx}``
    so every entry has a unique identifier.
    """
    flat: list[dict] = []
    for entry in metadata_list:
        for win_idx, window in enumerate(entry["t2w_windows"]):
            flat.append(
                {
                    **entry,
                    "uuid": f"{entry['uuid']}_w{win_idx}",
                    "t2w_windows": [window],
                }
            )
    return flat


def _load_sft_metadata_from_s3(
    s3_client,
    jsonl_url: str,
    min_frames: int,
    uuid_prefix: str = "",
    min_short_edge: int = 0,
) -> list[dict]:
    """Load SFT metadata from a single JSONL file on S3.

    Returns one entry per video.  Each entry keeps only the windows whose frame
    span is at least *min_frames*; videos with no qualifying windows are dropped.

    Args:
        s3_client: Boto3 S3 client
        jsonl_url: S3 URL to the JSONL metadata file
        min_frames: Minimum number of frames required per window
        uuid_prefix: Prefix prepended to each uuid for disambiguation when
            multiple JSONL files are loaded
        min_short_edge: Drop videos whose shortest spatial edge (min of width,
            height) is below this value.  0 disables the filter.
    """
    log.info(f"Downloading SFT metadata from {jsonl_url}", rank0_only=False)
    metadata_list: list[dict] = []
    num_raw_records = 0
    num_raw_windows = 0
    num_filtered_duration = 0
    num_filtered_windows = 0
    num_filtered_short_edge = 0

    with io.BytesIO() as buffer:
        if jsonl_url.startswith("s3://"):
            bucket, key = parse_s3_url(jsonl_url)
            s3_client.download_fileobj(Bucket=bucket, Key=key, Fileobj=buffer)
        else:
            path = Path(jsonl_url).absolute()
            jsonl_url = str(path)
            buffer.write(path.read_bytes())
        buffer.seek(0)
        log.info("Finished downloading. Decoding...", rank0_only=False)

        line_iter = gzip.open(buffer, "rb") if jsonl_url.endswith(".gz") else buffer
        for line in line_iter:
            num_raw_records += 1
            record = json.loads(line.decode("utf-8"))
            uuid = f"{uuid_prefix}{record['uuid']}" if uuid_prefix else record["uuid"]
            if record["duration"] > 61.0:
                print(f"Skipping video with too long duration: {uuid}, {record['duration']} > 61.0")
                num_filtered_duration += 1
                continue
            if min_short_edge > 0 and min(record["width"], record["height"]) < min_short_edge:
                num_filtered_short_edge += 1
                continue

            windows = record.get("t2w_windows")
            if not windows:
                continue

            kept_windows = []
            for window in windows:
                num_raw_windows += 1
                frames_in_window = window["end_frame"] - window["start_frame"] + 1
                if frames_in_window < min_frames:
                    num_filtered_windows += 1
                else:
                    kept_windows.append(window)

            if not kept_windows:
                continue

            vision_path = record["vision_path"]
            if "://" not in vision_path and not vision_path.startswith("/"):
                # Relative path to the JSONL file
                vision_path = f"{os.path.dirname(jsonl_url)}/{vision_path}"

            aspect_ratio = get_aspect_ratio(record["width"], record["height"])
            metadata_list.append(
                {
                    "uuid": uuid,
                    "vision_path": vision_path,
                    "width": record["width"],
                    "height": record["height"],
                    "nb_frames": record.get("nb_frames"),
                    "framerate": record.get("framerate"),
                    "aspect_ratio": aspect_ratio,
                    "t2w_windows": kept_windows,
                }
            )

    log.info(
        f"Finished decoding SFT metadata from {jsonl_url}. "
        f"Records: {num_raw_records}, "
        f"Duration > 61s: {num_filtered_duration}, "
        f"Short edge < {min_short_edge}: {num_filtered_short_edge}, "
        f"Windows: {num_raw_windows}, Windows < {min_frames}f: {num_filtered_windows}, "
        f"Videos kept: {len(metadata_list)}"
    )
    return metadata_list


def get_sft_dataset(
    jsonl_paths: str | list[str] = "s3://nv-00-10206-vfm/cosmos3_video_sft/human_1k/captions_full.jsonl",
    resolution: str = "720",
    num_video_frames: int = 93,
    temporal_interval_mode: str = "entire_chunk",
    frame_selection_mode: str = "center",
    tokenizer_config: Optional[Any] = None,
    cfg_dropout_rate: float = 0.1,
    use_system_prompt: bool = False,
    append_duration_fps_timestamps: bool = True,
    append_resolution_info: bool = True,
    cfg_dropout_keep_metadata: bool = False,
    sample_by_window: bool = False,
    min_short_edge: int = 0,
    caption_suffix: str = "",
    conditioning_fps: float = 24,
    conditioning_fps_noise_std: float = 0.0,
    conditioning_config: dict[int, float] | None = None,
    temporal_compression_factor: int = 4,
    **kwargs,
) -> SFTDataset:
    """Create SFT video dataset from one or more JSONL files on S3.

    Args:
        jsonl_paths: S3 path(s) to JSONL metadata file(s). A single string or
            a list of strings.  When multiple files are given, their samples are
            concatenated and each file's uuids are prefixed with ``<index>/``
            to avoid collisions.
        resolution: Output resolution (e.g., "720", "480")
        num_video_frames: Number of frames to extract from each video.
            Videos with fewer frames are skipped at decode time.
            Use -1 to take native chunks from the t2w_window metadata.
        temporal_interval_mode: How to compute the temporal interval between sampled frames.
            "force_one"    — always 1 (consecutive frames at original fps).
            "max_30fps"    — smallest interval that keeps effective fps <= 30.
            "entire_chunk" — spread num_video_frames evenly across the whole window.
        frame_selection_mode: Where to select frames within the window.
            "center" — center-crop temporally (default).
            "first"  — take the first num_video_frames from the window start.
        tokenizer_config: Config for the tokenizer
        cfg_dropout_rate: Dropout rate for the caption
        use_system_prompt: Whether to use the system prompt during tokenization
        append_duration_fps_timestamps: If True, appends duration/FPS text to captions
        append_resolution_info: If True, appends resolution text to captions
        cfg_dropout_keep_metadata: If True, CFG dropout fires before appending
            duration/FPS/resolution text so that metadata is preserved during
            unconditional guidance.  If False (default), dropout fires after
            and clears the entire caption including metadata.
        sample_by_window: If True, each t2w_window is treated as a separate
            sample (the dataset length equals the total number of windows).
            If False (default), each video uuid is one sample and a random
            window is chosen on every access.
        min_short_edge: Drop videos whose shortest spatial edge (min of width,
            height) is below this value.  0 (default) disables the filter.
        caption_suffix: Text appended to every caption before the
            duration/FPS/resolution templates, e.g.
            ``"Overall, the video is of poor quality."``.  Empty string
            (default) disables the suffix.
        conditioning_fps: FPS value used for duration/FPS conditioning.
            A positive value is used directly (default 24).  A negative
            value (e.g. ``-1``) means the actual effective FPS
            (``original_fps / temporal_interval``) is used instead.
        conditioning_fps_noise_std: Standard deviation of log-normal
            multiplicative noise applied to ``conditioning_fps``.  The FPS
            is multiplied by ``exp(N(0, std))``.  0.0 (default) disables
            the noise.
        conditioning_config: Weighted distribution mapping latent-frame counts
            to unnormalized probabilities for image-to-video conditioning.
            Example: ``{0: 0.7, 1: 0.2, 2: 0.1}``.  ``None`` disables
            conditioning (all frames are generation targets).
        temporal_compression_factor: VAE temporal compression factor used to
            convert pixel frame count to latent frame count.
    Returns:
        SFTDataset instance
    """
    log.info(f"Unknown kwargs for get_sft_dataset: {kwargs}")
    assert resolution in VIDEO_RES_SIZE_INFO.keys(), "The provided resolution cannot be found in VIDEO_RES_SIZE_INFO."

    if isinstance(jsonl_paths, str):
        jsonl_paths = [jsonl_paths]

    if INTERNAL:
        with open("credentials/gcs.secret", "r") as f:
            credentials = json.load(f)
    else:
        credentials = {}

    s3_client = boto3.client("s3", **credentials)

    metadata_list: list[dict] = []
    for idx, jsonl_url in enumerate(jsonl_paths):
        prefix = f"{idx}/" if len(jsonl_paths) > 1 else ""
        metadata_list.extend(
            _load_sft_metadata_from_s3(
                s3_client,
                jsonl_url,
                min_frames=61,
                uuid_prefix=prefix,
                min_short_edge=min_short_edge,
            )
        )

    total_windows = sum(len(m["t2w_windows"]) for m in metadata_list)
    log.info(
        f"Finished loading metadata from {len(jsonl_paths)} file(s). "
        f"Total videos: {len(metadata_list)}, total windows: {total_windows}"
    )

    if sample_by_window:
        metadata_list = _flatten_metadata_by_window(metadata_list)
        log.info(f"sample_by_window=True: flattened to {len(metadata_list)} samples (one per window)")

    # Deterministic shuffle based on the sha256 hash of uuid
    metadata_list.sort(key=lambda x: hashlib.sha256(x["uuid"].encode("utf-8")).hexdigest())

    dataset = SFTDataset(
        metadata=metadata_list,
        num_video_frames=num_video_frames,
        resolution=resolution,
        s3_credentials=credentials,
        temporal_interval_mode=temporal_interval_mode,
        frame_selection_mode=frame_selection_mode,
        tokenizer_config=tokenizer_config,
        cfg_dropout_rate=cfg_dropout_rate,
        use_system_prompt=use_system_prompt,
        append_duration_fps_timestamps=append_duration_fps_timestamps,
        append_resolution_info=append_resolution_info,
        cfg_dropout_keep_metadata=cfg_dropout_keep_metadata,
        caption_suffix=caption_suffix,
        conditioning_fps=conditioning_fps,
        conditioning_fps_noise_std=conditioning_fps_noise_std,
        conditioning_config=conditioning_config,
        temporal_compression_factor=temporal_compression_factor,
    )
    return dataset
