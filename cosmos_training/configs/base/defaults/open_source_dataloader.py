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

"""
Hydra ConfigStore registration for the open-source SFT dataloader.

This mirrors the inline ``dataloader_train`` block in
``configs/experiment/mixed_modality_sft_8b.yaml`` (cosmos-inference) so users
can pick it up via the Hydra defaults group::

    defaults:
      - data_train: open_source_sft_video_256p

or as a base to override in their own experiment configs::

    L(get_open_source_sft_dataloader)(
        jsonl_paths=["/path/to/data.jsonl"],
        resolution="256",
        max_sequence_length=45056,
    )

Original YAML reference target paths use the ``cosmos3._src.vfm.*`` namespace
(the OSS-release form of ``projects.cosmos3.vfm.*``); inside this released
tree the same modules live under ``cosmos.data.vfm.*``.
"""

from hydra.core.config_store import ConfigStore

from cosmos.data.vfm.joint_dataloader import (
    PackingDataLoader,
    RankPartitionedDataLoader,
)
from cosmos.data.vfm.local_datasets.sft_dataset import get_sft_dataset
from cosmos.utils.lazy_config import LazyCall as L
from configs.base.defaults.vlm import create_qwen2_tokenizer_with_download


# ---------------------------------------------------------------------------
# Inner: SFT video dataset (matches the inline ``get_sft_dataset`` call in the
# reference YAML).
# ---------------------------------------------------------------------------

def get_sft_video_dataset(
    *,
    jsonl_paths: list[str],
    resolution: str = "256",
    pretrained_model_name: str = "Qwen/Qwen3-VL-8B-Instruct",
    tokenizer_config_variant: str = "hf",
    num_video_frames: int = -1,
    temporal_compression_factor: int = 4,
    temporal_interval_mode: str = "max_30fps",
    min_short_edge: int = 0,
    frame_selection_mode: str = "first",
    sample_by_window: bool = False,
    append_duration_fps_timestamps: bool = True,
    append_resolution_info: bool = True,
    use_system_prompt: bool = False,
    caption_suffix: str = "",
    cfg_dropout_rate: float = 0.1,
    cfg_dropout_keep_metadata: bool = False,
    conditioning_config: dict[int, float] | None = None,
    conditioning_fps: int = -1,
    conditioning_fps_noise_std: float = 0.0,
):
    """LazyCall'd version of ``get_sft_dataset`` matching the reference YAML.

    Defaults reproduce ``mixed_modality_sft_8b.yaml``: 70% T2V / 20% I2V /
    10% V2V conditioning mix at 256p with the Qwen3-VL-8B tokenizer.
    """
    if conditioning_config is None:
        # 0: T2V (text-to-video) — 70%
        # 1: I2V (first-frame conditioning) — 20%
        # 2: V2V (first 5 frames → 2 latent frames) — 10%
        conditioning_config = {0: 0.7, 1: 0.2, 2: 0.1}

    return L(get_sft_dataset)(
        jsonl_paths=jsonl_paths,
        resolution=resolution,
        num_video_frames=num_video_frames,
        temporal_compression_factor=temporal_compression_factor,
        temporal_interval_mode=temporal_interval_mode,
        min_short_edge=min_short_edge,
        frame_selection_mode=frame_selection_mode,
        sample_by_window=sample_by_window,
        append_duration_fps_timestamps=append_duration_fps_timestamps,
        append_resolution_info=append_resolution_info,
        use_system_prompt=use_system_prompt,
        caption_suffix=caption_suffix,
        cfg_dropout_rate=cfg_dropout_rate,
        cfg_dropout_keep_metadata=cfg_dropout_keep_metadata,
        conditioning_config=conditioning_config,
        conditioning_fps=conditioning_fps,
        conditioning_fps_noise_std=conditioning_fps_noise_std,
        tokenizer_config=L(create_qwen2_tokenizer_with_download)(
            pretrained_model_name=pretrained_model_name,
            config_variant=tokenizer_config_variant,
        ),
    )


# ---------------------------------------------------------------------------
# Outer: full PackingDataLoader → RankPartitionedDataLoader → SFT dataset
# pipeline. This is the registered config_store node.
# ---------------------------------------------------------------------------

def get_open_source_sft_dataloader(
    *,
    jsonl_paths: list[str] | None = None,
    resolution: str = "256",
    batch_size: int = 1,
    max_sequence_length: int = 45056,
    max_samples_per_batch: int | None = None,
    num_workers: int = 4,
    prefetch_factor: int = 4,
    audio_sample_rate: int = 48000,
    patch_spatial: int = 2,
    tokenizer_spatial_compression_factor: int = 16,
    tokenizer_temporal_compression_factor: int = 4,
    sound_latent_fps: int = 0,
    dataset_name: str = "default",
    video_stream_ratio: float = 1.0,
):
    """Build the full open-source SFT dataloader (PackingDataLoader at top).

    ``jsonl_paths`` defaults to a Hydra-MISSING marker (``"???"``) so the
    user MUST override it at experiment time::

        ... dataloader_train.dataloader.datasets.video.dataset.jsonl_paths='[".../data.jsonl"]'
    """
    if jsonl_paths is None:
        # Hydra/OmegaConf "mandatory" sentinel — must be overridden.
        jsonl_paths = "???"  # type: ignore[assignment]

    return L(PackingDataLoader)(
        dataloader=L(RankPartitionedDataLoader)(
            batch_size=batch_size,
            datasets=dict(
                video=dict(
                    dataset=get_sft_video_dataset(
                        jsonl_paths=jsonl_paths,
                        resolution=resolution,
                    ),
                    ratio=video_stream_ratio,
                ),
            ),
            in_order=True,
            num_workers=num_workers,
            persistent_workers=True,
            pin_memory=True,
            prefetch_factor=prefetch_factor,
            sampler=None,
        ),
        audio_sample_rate=audio_sample_rate,
        dataset_name=dataset_name,
        max_samples_per_batch=max_samples_per_batch,
        max_sequence_length=max_sequence_length,
        patch_spatial=patch_spatial,
        sound_latent_fps=sound_latent_fps,
        tokenizer_spatial_compression_factor=tokenizer_spatial_compression_factor,
        tokenizer_temporal_compression_factor=tokenizer_temporal_compression_factor,
    )


# ---------------------------------------------------------------------------
# ConfigStore registration.
# ---------------------------------------------------------------------------

def register_open_source_dataloaders() -> None:
    """Register named dataloader configs under the ``data_train`` Hydra group.

    Pick them via experiment config::

        defaults:
          - data_train: open_source_sft_video_256p
          - data_train: open_source_sft_video_480p
          - data_train: open_source_sft_video_720p
    """
    cs = ConfigStore.instance()

    for res_str, max_seq in [
        ("256", 45056),
        ("480", 45056),
        ("720", 45056),
    ]:
        cs.store(
            group="data_train",
            package="dataloader_train",
            name=f"open_source_sft_video_{res_str}p",
            node=get_open_source_sft_dataloader(
                resolution=res_str,
                max_sequence_length=max_seq,
            ),
        )


# Auto-register on import.
register_open_source_dataloaders()
