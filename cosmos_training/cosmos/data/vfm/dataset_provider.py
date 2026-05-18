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

from typing import Callable, Optional

import omegaconf
from webdataset.handlers import warn_and_continue

import imaginaire.datasets.webdataset.decoders.image as image_decoders
import imaginaire.datasets.webdataset.decoders.pickle as pickle_decoders
import imaginaire.datasets.webdataset.distributors as distributors
import cosmos.data.vfm.decoders.video_decoder as video_decoder
import cosmos.data.vfm.webdataset as webdataset
from imaginaire.datasets.webdataset.config.schema import DatasetConfig
from cosmos.utils.lazy_config import LazyCall as L
from cosmos.utils.lazy_config import LazyDict
from cosmos.data.vfm.augmentor_provider import AUGMENTOR_OPTIONS
from cosmos.data.vfm.augmentors import sequence_plan as _sequence_plan
from cosmos.data.vfm.data_sources.data_registration import DATASET_OPTIONS
from cosmos.data.vfm.utils import IMAGE_RES_SIZE_INFO, VIDEO_RES_SIZE_INFO


def get_video_dataset(
    dataset_name: str,
    video_decoder_name: str,
    resolution: str,
    is_train: bool = True,
    num_video_frames: int = 121,
    chunk_size: int = 0,
    min_fps_thres: int = 10,
    max_fps_thres: int = 60,
    dataset_resolution_type: str = "all",
    augmentor_name: str = "video_basic_augmentor_v1",
    object_store: Optional[str] = "s3",
    caption_type: str = "t2w_qwen2p5_7b",
    embedding_type: str = "t5_xxl",
    detshuffle: bool = False,
    long_caption_ratio: int = 7,
    medium_caption_ratio: int = 2,
    short_caption_ratio: int = 1,
    user_caption_ratio: int = 90,
    dataset_info_fn: Optional[Callable] = None,
    use_native_fps: bool = True,
    use_original_fps: bool = False,
    tokenizer_config: Optional[LazyDict] = None,
    cfg_dropout_rate: float = 0.0,
    caption_config: dict | None = None,
    append_duration_fps_timestamps: bool = True,
    append_resolution_info: bool = True,
    use_dynamic_fps: bool = False,
    low_fps_bias: float = 0.5,
    min_frames: int | None = None,
    max_frames: int | None = None,
    resize_on_read: bool = False,
    conditioning_config: Optional[dict[int, float]] = None,
    uniform_conditioning: bool = False,
    temporal_compression_factor: int = 4,
    sound_generation_mode: str = "t2vs",
    audio_sample_rate: int = 48000,
    key_renames: dict[str, str] | None = None,
) -> omegaconf.dictconfig.DictConfig:
    assert resolution in VIDEO_RES_SIZE_INFO.keys(), "The provided resolution cannot be found in VIDEO_RES_SIZE_INFO."
    assert object_store in [
        "s3",
        "swiftstack",
        "gcp",
        "pdx_cosmos_gen",
        "team_sil_gws_data",
        False,
    ], "We support s3, swiftstack, gcp, pdx_cosmos_gen, team_sil_gws_data or False for local loading."
    basic_augmentor_names = [
        "video_basic_augmentor_v2",
        "video_basic_augmentor_v2_with_control",
        "video_basic_augmentor_v2_with_tokenization",
        "noframedrop_nocameramove_video_augmentor_v1",
        "video_basic_augmentor_v3",
        "video_basic_augmentor_v3_with_audio",
    ]
    if video_decoder_name == "video_naive_bytes":
        assert augmentor_name in basic_augmentor_names, (
            "We can only use video_basic_augmentor_v2 with video_naive_bytes decoder."
        )
    if augmentor_name in basic_augmentor_names:
        assert video_decoder_name == "video_naive_bytes", (
            "We can only use video_naive_bytes decoder with video_basic_augmentor_v2."
        )

    assert dataset_resolution_type in [
        "all",
        "gt480p",
        "gt720p",
        "gt1080p",
    ], f"The provided dataset resolution type {dataset_resolution_type} is not supported."
    # dataset_resolution_type
    # -- all - uses all dataset resolutions
    # -- gt720p - Uses only resolutions >= 720p
    # -- gt1080p - Uses only resolutions >= 1080p
    if not object_store:
        assert dataset_info_fn is not None, "dataset_info_fn is required for local loading."
        dataset_info = dataset_info_fn()
    else:
        dataset_info_fn = DATASET_OPTIONS[dataset_name]
        dataset_info = dataset_info_fn(
            object_store, caption_type, embedding_type, dataset_resolution_type, min_frames, max_frames
        )

    augmentator_kwargs = {}
    if augmentor_name in ("video_basic_augmentor_v3", "video_basic_augmentor_v3_with_audio"):
        augmentator_kwargs["resize_on_read"] = resize_on_read
    if augmentor_name == "video_basic_augmentor_v3_with_audio":
        augmentator_kwargs["audio_sample_rate"] = audio_sample_rate
        augmentator_kwargs["sound_generation_mode"] = sound_generation_mode
        if key_renames:
            augmentator_kwargs["key_renames"] = key_renames

    # v3 augmentors handle conditioning natively via **kwargs; all others get post-factory injection.
    _V3_AUGMENTORS = ("video_basic_augmentor_v3", "video_basic_augmentor_v3_with_audio")
    if (conditioning_config is not None or uniform_conditioning) and augmentor_name in _V3_AUGMENTORS:
        augmentator_kwargs["conditioning_config"] = conditioning_config
        augmentator_kwargs["uniform_conditioning"] = uniform_conditioning
        augmentator_kwargs["temporal_compression_factor"] = temporal_compression_factor

    augmentor = AUGMENTOR_OPTIONS[augmentor_name](
        resolution=resolution,
        caption_type=caption_type,
        embedding_type=embedding_type,
        min_fps=min_fps_thres,
        max_fps=max_fps_thres,
        long_caption_ratio=long_caption_ratio,
        medium_caption_ratio=medium_caption_ratio,
        short_caption_ratio=short_caption_ratio,
        user_caption_ratio=user_caption_ratio,
        num_video_frames=num_video_frames,
        use_native_fps=use_native_fps,
        use_original_fps=use_original_fps,
        tokenizer_config=tokenizer_config,
        cfg_dropout_rate=cfg_dropout_rate,
        caption_config=caption_config,
        append_duration_fps_timestamps=append_duration_fps_timestamps,
        append_resolution_info=append_resolution_info,
        use_dynamic_fps=use_dynamic_fps,
        low_fps_bias=low_fps_bias,
        dataset_resolution_type=dataset_resolution_type,
        **augmentator_kwargs,
    )
    if (conditioning_config is not None or uniform_conditioning) and augmentor_name not in _V3_AUGMENTORS:
        augmentor["sequence_plan"] = L(_sequence_plan.SequencePlanAugmentor)(
            input_keys=["video"],
            args={
                "conditioning_config": conditioning_config,
                "uniform_conditioning": uniform_conditioning,
                "temporal_compression_factor": temporal_compression_factor,
            },
        )

    distributor = distributors.ShardlistMultiAspectRatio(
        shuffle=True,
        split_by_node=True,
        split_by_worker=True,
        resume_flag=True,
        verbose=False,
        is_infinite_loader=is_train,
    )

    video_data_config = DatasetConfig(
        keys=[],  # use the per_dataset_keys in DatasetInfo instead
        buffer_size=25,
        streaming_download=True,
        dataset_info=dataset_info,
        distributor=distributor,
        decoders=[
            video_decoder.construct_video_decoder(
                video_decoder_name=video_decoder_name,
                sequence_length=num_video_frames,
                chunk_size=chunk_size,
                min_fps_thres=min_fps_thres,
                max_fps_thres=max_fps_thres,
            ),
            pickle_decoders.pkl_decoder,
        ],
        augmentation=augmentor,
        remove_extension_from_keys=True,
        sample_keys_full_list_path=None,
    )

    return webdataset.Dataset(config=video_data_config, decoder_handler=warn_and_continue, detshuffle=detshuffle)


def get_image_dataset(
    dataset_name: str,
    resolution: str,
    dataset_resolution_type: str = "all",
    is_train: bool = True,
    augmentor_name: str = "image_basic_augmentor",
    object_store: str = "s3",
    detshuffle: bool = False,
    caption_type: str = "ai_v3p1",
    embedding_type: str = "t5_xxl",
    train_on_captions: list[str] = [],
    tokenizer_config: Optional[LazyDict] = None,
    cfg_dropout_rate: float = 0.0,
    append_resolution_info: bool = True,
) -> omegaconf.dictconfig.DictConfig:
    assert resolution in IMAGE_RES_SIZE_INFO.keys(), "The provided resolution cannot be found in IMAGE_RES_SIZE_INFO."
    assert object_store in ["s3", "swiftstack", "gcp"], "We support s3, gcp and swiftstack only."
    assert dataset_resolution_type in [
        "all",
        "gt480p",
        "gt720p",
        "gt1080p",
    ], f"The provided dataset resolution type {dataset_resolution_type} is not supported."
    # dataset_resolution_type
    # -- all - uses all dataset resolutions
    # -- gt480p - Uses only resolutions >= 480p
    # -- gt720p - Uses only resolutions >= 720p
    # -- gt1080p - Uses only resolutions >= 1080p
    dataset_info_fn = DATASET_OPTIONS[dataset_name]
    dataset_info = dataset_info_fn(object_store, caption_type, embedding_type, dataset_resolution_type)
    augmentation = AUGMENTOR_OPTIONS[augmentor_name](
        resolution=resolution,
        caption_type=caption_type,
        embedding_type=embedding_type,
        train_on_captions=train_on_captions,
        tokenizer_config=tokenizer_config,
        cfg_dropout_rate=cfg_dropout_rate,
        dataset_resolution_type=dataset_resolution_type,
        append_resolution_info=append_resolution_info,
    )

    distributor = distributors.ShardlistMultiAspectRatio(
        shuffle=True,
        split_by_node=True,
        split_by_worker=True,
        resume_flag=True,
        verbose=False,
        is_infinite_loader=is_train,
    )

    image_data_config = DatasetConfig(
        keys=[],

        # https://gitlab-master.nvidia.com/dir/imaginaire4/-/issues/119
        buffer_size=25,
        streaming_download=True,
        dataset_info=dataset_info,
        distributor=distributor,
        decoders=[
            image_decoders.pil_loader,
            pickle_decoders.pkl_decoder,
        ],
        augmentation=augmentation,
    )

    return webdataset.Dataset(config=image_data_config, detshuffle=detshuffle)
