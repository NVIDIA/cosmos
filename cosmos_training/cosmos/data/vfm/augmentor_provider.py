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

from typing import Optional

import cosmos.data.imaginaire.webdataset.augmentors.image.normalize as normalize
import cosmos.data.imaginaire.webdataset.augmentors.image.padding as padding
import cosmos.data.imaginaire.webdataset.augmentors.image.resize as resize
import cosmos.data.vfm.augmentors.append_fps_frames_for_image as append_fps_frames_for_image
import cosmos.data.vfm.augmentors.audio_caption as audio_caption
import cosmos.data.vfm.augmentors.caption_filter as caption_filter
import cosmos.data.vfm.augmentors.duration_fps_text_timestamps as duration_fps_text_timestamps
import cosmos.data.vfm.augmentors.image_resolution_filter as image_resolution_filter
import cosmos.data.vfm.augmentors.merge_datadict as merge_datadict
import cosmos.data.vfm.augmentors.resolution_text_info as resolution_text_info
import cosmos.data.vfm.augmentors.sound_sequence_plan as sound_sequence_plan
import cosmos.data.vfm.augmentors.text_tokenizer as text_tokenizer
import cosmos.data.vfm.augmentors.text_transforms_for_image as text_transforms_for_image
import cosmos.data.vfm.augmentors.text_transforms_for_video as text_transforms_for_video
import cosmos.data.vfm.augmentors.video_parsing as video_parsing
from cosmos.utils.lazy_config import LazyCall as L
from cosmos.utils.lazy_config import LazyDict
from cosmos.utils import log
from cosmos.data.vfm.augmentors import sequence_plan
from cosmos.data.vfm.utils import IMAGE_RES_SIZE_INFO, VIDEO_RES_SIZE_INFO

AUGMENTOR_OPTIONS = {}

CAMERA_MOVEMENT_PHRASES = [
    # Panning
    "camera pan",
    "camera pans",
    "camera slowly pan",
    "camera slowly pans",
    "camera quickly pans",
    "camera fast pans",
    "panning shot",
    "panning camera",
    "slow pan",
    "quick pan",
    "fast pan",
    "pan across",
    "pan around",
    "pan shot",
    "panoramic shot",
    # Tracking / Dolly
    "camera moves",
    "camera slowly moves",
    "camera quickly moves",
    "moving camera",
    "tracking shot",
    "tracking camera",
    "dolly shot",
    "dolly in",
    "dolly out",
    "camera follows",
    "camera tracks",
    "tracking movement",
    # Sweeps / Rotations
    "sweeping camera",
    "camera sweep",
    "rotating camera",
    "camera rotation",
    "camera rotates",
    "camera circles around",
    # Tilts
    "camera tilt",
    "camera tilts",
    "camera slowly tilts",
    "tilting camera",
    "tilt up",
    "tilt down",
    # Zooms
    "camera zoom",
    "camera zooms",
    "zooming camera",
    "zoom in",
    "zoom out",
    # Handheld / Shake
    "handheld camera",
    "handheld shot",
    "shaky camera",
    "camera shake",
    "shaky shot",
    "handheld movement",
]


def augmentor_register(key):
    log.info(f"registering {key}...")

    def decorator(func):
        AUGMENTOR_OPTIONS[key] = func
        return func

    return decorator


def get_video_text_transform(
    caption_type: str,
    embedding_type: Optional[str] = "t5_xxl",
    long_caption_ratio: int = 7,
    medium_caption_ratio: int = 2,
    short_caption_ratio: int = 1,
    user_caption_ratio: int = 90,
    num_video_frames: int = -1,
):
    del num_video_frames
    if caption_type == "vila_caption":
        video_text_transform = L(text_transforms_for_video.TextTransformForVideo)(
            input_keys=[],
            args={
                "captions_key": "metas",
                "embeddings_key": embedding_type,
                "caption_windows_key": "windows",
                "caption_type": "vila_caption",
                "embedding_caption_type": "vila_caption",
                "t5_tokens": {"num": 512},
                "is_mask_all_ones": True,
            },
        )
    elif caption_type == "t2w_qwen2p5_7b":
        log.info(
            f"caption_type: {caption_type}, long_caption_ratio: {long_caption_ratio}, medium_caption_ratio: {medium_caption_ratio}, short_caption_ratio: {short_caption_ratio}, user_caption_ratio: {user_caption_ratio}"
        )
        video_text_transform = L(text_transforms_for_video.TextTransformForVideo)(
            input_keys=[],
            args={
                "captions_key": "metas",
                "embeddings_key": embedding_type,
                "caption_windows_key": "t2w_windows",
                "caption_type": "qwen2p5_7b_caption",
                "embedding_caption_type": "t2w_qwen2p5_7b",
                "t5_tokens": {"num": 512},
                "is_mask_all_ones": True,
                "caption_probs": {
                    "long": long_caption_ratio,
                    "medium": medium_caption_ratio,
                    "short": short_caption_ratio,
                    "user": user_caption_ratio,
                },
            },
        )
    elif caption_type == "i2w_qwen2p5_7b_later_frames":
        video_text_transform = L(text_transforms_for_video.TextTransformForVideo)(
            input_keys=[],
            args={
                "captions_key": "metas",
                "embeddings_key": embedding_type,
                "caption_windows_key": "i2w_windows_later_frames",
                "caption_type": "qwen2p5_7b_caption",
                "embedding_caption_type": "i2w_qwen2p5_7b_later_frames",
                "t5_tokens": {"num": 512},
                "is_mask_all_ones": True,
                "caption_probs": {
                    "long": long_caption_ratio,
                    "medium": medium_caption_ratio,
                    "short": short_caption_ratio,
                    "user": user_caption_ratio,
                },
            },
        )
    else:
        raise ValueError(f"Unsupported caption type ({caption_type}) for video data")

    return video_text_transform


@augmentor_register("video_basic_augmentor_v1")
def get_video_augmentor_v1(
    resolution: str,
    caption_type: str = "vila_caption",
    embedding_type: str = "t5_xxl",
    min_fps: int = 10,
    max_fps: int = 60,
    long_caption_ratio: int = 7,
    medium_caption_ratio: int = 2,
    short_caption_ratio: int = 1,
    user_caption_ratio: int = 90,
):
    """Video augmentor V1. It relies on a separate video decoder to decode videos of required number of frames.
    Augmentors here will resize the video, add reflection padding, and extract captions and embeddings.

    Supported caption_type include vila_caption.
    Supported embedding_type include t5_xxl.
    """
    assert caption_type == "vila_caption", f"Unsupported caption type ({caption_type}) for video data"
    assert embedding_type == "t5_xxl", f"Unsupported embeddings type ({embedding_type}) for video data"
    video_text_transform = get_video_text_transform(
        caption_type=caption_type,
        embedding_type=embedding_type,
        long_caption_ratio=long_caption_ratio,
        medium_caption_ratio=medium_caption_ratio,
        short_caption_ratio=short_caption_ratio,
        user_caption_ratio=user_caption_ratio,
    )

    return {
        "merge_datadict": L(merge_datadict.DataDictMerger)(
            input_keys=["video"],
            output_keys=[
                "video",
                "fps",
                "num_frames",
                "chunk_index",
                "frame_start",
                "frame_end",
                "n_orig_video_frames",
                "num_multiplier",  # Add frame skipping multiplier for duration/FPS calculations
            ],
        ),
        "resize_largest_side_aspect_ratio_preserving": L(resize.ResizeLargestSideAspectPreserving)(
            input_keys=["video"],
            args={"size": VIDEO_RES_SIZE_INFO[resolution]},
        ),
        "reflection_padding": L(padding.ReflectionPadding)(
            input_keys=["video"],
            args={"size": VIDEO_RES_SIZE_INFO[resolution]},
        ),
        "text_transform": video_text_transform,
    }


@augmentor_register("video_basic_augmentor_v2")
def get_video_augmentor_v2(
    resolution: str,
    caption_type: str = "t2w_qwen2p5_7b",
    embedding_type: Optional[str] = "t5_xxl",
    min_fps: int = 10,
    max_fps: int = 60,
    long_caption_ratio: int = 7,
    medium_caption_ratio: int = 2,
    short_caption_ratio: int = 1,
    user_caption_ratio: int = 90,
    num_video_frames: int = -1,
    use_native_fps: bool = True,
    use_original_fps: bool = False,
    use_random_consecutive_frames: bool = False,
    append_duration_fps_timestamps: bool = False,
    append_resolution_info: bool = False,
    use_dynamic_fps: bool = False,
    low_fps_bias: float = 0.5,
    dataset_resolution_type: str = "all",  # Unused here; resolution check is only in VideoParsingWithFullFrames (v3)
    **_kwargs,  # absorbs tokenizer_config, cfg_dropout_rate, caption_config, etc. passed by generic callers
):
    """
    num_video_frames: -1 means use all frames, otherwise use the number of frames specified.

    Video augmentor V2. It works with a naive video decoder ("video_naive_bytes") that does nothing.
    Augmentors here include:
    - a basic video decoder that fetches frames within a window and delegates further subsampling or duplication to the modeling code to produce videos with the required number of frames.
    - resize the video
    - add reflection padding
    - extract captions and embeddings.

    When use_random_consecutive_frames is True, the augmentor will sample random consecutive frames, preserving the original fps.

    Supported caption_type include t2w_qwen2p5_7b and i2w_qwen2p5_7b_later_frames.
    Supported embedding_type include t5_xxl and umt5_xxl.
    """
    video_text_transform = get_video_text_transform(
        caption_type=caption_type,
        embedding_type=embedding_type,
        long_caption_ratio=long_caption_ratio,
        medium_caption_ratio=medium_caption_ratio,
        short_caption_ratio=short_caption_ratio,
        user_caption_ratio=user_caption_ratio,
    )
    if caption_type == "t2w_qwen2p5_7b":
        key_for_caption = "t2w_windows"
    elif caption_type == "i2w_qwen2p5_7b_later_frames":
        key_for_caption = "i2w_windows_later_frames"
    else:
        f"Unsupported caption type ({caption_type}) for video data"
    if embedding_type is not None:
        assert embedding_type in (
            "t5_xxl",
            "umt5_xxl",
        ), f"Unsupported embeddings type ({embedding_type}) for video data"

    return {
        "video_parsing": L(video_parsing.VideoParsing)(
            input_keys=["metas", "video"],
            args={
                "key_for_caption": key_for_caption,
                "min_duration": 4.0,
                "min_fps": min_fps,
                "max_fps": max_fps,
                "video_decode_num_threads": 4,
                "num_video_frames": num_video_frames,
                "use_native_fps": use_native_fps,
                "use_original_fps": use_original_fps,
                "use_random_consecutive_frames": use_random_consecutive_frames,
                "use_dynamic_fps": use_dynamic_fps,
                "low_fps_bias": low_fps_bias,
            },
        ),
        "merge_datadict": L(merge_datadict.DataDictMerger)(
            input_keys=["video"],
            output_keys=[
                "video",
                "fps",
                "num_frames",
                "chunk_index",
                "frame_start",
                "frame_end",
                "n_orig_video_frames",
                "num_multiplier",  # Add frame skipping multiplier for duration/FPS calculations
                "conditioning_fps",  # Add conditioning FPS for RoPE modulation
            ],
        ),
        "resize_largest_side_aspect_ratio_preserving": L(resize.ResizeLargestSideAspectPreserving)(
            input_keys=["video"],
            args={"size": VIDEO_RES_SIZE_INFO[resolution]},
        ),
        "reflection_padding": L(padding.ReflectionPadding)(
            input_keys=["video"],
            args={"size": VIDEO_RES_SIZE_INFO[resolution]},
        ),
        "text_transform": video_text_transform,
        # Duration/FPS timestamp augmentor - appends metadata like "The video is 2.5 seconds long and is of 24 FPS."
        # To customize the template or separator, add them to the args dict below:
        #   "template": "Custom format: {duration:.2f} seconds at {fps:.0f} FPS"
        #   Must include "{duration} seconds" and "{fps} FPS" in the template for the visualization callback
        #   "separator": " - "  # Used when caption doesn't end with '.'
        "duration_fps_timestamps": L(duration_fps_text_timestamps.DurationFPSTextTimeStamps)(
            input_keys=["ai_caption", "video", "conditioning_fps"],
            output_keys=[
                "ai_caption",
                "conditioning_duration",
                "duration_fps_template",
            ],  # Add duration and template as output keys
            args={
                "caption_key": "ai_caption",
                "video_key": "video",
                "fps_key": "conditioning_fps",
                "enabled": append_duration_fps_timestamps,
            },
        ),
        # Resolution info augmentor - appends metadata like "This video is 480x854."
        # Reads final_height/final_width from CropToMultiple (required).
        "resolution_info": L(resolution_text_info.ResolutionTextInfo)(
            input_keys=["ai_caption", "video", "image_size"],
            output_keys=[
                "ai_caption",
            ],
            args={
                "caption_key": "ai_caption",
                "video_key": "video",
                "enabled": append_resolution_info,
            },
        ),
    }


@augmentor_register("video_basic_augmentor_v2_with_tokenization")
def get_video_augmentor_v2_with_tokenization(
    resolution: str,
    caption_type: str = "t2w_qwen2p5_7b",
    embedding_type: Optional[str] = "t5_xxl",
    min_fps: int = 10,
    max_fps: int = 60,
    long_caption_ratio: int = 7,
    medium_caption_ratio: int = 2,
    short_caption_ratio: int = 1,
    user_caption_ratio: int = 90,
    num_video_frames: int = -1,
    use_native_fps: bool = True,
    use_original_fps: bool = False,
    use_random_consecutive_frames: bool = False,
    tokenizer_config: Optional[LazyDict] = None,
    cfg_dropout_rate: float = 0.0,
    append_duration_fps_timestamps: bool = False,
    append_resolution_info: bool = False,
    use_dynamic_fps: bool = False,
    low_fps_bias: float = 0.5,
    caption_config: dict | None = None,
    use_system_prompt: bool = False,
    dataset_resolution_type: str = "all",  # Unused here; resolution check is only in VideoParsingWithFullFrames (v3)
):
    """
    num_video_frames: -1 means use all frames, otherwise use the number of frames specified.

    Video augmentor V2. It works with a naive video decoder ("video_naive_bytes") that does nothing.
    Augmentors here include:
    - a basic video decoder that fetches frames within a window and delegates further subsampling or duplication to the modeling code to produce videos with the required number of frames.
    - resize the video
    - add reflection padding
    - extract captions and embeddings.

    When use_random_consecutive_frames is True, the augmentor will sample random consecutive frames, preserving the original fps.

    Supported caption_type include t2w_qwen2p5_7b and i2w_qwen2p5_7b_later_frames.
    Supported embedding_type include t5_xxl and umt5_xxl.
    """
    video_text_transform = get_video_text_transform(
        caption_type=caption_type,
        embedding_type=embedding_type,
        long_caption_ratio=long_caption_ratio,
        medium_caption_ratio=medium_caption_ratio,
        short_caption_ratio=short_caption_ratio,
        user_caption_ratio=user_caption_ratio,
    )
    if caption_type == "t2w_qwen2p5_7b":
        key_for_caption = "t2w_windows"
    elif caption_type == "i2w_qwen2p5_7b_later_frames":
        key_for_caption = "i2w_windows_later_frames"
    else:
        f"Unsupported caption type ({caption_type}) for video data"
    if embedding_type is not None:
        assert embedding_type in (
            "t5_xxl",
            "umt5_xxl",
        ), f"Unsupported embeddings type ({embedding_type}) for video data"

    return {
        "video_parsing": L(video_parsing.VideoParsing)(
            input_keys=["metas", "video"],
            args={
                "key_for_caption": key_for_caption,
                "min_duration": 4.0,
                "min_fps": min_fps,
                "max_fps": max_fps,
                "video_decode_num_threads": 4,
                "num_video_frames": num_video_frames,
                "use_native_fps": use_native_fps,
                "use_original_fps": use_original_fps,
                "use_random_consecutive_frames": use_random_consecutive_frames,
                "use_dynamic_fps": use_dynamic_fps,
                "low_fps_bias": low_fps_bias,
            },
        ),
        "merge_datadict": L(merge_datadict.DataDictMerger)(
            input_keys=["video"],
            output_keys=[
                "video",
                "fps",
                "num_frames",
                "chunk_index",
                "frame_start",
                "frame_end",
                "n_orig_video_frames",
                "num_multiplier",  # Add frame skipping multiplier for duration/FPS calculations
                "conditioning_fps",  # Add conditioning FPS for RoPE modulation
            ],
        ),
        "resize_largest_side_aspect_ratio_preserving": L(resize.ResizeLargestSideAspectPreserving)(
            input_keys=["video"],
            args={"size": VIDEO_RES_SIZE_INFO[resolution]},
        ),
        "reflection_padding": L(padding.ReflectionPadding)(
            input_keys=["video"],
            args={"size": VIDEO_RES_SIZE_INFO[resolution]},
        ),
        "text_transform": video_text_transform,
        # Duration/FPS timestamp augmentor - appends metadata like "The video is 2.5 seconds long and is of 24 FPS."
        # To customize the template or separator, add them to the args dict below:
        #   "template": "Custom format: {duration:.2f} seconds at {fps:.0f} FPS"
        #   Must include "{duration} seconds" and "{fps} FPS" in the template for the visualization callback
        #   "separator": " - "  # Used when caption doesn't end with '.'
        "duration_fps_timestamps": L(duration_fps_text_timestamps.DurationFPSTextTimeStamps)(
            input_keys=["ai_caption", "video", "conditioning_fps"],
            output_keys=[
                "ai_caption",
                "conditioning_duration",
                "duration_fps_template",
            ],  # Add duration and template as output keys
            args={
                "caption_key": "ai_caption",
                "video_key": "video",
                "fps_key": "conditioning_fps",
                "enabled": append_duration_fps_timestamps,
            },
        ),
        # Resolution info augmentor - appends metadata like "This video is 480x854."
        # Reads final_height/final_width from CropToMultiple (required).
        "resolution_info": L(resolution_text_info.ResolutionTextInfo)(
            input_keys=["ai_caption", "video", "image_size"],
            output_keys=[
                "ai_caption",
            ],
            args={
                "caption_key": "ai_caption",
                "video_key": "video",
                "enabled": append_resolution_info,
            },
        ),
        "text_tokenization": L(text_tokenizer.TextTokenizerTransform)(
            input_keys=["ai_caption"],
            output_keys=["text_token_ids"],
            args={
                "tokenizer_config": tokenizer_config,
                "cfg_dropout_rate": cfg_dropout_rate,
                "use_system_prompt": use_system_prompt,
            },
        ),
    }


@augmentor_register("video_basic_augmentor_v3")
def get_video_augmentor_v3(
    resolution: str,
    caption_config: dict | None = None,
    tokenizer_config: Optional[LazyDict] = None,
    cfg_dropout_rate: float = 0.0,
    append_duration_fps_timestamps: bool = False,
    append_resolution_info: bool = False,
    use_dynamic_fps: bool = False,
    max_stride: int = 3,
    min_stride: int = 1,
    use_system_prompt: bool = False,
    resize_on_read: bool = False,
    dataset_resolution_type: str = "all",
    max_num_frames: int = 1000,
    **kwargs,
):
    """Build a video augmentation pipeline with parsing, resizing, captioning, and tokenization.

    Args:
        resolution: Target resolution key (e.g., "480p"). Looked up in VIDEO_RES_SIZE_INFO
            to determine the resize target size.
        caption_config: Caption sampling configuration mapping caption field names to
            selection ratios. Example::

                {
                    "caption_short": {"ratio": 0.1, "use_for": "all"},
                    "caption_rewrite_descriptive": {"ratio": 0.2, "use_for": "all"},
                    "caption_dense": {"ratio": 0.3, "use_for": "all"},
                }

        tokenizer_config: Lazy config for the text tokenizer. Passed to TextTokenizerTransform
            to convert captions into token IDs.
        cfg_dropout_rate: Probability of dropping the caption (replacing with empty string)
            for classifier-free guidance training.
        append_duration_fps_timestamps: If True, appends a duration/FPS metadata string
            (e.g., "The video is 2.5 seconds long and is of 24 FPS.") to the caption.
        append_resolution_info: If True, appends a resolution metadata string
            (e.g., "This video is 480x854.") to the caption.
        use_dynamic_fps: If True, enables dynamic FPS sampling during video parsing,
            allowing variable stride to simulate different frame rates.
        max_stride: Maximum temporal stride for frame sampling during video parsing.
        use_system_prompt: If True, prepends a system prompt to the tokenized caption
            that instructs the model it is a video generation assistant.
        resize_on_read: If True, resizes video frames during decoding rather than as a
            separate augmentation step, reducing peak CPU memory usage.
        **kwargs: Additional keyword arguments.

    Returns:
        dict: Ordered dictionary of augmentation stage name to LazyCall config.

    Augmentors include:
    - a basic video decoder that fetches frames within a window and delegates further
      subsampling or duplication to the modeling code to produce videos with the
      required number of frames.
    - resize the video
    - add reflection padding
    - extract captions and embeddings.

    Supported caption_type include t2w_qwen2p5_7b and i2w_qwen2p5_7b_later_frames.
    Supported embedding_type include t5_xxl and umt5_xxl.
    """

    conditioning_config = kwargs.get("conditioning_config", None)
    uniform_conditioning = kwargs.get("uniform_conditioning", False)
    temporal_compression_factor = kwargs.get("temporal_compression_factor", 4)

    print("Running video_basic_augmentor_v3...")
    augmentors = {
        "video_parsing": L(video_parsing.VideoParsingWithFullFrames)(
            input_keys=["metas", "video"],
            args={
                "video_decode_num_threads": 4,
                "max_num_frames": max_num_frames,
                "use_dynamic_fps": use_dynamic_fps,
                "max_stride": max_stride,
                "min_stride": min_stride,
                "seek_mode": "exact",  # Change to "approximate"?
                "dataset_resolution_type": dataset_resolution_type,
            },
        ),
        "merge_datadict": L(merge_datadict.DataDictMerger)(
            input_keys=["video"],
            output_keys=[
                "video",
                "fps",
                "num_frames",
                "frame_start",
                "frame_end",
                "n_orig_video_frames",
                "conditioning_fps",  # Add conditioning FPS for RoPE modulation
            ],
        ),
    }
    if conditioning_config is not None or uniform_conditioning:
        augmentors["sequence_plan"] = L(sequence_plan.SequencePlanAugmentor)(
            input_keys=["video"],
            args={
                "conditioning_config": conditioning_config,
                "uniform_conditioning": uniform_conditioning,
                "temporal_compression_factor": temporal_compression_factor,
            },
        )
    augmentors.update(
        {
            "resize_largest_side_aspect_ratio_preserving": L(resize.ResizeLargestSideAspectPreserving)(
                input_keys=["video"],
                args={"size": VIDEO_RES_SIZE_INFO[resolution]},
            ),
            "reflection_padding": L(padding.ReflectionPadding)(
                input_keys=["video"],
                args={"size": VIDEO_RES_SIZE_INFO[resolution]},
            ),
            "text_transform": L(text_transforms_for_video.TextTransformForVideoWithFullFrames)(
                input_keys=["metas", "ai_caption", "sequence_plan"],
                args={
                    "caption_config": caption_config,
                    "caption_prefix": kwargs.get("caption_prefix", None),
                },
            ),
            # Duration/FPS timestamp augmentor - appends metadata like "The video is 2.5 seconds long and is of 24 FPS."
            # To customize the template or separator, add them to the args dict below:
            #   "template": "Custom format: {duration:.2f} seconds at {fps:.0f} FPS"
            #   Must include "{duration} seconds" and "{fps} FPS" in the template for the visualization callback
            #   "separator": " - "  # Used when caption doesn't end with '.'
            "duration_fps_timestamps": L(duration_fps_text_timestamps.DurationFPSTextTimeStamps)(
                input_keys=["ai_caption", "video", "conditioning_fps"],
                output_keys=[
                    "ai_caption",
                    "conditioning_duration",
                    "duration_fps_template",
                ],  # Add duration and template as output keys
                args={
                    "caption_key": "ai_caption",
                    "video_key": "video",
                    "fps_key": "conditioning_fps",
                    "enabled": append_duration_fps_timestamps,
                },
            ),
            # Resolution info augmentor - appends metadata like "This video is 480x854."
            # Reads final_height/final_width from CropToMultiple (required).
            "resolution_info": L(resolution_text_info.ResolutionTextInfo)(
                input_keys=["ai_caption", "video", "image_size"],
                output_keys=[
                    "ai_caption",
                ],
                args={
                    "caption_key": "ai_caption",
                    "video_key": "video",
                    "enabled": append_resolution_info,
                },
            ),
            "text_tokenization": L(text_tokenizer.TextTokenizerTransform)(
                input_keys=["ai_caption"],
                output_keys=["text_token_ids"],
                args={
                    "tokenizer_config": tokenizer_config,
                    "cfg_dropout_rate": cfg_dropout_rate,
                    "use_system_prompt": use_system_prompt,
                },
            ),
        }
    )

    if resize_on_read:
        # When resize_on_read is True, we resize the video frames on read instead of during decoding.
        # This is useful for reducing CPU memory usage by avoiding the need to load the entire video into memory.
        augmentors["video_parsing"]["args"]["size"] = VIDEO_RES_SIZE_INFO[resolution]
        del augmentors["resize_largest_side_aspect_ratio_preserving"]
    return augmentors



# Use video_basic_augmentor_v3_json_caption instead.
@augmentor_register("video_basic_augmentor_v3_with_audio")
def get_video_augmentor_v3_with_audio(
    resolution: str,
    caption_config: dict | None = None,
    tokenizer_config: Optional[LazyDict] = None,
    cfg_dropout_rate: float = 0.0,
    append_duration_fps_timestamps: bool = False,
    use_dynamic_fps: bool = False,
    max_stride: int = 3,
    resize_on_read: bool = False,
    audio_sample_rate: int = 48000,
    sound_generation_mode: str = "t2vs",
    key_renames: dict[str, str] | None = None,
    extract_audio: bool = True,
    **kwargs,
):
    """
    Same as video_basic_augmentor_v3 but with audio extraction enabled.
    For use with V2A (tv2s) and T2VS datasets where both video and audio are loaded.

    Args:
        sound_generation_mode: One of "t2vs", "tv2s", "ts2v", "ti2sv".
            Controls how the SequencePlan is built for conditioning.
        key_renames: Optional mapping of old_key -> new_key to rename data_dict keys
            before the augmentor pipeline runs (e.g. {"metas_w_audio_caps": "metas"}).
        extract_audio: When True (default), decodes audio from video bytes.
            When False, emits placeholder sound=None and audio_sample_rate keys
            without decoding audio.  Useful for mixing video-only and audio
            datasets in the same dataloader with consistent output keys.
    """
    augmentors = get_video_augmentor_v3(
        resolution=resolution,
        caption_config=caption_config,
        tokenizer_config=tokenizer_config,
        cfg_dropout_rate=cfg_dropout_rate,
        append_duration_fps_timestamps=append_duration_fps_timestamps,
        use_dynamic_fps=use_dynamic_fps,
        max_stride=max_stride,
        resize_on_read=resize_on_read,
        **kwargs,
    )

    # Insert key renamer at the very start if key_renames is provided
    if key_renames:
        renamed_augmentors = {
            "key_rename": L(merge_datadict.KeyRenamer)(
                input_keys=[],
                args={"rename_map": key_renames},
            ),
        }
        renamed_augmentors.update(augmentors)
        augmentors = renamed_augmentors

    # Configure audio extraction in the video parser
    augmentors["video_parsing"]["args"]["extract_audio"] = extract_audio
    augmentors["video_parsing"]["args"]["audio_sample_rate"] = audio_sample_rate
    if not extract_audio:
        augmentors["video_parsing"]["args"]["emit_placeholder_sound"] = True

    # Add sound and audio_sample_rate to merge keys so they propagate through the pipeline
    augmentors["merge_datadict"]["output_keys"].extend(["sound", "audio_sample_rate"])

    # Tell text_transform to keep metas — AudioCaptionAppender will clean it up
    augmentors["text_transform"]["args"]["keep_metas"] = True

    # Insert audio caption appender BEFORE text_tokenization.
    # We rebuild the ordered dict to ensure correct pipeline order.
    text_tokenization = augmentors.pop("text_tokenization")
    augmentors["audio_caption"] = L(audio_caption.AudioCaptionAppender)(
        input_keys=["metas", "ai_caption"],
        args={
            "audio_caption_key": "caption_audio",
            "sound_key": "sound",
        },
    )
    augmentors["text_tokenization"] = text_tokenization

    # Add sequence plan builder at the end of the pipeline (after all data is ready)
    augmentors["sound_sequence_plan"] = L(sound_sequence_plan.SoundSequencePlanBuilder)(
        input_keys=[],
        args={
            "mode": sound_generation_mode,
            "video_key": "video",
            "sound_key": "sound",
        },
    )
    return augmentors


@augmentor_register("video_basic_augmentor_v3_json_caption")
def get_video_augmentor_v3_json_caption(
    resolution: str,
    caption_config: dict | None = None,
    tokenizer_config: Optional[LazyDict] = None,
    cfg_dropout_rate: float = 0.0,
    append_duration_fps_timestamps: bool = False,
    append_resolution_info: bool = False,
    use_dynamic_fps: bool = False,
    max_stride: int = 3,
    min_stride: int = 1,
    use_system_prompt: bool = False,
    resize_on_read: bool = False,
    dataset_resolution_type: str = "all",
    max_num_frames: int = 1000,
    audio_sample_rate: int = 48000,
    sound_generation_mode: str = "t2vs",
    extract_audio: bool = True,
    caption_key: str = "caption",
    **kwargs,
):
    """Build a video augmentation pipeline for JSON-captioned, chunked video datasets.

    The pipeline samples a single caption chunk per video, decodes only that chunk's
    frames, optionally extracts audio, and injects ``metas["caption_audio"]`` into the
    caption dict as an ``audio_description`` field when present.

    Args:
        resolution: Target resolution key (e.g., "480p"). Looked up in VIDEO_RES_SIZE_INFO
            to determine the resize target size.
        caption_config: Caption sampling configuration mapping caption field names to

                {
                    "json_field_dropout_rate": 0.1,
                }
        caption_key: Metadata key containing the chunked JSON caption string.
        tokenizer_config: Lazy config for the text tokenizer. Passed to TextTokenizerTransform
            to convert captions into token IDs.
        cfg_dropout_rate: Probability of dropping the caption (replacing with empty string)
            for classifier-free guidance training.
        append_duration_fps_timestamps: If True, injects duration/FPS fields into the
            caption dict.
        append_resolution_info: If True, injects resolution/aspect-ratio fields into the
            caption dict.
        use_dynamic_fps: If True, enables dynamic FPS sampling during video parsing,
            allowing variable stride to simulate different frame rates.
        max_stride: Maximum temporal stride for frame sampling during video parsing.
        use_system_prompt: If True, prepends a system prompt to the tokenized caption
            that instructs the model it is a video generation assistant.
        resize_on_read: If True, resizes video frames during decoding rather than as a
            separate augmentation step, reducing peak CPU memory usage.
        audio_sample_rate: Sample rate for audio extraction.
        sound_generation_mode: One of "t2vs", "tv2s", "ts2v", "ti2sv".
            Controls how the SequencePlan is built for conditioning.
        extract_audio: When True (default), decodes audio from video bytes.
            When False, emits placeholder sound=None and audio_sample_rate keys
            without decoding audio.  Useful for mixing video-only and audio
            datasets in the same dataloader with consistent output keys.
        **kwargs: Additional keyword arguments forwarded via ``conditioning_config``,
            ``uniform_conditioning``, ``temporal_compression_factor``.

    Returns:
        dict: Ordered dictionary of augmentation stage name to LazyCall config.
    """

    conditioning_config = kwargs.get("conditioning_config", None)
    uniform_conditioning = kwargs.get("uniform_conditioning", False)
    temporal_compression_factor = kwargs.get("temporal_compression_factor", 4)

    print("Running video_augmentor_v3_json_caption...")
    augmentors = {
        # Caption parsing runs BEFORE video parsing so that VideoParsingChunkedFrames can
        # decode only the frames belonging to a randomly sampled caption chunk.
        # keep_metas=True so that VideoParsingChunkedFrames still has framerate/width/height/nb_frames.
        "text_transform": L(text_transforms_for_video.TextTransformForVideoJsonCaption)(
            input_keys=["metas", "video"],
            args={
                "caption_config": caption_config,
                "caption_key": caption_key,
                "keep_metas": True,
            },
        ),
        "video_parsing": L(video_parsing.VideoParsingChunkedFrames)(
            input_keys=["metas", "video"],
            args={
                "video_decode_num_threads": 4,
                "max_num_frames": max_num_frames,
                "use_dynamic_fps": use_dynamic_fps,
                "max_stride": max_stride,
                "min_stride": min_stride,
                "seek_mode": "exact",
                "dataset_resolution_type": dataset_resolution_type,
                "extract_audio": extract_audio,
                "audio_sample_rate": audio_sample_rate,
                "emit_placeholder_sound": not extract_audio,
            },
        ),
        "merge_datadict": L(merge_datadict.DataDictMerger)(
            input_keys=["video"],
            output_keys=[
                "video",
                "fps",
                "num_frames",
                "frame_start",
                "frame_end",
                "n_orig_video_frames",
                "conditioning_fps",
                "sound",
                "audio_sample_rate",
            ],
        ),
    }

    if conditioning_config is not None or uniform_conditioning:
        augmentors["sequence_plan"] = L(sequence_plan.SequencePlanAugmentor)(
            input_keys=["video"],
            args={
                "conditioning_config": conditioning_config,
                "uniform_conditioning": uniform_conditioning,
                "temporal_compression_factor": temporal_compression_factor,
            },
        )
    augmentors.update(
        {
            "resize_largest_side_aspect_ratio_preserving": L(resize.ResizeLargestSideAspectPreserving)(
                input_keys=["video"],
                args={"size": VIDEO_RES_SIZE_INFO[resolution]},
            ),
            "reflection_padding": L(padding.ReflectionPadding)(
                input_keys=["video"],
                args={"size": VIDEO_RES_SIZE_INFO[resolution]},
            ),
            # Duration/FPS timestamp augmentor - appends metadata like "The video is 2.5 seconds long and is of 24 FPS."
            # To customize the template or separator, add them to the args dict below:
            #   "template": "Custom format: {duration:.2f} seconds at {fps:.0f} FPS"
            #   Must include "{duration} seconds" and "{fps} FPS" in the template for the visualization callback
            #   "separator": " - "  # Used when caption doesn't end with '.'
            "duration_fps_timestamps": L(duration_fps_text_timestamps.DurationFPSTextTimeStamps)(
                input_keys=["ai_caption", "video", "conditioning_fps"],
                output_keys=[
                    "ai_caption",
                    "conditioning_duration",
                    "duration_fps_template",
                ],  # Add duration and template as output keys
                args={
                    "caption_key": "ai_caption",
                    "video_key": "video",
                    "fps_key": "conditioning_fps",
                    "enabled": append_duration_fps_timestamps,
                },
            ),
            # Resolution info augmentor - appends metadata like "This video is 480x854."
            # Reads final_height/final_width from CropToMultiple (required).
            "resolution_info": L(resolution_text_info.ResolutionTextInfo)(
                input_keys=["ai_caption", "video", "image_size"],
                output_keys=[
                    "ai_caption",
                ],
                args={
                    "caption_key": "ai_caption",
                    "video_key": "video",
                    "enabled": append_resolution_info,
                },
            ),
            "text_tokenization": L(text_tokenizer.TextTokenizerTransform)(
                input_keys=["ai_caption"],
                output_keys=["text_token_ids"],
                args={
                    "tokenizer_config": tokenizer_config,
                    "cfg_dropout_rate": cfg_dropout_rate,
                    "use_system_prompt": use_system_prompt,
                },
            ),
            "sound_sequence_plan": L(sound_sequence_plan.SoundSequencePlanBuilder)(
                input_keys=[],
                args={
                    "mode": sound_generation_mode,
                    "video_key": "video",
                    "sound_key": "sound",
                },
            ),
        }
    )

    if resize_on_read:
        # When resize_on_read is True, we resize the video frames on read instead of during decoding.
        # This is useful for reducing CPU memory usage by avoiding the need to load the entire video into memory.
        augmentors["video_parsing"]["args"]["size"] = VIDEO_RES_SIZE_INFO[resolution]
        del augmentors["resize_largest_side_aspect_ratio_preserving"]
    return augmentors


@augmentor_register("noframedrop_nocameramove_video_augmentor_v1")
def get_noframedrop_nocameramove_video_augmentor_v1(
    resolution: str,
    caption_type: str = "t2w_qwen2p5_7b",
    embedding_type: Optional[str] = "t5_xxl",
    min_fps: int = 10,
    max_fps: int = 60,
    long_caption_ratio: int = 7,
    medium_caption_ratio: int = 2,
    short_caption_ratio: int = 1,
    user_caption_ratio: int = 90,
    num_video_frames: int = -1,
    use_native_fps: bool = True,
    use_original_fps: bool = False,
    use_random_consecutive_frames: bool = False,
    dataset_resolution_type: str = "all",  # Unused here; resolution check is only in VideoParsingWithFullFrames (v3)
):
    """
    This augmentor is v2 + the following:
    - no frame drop by ensure num_multipler is always 1
    - no camera move (indiciated by the camera related bad words in the caption)
    """
    video_text_transform = get_video_text_transform(
        caption_type=caption_type,
        embedding_type=embedding_type,
        long_caption_ratio=long_caption_ratio,
        medium_caption_ratio=medium_caption_ratio,
        short_caption_ratio=short_caption_ratio,
        user_caption_ratio=user_caption_ratio,
    )
    if caption_type == "t2w_qwen2p5_7b":
        key_for_caption = "t2w_windows"
    elif caption_type == "i2w_qwen2p5_7b_later_frames":
        key_for_caption = "i2w_windows_later_frames"
    else:
        f"Unsupported caption type ({caption_type}) for video data"
    if embedding_type is not None:
        assert embedding_type in (
            "t5_xxl",
            "umt5_xxl",
        ), f"Unsupported embeddings type ({embedding_type}) for video data"

    contain_keyword = False  # ensure no camera move
    augmentations = {
        "video_parsing": L(video_parsing.VideoParsing)(
            input_keys=["metas", "video"],
            args={
                "key_for_caption": key_for_caption,
                "min_duration": 4.0,
                "min_fps": min_fps,
                "max_fps": max_fps,
                "video_decode_num_threads": 4,
                "num_video_frames": num_video_frames,
                "use_native_fps": use_native_fps,
                "use_original_fps": use_original_fps,
                "use_random_consecutive_frames": use_random_consecutive_frames,
                # Both use_original_fps=True and "allowed_num_multiplers": [1] prevent frame dropping.
                # Key differences:
                # - use_original_fps=True: Hard-codes num_multiplier=1 and ignores allowed_num_multiplers setting.
                #   Won't skip entire videos, but may discard head/tail frames, potentially causing
                #   video-caption misalignment.
                # - "allowed_num_multiplers": [1]: Uses the multiplier system but restricts it to 1x only. May skip videos, causing slower dataloader
                "allowed_num_multiplers": [1],
            },
        ),
        "merge_datadict": L(merge_datadict.DataDictMerger)(
            input_keys=["video"],
            output_keys=[
                "video",
                "fps",
                "num_frames",
                "chunk_index",
                "frame_start",
                "frame_end",
                "n_orig_video_frames",
            ],
        ),
        "resize_largest_side_aspect_ratio_preserving": L(resize.ResizeLargestSideAspectPreserving)(
            input_keys=["video"],
            args={"size": VIDEO_RES_SIZE_INFO[resolution]},
        ),
        "reflection_padding": L(padding.ReflectionPadding)(
            input_keys=["video"],
            args={"size": VIDEO_RES_SIZE_INFO[resolution]},
        ),
        "text_transform": video_text_transform,
        "caption_filter": L(caption_filter.CaptionFilter)(
            input_keys=["ai_caption"],  # Works with ai_caption from TextTransformForVideo
            args={
                "keywords": CAMERA_MOVEMENT_PHRASES,
                "contain_keyword": contain_keyword,
                "log_filtered": False,  # Enable logging to see what gets filtered
                "filter_stats": True,
                # For 4k and physics AI datasets, even if this has camera movement, it is still good
                "dont_apply_on_webdataset_names": [
                    "4k_",
                    "a2d2_",
                    "agibot_",
                    "alpamayo_",
                    "bridgev2p1_",
                    "droid_",
                    "gr00t_",
                    "nexar",
                    "onex",
                    "openx",
                    "physical-ai-special",
                    "physics-cosmos-db",
                    "wisa",
                    "robomind",
                    "smartspace_",
                ],
            },
        ),
    }
    mode_str = "contain" if contain_keyword else "exclude"
    log.info(
        f"[video] noframedrop_nocameramove_video_augmentor_v1: Added caption filter in '{mode_str}' mode "
        f"with {len(CAMERA_MOVEMENT_PHRASES)} camera movement phrases"
    )
    return augmentations


@augmentor_register("nocameramove_video_augmentor_v1")
def get_nocameramove_video_augmentor_v1(
    resolution: str,
    caption_type: str = "t2w_qwen2p5_7b",
    embedding_type: Optional[str] = "t5_xxl",
    min_fps: int = 10,
    max_fps: int = 60,
    long_caption_ratio: int = 7,
    medium_caption_ratio: int = 2,
    short_caption_ratio: int = 1,
    user_caption_ratio: int = 90,
    num_video_frames: int = -1,
    use_native_fps: bool = True,
    use_original_fps: bool = False,
    use_random_consecutive_frames: bool = False,
):
    """
    This augmentor is based on noframedrop_nocameramove_video_augmentor_v1 but:
    - allows limited frame drop by setting allowed_num_multiplers to [1,2]
    - no camera move (indicated by the camera related bad words in the caption)
    """
    # Get the base augmentations from the no-frame-drop version
    augmentations = get_noframedrop_nocameramove_video_augmentor_v1(
        resolution=resolution,
        caption_type=caption_type,
        embedding_type=embedding_type,
        min_fps=min_fps,
        max_fps=max_fps,
        long_caption_ratio=long_caption_ratio,
        medium_caption_ratio=medium_caption_ratio,
        short_caption_ratio=short_caption_ratio,
        user_caption_ratio=user_caption_ratio,
        num_video_frames=num_video_frames,
        use_native_fps=use_native_fps,
        use_original_fps=use_original_fps,
        use_random_consecutive_frames=use_random_consecutive_frames,
    )

    # Modify only the allowed_num_multiplers parameter
    augmentations["video_parsing"].args["allowed_num_multiplers"] = [1, 2]

    log.info(
        "[video] nocameramove_video_augmentor_v1: Modified allowed_num_multiplers to [1, 2] "
        "for limited frame dropping capability"
    )
    return augmentations


@augmentor_register("image_basic_augmentor")
def get_image_augmentor(
    resolution: str,
    caption_type: str = "ai_v3p1",
    embedding_type: str = "t5_xxl",
    dataset_resolution_type: str = "all",
):
    augmentation = {
        "image_resolution_filter": L(image_resolution_filter.ImageResolutionFilter)(
            input_keys=["images"],
            args={"dataset_resolution_type": dataset_resolution_type, "image_key": "images"},
        ),
        "resize_largest_side_aspect_ratio_preserving": L(resize.ResizeLargestSideAspectPreserving)(
            input_keys=["images"],
            args={"size": IMAGE_RES_SIZE_INFO[resolution]},
        ),
        "reflection_padding": L(padding.ReflectionPadding)(
            input_keys=["images"],
            args={"size": IMAGE_RES_SIZE_INFO[resolution]},
        ),
        "normalize": L(normalize.Normalize)(
            input_keys=["images"],
            args={"mean": 0.5, "std": 0.5},
        ),
        "text_transform": L(text_transforms_for_image.TextTransformForImage)(
            input_keys=[],
            args={
                "caption_type": caption_type,
                "embedding_type": embedding_type,
                "weight_captions_gt": 0.05,
                "caption_probs": {"ground_truth": 0.05, "vfc_fidelity": 0.95},
                "t5_tokens": {"num": 512, "dim": 1024},
                "is_mask_all_ones": True,
            },
        ),
        "append_fps_frames": L(append_fps_frames_for_image.AppendFPSFramesForImage)(),
    }

    return augmentation


@augmentor_register("image_basic_augmentor_without_embeddings")
def get_image_augmentor_without_embeddings(
    resolution: str,
    caption_type: str = "ai_v3p1",
    embedding_type: Optional[str] = None,
    train_on_captions: list[str] = [],
    dataset_resolution_type: str = "all",
):
    augmentation = {
        "image_resolution_filter": L(image_resolution_filter.ImageResolutionFilter)(
            input_keys=["images"],
            args={"dataset_resolution_type": dataset_resolution_type, "image_key": "images"},
        ),
        "resize_largest_side_aspect_ratio_preserving": L(resize.ResizeLargestSideAspectPreserving)(
            input_keys=["images"],
            args={"size": IMAGE_RES_SIZE_INFO[resolution]},
        ),
        "reflection_padding": L(padding.ReflectionPadding)(
            input_keys=["images"],
            args={"size": IMAGE_RES_SIZE_INFO[resolution]},
        ),
        "normalize": L(normalize.Normalize)(
            input_keys=["images"],
            args={"mean": 0.5, "std": 0.5},
        ),
        "text_transform": L(text_transforms_for_image.TextTransformForImageWithoutEmbeddings)(
            input_keys=[],
            args={
                "caption_type": caption_type,
                "train_on_captions": train_on_captions,
            },
        ),
        "append_fps_frames": L(append_fps_frames_for_image.AppendFPSFramesForImage)(),
    }

    return augmentation


@augmentor_register("image_basic_augmentor_with_tokenization")
def image_basic_augmentor_with_tokenization(
    resolution: str,
    caption_type: str = "ai_v3p1",
    embedding_type: Optional[str] = None,
    tokenizer_config: Optional[LazyDict] = None,
    cfg_dropout_rate: float = 0.0,
    append_resolution_info: bool = False,
    use_system_prompt: bool = False,
    train_on_captions: list[str] | None = None,
    dataset_resolution_type: str = "all",
    **kwargs,
):
    """Build an image augmentation pipeline with resizing, captioning, and tokenization.

    Args:
        resolution: Target resolution key (e.g., "256p", "512p"). Looked up in
            IMAGE_RES_SIZE_INFO to determine the resize target size.
        caption_type: Caption field name to extract from the data sample
            (e.g., "ai_v3p1", "ai_caption").
        embedding_type: Unused. Kept for interface compatibility with other augmentors.
        tokenizer_config: Lazy config for the text tokenizer. Passed to TextTokenizerTransform
            to convert captions into token IDs.
        cfg_dropout_rate: Probability of dropping the caption (replacing with empty string)
            for classifier-free guidance training.
        append_resolution_info: If True, appends a resolution metadata string
            (e.g., "This image is 512x512.") to the caption.
        use_system_prompt: If True, prepends a system prompt to the tokenized caption
            that instructs the model it is an image generation assistant.
        train_on_captions: If non-empty, only use these caption types (e.g. ["dense"], ["qwen3vl_30B_v1_dense"]).
            If empty, caption type is inferred from the data.

    Returns:
        dict: Ordered dictionary of augmentation stage name to LazyCall config.
    """
    augmentation = {
        "image_resolution_filter": L(image_resolution_filter.ImageResolutionFilter)(
            input_keys=["images"],
            args={"dataset_resolution_type": dataset_resolution_type, "image_key": "images"},
        ),
        "resize_largest_side_aspect_ratio_preserving": L(resize.ResizeLargestSideAspectPreserving)(
            input_keys=["images"],
            args={"size": IMAGE_RES_SIZE_INFO[resolution]},
        ),
        "reflection_padding": L(padding.ReflectionPadding)(
            input_keys=["images"],
            args={"size": IMAGE_RES_SIZE_INFO[resolution]},
        ),
        "normalize": L(normalize.Normalize)(
            input_keys=["images"],
            args={"mean": 0.5, "std": 0.5},
        ),
        "text_transform": L(text_transforms_for_image.TextTransformForImageWithoutEmbeddings)(
            input_keys=[],
            args={
                "caption_type": caption_type,
                "train_on_captions": train_on_captions or [],
                "caption_prefix": kwargs.get("caption_prefix", None),
            },
        ),
        # Resolution info augmentor - appends metadata like "This image is 512x512."
        # Reads final_height/final_width from CropToMultiple (required).
        "resolution_info": L(resolution_text_info.ResolutionTextInfo)(
            input_keys=["ai_caption", "images", "image_size"],
            output_keys=[
                "ai_caption",
            ],
            args={
                "caption_key": "ai_caption",
                "image_key": "images",
                "enabled": append_resolution_info,
            },
        ),
        "text_tokenization": L(text_tokenizer.TextTokenizerTransform)(
            input_keys=["ai_caption"],
            output_keys=["text_token_ids"],
            args={
                "tokenizer_config": tokenizer_config,
                "cfg_dropout_rate": cfg_dropout_rate,
                "use_system_prompt": use_system_prompt,
            },
        ),
        "append_fps_frames": L(append_fps_frames_for_image.AppendFPSFramesForImage)(),
    }

    return augmentation


@augmentor_register("image_basic_augmentor_json_caption")
def image_basic_augmentor_json_caption(
    resolution: str,
    caption_type: str = "ai_v3p1",
    caption_config: dict | None = None,
    embedding_type: Optional[str] = None,
    tokenizer_config: Optional[LazyDict] = None,
    cfg_dropout_rate: float = 0.0,
    append_resolution_info: bool = False,
    use_system_prompt: bool = False,
    train_on_captions: list[str] | None = None,
    dataset_resolution_type: str = "all",
    **kwargs,
):
    """Build an image augmentation pipeline with resizing, json captioning, and tokenization.

    Args:
        resolution: Target resolution key (e.g., "256p", "512p"). Looked up in
            IMAGE_RES_SIZE_INFO to determine the resize target size.
        caption_type: Caption field name to extract from the data sample
            (e.g., "ai_v3p1", "ai_caption").
        embedding_type: Unused. Kept for interface compatibility with other augmentors.
        tokenizer_config: Lazy config for the text tokenizer. Passed to TextTokenizerTransform
            to convert captions into token IDs.
        cfg_dropout_rate: Probability of dropping the caption (replacing with empty string)
            for classifier-free guidance training.
        append_resolution_info: If True, appends a resolution metadata string
            (e.g., "This image is 512x512.") to the caption.
        use_system_prompt: If True, prepends a system prompt to the tokenized caption
            that instructs the model it is an image generation assistant.
        train_on_captions: If non-empty, only use these caption types (e.g. ["dense"], ["qwen3vl_30B_v1_dense"]).
            If empty, caption type is inferred from the data.

    Returns:
        dict: Ordered dictionary of augmentation stage name to LazyCall config.
    """
    assert caption_config is not None and "json_field_dropout_rate" in caption_config, (
        "image_basic_augmentor_json_caption requires caption_config with "
        "'json_field_dropout_rate'. Set it in your experiment config, e.g. "
        "caption_config={'json_field_dropout_rate': 0.05}."
    )
    augmentation = {
        "image_resolution_filter": L(image_resolution_filter.ImageResolutionFilter)(
            input_keys=["images"],
            args={"dataset_resolution_type": dataset_resolution_type, "image_key": "images"},
        ),
        "resize_largest_side_aspect_ratio_preserving": L(resize.ResizeLargestSideAspectPreserving)(
            input_keys=["images"],
            args={"size": IMAGE_RES_SIZE_INFO[resolution]},
        ),
        "reflection_padding": L(padding.ReflectionPadding)(
            input_keys=["images"],
            args={"size": IMAGE_RES_SIZE_INFO[resolution]},
        ),
        "normalize": L(normalize.Normalize)(
            input_keys=["images"],
            args={"mean": 0.5, "std": 0.5},
        ),
        "text_transform": L(text_transforms_for_image.TextTransformForImageJsonCaption)(
            input_keys=[],
            args={
                "caption_type": caption_type,
                "json_field_dropout_rate": caption_config["json_field_dropout_rate"],
            },
        ),
        # Resolution info augmentor - appends metadata like "This image is 512x512."
        # Reads final_height/final_width from CropToMultiple (required).
        "resolution_info": L(resolution_text_info.ResolutionTextInfo)(
            input_keys=["ai_caption", "images", "image_size"],
            output_keys=[
                "ai_caption",
            ],
            args={
                "caption_key": "ai_caption",
                "image_key": "images",
                "enabled": append_resolution_info,
            },
        ),
        "text_tokenization": L(text_tokenizer.TextTokenizerTransform)(
            input_keys=["ai_caption"],
            output_keys=["text_token_ids"],
            args={
                "tokenizer_config": tokenizer_config,
                "cfg_dropout_rate": cfg_dropout_rate,
                "use_system_prompt": use_system_prompt,
            },
        ),
        "append_fps_frames": L(append_fps_frames_for_image.AppendFPSFramesForImage)(),
    }

    return augmentation
