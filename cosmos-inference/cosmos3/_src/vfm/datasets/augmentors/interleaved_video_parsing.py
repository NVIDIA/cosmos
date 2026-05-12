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

import random
from collections.abc import Callable
from typing import Optional

import numpy as np
import omegaconf
import torch
from einops import rearrange
from torchcodec.decoders import VideoDecoder
from torchvision.transforms.v2 import Resize, UniformTemporalSubsample

from cosmos3._src.imaginaire.datasets.webdataset.augmentors.image.misc import obtain_augmentation_size
from cosmos3._src.imaginaire.utils import log
from cosmos3._src.vfm.datasets.augmentors.video_parsing import VideoParsingWithFullFrames

# Local copies of the torchcodec decoder helpers so this module does not depend on
# private symbols of ``video_parsing.py``. Behavior matches the originals.
_PostDecodeTransforms = list[Callable[[torch.Tensor], torch.Tensor]] | None
_SUPPORTS_VIDEO_DECODER_TRANSFORMS: bool | None = None
_WARNED_POST_DECODE_TRANSFORMS = False


def _create_video_decoder(
    video: bytes,
    seek_mode: str,
    num_ffmpeg_threads: int,
    transforms: _PostDecodeTransforms = None,
) -> tuple[VideoDecoder, _PostDecodeTransforms]:
    global _SUPPORTS_VIDEO_DECODER_TRANSFORMS, _WARNED_POST_DECODE_TRANSFORMS

    kwargs = {"seek_mode": seek_mode, "num_ffmpeg_threads": num_ffmpeg_threads}
    if transforms is None:
        return VideoDecoder(video, **kwargs), None

    if _SUPPORTS_VIDEO_DECODER_TRANSFORMS is not False:
        try:
            decoder = VideoDecoder(video, transforms=transforms, **kwargs)
            _SUPPORTS_VIDEO_DECODER_TRANSFORMS = True
            return decoder, None
        except TypeError as e:
            if "transforms" not in str(e):
                raise
            _SUPPORTS_VIDEO_DECODER_TRANSFORMS = False

    if not _WARNED_POST_DECODE_TRANSFORMS:
        log.warning(
            "Installed torchcodec does not support VideoDecoder(transforms=...); "
            "applying video transforms after frame decode.",
            rank0_only=False,
        )
        _WARNED_POST_DECODE_TRANSFORMS = True
    return VideoDecoder(video, **kwargs), transforms


def _apply_post_decode_transforms(
    frames: torch.Tensor, transforms: _PostDecodeTransforms
) -> torch.Tensor:  # frames: [T,C,H,W], returns: [T,C,H,W]
    if transforms is None:
        return frames

    for transform in transforms:
        frames = transform(frames)  # [T,C,H,W]
    return frames


class VideoTransferAlignedFullFramesParsing(VideoParsingWithFullFrames):
    """Decode RGB and precomputed control videos with one shared v3 frame plan.

    This is the variable-length counterpart of the fixed-window transfer parser.
    The RGB stream determines the sampled stride and frame indices. Any extra
    input video streams, such as depth or segmentation, are decoded with the same
    frame indices so the control video stays temporally aligned with the target.
    """

    def __init__(self, input_keys: list, output_keys: Optional[list] = None, args: Optional[dict] = None) -> None:
        assert len(input_keys) >= 2, "VideoTransferAlignedFullFramesParsing requires [metas, video, ...]."
        super().__init__(input_keys=input_keys[:2], output_keys=output_keys, args=args)
        self.input_keys = input_keys
        self.control_video_keys = input_keys[2:]
        self.min_stride_key = self.args.get("min_stride_key", "_full_frames_min_stride")

    def _build_rgb_decode_transform(self, data_dict: dict, meta_dict: dict) -> list[Resize] | None:
        if not self.perform_resize:
            return None

        img_size = obtain_augmentation_size(data_dict, {"size": self.size})
        assert isinstance(img_size, (tuple, omegaconf.listconfig.ListConfig)), (
            f"Arg size in resize should be a tuple, get {type(img_size)}, {img_size}"
        )
        img_w, img_h = img_size
        orig_w, orig_h = meta_dict["width"], meta_dict["height"]

        scaling_ratio = min((img_w / orig_w), (img_h / orig_h))
        target_size = (int(scaling_ratio * orig_h + 0.5), int(scaling_ratio * orig_w + 0.5))
        assert target_size[0] <= img_h and target_size[1] <= img_w, (
            f"Resize error. orig {(orig_w, orig_h)} desire {img_size} compute {target_size}"
        )
        return [Resize(target_size)]

    def _sample_frame_indices(self, decoder_len: int, min_stride_override: int | None = None) -> tuple[list[int], int]:
        min_stride = int(min_stride_override) if min_stride_override is not None else self.min_stride
        max_stride = max(self.max_stride, min_stride)
        stride = self._sample_stride_with_bias(max_stride, min_stride)
        frame_indices = np.arange(0, decoder_len, stride).tolist()
        max_num_frames = min(len(frame_indices), self.args.get("max_num_frames", 1000))
        if max_num_frames < 1:
            return [], stride

        # Wan VAE temporal compression expects 1 + 4N video frames.
        num_video_frames = 1 + 4 * ((max_num_frames - 1) // 4)
        return frame_indices[:num_video_frames], stride

    def _probe_video_len(self, video: bytes) -> int:
        video_decoder = VideoDecoder(
            video,
            seek_mode=self.seek_mode,
            num_ffmpeg_threads=self.video_decode_num_threads,
        )
        try:
            return len(video_decoder)
        finally:
            del video_decoder

    def _decode_frames_at(
        self,
        video: bytes,
        frame_indices: list[int],
        transforms: list[Resize] | None = None,
    ) -> torch.Tensor:  # returns [C,T,H,W]
        video_decoder, post_decode_transforms = _create_video_decoder(
            video,
            self.seek_mode,
            self.video_decode_num_threads,
            transforms,
        )
        try:
            frame_batch = video_decoder.get_frames_at(frame_indices)
            frames = frame_batch.data  # [T,C,H,W]
            frames = _apply_post_decode_transforms(frames, post_decode_transforms)  # [T,C,H,W]
            frames = frames.permute(1, 0, 2, 3)  # [C,T,H,W]
        finally:
            del video_decoder
        return frames  # [C,T,H,W]

    def __call__(self, data_dict: dict) -> dict | None:
        try:
            meta_dict = data_dict[self.meta_key]
            video = data_dict[self.video_key]
        except Exception:
            log.warning(
                f"Cannot find video. url: {data_dict['__url__']}, key: {data_dict['__key__']}",
                rank0_only=False,
            )
            return None

        if not self._validate_and_probe(video, meta_dict, data_dict):
            return None

        rgb_transform = self._build_rgb_decode_transform(data_dict, meta_dict)
        control_videos: dict[str, bytes] = {}
        try:
            decoder_len = self._probe_video_len(video)

            control_decoder_lens = []
            for control_video_key in self.control_video_keys:
                control_video = data_dict.get(control_video_key)
                if not isinstance(control_video, bytes):
                    log.warning(
                        f"VideoTransferAlignedFullFramesParsing: missing bytes for {control_video_key}. "
                        f"url: {data_dict['__url__']}, key: {data_dict['__key__']}",
                        rank0_only=False,
                    )
                    return None
                control_videos[control_video_key] = control_video
                control_decoder_lens.append(self._probe_video_len(control_video))

            # Precomputed control streams can be one frame shorter than RGB; sample
            # only frames present in every stream to keep all modalities aligned.
            aligned_decoder_len = min([decoder_len, *control_decoder_lens]) if control_decoder_lens else decoder_len

            min_stride_override = data_dict.pop(self.min_stride_key, None)
            frame_indices, stride = self._sample_frame_indices(
                aligned_decoder_len, min_stride_override=min_stride_override
            )
            if len(frame_indices) == 0:
                log.warning(
                    f"VideoTransferAlignedFullFramesParsing: no valid frame indices. "
                    f"url: {data_dict['__url__']}, key: {data_dict['__key__']}",
                    rank0_only=False,
                )
                return None

            video_frames = self._decode_frames_at(video, frame_indices, rgb_transform)  # [C,T,H,W]
        except Exception as e:
            log.warning(
                f"Failed to decode RGB video. url: {data_dict['__url__']}, key: {data_dict['__key__']}, error: {e}",
                rank0_only=False,
            )
            return None

        base_video_info = {
            "frame_start": frame_indices[0],
            "frame_end": frame_indices[-1],
            "frame_indices": frame_indices,
            "num_frames": len(frame_indices),
            "fps": meta_dict["framerate"],
            "conditioning_fps": meta_dict["framerate"] / stride,
            "num_multiplier": stride,
            "n_orig_video_frames": decoder_len,
        }
        data_dict[self.video_key] = {
            **base_video_info,
            "video": video_frames,  # [C,T,H,W]
        }

        for control_video_key, control_video in control_videos.items():
            try:
                control_frames = self._decode_frames_at(control_video, frame_indices)  # [C,T,H,W]
            except Exception as e:
                log.warning(
                    f"Failed to decode {control_video_key}. "
                    f"url: {data_dict['__url__']}, key: {data_dict['__key__']}, error: {e}",
                    rank0_only=False,
                )
                return None
            data_dict[control_video_key] = {
                **base_video_info,
                "video": control_frames,  # [C,T,H,W]
            }

        return data_dict


class VideoTransferAlignedLegacyChunkParsing(VideoTransferAlignedFullFramesParsing):
    """Decode legacy caption-window transfer streams with shared RGB/control frame indices."""

    def __init__(self, input_keys: list, output_keys: Optional[list] = None, args: Optional[dict] = None) -> None:
        super().__init__(input_keys=input_keys, output_keys=output_keys, args=args)
        self.key_for_caption = self.args["key_for_caption"]
        assert self.key_for_caption in [
            "t2w_windows",
            "i2w_windows_later_frames",
        ], "key_for_caption must be either t2w_windows or i2w_windows_later_frames"
        self.min_duration = self.args["min_duration"]
        self.num_frames = self.args["num_video_frames"]
        self.use_native_fps = self.args["use_native_fps"]
        self.use_original_fps = self.args["use_original_fps"]
        self.use_dynamic_fps = self.args.get("use_dynamic_fps", False)
        self.low_fps_bias = self.args.get("low_fps_bias", 0.5)
        assert 0.0 <= self.low_fps_bias <= 1.0, f"low_fps_bias must be in [0, 1], got {self.low_fps_bias}"
        mode_count = sum([self.use_dynamic_fps, self.use_native_fps, self.use_original_fps])
        assert mode_count <= 1, (
            f"Only one FPS mode can be enabled at a time. Got: "
            f"use_dynamic_fps={self.use_dynamic_fps}, "
            f"use_native_fps={self.use_native_fps}, "
            f"use_original_fps={self.use_original_fps}"
        )
        self.allowed_num_multiplers = self.args.get("allowed_num_multiplers", list(range(1, 100)))
        if self.num_frames > 0:
            self.sampler = UniformTemporalSubsample(self.num_frames)

    def _sample_legacy_stride_with_bias(self, max_stride: int) -> int:
        if max_stride == 1:
            return 1

        strides = np.arange(1, max_stride + 1)
        weights = np.linspace(1 - self.low_fps_bias, self.low_fps_bias, max_stride)
        weights = np.maximum(weights, 0.01)
        probs = weights / weights.sum()
        return int(np.random.choice(strides, p=probs))

    def _decode_all_streams_at(
        self,
        video: bytes,
        control_videos: dict[str, bytes],
        frame_indices: list[int],
        rgb_transform: list[Resize] | None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        video_frames = self._decode_frames_at(video, frame_indices, rgb_transform)  # [C,T,H,W]
        control_frames_by_key = {
            control_video_key: self._decode_frames_at(control_video, frame_indices)  # [C,T,H,W]
            for control_video_key, control_video in control_videos.items()
        }
        return video_frames, control_frames_by_key

    def _subsample_all_streams(
        self, video_frames: torch.Tensor, control_frames_by_key: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        video_frames = rearrange(
            self.sampler(rearrange(video_frames, "c t h w -> t c h w")), "t c h w -> c t h w"
        )  # [C,T,H,W]
        control_frames_by_key = {
            key: rearrange(self.sampler(rearrange(frames, "c t h w -> t c h w")), "t c h w -> c t h w")
            for key, frames in control_frames_by_key.items()
        }  # [C,T,H,W]
        return video_frames, control_frames_by_key

    def __call__(self, data_dict: dict) -> dict | None:
        try:
            meta_dict = data_dict[self.meta_key]
            video = data_dict[self.video_key]
        except Exception:
            log.warning(
                f"Cannot find video. url: {data_dict['__url__']}, key: {data_dict['__key__']}",
                rank0_only=False,
            )
            return None

        if not self._validate_and_probe(video, meta_dict, data_dict):
            return None

        control_videos: dict[str, bytes] = {}
        control_decoder_lens: list[int] = []
        try:
            decoder_len = self._probe_video_len(video)
            for control_video_key in self.control_video_keys:
                control_video = data_dict.get(control_video_key)
                if not isinstance(control_video, bytes):
                    log.warning(
                        f"VideoTransferAlignedLegacyChunkParsing: missing bytes for {control_video_key}. "
                        f"url: {data_dict['__url__']}, key: {data_dict['__key__']}",
                        rank0_only=False,
                    )
                    return None
                control_videos[control_video_key] = control_video
                control_decoder_lens.append(self._probe_video_len(control_video))
        except Exception as e:
            log.warning(
                f"Failed to probe video streams. url: {data_dict['__url__']}, key: {data_dict['__key__']}, error: {e}",
                rank0_only=False,
            )
            return None

        aligned_decoder_len = min([decoder_len, *control_decoder_lens]) if control_decoder_lens else decoder_len
        options: list = list((i, item) for i, item in enumerate(meta_dict[self.key_for_caption]))
        if len(options) > 1:
            options = options[:-1]
        random.shuffle(options)

        rgb_transform = self._build_rgb_decode_transform(data_dict, meta_dict)
        video_frames = None
        control_frames_by_key: dict[str, torch.Tensor] = {}
        dynamic_conditioning_fps = None
        num_multiplier: float | int = 1
        frame_indices: list[int] = []
        chunk_index = 0
        start_frame = 0
        end_frame = 0

        for chunk_index, option in options:
            start_frame = int(option["start_frame"])
            end_frame = min(int(option["end_frame"]), aligned_decoder_len)
            if (end_frame - start_frame) < self.min_duration * meta_dict["framerate"]:
                continue
            if self.use_native_fps or self.use_original_fps or self.use_dynamic_fps:
                if "alpamayo" in data_dict["__url__"].root:
                    start_frame += 5
                if (end_frame - start_frame) < self.num_frames:
                    continue
                total_frames = end_frame - start_frame
                if self.use_dynamic_fps:
                    max_stride = total_frames // self.num_frames
                    if max_stride < 1:
                        continue
                    num_multiplier = self._sample_legacy_stride_with_bias(max_stride)
                    dynamic_conditioning_fps = meta_dict["framerate"] / num_multiplier
                elif self.use_native_fps:
                    num_multiplier = total_frames // self.num_frames
                    if num_multiplier not in self.allowed_num_multiplers:
                        continue
                else:
                    num_multiplier = 1

                expected_length = self.num_frames * int(num_multiplier)
                if total_frames < expected_length:
                    continue
                frame_start = start_frame + (total_frames - expected_length) // 2
                frame_end = frame_start + expected_length
                frame_indices = list(range(frame_start, frame_end, int(num_multiplier)))
            else:
                frame_indices = list(range(start_frame, end_frame))
                if "alpamayo" in data_dict["__url__"].root:
                    if len(frame_indices) < 5:
                        continue
                    frame_indices = frame_indices[5:]
                    start_frame += 5

            try:
                video_frames, control_frames_by_key = self._decode_all_streams_at(
                    video, control_videos, frame_indices, rgb_transform
                )  # [C,T,H,W]
            except Exception as e:
                log.warning(
                    f"Failed to decode aligned video streams. "
                    f"url: {data_dict['__url__']}, key: {data_dict['__key__']}, error: {e}",
                    rank0_only=False,
                )
                return None
            break

        if video_frames is None:
            log.warning(
                f"No valid video frames found, return None. url: {data_dict['__url__']}, key: {data_dict['__key__']}",
                rank0_only=False,
            )
            return None

        if self.num_frames > 0 and not (self.use_dynamic_fps or self.use_native_fps or self.use_original_fps):
            video_frames, control_frames_by_key = self._subsample_all_streams(
                video_frames, control_frames_by_key
            )  # [C,T,H,W]
            num_multiplier = (end_frame - start_frame) / self.num_frames


        # variable-length fields like ``frame_indices`` here -- ``video_flatten_keys`` in
        # ``get_video_transfer_augmentor`` lists ``frame_indices``, and surfacing a
        # per-sample list there would crash ``custom_collate_fn`` (default_collate requires
        # equal-size elements across the batch).
        base_video_info = {
            "fps": meta_dict["framerate"],
            "n_orig_video_frames": meta_dict["nb_frames"],
            "chunk_index": chunk_index,
            "frame_start": start_frame,
            "frame_end": end_frame,
            "num_frames": end_frame - start_frame,
            "num_multiplier": num_multiplier,
            "conditioning_fps": dynamic_conditioning_fps or meta_dict["framerate"] / num_multiplier,
        }
        data_dict[self.video_key] = {
            **base_video_info,
            "video": video_frames,  # [C,T,H,W]
        }
        for control_video_key, control_frames in control_frames_by_key.items():
            data_dict[control_video_key] = {
                **base_video_info,
                "video": control_frames,  # [C,T,H,W]
            }
        return data_dict


class VideoTransferAlignedChunkedFramesParsing(VideoTransferAlignedFullFramesParsing):
    """Decode RGB and aligned control videos for a selected caption chunk."""

    def _sample_frame_indices_for_chunk(
        self,
        decoder_len: int,
        chunk_start: int,
        chunk_end: int,
        min_stride_override: int | None = None,
    ) -> tuple[list[int], int]:
        chunk_start = max(0, min(chunk_start, decoder_len))
        chunk_end = max(chunk_start, min(chunk_end, decoder_len))
        if chunk_end <= chunk_start:
            return [], 0

        min_stride = int(min_stride_override) if min_stride_override is not None else self.min_stride
        max_stride = max(self.max_stride, min_stride)
        stride = self._sample_stride_with_bias(max_stride, min_stride)
        frame_indices = np.arange(chunk_start, chunk_end, stride).tolist()
        max_num_frames = min(len(frame_indices), self.args.get("max_num_frames", 1000))
        if max_num_frames < 1:
            return [], stride

        # Wan VAE temporal compression expects 1 + 4N video frames.
        num_video_frames = 1 + 4 * ((max_num_frames - 1) // 4)
        return frame_indices[:num_video_frames], stride

    def __call__(self, data_dict: dict) -> dict | None:
        try:
            meta_dict = data_dict[self.meta_key]
            video = data_dict[self.video_key]
        except Exception:
            log.warning(
                f"Cannot find video. url: {data_dict['__url__']}, key: {data_dict['__key__']}",
                rank0_only=False,
            )
            return None

        if not self._validate_and_probe(video, meta_dict, data_dict):
            return None

        if "chunk_start_frame" not in data_dict or "chunk_end_frame" not in data_dict:
            log.warning(
                f"VideoTransferAlignedChunkedFramesParsing: missing chunk_start_frame/chunk_end_frame. "
                f"url: {data_dict['__url__']}, key: {data_dict['__key__']}",
                rank0_only=False,
            )
            return None
        chunk_start = int(data_dict["chunk_start_frame"])
        chunk_end = int(data_dict["chunk_end_frame"])

        rgb_transform = self._build_rgb_decode_transform(data_dict, meta_dict)
        control_videos: dict[str, bytes] = {}
        try:
            decoder_len = self._probe_video_len(video)

            control_decoder_lens = []
            for control_video_key in self.control_video_keys:
                control_video = data_dict.get(control_video_key)
                if not isinstance(control_video, bytes):
                    log.warning(
                        f"VideoTransferAlignedChunkedFramesParsing: missing bytes for {control_video_key}. "
                        f"url: {data_dict['__url__']}, key: {data_dict['__key__']}",
                        rank0_only=False,
                    )
                    return None
                control_videos[control_video_key] = control_video
                control_decoder_lens.append(self._probe_video_len(control_video))

            # Clamp the caption chunk to frames available in every loaded stream.
            aligned_decoder_len = min([decoder_len, *control_decoder_lens]) if control_decoder_lens else decoder_len
            min_stride_override = data_dict.pop(self.min_stride_key, None)
            frame_indices, stride = self._sample_frame_indices_for_chunk(
                aligned_decoder_len,
                chunk_start,
                chunk_end,
                min_stride_override=min_stride_override,
            )
            if len(frame_indices) == 0:
                log.warning(
                    f"VideoTransferAlignedChunkedFramesParsing: empty chunk after clamping/stride. "
                    f"chunk=[{chunk_start},{chunk_end}), aligned_decoder_len={aligned_decoder_len}, "
                    f"url: {data_dict['__url__']}, key: {data_dict['__key__']}",
                    rank0_only=False,
                )
                return None

            video_frames = self._decode_frames_at(video, frame_indices, rgb_transform)  # [C,T,H,W]
        except Exception as e:
            log.warning(
                f"Failed to decode RGB video. url: {data_dict['__url__']}, key: {data_dict['__key__']}, error: {e}",
                rank0_only=False,
            )
            return None

        base_video_info = {
            "frame_start": frame_indices[0],
            "frame_end": frame_indices[-1],
            "frame_indices": frame_indices,
            "num_frames": len(frame_indices),
            "fps": meta_dict["framerate"],
            "conditioning_fps": meta_dict["framerate"] / stride if stride > 0 else meta_dict["framerate"],
            "num_multiplier": stride,
            "n_orig_video_frames": decoder_len,
        }
        data_dict[self.video_key] = {
            **base_video_info,
            "video": video_frames,  # [C,T,H,W]
        }

        for control_video_key, control_video in control_videos.items():
            try:
                control_frames = self._decode_frames_at(control_video, frame_indices)  # [C,T,H,W]
            except Exception as e:
                log.warning(
                    f"Failed to decode {control_video_key}. "
                    f"url: {data_dict['__url__']}, key: {data_dict['__key__']}, error: {e}",
                    rank0_only=False,
                )
                return None
            data_dict[control_video_key] = {
                **base_video_info,
                "video": control_frames,  # [C,T,H,W]
            }

        data_dict.pop("chunk_start_frame", None)
        data_dict.pop("chunk_end_frame", None)
        return data_dict
