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

from collections.abc import Callable
from typing import Optional

import numpy as np
import omegaconf
import torch
from torchcodec.decoders import VideoDecoder
from torchvision.transforms.v2 import Resize

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

    def _sample_frame_indices(self, decoder_len: int) -> tuple[list[int], int]:
        stride = self._sample_stride_with_bias(self.max_stride, self.min_stride)
        frame_indices = np.arange(0, decoder_len, stride).tolist()
        max_num_frames = min(len(frame_indices), self.args.get("max_num_frames", 1000))
        if max_num_frames < 1:
            return [], stride

        # Wan VAE temporal compression expects 1 + 4N video frames.
        num_video_frames = 1 + 4 * ((max_num_frames - 1) // 4)
        return frame_indices[:num_video_frames], stride

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
        try:
            rgb_decoder = VideoDecoder(
                video,
                seek_mode=self.seek_mode,
                num_ffmpeg_threads=self.video_decode_num_threads,
            )
            decoder_len = len(rgb_decoder)
            del rgb_decoder

            frame_indices, stride = self._sample_frame_indices(decoder_len)
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

        for control_video_key in self.control_video_keys:
            control_video = data_dict.get(control_video_key)
            if not isinstance(control_video, bytes):
                log.warning(
                    f"VideoTransferAlignedFullFramesParsing: missing bytes for {control_video_key}. "
                    f"url: {data_dict['__url__']}, key: {data_dict['__key__']}",
                    rank0_only=False,
                )
                return None
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
