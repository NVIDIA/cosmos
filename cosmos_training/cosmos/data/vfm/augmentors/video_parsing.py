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
from typing import Optional

import numpy as np
import omegaconf
import torch
from einops import rearrange
from torchcodec.decoders import AudioDecoder, VideoDecoder
from torchvision.transforms.v2 import Resize, UniformTemporalSubsample

from cosmos.data.imaginaire.webdataset.augmentors.augmentor import Augmentor
from cosmos.data.imaginaire.webdataset.augmentors.image.misc import obtain_augmentation_size
from cosmos.utils import log
from cosmos.data.vfm.utils import VIDEO_RES_SIZE_INFO

# Map dataset_resolution_type to resolution tier key in VIDEO_RES_SIZE_INFO
_DATASET_RESOLUTION_TIER: dict[str, str] = {"gt480p": "480", "gt720p": "720", "gt1080p": "1080"}

_MIN_FPS = 10
_MAX_FPS = 60


class VideoParsing(Augmentor):
    """
    This augmentor is used to parse the video bytes and get the video frames.
    the return dict is back-compatible with old datasets, which video decoding happens in the decoder stage.

    Now uses torchcodec instead of decord for video decoding, with optional audio extraction.
    """

    def __init__(self, input_keys: list, output_keys: Optional[list] = None, args: Optional[dict] = None) -> None:
        super().__init__(input_keys, output_keys, args)
        assert len(input_keys) == 2, "VideoParsing augmentor only supports two input keys"
        self.meta_key = input_keys[0]
        self.video_key = input_keys[1]

        self.key_for_caption = args["key_for_caption"]
        assert self.key_for_caption in [
            "t2w_windows",
            "i2w_windows_later_frames",
        ], "key_for_caption must be either t2w_windows or i2w_windows_later_frames"
        self.min_duration = args["min_duration"]
        self.min_fps = args["min_fps"]
        self.max_fps = args["max_fps"]
        self.num_frames = args["num_video_frames"]
        self.use_native_fps = args["use_native_fps"]  # orginal fps if (total_frames // self.num_frames == 1).
        # a list of allowed num_multiplers (how many frames are skipped)
        # default is 1 - 100 which allows virtually any num_multipler possible
        self.allowed_num_multiplers = args.get("allowed_num_multiplers", list(range(1, 100)))
        log.info(f"allowed_num_multiplers in video_parsing with use_native_fps: {self.allowed_num_multiplers}")
        self.use_original_fps = args["use_original_fps"]  # use original fps without sampling

        # Dynamic FPS mode: sample stride from valid range based on video properties
        self.use_dynamic_fps = args.get("use_dynamic_fps", False)
        # low_fps_bias: 0.0 = favor original FPS (stride=1), 0.5 = uniform, 1.0 = favor slow-mo (high stride)
        self.low_fps_bias = args.get("low_fps_bias", 0.5)
        assert 0.0 <= self.low_fps_bias <= 1.0, f"low_fps_bias must be in [0, 1], got {self.low_fps_bias}"

        # Validate mutually exclusive modes
        mode_count = sum([self.use_dynamic_fps, self.use_native_fps, self.use_original_fps])
        assert mode_count <= 1, (
            f"Only one FPS mode can be enabled at a time. Got: "
            f"use_dynamic_fps={self.use_dynamic_fps}, "
            f"use_native_fps={self.use_native_fps}, "
            f"use_original_fps={self.use_original_fps}"
        )

        if self.use_dynamic_fps:
            log.info(
                f"use_dynamic_fps mode enabled: stride will be sampled from valid range per video "
                f"with low_fps_bias={self.low_fps_bias} (0.0=favor original FPS, 0.5=uniform, 1.0=favor slow-mo)"
            )

        if self.use_native_fps or self.use_original_fps:
            assert self.num_frames > 0, "num_frames must be greater than 0 when use_native_fps is True"
        if self.use_dynamic_fps:
            assert self.num_frames > 0, "num_frames must be greater than 0 when use_dynamic_fps is True"
        if self.num_frames > 0:
            self.sampler = UniformTemporalSubsample(self.num_frames)
        self.video_decode_num_threads = args.get("video_decode_num_threads", 1)

        # Audio extraction parameters
        self.extract_audio = args.get("extract_audio", False)
        self.audio_sample_rate = args.get("audio_sample_rate", 44100)
        self.seek_mode = args.get("seek_mode", "exact")

    def _extract_audio_chunk(
        self, video_bytes: bytes, video_fps: float, frame_indices: list[int]
    ) -> torch.Tensor | None:  # returns [C,N_audio] or None
        """
        Extract audio chunk corresponding to the given frame indices.

        Args:
            video_bytes: Raw video bytes
            video_fps: Video frames per second
            frame_indices: List of frame indices being extracted

        Returns:
            Audio tensor of shape (C, N) or None if audio extraction fails
        """
        try:
            # Create audio decoder
            audio_decoder = AudioDecoder(video_bytes)

            # Calculate time range for audio corresponding to video frames
            time_start = frame_indices[0] / video_fps
            time_end = (frame_indices[-1] + 1) / video_fps  # +1 to include the last frame's duration

            # Get audio samples for the specific time range
            audio_metadata = audio_decoder.metadata
            orig_sample_rate = audio_metadata.sample_rate

            audio_samples = audio_decoder.get_samples_played_in_range(start_seconds=time_start, stop_seconds=time_end)
            audio_chunk = audio_samples.data  # [C,N_orig]

            # Resample if needed
            if orig_sample_rate != self.audio_sample_rate:
                import librosa

                audio_np = audio_chunk.numpy()
                resampled_audio_np = librosa.resample(
                    audio_np, orig_sr=orig_sample_rate, target_sr=self.audio_sample_rate, axis=-1
                )
                audio_chunk = torch.from_numpy(resampled_audio_np)  # [C,N_resampled]

            # Clean up audio decoder
            del audio_decoder

            return audio_chunk

        except Exception as e:
            log.warning(f"Failed to extract audio: {e}", rank0_only=False)
            return None

    def _sample_stride_with_bias(self, max_stride: int) -> int:
        """Sample a stride from [1, max_stride] with bias controlled by low_fps_bias.

        Args:
            max_stride: Maximum valid stride value.

        Returns:
            Sampled stride value.

        The bias controls the probability distribution:
        - low_fps_bias=0.0: Favor stride=1 (original FPS)
        - low_fps_bias=0.5: Uniform distribution
        - low_fps_bias=1.0: Favor high strides (slow-mo / lower FPS)
        """
        if max_stride == 1:
            return 1

        # Linear interpolation from (1 - bias) to bias, clamped to min 0.01
        strides = np.arange(1, max_stride + 1)
        weights = np.linspace(1 - self.low_fps_bias, self.low_fps_bias, max_stride)
        weights = np.maximum(weights, 0.01)
        probs = weights / weights.sum()

        return int(np.random.choice(strides, p=probs))

    def __call__(self, data_dict: dict) -> dict | None:
        try:
            meta_dict = data_dict[self.meta_key]
            video = data_dict[self.video_key]
        except Exception as e:
            log.warning(
                f"Cannot find video. url: {data_dict['__url__']}, key: {data_dict['__key__']}", rank0_only=False
            )
            return None

        if not isinstance(video, bytes):
            return data_dict

        video_info = {
            "fps": meta_dict["framerate"],
            "n_orig_video_frames": meta_dict["nb_frames"],
        }

        if video_info["fps"] < self.min_fps:
            log.warning(f"Video FPS {video_info['fps']} is less than min_fps {self.min_fps}", rank0_only=False)
            return None
        if video_info["fps"] > self.max_fps:
            log.warning(f"Video FPS {video_info['fps']} is greater than max_fps {self.max_fps}", rank0_only=False)
            return None

        options: list = list((i, item) for i, item in enumerate(meta_dict[self.key_for_caption]))

        # Skip the last window if possible.
        # All windows except the last are 5 seconds long. The last window has a duration in the range [2.5s, 7.5), which is less preferred.
        if len(options) > 1:
            options = options[:-1]

        # shuffle options
        random.shuffle(options)
        video_frames = None
        dynamic_conditioning_fps = None  # Track conditioning FPS for dynamic mode
        for chunk_index, option in options:
            start_frame = option["start_frame"]
            end_frame = option["end_frame"]
            if (end_frame - start_frame) < self.min_duration * video_info["fps"]:
                continue

            if self.use_native_fps or self.use_original_fps or self.use_dynamic_fps:
                if (end_frame - start_frame) < self.num_frames:
                    continue

            # Create video decoder with torchcodec (directly from bytes)
            video_decoder = VideoDecoder(
                video, seek_mode=self.seek_mode, num_ffmpeg_threads=self.video_decode_num_threads
            )

            if self.use_dynamic_fps or self.use_native_fps or self.use_original_fps:
                # Shared: Handle alpamayo - skip first 5 frames
                if "alpamayo" in data_dict["__url__"].root:
                    start_frame += 5
                if (end_frame - start_frame) < self.num_frames:
                    continue

                total_frames = end_frame - start_frame

                # Compute num_multiplier based on mode
                if self.use_dynamic_fps:
                    # Dynamic FPS mode: compute valid strides and sample with bias
                    max_stride = total_frames // self.num_frames
                    if max_stride < 1:
                        # Not enough frames even for stride=1, skip this chunk
                        continue

                    # Sample stride with low_fps_bias controlling the distribution
                    num_multiplier = self._sample_stride_with_bias(max_stride)

                    # Compute conditioning FPS based on sampled stride
                    dynamic_conditioning_fps = video_info["fps"] / num_multiplier

                    fps_mode_desc = (
                        "original_fps (contiguous)" if num_multiplier == 1 else f"subsampled (stride={num_multiplier})"
                    )
                    log.info(
                        f"Dynamic FPS mode: video_fps={video_info['fps']}, total_frames={total_frames}, "
                        f"max_stride={max_stride}, sampled_stride={num_multiplier}, "
                        f"conditioning_fps={dynamic_conditioning_fps:.2f}, mode={fps_mode_desc}, "
                        f"low_fps_bias={self.low_fps_bias}",
                        rank0_only=False,
                    )
                elif self.use_native_fps:
                    # take mid self.num_frames frames from start frame to end frame.
                    # always try lower fps if possible.
                    num_multiplier = total_frames // self.num_frames
                    if num_multiplier not in self.allowed_num_multiplers:
                        log.debug(
                            f"Skipping chunk (native_fps): stride not allowed. num_multiplier={num_multiplier}, allowed={self.allowed_num_multiplers}"
                        )
                        continue
                else:  # self.use_original_fps
                    # Original FPS mode: no frame skipping
                    num_multiplier = 1

                # Shared: Check if we have enough frames for the selected stride
                expected_length = self.num_frames * num_multiplier
                if total_frames < expected_length:
                    log.info(
                        f"Skipping chunk: not enough frames for stride. total_frames={total_frames}, expected={expected_length}, num_multiplier={num_multiplier}",
                        rank0_only=False,
                    )
                    continue

                # Shared: Select frames from the center of the window
                _start_frame = start_frame + (total_frames - expected_length) // 2
                _end_frame = _start_frame + expected_length
                frame_indices = list(range(_start_frame, _end_frame, num_multiplier))
                assert len(frame_indices) == self.num_frames, "frame_indices length is not equal to num_frames"

                # Decode frames with torchcodec
                frame_batch = video_decoder.get_frames_at(frame_indices)
                video_frames = frame_batch.data  # [T,C,H,W]
                video_frames = video_frames.permute(1, 0, 2, 3)  # [C,T,H,W]

                # Clean up video decoder
                del video_decoder

                # Extract audio if requested
                audio_chunk = None
                if self.extract_audio:
                    audio_chunk = self._extract_audio_chunk(video, video_info["fps"], frame_indices)  # [C,N_audio]

                break

            else:
                frame_indices = list(range(start_frame, end_frame))
                num_multiplier = 1  # No frame skipping in this block of code.

                # online hot-fix for alpamayo data. Skip the first 5 frames as there is chance that the first five frames contain black frames.
                if "alpamayo" in data_dict["__url__"].root:
                    assert len(frame_indices) >= 5, (
                        "Getting less than 5 frames for alpamayo videos. There is no way to skip the first five frames."
                    )
                    frame_indices = frame_indices[5:]
                    start_frame += 5

                # Decode frames with torchcodec
                try:
                    frame_batch = video_decoder.get_frames_at(frame_indices)
                except Exception as e:
                    # Some segmentation videos for Transfer are not long enough as the target video, skip them.
                    log.warning(
                        f"Video is not long enough, return None. url: {data_dict['__url__']}, key: {data_dict['__key__']}, error: {e}, start_frame: {start_frame}, end_frame: {end_frame}, frame_indices: {frame_indices}",
                        rank0_only=False,
                    )
                    return None
                video_frames = frame_batch.data  # [T,C,H,W]
                video_frames = video_frames.permute(1, 0, 2, 3)  # [C,T,H,W]

                # Clean up video decoder
                del video_decoder

                # Extract audio if requested
                audio_chunk = None
                if self.extract_audio:
                    audio_chunk = self._extract_audio_chunk(video, video_info["fps"], frame_indices)  # [C,N_audio]

                break

        if video_frames is None:
            log.warning(
                f"No valid video frames found, return None. url: {data_dict['__url__']}, key: {data_dict['__key__']}",
                rank0_only=False,
            )
            return None

        video_info["chunk_index"] = chunk_index
        video_info["frame_start"] = start_frame
        video_info["frame_end"] = end_frame
        video_info["num_frames"] = end_frame - start_frame  # type: ignore
        if self.num_frames > 0 and not (self.use_dynamic_fps or self.use_native_fps or self.use_original_fps):
            # Uniform temporal subsampling mode (default when no FPS mode is enabled)
            video_frames = rearrange(
                self.sampler(rearrange(video_frames, "c t h w -> t c h w")), "t c h w -> c t h w"
            )  # [C,T_sub,H,W] where T_sub = self.num_frames
            num_multiplier = (
                end_frame - start_frame
            ) / self.num_frames  # Specifically for the uniform temporal subsampling case.

        video_info["video"] = video_frames
        video_info["num_multiplier"] = num_multiplier  # Store the frame skipping multiplier


        # 1. Our video parser stores the original video FPS of the video.
        # 2. We have multiple modes of frame selection -- consecutive chunk of frames or subsampled frames.
        # Here's what we do in each case:
        #
        # A. Dynamic FPS mode (use_dynamic_fps=True):
        #    - We compute max possible stride based on total_frames // num_frames.
        #    - We sample a stride uniformly from [1, max_stride].
        #    - We compute conditioning_fps = native_fps / stride.
        #    - This gives us a diverse range of effective FPS values.
        #
        # B. Consecutive chunk of frames (use_original_fps=True):
        #    - We use the stored FPS and the number of frames in the video.
        #    - We calculate the duration in seconds using the above two values.
        #    - conditioning_fps = native_fps (num_multiplier=1)
        #
        # C. Subsampled frames (use_native_fps=True or uniform subsampling):
        #    - We check the skipping_rate (1 / num_multiplier) in case of subsampling.
        #    - We adjust the conditioning FPS by the skipping_rate (faithful to original video's motion).
        #    - conditioning_fps = native_fps / num_multiplier
        #    - We calculate the duration in seconds using the adjusted conditioning FPS and the number of frames.
        if dynamic_conditioning_fps is not None:
            # Dynamic FPS mode: use the pre-computed conditioning FPS
            video_info["conditioning_fps"] = dynamic_conditioning_fps
        else:
            # Other modes: compute effective FPS from stride
            video_info["conditioning_fps"] = (
                video_info["fps"] / num_multiplier
            )  # Effective FPS for RoPE modulation and text timestamps

        # Add audio if extracted
        if audio_chunk is not None:
            video_info["audio"] = audio_chunk
            video_info["audio_sample_rate"] = self.audio_sample_rate

        # update data_dict, make it back-compatible with old datasets, which video decoding happens in the decoder stage.
        data_dict[self.video_key] = video_info

        return data_dict


class VideoParsingWithFullFrames(Augmentor):
    """
    This augmentor is used to parse the video bytes and get the video frames.
    The caption is assumed to be for the entire video frames, rather than VideoParsing which assume captions are for a specific chunk of frames
    """

    def __init__(self, input_keys: list, output_keys: Optional[list] = None, args: Optional[dict] = None) -> None:
        super().__init__(input_keys, output_keys, args)
        assert len(input_keys) == 2, "VideoParsingWithFullFrames augmentor only supports two input keys"
        self.meta_key = input_keys[0]
        self.video_key = input_keys[1]
        self.args = args

        # Dynamic FPS mode options
        # If use_dynamic_fps=True, then we sample fps from a valid range of values.
        # If use_dynamic_fps=False, then we use the original fps of the video (no frame skipping).
        self.use_dynamic_fps = args.get("use_dynamic_fps", False)
        # low_fps_bias: 0.0 = favor original FPS (stride=1), 0.5 = uniform, 1.0 = favor slow-mo (high stride)
        self.max_stride = args.get("max_stride", 3)
        self.min_stride = args.get("min_stride", 1)
        assert self.max_stride >= self.min_stride, (
            f"max_stride ({self.max_stride}) must be >= min_stride ({self.min_stride})"
        )
        self.min_fps = args.get("min_fps", _MIN_FPS)
        self.max_fps = args.get("max_fps", _MAX_FPS)
        if self.use_dynamic_fps:
            log.info(f"use_dynamic_fps mode enabled: stride will be sampled from valid range per video ")

        self.video_decode_num_threads = args.get("video_decode_num_threads", 1)
        self.seek_mode = args.get("seek_mode", "exact")

        self.size = args.get("size", None)
        self.perform_resize = self.size is not None

        # Audio extraction parameters
        self.extract_audio = args.get("extract_audio", False)
        self.audio_sample_rate = args.get("audio_sample_rate", 48000)
        # When True, emit placeholder sound=None and audio_sample_rate
        # without extracting audio.  Keeps output keys consistent across
        # datasets that share the same dataloader (some with audio, some
        # without).
        self.emit_placeholder_sound = args.get("emit_placeholder_sound", False)

        # Resolution filter: when not "all", skip samples whose (width, height) are below the
        # minimum for this aspect ratio in VIDEO_RES_SIZE_INFO[tier].
        self.dataset_resolution_type = args.get("dataset_resolution_type", "all")
        self.resolution_tier = _DATASET_RESOLUTION_TIER.get(self.dataset_resolution_type)

    def _sample_stride_with_bias(self, max_stride: int, min_stride: int = 1) -> int:
        """Sample a stride from [min_stride, max_stride] with bias controlled by low_fps_bias.

        Args:
            max_stride: Maximum valid stride value.
            min_stride: Minimum valid stride value.

        Returns:
            Sampled stride value.
            max_stride=3, min_stride=1, probs = [0.86681333, 0.11731043, 0.01587624]
            These values are chosen to approximately match our old ablations.
            TODO @pchattopadhy: Do ablations with this scheme
        """
        assert max_stride >= min_stride, f"max_stride ({max_stride}) must be >= min_stride ({min_stride})"
        if max_stride == min_stride:
            return min_stride

        # Samples native fps stride mostly and picks low fps with some probability.
        strides = np.arange(min_stride, max_stride + 1)
        weights = np.exp(-2 * strides)
        probs = weights / weights.sum()
        return int(np.random.choice(strides, p=probs))

    def _validate_and_probe(self, video: Optional[bytes], meta_dict: dict, data_dict: dict) -> bool:
        """Validate video bytes, back-fill missing metadata via probing, and
        enforce fps/resolution filters.
        Returns True if the video is valid, False otherwise.
        """

        if not isinstance(video, bytes):
            raise ValueError(f"Video is not bytes. url: {data_dict['__url__']}, key: {data_dict['__key__']}")

        if len(video) == 0:
            log.warning(
                f"Empty video bytes. url: {data_dict['__url__']}, key: {data_dict['__key__']}", rank0_only=False
            )
            return False

        # Back-fill missing metadata keys (width, height, framerate, nb_frames) by probing the
        # video stream header.  Also probe when the sidecar framerate looks abnormal to verify
        # against the actual video stream.
        _needs_probe = any(k not in meta_dict for k in ("width", "height", "framerate", "nb_frames"))
        _metadata_fps = meta_dict.get("framerate", 0)
        _fps_suspicious = _metadata_fps > _MAX_FPS or _metadata_fps < _MIN_FPS
        _needs_probe = _needs_probe or _fps_suspicious
        if _needs_probe:
            _probe = VideoDecoder(video, seek_mode=self.seek_mode)
            meta_dict.setdefault("width", _probe.metadata.width)
            meta_dict.setdefault("height", _probe.metadata.height)
            meta_dict.setdefault("nb_frames", _probe.metadata.num_frames)
            meta_dict["framerate"] = _probe.metadata.average_fps
            del _probe

        # Skip videos with framerates outside [min_fps, max_fps]
        if meta_dict["framerate"] > self.max_fps:
            log.warning(
                f"Skipping video with framerate {meta_dict['framerate']} > max_fps {self.max_fps}. "
                f"url: {data_dict['__url__']}, key: {data_dict['__key__']}",
                rank0_only=False,
            )
            return False
        if meta_dict["framerate"] < self.min_fps:
            log.warning(
                f"Skipping video with framerate {meta_dict['framerate']} < min_fps {self.min_fps}. "
                f"url: {data_dict['__url__']}, key: {data_dict['__key__']}",
                rank0_only=False,
            )
            return False

        # Resolution check: skip sample if (width, height) are below the minimum for this aspect ratio
        width = meta_dict["width"]
        height = meta_dict["height"]
        aspect_ratio: str | None = None

        if "__url__" in data_dict:
            aspect_ratio = data_dict["__url__"].meta.opts["aspect_ratio"]

        # If the resolution of the video is smaller than the minimum resolution for the aspect ratio, skip the sample. This will ensure that we do not upsample any video.
        if self.resolution_tier is not None:
            min_w, min_h = VIDEO_RES_SIZE_INFO[self.resolution_tier][aspect_ratio]
            if width < min_w and height < min_h:
                return False

        return True

    def __call__(self, data_dict: dict) -> dict | None:

        # if in future we need to train with batch size > 1, need to pad frames
        try:
            meta_dict = data_dict[self.meta_key]
            video = data_dict[self.video_key]
        except Exception as e:
            log.warning(
                f"Cannot find video. url: {data_dict['__url__']}, key: {data_dict['__key__']}", rank0_only=False
            )
            return None

        if not self._validate_and_probe(video, meta_dict, data_dict):
            return None

        # Resize video frames if size is specified. This computes a scaling ratio that fits the
        # video within the target size bounds while preserving the original aspect ratio.
        # The resize transform is applied during decoding via VideoDecoder's transforms parameter.
        if self.perform_resize:
            img_size = obtain_augmentation_size(data_dict, {"size": self.size})
            assert isinstance(img_size, (tuple, omegaconf.listconfig.ListConfig)), (
                f"Arg size in resize should be a tuple, get {type(img_size)}, {img_size}"
            )
            img_w, img_h = img_size
            orig_w, orig_h = meta_dict["width"], meta_dict["height"]

            # Compute uniform scaling ratio to fit video within target bounds (aspect-ratio preserving)
            scaling_ratio = min((img_w / orig_w), (img_h / orig_h))
            target_size = (int(scaling_ratio * orig_h + 0.5), int(scaling_ratio * orig_w + 0.5))

            assert target_size[0] <= img_h and target_size[1] <= img_w, (
                f"Resize error. orig {(orig_w, orig_h)} desire {img_size} compute {target_size}"
            )
            transform = [Resize(target_size)]
        else:
            transform = None

        # Adding try-expcept because some of the data is bad and video decoding call fail.
        try:
            video_decoder = VideoDecoder(
                video,
                seek_mode=self.seek_mode,
                num_ffmpeg_threads=self.video_decode_num_threads,
                transforms=transform,
            )
            num_video_frames = len(video_decoder)

            stride = self._sample_stride_with_bias(self.max_stride, self.min_stride)
            frame_indices = np.arange(0, num_video_frames, stride).tolist()

            # VAE compress temporal by 4x, with 1 as condition
            # thus the max_video_frames must be 1 + 4N
            num_video_frames = min(len(frame_indices), self.args.get("max_num_frames", 1000))
            N = (num_video_frames - 1) // 4
            num_video_frames = 1 + 4 * N
            frame_indices = frame_indices[0:num_video_frames]

            frame_batch = video_decoder.get_frames_at(frame_indices)
            video_frames = frame_batch.data  # [T,C,H,W]
            video_frames = video_frames.permute(1, 0, 2, 3)  # [C,T,H,W]  (T = num_video_frames)

            del video_decoder
        except Exception as e:
            log.warning(
                f"Failed to decode video. url: {data_dict['__url__']}, key: {data_dict['__key__']}, error: {e}",
                rank0_only=False,
            )
            return None

        video_info = {
            "frame_start": frame_indices[0],
            "frame_end": frame_indices[-1],
            "num_frames": len(frame_indices),
            "video": video_frames,
            "fps": meta_dict["framerate"],
            "conditioning_fps": meta_dict["framerate"] / stride,
            "n_orig_video_frames": num_video_frames,
        }

        # Extract audio for the same time range as the video frames
        if self.extract_audio:
            audio_chunk = self._extract_audio_chunk(
                video_bytes=video, video_fps=meta_dict["framerate"], frame_indices=frame_indices
            )
            if audio_chunk is not None:
                video_info["sound"] = audio_chunk
            else:
                video_info["sound"] = None
            # Always include audio_sample_rate when extract_audio is enabled,
            # even if audio extraction failed, so the collate function has a
            # consistent set of keys across all samples in the batch.
            video_info["audio_sample_rate"] = self.audio_sample_rate
        elif self.emit_placeholder_sound:
            video_info["sound"] = None
            video_info["audio_sample_rate"] = self.audio_sample_rate

        data_dict[self.video_key] = video_info

        return data_dict

    def _extract_audio_chunk(
        self, video_bytes: bytes, video_fps: float, frame_indices: list[int]
    ) -> torch.Tensor | None:  # returns [C,N_audio] or None
        """Load audio from the clip, resample, and truncate to match video duration.

        Args:
            video_bytes: Raw video bytes
            video_fps: Video frames per second, used to compute video duration for truncation.
            frame_indices: Frame indices extracted from the video.

        Returns:
            Audio tensor of shape (C, N) or None if extraction fails.
        """
        try:
            # Quick check: probe container for audio streams before AudioDecoder init.
            # AudioDecoder is slow when no audio stream exists. We use torchcodec._core
            # (internal API) to read container metadata without setting up a decode pipeline.
            # If this breaks on a future torchcodec upgrade, remove this block — AudioDecoder
            # will still work, just slower on videos without audio.
            try:
                from torchcodec._core import create_from_bytes, get_container_metadata

                _handle = create_from_bytes(video_bytes)
                _meta = get_container_metadata(_handle)
                _has_audio = _meta.best_audio_stream_index is not None
                del _handle, _meta
                if not _has_audio:
                    return None
            except (ImportError, AttributeError):
                pass  # Fall through to AudioDecoder if _core API is unavailable

            audio_decoder = AudioDecoder(video_bytes)
            all_samples = audio_decoder.get_all_samples()
            audio = all_samples.data  # [C,N_orig]
            orig_sr = all_samples.sample_rate
            del audio_decoder, all_samples

            if orig_sr != self.audio_sample_rate:
                import librosa

                audio = torch.from_numpy(
                    librosa.resample(audio.numpy(), orig_sr=orig_sr, target_sr=self.audio_sample_rate, axis=-1)
                )  # [C,N_resampled]

            # Truncate audio to match the extracted video frame duration.
            if len(frame_indices) > 0 and video_fps > 0:
                video_duration = (frame_indices[-1] + 1) / video_fps
                max_audio_samples = int(video_duration * self.audio_sample_rate)
                if audio.shape[-1] > max_audio_samples:
                    audio = audio[:, :max_audio_samples]  # [C,N_truncated]

            return audio.clone()  # [C,N_audio]

        except Exception as e:
            log.warning(f"Failed to extract audio: {e}", rank0_only=False)
            return None


class VideoParsingChunkedFrames(VideoParsingWithFullFrames):
    """
    This augmentor is used to parse the video bytes and get the video frames for a chunk of frames.
    In the new scheme, we process
    - Full frames if num_frames < 400
    - If num_frames >= 400, we caption only for the first n frame chunk
    In this case, the video extraction needs to only extract the first n frame chunk

    Additionally, in robotics and AV data, we do multi-chunk captioning.
    In this case, we need to sample a chunk uniformly at random and extract the video frames only for that chunk.

    The chunk's frame range is supplied by an upstream ``TextTransformForVideoJsonCaption``
    augmentor via ``data_dict["chunk_start_frame"]`` and ``data_dict["chunk_end_frame"]``.
    Only frames in ``[chunk_start_frame, chunk_end_frame)`` (and the matching audio range)
    are decoded.
    """

    def __init__(self, input_keys: list, output_keys: Optional[list] = None, args: Optional[dict] = None) -> None:
        super().__init__(input_keys, output_keys, args)

    def __call__(self, data_dict: dict) -> dict | None:

        # if in future we need to train with batch size > 1, need to pad frames
        try:
            meta_dict = data_dict[self.meta_key]
            video = data_dict[self.video_key]
        except Exception as e:
            log.warning(
                f"Cannot find video. url: {data_dict['__url__']}, key: {data_dict['__key__']}", rank0_only=False
            )
            return None

        if not self._validate_and_probe(video, meta_dict, data_dict):
            return None

        # The chunk frame range must be supplied by an upstream caption-parsing augmentor
        # (e.g. TextTransformForVideoJsonCaption).
        if "chunk_start_frame" not in data_dict or "chunk_end_frame" not in data_dict:
            log.warning(
                f"VideoParsingChunkedFrames: missing chunk_start_frame/chunk_end_frame in data_dict. "
                f"url: {data_dict['__url__']}, key: {data_dict['__key__']}",
                rank0_only=False,
            )
            return None
        chunk_start = int(data_dict["chunk_start_frame"])
        chunk_end = int(data_dict["chunk_end_frame"])

        # Resize video frames if size is specified. This computes a scaling ratio that fits the
        # video within the target size bounds while preserving the original aspect ratio.
        # The resize transform is applied during decoding via VideoDecoder's transforms parameter.
        if self.perform_resize:
            img_size = obtain_augmentation_size(data_dict, {"size": self.size})
            assert isinstance(img_size, (tuple, omegaconf.listconfig.ListConfig)), (
                f"Arg size in resize should be a tuple, get {type(img_size)}, {img_size}"
            )
            img_w, img_h = img_size
            orig_w, orig_h = meta_dict["width"], meta_dict["height"]

            # Compute uniform scaling ratio to fit video within target bounds (aspect-ratio preserving)
            scaling_ratio = min((img_w / orig_w), (img_h / orig_h))
            target_size = (int(scaling_ratio * orig_h + 0.5), int(scaling_ratio * orig_w + 0.5))

            assert target_size[0] <= img_h and target_size[1] <= img_w, (
                f"Resize error. orig {(orig_w, orig_h)} desire {img_size} compute {target_size}"
            )
            transform = [Resize(target_size)]
        else:
            transform = None

        # Adding try-expcept because some of the data is bad and video decoding call fail.
        try:
            video_decoder = VideoDecoder(
                video,
                seek_mode=self.seek_mode,
                num_ffmpeg_threads=self.video_decode_num_threads,
                transforms=transform,
            )
            decoder_len = len(video_decoder)

            # Clamp the chunk range to what the decoder actually has.
            chunk_start_clamped = max(0, min(chunk_start, decoder_len))
            chunk_end_clamped = max(chunk_start_clamped, min(chunk_end, decoder_len))
            if chunk_end_clamped <= chunk_start_clamped:
                log.warning(
                    f"VideoParsingChunkedFrames: empty chunk after clamping. "
                    f"chunk=[{chunk_start},{chunk_end}), decoder_len={decoder_len}, "
                    f"url: {data_dict['__url__']}, key: {data_dict['__key__']}",
                    rank0_only=False,
                )
                del video_decoder
                return None

            stride = self._sample_stride_with_bias(self.max_stride, self.min_stride)
            frame_indices = np.arange(chunk_start_clamped, chunk_end_clamped, stride).tolist()

            # VAE compress temporal by 4x, with 1 as condition
            # thus the max_video_frames must be 1 + 4N
            num_video_frames = min(len(frame_indices), self.args.get("max_num_frames", 1000))
            N = (num_video_frames - 1) // 4
            num_video_frames = 1 + 4 * N
            if num_video_frames < 1:
                log.warning(
                    f"VideoParsingChunkedFrames: chunk too short for stride. "
                    f"chunk=[{chunk_start_clamped},{chunk_end_clamped}), stride={stride}, "
                    f"url: {data_dict['__url__']}, key: {data_dict['__key__']}",
                    rank0_only=False,
                )
                del video_decoder
                return None
            frame_indices = frame_indices[0:num_video_frames]
            if len(frame_indices) == 0:
                del video_decoder
                return None

            frame_batch = video_decoder.get_frames_at(frame_indices)
            video_frames = frame_batch.data  # [T,C,H,W]
            video_frames = video_frames.permute(1, 0, 2, 3)  # [C,T,H,W]  (T = num_video_frames)

            del video_decoder
        except Exception as e:
            log.warning(
                f"Failed to decode video. url: {data_dict['__url__']}, key: {data_dict['__key__']}, error: {e}",
                rank0_only=False,
            )
            return None

        video_info = {
            "frame_start": frame_indices[0],
            "frame_end": frame_indices[-1],
            "num_frames": len(frame_indices),
            "video": video_frames,
            "fps": meta_dict["framerate"],
            "conditioning_fps": meta_dict["framerate"] / stride,
            "n_orig_video_frames": num_video_frames,
        }

        # Extract audio for the same time range as the chunk's video frames.
        if self.extract_audio:
            audio_chunk = self._extract_audio_chunk(
                video_bytes=video, video_fps=meta_dict["framerate"], frame_indices=frame_indices
            )
            if audio_chunk is not None:
                video_info["sound"] = audio_chunk
            else:
                video_info["sound"] = None
            # Always include audio_sample_rate when extract_audio is enabled,
            # even if audio extraction failed, so the collate function has a
            # consistent set of keys across all samples in the batch.
            video_info["audio_sample_rate"] = self.audio_sample_rate
        elif self.emit_placeholder_sound:
            video_info["sound"] = None
            video_info["audio_sample_rate"] = self.audio_sample_rate

        data_dict[self.video_key] = video_info

        # Cleanup: this augmentor is the last consumer of metas in the json-caption pipeline.
        # Also drop the chunk range markers now that the chunk has been decoded.
        data_dict.pop(self.meta_key, None)
        data_dict.pop("chunk_start_frame", None)
        data_dict.pop("chunk_end_frame", None)

        return data_dict
