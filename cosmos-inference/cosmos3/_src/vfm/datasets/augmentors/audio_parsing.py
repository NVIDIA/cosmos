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

"""Audio parsing augmentor for T2A (Text-to-Audio) datasets.

For audio-only datasets (AudioCaps, WavCaps, etc.) that have no video,
this augmentor:
1. Decodes audio from bytes
2. Creates a full-length dummy video (all zeros) matching the audio duration
3. Outputs data compatible with the v3 video training pipeline

The dummy video ensures compatibility with the model architecture which
requires vision tokens in the sequence (sound_gen requires vision_gen).
The dummy video is fully conditioned (all frames clean), so it contributes
no loss — effectively making this a tv2s (text+video→sound) mode where
the video is a placeholder.
"""

from typing import Optional

import torch
from torchcodec.decoders import AudioDecoder

from cosmos3._src.imaginaire.datasets.webdataset.augmentors.augmentor import Augmentor
from cosmos3._src.imaginaire.utils import log


class AudioParsingForFullClips(Augmentor):
    """Audio parsing augmentor for audio-only datasets.

    Loads audio from bytes, creates a dummy video of matching duration,
    and outputs data compatible with the VideoParsingWithFullFrames pipeline.

    Args:
        input_keys: [meta_key, audio_key] — keys to fetch metadata and audio bytes
        output_keys: Optional output keys
        args: Dictionary with:
            - target_sample_rate: Target audio sample rate (default: 48000)
            - target_channels: Target audio channels (default: 2 for stereo)
            - dummy_video_fps: FPS for dummy video (default: 24)
            - dummy_video_size: (H, W) for dummy video (default: (256, 256))
            - max_audio_duration_sec: Max audio duration in seconds (default: 30.0)
            - min_audio_duration_sec: Min audio duration in seconds (default: 1.0)
    """

    def __init__(self, input_keys: list, output_keys: Optional[list] = None, args: Optional[dict] = None) -> None:
        super().__init__(input_keys, output_keys, args)
        assert len(input_keys) == 2, "AudioParsingForFullClips requires two input keys: [meta_key, audio_key]"
        self.meta_key = input_keys[0]
        self.audio_key = input_keys[1]

        self.target_sample_rate = args.get("target_sample_rate", 48000)
        self.target_channels = args.get("target_channels", 2)
        self.dummy_video_fps = args.get("dummy_video_fps", 24.0)
        self.dummy_video_size = args.get("dummy_video_size", (256, 256))
        self.max_audio_duration_sec = args.get("max_audio_duration_sec", 30.0)
        self.min_audio_duration_sec = args.get("min_audio_duration_sec", 1.0)

    def __call__(self, data_dict: dict) -> dict | None:
        try:
            meta_dict = data_dict[self.meta_key]
            audio_bytes = data_dict[self.audio_key]
        except Exception:
            log.warning(
                f"Cannot find audio data. url: {data_dict.get('__url__', '?')}, key: {data_dict.get('__key__', '?')}",
                rank0_only=False,
            )
            return None

        if not isinstance(audio_bytes, bytes):
            log.warning("Audio data is not bytes, skipping", rank0_only=False)
            return None

        # Decode audio
        try:
            audio_decoder = AudioDecoder(audio_bytes)
            audio_metadata = audio_decoder.metadata
            orig_sample_rate = audio_metadata.sample_rate

            audio_samples = audio_decoder.get_samples_played_in_range()
            audio_chunk = audio_samples.data  # [C,N_orig]
            del audio_decoder
        except Exception as e:
            log.warning(f"Failed to decode audio: {e}", rank0_only=False)
            return None

        # Compute duration
        audio_duration_sec = audio_chunk.shape[1] / orig_sample_rate

        # Filter by duration
        if audio_duration_sec < self.min_audio_duration_sec:
            log.debug(f"Audio too short: {audio_duration_sec:.2f}s < {self.min_audio_duration_sec}s", rank0_only=False)
            return None
        if audio_duration_sec > self.max_audio_duration_sec:
            # Crop to max duration
            max_samples = int(self.max_audio_duration_sec * orig_sample_rate)
            audio_chunk = audio_chunk[:, :max_samples]
            audio_duration_sec = self.max_audio_duration_sec

        # Resample if needed
        if orig_sample_rate != self.target_sample_rate:
            import torchaudio

            audio_chunk = torchaudio.functional.resample(
                audio_chunk, orig_freq=orig_sample_rate, new_freq=self.target_sample_rate
            )  # [C,N_resampled]

        # Handle channel count (mono → stereo or vice versa)
        if audio_chunk.shape[0] == 1 and self.target_channels == 2:
            audio_chunk = audio_chunk.repeat(2, 1)  # [2,N_resampled]
        elif audio_chunk.shape[0] > self.target_channels:
            audio_chunk = audio_chunk[: self.target_channels]  # [C_target,N_resampled]

        # Create dummy video matching audio duration
        # VAE compress temporal by 4x, with 1 as condition → num_frames must be 1 + 4N
        num_video_frames = int(audio_duration_sec * self.dummy_video_fps)
        N = (num_video_frames - 1) // 4
        num_video_frames = max(1 + 4 * N, 1)

        h, w = self.dummy_video_size
        dummy_video = torch.zeros(3, num_video_frames, h, w, dtype=torch.uint8)  # [3,T,H,W]

        # Build output compatible with VideoParsingWithFullFrames
        video_info = {
            "frame_start": 0,
            "frame_end": num_video_frames - 1,
            "num_frames": num_video_frames,
            "video": dummy_video,
            "fps": self.dummy_video_fps,
            "conditioning_fps": self.dummy_video_fps,
            "n_orig_video_frames": num_video_frames,
            "sound": audio_chunk,
            "audio_sample_rate": self.target_sample_rate,
        }
        data_dict["video"] = video_info

        return data_dict
