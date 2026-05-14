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

"""Sound data utilities for building sequence plans and handling audio-video generation modes.

This module provides utilities for building SequencePlan objects based on sound generation modes,
similar to how action modes are handled in cosmos/data/vfm/action/data_utils.py.

Supported modes:
    - t2vs: Text → Video + Sound (joint generation)
    - tv2s: Text + Video → Sound (foley - video conditioned, sound generated)
    - ts2v: Text + Sound → Video (sound conditioned, video generated)
    - ti2sv: Text + Image → Sound + Video (first frame conditioned, rest + sound generated)
"""

from cosmos.data.vfm.sequence_packing import SequencePlan

# Valid generation modes for sound
VALID_SOUND_MODES = {"t2vs", "tv2s", "ts2v", "ti2sv"}


def build_sequence_plan_for_sound(
    mode: str,
    video_latent_length: int,
    sound_latent_length: int,
    has_text: bool = True,
) -> SequencePlan:
    """Build a SequencePlan based on the sound generation mode.

    This function determines the appropriate condition frame indexes for vision and sound
    based on the specified mode. It mirrors how `build_sequence_plan_from_mode` works
    for action in cosmos/data/vfm/action/data_utils.py.

    Args:
        mode: Generation mode. One of:
            - "t2vs": Text → Video + Sound (both generated)
            - "tv2s": Text + Video → Sound (video conditioned, sound generated)
            - "ts2v": Text + Sound → Video (sound conditioned, video generated)
            - "ti2sv": Text + Image → Sound + Video (first frame conditioned)
        video_latent_length: Number of video latent frames.
        sound_latent_length: Number of sound latent tokens.
        has_text: Whether text conditioning is available. Defaults to True.

    Returns:
        SequencePlan instance with appropriate settings.

    Raises:
        ValueError: If mode is not one of the supported modes.

    Example:
        >>> plan = build_sequence_plan_for_sound(
        ...     mode="tv2s",
        ...     video_latent_length=24,
        ...     sound_latent_length=100,
        ... )
        >>> plan.has_sound
        True
        >>> plan.condition_frame_indexes_vision  # All video conditioned
        [0, 1, 2, ..., 23]
        >>> plan.condition_frame_indexes_sound  # All sound generated
        []
    """
    if mode not in VALID_SOUND_MODES:
        raise ValueError(f"Invalid mode: {mode!r}. Must be one of {VALID_SOUND_MODES}")

    if mode == "t2vs":
        # Text → Video + Sound: both generated jointly
        return SequencePlan(
            has_text=has_text,
            has_vision=True,
            condition_frame_indexes_vision=[],  # All vision frames generated
            has_sound=True,
            condition_frame_indexes_sound=[],  # All sound tokens generated
        )
    elif mode == "tv2s":
        # Text + Video → Sound: video conditioned, sound generated (foley)
        return SequencePlan(
            has_text=has_text,
            has_vision=True,
            condition_frame_indexes_vision=list(range(video_latent_length)),  # All vision conditioned
            has_sound=True,
            condition_frame_indexes_sound=[],  # All sound tokens generated
        )
    elif mode == "ts2v":
        # Text + Sound → Video: sound conditioned, video generated
        return SequencePlan(
            has_text=has_text,
            has_vision=True,
            condition_frame_indexes_vision=[],  # All vision frames generated
            has_sound=True,
            condition_frame_indexes_sound=list(range(sound_latent_length)),  # All sound conditioned
        )
    elif mode == "ti2sv":
        # Text + Image → Sound + Video: first vision frame conditioned, rest + sound generated
        return SequencePlan(
            has_text=has_text,
            has_vision=True,
            condition_frame_indexes_vision=[0],  # First frame conditioned (image)
            has_sound=True,
            condition_frame_indexes_sound=[],  # All sound tokens generated
        )
    else:
        raise ValueError(f"Unhandled mode: {mode}")
