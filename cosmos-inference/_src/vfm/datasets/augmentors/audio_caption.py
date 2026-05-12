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

"""Augmentor that appends audio captions to video captions.

Reads an audio caption from the metadata JSON and appends it to the existing
video caption string before tokenization. This allows the model to condition
on both visual and audio descriptions.

Placed AFTER text_transform (which sets ai_caption) and BEFORE text_tokenization.
"""

import sys

from cosmos3._src.imaginaire.datasets.webdataset.augmentors.augmentor import Augmentor
from cosmos3._src.imaginaire.utils import log


def _debug(msg: str) -> None:
    """Write debug message to stderr (unbuffered, reliable in worker processes)."""
    sys.stderr.write(f"[AudioCaptionAppender] {msg}\n")
    sys.stderr.flush()


class AudioCaptionAppender(Augmentor):
    """Appends audio caption text from metadata to the video caption.

    Args:
        input_keys: Expected to be ["metas", "ai_caption"] but read from data_dict directly.
        output_keys: Not used.
        args: Dictionary with:
            - audio_caption_key: Metadata key for audio caption (default: "audio_caption")
            - separator: Text inserted between video and audio captions (default: " Audio description: ")
            - sound_key: Key to check if sound data exists (default: "sound")
    """

    def __init__(self, input_keys: list, output_keys: list | None = None, args: dict | None = None) -> None:
        super().__init__(input_keys, output_keys, args)
        args = args or {}
        self.audio_caption_key = args.get("audio_caption_key", "caption_audio")
        self.separator = args.get("separator", " Audio description: ")
        self.sound_key = args.get("sound_key", "sound")
        self.caption_key = "ai_caption"
        log.warning(
            f"AudioCaptionAppender initialized: audio_caption_key='{self.audio_caption_key}', "
            f"sound_key='{self.sound_key}', metas_key='{input_keys[0]}'",
            rank0_only=True,
        )

    def _find_audio_caption(self, meta_dict: dict) -> str | None:
        """Find audio caption in metas, supporting both flat and nested formats.

        Flat format (e.g., metas_w_audio_caps):
            {"caption_audio": "...", ...}

        Nested format (e.g., midtrain dataset):
            {"0_156": {"caption_sound": "..."}, ...}
            The key is a frame range like "0_156" containing a dict with "caption_sound".
        """
        # Try flat key first
        value = meta_dict.get(self.audio_caption_key)
        if isinstance(value, str) and len(value) > 0:
            return value

        # Try nested: look for a dict value containing "caption_sound"
        for key, val in meta_dict.items():
            if isinstance(val, dict) and "caption_sound" in val:
                caption = val["caption_sound"]
                if isinstance(caption, str) and len(caption) > 0:
                    return caption

        return None

    def __call__(self, data_dict: dict) -> dict | None:
        """Append audio caption to the video caption if available.

        Only appends when sound data is present in the sample. If the metadata
        does not contain the audio_caption_key, the video caption is left unchanged.
        Always cleans up metas from data_dict since this is the last augmentor that reads it.
        """
        metas_key = self.input_keys[0]
        has_sound = self.sound_key in data_dict and data_dict.get(self.sound_key) is not None
        meta_dict = data_dict.get(metas_key)

        if has_sound and meta_dict is not None:
            audio_caption = self._find_audio_caption(meta_dict)
            if isinstance(audio_caption, str) and len(audio_caption) > 0:
                current_caption = data_dict.get(self.caption_key, "")
                data_dict[self.caption_key] = current_caption + self.separator + audio_caption

        # Clean up metas from data_dict — this augmentor is the last consumer of metas
        if metas_key in data_dict:
            del data_dict[metas_key]

        return data_dict
