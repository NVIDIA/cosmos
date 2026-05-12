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

"""Augmentor that appends idle-frame count metadata to the caption.

The label is a Pi0.7-style episode-metadata field encoded as plain text. It
records how many frames of the action chunk were "idle" out of the total action
frames (i.e. the relative-pose delta is close to identity and the gripper
command does not change). The upstream dataset is responsible for populating
``data_dict[idle_frames_key]`` via
:func:`projects.cosmos3.vfm.datasets.action.pose_utils.compute_idle_frames`.

Per-field dropout (default 5%) is applied here, matching Pi0.7's approach of
independently dropping each metadata component. This is complementary to the
global ``cfg_dropout_rate`` in :class:`TextTokenizerTransform`, which still
empties the whole caption.
"""

from __future__ import annotations

import random

import torch

from cosmos3._src.imaginaire.datasets.webdataset.augmentors.augmentor import Augmentor
from cosmos3._src.imaginaire.utils import log

DEFAULT_TEMPLATE = "IdleFrames: {n} out of {m}."
FALLBACK_TEMPLATE = "IdleFrames: {n}."


class IdleFramesTextInfo(Augmentor):
    """Augmentor that appends ``IdleFrames: N out of M.`` to the caption.

    Reads ``data_dict[idle_frames_key]`` (set by the dataset layer) and appends
    a textual marker to the caption, modeled after
    :class:`ResolutionTextInfo` and :class:`DurationFPSTextTimeStamps`.

    Per-field dropout is supported: with probability ``dropout_rate`` the
    segment is omitted entirely (the caption is left unchanged). This is
    independent from the global classifier-free-guidance dropout in the
    tokenizer.

    Example:
        Original caption: "pick up the cup"
        Augmented:        "pick up the cup. IdleFrames: 0 out of 16."

    Args:
        input_keys (list): Input keys (not used, kept for API compatibility).
        output_keys (list): Output keys (not used, kept for API compatibility).
        args (dict): Configuration arguments:
            - caption_key (str): Key for caption in data_dict. Default ``"ai_caption"``.
            - idle_frames_key (str): Key for the idle-frame integer in data_dict.
              Default ``"idle_frames"``.
            - total_frames_key (str): Optional key for the total frame integer
              in data_dict. Default ``"idle_frames_total"``.
            - action_key (str): Key for the action tensor used to infer total
              frames when ``total_frames_key`` is missing. Default ``"action"``.
            - template (str): Format string for the appended segment.
              Default ``"IdleFrames: {n} out of {m}."``.
            - separator (str): Separator inserted between the original caption
              and the new segment. Default ``". "``.
            - dropout_rate (float): Probability of skipping the append step
              (per-field dropout). Default 0.05.
            - enabled (bool): Whether the augmentor is active. Default True.
    """

    def __init__(
        self,
        input_keys: list | None = None,
        output_keys: list | None = None,
        args: dict | None = None,
    ) -> None:
        super().__init__(input_keys, output_keys, args)

        args = args or {}
        self.caption_key: str = args.get("caption_key", "ai_caption")
        self.idle_frames_key: str = args.get("idle_frames_key", "idle_frames")
        self.total_frames_key: str = args.get("total_frames_key", "idle_frames_total")
        self.action_key: str = args.get("action_key", "action")
        self.template: str = args.get("template", DEFAULT_TEMPLATE)
        self.default_separator: str = args.get("separator", ". ")
        self.dropout_rate: float = float(args.get("dropout_rate", 0.05))
        self.enabled: bool = bool(args.get("enabled", True))

        if not 0.0 <= self.dropout_rate <= 1.0:
            raise ValueError(f"dropout_rate must be in [0, 1]; got {self.dropout_rate}")

    def _get_scalar_int(self, value: object, key: str) -> int | None:
        """Parse an optional scalar integer metadata value."""

        if value is None:
            return None

        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                log.warning(
                    f"IdleFramesTextInfo: expected scalar tensor at '{key}', got shape {tuple(value.shape)}. Skipping.",
                    rank0_only=False,
                )
                return None
            return int(value.item())

        try:
            return int(value)
        except (TypeError, ValueError):
            log.warning(
                f"IdleFramesTextInfo: expected integer-compatible value at "
                f"'{key}', got {type(value).__name__}. Skipping.",
                rank0_only=False,
            )
            return None

    def _get_total_frames(self, data_dict: dict) -> int | None:
        """Resolve the total action-frame count for the idle-frame text."""

        total_frames = self._get_scalar_int(data_dict.get(self.total_frames_key), self.total_frames_key)
        if total_frames is not None:
            return total_frames

        action = data_dict.get(self.action_key)
        if isinstance(action, torch.Tensor):
            if action.ndim == 0:
                log.warning(
                    f"IdleFramesTextInfo: expected action tensor at "
                    f"'{self.action_key}' to have a frame dimension. Skipping total frames.",
                    rank0_only=False,
                )
                return None
            return int(action.shape[0])

        try:
            return len(action) if action is not None else None
        except TypeError:
            return None

    def __call__(self, data_dict: dict) -> dict | None:
        """Append ``IdleFrames: N out of M.`` to ``data_dict[caption_key]`` in place.

        Returns the input dict unchanged when:

        - the augmentor is disabled,
        - the per-field dropout fires,
        - ``idle_frames_key`` is missing or ``None`` (e.g. non-action sample),
        - the caption is missing, empty, or not a string/dict (unconditional case).

        For dict-typed captions (the JSON-caption code path), the idle-frame
        integer is added under ``"idle_frames"`` and the total count, when
        available, is added under ``"idle_frames_total"``.
        """
        if not self.enabled:
            return data_dict

        if random.random() < self.dropout_rate:
            return data_dict

        n = self._get_scalar_int(data_dict.get(self.idle_frames_key), self.idle_frames_key)
        if n is None:
            return data_dict

        m = self._get_total_frames(data_dict)

        if self.caption_key not in data_dict:
            return data_dict
        caption = data_dict[self.caption_key]

        if isinstance(caption, str):
            if caption == "":
                return data_dict
            metadata_text = self.template.format(n=n, m=m) if m is not None else FALLBACK_TEMPLATE.format(n=n)
            separator = " " if caption.rstrip().endswith(".") else self.default_separator
            data_dict[self.caption_key] = caption + separator + metadata_text
        elif isinstance(caption, dict):
            data_dict[self.caption_key]["idle_frames"] = n
            if m is not None:
                data_dict[self.caption_key]["idle_frames_total"] = m
        else:
            return data_dict

        return data_dict
