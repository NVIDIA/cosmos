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

"""Dataset transform wrappers for the Action project.

This module provides the ``ActionTransformPipeline`` and spatial padding utilities.

The reflection padding snaps each sample to the closest predefined resolution from
``VIDEO_RES_SIZE_INFO`` (matching VFM's approach), guaranteeing a bounded set of
output shapes that are all multiples of 16.

See :func:`~.unified_dataset.wrap_dataset` for the convenience factory that
combines datasets with transforms, and :class:`~.unified_dataset.MapToIterableAdapter`
for the map-to-iterable wrapper.
"""

from __future__ import annotations

import torch
import torchvision.transforms.functional as transforms_F

from cosmos.utils import log
from cosmos.data.vfm.action.json_formatter import ActionPromptJsonFormatter
from cosmos.data.vfm.action.viewpoint_utils import ViewpointTextInfo
from cosmos.data.vfm.augmentors.duration_fps_text_timestamps import DurationFPSTextTimeStamps
from cosmos.data.vfm.augmentors.idle_frames_text_info import IdleFramesTextInfo
from cosmos.data.vfm.augmentors.resolution_text_info import ResolutionTextInfo
from cosmos.data.vfm.augmentors.text_tokenizer import TextTokenizerTransform
from cosmos.data.vfm.sequence_packing import SequencePlan
from cosmos.data.vfm.utils import VIDEO_RES_SIZE_INFO
from cosmos.utils.vfm.data_utils import get_vision_data_resolution


def _should_append_idle_frame_info(mode: object) -> bool:
    """Return whether idle-frame prompt metadata should be surfaced."""
    return mode != "inverse_dynamics"


def pad_action_to_max_dim(action: torch.Tensor, max_action_dim: int) -> torch.Tensor:
    """Pad action tensor to max_action_dim along the last dimension.

    Args:
        action: Action tensor of shape (T, D) where D is the current action dimension.
        max_action_dim: Target action dimension to pad to.

    Returns:
        Padded action tensor of shape (T, max_action_dim).
    """
    if action.shape[-1] > max_action_dim:
        raise ValueError(f"Action dimension {action.shape[-1]} is greater than max_action_dim {max_action_dim}")
    elif action.shape[-1] == max_action_dim:
        return action
    else:
        padding_size = max_action_dim - action.shape[-1]
        zero_padding = torch.zeros(
            *action.shape[:-1], padding_size, dtype=action.dtype, device=action.device
        )  # [T,padding_size]
        return torch.cat([action, zero_padding], dim=-1)  # [T,max_action_dim]


def find_closest_target_size(h: int, w: int, resolution: str | int) -> tuple[int, int]:
    """Find the closest predefined target size for a given input resolution.

    Looks up ``VIDEO_RES_SIZE_INFO[resolution]`` and selects the aspect ratio
    whose ``H/W`` ratio is closest to the input ``h/w``.

    Args:
        h: Input height in pixels.
        w: Input width in pixels.
        resolution: Resolution tier key (e.g. ``"256"``, ``"480"``, ``"720"``).

    Returns:
        ``(target_w, target_h)`` from the predefined table.

    Raises:
        ValueError: If *resolution* is not a key in ``VIDEO_RES_SIZE_INFO``.
    """
    if isinstance(resolution, int):
        resolution = str(resolution)
    if resolution not in VIDEO_RES_SIZE_INFO:
        raise ValueError(
            f"Resolution '{resolution}' not found in VIDEO_RES_SIZE_INFO. Available: {list(VIDEO_RES_SIZE_INFO.keys())}"
        )

    candidates = VIDEO_RES_SIZE_INFO[resolution]
    input_ratio = h / w

    best_key: str | None = None
    best_diff = float("inf")
    for aspect_key, (cand_w, cand_h) in candidates.items():
        cand_ratio = cand_h / cand_w
        diff = abs(input_ratio - cand_ratio)
        if diff < best_diff:
            best_diff = diff
            best_key = aspect_key

    assert best_key is not None
    target_w, target_h = candidates[best_key]
    return target_w, target_h


def reflection_pad_to_target(
    data_dict: dict,
    keys: list[str],
    keep_aspect_ratio: bool,
    target_w: int,
    target_h: int,
) -> dict:
    """Resize (aspect-preserving) and reflection-pad tensors to exact target size.

    For each key in *keys*, the tensor is:

    1. Resized so its spatial dimensions fit within ``(target_h, target_w)``
       while preserving the aspect ratio (matching VFM's
       ``ResizeLargestSideAspectPreserving``).
    2. Reflection-padded (or edge-padded when the padding exceeds the spatial
       dimension) to reach exactly ``(target_h, target_w)`` (matching VFM's
       ``ReflectionPadding``).

    After processing, the following entries are added to *data_dict*:

    - ``"image_size"``: ``torch.Tensor`` of shape ``(4,)`` containing
      ``[target_h, target_w, orig_h_resized, orig_w_resized]`` where
      ``target_h/w`` is the padded canvas size and ``orig_h/w_resized``
      is the original spatial size after aspect-preserving resize (i.e.
      the content region before padding).  After ``default_collate``
      this becomes ``(B, 4)``;  the ``IterativeJointDataLoader`` then
      splits it into per-sample ``(1, 4)`` tensors so the model can
      index as ``data_batch["image_size"][i][0][0]``.

    Args:
        data_dict: The sample dictionary (mutated in-place).
        keys: Data-dict keys whose tensors should be resized and padded.
            Tensors must have shape ``(C, H, W)`` or ``(C, T, H, W)``.
        keep_aspect_ratio: Whether to keep the aspect ratio of the input tensor.
        target_w: Target width in pixels.
        target_h: Target height in pixels.

    Returns:
        The mutated *data_dict*.
    """
    orig_h_resized: int = 0
    orig_w_resized: int = 0

    for key in keys:
        if key not in data_dict:
            continue
        tensor = data_dict[key]
        if not isinstance(tensor, torch.Tensor):
            continue

        # Extract spatial dims
        if tensor.ndim == 3:
            orig_h, orig_w = tensor.shape[-2:]
        elif tensor.ndim == 4:
            orig_h, orig_w = tensor.shape[-2:]
        else:
            raise ValueError(f"Unexpected tensor ndim={tensor.ndim} for key '{key}', expected 3 or 4")

        # Step 1: aspect-preserving resize to fit within (target_h, target_w)
        if keep_aspect_ratio:
            # Prevent upscaling the video by setting the upper bound of scaling_ratio to 1.0.
            scaling_ratio = min(target_w / orig_w, target_h / orig_h, 1.0)
            orig_h_resized = int(scaling_ratio * orig_h + 0.5)
            orig_w_resized = int(scaling_ratio * orig_w + 0.5)
            assert orig_h_resized <= target_h and orig_w_resized <= target_w, (
                f"Resize error: orig ({orig_h}, {orig_w}) target ({target_h}, {target_w}) "
                f"computed ({orig_h_resized}, {orig_w_resized})"
            )
        else:
            orig_h_resized = target_h
            orig_w_resized = target_w

        if orig_h_resized != orig_h or orig_w_resized != orig_w:
            tensor = transforms_F.resize(
                tensor,
                size=[orig_h_resized, orig_w_resized],
                interpolation=transforms_F.InterpolationMode.BICUBIC,
                antialias=True,
            )

        # Step 2: padding to exact target size (bottom and right only)
        if orig_w_resized != target_w or orig_h_resized != target_h:
            padding_right = target_w - orig_w_resized
            padding_bottom = target_h - orig_h_resized
            padding = [0, 0, padding_right, padding_bottom]

            if padding_right >= orig_w_resized or padding_bottom >= orig_h_resized:
                tensor = transforms_F.pad(tensor, padding, padding_mode="edge")
            else:
                tensor = transforms_F.pad(tensor, padding, padding_mode="reflect")

        data_dict[key] = tensor

    # image_size: shape (4,) — [target_h, target_w, orig_h_resized, orig_w_resized].
    # Matches VFM's item_dataset convention.  default_collate stacks to (B, 4);
    # IterativeJointDataLoader._get_next_sample slices to (1, 4) per sample so
    # the model can index [i][0][0].
    data_dict["image_size"] = torch.tensor(
        [target_h, target_w, orig_h_resized, orig_w_resized], dtype=torch.float
    )  # [4]

    return data_dict


def remove_reflection_padding(
    tensor: torch.Tensor,
    image_size: torch.Tensor,
) -> torch.Tensor:
    """Remove reflection padding added by :func:`reflection_pad_to_target`.

    Content is at top-left; crops to ``(orig_h_resized, orig_w_resized)``.

    Args:
        tensor: Tensor whose last two dimensions are the padded spatial dims.
            Supports any leading dimensions, e.g. ``(C, T, H, W)`` or
            ``(C, H, W)``.
        image_size: 1-D tensor of shape ``(4,)`` containing
            ``[target_h, target_w, orig_h_resized, orig_w_resized]`` where
            ``orig_h/w_resized`` is the original spatial size after
            aspect-preserving resize (i.e. the content region before
            padding) — the same convention stored by
            :func:`reflection_pad_to_target` and VFM's
            ``ReflectionPadding``.

    Returns:
        Cropped tensor of shape ``(..., orig_h_resized, orig_w_resized)``.
    """
    target_h = int(image_size[0].item())
    target_w = int(image_size[1].item())
    orig_h_resized = int(image_size[2].item())
    orig_w_resized = int(image_size[3].item())

    if orig_h_resized == target_h and orig_w_resized == target_w:
        return tensor

    return tensor[..., :orig_h_resized, :orig_w_resized].contiguous()


def build_sequence_plan_from_mode(
    mode: str,
    video_length: int,
    action_length: int,
    has_text: bool = True,
    video_temporal_downsample: int = 4,
    num_history_actions: int = 0,
) -> SequencePlan:
    """Build a SequencePlan based on the training mode.

    This function determines whether action should be included and computes the
    appropriate condition frame indexes for vision and action based on the mode.

    Args:
        mode: Training mode. One of:
            - "image2video": Image-to-video generation (no action)
            - "forward_dynamics": Predict video given first frame and all actions
            - "inverse_dynamics": Predict actions given all video frames
            - "policy": Predict both actions and video given first frame
        video_length: Number of video frames (including the conditioning frame).
        action_length: Number of action steps (typically video_length - 1).
        has_text: Whether text conditioning is available. Defaults to True.
        video_temporal_downsample: Temporal downsampling factor of the video
            tokenizer. Used to compute condition frame indexes for inverse
            dynamics mode. Defaults to 4.

    Returns:
        SequencePlan instance with appropriate settings.
        Use ``sequence_plan.has_action`` to check if action should be included.

    Raises:
        ValueError: If mode is not one of the supported modes.

    Example:
        >>> sequence_plan = build_sequence_plan_from_mode(
        ...     mode="policy",
        ...     video_length=5,
        ...     action_length=4,
        ... )
        >>> sequence_plan.has_action
        True
        >>> sequence_plan.as_dict()
        {'has_text': True, 'has_vision': True, 'has_action': True,
         'condition_frame_indexes_vision': [0], 'condition_frame_indexes_action': []}
    """
    valid_modes = ["image2video", "forward_dynamics", "inverse_dynamics", "policy"]
    if mode not in valid_modes:
        raise ValueError(f"Invalid mode: {mode!r}. Must be one of {valid_modes}")

    # Determine if action should be included based on mode
    # image2video mode: no action (pure image-to-video generation)
    # forward_dynamics, inverse_dynamics, policy: action is needed
    has_action = mode != "image2video"

    # Determine condition frame indexes based on mode
    # image2video/forward_dynamics/policy: first frame is clean (conditioning)
    # inverse_dynamics: all frames are provided as context
    if mode in ["image2video", "forward_dynamics", "policy"]:
        condition_frame_indexes_vision = [0]
    elif mode == "inverse_dynamics":
        # All frames are observed for inverse dynamics
        condition_frame_indexes_vision = list(range(0, (video_length - 1) // video_temporal_downsample + 1))
    else:
        condition_frame_indexes_vision = []

    # For action conditioning indexes:
    # forward_dynamics: all action steps are clean (conditioning)
    # inverse_dynamics/policy: action is supervised (predicted)
    # History frames (prepended) are always conditioning.
    base_action_length = action_length - num_history_actions
    if mode == "forward_dynamics":
        condition_frame_indexes_action = list(range(action_length))

    # This currently assumes that the action length is the same as the video length - 1
    # and if action length is the same as the video length, then the first action is the conditioning action
    elif base_action_length == video_length - 1:
        condition_frame_indexes_action = list(range(num_history_actions))
    elif base_action_length == video_length:
        condition_frame_indexes_action = list(range(num_history_actions + 1))

    if base_action_length == video_length - 1:
        action_start_frame_offset = 1 - num_history_actions
    if base_action_length == video_length:
        action_start_frame_offset = -num_history_actions

    return SequencePlan(
        has_text=has_text,
        has_vision=True,
        has_action=has_action,
        condition_frame_indexes_vision=condition_frame_indexes_vision,
        condition_frame_indexes_action=condition_frame_indexes_action,
        action_start_frame_offset=action_start_frame_offset,
    )


class VideoResize:
    """Resize and reflection-pad video-aligned tensors for a single sample.

    Resolution is supplied at call time. When ``resolution`` is ``None``, the
    tier is auto-detected from the sample's ``"video"`` spatial dimensions.

    Args:
        pad_keys: Data-dict keys whose values should be resized and padded.
            Pass an empty list to disable padding entirely. Defaults to
            ``["video"]``.
        keep_aspect_ratio: Whether to resize aspect-preservingly to the closest
            predefined target size before padding. Defaults to ``True``.
        log_prefix: Prefix used in debug logging.
    """

    def __init__(
        self,
        pad_keys: list[str] | None = None,
        keep_aspect_ratio: bool = True,
        log_prefix: str = "VideoResize",
    ) -> None:
        self.pad_keys = pad_keys if pad_keys is not None else ["video"]
        self.keep_aspect_ratio = keep_aspect_ratio
        self.log_prefix = log_prefix

    def __call__(self, data_dict: dict, resolution: str | int | None) -> dict:
        """Resize and pad a sample in-place.

        Args:
            data_dict: Sample dictionary containing a ``"video"`` entry.
            resolution: Resolution tier key (e.g. ``"256"``, ``"480"``,
                ``"720"``). When ``None``, auto-detected from video dimensions.

        Returns:
            The same dictionary, mutated in-place with padded tensors and an
            ``"image_size"`` entry.
        """
        video = data_dict.get("video")
        assert isinstance(video, torch.Tensor), "video is required for reflection padding"
        h, w = video.shape[-2:]

        if resolution is None:
            resolution = get_vision_data_resolution((h, w))

        if self.keep_aspect_ratio:
            target_w, target_h = find_closest_target_size(h, w, resolution)
        else:
            target_w = int(resolution)
            target_h = int(resolution)
        reflection_pad_to_target(data_dict, self.pad_keys, self.keep_aspect_ratio, target_w, target_h)

        return data_dict

    def _log_shapes(self, data_dict: dict, when: str) -> None:
        """Log tensor shapes for the configured pad keys."""
        for key in self.pad_keys:
            val = data_dict.get(key)
            if isinstance(val, torch.Tensor):
                log.debug(f"{self.log_prefix}: {when} padding '{key}' shape = {tuple(val.shape)}")


class ActionTransformPipeline:
    """A composable transform pipeline that chains ``VideoResize``, text
    tokenization, and automatic sequence plan construction.

    Reflection padding snaps each sample to the closest predefined aspect
    ratio from ``VIDEO_RES_SIZE_INFO[resolution]``, resizes
    (aspect-preserving) to fit within the target, then reflection-pads to
    the exact target size.  This guarantees a bounded set of output shapes
    (5 per resolution tier), all multiples of 16.  Resolution is supplied
    at call time via the required ``resolution`` argument to ``__call__``;
    when ``resolution`` is ``None``, the tier is auto-detected from the
    video's spatial dimensions via ``get_vision_data_resolution``.

    Text tokenization is enabled when ``tokenizer_config`` is provided.

    When the data dictionary contains a ``"mode"`` key, the pipeline automatically
    builds a ``SequencePlan`` via :func:`build_sequence_plan_from_mode` and attaches
    it as ``data_dict["sequence_plan"]``.  For modes where action is not needed
    (e.g. ``"image2video"``), the ``"action"`` and ``"domain_id"`` keys are set to
    ``None``.

    Args:
        pad_keys: Data-dict keys whose values should be resized and padded. Pass
            an empty list to disable padding entirely. Defaults to ``["video"]``.
        tokenizer_config: A lazy-instantiable config dict for the VLM tokenizer. When
            ``None``, text tokenization is skipped. Defaults to ``None``.
        cfg_dropout_rate: Probability of replacing the caption with an empty string for
            classifier-free guidance. Only used when text tokenization is enabled.
            Defaults to ``0.0``.
        caption_key: The data-dict key that contains the input caption string.
            Defaults to ``"ai_caption"``.
        text_token_key: The data-dict key where tokenized text IDs will be stored.
            Defaults to ``"text_token_ids"``.
        video_temporal_downsample: Temporal downsampling factor of the video tokenizer.
            Used when building a ``SequencePlan`` for ``"inverse_dynamics"`` mode.
            Defaults to 4.
        max_action_dim: Target action dimension to pad to.  The ``"action"`` tensor
            in every sample is padded along its last dimension via
            :func:`pad_action_to_max_dim`.  Defaults to 32.
        action_channel_masking: When ``True`` (default), the original action
            dimension is stored in ``"raw_action_dim"`` so that the model masks
            loss/noise/velocity on zero-padded action channels.  When ``False``,
            ``"raw_action_dim"`` is set to ``None`` and the model treats all
            ``max_action_dim`` channels equally (original main-branch behavior).
        append_viewpoint_info: Whether to append viewpoint type metadata to the
            caption (via ``ViewpointTextInfo`` augmentor).  Requires that
            samples contain a ``"viewpoint"`` key.  Defaults to ``True``.
        append_duration_fps_timestamps: Whether to append duration and FPS metadata to the
            caption (matching VFM's ``DurationFPSTextTimeStamps`` augmentor).
            Defaults to ``True``.
        append_resolution_info: Whether to append resolution metadata to the
            caption (matching VFM's ``ResolutionTextInfo`` augmentor).
            Defaults to ``True``.
        append_idle_frames: Whether to append the idle-frame count out of the
            total action frames to the caption (Pi0.7-style metadata, via
            ``IdleFramesTextInfo`` augmentor).  The dataset is responsible for
            populating ``data_dict["idle_frames"]``; samples without it are
            silently skipped.  Idle-frame text is skipped only for
            ``"inverse_dynamics"`` mode.  Defaults to ``False`` so existing
            experiments are unaffected.
        idle_frames_dropout: Per-field dropout rate for the idle-frame segment.
            With this probability the augmentor leaves the caption unchanged
            (matching Pi0.7's ~5% per-component dropout).  Independent of the
            global ``cfg_dropout_rate``, which empties the whole caption.
            Defaults to 0.05.
        format_prompt_as_json: Whether to replace the plain text prompt with a
            structured JSON-compatible dictionary before tokenization.  When
            enabled, legacy string metadata appenders are skipped and the JSON
            formatter owns viewpoint, action, resolution, duration, FPS, and
            idle-frame fields.  Defaults to ``False``.
    """

    def __init__(
        self,
        pad_keys: list[str] | None = None,
        keep_aspect_ratio: bool = True,
        tokenizer_config: dict | None = None,
        cfg_dropout_rate: float = 0.0,
        caption_key: str = "ai_caption",
        text_token_key: str = "text_token_ids",
        video_temporal_downsample: int = 4,
        max_action_dim: int = 32,
        action_channel_masking: bool = True,
        append_viewpoint_info: bool = True,
        append_duration_fps_timestamps: bool = True,
        append_resolution_info: bool = True,
        append_idle_frames: bool = False,
        idle_frames_dropout: float = 0.05,
        format_prompt_as_json: bool = False,
    ) -> None:
        self.caption_key: str = caption_key
        self.video_temporal_downsample: int = video_temporal_downsample
        self.max_action_dim: int = max_action_dim
        self.action_channel_masking: bool = action_channel_masking

        # --- Spatial resize/padding stage (resolution supplied at call time) ---
        self.video_resize: VideoResize = VideoResize(
            pad_keys=pad_keys,
            keep_aspect_ratio=keep_aspect_ratio,
            log_prefix="ActionTransformPipeline",
        )
        self.pad_keys: list[str] = self.video_resize.pad_keys
        self.keep_aspect_ratio: bool = self.video_resize.keep_aspect_ratio

        self.prompt_json_formatter: ActionPromptJsonFormatter | None = None
        if format_prompt_as_json:
            self.prompt_json_formatter = ActionPromptJsonFormatter(caption_key=caption_key)

        # --- Viewpoint text augmentor (runs after ai_caption, before duration/FPS) ---
        self.viewpoint_augmentor: ViewpointTextInfo | None = None
        if append_viewpoint_info and self.prompt_json_formatter is None:
            self.viewpoint_augmentor = ViewpointTextInfo(
                input_keys=[caption_key, "viewpoint"],
                output_keys=[caption_key],
                args={"caption_key": caption_key, "viewpoint_key": "viewpoint", "enabled": True},
            )

        # --- Duration/FPS text augmentor (runs before tokenization) ---
        self.duration_fps_augmentor: DurationFPSTextTimeStamps | None = None
        if append_duration_fps_timestamps and self.prompt_json_formatter is None:
            self.duration_fps_augmentor = DurationFPSTextTimeStamps(
                input_keys=[caption_key, "video", "conditioning_fps"],
                output_keys=[caption_key],
                args={"caption_key": caption_key, "video_key": "video", "fps_key": "conditioning_fps"},
            )

        # --- Resolution text augmentor (runs before tokenization) ---
        self.resolution_info_augmentor: ResolutionTextInfo | None = None
        if append_resolution_info and self.prompt_json_formatter is None:
            self.resolution_info_augmentor = ResolutionTextInfo(
                input_keys=[caption_key, "video", "image_size"],
                output_keys=[caption_key],
                args={"caption_key": caption_key, "video_key": "video", "enabled": True},
            )

        # --- IdleFrames text augmentor (Pi0.7-style episode metadata) ---
        # Runs after resolution info, before tokenization. Per-field dropout is
        # independent from the tokenizer's global cfg_dropout_rate.
        self.idle_frames_augmentor: IdleFramesTextInfo | None = None
        if append_idle_frames and self.prompt_json_formatter is None:
            self.idle_frames_augmentor = IdleFramesTextInfo(
                input_keys=[caption_key, "idle_frames", "action"],
                output_keys=[caption_key],
                args={
                    "caption_key": caption_key,
                    "idle_frames_key": "idle_frames",
                    "action_key": "action",
                    "dropout_rate": idle_frames_dropout,
                    "enabled": True,
                },
            )

        # --- Text tokenizer augmentor ---
        self.text_tokenizer: TextTokenizerTransform | None = None
        if tokenizer_config is not None:
            self.text_tokenizer = TextTokenizerTransform(
                input_keys=[caption_key],
                output_keys=[text_token_key],
                args={
                    "tokenizer_config": tokenizer_config,
                    "cfg_dropout_rate": cfg_dropout_rate,
                },
            )

    def __call__(self, data_dict: dict, resolution: str | None) -> dict:
        """Apply the transform pipeline to a single data dictionary.

        Resolution is required at call time and is the only source of truth
        for this sample. When ``resolution`` is ``None``, the tier is
        auto-detected from the video's spatial dimensions.

        The pipeline runs in order:

        1. Resize + reflection-pad spatial dimensions to the closest
           predefined target from ``VIDEO_RES_SIZE_INFO[resolution]``.
        2. Format the caption as a structured JSON prompt (if enabled).
        3. Otherwise, append viewpoint type metadata to caption (if enabled).
        4. Append duration/FPS metadata to caption (if enabled).
        5. Append resolution metadata to caption (if enabled).
        6. Append idle-frame metadata (Pi0.7-style) to caption unless the
           sample is in inverse dynamics mode (if enabled).
        7. Tokenize caption text (if enabled).
        8. Build a ``SequencePlan`` from the ``"mode"`` key (if present).
        9. If action is needed by the plan, pad ``"action"`` to ``max_action_dim``.
        10. Otherwise, nullify ``"action"`` and ``"domain_id"`` (e.g. in
           ``"image2video"`` mode).

        Args:
            data_dict: A sample dictionary as returned by a Action dataset.
            resolution: Resolution tier key (e.g. ``"256"``, ``"480"``, ``"720"``)
                for this sample. When ``None``, auto-detected from video dimensions.

        Returns:
            The same dictionary, mutated in-place with padded tensors,
            ``image_size``, tokenized text IDs, and a
            ``"sequence_plan"`` entry added.
        """
        mode = data_dict.get("mode")
        assert mode is not None, "mode is required"

        # 1. Resize + reflection-pad spatial dimensions to the closest predefined target from ``VIDEO_RES_SIZE_INFO[resolution]``.
        data_dict = self.video_resize(data_dict, resolution)

        # 2. Format the caption as structured JSON when requested; otherwise run the legacy string appenders.
        if self.prompt_json_formatter is not None:
            data_dict = self.prompt_json_formatter(data_dict)
        else:
            # 3. Append viewpoint type metadata to caption (if enabled).
            if self.viewpoint_augmentor is not None:
                result = self.viewpoint_augmentor(data_dict)
                if result is not None:
                    data_dict = result

            # 4. Append duration/FPS metadata to caption (if enabled).
            if self.duration_fps_augmentor is not None:
                result = self.duration_fps_augmentor(data_dict)
                if result is not None:
                    data_dict = result

            # 5. Append resolution metadata to caption (if enabled).
            if self.resolution_info_augmentor is not None:
                result = self.resolution_info_augmentor(data_dict)
                if result is not None:
                    data_dict = result

            # 6. Append idle-frame metadata to caption (if enabled for this mode).
            if self.idle_frames_augmentor is not None and _should_append_idle_frame_info(mode):
                result = self.idle_frames_augmentor(data_dict)
                if result is not None:
                    data_dict = result

        # 7. Tokenize caption text (if enabled).
        if self.text_tokenizer is not None:
            data_dict = self.text_tokenizer(data_dict)

        # 8. Build a ``SequencePlan`` from the ``"mode"`` key (if present).
        video = data_dict.get("video")
        action = data_dict.get("action")
        assert video is not None, "video is required"
        video_length = video.shape[1]  # [C,T,H,W] -> T
        action_length = action.shape[0] if isinstance(action, torch.Tensor) else max(video_length - 1, 0)

        # Prepend history action frames (ground-truth conditioning) if present.
        history_action = data_dict.pop("history_action", None)
        num_history_actions = 0
        if history_action is not None and isinstance(action, torch.Tensor):
            num_history_actions = history_action.shape[0]
            action = torch.cat([history_action, action], dim=0)
            action_length += num_history_actions

        sequence_plan = build_sequence_plan_from_mode(
            mode=mode,
            video_length=video_length,
            action_length=action_length,
            video_temporal_downsample=self.video_temporal_downsample,
            num_history_actions=num_history_actions,
        )
        data_dict["sequence_plan"] = sequence_plan

        if sequence_plan.has_action:
            assert isinstance(action, torch.Tensor), "action tensor is required when sequence plan has action"
            data_dict["raw_action_dim"] = torch.tensor(action.shape[1]) if self.action_channel_masking else None
            data_dict["action"] = pad_action_to_max_dim(action, self.max_action_dim)
        else:
            # Nullify action-related fields when action is not needed so the
            # collate function can simply stack all non-None actions.
            data_dict["raw_action_dim"] = None
            data_dict["action"] = None
            data_dict["domain_id"] = None

        return data_dict
