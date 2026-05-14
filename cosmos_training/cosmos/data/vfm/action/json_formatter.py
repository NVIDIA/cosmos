from __future__ import annotations

import math

import torch

from cosmos.utils import log
from cosmos.data.vfm.action.viewpoint_utils import DEFAULT_VIEWPOINT_TEMPLATES
from cosmos.data.vfm.utils import VIDEO_RES_SIZE_INFO


def _should_append_idle_frame_info(mode: object) -> bool:
    """Return whether idle-frame prompt metadata should be surfaced."""
    return mode != "inverse_dynamics"


class ActionPromptJsonFormatter:
    """Format action prompts into a structured JSON-compatible dictionary.

    JSON fields are emitted in this order: ``cinematography``, ``actions``,
    ``duration``, ``fps``, ``resolution``, then ``aspect_ratio``. Like video JSON
    prompts, ``cinematography`` is a dictionary, duration is truncated to an
    integer-second string such as ``"2s"``, and aspect ratio is stored as a
    comma-separated string such as ``"16,9"``. If ``data_dict["mode"]`` is
    ``"inverse_dynamics"``, idle-frame metadata is omitted from the prompt.
    """

    def __init__(
        self,
        caption_key: str = "ai_caption",
        viewpoint_key: str = "viewpoint",
        video_key: str = "video",
        fps_key: str = "conditioning_fps",
        image_size_key: str = "image_size",
        idle_frames_key: str = "idle_frames",
        total_frames_key: str = "idle_frames_total",
        action_key: str = "action",
        viewpoint_templates: dict[str, str] | None = None,
    ) -> None:
        self.caption_key: str = caption_key
        self.viewpoint_key: str = viewpoint_key
        self.video_key: str = video_key
        self.fps_key: str = fps_key
        self.image_size_key: str = image_size_key
        self.idle_frames_key: str = idle_frames_key
        self.total_frames_key: str = total_frames_key
        self.action_key: str = action_key
        self.viewpoint_templates: dict[str, str] = (
            viewpoint_templates if viewpoint_templates is not None else DEFAULT_VIEWPOINT_TEMPLATES
        )

    def __call__(self, data_dict: dict) -> dict:
        """Replace the caption with the action JSON prompt structure."""
        additional_view_description = data_dict.pop("additional_view_description", None)
        caption = data_dict.get(self.caption_key)
        if not isinstance(caption, str) or caption == "":
            return data_dict

        height, width = self._get_resolution(data_dict)
        fps = self._get_scalar_float(data_dict.get(self.fps_key), self.fps_key)
        if fps <= 0:
            raise ValueError(f"ActionPromptJsonFormatter: '{self.fps_key}' must be positive, got {fps}")

        video = data_dict.get(self.video_key)
        if not isinstance(video, torch.Tensor) or video.ndim < 2:
            raise ValueError(
                f"ActionPromptJsonFormatter: expected '{self.video_key}' to be a video tensor with shape "
                f"(C, T, H, W), got {type(video).__name__}"
            )
        duration_seconds = video.shape[1] / fps
        duration = self._truncate_seconds(duration_seconds)
        action_end_time = self._round_time_seconds(duration_seconds)

        prompt = {
            "cinematography": {
                "framing": self._get_viewpoint_caption(data_dict, additional_view_description),
            },
            "actions": [
                {
                    "time": f"0:00-{self._format_time_mss(action_end_time)}",
                    "description": self._ensure_sentence(caption),
                    "idle_frame": self._get_idle_frame_info(data_dict),
                }
            ],
            "duration": f"{duration}s",
            "fps": float(fps),
            "resolution": {"H": height, "W": width},
            "aspect_ratio": self._get_aspect_ratio(width, height),
        }
        cleaned_prompt = self._drop_empty_fields(prompt)
        self._raise_if_empty_fields(cleaned_prompt)
        data_dict[self.caption_key] = cleaned_prompt
        return data_dict

    def _truncate_seconds(self, seconds: float) -> int:
        """Truncate duration to integer seconds, matching video JSON-caption augmentors."""
        if seconds < 0 or not math.isfinite(seconds):
            return 0
        return int(seconds)

    def _round_time_seconds(self, seconds: float) -> int:
        """Round an action timestamp to integer seconds, matching video captioning."""
        if seconds < 0 or not math.isfinite(seconds):
            return 0
        return round(seconds)

    def _format_time_mss(self, seconds: int) -> str:
        """Format integer seconds as M:SS for JSON prompt time ranges."""
        minutes, remaining_seconds = divmod(seconds, 60)
        return f"{minutes}:{remaining_seconds:02d}"

    def _get_aspect_ratio(self, width: int, height: int) -> str:
        """Return the canonical width,height aspect ratio string when known."""
        for aspect_ratio_sizes in VIDEO_RES_SIZE_INFO.values():
            for aspect_ratio, (candidate_w, candidate_h) in aspect_ratio_sizes.items():
                if width == candidate_w and height == candidate_h:
                    return aspect_ratio

        divisor = math.gcd(width, height)
        if divisor == 0:
            raise ValueError(
                f"ActionPromptJsonFormatter: width and height must be non-zero, got width={width}, height={height}."
            )
        return f"{width // divisor},{height // divisor}"

    def _get_viewpoint_caption(self, data_dict: dict, additional_view_description: object | None) -> str | None:
        """Resolve the viewpoint text used in the ``cinematography`` field."""
        viewpoint = data_dict.get(self.viewpoint_key)
        template = self.viewpoint_templates.get(viewpoint) if isinstance(viewpoint, str) else None

        if template is None:
            if viewpoint is not None:
                log.warning(
                    f"ActionPromptJsonFormatter: unrecognized viewpoint {viewpoint!r}. "
                    f"Known viewpoints: {sorted(self.viewpoint_templates.keys())}. "
                    f"Using additional view description when available.",
                    rank0_only=False,
                )
            return self._get_optional_text(additional_view_description)

        if additional_view_description:
            separator = " " if template.endswith(".") else ". "
            template = template + separator + str(additional_view_description).rstrip()
        return template

    def _get_resolution(self, data_dict: dict) -> tuple[int, int]:
        """Resolve ``(height, width)`` from the post-padding image size."""
        image_size = data_dict.get(self.image_size_key)
        if image_size is None:
            raise ValueError(f"ActionPromptJsonFormatter: missing '{self.image_size_key}' in data_dict.")

        if isinstance(image_size, torch.Tensor):
            if image_size.numel() < 2:
                raise ValueError(
                    f"ActionPromptJsonFormatter: expected '{self.image_size_key}' to contain at least "
                    f"height and width, got shape {tuple(image_size.shape)}"
                )
            return int(image_size[0].item()), int(image_size[1].item())

        try:
            return int(image_size[0]), int(image_size[1])
        except (TypeError, ValueError, IndexError) as e:
            raise ValueError(
                f"ActionPromptJsonFormatter: expected '{self.image_size_key}' to contain height and width."
            ) from e

    def _get_scalar_float(self, value: object, key: str) -> float:
        """Parse a required scalar float from a tensor or Python value."""
        if value is None:
            raise ValueError(f"ActionPromptJsonFormatter: missing '{key}' in data_dict.")

        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                raise ValueError(
                    f"ActionPromptJsonFormatter: expected scalar tensor at '{key}', got shape {tuple(value.shape)}"
                )
            return float(value.item())

        if isinstance(value, (str, int, float)):
            try:
                return float(value)
            except ValueError as e:
                raise ValueError(
                    f"ActionPromptJsonFormatter: expected scalar float-compatible value at '{key}'."
                ) from e
        raise ValueError(f"ActionPromptJsonFormatter: expected scalar float-compatible value at '{key}'.")

    def _get_optional_scalar_int(self, value: object, key: str) -> int | None:
        """Parse an optional scalar integer metadata value."""
        if value is None:
            return None

        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                log.warning(
                    f"ActionPromptJsonFormatter: expected scalar tensor at '{key}', got shape "
                    f"{tuple(value.shape)}. Skipping.",
                    rank0_only=False,
                )
                return None
            return int(value.item())

        if isinstance(value, (str, int, float)):
            try:
                return int(value)
            except ValueError:
                pass
        log.warning(
            f"ActionPromptJsonFormatter: expected integer-compatible value at "
            f"'{key}', got {type(value).__name__}. Skipping.",
            rank0_only=False,
        )
        return None

    def _get_total_frames(self, data_dict: dict) -> int | None:
        """Resolve the total action-frame count for idle-frame text."""
        total_frames = self._get_optional_scalar_int(data_dict.get(self.total_frames_key), self.total_frames_key)
        if total_frames is not None:
            return total_frames

        action = data_dict.get(self.action_key)
        if isinstance(action, torch.Tensor):
            if action.ndim == 0:
                log.warning(
                    f"ActionPromptJsonFormatter: expected action tensor at "
                    f"'{self.action_key}' to have a frame dimension. Skipping total frames.",
                    rank0_only=False,
                )
                return None
            return int(action.shape[0])

        try:
            return len(action) if action is not None else None
        except TypeError:
            return None

    def _get_idle_frame_info(self, data_dict: dict) -> str | None:
        """Build the idle-frame string for the action object."""
        if not _should_append_idle_frame_info(data_dict.get("mode")):
            return None

        idle_frames = self._get_optional_scalar_int(data_dict.get(self.idle_frames_key), self.idle_frames_key)
        total_frames = self._get_total_frames(data_dict)

        if idle_frames is not None and total_frames is not None:
            return f"{idle_frames} out of {total_frames}."
        if idle_frames is not None:
            return f"{idle_frames}."
        return None

    def _ensure_sentence(self, text: str) -> str:
        """Return text with terminal sentence punctuation."""
        text = text.strip()
        if text.endswith((".", "!", "?")):
            return text
        return f"{text}."

    def _get_optional_text(self, value: object) -> str | None:
        """Return stripped text, leaving empty optional text for the final prune pass."""
        if value is None:
            return None
        text = str(value).rstrip()
        return text if text else None

    def _drop_empty_fields(self, value: object) -> object:
        """Recursively remove empty strings, dictionaries, lists, and ``None`` values."""
        if isinstance(value, dict):
            return {
                key: cleaned
                for key, item in value.items()
                if not self._is_empty(cleaned := self._drop_empty_fields(item))
            }
        if isinstance(value, list):
            return [cleaned for item in value if not self._is_empty(cleaned := self._drop_empty_fields(item))]
        return value

    def _is_empty(self, value: object) -> bool:
        """Return whether a JSON field should be dropped."""
        return value is None or value == "" or value == [] or value == {}

    def _raise_if_empty_fields(self, value: object, path: str = "prompt") -> None:
        """Validate that no empty JSON fields remain after pruning."""
        if self._is_empty(value):
            raise ValueError(f"ActionPromptJsonFormatter: empty field remains at {path}.")

        if isinstance(value, dict):
            for key, item in value.items():
                self._raise_if_empty_fields(item, f"{path}.{key}")
        elif isinstance(value, list):
            for index, item in enumerate(value):
                self._raise_if_empty_fields(item, f"{path}[{index}]")
