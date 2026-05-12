# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import json
import math
import os
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Literal, Self, cast, override

import pydantic
import pynvml
from typing_extensions import assert_never
from tyro.conf import Suppress

from cosmos3.common.args import (
    IMAGE_EXTENSIONS,
    VIDEO_EXTENSIONS,
    ArgsBase,
    CfgpSize,
    CheckpointConfig,
    OverridesBase,
    ResolvedFilePath,
    ResolvedFilePathOrUrl,
    SampleArgs,
    SampleOverrides,
    SetupArgs,
    SetupOverrides,
    StrEnum,
    Training,
    _deep_merge,
    download_file,
)
from cosmos3.flags import EARLY_ACCESS
from cosmos3._src.imaginaire.flags import SMOKE, TRAINING
from cosmos3._src.imaginaire.utils import log
from cosmos3._src.imaginaire.utils.checkpoint_db import CheckpointDirHf

if TYPE_CHECKING:
    from cosmos3.common.inference import Inference
    from cosmos3._src.vfm.configs.base.defaults.model_config import OmniMoTModelConfig

_PACKAGE_DIR = Path(__file__).parent


@cache
def _load_modality_defaults(model_mode: str) -> dict[str, Any]:
    default_file = _PACKAGE_DIR / f"defaults/{model_mode}/sample_args.json"
    if not default_file.exists():
        raise FileNotFoundError(f"Missing modality defaults: {default_file}")
    return json.loads(default_file.read_text())


Guidance = Annotated[float, pydantic.Field(ge=0, le=7)]
GuidanceInterval = tuple[pydantic.NonNegativeFloat, pydantic.NonNegativeFloat]


class SamplingArgs(ArgsBase):
    num_steps: pydantic.PositiveInt
    guidance: Guidance
    guidance_interval: GuidanceInterval | None
    normalize_cfg: bool
    shift: float
    sigma_max: float


class SamplingOverrides(OverridesBase):
    """Sampling arguments for 'OmniMoTModel.generate_samples'."""

    num_steps: Training[pydantic.PositiveInt | None] = None
    """Number of steps for the diffusion model."""
    guidance: Training[Guidance | None] = None
    """Guidance scale for the diffusion model."""
    guidance_interval: Training[GuidanceInterval | None] = None
    """Guidance interval for the diffusion model."""
    normalize_cfg: Training[bool | None] = None
    """If True, normalize the CFG output."""
    shift: Training[float | None] = None
    """Shift in the UniPC sampler. Ignored when sampler='edm'."""
    sigma_max: Training[float | None] = None
    """Maximum sigma for the EDM sampler. Ignored when sampler='unipc'."""

    def _build_sampling(self, model_config: "OmniMoTModelConfig", sample_meta: "SampleMeta"):
        assert self.num_steps is not None
        if SMOKE:
            self.num_steps = min(self.num_steps, 1)


InferenceResolution = Literal["256", "480", "720", "1080"]
if TRAINING:
    Resolution = Literal["256", "480", "704", "720", "1080"]
else:
    Resolution = InferenceResolution
AspectRatio = Literal["1,1", "4,3", "3,4", "16,9", "9,16"]

# Resolutions that only support image generation (num_frames == 1). Video
# generation at these resolutions is rejected by ``_build_vision_data`` because
# the model wasn't trained on temporal data above 720p and ``MAX_NUM_FRAMES``
# has no entry for them.
IMAGE_ONLY_RESOLUTIONS: frozenset[str] = frozenset({"1080"})

MIN_NUM_FRAMES = 24
MAX_NUM_FRAMES: dict[Resolution, int] = {
    "256": 400,
    "480": 300,
    "704": 200,
    "720": 200,
}


class ModelVariant(StrEnum):
    VFM = "vfm"
    ACTION = "action"


ModelSize = Literal["0.6B", "2B", "8B", "30B-A3B", "32B", "235B-A22B"]
PromptUpsamplerVariant = Literal["8B", "32B"]


class ModelMode(StrEnum):
    TEXT2IMAGE = "text2image"
    TEXT2VIDEO = "text2video"
    IMAGE2IMAGE = "image2image"
    IMAGE2VIDEO = "image2video"
    VIDEO2VIDEO = "video2video"

    # Action
    FORWARD_DYNAMICS = "forward_dynamics"
    INVERSE_DYNAMICS = "inverse_dynamics"
    POLICY = "policy"


class VisionMode(StrEnum):
    IMAGE = "image"
    VIDEO = "video"


class ConditionVisionMode(StrEnum):
    IMAGE = "image"
    VIDEO = "video"


class ActionMode(StrEnum):
    POLICY = "policy"
    FORWARD_DYNAMICS = "forward_dynamics"
    INVERSE_DYNAMICS = "inverse_dynamics"
    IMAGE2VIDEO = "image2video"


class NegativeMetadataMode(StrEnum):
    NONE = "none"
    SAME = "same"
    INVERSE = "inverse"


class TransferHintKey(StrEnum):
    EDGE = "edge"
    BLUR = "blur"
    DEPTH = "depth"
    SEG = "seg"
    WSM = "wsm"


class PresetEdgeThreshold(StrEnum):
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class PresetBlurStrength(StrEnum):
    NONE = "none"
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class TransferArgs(ArgsBase):
    """Resolved transfer inference arguments for a single control hint."""

    control_path: ResolvedFilePathOrUrl | None = None


class EdgeTransferArgs(TransferArgs):
    preset_edge_threshold: PresetEdgeThreshold = PresetEdgeThreshold.MEDIUM


class BlurTransferArgs(TransferArgs):
    preset_blur_strength: PresetBlurStrength = PresetBlurStrength.MEDIUM


class TransferOverrides(OverridesBase):
    """Transfer inference overrides for a single control hint (all optional)."""

    control_path: ResolvedFilePathOrUrl | None = None
    """Path or URL to pre-computed control input."""

    def download(self, output_dir: Path):
        if self.control_path is not None:
            self.control_path = download_file(self.control_path, output_dir, "transfer_control")


class EdgeTransferOverrides(TransferOverrides):
    preset_edge_threshold: PresetEdgeThreshold | None = None
    """Edge detection threshold preset."""


class BlurTransferOverrides(TransferOverrides):
    preset_blur_strength: PresetBlurStrength | None = None
    """Blur strength preset."""


class SampleMeta(pydantic.BaseModel):
    model_variant: ModelVariant
    model_mode: ModelMode
    vision_mode: VisionMode
    condition_vision_mode: ConditionVisionMode | None
    action_mode: ActionMode | None = None


RESOLUTION_ADAPTER = pydantic.TypeAdapter(Resolution)
ASPECT_RATIO_ADAPTER = pydantic.TypeAdapter(AspectRatio)

DEFAULT_CONDITION_FRAME_INDEXES_VISION: dict[ConditionVisionMode, list[int]] = {
    ConditionVisionMode.IMAGE: [0],
    ConditionVisionMode.VIDEO: [0, 1],
}


class TextDataArgs(ArgsBase):
    prompt: str

    negative_prompt: str | None

    duration_template: str | None
    resolution_template: str | None
    negative_metadata_mode: NegativeMetadataMode
    inverse_duration_template: str
    inverse_resolution_template: str
    negative_prompt_keep_metadata: bool


class TextDataOverrides(OverridesBase):
    prompt_path: ResolvedFilePath | None = None
    """Path to a .txt file containing the prompt. Only one of 'prompt' or 'prompt_path' should be provided."""
    prompt: str | None = None
    """Text prompt for generation. Only one of 'prompt' or 'prompt_path' should be provided."""

    negative_prompt: str | None = None
    """Negative prompt - describing what you don't want in the generated video."""

    duration_template: Training[str | None] = None
    """Template string for appending duration/fps to prompt. Use {duration} and {fps} placeholders."""
    resolution_template: Training[str | None] = None
    """Template string for appending resolution to prompt. Use {height} and {width} placeholders."""
    negative_metadata_mode: Training[NegativeMetadataMode | None] = None
    """Negative prompt metadata mode: 'none', 'same', or 'inverse'."""
    inverse_duration_template: Training[str | None] = None
    """Inverse template for duration/fps metadata in the negative prompt."""
    inverse_resolution_template: Training[str | None] = None
    """Inverse template for resolution metadata in the negative prompt."""
    negative_prompt_keep_metadata: Training[bool | None] = None
    """Compatibility flag. If True and mode is 'none', mode is promoted to 'same'."""

    def _build_text_data(self, model_config: "OmniMoTModelConfig", sample_meta: SampleMeta):
        if self.prompt_path is not None:
            self.prompt = self.prompt_path.read_text().strip()
        if self.prompt is None:
            self.prompt = ""

        if self.negative_prompt_keep_metadata and self.negative_metadata_mode == NegativeMetadataMode.NONE:
            self.negative_metadata_mode = NegativeMetadataMode.SAME


Fps = Annotated[int, pydantic.Field(ge=1)]
VideoSaveQuality = Annotated[int, pydantic.Field(ge=0, le=10)]
ImageSaveQuality = Annotated[int, pydantic.Field(ge=0, le=100)]


class _VisionDataBase:
    @property
    def vision_mode(self) -> VisionMode:
        self = cast(VisionDataOverrides, self)
        return VisionMode.IMAGE if self.num_frames == 1 else VisionMode.VIDEO

    @property
    def condition_vision_mode(self) -> ConditionVisionMode | None:
        self = cast(VisionDataOverrides, self)
        if self.vision_path is not None:
            vision_ext = Path(self.vision_path).suffix.lower()
            if vision_ext in IMAGE_EXTENSIONS:
                return ConditionVisionMode.IMAGE
            elif vision_ext in VIDEO_EXTENSIONS:
                return ConditionVisionMode.VIDEO
            else:
                raise ValueError(f"Invalid vision extension: {vision_ext}")
        else:
            return None


class VisionDataArgs(ArgsBase, _VisionDataBase):
    vision_path: ResolvedFilePath | None
    condition_frame_indexes_vision: list[int]

    resolution: Resolution | None
    aspect_ratio: AspectRatio | None
    fps: pydantic.PositiveInt
    num_frames: pydantic.PositiveInt
    video_save_quality: VideoSaveQuality
    image_save_quality: ImageSaveQuality

    @property
    def duration(self) -> float:
        return self.num_frames / self.fps

    @property
    def vision_size(self) -> tuple[int, int]:
        """Vision size (width, height) in pixels."""
        from cosmos3._src.vfm.datasets.utils import IMAGE_RES_SIZE_INFO, VIDEO_RES_SIZE_INFO

        assert self.resolution
        assert self.aspect_ratio
        if self.num_frames == 1:
            return IMAGE_RES_SIZE_INFO[self.resolution][self.aspect_ratio]
        else:
            return VIDEO_RES_SIZE_INFO[self.resolution][self.aspect_ratio]

    @property
    def vision_extension(self) -> str:
        match self.vision_mode:
            case VisionMode.IMAGE:
                return ".jpg"
            case VisionMode.VIDEO:
                return ".mp4"
            case _:
                assert_never(self.vision_mode)


class VisionDataOverrides(OverridesBase, _VisionDataBase):
    # Vision condition fields
    vision_path: ResolvedFilePathOrUrl | None = None
    """Path or URL to conditioning image/video."""
    condition_frame_indexes_vision: Training[list[int] | None] = None
    """Latent frame indices to condition on. Defaults to [0] for image, [0, 1] for video."""

    # Vision fields
    resolution: Resolution | None = None
    """Vision resolution.
    
    Defaults to model config resolution.
    """
    aspect_ratio: AspectRatio | None = None
    """Vision aspect ratio. When None, image_edit preserves the input image's native
    aspect ratio; all other modes default to 16:9."""
    fps: Fps | None = None
    """Vision frames per second. Recommended range [10, 30]; quality may be degraded outside of this range."""
    num_frames: pydantic.PositiveInt | None = None
    """Number of vision frames.

    Range by resolution: 256p: [24, 400], 480p: [24, 300], 720p: [24, 200].
    Image-only resolutions (e.g. 1080p) require num_frames=1.
    """
    video_save_quality: Training[VideoSaveQuality | None] = None
    """Quality of the saved video (0-10)."""
    image_save_quality: Training[ImageSaveQuality | None] = None
    """Quality of the saved image (0-100)."""

    @override
    def download(self, output_dir: Path):
        super().download(output_dir)
        self.vision_path = download_file(self.vision_path, output_dir, "vision")

    def _build_vision_data(self, model_config: "OmniMoTModelConfig", sample_meta: SampleMeta):
        """Finalize and validate in-place."""
        if self.vision_path and "://" in self.vision_path:
            raise ValueError("Must call `download()` before building vision data")

        if self.condition_frame_indexes_vision is None:
            if sample_meta.condition_vision_mode:
                self.condition_frame_indexes_vision = DEFAULT_CONDITION_FRAME_INDEXES_VISION[
                    sample_meta.condition_vision_mode
                ]
            else:
                self.condition_frame_indexes_vision = []

        # Image edit defaults to input image size
        if sample_meta.model_mode != ModelMode.IMAGE2IMAGE:
            if self.resolution is None:
                self.resolution = RESOLUTION_ADAPTER.validate_python(model_config.resolution)

        assert self.num_frames is not None
        if self.fps is not None and (self.fps < 10 or self.fps > 30):
            log.warning(f"FPS {self.fps} is outside the recommended range [10, 30]. Quality may be degraded.")
        if self.num_frames > 1:
            assert self.resolution is not None
            if self.resolution in IMAGE_ONLY_RESOLUTIONS:
                raise ValueError(
                    f"Resolution {self.resolution!r} only supports image generation (num_frames=1). "
                    f"For video, use one of: {sorted(MAX_NUM_FRAMES)}"
                )
            if self.num_frames < MIN_NUM_FRAMES or self.num_frames > MAX_NUM_FRAMES[self.resolution]:
                log.warning(
                    f"Number of frames {self.num_frames} is outside the recommended range [{MIN_NUM_FRAMES}, {MAX_NUM_FRAMES[self.resolution]}]. Quality may be degraded."
                )
        if SMOKE:
            self.num_frames = min(self.num_frames, 2)
        temporal_compression_factor = model_config.tokenizer.temporal_compression_factor
        self.num_frames = (
            math.ceil((self.num_frames - 1) / temporal_compression_factor) * temporal_compression_factor + 1
        )


class SoundDataArgs(ArgsBase):
    enable_sound: bool = False


class SoundDataOverrides(OverridesBase):
    """Sound data overrides."""



class ActionDataArgs(ArgsBase):
    action_mode: ActionMode | None = None
    action_path: ResolvedFilePath | None = None
    domain_name: str = ""
    image_size: pydantic.PositiveInt = 256
    action_chunk_size: pydantic.PositiveInt = 16
    raw_action_dim: int | None = None


class ActionDataOverrides(OverridesBase):
    """Action data overrides."""

    action_mode: Training[ActionMode | None] = None
    """Action mode. When set, activates Action batch construction."""
    action_path: Training[ResolvedFilePathOrUrl | None] = None
    """Path to action JSON file. Required for forward_dynamics mode."""
    domain_name: Training[str | None] = None
    """Action domain name passed to get_domain_id()."""
    image_size: Training[pydantic.PositiveInt | None] = None
    """Target image height in pixels (aspect-ratio-preserving resize)."""
    action_chunk_size: Training[pydantic.PositiveInt | None] = None
    """Number of action steps to predict."""
    raw_action_dim: Training[pydantic.PositiveInt | None] = None
    """Dimension of the raw action data. Required when action_path is not provided."""

    @override
    def download(self, output_dir: Path):
        super().download(output_dir)
        self.action_path = download_file(self.action_path, output_dir, "action")

    def _build_action_data(self, model_config: "OmniMoTModelConfig", sample_meta: SampleMeta):
        if self.action_mode is not None:
            match self.action_mode:
                case ActionMode.IMAGE2VIDEO:
                    pass
                case ActionMode.FORWARD_DYNAMICS:
                    if self.action_path is None:
                        raise ValueError("'action_path' is required")
                case ActionMode.INVERSE_DYNAMICS | ActionMode.POLICY:
                    if self.raw_action_dim is None:
                        raise ValueError("'raw_action_dim' is required")
                case _:
                    assert_never(self.action_mode)
        if self.action_path and "://" in self.action_path:
            raise ValueError("Must call `download()` before building action data")

        if self.domain_name is None:
            self.domain_name = ""
        if self.image_size is None:
            self.image_size = 256
        if self.action_chunk_size is None:
            self.action_chunk_size = 16




class _SampleDataBase:
    @property
    def model_variant(self) -> ModelVariant:
        self = cast(SampleDataOverrides, self)
        if self.action_mode is not None:
            return ModelVariant.ACTION
        return ModelVariant.VFM

    @property
    def model_mode(self) -> ModelMode:
        self = cast(SampleDataOverrides, self)
        match self.model_variant:
            case ModelVariant.VFM:
                input_mode = self.condition_vision_mode or "text"
                output_mode = self.vision_mode
                return ModelMode(f"{input_mode}2{output_mode}")
            case ModelVariant.ACTION:
                assert self.action_mode
                return ModelMode(self.action_mode.value)
            case _:
                assert_never(self.model_variant)

    @property
    def sample_meta(self) -> SampleMeta:
        self = cast(SampleDataOverrides, self)
        return SampleMeta(
            model_variant=self.model_variant,
            model_mode=self.model_mode,
            vision_mode=self.vision_mode,
            condition_vision_mode=self.condition_vision_mode,
            action_mode=self.action_mode,
        )


class SampleDataArgs(
    _SampleDataBase,
    TextDataArgs,
    VisionDataArgs,
    SoundDataArgs,
    ActionDataArgs,
): ...


class SampleDataOverrides(
    _SampleDataBase,
    TextDataOverrides,
    VisionDataOverrides,
    SoundDataOverrides,
    ActionDataOverrides,
):
    """Sample data arguments for 'OmniMoTModel.generate_samples'."""


class OmniSampleArgs(SampleArgs, SamplingArgs, SampleDataArgs): ...


class OmniSampleOverrides(SampleOverrides, SamplingOverrides, SampleDataOverrides):
    defaults_file: ResolvedFilePath | None = None
    """Path to a JSON file of per-modality default sample fields. Overrides the built-in defaults."""


    _VLM_MODEL_SIZE: ClassVar[dict[str, ModelSize]] = {
        "Qwen/Qwen3-0.6B": "0.6B",
        "Qwen/Qwen3-VL-2B-Instruct": "2B",
        "Qwen/Qwen3-VL-8B-Instruct": "8B",
        "Qwen/Qwen3-VL-32B-Instruct": "32B",
        "Qwen/Qwen3-VL-30B-A3B-Instruct": "30B-A3B",
        "Qwen/Qwen3-VL-235B-A22B-Instruct": "235B-A22B",
    }

    _RESOLUTION_SHIFT_DEFAULTS: ClassVar[dict[(ModelSize, Resolution), float]] = {
        ("8B", "256"): 3.0,
        ("8B", "480"): 5.0,
        ("8B", "720"): 10.0,
        ("32B", "256"): 5.0,
        ("32B", "480"): 5.0,
        ("32B", "720"): 5.0,
    }

    @override
    def build_sample(self, *, model_config: Any) -> OmniSampleArgs:
        model_config = cast("OmniMoTModelConfig", model_config)
        sample_meta = self.sample_meta

        # Apply per-modality defaults from JSON config files.
        # User-provided values take precedence over JSON defaults.
        if self.defaults_file is not None:
            defaults = json.loads(self.defaults_file.read_text())
        else:
            defaults = _load_modality_defaults(sample_meta.model_mode)
        overrides = self.model_dump(exclude_none=True)
        user_specified_shift = "shift" in overrides
        merged_data = _deep_merge(defaults, overrides)
        merged_data = {k: v for k, v in merged_data.items() if k in type(self).model_fields}
        merged = type(self).model_validate(merged_data)

        self.__dict__.update(merged.__dict__)

        self._build_sample()
        self._build_sampling(model_config=model_config, sample_meta=sample_meta)
        self._build_text_data(model_config=model_config, sample_meta=sample_meta)
        self._build_vision_data(model_config=model_config, sample_meta=sample_meta)



        self._build_action_data(model_config=model_config, sample_meta=sample_meta)


        if not user_specified_shift:
            model_size = self._VLM_MODEL_SIZE[model_config.vlm_config.model_name]
            key = (model_size, self.resolution)
            if key in self._RESOLUTION_SHIFT_DEFAULTS:
                self.shift = self._RESOLUTION_SHIFT_DEFAULTS[key]

        return self._build(OmniSampleArgs)



_MODEL_MEMORY_FACTOR: int = int(1e9) * 2 * 2  # 1B params/tower * 2 bytes/param (bfloat16) * 2 towers
MODEL_MEMORY_BYTES_BY_SIZE: dict[ModelSize, int] = {
    "0.6B": round(0.6 * _MODEL_MEMORY_FACTOR),
    "2B": 2 * _MODEL_MEMORY_FACTOR,
    "8B": 8 * _MODEL_MEMORY_FACTOR,
    "30B-A3B": 30 * _MODEL_MEMORY_FACTOR,
    "32B": 32 * _MODEL_MEMORY_FACTOR,
    "235B-A22B": 235 * _MODEL_MEMORY_FACTOR,
}

_CHECKPOINTS_EXPERIMENTAL: dict[str, CheckpointConfig] = {
    # 0.6B
    "5d561d7d-080f-45cb-a455-920d444e40cc": CheckpointConfig(
        model_memory_bytes=MODEL_MEMORY_BYTES_BY_SIZE["0.6B"],
        s3_uri="s3://bucket/cosmos3_vfm/t2w_mot_0p6b_qwen3_vl_runs/t2w_mot_dryrun_exp200_001_qwen3_vl_0p6b_480res_qwen3_captions_mrope_v2/checkpoints/iter_000079500/",
        hf=CheckpointDirHf(
            repository="nvidia/Cosmos3-Experimental",
            subdirectory="5d561d7d-080f-45cb-a455-920d444e40cc",
            revision="844eb561ec6a8d6a917aec463464cdd594d5e965",
        ),
    ),
    # 8B
    "5fabe660-7021-4286-96ec-e1858d194b82": CheckpointConfig(
        model_memory_bytes=MODEL_MEMORY_BYTES_BY_SIZE["8B"],
        s3_uri="s3://bucket/cosmos3_vfm/t2w_mot_8b_qwen3_vl_runs/t2w_mot_exp506_000_qwen3_vl_8b_multires_recipe_midtraining_v1/checkpoints/iter_000040000/",
        hf=CheckpointDirHf(
            repository="nvidia/Cosmos3-Nano-Internal",
            revision="dd232bc1219d8dd724f54027a6f5c987b91f0623",
        ),
    ),
    # 32B
    "7415f3d4-91e5-4df4-baaa-f09ff4dafd5e": CheckpointConfig(
        model_memory_bytes=MODEL_MEMORY_BYTES_BY_SIZE["32B"],
        s3_uri="s3://bucket/cosmos3_vfm/t2w_mot_32b_qwen3_vl_runs/t2w_mot_exp305_000_qwen3_vl_32b_multires_v7/checkpoints/iter_000085000/",
        hf=CheckpointDirHf(
            repository="nvidia/Cosmos3-Super-Internal",
            revision="0cb696a450e36e2be48402d5815fdcca2d11050d",
        ),
    ),
}
_CHECKPOINTS_EA = {
    "5fabe660-7021-4286-96ec-e1858d194b82": CheckpointConfig(
        model_memory_bytes=MODEL_MEMORY_BYTES_BY_SIZE["8B"],
        s3_uri="s3://bucket/cosmos3_vfm/t2w_mot_8b_qwen3_vl_runs/t2w_mot_exp506_000_qwen3_vl_8b_multires_recipe_midtraining_v1/checkpoints/iter_000040000/",
        hf=CheckpointDirHf(
            repository="nvidia-cosmos-ea/Cosmos3-Nano",
            revision="ebbe03fe665661d0eab87130a341184efecca365",
        ),
    ),
    "7415f3d4-91e5-4df4-baaa-f09ff4dafd5e": CheckpointConfig(
        model_memory_bytes=MODEL_MEMORY_BYTES_BY_SIZE["32B"],
        s3_uri="s3://bucket/cosmos3_vfm/t2w_mot_32b_qwen3_vl_runs/t2w_mot_exp305_000_qwen3_vl_32b_multires_v7/checkpoints/iter_000085000/",
        hf=CheckpointDirHf(
            repository="nvidia-cosmos-ea/Cosmos3-Super",
            revision="824198211d88e556645ec1c38bd37c3581775268",
        ),
    ),
}
_CHECKPOINTS = _CHECKPOINTS_EXPERIMENTAL.copy()
if EARLY_ACCESS:
    _CHECKPOINTS.update(_CHECKPOINTS_EA)
_CHECKPOINTS["Cosmos3-Test"] = _CHECKPOINTS["Cosmos3-0.6B"] = _CHECKPOINTS["5d561d7d-080f-45cb-a455-920d444e40cc"]
_CHECKPOINTS["Cosmos3-Nano"] = _CHECKPOINTS["Cosmos3-8B"] = _CHECKPOINTS["5fabe660-7021-4286-96ec-e1858d194b82"]
_CHECKPOINTS["Cosmos3-Super"] = _CHECKPOINTS["Cosmos3-32B"] = _CHECKPOINTS["7415f3d4-91e5-4df4-baaa-f09ff4dafd5e"]
DEFAULT_CHECKPOINT_NAME = "Cosmos3-Test" if SMOKE else "Cosmos3-Nano"
DEFAULT_CHECKPOINT = _CHECKPOINTS[DEFAULT_CHECKPOINT_NAME]

# CP size cannot exceed number of KV heads (8)
MAX_CP_SIZE = 8
CpSize = Annotated[int, pydantic.Field(ge=1, le=MAX_CP_SIZE)]


class OmniSetupArgs(SetupArgs):
    variant: Suppress[Literal["omni"]] = "omni"
    """Discriminator."""

    # pyrefly: ignore[bad-override]
    sample_overrides: OmniSampleOverrides

    sampler: Literal["unipc", "edm"]

    # Override defaults
    cp_size: CpSize

    @override
    @classmethod
    def get_sample_overrides_cls(cls) -> type[SampleOverrides]:
        return OmniSampleOverrides

    @override
    @classmethod
    def get_sample_args_cls(cls) -> type[SampleArgs]:
        return OmniSampleArgs

    @override
    @classmethod
    def get_inference_cls(cls) -> type["Inference"]:
        from cosmos3.inference import OmniInference

        return OmniInference

    @pydantic.model_validator(mode="after")
    def _validate_parallelism(self) -> Self:
        world_size = int(os.environ.get("WORLD_SIZE", "0"))

        if world_size:
            if self.dp_shard_size * self.dp_replicate_size > world_size:
                raise ValueError(
                    f"dp_shard_size({self.dp_shard_size}) * dp_replicate_size({self.dp_replicate_size}) must be <= WORLD_SIZE({world_size})"
                )
            if world_size % (self.dp_shard_size * self.dp_replicate_size) != 0:
                raise ValueError(
                    f"dp_shard_size({self.dp_shard_size}) * dp_replicate_size({self.dp_replicate_size}) must divide WORLD_SIZE({world_size})"
                )

        if world_size:
            if self.cp_size * self.cfgp_size > world_size:
                raise ValueError(
                    f"cp_size({self.cp_size}) * cfgp_size({self.cfgp_size}) must be <= WORLD_SIZE({world_size})"
                )
            if world_size % (self.cp_size * self.cfgp_size) != 0:
                raise ValueError(
                    f"cp_size({self.cp_size}) * cfgp_size({self.cfgp_size}) must divide WORLD_SIZE({world_size})"
                )

        return self


class OmniSetupOverrides(SetupOverrides):
    variant: Suppress[Literal["omni"]] = "omni"
    """Discriminator."""

    CHECKPOINTS: ClassVar[dict[str, CheckpointConfig]] = _CHECKPOINTS

    sample_overrides: OmniSampleOverrides = OmniSampleOverrides()

    model_size: Training[ModelSize | None] = None

    sampler: Literal["unipc", "edm"] = "unipc"

    # Override defaults
    dp_replicate_size: pydantic.NonNegativeInt = 0
    dp_shard_size: pydantic.NonNegativeInt = 0
    cp_size: CpSize | Literal[0] = 0
    cfgp_size: CfgpSize | Literal[0] = 0

    use_cuda_graphs: bool = False

    compiled_region: Literal["all", "language"] = "all"
    # Unsupported
    tp_size: Suppress[pydantic.NonNegativeInt] = 1

    def _build_model_parallelism(self, world_size: int, device_memory_bytes: int):
        if not self.dp_shard_size:
            if self.model_memory_bytes:
                self.dp_shard_size = _get_dp_shard_size(
                    model_memory_bytes=self.model_memory_bytes,
                    device_memory_bytes=device_memory_bytes,
                    device_memory_utilization=self.device_memory_utilization,
                )
            else:
                self.dp_shard_size = 1
        if not self.dp_replicate_size:
            self.dp_replicate_size = max(1, world_size // self.dp_shard_size)

    def _build_context_parallelism(self, world_size: int):
        if not self.cfgp_size:
            match self.parallelism_preset:
                case "throughput":
                    self.cfgp_size = 1
                case "latency":
                    self.cfgp_size = max(1, min(2, world_size))
                case _:
                    assert_never(self.parallelism_preset)
        if not self.cp_size:
            match self.parallelism_preset:
                case "throughput":
                    self.cp_size = 1
                case "latency":
                    self.cp_size = max(1, min(MAX_CP_SIZE, world_size // self.cfgp_size))
                case _:
                    assert_never(self.parallelism_preset)

    @override
    def _build_parallelism(self, world_size: int | None, local_world_size: int | None, device_memory_bytes: int | None):
        if world_size is None:
            world_size = int(os.environ.get("WORLD_SIZE", "0"))
        if local_world_size is None:
            local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", str(world_size)))
        if device_memory_bytes is None:
            device_memory_bytes = _get_device_memory_bytes()

        if self.model_memory_bytes is None and self.model_size is not None:
            self.model_memory_bytes = MODEL_MEMORY_BYTES_BY_SIZE[self.model_size]
        self._build_model_parallelism(world_size=world_size, device_memory_bytes=device_memory_bytes)
        self._build_context_parallelism(world_size=world_size)

    @override
    def build_setup(
        self, world_size: int | None = None, local_world_size: int | None = None, device_memory_bytes: int | None = None
    ) -> OmniSetupArgs:
        self._build_setup()
        self._build_checkpoint(checkpoints=self.CHECKPOINTS)
        self._build_parallelism(
            world_size=world_size, local_world_size=local_world_size, device_memory_bytes=device_memory_bytes
        )
        return self._build(OmniSetupArgs)


def _get_dp_shard_size(
    model_memory_bytes: int, device_memory_bytes: int, device_memory_utilization: float = 0.75
) -> int:
    return math.ceil(model_memory_bytes / device_memory_bytes / device_memory_utilization)


@cache
def _get_device_memory_bytes() -> int:
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    pynvml.nvmlShutdown()
    return info.total
