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

import json
import pickle
import random
from collections.abc import Generator, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence, cast, override

import attrs
import cattrs
import cattrs.preconf.json
import safetensors.torch
import torch
from PIL import Image
from torch.utils._pytree import tree_map_only
from torch.utils.data import Dataset
from typing_extensions import Self

from cosmos3.args import (
    ModelMode,
    NegativeMetadataMode,
    OmniSampleArgs,
    OmniSetupArgs,
)
from cosmos3.common.args import (
    CheckpointType,
    ConfigFileType,
    ParallelismArgs,
    SampleArgs,
    SampleOutput,
    SampleOutputs,
    SetupArgs,
)
from cosmos3.common.inference import Inference, sync_distributed_errors
from cosmos3.common.init import get_rank, get_world_size
from cosmos3.model import Cosmos3OmniConfig, Cosmos3OmniModel
from cosmos3.vision import (
    build_conditioned_video_batch,
    build_image_edit_batch,
    load_conditioning_image,
    load_conditioning_video,
    pil_to_conditioning_frames,
    resize_pil_image,
)
from cosmos3._src.imaginaire.utils import log
from cosmos3._src.imaginaire.visualize.video import save_img_or_video
from cosmos3._src.vfm.configs.base.defaults.model_config import ParallelismConfig
from cosmos3._src.vfm.models.omni_mot_model import OmniMoTModel
from cosmos3._src.vfm.models.vlm.qwen3_vl.utils import _SYSTEM_PROMPT_IMAGE_EDITING

if TYPE_CHECKING:
    from cosmos3._src.vfm.configs.base.defaults.model_config import OmniMoTModelConfig


def _iter_batch_ranges(
    sampler_indices: list[int],
    sample_args_list: Sequence[OmniSampleArgs],
    model: OmniMoTModel,
    max_model_len: int | None,
    max_num_seqs: int | None,
) -> Generator[tuple[int, int]]:
    """Yield ``(start, end)`` position pairs into *sampler_indices* for each batch.

    The boundary logic accounts for either a token budget (``max_model_len``)
    or a sequence-count budget (``max_num_seqs``).  Exactly one must be set.
    """
    batch_start = 0
    running_tokens = 0
    running_seqs = 0

    for pos, sample_idx in enumerate(sampler_indices):
        sa = sample_args_list[sample_idx]
        assert isinstance(sa, OmniSampleArgs)
        assert sa.num_outputs == 1, "num_outputs must be 1"

        if max_model_len is not None:
            num_tokens = _compute_num_tokens_for_sample(sa, model)
            if running_tokens > 0 and running_tokens + num_tokens > max_model_len:
                yield (batch_start, pos)
                batch_start = pos
                running_tokens = 0
            running_tokens += num_tokens

        elif max_num_seqs is not None:
            if running_seqs > 0 and running_seqs + 1 > max_num_seqs:
                yield (batch_start, pos)
                batch_start = pos
                running_seqs = 0
            running_seqs += 1

        else:
            raise ValueError("Either max_model_len or max_num_seqs must be set")

    if running_tokens > 0 or running_seqs > 0:
        yield (batch_start, len(sampler_indices))


def _compute_num_tokens_for_sample(sample_args: OmniSampleArgs, model: OmniMoTModel) -> int:
    """Estimate the number of tokens for a single inference sample.

    Follows the counting logic in
    ``JointDataLoader._compute_num_tokens_per_sample`` (vision + text + EOS).
    """
    w, h = sample_args.vision_size
    T = sample_args.num_frames

    spatial_cf = cast(int, model.tokenizer_vision_gen.spatial_compression_factor)
    temporal_cf = cast(int, model.tokenizer_vision_gen.temporal_compression_factor)
    patch_spatial: int = model.config.diffusion_expert_config.patch_spatial

    vae_spatial_downsample = spatial_cf * patch_spatial
    vae_temporal_downsample = temporal_cf

    latent_h = h // vae_spatial_downsample
    latent_w = w // vae_spatial_downsample
    latent_t = 1 + (T - 1) // vae_temporal_downsample
    num_vision_tokens = latent_h * latent_w * latent_t


    # small compared to vision tokens, so we can ignore them for now.

    return num_vision_tokens


def _format_prompt_with_template(
    prompt: str,
    *,
    fps: int,
    num_frames: int,
    duration_template: str | None,
    resolution_template: str | None,
    h: int,
    w: int,
    force_duration_template: bool = False,
) -> str:
    """Append duration/fps and resolution metadata to a prompt."""
    prompt = prompt.strip()
    if duration_template is not None and (num_frames > 1 or force_duration_template):
        duration = num_frames / fps
        dur_text = duration_template.format(duration=duration, fps=fps)
        prompt = prompt.rstrip(".") + ". " + dur_text

    prompt = prompt.strip()
    if resolution_template is not None:
        res_text = resolution_template.format(height=h, width=w)
        prompt = prompt.rstrip(".") + ". " + res_text

    return prompt


def _parse_json_object_prompt(prompt: str) -> dict | None:
    """Return the parsed dict iff ``prompt`` is a JSON object string; else ``None``.

    JSON arrays / numbers / strings / nulls are NOT considered "JSON-object
    prompts" and return ``None`` so they continue down the plain-text path.
    """
    try:
        obj = json.loads(prompt)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    return obj if isinstance(obj, dict) else None


def _format_json_prompt_with_template(
    prompt_obj: dict,
    *,
    fps: int,
    num_frames: int,
    aspect_ratio: str | None,
    h: int,
    w: int,
) -> str:
    """JSON-prompt counterpart to ``_format_prompt_with_template``.

    Injects structured metadata fields directly into the parsed prompt object,
    matching the training-time augmentors so the tokenizer sees the exact
    schema the model was trained on:

        - ``ResolutionTextInfo``        -> ``resolution: {"H": int, "W": int}``, ``aspect_ratio: str``
        - ``DurationFPSTextTimeStamps`` -> ``duration: "<int>s"``, ``fps: float``

    Always overwrites existing values for these keys, mirroring the augmentors'
    ``dict.update(...)`` semantics: the actual generation specs are the source
    of truth, regardless of what the input prompt may have specified.
    """
    duration_seconds = int(num_frames / fps) if fps > 0 else 0
    metadata: dict[str, Any] = {
        "duration": f"{duration_seconds}s",
        "fps": float(fps),
        "resolution": {"H": int(h), "W": int(w)},
    }
    if aspect_ratio is not None:
        metadata["aspect_ratio"] = aspect_ratio

    prompt_obj.update(metadata)
    log.debug(f"Injected JSON-prompt metadata fields: {sorted(metadata.keys())}")

    return json.dumps(prompt_obj)


def _get_prompt_sample_data(sample_args: OmniSampleArgs, model: OmniMoTModel, *, h: int, w: int, device: Any) -> dict:
    duration_template = sample_args.duration_template
    inverse_duration_template = sample_args.inverse_duration_template
    prompt_obj = _parse_json_object_prompt(sample_args.prompt)
    prompt_is_json = prompt_obj is not None
    if not prompt_is_json:
        prompt = _format_prompt_with_template(
            sample_args.prompt,
            fps=sample_args.fps,
            num_frames=sample_args.num_frames,
            duration_template=duration_template,
            resolution_template=sample_args.resolution_template,
            h=h,
            w=w,
        )
    else:
        assert prompt_obj is not None  # type-narrowing
        prompt = _format_json_prompt_with_template(
            prompt_obj,
            fps=sample_args.fps,
            num_frames=sample_args.num_frames,
            aspect_ratio=sample_args.aspect_ratio,
            h=h,
            w=w,
        )
    out = {
        model.input_caption_key: [prompt] * sample_args.num_outputs,
    }

    negative_prompt = sample_args.negative_prompt
    if sample_args.negative_metadata_mode == NegativeMetadataMode.SAME:
        negative_prompt = (
            _format_prompt_with_template(
                negative_prompt if negative_prompt is not None else "",
                fps=sample_args.fps,
                num_frames=sample_args.num_frames,
                duration_template=duration_template,
                resolution_template=sample_args.resolution_template,
                h=h,
                w=w,
            )
            .lstrip(".")
            .strip()
        )
    elif sample_args.negative_metadata_mode == NegativeMetadataMode.INVERSE:
        negative_prompt = (
            _format_prompt_with_template(
                negative_prompt if negative_prompt is not None else "",
                fps=sample_args.fps,
                num_frames=sample_args.num_frames,
                duration_template=inverse_duration_template,
                resolution_template=sample_args.inverse_resolution_template,
                h=h,
                w=w,
                force_duration_template=True,
            )
            .lstrip(".")
            .strip()
        )

    if negative_prompt:
        neg_key = "neg_" + model.input_caption_key
        out[neg_key] = [negative_prompt] * sample_args.num_outputs

    return out


def _get_image_edit_sample_data(
    sample_args: OmniSampleArgs,
    model: OmniMoTModel,
    *,
    device: Any,
) -> dict:
    """Create a sample batch for image-edit generation."""
    assert sample_args.vision_path is not None
    if sample_args.resolution and sample_args.aspect_ratio:
        w, h = sample_args.vision_size
        conditioning_frames = load_conditioning_image(Path(sample_args.vision_path), target_h=h, target_w=w)
    else:
        pil_img = Image.open(sample_args.vision_path).convert("RGB")
        pil_img = resize_pil_image(pil_img, max_size=512, padding_constant=32)
        conditioning_frames, h, w = pil_to_conditioning_frames(pil_img)

    conditioning_frames = conditioning_frames.to(device=device)
    batch = build_image_edit_batch(conditioning_frames, h=h, w=w, batch_size=sample_args.num_outputs)
    batch["system_prompt"] = _SYSTEM_PROMPT_IMAGE_EDITING
    batch |= _get_prompt_sample_data(sample_args, model, h=h, w=w, device=device)
    return batch


def get_sample_data(
    sample_args: OmniSampleArgs,
    model: OmniMoTModel,
    *,
    device: Any = "cuda",
) -> dict:
    """Create a sample batch for generation."""

    if sample_args.model_mode.is_action:
        from cosmos3.action import get_action_sample_data

        assert sample_args.vision_path is not None
        return get_action_sample_data(
            model_config=model,
            batch_size=sample_args.num_outputs,
            prompt=sample_args.prompt,
            vision_path=sample_args.vision_path,
            model_mode=sample_args.model_mode,
            action_path=sample_args.action_path,
            domain_name=sample_args.domain_name,
            resolution=str(sample_args.image_size),
            aspect_ratio=sample_args.aspect_ratio,
            action_chunk_size=sample_args.action_chunk_size,
            max_action_dim=model.config.max_action_dim,
            raw_action_dim=sample_args.raw_action_dim,
            duration_template=sample_args.duration_template,
            resolution_template=sample_args.resolution_template,
            fps=sample_args.fps,
            device=device,
        )


    if sample_args.model_mode == ModelMode.IMAGE2IMAGE:
        return _get_image_edit_sample_data(sample_args, model, device=device)

    w, h = sample_args.vision_size
    if sample_args.num_frames == 1:
        input_vision_key = model.input_image_key
    else:
        input_vision_key = model.input_video_key

    with torch.device(device):
        match sample_args.condition_vision_mode:
            case "image":
                assert sample_args.vision_path is not None
                conditioning_frames = load_conditioning_image(Path(sample_args.vision_path), target_h=h, target_w=w)
            case "video":
                assert sample_args.vision_path is not None
                assert sample_args.condition_frame_indexes_vision is not None
                num_condition_latent_frames = max(sample_args.condition_frame_indexes_vision) + 1
                max_frames = model.tokenizer_vision_gen.get_pixel_num_frames(num_condition_latent_frames)
                conditioning_frames = load_conditioning_video(
                    Path(sample_args.vision_path),
                    target_h=h,
                    target_w=w,
                    max_frames=max_frames,
                    keep=sample_args.condition_video_keep or "first",
                )
            case _:
                conditioning_frames = None

        if conditioning_frames is not None:
            assert sample_args.condition_frame_indexes_vision is not None
            conditioned = build_conditioned_video_batch(
                conditioning_frames,
                condition_frames_vision=sample_args.condition_frame_indexes_vision,
                w=w,
                h=h,
                num_frames=sample_args.num_frames,
                fps=sample_args.fps,
                batch_size=sample_args.num_outputs,
            )
            video_tensor = torch.cat(conditioned["video"], dim=0).to(device=device)  # [1,3,T,H,W]
            sequence_plan = conditioned["sequence_plan"]
        else:
            video_tensor = [
                torch.zeros(1, 3, sample_args.num_frames, h, w) for _ in range(sample_args.num_outputs)
            ]  # list of [1,3,T,H,W]
            sequence_plan = None

        out: dict = {
            input_vision_key: video_tensor,
            "image_size": [
                torch.tensor([[h, w, h, w]], dtype=torch.float32) for _ in range(sample_args.num_outputs)
            ],  # list of [1,4]
            "t5_text_embeddings": torch.randn(sample_args.num_outputs, 512, 1024, dtype=torch.bfloat16),  # [B,512,1024]
            "fps": torch.full((sample_args.num_outputs,), float(sample_args.fps)),  # [B]
            "conditioning_fps": torch.full((sample_args.num_outputs,), float(sample_args.fps)),  # [B]
            "num_frames": torch.full((sample_args.num_outputs,), sample_args.num_frames),  # [B]
            "is_preprocessed": True,
        }
        if sequence_plan is not None:
            out["sequence_plan"] = sequence_plan

        out |= _get_prompt_sample_data(sample_args, model, w=w, h=h, device=device)


        return out


def _merge_data_batches(batches: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge multiple single-sample data dicts into one batched dict.

    Values that are lists are concatenated. Tensors with a batch dimension are
    concatenated along dim 0. Scalar/bool values are taken from the first batch.

    Args:
        batches (list[dict[str, Any]]): List of data batches to merge.

    Returns:
        dict[str, Any]: Merged data batch.

    Raises:
        ValueError: If the batches have different keys.
        ValueError: If the scalar/bool values are not the same across all batches.
    """
    if len(batches) == 1:
        return batches[0]

    # First ensure that all batches have the same keys.
    reference_keys = set(batches[0].keys())
    for i, batch in enumerate(batches[1:], start=1):
        if set(batch.keys()) != reference_keys:
            raise ValueError(f"Batch {i} keys {set(batch.keys())} differ from batch 0 keys {reference_keys}")

    # Then merge the batches.
    merged: dict[str, Any] = {}
    keys = batches[0].keys()
    for key in keys:
        values = [b[key] for b in batches if key in b]
        first = values[0]
        if isinstance(first, list):
            merged[key] = [item for v in values for item in v]
        elif isinstance(first, torch.Tensor):
            assert first.ndim > 0, "Tensor must have at least one (batch) dimension"
            merged[key] = torch.cat(values, dim=0)
        else:
            if not all(v == values[0] for v in values):
                raise ValueError(f"Key {key} values are not the same: {values}")
            merged[key] = first
    return merged


def _finalize_sample_args_list(sample_args_list: Sequence[OmniSampleArgs], model: OmniMoTModel) -> list[OmniSampleArgs]:
    """
    Finalize sample arguments by checking and validating.
    Also, expand the sample arguments to multiple outputs if num_outputs > 1.
    """
    if all(sample_args.num_outputs == 1 for sample_args in sample_args_list):
        return list(sample_args_list)

    finalized_sample_args_list = []

    for sample_args in sample_args_list:
        seed = sample_args.seed
        num_outputs = sample_args.num_outputs
        output_dir = sample_args.output_dir

        for i in range(num_outputs):
            sample_args_i = sample_args.model_copy(deep=True)
            sample_args_i.seed = (seed + i) if seed is not None else None
            sample_args_i.num_outputs = 1
            sample_args_i.output_dir = output_dir / f"{i}"
            finalized_sample_args_list.append(sample_args_i)

    return finalized_sample_args_list


def create_batches_from_dataset(
    samples: Iterable[tuple[OmniSampleArgs, dict[str, Any]]],
    model: OmniMoTModel,
    *,
    max_num_seqs: int | None = None,
    max_model_len: int | None = None,
) -> Generator[tuple[list[OmniSampleArgs], dict[str, Any], list[dict[str, Any]]]]:
    """Create batches from pre-loaded (sample_args, data_batch) pairs.

    Reuses the same token-count / sample-count batching logic as
    ``OmniInference.create_batches``, but works with dataset iterators that
    already provide data. Samples with ``num_outputs > 1`` are multi-seed
    expanded via ``_finalize_sample_args_list``; callers that want only a
    subset of samples expanded should set ``num_outputs`` accordingly before
    yielding each sample.

    Args:
        samples: Iterable of ``(OmniSampleArgs, data_batch)`` pairs.
        model: The model, used for token counting and seed expansion.
        max_num_seqs: Maximum number of sequences per batch.
        max_model_len: Maximum total tokens per batch.
            Exactly one of ``max_num_seqs`` or ``max_model_len`` must be set.

    Yields:
        ``(sample_args_list, merged_data_batch, per_sample_data_batches)`` tuples.
        ``per_sample_data_batches`` is the list of individual data dicts before
        merging, useful when callers need per-sample post-processing.
    """
    assert max_model_len is not None or max_num_seqs is not None, "Either max_model_len or max_num_seqs must be set"
    assert max_model_len is None or max_num_seqs is None, "Either max_model_len or max_num_seqs must be set, not both"

    # Tensor keys whose non-batch dims may differ across samples and must be
    # promoted to a length-1 ``list[Tensor]`` so ``_merge_data_batches`` can
    # flatten them via list-extension instead of failing in ``torch.cat``.
    _VARIABLE_SHAPE_TENSOR_KEYS = {"video", "action"}

    def _prepare_for_merge(db: dict[str, Any]) -> dict[str, Any]:
        """Reshape a per-sample data dict so ``_merge_data_batches`` can combine it.

        Returns a shallow copy of ``db`` with two key-specific rewrites; all
        other values are passed through by reference.

        - **0-dim tensors** are promoted to 1-D via ``unsqueeze(0)``. Without
          this, ``_merge_data_batches`` would route them to ``torch.stack``
          (adding a new batch dim), inconsistent with the ``torch.cat`` path
          taken by every other tensor key.
        - **``"video"`` / ``"action"``** are converted from
          ``[1, *variable_dims]`` tensors (the shape produced by
          ``_collate_sample``'s ``unsqueeze(0)``) into length-1 ``list[Tensor]``
          of shape ``[*variable_dims]``. ``_merge_data_batches`` then flattens
          these per-sample lists across the batch instead of calling
          ``torch.cat`` on the tensors, which would fail when samples in the
          same chunk have different shapes (e.g. videos with different aspect
          ratios like 544x736 vs 640x640, or actions with different sequence
          lengths like ``[148, D]`` vs ``[104, D]`` from variable-length clips
          in the camera-480 dataset). Both keys are eventually consumed as
          per-sample lists by ``pack_action`` / video tokenization, so the
          list form matches the downstream contract.

        Args:
            db: Single sample's data dict, typically straight out of
                ``_collate_sample``.

        Returns:
            A new dict with the rewrites applied; the original ``db`` is not
            mutated.
        """
        updated_db: dict[str, Any] = {}
        for key, value in db.items():
            if isinstance(value, torch.Tensor) and value.ndim == 0:
                updated_db[key] = value.unsqueeze(0)
            elif key in _VARIABLE_SHAPE_TENSOR_KEYS and isinstance(value, torch.Tensor):
                updated_db[key] = [value.squeeze(0)]
            else:
                updated_db[key] = value
        return updated_db

    # Materialize all samples and apply seed expansion.
    all_args: list[OmniSampleArgs] = []
    all_data: list[dict[str, Any]] = []
    for sa, db in samples:
        db = _prepare_for_merge(db)
        if sa.num_outputs > 1:
            expanded = _finalize_sample_args_list([sa], model)
        else:
            expanded = [sa]
        for exp_sa in expanded:
            all_args.append(exp_sa)
            all_data.append({k: list(v) if isinstance(v, list) else v for k, v in db.items()})

    if not all_args:
        return

    # Reuse _iter_batch_ranges for smart batching.
    indices = list(range(len(all_args)))
    for batch_start, batch_end in _iter_batch_ranges(indices, all_args, model, max_model_len, max_num_seqs):
        chunk_args = [all_args[indices[pos]] for pos in range(batch_start, batch_end)]
        chunk_data = [all_data[indices[pos]] for pos in range(batch_start, batch_end)]
        yield chunk_args, _merge_data_batches(chunk_data), chunk_data


def _finalize_data_batch(data_batch: dict[str, Any], batch_size: int, model: OmniMoTModel):
    """Finalize and validate data batch in-place."""
    for old_key, new_key in [
        ("video", model.input_video_key),
        ("images", model.input_image_key),
        ("ai_caption", model.input_caption_key),
    ]:
        if old_key in data_batch and new_key != old_key:
            if new_key in data_batch:
                raise ValueError(f"Conflicting keys: '{old_key}' and '{new_key}'")
            data_batch[new_key] = data_batch.pop(old_key)

    # Unstack variable length tensors
    _multi_item_keys = {
        "text_token_ids",
        "action",
        model.input_video_key,
        model.input_image_key,
    }
    for key in _multi_item_keys:
        if key in data_batch and isinstance(data_batch[key], torch.Tensor):
            if key == model.input_image_key:
                data_batch[key] = [
                    t.unsqueeze(0).squeeze(2) for t in torch.unbind(data_batch[key])
                ]  # list of [1,C,H,W]
            elif key == model.input_video_key:
                if data_batch.get("is_preprocessed", False):
                    data_batch[key] = [t.unsqueeze(0) for t in torch.unbind(data_batch[key])]  # list of [1,C,T,H,W]
                else:
                    data_batch[key] = list(torch.unbind(data_batch[key]))
            else:
                data_batch[key] = [[t] for t in torch.unbind(data_batch[key])]

    # Validate
    if len(data_batch[model.input_caption_key]) != batch_size:
        raise ValueError(
            f"Data batch length ({len(data_batch[model.input_caption_key])}) does not match batch size ({batch_size})"
        )


class SampleDataset(Dataset):
    """PyTorch map-style dataset over inference sample args.

    Each item is a ``(SampleArgs, data_dict)`` tuple where the data dict is
    lazily prepared on access via ``__getitem__``.
    """

    def __init__(self, sample_args_list: Sequence[SampleArgs], model: OmniMoTModel) -> None:
        self._sample_args_list = list(sample_args_list)
        self._model = model

    def __len__(self) -> int:
        return len(self._sample_args_list)

    def __getitem__(self, idx: int) -> tuple[SampleArgs, dict[str, Any]]:
        sample_args = self._sample_args_list[idx]
        assert isinstance(sample_args, OmniSampleArgs)
        assert sample_args.output_dir is not None
        data_batch = sample_args.get_data(device="cuda")
        if not data_batch:
            data_batch = get_sample_data(sample_args=sample_args, model=self._model)
        return sample_args, data_batch


@dataclass
class OmniInference(Inference):
    # pyrefly: ignore[bad-override]
    model: OmniMoTModel
    vae_decode_stream: torch.cuda.Stream | None = None

    @property
    def model_config(self) -> "OmniMoTModelConfig":
        return self.model.config

    @classmethod
    def _get_parallelism_config(cls, setup_args: ParallelismArgs) -> ParallelismConfig:
        return ParallelismConfig(
            enable_inference_mode=True,
            data_parallel_shard_degree=setup_args.dp_shard_size,
            context_parallel_shard_degree=setup_args.cp_size,
            cfg_parallel_shard_degree=setup_args.cfgp_size,
            use_torch_compile=setup_args.use_torch_compile,
            use_cuda_graphs=setup_args.use_cuda_graphs
            and setup_args.dp_shard_size * setup_args.cp_size * setup_args.cfgp_size == 1,
            compiled_region=setup_args.compiled_region,
            compile_dynamic=setup_args.compile_dynamic,
            use_activation_checkpointing=False,
        )

    @override
    @classmethod
    def _create(cls, setup_args: SetupArgs, **kwargs: Any) -> Self:
        assert isinstance(setup_args, OmniSetupArgs)
        assert setup_args.output_dir is not None

        sampler_override = setup_args.sampler
        parallelism_config = cls._get_parallelism_config(setup_args)
        if setup_args.checkpoint_type == CheckpointType.DCP and setup_args.config_file_type == ConfigFileType.MODULE:
            from cosmos3.common.config import save_config
            from cosmos3._src.vfm.utils.model_loader import load_model_from_checkpoint

            if not setup_args.experiment:
                raise ValueError("'experiment' is required")
            if not setup_args.config_file:
                raise ValueError("'config_file' is required")

            Cosmos3OmniModel.before_load_model()
            model, config = load_model_from_checkpoint(
                experiment_name=setup_args.experiment,
                config_file=setup_args.config_file,
                checkpoint_path=setup_args.checkpoint_path,
                credential_path=setup_args.credential_path or None,
                parallelism_config=attrs.asdict(parallelism_config),
                load_ema_to_reg=setup_args.use_ema_weights,
                experiment_opts=[
                    *setup_args.experiment_overrides,
                    f"model.config.rectified_flow_inference_config.scheduler_type={sampler_override}",
                ],
                use_cache_checkpoint=setup_args.checkpoint_cache_dir is not None,
                cache_checkpoint_rootdir=str(setup_args.checkpoint_cache_dir or ""),
            )
            model = cast("OmniMoTModel", model)
            Cosmos3OmniModel.after_load_model(model)
            save_config(config, setup_args.output_dir)
        else:
            checkpoint_path = setup_args.download_checkpoint()
            if setup_args.config_file_type == ConfigFileType.MODULE:
                config = None
            else:
                model_dict = setup_args.load_model_config_dict()
                config = Cosmos3OmniConfig(model=model_dict)
            model = Cosmos3OmniModel.from_pretrained_dcp(
                checkpoint_path, config=config, parallelism_config=parallelism_config
            ).model
            if model.config.rectified_flow_inference_config.scheduler_type != sampler_override:
                model.config.rectified_flow_inference_config.scheduler_type = sampler_override
                model.set_up_scheduler_and_sampler()
                log.debug(f"Sampler overridden to: {sampler_override}")

        vae_decode_stream: torch.cuda.Stream | None = None
        if setup_args.use_separate_pipeline_vision_decode_gpu:
            # The CP/CFGP ranks are partitioned into replica-local groups of size
            # cp_size * cfgp_size. Only the first rank in each group owns separate-VAE
            # decode work. For example, with cp_size=2 and cfgp_size=1, ranks [0,1]
            # form one replica and only rank 0 returns True here.
            replica_size = setup_args.cp_size * setup_args.cfgp_size

            is_vae_output_rank = (replica_size <= 1) or (get_rank() % replica_size == 0)

            vae_device_index = setup_args.cp_size * setup_args.cfgp_size
            if torch.cuda.device_count() <= vae_device_index:
                raise RuntimeError(
                    "--use-separate-pipeline-vision-decode-gpu requires a spare visible local GPU on the "
                    "same node as the decode-owning rank, but the configured local decode GPU index "
                    f"{vae_device_index} is unavailable with only {torch.cuda.device_count()} visible local GPUs."
                )
            if is_vae_output_rank:
                vae_device = torch.device("cuda", vae_device_index)
                inference_device = torch.device("cuda", torch.cuda.current_device())
                vae_decode_stream = torch.cuda.Stream(device=vae_device)
                vae = model.tokenizer_vision_gen.model
                vae.device = str(vae_device)
                vae.model = vae.model.to(device=vae_device)
                vae.scale = tree_map_only(torch.Tensor, lambda tensor: tensor.to(device=vae_device), vae.scale)

                original_encode = model.encode
                original_decode = model.decode

                def encode_on_vae(state: torch.Tensor) -> torch.Tensor:
                    return original_encode(state.to(device=vae_device, non_blocking=True)).to(
                        device=inference_device, non_blocking=True
                    )

                def decode_on_vae(latent: torch.Tensor) -> torch.Tensor:
                    return original_decode(latent.to(device=vae_device, non_blocking=True))

                model.encode = encode_on_vae
                model.decode = decode_on_vae
                log.info(
                    f"Configured vision VAE on device '{vae_device}' while inference remains on '{inference_device}'",
                    rank0_only=False,
                )

        return cls(setup_args=setup_args, model=model, vae_decode_stream=vae_decode_stream, **kwargs)

    @classmethod
    def save_data(
        cls,
        data: dict[str, Any],
        *,
        output_dir: Path,
        output_name: str,
        truncate_action_dim: bool = True,
    ) -> list[Path]:
        """Save data to disk in multiple formats.

        Tensors are saved as ``<output_name>.safetensors``, non-tensor values as
        ``<output_name>.pickle``. If ``truncate_action_dim`` is True and both ``action``
        and ``raw_action_dim`` are present in ``data``, the action tensor's last dimension
        is truncated to ``raw_action_dim`` before saving.

        Returns a list of paths to all files written.
        """
        files: list[Path] = []
        data_tensors: dict[str, torch.Tensor] = {}
        data_pickle: dict[str, Any] = {}
        for k, v in data.items():
            if isinstance(v, list) and len(v) > 0 and isinstance(v[0], torch.Tensor):
                for i, x in enumerate(v):
                    data_tensors[f"{k}[{i}]"] = x
            elif isinstance(v, torch.Tensor):
                data_tensors[k] = v
            else:
                data_pickle[k] = v

        # Truncate `action` tensor's last dimension to `raw_action_dim` if available;
        # otherwise use the full action tensor as-is.
        if truncate_action_dim and "action" in data_tensors and "raw_action_dim" in data_tensors:
            raw_action_dim = data_tensors["raw_action_dim"][0]
            action = data_tensors["action"][..., :raw_action_dim]
            data_tensors["action"] = action
            log.debug(f"Truncated 'action' tensor to shape={action.shape}")

        if data_tensors:
            tensors_file = output_dir / f"{output_name}.safetensors"
            safetensors.torch.save_file(
                {k: v.detach().cpu().contiguous() for k, v in data_tensors.items()}, tensors_file
            )
            files.append(tensors_file)

        if data_pickle:
            pickle_file = output_dir / f"{output_name}.pickle"
            with pickle_file.open("wb") as f:
                pickle.dump(data_pickle, f)
            files.append(pickle_file)

        return files

    @override
    def create_batches(
        self, sample_args_list: Sequence[SampleArgs]
    ) -> Generator[tuple[list[SampleArgs], dict[str, Any]]]:
        assert isinstance(self.setup_args, OmniSetupArgs)
        max_model_len = self.setup_args.max_model_len
        max_num_seqs = self.setup_args.max_num_seqs
        assert max_model_len is not None or max_num_seqs is not None, "Either max_model_len or max_num_seqs must be set"
        assert max_model_len is None or max_num_seqs is None, (
            "Either max_model_len or max_num_seqs must be set, not both"
        )

        sample_args_list = _finalize_sample_args_list(cast(Sequence[OmniSampleArgs], sample_args_list), self.model)
        dataset = SampleDataset(sample_args_list, self.model)

        # Mod-shard the dataset indices across replicas.
        sampler_indices = list(range(self.replica_id, len(dataset), self.num_replicas))

        # --- Phase 1: pre-compute batch boundaries (cheap, no data prep) ---
        batch_ranges = list(
            _iter_batch_ranges(sampler_indices, sample_args_list, self.model, max_model_len, max_num_seqs)
        )
        num_local_batches = len(batch_ranges)

        log.debug(f"Number of local batches: {num_local_batches}", rank0_only=False)

        # --- Phase 2: synchronize batch count across replicas ---
        # All ranks within a replica share the same replica_id and therefore
        # the same local batch count, so a global MAX all-reduce is sufficient
        # to align all replicas.
        if torch.distributed.is_initialized() and self.num_replicas > 1:
            count_tensor = torch.tensor([num_local_batches], dtype=torch.long, device="cuda")
            torch.distributed.all_reduce(count_tensor, op=torch.distributed.ReduceOp.MAX)
            global_max_batches = int(count_tensor.item())
        else:
            global_max_batches = num_local_batches

        log.debug(f"Number of global batches: {global_max_batches}")
        log.debug(f"Number of padding batches: {global_max_batches - num_local_batches}", rank0_only=False)

        # --- Phase 3: yield real batches (lazily prepare data) ---
        batches_yielded = 0

        for batch_start, batch_end in batch_ranges:
            chunk_args: list[SampleArgs] = []
            chunk_data: list[dict[str, Any]] = []

            for pos in range(batch_start, batch_end):
                sample_idx = sampler_indices[pos]
                sample_args, data_batch = dataset[sample_idx]

                if self.setup_args.debug and self.should_process_sample(sample_args):
                    assert sample_args.output_dir is not None
                    sample_args.output_dir.mkdir(parents=True, exist_ok=True)
                    self.save_data(
                        data_batch,
                        output_dir=sample_args.output_dir,
                        output_name="sample_data",
                    )

                chunk_args.append(sample_args)
                chunk_data.append(data_batch)

            yield chunk_args, _merge_data_batches(chunk_data)
            batches_yielded += 1

        assert batches_yielded == num_local_batches

        # --- Phase 4: pad with dummy batches so every replica calls
        #     generate_batch the same number of times (prevents collective
        #     deadlocks in context-parallel / CFG-parallel communication). ---
        dummy_sa = sample_args_list[0].model_copy(update={"output_dir": None, "name": "padding"})
        dummy_data = dataset[0][1]
        while batches_yielded < global_max_batches:
            yield [dummy_sa], dummy_data
            batches_yielded += 1
        assert batches_yielded == global_max_batches

    @torch.no_grad()
    @override
    def generate_batch(
        self, sample_args_list: Sequence[SampleArgs], data_batch: dict[str, Any], *, warmup: bool = False
    ) -> list[SampleOutputs]:
        assert all(isinstance(sa, OmniSampleArgs) for sa in sample_args_list)


        # Process inputs
        try:
            with sync_distributed_errors():
                for sample_args in sample_args_list:
                    if self.should_process_sample(sample_args) and not warmup:
                        log.debug(f"{sample_args.__class__.__name__}({sample_args})")
                        assert sample_args.output_dir is not None
                        sample_args.output_dir.mkdir(parents=True, exist_ok=True)
                        sample_args_file = sample_args.output_dir / "sample_args.json"
                        sample_args_file.write_text(sample_args.model_dump_json())
                        log.info(f"Saved sample args to '{sample_args_file}'", rank0_only=False)

                assert all(sa.num_outputs == 1 for sa in sample_args_list), "num_outputs must be 1"
                _finalize_data_batch(data_batch=data_batch, batch_size=len(sample_args_list), model=self.model)
        except Exception as e:
            return [
                self._handle_sample_exception(args, e)
                for args in sample_args_list
                if self.should_process_sample(args) and not warmup
            ]

        # Generate samples
        #
        # Can't catch exceptions here. This code contains collective operations
        # that will hang if any rank fails. If a rank fails, we must restart
        # the entire distributed environment.
        #
        # Use the first sample's sampling parameters for the whole batch.
        # All samples in a batch share guidance, num_steps, shift, etc.
        def _getattr(sample_args_list: Sequence[OmniSampleArgs], attr: str) -> Any:
            attr_values = [getattr(sa, attr) for sa in sample_args_list]
            if all(v == attr_values[0] for v in attr_values):
                return attr_values[0]
            else:
                raise ValueError(f"Attribute '{attr}' is not the same for all samples: {attr_values}")

        is_distilled = self.model.config.fixed_step_sampler_config is not None
        if is_distilled:
            sampler = self.model.fixed_step_sampler
            guidance = 1.0
        else:
            sampler = None
            guidance = _getattr(sample_args_list, "guidance")

        should_decode_outputs = self.should_process_sample(sample_args_list[0])

        def decode_vision(vision_latent: torch.Tensor) -> torch.Tensor:
            """
            Handles decoding of vision latents, either on the inference device or on a separate VAE device if configured.
            """
            if not should_decode_outputs:
                tokenizer_vision_gen = self.model.tokenizer_vision_gen
                return vision_latent.new_zeros(
                    (
                        vision_latent.shape[0],
                        3,
                        tokenizer_vision_gen.get_pixel_num_frames(int(vision_latent.shape[2])),
                        int(vision_latent.shape[3]) * tokenizer_vision_gen.spatial_compression_factor,
                        int(vision_latent.shape[4]) * tokenizer_vision_gen.spatial_compression_factor,
                    )
                )
            if self.vae_decode_stream is None:
                # We are not using a separate GPU for VAE decoding, so decode directly on the inference device
                vision = self.model.decode(vision_latent)  # [B,C,T,H,W]
                return ((1.0 + vision) / 2).clamp(0, 1)  # [B,C,T,H,W]
            # We are using a separate GPU for VAE decoding, so we need to issue decode on the VAE device
            vision_ready = torch.cuda.Event()
            torch.cuda.current_stream(device=vision_latent.device).record_event(vision_ready)
            self.vae_decode_stream.wait_event(vision_ready)
            with torch.cuda.stream(self.vae_decode_stream):
                vision = self.model.decode(vision_latent)  # [B,C,T,H,W]
                return ((1.0 + vision) / 2).clamp(0, 1)  # [B,C,T,H,W]

        seed = [sa.seed if sa.seed is not None else random.randint(0, 10000) for sa in sample_args_list]
        outputs: dict[str, Any] | None = None

        if outputs is None:
            assert all(sa.num_outputs == 1 for sa in sample_args_list), "num_outputs must be 1"
            n_sample = sum(cast(OmniSampleArgs, sa).num_outputs for sa in sample_args_list)
            neg_key = "neg_" + self.model.input_caption_key

            with self._get_timer(f"{self.model.__class__.__name__}.generate_samples_from_batch"):
                outputs = self.model.generate_samples_from_batch(
                    data_batch,
                    sampler=sampler,
                    guidance=guidance,
                    guidance_interval=_getattr(sample_args_list, "guidance_interval"),
                    seed=seed,
                    num_steps=_getattr(sample_args_list, "num_steps"),
                    shift=_getattr(sample_args_list, "shift"),
                    sigma_max=_getattr(sample_args_list, "sigma_max"),
                    has_negative_prompt=neg_key in data_batch,
                    n_sample=n_sample,
                    normalize_cfg=_getattr(sample_args_list, "normalize_cfg"),
                )

            with self._get_timer(f"{self.model.__class__.__name__}.decode"):
                output_vision = outputs.pop("vision")
                decoded_vision = [decode_vision(vision) for vision in output_vision]
                outputs["vision"] = [cast(torch.Tensor, vision) for vision in decoded_vision]
                if self.vae_decode_stream is not None:
                    # If we are using a separate GPU for VAE decoding, wait for results to be ready
                    torch.cuda.current_stream(device=outputs["vision"][0].device).wait_stream(self.vae_decode_stream)
        for k, v in outputs.items():
            if len(v) != len(sample_args_list):
                raise ValueError(f"Output key '{k}' has length {len(v)} but expected {len(sample_args_list)}")

        if "sound" in outputs:
            with self._get_timer(f"{self.model.__class__.__name__}.decode_sound"):
                outputs["sound"] = [self.model.decode_sound(sound) for sound in outputs.pop("sound")]

        if warmup:
            return []

        # Save outputs
        sample_outputs: list[SampleOutputs] = []
        try:
            with sync_distributed_errors():
                for sample_idx, sample_args in enumerate(sample_args_list):
                    if self.should_process_sample(sample_args):
                        assert isinstance(sample_args, OmniSampleArgs)
                        assert sample_args.output_dir is not None
                        assert sample_args.num_outputs == 1
                        output = {k: v[sample_idx].squeeze(0) for k, v in outputs.items()}
                        vision_cthw = output.pop("vision")

                        # Run guardrails
                        self._run_text_guardrail(
                            str(sample_args.output_dir), data_batch[self.model.input_caption_key][sample_idx]
                        )
                        vision_cthw = self._run_video_guardrail(str(sample_args.output_dir), vision_cthw)
                        output["vision"] = vision_cthw

                        content: dict[str, Any] = {}
                        files: list[Path] = []

                        # Save debug
                        if self.setup_args.debug:
                            files.extend(
                                self.save_data(output, output_dir=sample_args.output_dir, output_name="output")
                            )

                        # Save vision
                        if vision_cthw.shape[1] == 1:
                            quality = sample_args.image_save_quality
                        else:
                            quality = sample_args.video_save_quality
                        vision_file = sample_args.output_dir / f"vision{sample_args.vision_extension}"
                        output_fps = sample_args.fps
                        save_img_or_video(
                            vision_cthw, str(vision_file.with_suffix("")), fps=output_fps, quality=quality
                        )
                        assert vision_file.is_file(), vision_file
                        files.append(vision_file)

                        if "sound" in output:
                            from cosmos3.sound import (
                                get_audio_tokenizer_info,
                                save_sound,
                            )

                            audio_info = get_audio_tokenizer_info(self.model)
                            decoded_audio = output["sound"]
                            sound_file = sample_args.output_dir / "sound.wav"
                            save_sound(decoded_audio, sound_file, audio_info.sample_rate)
                            files.append(sound_file)

                        if "action" in output:
                            pred_action = output["action"]
                            if sample_args.raw_action_dim is not None:
                                raw_action_dim = int(sample_args.raw_action_dim)

                                assert pred_action.shape[-1] >= raw_action_dim, (
                                    f"invalid raw_action_dim={raw_action_dim} for action with shape {pred_action.shape}"
                                )
                                pred_action = pred_action[..., :raw_action_dim]
                            content["action"] = pred_action.detach().cpu().tolist()

                        sample_output = SampleOutputs(
                            args=sample_args.model_dump(mode="json"),
                            outputs=[SampleOutput(content=content, files=files)],
                        )
                        sample_outputs_file = sample_args.output_dir / "sample_outputs.json"
                        sample_outputs_file.write_text(sample_output.model_dump_json())
                        log.success(f"Saved sample outputs to '{sample_outputs_file}'", rank0_only=False)

                        sample_outputs.append(sample_output)

        except Exception as e:
            return [
                self._handle_sample_exception(sample_args, e)
                for sample_args in sample_args_list
                if self.should_process_sample(sample_args)
            ]

        return sample_outputs


    @property
    def replica_size(self) -> int:
        """
        The ranks are divided into computation replicas. The replica size is
        the product of the context parallelism and CFG parallelism sizes.
        """
        if not hasattr(self.model, "parallel_dims") or self.model.parallel_dims is None:
            return 1
        else:
            return self.model.parallel_dims.cp_size * self.model.parallel_dims.cfgp_size

    @property
    def num_replicas(self) -> int:
        assert get_world_size() % self.replica_size == 0
        return get_world_size() // self.replica_size

    @property
    def replica_id(self) -> int:
        return get_rank() // self.replica_size

    @property
    def index_in_replica(self) -> int:
        return get_rank() % self.replica_size

    def should_process_sample(self, sample_args: SampleArgs) -> bool:
        """Whether the sample should be processed by the current rank."""
        return sample_args.output_dir is not None and self.index_in_replica == 0


_data_converter = cattrs.preconf.json.make_converter()


# torch.Tensor
@_data_converter.register_unstructure_hook
def _unstructure_torch_tensor(obj: torch.Tensor) -> Any:
    return {
        "shape": obj.shape,
        "dtype": str(obj.dtype),
        "device": str(obj.device),
        "values": obj.detach().flatten()[:5].cpu().tolist(),
    }
