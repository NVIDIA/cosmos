from __future__ import annotations

import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from diffusers import Cosmos3OmniBlocks, Cosmos3OmniModularPipeline
from diffusers.models.transformers.transformer_cosmos3 import Cosmos3OmniTransformer
from diffusers.modular_pipelines import BlockState, ComponentSpec, InputParam, ModularPipelineBlocks, OutputParam

from .config import StreamingConfig
from .fp8 import DEFAULT_FP8_MODEL_ID, load_fp8_transformer, materialize_fp8_checkpoint
from .offloading import offload_groups
from .runtime import StreamingCosmosRuntime


class StreamingCosmos3LoopDenoiser(ModularPipelineBlocks):
    """External streamed replacement for Diffusers' multimodal denoiser."""

    model_name = "cosmos3-omni"

    def __init__(self, runtime: StreamingCosmosRuntime) -> None:
        super().__init__()
        self.runtime = runtime

    @property
    def description(self) -> str:
        return "Predicts Cosmos 3 modality velocities with cached understanding and streamed generation blocks."

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [ComponentSpec("transformer", Cosmos3OmniTransformer)]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("denoiser_input_fields"),
            InputParam(
                name="guidance_scale",
                type_hint=float,
                default=6.0,
                description="Scale for classifier-free guidance.",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "velocity_vision",
                type_hint=torch.Tensor,
                description="Predicted velocity for vision latents.",
            ),
            OutputParam("velocity_sound", type_hint=torch.Tensor, description="Predicted velocity for sound latents."),
            OutputParam("velocity_action", type_hint=torch.Tensor, description="Predicted velocity for action latents."),
        ]

    @torch.no_grad()
    def __call__(
        self,
        components: Cosmos3OmniModularPipeline,
        block_state: BlockState,
        i: int,
        t: torch.Tensor,
    ) -> tuple[Cosmos3OmniModularPipeline, BlockState]:
        del t  # The per-step timestep tensor is already present in block_state.
        self.runtime.trace_checkpoint("denoise_step.start", step=i)

        denoiser_input_fields = block_state.denoiser_input_fields
        loop_input_fields = block_state.as_dict()
        has_sound = "sound_tokens" in loop_input_fields
        has_action = "action_tokens" in loop_input_fields

        transformer_args = set(inspect.signature(components.transformer.forward).parameters)
        do_cfg = block_state.guidance_scale != 1.0
        pass_names = ["cond", "uncond"] if do_cfg else ["cond"]
        branches = {
            pass_name: self._collect_transformer_kwargs(
                pass_name,
                denoiser_input_fields,
                loop_input_fields,
                transformer_args,
            )
            for pass_name in pass_names
        }
        predictions = self.runtime.predict_modalities(components.transformer, branches)

        velocities: dict[str, tuple[torch.Tensor, torch.Tensor | None, None]] = {}
        for pass_name, prediction in predictions.items():
            velocities[pass_name] = components._mask_velocity_predictions(
                prediction.vision,
                prediction.sound,
                vision_condition_mask=[loop_input_fields["vision_condition_mask"]],
                sound_condition_mask=[loop_input_fields["sound_condition_mask"]] if has_sound else None,
                preds_action=prediction.action,
                action_condition_mask=[loop_input_fields["action_condition_mask"]] if has_action else None,
                raw_action_dim=loop_input_fields.get("raw_action_dim_resolved"),
            )

        cond_velocity, cond_velocity_sound, cond_velocity_action = velocities["cond"]
        if do_cfg:
            uncond_velocity, uncond_velocity_sound, uncond_velocity_action = velocities["uncond"]
            block_state.velocity_vision = uncond_velocity + block_state.guidance_scale * (
                cond_velocity - uncond_velocity
            )
            block_state.velocity_sound = (
                uncond_velocity_sound + block_state.guidance_scale * (cond_velocity_sound - uncond_velocity_sound)
                if has_sound
                else None
            )
            block_state.velocity_action = (
                uncond_velocity_action
                + block_state.guidance_scale * (cond_velocity_action - uncond_velocity_action)
                if has_action
                else None
            )
        else:
            block_state.velocity_vision = cond_velocity
            block_state.velocity_sound = cond_velocity_sound if has_sound else None
            block_state.velocity_action = cond_velocity_action if has_action else None
        self.runtime.trace_checkpoint("denoise_step.end", step=i)
        return components, block_state

    @staticmethod
    def _collect_transformer_kwargs(
        pass_name: str,
        denoiser_input_fields: dict[str, Any],
        loop_input_fields: dict[str, Any],
        transformer_args: set[str],
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        prefix = f"{pass_name}_"
        for field_name, field_value in denoiser_input_fields.items():
            if field_name.startswith(prefix):
                transformer_field_name = field_name.removeprefix(prefix)
                if transformer_field_name.endswith("_segment"):
                    kwargs.update(field_value)
                else:
                    kwargs[transformer_field_name] = field_value
            elif field_name in transformer_args:
                kwargs[field_name] = field_value

        kwargs.update(
            {
                field_name: field_value
                for field_name, field_value in loop_input_fields.items()
                if field_name in transformer_args
            }
        )
        kwargs = {name: value for name, value in kwargs.items() if name in transformer_args}

        required = {
            "input_ids",
            "text_indexes",
            "position_ids",
            "und_len",
            "sequence_length",
            "vision_tokens",
            "vision_token_shapes",
            "vision_sequence_indexes",
            "vision_mse_loss_indexes",
            "vision_timesteps",
            "vision_noisy_frame_indexes",
        }
        missing = sorted(required - kwargs.keys())
        if missing:
            raise KeyError(f"missing Cosmos transformer inputs for {pass_name}: {missing}")
        return kwargs


class StreamingCosmos3TransferLoopDenoiser(ModularPipelineBlocks):
    """Streamed replacement for the three-axis transfer denoiser."""

    model_name = "cosmos3-omni"

    def __init__(self, runtime: StreamingCosmosRuntime) -> None:
        super().__init__()
        self.runtime = runtime

    @property
    def description(self) -> str:
        return "Predicts transfer velocity with streamed control and text guidance branches."

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [ComponentSpec("transformer", Cosmos3OmniTransformer)]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("denoiser_input_fields"),
            InputParam(name="vision_tokens_full", type_hint=list[torch.Tensor], required=True),
            InputParam(name="vision_tokens_target", type_hint=list[torch.Tensor], required=True),
            InputParam(name="vision_timesteps", type_hint=torch.Tensor, required=True),
            InputParam(name="velocity_mask", type_hint=torch.Tensor, required=True),
            InputParam(name="guidance_scale", type_hint=float, default=6.0),
            InputParam(name="control_guidance", type_hint=float, default=1.0),
            InputParam(name="guidance_interval", type_hint=tuple, default=None),
            InputParam(name="control_guidance_interval", type_hint=tuple, default=None),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [OutputParam("velocity", type_hint=torch.Tensor, description="Predicted transfer velocity.")]

    @staticmethod
    def _transformer_kwargs(
        static: dict[str, Any],
        vision_tokens: list[torch.Tensor],
        vision_timesteps: torch.Tensor,
    ) -> dict[str, Any]:
        return {
            "input_ids": static["input_ids"],
            "text_indexes": static["text_indexes"],
            "position_ids": static["position_ids"],
            "und_len": static["und_len"],
            "sequence_length": static["sequence_length"],
            "vision_tokens": vision_tokens,
            "vision_token_shapes": static["vision_token_shapes"],
            "vision_sequence_indexes": static["vision_sequence_indexes"],
            "vision_mse_loss_indexes": static["vision_mse_loss_indexes"],
            "vision_timesteps": vision_timesteps,
            "vision_noisy_frame_indexes": static["vision_noisy_frame_indexes"],
        }

    @staticmethod
    def _active(interval: tuple | None, timestep: torch.Tensor) -> bool:
        return interval is None or float(interval[0]) <= float(timestep.item()) <= float(interval[1])

    @torch.no_grad()
    def __call__(
        self,
        components: Cosmos3OmniModularPipeline,
        block_state: BlockState,
        i: int,
        t: torch.Tensor,
    ) -> tuple[Cosmos3OmniModularPipeline, BlockState]:
        self.runtime.trace_checkpoint("transfer_denoise_step.start", step=i)
        step_guidance = (
            block_state.guidance_scale
            if self._active(block_state.guidance_interval, t)
            else 1.0
        )
        step_control = (
            block_state.control_guidance
            if self._active(block_state.control_guidance_interval, t)
            else 1.0
        )
        needs_text_cfg = step_guidance > 1.0
        needs_control_cfg = step_control != 1.0

        static = block_state.denoiser_input_fields
        branches = {
            "cond_full": self._transformer_kwargs(
                static["cond_full_static"],
                block_state.vision_tokens_full,
                block_state.vision_timesteps,
            )
        }
        if needs_control_cfg:
            branches["cond_no_control"] = self._transformer_kwargs(
                static["cond_no_control_static"],
                block_state.vision_tokens_target,
                block_state.vision_timesteps,
            )
        if needs_text_cfg:
            branches["uncond_full"] = self._transformer_kwargs(
                static["uncond_full_static"],
                block_state.vision_tokens_full,
                block_state.vision_timesteps,
            )
        predictions = self.runtime.predict_modalities(components.transformer, branches)
        cond_full = predictions["cond_full"].vision[-1]

        if needs_control_cfg:
            cond_no_control = predictions["cond_no_control"].vision[-1]
        if needs_text_cfg:
            uncond_full = predictions["uncond_full"].vision[-1]

        if needs_control_cfg and needs_text_cfg:
            control_cond = cond_no_control + step_control * (cond_full - cond_no_control)
            velocity = uncond_full + step_guidance * (control_cond - uncond_full)
        elif needs_control_cfg:
            velocity = cond_no_control + step_control * (cond_full - cond_no_control)
        elif needs_text_cfg:
            velocity = uncond_full + step_guidance * (cond_full - uncond_full)
        else:
            velocity = cond_full

        block_state.velocity = velocity * block_state.velocity_mask
        self.runtime.trace_checkpoint("transfer_denoise_step.end", step=i)
        return components, block_state


class StreamingCosmos3Pipeline(Cosmos3OmniModularPipeline):
    """Thin device/lifecycle adapter over the official modular pipeline."""

    def __init__(
        self,
        *args: Any,
        streaming_runtime: StreamingCosmosRuntime,
        execution_device: str | torch.device = "cuda",
        **kwargs: Any,
    ) -> None:
        self._streaming_runtime = streaming_runtime
        self._streaming_execution_device = torch.device(execution_device)
        super().__init__(*args, **kwargs)

    @property
    def _execution_device(self) -> torch.device:
        return self._streaming_execution_device

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self._streaming_runtime.begin_request()
        failed = True
        try:
            result = super().__call__(*args, **kwargs)
            failed = False
            return result
        finally:
            try:
                if failed:
                    # An encoder/decoder can fail outside the streamed runtime.
                    # VAE leaf hooks have no prefetch trace and remain reusable.
                    if self.vae is not None:
                        offload_groups(self.vae)
                    if self.sound_tokenizer is not None:
                        self.sound_tokenizer.to(self._streaming_runtime.offload_device)
            finally:
                self._streaming_runtime.end_request(failed=failed)

    def _apply_video_safety_check(self, *args: Any, **kwargs: Any) -> Any:
        if self._streaming_runtime.release_cache_before_safety:
            self._streaming_runtime.trace_checkpoint("pre_safety_cache_clear.start")
            torch.cuda.empty_cache()
            self._streaming_runtime.trace_checkpoint("pre_safety_cache_clear.end")
        self._streaming_runtime.trace_checkpoint("safety_check.start")
        try:
            return super()._apply_video_safety_check(*args, **kwargs)
        finally:
            self._streaming_runtime.trace_checkpoint("safety_check.end")


class _MemoryTracedBlock(ModularPipelineBlocks):
    """Transparent telemetry wrapper around an official modular block."""

    def __init__(self, block: ModularPipelineBlocks, runtime: StreamingCosmosRuntime, stage: str) -> None:
        super().__init__()
        self.block = block
        self.runtime = runtime
        self.stage = stage
        self.model_name = block.model_name

    @property
    def description(self) -> str:
        return self.block.description

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return self.block.expected_components

    @property
    def expected_configs(self) -> list[Any]:
        return self.block.expected_configs

    @property
    def inputs(self) -> list[InputParam]:
        return self.block.inputs

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return self.block.intermediate_outputs

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.runtime.trace_checkpoint(f"pipeline.{self.stage}.start")
        result = self.block(*args, **kwargs)
        self.runtime.trace_checkpoint(f"pipeline.{self.stage}.end")
        return result


@dataclass(frozen=True, slots=True)
class _VaeMemoryPolicy:
    tile_size: int | None
    tile_stride: int | None
    release_cache_before_safety: bool


def _resolve_vae_memory_policy(profile: str, total_vram_bytes: int) -> _VaeMemoryPolicy:
    """Resolve internal VAE details from one researcher-facing memory profile."""

    if profile == "auto" and total_vram_bytes < 14 * 2**30:
        return _VaeMemoryPolicy(tile_size=512, tile_stride=384, release_cache_before_safety=True)
    if profile == "low_memory" or (profile == "auto" and total_vram_bytes < 20 * 2**30):
        return _VaeMemoryPolicy(tile_size=768, tile_stride=576, release_cache_before_safety=True)
    return _VaeMemoryPolicy(tile_size=None, tile_stride=None, release_cache_before_safety=True)


def _vision_denoise_loop(blocks: Cosmos3OmniBlocks) -> ModularPipelineBlocks:
    """Return the shared vision-only loop in Diffusers' auto-routing Cosmos graph."""

    try:
        return blocks.sub_blocks["denoise"].sub_blocks["vision"].sub_blocks["denoise"]
    except KeyError as error:
        raise RuntimeError(
            "the pinned Cosmos modular block contract changed: expected "
            "'denoise.vision.denoise' in the auto-routing workflow"
        ) from error


def _vision_sound_denoise_loop(blocks: Cosmos3OmniBlocks) -> ModularPipelineBlocks:
    """Return the official joint vision-and-sound loop in the auto-routing graph."""

    try:
        return blocks.sub_blocks["denoise"].sub_blocks["vision_sound"].sub_blocks["denoise"]
    except KeyError as error:
        raise RuntimeError(
            "the pinned Cosmos modular block contract changed: expected "
            "'denoise.vision_sound.denoise' in the auto-routing workflow"
        ) from error


def _vision_action_denoise_loop(blocks: Cosmos3OmniBlocks) -> ModularPipelineBlocks:
    """Return the official joint vision-and-action loop in the auto-routing graph."""

    try:
        return blocks.sub_blocks["denoise"].sub_blocks["vision_action"].sub_blocks["denoise"]
    except KeyError as error:
        raise RuntimeError(
            "the pinned Cosmos modular block contract changed: expected "
            "'denoise.vision_action.denoise' in the auto-routing workflow"
        ) from error


def _vision_sound_action_denoise_loop(blocks: Cosmos3OmniBlocks) -> ModularPipelineBlocks:
    """Return the joint vision, sound, and action loop in the auto-routing graph."""

    try:
        return blocks.sub_blocks["denoise"].sub_blocks["vision_sound_action"].sub_blocks["denoise"]
    except KeyError as error:
        raise RuntimeError(
            "the pinned Cosmos modular block contract changed: expected "
            "'denoise.vision_sound_action.denoise' in the auto-routing workflow"
        ) from error


def _transfer_denoise_loop(blocks: Cosmos3OmniBlocks) -> ModularPipelineBlocks:
    """Return the per-chunk transfer loop in Diffusers' auto-routing graph."""

    try:
        return (
            blocks.sub_blocks["denoise"]
            .sub_blocks["transfer"]
            .sub_blocks["chunk_denoise"]
            .sub_blocks["denoise"]
        )
    except KeyError as error:
        raise RuntimeError(
            "the pinned Cosmos modular block contract changed: expected "
            "'denoise.transfer.chunk_denoise.denoise' in the auto-routing workflow"
        ) from error


def _fp8_component_load_overrides(
    pipe: Cosmos3OmniModularPipeline,
    checkpoint_dir: str,
) -> tuple[dict[str, str], dict[str, None]]:
    """Use complete local paths for non-transformer FP8 components.

    The official FP8 snapshot also contains a root ``modelopt_state.pth`` for
    its non-Diffusers loading path. ModelOpt 0.44 ignores ``subfolder=`` while
    looking for that file, so passing ``root + subfolder`` would incorrectly
    try to restore the transformer quantizer graph into the VAE and tokenizer.
    """

    root = Path(checkpoint_dir)
    pretrained_paths: dict[str, str] = {}
    subfolders: dict[str, None] = {}
    for name, spec in pipe._component_specs.items():
        if name == "transformer" or spec.default_creation_method != "from_pretrained":
            continue
        if not spec.subfolder:
            continue
        component_dir = root / spec.subfolder
        if component_dir.is_dir():
            pretrained_paths[name] = str(component_dir)
            subfolders[name] = None
    return pretrained_paths, subfolders


def build_pipeline(
    model_id: str | None = None,
    *,
    config: StreamingConfig | None = None,
    dtype: torch.dtype = torch.bfloat16,
    checkpoint_precision: Literal["bf16", "fp8"] = "bf16",
    revision: str | None = None,
    local_files_only: bool = False,
    enable_safety_checker: bool = True,
) -> StreamingCosmos3Pipeline:
    """Build one auto-routing Cosmos pipeline for vision, sound, action, and transfer generation."""

    if checkpoint_precision not in ("bf16", "fp8"):
        raise ValueError(f"unsupported checkpoint precision: {checkpoint_precision}")
    if checkpoint_precision == "fp8" and dtype != torch.bfloat16:
        raise ValueError("the Cosmos 3 FP8 checkpoint requires BF16 compute inputs and outputs")

    if model_id is None:
        model_id = DEFAULT_FP8_MODEL_ID if checkpoint_precision == "fp8" else "nvidia/Cosmos3-Nano"
    component_model_id = model_id
    component_revision = revision
    fp8_transformer: torch.nn.Module | None = None
    if checkpoint_precision == "fp8":
        checkpoint_dir = materialize_fp8_checkpoint(
            model_id,
            revision=revision,
            local_files_only=local_files_only,
        )
        component_model_id = str(checkpoint_dir)
        component_revision = None
        fp8_transformer = load_fp8_transformer(checkpoint_dir, dtype=dtype)

    config = config or StreamingConfig()
    runtime = StreamingCosmosRuntime(config)
    blocks = Cosmos3OmniBlocks()

    denoise_loops = (
        _vision_denoise_loop(blocks),
        _vision_sound_denoise_loop(blocks),
        _vision_action_denoise_loop(blocks),
        _vision_sound_action_denoise_loop(blocks),
    )
    for denoise_loop in denoise_loops:
        if "denoiser" not in denoise_loop.sub_blocks:
            raise RuntimeError("a Cosmos denoise loop no longer exposes a 'denoiser' sub-block")
        denoise_loop.sub_blocks["denoiser"] = StreamingCosmos3LoopDenoiser(runtime)
    transfer_denoise_loop = _transfer_denoise_loop(blocks)
    if "denoiser" not in transfer_denoise_loop.sub_blocks:
        raise RuntimeError("the Cosmos transfer denoise loop no longer exposes a 'denoiser' sub-block")
    transfer_denoise_loop.sub_blocks["denoiser"] = StreamingCosmos3TransferLoopDenoiser(runtime)

    if runtime.memory_trace is not None:
        for denoise_loop in denoise_loops:
            for nested_name in ("prepare_vision", "prepare_sound", "update_vision", "update_sound"):
                if nested_name not in denoise_loop.sub_blocks:
                    continue
                nested = denoise_loop.sub_blocks[nested_name]
                denoise_loop.sub_blocks[nested_name] = _MemoryTracedBlock(
                    nested,
                    runtime,
                    f"denoise.{nested_name}",
                )
        for nested_name in ("prepare_transfer", "update_transfer"):
            nested = transfer_denoise_loop.sub_blocks[nested_name]
            transfer_denoise_loop.sub_blocks[nested_name] = _MemoryTracedBlock(
                nested,
                runtime,
                f"denoise.{nested_name}",
            )
        for block_name in list(blocks.sub_blocks):
            block = blocks.sub_blocks[block_name]
            blocks.sub_blocks[block_name] = _MemoryTracedBlock(block, runtime, block_name)

    pipeline_kwargs: dict[str, Any] = {"local_files_only": local_files_only}
    if component_revision is not None:
        pipeline_kwargs["revision"] = component_revision
    pipe = StreamingCosmos3Pipeline(
        blocks=blocks,
        pretrained_model_name_or_path=component_model_id,
        streaming_runtime=runtime,
        execution_device=config.device,
        **pipeline_kwargs,
    )
    component_load_kwargs: dict[str, Any] = {
        "dtype": dtype,
        "pretrained_model_name_or_path": component_model_id,
        "local_files_only": local_files_only,
    }
    if component_revision is not None:
        component_load_kwargs["revision"] = component_revision
    if fp8_transformer is not None:
        pipe.update_components(transformer=fp8_transformer)
        component_paths, component_subfolders = _fp8_component_load_overrides(pipe, component_model_id)
        component_load_kwargs["pretrained_model_name_or_path"] = component_paths
        component_load_kwargs["subfolder"] = component_subfolders
    # This superset loads the sound tokenizer once so calls can opt in with
    # ``enable_sound=True`` without rebuilding a pipeline or changing graphs.
    pipe.load_components(workflow="text2video_with_sound", **component_load_kwargs)

    original_vae_encode = pipe.vae.encode
    original_vae_decode = pipe.vae.decode

    def managed_vae_encode(*args: Any, **kwargs: Any) -> Any:
        if runtime.memory_trace is not None:
            runtime.trace_checkpoint("vae_encode.start")
        try:
            return original_vae_encode(*args, **kwargs)
        finally:
            if runtime.memory_trace is not None:
                runtime.trace_checkpoint("vae_encode.end")

    def managed_vae_decode(*args: Any, **kwargs: Any) -> Any:
        runtime.prepare_for_decode()
        if runtime.memory_trace is not None:
            runtime.trace_checkpoint("vae_decode.start")
        try:
            return original_vae_decode(*args, **kwargs)
        finally:
            if runtime.memory_trace is not None:
                runtime.trace_checkpoint("vae_decode.end")

    pipe.vae.encode = managed_vae_encode
    pipe.vae.decode = managed_vae_decode

    original_decode_sound = pipe.decode_sound

    def managed_decode_sound(*args: Any, **kwargs: Any) -> Any:
        """Run the official audio decoder on GPU, then return its weights to RAM."""

        if pipe.sound_tokenizer is None:
            raise RuntimeError("sound generation was requested but the checkpoint has no sound tokenizer")
        runtime.prepare_for_decode()
        runtime.trace_checkpoint("sound_decode.start")
        pipe.sound_tokenizer.to(torch.device(config.device))
        try:
            return original_decode_sound(*args, **kwargs)
        finally:
            pipe.sound_tokenizer.to(torch.device(config.offload_device))
            runtime.trace_checkpoint("sound_decode.end")

    pipe.decode_sound = managed_decode_sound
    if enable_safety_checker:
        pipe.enable_safety_checker()
    else:
        pipe.disable_safety_checker()

    if config.vae_leaf_offload:
        pipe.vae.enable_group_offload(
            onload_device=torch.device(config.device),
            offload_device=torch.device(config.offload_device),
            offload_type="leaf_level",
            use_stream=False,
        )
    device = torch.device(config.device)
    total_vram = torch.cuda.get_device_properties(device).total_memory if device.type == "cuda" else 0
    vae_policy = _resolve_vae_memory_policy(config.memory_profile, total_vram)
    runtime.resolve_memory_policy(total_vram)
    runtime.vae_decode_mode = "tiled" if vae_policy.tile_size is not None else "untiled"
    runtime.release_cache_before_safety = vae_policy.release_cache_before_safety
    if vae_policy.tile_size is not None and vae_policy.tile_stride is not None:
        pipe.vae.enable_tiling(
            tile_sample_min_height=vae_policy.tile_size,
            tile_sample_min_width=vae_policy.tile_size,
            tile_sample_stride_height=vae_policy.tile_stride,
            tile_sample_stride_width=vae_policy.tile_stride,
        )
    return pipe


def build_text_to_video_pipeline(*args: Any, **kwargs: Any) -> StreamingCosmos3Pipeline:
    """Compatibility alias for :func:`build_pipeline`.

    The returned pipeline is auto-routing and can also run T2I, I2V, V2V,
    sound, action, and transfer workflows.
    """

    return build_pipeline(*args, **kwargs)
