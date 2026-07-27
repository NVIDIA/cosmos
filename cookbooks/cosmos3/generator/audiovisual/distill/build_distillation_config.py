# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Build short Cosmos3-Super DMD2 distillation configs from public APIs."""

import argparse
import copy
from pathlib import Path
from typing import Literal

import tomllib
from pydantic import BaseModel, ConfigDict

from cosmos_framework.callbacks.dmd2_metrics import DMD2Metrics
from cosmos_framework.callbacks.grad_clip_distillation import GradClip
from cosmos_framework.callbacks.iter_speed import IterSpeed
from cosmos_framework.callbacks.manual_gc import ManualGarbageCollection
from cosmos_framework.callbacks.skip_nan_step import SkipNaNStep
from cosmos_framework.callbacks.wandb_log import WandbCallback
from cosmos_framework.checkpoint.dcp_distill import DistributedCheckpointer
from cosmos_framework.configs.base.config import make_config
from cosmos_framework.configs.base.defaults.tokenizer import Wan2pt2VAEConfig
from cosmos_framework.configs.base.experiment.distillation.dmd2_config import (
    DMD2OptimizerConfig,
    DMD2RFConfig,
)
from cosmos_framework.configs.base.experiment.sft.models.super_model_config import (
    SUPER_MODEL_CONFIG,
)
from cosmos_framework.data.generator.joint_dataloader import (
    PackingDataLoader,
    RankPartitionedDataLoader,
)
from cosmos_framework.data.generator.local_datasets.sft_dataset import get_sft_dataset
from cosmos_framework.inference.common.config import (
    deserialize_config,
    serialize_config,
    structure_config,
)
from cosmos_framework.model.generator.distillation.dmd2_rf import DMD2RFModel
from cosmos_framework.model.generator.reasoner.qwen3_vl import (
    configs as qwen3_vl_configs,
)
from cosmos_framework.trainer.distillation import DistillationTrainer
from cosmos_framework.utils.callback import LowPrecisionCallback
from cosmos_framework.utils.config import CheckpointConfig, Config
from cosmos_framework.utils.generator.optimizer import build_lr_scheduler
from cosmos_framework.utils.lazy_config import PLACEHOLDER, LazyDict
from cosmos_framework.utils.lazy_config import LazyCall as L

Mode = Literal["t2i", "i2v"]

_RECIPE_MODEL_CONFIG = ConfigDict(extra="forbid")
_QWEN3_VL_CONFIG_PACKAGE_FILE = qwen3_vl_configs.__file__
assert _QWEN3_VL_CONFIG_PACKAGE_FILE is not None
_QWEN3_VL_32B_CONFIG = Path(_QWEN3_VL_CONFIG_PACKAGE_FILE).with_name(
    "Qwen3-VL-32B-Instruct.json"
)


class JobRecipeConfig(BaseModel):
    """User-facing run identity."""

    model_config = _RECIPE_MODEL_CONFIG

    project: str
    group: str
    name: str
    wandb_mode: str


class ParallelismRecipeConfig(BaseModel):
    """The canonical 32-GPU training topology."""

    model_config = _RECIPE_MODEL_CONFIG

    data_parallel_shard_degree: int
    context_parallel_shard_degree: int


class FixedStepSamplerRecipeConfig(BaseModel):
    """Four-step student sampling schedule."""

    model_config = _RECIPE_MODEL_CONFIG

    sample_type: str
    t_list: list[float]


class ModelRecipeConfig(BaseModel):
    """DMD2 model and loss knobs exposed by the cookbook."""

    model_config = _RECIPE_MODEL_CONFIG

    resolution: str
    max_num_tokens_after_packing: int
    lora_enabled: bool
    ema_enabled: bool
    compile_enabled: bool
    base_fps: int
    rectified_flow_shift: dict[str, int]
    rectified_flow_loss_scale: float
    train_time_video_distribution: str
    simulation_mode: str
    teacher_guidance: float
    student_update_freq: int
    vsd_loss_reduction: str
    fake_score_loss_reduction: str
    warmup_student_steps: int
    warmup_critic_steps: int
    grad_clip: bool
    parallelism: ParallelismRecipeConfig
    fixed_step_sampler: FixedStepSamplerRecipeConfig


class OptimizerGroupRecipeConfig(BaseModel):
    """One DMD2 optimizer parameter group."""

    model_config = _RECIPE_MODEL_CONFIG

    lr: float
    betas: list[float]


class OptimizerRecipeConfig(BaseModel):
    """Student and fake-score optimizer settings."""

    model_config = _RECIPE_MODEL_CONFIG

    net: OptimizerGroupRecipeConfig
    fake_score: OptimizerGroupRecipeConfig


class TrainerRecipeConfig(BaseModel):
    """Short smoke-run trainer settings."""

    model_config = _RECIPE_MODEL_CONFIG

    distributed_parallelism: str
    grad_accum_iter: int
    logging_iter: int
    max_iter: int
    seed: int
    recompile_limit: int


class CheckpointRecipeConfig(BaseModel):
    """Strict DCP save and resume settings."""

    model_config = _RECIPE_MODEL_CONFIG

    save_iter: int
    strict_resume: bool


class DataloaderTrainRecipeConfig(BaseModel):
    """Mode-specific BridgeData2 data view."""

    model_config = _RECIPE_MODEL_CONFIG

    num_video_frames: int
    conditioning_config: dict[int, float] | None = None
    append_duration_fps_timestamps: bool
    conditioning_fps_noise_std: float
    max_caption_tokens: int
    max_samples_per_batch: int
    batch_size: int
    num_workers: int
    prefetch_factor: int


class DistillationRecipeConfig(BaseModel):
    """Validated TOML surface for one cookbook mode."""

    model_config = _RECIPE_MODEL_CONFIG

    mode: Mode
    job: JobRecipeConfig
    model: ModelRecipeConfig
    optimizer: OptimizerRecipeConfig
    trainer: TrainerRecipeConfig
    checkpoint: CheckpointRecipeConfig
    dataloader_train: DataloaderTrainRecipeConfig


def load_recipe(recipe_path: Path) -> DistillationRecipeConfig:
    """Load and strictly validate one distillation TOML recipe."""
    with recipe_path.open("rb") as recipe_file:
        return DistillationRecipeConfig.model_validate(tomllib.load(recipe_file))


def _build_model_config(
    *,
    recipe: DistillationRecipeConfig,
    teacher_checkpoint: Path,
    student_checkpoint: Path,
) -> DMD2RFConfig:
    model_config = structure_config(copy.deepcopy(SUPER_MODEL_CONFIG), DMD2RFConfig)

    model_config.resolution = recipe.model.resolution
    model_config.max_num_tokens_after_packing = (
        recipe.model.max_num_tokens_after_packing
    )
    model_config.lora_enabled = recipe.model.lora_enabled
    model_config.ema.enabled = recipe.model.ema_enabled
    model_config.parallelism.data_parallel_shard_degree = (
        recipe.model.parallelism.data_parallel_shard_degree
    )
    model_config.parallelism.context_parallel_shard_degree = (
        recipe.model.parallelism.context_parallel_shard_degree
    )
    model_config.compile.enabled = recipe.model.compile_enabled
    tokenizer_config = copy.deepcopy(Wan2pt2VAEConfig)
    tokenizer_config.update(model_config.tokenizer)
    model_config.tokenizer = tokenizer_config
    model_config.tokenizer.vae_path = "${oc.env:WAN_VAE_PATH}"
    model_config.tokenizer.encode_chunk_frames = {
        "256": 68,
        "480": 24,
        "720": 12,
        "768": 12,
    }
    model_config.tokenizer.encode_exact_durations = None
    model_config.vlm_config.pretrained_weights.credentials_path = ""
    model_config.vlm_config.pretrained_weights.enable_gcs_patch_in_boto3 = False

    model_config.diffusion_expert_config.base_fps = recipe.model.base_fps
    model_config.rectified_flow_training_config.shift = dict(
        recipe.model.rectified_flow_shift
    )
    model_config.rectified_flow_training_config.loss_scale = (
        recipe.model.rectified_flow_loss_scale
    )
    model_config.rectified_flow_training_config.train_time_video_distribution = (
        recipe.model.train_time_video_distribution
    )

    model_config.vlm_config.model_instance["config"]["base_config"]["json_file"] = str(
        _QWEN3_VL_32B_CONFIG
    )
    model_config.teacher_load_from = LazyDict(
        {"load_path": str(teacher_checkpoint), "credentials": ""},
        flags={"allow_objects": True},
    )
    model_config.student_load_from = LazyDict(
        {"load_path": str(student_checkpoint), "credentials": ""},
        flags={"allow_objects": True},
    )
    model_config.load_teacher_weights = True
    model_config.vlm_config_teacher = copy.deepcopy(model_config.vlm_config)
    model_config.vlm_config_fake_score = copy.deepcopy(model_config.vlm_config)
    model_config.fixed_step_sampler_config.t_list = list(
        recipe.model.fixed_step_sampler.t_list
    )
    model_config.fixed_step_sampler_config.sample_type = (
        recipe.model.fixed_step_sampler.sample_type
    )
    model_config.simulation_mode = recipe.model.simulation_mode
    model_config.teacher_guidance = recipe.model.teacher_guidance
    model_config.student_update_freq = recipe.model.student_update_freq
    model_config.vsd_loss_reduction = recipe.model.vsd_loss_reduction
    model_config.fake_score_loss_reduction = recipe.model.fake_score_loss_reduction
    model_config.warmup_student_steps = recipe.model.warmup_student_steps
    model_config.warmup_critic_steps = recipe.model.warmup_critic_steps
    model_config.grad_clip = recipe.model.grad_clip
    return model_config


def _build_optimizer(recipe: DistillationRecipeConfig) -> LazyDict:
    optimizer = DMD2OptimizerConfig()
    optimizer.net.lr = recipe.optimizer.net.lr
    optimizer.net.betas = list(recipe.optimizer.net.betas)
    optimizer.fake_score.lr = recipe.optimizer.fake_score.lr
    optimizer.fake_score.betas = list(recipe.optimizer.fake_score.betas)
    return LazyDict(
        {"net": optimizer.net, "fake_score": optimizer.fake_score},
        flags={"allow_objects": True},
    )


def _build_dataloader(
    *, recipe: DistillationRecipeConfig, dataset_path: Path
) -> LazyDict:
    return L(PackingDataLoader)(
        audio_sample_rate=48_000,
        dataset_name=f"bridge_data2_{recipe.mode}",
        max_samples_per_batch=recipe.dataloader_train.max_samples_per_batch,
        max_sequence_length=None,
        patch_spatial=2,
        sound_latent_fps=0,
        tokenizer_spatial_compression_factor=16,
        tokenizer_temporal_compression_factor=4,
        dataloader=L(RankPartitionedDataLoader)(
            batch_size=recipe.dataloader_train.batch_size,
            datasets={
                "video": {
                    "ratio": 1,
                    "dataset": L(get_sft_dataset)(
                        append_duration_fps_timestamps=(
                            recipe.dataloader_train.append_duration_fps_timestamps
                        ),
                        append_resolution_info=True,
                        caption_suffix="",
                        cfg_dropout_keep_metadata=False,
                        cfg_dropout_rate=0.1,
                        conditioning_config=recipe.dataloader_train.conditioning_config,
                        conditioning_fps=-1,
                        conditioning_fps_noise_std=(
                            recipe.dataloader_train.conditioning_fps_noise_std
                        ),
                        frame_selection_mode="first",
                        jsonl_paths=[
                            str(dataset_path / "train/video_dataset_file.jsonl")
                        ],
                        max_caption_tokens=recipe.dataloader_train.max_caption_tokens,
                        min_short_edge=0,
                        num_video_frames=recipe.dataloader_train.num_video_frames,
                        resolution=recipe.model.resolution,
                        sample_by_window=False,
                        temporal_compression_factor=4,
                        temporal_interval_mode="max_30fps",
                        tokenizer_config="${model.config.vlm_config.tokenizer}",
                        use_system_prompt=False,
                    ),
                }
            },
            in_order=True,
            num_workers=recipe.dataloader_train.num_workers,
            persistent_workers=True,
            pin_memory=True,
            prefetch_factor=recipe.dataloader_train.prefetch_factor,
            sampler=None,
        ),
    )


def build_config(
    *,
    recipe_path: Path,
    dataset_path: Path,
    teacher_checkpoint: Path,
    student_checkpoint: Path,
) -> Config:
    """Build one mode-specific config for the two-phase distillation smoke workflow."""
    recipe = load_recipe(recipe_path)
    model_config = _build_model_config(
        recipe=recipe,
        teacher_checkpoint=teacher_checkpoint,
        student_checkpoint=student_checkpoint,
    )

    config = make_config()
    config.job.project = recipe.job.project
    config.job.group = recipe.job.group
    config.job.name = recipe.job.name
    config.job.wandb_mode = recipe.job.wandb_mode

    config.model = L(DMD2RFModel)(config=model_config, _recursive_=False)
    config.optimizer = _build_optimizer(recipe)
    config.scheduler = L(build_lr_scheduler)(
        optimizer=PLACEHOLDER,
        lr_scheduler_type="LambdaLinear",
        warm_up_steps=[0],
        cycle_lengths=[10_000_000_000_000],
        f_start=[1.0],
        f_max=[1.0],
        f_min=[1.0],
    )
    config.dataloader_train = _build_dataloader(
        recipe=recipe, dataset_path=dataset_path
    )
    config.dataloader_val = None

    config.trainer.type = DistillationTrainer
    config.trainer.distributed_parallelism = recipe.trainer.distributed_parallelism
    config.trainer.grad_accum_iter = recipe.trainer.grad_accum_iter
    config.trainer.logging_iter = recipe.trainer.logging_iter
    config.trainer.max_iter = recipe.trainer.max_iter
    config.trainer.run_validation = False
    config.trainer.run_validation_on_start = False
    config.trainer.save_zero_checkpoint = False
    config.trainer.seed = recipe.trainer.seed
    config.trainer.straggler_detection.enabled = False
    config.trainer.compile_config.recompile_limit = recipe.trainer.recompile_limit
    config.trainer.callbacks = LazyDict(
        {
            "dmd2_metrics": L(DMD2Metrics)(output_path=""),
            "grad_clip": L(GradClip)(clip_norm=1.0),
            "iter_speed": L(IterSpeed)(
                every_n=1, save_s3=False, save_s3_every_log_n=500, hit_thres=200
            ),
            "low_precision": L(LowPrecisionCallback)(
                update_iter=1, config=None, trainer=None
            ),
            "manual_gc": L(ManualGarbageCollection)(every_n=5),
            "skip_nan_step": L(SkipNaNStep)(max_consecutive_nan=100),
            "wandb": L(WandbCallback)(
                logging_iter_multipler=1,
                save_logging_iter_multipler=10,
                save_s3=False,
            ),
        },
        flags={"allow_objects": True},
    )

    config.checkpoint = CheckpointConfig(
        type=L(DistributedCheckpointer)(),
        dcp_async_mode_enabled=False,
        dcp_load_dedup=True,
        save_iter=recipe.checkpoint.save_iter,
        strict_resume=recipe.checkpoint.strict_resume,
        load_path="",
        load_training_state=False,
        broadcast_via_filesystem=True,
    )
    config.checkpoint.hf_export.enabled = False
    config.upload_reproducible_setup = False
    return config


def write_config(config: Config, output_path: Path) -> None:
    """Serialize a training config and verify that Cosmos Framework can read it."""
    serialize_config(config, output_path)
    deserialize_config(output_path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe-toml", type=Path, required=True)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--teacher-checkpoint", type=Path, required=True)
    parser.add_argument("--student-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Build, serialize, and round-trip validate one cookbook config."""
    args = _parse_args()
    config = build_config(
        recipe_path=args.recipe_toml,
        dataset_path=args.dataset_path,
        teacher_checkpoint=args.teacher_checkpoint,
        student_checkpoint=args.student_checkpoint,
    )
    write_config(config, args.output)
    if args.validate_only:
        recipe = load_recipe(args.recipe_toml)
        print(f"Validated {recipe.mode} distillation config: {args.output}")


if __name__ == "__main__":
    main()
