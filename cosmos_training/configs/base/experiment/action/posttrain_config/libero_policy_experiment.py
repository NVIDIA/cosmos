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

# Policy Experiment for LIBERO - training a policy network to predict video and action jointly

import copy
import os
from datetime import datetime

from hydra.core.config_store import ConfigStore
from loguru import logger as log

from configs.base.experiment.action._experiment_helpers import (
    LIBERO_BASELINE_BATCH_SIZE,
    LIBERO_BASELINE_NUM_WORKERS,
    LIBERO_BASELINE_TRAINING_ITERATIONS,
    LIBERO_LOCAL_ROOT_ENV,
    make_libero_dataset,
)
from configs.base.experiment.action.pretrained_config.cosmos3_2b import make_2b_experiment
from configs.base.experiment.action.pretrained_config.cosmos3_8b import make_8b_experiment

ACTION_STATS_PATH = "cosmos/data/vfm/action/libero_action_stats_10k.json"
NATIVE_FRAME_WISE_ROT6D_ACTION_STATS_PATH = (
    "cosmos/data/vfm/action/normalizers/libero_native_frame_wise_relative_rot6d.json"
)
cs = ConfigStore.instance()

_LIBERO_REPO_IDS = ["libero_10", "libero_object", "libero_spatial", "libero_goal"]

# Cluster lustre layout (default).
_LIBERO_ROOTS_LUSTRE = [
    "/lustre/fsw/portfolios/dir/projects/dir_cosmos_base_lustre/maxzhaoshuol/dataset/libero_10_no_noops_1.0.0_lerobot_aligned",
    "/lustre/fsw/portfolios/dir/projects/dir_cosmos_base_lustre/maxzhaoshuol/dataset/libero_object_no_noops_1.0.0_lerobot_aligned",
    "/lustre/fsw/portfolios/dir/projects/dir_cosmos_base_lustre/maxzhaoshuol/dataset/libero_spatial_no_noops_1.0.0_lerobot",
    "/lustre/fsw/portfolios/dir/projects/dir_cosmos_base_lustre/maxzhaoshuol/dataset/libero_goal_no_noops_1.0.0_lerobot",
]

# Local-mount layout (matches gs://nv-00-10206-robot/lerobot_v30/). Order matches
# _LIBERO_REPO_IDS. Mirrors LIBERO_LOCAL_SUITE_DIRS in _experiment_helpers.py.
_LIBERO_LOCAL_SUITE_DIRS = [
    "libero_10_no_noops_1.0.0_lerobot_aligned_20260124",
    "libero_object_no_noops_1.0.0_lerobot_aligned_20260124",
    "libero_spatial_no_noops_1.0.0_lerobot_20260124",
    "libero_goal_no_noops_1.0.0_lerobot_20260124",
]


def _resolve_libero_roots() -> list[str]:
    """Resolve the libero ``root`` list, honoring ``$LIBERO_LOCAL_DATA_ROOT``.

    Mirrors ``_resolve_libero_default_roots`` in ``_experiment_helpers.py`` so
    debug machines without the lustre share can opt into a local mount via the
    same env var that the other libero experiments already respect.
    """
    base = os.environ.get(LIBERO_LOCAL_ROOT_ENV)
    if not base:
        return list(_LIBERO_ROOTS_LUSTRE)

    candidates = [os.path.join(base, suite) for suite in _LIBERO_LOCAL_SUITE_DIRS]
    missing = [p for p in candidates if not os.path.isdir(p)]
    if missing:
        raise FileNotFoundError(
            f"${LIBERO_LOCAL_ROOT_ENV}={base} is set but the following LIBERO suite directories are missing: {missing}"
        )
    log.info(f"[libero] using local LIBERO mount at {base} (via ${LIBERO_LOCAL_ROOT_ENV})")
    return candidates


_LIBERO_ROOTS = _resolve_libero_roots()


def _build_libero_policy_base(
    exp_name: str,
    *,
    training_iterations: int = LIBERO_BASELINE_TRAINING_ITERATIONS,
    batch_size: int = LIBERO_BASELINE_BATCH_SIZE,
    num_workers: int = LIBERO_BASELINE_NUM_WORKERS,
    job_group: str = "debugging",
    job_project: str | None = None,
    mode: str = "policy",
    **dataset_kwargs,
) -> dict:
    """Build a libero policy-mode experiment.

    Mirrors the original ``libero_exp`` setup (callbacks override, scheduler
    f_max=1.0, warm_up = iter//20). ``mode`` defaults to ``"policy"`` but can
    be overridden (e.g., the ``normalized_action_joint_exp`` chain uses other
    modes despite living in this file).
    """
    dataset_kwargs.setdefault("mode", mode)
    exp = make_2b_experiment(
        exp_name=exp_name,
        datasets=make_libero_dataset(**dataset_kwargs),
        batch_size=batch_size,
        num_workers=num_workers,
        training_iterations=training_iterations,
    )

    # Match libero_experiment.libero_exp post-build mutations.
    for i, d in enumerate(exp["defaults"]):
        if isinstance(d, dict) and "override /callbacks" in d:
            exp["defaults"][i] = {
                "override /callbacks": [
                    "basic",
                    "optimization",
                    "job_monitor",
                    "training_stats",
                ]
            }
            break
    exp["scheduler"]["f_max"] = [1.0]
    exp["scheduler"]["warm_up_steps"] = [training_iterations // 20]

    exp["job"]["group"] = job_group
    if job_project is not None:
        exp["job"]["project"] = job_project
    return exp


def _build_libero_policy_base_8b(
    exp_name: str,
    *,
    training_iterations: int = LIBERO_BASELINE_TRAINING_ITERATIONS,
    batch_size: int = LIBERO_BASELINE_BATCH_SIZE,
    num_workers: int = LIBERO_BASELINE_NUM_WORKERS,
    job_group: str = "debugging",
    job_project: str | None = None,
    mode: str = "policy",
    keys_to_skip_loading: list[str] | None = None,
    use_deterministic_seed: bool = False,
    **dataset_kwargs,
) -> dict:
    """8B counterpart to ``_build_libero_policy_base``.

    Mirrors the libero post-build mutations (callbacks override, scheduler
    f_max=1.0, warm_up = iter//20) but builds from ``make_8b_experiment``
    instead of the 2B base.
    """
    dataset_kwargs.setdefault("mode", mode)
    extra_kwargs: dict = {}
    if keys_to_skip_loading is not None:
        extra_kwargs["keys_to_skip_loading"] = keys_to_skip_loading
    exp = make_8b_experiment(
        exp_name=exp_name,
        datasets=make_libero_dataset(**dataset_kwargs),
        batch_size=batch_size,
        num_workers=num_workers,
        training_iterations=training_iterations,
        use_deterministic_seed=use_deterministic_seed,
        **extra_kwargs,
    )

    for i, d in enumerate(exp["defaults"]):
        if isinstance(d, dict) and "override /callbacks" in d:
            exp["defaults"][i] = {
                "override /callbacks": [
                    "basic",
                    "optimization",
                    "job_monitor",
                    "training_stats",
                ]
            }
            break
    exp["scheduler"]["f_max"] = [1.0]
    exp["scheduler"]["warm_up_steps"] = [training_iterations // 20]

    exp["trainer"]["callbacks"]["compile_tokenizer"]["enabled"] = True
    exp["trainer"]["callbacks"]["compile_tokenizer"]["warmup_resolutions"] = ["256", "480", "720"]

    exp["job"]["group"] = job_group
    if job_project is not None:
        exp["job"]["project"] = job_project
    return exp


# ---------------------------------------------------------------------------
# libero_exp_policy — base policy experiment
# ---------------------------------------------------------------------------
libero_exp_policy = _build_libero_policy_base(
    exp_name=f"libero_exp_policy_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
)
cs.store(
    group="experiment",
    package="_global_",
    name="libero_exp_policy",
    node=libero_exp_policy,
)


# ---------------------------------------------------------------------------
# LR sweep on libero_exp_policy
# ---------------------------------------------------------------------------
def lr_sweep():
    for lr_value, suffix in [(4e-4, "lr4e4"), (2e-4, "lr2e4"), (8e-4, "lr8e4")]:
        name = f"libero_exp_policy_{suffix}"
        exp = _build_libero_policy_base(exp_name=name, job_group="action_libero")
        exp["optimizer"]["lr"] = lr_value
        cs.store(group="experiment", package="_global_", name=name, node=exp)


# ---------------------------------------------------------------------------
# Multi-view policy experiment (camera_mode=concat_view)
# ---------------------------------------------------------------------------
def multi_view_policy_exp():
    """
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. torchrun --nproc_per_node=1 --master_port=12341 scripts/train.py --config=configs/base/config.py -- experiment=libero_exp_policy_multi_view_lr2e4
    """
    name = "libero_exp_policy_multi_view_lr2e4"
    exp = _build_libero_policy_base(
        exp_name=name,
        job_group="action_libero",
        job_project="cosmos3_action_libero",
        camera_mode="concat_view",
    )
    cs.store(group="experiment", package="_global_", name=name, node=exp)

    """
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. torchrun --nproc_per_node=1 --master_port=12341 scripts/train.py --config=configs/base/config.py -- experiment=libero_exp_policy_multi_view_lr2e4_20k
    """
    name_20k = "libero_exp_policy_multi_view_lr2e4_20k"
    exp_20k = _build_libero_policy_base(
        exp_name=name_20k,
        training_iterations=20_000,
        job_group="action_libero",
        job_project="cosmos3_action_libero",
        camera_mode="concat_view",
    )
    cs.store(group="experiment", package="_global_", name=name_20k, node=exp_20k)


# ---------------------------------------------------------------------------
# Single-view 60k iter policy experiment
# ---------------------------------------------------------------------------
def single_view_policy_exp():
    """
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. torchrun --nproc_per_node=1 --master_port=12341 scripts/train.py --config=configs/base/config.py -- experiment=libero_exp_policy_single_view_lr2e4_60k
    """
    name = "libero_exp_policy_single_view_lr2e4_60k"
    exp = _build_libero_policy_base(
        exp_name=name,
        training_iterations=60_000,
        batch_size=256,
        job_group="action_libero",
        job_project="cosmos3_action_libero",
    )
    exp["optimizer"]["lr"] = 2e-4
    exp["scheduler"]["warm_up_steps"] = [100]
    exp["scheduler"]["cycle_lengths"] = [60_000]
    cs.store(group="experiment", package="_global_", name=name, node=exp)


# ---------------------------------------------------------------------------
# Normalized-action policy experiment (with image / wrist_image variants)
# ---------------------------------------------------------------------------
def normalized_action_policy_exp():
    """
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. torchrun --nproc_per_node=1 --master_port=12341 scripts/train.py --config=configs/base/config.py -- experiment=libero_exp_policy_norm_action_lr2e4_60k
    """
    # Use the shared action stats JSON through LIBERODataset's new normalization API.
    name = "libero_exp_policy_norm_action_lr2e4_60k"
    exp = _build_libero_policy_base(
        exp_name=name,
        training_iterations=60_000,
        batch_size=256,
        job_group="action_libero",
        job_project="cosmos3_action_libero",
        action_normalization="minmax",
        action_stats_path=ACTION_STATS_PATH,
    )
    exp["optimizer"]["lr"] = 2e-4
    exp["scheduler"]["warm_up_steps"] = [100]
    exp["scheduler"]["cycle_lengths"] = [60_000]
    exp["checkpoint"]["load_from_object_store"] = dict(bucket="nv-00-10206-checkpoint")
    cs.store(group="experiment", package="_global_", name=name, node=exp)

    """
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. torchrun --nproc_per_node=1 --master_port=12341 scripts/train.py --config=configs/base/config.py -- experiment=libero_exp_policy_norm_action_wrist_lr2e4_60k
    """
    wrist_name = "libero_exp_policy_norm_action_wrist_lr2e4_60k"
    wrist_exp = _build_libero_policy_base(
        exp_name=wrist_name,
        training_iterations=60_000,
        batch_size=256,
        job_group="action_libero",
        job_project="cosmos3_action_libero",
        action_normalization="minmax",
        action_stats_path=ACTION_STATS_PATH,
        camera_mode="wrist_image",
    )
    wrist_exp["optimizer"]["lr"] = 2e-4
    wrist_exp["scheduler"]["warm_up_steps"] = [100]
    wrist_exp["scheduler"]["cycle_lengths"] = [60_000]
    wrist_exp["checkpoint"]["load_from_object_store"] = dict(bucket="nv-00-10206-checkpoint")
    cs.store(group="experiment", package="_global_", name=wrist_name, node=wrist_exp)


# ---------------------------------------------------------------------------
# Wrist-view 60k iter policy experiment
# ---------------------------------------------------------------------------
def wrist_view_policy_exp():
    """
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. torchrun --nproc_per_node=1 --master_port=12341 scripts/train.py --config=configs/base/config.py -- experiment=libero_exp_policy_wrist_view_lr2e4_60k
    """
    name = "libero_exp_policy_wrist_view_lr2e4_60k"
    exp = _build_libero_policy_base(
        exp_name=name,
        training_iterations=60_000,
        batch_size=256,
        job_group="action_libero",
        job_project="cosmos3_action_libero",
        camera_mode="wrist_image",
    )
    exp["optimizer"]["lr"] = 2e-4
    exp["scheduler"]["warm_up_steps"] = [100]
    exp["scheduler"]["cycle_lengths"] = [60_000]
    exp["checkpoint"]["load_from_object_store"] = dict(bucket="nv-00-10206-checkpoint")
    cs.store(group="experiment", package="_global_", name=name, node=exp)


# ---------------------------------------------------------------------------
# Sweep camera_mode × action_space × rotation_space (lr=2e-4)
# ---------------------------------------------------------------------------
def action_camera_rotation_sweep():
    camera_modes = ["image", "wrist_image"]
    action_spaces = ["relative", "frame_wise_relative"]
    rotation_spaces = ["9d", "6d", "3d"]

    for camera_mode in camera_modes:
        for action_space in action_spaces:
            for rotation_space in rotation_spaces:
                cam_short = "img" if camera_mode == "image" else "wrist"
                act_short = "rel" if action_space == "relative" else "fwrel"
                rot_short = rotation_space
                name = f"libero_exp_policy_{cam_short}_{act_short}_{rot_short}_varlen"

                exp = _build_libero_policy_base(
                    exp_name=name,
                    job_group="action_libero",
                    camera_mode=camera_mode,
                    action_space=action_space,
                    rotation_space=rotation_space,
                )
                exp["optimizer"]["lr"] = 2e-4
                cs.store(group="experiment", package="_global_", name=name, node=exp)


# ---------------------------------------------------------------------------
# Normalized-action joint experiments (libero-90 removed for faster convergence).
# These use mode="policy" or mode="forward_dynamics" despite the name.
# Build the libero dataset then layer further variants.
# ---------------------------------------------------------------------------
def normalized_action_joint_exp():
    # Common dataset-kwargs base for the libero wrist, normalized variants.
    wrist_norm_kwargs = dict(
        repo_id=copy.deepcopy(_LIBERO_REPO_IDS),
        root=copy.deepcopy(_LIBERO_ROOTS),
        camera_mode="wrist_image",
        action_normalization="minmax",
        action_stats_path=ACTION_STATS_PATH,
    )

    _UNSET = object()

    def _make(
        exp_name: str,
        *,
        mode: str = "policy",
        max_iter: int = 20_000,
        cycle_lengths: list | None = None,
        warm_up_steps: list | None = None,
        f_min: list | None = None,
        lr: float = 2e-4,
        batch_size: int = 256,
        num_workers: int = 24,
        cfg_dropout_rate: float | None = None,
        action_loss_weight: float | None = None,
        max_samples_per_batch: int | None = None,
        max_sequence_length=_UNSET,  # sentinel: pass _UNSET to keep parent default
        run_validation_on_start: bool | None = None,
        logging_iter: int | None = None,
        validation_iter: int | None = None,
        max_val_iter: int | None = None,
        load_path: str | None = None,
        action_channel_masking: bool | None = None,
        fps: int | None = None,
    ) -> dict:
        ds_kwargs = dict(wrist_norm_kwargs)
        ds_kwargs["mode"] = mode
        if fps is not None:
            ds_kwargs["fps"] = fps

        exp = _build_libero_policy_base(
            exp_name=exp_name,
            training_iterations=max_iter,
            batch_size=batch_size,
            num_workers=num_workers,
            job_group="action_libero",
            job_project="cosmos3_action_libero",
            mode=mode,
            **{k: v for k, v in ds_kwargs.items() if k != "mode"},
        )
        exp["optimizer"]["lr"] = lr
        exp["scheduler"]["f_min"] = f_min if f_min is not None else [0.0]
        exp["scheduler"]["warm_up_steps"] = warm_up_steps if warm_up_steps is not None else [2000]
        exp["scheduler"]["cycle_lengths"] = cycle_lengths if cycle_lengths is not None else [max_iter]
        exp["trainer"]["max_iter"] = max_iter
        exp["trainer"]["run_validation"] = False
        if logging_iter is not None:
            exp["trainer"]["logging_iter"] = logging_iter
        if run_validation_on_start is not None:
            exp["trainer"]["run_validation_on_start"] = run_validation_on_start
        if validation_iter is not None:
            exp["trainer"]["validation_iter"] = validation_iter
        if max_val_iter is not None:
            exp["trainer"]["max_val_iter"] = max_val_iter
        exp["checkpoint"]["load_from_object_store"] = dict(bucket="nv-00-10206-checkpoint-experiments")
        exp["checkpoint"]["save_to_object_store"] = dict(bucket="nv-00-10206-checkpoint-experiments")
        if load_path is not None:
            exp["checkpoint"]["load_path"] = load_path

        # dataloader-level overrides (action_channel_masking, cfg_dropout_rate, max_samples_per_batch).
        ds_node = exp["dataloader_train"]["dataloaders"]["action_data"]["dataloader"]["dataset"]
        if cfg_dropout_rate is not None:
            ds_node["cfg_dropout_rate"] = cfg_dropout_rate
        if action_channel_masking is not None:
            ds_node["action_channel_masking"] = action_channel_masking
        if max_samples_per_batch is not None:
            exp["dataloader_train"]["max_samples_per_batch"] = max_samples_per_batch
        if max_sequence_length is not _UNSET:
            # Setting to None disables sequence packing (each sample its own batch slot).
            # Default from make_2b_experiment is the ${model.config.max_num_tokens_after_packing}
            # interpolation; pass _UNSET to keep that.
            exp["dataloader_train"]["max_sequence_length"] = max_sequence_length

        if action_loss_weight is not None:
            exp["model"]["config"]["rectified_flow_training_config"]["action_loss_weight"] = action_loss_weight

        return exp

    # libero_exp_policy_norm_action_wrist_lr2e4_20k — the root of this chain.
    name = "libero_exp_policy_norm_action_wrist_lr2e4_20k"
    exp = _make(name, mode="policy", lr=2e-4, max_iter=20_000)
    cs.store(group="experiment", package="_global_", name=name, node=exp)

    # _no_cfg_20k = parent + cfg_dropout_rate=0.0
    name = "libero_exp_policy_norm_action_wrist_lr2e4_no_cfg_20k"
    exp = _make(name, mode="policy", lr=2e-4, max_iter=20_000, cfg_dropout_rate=0.0)
    cs.store(group="experiment", package="_global_", name=name, node=exp)

    # _no_cfg_bs64_20k = parent + batch_size=64
    name = "libero_exp_policy_norm_action_wrist_lr2e4_no_cfg_bs64_20k"
    exp = _make(name, mode="policy", lr=2e-4, max_iter=20_000, cfg_dropout_rate=0.0, batch_size=64)
    cs.store(group="experiment", package="_global_", name=name, node=exp)

    # _lr1e3_no_cfg_bs64_20k = bs64_20k + lr=1e-3
    name = "libero_exp_policy_norm_action_wrist_lr1e3_no_cfg_bs64_20k"
    exp = _make(name, mode="policy", lr=1e-3, max_iter=20_000, cfg_dropout_rate=0.0, batch_size=64)
    cs.store(group="experiment", package="_global_", name=name, node=exp)

    # _lr1e3_no_cfg_bs256_20k = no_cfg_20k + lr=1e-3 (bs=256 inherited)
    name = "libero_exp_policy_norm_action_wrist_lr1e3_no_cfg_bs256_20k"
    exp = _make(name, mode="policy", lr=1e-3, max_iter=20_000, cfg_dropout_rate=0.0)
    cs.store(group="experiment", package="_global_", name=name, node=exp)

    # action10 variants — action_loss_weight=100.0 + various LR
    for lr_v, suffix in [(4e-5, "lr4e5"), (2e-5, "lr2e5"), (1e-4, "lr1e4"), (3e-4, "lr3e4")]:
        name = f"libero_exp_policy_norm_action_wrist_{suffix}_no_cfg_bs256_action10_20k"
        exp = _make(
            name,
            mode="policy",
            lr=lr_v,
            max_iter=20_000,
            cfg_dropout_rate=0.0,
            action_loss_weight=100.0,
        )
        cs.store(group="experiment", package="_global_", name=name, node=exp)

    # _lr5e4_no_cfg_30k — adds validation overrides + lr=5e-4 + cfg=0
    name = "libero_exp_policy_norm_action_wrist_lr5e4_no_cfg_30k"
    exp = _make(
        name,
        mode="policy",
        lr=5e-4,
        max_iter=20_000,
        cfg_dropout_rate=0.0,
        run_validation_on_start=False,
        validation_iter=1000,
        max_val_iter=40,
    )
    cs.store(group="experiment", package="_global_", name=name, node=exp)

    # _lr5e4_no_cfg_30k_resume — same + load_path override
    name = "libero_exp_policy_norm_action_wrist_lr5e4_no_cfg_30k_resume"
    exp = _make(
        name,
        mode="policy",
        lr=5e-4,
        max_iter=20_000,
        cfg_dropout_rate=0.0,
        run_validation_on_start=False,
        validation_iter=1000,
        max_val_iter=40,
        load_path="cosmos3_uva_libero/uva_libero/libero_exp_policy_norm_action_wrist_lr2e4_no_cfg_bs64_20k/checkpoints/iter_000018500",
    )
    cs.store(group="experiment", package="_global_", name=name, node=exp)

    # _lr5e4_no_cfg_40k — 40k iters, max_samples_per_batch=256, max_sequence_length=None,
    # logging_iter=10, run_validation_on_start=False, lr=1e-4 (override at the end)
    name = "libero_exp_policy_norm_action_wrist_lr5e4_no_cfg_40k"
    exp = _make(
        name,
        mode="policy",
        lr=1e-4,
        max_iter=40_000,
        cycle_lengths=[40_000],
        cfg_dropout_rate=0.0,
        run_validation_on_start=False,
        logging_iter=10,
        max_samples_per_batch=256,
        max_sequence_length=None,
    )
    cs.store(group="experiment", package="_global_", name=name, node=exp)

    # _lr5e4_no_cfg_40k_20fps_re — 40k iters, fps=20, lr=5e-5
    name = "libero_exp_policy_norm_action_wrist_lr5e4_no_cfg_40k_20fps_re"
    exp_40k_20fps = _make(
        name,
        mode="policy",
        lr=5e-5,
        max_iter=40_000,
        cycle_lengths=[40_000],
        cfg_dropout_rate=0.0,
        run_validation_on_start=False,
        logging_iter=100,
        max_samples_per_batch=256,
        max_sequence_length=None,
        fps=20,
    )
    cs.store(group="experiment", package="_global_", name=name, node=exp_40k_20fps)

    # _fd_norm_action_wrist_lr5e4_no_cfg_20k_20fps — same chain, 20k iters, mode=forward_dynamics
    name = "libero_exp_fd_norm_action_wrist_lr5e4_no_cfg_20k_20fps"
    exp = _make(
        name,
        mode="forward_dynamics",
        lr=5e-5,
        max_iter=20_000,
        cycle_lengths=[20_000],
        cfg_dropout_rate=0.0,
        run_validation_on_start=False,
        logging_iter=100,
        max_samples_per_batch=256,
        max_sequence_length=None,
        fps=20,
    )
    cs.store(group="experiment", package="_global_", name=name, node=exp)


# ---------------------------------------------------------------------------
# Action-design ablation study — joint and policy mode variants.
# All inherit settings from libero_exp_policy_norm_action_wrist_lr5e4_no_cfg_40k_20fps_re
# (libero wrist, action_normalization="minmax", fps=20, max_samples_per_batch=256).
# ---------------------------------------------------------------------------
def action_design_ablation():
    wrist_norm_kwargs = dict(
        repo_id=copy.deepcopy(_LIBERO_REPO_IDS),
        root=copy.deepcopy(_LIBERO_ROOTS),
        camera_mode="wrist_image",
        action_normalization="minmax",
        action_stats_path=ACTION_STATS_PATH,
        fps=20,
    )

    def _make_ablation_2b(
        exp_name: str,
        *,
        mode: str,
        action_loss_weight: float = 100.0,
        action_channel_masking: bool = True,
        lr_multipliers: dict | None = None,
        cfg_dropout_rate: float = 0.1,
        dataset_overrides: dict | None = None,
    ) -> dict:
        ds_kwargs = dict(wrist_norm_kwargs)
        if dataset_overrides is not None:
            ds_kwargs.update(dataset_overrides)
        exp = _build_libero_policy_base(
            exp_name=exp_name,
            training_iterations=20_000,
            batch_size=256,
            num_workers=24,
            job_group="action_libero",
            job_project="cosmos3_action_libero",
            mode=mode,
            **ds_kwargs,
        )
        exp["optimizer"]["lr"] = 1e-4
        if lr_multipliers is not None:
            exp["optimizer"]["lr_multipliers"] = lr_multipliers
        exp["scheduler"]["f_min"] = [0.0]
        exp["scheduler"]["warm_up_steps"] = [2000]
        exp["scheduler"]["cycle_lengths"] = [20_000]
        exp["trainer"]["max_iter"] = 20_000
        exp["trainer"]["logging_iter"] = 100
        exp["trainer"]["run_validation"] = False
        exp["trainer"]["run_validation_on_start"] = False
        exp["model"]["config"]["rectified_flow_training_config"]["action_loss_weight"] = action_loss_weight
        # cfg_dropout_rate + action_channel_masking are inside dataloader_train.
        ds_node = exp["dataloader_train"]["dataloaders"]["action_data"]["dataloader"]["dataset"]
        ds_node["cfg_dropout_rate"] = cfg_dropout_rate
        ds_node["action_channel_masking"] = action_channel_masking
        exp["dataloader_train"]["max_samples_per_batch"] = 256
        exp["dataloader_train"]["max_sequence_length"] = None
        exp["checkpoint"]["load_from_object_store"] = dict(bucket="nv-00-10206-checkpoint-experiments")
        exp["checkpoint"]["save_to_object_store"] = dict(bucket="nv-00-10206-checkpoint-experiments")
        return exp

    # First register the lr_sweep_lr1e4 anchor used by ablations as their parent
    # (mode=joint, cfg_dropout_rate=0.1, action_loss_weight=10.0 inherited).
    name = "libero_exp_policy_norm_action_wrist_no_cfg_40k_20fps_lr_sweep_lr1e4"
    exp = _make_ablation_2b(
        name,
        mode="joint",
        action_loss_weight=10.0,  # inherited from base 2B make_2b_experiment
    )
    # _lr_sweep_lr1e4 keeps the 40k schedule, not 20k.
    exp["trainer"]["max_iter"] = 40_000
    exp["scheduler"]["cycle_lengths"] = [40_000]
    cs.store(group="experiment", package="_global_", name=name, node=exp)

    # Joint ablations — all 20k, action_loss_weight=100 except *_wo_action_loss_reweighting
    cs.store(
        group="experiment",
        package="_global_",
        name="libero_exp_ablation_study_joint_baseline",
        node=_make_ablation_2b(
            "libero_exp_ablation_study_joint_baseline",
            mode="joint",
            action_loss_weight=100.0,
        ),
    )

    cs.store(
        group="experiment",
        package="_global_",
        name="libero_exp_ablation_study_joint_wo_action_masking",
        node=_make_ablation_2b(
            "libero_exp_ablation_study_joint_wo_action_masking",
            mode="joint",
            action_loss_weight=100.0,
            action_channel_masking=False,
        ),
    )

    cs.store(
        group="experiment",
        package="_global_",
        name="libero_exp_ablation_study_joint_wo_action_loss_reweighting",
        node=_make_ablation_2b(
            "libero_exp_ablation_study_joint_wo_action_loss_reweighting",
            mode="joint",
            action_loss_weight=10.0,
        ),
    )

    cs.store(
        group="experiment",
        package="_global_",
        name="libero_exp_ablation_study_joint_w_action_lr_scaling",
        node=_make_ablation_2b(
            "libero_exp_ablation_study_joint_w_action_lr_scaling",
            mode="joint",
            action_loss_weight=10.0,
            lr_multipliers={
                "action2llm": 5.0,
                "llm2action": 5.0,
                "action_modality_embed": 5.0,
            },
        ),
    )

    # Policy ablations — same as joint but mode=policy
    cs.store(
        group="experiment",
        package="_global_",
        name="libero_exp_ablation_study_policy_baseline",
        node=_make_ablation_2b(
            "libero_exp_ablation_study_policy_baseline",
            mode="policy",
            action_loss_weight=100.0,
        ),
    )

    cs.store(
        group="experiment",
        package="_global_",
        name="libero_exp_ablation_study_policy_wo_action_masking",
        node=_make_ablation_2b(
            "libero_exp_ablation_study_policy_wo_action_masking",
            mode="policy",
            action_loss_weight=100.0,
            action_channel_masking=False,
        ),
    )

    cs.store(
        group="experiment",
        package="_global_",
        name="libero_exp_ablation_study_policy_wo_action_loss_reweighting",
        node=_make_ablation_2b(
            "libero_exp_ablation_study_policy_wo_action_loss_reweighting",
            mode="policy",
            action_loss_weight=10.0,
        ),
    )

    cs.store(
        group="experiment",
        package="_global_",
        name="libero_exp_ablation_study_policy_w_action_lr_scaling",
        node=_make_ablation_2b(
            "libero_exp_ablation_study_policy_w_action_lr_scaling",
            mode="policy",
            action_loss_weight=10.0,
            lr_multipliers={
                "action2llm": 5.0,
                "llm2action": 5.0,
                "action_modality_embed": 5.0,
            },
        ),
    )


# ---------------------------------------------------------------------------
# 8B variants of libero_exp_ablation_study_policy_w_action_lr_scaling.
#   * _8b              — built on the default 8B pretrained checkpoint from
#                         ``make_8b_experiment`` (cosmos3 8B multires recipe).
#   * _8b_midtrain_v1p1 — built on the action mid-training checkpoint
#                         ``cosmos3_action/midtrain_v1p1/action_midtrain_exp005_v1p1_equalTokens
#                         /checkpoints/iter_000013000/`` (action layers loaded too).
#   * _8b_midtrain_v1p1_concat_view — same mid-training checkpoint, but with
#                         ``camera_mode="concat_view"`` for concatenated third-person
#                         and wrist views.
#   * _8b_midtrain_v1p1_concat_view_rot6d_native — same as concat_view, but
#                         with native-frame 6D rotations and no OpenCV pose
#                         conversion.
#   * _8b_midtrain_v1p1_concat_view_rot6d_native_no_rot_norm_quantile — same as
#                         rot6d_native, but uses quantile action normalization
#                         while leaving 6D rotation dims unnormalized.
#   * _8b_midtrain_v1p1_concat_view_rot6d_native_rot_norm_quantile — same as
#                         rot6d_native, but normalizes 6D rotation dims too
#                         with quantile action normalization.
#   * *_det_seed       — deterministic-dataloader-seed variants of the two
#                         native quantile experiments.
#   * *_det_seed_iter40k — same as the rot_norm_quantile deterministic-seed
#                         variant, but warm-starts from the 40k action-midtrain
#                         checkpoint.
#   * _8b_exp506_midtraining_v1_*_iter40k — same deterministic native normalization
#                         setups, but warm-start from the exp506 VFM base
#                         checkpoint and initialize action layers fresh.
#   * _8b_concat_view  — same as _8b but with ``camera_mode="concat_view"`` so the
#                         third-person and wrist views are concatenated along
#                         width into a single (H, 2*W) frame fed to the model.
# All other hparams (lr, schedule, dataloader settings, action_loss_weight,
# lr_multipliers) are kept identical to the 2B parent.
# ---------------------------------------------------------------------------
def action_design_ablation_8b():
    def _wrist_norm_kwargs(camera_mode: str) -> dict:
        return dict(
            repo_id=copy.deepcopy(_LIBERO_REPO_IDS),
            root=copy.deepcopy(_LIBERO_ROOTS),
            camera_mode=camera_mode,
            action_normalization="minmax",
            action_stats_path=ACTION_STATS_PATH,
            fps=20,
        )

    def _make_ablation_8b(
        exp_name: str,
        *,
        camera_mode: str = "wrist_image",
        load_path: str | None = None,
        keys_to_skip_loading: list[str] | None = None,
        dataset_overrides: dict | None = None,
        use_deterministic_seed: bool = False,
        batch_size: int = 256,
        num_workers: int = 24,
        lr: float = 1e-4,
        training_iterations: int = 20_000,
        warm_up_steps: int = 2000,
        grad_accum_iter: int = 1,
        max_samples_per_batch: int | None = 256,
        cfg_dropout_rate: float = 0.1,
        action_channel_masking: bool = True,
        action_loss_weight: float = 10.0,
        action_param_lr_multiplier: float = 5.0,
        shard_across_workers: bool | None = None,
    ) -> dict:
        ds_kwargs = _wrist_norm_kwargs(camera_mode)
        if dataset_overrides is not None:
            ds_kwargs.update(dataset_overrides)
        exp = _build_libero_policy_base_8b(
            exp_name=exp_name,
            training_iterations=training_iterations,
            batch_size=batch_size,
            num_workers=num_workers,
            job_group="action_libero",
            job_project="cosmos3_action_libero",
            mode="policy",
            keys_to_skip_loading=keys_to_skip_loading,
            use_deterministic_seed=use_deterministic_seed,
            **ds_kwargs,
        )
        exp["optimizer"]["lr"] = lr
        exp["optimizer"]["lr_multipliers"] = {
            "action2llm": action_param_lr_multiplier,
            "llm2action": action_param_lr_multiplier,
            "action_modality_embed": action_param_lr_multiplier,
        }
        exp["scheduler"]["f_min"] = [0.0]
        exp["scheduler"]["warm_up_steps"] = [warm_up_steps]
        exp["scheduler"]["cycle_lengths"] = [training_iterations]
        exp["trainer"]["max_iter"] = training_iterations
        exp["trainer"]["logging_iter"] = 100
        exp["trainer"]["run_validation"] = False
        exp["trainer"]["run_validation_on_start"] = False
        exp["trainer"]["grad_accum_iter"] = grad_accum_iter
        exp["model"]["config"]["rectified_flow_training_config"]["action_loss_weight"] = action_loss_weight
        ds_node = exp["dataloader_train"]["dataloaders"]["action_data"]["dataloader"]["dataset"]
        ds_node["cfg_dropout_rate"] = cfg_dropout_rate
        ds_node["action_channel_masking"] = action_channel_masking
        if shard_across_workers is not None:
            ds_node["shard_across_workers"] = shard_across_workers
        exp["dataloader_train"]["max_samples_per_batch"] = max_samples_per_batch
        exp["dataloader_train"]["max_sequence_length"] = None
        exp["checkpoint"]["load_from_object_store"] = dict(bucket="nv-00-10206-checkpoint-experiments")
        exp["checkpoint"]["save_to_object_store"] = dict(bucket="nv-00-10206-checkpoint-experiments")
        if load_path is not None:
            exp["checkpoint"]["load_path"] = load_path
        return exp

    cs.store(
        group="experiment",
        package="_global_",
        name="libero_exp_ablation_study_policy_w_action_lr_scaling_8b",
        node=_make_ablation_8b("libero_exp_ablation_study_policy_w_action_lr_scaling_8b"),
    )

    # bucket nv-00-10206-checkpoint-experiments matches the gcp default, so we
    # only need to override load_path. keys_to_skip_loading=[] preserves the
    # action layers that are already trained in the mid-training checkpoint.
    cs.store(
        group="experiment",
        package="_global_",
        name="libero_exp_ablation_study_policy_w_action_lr_scaling_8b_midtrain_v1p1",
        node=_make_ablation_8b(
            "libero_exp_ablation_study_policy_w_action_lr_scaling_8b_midtrain_v1p1",
            load_path=(
                "cosmos3_action/midtrain_v1p1/action_midtrain_exp005_v1p1_equalTokens/checkpoints/iter_000013000/"
            ),
            keys_to_skip_loading=[],
        ),
    )

    cs.store(
        group="experiment",
        package="_global_",
        name="libero_exp_ablation_study_policy_w_action_lr_scaling_8b_midtrain_v1p1_concat_view",
        node=_make_ablation_8b(
            "libero_exp_ablation_study_policy_w_action_lr_scaling_8b_midtrain_v1p1_concat_view",
            camera_mode="concat_view",
            load_path=(
                "cosmos3_action/midtrain_v1p1/action_midtrain_exp005_v1p1_equalTokens/checkpoints/iter_000013000/"
            ),
            keys_to_skip_loading=[],
        ),
    )

    cs.store(
        group="experiment",
        package="_global_",
        name="libero_exp_ablation_study_policy_w_action_lr_scaling_8b_exp506_midtraining_v1_concat_view_rot6d_native_no_rot_norm_quantile_det_seed_iter40k",
        node=_make_ablation_8b(
            "libero_exp_ablation_study_policy_w_action_lr_scaling_8b_exp506_midtraining_v1_concat_view_rot6d_native_no_rot_norm_quantile_det_seed_iter40k",
            camera_mode="concat_view",
            load_path=(
                "cosmos3_vfm/t2w_mot_8b_qwen3_vl_runs/"
                "t2w_mot_exp506_000_qwen3_vl_8b_multires_recipe_midtraining_v1/checkpoints/iter_000040000/"
            ),
            use_deterministic_seed=True,
            dataset_overrides={
                "action_space": "frame_wise_relative",
                "rotation_space": "6d",
                "pose_coordinate_frame": "native",
                "action_normalization": "quantile",
                "action_stats_path": NATIVE_FRAME_WISE_ROT6D_ACTION_STATS_PATH,
            },
        ),
    )

    cs.store(
        group="experiment",
        package="_global_",
        name="libero_exp_ablation_study_policy_w_action_lr_scaling_8b_exp506_midtraining_v1_concat_view_rot6d_native_rot_norm_quantile_det_seed_iter40k",
        node=_make_ablation_8b(
            "libero_exp_ablation_study_policy_w_action_lr_scaling_8b_exp506_midtraining_v1_concat_view_rot6d_native_rot_norm_quantile_det_seed_iter40k",
            camera_mode="concat_view",
            load_path=(
                "cosmos3_vfm/t2w_mot_8b_qwen3_vl_runs/"
                "t2w_mot_exp506_000_qwen3_vl_8b_multires_recipe_midtraining_v1/checkpoints/iter_000040000/"
            ),
            use_deterministic_seed=True,
            dataset_overrides={
                "action_space": "frame_wise_relative",
                "rotation_space": "6d",
                "pose_coordinate_frame": "native",
                "action_normalization": "quantile_rot",
                "action_stats_path": NATIVE_FRAME_WISE_ROT6D_ACTION_STATS_PATH,
            },
        ),
    )

    cs.store(
        group="experiment",
        package="_global_",
        name="libero_exp_ablation_study_policy_w_action_lr_scaling_8b_exp506_midtraining_v1_concat_view_rot6d_native_no_rot_norm_minmax_det_seed_iter40k",
        node=_make_ablation_8b(
            "libero_exp_ablation_study_policy_w_action_lr_scaling_8b_exp506_midtraining_v1_concat_view_rot6d_native_no_rot_norm_minmax_det_seed_iter40k",
            camera_mode="concat_view",
            load_path=(
                "cosmos3_vfm/t2w_mot_8b_qwen3_vl_runs/"
                "t2w_mot_exp506_000_qwen3_vl_8b_multires_recipe_midtraining_v1/checkpoints/iter_000040000/"
            ),
            use_deterministic_seed=True,
            dataset_overrides={
                "action_space": "frame_wise_relative",
                "rotation_space": "6d",
                "pose_coordinate_frame": "native",
                "action_normalization": "minmax",
                "action_stats_path": NATIVE_FRAME_WISE_ROT6D_ACTION_STATS_PATH,
            },
        ),
    )

    cs.store(
        group="experiment",
        package="_global_",
        name="libero_exp_ablation_study_policy_w_action_lr_scaling_8b_exp506_midtraining_v1_concat_view_rot6d_native_rot_norm_minmax_det_seed_iter40k",
        node=_make_ablation_8b(
            "libero_exp_ablation_study_policy_w_action_lr_scaling_8b_exp506_midtraining_v1_concat_view_rot6d_native_rot_norm_minmax_det_seed_iter40k",
            camera_mode="concat_view",
            load_path=(
                "cosmos3_vfm/t2w_mot_8b_qwen3_vl_runs/"
                "t2w_mot_exp506_000_qwen3_vl_8b_multires_recipe_midtraining_v1/checkpoints/iter_000040000/"
            ),
            use_deterministic_seed=True,
            dataset_overrides={
                "action_space": "frame_wise_relative",
                "rotation_space": "6d",
                "pose_coordinate_frame": "native",
                "action_normalization": "minmax",
                "action_stats_path": NATIVE_FRAME_WISE_ROT6D_ACTION_STATS_PATH,
            },
        ),
    )

    # Concatenated third-person + wrist view (camera_mode="concat_view" stitches the
    # two cameras horizontally inside LIBERODataset, producing a (H, 2*W) frame
    # tagged with viewpoint="concat_view").
    cs.store(
        group="experiment",
        package="_global_",
        name="libero_exp_ablation_study_policy_w_action_lr_scaling_8b_concat_view",
        node=_make_ablation_8b(
            "libero_exp_ablation_study_policy_w_action_lr_scaling_8b_concat_view",
            camera_mode="concat_view",
        ),
    )


def action_policy_sft_8b_experiments():
    """8B policy SFT on LIBERO-4 with frame-wise relative actions, 6D rotation, concat_view.

    Converted from cosmos3-internal configs/experiment/action_policy_sft_8b.yaml.

    API caveats vs the original YAML:
    - action_normalization='quantile_rot' is not in i4 (i4 only has normalize_action: bool).
      To restore full fidelity: add normalizers/libero_native_frame_wise_relative_rot6d.json,
      then set normalize_action=True + action_stats_path pointing to that file.
    - format_prompt_as_json and pose_coordinate_frame are cosmos3-internal-only; omitted.
    - load_path is a local outputs/ path from the original; override for cluster/S3 runs.
    """
    exp = _build_libero_policy_base_8b(
        "action_policy_sft_8b",
        training_iterations=16_000,
        batch_size=256,
        num_workers=4,
        job_group="action_libero",
        job_project="cosmos3_action_libero",
        mode="policy",
        keys_to_skip_loading=["net_ema.", "action2llm", "llm2action", "action_modality_embed", "action_pos_embed"],
        # LIBERODataset kwargs forwarded via **dataset_kwargs:
        repo_id=copy.deepcopy(_LIBERO_REPO_IDS),
        root=copy.deepcopy(_LIBERO_ROOTS),
        fps=20,
        camera_mode="concat_view",
        action_space="frame_wise_relative",
        rotation_space="6d",
        chunk_length=16,
        seed=0,
        val_ratio=0.01,
    )

    # Scheduler: f_max=1.0, warm_up=500; cycle_lengths=20000 intentionally longer than max_iter
    exp["scheduler"]["f_max"] = [1.0]
    exp["scheduler"]["f_min"] = [0.0]
    exp["scheduler"]["warm_up_steps"] = [500]
    exp["scheduler"]["cycle_lengths"] = [20_000]

    # Optimizer: lr=5e-5
    exp["optimizer"]["lr"] = 5e-5

    # Trainer
    exp["trainer"]["max_iter"] = 16_000
    exp["trainer"]["logging_iter"] = 100
    exp["trainer"]["run_validation"] = False
    exp["trainer"]["run_validation_on_start"] = False

    # Checkpoint: set to None by default; override via checkpoint.load_path= in the launch script
    # or set LIBERO_ACTION_CHECKPOINT_PATH env var to the pretrained checkpoint directory.
    exp["checkpoint"]["load_path"] = "outputs/checkpoints/action_policy_sft_8b"
    exp["checkpoint"]["load_training_state"] = False
    exp["checkpoint"]["save_iter"] = 500

    # compile_tokenizer: make_8b_experiment sets enabled=True but drops warmup_resolutions
    exp["trainer"]["callbacks"]["compile_tokenizer"]["warmup_resolutions"] = ["256", "480", "720"]

    # Model overrides
    exp["model"]["config"]["rectified_flow_training_config"]["action_loss_weight"] = 10.0
    exp["model"]["config"]["num_embodiment_domains"] = 32

    # wrap_dataset overrides
    ds_node = exp["dataloader_train"]["dataloaders"]["action_data"]["dataloader"]["dataset"]
    ds_node["action_channel_masking"] = True

    # Dataloader: disable sequence packing (max_sequence_length=None), cap batch budget
    exp["dataloader_train"]["max_samples_per_batch"] = 256
    exp["dataloader_train"]["max_sequence_length"] = None

    cs.store(group="experiment", package="_global_", name="action_policy_sft_8b", node=exp)


lr_sweep()
multi_view_policy_exp()
single_view_policy_exp()
action_camera_rotation_sweep()
normalized_action_policy_exp()
wrist_view_policy_exp()
normalized_action_joint_exp()
action_design_ablation()
action_design_ablation_8b()
action_policy_sft_8b_experiments()
