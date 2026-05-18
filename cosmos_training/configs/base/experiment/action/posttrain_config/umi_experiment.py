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

# UMI experiment — 8B multires pretrained base (modality-offset mRoPE)
#
# Uses make_8b_experiment() with multires overrides matching the
# _make_ablation_experiment pattern from mixed_res_dataset_ablations.py.
#
# Experiments:
#   joint_multires_8b_exp_umi4tasks_mrope          — base (25K iters)
#   joint_multires_8b_exp_umi4tasks_mrope_overfit4K — 4K iter overfit
#   ..._overfit4K_for_eval                          — 4K + val dataloader
#   joint_multires_8b_exp_umi4tasks_add3drope_overfit4K      — 3D RoPE, 4K
#   ..._add3drope_overfit4K_for_eval                         — 3D RoPE + val

import copy

from hydra.core.config_store import ConfigStore

from cosmos.utils.lazy_config import LazyCall as L
from configs.base.experiment.action.pretrained_config.cosmos3_8b import make_8b_experiment
from cosmos.data.vfm.action.dataloaders import InfiniteDataLoader
from cosmos.data.vfm.action.umi_dataset import get_umi_dataset
from cosmos.data.vfm.action.unified_dataset import dataset_entry, wrap_dataset
from cosmos.data.vfm.joint_dataloader import IterativeJointDataLoader

cs = ConfigStore.instance()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CHECKPOINT_MULTIRES_302_003 = (
    "cosmos3_vfm/t2w_mot_8b_qwen3_vl_runs/"
    "t2w_mot_exp302_003_qwen3_vl_8b_multires_modality_offset/"
    "checkpoints/iter_000006750/"
)

UMI_DATASET_NAMES = (
    "umi/cup_arrangement_0,"
    "data_scaling_law/towel_folding_0,"
    "data_scaling_law/mouse_arrangement_0,"
    "data_scaling_law/water_pouring_1"
)

# ---------------------------------------------------------------------------
# Dataset definition (shared across all experiments)
# ---------------------------------------------------------------------------
DATASET_UMI4TASKS = [
    L(dataset_entry)(
        name="umi",
        dataset=L(get_umi_dataset)(
            dataset_name=UMI_DATASET_NAMES,
            dataset_type="umi_multi_task",
            is_val=False,
            mode="joint",
        ),
        ratio=1,
    ),
]


# ---------------------------------------------------------------------------
# Multires overrides — transforms 480p single-res base into the 720p multires
# modality-offset configuration matching _make_ablation_experiment.
# ---------------------------------------------------------------------------
def _apply_multires_overrides(exp: dict) -> dict:
    """Apply 720p multires modality-offset overrides to a make_8b_experiment() config."""
    # Model: 720p multires with per-resolution shift
    exp["model"]["config"]["resolution"] = "720"
    exp["model"]["config"]["max_num_tokens_after_packing"] = 45056
    exp["model"]["config"]["rectified_flow_training_config"]["shift"] = {
        "256": 1,
        "480": 2,
        "720": 3,
    }

    # Diffusion expert: modality margin; drop single-res rope extrapolation fields
    diff_cfg = exp["model"]["config"]["diffusion_expert_config"]
    diff_cfg["unified_3d_mrope_temporal_modality_margin"] = 15000
    for key in [
        "rope_h_extrapolation_ratio",
        "rope_w_extrapolation_ratio",
        "rope_t_extrapolation_ratio",
        "max_vae_latent_side_after_patchify",
    ]:
        diff_cfg.pop(key, None)

    # Tokenizer: multires chunk/duration settings
    tok_cfg = exp["model"]["config"]["tokenizer"]
    tok_cfg["encode_chunk_frames"] = {"256": 68, "480": 24, "720": 12}
    tok_cfg["encode_exact_durations"] = [17, 73]

    # Optimizer: lower LR for multires
    exp["optimizer"]["lr"] = 1e-4

    # Scheduler: no warmup
    exp["scheduler"]["warm_up_steps"] = [0]

    # Checkpoint: exp302_003 multires pretrained, save every 1K
    exp["checkpoint"]["save_iter"] = 1000
    exp["checkpoint"]["load_path"] = CHECKPOINT_MULTIRES_302_003

    # Trainer: tokenizer compile warmup for 3 resolutions, higher recompile limit
    exp["trainer"]["callbacks"]["compile_tokenizer"] = dict(
        enabled=True,
        warmup_resolutions=["256", "480", "720"],
    )
    exp["trainer"]["compile_config"]["recompile_limit"] = 100

    return exp


# ---------------------------------------------------------------------------
# Val dataloader (used by *_for_eval variants)
# ---------------------------------------------------------------------------
def _make_val_dataloader() -> L:
    """Build the UMI val dataloader matching the original eval config."""
    return L(IterativeJointDataLoader)(
        dataloaders={
            "umi_data": dict(
                dataloader=L(InfiniteDataLoader)(
                    dataset=L(wrap_dataset)(
                        list_of_datasets=[
                            L(dataset_entry)(
                                name="umi",
                                dataset=L(get_umi_dataset)(
                                    dataset_name=UMI_DATASET_NAMES,
                                    dataset_type="umi_multi_task",
                                    is_val=True,
                                    mode="joint",
                                ),
                                ratio=1.0,
                            ),
                        ],
                        tokenizer_config="${model.config.vlm_config.tokenizer}",
                        cfg_dropout_rate=0.0,
                        max_action_dim="${model.config.max_action_dim}",
                    ),
                    batch_size=2,
                    num_workers=0,
                    pin_memory=True,
                    drop_last=True,
                    seed=0,
                ),
                ratio=1,
            ),
        },
        tokenizer_spatial_compression_factor="${model.config.tokenizer.spatial_compression_factor}",
        tokenizer_temporal_compression_factor="${model.config.tokenizer.temporal_compression_factor}",
        patch_spatial="${model.config.diffusion_expert_config.patch_spatial}",
        max_sequence_length="${model.config.max_num_tokens_after_packing}",
    )


# ---------------------------------------------------------------------------
# Base experiment — multires 8B UMI 4 tasks (mRoPE), 25K iters
# ---------------------------------------------------------------------------
joint_multires_8b_exp_umi4tasks_mrope = _apply_multires_overrides(
    make_8b_experiment(
        exp_name="joint_multires_8b_exp_umi4tasks_mrope",
        datasets=copy.deepcopy(DATASET_UMI4TASKS),
    ),
)

cs.store(
    group="experiment",
    package="_global_",
    name="joint_multires_8b_exp_umi4tasks_mrope",
    node=joint_multires_8b_exp_umi4tasks_mrope,
)


# ---------------------------------------------------------------------------
# Overfit 4K — mRoPE
# ---------------------------------------------------------------------------
joint_multires_8b_exp_umi4tasks_mrope_overfit4K = dict(
    defaults=[
        "/experiment/joint_multires_8b_exp_umi4tasks_mrope",
        "_self_",
    ],
    scheduler=dict(
        cycle_lengths=[4000],
    ),
    trainer=dict(
        max_iter=4000,
    ),
)

cs.store(
    group="experiment",
    package="_global_",
    name="joint_multires_8b_exp_umi4tasks_mrope_overfit4K",
    node=joint_multires_8b_exp_umi4tasks_mrope_overfit4K,
)


# ---------------------------------------------------------------------------
# Overfit 4K for eval — mRoPE + val dataloader
# ---------------------------------------------------------------------------
joint_multires_8b_exp_umi4tasks_mrope_overfit4K_for_eval = dict(
    defaults=[
        "/experiment/joint_multires_8b_exp_umi4tasks_mrope_overfit4K",
        "_self_",
    ],
    dataloader_val=_make_val_dataloader(),
)

cs.store(
    group="experiment",
    package="_global_",
    name="joint_multires_8b_exp_umi4tasks_mrope_overfit4K_for_eval",
    node=joint_multires_8b_exp_umi4tasks_mrope_overfit4K_for_eval,
)


# ---------------------------------------------------------------------------
# Overfit 4K — additive 3D RoPE (switches position embedding from mRoPE)
# ---------------------------------------------------------------------------
joint_multires_8b_exp_umi4tasks_add3drope_overfit4K = dict(
    defaults=[
        "/experiment/joint_multires_8b_exp_umi4tasks_mrope",
        "_self_",
    ],
    scheduler=dict(
        cycle_lengths=[4000],
    ),
    trainer=dict(
        max_iter=4000,
    ),
    model=dict(
        config=dict(
            diffusion_expert_config=dict(
                position_embedding_type="3d_rope",
            ),
        ),
    ),
    checkpoint=dict(
        keys_to_skip_loading=[
            "action2llm",
            "llm2action",
            "action_modality_embed",
            "action_pos_embed",
            "latent_pos_embed",
        ],
    ),
)

cs.store(
    group="experiment",
    package="_global_",
    name="joint_multires_8b_exp_umi4tasks_add3drope_overfit4K",
    node=joint_multires_8b_exp_umi4tasks_add3drope_overfit4K,
)


# ---------------------------------------------------------------------------
# Overfit 4K for eval — additive 3D RoPE + val dataloader
# ---------------------------------------------------------------------------
joint_multires_8b_exp_umi4tasks_add3drope_overfit4K_for_eval = dict(
    defaults=[
        "/experiment/joint_multires_8b_exp_umi4tasks_add3drope_overfit4K",
        "_self_",
    ],
    dataloader_val=_make_val_dataloader(),
)

cs.store(
    group="experiment",
    package="_global_",
    name="joint_multires_8b_exp_umi4tasks_add3drope_overfit4K_for_eval",
    node=joint_multires_8b_exp_umi4tasks_add3drope_overfit4K_for_eval,
)
