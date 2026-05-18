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

# AV policy experiment on the multires 8B model with modality-offset mRoPE.
#
# Built on make_8b_experiment() from cosmos3_8b.py with overrides for:
# - 720p multires (256/480/720) with per-resolution shift
# - Modality-offset mRoPE (temporal_modality_margin=15000)
# - exp302_003 checkpoint (multires modality offset pretrained)
# - Extended tokenizer warmup for multi-resolution

import copy

from hydra.core.config_store import ConfigStore

from cosmos.utils.lazy_config import LazyCall as L
from cosmos.utils.lazy_config import LazyDict
from configs.base.experiment.action.pretrained_config.cosmos3_8b import make_8b_experiment
from cosmos.data.vfm.action.av_dataset import AVDataset
from cosmos.data.vfm.action.unified_dataset import dataset_entry

cs = ConfigStore.instance()


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
AV_DATASET_I2V = [
    L(dataset_entry)(
        name="av",
        dataset=L(AVDataset)(
            split="train",
            fps=10,
            mode="image2video",
            history_len=0.1,
            future_len=6.0,
            rotation_format="rot6d",
            pose_convention="backward_anchored",
            credential_path="${job.cluster.object_store_credential_data}",
            shuffle=True,
            translation_scale=0.01,
        ),
        ratio=6,
    ),
]

# ---------------------------------------------------------------------------
# Base experiment — i2v mode
# ---------------------------------------------------------------------------
av_exp302_003_qwen3_vl_8b_multires_modality_offset_i2v = make_8b_experiment(
    exp_name="av_exp302_003_qwen3_vl_8b_multires_modality_offset_i2v",
    datasets=AV_DATASET_I2V,
    training_iterations=25_000,
)

# -- Override job --
av_exp302_003_qwen3_vl_8b_multires_modality_offset_i2v["job"]["group"] = "action_av"
av_exp302_003_qwen3_vl_8b_multires_modality_offset_i2v["job"]["name"] = (
    "av_exp302_003_qwen3_vl_8b_multires_modality_offset_i2v"
)

# -- Override model for multires 720p --
av_exp302_003_qwen3_vl_8b_multires_modality_offset_i2v["model"]["config"]["resolution"] = "720"
av_exp302_003_qwen3_vl_8b_multires_modality_offset_i2v["model"]["config"]["max_num_tokens_after_packing"] = 45056
av_exp302_003_qwen3_vl_8b_multires_modality_offset_i2v["model"]["config"]["rectified_flow_training_config"]["shift"] = {
    "256": 1,
    "480": 2,
    "720": 3,
}
av_exp302_003_qwen3_vl_8b_multires_modality_offset_i2v["model"]["config"]["diffusion_expert_config"][
    "unified_3d_mrope_temporal_modality_margin"
] = 15000
# Remove extrapolation ratios and max_vae_latent_side (not in original multires config)
for _key in [
    "rope_h_extrapolation_ratio",
    "rope_w_extrapolation_ratio",
    "rope_t_extrapolation_ratio",
    "max_vae_latent_side_after_patchify",
]:
    av_exp302_003_qwen3_vl_8b_multires_modality_offset_i2v["model"]["config"]["diffusion_expert_config"].pop(_key, None)

# -- Override tokenizer --
av_exp302_003_qwen3_vl_8b_multires_modality_offset_i2v["model"]["config"]["tokenizer"]["encode_chunk_frames"] = {
    "256": 68,
    "480": 24,
    "720": 12,
}
# Remove fields not in original
for _key in ["encode_exact_durations"]:
    av_exp302_003_qwen3_vl_8b_multires_modality_offset_i2v["model"]["config"]["tokenizer"].pop(_key, None)

# -- Override optimizer --
av_exp302_003_qwen3_vl_8b_multires_modality_offset_i2v["optimizer"]["lr"] = 1e-4

# -- Override scheduler --
av_exp302_003_qwen3_vl_8b_multires_modality_offset_i2v["scheduler"]["warm_up_steps"] = [0]
av_exp302_003_qwen3_vl_8b_multires_modality_offset_i2v["scheduler"]["cycle_lengths"] = [200_000]

# -- Override trainer --
av_exp302_003_qwen3_vl_8b_multires_modality_offset_i2v["trainer"]["logging_iter"] = 50
av_exp302_003_qwen3_vl_8b_multires_modality_offset_i2v["trainer"]["callbacks"]["norm_monitor"] = dict(every_n=100)
av_exp302_003_qwen3_vl_8b_multires_modality_offset_i2v["trainer"]["callbacks"]["sigma_loss_analysis"] = dict(
    every_n=500,
    every_n_viz=500,
)
av_exp302_003_qwen3_vl_8b_multires_modality_offset_i2v["trainer"]["callbacks"]["compile_tokenizer"] = dict(
    enabled=True,
    warmup_resolutions=["256", "480", "720"],
)
av_exp302_003_qwen3_vl_8b_multires_modality_offset_i2v["trainer"]["compile_config"]["recompile_limit"] = 100

# -- Override checkpoint (exp302_003 multires) --
av_exp302_003_qwen3_vl_8b_multires_modality_offset_i2v["checkpoint"]["save_iter"] = 250
av_exp302_003_qwen3_vl_8b_multires_modality_offset_i2v["checkpoint"]["load_path"] = (
    "cosmos3_vfm/t2w_mot_8b_qwen3_vl_runs/"
    "t2w_mot_exp302_003_qwen3_vl_8b_multires_modality_offset/"
    "checkpoints/iter_000006750/"
)
av_exp302_003_qwen3_vl_8b_multires_modality_offset_i2v["checkpoint"]["load_from_object_store"] = dict(enabled=True)

# Remove vlm_config.pretrained_weights.enabled (not in original — inherits default)
av_exp302_003_qwen3_vl_8b_multires_modality_offset_i2v["model"]["config"]["vlm_config"]["pretrained_weights"].pop(
    "enabled", None
)

# Wrap in LazyDict for allow_objects support
av_exp302_003_qwen3_vl_8b_multires_modality_offset_i2v = LazyDict(
    av_exp302_003_qwen3_vl_8b_multires_modality_offset_i2v,
    flags={"allow_objects": True},
)

cs.store(
    group="experiment",
    package="_global_",
    name="av_exp302_003_qwen3_vl_8b_multires_modality_offset_i2v",
    node=av_exp302_003_qwen3_vl_8b_multires_modality_offset_i2v,
)

# ---------------------------------------------------------------------------
# Policy variant
# ---------------------------------------------------------------------------
AV_DATASET_POLICY = copy.deepcopy(AV_DATASET_I2V)
AV_DATASET_POLICY[0]["dataset"]["mode"] = "policy"

av_exp302_003_qwen3_vl_8b_multires_modality_offset_policy = LazyDict(
    dict(
        defaults=[
            "/experiment/av_exp302_003_qwen3_vl_8b_multires_modality_offset_i2v",
            "_self_",
        ],
        job=dict(
            group="action_av",
            name="av_exp302_003_qwen3_vl_8b_multires_modality_offset_policy",
        ),
        dataloader_train=dict(
            dataloaders=dict(
                action_data=dict(
                    dataloader=dict(
                        dataset=dict(
                            list_of_datasets=[
                                L(dataset_entry)(
                                    name="av",
                                    dataset=L(AVDataset)(
                                        split="train",
                                        fps=10,
                                        mode="policy",
                                        history_len=0.1,
                                        future_len=6.0,
                                        rotation_format="rot6d",
                                        pose_convention="backward_anchored",
                                        credential_path="${job.cluster.object_store_credential_data}",
                                        shuffle=True,
                                        translation_scale=0.01,
                                    ),
                                    ratio=6,
                                ),
                            ],
                        ),
                    ),
                ),
            ),
        ),
    ),
    flags={"allow_objects": True},
)

cs.store(
    group="experiment",
    package="_global_",
    name="av_exp302_003_qwen3_vl_8b_multires_modality_offset_policy",
    node=av_exp302_003_qwen3_vl_8b_multires_modality_offset_policy,
)
