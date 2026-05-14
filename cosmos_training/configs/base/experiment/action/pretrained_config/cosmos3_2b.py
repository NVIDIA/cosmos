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

"""Centralized Cosmos3 VFM 2B pretrained model config for Action experiments.

Provides make_2b_experiment(), a factory that builds complete experiment configs
from the 2B mrope pretrained base (480p single-res).

Usage — downstream experiment files just pass datasets::

    from configs.base.experiment.action.pretrained_config.cosmos3_2b import (
        make_2b_experiment,
    )

    exp = make_2b_experiment("my_exp", [DATASET_BRIDGE, DATASET_UMI])
    cs.store(group="experiment", package="_global_", name="my_exp", node=exp)

    # Override specific fields after creation:
    exp["scheduler"]["cycle_lengths"] = [4000]
    exp["trainer"]["max_iter"] = 4000

The raw ``cosmos3_pretrained_2b`` dict is also registered in ConfigStore for
Hydra defaults inheritance (``defaults=["/experiment/cosmos3_pretrained_2b", "_self_"]``).
"""

import copy

from hydra.core.config_store import ConfigStore

from cosmos.utils.lazy_config import LazyCall as L
from cosmos.utils import log
from cosmos.data.vfm.action.dataloaders import InfiniteDataLoader
from cosmos.data.vfm.action.unified_dataset import wrap_dataset
from cosmos.data.vfm.joint_dataloader import IterativeJointDataLoader

PRETRAINED_CHECKPOINT = (
    "cosmos3_vfm/t2w_mot_2b_qwen3_vl_runs/"
    "t2w_mot_dryrun_exp202_001_qwen3_vl_2b_480res_qwen3_captions_v4_300_frames_mrope/"
    "checkpoints/iter_000026000/"
)


_DEFAULT_KEYS_TO_SKIP = ["action2llm", "llm2action", "action_modality_embed", "action_pos_embed"]


def make_2b_experiment(
    exp_name: str,
    datasets: list,
    *,
    batch_size: int = 4,
    num_workers: int = 16,
    training_iterations: int = 25_000,
    use_deterministic_seed: bool = False,
    action_data_ratio: int = 1,
    max_action_dim: int = 64,
    action_param_lr_multipliers: float = 5.0,
    keys_to_skip_loading: list[str] = _DEFAULT_KEYS_TO_SKIP,
) -> dict:
    """Build a Cosmos3 2B Action experiment config with the given dataset subset."""
    if keys_to_skip_loading:
        log.warning(
            f"keys_to_skip_loading={keys_to_skip_loading} — action layers will NOT be loaded from the "
            "checkpoint. Set keys_to_skip_loading=[] when resuming from an action "
            "checkpoint to avoid dropping trained action layers."
        )
    return dict(
        defaults=[
            {"override /data_train": None},
            {"override /data_val": None},
            {"override /model": "mot_fsdp"},
            {"override /optimizer": "fusedadamw"},
            {"override /scheduler": "lambdalinear"},
            {"override /tokenizer": "wan2pt2_tokenizer"},
            {"override /cluster": "gcp_iad_gb200"},
            {"override /vlm_config": "qwen3_vl_mot_vlm_2b_instruct_gcp"},
            {"override /checkpoint": "gcp"},
            {
                "override /callbacks": [
                    "basic",
                    "optimization",
                    "job_monitor",
                    # "training_stats",
                ]
            },
            "_self_",
        ],
        job=dict(
            project="cosmos3_action",
            group="cosmos3_action_2b_pretrained",
            name=exp_name,
        ),
        model=dict(
            config=dict(
                action_gen=True,
                max_action_dim=max_action_dim,
                joint_attn_implementation="two_way",
                resolution="480",
                state_ch=48,
                latent_downsample_factor=16,
                state_t=300,
                max_num_tokens_after_packing=30000,
                rectified_flow_training_config=dict(
                    train_time_weight="uniform",
                    loss_scale=10.0,
                    use_discrete_rf=False,
                    shift=2,
                ),
                diffusion_expert_config=dict(
                    patch_spatial=2,
                    position_embedding_type="unified_3d_mrope",
                    unified_3d_mrope_reset_spatial_ids=True,
                    rope_h_extrapolation_ratio=2.0,
                    rope_w_extrapolation_ratio=2.0,
                    rope_t_extrapolation_ratio=1.0,
                    max_vae_latent_side_after_patchify=52,
                    enable_fps_modulation=True,
                    base_fps=24,
                ),
                tokenizer=dict(
                    bucket_name="${job.cluster.object_store_bucket_pretrained}",
                    object_store_credential_path_pretrained="${job.cluster.object_store_credential_pretrained}",
                    encode_chunk_frames={"256": 68, "480": 24, "720": 12},
                    encode_exact_durations=[17, 61, 73, 93],
                ),
                parallelism=dict(
                    use_activation_checkpointing=True,
                    use_torch_compile=True,
                    data_parallel_shard_degree=8,
                ),
                vlm_config=dict(
                    use_system_prompt=False,
                    pretrained_weights=dict(enabled=True),
                ),
            ),
        ),
        optimizer=dict(
            lr=2e-4,
            betas=[0.9, 0.99],
            weight_decay=0.05,
            keys_to_select=[
                "moe_gen",
                "time_embedder",
                "vae2llm",
                "llm2vae",
                "action2llm",
                "llm2action",
                "action_modality_embed",
            ],
            lr_multipliers={
                "action2llm": action_param_lr_multipliers,
                "llm2action": action_param_lr_multipliers,
                "action_modality_embed": action_param_lr_multipliers,
            },
        ),
        scheduler=dict(
            f_max=[0.5],
            f_min=[0.2],
            warm_up_steps=[2_000],
            cycle_lengths=[training_iterations],
        ),
        trainer=dict(
            max_iter=training_iterations,
            logging_iter=200,
            callbacks=dict(
                grad_clip=dict(clip_norm=1.0),
                manual_gc=dict(every_n=200),
                compile_tokenizer=dict(enabled=True, warmup_resolutions=["256", "480", "720"]),
                straggler_detection=dict(enabled=True, report_freq=50),
            ),
            compile_config=dict(recompile_limit=32, use_duck_shape=False),
        ),
        checkpoint=dict(
            save_iter=500,
            load_path=PRETRAINED_CHECKPOINT,
            load_training_state=False,
            strict_resume=True,
            keys_to_skip_loading=keys_to_skip_loading,
        ),
        dataloader_train=L(IterativeJointDataLoader)(
            dataloaders={
                "action_data": dict(
                    dataloader=L(InfiniteDataLoader)(
                        dataset=L(wrap_dataset)(
                            list_of_datasets=copy.deepcopy(datasets),
                            tokenizer_config="${model.config.vlm_config.tokenizer}",
                            cfg_dropout_rate=0.1,
                            max_action_dim="${model.config.max_action_dim}",
                            shard_across_workers=True,
                            append_duration_fps_timestamps=True,
                            append_resolution_info=True,
                        ),
                        batch_size=batch_size,
                        num_workers=num_workers,
                        pin_memory=True,
                        use_deterministic_seed=use_deterministic_seed,
                        in_order=False,
                        multiprocessing_context="spawn",
                    ),
                    ratio=action_data_ratio,
                ),
            },
            tokenizer_spatial_compression_factor="${model.config.tokenizer.spatial_compression_factor}",
            tokenizer_temporal_compression_factor="${model.config.tokenizer.temporal_compression_factor}",
            patch_spatial="${model.config.diffusion_expert_config.patch_spatial}",
            max_sequence_length="${model.config.max_num_tokens_after_packing}",
        ),
    )


# ---------------------------------------------------------------------------
# Hydra defaults inheritance config generated from make_2b_experiment().
# ---------------------------------------------------------------------------
cosmos3_pretrained_2b = make_2b_experiment("cosmos3_pretrained_2b", datasets=[])
cosmos3_pretrained_2b["job"]["group"] = "pretrained_config"
cosmos3_pretrained_2b["job"]["name"] = "cosmos3_pretrained_2b"

cs = ConfigStore.instance()
cs.store(
    group="experiment",
    package="_global_",
    name="cosmos3_pretrained_2b",
    node=cosmos3_pretrained_2b,
)
