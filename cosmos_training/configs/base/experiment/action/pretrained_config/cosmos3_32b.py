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

import copy

from hydra.core.config_store import ConfigStore

from cosmos.utils.lazy_config import LazyCall as L
from cosmos.utils import log
from cosmos.data.vfm.action.dataloaders import InfiniteDataLoader
from cosmos.data.vfm.action.unified_dataset import wrap_dataset
from cosmos.data.vfm.joint_dataloader import IterativeJointDataLoader

PRETRAINED_CHECKPOINT = (
    "cosmos3_vfm/t2w_mot_32b_qwen3_vl_runs/t2w_mot_exp305_000_qwen3_vl_32b_multires_v7/checkpoints/iter_000085000/"
)


_DEFAULT_KEYS_TO_SKIP = ["action2llm", "llm2action", "action_modality_embed", "action_pos_embed"]


def make_32b_experiment(
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
    extra_dataloaders: dict[str, dict] | None = None,
    keys_to_skip_loading: list[str] = _DEFAULT_KEYS_TO_SKIP,
) -> dict:
    """ """
    if keys_to_skip_loading:
        log.warning(
            f"keys_to_skip_loading={keys_to_skip_loading} — action layers will NOT be loaded from the "
            "checkpoint. Set keys_to_skip_loading=[] when resuming from an action "
            "checkpoint to avoid dropping trained action layers."
        )

    dataloaders: dict[str, dict] = {}
    if extra_dataloaders is not None:
        dataloaders.update(extra_dataloaders)
    dataloaders["action_data"] = dict(
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
            {"override /vlm_config": "qwen3_vl_mot_vlm_32b_instruct_gcp"},
            {"override /checkpoint": "gcp"},
            {
                "override /callbacks": [
                    "basic",
                    "optimization",
                    "job_monitor",
                    "training_stats",
                ]
            },
            "_self_",
        ],
        job=dict(
            project="cosmos3_action",
            group="cosmos3_action_32b_pretrained",
            name=exp_name,
        ),
        model=dict(
            config=dict(
                action_gen=True,
                max_action_dim=max_action_dim,
                joint_attn_implementation="two_way",
                resolution="720",
                state_ch=48,
                latent_downsample_factor=16,
                state_t=300,
                max_num_tokens_after_packing=45056,
                rectified_flow_training_config=dict(
                    train_time_weight="uniform",
                    train_time_video_distribution="waver",
                    loss_scale=10.0,
                    use_discrete_rf=False,
                    shift={"256": 1, "480": 2, "720": 3},
                ),
                diffusion_expert_config=dict(
                    patch_spatial=2,
                    position_embedding_type="unified_3d_mrope",
                    unified_3d_mrope_reset_spatial_ids=True,
                    unified_3d_mrope_temporal_modality_margin=15000,
                    enable_fps_modulation=True,
                    base_fps=24,
                ),
                tokenizer=dict(
                    bucket_name="${job.cluster.object_store_bucket_pretrained}",
                    object_store_credential_path_pretrained="${job.cluster.object_store_credential_pretrained}",
                    encode_chunk_frames={"256": 68, "480": 24, "720": 12},
                    encode_exact_durations=[17, 61, 73],
                ),
                parallelism=dict(
                    data_parallel_shard_degree=64,
                    use_activation_checkpointing=True,
                    use_torch_compile=True,
                ),
                vlm_config=dict(
                    use_system_prompt=False,
                    load_pretrained=True,
                ),
            ),
        ),
        optimizer=dict(
            lr=1e-4,
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
            f_max=[0.95],
            f_min=[0.3],
            warm_up_steps=[0],
            cycle_lengths=[training_iterations],
        ),
        trainer=dict(
            max_iter=training_iterations,
            logging_iter=50,
            callbacks=dict(
                grad_clip=dict(clip_norm=1.0),
                manual_gc=dict(every_n=200),
                compile_tokenizer=dict(enabled=True),
                straggler_detection=dict(enabled=True, report_freq=50),
                norm_monitor=dict(every_n=100),
                sigma_loss_analysis=dict(every_n=500, every_n_viz=500),
            ),
            compile_config=dict(recompile_limit=100, use_duck_shape=False),
        ),
        checkpoint=dict(
            save_iter=500,
            load_path=PRETRAINED_CHECKPOINT,
            load_training_state=False,
            strict_resume=True,
            keys_to_skip_loading=keys_to_skip_loading,
        ),
        dataloader_train=L(IterativeJointDataLoader)(
            dataloaders=dataloaders,
            tokenizer_spatial_compression_factor="${model.config.tokenizer.spatial_compression_factor}",
            tokenizer_temporal_compression_factor="${model.config.tokenizer.temporal_compression_factor}",
            patch_spatial="${model.config.diffusion_expert_config.patch_spatial}",
            max_sequence_length="${model.config.max_num_tokens_after_packing}",
        ),
    )


cosmos3_pretrained_32b = make_32b_experiment("cosmos3_pretrained_32b", datasets=[])
cosmos3_pretrained_32b["job"]["group"] = "pretrained_config"
cosmos3_pretrained_32b["job"]["name"] = "cosmos3_pretrained_32b"

cs = ConfigStore.instance()
cs.store(
    group="experiment",
    package="_global_",
    name="cosmos3_pretrained_32b",
    node=cosmos3_pretrained_32b,
)
