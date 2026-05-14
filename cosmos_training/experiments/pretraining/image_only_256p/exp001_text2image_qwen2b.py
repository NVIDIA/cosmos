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

from hydra.core.config_store import ConfigStore

from cosmos.utils.lazy_config import LazyCall as L
from cosmos.utils.lazy_config import LazyDict
from cosmos.data.vfm.cached_replay_dataloader import get_cached_replay_dataloader
from cosmos.data.vfm.dataset_provider import get_image_dataset
from cosmos.data.vfm.joint_dataloader import IterativeJointDataLoader

cs = ConfigStore.instance()

"""
torchrun --nproc_per_node=4 --master_port=12341 -m scripts/train.py --config=configs/base/config.py  -- experiment=t2i_mot_exp000_000_qwen0p5b_256res_frozen_llm trainer.callbacks.generation.every_n=10
"""

t2i_mot_exp001_000_qwen1p7b_256res_frozen_llm = dict(
    defaults=[
        {"override /data_train": None},
        {"override /data_val": None},
        {"override /model": "mot_fsdp"},
        {"override /optimizer": "fusedadamw"},
        {"override /scheduler": "lambdalinear"},
        {"override /tokenizer": "wan2pt2_tokenizer"},
        {"override /cluster": "aws_iad_h100"},
        {"override /vlm_config": "qwen3_mot_1p7b"},
        {
            "override /callbacks": [
                "basic",
                "optimization",
                "job_monitor",
                "viz_online_sampling",
            ]
        },
        "_self_",
    ],
    trainer=dict(
        max_iter=1_000_000,
        logging_iter=200,
        callbacks=dict(
            grad_clip=dict(
                clip_norm=1.0,
            ),
            manual_gc=dict(
                every_n=200,
            ),
            every_n_sample_reg=dict(
                every_n=2_500,
            ),
            every_n_sample_ema=dict(
                every_n=2_500,
            ),
        ),
        compile_config=dict(
            recompile_limit=32,
            use_duck_shape=False,
        ),
    ),
    model=dict(
        config=dict(
            resolution="256",
            max_num_tokens_after_packing=8192,
            rectified_flow_training_config=dict(
                train_time_weight="uniform",
                loss_scale=10.0,
                use_discrete_rf=False,
                shift=1,
            ),
            state_ch=48,
            latent_downsample_factor=16,
            diffusion_expert_config=dict(
                patch_spatial=1,
                position_embedding_type="flattened_sin_cos",
            ),
            tokenizer=dict(
                bucket_name="${job.cluster.object_store_bucket_pretrained}",
                object_store_credential_path_pretrained="${job.cluster.object_store_credential_pretrained}",
            ),
            parallelism=dict(
                data_parallel_shard_degree=32,
                use_activation_checkpointing=True,
                use_torch_compile=True,
            ),
            vlm_config=dict(
                use_system_prompt=True,
            ),
        ),
    ),
    optimizer=dict(
        lr=1e-3,
        betas=[0.9, 0.99],
        weight_decay=0.05,
        keys_to_select=["moe_gen", "q_norm", "k_norm", "time_embedder", "vae2llm", "llm2vae", "latent_pos_embed"],
    ),
    job=dict(
        group="t2i_mot_1p7b_qwen3_ablations_small_scale",
        name="t2i_mot_exp001_000_qwen1p7b_256res_frozen_llm",
    ),
    scheduler=dict(
        f_max=[1.0],
        f_min=[0.3],
        warm_up_steps=[2_000],
        cycle_lengths=[1_000_000],
    ),
    checkpoint=dict(
        save_iter=2_500,
    ),
    dataloader_train=L(IterativeJointDataLoader)(
        dataloaders={
            "image_data_qwen_2p5_7b_v4_captions": dict(
                dataloader=L(get_cached_replay_dataloader)(
                    dataset=L(get_image_dataset)(
                        dataset_name="cosmos_pretrain_and_synthetic_photoreal_20250805_image_whole",
                        object_store="s3",
                        resolution="256",
                        is_train=True,
                        caption_type="qwen2p5_7b_v4",
                        dataset_resolution_type="all",
                        embedding_type=None,
                        augmentor_name="image_basic_augmentor_with_tokenization",
                        tokenizer_config="${model.config.vlm_config.tokenizer}",
                        cfg_dropout_rate=0.1,
                    ),
                    batch_size=32,
                    num_workers=16,
                    prefetch_factor=4,
                    sampler=None,
                    persistent_workers=False,
                    pin_memory=True,
                    cache_replay_name="image_dataloader",
                ),
                ratio=1,
            ),
            "image_data_prompt_captions": dict(
                dataloader=L(get_cached_replay_dataloader)(
                    dataset=L(get_image_dataset)(
                        dataset_name="cosmos_synthetic_filtered_combined_20250805_image_whole",
                        object_store="s3",
                        resolution="256",
                        is_train=True,
                        caption_type="prompts",
                        dataset_resolution_type="all",
                        embedding_type=None,
                        augmentor_name="image_basic_augmentor_with_tokenization",
                        tokenizer_config="${model.config.vlm_config.tokenizer}",
                        cfg_dropout_rate=0.1,
                    ),
                    batch_size=32,
                    num_workers=16,
                    prefetch_factor=4,
                    sampler=None,
                    persistent_workers=False,
                    pin_memory=True,
                    cache_replay_name="image_dataloader",
                ),
                ratio=1,
            ),
        },
        tokenizer_spatial_compression_factor="${model.config.tokenizer.spatial_compression_factor}",
        tokenizer_temporal_compression_factor="${model.config.tokenizer.temporal_compression_factor}",
        patch_spatial="${model.config.diffusion_expert_config.patch_spatial}",
        max_sequence_length="${model.config.max_num_tokens_after_packing}",
    ),
)

t2i_mot_exp001_001_qwen1p7b_256res_frozen_llm_lr_1e4 = LazyDict(
    dict(
        defaults=[
            f"/experiment/{t2i_mot_exp001_000_qwen1p7b_256res_frozen_llm['job']['name']}",
            {"override /tokenizer": "wan2pt2_tokenizer"},
            "_self_",
        ],
        job=dict(
            group="cosmos3_ablations",
            name="t2i_mot_exp001_001_qwen1p7b_256res_frozen_llm_lr_1e4",
        ),
        optimizer=dict(
            lr=1e-4,
        ),
    ),
    flags={"allow_objects": True},
)

t2i_mot_exp001_002_qwen1p7b_256res_frozen_llm_lr_3e3 = LazyDict(
    dict(
        defaults=[
            f"/experiment/{t2i_mot_exp001_000_qwen1p7b_256res_frozen_llm['job']['name']}",
            {"override /tokenizer": "wan2pt2_tokenizer"},
            "_self_",
        ],
        job=dict(
            group="cosmos3_ablations",
            name="t2i_mot_exp001_001_qwen1p7b_256res_frozen_llm_lr_1e4",
        ),
        optimizer=dict(
            lr=3e-3,
        ),
    ),
    flags={"allow_objects": True},
)


t2i_mot_exp001_003_qwen1p7b_256res_frozen_llm_lr_1e4_256nodes_gcp = LazyDict(
    dict(
        defaults=[
            f"/experiment/{t2i_mot_exp001_000_qwen1p7b_256res_frozen_llm['job']['name']}",
            {"override /tokenizer": "wan2pt1_tokenizer"},
            {"override /vlm_config": "qwen3_mot_1p7b_gcp"},
            {"override /cluster": "gcp_iad_gb200"},
            {"override /checkpoint": "gcp"},
            "_self_",
        ],
        job=dict(
            group="cosmos3_ablations",
            name="t2i_mot_exp001_003_qwen1p7b_256res_frozen_llm_lr_1e4_256nodes_gcp",
        ),
        optimizer=dict(
            lr=1e-4,
        ),
        model=dict(
            config=dict(
                diffusion_expert_config=dict(
                    patch_spatial=2,
                    position_embedding_type="3d_rope",
                ),
                state_ch=16,
                latent_downsample_factor=8,
            ),
        ),
        dataloader_train=dict(
            dataloaders=dict(
                image_data_qwen_2p5_7b_v4_captions=dict(
                    dataloader=dict(
                        dataset=dict(object_store="gcp"),
                    )
                ),
                image_data_prompt_captions=dict(
                    dataloader=dict(
                        dataset=dict(object_store="gcp"),
                    )
                ),
            ),
        ),
        checkpoint=dict(
            save_iter=2_000,
        ),
    ),
    flags={"allow_objects": True},
)

t2i_mot_exp001_004_qwen1p7b_256res_frozen_llm_lr_4e4_gcp = LazyDict(
    dict(
        defaults=[
            f"/experiment/{t2i_mot_exp001_003_qwen1p7b_256res_frozen_llm_lr_1e4_256nodes_gcp['job']['name']}",
            {"override /tokenizer": "wan2pt2_tokenizer"},
            "_self_",
        ],
        job=dict(
            group="cosmos3_ablations",
            name="t2i_mot_exp001_004_qwen1p7b_256res_frozen_llm_lr_4e4_gcp",
        ),
        optimizer=dict(
            lr=4e-4,
        ),
        model=dict(
            config=dict(
                diffusion_expert_config=dict(
                    patch_spatial=1,
                    position_embedding_type="flattened_sin_cos",
                ),
                state_ch=48,
                latent_downsample_factor=16,
            ),
        ),
    ),
    flags={"allow_objects": True},
)

t2i_mot_exp001_005_qwen1p7b_256res_frozen_llm_lr_6e5_gcp = LazyDict(
    dict(
        defaults=[
            f"/experiment/{t2i_mot_exp001_003_qwen1p7b_256res_frozen_llm_lr_1e4_256nodes_gcp['job']['name']}",
            {"override /tokenizer": "wan2pt2_tokenizer"},
            "_self_",
        ],
        job=dict(
            group="cosmos3_ablations",
            name="t2i_mot_exp001_005_qwen1p7b_256res_frozen_llm_lr_6e5_gcp",
        ),
        optimizer=dict(
            lr=6e-5,
        ),
        model=dict(
            config=dict(
                diffusion_expert_config=dict(
                    patch_spatial=1,
                    position_embedding_type="flattened_sin_cos",
                ),
                state_ch=48,
                latent_downsample_factor=16,
            ),
        ),
    ),
    flags={"allow_objects": True},
)

t2i_mot_exp001_006_qwen1p7b_256res_frozen_llm_lr_2e5_gcp = LazyDict(
    dict(
        defaults=[
            f"/experiment/{t2i_mot_exp001_003_qwen1p7b_256res_frozen_llm_lr_1e4_256nodes_gcp['job']['name']}",
            {"override /tokenizer": "wan2pt2_tokenizer"},
            "_self_",
        ],
        job=dict(
            group="cosmos3_ablations",
            name="t2i_mot_exp001_006_qwen1p7b_256res_frozen_llm_lr_2e5_gcp",
        ),
        optimizer=dict(
            lr=2e-5,
        ),
        model=dict(
            config=dict(
                diffusion_expert_config=dict(
                    patch_spatial=1,
                    position_embedding_type="flattened_sin_cos",
                ),
                state_ch=48,
                latent_downsample_factor=16,
            ),
        ),
    ),
    flags={"allow_objects": True},
)

t2i_mot_exp001_007_qwen1p7b_256res_frozen_llm_lr_1e4_wd_0p01_gcp = LazyDict(
    dict(
        defaults=[
            f"/experiment/{t2i_mot_exp001_003_qwen1p7b_256res_frozen_llm_lr_1e4_256nodes_gcp['job']['name']}",
            {"override /tokenizer": "wan2pt2_tokenizer"},
            "_self_",
        ],
        job=dict(
            group="cosmos3_ablations",
            name="t2i_mot_exp001_007_qwen1p7b_256res_frozen_llm_lr_1e4_wd_0p01_gcp",
        ),
        optimizer=dict(
            lr=1e-4,
            weight_decay=0.01,
        ),
        model=dict(
            config=dict(
                diffusion_expert_config=dict(
                    patch_spatial=1,
                    position_embedding_type="flattened_sin_cos",
                ),
                state_ch=48,
                latent_downsample_factor=16,
            ),
        ),
        dataloader_train=dict(
            tokenizer_spatial_compression_factor=16,
        ),
    ),
    flags={"allow_objects": True},
)

t2i_mot_exp001_008_qwen1p7b_256res_frozen_llm_lr_4e4_256nodes_gcp = LazyDict(
    dict(
        defaults=[
            f"/experiment/{t2i_mot_exp001_003_qwen1p7b_256res_frozen_llm_lr_1e4_256nodes_gcp['job']['name']}",
            "_self_",
        ],
        job=dict(
            group="cosmos3_ablations",
            name="t2i_mot_exp001_008_qwen1p7b_256res_frozen_llm_lr_4e4_256nodes_gcp",
        ),
        optimizer=dict(
            lr=4e-4,
        ),
    ),
    flags={"allow_objects": True},
)

# Use 2B VL model
# We load this checkpoint from t2i_mot_exp001_003_qwen1p7b_256res_frozen_llm_lr_1e4_256nodes_gcp
t2i_mot_exp001_009_qwen3_vl_2b_256res_frozen_llm = LazyDict(
    dict(
        defaults=[
            f"/experiment/{t2i_mot_exp001_003_qwen1p7b_256res_frozen_llm_lr_1e4_256nodes_gcp['job']['name']}",
            {"override /vlm_config": "qwen3_vl_mot_vlm_2b_instruct_gcp"},
            "_self_",
        ],
        job=dict(
            group="t2i_mot_2b_qwen3_vl_runs",
            name="t2i_mot_exp001_009_qwen3_vl_2b_256res_frozen_llm",
        ),
        model=dict(
            config=dict(
                joint_attn_implementation="flex",  # "flex" is DEPRECATED: Please use "two_way" for new experiments
            ),
        ),
        checkpoint=dict(
            load_from_object_store=dict(
                enabled=True,
            ),
            load_path="cosmos3_vfm/t2i_mot_1p7b_qwen3_ablations/t2i_mot_exp001_003_qwen1p7b_256res_frozen_llm_lr_1e4_256nodes_gcp/checkpoints/iter_000952000/",
            load_training_state=True,
            strict_resume=False,
        ),
        trainer=dict(
            max_iter=1_250_000,
            logging_iter=200,
        ),
        scheduler=dict(
            f_max=[0.4],
            f_min=[0.3],
            warm_up_steps=[0],
            cycle_lengths=[1_250_000],
        ),
        optimizer=dict(
            keys_to_select=["moe_gen", "time_embedder", "vae2llm", "llm2vae", "latent_pos_embed"],
        ),
    ),
    flags={"allow_objects": True},
)

t2i_mot_exp001_010_qwen3_vl_2b_256res_frozen_llm_small_scale = LazyDict(
    dict(
        defaults=[
            f"/experiment/{t2i_mot_exp001_003_qwen1p7b_256res_frozen_llm_lr_1e4_256nodes_gcp['job']['name']}",
            {"override /vlm_config": "qwen3_vl_mot_vlm_2b_instruct_gcp"},
            "_self_",
        ],
        job=dict(
            group="t2i_mot_2b_qwen3_vl_runs",
            name="t2i_mot_exp001_010_qwen3_vl_2b_256res_frozen_llm_small_scale",
        ),
        model=dict(
            config=dict(
                joint_attn_implementation="flex",  # "flex" is DEPRECATED: Please use "two_way" for new experiments
            ),
        ),
        trainer=dict(
            max_iter=500_000,
            logging_iter=200,
        ),
        scheduler=dict(
            f_max=[1.0],
            f_min=[0.3],
            warm_up_steps=[0],
            cycle_lengths=[500_000],
        ),
        optimizer=dict(
            keys_to_select=["moe_gen", "time_embedder", "vae2llm", "llm2vae", "latent_pos_embed"],
        ),
    ),
    flags={"allow_objects": True},
)


t2i_mot_exp001_010_qwen3_vl_2b_256res_frozen_llm_small_scale_two_way = LazyDict(
    dict(
        defaults=[
            f"/experiment/{t2i_mot_exp001_010_qwen3_vl_2b_256res_frozen_llm_small_scale['job']['name']}",
            "_self_",
        ],
        job=dict(
            group="t2i_mot_2b_qwen3_vl_ablations",
            name="t2i_mot_exp001_010_qwen3_vl_2b_256res_frozen_llm_small_scale_two_way",
        ),
        model=dict(
            config=dict(
                joint_attn_implementation="two_way",
                max_num_tokens_after_packing=32768,
            ),
        ),
    ),
    flags={"allow_objects": True},
)

t2i_mot_exp001_010_qwen3_vl_2b_256res_frozen_llm_small_scale_pchattopadhy_explicit_dur_fps_inclusion = LazyDict(
    dict(
        defaults=[
            f"/experiment/{t2i_mot_exp001_010_qwen3_vl_2b_256res_frozen_llm_small_scale_two_way['job']['name']}",
            "_self_",
        ],
        job=dict(
            group="t2i_mot_2b_qwen3_vl_ablations",
            name="t2i_mot_exp001_010_qwen3_vl_2b_256res_frozen_llm_small_scale_pchattopadhy_explicit_dur_fps_inclusion",
        ),
    ),
    flags={"allow_objects": True},
)

t2i_mot_exp001_010_qwen3_vl_2b_256res_frozen_llm_small_scale_feb08_commit626e6b = LazyDict(
    dict(
        defaults=[
            f"/experiment/{t2i_mot_exp001_010_qwen3_vl_2b_256res_frozen_llm_small_scale_two_way['job']['name']}",
            "_self_",
        ],
        job=dict(
            group="t2i_mot_2b_qwen3_vl_ablations",
            name="t2i_mot_exp001_010_qwen3_vl_2b_256res_frozen_llm_small_scale_feb08_commit626e6b",
        ),
    ),
    flags={"allow_objects": True},
)

t2i_mot_exp001_010_qwen3_vl_2b_256res_frozen_llm_small_scale_mot_refactor_jan15_commit7df3e0 = LazyDict(
    dict(
        defaults=[
            f"/experiment/{t2i_mot_exp001_010_qwen3_vl_2b_256res_frozen_llm_small_scale_two_way['job']['name']}",
            "_self_",
        ],
        job=dict(
            group="t2i_mot_2b_qwen3_vl_ablations",
            name="t2i_mot_exp001_010_qwen3_vl_2b_256res_frozen_llm_small_scale_mot_refactor_jan15_commit7df3e0",
        ),
    ),
    flags={"allow_objects": True},
)

t2i_mot_exp001_010_qwen3_vl_2b_256res_frozen_llm_small_scale_mar09_commitc2ccaf = LazyDict(
    dict(
        defaults=[
            f"/experiment/{t2i_mot_exp001_010_qwen3_vl_2b_256res_frozen_llm_small_scale_two_way['job']['name']}",
            "_self_",
        ],
        job=dict(
            group="t2i_mot_2b_qwen3_vl_ablations",
            name="t2i_mot_exp001_010_qwen3_vl_2b_256res_frozen_llm_small_scale_mar09_commitc2ccaf",
        ),
    ),
    flags={"allow_objects": True},
)

t2i_mot_exp001_010_qwen3_vl_2b_256res_frozen_llm_small_scale_mar24_commit32b97f = LazyDict(
    dict(
        defaults=[
            f"/experiment/{t2i_mot_exp001_010_qwen3_vl_2b_256res_frozen_llm_small_scale_two_way['job']['name']}",
            "_self_",
        ],
        job=dict(
            group="t2i_mot_2b_qwen3_vl_ablations",
            name="t2i_mot_exp001_010_qwen3_vl_2b_256res_frozen_llm_small_scale_mar24_commit32b97f",
        ),
    ),
    flags={"allow_objects": True},
)


# Config for batch size 128 - used for model size comparison
t2i_mot_exp001_011_qwen3_vl_2b_256res_frozen_llm_bs128 = LazyDict(
    dict(
        defaults=[
            f"/experiment/{t2i_mot_exp001_003_qwen1p7b_256res_frozen_llm_lr_1e4_256nodes_gcp['job']['name']}",
            {"override /vlm_config": "qwen3_vl_mot_vlm_2b_instruct_gcp"},
            {"override /model": "mot_fsdp"},
            "_self_",
        ],
        job=dict(
            group="t2i_mot_2b_qwen3_vl_runs",
            name="t2i_mot_exp001_011_qwen3_vl_2b_256res_frozen_llm_bs128",
        ),
        trainer=dict(
            max_iter=1_000_000,
            logging_iter=200,
        ),
        optimizer=dict(
            lr=1e-4,
            keys_to_select=["moe_gen", "time_embedder", "vae2llm", "llm2vae", "latent_pos_embed"],
        ),
        scheduler=dict(
            f_max=[1.0],
            f_min=[0.3],
            warm_up_steps=[2_000],
            cycle_lengths=[1_000_000],
        ),
        model=dict(
            config=dict(
                joint_attn_implementation="flex",
                max_num_tokens_after_packing=40960,
                parallelism=dict(
                    data_parallel_shard_degree=8,
                ),
            ),
        ),
    ),
    flags={"allow_objects": True},
)


t2i_mot_exp001_011_qwen3_vl_2b_256res_frozen_llm_bs128_twoway = LazyDict(
    dict(
        defaults=[
            f"/experiment/{t2i_mot_exp001_011_qwen3_vl_2b_256res_frozen_llm_bs128['job']['name']}",
            "_self_",
        ],
        job=dict(
            group="t2i_mot_2b_qwen3_vl_runs",
            name="t2i_mot_exp001_011_qwen3_vl_2b_256res_frozen_llm_bs128_twoway",
        ),
        model=dict(
            config=dict(
                joint_attn_implementation="two_way",
            ),
        ),
    ),
    flags={"allow_objects": True},
)

# three-way variant, no sparsity
t2i_mot_exp001_011_qwen3_vl_2b_256res_frozen_llm_bs128_threeway = LazyDict(
    dict(
        defaults=[
            f"/experiment/{t2i_mot_exp001_011_qwen3_vl_2b_256res_frozen_llm_bs128_twoway['job']['name']}",
        ],
        job=dict(
            group="t2i_mot_2b_qwen3_vl_runs",
            name="t2i_mot_exp001_011_qwen3_vl_2b_256res_frozen_llm_bs128_threeway",
        ),
        model=dict(
            config=dict(
                joint_attn_implementation="three_way",
            ),
        ),
    ),
    flags={"allow_objects": True},
)

# three-way variant, 75% sparsity in all layers
t2i_mot_exp001_011_qwen3_vl_2b_256res_frozen_llm_bs128_threeway_sparseattn_75pct_alllayers = LazyDict(
    dict(
        defaults=[
            f"/experiment/{t2i_mot_exp001_011_qwen3_vl_2b_256res_frozen_llm_bs128_threeway['job']['name']}",
        ],
        job=dict(
            group="t2i_mot_2b_qwen3_vl_runs",
            name="t2i_mot_exp001_011_qwen3_vl_2b_256res_frozen_llm_bs128_threeway_sparseattn_75pct_alllayers",
        ),
        model=dict(
            config=dict(
                joint_attn_implementation="three_way",
                natten_parameter_list=[{"window_size_float": (0.5, 0.5)} for layer_idx in range(28)],
            ),
        ),
    ),
    flags={"allow_objects": True},
)

# three-way variant, GPT-OSS style sparsity (every other layer), with 75% sparsity in each sparse layer
t2i_mot_exp001_011_qwen3_vl_2b_256res_frozen_llm_bs128_threeway_sparseattn_75pct_evenlayers = LazyDict(
    dict(
        defaults=[
            f"/experiment/{t2i_mot_exp001_011_qwen3_vl_2b_256res_frozen_llm_bs128_threeway['job']['name']}",
        ],
        job=dict(
            group="t2i_mot_2b_qwen3_vl_runs",
            name="t2i_mot_exp001_011_qwen3_vl_2b_256res_frozen_llm_bs128_threeway_sparseattn_75pct_evenlayers",
        ),
        model=dict(
            config=dict(
                joint_attn_implementation="three_way",
                natten_parameter_list=[
                    None if layer_idx % 2 != 0 else {"window_size_float": (0.5, 0.5)} for layer_idx in range(28)
                ],
            ),
        ),
    ),
    flags={"allow_objects": True},
)

# three-way variant, GPT-OSS style sparsity (every other layer), with 90% sparsity in each sparse layer
t2i_mot_exp001_011_qwen3_vl_2b_256res_frozen_llm_bs128_threeway_sparseattn_90pct_evenlayers = LazyDict(
    dict(
        defaults=[
            f"/experiment/{t2i_mot_exp001_011_qwen3_vl_2b_256res_frozen_llm_bs128_threeway['job']['name']}",
        ],
        job=dict(
            group="t2i_mot_2b_qwen3_vl_runs",
            name="t2i_mot_exp001_011_qwen3_vl_2b_256res_frozen_llm_bs128_threeway_sparseattn_90pct_evenlayers",
        ),
        model=dict(
            config=dict(
                joint_attn_implementation="three_way",
                natten_parameter_list=[
                    # 1 - 0.32 * 0.32 ~= 0.9
                    None if layer_idx % 2 != 0 else {"window_size_float": (0.32, 0.32)}
                    for layer_idx in range(28)
                ],
            ),
        ),
    ),
    flags={"allow_objects": True},
)

# three-way variant, "DiNAT-style"
t2i_mot_exp001_011_qwen3_vl_2b_256res_frozen_llm_bs128_threeway_sparseattn_75pct_dinat = LazyDict(
    dict(
        defaults=[
            f"/experiment/{t2i_mot_exp001_011_qwen3_vl_2b_256res_frozen_llm_bs128_threeway['job']['name']}",
        ],
        job=dict(
            group="t2i_mot_2b_qwen3_vl_runs",
            name="t2i_mot_exp001_011_qwen3_vl_2b_256res_frozen_llm_bs128_threeway_sparseattn_75pct_dinat",
        ),
        model=dict(
            config=dict(
                joint_attn_implementation="three_way",
                natten_parameter_list=[
                    {
                        "window_size_float": (0.5, 0.5),
                        "dilation_float": (1.0, 1.0),
                    }
                    if layer_idx % 2 != 0
                    else {"window_size_float": (0.5, 0.5)}
                    for layer_idx in range(28)
                ],
            ),
        ),
    ),
    flags={"allow_objects": True},
)

for _item in [
    t2i_mot_exp001_000_qwen1p7b_256res_frozen_llm,
    t2i_mot_exp001_001_qwen1p7b_256res_frozen_llm_lr_1e4,
    t2i_mot_exp001_002_qwen1p7b_256res_frozen_llm_lr_3e3,
    t2i_mot_exp001_003_qwen1p7b_256res_frozen_llm_lr_1e4_256nodes_gcp,
    t2i_mot_exp001_004_qwen1p7b_256res_frozen_llm_lr_4e4_gcp,
    t2i_mot_exp001_005_qwen1p7b_256res_frozen_llm_lr_6e5_gcp,
    t2i_mot_exp001_006_qwen1p7b_256res_frozen_llm_lr_2e5_gcp,
    t2i_mot_exp001_007_qwen1p7b_256res_frozen_llm_lr_1e4_wd_0p01_gcp,
    t2i_mot_exp001_008_qwen1p7b_256res_frozen_llm_lr_4e4_256nodes_gcp,
    t2i_mot_exp001_009_qwen3_vl_2b_256res_frozen_llm,
    t2i_mot_exp001_010_qwen3_vl_2b_256res_frozen_llm_small_scale,
    t2i_mot_exp001_010_qwen3_vl_2b_256res_frozen_llm_small_scale_two_way,
    t2i_mot_exp001_010_qwen3_vl_2b_256res_frozen_llm_small_scale_mot_refactor_jan15_commit7df3e0,
    t2i_mot_exp001_010_qwen3_vl_2b_256res_frozen_llm_small_scale_feb08_commit626e6b,
    t2i_mot_exp001_010_qwen3_vl_2b_256res_frozen_llm_small_scale_mar09_commitc2ccaf,
    t2i_mot_exp001_010_qwen3_vl_2b_256res_frozen_llm_small_scale_mar24_commit32b97f,
    t2i_mot_exp001_011_qwen3_vl_2b_256res_frozen_llm_bs128,
    t2i_mot_exp001_011_qwen3_vl_2b_256res_frozen_llm_bs128_twoway,
    t2i_mot_exp001_011_qwen3_vl_2b_256res_frozen_llm_bs128_threeway,
    t2i_mot_exp001_011_qwen3_vl_2b_256res_frozen_llm_bs128_threeway_sparseattn_75pct_alllayers,
    t2i_mot_exp001_011_qwen3_vl_2b_256res_frozen_llm_bs128_threeway_sparseattn_75pct_evenlayers,
    t2i_mot_exp001_011_qwen3_vl_2b_256res_frozen_llm_bs128_threeway_sparseattn_75pct_dinat,
    t2i_mot_exp001_011_qwen3_vl_2b_256res_frozen_llm_bs128_threeway_sparseattn_90pct_evenlayers,
    t2i_mot_exp001_010_qwen3_vl_2b_256res_frozen_llm_small_scale_pchattopadhy_explicit_dur_fps_inclusion,
    t2i_mot_exp001_011_qwen3_vl_2b_256res_frozen_llm_bs128_twoway,
]:
    experiment_name = [name.lower() for name, value in globals().items() if value is _item][0]
    cs.store(
        group="experiment",
        package="_global_",
        name=experiment_name,
        node=_item,
    )
