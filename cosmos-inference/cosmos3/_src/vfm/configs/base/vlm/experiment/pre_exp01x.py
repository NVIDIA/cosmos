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

from cosmos3._src.imaginaire.lazy_config import LazyDict

cs = ConfigStore.instance()

"""
Bundled VLM training experiments registered under the ``experiment`` group.

Usage:
torchrun --nproc_per_node=8 --master_port=12341 -m scripts.train \
    --config=cosmos3/_src/vfm/configs/base/vlm/config.py \
    -- experiment=pre_exp010_000_eagle_er_1p7b_joint_reasoner_tl_722_5vs5_no_predict2_s3_webloader
"""


pre_exp010_000_eagle_er_1p7b_joint_reasoner_tl_722_5vs5_no_predict2_s3_webloader = LazyDict(
    dict(
        defaults=[
            {"override /checkpoint": "s3"},
            {"override /data_train": "joint_reasoner_tl_722_5vs5_no_predict2_s3_webloader"},
            {"override /data_val": "debug_image_data_qwen"},
            {"override /model": "vlm_fsdp"},
            {"override /vlm_policy": "eagle_er_1p7b"},
            {"override /callbacks": ["basic_vlm", "simple_log"]},
            "_self_",
        ],
        job=dict(
            group="debug",
        ),
        trainer=dict(
            callbacks=dict(log_tensor_shape=dict(num_log=100)),
            max_iter=32000,
        ),
        checkpoint=dict(save_iter=5000),
    )
)

pre_exp010_010_internvl3_5_1b_joint_reasoner_tl_722_5vs5_no_predict2_s3_webloader = LazyDict(
    dict(
        defaults=[
            {"override /checkpoint": "s3"},
            {"override /data_train": "joint_reasoner_tl_722_5vs5_no_predict2_s3_webloader"},
            {"override /data_val": "debug_image_data_qwen"},
            {"override /model": "vlm_fsdp"},
            {"override /vlm_policy": "internvl3_5_1b"},
            {"override /callbacks": ["basic_vlm", "simple_log"]},
            "_self_",
        ],
        job=dict(
            group="debug",
        ),
        trainer=dict(
            callbacks=dict(log_tensor_shape=dict(num_log=100)),
        ),
        checkpoint=dict(save_iter=2000),
    )
)

pre_exp010_020_internvl3_5_2b_joint_reasoner_tl_722_5vs5_no_predict2_s3_webloader = LazyDict(
    dict(
        defaults=[
            {"override /checkpoint": "s3"},
            {"override /data_train": "joint_reasoner_tl_722_5vs5_no_predict2_s3_webloader"},
            {"override /data_val": "debug_image_data_qwen"},
            {"override /model": "vlm_fsdp"},
            {"override /vlm_policy": "internvl3_5_2b"},
            {"override /callbacks": ["basic_vlm", "simple_log"]},
            "_self_",
        ],
        job=dict(
            group="debug",
        ),
        trainer=dict(
            callbacks=dict(log_tensor_shape=dict(num_log=100)),
            max_iter=32000,
        ),
        checkpoint=dict(save_iter=2000),
    )
)

"""
torchrun --nproc_per_node=8 --master_port=12341 -m projects.cosmos3.vlm.train     --config=projects/cosmos3/vlm/configs/base/config.py -- experiment=pre_exp011_000_qwen3_vl_2b
"""
pre_exp011_000_qwen3_vl_2b = LazyDict(
    dict(
        defaults=[
            {"override /checkpoint": "s3"},
            {"override /data_train": "08_eagle_sft_full_mul_repeat_default_s3"},
            {"override /data_val": "debug_image_data_qwen"},
            {"override /model": "vlm_fsdp"},
            {"override /vlm_policy": "qwen3_vl_2b"},
            {"override /callbacks": ["basic_vlm", "simple_log"]},
            "_self_",
        ],
        job=dict(
            group="debug",
        ),
        trainer=dict(
            callbacks=dict(log_tensor_shape=dict(num_log=2)),
            max_iter=8000,
        ),
        checkpoint=dict(save_iter=2000),
    )
)


"""
torchrun --nproc_per_node=8 --master_port=12341 -m projects.cosmos3.vlm.train --config=projects/cosmos3/vlm/configs/base/config.py -- experiment=pre_exp011_020_qwen3_vl_2b_vit2k8k
"""
pre_exp011_020_qwen3_vl_2b_vit2k8k = LazyDict(
    dict(
        defaults=[
            {"override /checkpoint": "s3"},
            {"override /data_train": "08_eagle_sft_full_mul_repeat_default_s3"},
            {"override /data_val": "debug_image_data_qwen"},
            {"override /model": "vlm_fsdp"},
            {"override /vlm_policy": "qwen3_vl_2b"},
            {"override /callbacks": ["basic_vlm", "simple_log"]},
            "_self_",
        ],
        job=dict(
            group="debug",
        ),
        trainer=dict(
            callbacks=dict(log_tensor_shape=dict(num_log=2)),
            max_iter=8000,
        ),
        optimizer=dict(
            config=dict(
                lr=5e-5,
                fused=True,
            ),
        ),
        model=dict(
            config=dict(
                policy=dict(
                    parallelism=dict(
                        data_parallel_shard_degree=8,
                        data_parallel_replicate_degree=-1,
                    ),
                ),
            ),
        ),
        data_setting=dict(
            qwen_max_video_token_length=8192,
            qwen_max_image_token_length=2048,
        ),
        checkpoint=dict(save_iter=2000),
    )
)

"""
torchrun --nproc_per_node=8 --master_port=12341 -m projects.cosmos3.vlm.train --config=projects/cosmos3/vlm/configs/base/config.py -- experiment=pre_exp011_030_qwen3_vl_2b_vit2k8k_mbs8
"""
pre_exp011_030_qwen3_vl_2b_vit2k8k_mbs8 = LazyDict(
    dict(
        defaults=[
            {"override /checkpoint": "s3"},
            {"override /data_train": "08_eagle_sft_full_mul_repeat_default_s3"},
            {"override /data_val": "debug_image_data_qwen"},
            {"override /model": "vlm_fsdp"},
            {"override /vlm_policy": "qwen3_vl_2b"},
            {"override /callbacks": ["basic_vlm", "simple_log"]},
            "_self_",
        ],
        job=dict(
            group="debug",
        ),
        trainer=dict(
            callbacks=dict(log_tensor_shape=dict(num_log=2)),
            max_iter=200_000,
        ),
        optimizer=dict(
            config=dict(
                lr=5e-5,
                fused=True,
            ),
        ),
        data_setting=dict(
            qwen_max_video_token_length=8192,
            qwen_max_image_token_length=2048,
        ),
        model=dict(
            config=dict(
                policy=dict(
                    parallelism=dict(
                        data_parallel_shard_degree=8,
                        data_parallel_replicate_degree=-1,
                    ),
                ),
            ),
        ),
        dataloader_train=dict(
            max_batch_size=8,
            max_tokens=16000,
        ),
        checkpoint=dict(save_iter=2000),
    )
)

"""
torchrun --nproc_per_node=8 --master_port=12341 -m projects.cosmos3.vlm.train --config=projects/cosmos3/vlm/configs/base/config.py -- experiment=pre_exp011_030_qwen3_vl_2b_vit2k8k_mbs8
"""
pre_exp011_040_qwen3_vl_2b_vit2k8k_mbs8_flop3s = LazyDict(
    dict(
        defaults=[
            {"override /checkpoint": "s3"},
            {"override /data_train": "08_eagle_sft_full_mul_repeat_default_s3"},
            {"override /data_val": "debug_image_data_qwen"},
            {"override /model": "vlm_fsdp"},
            {"override /vlm_policy": "qwen3_vl_2b"},
            {"override /callbacks": ["basic_vlm", "simple_log"]},
            "_self_",
        ],
        job=dict(
            group="debug",
        ),
        trainer=dict(
            callbacks=dict(log_tensor_shape=dict(num_log=2)),
            max_iter=200_000,
        ),
        optimizer=dict(
            config=dict(
                lr=5e-5,
                fused=True,
            ),
        ),
        data_setting=dict(
            qwen_max_video_token_length=8192,
            qwen_max_image_token_length=2048,
            max_tokens=16000,
            max_batch_size=8,
        ),
        model=dict(
            config=dict(
                policy=dict(
                    parallelism=dict(
                        data_parallel_shard_degree=8,
                        data_parallel_replicate_degree=-1,
                    ),
                ),
            ),
        ),
        dataloader_train=dict(
            enable_flop_based_batching=True,
            target_runtime_seconds=3.0,
        ),
        checkpoint=dict(save_iter=2000),
    )
)

pre_exp011_041_qwen3_vl_2b_vit2k8k_mbs8_flop3s_mix_text_only = LazyDict(
    dict(
        defaults=[
            {"override /checkpoint": "s3"},
            {"override /data_train": "08_eagle_sft_full_mul_repeat_default_s3"},
            {"override /data_val": "debug_image_data_qwen"},
            {"override /model": "vlm_fsdp"},
            {"override /vlm_policy": "qwen3_vl_2b"},
            {"override /callbacks": ["basic_vlm", "simple_log"]},
            "_self_",
        ],
        job=dict(
            group="debug",
        ),
        trainer=dict(
            callbacks=dict(log_tensor_shape=dict(num_log=2)),
            max_iter=200_000,
        ),
        optimizer=dict(
            config=dict(
                lr=5e-5,
                fused=True,
            ),
        ),
        data_setting=dict(
            qwen_max_video_token_length=8192,
            qwen_max_image_token_length=2048,
            max_tokens=16000,
            max_batch_size=8,
        ),
        model=dict(
            config=dict(
                policy=dict(
                    parallelism=dict(
                        data_parallel_shard_degree=8,
                        data_parallel_replicate_degree=-1,
                    ),
                ),
            ),
        ),
        dataloader_train=dict(
            dataloaders=dict(
                with_visual=dict(
                    dataloader=dict(
                        enable_flop_based_batching=True,
                        target_runtime_seconds=3.0,
                    ),
                ),
                text_only=dict(
                    dataloader=dict(
                        enable_flop_based_batching=True,
                        target_runtime_seconds=3.0,
                    ),
                ),
            )
        ),
        checkpoint=dict(save_iter=2000),
    )
)


pre_exp011_100_qwen3_vl_2b_align = LazyDict(
    dict(
        defaults=[
            {"override /checkpoint": "s3"},
            {"override /data_train": "07_eagle_pretrain_default_s3"},
            {"override /data_val": "debug_image_data_qwen"},
            {"override /model": "vlm_fsdp"},
            {"override /vlm_policy": "qwen3_vl_2b"},
            {"override /callbacks": ["basic_vlm", "simple_log"]},
            "_self_",
        ],
        job=dict(
            group="debug",
        ),
        optimizer=dict(
            config=dict(
                freeze_vision_encoder=True,
                freeze_llm=True,
                lr=1e-5,
                init_lr=1e-7,
                end_lr=5e-6,
            ),
        ),
        trainer=dict(
            callbacks=dict(log_tensor_shape=dict(num_log=2)),
            max_iter=8000,
        ),
        checkpoint=dict(save_iter=2000),
    )
)

# reinit the llm and/or projector weights for internvl3_5_2b
pre_exp011_300_internvl3_5_2b_reinit_align = LazyDict(
    dict(
        defaults=[
            {"override /checkpoint": "s3"},
            {"override /data_train": "07_eagle_pretrain_default_s3"},
            {"override /data_val": "debug_image_data_qwen"},
            {"override /model": "vlm_fsdp"},
            {"override /vlm_policy": "internvl3_5_2b"},
            {"override /callbacks": ["basic_vlm", "simple_log"]},
            "_self_",
        ],
        job=dict(
            group="debug",
        ),
        optimizer=dict(
            config=dict(
                freeze_vision_encoder=True,
                freeze_llm=True,
                lr=1e-5,
                init_lr=1e-7,
                end_lr=5e-6,
            ),
        ),
        trainer=dict(
            callbacks=dict(log_tensor_shape=dict(num_log=2)),
            max_iter=8000,
        ),
        model=dict(
            config=dict(
                policy=dict(
                    model_name_or_path="OpenGVLab/InternVL3_5-2B-HF-ReinitLLMProj",
                ),
            ),
        ),
        checkpoint=dict(save_iter=2000),
    )
)


# reinit the llm and/or projector weights for internvl3_5_2b
pre_exp011_400_internvl3_5_2b_reinit_e2e = LazyDict(
    dict(
        defaults=[
            {"override /checkpoint": "s3"},
            {"override /data_train": "07_eagle_pretrain_default_s3"},
            {"override /data_val": "debug_image_data_qwen"},
            {"override /model": "vlm_fsdp"},
            {"override /vlm_policy": "internvl3_5_2b"},
            {"override /callbacks": ["basic_vlm", "simple_log"]},
            "_self_",
        ],
        job=dict(
            group="debug",
        ),
        trainer=dict(
            callbacks=dict(log_tensor_shape=dict(num_log=2)),
            max_iter=8000,
        ),
        model=dict(
            config=dict(
                policy=dict(
                    model_name_or_path="OpenGVLab/InternVL3_5-2B-HF-ReinitLLMProj",
                ),
            ),
        ),
        checkpoint=dict(save_iter=2000),
    )
)

"""
torchrun --nproc_per_node=8 --master_port=12341 -m projects.cosmos3.vlm.train --config=projects/cosmos3/vlm/configs/base/config.py -- experiment=pre_exp014_000_qwen3_vl_8b_instruct_vit2k8k_mbs8_flop3s
- reduce lr to 2e-6 cause this is SFT
"""
pre_exp015_000_qwen3_vl_8b_instruct_vit2k8k_mbs1_flop3s = LazyDict(
    dict(
        defaults=[
            {"override /checkpoint": "s3"},
            {"override /data_train": "08_eagle_sft_full_mul_repeat_default_s3"},
            {"override /data_val": "dummy_image_data_qwen"},
            {"override /model": "vlm_fsdp"},
            {"override /vlm_policy": "qwen3_vl_8b_instruct"},
            {"override /callbacks": ["basic_vlm", "simple_log"]},
            "_self_",
        ],
        job=dict(
            group="debug",
        ),
        trainer=dict(
            callbacks=dict(log_tensor_shape=dict(num_log=2)),
            max_iter=8000,
        ),
        optimizer=dict(
            config=dict(
                lr=1e-5,
                fused=True,
            ),
        ),
        data_setting=dict(
            qwen_max_video_token_length=8192,
            qwen_max_image_token_length=2048,
            max_tokens=16000,
            max_batch_size=1,
        ),
        model=dict(
            config=dict(
                policy=dict(
                    parallelism=dict(
                        data_parallel_shard_degree=8,
                        data_parallel_replicate_degree=-1,
                    ),
                ),
            ),
        ),
        dataloader_train=dict(
            enable_flop_based_batching=True,
            target_runtime_seconds=3.0,  # Used when max_batch_size > 1
        ),
        checkpoint=dict(save_iter=2000),
    )
)

"""
torchrun --nproc_per_node=8 --master_port=12341 -m projects.cosmos3.vlm.train --config=projects/cosmos3/vlm/configs/base/config.py -- experiment=pre_exp015_001_qwen3_vl_8b_instruct_vit2k8k_mbs1_flop3s_mix_text_only data_train=m02_visual_5_mix_text_1__joint_reason2p0_2p1_joint_s3
"""
pre_exp015_001_qwen3_vl_8b_instruct_vit2k8k_mbs1_flop3s_mix_text_only = LazyDict(
    dict(
        defaults=[
            {"override /checkpoint": "s3"},
            {"override /data_train": "08_eagle_sft_full_mul_repeat_default_s3"},
            {"override /data_val": "debug_image_data_qwen"},
            {"override /model": "vlm_fsdp"},
            {"override /vlm_policy": "qwen3_vl_8b_instruct"},
            {"override /callbacks": ["basic_vlm", "simple_log"]},
            "_self_",
        ],
        job=dict(
            group="debug",
        ),
        trainer=dict(
            callbacks=dict(log_tensor_shape=dict(num_log=2)),
            max_iter=10000,
        ),
        optimizer=dict(
            config=dict(
                lr=1e-5,
                fused=True,
            ),
        ),
        data_setting=dict(
            qwen_max_video_token_length=8192,
            qwen_max_image_token_length=2048,
            max_tokens=16000,
            max_batch_size=1,
        ),
        model=dict(
            config=dict(
                policy=dict(
                    parallelism=dict(
                        data_parallel_shard_degree=8,
                        data_parallel_replicate_degree=-1,
                    ),
                ),
            ),
        ),
        dataloader_train=dict(
            dataloaders=dict(
                with_visual=dict(
                    dataloader=dict(
                        enable_flop_based_batching=True,
                        target_runtime_seconds=3.0,
                    ),
                ),
                text_only=dict(
                    dataloader=dict(
                        enable_flop_based_batching=True,
                        target_runtime_seconds=3.0,
                    ),
                ),
            )
        ),
        checkpoint=dict(save_iter=2000),
    )
)

"""
torchrun --nproc_per_node=8 --master_port=12341 -m projects.cosmos3.vlm.train --config=projects/cosmos3/vlm/configs/base/config.py -- experiment=pre_exp014_000_qwen3_vl_8b_instruct_vit2k8k_mbs8_flop3s
- reduce lr to 2e-6 cause this is SFT
"""
pre_exp016_000_qwen3_vl_8b_thinking_vit2k8k_mbs1_flop3s = LazyDict(
    dict(
        defaults=[
            {"override /checkpoint": "s3"},
            {"override /data_train": "08_eagle_sft_full_mul_repeat_default_s3"},
            {"override /data_val": "debug_image_data_qwen"},
            {"override /model": "vlm_fsdp"},
            {"override /vlm_policy": "qwen3_vl_8b_thinking"},
            {"override /callbacks": ["basic_vlm", "simple_log"]},
            "_self_",
        ],
        job=dict(
            group="debug",
        ),
        trainer=dict(
            callbacks=dict(log_tensor_shape=dict(num_log=2)),
            max_iter=8000,
        ),
        optimizer=dict(
            config=dict(
                lr=1e-5,
                fused=True,
            ),
        ),
        data_setting=dict(
            qwen_max_video_token_length=8192,
            qwen_max_image_token_length=2048,
            max_tokens=16000,
            max_batch_size=1,
        ),
        model=dict(
            config=dict(
                policy=dict(
                    parallelism=dict(
                        data_parallel_shard_degree=8,
                        data_parallel_replicate_degree=-1,
                    ),
                ),
            ),
        ),
        dataloader_train=dict(
            enable_flop_based_batching=True,
            target_runtime_seconds=3.0,  # Used when max_batch_size > 1
        ),
        checkpoint=dict(save_iter=2000),
    )
)


"""
torchrun --nproc_per_node=8 --master_port=12341 -m projects.cosmos3.vlm.train --config=projects/cosmos3/vlm/configs/base/config.py -- experiment=pre_exp017_000_nemotron_vl_12b_mbs1
"""
pre_exp017_000_nemotron_vl_12b_mbs1 = LazyDict(
    dict(
        defaults=[
            {"override /checkpoint": "s3"},
            {"override /data_train": "51_joint_reason1p0_1p1_2_1126_1126_s3"},
            {"override /data_val": "debug_image_data_qwen"},
            {"override /model": "vlm_fsdp"},
            {"override /vlm_policy": "nemotron_nano_12b_v2_vl_bf16"},
            {"override /callbacks": ["basic_vlm", "simple_log"]},
            "_self_",
        ],
        job=dict(
            group="debug",
        ),
        trainer=dict(
            callbacks=dict(log_tensor_shape=dict(num_log=2)),
            max_iter=8000,
        ),
        optimizer=dict(
            config=dict(
                lr=1e-5,
                fused=True,
            ),
        ),
        data_setting=dict(
            max_tokens=16000,
            max_batch_size=1,
        ),
        model=dict(
            config=dict(
                policy=dict(
                    parallelism=dict(
                        data_parallel_shard_degree=8,
                        data_parallel_replicate_degree=-1,
                    ),
                ),
            ),
        ),
        checkpoint=dict(save_iter=2000),
    )
)

"""
torchrun --nproc_per_node=8 --master_port=12341 -m projects.cosmos3.vlm.train --config=projects/cosmos3/vlm/configs/base/config.py -- experiment=pre_exp018_000_qwen3_vl_8b_instruct_vit2k8k_mbs1_flop3s_pretrain
"""
pre_exp018_000_qwen3_vl_8b_instruct_vit2k8k_mbs1_flop3s_pretrain = LazyDict(
    dict(
        defaults=[
            {"override /checkpoint": "s3"},
            {"override /data_train": "200_nanov2_stage_1_0218_34m_uniform_pretrain_repeat_s3_vlmdb"},
            {"override /data_val": "200_nanov2_stage_1_0218_34m_uniform_pretrain_repeat_s3_vlmdb"},
            {"override /model": "vlm_fsdp"},
            {"override /vlm_policy": "qwen3_vl_8b_instruct"},
            {"override /callbacks": ["basic_vlm", "simple_log"]},
            "_self_",
        ],
        job=dict(
            group="debug",
        ),
        trainer=dict(
            callbacks=dict(log_tensor_shape=dict(num_log=2)),
            max_iter=60000,
            run_validation=True,
            validation_iter=40,
            max_val_iter=1,
        ),
        optimizer=dict(
            config=dict(
                lr=2e-5,
                fused=True,
                weight_decay=0.05,
                betas=[0.9, 0.999],
                freeze_vision_encoder=True,
                freeze_mm_projector=True,
            ),
        ),
        scheduler=dict(
            warmup_iters=6000,
        ),
        data_setting=dict(
            qwen_max_video_token_length=8192,
            qwen_max_image_token_length=2048,
            max_tokens=16000,
            max_batch_size=1,
            distributor_type="no_replace",
            distributor_seed=1993,
            val_split_ratio=0.1,
        ),
        model=dict(
            config=dict(
                policy=dict(
                    parallelism=dict(
                        data_parallel_shard_degree=8,
                        data_parallel_replicate_degree=-1,
                    ),
                    monkey_patch_for_text_only_data=True,
                    pretrain_weights_path_llm="Qwen/Qwen3-8B",
                ),
            ),
        ),
        dataloader_train=dict(
            enable_flop_based_batching=True,
            target_runtime_seconds=3.0,  # Used when max_batch_size > 1
        ),
        checkpoint=dict(
            save_iter=2000,
        ),
    )
)

"""
torchrun --nproc_per_node=8 --master_port=12341 -m projects.cosmos3.vlm.train --config=projects/cosmos3/vlm/configs/base/config.py -- experiment=pre_exp019_000_qwen3_vl_8b_instruct_vit2k8k_mbs1_flop3s_posttrain
"""
pre_exp019_000_qwen3_vl_8b_instruct_vit2k8k_mbs1_flop3s_posttrain = LazyDict(
    dict(
        defaults=[
            {"override /checkpoint": "s3"},
            {"override /data_train": "201_nanov2_stage_1_0218_34m_uniform_posttrain_repeat_s3_vlmdb"},
            {"override /data_val": "201_nanov2_stage_1_0218_34m_uniform_posttrain_repeat_s3_vlmdb"},
            {"override /model": "vlm_fsdp"},
            {"override /vlm_policy": "qwen3_vl_8b_instruct"},
            {"override /callbacks": ["basic_vlm", "simple_log"]},
            "_self_",
        ],
        job=dict(
            group="debug",
        ),
        trainer=dict(
            callbacks=dict(log_tensor_shape=dict(num_log=2)),
            max_iter=8000,
            run_validation=True,
            validation_iter=40,
            max_val_iter=1,
        ),
        optimizer=dict(
            config=dict(
                lr=1e-5,
                fused=True,
                weight_decay=0.05,
                betas=[0.9, 0.999],
                freeze_vision_encoder=True,
                freeze_mm_projector=True,
            ),
        ),
        scheduler=dict(
            warmup_iters=800,
        ),
        data_setting=dict(
            qwen_max_video_token_length=8192,
            qwen_max_image_token_length=2048,
            max_tokens=16000,
            max_batch_size=1,
            distributor_type="no_replace",
            distributor_seed=1996,
            val_split_ratio=0.05,
        ),
        model=dict(
            config=dict(
                policy=dict(
                    parallelism=dict(
                        data_parallel_shard_degree=8,
                        data_parallel_replicate_degree=-1,
                    ),
                    monkey_patch_for_text_only_data=True,
                ),
            ),
        ),
        dataloader_train=dict(
            enable_flop_based_batching=True,
            target_runtime_seconds=3.0,  # Used when max_batch_size > 1
        ),
        checkpoint=dict(
            save_iter=1000,
            load_path="cosmos_reason2/pre_exp015/pre_exp015_288_34m_v1352_repeat_uniform_1_epoch_n64/checkpoints/iter_000060000/",
        ),
    )
)


pre_exp020_001_qwen3_vl_30b_a3b_instruct_ep = LazyDict(
    dict(
        defaults=[
            {"override /checkpoint": "s3"},
            {"override /data_train": "nemotron_nanov2_stage_1_0218_34m_uniform_pretrain_s3_vlmdb_recipe"},
            {"override /data_val": "nemotron_nanov2_stage_1_0218_34m_uniform_pretrain_s3_vlmdb_recipe"},
            {"override /model": "vlm_fsdp"},
            {"override /vlm_policy": "qwen3_vl_30b_a3b_instruct"},
            {"override /callbacks": ["basic_vlm", "simple_log"]},
            "_self_",
        ],
        job=dict(group="debug"),
        trainer=dict(
            max_iter=8000,
            logging_iter=1,
        ),
        optimizer=dict(config=dict(lr=5e-5, fused=True)),
        data_setting=dict(
            qwen_max_video_token_length=8192,
            qwen_max_image_token_length=2048,
            max_tokens=16000,
            max_batch_size=1,
            distributor_type="no_replace",
            distributor_seed=1993,
            val_split_ratio=0.1,
            webdataset_detshuffle=True,
        ),
        model=dict(
            config=dict(
                policy=dict(
                    parallelism=dict(
                        data_parallel_shard_degree=2,
                        data_parallel_replicate_degree=1,
                    ),
                ),
            ),
        ),
        dataloader_train=dict(
            enable_flop_based_batching=True,
            target_runtime_seconds=3.0,  # Used when max_batch_size > 1
        ),
        checkpoint=dict(save_iter=2000),
    )
)


for _item in [
    pre_exp010_000_eagle_er_1p7b_joint_reasoner_tl_722_5vs5_no_predict2_s3_webloader,
    pre_exp010_010_internvl3_5_1b_joint_reasoner_tl_722_5vs5_no_predict2_s3_webloader,
    pre_exp010_020_internvl3_5_2b_joint_reasoner_tl_722_5vs5_no_predict2_s3_webloader,
    pre_exp011_000_qwen3_vl_2b,
    pre_exp011_020_qwen3_vl_2b_vit2k8k,
    pre_exp011_030_qwen3_vl_2b_vit2k8k_mbs8,
    pre_exp011_040_qwen3_vl_2b_vit2k8k_mbs8_flop3s,
    pre_exp011_041_qwen3_vl_2b_vit2k8k_mbs8_flop3s_mix_text_only,
    pre_exp011_100_qwen3_vl_2b_align,
    pre_exp011_300_internvl3_5_2b_reinit_align,
    pre_exp011_400_internvl3_5_2b_reinit_e2e,
    pre_exp015_000_qwen3_vl_8b_instruct_vit2k8k_mbs1_flop3s,
    pre_exp015_001_qwen3_vl_8b_instruct_vit2k8k_mbs1_flop3s_mix_text_only,
    pre_exp016_000_qwen3_vl_8b_thinking_vit2k8k_mbs1_flop3s,
    pre_exp017_000_nemotron_vl_12b_mbs1,
    pre_exp018_000_qwen3_vl_8b_instruct_vit2k8k_mbs1_flop3s_pretrain,
    pre_exp019_000_qwen3_vl_8b_instruct_vit2k8k_mbs1_flop3s_posttrain,
    pre_exp020_001_qwen3_vl_30b_a3b_instruct_ep,
]:
    experiment_name = [name.lower() for name, value in globals().items() if value is _item][0]
    if "job" not in _item:
        _item["job"] = dict(name=experiment_name + "_${now:%Y-%m-%d}_${now:%H-%M-%S}")
    else:
        _item["job"]["name"] = experiment_name + "_${now:%Y-%m-%d}_${now:%H-%M-%S}"

    cs.store(group="experiment", package="_global_", name=experiment_name, node=_item)
