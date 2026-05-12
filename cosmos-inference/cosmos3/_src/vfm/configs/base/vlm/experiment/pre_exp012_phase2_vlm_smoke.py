# -----------------------------------------------------------------------------
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# This codebase constitutes NVIDIA proprietary technology and is strictly
# confidential. Any unauthorized reproduction, distribution, or disclosure
# of this code, in whole or in part, outside NVIDIA is strictly prohibited
# without prior written consent.
#
# For inquiries regarding the use of this code in other NVIDIA CORPORATION
# projects, please contact the Deep Imagination Research Team at
# dir@exchange.nvidia.com.
# -----------------------------------------------------------------------------
"""Phase 2 VFM VLMModel smoke test experiments.

These configs exercise the Phase 2 HFModel/FSDP2 path end-to-end on 4 GPUs.
They are NOT training runs — max_iter=10 is intentionally minimal.

Usage:
    torchrun --nproc_per_node=4 --master_port=12341 -m scripts.train \\
        --config=cosmos3/_src/vfm/configs/base/vlm/config.py \\
        -- experiment=pre_exp012_000_phase2_vlm_smoke_4gpu
"""

from hydra.core.config_store import ConfigStore

from cosmos3._src.imaginaire.lazy_config import LazyDict

cs = ConfigStore.instance()

# ---------------------------------------------------------------------------
# Smoke test: 4-GPU FSDP2 with qwen3_vl_2b, debug data, 10 iterations.
# Exercises: HFModel meta-init → parallelize() → forward → CE loss → backward
# Requires: trainable_params (Phase 2 mandatory), data_parallel_shard_degree=4
# ---------------------------------------------------------------------------
pre_exp012_000_phase2_vlm_smoke_4gpu_8b = LazyDict(
    dict(
        defaults=[
            {"override /checkpoint": "s3"},
            {"override /data_train": "nemotron_nanov2_stage_1_0218_34m_uniform_pretrain_s3_vlmdb_recipe"},
            {"override /data_val": "nemotron_nanov2_stage_1_0218_34m_uniform_pretrain_s3_vlmdb_recipe"},
            {"override /model": "vlm_fsdp"},
            {"override /vlm_policy": "qwen3_vl_8b_instruct"},
            {"override /callbacks": ["basic_vlm", "simple_log"]},
            "_self_",
        ],
        job=dict(
            group="phase2_smoke",
        ),
        trainer=dict(
            max_iter=10,
            run_validation=False,
            logging_iter=1,
        ),
        optimizer=dict(
            config=dict(
                # Phase 2 REQUIRED: trainable_params regex list.
                # ".*" trains all parameters — appropriate for smoke test.
                trainable_params=[".*"],
                lr=1e-5,
                fused=True,
            ),
        ),
        model=dict(
            config=dict(
                policy=dict(
                    parallelism=dict(
                        data_parallel_shard_degree=4,
                        data_parallel_replicate_degree=-1,
                    ),
                ),
            ),
        ),
        # dataloader_train=dict(
        #     enable_flop_based_batching=True,
        #     target_runtime_seconds=3.0,  # Used when max_batch_size > 1
        # ),
        checkpoint=dict(
            # Don't save checkpoints during smoke test
            save_iter=100000,
        ),
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
    )
)


for _item in [
    pre_exp012_000_phase2_vlm_smoke_4gpu_8b,
]:
    experiment_name = [name.lower() for name, value in globals().items() if value is _item][0]
    if "job" not in _item:
        _item["job"] = dict(name=experiment_name + "_${now:%Y-%m-%d}_${now:%H-%M-%S}")
    else:
        _item["job"]["name"] = experiment_name + "_${now:%Y-%m-%d}_${now:%H-%M-%S}"

    cs.store(group="experiment", package="_global_", name=experiment_name, node=_item)
