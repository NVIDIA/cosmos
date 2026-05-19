# -----------------------------------------------------------------------------
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# This codebase constitutes NVIDIA proprietary technology and is strictly
# confidential. Any unauthorized reproduction, distribution, or disclosure
# of this code, in whole or in part, outside NVIDIA is strictly prohibited
# without prior written consent.
# -----------------------------------------------------------------------------

"""VFM T2W SFT — Qwen3-VL-8B, 720p, local dataset via DataPackerDataLoader.

Equivalent to t2w_sft_8b_local.py but uses the OSS-facing DataPackerDataLoader
instead of PackingDataLoader + RankPartitionedDataLoader.

Demonstrates the intended OSS usage pattern:
  1. Define a custom DataPacker subclass (SFTDataPacker) in the experiment file.
  2. Wire a data_source (get_sft_dataset_no_dp) into DataPackerDataLoader.
  3. DataPackerDataLoader handles sequence packing and data-parallel sharding.

get_sft_dataset_no_dp disables SFTDataset's internal DP sharding so that
_IterableWrapper inside DataPackerDataLoader is the sole DP sharding layer,
avoiding double-sharding.

The DataPackerDataLoader must be initialized with ``batch_size=None`` to
disable PyTorch's default collation — the dataset already returns a fully
collated batch dict, and re-collating it would add a spurious outer batch
dimension that breaks the model forward pass.

Prerequisites
-------------
Please refer to: https://github.com/nvidia-cosmos/cosmos3-internal/blob/main/docs/training.md

- DCP mid-train checkpoint at ``checkpoint.load_path`` (8B weights).
- GCP credentials at ``credentials/gcp_checkpoint.secret`` for:
    - VAE weights (``model.config.tokenizer.bucket_name``).
    - VLM tokenizer files (downloaded automatically on startup).
- Wan2.2 VAE bucket: ``nv-00-10206-checkpoint``.
- ``credentials/gcs.secret`` must exist (symlink to ``gcp_checkpoint.secret``
  is sufficient) — required by SFTDataset on INTERNAL builds.

Production run (4 GPUs, GCP GB200)::

    torchrun --nproc_per_node=4 --master_port=12342 -m scripts.train \\
        --config=configs/base/config.py -- \\
        experiment=t2w_sft_8b_local_datapacker \\
        checkpoint.load_path=/path/to/midtrain/checkpoint \\
        model.config.tokenizer.bucket_name=nv-00-10206-checkpoint \\
        model.config.tokenizer.object_store_credential_path_pretrained=credentials/gcp_checkpoint.secret \\
        'dataloader_train.data_source.jsonl_paths=[/path/to/video_dataset_file.jsonl]'

Smoke run (4 GPUs, no checkpoint, flex attention for non-GB200 nodes)::

    torchrun --nproc_per_node=4 --master_port=12342 -m scripts.train \\
        --config=configs/base/config.py -- \\
        experiment=t2w_sft_8b_local_datapacker \\
        trainer.max_iter=10 trainer.logging_iter=1 \\
        job.wandb_mode=disabled upload_reproducible_setup=false \\
        checkpoint.load_path=/path/to/midtrain/checkpoint \\
        checkpoint=local ckpt_type=dummy \\
        model.config.parallelism.data_parallel_shard_degree=-1 \\
        model.config.parallelism.use_torch_compile=false \\
        model.config.joint_attn_implementation=flex \\
        model.config.tokenizer.bucket_name=nv-00-10206-checkpoint \\
        model.config.tokenizer.object_store_credential_path_pretrained=credentials/gcp_checkpoint.secret \\
        dataloader_train.data_source.num_video_frames=61 \\
        'dataloader_train.data_source.jsonl_paths=[/path/to/video_dataset_file.jsonl]'

See also: ``launch_vfm_datapacker.sh`` for a ready-to-run shell script.
"""

from __future__ import annotations

import math

import torch
from hydra.core.config_store import ConfigStore

from cosmos.utils.lazy_config import LazyCall as L
from cosmos.utils.lazy_config import LazyDict
from cosmos.data.vfm.data_packer import DataPacker
from cosmos.data.vfm.data_packer_dataloader import DataPackerDataLoader
from cosmos.data.vfm.local_datasets.sft_dataset import get_sft_dataset as _get_sft_dataset

cs = ConfigStore.instance()

# ---------------------------------------------------------------------------
# Custom DataPacker for SFTDataset output
#
# SFTDataset already performs S3 download, ffmpeg decode, and tokenization.
# This packer is a thin adapter: sft_process_sample passes the sample through
# unchanged; compute_num_tokens reads the video shape directly to compute the
# VAE token budget; sft_collate_fn assembles the batch in the format expected
# by OmniMoTModel (matching PackingDataLoader's custom_collate_fn output).
#
# OSS users writing their own DataPacker for a custom dataset format should
# follow this pattern: subclass DataPacker and place the class here.
# ---------------------------------------------------------------------------


class SFTDataPacker(DataPacker):
    """DataPacker adapter for SFTDataset output.

    SFTDataset yields fully processed samples with pre-decoded video tensors
    and pre-tokenized text.  This packer passes them through unchanged, counts
    tokens from the video shape, and collates into the format OmniMoTModel
    expects (matching PackingDataLoader's custom_collate_fn output).
    """

    def __init__(
        self,
        tokenizer_spatial_compression_factor: int = 16,
        tokenizer_temporal_compression_factor: int = 4,
        patch_spatial: int = 2,
    ):
        self.spatial_compression = tokenizer_spatial_compression_factor
        self.temporal_compression = tokenizer_temporal_compression_factor
        self.patch_spatial = patch_spatial

    def sft_process_sample(self, item: dict) -> dict:
        # This is a special case because SFTDataset has already handle all the preprocessing logic.
        # So we just return the item.
        return item

    def compute_num_tokens(self, sample: dict) -> int:
        tokens = 1 + len(sample.get("text_token_ids", []))
        v = sample.get("video")
        if v is not None:
            _, T, H, W = v.shape
            latent_h = math.ceil(H / (self.spatial_compression * self.patch_spatial))
            latent_w = math.ceil(W / (self.spatial_compression * self.patch_spatial))
            latent_t = 1 + (T - 1) // self.temporal_compression
            tokens += latent_h * latent_w * latent_t + 2
        return tokens

    def sft_collate_fn(self, samples: list, max_len: int, ignore_label_id: int = -100) -> dict:
        return {
            # Nested list [[tensor]] to match PackingDataLoader's _MULTI_ITEM_KEYS append format
            "text_token_ids": [[s["text_token_ids"]] for s in samples],
            "video": [s.get("video") for s in samples],
            "padding_mask": [s.get("padding_mask") for s in samples],
            "image_size": [s.get("image_size") for s in samples],
            "fps": torch.tensor([float(s.get("fps", 0.0)) for s in samples]),
            "conditioning_fps": torch.tensor([float(s.get("conditioning_fps", 0.0)) for s in samples]),
            "ai_caption": [s.get("ai_caption", "") for s in samples],
        }


# ---------------------------------------------------------------------------
# get_sft_dataset wrapper that disables internal DP sharding
#
# SFTDataset auto-detects dp_rank from torch.distributed when shard_world_size
# is None (the default).  DataPackerDataLoader's _IterableWrapper also shards
# by (dp_rank × num_workers), so using the bare SFTDataset would double-shard.
# Setting shard_world_size=1 makes SFTDataset iterate all items; _IterableWrapper
# handles DP sharding exclusively.
# ---------------------------------------------------------------------------


def get_sft_dataset_no_dp(**kwargs):
    dataset = _get_sft_dataset(**kwargs)
    dataset.shard_world_size = 1
    dataset.shard_rank = 0
    dataset.shard_id = 0
    return dataset


# ---------------------------------------------------------------------------
# Experiment config
# ---------------------------------------------------------------------------

t2w_sft_8b_local_datapacker = LazyDict(
    dict(
        defaults=[
            {"override /model": "mot_fsdp"},
            {"override /optimizer": "adamw"},
            {"override /scheduler": "lambdacosine"},
            {"override /checkpoint": "local"},
            {"override /callbacks": ["basic", "optimization", "job_monitor"]},
            {"override /ema": "power"},
            {"override /tokenizer": "wan2pt2_tokenizer"},
            {"override /vlm_config": "qwen3_vl_mot_vlm_8b_instruct_gcp"},
            {"override /cluster": "gcp_iad_gb200"},
            "_self_",
        ],
        job=dict(
            project="cosmos3_vfm",
            group="t2w_mot_sft_runs",
            name="t2w_mot_sft_example_local_datapacker",
            wandb_mode="disabled",
        ),
        trainer=dict(
            max_iter=2000,
            logging_iter=10,
            run_validation=False,
            seed=42,
            callbacks=dict(
                iter_speed=dict(every_n=10, hit_thres=50),
                grad_clip=dict(clip_norm=0.1, force_finite=True),
            ),
        ),
        model=dict(
            config=dict(
                state_ch=48,
                state_t=300,
                latent_downsample_factor=16,
                resolution="720",
                max_num_tokens_after_packing=45056,
                joint_attn_implementation="two_way",
                vision_gen=True,
                action_gen=False,
                sound_gen=False,
                # VLMConfig schema unified (commit 63161d8b80 "v12 + dingm"):
                # `load_pretrained` collapsed into `pretrained_weights: PretrainedWeightsConfig`.
                vlm_config=dict(
                    pretrained_weights=dict(enabled=False),
                    tokenizer=dict(config_variant="hf"),
                ),
                # Local-only mode: bypass GCP object-store bucket/creds interpolations
                # from the wan2pt2_tokenizer default + gcp_iad_gb200 cluster.
                tokenizer=dict(
                    bucket_name="",
                    object_store_credential_path_pretrained="",
                ),
                diffusion_expert_config=dict(
                    patch_spatial=2,
                    position_embedding_type="unified_3d_mrope",
                    unified_3d_mrope_reset_spatial_ids=True,
                    enable_fps_modulation=True,
                    base_fps=24,
                    unified_3d_mrope_temporal_modality_margin=15000,
                    load_weights_from_pretrained=True,
                ),
                rectified_flow_training_config=dict(
                    train_time_video_distribution="waver",
                    train_time_image_distribution="logitnormal",
                    train_time_weight="uniform",
                    loss_scale=1.0,
                    image_loss_scale=1.0,
                    use_discrete_rf=False,
                    use_dynamic_shift=False,
                    use_high_sigma_strategy=False,
                    shift={"256": 1, "480": 2, "720": 3},
                ),
                parallelism=dict(
                    data_parallel_shard_degree=8,
                    use_activation_checkpointing=True,
                    use_torch_compile=True,
                    compiled_region="language",
                    compile_dynamic=True,
                    precision="bfloat16",
                ),
            ),
        ),
        optimizer=dict(
            lr=1e-6,
            weight_decay=1e-4,
            betas=[0.9, 0.95],
            eps=1e-6,
            fused=True,
            keys_to_select=["moe_gen", "time_embedder", "vae2llm", "llm2vae"],
        ),
        scheduler=dict(
            warm_up_steps=[200],
            cycle_lengths=[2000],
            f_max=[1.0],
            f_min=[0.1],
            f_start=[0.0],
        ),
        checkpoint=dict(
            save_iter=100,
            load_path="outputs/checkpoints/midtrain",
            load_training_state=False,
            strict_resume=True,
            keys_to_skip_loading=["net_ema."],
            # Local-only mode: disable the gcp checkpoint preset's object-store IO.
            load_from_object_store=dict(enabled=False),
            save_to_object_store=dict(enabled=False),
        ),
        upload_reproducible_setup=False,
        # ---------------------------------------------------------------------------
        # Dataloader — DataPackerDataLoader with SFTDataPacker
        #
        # data_source: get_sft_dataset_no_dp disables SFTDataset's internal DP
        #   sharding so _IterableWrapper in DataPackerDataLoader handles it.
        # data_packer: SFTDataPacker (defined above) is the custom DataPacker
        #   that bridges SFTDataset output to the packing engine.
        # ---------------------------------------------------------------------------
        dataloader_train=L(DataPackerDataLoader)(
            data_source=L(get_sft_dataset_no_dp)(
                jsonl_paths=["outputs/sft_dataset_bridge/train/video_dataset_file.jsonl"],
                resolution="480",
                num_video_frames=61,
                frame_selection_mode="first",
                temporal_interval_mode="max_30fps",
                cfg_dropout_rate=0.1,
                cfg_dropout_keep_metadata=False,
                use_system_prompt=False,
                conditioning_fps=-1,
                conditioning_fps_noise_std=0.0,
                append_duration_fps_timestamps=True,
                append_resolution_info=True,
                sample_by_window=False,
                caption_suffix="",
                temporal_compression_factor=4,
                tokenizer_config="${model.config.vlm_config.tokenizer}",
            ),
            data_packer=L(SFTDataPacker)(
                tokenizer_spatial_compression_factor="${model.config.tokenizer.spatial_compression_factor}",
                tokenizer_temporal_compression_factor="${model.config.tokenizer.temporal_compression_factor}",
                patch_spatial="${model.config.diffusion_expert_config.patch_spatial}",
            ),
            max_tokens="${model.config.max_num_tokens_after_packing}",
            pool_size=16,
            max_batch_size=1,
            num_workers=4,
            prefetch_factor=4,
            persistent_workers=True,
            pin_memory=True,
        ),
        dataloader_val=None,
    ),
    flags={"allow_objects": True},
)

for _item in [t2w_sft_8b_local_datapacker]:
    _name = [k for k, v in globals().items() if v is _item][0]
    _item["job"]["name"] = _name + "_${now:%Y-%m-%d}_${now:%H-%M-%S}"
    cs.store(group="experiment", package="_global_", name=_name, node=_item)
