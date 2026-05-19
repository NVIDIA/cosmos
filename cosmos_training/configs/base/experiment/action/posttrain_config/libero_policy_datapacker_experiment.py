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

"""LIBERO policy SFT — 8B, DataPackerDataLoader variant.

Equivalent to ``action_policy_sft_8b`` but replaces the 3-layer
  IterativeJointDataLoader → InfiniteDataLoader → ActionUnifiedIterableDataset
with the single-layer OSS-facing DataPackerDataLoader + ActionDataPacker.

Batch semantics are preserved: ``max_batch_size=256`` caps the sample count;
``max_tokens=999_999`` is set high so the token budget never fires first.

Usage (smoke test — same overrides as launch_action_from_vfm.sh)::

    LIBERO_LOCAL_DATA_ROOT=outputs/libero_datasets torchrun \\
        --nproc_per_node=4 --master_port=12342 -m scripts.train \\
        --config=configs/base/config.py -- \\
        experiment=action_policy_sft_8b_datapacker \\
        trainer.max_iter=10 trainer.logging_iter=1 \\
        job.group=debug job.wandb_mode=disabled \\
        upload_reproducible_setup=false \\
        checkpoint=local ckpt_type=dummy \\
        model.config.parallelism.data_parallel_shard_degree=-1 \\
        model.config.parallelism.use_torch_compile=false
"""

from __future__ import annotations

import copy
import math

import torch
from hydra.core.config_store import ConfigStore

from cosmos.utils.lazy_config import LazyCall as L
from configs.base.experiment.action.posttrain_config.libero_policy_experiment import (
    _LIBERO_REPO_IDS,
    _LIBERO_ROOTS,
    _build_libero_policy_base_8b,
)
from cosmos.data.vfm.action.libero_dataset import LIBERODataset
from cosmos.data.vfm.action.transforms import ActionTransformPipeline
from cosmos.data.vfm.action.unified_dataset import MapToIterableAdapter
from cosmos.data.vfm.data_packer import DataPacker
from cosmos.data.vfm.data_packer_dataloader import DataPackerDataLoader

cs = ConfigStore.instance()


# ---------------------------------------------------------------------------
# data_source factory
#
# DataPackerDataLoader requires any Python iterable.  LIBERODataset is
# map-style (torch.utils.data.Dataset), so we wrap it in MapToIterableAdapter
# which yields uniformly random items on each iteration.
# ---------------------------------------------------------------------------


def get_libero_iterable_dataset(**kwargs) -> MapToIterableAdapter:
    """Wrap LIBERODataset in MapToIterableAdapter for use as DataPackerDataLoader data_source.

    MapToIterableAdapter yields random items indefinitely, making the
    map-style LIBERODataset compatible with _IterableWrapper's DP sharding.
    """
    return MapToIterableAdapter(LIBERODataset(**kwargs))


# ---------------------------------------------------------------------------
# ActionDataPacker
#
# OSS users writing a DataPacker for a custom robot dataset should follow
# this pattern: subclass DataPacker, implement three methods, and place the
# class next to the experiment config that uses it.
# ---------------------------------------------------------------------------


class ActionDataPacker(DataPacker):
    """DataPacker adapter for LIBERODataset + ActionTransformPipeline.

    Bridges raw LIBERODataset items into the DataPackerDataLoader packing
    engine with the same transform pipeline and batch format as the
    IterativeJointDataLoader-based ``action_policy_sft_8b`` experiment.

    Three responsibilities:
    - ``sft_process_sample``: run the full ActionTransformPipeline on a
      raw LIBERODataset sample (resize, tokenize, pad action, build SequencePlan).
    - ``compute_num_tokens``: count video + text tokens for packing budget.
    - ``sft_collate_fn``: assemble the batch dict OmniMoTModel expects.
    """

    def __init__(
        self,
        tokenizer_spatial_compression_factor: int = 16,
        tokenizer_temporal_compression_factor: int = 4,
        patch_spatial: int = 2,
        tokenizer_config=None,
        cfg_dropout_rate: float = 0.1,
        max_action_dim: int = 64,
        action_channel_masking: bool = True,
    ) -> None:
        self._spatial = tokenizer_spatial_compression_factor
        self._temporal = tokenizer_temporal_compression_factor
        self._patch = patch_spatial
        self._transform = ActionTransformPipeline(
            pad_keys=["video"],
            tokenizer_config=tokenizer_config,
            cfg_dropout_rate=cfg_dropout_rate,
            max_action_dim=max_action_dim,
            action_channel_masking=action_channel_masking,
            append_duration_fps_timestamps=True,
            append_resolution_info=True,
        )

    def sft_process_sample(self, item: dict) -> dict:
        """Apply ActionTransformPipeline to one raw LIBERODataset sample."""
        return self._transform(item, resolution=None)

    def compute_num_tokens(self, sample: dict) -> int:
        """Count video latent tokens + text tokens for the packing budget."""
        tokens = 1 + len(sample.get("text_token_ids", []))
        v = sample.get("video")  # [C,T,H,W] or None
        if v is not None:
            _, T, H, W = v.shape  # [C,T,H,W]
            latent_h = math.ceil(H / (self._spatial * self._patch))
            latent_w = math.ceil(W / (self._spatial * self._patch))
            latent_t = 1 + (T - 1) // self._temporal
            tokens += latent_h * latent_w * latent_t + 2
        return tokens

    def sft_collate_fn(self, samples: list, max_len: int, ignore_label_id: int = -100) -> dict:
        """Assemble a list of processed samples into the OmniMoTModel batch dict.

        Mirrors the format produced by InfiniteDataLoader + custom_collate_fn
        in action_policy_sft_8b so OmniMoTModel.training_step() needs no changes.
        Variable-length or custom-type fields are kept as lists; scalar numeric
        fields are stacked into tensors.
        """
        return {
            # Nested list [[tensor]] matches PackingDataLoader's _MULTI_ITEM_KEYS format
            "text_token_ids": [[s["text_token_ids"]] for s in samples],
            "video": [s.get("video") for s in samples],
            "action": [s.get("action") for s in samples],
            "padding_mask": [s.get("padding_mask") for s in samples],
            "image_size": [s.get("image_size") for s in samples],
            "fps": torch.tensor([float(s.get("fps", 0.0)) for s in samples]),  # [N]
            "domain_id": [s.get("domain_id") for s in samples],
            "sequence_plan": [s.get("sequence_plan") for s in samples],
            "raw_action_dim": [s.get("raw_action_dim") for s in samples],
            "ai_caption": [s.get("ai_caption", "") for s in samples],
        }


# ---------------------------------------------------------------------------
# Experiment registration
# ---------------------------------------------------------------------------


def action_policy_sft_8b_datapacker_experiments() -> None:
    """Register action_policy_sft_8b_datapacker in Hydra ConfigStore.

    Builds from the same base as action_policy_sft_8b (identical model,
    optimizer, scheduler, checkpoint, dataset params), then replaces
    dataloader_train with DataPackerDataLoader + ActionDataPacker.
    """
    # Build from the same base as action_policy_sft_8b.
    # batch_size/num_workers here configure the (discarded) IterativeJointDataLoader;
    # DataPackerDataLoader's own num_workers=4 is set below when replacing dataloader_train.
    exp = _build_libero_policy_base_8b(
        "action_policy_sft_8b_datapacker",
        training_iterations=16_000,
        batch_size=256,
        num_workers=4,
        job_group="action_libero",
        job_project="cosmos3_action_libero",
        mode="policy",
        keys_to_skip_loading=["net_ema.", "action2llm", "llm2action", "action_modality_embed", "action_pos_embed"],
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

    # Post-build overrides — identical to action_policy_sft_8b
    exp["scheduler"]["f_max"] = [1.0]
    exp["scheduler"]["f_min"] = [0.0]
    exp["scheduler"]["warm_up_steps"] = [500]
    exp["scheduler"]["cycle_lengths"] = [20_000]
    exp["optimizer"]["lr"] = 5e-5
    exp["trainer"]["max_iter"] = 16_000
    exp["trainer"]["logging_iter"] = 100
    exp["trainer"]["run_validation"] = False
    exp["trainer"]["run_validation_on_start"] = False
    exp["checkpoint"]["load_path"] = "outputs/checkpoints/action_policy_sft_8b"
    exp["checkpoint"]["load_training_state"] = False
    exp["checkpoint"]["save_iter"] = 500
    exp["trainer"]["callbacks"]["compile_tokenizer"]["warmup_resolutions"] = ["256", "480", "720"]
    exp["model"]["config"]["rectified_flow_training_config"]["action_loss_weight"] = 10.0
    exp["model"]["config"]["num_embodiment_domains"] = 32

    # Local-only mode: bypass internal GCP object-store paths in favor of the
    # local checkpoint_path / wan_vae_path declared in toml/launch_action_libero.toml
    # and the HF tokenizer/weights stack. Without these, defaults pull the GCP
    # bucket + qwen3_vl_mot_vlm_8b_instruct_gcp variant.
    exp["model"]["config"]["tokenizer"]["bucket_name"] = ""
    exp["model"]["config"]["tokenizer"]["object_store_credential_path_pretrained"] = ""
    exp["checkpoint"]["load_from_object_store"] = dict(enabled=False)
    exp["checkpoint"]["save_to_object_store"] = dict(enabled=False)
    exp["model"]["config"]["vlm_config"]["tokenizer"] = dict(config_variant="hf")
    exp["model"]["config"]["vlm_config"]["pretrained_weights"]["enabled"] = False

    # Replace the 3-layer IterativeJointDataLoader stack with DataPackerDataLoader.
    # max_tokens=999_999 ensures the token budget never triggers;
    # max_batch_size=256 is the effective batch boundary (matches action_policy_sft_8b).
    exp["dataloader_train"] = L(DataPackerDataLoader)(
        data_source=L(get_libero_iterable_dataset)(
            repo_id=copy.deepcopy(_LIBERO_REPO_IDS),
            root=copy.deepcopy(_LIBERO_ROOTS),
            fps=20,
            camera_mode="concat_view",
            action_space="frame_wise_relative",
            rotation_space="6d",
            chunk_length=16,
            seed=0,
            val_ratio=0.01,
            mode="policy",
        ),
        data_packer=L(ActionDataPacker)(
            tokenizer_spatial_compression_factor="${model.config.tokenizer.spatial_compression_factor}",
            tokenizer_temporal_compression_factor="${model.config.tokenizer.temporal_compression_factor}",
            patch_spatial="${model.config.diffusion_expert_config.patch_spatial}",
            tokenizer_config="${model.config.vlm_config.tokenizer}",
            cfg_dropout_rate=0.1,
            max_action_dim="${model.config.max_action_dim}",
            action_channel_masking=True,
        ),
        max_tokens=999_999,
        max_batch_size=256,
        pool_size=16,
        num_workers=4,
        prefetch_factor=4,
        persistent_workers=True,
        pin_memory=True,
    )

    cs.store(group="experiment", package="_global_", name="action_policy_sft_8b_datapacker", node=exp)


action_policy_sft_8b_datapacker_experiments()
