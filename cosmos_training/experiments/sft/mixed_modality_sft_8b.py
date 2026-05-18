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

"""Mixed-modality SFT (T2V / I2V / V2V) experiment for Qwen3-VL-8B.

Python port of ``cosmos-inference/configs/experiment/mixed_modality_sft_8b.yaml``.
Mirrors the YAML structurally: defaults groups, optimizer (selecting only
generation params), warmup-cosine scheduler, OmniMoTModel with Qwen3-VL-8B
backbone and Wan2.2 VAE tokenizer, mixed conditioning (70% T2V, 20% I2V,
10% V2V).

Usage::

    torchrun --nproc_per_node=8 --master_port=12341 -m scripts.train \\
        --config=configs/base/config.py \\
        -- experiment=mixed_modality_sft_8b \\
           dataloader_train.dataloader.datasets.video.dataset.jsonl_paths='["/path/to/sft.jsonl"]' \\
           checkpoint.load_path='/path/to/pretrained/cosmos3_8b' \\
           job.group=sft
"""

from hydra.core.config_store import ConfigStore

from cosmos.utils.lazy_config import LazyCall as L
from cosmos.utils.lazy_config import LazyDict

from configs.base.defaults.vlm import create_qwen2_tokenizer_with_download

cs = ConfigStore.instance()


# ---------------------------------------------------------------------------
# mixed_modality_sft_8b — Qwen3-VL-8B SFT with mixed T2V/I2V/V2V conditioning
# ---------------------------------------------------------------------------
mixed_modality_sft_8b = LazyDict(
    dict(
        defaults=[
            # Compose preset groups (matches the YAML `defaults:` block).
            {"override /model": "mot_fsdp"},
            {"override /optimizer": "adamw"},
            {"override /scheduler": "lambdacosine"},
            {"override /checkpoint": "s3"},
            {"override /callbacks": ["basic", "optimization", "job_monitor", "generation"]},
            {"override /ema": "power"},
            {"override /tokenizer": "wan2pt2_tokenizer"},
            {"override /cluster": "gcp_iad_gb200"},
            {"override /ckpt_type": "dcp"},
            # Data: pick the open-source SFT dataloader we registered.
            # The user MUST override jsonl_paths at launch time (Hydra "???").
            {"override /data_train": "open_source_sft_video_256p"},
            # VLM backbone preset — supplies vlm_config.model_instance
            # (Qwen3VLTextForCausalLM + its nested base_config). The inline
            # vlm_config block below only sets release-specific overrides on
            # top (credentials, layer_module, etc.).
            {"override /vlm_config": "qwen3_vl_mot_vlm_8b_instruct"},
            "_self_",
        ],
        trainer=dict(
            # Smoke-variant knobs (match cosmos-inference reference YAML).
            max_iter=500,
            grad_accum_iter=2,
            logging_iter=1,
            validation_iter=100,
            run_validation=False,
            seed=42,
            compile_config=dict(use_duck_shape=False),
            callbacks=dict(
                every_n_sample_ema=dict(every_n=999_999, save_s3=False),
                every_n_sample_reg=dict(every_n=999_999, save_s3=False),
                grad_clip=dict(clip_norm=0.1),
                iter_speed=dict(every_n=1),
                manual_gc=dict(warm_up=1),
                norm_monitor=dict(every_n=100),
                sequence_packing_padding=dict(every_n=1),
                sigma_loss_analysis=dict(every_n=500, every_n_viz=500),
                wandb_2x=dict(logging_iter_multipler=1),
                # NOTE: keep the preset-added callbacks (mfu, moe_*, ofu,
                # termination_signal_checkpoint) — earlier we set them to None
                # to match the reference YAML cosmetically, but
                # cosmos/utils/callback.py iterates each entry expecting a dict.
                # Smoke knobs on those callbacks (every_n etc.) inherit preset
                # defaults — fine for this run.
            ),
        ),
        # --------------------------------------------------------------- model
        model=dict(
            config=dict(
                resolution="720",
                # Generation gates
                action_gen=True,
                vision_gen=True,
                sound_gen=False,
                # Causal / temporal
                causal_training_strategy="none",
                video_temporal_causal=False,
                # Attention / packing
                joint_attn_implementation="two_way",
                max_num_tokens_after_packing=45056,
                latent_downsample_factor=16,
                # State
                state_ch=48,
                state_t=300,
                max_action_dim=64,
                num_embodiment_domains=32,
                # Keys
                input_caption_key="ai_caption",
                input_image_key="images",
                input_video_key="video",
                sound_dim=None,
                sound_latent_fps=25,
                sound_tokenizer=None,
                log_enc_time_every_n=100,
                # Diffusion expert
                diffusion_expert_config=dict(
                    base_fps=24,
                    enable_fps_modulation=True,
                    load_weights_from_pretrained=True,
                    max_vae_latent_side_after_patchify=20,
                    patch_spatial=2,
                    position_embedding_type="unified_3d_mrope",
                    rope_h_extrapolation_ratio=1.0,
                    rope_t_extrapolation_ratio=1.0,
                    rope_w_extrapolation_ratio=1.0,
                    timestep_range=1.0,
                    unified_3d_mrope_reset_spatial_ids=True,
                    unified_3d_mrope_temporal_modality_margin=15000,
                ),
                # EMA
                ema=dict(
                    enabled=True,
                    iteration_shift=0,
                    rate=0.1,
                ),
                # Load-balancing loss
                lbl=dict(
                    coeff_gen=None,
                    coeff_und=None,
                    method="local",
                ),
                # Parallelism
                parallelism=dict(
                    cfg_parallel_shard_degree=1,
                    compile_dynamic=True,
                    compiled_region="language",
                    context_parallel_shard_degree=1,
                    coordinate_descent_tuning=False,
                    data_parallel_shard_degree=8,
                    enable_inference_mode=False,
                    max_autotune_pointwise=False,
                    precision="bfloat16",
                    use_activation_checkpointing=True,
                    use_cuda_graphs=False,
                    use_torch_compile=True,
                ),
                # Rectified-flow training config
                rectified_flow_training_config=dict(
                    action_loss_weight=10.0,
                    high_sigma_ratio=0.05,
                    high_sigma_timesteps_max=1000,
                    high_sigma_timesteps_min=995,
                    image_loss_scale=1.0,
                    independent_action_schedule=False,
                    loss_scale=1.0,
                    normalize_loss_by_active=False,
                    shift={"256": 3, "480": 5, "720": 10},
                    shift_action=None,
                    sound_loss_scale=None,
                    train_time_action_distribution="logitnormal",
                    train_time_image_distribution="logitnormal",
                    train_time_sound_distribution="logitnormal",
                    train_time_video_distribution="waver",
                    train_time_weight="uniform",
                    use_discrete_rf=False,
                    use_dynamic_shift=False,
                    use_high_sigma_strategy=False,
                    use_high_sigma_strategy_action=False,
                ),
                # Rectified-flow inference config
                rectified_flow_inference_config=dict(
                    num_train_timesteps=1000,
                    scheduler_type="unipc",
                    shift=1,
                    use_dynamic_shifting=False,
                ),
                # VLM backbone (Qwen3-VL-8B Instruct)
                # Schema unified by the v12 + dingm merge (commit 63161d8b80):
                # checkpoint_path / credential_path / load_pretrained etc. are
                # now nested under pretrained_weights: PretrainedWeightsConfig.
                # qk_norm_for_{diffusion,text} collapsed to a single qk_norm.
                # vlm_checkpoint_format → pretrained_weights.checkpoint_format.
                vlm_config=dict(
                    model_name="Qwen/Qwen3-VL-8B-Instruct",
                    layer_module="Qwen2MoTDecoderLayer",
                    qk_norm=True,
                    tie_word_embeddings=False,
                    use_system_prompt=False,
                    pretrained_weights=dict(
                        enabled=False,
                        backbone_path="s3://bucket/cosmos3/pretrained/huggingface/Qwen/Qwen3-VL-8B-Instruct/",
                        credentials_path="credentials/gcp_checkpoint.secret",
                        enable_gcs_patch_in_boto3=True,
                        checkpoint_format=None,
                    ),
                    # Use the create_qwen2_tokenizer_with_download path with
                    # config_variant="hf" so the tokenizer is fetched from
                    # HuggingFace (no GCP/S3 credentials needed). OmegaConf
                    # merges this dict with the preset's tokenizer block; extra
                    # kwargs (e.g. tokenizer_type) leak through and get
                    # absorbed by the **_unused_kwargs added in vlm.py.
                    tokenizer=L(create_qwen2_tokenizer_with_download)(
                        pretrained_model_name="Qwen/Qwen3-VL-8B-Instruct",
                        config_variant="hf",
                    ),
                ),
            ),
        ),
        # ----------------------------------------------------------- optimizer
        optimizer=dict(
            # AdamW with betas (0.9, 0.95) / weight_decay 0
            # Train ONLY generation-specific params (VLM backbone stays frozen).
            betas=(0.9, 0.95),
            eps=1.0e-6,
            fused=True,
            keys_to_select=["moe_gen", "time_embedder", "vae2llm", "llm2vae"],
            lr=2.0e-5,
            optimizer_type="AdamW",
            weight_decay=0,
        ),
        # ----------------------------------------------------------- scheduler
        scheduler=dict(
            cycle_lengths=[1000],
            f_max=[1.0],
            f_min=[0.0],
            f_start=[0.0],
            warm_up_steps=[50],
            verbosity_interval=0,
        ),
        # ---------------------------------------------------------- checkpoint
        checkpoint=dict(
            save_iter=100,
            # User MUST override load_path at launch time.
            # load_path="???",
            load_training_state=False,
            strict_resume=True,
            keys_to_skip_loading=["net_ema."],
            broadcast_via_filesystem=False,
            dcp_async_mode_enabled=False,
            enable_gcs_patch_in_boto3=True,
            load_from_object_store=dict(bucket="", credentials="", enabled=False),
            save_to_object_store=dict(bucket="", credentials="", enabled=False),
        ),
        # --------------------------------------------------------- job/cluster
        # Reference YAML had explicit empty cluster bucket/credentials —
        # mirror that here so we don't pick up the gcp_iad_gb200 preset's
        # production bucket names.
        job=dict(
            project="cosmos3",
            group="sft",
            name="mixed_modality_sft_8b",
            wandb_mode="disabled",
            cluster=dict(
                object_store_bucket_checkpoint="",
                object_store_bucket_data="",
                object_store_bucket_pretrained="",
                object_store_credential_checkpoint="",
                object_store_credential_data="",
                object_store_credential_pretrained="",
            ),
        ),
        # --------------------------------------------------------- data setting
        data_setting=dict(
            qwen_max_video_token_length=8192,
        ),
        # ---------------------------------------------- runtime infra flags
        upload_reproducible_setup=False,
    )
)


# ---------------------------------------------------------------------------
# Register the experiment under Hydra group "experiment" so it's reachable
# via `experiment=mixed_modality_sft_8b`. Mirrors the pattern used by
# pre_exp012_phase2_vlm_smoke.py.
# ---------------------------------------------------------------------------
for _item in [
    mixed_modality_sft_8b,
]:
    experiment_name = [name.lower() for name, value in globals().items() if value is _item][0]
    if "job" not in _item:
        _item["job"] = dict(name=experiment_name + "_${now:%Y-%m-%d}_${now:%H-%M-%S}")
    else:
        _item["job"]["name"] = experiment_name + "_${now:%Y-%m-%d}_${now:%H-%M-%S}"

    cs.store(group="experiment", package="_global_", name=experiment_name, node=_item)
