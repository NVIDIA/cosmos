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


from typing import Any, Literal

import attrs

from cosmos3._src.imaginaire.config import Config
from cosmos3._src.imaginaire.lazy_config import LazyDict
from cosmos3._src.vfm.configs.base.defaults.ema import EMAConfig
from cosmos3._src.vfm.configs.base.defaults.vlm import VLMConfig
from cosmos3._src.vfm.configs.base.vlm.defaults.training import PolicyConfig, TrainConfig


@attrs.define(slots=False)
class ModelConfig:
    """Typed base for project model configs.

    Subclasses override validate to add family-specific checks. The receiver is
    a fresh attrs copy from OmegaConf.to_object, so mutations to self are
    discarded; write through root_config for any propagating side effects.

    The ema field is required (disabled by default) so trainer/callback reads
    stay as model.config.ema.enabled across every family; subclasses that opt in
    (e.g. OmniMoTModelConfig) override with their own EMAConfig.
    """

    ema: EMAConfig = EMAConfig(enabled=False)

    def validate(self, root_config: Config) -> None:
        return


@attrs.define(slots=False)
class DiffusionExpertConfig:
    # This determines the range of timesteps before the fourier feature embedding is applied.
    timestep_range: float = 1.0
    # Whether to load the generation pathway weights from pretrained LLM/VLM weights.
    load_weights_from_pretrained: bool = True

    patch_spatial: int = 2
    max_vae_latent_side_after_patchify: int = (
        20  # Max dimension (h or w) of the VAE latent after patchification (320/(8*2))
    )
    # Position embedding type for vision tokens:
    #   - "3d_rope": Additive 3D RoPE embeddings (VideoRopePosition3DEmb) + 1D position IDs for attention
    #   - "flattened_sin_cos": Additive flattened sin/cos embeddings + 1D position IDs for attention
    #   - "unified_3d_mrope": No additive embedding + 3D position IDs for Qwen3VL-style mRoPE attention
    position_embedding_type: str = "3d_rope"
    # When finetuning from lower resolution to higher resolution, the spatial resolution of videos increase.
    # So, we need to adjust the position embedding.
    # We use NTK based RoPE extrapolation to adjust the position embedding.
    # Reference: (https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/)
    # Design adapted from Cosmos2.5 (https://arxiv.org/pdf/2511.00062)
    # extrapolation_ratio here is how the base of the RoPE is scaled
    # b' = b * extrapolation_ratio^(dim / (dim - 2))
    rope_h_extrapolation_ratio: float = 1.0
    rope_w_extrapolation_ratio: float = 1.0
    rope_t_extrapolation_ratio: float = 1.0
    enable_fps_modulation: bool = False
    base_fps: int = 24
    # For unified_3d_mrope: whether spatial (H, W) indices reset to 0 for each vision segment
    unified_3d_mrope_reset_spatial_ids: bool = True
    # Setting the temporal gap on the boundary of the different modalities, default is 0, using a value greater than 0 will add an additional offset on the accumulated temporal offset.
    unified_3d_mrope_temporal_modality_margin: int = 0


@attrs.define(slots=False)
class ParallelismConfig:
    # Activation checkpointing is used to reduce the memory usage of the model.
    # The outputs of each layer are checkpointed, the intermediate results are not saved.
    use_activation_checkpointing: bool = False

    # Torch compile is used to compile the model for faster training.
    use_torch_compile: bool = False

    # Whether to use CUDA graphs for faster inference. This option does not work during training.
    use_cuda_graphs: bool = False

    # Whether the entire Cosmos3 VFM network is compiled, or only a specific region is compiled.
    # Use "language" to compile only individual layers in the MOT model.
    # Use "all" to compile the the MOT model, as well as encode/decode functions.
    compiled_region: Literal["all", "language"] = "language"

    # Whether torch.compile should generate symbolic-shape (dynamic) kernels
    # (maps to ``torch.compile(dynamic=...)``).  Defaults to True for training,
    # which sees varying shapes across batches (sequence length, CP sharding, ...);
    # specializing would recompile continuously.  See ParallelismOverrides in
    # packages/cosmos3/cosmos3/common/args.py for the inference-side rationale
    # (where dynamic=False is preferred for stable AR shapes).
    compile_dynamic: bool = True

    # Enable autotuning for pointwise/reduction Triton kernels (e.g. RMSNorm).
    # Explores 6 candidate configs instead of the default 1, improving kernel performance
    # at the cost of longer first-iteration compilation time.
    max_autotune_pointwise: bool = False

    # Enable coordinate descent tuning after autotuning. Starts from the best autotuned
    # config and explores nearby configs by adjusting one parameter at a time.
    # Requires max_autotune_pointwise=True to have effect on reduction kernels.
    coordinate_descent_tuning: bool = False

    # Whether to enable inference mode.
    enable_inference_mode: bool = False

    # Number of ranks for sharding the model weights.
    data_parallel_shard_degree: int = 1

    # Number of ranks for context parallelism.
    context_parallel_shard_degree: int = 1

    # Number of ranks for CFG parallelism.
    cfg_parallel_shard_degree: int = 1

    # Precision for the model.
    precision: str = "bfloat16"


@attrs.define(slots=False)
class LBLConfig:
    # For load balancing loss computation.
    # - "local": Use the fraction of tokens routed to each expert only for the local rank.
    # - "global": Use the fraction of tokens routed to each expert across all ranks.
    method: str = "local"

    # Coefficients for the load balancing loss.
    # - "und": Coefficient for the load balancing loss for the "und" pathway.
    # - "gen": Coefficient for the load balancing loss for the "gen" pathway.
    coeff_und: float | None = None
    coeff_gen: float | None = None


@attrs.define(slots=False)
class RectifiedFlowTrainingConfig:
    shift: Any = 5  # Training time shift. If dict, maps resolution (str) to shift value (int)
    use_dynamic_shift: bool = False  # Whether to use dynamic shifting
    train_time_image_distribution: str = "logitnormal"  # Training time distribution for images
    train_time_video_distribution: str = "logitnormal"  # Training time distribution for videos
    train_time_action_distribution: str = "logitnormal"  # Training time distribution for actions
    train_time_sound_distribution: str = "logitnormal"  # Training time distribution for sound
    train_time_weight: str = "uniform"  # Training time weight
    loss_scale: float = 1.0  # Loss scale
    image_loss_scale: float | None = None  # If set, overrides loss_scale for images
    sound_loss_scale: float | None = None  # If set, overrides loss_scale for sound
    use_high_sigma_strategy: bool = False  # Whether to use high sigma strategy
    high_sigma_ratio: float = 0.05  # Ratio of using high sigmas
    high_sigma_timesteps_min: int = 995  # Minimum timestep for high sigma
    high_sigma_timesteps_max: int = 1000  # Maximum timestep for high sigma
    use_discrete_rf: bool = False  # Whether to use discrete formulation of rectified flow

    # user: please adjust this value according to loss_scale to balance the action loss with the video loss.
    # default is 10.0 to align with previous training settings.
    action_loss_weight: float = 10.0

    # Independent noise schedule for action. When False (default), action shares the sigma
    # sampled from the vision RF on every step — legacy behavior. When True, action samples
    # its own sigma from `rectified_flow_action` using `shift_action` and
    # `use_high_sigma_strategy_action`. Action always uses a shared scalar sigma per sample
    # ([B,1]), independent of vision's DF mode. If action opts in to the high-sigma strategy,
    # it reuses the global ratio / min / max.
    independent_action_schedule: bool = False
    shift_action: int | None = None  # must be int; None → inherit `shift` (which must also be int)
    use_high_sigma_strategy_action: bool = False

    # When True, per-instance flow-matching loss is normalized by the count of
    # active (noisy) elements rather than all elements — preserves sum/active_count
    # semantics so conditioning-heavy samples (e.g. I2V, forward_dynamics, diffusion
    # forcing, AR rollout teacher-forcing) contribute gradient on par with K=0
    # samples. With .mean() the gradient of a K-conditioned sample is scaled by
    # (T-K)/T, which undertrains the attend-to-clean-history dynamics. Kept
    # False by default to preserve legacy loss magnitudes; enable for AR/DF training.
    normalize_loss_by_active: bool = False


@attrs.define(slots=False)
class RectifiedFlowInferenceConfig:
    scheduler_type: str = "unipc"  # Scheduler type
    num_train_timesteps: int = 1000
    shift: int = 1
    use_dynamic_shifting: bool = False


@attrs.define(slots=False)
class FixedStepSamplerConfig:
    """Config for the fixed-step sampler used by distilled models.

    Uses a fixed sigma schedule instead of a smooth multi-step solver.

    Mirrors the constructor args of ``FixedStepSampler``.
    """

    # Discrete noise-level schedule (descending, excluding the final 0.0 step).
    # Convention: exclude the final 0.0 step — FixedStepSampler appends it automatically.
    # Values must be descending. Using 0.999 instead of 1.0 avoids numeric edge cases at sigma=1.
    t_list: list[float] = [0.999, 0.75, 0.5, 0.25]
    # Integrator type: "ode" (deterministic Euler) or "sde" (stochastic re-noising at each step).
    sample_type: str = "ode"


# Don't have any defaults and init only in config file.
@attrs.define(slots=False)
class OmniMoTModelConfig(ModelConfig):
    """
    Config for Omni MoT model.
    """

    tokenizer: LazyDict = None
    net: LazyDict = None
    ema: EMAConfig = EMAConfig()
    parallelism: ParallelismConfig = ParallelismConfig()

    # Rectified flow configs
    rectified_flow_training_config: RectifiedFlowTrainingConfig = RectifiedFlowTrainingConfig()
    rectified_flow_inference_config: RectifiedFlowInferenceConfig = RectifiedFlowInferenceConfig()

    # Optional fixed-step sampler for distilled models (None for base models).
    fixed_step_sampler_config: FixedStepSamplerConfig | None = None

    # Model configs
    vlm_config: VLMConfig = VLMConfig()
    diffusion_expert_config: DiffusionExpertConfig = DiffusionExpertConfig()
    # Training data keys
    input_video_key: str = "video"
    input_image_key: str = "images"  # key to fetch input image from data_batch
    input_caption_key: str = "ai_caption"  # Key used to fetch input captions

    # State and sequence shapes
    state_ch: int = 16  # for latent model, ref to the latent channel number
    state_t: int = 8  # for latent model, ref to the latent number of frames
    latent_downsample_factor: int = 8
    resolution: str = "512"
    max_num_tokens_after_packing: int = 13312  # Final num tokens after sequence packing

    # Attention implementation for joint understanding + generation
    # Note "two_way" and "three_way" disallow and remove "End-of-Vision" or other text token in the generation tower.
    # "three_way" must only be used when introducing sparsity
    joint_attn_implementation: str = (
        "two_way"  # "two_way", "three_way" or "flex" (NOTICE: We are planning to remove "flex" soon)
    )

    # Per-layer NATTEN parameters
    # Must use "three_way" attention if used.
    # If None, all attention layers remain dense.
    # If not None, must be a list exactly the size of number of layers, and each layer can be either
    # None (dense) or a dictionary, with at least 'kernel_size' or 'kernel_size_float' keys
    # specifying sparsity. NATTEN parameters 'dilation' and 'stride' may also be specified either as
    # static integers, or as floating point values that will be mapped to their domain during
    # runtime. Integer parameters should never be mixed with floating point ones.
    #
    # Floating point parameters are highly recommended, unless the use case will have a fixed token
    # layout (input resolution).
    #
    # Examples:
    #   Interleaved sliding window layers, "GPT-OSS"-style, with static window size:
    #     natten_parameter_list = [None if layer_idx % 2 != 0 else {"kernel_size": (8, 8)}]
    #   Layers with odd indices ("None"s) will use dense attention, and layers with an even indices
    #   will use a static sliding window size of 8x8.
    #
    #   Interleaved sliding window layers, "GPT-OSS"-style, with input-dependent window size:
    #     natten_parameter_list = [None if layer_idx % 2 != 0 else {"kernel_size_float": (0.5, 0.5)}]
    #   Layers with odd indices ("None"s) will use dense attention, and layers with an even indices
    #   will use a dynamic window size that is 50% of the input along each of the two dimensions.
    #
    #   Interleaved sliding window and dilated layers, "DiNAT"-style:
    #     natten_parameter_list = [
    #       {
    #           "kernel_size_float": (0.5, 0.5),
    #           "dilation_float": (1.0, 1.0),
    #       } if layer_idx % 2 != 0 else {
    #           "kernel_size_float": (0.5, 0.5),
    #       }
    #     ]
    #   All layers will use a dynamic window size that is 50% of the input along each of the two
    #   dimensions. Layers with odd indices will also dilate to the maximum level possible.
    #
    natten_parameter_list: list | None = None

    # Temporal causality for training autoregressive video generation models.
    # When enabled, applies temporal causal attention to generation supertokens.
    # Each supertoken is num_action_tokens_per_supertoken action tokens followed
    # by H*W vision tokens; the value is stamped onto the packed sequence by the
    # temporal-causal packer and read by attention/KV-cache code unchanged.
    # Only supports image2video modes (with or without actions).
    # Requires joint_attn_implementation="three_way".
    video_temporal_causal: bool = False
    # "none":             standard joint denoising (shared σ, no clean context)
    # "teacher_forcing":  all frames noised with shared σ; clean history via cross-attention
    # "diffusion_forcing": each latent frame gets independent σ ~ Uniform[0,1]
    # "teacher_forcing_dcm": replayed teacher-forcing discrete-time consistency distillation
    causal_training_strategy: Literal["none", "teacher_forcing", "diffusion_forcing", "teacher_forcing_dcm"] = "none"

    # Load balancing loss config.
    lbl: LBLConfig = LBLConfig()

    # vision configs
    vision_gen: bool = True  # whether to use vision related parameters and condition/generate vision tokens

    # action configs
    action_gen: bool = False  # whether to use action related parameters and condition/generate action tokens
    max_action_dim: int = 32  # maximum dimension of the action space, we need to pad the data to this dimension.
    num_embodiment_domains: int = 32  # number of domains for the domain-aware linear layer

    # sound configs
    sound_gen: bool = False  # whether to use sound related parameters and condition/generate sound tokens
    sound_tokenizer: LazyDict | None = None  # Sound tokenizer config (e.g., AVAE)
    sound_dim: int | None = None  # Sound latent channel size (e.g., 64 for AVAE 48kHz)
    sound_latent_fps: int = 25  # Sound tokenizer's latent rate (e.g., 48kHz / 1920 hop = 25 Hz)

    log_enc_time_every_n: int = 100  # Frequency of logging encoding time to W&B

    def validate(self, root_config: Config) -> None:
        """Skip pretrained loading if a training checkpoint exists.

        Mutates root_config.model.config.* directly because the receiver self
        is a fresh attrs copy from OmegaConf.to_object and its writes would be
        dropped.
        """
        from cosmos3._src.imaginaire.utils import log
        from cosmos3._src.vfm.checkpointer.dcp import DistributedCheckpointer

        # There are three cases to consider:
        # 1. Model is being trained from scratch (using weights from Hugging Face).
        #    (both _read_latest_checkpoint_file() and load_path are None).
        #    In this case, we should load the understanding pathway weights from HF weights,
        #    Additionally, we must copy the understanding pathway weights to the generation
        #    pathway.
        #
        # 2. Model is being trained from a previous checkpoint.
        #    (_read_latest_checkpoint_file() is not None and load_path can be None or not).
        #    In this case, the model weights have been already loaded from DCP checkpoint
        #    (checkpointer/dcp.py). We must skip both loading understanding pathway weights,
        #    and copying the understanding pathway weights to the generation pathway.

        # 3. Model is being warm-started from a load_path (but no previous checkpoint exists).
        #    (_read_latest_checkpoint_file() is None and load_path is not None).
        #    In this case, the model weights have been already loaded from DCP checkpoint
        #    due to load_path being specified (checkpointer/dcp.py). However, we must still
        #    load the understanding weights from HF weights (since the understanding model
        #    may be moved from Qwen3-VL to Cosmos-Reason2 for example). We should not copy
        #    the understanding pathway weights to the generation pathway (since the generation
        #    pathway has already been pretrained using the previous model weights, for example,
        #    the Qwen3-VL weights). But the understanding weights are always kept unchanged.

        if not self.vlm_config.load_pretrained and not self.diffusion_expert_config.load_weights_from_pretrained:
            # Neither if branch below is taken; no need to create checkpointer.
            return

        checkpointer = DistributedCheckpointer(
            root_config.checkpoint, root_config.job, callbacks=None, disable_async=True
        )

        if self.vlm_config.load_pretrained:
            if checkpointer._read_latest_checkpoint_file() is not None:
                log.info(
                    "Checkpoint found: disabling pretrained model loading to avoid double loading. "
                    "Model weights will be loaded from checkpoint instead of safetensors."
                )
                root_config.model.config.vlm_config.load_pretrained = False

            if self.diffusion_expert_config.load_weights_from_pretrained:
                if checkpointer.load_path is not None:
                    log.info(
                        "Load path found: disabling pretrained model loading for generation pathway. "
                        "Generation pathway weights will be loaded from load_path instead of safetensors."
                    )
                    root_config.model.config.diffusion_expert_config.load_weights_from_pretrained = False


@attrs.define(slots=False)
class VLMModelConfig(ModelConfig):
    """
    Config for VLM model.
    """

    policy: PolicyConfig = PolicyConfig()
    train: TrainConfig = TrainConfig()
