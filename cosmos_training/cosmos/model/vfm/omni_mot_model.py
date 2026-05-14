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

from __future__ import annotations

import collections
from contextlib import contextmanager
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
from einops import rearrange
from torch.distributed._composable.fsdp import FSDPModule
from torch.nn.modules.module import _IncompatibleKeys

from cosmos.utils.flags import DEVICE, TRAINING, Device
from cosmos.utils.lazy_config import LazyDict
from cosmos.utils.lazy_config import instantiate as lazy_instantiate
from cosmos.model._base import ImaginaireModel
from cosmos.utils import log, misc
from cosmos.utils.count_params import count_params
from cosmos.utils.timer import Timer
from cosmos.model.vfm.algorithm.loss.flow_matching import compute_flow_matching_loss
from cosmos.model.vfm.algorithm.loss.load_balancing import compute_load_balancing_loss
from configs.base.defaults.model_config import OmniMoTModelConfig
from cosmos.data.vfm.sequence_packing import (
    PackedSequence,
    SequencePlan,
    add_special_tokens,
    build_sequence_plans_from_data_batch,
    pack_input_sequence,
)
from cosmos.data.vfm.utils import VIDEO_RES_SIZE_INFO
from cosmos.model.vfm.diffusion.rectified_flow import RectifiedFlow
from cosmos.model.vfm.diffusion.samplers.edm import EDMSampler
from cosmos.model.vfm.diffusion.samplers.fixed_step import FixedStepSampler
from cosmos.model.vfm.diffusion.samplers.unipc import UniPCSampler, UniPCSamplerConfig
from cosmos.model.vfm.mot.cosmos3_vfm_network import Cosmos3VFMNetwork, Cosmos3VFMNetworkConfig
from cosmos.model.vfm.mot.modeling_utils import has_noisy_tokens
from cosmos.model.vfm.mot.parallelize_vfm_network import parallelize_vfm_network
from cosmos.model.vfm.utils.data_and_condition import (
    GenerationDataClean,
    GenerationDataNoised,
    _expand_per_sample_to_per_vision_item,
    build_dense_sound_schedule,
    unwrap_and_densify,
)
from cosmos.model.vfm.utils.memory import MemoryState
from cosmos.model.vfm.utils.safetensors_loader import load_language_model as load_language_model_safetensors
from cosmos.model.vfm.vlm.qwen3_vl.utils import tokenize_caption
from cosmos.model.vfm.tokenizers.interface import VideoTokenizerInterface
from cosmos.utils.vfm.data_utils import get_vision_data_resolution
from cosmos.utils.vfm.dtensor_helper import DTensorFastEmaModelUpdater
from cosmos.utils.vfm.model_weights_stats import WeightTrainingStat
from cosmos.utils.vfm.parallelism import ParallelDims


class OmniMoTModel(ImaginaireModel):
    """
    Mixture of Transformers (MoT) model to be trained with the flow matching objective
    for visual / sound / action generation.
    """

    def __init__(self, config: OmniMoTModelConfig):
        super().__init__()
        self.config = config
        log.info(f"OmniMoTModel: config {self.config}")
        # 0. Set up precision
        self.set_precision()

        # 1. Set data keys and data information
        self.set_up_data_key()

        # 2. Text, vision, audio, action tokenizers
        self.set_up_tokenizers()

        # 3. FSDP setup. Note: call this before building the model.
        self.set_up_parallelism()

        # 4. Build the denoiser network
        self.set_up_model()

        # 5. Set up training time scheduler and inference time sampler
        self.set_up_scheduler_and_sampler()

        self.log_enc_time_every_n = config.log_enc_time_every_n

    def set_precision(self) -> None:
        self.precision = getattr(torch, self.config.parallelism.precision)
        self.tensor_kwargs = {"device": DEVICE, "dtype": self.precision}
        self.tensor_kwargs_fp32 = {"device": DEVICE, "dtype": torch.float32}
        log.warning(f"OmniMoTModel: precision {self.precision}")

        # Disable TF32 for CUDA matrix multiplications since this may impact model quality.
        torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = False

    def set_up_data_key(self) -> None:

        self.input_video_key = self.config.input_video_key  # by default it is video key for Video diffusion model
        self.input_image_key = self.config.input_image_key
        self.input_caption_key = self.config.input_caption_key

    @misc.timer("OmniMoTModel: set_up_tokenizers")
    def set_up_tokenizers(self) -> None:
        """
        Variable names follow the naming convention:
        - tokenizer_<modality_type>_gen if used for generation branch
        - tokenizer_<modality_type>_und if used for understanding branch
        """
        # 1. Text tokenizer
        self.vlm_config = self.config.vlm_config
        _vlm_proc = lazy_instantiate(self.vlm_config.tokenizer)
        vlm_tokenizer = (
            _vlm_proc.processor.tokenizer if getattr(_vlm_proc, "is_unified_processor", False) else _vlm_proc
        )
        vlm_tokenizer, special_tokens = add_special_tokens(vlm_tokenizer)
        self.vlm_tokenizer = vlm_tokenizer

        self.llm_special_tokens = special_tokens
        self.llm_special_tokens["eos_token_id"] = vlm_tokenizer.eos_token_id

        # 2. Vision tokenizer (images/videos) for generation.
        self.tokenizer_vision_gen: VideoTokenizerInterface = lazy_instantiate(self.config.tokenizer)
        assert self.tokenizer_vision_gen.latent_ch == self.config.state_ch, (
            f"vision tokenizer latent_ch {self.tokenizer_vision_gen.latent_ch} != state_shape {self.config.state_ch}"
        )
        if hasattr(self.tokenizer_vision_gen, "reset_dtype"):
            self.tokenizer_vision_gen.reset_dtype()

        # 3. Sound/audio tokenizer (optional)
        if self.config.sound_gen:
            assert self.config.sound_tokenizer is not None, "sound_tokenizer must be provided when sound_gen is True"
            self.tokenizer_sound_gen = lazy_instantiate(self.config.sound_tokenizer)
            assert self.config.sound_dim is not None, "sound_dim must be provided when sound_gen is True"
            assert self.tokenizer_sound_gen.latent_ch == self.config.sound_dim, (
                f"sound tokenizer latent_ch {self.tokenizer_sound_gen.latent_ch} != sound_dim {self.config.sound_dim}"
            )
            if hasattr(self.tokenizer_sound_gen, "reset_dtype"):
                self.tokenizer_sound_gen.reset_dtype()
            log.info(f"Sound tokenizer initialized: {type(self.tokenizer_sound_gen).__name__}")
        else:
            self.tokenizer_sound_gen = None



    def build_net(self, dtype: torch.dtype):
        # Build model network and parallelize it.
        with torch.device("meta"):
            assert self.vlm_config.model_instance is not None, "Model instance should be specified"

            language_model = lazy_instantiate(self.vlm_config.model_instance)

            # (i.e., roughly [0, num_train_timesteps]). The MoT network expects to internally
            # rescale timesteps before embedding; avoid hard-coding 1e-3 by computing it from
            # the configured scheduler resolution.
            num_train_timesteps = self.config.rectified_flow_inference_config.num_train_timesteps
            network_config = Cosmos3VFMNetworkConfig(
                vlm_config=language_model.config,
                latent_patch_size=self.config.diffusion_expert_config.patch_spatial,
                latent_downsample_factor=self.config.latent_downsample_factor,
                latent_channel_size=self.config.state_ch,
                max_latent_h=self.config.diffusion_expert_config.max_vae_latent_side_after_patchify,
                max_latent_w=self.config.diffusion_expert_config.max_vae_latent_side_after_patchify,
                max_latent_t=self.config.state_t,
                rope_h_extrapolation_ratio=self.config.diffusion_expert_config.rope_h_extrapolation_ratio,
                rope_w_extrapolation_ratio=self.config.diffusion_expert_config.rope_w_extrapolation_ratio,
                rope_t_extrapolation_ratio=self.config.diffusion_expert_config.rope_t_extrapolation_ratio,
                enable_fps_modulation=self.config.diffusion_expert_config.enable_fps_modulation,
                base_fps=self.config.diffusion_expert_config.base_fps,
                vision_gen=self.config.vision_gen,
                action_gen=self.config.action_gen,
                sound_gen=self.config.sound_gen,
                position_embedding_type=self.config.diffusion_expert_config.position_embedding_type,
                joint_attn_implementation=self.config.joint_attn_implementation,
                timestep_scale=1.0 / float(num_train_timesteps) * self.config.diffusion_expert_config.timestep_range,
                action_dim=self.config.max_action_dim,
                num_embodiment_domains=self.config.num_embodiment_domains,
                temporal_compression_factor_vision=self.tokenizer_vision_gen.temporal_compression_factor,
                natten_parameter_list=self.config.natten_parameter_list,
                video_temporal_causal=self.config.video_temporal_causal,
                # Sound generation parameters
                sound_dim=self.config.sound_dim,
                sound_latent_fps=self.config.sound_latent_fps,
            )
            network_config._attn_implementation_internal = "eager"
            net = Cosmos3VFMNetwork(
                language_model=language_model,
                config=network_config,
            )
            net.pad_for_cuda_graphs = self.config.parallelism.use_cuda_graphs

            # Inject LoRA BEFORE FSDP wrap, while still on meta device. The
            # injector must see unsharded Linear shapes; injecting post-FSDP causes
            # lora_B to be created at the per-rank shard size and crashes at
            # forward time. See `OmniMoTModel.add_lora` for details.
            if getattr(self.config, "lora_enabled", False):
                net = self.add_lora(
                    net,
                    lora_rank=self.config.lora_rank,
                    lora_alpha=self.config.lora_alpha,
                    lora_target_modules=self.config.lora_target_modules,
                )

        self.install_attention_dispatch(net)

        net = parallelize_vfm_network(
            net,
            parallel_dims=self.parallel_dims,
            config=self.config.parallelism,
        )

        with misc.timer("meta to cuda and broadcast model states"):
            net = net.to(dtype=dtype)
            net.to_empty(device=DEVICE)
            if DEVICE == Device.CUDA:
                # Weight initialization is not needed for other devices (cpu,
                # meta), since they are only for checkpoint conversion and smoke
                # tests.
                net.init_weights(buffer_device=DEVICE)
                if getattr(self.config, "lora_enabled", False):
                    self._init_lora_weights_post_materialization(net)

        return net

    def load_pretrained_model_if_needed(self):
        """
        This function is used to load the pretrained model weights from HF if needed.

        1. If self.vlm_config.load_pretrained is False, we skip loading the pretrained
           model weights.
        2. If self.vlm_config.load_pretrained is True, and
           self.config.diffusion_expert_config.load_weights_from_pretrained is True,
           we load the understanding pathway weights from HF, and copy them to the
           generation pathway.
        3. If self.vlm_config.load_pretrained is True, and
           self.config.diffusion_expert_config.load_weights_from_pretrained is False,
           we load the understanding pathway weights from HF, but do not copy them to
           the generation pathway. This is used when we warm-start from a load_path
           (but no previous checkpoint exists), and we want to switch the understanding
           pathway weights to a new model (e.g., Qwen3-VL to Cosmos-Reason2).
        """
        if not self.vlm_config.load_pretrained:
            return

        def _load_language_model(net: torch.nn.Module):
            load_language_model_safetensors(
                model=net.language_model,
                checkpoint_path=self.vlm_config.checkpoint_path,
                credential_path=self.vlm_config.credential_path,
                parallel_dims=self.parallel_dims,
                checkpoint_format=getattr(self.vlm_config, "vlm_checkpoint_format", None),
            )

        # When specified, we load pretrained LLM weights.
        log.info(f"Loading understanding pathway weights from {self.vlm_config.checkpoint_path}")
        _load_language_model(self.net)
        if self.config.ema.enabled:
            _load_language_model(self.net_ema)
        log.info("Successfully loaded understanding pathway weights.")

        if self.config.diffusion_expert_config.load_weights_from_pretrained:
            log.info("Copying understanding pathway weights to generation pathway.")
            self.net.language_model.init_moe()
            if self.config.ema.enabled:
                self.net_ema.language_model.init_moe()
            log.info("Successfully copied understanding pathway weights to generation pathway.")

    @misc.timer("OmniMoTModel: set_up_model")
    def set_up_model(self):
        assert hasattr(self, "parallel_dims"), "parallel_dims must be set"
        config = self.config
        with misc.timer("Creating PyTorch model and ema if enabled"):
            self.net = self.build_net(dtype=self.precision)
            self._param_count = count_params(self.net, verbose=False)

            if config.ema.enabled:
                self.net_ema = self.build_net(dtype=torch.float32)
                self.net_ema.requires_grad_(False)

                self.net_ema_worker = DTensorFastEmaModelUpdater()


                s = config.ema.rate
                self.ema_exp_coefficient = np.roots([1, 7, 16 - s**-2, 12 - s**-2]).real.max()

                self.net_ema_worker.copy_to(src_model=self.net, tgt_model=self.net_ema)

        self.set_up_memory()

        torch.cuda.empty_cache()

    def install_attention_dispatch(self, net: torch.nn.Module) -> None:
        """Install a custom attention dispatch function on the network.

        Called during ``build_net()`` after the network is constructed but
        before parallelization.  The base implementation is a no-op;
        ``OmniMoTCausalModel`` overrides this to install
        ``dispatch_attention_with_memory`` on every attention layer.
        """
        pass

    def set_up_memory(self) -> None:
        """Initialize memory state used during training (e.g. KV caches).

        The base implementation is a no-op.  ``OmniMoTCausalModel`` overrides
        this to allocate a KV cache.
        """
        pass

    def set_up_parallelism(self) -> None:
        """Set up the fsdp for the model."""
        if not torch.distributed.is_initialized():
            self.parallel_dims = None
            return

        self.parallel_dims = ParallelDims(
            enable_inference_mode=self.config.parallelism.enable_inference_mode,
            world_size=torch.distributed.get_world_size(),
            dp_shard=self.config.parallelism.data_parallel_shard_degree,
            cfgp=self.config.parallelism.cfg_parallel_shard_degree,
            cp=self.config.parallelism.context_parallel_shard_degree,
        )
        self.parallel_dims.build_meshes(device_type=DEVICE)

    def set_up_scheduler_and_sampler(self):
        # Get shift value - support both int and dict-based resolution lookup
        # For scheduler initialization, use model's configured resolution
        shift_config = self.config.rectified_flow_training_config.shift
        if isinstance(shift_config, int):
            shift = shift_config
        else:
            # shift set in RectifiedFlow is only used during inference.
            # So, set it to the resolution of the model.
            # This part gets executed only when we specify shift as a dict
            # This is needed during multi-resolution training.
            shift_dict = dict(shift_config)
            resolution = self.config.resolution
            if resolution not in shift_dict:
                raise ValueError(
                    f"Resolution '{resolution}' not found in shift dict. Available resolutions: {list(shift_dict.keys())}"
                )
            shift = shift_dict[resolution]

        # Rectified Flow timestep scheduler and sampler for training (separate for image and video)
        if self.config.vision_gen:
            self.rectified_flow_image = RectifiedFlow(
                velocity_field=self.net,
                train_time_distribution=self.config.rectified_flow_training_config.train_time_image_distribution,
                use_dynamic_shift=self.config.rectified_flow_training_config.use_dynamic_shift,
                shift=shift,
                train_time_weight_method=self.config.rectified_flow_training_config.train_time_weight,
                device=torch.device(DEVICE),
                dtype=self.tensor_kwargs_fp32["dtype"],
            )
            self.rectified_flow_video = RectifiedFlow(
                velocity_field=self.net,
                train_time_distribution=self.config.rectified_flow_training_config.train_time_video_distribution,
                use_dynamic_shift=self.config.rectified_flow_training_config.use_dynamic_shift,
                shift=shift,
                train_time_weight_method=self.config.rectified_flow_training_config.train_time_weight,
                device=torch.device(DEVICE),
                dtype=self.tensor_kwargs_fp32["dtype"],
            )
        if self.config.action_gen:
            self.rectified_flow_action = RectifiedFlow(
                velocity_field=self.net,
                train_time_distribution=self.config.rectified_flow_training_config.train_time_action_distribution,
                use_dynamic_shift=self.config.rectified_flow_training_config.use_dynamic_shift,
                shift=shift,
                train_time_weight_method=self.config.rectified_flow_training_config.train_time_weight,
                device=torch.device(DEVICE),
                dtype=self.tensor_kwargs_fp32["dtype"],
            )
        if self.config.sound_gen:
            self.rectified_flow_sound = RectifiedFlow(
                velocity_field=self.net,
                train_time_distribution=self.config.rectified_flow_training_config.train_time_sound_distribution,
                use_dynamic_shift=self.config.rectified_flow_training_config.use_dynamic_shift,
                shift=shift,
                train_time_weight_method=self.config.rectified_flow_training_config.train_time_weight,
                device=torch.device(DEVICE),
                dtype=self.tensor_kwargs_fp32["dtype"],
            )

        # Denoising sampler (solver) for inference
        assert self.config.rectified_flow_inference_config.scheduler_type in ["unipc", "edm"]
        if self.config.rectified_flow_inference_config.scheduler_type == "unipc":
            unipc_sampler_config = UniPCSamplerConfig(
                num_train_timesteps=self.config.rectified_flow_inference_config.num_train_timesteps,
                shift=self.config.rectified_flow_inference_config.shift,
                use_dynamic_shifting=self.config.rectified_flow_inference_config.use_dynamic_shifting,
            )
            self.sampler = UniPCSampler(cfg=unipc_sampler_config, tensor_kwargs=self.tensor_kwargs)
        else:
            self.sampler = EDMSampler()

        # Fixed-step sampler for distilled models (None for base models)
        if self.config.fixed_step_sampler_config is not None:
            cfg = self.config.fixed_step_sampler_config
            self.fixed_step_sampler = FixedStepSampler(
                t_list=list(cfg.t_list),
                sample_type=cfg.sample_type,
                num_train_timesteps=float(self.config.rectified_flow_inference_config.num_train_timesteps),
            )
        else:
            self.fixed_step_sampler = None

    def init_optimizer_scheduler(
        self, optimizer_config: LazyDict, scheduler_config: LazyDict
    ) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
        """Creates the optimizer and scheduler for the model.

        Args:
            optimizer_config (LazyDict): The lazy config for the optimizer.
            scheduler_config (LazyDict): The lazy config for the learning rate scheduler.

        Returns:
            optimizer (torch.optim.Optimizer): The model optimizer.
            scheduler (torch.optim.lr_scheduler.LRScheduler): The optimization scheduler.
        """

        optimizer = lazy_instantiate(optimizer_config, model=self.net)
        scheduler = lazy_instantiate(scheduler_config, optimizer=optimizer)
        return optimizer, scheduler

    def _derive_include_end_of_generation_token(self) -> bool:
        impl = self.config.joint_attn_implementation
        assert impl in ("flex", "two_way", "three_way"), (
            f"Invalid joint_attn_implementation: {impl}. Must be 'flex', 'two_way', or 'three_way'."
        )
        return impl == "flex"

    # ------------------------ training hooks ------------------------
    def on_before_zero_grad(
        self, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.LRScheduler, iteration: int
    ) -> None:
        """
        update the net_ema
        """
        del scheduler, optimizer

        if self.config.ema.enabled:
            # calculate beta for EMA update
            ema_beta = self.ema_beta(iteration)
            self.net_ema_worker.update_average(self.net, self.net_ema, beta=ema_beta)

    # ------------------------ helpers ------------------------

    def _pack_input_sequence(
        self,
        sequence_plans: list[SequencePlan],
        input_text_indexes: list[list[int]],
        gen_data_clean: GenerationDataClean,
        input_timesteps: torch.Tensor,
        include_end_of_generation_token: bool = False,
        skip_text_tokens: bool = False,
        initial_mrope_temporal_offset: int | float = 0,
    ) -> PackedSequence:
        """Wrap ``pack_input_sequence`` with all config-derived args pre-filled.

        Centralises the 9 config-derived positional/embedding args so callers only
        supply the four per-call arguments (sequence_plans, text tokens, data, timesteps)
        plus three optional flags.
        """
        assert self.tokenizer_vision_gen is not None
        return pack_input_sequence(
            sequence_plans=sequence_plans,
            input_text_indexes=input_text_indexes,
            gen_data_clean=gen_data_clean,
            input_timesteps=input_timesteps,
            special_tokens=self.llm_special_tokens,
            latent_patch_size=self.config.diffusion_expert_config.patch_spatial,
            skip_text_tokens=skip_text_tokens,
            include_end_of_generation_token=include_end_of_generation_token,
            position_embedding_type=self.config.diffusion_expert_config.position_embedding_type,
            unified_3d_mrope_reset_spatial_ids=self.config.diffusion_expert_config.unified_3d_mrope_reset_spatial_ids,
            unified_3d_mrope_temporal_modality_margin=self.config.diffusion_expert_config.unified_3d_mrope_temporal_modality_margin,
            enable_fps_modulation=self.config.diffusion_expert_config.enable_fps_modulation,
            base_fps=float(self.config.diffusion_expert_config.base_fps),
            temporal_compression_factor=self.tokenizer_vision_gen.temporal_compression_factor,
            video_temporal_causal=self.config.video_temporal_causal,
            action_dim=self.config.max_action_dim,
            initial_mrope_temporal_offset=initial_mrope_temporal_offset,
        )

    # ------------------------ training ------------------------

    def memory_init_training(
        self,
        gen_data_clean: GenerationDataClean,
        data_batch: dict[str, torch.Tensor],
        input_text_indexes: list[list[int]],
    ) -> tuple[GenerationDataClean, dict]:
        """Prepare the memory for a single training step.

        Called at the start of ``training_step`` to give the causal subclass
        an injection point for memory-based segment handling (frame trimming,
        segment bookkeeping, cache resets, packing overrides).

        The base implementation returns *gen_data_clean* unmodified and a
        default memory_info dict that does not support memory-backed training.

        The ``skip_text`` and ``initial_temporal`` offset fields are required,
        and are used for both sequence packing and memory.

        Returns:
            ``(gen_data_clean, memory_info)`` where *memory_info* is a dict with keys:
            ``skip_text``, ``initial_temporal_offset``
        """
        return gen_data_clean, {
            "skip_text": False,
            "initial_temporal_offset": 0,
        }

    def build_memory_state(
        self,
        packed_seq: PackedSequence,
        memory_info: dict,
    ) -> MemoryState | None:
        """Construct a ``MemoryState`` from a packed sequence and context dict.

        Called after packing in ``training_step()``, and before ``denoise()``
        in AR inference.  The base implementation returns ``None`` (no
        persistent memory).  ``OmniMoTCausalModel`` overrides this to build
        the appropriate ``ARMemoryState`` or ``KVCacheTrainMemoryState``.

        Args:
            packed_seq: The packed multi-modal sequence produced by
                ``_pack_input_sequence``.
            memory_info: Context dict returned by ``memory_init_training()``
                (for the training path) or constructed by the AR inference
                caller.  See ``memory_init_training()`` for the base keys.
        """
        return None

    def pre_noise_memory_hook(
        self,
        packed_sequence: PackedSequence,
        gen_data_clean: GenerationDataClean,
        memory_info: dict,
    ) -> dict:
        """Hook called after sequence packing and before noising. Returns (possibly updated) memory_info.

        The packed sequence still contains clean tokens at this point.
        Override in subclasses to run a clean forward pass (e.g. for teacher forcing).
        """
        return memory_info

    def training_step(
        self, data_batch: dict[str, torch.Tensor], iteration: int
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """
        Performs a single training step for the rectified-flow (flow-matching) model.

        This method executes one iteration of the model's training. It involves:
        1. Tokenizing generation modalities (vision/action/sound) into latents (tokens).
        2. Sampling a training timestep (t) for each modality and constructing noised latents (xt)
           per the rectified-flow formulation.
        3. Packing text + generation tokens into a single sequence and running the MoT network to predict
           the flow field velocity at the given t.
        4. Computing flow-matching loss (plus optional auxiliary load-balancing losses).

        Args:
            data_batch (dict): raw data batch draw from the training data loader.
            iteration (int): Current iteration number.

        Returns:
            tuple: A tuple containing two elements:
                - dict: additional data that used to debug / logging / callbacks
                - Tensor: The computed loss for the training step as a PyTorch Tensor.

        """
        if self.parallel_dims is None or self.parallel_dims.cp_rank == 0:
            self._update_train_stats(data_batch)

        # Load, apply dropout, and tokenize input captions
        input_text_indexes = self._load_and_tokenize_text_data(data_batch, iteration)

        # Build sequence plans if not present. SequencePlan has the conditioning information.
        sequence_plans = build_sequence_plans_from_data_batch(
            data_batch=data_batch,
            input_video_key=self.input_video_key,
            input_image_key=self.input_image_key,
        )

        # Get data from raw data batch and tokenize into corresponding tokens for *generation* task
        # The unnoised, tokenized data for the generation task.
        gen_data_clean = self.get_data_and_condition(data_batch, iteration=iteration)

        gen_data_clean, memory_info = self.memory_init_training(gen_data_clean, data_batch, input_text_indexes)

        # Compute resolution per sample for per-sample shift lookup
        # image_size[i] may be (1, 4) from IterativeJointDataLoader or (4,) from custom_collate_fn.
        if "image_size" in data_batch:
            data_resolutions = []
            for i in range(gen_data_clean.batch_size):
                img_size = data_batch["image_size"][i]
                if img_size.dim() == 2:
                    img_size = img_size[0]
                target_h = int(img_size[0].item())
                target_w = int(img_size[1].item())
                data_resolutions.append(get_vision_data_resolution((target_h, target_w)))
        else:
            data_resolutions = None

        # Calculate number of tokens per sample (before 2x2 merge) for dynamic shift
        # gen_data_clean.x0_tokens_vision: B, C, T, H, W
        assert all(x.shape[0] == 1 for x in gen_data_clean.x0_tokens_vision), (
            "Batch size must be 1 for individual samples"
        )
        num_tokens_per_sample = [x.shape[2] * x.shape[3] * x.shape[4] for x in gen_data_clean.x0_tokens_vision]

        # Sample a random noise level (sigma) and corresponding interpolation coefficient ("timesteps" in RF)
        # Apply shift per sample based on each sample's resolution
        num_vision_latent_frames = [x.shape[2] for x in gen_data_clean.x0_tokens_vision]
        timesteps_vision, sigmas_vision = self._get_train_noise_level_vision(
            batch_size=gen_data_clean.batch_size,
            is_image_batch=gen_data_clean.is_image_batch,
            resolutions=data_resolutions,
            num_vision_latent_frames=num_vision_latent_frames,
            num_tokens=num_tokens_per_sample,
            iteration=iteration,
        )  # [B, T_vis] each

        # Optional independent action schedule (sampled from rectified_flow_action with
        # action-specific shift/high-sigma overrides). Only active when the config opts in and
        # the batch contains action data.
        #
        # Mixed-batch indexing: gen_data_clean.x0_tokens_action (and every packed_sequence.action.*
        # field) is *dense* — one entry per sample with has_action=True, in the original batch order
        # but skipping non-action samples. To feed each dense action entry its sample's sigma, we
        # sample σ for the full batch and reindex with action_sample_indices (the batch positions
        # of action-bearing samples). This avoids the mismatch that happens when, e.g., batch
        # sample 1 has action but the dense entry 0 would otherwise read σ from batch position 0.
        rf_cfg = self.config.rectified_flow_training_config
        action_sample_indices = [i for i, plan in enumerate(sequence_plans) if plan.has_action]
        if rf_cfg.independent_action_schedule and action_sample_indices:
            ts_full, sg_full = self._get_train_noise_level_action(
                batch_size=gen_data_clean.batch_size, iteration=iteration
            )  # [B, 1] each
            idx = torch.tensor(action_sample_indices, dtype=torch.long)  # [n_action]
            timesteps_action = ts_full[idx]  # [n_action, 1]
            sigmas_action = sg_full[idx]  # [n_action, 1]
        else:
            timesteps_action, sigmas_action = (None, None)

        # Optional independent sound schedule: sample a scalar sound sigma per batch
        # slot, then reindex to the dense audio-bearing subset.
        sound_sample_indices = [i for i, plan in enumerate(sequence_plans) if getattr(plan, "has_sound", False)]
        if getattr(rf_cfg, "independent_sound_schedule", False) and sound_sample_indices:
            ts_sound_full, sg_sound_full = self._get_train_noise_level_sound(
                batch_size=gen_data_clean.batch_size
            )  # [B,1] each
            timesteps_sound, sigmas_sound = build_dense_sound_schedule(
                sequence_plans,
                gen_data_clean.x0_tokens_sound,
                ts_sound_full,
                sg_sound_full,
            )  # [n_sound,1], [n_sound,1]
        else:
            timesteps_sound, sigmas_sound = (None, None)

        # Broadcast timesteps/sigmas across CP group to ensure consistency
        if self.parallel_dims is not None and self.parallel_dims.cp_enabled:
            src_rank = 0  # use cp rank 0 to broadcast timesteps/sigmas
            cp_group = self.parallel_dims.cp_mesh.get_group()
            global_src_rank = torch.distributed.get_global_rank(cp_group, src_rank)
            timesteps_vision = timesteps_vision.contiguous()
            sigmas_vision = sigmas_vision.contiguous()
            torch.distributed.broadcast(timesteps_vision, src=global_src_rank, group=cp_group)
            torch.distributed.broadcast(sigmas_vision, src=global_src_rank, group=cp_group)
            if sigmas_action is not None:
                timesteps_action = timesteps_action.contiguous()
                sigmas_action = sigmas_action.contiguous()
                torch.distributed.broadcast(timesteps_action, src=global_src_rank, group=cp_group)
                torch.distributed.broadcast(sigmas_action, src=global_src_rank, group=cp_group)
            if sigmas_sound is not None:
                timesteps_sound = timesteps_sound.contiguous()  # [n_sound,1]
                sigmas_sound = sigmas_sound.contiguous()  # [n_sound,1]
                torch.distributed.broadcast(timesteps_sound, src=global_src_rank, group=cp_group)
                torch.distributed.broadcast(sigmas_sound, src=global_src_rank, group=cp_group)

        if timesteps_sound is None:
            # Sound tensors are dense over audio-bearing samples, while the vision timestep/sigma schedule
            # is indexed by original batch position. Reindex here so mixed audio/no-audio batches use each
            # sound sample's own schedule for noising and loss weighting.
            timesteps_sound, sigmas_sound = build_dense_sound_schedule(
                sequence_plans,
                gen_data_clean.x0_tokens_sound,
                timesteps_vision,
                sigmas_vision,
            )  # [n_sound,T_vis] or None, [n_sound,T_vis] or None

        packed_sequence = self._pack_input_sequence(
            sequence_plans,
            input_text_indexes,
            gen_data_clean,
            timesteps_vision.cpu(),
            skip_text_tokens=memory_info["skip_text"],
            initial_mrope_temporal_offset=memory_info["initial_temporal_offset"],
        )

        # Under independent_action_schedule, overwrite the vision-based action timestep the
        # packer injected with the action timestep, so the denoiser's action timestep embedding
        # matches the sigma used to noise action tokens.
        if timesteps_action is not None and packed_sequence.action is not None:
            action_has_noisy_tokens = any(nfi.numel() > 0 for nfi in packed_sequence.action.noisy_frame_indexes)
            if action_has_noisy_tokens:
                sample_ts = timesteps_action.squeeze(1).cpu()  # [n_action]
                packed_sequence.action.timesteps = torch.cat(
                    [
                        sample_ts[i : i + 1].expand(nfi.numel())
                        for i, nfi in enumerate(packed_sequence.action.noisy_frame_indexes)
                    ]
                ).to(dtype=torch.float32)  # [N_action_noisy]
            else:
                timesteps_action, sigmas_action = (None, None)

        # Under independent_sound_schedule, overwrite the vision-based sound timestep the packer
        # injected with the sound timestep, so the denoiser's sound timestep embedding matches
        # the sigma used to noise sound tokens.
        if (
            getattr(rf_cfg, "independent_sound_schedule", False)
            and timesteps_sound is not None
            and packed_sequence.sound is not None
        ):
            sound_has_noisy_tokens = any(nfi.numel() > 0 for nfi in packed_sequence.sound.noisy_frame_indexes)
            if sound_has_noisy_tokens:
                sample_ts = timesteps_sound.squeeze(1).cpu()  # [n_sound]
                packed_sequence.sound.timesteps = torch.cat(
                    [
                        sample_ts[i : i + 1].expand(nfi.numel())
                        for i, nfi in enumerate(packed_sequence.sound.noisy_frame_indexes)
                    ]
                ).to(dtype=torch.float32)  # [N_sound_noisy]
            else:
                timesteps_sound, sigmas_sound = (None, None)

        # For image editing (multi-item vision), expand per-sample timesteps/sigmas to
        # per-vision-item so downstream noise/loss indexing matches the flat x0_tokens_vision
        # list. No-op when num_vision_items_per_sample is None (standard T2I/T2V/policy cases).
        # Conditioning items get sigma=0 via their condition_mask, so the actual timestep value
        # for them does not matter.
        timesteps_vision = _expand_per_sample_to_per_vision_item(
            timesteps_vision, gen_data_clean.num_vision_items_per_sample
        )  # [B_items, T_vis]
        sigmas_vision = _expand_per_sample_to_per_vision_item(
            sigmas_vision, gen_data_clean.num_vision_items_per_sample
        )  # [B_items, T_vis]

        memory_info = self.pre_noise_memory_hook(packed_sequence, gen_data_clean, memory_info)

        # Flow matching/diffusion forward process: noise the input signal with the sampled noise level
        gen_data_noised = self._add_noise_to_input(
            gen_data_clean,
            packed_sequence,
            sigmas_vision,
            sigmas_action=sigmas_action,
            sigmas_sound=sigmas_sound,
        )
        self._replace_clean_with_noised(packed_sequence, gen_data_noised)

        # Move packed sequence to CUDA
        packed_sequence.to_cuda()

        # Network forward pass
        memory = self.build_memory_state(packed_sequence, memory_info)  # pylint: disable=assignment-from-none
        out_net = self.denoise(
            data_batch_packed=packed_sequence,
            fps_vision=gen_data_clean.fps_vision,
            fps_action=gen_data_clean.fps_action,
            fps_sound=gen_data_clean.fps_sound,
            memory=memory,
        )

        loss, losses_dict = self._compute_losses(
            out_net=out_net,
            data_batch_packed=packed_sequence,
            gen_data_noised=gen_data_noised,
            timesteps=timesteps_vision,
            is_image_batch=gen_data_clean.is_image_batch,
            timesteps_action=timesteps_action,
            timesteps_sound=timesteps_sound,
        )

        # Pixel-space video shapes for VAE FLOPs estimation in callbacks (e.g. MFU).
        _vae_pixel_shapes: list[tuple[int, int, int]] = []
        if gen_data_clean.raw_state_vision is not None:
            for _v in gen_data_clean.raw_state_vision:
                if _v is not None:
                    assert _v.dim() in [4, 5], (
                        "Currently only [C, T, H, W] and [B, C, T, H, W] formats are supported for the VAE encoding."
                    )
                    t_h_w = (
                        (int(_v.shape[2]), int(_v.shape[3]), int(_v.shape[4]))
                        if _v.dim() == 5
                        else (int(_v.shape[1]), int(_v.shape[2]), int(_v.shape[3]))
                    )
                    _vae_pixel_shapes.append(t_h_w)

        _vision_tokens = len(packed_sequence.vision.sequence_indexes) if packed_sequence.vision else 0
        _action_tokens = len(packed_sequence.action.sequence_indexes) if packed_sequence.action else 0
        _sound_tokens = len(packed_sequence.sound.sequence_indexes) if packed_sequence.sound else 0

        output_batch = {
            "x0": gen_data_clean.x0_tokens_vision,
            "xt": gen_data_noised.xt_tokens_vision,
            "sigma": sigmas_vision,  # [B_items, T_vis]
            "model_pred": out_net["preds_vision"],
            "condition_mask_vision": packed_sequence.vision.condition_mask if packed_sequence.vision else None,
            "condition_mask_action": packed_sequence.action.condition_mask if packed_sequence.action else None,
            "und_token_length": packed_sequence.text_indexes.shape[0],
            "gen_token_length": packed_sequence.sequence_length - packed_sequence.text_indexes.shape[0],
            "vision_token_length": _vision_tokens,
            "action_token_length": _action_tokens,
            "sound_token_length": _sound_tokens,
            "is_image_batch": gen_data_clean.is_image_batch,
            "batch_size": gen_data_clean.batch_size,
            "split_lens": packed_sequence.split_lens,
            "attn_modes": packed_sequence.attn_modes,
            "vae_pixel_shapes": _vae_pixel_shapes,
            **losses_dict,
        }
        if sigmas_action is not None:
            output_batch["sigma_action"] = sigmas_action  # [n_action, 1] — dense over action-bearing samples
        if getattr(rf_cfg, "independent_sound_schedule", False) and sigmas_sound is not None:
            output_batch["sigma_sound"] = sigmas_sound  # [n_sound, 1] — dense over sound-bearing samples

        return output_batch, loss

    def _compute_flow_matching_loss(
        self,
        pred: list[torch.Tensor],
        target: list[torch.Tensor],
        condition_mask: list[torch.Tensor],
        timesteps: torch.Tensor,
        has_valid_tokens: bool,
        rectified_flow: RectifiedFlow,
        loss_scale: float | None = None,
        raw_action_dim: list[torch.Tensor] | None = None,
        normalize_by_active: bool = False,
    ) -> torch.Tensor:
        """Compute flow matching loss for a modality.

        Args:
            pred: Predicted velocity field (list of tensors, one per sample).
            target: Target velocity field (list of tensors, one per sample).
                Under rectified flow the target is ``v = eps - x0``.
            condition_mask: Mask where 1 = clean/conditioning, 0 = noisy/generation (list of tensors).
            timesteps: Diffusion timesteps for time weighting. Shape [B,1] for
                base/teacher_forcing (all frames share one timestep) or [B,T_max]
                for diffusion_forcing (per-frame independent timesteps). Time weights
                are applied per-frame before averaging, so non-uniform weight functions
                are handled correctly.
            has_valid_tokens: Whether this modality has valid noisy tokens.
            rectified_flow: The rectified flow object for time weighting.
            loss_scale: Optional per-modality loss scale. Falls back to the global
                ``rectified_flow_training_config.loss_scale`` when *None*.
            normalize_by_active: When True, normalize per-instance loss by the count of
                active (noisy) elements rather than all elements. Preserves the
                ``sum / active_count`` semantics needed for distillation critics where
                conditioned frames contribute no signal and should not dilute the
                denominator.

        Returns:
            tuple: A tuple containing two elements:
                - Flow matching loss (or dummy loss for gradient consistency).
                - Per-instance loss (or dummy loss for gradient consistency).
        """
        return compute_flow_matching_loss(
            pred=pred,
            target=target,
            condition_mask=condition_mask,
            timesteps=timesteps,
            has_valid_tokens=has_valid_tokens,
            rectified_flow=rectified_flow,
            tensor_kwargs_fp32=self.tensor_kwargs_fp32,
            loss_scale=loss_scale,
            raw_action_dim=raw_action_dim,
            normalize_by_active=normalize_by_active,
        )

    def _compute_losses(
        self,
        out_net: dict,
        data_batch_packed: PackedSequence,
        gen_data_noised: GenerationDataNoised,
        timesteps: torch.Tensor,
        is_image_batch: bool,
        timesteps_action: torch.Tensor | None = None,
        timesteps_sound: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute flow matching loss and auxiliary load balancing losses.

        ``timesteps_action`` is an optional ``[n_action, 1]`` override for the action loss
        time-weighting — dense over action-bearing samples, matching ``data_batch_packed.action.*``.
        When None, action reuses ``timesteps`` (vision timesteps, legacy behavior). Set by
        ``training_step`` under ``independent_action_schedule=True``.

        ``timesteps_sound`` is an optional dense sound timestep tensor, matching
        ``data_batch_packed.sound.*``. When None, sound reuses ``timesteps``.
        """
        total_loss = 0.0
        losses_dict = {}
        # ts_action shape: vision fallback [B_items, T_vis] (legacy) or [n_action, 1] (independent).
        ts_action = timesteps if timesteps_action is None else timesteps_action  # [B_items,T_vis] or [n_action,1]
        # ts_sound shape: vision fallback [B_items,T_vis] or dense sound schedule [n_sound,...].
        ts_sound = timesteps if timesteps_sound is None else timesteps_sound  # [B_items,T_vis] or [n_sound,...]

        rf_cfg = self.config.rectified_flow_training_config
        normalize_by_active = rf_cfg.normalize_loss_by_active
        if self.config.vision_gen:
            assert data_batch_packed.vision is not None, "Vision packed data required when vision_gen is True"
            assert isinstance(data_batch_packed.vision.condition_mask, list), (
                "Vision condition mask must be a list of tensors for loss computation"
            )
            rectified_flow_vision = self.rectified_flow_image if is_image_batch else self.rectified_flow_video

            fm_loss_vision, fm_loss_vision_per_instance = self._compute_flow_matching_loss(
                pred=out_net["preds_vision"],
                target=gen_data_noised.vt_target_vision,
                condition_mask=data_batch_packed.vision.condition_mask,
                timesteps=timesteps,
                has_valid_tokens=has_noisy_tokens(data_batch_packed.vision),
                rectified_flow=rectified_flow_vision,
                normalize_by_active=normalize_by_active,
            )
            loss_scale = (
                rf_cfg.image_loss_scale if is_image_batch and rf_cfg.image_loss_scale is not None else rf_cfg.loss_scale
            )
            total_loss += fm_loss_vision * loss_scale
            losses_dict["flow_matching_loss_vision"] = fm_loss_vision
            losses_dict["flow_matching_loss_vision_per_instance"] = fm_loss_vision_per_instance
        else:
            losses_dict["flow_matching_loss_vision"] = torch.tensor(0.0, **self.tensor_kwargs_fp32)

        if self.config.action_gen:
            if data_batch_packed.action is not None:
                assert isinstance(data_batch_packed.action.condition_mask, list), (
                    "Action condition mask must be a list of tensors for loss computation"
                )
                assert gen_data_noised.vt_target_action is not None, "Action targets required when action_gen is True"
                fm_loss_action, _ = self._compute_flow_matching_loss(
                    pred=out_net["preds_action"],
                    target=gen_data_noised.vt_target_action,
                    condition_mask=data_batch_packed.action.condition_mask,
                    timesteps=ts_action,
                    has_valid_tokens=has_noisy_tokens(data_batch_packed.action),
                    rectified_flow=self.rectified_flow_action,
                    raw_action_dim=data_batch_packed.action.raw_action_dim,
                    normalize_by_active=normalize_by_active,
                )

                # Yihuai: In case the video loss is too large (1.5) and covers the action loss (0.05), we scale up the action loss to match the video loss to improve action precision.
                total_loss += fm_loss_action * rf_cfg.action_loss_weight
                losses_dict["flow_matching_loss_action"] = fm_loss_action
            else:
                # No action data in this batch. Connect the network's dummy preds_action
                # to the loss so action-specific params
                # (llm2action, action2llm, action_modality_embed) stay in the backward
                # graph. Without this, FSDP reduce-scatter / DDP all-reduce will hang
                # when other ranks do have action data.
                dummy_loss = 0.0 * sum(p.sum() for p in out_net["preds_action"])
                total_loss += dummy_loss
                losses_dict["flow_matching_loss_action"] = dummy_loss
        else:
            losses_dict["flow_matching_loss_action"] = torch.tensor(0.0, **self.tensor_kwargs_fp32)

        if self.config.sound_gen:
            if data_batch_packed.sound is not None:
                assert isinstance(data_batch_packed.sound.condition_mask, list), (
                    "Sound condition mask must be a list of tensors for loss computation"
                )
                assert gen_data_noised.vt_target_sound is not None, "Sound targets required when sound_gen is True"
                # Sound preds/targets are (C, T); condition_mask is (T, 1) — transpose to (1, T) for broadcasting
                fm_loss_sound, _ = self._compute_flow_matching_loss(
                    pred=out_net["preds_sound"],
                    target=gen_data_noised.vt_target_sound,
                    condition_mask=[m.T for m in data_batch_packed.sound.condition_mask],
                    timesteps=ts_sound,
                    has_valid_tokens=has_noisy_tokens(data_batch_packed.sound),
                    rectified_flow=self.rectified_flow_sound,
                    normalize_by_active=normalize_by_active,
                )
                loss_scale = rf_cfg.sound_loss_scale if rf_cfg.sound_loss_scale is not None else rf_cfg.loss_scale
                total_loss += fm_loss_sound * loss_scale
                losses_dict["flow_matching_loss_sound"] = fm_loss_sound
            else:
                # No sound data in this batch. Connect the network's dummy preds_sound
                # to the loss so sound-specific params (sound2llm, llm2sound,
                # sound_modality_embed) stay in the backward graph. Without this,
                # FSDP gradient reduce hangs when other ranks do have sound data.
                dummy_loss = 0.0 * sum(p.sum() for p in out_net["preds_sound"])
                total_loss += dummy_loss
                losses_dict["flow_matching_loss_sound"] = dummy_loss
        else:
            losses_dict["flow_matching_loss_sound"] = torch.tensor(0.0, **self.tensor_kwargs_fp32)

        # 2. Load balancing auxiliary losses
        for load_balancing_type in ["und", "gen"]:
            lbl_metadata = out_net.get(f"lbl_metadata_{load_balancing_type}", None)
            if lbl_metadata is None:
                continue
            load_balancing_loss = compute_load_balancing_loss(
                lbl_metadata,
                coeff=getattr(self.config.lbl, f"coeff_{load_balancing_type}"),
                method=self.config.lbl.method,
                device_mesh=self.parallel_dims.dp_mesh if self.parallel_dims else None,
            )
            if load_balancing_loss is not None:
                total_loss += load_balancing_loss
                losses_dict[f"aux_loss_{load_balancing_type}"] = load_balancing_loss

        return total_loss, losses_dict

    def _update_train_stats(self, data_batch: dict[str, torch.Tensor]) -> None:
        is_image = self.is_image_batch(data_batch)
        input_key = self.input_image_key if is_image else self.input_video_key
        if isinstance(self.net, WeightTrainingStat):
            val = data_batch[input_key]
            # For image editing data_batch[input_key] is a list-of-lists, not a tensor.
            sample_count = len(val) if isinstance(val, list) else val.shape[0]
            if is_image:
                self.net.accum_image_sample_counter += sample_count
            else:
                self.net.accum_video_sample_counter += sample_count

    def _load_and_tokenize_text_data(self, data_batch: dict[str, torch.Tensor], iteration: int) -> list[list[int]]:
        """
        Load and tokenize the text data from the data batch.

        Args:
            data_batch (dict[str, torch.Tensor]): The data batch.
            iteration (int): The current iteration number.

        Returns:
            list[torch.Tensor]: The input text tokens.
        """
        input_text_indexes = []

        input_captions = data_batch[self.input_caption_key]
        input_text_tokens = data_batch["text_token_ids"]
        if isinstance(input_text_tokens, list):
            # Convert text tokens to list of lists of ints
            input_text_tokens = [tokens.tolist() for x in input_text_tokens for tokens in x]
        else:
            input_text_tokens = [tokens.squeeze(0).tolist() for tokens in input_text_tokens]

        return input_text_tokens

    def _get_train_noise_level_vision(
        self,
        batch_size: int,
        is_image_batch: bool,
        num_vision_latent_frames: list[int],
        resolutions: list[str] | str | None = None,
        num_tokens: list[int] | None = None,
        iteration: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sample the rectified flow interpolation coefficient (timesteps), optionally adjust the sampled
        timesteps with high sigma strategy, and obtain the corresponding normalized timestep.

        Args:
            batch_size: Batch size for sampling timesteps.
            is_image_batch: Whether this is an image batch (vs video).
            num_vision_latent_frames: Per-sample vision latent frame counts [T_0, ..., T_{B-1}].
                         For causal_training_strategy="diffusion_forcing", resamples B*T_max independent
                         times and returns tensors of shape [B,T_max]. For base/TF strategies, ignored —
                         returns shape [B,1] (all frames share the same sigma).
            resolutions: Resolution string(s) (e.g., "256", "512") for dict-based shift lookup.
                         Can be a single string (applied to all samples) or a list of strings (one per sample).
                         If None, defaults to self.config.resolution (can be used for other modalities).
            num_tokens: Number of tokens for each sample (before 2x2 merge). Needed for dynamic shift.

        Returns:
            (timesteps, sigmas): Both [B,1] for TF/base, or [B,T_max] for diffusion_forcing.
        """


        rectified_flow = self.rectified_flow_image if is_image_batch else self.rectified_flow_video

        assert not self.config.rectified_flow_training_config.use_discrete_rf, (
            "Discrete RF is not supported for Cosmos3"
        )
        # Continuous RF implementation
        max_timestep = rectified_flow.noise_scheduler.config.num_train_timesteps

        # Get shift value(s) - support both int and dict-based resolution lookup
        shift_config = self.config.rectified_flow_training_config.shift
        if isinstance(shift_config, int):
            # Int-based shift: use directly for all samples
            shifts = torch.full((batch_size,), shift_config, dtype=torch.float32)
        else:
            # Convert to plain dict to avoid traceback-based memory leaks when GC is disabled
            # (OmegaConf's `in` operator uses exception control flow internally).
            shift_dict = dict(shift_config)
            if not is_image_batch and "dynamic_shift_base_num_tokens_video" in shift_dict:
                # Dynamic shift based on token count
                assert num_tokens is not None and len(num_tokens) == batch_size
                base_num_tokens = shift_dict["dynamic_shift_base_num_tokens_video"]
                shifts = torch.sqrt(torch.tensor(num_tokens, dtype=torch.float32) / base_num_tokens)
            elif is_image_batch and "dynamic_shift_base_num_tokens_image" in shift_dict:
                assert num_tokens is not None and len(num_tokens) == batch_size
                base_num_tokens = shift_dict["dynamic_shift_base_num_tokens_image"]
                shifts = torch.sqrt(torch.tensor(num_tokens, dtype=torch.float32) / base_num_tokens)
            else:
                # Dict-based shift: lookup per sample
                if resolutions is None:
                    raise ValueError("Resolutions must be provided when shift is a dict")

                # Normalize to list format
                if isinstance(resolutions, str):
                    resolutions = [resolutions] * batch_size

                assert len(resolutions) == batch_size, (
                    f"Number of resolutions ({len(resolutions)}) must match batch_size ({batch_size})"
                )

                # Lookup shift per sample
                shifts_list = []
                for resolution in resolutions:
                    if resolution not in shift_dict:
                        raise ValueError(
                            f"Resolution '{resolution}' not found in shift dict. Available resolutions: {list(shift_dict.keys())}"
                        )
                    shifts_list.append(shift_dict[resolution])
                shifts = torch.tensor(shifts_list, dtype=torch.float32)

        # Sample noise times: B×T_max for DF (one per video latent frame), B×1 for base/TF
        if self.config.causal_training_strategy == "diffusion_forcing":
            # T_max = max(num_vision_latent_frames) across the batch; trailing entries for shorter
            # sequences are unused (sliced away in _add_noise_to_input).
            T_max = max(num_vision_latent_frames)
            t_raw = (
                rectified_flow.sample_train_time(batch_size * T_max, iteration=iteration)
                .to(**self.tensor_kwargs_fp32)
                .reshape(batch_size, T_max)
            )  # [B,T_max]
        else:
            t_raw = (
                rectified_flow.sample_train_time(batch_size, iteration=iteration)
                .to(**self.tensor_kwargs_fp32)
                .unsqueeze(1)
            )  # [B,1]

        # Apply shift and scale: t_raw ∈ [0,1] → timesteps ∈ [0,max_timestep]
        # shifts.unsqueeze(1) → [B,1], broadcasts with both [B,1] (base/TF) and [B,T_max] (DF)
        t = 1 - t_raw  # [B,1] or [B,T_max]
        shifts_2d = shifts.unsqueeze(1).to(t_raw.device)  # [B,1], broadcasts with [B,1] and [B,T_max]
        timesteps = shifts_2d * t / (1 + (shifts_2d - 1) * t) * max_timestep  # [B,1] or [B,T_max]

        if self.config.rectified_flow_training_config.use_high_sigma_strategy:
            timesteps = self._apply_high_noise_strategy(timesteps, max_timestep)  # [B,1] or [B,T_max]

        sigmas = timesteps / max_timestep  # [B,1] for base/TF, [B,T_max] for DF
        return timesteps, sigmas

    def _apply_high_noise_strategy(self, timesteps: torch.Tensor, max_timestep: int) -> torch.Tensor:
        """
        Update the sampled RF timesteps to shift the distribution towards higher noise levels (high sigmas).

        Args:
            timesteps (torch.Tensor): Input timesteps. Shape [B,1] for base/TF or [B,T_max] for DF.
            max_timestep (int): The maximum timestep value.

        Returns:
            torch.Tensor: Timesteps with the same shape as input — [B,1] or [B,T_max].
        """
        mask = (
            torch.rand(timesteps.shape, device=timesteps.device)
            < self.config.rectified_flow_training_config.high_sigma_ratio
        )
        new_timesteps = (
            torch.rand(timesteps.shape, device=timesteps.device).type_as(timesteps)
            * (
                self.config.rectified_flow_training_config.high_sigma_timesteps_max
                - self.config.rectified_flow_training_config.high_sigma_timesteps_min
            )
            + self.config.rectified_flow_training_config.high_sigma_timesteps_min
        )
        timesteps = torch.where(mask, new_timesteps, timesteps)

        return timesteps

    def _get_train_noise_level_action(
        self, batch_size: int, iteration: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample ``(timesteps, sigmas)`` of shape ``[batch_size, 1]`` from ``rectified_flow_action``.

        This helper is locally-scoped: it just draws ``batch_size`` independent σ values and
        applies action-specific shift / high-sigma config. The caller decides what ``batch_size``
        means semantically — ``training_step`` passes the full batch size and then reindexes to
        the dense action-bearing subset with ``action_sample_indices``.

        ``shift_action`` must be an int (or ``None`` to inherit ``shift``). Dict-keyed
        per-resolution shifts are vision-only — multi-resolution action training would need
        per-sample lookup, which this helper does not implement; if the global ``shift`` is a
        dict and ``shift_action`` is None, this raises so the user sets shift_action explicitly.
        ``use_high_sigma_strategy_action`` toggles the high-σ strategy for action; when on, the
        global ``high_sigma_ratio`` / ``_min`` / ``_max`` apply. σ is a shared scalar per input
        slot (no per-frame σ for action).
        """
        rf_cfg = self.config.rectified_flow_training_config
        rf = self.rectified_flow_action
        max_timestep = rf.noise_scheduler.config.num_train_timesteps  # int

        # Resolve shift. shift_action, when provided, must be an int.
        if rf_cfg.shift_action is not None:
            if not isinstance(rf_cfg.shift_action, int):
                raise ValueError(
                    f"shift_action must be an int; got {type(rf_cfg.shift_action).__name__}. "
                    "Dict-keyed per-resolution shifts are vision-only."
                )
            shift_val = rf_cfg.shift_action  # int
        elif isinstance(rf_cfg.shift, int):
            shift_val = rf_cfg.shift  # inherit the global int shift
        else:
            raise ValueError(
                "shift_action=None requires the global `shift` to be an int. When `shift` is a "
                f"dict (multi-resolution vision training), set shift_action explicitly as an int. "
                f"Got shift={rf_cfg.shift!r}."
            )

        t_raw = (
            rf.sample_train_time(batch_size, iteration=iteration).to(**self.tensor_kwargs_fp32).unsqueeze(1)
        )  # [B,1]
        t = 1 - t_raw  # [B,1]
        shifts_2d = torch.full((batch_size, 1), shift_val, dtype=torch.float32, device=t_raw.device)  # [B,1]
        timesteps = shifts_2d * t / (1 + (shifts_2d - 1) * t) * max_timestep  # [B,1]

        if rf_cfg.use_high_sigma_strategy_action:
            timesteps = self._apply_high_noise_strategy(timesteps, max_timestep)  # [B,1]

        sigmas = timesteps / max_timestep  # [B,1]
        return timesteps, sigmas

    def _get_train_noise_level_sound(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample ``(timesteps, sigmas)`` of shape ``[batch_size, 1]`` from ``rectified_flow_sound``.

        Sound uses a shared scalar sigma per audio-bearing sample, then training_step
        reindexes the full-batch samples to the dense sound tensor list.
        """
        rf_cfg = self.config.rectified_flow_training_config
        rf = self.rectified_flow_sound
        max_timestep = rf.noise_scheduler.config.num_train_timesteps  # int

        # Resolve shift. shift_sound, when provided, must be an int.
        if rf_cfg.shift_sound is not None:
            if not isinstance(rf_cfg.shift_sound, int):
                raise ValueError(
                    f"shift_sound must be an int; got {type(rf_cfg.shift_sound).__name__}. "
                    "Dict-keyed per-resolution shifts are vision-only."
                )
            shift_val = rf_cfg.shift_sound  # int
        elif isinstance(rf_cfg.shift, int):
            shift_val = rf_cfg.shift  # inherit the global int shift
        else:
            raise ValueError(
                "shift_sound=None requires the global `shift` to be an int. When `shift` is a "
                f"dict (multi-resolution vision training), set shift_sound explicitly as an int. "
                f"Got shift={rf_cfg.shift!r}."
            )

        t_raw = rf.sample_train_time(batch_size).to(**self.tensor_kwargs_fp32).unsqueeze(1)  # [B,1]
        t = 1 - t_raw  # [B,1]
        shifts_2d = torch.full((batch_size, 1), shift_val, dtype=torch.float32, device=t_raw.device)  # [B,1]
        timesteps = shifts_2d * t / (1 + (shifts_2d - 1) * t) * max_timestep  # [B,1]

        if rf_cfg.use_high_sigma_strategy_sound:
            timesteps = self._apply_high_noise_strategy(timesteps, max_timestep)  # [B,1]

        sigmas = timesteps / max_timestep  # [B,1]
        return timesteps, sigmas

    def _add_noise_to_input(
        self,
        gen_data_clean: GenerationDataClean,
        packed_sequence: PackedSequence,
        sigmas: torch.Tensor,
        sigmas_action: torch.Tensor | None = None,
        sigmas_sound: torch.Tensor | None = None,
    ) -> GenerationDataNoised:
        """
        Diffusion / Flow matching forward process: apply noise of given noise level (sigmas) to input data.

        Args:
            gen_data_clean (GenerationDataClean): The input dataclass containing the clean data *latents* (tokens).
            packed_sequence (PackedSequence): Packed sequence with condition masks attached to modalities.
            sigmas (torch.Tensor): The noise levels. Shape [B,1] for base/teacher_forcing (all video
                latent frames share the same sigma) or [B,T_max] for diffusion_forcing (per-latent-frame
                independent sigma). T_max is the number of video latent frames (temporally compressed
                tokens), not RGB frames. In all modes, sigmas are multiplied by (1 - condition_mask)
                so conditioning latent frames get sigma_eff=0 and only non-conditioned frames contribute
                to the loss.
            sigmas_action: Optional ``[n_action, 1]`` override for action noising — dense over
                action-bearing samples, matching ``packed_sequence.action.*``. When None, action
                reuses ``sigmas`` (vision σ, legacy behavior). Set by ``training_step`` when
                ``independent_action_schedule=True``.
            sigmas_sound: Optional dense sound sigma tensor matching ``packed_sequence.sound.*``.
                When None, sound reuses ``sigmas``.

        Returns:
            GenerationDataNoised: A dataclass containing the noise, noisy data (xt), and velocity field (vt).
        """
        # Action sigma defaults to the shared vision sigma (legacy behavior).
        # Legacy (sigmas_action=None): vision σ of shape [B_items, T_vis].
        # Independent (sigmas_action provided): dense action σ of shape [n_action, 1].
        sigmas_for_action = sigmas if sigmas_action is None else sigmas_action  # [B_items,T_vis] or [n_action,1]
        # Sound uses a dense view of the per-sample vision schedule so mixed audio/no-audio
        # batches do not index full-batch sigmas with dense sound positions.
        sigmas_for_sound = sigmas if sigmas_sound is None else sigmas_sound  # [B_items,T_vis] or [n_sound,...]
        # Vision
        x0_vision = gen_data_clean.x0_tokens_vision  # list of [C,T,H,W]
        epsilon_vision = [
            torch.randn(x0_vision_i.size()).to(**self.tensor_kwargs_fp32) for x0_vision_i in x0_vision
        ]  # list of [C,T,H,W]

        # Derive noisy mask (1 for noised, 0 for clean) for sigmas computation
        assert packed_sequence.vision is not None, "Packed vision data required for noise scheduling"
        assert packed_sequence.vision.condition_mask is not None, "Vision condition mask required for noise scheduling"
        assert isinstance(packed_sequence.vision.condition_mask, list), (
            "Vision condition mask must be a list of tensors for noise scheduling"
        )

        # Compute sigmas per vision item (supports variable shapes).
        # For image editing, x0_tokens_vision is a flat list with multiple items per sample
        # and sigmas has already been expanded to match (see _expand_per_sample_to_per_vision_item).
        # Conditioning latent frames are zeroed via (1 - condition_mask) in all modes (base/TF/DF).
        # view(-1,1,1)[:T_latent]: for base/TF sigmas[i] is (1,), view gives (1,1,1) and the slice is a no-op;
        # for DF sigmas[i] is (T_max,) — one sigma per video latent frame — view gives (T_max,1,1)
        # and [:T_latent] slices to (T_latent,1,1) matching the per-item latent frame count.
        num_vision_items = len(packed_sequence.vision.condition_mask)
        noisy_mask_vision = [1.0 - cond_mask for cond_mask in packed_sequence.vision.condition_mask]
        sigmas_vision = [
            sigmas[i].view(-1, 1, 1)[: x0_vision[i].shape[2]] * noisy_mask_vision[i] for i in range(num_vision_items)
        ]
        rectified_flow_vision = (
            self.rectified_flow_image if gen_data_clean.is_image_batch else self.rectified_flow_video
        )
        xt_vision, vt_vision = rectified_flow_vision.get_interpolation(
            epsilon_vision, x0_vision, sigmas_vision
        )  # list of [C,T,H,W], list of [C,T,H,W]

        xt_vision = [
            xt_vision_i.to(**self.tensor_kwargs) for xt_vision_i in xt_vision
        ]  # list of [C,T,H,W]; to make tensor compatible with the precision of the model

        # Action (x0_tokens_action is already a dense list with no None entries).
        # Gate on action_gen: the dataset may emit action tensors for models that
        # don't consume them (e.g. camera dataset on a vision-only config), in
        # which case packed_sequence.action is None and we must skip this block.
        x0_action = gen_data_clean.x0_tokens_action  # list of [T,action_dim]
        if self.config.action_gen and x0_action is not None and len(x0_action) > 0:
            assert packed_sequence.action is not None, "Packed action data required when action tokens exist"
            assert packed_sequence.action.condition_mask is not None, (
                "Action condition mask required when action tokens exist"
            )
            action_batch_size = len(packed_sequence.action.condition_mask)
            all_actions_are_conditioning = all(
                torch.all(condition_mask == 1).item() for condition_mask in packed_sequence.action.condition_mask
            )
            if all_actions_are_conditioning:
                epsilon_action = [
                    torch.zeros(x0_action_i.size(), **self.tensor_kwargs_fp32) for x0_action_i in x0_action
                ]  # list of [T,action_dim]
                sigmas_action = [
                    torch.zeros_like(condition_mask, dtype=torch.float32, device=condition_mask.device)
                    for condition_mask in packed_sequence.action.condition_mask
                ]  # list of [T,1]
                xt_action = [
                    x0_action_i.to(**self.tensor_kwargs) for x0_action_i in x0_action
                ]  # list of [T,action_dim]
                vt_action = [
                    torch.zeros(x0_action_i.size(), **self.tensor_kwargs_fp32) for x0_action_i in x0_action
                ]  # list of [T,action_dim]
            else:
                epsilon_action = [
                    torch.randn(x0_action_i.size()).to(**self.tensor_kwargs_fp32) for x0_action_i in x0_action
                ]  # list of [T,action_dim]
                # Conditioning action timesteps are zeroed via (1 - condition_mask) in all modes (base/TF/DF).
                # Action timesteps are aligned 1-to-1 with video latent frames, not RGB frames.
                # view(-1,1)[:T_i]: for base/TF sigmas[i] is (1,) → (1,1), slice is a no-op;
                # for DF sigmas[i] is (T_max,) → (T_max,1) → (T_i,1) per-action-timestep sigmas.
                # condition_mask[i] shape [T_i,1]; result broadcasts with x0 shape [T_i,C].
                sigmas_action = [
                    sigmas_for_action[i].view(-1, 1)[: x0_action[i].shape[0]]
                    * (1.0 - packed_sequence.action.condition_mask[i])
                    for i in range(action_batch_size)
                ]  # list of [T_i,1]
                xt_action, vt_action = self.rectified_flow_action.get_interpolation(
                    epsilon_action, x0_action, sigmas_action
                )  # list of [T,action_dim], list of [T,action_dim]
                xt_action = [
                    xt_action_i.to(**self.tensor_kwargs) for xt_action_i in xt_action
                ]  # list of [T,action_dim]; to make tensor compatible with the precision of the model
            for i in range(len(xt_action)):
                if gen_data_clean.raw_action_dim is not None and gen_data_clean.raw_action_dim[i] is not None:
                    xt_action[i][:, gen_data_clean.raw_action_dim[i] :] = 0

        else:
            epsilon_action = None
            sigmas_action = None
            xt_action = None
            vt_action = None

        # Sound (x0_tokens_sound is a list of [C, T] tensors, or None)
        x0_sound = gen_data_clean.x0_tokens_sound  # list of [sound_channels,T_sound]
        if x0_sound is not None and len(x0_sound) > 0:
            assert packed_sequence.sound is not None, "Packed sound data required when sound tokens exist"
            assert packed_sequence.sound.condition_mask is not None, (
                "Sound condition mask required when sound tokens exist"
            )
            sound_batch_size = len(packed_sequence.sound.condition_mask)
            epsilon_sound = [
                torch.randn(x0_i.size()).to(**self.tensor_kwargs_fp32) for x0_i in x0_sound
            ]
            # Conditioning frames are zeroed via (1 - condition_mask) in all modes (base/TF/DF).
            # view(-1,1)[:T_sound].T: for base/TF sigmas[i] is (1,) → (1,1) → no-op → (1,1);
            # for DF sigmas[i] is (T_max,) → (T_max,1) → (T_sound,1) → (1,T_sound).
            # condition_mask[i] shape [T_sound,1]; .T gives [1,T_sound]; result broadcasts with x0 [C,T_sound].
            sigmas_sound = [
                sigmas_for_sound[i].view(-1, 1)[: x0_sound[i].shape[1]].T
                * (1.0 - packed_sequence.sound.condition_mask[i].T)
                for i in range(sound_batch_size)
            ]
            xt_sound, vt_sound = self.rectified_flow_sound.get_interpolation(epsilon_sound, x0_sound, sigmas_sound)
            xt_sound = [xt_i.to(**self.tensor_kwargs) for xt_i in xt_sound]
        else:
            epsilon_sound = None
            sigmas_sound = None
            xt_sound = None
            vt_sound = None

        # create the GenerationDataNoised object
        gen_data_noised = GenerationDataNoised(
            batch_size=gen_data_clean.batch_size,
            # vision
            epsilon_vision=epsilon_vision,
            xt_tokens_vision=xt_vision,
            vt_target_vision=vt_vision,
            sigmas_vision=sigmas_vision,
            # action
            epsilon_action=epsilon_action,
            xt_tokens_action=xt_action,
            vt_target_action=vt_action,
            sigmas_action=sigmas_action,
            raw_action_dim=gen_data_clean.raw_action_dim,
            # sound
            epsilon_sound=epsilon_sound,
            xt_tokens_sound=xt_sound,
            vt_target_sound=vt_sound,
            sigmas_sound=sigmas_sound,
        )

        return gen_data_noised

    def _replace_clean_with_noised(
        self,
        packed_sequence: PackedSequence,
        gen_data_noised: GenerationDataNoised,
    ) -> None:
        """Replace packed clean tokens with noised tokens."""
        if packed_sequence.vision is not None:
            packed_sequence.vision.tokens = gen_data_noised.xt_tokens_vision
        if packed_sequence.action is not None and gen_data_noised.xt_tokens_action is not None:
            action_all_conditioning = all(
                torch.all(condition_mask == 1).item() for condition_mask in packed_sequence.action.condition_mask
            )
            if not action_all_conditioning:
                packed_sequence.action.tokens = gen_data_noised.xt_tokens_action
        if packed_sequence.sound is not None and gen_data_noised.xt_tokens_sound is not None:
            packed_sequence.sound.tokens = gen_data_noised.xt_tokens_sound

    # ------------------------ Inference Utils ------------------------
    def _get_inference_text_tokens(
        self, data_batch: dict, has_negative_prompt: bool
    ) -> tuple[list[list[int]], list[list[int]]]:
        """Tokenize conditional and unconditional captions for inference."""
        use_system_prompt = self.vlm_config.use_system_prompt
        system_prompt: str | None = data_batch.get("system_prompt")

        cond_tokens = [
            tokenize_caption(
                c,
                self.vlm_tokenizer,
                is_video=False,
                use_system_prompt=use_system_prompt,
                system_prompt=system_prompt,
            )
            for c in data_batch[self.input_caption_key]
        ]

        if has_negative_prompt:
            neg_key = "neg_" + self.input_caption_key
            assert neg_key in data_batch, f"Negative prompt ({neg_key}) not found"
            uncond_captions = data_batch[neg_key]
        else:
            uncond_captions = [""] * len(cond_tokens)

        uncond_tokens = [
            tokenize_caption(
                c,
                self.vlm_tokenizer,
                is_video=False,
                use_system_prompt=use_system_prompt,
                system_prompt=system_prompt,
            )
            for c in uncond_captions
        ]
        return cond_tokens, uncond_tokens

    def _prepare_inference_data(
        self,
        data_batch: dict,
        seed: list[int],
        has_negative_prompt: bool = False,
    ) -> tuple[
        list[SequencePlan],
        GenerationDataClean,
        list[list[int]],
        list[list[int]],
        list[torch.Tensor],
    ]:
        """
        Prepare all data needed for inference sampling.
        Mirrors training_step's data preparation flow.

        This method:
        1. Builds sequence plans (conditioning information)
        2. Gets data and condition (encodes vision)
        3. Tokenizes text (conditional and unconditional for CFG)
        4. Builds a packed sequence to fetch conditioning masks
        5. Initializes noise with conditioning applied (as lists for variable shapes)
        6. If action_gen is True, concatenates action noise with vision noise

        Args:
            data_batch: Raw data batch from dataloader.
            seed: Random seed(s) for noise generation.
            has_negative_prompt: If True, use negative prompt for unconditional branch.

        Returns:
            Tuple of:
                - sequence_plans: List of SequencePlan objects
                - gen_data_clean: GenerationDataClean with encoded tokens
                - cond_text_tokens: Conditional text tokens
                - uncond_text_tokens: Unconditional text tokens (for CFG)
                - initial_noise: List of noise tensors (one per sample), each containing
                  flattened vision (and optionally action) noise concatenated
        """
        # 1. Build sequence plans (same as training)
        sequence_plans = build_sequence_plans_from_data_batch(
            data_batch=data_batch,
            input_video_key=self.input_video_key,
            input_image_key=self.input_image_key,
        )

        # 2. Get data and condition (same as training)
        # This encodes vision to x0_tokens
        gen_data_clean = self.get_data_and_condition(data_batch)

        num_items_per_sample = gen_data_clean.num_vision_items_per_sample  # None for standard T2I/T2V

        # 3. Tokenize text (similar to training's _load_and_tokenize_text_data)
        cond_text_tokens, uncond_text_tokens = self._get_inference_text_tokens(data_batch, has_negative_prompt)

        # 4. Build packed sequence to fetch conditioning masks
        mask_timesteps = torch.zeros((gen_data_clean.batch_size,), dtype=torch.float32)  # [B]
        packed_sequence = self._pack_input_sequence(
            sequence_plans,
            cond_text_tokens,
            gen_data_clean,
            mask_timesteps,
            include_end_of_generation_token=self._derive_include_end_of_generation_token(),
        )

        # 5. Initialize vision noise with conditioning
        assert packed_sequence.vision is not None, "Packed vision data required for inference noise"
        assert packed_sequence.vision.condition_mask is not None, "Vision condition mask required for inference noise"
        assert isinstance(packed_sequence.vision.condition_mask, list), (
            "Vision condition mask must be a list of tensors for inference noise"
        )
        assert gen_data_clean.x0_tokens_vision is not None, "Vision data required for inference noise"
        n_sample = (
            len(gen_data_clean.x0_tokens_vision)
            if gen_data_clean.num_vision_items_per_sample is None
            else len(gen_data_clean.num_vision_items_per_sample)
        )

        assert len(seed) == n_sample, (
            f"Seed list length {len(seed)} must have the same length as the number of samples {n_sample}"
        )

        # For image2image, num_items_per_sample could be > 1 (multi-vision),
        # so we need to repeat the seed for each vision item.
        seed_dict = {"vision": [], "action": [], "sound": []}
        for sample_idx in range(n_sample):
            num_vision_items = num_items_per_sample[sample_idx] if num_items_per_sample is not None else 1
            seed_dict["vision"].extend([seed[sample_idx]] * num_vision_items)
            seed_dict["action"].append(seed[sample_idx])
            seed_dict["sound"].append(seed[sample_idx])

        # Generate noise and apply conditioning per vision item (supports variable shapes)
        noise_vision_list: list[torch.Tensor] = []
        for i, (x0_token, cond_mask) in enumerate(
            zip(gen_data_clean.x0_tokens_vision, packed_sequence.vision.condition_mask, strict=True)
        ):
            pure_noise_i = misc.arch_invariant_rand(
                tuple(x0_token.shape),
                self.tensor_kwargs["dtype"],
                self.tensor_kwargs["device"],
                seed_dict["vision"][i],  # Different seed per sample for diversity
            )  # [C,T,H,W]
            noise_i = cond_mask * x0_token.to(**self.tensor_kwargs) + (1.0 - cond_mask) * pure_noise_i  # [C,T,H,W]
            noise_vision_list.append(noise_i)

        # 6. Initialize action noise if action_gen is True
        has_action = self.config.action_gen and any(plan.has_action for plan in sequence_plans)
        noise_action_list: list[torch.Tensor] | None = None

        if has_action:
            assert gen_data_clean.x0_tokens_action is not None, "Action data required when sequence plan has action"
            assert packed_sequence.action is not None, "Packed action data required when action_gen is True"
            assert packed_sequence.action.condition_mask is not None, "Action condition mask required"
            assert isinstance(packed_sequence.action.condition_mask, list), (
                "Action condition mask must be a list of tensors for inference noise"
            )

            # Generate action noise per sample (x0_tokens_action is already dense, no None entries)
            noise_action_list = []
            for i, (x0_action, cond_mask_action) in enumerate(
                zip(gen_data_clean.x0_tokens_action, packed_sequence.action.condition_mask, strict=True)
            ):
                pure_noise_action_i = misc.arch_invariant_rand(
                    tuple(x0_action.shape),
                    self.tensor_kwargs["dtype"],
                    self.tensor_kwargs["device"],
                    seed_dict["action"][i],  # Different seed per sample for diversity
                )  # [T,action_dim]
                noise_action_i = (
                    cond_mask_action * x0_action.to(**self.tensor_kwargs)
                    + (1.0 - cond_mask_action) * pure_noise_action_i
                )
                if gen_data_clean.raw_action_dim is not None and gen_data_clean.raw_action_dim[i] is not None:
                    noise_action_i[:, gen_data_clean.raw_action_dim[i] :] = 0
                noise_action_list.append(noise_action_i)

        # 7. Initialize sound noise if sound_gen is True
        has_sound = self.config.sound_gen and any(plan.has_sound for plan in sequence_plans)
        noise_sound_list: list[torch.Tensor] | None = None

        if has_sound:
            assert gen_data_clean.x0_tokens_sound is not None, "Sound data required when sequence plan has sound"
            assert packed_sequence.sound is not None, "Packed sound data required when sound_gen is True"
            assert packed_sequence.sound.condition_mask is not None, "Sound condition mask required"
            assert isinstance(packed_sequence.sound.condition_mask, list), (
                "Sound condition mask must be a list of tensors for inference noise"
            )

            noise_sound_list = []
            for i, (x0_sound, cond_mask_sound) in enumerate(
                zip(gen_data_clean.x0_tokens_sound, packed_sequence.sound.condition_mask, strict=True)
            ):
                pure_noise_sound_i = misc.arch_invariant_rand(
                    tuple(x0_sound.shape),
                    self.tensor_kwargs["dtype"],
                    self.tensor_kwargs["device"],
                    seed_dict["sound"][i],  # Different seed per sample for diversity
                )  # [sound_channels,T_sound]
                # cond_mask_sound is (T, 1), x0_sound is (C, T) — transpose mask for broadcasting
                noise_sound_i = (
                    cond_mask_sound.T * x0_sound.to(**self.tensor_kwargs)
                    + (1.0 - cond_mask_sound.T) * pure_noise_sound_i
                )  # [sound_channels,T_sound]
                noise_sound_list.append(noise_sound_i)

        # 8. Concatenate vision, action, and sound noise per sample (flattened)
        # Order: [vision | action (if present) | sound (if present)]
        # noise_action_list and noise_sound_list are dense (only modality-having samples),
        # so we use separate indexes.
        initial_noise: list[torch.Tensor] = []
        idx_vision = 0
        idx_action = 0
        idx_sound = 0

        for i in range(n_sample):
            parts = []

            # Flatten and concatenate all vision items for this sample
            num_vis = num_items_per_sample[i] if num_items_per_sample is not None else 1
            for _ in range(num_vis):
                parts.append(noise_vision_list[idx_vision].reshape(-1))
                idx_vision += 1

            if noise_action_list is not None and sequence_plans[i].has_action:
                parts.append(noise_action_list[idx_action].reshape(-1))
                idx_action += 1

            if noise_sound_list is not None and sequence_plans[i].has_sound:
                parts.append(noise_sound_list[idx_sound].reshape(-1))
                idx_sound += 1

            initial_noise.append(torch.cat(parts, dim=0))  # [N_tokens_flat]

        return (
            sequence_plans,
            gen_data_clean,
            cond_text_tokens,
            uncond_text_tokens,
            initial_noise,
        )

    def _get_velocity(
        self,
        *,
        net: torch.nn.Module | None = None,
        noise_x: list[torch.Tensor],
        timestep: torch.Tensor,
        text_tokens: list[list[int]],
        sequence_plans: list[SequencePlan],
        gen_data_clean: GenerationDataClean,
        skip_text_tokens: bool = False,
    ) -> list[torch.Tensor]:
        """
        Compute velocity prediction for a single sampling step.

        This method handles the full pipeline for one denoising step:
        1. Splits flattened noise_x into vision (and action) parts per sample
        2. Packs the input sequence with current noisy latents
        3. Runs the network via self.denoise()
        4. Applies velocity masks (zeroes out conditioned parts)
        5. Returns flattened velocities (concatenated vision + action per sample)

        Args:
            noise_x: List of noisy latents, each containing concatenated
                     vision (and optionally action) noise.
                     len(noise_x) == B, noise_x[i] is shape (D)
            timestep: Current timestep for each sample
            text_tokens: Tokenized text for each sample
            sequence_plans: Pre-computed sequence plans (from _prepare_inference_data)
            gen_data_clean: Pre-computed clean data (from _prepare_inference_data)
            skip_text_tokens: If True, skip text tokens (for CFG unconditional branch)

        Returns:
            Stacked flattened velocity tensors (one per sample), each containing
            concatenated vision (and optionally action) velocity
        """
        n_samples = len(noise_x)
        is_image_batch = gen_data_clean.is_image_batch
        has_action = self.config.action_gen and any(plan.has_action for plan in sequence_plans)
        num_items = gen_data_clean.num_vision_items_per_sample  # None for standard T2I/T2V
        has_sound = self.config.sound_gen and any(plan.has_sound for plan in sequence_plans)

        # Split flattened noise_x into vision, action, and sound parts per sample
        # Order must match _prepare_inference_data: [vision | action (if present) | sound (if present)]
        noise_x_vision: list[torch.Tensor] = []
        noise_x_action: list[torch.Tensor] | None = [] if has_action else None
        noise_x_sound: list[torch.Tensor] | None = [] if has_sound else None

        vision_offset = 0  # tracks position in the flat x0_tokens_vision list
        idx_action = 0
        idx_sound = 0
        for i in range(n_samples):
            n_vis = num_items[i] if num_items is not None else 1
            offset = 0
            for j in range(n_vis):
                vision_shape = gen_data_clean.x0_tokens_vision[vision_offset + j].shape
                vision_dim = int(torch.prod(torch.tensor(vision_shape)))
                noise_vision_ij = noise_x[i][offset : offset + vision_dim].reshape(vision_shape)
                noise_x_vision.append(noise_vision_ij)
                offset += vision_dim
            vision_offset += n_vis

            if has_action and noise_x_action is not None:
                assert gen_data_clean.x0_tokens_action is not None
                action_shape = gen_data_clean.x0_tokens_action[idx_action].shape
                action_dim = int(torch.prod(torch.tensor(action_shape)))
                noise_x_action.append(noise_x[i][offset : offset + action_dim].reshape(action_shape))  # [T,action_dim]
                offset += action_dim
                idx_action += 1

            # Extract sound if present for this sample
            if has_sound and noise_x_sound is not None and sequence_plans[i].has_sound:
                assert gen_data_clean.x0_tokens_sound is not None
                sound_shape = gen_data_clean.x0_tokens_sound[idx_sound].shape
                sound_dim = int(torch.prod(torch.tensor(sound_shape)))
                noise_x_sound.append(
                    noise_x[i][offset : offset + sound_dim].reshape(sound_shape)
                )  # [sound_channels,T_sound]
                offset += sound_dim
                idx_sound += 1

        gen_data_for_packing = GenerationDataClean(
            batch_size=n_samples,
            is_image_batch=is_image_batch,
            raw_state_vision=gen_data_clean.raw_state_vision,
            x0_tokens_vision=noise_x_vision,
            fps_vision=gen_data_clean.fps_vision,
            # Action fields
            raw_state_action=gen_data_clean.raw_state_action if has_action else None,
            x0_tokens_action=noise_x_action if has_action else None,
            action_domain_id=gen_data_clean.action_domain_id if has_action else None,
            fps_action=gen_data_clean.fps_action if has_action else None,
            raw_action_dim=gen_data_clean.raw_action_dim if has_action else None,
            # Sound fields
            raw_state_sound=gen_data_clean.raw_state_sound if has_sound else None,
            x0_tokens_sound=noise_x_sound if has_sound else None,
            fps_sound=gen_data_clean.fps_sound if has_sound else None,
            num_vision_items_per_sample=num_items,
        )

        packed_sequence = self._pack_input_sequence(
            sequence_plans,
            text_tokens,
            gen_data_for_packing,
            timestep.cpu(),
            include_end_of_generation_token=self._derive_include_end_of_generation_token(),
            skip_text_tokens=skip_text_tokens,
        )

        # Set the actual noisy latents (as lists)
        if packed_sequence.vision is not None:
            packed_sequence.vision.tokens = [x.to(**self.tensor_kwargs) for x in noise_x_vision]

        if has_action and noise_x_action is not None:
            assert packed_sequence.action is not None, "packed_sequence.action must exist when has_action is True"
            packed_sequence.action.tokens = [x.to(**self.tensor_kwargs) for x in noise_x_action]
            packed_sequence.action.domain_id = gen_data_clean.action_domain_id

        if has_sound and noise_x_sound is not None:
            assert packed_sequence.sound is not None, "packed_sequence.sound must exist when has_sound is True"
            packed_sequence.sound.tokens = [x.to(**self.tensor_kwargs) for x in noise_x_sound]

        packed_sequence.to_cuda()

        # --- Network forward ---
        fps_action = gen_data_clean.fps_action if has_action else None
        fps_sound = gen_data_clean.fps_sound if has_sound else None
        out = self.denoise(
            net=net,
            data_batch_packed=packed_sequence,
            fps_vision=gen_data_clean.fps_vision,
            fps_action=fps_action,
            fps_sound=fps_sound,
        )

        # --- Apply velocity masks ---
        # Zero out velocity for conditioned parts (they don't change during sampling)
        assert packed_sequence.vision is not None, "packed_sequence.vision must exist for velocity masking"
        assert packed_sequence.vision.condition_mask is not None, "Vision condition mask required for masking"
        assert isinstance(packed_sequence.vision.condition_mask, list), (
            "Vision condition mask must be a list of tensors for masking"
        )
        # Compute noisy_mask per sample (supports variable shapes)
        noisy_mask_vision = [1.0 - cond_mask for cond_mask in packed_sequence.vision.condition_mask]

        # Apply velocity mask per element - check if each sample has noisy tokens
        velocity_vision: list[torch.Tensor] = []
        for i, (pred, noisy_mask) in enumerate(zip(out["preds_vision"], noisy_mask_vision)):
            # pred: [C,T,H,W], noisy_mask: [T,1,1]
            has_noisy_tokens_i = noisy_mask.sum() > 0
            if has_noisy_tokens_i:
                # Apply mask to prediction
                velocity_vision.append(pred * noisy_mask.to(dtype=pred.dtype, device=pred.device))  # [C,T,H,W]
            else:
                # All tokens are conditioned - velocity should be zero
                velocity_vision.append(torch.zeros_like(pred))  # [C,T,H,W]

        # Handle action velocity
        velocity_action: list[torch.Tensor] | None = None
        if (
            has_action
            and packed_sequence.action is not None
            and packed_sequence.action.condition_mask is not None
            and isinstance(packed_sequence.action.condition_mask, list)
        ):
            noisy_mask_action = [1.0 - cond_mask for cond_mask in packed_sequence.action.condition_mask]

            velocity_action = []
            for i, (pred, noisy_mask) in enumerate(zip(out["preds_action"], noisy_mask_action)):
                # pred: [T,action_dim], noisy_mask: [T,1]
                has_noisy_tokens_i = noisy_mask.sum() > 0
                if has_noisy_tokens_i:
                    v = pred * noisy_mask.to(dtype=pred.dtype, device=pred.device)  # [T,action_dim]
                else:
                    v = torch.zeros_like(pred)  # [T,action_dim]
                if gen_data_clean.raw_action_dim is not None and gen_data_clean.raw_action_dim[i] is not None:
                    v[:, gen_data_clean.raw_action_dim[i] :] = 0
                velocity_action.append(v)

        # Handle sound velocity
        velocity_sound: list[torch.Tensor] | None = None
        if (
            has_sound
            and packed_sequence.sound is not None
            and packed_sequence.sound.condition_mask is not None
            and isinstance(packed_sequence.sound.condition_mask, list)
        ):
            noisy_mask_sound = [1.0 - cond_mask for cond_mask in packed_sequence.sound.condition_mask]

            velocity_sound = []
            for i, (pred, noisy_mask) in enumerate(zip(out["preds_sound"], noisy_mask_sound)):
                # pred: [sound_channels,T_sound], noisy_mask: [T_sound,1]
                has_noisy_tokens_i = noisy_mask.sum() > 0
                if has_noisy_tokens_i:
                    # noisy_mask is (T, 1), pred is (C, T) — transpose mask for broadcasting
                    velocity_sound.append(
                        pred * noisy_mask.T.to(dtype=pred.dtype, device=pred.device)
                    )  # [sound_channels,T_sound]
                else:
                    velocity_sound.append(torch.zeros_like(pred))  # [sound_channels,T_sound]

        # Concatenate vision, action, and sound velocities per sample (flattened)
        # Order must match _prepare_inference_data: [vision | action | sound]
        velocity_output: list[torch.Tensor] = []
        vis_offset = 0
        idx_action = 0
        idx_sound = 0
        for i in range(n_samples):
            parts = []
            n_vis = num_items[i] if num_items is not None else 1

            for _ in range(n_vis):
                parts.append(velocity_vision[vis_offset].reshape(-1))
                vis_offset += 1

            if velocity_action is not None and sequence_plans[i].has_action:
                parts.append(velocity_action[idx_action].reshape(-1))
                idx_action += 1

            if velocity_sound is not None and sequence_plans[i].has_sound:
                parts.append(velocity_sound[idx_sound].reshape(-1))
                idx_sound += 1

            velocity_output.append(torch.cat(parts, dim=0))  # [N_tokens_flat]

        return velocity_output

    def _remove_padding_from_latent(
        self, x0_tokens_vision: list[torch.Tensor], frame_size: list[torch.Tensor]
    ) -> list[torch.Tensor]:
        """
        Remove reflection padding from encoded latent vision tokens.

        Each sample in the batch may have different original dimensions, so we process
        each sample individually and return a list of latents with varying spatial sizes.

        The padding coordinates are scaled down by the spatial compression factor since
        we're operating in latent space.

        Args:
            x0_tokens_vision (list[torch.Tensor]): List of encoded latent tensors,
                each of shape (1, C, T, H_latent, W_latent)
                where H_latent, W_latent include scaled padding.
            frame_size (list[torch.Tensor]): List of tensors, each of shape (1,4) or (4,) containing
                [target_h, target_w, orig_h, orig_w] for each sample (in pixel space).

        Returns:
            list[torch.Tensor]: List of cropped latent tokens, each of shape (1, C, T, H_latent_cropped, W_latent_cropped).
                Each element may have different spatial sizes based on original image dimensions.
        """
        batch_size = len(x0_tokens_vision)
        spatial_factor = self.tokenizer_vision_gen.spatial_compression_factor
        cropped_latents = []
        for i in range(batch_size):
            # frame_size: [target_h, target_w, orig_h, orig_w] in pixel space
            # Normalize: frame_size[i] may be (1, 4) from IterativeJointDataLoader
            # or (4,) when loaded from safetensors in the eval/export path.
            fs = frame_size[i]
            if fs.dim() == 2:
                fs = fs[0]
            orig_h = int(fs[2].item())
            orig_w = int(fs[3].item())

            # Scale to latent space
            if orig_h // spatial_factor == 0 or orig_w // spatial_factor == 0:
                log.warning(
                    f"Zero-sized latent found: orig_h: {orig_h}, orig_w: {orig_w}, spatial_factor: {spatial_factor}"
                )

            orig_h_latent = max(orig_h // spatial_factor, 1)
            orig_w_latent = max(orig_w // spatial_factor, 1)

            # Crop to remove padding: x0_tokens_vision[i] shape is (1, C, T, H, W)
            cropped_latent = x0_tokens_vision[i][:, :, :, :orig_h_latent, :orig_w_latent].contiguous()
            cropped_latents.append(cropped_latent)

        return cropped_latents

    def _run_classifier_free_guidance(
        self,
        cond_tokens: list[list[int]],
        uncond_tokens: list[list[int]],
        skip_text_tokens_for_cfg: bool,
        single_velocity_fn: Callable[[list[list[int]], bool], list[torch.Tensor]],
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Run classifier-free guidance, optionally in parallel via CFG parallelism.

        Args:
            cond_tokens: Tokenized text for the conditional branch.
            uncond_tokens: Tokenized text for the unconditional branch.
            skip_text_tokens_for_cfg: If True, skip text tokens in the
                unconditional branch.
            single_velocity_fn: Computes velocity for a given set of tokens.
                Accepts ``(tokens, skip_text_tokens)`` and returns a list of
                velocity tensors (one per sample).

        Returns:
            A tuple ``(cond_v, uncond_v)`` where each element is a list of
            velocity tensors (one per sample).
        """
        if self.parallel_dims is None or not self.parallel_dims.cfgp_enabled:
            return (
                single_velocity_fn(cond_tokens, False),
                single_velocity_fn(uncond_tokens, skip_text_tokens_for_cfg),
            )

        cfgp_rank = self.parallel_dims.cfgp_rank
        cfgp_size = self.parallel_dims.cfgp_size
        cfgp_group = self.parallel_dims.cfgp_mesh.get_group()
        cfgp_peer = (cfgp_rank + 1) % cfgp_size

        if cfgp_rank == 0:
            v_list = single_velocity_fn(cond_tokens, False)
        else:
            v_list = single_velocity_fn(uncond_tokens, skip_text_tokens_for_cfg)

        other_v_list = [torch.empty_like(v_i) for v_i in v_list]

        ops: list[dist.P2POp] = []
        for v_i, other_v_i in zip(v_list, other_v_list):
            ops.append(dist.P2POp(op=dist.isend, tensor=v_i, group_peer=cfgp_peer, group=cfgp_group))
            ops.append(dist.P2POp(op=dist.irecv, tensor=other_v_i, group_peer=cfgp_peer, group=cfgp_group))

        reqs = dist.batch_isend_irecv(ops)
        for req in reqs:
            req.wait()

        if cfgp_rank == 0:
            return v_list, other_v_list
        else:
            return other_v_list, v_list

    @torch.no_grad()
    def generate_samples_from_batch(
        self,
        data_batch: Dict,
        net: torch.nn.Module | None = None,
        sampler: Any | None = None,
        guidance: float = 1.5,
        guidance_interval: Optional[list[float]] = None,
        seed: list[int] | int = 1,
        n_sample: int | None = None,
        has_negative_prompt: bool = False,
        num_steps: int = 35,
        shift: float = 5.0,
        sigma_max: float = 80.0,
        skip_text_tokens_for_cfg: bool = False,
        normalize_cfg: bool = False,
        **kwargs,
    ) -> dict[str, list[torch.Tensor]]:
        """
        Generate samples from the batch. Based on given batch, it will automatically determine
        whether to generate image or video samples.

        This method follows the same structure as training_step:
        1. Build sequence plans
        2. Get data and condition (encode vision)
        3. Initialize noise with conditioning (as lists for variable shapes)
        4. Run sampling loop with velocity function
        5. Return latents as lists (supports variable shapes)

        Args:
            data_batch (dict): Raw data batch from the dataloader.
            guidance (float): Classifier-free guidance weight.
            guidance_interval (list[float] | None): Optional timestep interval to apply guidance.
                For the timesteps (ranging between 0-1000) that fall between the interval, we perform CFG, otherwise, we skip the unconditional generation.
            seed (list[int] | int): Random seeds for noise generation. For all new use-cases,
                we use a list of seeds, one for each sample. The length of the list must match
                the number of samples. Legacy use-cases use a single integer seed which is
                incremented by 1 for each sample. But this is not supported anymore, and will
                raise an error if used.
            n_sample (int | None): Number of samples to generate; defaults to batch size.
            has_negative_prompt (bool): If True, use negative prompt for unconditional branch.
            num_steps (int): Number of sampling steps for the diffusion process.
            shift (float): Time shift parameter for the sampler.
            sigma_max (float): Maximum sigma for the EDM sampler.
            skip_text_tokens_for_cfg (bool): If True, skip text tokens in unconditional branch.
            normalize_cfg (bool): If True, normalize the CFG output.

        Returns:
            Dict with keys:
                - "vision": List of vision latent tensors (one per sample, variable shapes)
                - "action": List of action latent tensors or None (only present when action_gen=True and has_action)

        Raises:
            ValueError: If the number of samples does not match the number of noise tensors or seeds.
            ValueError: If the seed is a single integer. This is not supported anymore: `seed` must be
                a list of integers, one for each sample.
        """
        if isinstance(seed, int):
            raise ValueError(
                "Single integer seed is not supported anymore: `seed` must be a list of integers, one for each sample."
            )
        assert isinstance(seed, list)

        if self.parallel_dims is not None and self.parallel_dims.cp_enabled:
            seed = _broadcast_seed(seed, self.parallel_dims.cp_mesh.get_group(), self.parallel_dims.cp_rank)

        if self.parallel_dims is not None and self.parallel_dims.cfgp_enabled:
            seed = _broadcast_seed(seed, self.parallel_dims.cfgp_mesh.get_group(), self.parallel_dims.cfgp_rank)

        # Prepare all data (initial noise as list of flattened tensors per sample)
        (
            sequence_plans,
            gen_data_clean,
            cond_tokens,
            uncond_tokens,
            initial_noise,
        ) = self._prepare_inference_data(data_batch, seed, has_negative_prompt)

        if n_sample is not None:
            assert n_sample == len(initial_noise), (
                f"Number of samples {n_sample} must match number of noise tensors {len(initial_noise)}"
            )
        else:
            n_sample = len(initial_noise)

        assert n_sample == len(seed), f"Number of samples {n_sample} must match number of seeds {len(seed)}"

        # Create a velocity function for a single sample (for use with self.sampler).

        def velocity_fn(noise_x: list[torch.Tensor], timestep: torch.Tensor) -> list[torch.Tensor]:
            # len(noise_x) == B, noise_x[i] is shape (D)
            # timestep is shape (B, 1)
            torch.compiler.cudagraph_mark_step_begin()

            assert timestep.ndim == 2, f"timestep must be 2D, got {timestep.shape}"
            assert timestep.shape == (1, 1), f"timestep must be (1, 1), got {timestep.shape}"

            # Expand timestep to (B, 1)
            timestep = timestep.repeat(len(noise_x), 1)

            def _single_velocity_fn(tokens: list[list[int]], skip_text_tokens: bool):
                return self._get_velocity(
                    net=net,
                    noise_x=noise_x,
                    timestep=timestep,
                    text_tokens=tokens,
                    sequence_plans=sequence_plans,
                    gen_data_clean=gen_data_clean,
                    skip_text_tokens=skip_text_tokens,
                )

            # Skip unconditional branch when outside the guidance interval
            needs_cfg = guidance != 1.0
            if needs_cfg and guidance_interval is not None:
                assert len(guidance_interval) == 2, f"guidance_interval must be [lo, hi], got {guidance_interval}"
                t_lo, t_hi = guidance_interval
                needs_cfg = t_lo < timestep[0].item() < t_hi

            if not needs_cfg:
                return _single_velocity_fn(cond_tokens, skip_text_tokens=False)

            cond_v, uncond_v = self._run_classifier_free_guidance(
                cond_tokens=cond_tokens,
                uncond_tokens=uncond_tokens,
                skip_text_tokens_for_cfg=skip_text_tokens_for_cfg,
                single_velocity_fn=_single_velocity_fn,
            )

            v_pred = [u_i + guidance * (c_i - u_i) for c_i, u_i in zip(cond_v, uncond_v)]

            if normalize_cfg:
                v_pred = [
                    v_i * (torch.norm(c_i) / (torch.norm(v_i) + 1e-8)).clamp(min=0.0, max=1.0)
                    for v_i, c_i in zip(v_pred, cond_v)
                ]

            return v_pred

        # Run sampler for all samples at once.
        sampler = sampler or self.sampler
        scheduler_type = self.config.rectified_flow_inference_config.scheduler_type
        if scheduler_type == "unipc":
            log.info(f"Using sampler: UniPC (shift={shift}, num_steps={num_steps})")
        else:
            log.info(f"Using sampler: EDM (sigma_max={sigma_max}, num_steps={num_steps})")

        if scheduler_type == "unipc":
            latents = sampler(
                velocity_fn,
                initial_noise,
                num_steps=num_steps,
                shift=shift,
                seed=seed,
            )
        else:
            # EDM Sampler
            chunk_sizes = [_x.shape[0] for _x in initial_noise]
            initial_noise = torch.cat(initial_noise, dim=0)

            def x0_fn(noise_x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
                assert sigma.ndim == 0, f"sigma must be 0D, got {sigma.shape}"
                timestep_rf = sigma * float(self.config.rectified_flow_inference_config.num_train_timesteps)

                # Convert noise_x to list of tensors for velocity_fn, and then
                # concatenate the results back into a single tensor.
                _noise_x = list(torch.split(noise_x, chunk_sizes, dim=0))
                _velocity_pred = velocity_fn(_noise_x, timestep_rf.reshape(1, 1))
                velocity_pred = torch.cat(_velocity_pred, dim=0)

                x0_pred = noise_x - sigma * velocity_pred
                return x0_pred

            latents = sampler(
                x0_fn,
                initial_noise,
                num_steps=num_steps,
                sigma_max=sigma_max,
                sigma_min=0.002,
                solver_option="2ab",
            )
            latents = list(torch.split(latents, chunk_sizes, dim=0))

        # Split flattened latents back into vision, action, and sound
        # Mirror the per-sample logic from _prepare_inference_data:
        # Order: [vision | action (if present) | sound (if present)]
        # action/sound lists are dense (only modality-having samples), so use separate indexes.
        result_vision: list[torch.Tensor] = []
        result_action: list[torch.Tensor] = []
        result_sound: list[torch.Tensor] = []
        idx_vision = 0
        idx_action = 0
        idx_sound = 0
        num_vision_items = gen_data_clean.num_vision_items_per_sample

        for i in range(n_sample):
            offset = 0

            # Extract vision
            n_vis = num_vision_items[i] if num_vision_items is not None else 1
            for j in range(n_vis):
                vision_shape = gen_data_clean.x0_tokens_vision[idx_vision + j].shape
                vision_dim = int(torch.prod(torch.tensor(vision_shape)))
                if j == n_vis - 1:  # the last vision item is the only target for each sample.

                    result_vision.append(latents[i][offset : offset + vision_dim].reshape(vision_shape))
                else:  # the other vision items are the condition inputs that we don't need to return
                    pass
                offset += vision_dim
            idx_vision += n_vis

            # Extract action if present
            if self.config.action_gen and sequence_plans[i].has_action:
                assert gen_data_clean.x0_tokens_action is not None
                action_shape = gen_data_clean.x0_tokens_action[idx_action].shape
                action_dim = int(torch.prod(torch.tensor(action_shape)))
                result_action.append(latents[i][offset : offset + action_dim].reshape(action_shape))
                offset += action_dim
                idx_action += 1

            # Extract sound if present
            if self.config.sound_gen and sequence_plans[i].has_sound:
                assert gen_data_clean.x0_tokens_sound is not None
                sound_shape = gen_data_clean.x0_tokens_sound[idx_sound].shape
                sound_dim = int(torch.prod(torch.tensor(sound_shape)))
                result_sound.append(latents[i][offset : offset + sound_dim].reshape(sound_shape))
                offset += sound_dim
                idx_sound += 1

        result: dict[str, list[torch.Tensor]] = {"vision": result_vision}
        if self.config.action_gen and len(result_action) > 0:
            result["action"] = result_action
        if self.config.sound_gen and len(result_sound) > 0:
            result["sound"] = result_sound
        return result

    def _extract_condition_images_for_visualization(
        self,
        gen_data_clean: GenerationDataClean,
        sequence_plans: list[SequencePlan],
        n_samples: int,
    ) -> list[torch.Tensor | None]:
        """Extract condition images from gen_data_clean for visualization.

        For image editing, raw_state_vision is a flat list of individually-encoded
        images (e.g. [src1, tgt1, src2, tgt2, ...]).  The first vision item for
        each sample is the condition (source) image.  This method extracts it and
        resizes to match the target for side-by-side display.

        Args:
            gen_data_clean: Clean data containing raw vision states.
            sequence_plans: Sequence plans for each sample.
            n_samples: Number of samples to process.

        Returns:
            List of condition image tensors (one per sample with condition frames).
        """
        condition_images: list[torch.Tensor | None] = []

        if gen_data_clean.num_vision_items_per_sample is not None:
            # Multi-item (image editing): raw_state_vision is flat [src1, tgt1, src2, tgt2, ...]
            vision_offset = 0
            for i in range(n_samples):
                num_items = gen_data_clean.num_vision_items_per_sample[i]
                if num_items >= 2:
                    cond_frame = gen_data_clean.raw_state_vision[vision_offset]  # (1, C, 1, H_s, W_s)
                    target_frame = gen_data_clean.raw_state_vision[vision_offset + 1]  # (1, C, 1, H_t, W_t)
                    # Resize condition frame to match target size for visualization
                    if cond_frame.shape[-2:] != target_frame.shape[-2:]:
                        cond_frame = torch.nn.functional.interpolate(
                            cond_frame.squeeze(2),  # (1, C, H, W)
                            size=target_frame.shape[-2:],
                            mode="bilinear",
                            align_corners=False,
                        ).unsqueeze(2)  # (1, C, 1, H, W)
                    condition_images.append(cond_frame)
                else:
                    condition_images.append(None)
                vision_offset += num_items
        else:
            # Standard single-item mode: check condition_frame_indexes_vision
            for i in range(n_samples):
                plan = sequence_plans[i]
                if len(plan.condition_frame_indexes_vision) > 0 and gen_data_clean.raw_state_vision is not None:
                    raw_vision = gen_data_clean.raw_state_vision[i]  # (1, C, T, H, W)
                    condition_images.append(raw_vision[:, :, 0:1, :, :])
                else:
                    condition_images.append(None)

        return condition_images

    def _slice_gen_data_clean(self, gen_data_clean: GenerationDataClean, start: int, limit: int) -> GenerationDataClean:
        """Extract a subset of GenerationDataClean for inference.

        The samples in [start:limit] are extracted from the original GenerationDataClean.

        For image editing (``num_vision_items_per_sample`` is set), the sample index refers to
        the *real sample* index. The method computes the correct slice of the flat
        ``x0_tokens_vision`` / ``raw_state_vision`` lists using the item counts and
        preserves ``num_vision_items_per_sample`` on the returned subset so that
        downstream packing works correctly.

        Args:
            gen_data_clean: GenerationDataClean to slice.
            start: Start index of the slice.
            limit: Limit index of the slice.

        Returns:
            Sliced GenerationDataClean.
        """
        # x0_tokens_action can be an empty list (e.g. image2video mode), not just None
        has_action = bool(gen_data_clean.x0_tokens_action)
        has_sound = bool(gen_data_clean.x0_tokens_sound)

        # Determine vision slice for this sample
        num_items = gen_data_clean.num_vision_items_per_sample
        if num_items is not None:
            # Multi-item mode: compute flat-list offset
            vis_start = sum(num_items[:start])  # number of all the vision tokens before the start
            vis_end = sum(num_items[:limit])
            subset_x0_vision = gen_data_clean.x0_tokens_vision[vis_start:vis_end]
            subset_raw_vision = (
                gen_data_clean.raw_state_vision[vis_start:vis_end] if gen_data_clean.raw_state_vision else None
            )
            subset_num_items = num_items[start:limit]
        else:
            # Standard single-item mode
            subset_x0_vision = gen_data_clean.x0_tokens_vision[start:limit]
            subset_raw_vision = (
                gen_data_clean.raw_state_vision[start:limit] if gen_data_clean.raw_state_vision else None
            )
            subset_num_items = None
        fps_vision = gen_data_clean.fps_vision[start:limit] if gen_data_clean.fps_vision is not None else None

        if has_action:
            subset_raw_action = (
                gen_data_clean.raw_state_action[start:limit] if gen_data_clean.raw_state_action else None
            )
            x0_tokens_action = gen_data_clean.x0_tokens_action[start:limit]
            fps_action = gen_data_clean.fps_action[start:limit] if gen_data_clean.fps_action is not None else None
            action_domain_id = gen_data_clean.action_domain_id[start:limit] if gen_data_clean.action_domain_id else None
            raw_action_dim = gen_data_clean.raw_action_dim[start:limit] if gen_data_clean.raw_action_dim else None
        else:
            subset_raw_action = None
            x0_tokens_action = None
            fps_action = None
            action_domain_id = None
            raw_action_dim = None

        if has_sound:
            subset_raw_sound = gen_data_clean.raw_state_sound[start:limit] if gen_data_clean.raw_state_sound else None
            x0_tokens_sound = gen_data_clean.x0_tokens_sound[start:limit]
            fps_sound = gen_data_clean.fps_sound[start:limit] if gen_data_clean.fps_sound is not None else None
        else:
            subset_raw_sound = None
            x0_tokens_sound = None
            fps_sound = None

        return GenerationDataClean(
            batch_size=limit - start,
            is_image_batch=gen_data_clean.is_image_batch,
            raw_state_vision=subset_raw_vision,
            raw_state_action=subset_raw_action,
            raw_state_sound=subset_raw_sound,
            x0_tokens_vision=subset_x0_vision,
            x0_tokens_action=x0_tokens_action,
            x0_tokens_sound=x0_tokens_sound,
            fps_vision=fps_vision,
            fps_action=fps_action,
            fps_sound=fps_sound,
            action_domain_id=action_domain_id,
            raw_action_dim=raw_action_dim,
            num_vision_items_per_sample=subset_num_items,
        )

    @torch.no_grad()
    def validation_step(self, data_batch: dict[str, torch.Tensor], iteration: int):
        pass

    @torch.no_grad()
    def forward(self, xt, t):
        pass

    def get_data_and_condition(self, data_batch: dict[str, torch.Tensor], iteration: int = 1) -> GenerationDataClean:
        """
        - Get raw data of different modalities from databatch
        - Tokenize into corresponding latents
        - Load other conditioning information if any (fps, etc.)
        """
        # Detect whether any sample has multiple vision items (e.g. image editing).
        # If so, track the count per sample before all vision items from this batch are flattened into a list.
        is_image_batch = self.is_image_batch(data_batch)
        sample_vision_list = data_batch[self.input_image_key if is_image_batch else self.input_video_key]


        # we should always get this information here during training. If we can read this field
        # from data_batch it means we are in the visualization callback:
        if "num_vision_items_per_sample" not in data_batch:
            # Each element must be a list/tuple of tensors (not a bare tensor) to count
            # as multi-vision.  A bare tensor's len() returns its first dim size (e.g. C=3),
            # which would incorrectly trigger the multi-vision path for regular video batches.
            has_multiple_vision_per_sample = any(
                isinstance(v, (list, tuple)) and len(v) > 1 for v in sample_vision_list
            )
            num_vision_items_per_sample: list[int] | None = (
                [len(v) for v in sample_vision_list] if has_multiple_vision_per_sample else None
            )

            # information is only stored in the GenerationDataClean object which will be discarded
            # outside the training loop. Error will be raised when the data batch is passed to the
            # visualization callbacks.
            data_batch["num_vision_items_per_sample"] = num_vision_items_per_sample

            # if has_multiple_vision_per_sample, this means that the input media is a list of lists of tensors, we need to flatten it to a list of tensors
            if has_multiple_vision_per_sample:
                media_key = self.input_video_key if not is_image_batch else self.input_image_key
                data_batch[media_key] = [item.unsqueeze(0) for sublist in sample_vision_list for item in sublist]
                if data_batch[media_key][0].dtype == torch.float32 and not is_image_batch:
                    data_batch["is_preprocessed"] = (
                        True  # for video batch, is_processed = True means the video data is normalized. However, for the image batch, is_processed = True means the image data is augmented with a temporal dimension.
                    )
        else:
            num_vision_items_per_sample = data_batch["num_vision_items_per_sample"]

        batch_size = (
            len(sample_vision_list) if num_vision_items_per_sample is None else len(num_vision_items_per_sample)
        )

        log_enc_time = False
        timer = None
        if TRAINING:
            import wandb

            log_enc_time = iteration % self.log_enc_time_every_n == 0 and wandb.run
            if log_enc_time:
                timer = Timer(unit="s")
                timer.start()
        # Vision (image/video) raw state and tokenized latent state
        self._normalize_video_databatch_inplace(data_batch)
        self._augment_image_dim_inplace(data_batch)  # converts each image tensor to (1, C, 1, H, W)
        raw_state_vision = data_batch[self.input_image_key if is_image_batch else self.input_video_key]
        x0_tokens_vision = [
            self.encode(raw_state_vision_i).contiguous().float() for raw_state_vision_i in raw_state_vision
        ]

        frame_size = data_batch.get("image_size", None)
        if frame_size is not None:
            x0_tokens_vision = self._remove_padding_from_latent(x0_tokens_vision, frame_size)

        # Action – extract dense action / domain_id without mutating data_batch,
        # so downstream callbacks can still read the original per-sample domain_ids.
        raw_state_action, action_domain_id = self._normalize_action_databatch(data_batch)
        x0_tokens_action = raw_state_action
        raw_action_dim = data_batch.get("raw_action_dim", None)

        # Sound/audio - normalize, encode if present and sound_gen is enabled
        self._normalize_sound_databatch_inplace(data_batch)
        raw_state_sound = data_batch.get("sound", None)
        if raw_state_sound is not None and self.tokenizer_sound_gen is not None:
            x0_tokens_sound = [self.encode_sound(s).contiguous().float() for s in raw_state_sound]
        else:
            x0_tokens_sound = None

        # We pass the conditioning FPS along to the denoising function
        # It will not be used for RoPE FPS modulation unless enabled in the training config
        # Note: conditioning_fps from data is converted to TPS via temporal_compression_factor
        # in VideoRopePosition3DEmb.
        fps_raw = data_batch.get("conditioning_fps", None)
        if isinstance(fps_raw, list):
            fps_raw = torch.stack(fps_raw).flatten()  # list of scalar tensors -> (B,)
        fps_vision = fps_raw.to(**self.tensor_kwargs) if fps_raw is not None else None
        fps_action = fps_raw.to(**self.tensor_kwargs) if fps_raw is not None else None

        # Sound FPS for RoPE alignment (constant, from config)
        if x0_tokens_sound is not None:
            sound_batch_size = len(x0_tokens_sound)
            fps_sound = torch.full(
                (sound_batch_size,),
                self._get_sound_fps_for_rope(),
                dtype=torch.float32,
            ).to(**self.tensor_kwargs)
        else:
            fps_sound = None

        if TRAINING and log_enc_time and timer is not None:
            timer.end()
            elapsed = timer.get_cuda_time()
            h, w = raw_state_vision[0].shape[-2], raw_state_vision[0].shape[-1]
            resolution_label = "unknown"
            for res_name, aspect_ratios in VIDEO_RES_SIZE_INFO.items():
                if (h, w) in aspect_ratios.values():
                    resolution_label = res_name
                    if res_name == "704":
                        # 720 shares some aspect ratios with 704 (e.g., 1:1 at 960x960); prefer 720.
                        if (h, w) in VIDEO_RES_SIZE_INFO.get("720", {}).values():
                            resolution_label = "720"
                    break
            wandb.log(
                {
                    f"timer/encoding_{resolution_label}p": elapsed,
                    "timer/encoding": elapsed,
                },
                step=iteration,
            )
        return GenerationDataClean(
            batch_size=batch_size,
            is_image_batch=is_image_batch,
            raw_state_vision=raw_state_vision,
            raw_state_action=raw_state_action,
            raw_state_sound=raw_state_sound,
            x0_tokens_vision=x0_tokens_vision,
            x0_tokens_action=x0_tokens_action,
            x0_tokens_sound=x0_tokens_sound,
            fps_vision=fps_vision,
            fps_action=fps_action,
            fps_sound=fps_sound,
            action_domain_id=action_domain_id,
            num_vision_items_per_sample=num_vision_items_per_sample,
            raw_action_dim=raw_action_dim,
        )

    def _normalize_video_databatch_inplace(
        self, data_batch: dict[str, torch.Tensor], input_key: str | None = None
    ) -> None:
        """
        Normalizes video data in-place on a CUDA device to reduce data loading overhead.

        This function modifies the video data tensor within the provided data_batch dictionary
        in-place, scaling the uint8 data from the range [0, 255] to the normalized range [-1, 1].

        Args:
            data_batch (dict[str, Tensor]): A dictionary containing the video data under a specific key.
                This tensor is expected to be on a CUDA device and have dtype of torch.uint8.
            input_key (str | None): The key for the video tensor in the data_batch. Defaults to
                `self.input_video_key` if not provided.

        Side Effects:
            Modifies the tensor at `input_key` within `data_batch` in-place.

        Note:
            This operation is performed directly on the CUDA device to avoid the overhead associated
            with moving data to/from the GPU. Ensure that the tensor is already on the appropriate device
            and has the correct dtype (torch.uint8) to avoid unexpected behaviors.
        """
        IS_PREPROCESSED_KEY = "is_preprocessed"
        input_key = self.input_video_key if input_key is None else input_key
        # only handle video batch
        if input_key in data_batch:
            if IS_PREPROCESSED_KEY in data_batch and data_batch[IS_PREPROCESSED_KEY] is True:
                for i in range(len(data_batch[input_key])):
                    assert torch.is_floating_point(data_batch[input_key][i]), "Video data is not in float format."
                    assert torch.all((data_batch[input_key][i] >= -1.0001) & (data_batch[input_key][i] <= 1.0001)), (
                        f"Video data is not in the range [-1, 1]. get data range "
                        f"[{data_batch[input_key][i].min()}, {data_batch[input_key][i].max()}]"
                    )
            else:
                for i in range(len(data_batch[input_key])):
                    item = data_batch[input_key][i]
                    if isinstance(item, torch.Tensor):
                        item = [item]
                    assert item[0].dtype == torch.uint8, "Video data is not in uint8 format."
                    data_batch[input_key][i] = torch.stack(item).to(**self.tensor_kwargs) / 127.5 - 1.0
                data_batch[IS_PREPROCESSED_KEY] = True

    def _normalize_action_databatch(
        self, data_batch: dict[str, torch.Tensor]
    ) -> tuple[list[torch.Tensor] | None, list[torch.Tensor] | None]:
        """Extract dense action and domain_id lists from the data batch.

        The joint dataloader produces action and domain_id data as
        ``[[tensor], [None], [tensor], ...]`` (each sample wrapped in a
        single-element list).  This method unwraps inner lists and filters
        out ``None`` entries to produce dense lists suitable for the model,
        **without mutating** ``data_batch``.

        Returns:
            (dense_action, dense_domain_id): Each is a list of device tensors
            containing only non-None entries, or ``None`` if all entries are
            ``None`` / the key is absent.
        """
        dense_action = unwrap_and_densify(data_batch.get("action", None), self.tensor_kwargs)
        dense_domain_id = unwrap_and_densify(
            data_batch.get("domain_id", None), {"device": self.tensor_kwargs["device"]}
        )
        return dense_action, dense_domain_id

    def _normalize_sound_databatch_inplace(self, data_batch: dict[str, torch.Tensor]) -> None:
        """Flatten and densify nested sound lists in-place.

        The joint dataloader produces sound data as
        ``[[tensor], [None], [tensor], ...]`` (each sample wrapped in a single-element
        list).  This method:

        1. Unwraps inner lists: ``[[t], [None], [t]]`` -> ``[t, None, t]``
        2. Clears ``sequence_plan.has_sound`` for samples whose sound is ``None``
           (kept aligned by ``custom_collate_fn`` preserving ``None`` placeholders).
        3. Filters out None entries: ``[t, None, t]`` -> ``[t, t]``
        4. Moves tensors to the model device.
        5. Sets ``data_batch["sound"]`` to ``None`` if no valid sound data remains.

        Alignment invariant: ``custom_collate_fn`` keeps the ``"sound"`` key
        as a list with ``None`` placeholders for samples that lack audio (e.g.
        audio-extraction failures), so the unwrapped ``raw_state_sound`` is
        1:1 with ``sequence_plan``.  ``SoundSequencePlanBuilder`` already sets
        each plan's ``has_sound`` according to that sample's actual sound
        presence, so clearing flags for ``None`` slots here is just defensive.
        """
        raw_state_sound = data_batch.get("sound", None)
        sequence_plans = data_batch.get("sequence_plan", None)
        sound_enabled = self.tokenizer_sound_gen is not None

        def _disable_sound_on_plans() -> None:
            if isinstance(sequence_plans, list):
                for plan in sequence_plans:
                    if hasattr(plan, "has_sound"):
                        plan.has_sound = False
                        plan.condition_frame_indexes_sound = []

        if not isinstance(raw_state_sound, list) or len(raw_state_sound) == 0:
            # No sound entries at all (image-only batches, or every sample
            # came from a non-audio stream).  Defensively clear has_sound on
            # any plan that somehow has it set so packing does not look up
            # missing tensors.
            _disable_sound_on_plans()
            data_batch["sound"] = None
            return

        # Unwrap single-element inner lists produced by IterativeJointDataLoader
        if isinstance(raw_state_sound[0], list):
            raw_state_sound = [item[0] if isinstance(item, list) else item for item in raw_state_sound]

        if not sound_enabled:
            # Model is not configured for sound generation: drop tensors and
            # clear any has_sound flags so packing skips the sound path.
            _disable_sound_on_plans()
            data_batch["sound"] = None
            return

        if isinstance(sequence_plans, list):
            if len(sequence_plans) == len(raw_state_sound):
                # Expected path: 1:1 alignment between plans and per-sample
                # sound slots.  Clear has_sound where the per-sample tensor
                # is None so sequence_packing's idx_sound counter stays in
                # sync with the filtered dense list.
                for plan, sound in zip(sequence_plans, raw_state_sound, strict=True):
                    if hasattr(plan, "has_sound") and sound is None:
                        plan.has_sound = False
                        plan.condition_frame_indexes_sound = []
            else:
                # Length mismatch can only happen if some upstream code path
                # (e.g. a stale collate that drops "sound" when any sample is
                # None) leaves the dense list shorter than the plans.  Without
                # 1:1 alignment we cannot safely associate tensors with plans,
                # so we conservatively disable sound for the whole batch.
                # This trades a small amount of training signal for guaranteed
                # correctness — better than silently feeding sound from one
                # sample into another sample's plan.
                log.warning(
                    f"Sound/plan length mismatch ({len(sequence_plans)} plans vs "
                    f"{len(raw_state_sound)} sound entries). Disabling sound for "
                    "this batch.  Check that custom_collate_fn preserves the "
                    "'sound' key with None placeholders."
                )
                _disable_sound_on_plans()
                data_batch["sound"] = None
                return

        # Filter out None entries (samples without audio) and move to device.
        # After the alignment step above, the remaining dense list has the
        # same cardinality as plans with has_sound=True.
        raw_state_sound = [
            s.to(self.tensor_kwargs["device"]) for s in raw_state_sound if s is not None
        ]  # list of [C,T_audio]

        if len(raw_state_sound) == 0:
            _disable_sound_on_plans()
            data_batch["sound"] = None
        else:
            data_batch["sound"] = raw_state_sound

    def _augment_image_dim_inplace(self, data_batch: dict[str, torch.Tensor], input_key: str = None) -> None:
        """
        Augments image tensors by adding a temporal dimension (B, C, H, W) -> (B, C, 1, H, W).

        Args:
            data_batch (dict[str, Tensor]): A dictionary containing the image data.
            input_key (str | None): The key for the image tensor. Defaults to `self.input_image_key`.

        Side Effects:
            Modifies the tensor at `input_key` within `data_batch` in-place.
        """
        IS_PREPROCESSED_KEY = "is_preprocessed"

        input_key = self.input_image_key if input_key is None else input_key
        if input_key in data_batch:
            # Check if the data has already been augmented and avoid re-augmenting
            if IS_PREPROCESSED_KEY in data_batch and data_batch[IS_PREPROCESSED_KEY] is True:
                for i in range(len(data_batch[input_key])):
                    assert data_batch[input_key][i].shape[2] == 1, (
                        f"Image data is claimed be augmented while its shape is {data_batch[input_key][i].shape} for sample {i}"
                    )
                return
            else:
                new_image_tensor_list = []
                for i in range(len(data_batch[input_key])):
                    for img_tensor in data_batch[input_key][i]:
                        img_tensor = rearrange(img_tensor, "c h w -> 1 c 1 h w").contiguous()
                        if img_tensor.dtype == torch.uint8:
                            img_tensor = img_tensor.to(**self.tensor_kwargs) / 127.5 - 1.0
                        new_image_tensor_list.append(img_tensor)
                data_batch[input_key] = new_image_tensor_list
                data_batch[IS_PREPROCESSED_KEY] = True

    # ------------------ Checkpointing ------------------

    def state_dict(self, prefix: str = "", **kwargs) -> Dict[str, Any]:
        final_state_dict = self.net.state_dict(prefix=prefix + "net.", **kwargs)
        if self.config.ema.enabled:
            ema_state_dict = self.net_ema.state_dict(prefix=prefix + "net_ema.", **kwargs)
            final_state_dict.update(ema_state_dict)
        return final_state_dict

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False):
        """
        Loads a state dictionary into the model and optionally its EMA counterpart.

        Parameters:
            state_dict (Mapping[str, Any]): A dictionary containing separate state
                dictionaries for the model and potentially for an EMA version of the model
                under the keys 'net' and 'net_ema', respectively.
            strict (bool, optional): If True, the method will enforce that the keys in
                the state dict match exactly those in the model and EMA model (if applicable).
                Defaults to True.
            assign (bool, optional): If True and in strict mode, will assign the state dictionary
                directly rather than matching keys one-by-one. This is typically used when loading
                parts of state dicts or using customized loading procedures. Defaults to False.
        """
        if not strict:
            raise ValueError("Strict mode is required for OmniMoTModel load_state_dict")
        if assign:
            raise ValueError("Assign mode is not supported for OmniMoTModel load_state_dict")

        _reg_state_dict = collections.OrderedDict()
        _ema_state_dict = collections.OrderedDict()
        for k, v in state_dict.items():
            if k.startswith("net."):
                _reg_state_dict[k.replace("net.", "")] = v
            elif k.startswith("net_ema."):
                _ema_state_dict[k.replace("net_ema.", "")] = v

        state_dict = _reg_state_dict

        reg_results: _IncompatibleKeys = self.net.load_state_dict(_reg_state_dict, strict=True, assign=False)
        missing_keys = reg_results.missing_keys
        unexpected_keys = reg_results.unexpected_keys

        if self.config.ema.enabled:
            ema_results: _IncompatibleKeys = self.net_ema.load_state_dict(_ema_state_dict, strict=True, assign=False)
            missing_keys += ema_results.missing_keys
            unexpected_keys += ema_results.unexpected_keys
        else:
            assert len(_ema_state_dict) == 0, f"EMA is disabled but EMA state dict is not empty: {len(_ema_state_dict)}"

        return _IncompatibleKeys(missing_keys=missing_keys, unexpected_keys=unexpected_keys)

    # ------------------ public methods ------------------

    def ema_beta(self, iteration: int) -> float:
        """
        Calculate the beta value for EMA update.
        weights = weights * beta + (1 - beta) * new_weights

        Args:
            iteration (int): Current iteration number.

        Returns:
            float: The calculated beta value.
        """
        iteration = iteration + self.config.ema.iteration_shift
        if iteration < 1:
            return 0.0
        return (1 - 1 / (iteration + 1)) ** (self.ema_exp_coefficient + 1)

    def model_param_stats(self) -> Dict[str, int]:
        return {"total_learnable_param_num": self._param_count}

    def is_image_batch(self, data_batch: dict[str, torch.Tensor]) -> bool:
        """Check if the data_batch contains images (vs. videos).

        We handle two types of data_batch: one from a joint_dataloader where "dataset_name" can
        differentiate image_batch and video_batch, another from a single dataloader which we
        assume as video_data by default.
        """
        is_image = self.input_image_key in data_batch
        is_video = self.input_video_key in data_batch
        assert is_image != is_video, (
            "Only one of the input_image_key or input_video_key should be present in the data_batch."
        )
        return is_image

    def denoise(
        self,
        net: torch.nn.Module | None = None,
        data_batch_packed: PackedSequence | None = None,
        fps_vision: torch.Tensor | None = None,
        fps_action: torch.Tensor | None = None,
        fps_sound: torch.Tensor | None = None,
        memory: MemoryState | None = None,
    ) -> dict:
        """
        Runs the MoT network on a packed multi-modal sequence to predict velocity (v) targets.

        Args:
            data_batch_packed: PackedSequence from `pack_input_sequence(...)`.
            fps_vision: Optional FPS tensor used for RoPE FPS modulation (if enabled in config).
            fps_action: Optional FPS tensor used for action RoPE FPS modulation (if enabled in config).
            fps_sound: Optional FPS tensor for sound RoPE modulation (e.g., sound_latent_fps=25).
            memory: Optional pre-built MemoryState for autoregressive generation
                or KV-cache training.

        Returns:
            dict containing:
                - "preds_vision": list[Tensor[C,T,H,W]], one per sample.
                - "preds_action": Velocity prediction for action modality (if action_gen enabled).
                - "preds_sound": Velocity prediction for sound modality (if sound_gen enabled).
                - "lbl_metadata_und": Load balancing metadata for understanding pathway (if present).
                - "lbl_metadata_gen": Load balancing metadata for generation pathway (if present).
        """
        net = net or self.net
        out_net = net(
            packed_seq=data_batch_packed,
            fps_vision=fps_vision,
            fps_action=fps_action,
            fps_sound=fps_sound,
            memory=memory,
        )
        output_dict = dict()
        output_dict["preds_vision"] = out_net["preds_vision"]
        if self.config.action_gen and "preds_action" in out_net:
            output_dict["preds_action"] = out_net["preds_action"]
        if self.config.sound_gen and "preds_sound" in out_net:
            output_dict["preds_sound"] = out_net["preds_sound"]
        for key, value in out_net.items():
            if "lbl_metadata_" in key:
                output_dict[key] = value

        return output_dict

    @torch.no_grad()
    def encode(self, state: torch.Tensor) -> torch.Tensor:
        return self.tokenizer_vision_gen.encode(state)

    @torch.no_grad()
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.tokenizer_vision_gen.decode(latent)

    @torch.no_grad()
    def encode_sound(self, waveform: torch.Tensor) -> torch.Tensor:
        """Encode audio waveform into latent tokens.

        Args:
            waveform: Audio tensor of shape (C, N). A batch dim is added/removed
                      internally since AVAE expects (B, C, N).
                      Mono audio is duplicated to stereo if the tokenizer expects 2 channels.
        """
        assert self.tokenizer_sound_gen is not None, "Sound tokenizer not initialized"
        # Ensure correct number of channels (AVAE typically expects stereo)
        expected_channels = self.tokenizer_sound_gen.audio_channels
        if waveform.shape[0] == 1 and expected_channels == 2:
            waveform = waveform.repeat(2, 1)  # mono → stereo
        elif waveform.shape[0] > expected_channels:
            waveform = waveform[:expected_channels]
        # AVAE expects (B, C, N)
        latent = self.tokenizer_sound_gen.encode(waveform.unsqueeze(0))  # [1,sound_channels,T_sound]
        return latent.squeeze(0)  # [sound_channels,T_sound]

    @torch.no_grad()
    def decode_sound(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode sound latent tokens back to waveform.

        Args:
            latent: Sound latent tensor of shape (C, T). A batch dim is added/removed
                    internally since AVAE expects (B, C, T).
        """
        assert self.tokenizer_sound_gen is not None, "Sound tokenizer not initialized"
        # AVAE expects (B, C, T)
        waveform = self.tokenizer_sound_gen.decode(latent.unsqueeze(0))  # [1,audio_channels,N_samples]
        return waveform.squeeze(0)  # [audio_channels,N_samples]

    def _get_sound_fps_for_rope(self) -> float:
        """Compute the sound FPS to pass to RoPE for temporal alignment with video.

        Returns the sound tokenizer's latent rate (e.g., 25 Hz for 48kHz/1920 hop).
        This is passed as input_fps to the sound RoPE's generate_embeddings(), where
        the FPS modulation formula aligns sound indices with video indices.
        """
        return float(self.config.sound_latent_fps)

    def get_video_height_width(self) -> Tuple[int, int]:
        return VIDEO_RES_SIZE_INFO[self.config.resolution]["9,16"]

    def get_video_latent_height_width(self) -> Tuple[int, int]:
        height, width = VIDEO_RES_SIZE_INFO[self.config.resolution]["9,16"]
        return (
            height // self.tokenizer_vision_gen.spatial_compression_factor,
            width // self.tokenizer_vision_gen.spatial_compression_factor,
        )

    def get_num_video_latent_frames(self) -> int:
        return self.config.state_t

    @contextmanager
    def ema_scope(self, context=None, is_cpu=False):
        if self.config.ema.enabled:
            # https://github.com/pytorch/pytorch/issues/144289
            for module in self.net.modules():
                if isinstance(module, FSDPModule):
                    module.reshard()
            self.net_ema_worker.cache(self.net.parameters(), is_cpu=is_cpu)
            self.net_ema_worker.copy_to(src_model=self.net_ema, tgt_model=self.net)
            if context is not None:
                log.info(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.config.ema.enabled:
                for module in self.net.modules():
                    if isinstance(module, FSDPModule):
                        module.reshard()
                self.net_ema_worker.restore(self.net.parameters())
                if context is not None:
                    log.info(f"{context}: Restored training weights")

    def add_lora(
        self,
        network: torch.nn.Module,
        lora_rank: int,
        lora_alpha: int,
        lora_target_modules: str,
    ) -> torch.nn.Module:
        """Pre-FSDP LoRA injection — see :func:`inject_lora_pre_fsdp` for details."""
        from cosmos.utils.vfm.lora import inject_lora_pre_fsdp

        self.lora_alpha = lora_alpha
        return inject_lora_pre_fsdp(
            network,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_target_modules=lora_target_modules,
        )

    def _init_lora_weights_post_materialization(self, network: torch.nn.Module) -> None:
        """Post-materialization LoRA init — see :func:`init_lora_weights_post_materialization`."""
        from cosmos.utils.vfm.lora import init_lora_weights_post_materialization

        init_lora_weights_post_materialization(network)


def _broadcast_seed(seed: list[int], group: dist.ProcessGroup, rank: int) -> list[int]:
    global_src_rank = torch.distributed.get_global_rank(group, 0)

    if rank == 0:
        seed_tensor = torch.tensor(seed, dtype=torch.int64, device=DEVICE)  # [len(seed)]
    else:
        seed_tensor = torch.zeros(len(seed), dtype=torch.int64, device=DEVICE)  # [len(seed)]

    torch.distributed.broadcast(seed_tensor, src=global_src_rank, group=group)
    return seed_tensor.tolist()
