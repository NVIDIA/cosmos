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

import gc
import getpass
import math
import os

import pytest
import torch
import torch.distributed as dist
from transformers import AutoTokenizer

from cosmos.utils.lazy_config import instantiate as lazy_instantiate
from cosmos.utils import log
from cosmos.utils.easy_io import easy_io
from cosmos.model.vfm.algorithm.loss.load_balancing import compute_load_balancing_loss
from cosmos.callbacks.expert_heatmap import compute_expert_heatmap
from configs.base.defaults.cluster import GCPIADGB200Config as CLUSTER_CONFIG
from configs.base.defaults.model_config import ParallelismConfig
from configs.base.defaults.vlm import (
    Qwen3MoT_LLM_0p6b_GCP_Config,
    Qwen3VLMoT_VLM_2b_Instruct_GCP_Config,
    Qwen3VLMoT_VLM_4b_Instruct_GCP_Config,
    Qwen3VLMoT_VLM_8b_Instruct_GCP_Config,
    Qwen3VLMoT_VLM_30b_a3b_Instruct_GCP_Config,
    Qwen3VLMoT_VLM_32b_Instruct_GCP_Config,
)
from cosmos.data.vfm.sequence_packing import (
    PackedSequence,
    add_special_tokens,
    build_sequence_plans_from_data_batch,
    pack_input_sequence,
)
from cosmos.model.vfm.mot.context_parallel_utils import context_parallel_broadcast
from cosmos.model.vfm.mot.cosmos3_vfm_network import Cosmos3VFMNetwork, Cosmos3VFMNetworkConfig
from cosmos.model.vfm.mot.parallelize_vfm_network import parallelize_vfm_network
from cosmos.model.vfm.utils.data_and_condition import GenerationDataClean
from cosmos.model.vfm.utils.safetensors_loader import load_language_model as load_language_model_safetensors
from cosmos.utils.vfm.parallelism import ParallelDims

"""
Unified test file for Qwen3-VL network configurations supporting both Dense and MoE models.
Supports 2B, 4B, 8B, 32B Dense models and 30B A3B MoE models.

We support 3 tests in this script:
1. Embedding match test: We compare the text embeddings of our model with the embeddings of the HF model.
2. Weight match test: We compare the weights of the first layer of our model with the weights of the HF model.
3. Cosmos3 network test: We test the shapes of the output from our model with our expected shapes.

Usage Examples:
    # Dense LLM models
    # We have 0.6B dense LLM model support
    export HF_HOME=/nfs/dir/dir_cosmos_base/users/gekurian/hf_cache
    export QWEN3_VL_TEST_MODELS=qwen3_vl_0p6b_llm
    pytest cosmos/model/vfm/mot/cosmos3_vfm_qwen3_vl_network_test.py --all -s -v

    # Dense models
    # 2B, 4B, 8B don't need any parallelization, we run 1, 2, 3 tests from above for these models
    export HF_HOME=/nfs/dir/dir_cosmos_base/users/gekurian/hf_cache
    export QWEN3_VL_TEST_MODELS=qwen3_vl_2b_instruct,qwen3_vl_4b_instruct,qwen3_vl_8b_instruct
    pytest cosmos/model/vfm/mot/cosmos3_vfm_qwen3_vl_network_test.py --all -s -v

    # 32B dense needs distributed test, we run 1, 3 tests from above for this model
    export HF_HOME=/nfs/dir/dir_cosmos_base/users/gekurian/hf_cache
    export QWEN3_VL_TEST_MODELS=qwen3_vl_32b_instruct
    torchrun --standalone --nproc_per_node=4 -m pytest cosmos/model/vfm/mot/cosmos3_vfm_qwen3_vl_network_test.py::test_unified_llm_outputs_with_hf --all -s -v
    torchrun --standalone --nproc_per_node=4 -m pytest cosmos/model/vfm/mot/cosmos3_vfm_qwen3_vl_network_test.py::test_unified_cosmos3_network --all -s -v

    # MoE models (distributed)
    # 30B A3B needs distributed test, we run 1, 3 tests from above for this model
    export HF_HOME=/nfs/dir/dir_cosmos_base/users/gekurian/hf_cache
    export QWEN3_VL_TEST_MODELS=qwen3_vl_30b_a3b_instruct
    torchrun --standalone --nproc_per_node=4 -m pytest cosmos/model/vfm/mot/cosmos3_vfm_qwen3_vl_network_test.py::test_unified_llm_outputs_with_hf --all -s -v
    torchrun --standalone --nproc_per_node=4 -m pytest cosmos/model/vfm/mot/cosmos3_vfm_qwen3_vl_network_test.py::test_unified_cosmos3_network --all -s -v

Environment Variables:
    QWEN3_VL_TEST_MODELS: Comma-separated list of models to test
"""

# Unified model configurations
MODEL_CONFIGS = {
    # Dense models
    "0p6b": Qwen3MoT_LLM_0p6b_GCP_Config,
    "2b": Qwen3VLMoT_VLM_2b_Instruct_GCP_Config,
    "4b": Qwen3VLMoT_VLM_4b_Instruct_GCP_Config,
    "8b": Qwen3VLMoT_VLM_8b_Instruct_GCP_Config,
    "32b": Qwen3VLMoT_VLM_32b_Instruct_GCP_Config,
    # MoE models
    "30b_a3b": Qwen3VLMoT_VLM_30b_a3b_Instruct_GCP_Config,
}

# Unified test configurations
TEST_CONFIGS = {
    # Dense models (single GPU)
    "qwen3_vl_0p6b_llm": {
        "config": MODEL_CONFIGS["0p6b"],
        "parallelism": ParallelismConfig(
            use_activation_checkpointing=False,
            use_torch_compile=False,
            data_parallel_shard_degree=1,
            context_parallel_shard_degree=1,
        ),
        "model_name": "Qwen/Qwen3-0.6B",
        "model_type": "dense_llm",
        "max_tokens": 4096,
        "latent_shapes": [(16, 1, 128, 128)],
        "backend_credentials": "gcp_checkpoint.secret",
        "requires_dcp": True,
        "input_samples": 3,
        "test_latent_size": (256, 256),
        "network_latent_size": (512, 512),
        "network_samples": 3,
        "network_max_tokens": 6144,
        "network_latent_shapes": [(16, 1, 64, 64)] * 3,
        "requires_gc": True,
        "requires_distributed": False,
        "output_tolerance": 0.04,
        "joint_attn_implementation": "two_way",
    },
    "qwen3_vl_2b_instruct": {
        "config": MODEL_CONFIGS["2b"],
        "parallelism": ParallelismConfig(
            use_activation_checkpointing=False,
            use_torch_compile=False,
            data_parallel_shard_degree=1,
            context_parallel_shard_degree=1,
        ),
        "model_name": "Qwen/Qwen3-VL-2B-Instruct",
        "model_type": "dense",
        "max_tokens": 4096,
        "latent_shapes": [(16, 1, 128, 128)],
        "backend_credentials": "gcp_checkpoint.secret",
        "requires_dcp": True,
        "input_samples": 3,
        "test_latent_size": (256, 256),
        "network_latent_size": (512, 512),
        "network_samples": 3,
        "network_max_tokens": 6144,
        "network_latent_shapes": [(16, 1, 64, 64)] * 3,
        "requires_gc": True,
        "requires_distributed": False,
        "output_tolerance": 0.04,
        "joint_attn_implementation": "two_way",
    },
    "qwen3_vl_4b_instruct": {
        "config": MODEL_CONFIGS["4b"],
        "parallelism": ParallelismConfig(
            use_activation_checkpointing=False,
            use_torch_compile=False,
            data_parallel_shard_degree=1,
            context_parallel_shard_degree=1,
        ),
        "model_name": "Qwen/Qwen3-VL-4B-Instruct",
        "model_type": "dense",
        "max_tokens": 4096,
        "latent_shapes": [(16, 1, 64, 64)],
        "backend_credentials": "gcp_checkpoint.secret",
        "requires_dcp": True,
        "input_samples": 3,
        "test_latent_size": (256, 256),
        "network_latent_size": (512, 512),
        "network_samples": 3,
        "network_max_tokens": 6144,
        "network_latent_shapes": [(16, 1, 64, 64)] * 3,
        "requires_gc": True,
        "requires_distributed": False,
        "output_tolerance": 0.04,
        "joint_attn_implementation": "two_way",
    },
    "qwen3_vl_8b_instruct": {
        "config": MODEL_CONFIGS["8b"],
        "parallelism": ParallelismConfig(
            use_activation_checkpointing=False,
            use_torch_compile=False,
            data_parallel_shard_degree=1,
            context_parallel_shard_degree=1,
        ),
        "model_name": "Qwen/Qwen3-VL-8B-Instruct",
        "model_type": "dense",
        "max_tokens": 3072,
        "latent_shapes": [(16, 1, 32, 32)],
        "backend_credentials": "gcp_checkpoint.secret",
        "requires_dcp": True,
        "input_samples": 3,
        "test_latent_size": (256, 256),
        "network_latent_size": (256, 256),
        "network_samples": 2,
        "network_max_tokens": 3072,
        "network_latent_shapes": [(16, 1, 32, 32)] * 2,
        "requires_gc": True,
        "requires_distributed": False,
        "output_tolerance": 0.04,
        "joint_attn_implementation": "two_way",
    },
    "qwen3_vl_32b_instruct": {
        "config": MODEL_CONFIGS["32b"],
        "parallelism": ParallelismConfig(
            use_activation_checkpointing=True,
            use_torch_compile=True,
            data_parallel_shard_degree=4,
            context_parallel_shard_degree=2,
        ),
        "model_name": "Qwen/Qwen3-VL-32B-Instruct",
        "model_type": "dense",
        "max_tokens": 3072,
        "latent_shapes": [(16, 1, 32, 32)],
        "backend_credentials": "gcp_checkpoint.secret",
        "requires_dcp": True,
        "input_samples": 3,
        "test_latent_size": (256, 256),
        "network_latent_size": (256, 256),
        "network_samples": 2,
        "network_max_tokens": 3072,
        "network_latent_shapes": [(16, 1, 32, 32)] * 2,
        "requires_gc": True,
        "requires_distributed": True,
        "output_tolerance": 0.05,
        "joint_attn_implementation": "two_way",
    },
    # MoE models (distributed)
    "qwen3_vl_30b_a3b_instruct": {
        "config": MODEL_CONFIGS["30b_a3b"],
        "parallelism": ParallelismConfig(
            use_activation_checkpointing=True,
            use_torch_compile=True,
            data_parallel_shard_degree=4,
            context_parallel_shard_degree=2,
        ),
        "model_name": "Qwen/Qwen3-VL-30B-A3B-Instruct",
        "model_type": "moe",
        "max_tokens": 4096,
        "latent_shapes": [(16, 1, 128, 128)],
        "backend_credentials": "gcp_checkpoint.secret",
        "requires_dcp": True,
        "input_samples": 3,
        "test_latent_size": (256, 256),
        "network_latent_size": (512, 512),
        "network_samples": 3,
        "network_max_tokens": 6144,
        "network_latent_shapes": [(16, 1, 64, 64)] * 3,
        "requires_gc": True,
        "requires_distributed": True,
        "output_tolerance": 0.12,  # Higher tolerance for MoE
        "joint_attn_implementation": "two_way",
    },
}


def get_models_to_test():
    """Get the list of models to test from environment variable or return all."""
    # Check environment variable
    models_arg = os.environ.get("QWEN3_VL_TEST_MODELS")

    # If not specified, test all models
    if not models_arg:
        return list(TEST_CONFIGS.keys())

    # Parse comma-separated models
    requested_models = [model.strip() for model in models_arg.split(",")]

    # Validate requested models
    models_to_test = []
    for model in requested_models:
        if model in TEST_CONFIGS:
            models_to_test.append(model)
        else:
            available_models = ", ".join(TEST_CONFIGS.keys())
            raise ValueError(f"Invalid model '{model}'. Available models: {available_models}")

    return models_to_test


# Get models to test at module level
MODELS_TO_TEST = get_models_to_test()


def memory_cleanup(config_name: str) -> None:
    """Memory cleanup based on model size."""
    config = TEST_CONFIGS[config_name]
    torch.cuda.empty_cache()
    if config["requires_gc"]:
        gc.collect()
        torch.cuda.empty_cache()


def init_parallel(
    data_parallel_shard_degree: int,
    context_parallel_shard_degree: int,
) -> ParallelDims | None:
    """Initialize distributed device mesh for MoE models. Returns None for dense models."""
    # Use deterministic algorithms for reproducibility.
    # For example, torch.scatter_add operation (used in MoE models) is not
    # deterministic by default unless this flag is set.
    torch.use_deterministic_algorithms(True, warn_only=True)

    if not dist.is_initialized():
        try:
            dist.init_process_group("nccl")
        except RuntimeError:
            # Already initialized or not in distributed environment
            pass

    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
        print(f"Rank {rank}/{world_size} using device {device}")

        parallel_dims = ParallelDims(
            enable_inference_mode=False,
            world_size=world_size,
            dp_shard=data_parallel_shard_degree,
            cp=context_parallel_shard_degree,
        )
        parallel_dims.build_meshes(device_type="cuda")
        return parallel_dims
    else:
        return None


def load_model_weights(model, config_name: str, parallel_dims: ParallelDims | None) -> None:
    """Load model weights using automatic key remapping."""
    config = TEST_CONFIGS[config_name]
    vlm_config = config["config"]

    load_language_model_safetensors(
        model=model.language_model,
        checkpoint_path=vlm_config.pretrained_weights.backbone_path,
        credential_path=f"credentials/{config['backend_credentials']}",
        parallel_dims=parallel_dims,
    )


def build_model_unified(config_name: str, parallel_dims: ParallelDims | None = None) -> tuple:
    """Build model with unified configuration supporting both dense and MoE."""
    config = TEST_CONFIGS[config_name]
    vlm_config = config["config"]

    easy_io.set_s3_backend(
        backend_args={
            "backend": "s3",
            "s3_credential_path": CLUSTER_CONFIG.object_store_credential_pretrained,
        }
    )

    # Create models on meta device for memory efficiency
    with torch.device("meta"):
        vlm = lazy_instantiate(vlm_config.model_instance)
        cosmos3_network_config = Cosmos3VFMNetworkConfig(
            vlm_config=vlm.config,
            latent_downsample_factor=8,
            latent_channel_size=16,
            latent_patch_size=2,
            max_latent_h=256,
            max_latent_w=256,
            max_latent_t=1,
            vision_gen=True,
            joint_attn_implementation=config["joint_attn_implementation"],
        )
        cosmos3_network_config._attn_implementation_internal = "eager"
        cosmos3_network_config.dtype = torch.bfloat16
        model_cosmos3 = Cosmos3VFMNetwork(
            language_model=vlm,
            config=cosmos3_network_config,
        )

    # Model optimization: parallelization, compilation, and activation checkpointing.
    model_cosmos3 = parallelize_vfm_network(
        model_cosmos3,
        parallel_dims=parallel_dims,
        config=config["parallelism"],
    )

    # Transfer from meta to cuda and initialize.
    assert isinstance(vlm, torch.nn.Module)
    model_cosmos3 = model_cosmos3.to(torch.bfloat16)
    model_cosmos3.to_empty(device="cuda")
    model_cosmos3.init_weights(buffer_device="cuda")
    model_cosmos3.train()

    # Load weights and initialize MoE
    load_model_weights(model_cosmos3, config_name, parallel_dims)
    model_cosmos3.language_model.init_moe()

    # Setup tokenizer
    _proc = lazy_instantiate(vlm_config.tokenizer)
    tokenizer_cosmos3 = _proc.tokenizer
    tokenizer_cosmos3, _ = add_special_tokens(tokenizer_cosmos3)

    # Add special tokens
    special_tokens = {
        "eos_token_id": tokenizer_cosmos3.eos_token_id,
        "start_of_generation": tokenizer_cosmos3.convert_tokens_to_ids("<|vision_start|>"),
        "end_of_generation": tokenizer_cosmos3.convert_tokens_to_ids("<|vision_end|>"),
    }

    return model_cosmos3, tokenizer_cosmos3, special_tokens


def create_multiple_test_inputs_for_hf_comparison(config_name: str, tokenizer, special_tokens) -> tuple:
    """Create multiple test inputs for HF comparison test based on config input_samples."""
    config = TEST_CONFIGS[config_name]
    num_messages = config["input_samples"]

    # Define message templates
    message_templates = [
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What's the best way to learn Python?"},
        ],
        [
            {"role": "system", "content": "You are a coding expert."},
            {"role": "user", "content": "Explain machine learning in simple terms."},
        ],
        [
            {"role": "system", "content": "You are a creative writer."},
            {"role": "user", "content": "Write a short story about space exploration."},
        ],
        [
            {"role": "system", "content": "You are a science teacher."},
            {"role": "user", "content": "How does photosynthesis work?"},
        ],
        [
            {"role": "system", "content": "You are a travel guide."},
            {"role": "user", "content": "Recommend places to visit in Japan."},
        ],
    ]

    # Select the number of messages based on config
    selected_messages = message_templates[:num_messages]

    # Tokenize all selected messages
    all_model_inputs = []
    for messages in selected_messages:
        text_tokens = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text_tokens])
        all_model_inputs.append(model_inputs)

    # Create images for each message
    input_images = [torch.randn(3, 1, *config["test_latent_size"]) for _ in range(num_messages)]  # [3,1,H,W] each
    input_timesteps = [0.0] * num_messages

    return selected_messages, all_model_inputs, input_images, input_timesteps


def create_packed_sequence(
    input_text_tokens: list[list[int]],
    latent_shapes: list[tuple[int, int, int, int]],
    input_timesteps: list[float],
    special_tokens: dict[str, int],
    latent_patch_size: int = 1,
) -> PackedSequence:
    """Create a PackedSequence following the training_step flow.

    This function simulates the data preparation pipeline used in training:
    1. Build sequence plans from data batch
    2. Create tokenized latents (simulating VAE encoder output)
    3. Create GenerationDataClean
    4. Pack input sequence (builds condition masks)
    5. Create GenerationDataNoised and attach tokens

    Args:
        input_text_tokens: List of tokenized text sequences, one per sample.
        latent_shapes: List of (C, T, H, W) shapes for each sample's latent.
        input_timesteps: List of diffusion timesteps for each sample.
        special_tokens: Dictionary of special token IDs.

    Returns:
        PackedSequence containing all packed tensors and metadata.
    """
    num_samples = len(input_text_tokens)

    # Create dummy raw images (before VAE encoding)
    input_images = [
        torch.randn(3, 1, shape[2] * 8, shape[3] * 8).to(torch.bfloat16)  # [3,1,H*8,W*8]
        for shape in latent_shapes
    ]

    # 1. Build sequence plans
    sequence_plans = build_sequence_plans_from_data_batch(
        data_batch={"images": input_images}, input_video_key="video", input_image_key="images"
    )

    # 2. Stack images into a batch tensor (raw state)
    input_images_stacked = torch.stack(input_images, dim=0)  # [B,3,1,H*8,W*8]

    # 3. Create tokenized latents (simulating encoder output)
    x0_tokens_vision = torch.stack(
        [torch.randn(1, *shape).to(torch.bfloat16) for shape in latent_shapes], dim=0
    )  # [B,1,C,T,H,W]

    # 4. Create GenerationDataClean (like get_data_and_condition output)
    gen_data_clean = GenerationDataClean(
        batch_size=num_samples,
        is_image_batch=True,
        raw_state_vision=input_images_stacked,
        x0_tokens_vision=x0_tokens_vision,
        raw_state_action=None,
    )

    timesteps_tensor = torch.tensor(input_timesteps)  # [num_samples]

    # 5. Pack input sequence (builds condition masks)
    packed_sequence = pack_input_sequence(
        sequence_plans=sequence_plans,
        input_text_indexes=input_text_tokens,
        gen_data_clean=gen_data_clean,
        input_timesteps=timesteps_tensor,
        special_tokens=special_tokens,
        latent_patch_size=latent_patch_size,
    )

    # 6. Create GenerationDataNoised (simulating _add_noise_to_input)
    assert packed_sequence.vision is not None
    assert packed_sequence.vision.condition_mask is not None

    return packed_sequence


def compare_single_weight(our_state_dict, hf_state_dict, our_key, hf_key, weight_name):
    """Compare a single weight between two models (Dense models only)."""
    result = {
        "weight_name": weight_name,
        "our_key": our_key,
        "hf_key": hf_key,
        "found_both": False,
        "shapes_match": False,
        "max_diff": float("inf"),
        "mean_diff": float("inf"),
        "identical": False,
    }

    if our_key in our_state_dict and hf_key in hf_state_dict:
        result["found_both"] = True

        our_weight = our_state_dict[our_key]
        hf_weight = hf_state_dict[hf_key]

        result["our_shape"] = list(our_weight.shape)
        result["hf_shape"] = list(hf_weight.shape)
        result["shapes_match"] = our_weight.shape == hf_weight.shape

        if result["shapes_match"]:
            # Compare weights
            atol = 1e-6
            diff = (our_weight - hf_weight).abs()
            result["max_diff"] = diff.max().item()
            result["mean_diff"] = diff.mean().item()
            result["identical"] = torch.allclose(our_weight, hf_weight, atol=atol)
            print(f"Weight name: {weight_name}")
            print(f"Our key: {our_key}")
            print(f"HF key: {hf_key}")
            print(f"Our shape: {result['our_shape']}")
            print(f"HF shape: {result['hf_shape']}")
            print(f"Our weight mean: {our_weight.mean().item()}")
            print(f"HF weight mean: {hf_weight.mean().item()}")
            print(f"Max diff: {result['max_diff']:.2e}")
            print(f"Mean diff: {result['mean_diff']:.2e}")
            print(f"Identical: {result['identical']}")
        else:
            print(f"Shape mismatch for {weight_name}: our={result['our_shape']} vs hf={result['hf_shape']}")
    else:
        if our_key not in our_state_dict:
            print(f"Missing key in our model: {our_key}")
        if hf_key not in hf_state_dict:
            print(f"Missing key in HF model: {hf_key}")

    return result


@pytest.mark.L1
@pytest.mark.parametrize("config_name", MODELS_TO_TEST)
def test_unified_llm_outputs_with_hf(config_name: str):
    """
    Unified test for checking LLM text outputs from our model matches HF model.
    Handles both dense (single GPU) and MoE (distributed) models.
    """
    config = TEST_CONFIGS[config_name]
    is_distributed = config["requires_distributed"]

    # Initialize distributed setup if needed
    parallel_dims = None
    if is_distributed:
        parallel_dims = init_parallel(
            data_parallel_shard_degree=config["parallelism"].data_parallel_shard_degree,
            context_parallel_shard_degree=config["parallelism"].context_parallel_shard_degree,
        )
        rank = dist.get_rank()
    else:
        rank = 0

    log.info(f"\n=== Testing {config_name} ({config['model_type']}) ===")

    #########################################################
    # Step 1: Get outputs from our model, then delete it

    print(f"Loading {config['model_type']} model...")
    model_cosmos3, tokenizer_cosmos3, special_tokens = build_model_unified(config_name, parallel_dims)

    # Create test inputs
    all_messages, all_model_inputs_cosmos3, input_images, input_timesteps = (
        create_multiple_test_inputs_for_hf_comparison(config_name, tokenizer_cosmos3, special_tokens)
    )

    all_text_embeds = []

    # Process each message separately
    for i, (messages, model_inputs_cosmos3) in enumerate(zip(all_messages, all_model_inputs_cosmos3)):
        log.info(f"Processing message {i + 1}/{len(all_messages)}")

        # Create packed sequence following training_step flow
        data_batch = create_packed_sequence(
            input_text_tokens=model_inputs_cosmos3.input_ids,
            latent_shapes=config["latent_shapes"],
            input_timesteps=[input_timesteps[i]],
            special_tokens=special_tokens,
            latent_patch_size=model_cosmos3.config.latent_patch_size,
        )

        # Move to CUDA
        data_batch.to_cuda()

        # Forward pass
        with torch.no_grad():
            output = model_cosmos3(packed_seq=data_batch)

        if config["model_type"] == "moe":
            lbl_und = compute_load_balancing_loss(
                output["lbl_metadata_und"],
                coeff=1.0,
                method="global",
                device_mesh=parallel_dims.dp_mesh,
            )
            lbl_gen = compute_load_balancing_loss(
                output["lbl_metadata_gen"],
                coeff=1.0,
                method="local",
                device_mesh=parallel_dims.dp_mesh,
            )
            log.info(f"Cosmos3 model aux loss: {lbl_und:.2f}, {lbl_gen:.2f}")

        # Extract text embeddings for this message
        num_text_tokens = len(model_inputs_cosmos3.input_ids[0])
        text_embeds = output["last_hidden_state"][0:num_text_tokens].cpu()
        all_text_embeds.append(text_embeds)

        # Add rank-specific logging for distributed models
        log.info(f"Cosmos3 model output norm: {text_embeds.norm()}")

    if config["model_type"] == "moe":
        vlm_config = model_cosmos3.language_model.config
        expert_heatmaps = compute_expert_heatmap(model_cosmos3)
        for tower, heatmap in expert_heatmaps.items():
            assert heatmap.ndim == 2, f"Expected 2 dimensions, got {heatmap.ndim}"
            assert heatmap.shape[0] == vlm_config.num_hidden_layers, (
                f"Expected {vlm_config.num_hidden_layers} layers, got {heatmap.shape[0]}"
            )
            assert heatmap.shape[1] == vlm_config.num_experts, (
                f"Expected {vlm_config.num_experts} experts, got {heatmap.shape[1]}"
            )
            for layer in range(vlm_config.num_hidden_layers):
                heatmap_sum = heatmap[layer].sum().cpu().item()
                torch.testing.assert_close(heatmap_sum, float(vlm_config.num_experts_per_tok), atol=1e-4, rtol=1e-4)

    # Delete our model to free GPU memory
    del model_cosmos3
    memory_cleanup(config_name)
    log.info(f"{config['model_type']} model deleted from GPU, memory freed")

    # For distributed models, only rank 0 continues with HF comparison
    if is_distributed and rank != 0:
        dist.barrier()
        dist.destroy_process_group()
        return

    #########################################################
    # Step 2: Get outputs from HF model, then delete it

    log.info(f"Loading HF {config['model_type']} model...")

    # Dynamic import based on model type
    if config["model_type"] == "moe":
        from transformers import Qwen3VLMoeForConditionalGeneration as HFModelClass
    elif config["model_type"] == "dense_llm":
        from transformers import Qwen3ForCausalLM as HFModelClass
    else:
        from transformers import Qwen3VLForConditionalGeneration as HFModelClass

    # Create the HF model
    tokenizer_hf = AutoTokenizer.from_pretrained(
        config["model_name"],
        cache_dir=f"/nfs/dir/dir_cosmos_base/users/{getpass.getuser()}/hf_cache/",
    )
    hf_vlm_model = HFModelClass.from_pretrained(
        config["model_name"],
        dtype=torch.bfloat16,
        device_map="auto",
        cache_dir=f"/nfs/dir/dir_cosmos_base/users/{getpass.getuser()}/hf_cache/",
    )
    if config["model_type"] == "dense_llm":
        hf_model = hf_vlm_model
    else:
        hf_model = hf_vlm_model.language_model

    all_hf_embeds = []

    # Process each message with HF model
    for i, (messages, model_inputs_cosmos3) in enumerate(zip(all_messages, all_model_inputs_cosmos3)):
        log.info(f"Processing HF model for message {i + 1}/{len(all_messages)}")

        # Process with HF model
        text = tokenizer_hf.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs_hf = tokenizer_hf([text], return_tensors="pt").to(hf_model.device)

        # Verify tokenization consistency
        assert (model_inputs_hf.input_ids.cpu() == torch.LongTensor(model_inputs_cosmos3.input_ids)).all(), (
            f"Tokenization outputs are not the same as hf for message {i + 1}"
        )

        # Forward pass with HF model
        hf_model.train()
        with torch.no_grad():
            outs_hf = hf_model(**model_inputs_hf, output_hidden_states=True)
        outs_hf = outs_hf.hidden_states[-1]
        outs_hf = outs_hf[0].cpu()  # Move to CPU for comparison
        all_hf_embeds.append(outs_hf)

        # Add rank-specific logging for distributed models
        log.info(f"HF model output norm: {outs_hf.norm()}")

    # Delete HF model to free GPU memory
    del hf_vlm_model
    del hf_model
    memory_cleanup(config_name)
    log.info(f"HF {config['model_type']} model deleted from GPU, memory freed")

    #########################################################
    # Step 3: Compare outputs for all messages

    # Compare all embeddings
    diffs = []
    for i, (text_embeds, outs_hf) in enumerate(zip(all_text_embeds, all_hf_embeds)):
        # Move text_embeds to same device as outs_hf for comparison
        text_embeds = text_embeds.to(outs_hf.device)

        # Compare if text_embeds and outs_hf are close
        diff = (text_embeds - outs_hf).norm() / text_embeds.norm()
        diffs.append(diff)
        log.info(f"Text embeds diff for message {i + 1}: {diff}")

    log.info(f"All {len(all_messages)} message comparisons passed!")
    log.info(f"Differences: {diffs}")

    # Use model-specific tolerance
    tolerance = config["output_tolerance"]
    assert all(diff < tolerance for diff in diffs), (
        f"Text embeds are not close for all messages (tolerance: {tolerance})"
    )

    # Cleanup distributed setup if needed
    if is_distributed:
        dist.barrier()
        dist.destroy_process_group()


@pytest.mark.L1
@pytest.mark.parametrize("config_name", [c for c in MODELS_TO_TEST if not TEST_CONFIGS[c]["requires_distributed"]])
def test_weight_comparison_embeddings_and_layer_0(config_name: str):
    """
    Test that compares embeddings and layer 0 weights between our model and HF model.
    Runs only for non-distributed models.
    """
    print(f"\n=== Weight Comparison Test (Embeddings + Layer 0) for {config_name} ===")
    config = TEST_CONFIGS[config_name]

    # Load our model (no device mesh for dense models)
    print("Loading our model...")
    model_cosmos3, _, _ = build_model_unified(config_name)

    # Load HF model
    print("Loading HF model...")
    if config["model_type"] == "dense_llm":
        from transformers import Qwen3ForCausalLM as HFModelClass
    else:
        from transformers import Qwen3VLForConditionalGeneration as HFModelClass

    hf_vlm_model = HFModelClass.from_pretrained(
        config["model_name"],
        dtype=torch.bfloat16,
        device_map="auto",
    )
    if config["model_type"] == "dense_llm":
        hf_model = hf_vlm_model
    else:
        hf_model = hf_vlm_model.language_model

    # Get state dicts
    our_state_dict = model_cosmos3.language_model.state_dict()
    hf_state_dict = hf_model.state_dict()

    # Check which keys our model has that HF doesn't
    # Strip "model." prefix from our keys for fair comparison
    if config["model_type"] == "dense_llm":
        our_keys_stripped = our_state_dict.keys()
    else:
        our_keys_stripped = {k.replace("model.", "") if k.startswith("model.") else k for k in our_state_dict.keys()}
    hf_keys = set(hf_state_dict.keys())

    our_only_keys = our_keys_stripped - hf_keys
    print(f"Keys in our model but not in HF (after stripping 'model.'): {list(our_only_keys)[:10]}")

    # Assert 1: Check if all diff keys have "moe_gen" substring
    assert all("moe_gen" in key for key in our_only_keys if "lm_head" not in key), (
        f"Not all diff keys contain 'moe_gen': {our_only_keys}"
    )

    # Assert 2: Check if number of diff keys equals number of HF keys
    if config["model_type"] == "dense_llm":
        # LLM models have embed_tokens and lm_head in state dict
        # gen keys don't have these two keys
        hf_keys_without_embed_and_lm_head = [
            key for key in hf_keys if ("embed_tokens" not in key and "lm_head" not in key)
        ]
        assert len(our_only_keys) == len(hf_keys_without_embed_and_lm_head), (
            f"Number of diff keys ({len(our_only_keys)}) != number of HF keys ({len(hf_keys_without_embed_and_lm_head)})"
        )
    else:
        assert len(our_only_keys) == len(hf_keys), (
            f"Number of diff keys ({len(our_only_keys)}) != number of HF keys ({len(hf_keys)})"
        )

    print(f"Our model has {len(our_state_dict)} parameters")
    print(f"HF model has {len(hf_state_dict)} parameters")

    # Compare embeddings
    embed_result = compare_single_weight(
        our_state_dict,
        hf_state_dict,
        our_key="model.embed_tokens.weight",
        hf_key="model.embed_tokens.weight" if config["model_type"] == "dense_llm" else "embed_tokens.weight",
        weight_name="embed_tokens",
    )

    allclose_results = {
        "embed_weights": embed_result["identical"],
    }

    print(f"\nEmbeddings comparison:")
    if embed_result["found_both"] and embed_result["shapes_match"]:
        print(f"  Shape: {embed_result['our_shape']} [PASS]")
        print(f"  Max diff: {embed_result['max_diff']:.2e}")
        print(f"  Mean diff: {embed_result['mean_diff']:.2e}")
        print(f"  Identical: {'[IDENTICAL]' if embed_result['identical'] else '[DIFFERENT]'}")
    else:
        print("  Embeddings comparison failed [FAIL]")

    # Compare layer 0 weights
    layer_0_weights = [
        "layers.0.input_layernorm.weight",
        "layers.0.self_attn.q_proj.weight",
        "layers.0.self_attn.k_proj.weight",
        "layers.0.self_attn.v_proj.weight",
        "layers.0.self_attn.o_proj.weight",
        "layers.0.self_attn.q_norm.weight",
        "layers.0.self_attn.k_norm.weight",
        "layers.0.mlp.gate_proj.weight",
        "layers.0.mlp.up_proj.weight",
        "layers.0.mlp.down_proj.weight",
        "layers.0.post_attention_layernorm.weight",
    ]

    print("Layer 0 weights comparison:")
    for weight_type in layer_0_weights:
        allclose_results[weight_type] = compare_single_weight(
            our_state_dict,
            hf_state_dict,
            our_key=f"model.{weight_type}",
            hf_key=f"model.{weight_type}" if config["model_type"] == "dense_llm" else weight_type,
            weight_name=weight_type,
        )["identical"]

    # Overall assessment
    print("SUMMARY:")
    print(f"  Weights compared: {len(allclose_results)}")
    print(f"  Identical weights: {sum(allclose_results.values())}/{len(allclose_results)}")

    # Assert weights are close enough
    assert all(allclose_results.values()), "Weights differ too much"

    # Cleanup
    del model_cosmos3
    del hf_vlm_model
    del hf_model
    memory_cleanup(config_name)


@pytest.mark.L1
@pytest.mark.parametrize("config_name", MODELS_TO_TEST)
def test_unified_cosmos3_network(config_name: str):
    """
    Unified test for Cosmos3 network functionality across different model types.
    """
    config = TEST_CONFIGS[config_name]
    is_distributed = config["requires_distributed"]

    # Initialize distributed setup if needed
    parallel_dims = None
    if is_distributed:
        parallel_dims = init_parallel(
            data_parallel_shard_degree=config["parallelism"].data_parallel_shard_degree,
            context_parallel_shard_degree=config["parallelism"].context_parallel_shard_degree,
        )

    log.info(f"\n=== Testing {config_name} network functionality ===")

    # Build model
    model_cosmos3, tokenizer_cosmos3, special_tokens = build_model_unified(config_name, parallel_dims)

    # Tokenize input strings
    num_samples = config["network_samples"]
    input_strings = ["Hello world", "How are you?", "I am fine"][:num_samples]
    input_text_tokens = [tokenizer_cosmos3.encode(s, add_special_tokens=False) for s in input_strings]
    input_timesteps = [0.0, 0.5, 0.9][:num_samples]

    # Create test inputs and pack sequence following training_step flow
    data_batch = create_packed_sequence(
        input_text_tokens=input_text_tokens,
        latent_shapes=config["network_latent_shapes"],
        input_timesteps=input_timesteps,
        special_tokens=special_tokens,
        latent_patch_size=model_cosmos3.config.latent_patch_size,
    )
    if parallel_dims is not None and parallel_dims.cp_enabled:
        log.info(f"Broadcasting data batch to context parallel ranks.")
        data_batch = context_parallel_broadcast(data_batch, parallel_dims, iteration=0)

    # Test patchify and unpatchify (narrow type after broadcast for type checker)
    assert isinstance(data_batch, PackedSequence)
    packed_latent, original_latent_shapes = model_cosmos3.patchify_and_pack_latents(
        tokens_vision=data_batch.vision.tokens,
        token_shapes_vision=data_batch.vision.token_shapes,
    )
    unpatchified_latent = model_cosmos3.unpatchify_and_unpack_latents(
        packed_mse_preds=packed_latent,
        token_shapes_vision=data_batch.vision.token_shapes,
        noisy_frame_indexes_vision=data_batch.vision.noisy_frame_indexes,
        original_latent_shapes=original_latent_shapes,
    )
    for i in range(len(unpatchified_latent)):
        isclose = torch.allclose(unpatchified_latent[i], data_batch.vision.tokens[i])
        assert isclose, f"Patchify and unpatchify are not inverses for {config_name}"

    # Move tensors to GPU
    data_batch.to_cuda()

    # Forward pass
    output = model_cosmos3(packed_seq=data_batch)

    # Verify output shapes
    preds_vision = output["preds_vision"]
    for i in range(len(preds_vision)):
        assert preds_vision[i].shape == data_batch.vision.tokens[i].shape, (
            f"Shape mismatch for {config_name}: {preds_vision[i].shape} vs {data_batch.vision.tokens[i].shape}"
        )

    # Cleanup
    del model_cosmos3
    memory_cleanup(config_name)

    # Cleanup distributed setup if needed
    if is_distributed:
        dist.destroy_process_group()


@pytest.mark.L1
@pytest.mark.parametrize("config_name", [c for c in MODELS_TO_TEST if not TEST_CONFIGS[c]["requires_distributed"]])
def test_patchify_unpatchify_non_divisible_shapes(config_name: str):
    """
    Unit test for patchify_and_pack_latents and unpatchify_and_unpack_latents
    with latent shapes not divisible by patch size.
    Tests that zero-padding and unpadding work correctly to preserve the original latent.
    """
    config = TEST_CONFIGS[config_name]

    log.info(f"\n=== Testing non-divisible latent shapes for {config_name} ===")

    # Build model (non-distributed only for simplicity)
    model_cosmos3, _, _ = build_model_unified(config_name, parallel_dims=None)

    p = model_cosmos3.config.latent_patch_size
    log.info(f"Testing non-divisible latent shapes with patch_size={p}")

    # Create latents with shapes not divisible by patch size
    for shape_to_test in [(63, 65), (33, 31), (111, 191)]:
        non_div_shapes = [(1, shape_to_test[0], shape_to_test[1])] * config["network_samples"]
        non_div_tokens = [
            torch.randn(1, model_cosmos3.latent_channel, t, h, w) for t, h, w in non_div_shapes
        ]  # [1,C,t,h,w] each
        non_div_shapes_after_patchify = [(t, math.ceil(h / p), math.ceil(w / p)) for t, h, w in non_div_shapes]

        # Create dummy condition mask (all noisy)

        latent_t = max(s[0] for s in non_div_shapes)
        noisy_frame_indexes = torch.arange(latent_t, device=non_div_tokens[0].device, dtype=torch.long)  # [latent_t]
        noisy_frame_indexes = noisy_frame_indexes.unsqueeze(0).expand(len(non_div_shapes), -1)  # [num_samples,latent_t]

        packed_non_div, orig_shapes_non_div = model_cosmos3.patchify_and_pack_latents(
            tokens_vision=non_div_tokens,
            token_shapes_vision=non_div_shapes_after_patchify,
        )
        unpatchified_non_div = model_cosmos3.unpatchify_and_unpack_latents(
            packed_mse_preds=packed_non_div,
            token_shapes_vision=non_div_shapes_after_patchify,
            noisy_frame_indexes_vision=noisy_frame_indexes,
            original_latent_shapes=orig_shapes_non_div,
        )

        for i in range(len(unpatchified_non_div)):
            expected_shape = non_div_tokens[i].shape
            actual_shape = unpatchified_non_div[i].shape
            assert actual_shape == expected_shape, (
                f"Non-divisible shape mismatch for {config_name}: expected {expected_shape}, got {actual_shape}"
            )
            isclose = torch.allclose(unpatchified_non_div[i], non_div_tokens[i])
            assert isclose, f"Non-divisible patchify/unpatchify not inverses for {config_name}, shape {expected_shape}"

            log.info(f"Non-divisible shape test PASSED for {config_name}, shape {shape_to_test}")

    # Cleanup
    del model_cosmos3
    memory_cleanup(config_name)
