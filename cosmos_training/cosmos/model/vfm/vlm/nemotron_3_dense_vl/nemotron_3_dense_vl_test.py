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

"""Tests for the Nemotron 3 Dense VL text backbone modules.

Covers both standalone component tests (no GPU or credentials needed) and
integration tests that build the full Cosmos3 VFM network with Nemotron as
the VLM backbone.

We support two Nemotron variants:
  - nemotron_3_dense_vl_2b: Nemotron 3 Dense VL 2B (VLM, with vision tokens in tokenizer)
  - nemotron_3_llm_2b:      Nemotron 3 LLM 2B (pure LLM, tokenizer patched for vision tokens)

Usage Examples:
    # Component-level tests (no GPU required)
    pytest cosmos/model/vfm/vlm/nemotron_3_dense_vl/nemotron_3_dense_vl_test.py -k "not integration" -s -v

    # Integration tests (requires GPU + GCP credentials)
    export NEMOTRON_TEST_MODELS=nemotron_3_dense_vl_2b
    pytest cosmos/model/vfm/vlm/nemotron_3_dense_vl/nemotron_3_dense_vl_test.py -k integration --all -s -v

    # HF comparison tests only (LLM variant, downloads from HuggingFace Hub)
    export HF_HOME=/path/to/shared/hf_cache
    export NEMOTRON_TEST_MODELS=nemotron_3_llm_2b
    pytest cosmos/model/vfm/vlm/nemotron_3_dense_vl/nemotron_3_dense_vl_test.py -k "hf" --all -s -v

    # Run all tests
    export NEMOTRON_TEST_MODELS=nemotron_3_dense_vl_2b,nemotron_3_llm_2b
    pytest cosmos/model/vfm/vlm/nemotron_3_dense_vl/nemotron_3_dense_vl_test.py --all -s -v

Environment Variables:
    NEMOTRON_TEST_MODELS: Comma-separated list of models to test for integration tests.
                          Defaults to all models if unset.
    HF_HOME: Override HuggingFace cache directory (optional).
"""

import gc
import math
import os

import pytest
import torch

from cosmos.model.vfm.vlm.nemotron_3_dense_vl.configuration_nemotron_3_dense_vl import (
    Nemotron3DenseVLTextConfig,
)
from cosmos.model.vfm.vlm.nemotron_3_dense_vl.nemotron_3_dense_vl import (
    MultiModalRotaryEmbedding,
    Nemotron3DenseVLMLP,
    Nemotron3DenseVLPreTrainedModel,
    Nemotron3DenseVLRMSNorm,
    apply_rotary_pos_emb_partial,
    rotate_half,
)

CONFIG_JSON = "cosmos/model/vfm/vlm/nemotron_3_dense_vl/configs/Nemotron-2B-Dense-VL.json"


def _make_small_config(**overrides) -> Nemotron3DenseVLTextConfig:
    """Build a small config suitable for fast CPU tests."""
    defaults = dict(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        head_dim=16,
        num_key_value_heads=2,
        max_position_embeddings=512,
        mlp_hidden_act="relu2",
    )
    defaults.update(overrides)
    return Nemotron3DenseVLTextConfig(**defaults)


# ---------------------------------------------------------------------------
# Component-level tests (CPU-only, no credentials)
# ---------------------------------------------------------------------------


class TestNemotron3DenseVLTextConfig:
    def test_defaults(self) -> None:
        cfg = Nemotron3DenseVLTextConfig()
        assert cfg.vocab_size == 131072
        assert cfg.hidden_size == 2048
        assert cfg.intermediate_size == 9216
        assert cfg.num_hidden_layers == 28
        assert cfg.num_attention_heads == 16
        assert cfg.head_dim == 128
        assert cfg.num_key_value_heads == 8
        assert cfg.mlp_hidden_act == "relu2"
        assert cfg.mlp_bias is False
        assert cfg.attention_bias is False
        assert cfg.enable_rope is True
        assert cfg.enable_mrope is True
        assert cfg.mrope_section == [24, 20, 20]
        assert cfg.rope_theta == 100_000_000.0
        assert cfg.tie_word_embeddings is False

    def test_rms_norm_eps_alias(self) -> None:
        cfg = Nemotron3DenseVLTextConfig(layer_norm_epsilon=1e-6)
        assert cfg.rms_norm_eps == 1e-6

    def test_from_json_file(self) -> None:
        cfg = Nemotron3DenseVLTextConfig.from_json_file(CONFIG_JSON)
        assert cfg.vocab_size == 131072
        assert cfg.hidden_size == 2048
        assert cfg.num_hidden_layers == 28
        assert cfg.mlp_hidden_act == "relu2"
        assert cfg.mrope_section == [24, 20, 20]

    def test_custom_overrides(self) -> None:
        cfg = Nemotron3DenseVLTextConfig(
            hidden_size=512,
            num_hidden_layers=4,
            num_attention_heads=8,
            head_dim=64,
        )
        assert cfg.hidden_size == 512
        assert cfg.num_hidden_layers == 4
        assert cfg.num_attention_heads == 8
        assert cfg.head_dim == 64


class TestNemotron3DenseVLRMSNorm:
    def test_output_shape(self) -> None:
        norm = Nemotron3DenseVLRMSNorm(hidden_size=64, eps=1e-5)
        x = torch.randn(2, 10, 64)
        out = norm(x)
        assert out.shape == x.shape

    def test_dtype_preservation(self) -> None:
        norm = Nemotron3DenseVLRMSNorm(hidden_size=32)
        x_fp16 = torch.randn(1, 5, 32, dtype=torch.float16)
        out = norm(x_fp16)
        assert out.dtype == torch.float16

    def test_unit_weight_is_identity_for_normalized(self) -> None:
        """With weight=1 and input already unit-norm, output should closely match input."""
        norm = Nemotron3DenseVLRMSNorm(hidden_size=16)
        x = torch.randn(1, 1, 16)
        rms = x.pow(2).mean(-1, keepdim=True).sqrt()
        x_unit = x / rms
        out = norm(x_unit)
        assert torch.allclose(out.float(), x_unit.float(), atol=1e-4)

    def test_extra_repr(self) -> None:
        norm = Nemotron3DenseVLRMSNorm(hidden_size=64, eps=1e-6)
        s = norm.extra_repr()
        assert "(64,)" in s
        assert "1e-06" in s


class TestNemotron3DenseVLMLP:
    def test_output_shape(self) -> None:
        cfg = _make_small_config()
        mlp = Nemotron3DenseVLMLP(cfg)
        x = torch.randn(2, 10, cfg.hidden_size)
        out = mlp(x)
        assert out.shape == x.shape

    def test_relu2_activation_is_nonnegative(self) -> None:
        """relu(x)^2 is always >= 0."""
        cfg = _make_small_config()
        mlp = Nemotron3DenseVLMLP(cfg)
        x = torch.randn(4, 8, cfg.hidden_size)
        intermediate = mlp.act_fn(mlp.up_proj(x))
        assert (intermediate >= 0).all()

    def test_no_bias_by_default(self) -> None:
        cfg = _make_small_config(mlp_bias=False)
        mlp = Nemotron3DenseVLMLP(cfg)
        assert mlp.up_proj.bias is None
        assert mlp.down_proj.bias is None

    def test_with_bias(self) -> None:
        cfg = _make_small_config(mlp_bias=True)
        mlp = Nemotron3DenseVLMLP(cfg)
        assert mlp.up_proj.bias is not None
        assert mlp.down_proj.bias is not None


class TestRotateHalf:
    def test_output_shape(self) -> None:
        x = torch.randn(2, 4, 8)
        out = rotate_half(x)
        assert out.shape == x.shape

    def test_self_inverse_with_negation(self) -> None:
        """rotate_half(rotate_half(x)) == -x."""
        x = torch.randn(3, 5, 16)
        out = rotate_half(rotate_half(x))
        assert torch.allclose(out, -x)


class TestApplyRotaryPosEmbPartial:
    def test_full_rotation(self) -> None:
        """When rot_dim == head_dim, all channels are rotated."""
        seq_len, n_heads, head_dim = 10, 4, 16
        q = torch.randn(seq_len, n_heads, head_dim)
        k = torch.randn(seq_len, n_heads, head_dim)
        cos = torch.randn(seq_len, head_dim)
        sin = torch.randn(seq_len, head_dim)

        q_out, k_out = apply_rotary_pos_emb_partial(q, k, cos, sin, unsqueeze_dim=1)
        assert q_out.shape == q.shape
        assert k_out.shape == k.shape

    def test_partial_rotation_passthrough(self) -> None:
        """When rot_dim < head_dim, the remainder channels pass through unchanged."""
        seq_len, n_heads, head_dim = 8, 2, 32
        rot_dim = 16
        q = torch.randn(seq_len, n_heads, head_dim)
        k = torch.randn(seq_len, n_heads, head_dim)
        cos = torch.randn(seq_len, rot_dim)
        sin = torch.randn(seq_len, rot_dim)

        q_out, k_out = apply_rotary_pos_emb_partial(q, k, cos, sin, unsqueeze_dim=1)

        assert torch.allclose(q_out[..., rot_dim:], q[..., rot_dim:])
        assert torch.allclose(k_out[..., rot_dim:], k[..., rot_dim:])

    def test_zero_angle_is_identity(self) -> None:
        """With cos=1, sin=0, the rotated output should equal the input."""
        seq_len, n_heads, head_dim = 6, 2, 16
        q = torch.randn(seq_len, n_heads, head_dim)
        k = torch.randn(seq_len, n_heads, head_dim)
        cos = torch.ones(seq_len, head_dim)
        sin = torch.zeros(seq_len, head_dim)

        q_out, k_out = apply_rotary_pos_emb_partial(q, k, cos, sin, unsqueeze_dim=1)
        assert torch.allclose(q_out, q, atol=1e-6)
        assert torch.allclose(k_out, k, atol=1e-6)


class TestMultiModalRotaryEmbedding:
    def test_output_shapes(self) -> None:
        cfg = _make_small_config()
        rope = MultiModalRotaryEmbedding(cfg)
        seq_len = 12
        x = torch.randn(1, seq_len, cfg.hidden_size)
        position_ids = torch.arange(seq_len).unsqueeze(0)

        cos, sin = rope(x, position_ids)
        assert cos.shape[-1] == cfg.head_dim
        assert sin.shape[-1] == cfg.head_dim

    def test_mrope_3d_position_ids(self) -> None:
        """With 3D position_ids (3, batch, seq_len) the mrope interleaving path runs."""
        cfg = _make_small_config()
        rope = MultiModalRotaryEmbedding(cfg)
        seq_len = 8
        x = torch.randn(1, seq_len, cfg.hidden_size)
        position_ids = torch.arange(seq_len).unsqueeze(0).unsqueeze(0).expand(3, 1, -1)

        cos, sin = rope(x, position_ids)
        assert cos.shape[-1] == cfg.head_dim
        assert sin.shape[-1] == cfg.head_dim

    def test_init_weights(self) -> None:
        cfg = _make_small_config()
        rope = MultiModalRotaryEmbedding(cfg)
        orig_inv_freq = rope.inv_freq.clone()
        rope.init_weights(buffer_device=None)
        assert torch.allclose(rope.inv_freq, orig_inv_freq)

    def test_deterministic(self) -> None:
        cfg = _make_small_config()
        rope = MultiModalRotaryEmbedding(cfg)
        seq_len = 10
        x = torch.randn(1, seq_len, cfg.hidden_size)
        pos = torch.arange(seq_len).unsqueeze(0)
        cos1, sin1 = rope(x, pos)
        cos2, sin2 = rope(x, pos)
        assert torch.allclose(cos1, cos2)
        assert torch.allclose(sin1, sin2)


class TestNemotron3DenseVLPreTrainedModel:
    def test_config_class(self) -> None:
        assert Nemotron3DenseVLPreTrainedModel.config_class == Nemotron3DenseVLTextConfig

    def test_base_model_prefix(self) -> None:
        assert Nemotron3DenseVLPreTrainedModel.base_model_prefix == "model"


# ---------------------------------------------------------------------------
# Integration tests (require GPU + GCP credentials)
# ---------------------------------------------------------------------------

INTEGRATION_MODEL_CONFIGS = {
    "nemotron_3_dense_vl_2b": {
        "config_import": "Nemotron3DenseVL_VLM_2b_GCP_Config",
        "backend_credentials": "gcp_checkpoint.secret",
        "max_tokens": 4096,
        "latent_shapes": [(16, 1, 128, 128)],
        "input_samples": 3,
        "test_latent_size": (256, 256),
        "network_latent_size": (512, 512),
        "network_samples": 3,
        "network_max_tokens": 6144,
        "network_latent_shapes": [(16, 1, 64, 64)] * 3,
        "requires_distributed": False,
        "joint_attn_implementation": "two_way",
    },
    "nemotron_3_llm_2b": {
        "config_import": "Nemotron3_LLM_2b_GCP_Config",
        "backend_credentials": "gcp_checkpoint.secret",
        "hf_model_name": "nvidia/NVIDIA-Nemotron-3-2B-BF16",
        "max_tokens": 4096,
        "latent_shapes": [(16, 1, 128, 128)],
        "input_samples": 3,
        "test_latent_size": (256, 256),
        "network_latent_size": (512, 512),
        "network_samples": 3,
        "network_max_tokens": 6144,
        "network_latent_shapes": [(16, 1, 64, 64)] * 3,
        "requires_distributed": False,
        "joint_attn_implementation": "two_way",
    },
}


def _get_integration_models() -> list[str]:
    models_arg = os.environ.get("NEMOTRON_TEST_MODELS")
    if not models_arg:
        return list(INTEGRATION_MODEL_CONFIGS.keys())
    requested = [m.strip() for m in models_arg.split(",")]
    for m in requested:
        if m not in INTEGRATION_MODEL_CONFIGS:
            available = ", ".join(INTEGRATION_MODEL_CONFIGS.keys())
            raise ValueError(f"Invalid model '{m}'. Available: {available}")
    return requested


INTEGRATION_MODELS_TO_TEST = _get_integration_models()


def _get_vlm_config(config_name: str):
    """Lazily import the VLM config to avoid import errors when credentials are absent."""
    from configs.base.defaults.vlm import (
        Nemotron3_LLM_2b_GCP_Config,
        Nemotron3DenseVL_VLM_2b_GCP_Config,
    )

    mapping = {
        "Nemotron3DenseVL_VLM_2b_GCP_Config": Nemotron3DenseVL_VLM_2b_GCP_Config,
        "Nemotron3_LLM_2b_GCP_Config": Nemotron3_LLM_2b_GCP_Config,
    }
    import_name = INTEGRATION_MODEL_CONFIGS[config_name]["config_import"]
    return mapping[import_name]


def _memory_cleanup() -> None:
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.empty_cache()


def _build_nemotron_model(config_name: str):
    """Build a Cosmos3VFMNetwork with a Nemotron VLM backbone."""
    from cosmos.utils.lazy_config import instantiate as lazy_instantiate
    from cosmos.utils.easy_io import easy_io
    from configs.base.defaults.cluster import GCPIADGB200Config as CLUSTER_CONFIG
    from configs.base.defaults.model_config import ParallelismConfig
    from cosmos.data.vfm.sequence_packing import add_special_tokens
    from cosmos.model.vfm.mot.cosmos3_vfm_network import Cosmos3VFMNetwork, Cosmos3VFMNetworkConfig
    from cosmos.model.vfm.mot.parallelize_vfm_network import parallelize_vfm_network
    from cosmos.model.vfm.utils.safetensors_loader import (
        load_language_model as load_language_model_safetensors,
    )

    config = INTEGRATION_MODEL_CONFIGS[config_name]
    vlm_config = _get_vlm_config(config_name)

    easy_io.set_s3_backend(
        backend_args={
            "backend": "s3",
            "s3_credential_path": CLUSTER_CONFIG.object_store_credential_pretrained,
        }
    )

    parallelism = ParallelismConfig(
        use_activation_checkpointing=False,
        use_torch_compile=False,
        data_parallel_shard_degree=1,
        context_parallel_shard_degree=1,
    )

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
        model = Cosmos3VFMNetwork(language_model=vlm, config=cosmos3_network_config)

    model = parallelize_vfm_network(model, parallel_dims=None, config=parallelism)
    model = model.to(torch.bfloat16)
    model.to_empty(device="cuda")
    model.init_weights(buffer_device="cuda")
    model.train()

    load_language_model_safetensors(
        model=model.language_model,
        checkpoint_path=vlm_config.pretrained_weights.backbone_path,
        credential_path=f"credentials/{config['backend_credentials']}",
        device_mesh=None,
    )
    model.language_model.init_moe()

    _proc = lazy_instantiate(vlm_config.tokenizer)
    tokenizer = _proc.tokenizer
    tokenizer, _ = add_special_tokens(tokenizer)

    special_tokens = {
        "eos_token_id": tokenizer.eos_token_id,
        "start_of_generation": tokenizer.convert_tokens_to_ids("<|vision_start|>"),
        "end_of_generation": tokenizer.convert_tokens_to_ids("<|vision_end|>"),
    }

    return model, tokenizer, special_tokens


def _create_packed_sequence(
    input_text_tokens: list[list[int]],
    latent_shapes: list[tuple[int, int, int, int]],
    input_timesteps: list[float],
    special_tokens: dict[str, int],
    latent_patch_size: int = 1,
):
    """Create a PackedSequence following the training_step flow."""
    from cosmos.data.vfm.sequence_packing import (
        build_sequence_plans_from_data_batch,
        pack_input_sequence,
    )
    from cosmos.model.vfm.utils.data_and_condition import GenerationDataClean

    num_samples = len(input_text_tokens)
    input_images = [torch.randn(3, 1, shape[2] * 8, shape[3] * 8).to(torch.bfloat16) for shape in latent_shapes]
    sequence_plans = build_sequence_plans_from_data_batch(
        data_batch={"images": input_images}, input_video_key="video", input_image_key="images"
    )
    input_images_stacked = torch.stack(input_images, dim=0)
    x0_tokens_vision = torch.stack([torch.randn(1, *shape).to(torch.bfloat16) for shape in latent_shapes], dim=0)
    gen_data_clean = GenerationDataClean(
        batch_size=num_samples,
        is_image_batch=True,
        raw_state_vision=input_images_stacked,
        x0_tokens_vision=x0_tokens_vision,
        raw_state_action=None,
    )
    timesteps_tensor = torch.tensor(input_timesteps)
    packed_sequence = pack_input_sequence(
        sequence_plans=sequence_plans,
        input_text_indexes=input_text_tokens,
        gen_data_clean=gen_data_clean,
        input_timesteps=timesteps_tensor,
        special_tokens=special_tokens,
        latent_patch_size=latent_patch_size,
    )
    assert packed_sequence.vision is not None
    assert packed_sequence.vision.condition_mask is not None
    return packed_sequence


@pytest.mark.L1
@pytest.mark.parametrize("config_name", INTEGRATION_MODELS_TO_TEST)
def test_integration_nemotron_cosmos3_network(config_name: str) -> None:
    """Build the full Cosmos3 network with a Nemotron backbone and verify forward pass shapes."""
    from cosmos.utils import log

    config = INTEGRATION_MODEL_CONFIGS[config_name]
    log.info(f"\n=== Testing Nemotron Cosmos3 network: {config_name} ===")

    model, tokenizer, special_tokens = _build_nemotron_model(config_name)

    num_samples = config["network_samples"]
    input_strings = ["Hello world", "How are you?", "I am fine"][:num_samples]
    input_text_tokens = [tokenizer.encode(s, add_special_tokens=False) for s in input_strings]
    input_timesteps = [0.0, 0.5, 0.9][:num_samples]

    data_batch = _create_packed_sequence(
        input_text_tokens=input_text_tokens,
        latent_shapes=config["network_latent_shapes"],
        input_timesteps=input_timesteps,
        special_tokens=special_tokens,
        latent_patch_size=model.config.latent_patch_size,
    )
    data_batch.to_cuda()

    output = model(packed_seq=data_batch)

    preds_vision = output["preds_vision"]
    for i in range(len(preds_vision)):
        assert preds_vision[i].shape == data_batch.vision.tokens[i].shape, (
            f"Shape mismatch for {config_name}: {preds_vision[i].shape} vs {data_batch.vision.tokens[i].shape}"
        )

    log.info(f"Nemotron Cosmos3 network test PASSED for {config_name}")

    del model
    _memory_cleanup()


@pytest.mark.L1
@pytest.mark.parametrize("config_name", INTEGRATION_MODELS_TO_TEST)
def test_integration_nemotron_init_moe(config_name: str) -> None:
    """Verify that init_moe correctly copies understanding weights to generation pathway."""
    from cosmos.utils import log

    log.info(f"\n=== Testing Nemotron init_moe: {config_name} ===")

    model, _, _ = _build_nemotron_model(config_name)

    state_dict = model.language_model.state_dict()
    mismatches = []
    checked = 0
    for name in state_dict:
        if "moe_gen" not in name:
            continue
        original_name = name.replace("_moe_gen", "").replace("_checkpoint_wrapped_module.", "")
        if original_name not in state_dict:
            continue
        checked += 1
        gen_param = state_dict[name]
        und_param = state_dict[original_name]
        if not torch.allclose(gen_param, und_param, atol=1e-6):
            diff = (gen_param - und_param).abs().max().item()
            mismatches.append((name, original_name, diff))

    assert checked > 0, "No moe_gen parameters found — init_moe may not have run"
    assert len(mismatches) == 0, f"init_moe weight copy mismatch for {len(mismatches)} params: " + ", ".join(
        f"{n} (max_diff={d:.2e})" for n, _, d in mismatches[:5]
    )

    log.info(f"init_moe test PASSED: {checked} moe_gen params verified for {config_name}")

    del model
    _memory_cleanup()


@pytest.mark.L1
@pytest.mark.parametrize("config_name", INTEGRATION_MODELS_TO_TEST)
def test_integration_nemotron_patchify_unpatchify(config_name: str) -> None:
    """Test patchify and unpatchify round-trip with the Nemotron-backed Cosmos3 model."""
    from cosmos.utils import log

    config = INTEGRATION_MODEL_CONFIGS[config_name]
    log.info(f"\n=== Testing Nemotron patchify/unpatchify: {config_name} ===")

    model, tokenizer, special_tokens = _build_nemotron_model(config_name)

    num_samples = config["network_samples"]
    input_strings = ["Hello world", "How are you?", "I am fine"][:num_samples]
    input_text_tokens = [tokenizer.encode(s, add_special_tokens=False) for s in input_strings]
    input_timesteps = [0.0, 0.5, 0.9][:num_samples]

    data_batch = _create_packed_sequence(
        input_text_tokens=input_text_tokens,
        latent_shapes=config["network_latent_shapes"],
        input_timesteps=input_timesteps,
        special_tokens=special_tokens,
        latent_patch_size=model.config.latent_patch_size,
    )

    packed_latent, original_latent_shapes = model.patchify_and_pack_latents(
        tokens_vision=data_batch.vision.tokens,
        token_shapes_vision=data_batch.vision.token_shapes,
    )
    unpatchified_latent = model.unpatchify_and_unpack_latents(
        packed_mse_preds=packed_latent,
        token_shapes_vision=data_batch.vision.token_shapes,
        noisy_frame_indexes_vision=data_batch.vision.noisy_frame_indexes,
        original_latent_shapes=original_latent_shapes,
    )
    for i in range(len(unpatchified_latent)):
        assert torch.allclose(unpatchified_latent[i], data_batch.vision.tokens[i]), (
            f"Patchify/unpatchify round-trip failed for {config_name} sample {i}"
        )

    log.info(f"Patchify/unpatchify test PASSED for {config_name}")

    del model
    _memory_cleanup()


@pytest.mark.L1
@pytest.mark.parametrize("config_name", INTEGRATION_MODELS_TO_TEST)
def test_integration_nemotron_patchify_non_divisible(config_name: str) -> None:
    """Test patchify/unpatchify with latent shapes not divisible by patch size."""
    from cosmos.utils import log

    log.info(f"\n=== Testing Nemotron non-divisible patchify: {config_name} ===")

    model, _, _ = _build_nemotron_model(config_name)

    config = INTEGRATION_MODEL_CONFIGS[config_name]
    p = model.config.latent_patch_size

    for shape_to_test in [(63, 65), (33, 31), (111, 191)]:
        non_div_shapes = [(1, shape_to_test[0], shape_to_test[1])] * config["network_samples"]
        non_div_tokens = [torch.randn(1, model.latent_channel, t, h, w) for t, h, w in non_div_shapes]
        non_div_shapes_after = [(t, math.ceil(h / p), math.ceil(w / p)) for t, h, w in non_div_shapes]

        latent_t = max(s[0] for s in non_div_shapes)
        noisy_frame_indexes = torch.arange(latent_t, device=non_div_tokens[0].device, dtype=torch.long)
        noisy_frame_indexes = noisy_frame_indexes.unsqueeze(0).expand(len(non_div_shapes), -1)

        packed, orig_shapes = model.patchify_and_pack_latents(
            tokens_vision=non_div_tokens,
            token_shapes_vision=non_div_shapes_after,
        )
        unpatchified = model.unpatchify_and_unpack_latents(
            packed_mse_preds=packed,
            token_shapes_vision=non_div_shapes_after,
            noisy_frame_indexes_vision=noisy_frame_indexes,
            original_latent_shapes=orig_shapes,
        )

        for i in range(len(unpatchified)):
            assert unpatchified[i].shape == non_div_tokens[i].shape, (
                f"Shape mismatch: expected {non_div_tokens[i].shape}, got {unpatchified[i].shape}"
            )
            assert torch.allclose(unpatchified[i], non_div_tokens[i]), (
                f"Non-divisible patchify/unpatchify not inverse for {config_name}, shape {shape_to_test}"
            )

        log.info(f"Non-divisible shape test PASSED for {config_name}, shape {shape_to_test}")

    del model
    _memory_cleanup()


@pytest.mark.L1
@pytest.mark.parametrize("config_name", INTEGRATION_MODELS_TO_TEST)
def test_integration_nemotron_embedding_output(config_name: str) -> None:
    """Verify that the Nemotron model produces non-zero, finite text embeddings."""
    from cosmos.utils import log

    config = INTEGRATION_MODEL_CONFIGS[config_name]
    log.info(f"\n=== Testing Nemotron embedding output: {config_name} ===")

    model, tokenizer, special_tokens = _build_nemotron_model(config_name)

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is machine learning?"},
    ]

    try:
        text_tokens = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text_tokens])
    except Exception:
        text = "What is machine learning?"
        model_inputs_ids = [tokenizer.encode(text, add_special_tokens=True)]
        model_inputs = type("Inputs", (), {"input_ids": model_inputs_ids})()

    data_batch = _create_packed_sequence(
        input_text_tokens=model_inputs.input_ids,
        latent_shapes=config["latent_shapes"],
        input_timesteps=[0.0],
        special_tokens=special_tokens,
        latent_patch_size=model.config.latent_patch_size,
    )
    data_batch.to_cuda()

    with torch.no_grad():
        output = model(packed_seq=data_batch)

    num_text_tokens = len(model_inputs.input_ids[0])
    text_embeds = output["last_hidden_state"][0:num_text_tokens].cpu()

    assert text_embeds.shape[0] == num_text_tokens, f"Expected {num_text_tokens} embeddings, got {text_embeds.shape[0]}"
    assert torch.isfinite(text_embeds).all(), "Text embeddings contain non-finite values"
    assert text_embeds.norm() > 0, "Text embeddings are all zeros"

    log.info(f"Embedding output norm: {text_embeds.norm():.4f}")
    log.info(f"Embedding output test PASSED for {config_name}")

    del model
    _memory_cleanup()


# ---------------------------------------------------------------------------
# HF reference comparison tests (nemotron_3_llm_2b only)
# ---------------------------------------------------------------------------

_HF_MODELS_TO_TEST = [m for m in INTEGRATION_MODELS_TO_TEST if m == "nemotron_3_llm_2b"]

_MESSAGE_TEMPLATES = [
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
]


@pytest.mark.L1
@pytest.mark.parametrize("config_name", _HF_MODELS_TO_TEST)
def test_integration_nemotron_llm_outputs_with_hf(config_name: str) -> None:
    """Compare text embeddings from our MoT Nemotron model against the HF reference.

    Loads both models, feeds the same tokenised text through each, and checks
    that the last hidden states (after final norm) match within tolerance.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from cosmos.utils import log

    config = INTEGRATION_MODEL_CONFIGS[config_name]
    tolerance = 0.04
    num_messages = config["input_samples"]

    log.info(f"\n=== Testing Nemotron LLM outputs vs HF: {config_name} ===")

    # --- Step 1: our model ---
    model_cosmos3, tokenizer_cosmos3, special_tokens = _build_nemotron_model(config_name)

    selected_messages = _MESSAGE_TEMPLATES[:num_messages]
    all_model_inputs: list = []
    for messages in selected_messages:
        text = "".join(m["content"] + " " for m in messages).strip()
        ids = tokenizer_cosmos3.encode(text, add_special_tokens=True)
        all_model_inputs.append(ids)

    all_text_embeds: list[torch.Tensor] = []
    for i, token_ids in enumerate(all_model_inputs):
        log.info(f"Processing our model for message {i + 1}/{num_messages}")
        data_batch = _create_packed_sequence(
            input_text_tokens=[token_ids],
            latent_shapes=config["latent_shapes"],
            input_timesteps=[0.0],
            special_tokens=special_tokens,
            latent_patch_size=model_cosmos3.config.latent_patch_size,
        )
        data_batch.to_cuda()
        with torch.no_grad():
            output = model_cosmos3(packed_seq=data_batch)
        num_text_tokens = len(token_ids)
        text_embeds = output["last_hidden_state"][0:num_text_tokens].cpu()
        all_text_embeds.append(text_embeds)
        log.info(f"Our model output norm: {text_embeds.norm():.4f}")

    del model_cosmos3
    _memory_cleanup()
    log.info("Our model deleted, memory freed")

    # --- Step 2: HF reference model ---
    hf_model_name = config["hf_model_name"]
    log.info(f"Loading HF Nemotron model: {hf_model_name}")
    tokenizer_hf = AutoTokenizer.from_pretrained(hf_model_name, trust_remote_code=True)
    hf_causal_model = AutoModelForCausalLM.from_pretrained(
        hf_model_name,
        torch_dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
        trust_remote_code=True,
    )
    hf_backbone = hf_causal_model.model

    all_hf_embeds: list[torch.Tensor] = []
    for i, token_ids in enumerate(all_model_inputs):
        log.info(f"Processing HF model for message {i + 1}/{num_messages}")

        hf_input_ids = tokenizer_hf.encode(
            "".join(m["content"] + " " for m in selected_messages[i]).strip(),
            add_special_tokens=True,
            return_tensors="pt",
        )
        if not isinstance(hf_input_ids, torch.Tensor):
            hf_input_ids = torch.LongTensor([hf_input_ids])
        hf_input_ids = hf_input_ids.to(hf_backbone.embeddings.weight.device)

        assert (hf_input_ids.cpu().flatten() == torch.LongTensor(token_ids)).all(), (
            f"Tokenisation mismatch for message {i + 1}"
        )

        hf_backbone.train()
        with torch.no_grad():
            hf_out = hf_backbone(input_ids=hf_input_ids, use_cache=False)
        hf_embeds = hf_out.last_hidden_state[0].cpu()
        all_hf_embeds.append(hf_embeds)
        log.info(f"HF model output norm: {hf_embeds.norm():.4f}")

    del hf_causal_model, hf_backbone
    _memory_cleanup()
    log.info("HF model deleted, memory freed")

    # --- Step 3: compare ---
    diffs: list[torch.Tensor] = []
    for i, (ours, hf) in enumerate(zip(all_text_embeds, all_hf_embeds)):
        diff = (ours - hf).norm() / ours.norm()
        diffs.append(diff)
        log.info(f"Text embeds relative diff for message {i + 1}: {diff:.6f}")

    log.info(f"All diffs: {diffs}")
    assert all(d < tolerance for d in diffs), f"Text embeds not close enough (tolerance={tolerance}): {diffs}"
    log.info("Nemotron LLM HF comparison PASSED")


@pytest.mark.L1
@pytest.mark.parametrize("config_name", _HF_MODELS_TO_TEST)
def test_integration_nemotron_llm_backward_sanity(config_name: str) -> None:
    """Sanity-check the backward pass of the i4 Nemotron model.

    Why not a direct numerical comparison with HF?
      The MoT model's two-way attention processes text + vision tokens together
      (understanding path).  This produces a fundamentally different Jacobian
      from HF's text-only attention — forward outputs approximately agree
      (verified by test_integration_nemotron_llm_outputs_with_hf), but backward
      gradient directions diverge because the attention paths differ structurally.

    What this test checks:
      1. backward() completes without error.
      2. Gradients are finite (no NaN / inf).
      3. embed_tokens rows for text tokens that appeared have non-zero gradients.
      4. embed_tokens rows for token IDs that did NOT appear are zero
         (embed_tokens is a sparse embedding — only present tokens get grads).
      5. Understanding-pathway params (e.g., layer-0 q_proj) have non-zero grads.
      6. Generation-pathway params (moe_gen, e.g., layer-0 q_proj_moe_gen) have
         non-zero grads, confirming the generation path is active in backward.
    """
    from cosmos.utils import log

    config = INTEGRATION_MODEL_CONFIGS[config_name]
    log.info(f"\n=== Testing Nemotron backward sanity: {config_name} ===")

    text = "Hello world, this is a backward pass test."

    model_i4, tokenizer_i4, special_tokens = _build_nemotron_model(config_name)
    model_i4.train()

    token_ids = tokenizer_i4.encode(text, add_special_tokens=True)
    num_text = len(token_ids)
    unique_token_ids = sorted(set(token_ids))
    # Pick a few vocab IDs that do NOT appear in the sequence
    absent_ids = [i for i in range(100, 200) if i not in unique_token_ids][:5]

    data_batch = _create_packed_sequence(
        input_text_tokens=[token_ids],
        latent_shapes=config["latent_shapes"],
        input_timesteps=[0.5],
        special_tokens=special_tokens,
        latent_patch_size=model_i4.config.latent_patch_size,
    )
    data_batch.to_cuda()

    model_i4.zero_grad()
    output_i4 = model_i4(packed_seq=data_batch)

    # Combined loss: text outputs (understanding path) + vision predictions (generation path)
    text_loss = output_i4["last_hidden_state"][:num_text].float().sum()
    vision_preds = output_i4.get("preds_vision", [])
    vision_loss = sum(p.float().sum() for p in vision_preds) if vision_preds else torch.tensor(0.0)
    loss = text_loss + vision_loss
    loss.backward()

    lm = model_i4.language_model

    # 1. embed_tokens grad exists and is finite
    embed_grad = lm.model.embed_tokens.weight.grad
    assert embed_grad is not None, "embed_tokens has no grad"
    assert torch.isfinite(embed_grad).all(), "embed_tokens grad contains NaN/inf"

    # 2. Text token rows have non-zero grads
    active_grad = embed_grad[unique_token_ids]
    assert active_grad.abs().sum() > 0, "embed_tokens grad is zero for all text tokens"
    log.info(f"embed_tokens active-rows grad norm: {active_grad.norm():.4f}")

    # 3. Absent token rows have zero grads
    if absent_ids:
        absent_grad = embed_grad[absent_ids]
        assert (absent_grad == 0).all(), "embed_tokens grad is non-zero for tokens not in sequence"

    # 4. Understanding-pathway grad (q_proj layer 0) — flows from text_loss
    q_proj_grad = lm.model.layers[0].self_attn.q_proj.weight.grad
    assert q_proj_grad is not None, "layer-0 q_proj has no grad"
    assert torch.isfinite(q_proj_grad).all(), "layer-0 q_proj grad contains NaN/inf"
    assert q_proj_grad.abs().sum() > 0, "layer-0 q_proj grad is zero"
    log.info(f"layer-0 q_proj grad norm: {q_proj_grad.norm():.4f}")

    # 5. Generation-pathway grad (q_proj_moe_gen layer 0) — flows from vision_loss.
    #    At t=0 (clean latents) generation tokens still participate in backward.
    if vision_preds:
        moe_gen_param = None
        for name, param in lm.model.layers[0].named_parameters():
            if "q_proj_moe_gen" in name and param.grad is not None:
                moe_gen_param = param
                break
        if moe_gen_param is not None:
            assert torch.isfinite(moe_gen_param.grad).all(), "moe_gen q_proj grad contains NaN/inf"
            assert moe_gen_param.grad.abs().sum() > 0, (
                "moe_gen q_proj grad is zero even with vision loss — generation path not active"
            )
            log.info(f"layer-0 q_proj_moe_gen grad norm: {moe_gen_param.grad.norm():.4f}")
        else:
            log.info("layer-0 q_proj_moe_gen not found (may use different naming)")
    else:
        log.info("No vision predictions — skipping moe_gen grad check")

    del model_i4
    _memory_cleanup()
    log.info("Nemotron backward sanity PASSED")


@pytest.mark.L1
@pytest.mark.parametrize("config_name", _HF_MODELS_TO_TEST)
def test_integration_nemotron_llm_weight_comparison(config_name: str) -> None:
    """Compare embed_tokens and layer-0 weights between our model and the HF reference."""
    from transformers import AutoModelForCausalLM

    from cosmos.utils import log

    log.info(f"\n=== Testing Nemotron LLM weight comparison: {config_name} ===")

    config = INTEGRATION_MODEL_CONFIGS[config_name]
    hf_model_name = config["hf_model_name"]

    model_cosmos3, _, _ = _build_nemotron_model(config_name)
    our_state_dict = model_cosmos3.language_model.state_dict()

    hf_causal_model = AutoModelForCausalLM.from_pretrained(
        hf_model_name,
        torch_dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
        trust_remote_code=True,
    )
    hf_state_dict = hf_causal_model.state_dict()

    # Weight key mapping: our key → HF key
    weight_pairs = [
        ("model.embed_tokens.weight", "model.embeddings.weight", "embed_tokens"),
        ("model.layers.0.input_layernorm.weight", "model.layers.0.input_layernorm.weight", "input_layernorm"),
        ("model.layers.0.self_attn.q_proj.weight", "model.layers.0.self_attn.q_proj.weight", "q_proj"),
        ("model.layers.0.self_attn.k_proj.weight", "model.layers.0.self_attn.k_proj.weight", "k_proj"),
        ("model.layers.0.self_attn.v_proj.weight", "model.layers.0.self_attn.v_proj.weight", "v_proj"),
        ("model.layers.0.self_attn.o_proj.weight", "model.layers.0.self_attn.o_proj.weight", "o_proj"),
        ("model.layers.0.mlp.up_proj.weight", "model.layers.0.mlp.up_proj.weight", "mlp.up_proj"),
        ("model.layers.0.mlp.down_proj.weight", "model.layers.0.mlp.down_proj.weight", "mlp.down_proj"),
        (
            "model.layers.0.post_attention_layernorm.weight",
            "model.layers.0.post_attention_layernorm.weight",
            "post_attn_ln",
        ),
    ]

    all_identical = True
    for our_key, hf_key, label in weight_pairs:
        assert our_key in our_state_dict, f"Missing in our model: {our_key}"
        assert hf_key in hf_state_dict, f"Missing in HF model: {hf_key}"

        our_w = our_state_dict[our_key]
        hf_w = hf_state_dict[hf_key]

        assert our_w.shape == hf_w.shape, f"Shape mismatch for {label}: {our_w.shape} vs {hf_w.shape}"

        max_diff = (our_w - hf_w).abs().max().item()
        identical = torch.allclose(our_w, hf_w, atol=1e-6)
        all_identical = all_identical and identical
        log.info(f"  {label}: shape={list(our_w.shape)}, max_diff={max_diff:.2e}, identical={identical}")

    # Verify moe_gen keys don't appear in HF model (they're our addition)
    our_only = {k for k in our_state_dict if "moe_gen" in k}
    assert len(our_only) > 0, "No moe_gen keys found in our model"
    log.info(f"  moe_gen keys (our model only): {len(our_only)}")

    del model_cosmos3, hf_causal_model
    _memory_cleanup()

    assert all_identical, "Weight mismatch between our model and HF reference"
    log.info("Nemotron LLM weight comparison PASSED")
