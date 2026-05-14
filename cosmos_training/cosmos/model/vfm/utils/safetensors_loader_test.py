# -----------------------------------------------------------------------------
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# -----------------------------------------------------------------------------
"""
Unit tests for safetensors_loader helpers and load_vlm_model.

pytest cosmos/model/vfm/utils/safetensors_loader_test.py -v
"""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch
from safetensors.torch import save_file

from cosmos.model.vfm.utils.safetensors_loader import (
    MultiRankCheckpointLoader,
    _build_model_key_by_tail,
    _get_dp_shard_mesh,
    _is_moe_vlm,
    _make_name_converter,
    _shard_first_dim,
    load_vlm_model,
)


class _StubConfig:
    """Minimal stand-in for an HF config object."""

    def __init__(self, **kwargs: Any):
        for k, v in kwargs.items():
            setattr(self, k, v)


class _StubModel(torch.nn.Module):
    """A torch module with dotted-name parameters so load_vlm_model can
    treat it like an HF model. Non-identifier dots in names prevent using
    register_parameter; override state_dict() instead."""

    def __init__(self, param_shapes: dict[str, tuple[int, ...]], config: _StubConfig):
        super().__init__()
        self._params: dict[str, torch.nn.Parameter] = {}
        for name, shape in param_shapes.items():
            p = torch.nn.Parameter(torch.zeros(shape), requires_grad=False)
            self._params[name] = p
        self.config = config

    def state_dict(self, *args, **kwargs):
        return dict(self._params)


def _make_safetensors(tmp_path: Path, tensors: dict[str, torch.Tensor]) -> Path:
    """Write ``tensors`` into a single model.safetensors under tmp_path.

    Creates ``tmp_path`` and the ``ckpt`` subdirectory if missing (supports
    callers that pass a nested tmp_path like ``tmp_path / "with"``).
    """
    ckpt_dir = tmp_path / "ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(ckpt_dir / "model.safetensors"))
    return ckpt_dir



#
# The single-rank CPU fallback is reached by passing ``parallel_dims=None``
# (the documented escape hatch — see ``load_vlm_model`` docstring). All
# end-to-end tests below use that path; multi-rank behavior is covered in
# the GPU-marked tests under ``projects/cosmos3/vfm/models/mot/``.
#
# Do NOT introduce a "fake" ``ParallelDims`` MagicMock fixture for this
# fallback: ``MagicMock.__getitem__`` returns another MagicMock rather than
# raising, which silently bypasses the loader's real None-handling path.


@pytest.mark.L0
@pytest.mark.CPU
def test_shard_first_dim_even_split():
    """Even split: each rank gets row_size / world_size rows."""
    tensor = torch.arange(16, dtype=torch.float32).view(8, 2)
    shard_r0 = _shard_first_dim(tensor, world_size=4, rank=0)
    shard_r3 = _shard_first_dim(tensor, world_size=4, rank=3)

    assert shard_r0.shape == (2, 2)
    assert torch.equal(shard_r0, tensor[0:2])
    assert shard_r3.shape == (2, 2)
    assert torch.equal(shard_r3, tensor[6:8])


@pytest.mark.L0
@pytest.mark.CPU
def test_shard_first_dim_uneven_split_last_rank_smaller():
    """Uneven split: avg = ceil(D/N); last rank gets remainder rows."""
    # D=17, N=4 → avg=5, rank 0..2 get 5 rows each, rank 3 gets 2 rows.
    tensor = torch.arange(17 * 2, dtype=torch.float32).view(17, 2)

    shard_r0 = _shard_first_dim(tensor, world_size=4, rank=0)
    shard_r2 = _shard_first_dim(tensor, world_size=4, rank=2)
    shard_r3 = _shard_first_dim(tensor, world_size=4, rank=3)

    assert shard_r0.shape == (5, 2)
    assert torch.equal(shard_r0, tensor[0:5])
    assert shard_r2.shape == (5, 2)
    assert torch.equal(shard_r2, tensor[10:15])
    assert shard_r3.shape == (2, 2)  # remainder
    assert torch.equal(shard_r3, tensor[15:17])


@pytest.mark.L0
@pytest.mark.CPU
def test_shard_first_dim_single_rank_returns_full_tensor():
    tensor = torch.arange(6, dtype=torch.float32).view(3, 2)
    shard = _shard_first_dim(tensor, world_size=1, rank=0)
    assert torch.equal(shard, tensor)


# --- MultiRankCheckpointLoader.__init__ ------------------------------------


@pytest.mark.L0
@pytest.mark.CPU
def test_loader_init_with_mesh_reads_group_rank_size_directly():
    """The class takes a 1-D ``dp_shard_mesh`` and reads (group, rank, size)
    straight off it — no subscripting, no flattened-name lookup."""
    mesh = MagicMock()
    mesh.get_group.return_value = "fake_group"
    mesh.get_local_rank.return_value = 2
    mesh.size.return_value = 4

    loader = MultiRankCheckpointLoader(mesh)

    mesh.get_group.assert_called_once_with()
    mesh.get_local_rank.assert_called_once_with()
    mesh.size.assert_called_once_with()
    # The class must NOT subscript the mesh anymore (callers pass the 1-D
    # sub-mesh directly via ``_get_dp_shard_mesh``).
    mesh.__getitem__.assert_not_called()
    assert loader.group == "fake_group"
    assert loader.rank == 2
    assert loader.world_size == 4


@pytest.mark.L0
@pytest.mark.CPU
def test_loader_init_handles_none_mesh():
    """``None`` is the documented single-rank fallback (no distributed setup)."""
    loader = MultiRankCheckpointLoader(None)
    assert loader.group is None
    assert loader.rank == 0
    assert loader.world_size == 1


# --- _get_dp_shard_mesh helper ---------------------------------------------


@pytest.mark.L0
@pytest.mark.CPU
def test_get_dp_shard_mesh_returns_none_for_none_parallel_dims():
    """No ParallelDims → no mesh; matches the loader's single-rank fallback."""
    assert _get_dp_shard_mesh(None) is None


@pytest.mark.L0
@pytest.mark.CPU
def test_get_dp_shard_mesh_returns_none_when_dp_shard_disabled():
    """``dp_shard <= 1`` is treated as no FSDP shard axis → None."""
    pd = MagicMock()
    pd.dp_shard_enabled = False
    pd.dp_shard_mesh = MagicMock()  # populated, but should be ignored
    assert _get_dp_shard_mesh(pd) is None


@pytest.mark.L0
@pytest.mark.CPU
def test_get_dp_shard_mesh_returns_mesh_when_enabled():
    """``dp_shard > 1`` → forward the ParallelDims property unchanged."""
    pd = MagicMock()
    pd.dp_shard_enabled = True
    sentinel = MagicMock(name="dp_shard_mesh")
    pd.dp_shard_mesh = sentinel
    assert _get_dp_shard_mesh(pd) is sentinel


# --- suffix-lookup helpers --------------------------------------------------


@pytest.mark.L0
@pytest.mark.CPU
def test_build_model_key_by_tail_longest_prefix_wins():
    """Longer prefixes take precedence — shortest tail wins."""
    state_dict_keys = [
        "model.embed_tokens.weight",
        "model.language_model.layers.0.input_layernorm.weight",
        "visual.blocks.0.norm.weight",
        "lm_head.weight",
    ]
    table = _build_model_key_by_tail({k: None for k in state_dict_keys})

    # "model.language_model." prefix strips more than "model.", so the tail
    # for the layer 0 entry is "layers.0.input_layernorm.weight" (shortest).
    assert table["layers.0.input_layernorm.weight"] == ("model.language_model.layers.0.input_layernorm.weight")
    # "model." strips for embed_tokens
    assert table["embed_tokens.weight"] == "model.embed_tokens.weight"
    # No prefix matches — empty prefix → tail == self
    assert table["visual.blocks.0.norm.weight"] == "visual.blocks.0.norm.weight"
    assert table["lm_head.weight"] == "lm_head.weight"


@pytest.mark.L0
@pytest.mark.CPU
def test_name_converter_direct_match_wins():
    """If the checkpoint key is already in the state dict, return unchanged."""
    state_dict = {"model.embed_tokens.weight": None}
    convert = _make_name_converter(state_dict, hf_conv_map=None)
    assert convert("model.embed_tokens.weight") == "model.embed_tokens.weight"


@pytest.mark.L0
@pytest.mark.CPU
def test_name_converter_suffix_lookup_across_prefixes():
    """Suffix lookup handles HF-native, model.-prefixed, and language_model.-prefixed layouts."""
    state_dict = {
        "model.language_model.layers.0.self_attn.q_proj.weight": None,
        "visual.blocks.0.norm.weight": None,
    }
    convert = _make_name_converter(state_dict, hf_conv_map=None)

    # Checkpoint with "model." prefix
    assert (
        convert("model.language_model.layers.0.self_attn.q_proj.weight")
        == "model.language_model.layers.0.self_attn.q_proj.weight"
    )
    # Checkpoint with "language_model." prefix (no "model." outer wrapper)
    assert (
        convert("language_model.layers.0.self_attn.q_proj.weight")
        == "model.language_model.layers.0.self_attn.q_proj.weight"
    )
    # HF-native visual checkpoint key
    assert convert("visual.blocks.0.norm.weight") == "visual.blocks.0.norm.weight"


@pytest.mark.L0
@pytest.mark.CPU
def test_name_converter_hf_conv_map_takes_precedence():
    """When _checkpoint_conversion_mapping is set, use its regex mapping."""
    state_dict = {"model.new_name.weight": None}
    convert = _make_name_converter(
        state_dict,
        hf_conv_map={r"^old_name\.": "model.new_name."},
    )
    assert convert("old_name.weight") == "model.new_name.weight"
    # Unrelated keys pass through
    assert convert("foo.bar") == "foo.bar"


@pytest.mark.L0
@pytest.mark.CPU
def test_name_converter_unmappable_key_returns_unchanged():
    """If no rule matches, return the input unchanged (caller handles miss)."""
    state_dict = {"model.embed_tokens.weight": None}
    convert = _make_name_converter(state_dict, hf_conv_map=None)
    assert convert("totally.unknown.key") == "totally.unknown.key"


# --- _is_moe_vlm detection --------------------------------------------------


@pytest.mark.L0
@pytest.mark.CPU
def test_is_moe_vlm_detects_num_experts_in_text_config():
    model = MagicMock()
    model.config.text_config.num_experts = 128
    model.state_dict.return_value = {}
    assert _is_moe_vlm(model) is True


@pytest.mark.L0
@pytest.mark.CPU
def test_is_moe_vlm_detects_num_local_experts():
    model = MagicMock()
    del model.config.text_config  # no text_config → fallback to top-level config
    model.config.num_experts = None
    model.config.num_local_experts = 8
    model.state_dict.return_value = {}
    assert _is_moe_vlm(model) is True


@pytest.mark.L0
@pytest.mark.CPU
def test_is_moe_vlm_detects_mlp_experts_in_state_dict():
    model = MagicMock()
    model.config.text_config.num_experts = None
    model.config.text_config.num_local_experts = None
    model.state_dict.return_value = {
        "model.layers.0.mlp.experts.0.gate_proj.weight": torch.empty(0),
    }
    assert _is_moe_vlm(model) is True


@pytest.mark.L0
@pytest.mark.CPU
def test_is_moe_vlm_false_for_dense_model():
    model = MagicMock()
    model.config.text_config.num_experts = None
    model.config.text_config.num_local_experts = None
    model.state_dict.return_value = {
        "model.layers.0.mlp.gate_proj.weight": torch.empty(0),  # dense, no .experts.
    }
    assert _is_moe_vlm(model) is False


# --- load_vlm_model end-to-end (single-rank, CPU) ---------------------------


@pytest.mark.L0
@pytest.mark.CPU
def test_load_vlm_model_copies_matching_keys(tmp_path):
    """Checkpoint keys that resolve via suffix lookup get copied into model params.

    Model state_dict mirrors Qwen3-VLForConditionalGeneration layout
    (``model.model.<inner>`` for the language model inside the outer wrapper).
    Checkpoint uses the HF-native layout — matches verbatim for both paths."""
    cfg = _StubConfig(
        text_config=_StubConfig(num_experts=None, num_local_experts=None),
        tie_word_embeddings=False,
    )
    model = _StubModel(
        param_shapes={
            "model.model.embed_tokens.weight": (4, 2),
            "model.model.layers.0.input_layernorm.weight": (4,),
            "lm_head.weight": (4, 2),
        },
        config=cfg,
    )
    expected_embed = torch.arange(8, dtype=torch.float32).view(4, 2)
    expected_norm = torch.arange(4, dtype=torch.float32)
    expected_head = torch.arange(8, 16, dtype=torch.float32).view(4, 2)
    ckpt = {
        "model.model.embed_tokens.weight": expected_embed,
        "model.model.layers.0.input_layernorm.weight": expected_norm,
        "lm_head.weight": expected_head,
    }
    ckpt_dir = _make_safetensors(tmp_path, ckpt)

    keys_loaded = load_vlm_model(
        model=model,
        checkpoint_path=str(ckpt_dir),
        credential_path=None,
        parallel_dims=None,
    )

    assert "model.model.embed_tokens.weight" in keys_loaded
    assert "lm_head.weight" in keys_loaded
    assert torch.equal(model._params["model.model.embed_tokens.weight"].data, expected_embed)
    assert torch.equal(model._params["model.model.layers.0.input_layernorm.weight"].data, expected_norm)
    assert torch.equal(model._params["lm_head.weight"].data, expected_head)


@pytest.mark.L0
@pytest.mark.CPU
def test_load_vlm_model_tolerates_missing_lm_head_when_tied(tmp_path):
    """tie_word_embeddings=True + no lm_head in checkpoint → no error,
    AND lm_head.weight.data must equal embed_tokens.weight.data after load
    (because the caller pre-tied them via shared storage)."""
    cfg = _StubConfig(
        text_config=_StubConfig(num_experts=None, num_local_experts=None),
        tie_word_embeddings=True,
    )
    model = _StubModel(
        param_shapes={
            "model.embed_tokens.weight": (4, 2),
            "lm_head.weight": (4, 2),
        },
        config=cfg,
    )
    # Tie: make lm_head.weight SHARE STORAGE with embed_tokens.weight.  This
    # is what HFModel.tie_embeddings() does at the live call site.
    model._params["lm_head.weight"] = model._params["model.embed_tokens.weight"]

    expected = torch.arange(8, dtype=torch.float32).view(4, 2)
    ckpt = {
        "model.embed_tokens.weight": expected,
        # deliberately NO lm_head.weight in the checkpoint — tied.
    }
    ckpt_dir = _make_safetensors(tmp_path, ckpt)

    # Must not raise on the missing lm_head key.
    load_vlm_model(
        model=model,
        checkpoint_path=str(ckpt_dir),
        credential_path=None,
        parallel_dims=None,
    )

    # Tie invariant: both parameters now reflect the checkpoint's embed_tokens.
    assert torch.equal(model._params["model.embed_tokens.weight"].data, expected)
    assert torch.equal(model._params["lm_head.weight"].data, expected)
    # And they must STILL share storage — the load must not have rebound
    # lm_head to a fresh tensor.
    assert model._params["lm_head.weight"].data.data_ptr() == model._params["model.embed_tokens.weight"].data.data_ptr()


@pytest.mark.L0
@pytest.mark.CPU
def test_load_vlm_model_rejects_moe(tmp_path):
    """MoE VLMs raise NotImplementedError — MoE is not yet supported."""
    cfg = _StubConfig(
        text_config=_StubConfig(num_experts=128, num_local_experts=None),
        tie_word_embeddings=False,
    )
    model = _StubModel(param_shapes={"model.embed_tokens.weight": (4, 2)}, config=cfg)
    expected = torch.arange(8, dtype=torch.float32).view(4, 2)
    ckpt_dir = _make_safetensors(tmp_path, {"model.embed_tokens.weight": expected})

    with pytest.raises(NotImplementedError, match="MoE VLMs"):
        load_vlm_model(
            model=model,
            checkpoint_path=str(ckpt_dir),
            credential_path=None,
            parallel_dims=None,
        )


@pytest.mark.L0
@pytest.mark.CPU
def test_load_vlm_model_cpu_localview_stays_off_gpu(tmp_path):
    """When model params live on CPU (fsdp_offload mode), the loader MUST
    keep the broadcast/iterate device on CPU — it must not drag the full
    tensor through CUDA.  This single-process test cannot directly observe
    the broadcast device, but it CAN verify that a load succeeds end-to-end
    with CPU params and that the resulting shard lands on CPU."""
    cfg = _StubConfig(
        text_config=_StubConfig(num_experts=None, num_local_experts=None),
        tie_word_embeddings=False,
    )
    model = _StubModel(
        param_shapes={"model.model.embed_tokens.weight": (4, 2)},
        config=cfg,
    )
    # Stub params are already on CPU (torch.zeros default) — this mirrors
    # the fsdp_offload materialization path that keeps FSDP local_views on CPU.
    assert model._params["model.model.embed_tokens.weight"].device.type == "cpu"

    expected = torch.arange(8, dtype=torch.float32).view(4, 2)
    ckpt_dir = _make_safetensors(tmp_path, {"model.model.embed_tokens.weight": expected})

    load_vlm_model(
        model=model,
        checkpoint_path=str(ckpt_dir),
        credential_path=None,
        parallel_dims=None,
    )

    # Value matches AND device stayed CPU.
    loaded = model._params["model.model.embed_tokens.weight"]
    assert loaded.device.type == "cpu"
    assert torch.equal(loaded.data, expected)


@pytest.mark.L0
@pytest.mark.CPU
def test_load_vlm_model_skip_patterns_on_model_keys(tmp_path):
    """Skip patterns match MODEL keys (post-name_converter), and skipped
    MODEL keys are tolerated by the completeness check in both branches."""
    cfg = _StubConfig(
        text_config=_StubConfig(num_experts=None, num_local_experts=None),
        tie_word_embeddings=False,
    )
    model = _StubModel(
        param_shapes={
            "model.embed_tokens.weight": (2, 2),
            # A buffer the model has but should be initialized in-place,
            # not loaded from ckpt.  Modeled as a model state-dict entry.
            "vision_model.radio_model.summary_idxs": (8,),
        },
        config=cfg,
    )
    skip = [r"vision_model\.radio_model\.summary_idxs"]

    # Subtest 1: ckpt has the key → Phase 5 branch.
    ckpt_with = {
        "model.embed_tokens.weight": torch.ones(2, 2),
        "vision_model.radio_model.summary_idxs": torch.arange(8, dtype=torch.float32),
    }
    ckpt_dir = _make_safetensors(tmp_path / "with", ckpt_with)
    keys_loaded = load_vlm_model(
        model=model,
        checkpoint_path=str(ckpt_dir),
        credential_path=None,
        parallel_dims=None,
        tensor_names_to_skip=skip,
    )
    assert "model.embed_tokens.weight" in keys_loaded
    assert "vision_model.radio_model.summary_idxs" not in keys_loaded

    # Subtest 2: ckpt omits the key → Phase 6 branch must tolerate.
    ckpt_without = {"model.embed_tokens.weight": torch.ones(2, 2)}
    ckpt_dir2 = _make_safetensors(tmp_path / "without", ckpt_without)
    keys_loaded = load_vlm_model(
        model=model,
        checkpoint_path=str(ckpt_dir2),
        credential_path=None,
        parallel_dims=None,
        tensor_names_to_skip=skip,
    )
    assert "model.embed_tokens.weight" in keys_loaded


# ──────────────────────────────────────────────────────────────────────────────
# extra_skip_patterns — overlay-friendly skip list for pretrain_weights_path_llm
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.L0
@pytest.mark.CPU
def test_load_vlm_model_extra_skip_patterns_tolerates_declared_missing(tmp_path):
    """extra_skip_patterns: a regex matching a model key that is absent from
    the checkpoint is demoted from ValueError to silent skip. Present keys
    still get their values loaded."""
    config = _StubConfig(
        model_type="qwen3_vl",
        vision_config=object(),  # truthy — bypass LLM-only code paths
        tie_word_embeddings=False,
    )
    model = _StubModel(
        {
            "model.language_model.layers.0.weight": (4, 4),
            "model.visual.blocks.0.weight": (4, 4),
        },
        config,
    )
    sentinel = torch.full((4, 4), 7.0)
    with torch.no_grad():
        model._params["model.visual.blocks.0.weight"].data.copy_(sentinel)

    # Checkpoint with only the language_model key (what an LLM overlay looks like).
    llm_val = torch.full((4, 4), 3.0)
    ckpt_dir = _make_safetensors(tmp_path / "overlay", {"model.language_model.layers.0.weight": llm_val})

    keys_loaded = load_vlm_model(
        model=model,
        checkpoint_path=str(ckpt_dir),
        credential_path=None,
        parallel_dims=None,
        extra_skip_patterns=[r"^model\.visual\..*"],
    )

    # Language-model param was overlaid.
    assert torch.allclose(model._params["model.language_model.layers.0.weight"].data, llm_val)
    # Visual param was left untouched (sentinel preserved).
    assert torch.allclose(model._params["model.visual.blocks.0.weight"].data, sentinel)
    # keys_loaded reports only the language-model key.
    assert "model.language_model.layers.0.weight" in keys_loaded
    assert "model.visual.blocks.0.weight" not in keys_loaded


@pytest.mark.L0
@pytest.mark.CPU
def test_load_vlm_model_extra_skip_patterns_still_raises_on_unexpected_missing(tmp_path):
    """extra_skip_patterns only tolerates keys matched by a skip pattern.
    A missing key OUTSIDE the skip list still raises ValueError."""
    config = _StubConfig(
        model_type="qwen3_vl",
        vision_config=object(),
        tie_word_embeddings=False,
    )
    model = _StubModel(
        {
            "model.language_model.layers.0.weight": (4, 4),
            "model.visual.blocks.0.weight": (4, 4),
        },
        config,
    )
    # Checkpoint is empty of expected keys — visual is tolerated, LM is not.
    ckpt_dir = _make_safetensors(tmp_path / "empty", {"unrelated.weight": torch.zeros(1)})

    with pytest.raises(ValueError, match=r"required model parameter"):
        load_vlm_model(
            model=model,
            checkpoint_path=str(ckpt_dir),
            credential_path=None,
            parallel_dims=None,
            extra_skip_patterns=[r"^model\.visual\..*"],
        )


@pytest.mark.L0
@pytest.mark.CPU
def test_load_vlm_model_tail_matches_raw_llm_keys_to_language_model(tmp_path):
    """REGRESSION GUARD for the v2 simplification: raw LLM checkpoint keys
    (``model.layers.*``, ``model.embed_tokens.weight``, ``model.norm.weight``,
    ``lm_head.weight``) must resolve to the VLM's ``model.language_model.*``
    / ``lm_head.weight`` state-dict keys via the existing ``_VLM_KEY_PREFIXES``
    tail matcher. If this ever breaks (e.g. a future edit removes the
    ``"model."`` entry from ``_VLM_KEY_PREFIXES`` or changes the tail-match
    logic), the production pretrain_weights_path_llm overlay silently stops
    working — this L0 catches it before smoke testing."""
    config = _StubConfig(
        model_type="qwen3_vl",
        vision_config=object(),
        tie_word_embeddings=False,
    )
    # Model keys as the VLM actually exposes them.
    model = _StubModel(
        {
            "model.language_model.layers.0.self_attn.q_proj.weight": (4, 4),
            "model.language_model.embed_tokens.weight": (4, 4),
            "model.language_model.norm.weight": (4,),
            "lm_head.weight": (4, 4),
            "model.visual.blocks.0.weight": (4, 4),
        },
        config,
    )
    # Sentinel on the visual param — must not be touched by the overlay.
    sentinel = torch.full((4, 4), 9.0)
    with torch.no_grad():
        model._params["model.visual.blocks.0.weight"].data.copy_(sentinel)

    # Raw LLM-style checkpoint keys (no ``language_model.`` prefix).
    llm_vals = {
        "model.layers.0.self_attn.q_proj.weight": torch.full((4, 4), 1.0),
        "model.embed_tokens.weight": torch.full((4, 4), 2.0),
        "model.norm.weight": torch.full((4,), 3.0),
        "lm_head.weight": torch.full((4, 4), 4.0),
    }
    ckpt_dir = _make_safetensors(tmp_path / "raw_llm", llm_vals)

    keys_loaded = load_vlm_model(
        model=model,
        checkpoint_path=str(ckpt_dir),
        credential_path=None,
        parallel_dims=None,
        extra_skip_patterns=[r"^model\.visual\..*"],
    )

    # Each raw LLM key landed on the expected VLM key.
    assert torch.allclose(
        model._params["model.language_model.layers.0.self_attn.q_proj.weight"].data,
        llm_vals["model.layers.0.self_attn.q_proj.weight"],
    )
    assert torch.allclose(
        model._params["model.language_model.embed_tokens.weight"].data,
        llm_vals["model.embed_tokens.weight"],
    )
    assert torch.allclose(
        model._params["model.language_model.norm.weight"].data,
        llm_vals["model.norm.weight"],
    )
    assert torch.allclose(
        model._params["lm_head.weight"].data,
        llm_vals["lm_head.weight"],
    )
    # Visual param still untouched.
    assert torch.allclose(model._params["model.visual.blocks.0.weight"].data, sentinel)
    # keys_loaded reports the resolved (VLM) names.
    assert "model.language_model.layers.0.self_attn.q_proj.weight" in keys_loaded
    assert "model.language_model.embed_tokens.weight" in keys_loaded
    assert "model.language_model.norm.weight" in keys_loaded
    assert "lm_head.weight" in keys_loaded


@pytest.mark.L0
@pytest.mark.CPU
def test_load_vlm_model_merges_tensor_names_to_skip_and_extra_skip_patterns(tmp_path):
    """Both kwargs coexist: model keys matched by EITHER list are tolerated in
    Phase-6 completeness. Catches the case where a caller like HFModel.load_weights
    passes model-type fixed skips (``tensor_names_to_skip``) AND overlay-specific
    skips (``extra_skip_patterns``) at the same time — the lists must be
    concatenated, not one overriding the other.
    """
    config = _StubConfig(
        model_type="qwen3_vl",
        vision_config=object(),
        tie_word_embeddings=False,
    )
    model = _StubModel(
        {
            "model.language_model.layers.0.weight": (4, 4),  # will be loaded
            "model.visual.blocks.0.weight": (4, 4),  # tolerated via extra_skip_patterns
            "fixed_skip.weight": (4, 4),  # tolerated via tensor_names_to_skip
        },
        config,
    )
    ckpt_dir = _make_safetensors(
        tmp_path / "merged",
        {"model.language_model.layers.0.weight": torch.full((4, 4), 5.0)},
    )

    keys_loaded = load_vlm_model(
        model=model,
        checkpoint_path=str(ckpt_dir),
        credential_path=None,
        parallel_dims=None,
        tensor_names_to_skip=[r"^fixed_skip\..*"],
        extra_skip_patterns=[r"^model\.visual\..*"],
    )

    assert "model.language_model.layers.0.weight" in keys_loaded
    # Neither of the two skipped keys was loaded (checkpoint didn't have them).
    assert "model.visual.blocks.0.weight" not in keys_loaded
    assert "fixed_skip.weight" not in keys_loaded


@pytest.mark.L0
@pytest.mark.CPU
def test_load_vlm_model_qwen2_5_bare_visual_keys_overlay(tmp_path):
    """End-to-end overlay on a Qwen2.5-VL-shaped state dict (BARE ``visual.*``).

    Qwen2_5_VLForConditionalGeneration puts the visual encoder at the top level
    (no leading ``model.``), while the LLM lives under ``model.*`` — see
    projects/cosmos3/vlm/scripts/convert_qwenvl_ckpt.py:101-118.

    Exercises BOTH Phase-5 (skip present visual keys in the checkpoint — sentinel
    visual param survives) AND Phase-6 (tolerate absent visual keys — merger is in
    the model but not in the checkpoint), using the actual pattern emitted by
    ``_get_overlay_config("qwen2_5_vl")`` so the test and the helper cannot drift.
    """
    # Import the production helper pattern instead of retyping — guarantees drift
    # between test and source is impossible.
    from cosmos.model.vfm.vlm_model import _get_overlay_config

    overlay_skip_patterns, _ = _get_overlay_config("qwen2_5_vl")

    config = _StubConfig(
        model_type="qwen2_5_vl",
        vision_config=object(),
        tie_word_embeddings=False,
    )
    model = _StubModel(
        {
            "model.layers.0.weight": (4, 4),  # LLM param, will be loaded
            "visual.blocks.0.weight": (4, 4),  # BARE visual: present in ckpt → Phase-5 skip
            "visual.merger.mlp.0.weight": (4, 4),  # BARE merger: absent from ckpt → Phase-6 tolerate
            "lm_head.weight": (4, 4),  # LM head, will be loaded
        },
        config,
    )
    # Seed the model's visual param with a sentinel so we can assert it survives
    # the overlay (i.e., Phase-5 actually skipped the checkpoint tensor for this key).
    sentinel = torch.full((4, 4), 42.0)
    model._params["visual.blocks.0.weight"].data.copy_(sentinel)

    ckpt_dir = _make_safetensors(
        tmp_path / "qwen25_overlay",
        {
            "model.layers.0.weight": torch.full((4, 4), 3.0),
            "lm_head.weight": torch.full((4, 4), 7.0),
            # Present bare visual tensor — Phase-5 must SKIP this, not copy it over.
            "visual.blocks.0.weight": torch.full((4, 4), -1.0),
        },
    )

    keys_loaded = load_vlm_model(
        model=model,
        checkpoint_path=str(ckpt_dir),
        credential_path=None,
        parallel_dims=None,
        extra_skip_patterns=overlay_skip_patterns,
    )

    assert "model.layers.0.weight" in keys_loaded
    assert "lm_head.weight" in keys_loaded
    # Bare visual keys: Phase-5 skip for the present one, Phase-6 tolerance for the absent one.
    assert "visual.blocks.0.weight" not in keys_loaded
    assert "visual.merger.mlp.0.weight" not in keys_loaded
    # Phase-5 actually ran: the sentinel survived, proving the ckpt's -1.0 tensor
    # did NOT overwrite the visual param.
    assert torch.equal(model._params["visual.blocks.0.weight"].data, sentinel)
    # Non-visual keys that WERE in the checkpoint made it through.
    assert torch.equal(model._params["model.layers.0.weight"].data, torch.full((4, 4), 3.0))
    assert torch.equal(model._params["lm_head.weight"].data, torch.full((4, 4), 7.0))
