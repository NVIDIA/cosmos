import random
from typing import cast

import pytest
import torch

import cosmos.model.vfm.mot.attention as attention
from cosmos.model.attention.natten import NATTEN_SUPPORTED
from cosmos.data.vfm.sequence_packing import (
    get_all_seq,
    get_gen_seq,
    get_und_seq,
    set_gen_seq,
    set_und_seq,
    zeros_like,
)
from cosmos.model.vfm.mot.attention import (
    build_packed_sequence,
)

MAX_SEQ_LEN = 24
SEQS_PER_BATCH = 4


def unwrap(fn):
    import torch.utils._pytree as pytree

    def unwrap_fn(a, s):
        args, kwargs = pytree.tree_unflatten(a, s)
        return fn(*args, **kwargs)

    return unwrap_fn


def wrap(fn):
    import torch.utils._pytree as pytree

    def wrap_fn(*args, **kwargs):
        a, s = pytree.tree_flatten((args, kwargs))
        return fn(a, s)

    return wrap_fn


def _test_attention_impls(
    impl_1: str,
    impl_2: str,
    atol_self: float = 1e-4,
    rtol_self: float = 0,
    atol_cmp: float = 1e-1,
    rtol_cmp: float = 0,
    atol_bwd_self: float = 1e-1,
    rtol_bwd_self: float = 0,
    atol_bwd_cmp: float = 1.5,
    rtol_bwd_cmp: float = 0,
):
    random.seed(42)
    torch.manual_seed(42)

    # Reset cache for every new test to avoid reusing cache from previous ones
    torch.compiler.reset()

    IMPL_TO_FN = {
        "flex": attention.block_flex_attention,
        "two_way": attention.two_way_attention,
        "three_way": attention.three_way_attention,
    }

    assert impl_1 in IMPL_TO_FN
    assert impl_2 in IMPL_TO_FN
    assert impl_1 != impl_2

    fn_1 = IMPL_TO_FN[impl_1]
    fn_2 = IMPL_TO_FN[impl_2]

    use_compile = True
    test_backward: bool = True
    device = torch.device("cuda")
    num_q_heads = 32
    num_kv_heads = 4
    head_dim = 128
    text_on_und_mode_only = True
    num_layers = 1

    # smaller seq length to expose off-by-one errors
    sample_lens = torch.randint(4, MAX_SEQ_LEN, (SEQS_PER_BATCH,), device=device, dtype=torch.int32)
    sample_lens = sample_lens.tolist()

    full_length = int(sum(sample_lens))

    # Generate `split_ids` with two splits per sample: always include 0, and a random int within range as intermediate for each sample in `sample_lens`.
    # packed_und_token_indexes takes the first split plust the first and last token of the second split.
    split_lens = []
    start = 0
    packed_und_token_indexes = []
    packed_gen_token_indexes = []
    position_ids = []
    attn_modes = ["causal", "full"] * len(sample_lens)
    token_shapes = []
    for length in sample_lens:
        assert length >= 4, f"sample_len must be >= 4, got {length}"

        und_extra = 1 if text_on_und_mode_only else 0
        gen_minus = 0 if text_on_und_mode_only else 1

        causal_len = int(torch.randint(1, length - 2 + und_extra, ()))
        split_lens.extend((causal_len, length - causal_len))

        und_len = causal_len if text_on_und_mode_only else causal_len + 1

        packed_und_token_indexes.extend(range(start, start + und_len))
        # generation part (latent noise)
        packed_gen_token_indexes.extend(range(start + und_len, start + length - gen_minus))
        if not text_on_und_mode_only:
            # final <IMGEND> token
            packed_und_token_indexes.append(start + length - 1)

        position_ids.extend(range(length))
        start += length

        token_shapes.append((1, length))

    real_len = sum(sample_lens)

    # Precompute LongTensor indices and common kwargs
    packed_und_idx_t = cast(torch.LongTensor, torch.tensor(packed_und_token_indexes, device=device, dtype=torch.long))
    packed_gen_idx_t = cast(torch.LongTensor, torch.tensor(packed_gen_token_indexes, device=device, dtype=torch.long))

    # Builders: return only the pack; retrieve the attention_meta explicitly when needed
    def _make_pack_decomposed(x, impl: str):
        return build_packed_sequence(
            impl,
            packed_sequence=x,
            attn_modes=attn_modes,
            split_lens=split_lens,
            sample_lens=sample_lens,
            packed_und_token_indexes=packed_und_idx_t,
            packed_gen_token_indexes=packed_gen_idx_t,
            num_heads=num_q_heads,
            head_dim=head_dim,
            num_layers=num_layers,
            token_shapes=token_shapes,
        )[0]

    def make_pack_two_way(x):
        return _make_pack_decomposed(x, "two_way")

    def make_pack_three_way(x):
        return _make_pack_decomposed(x, "three_way")

    def make_pack_flex(x):
        return build_packed_sequence(
            "flex",
            packed_sequence=x,
            attn_modes=attn_modes,
            split_lens=split_lens,
            sample_lens=sample_lens,
            packed_und_token_indexes=packed_und_idx_t,
            packed_gen_token_indexes=packed_gen_idx_t,
            num_heads=num_q_heads,
            head_dim=head_dim,
            num_layers=num_layers,
            token_shapes=token_shapes,
        )[0]

    IMPL_TO_MAKE_PACK = {
        "flex": make_pack_flex,
        "two_way": make_pack_two_way,
        "three_way": make_pack_three_way,
    }

    packed_und_token_indexes = torch.tensor(packed_und_token_indexes, device=device, dtype=torch.int32)
    packed_gen_token_indexes = torch.tensor(packed_gen_token_indexes, device=device, dtype=torch.int32)
    position_ids = torch.tensor(position_ids, device=device, dtype=torch.int32)

    packed_qkv11 = torch.randn(
        full_length,
        num_q_heads + 2 * num_kv_heads,
        head_dim,
        device=device,
        dtype=torch.bfloat16,
        requires_grad=test_backward,
    )
    packed_qkv12 = packed_qkv11.detach().clone().requires_grad_(test_backward)
    packed_qkv21 = packed_qkv11.detach().clone().requires_grad_(test_backward)
    packed_qkv22 = packed_qkv11.detach().clone().requires_grad_(test_backward)

    def split_qkv(qkv, make_pack):
        query = qkv[:, :num_q_heads, :]
        key = qkv[:, num_q_heads : num_q_heads + num_kv_heads, :]
        value = qkv[:, num_q_heads + num_kv_heads :, :]

        query_packed = make_pack(query.clone())
        key_packed = make_pack(key.clone())
        value_packed = make_pack(value.clone())

        if test_backward:
            # if we are running backward we cannot modify in place.
            query_packed2 = zeros_like(query_packed)
            key_packed2 = zeros_like(key_packed)
            value_packed2 = zeros_like(value_packed)

            set_gen_seq(query_packed2, get_gen_seq(query_packed))
            set_gen_seq(key_packed2, get_gen_seq(key_packed))
            set_gen_seq(value_packed2, get_gen_seq(value_packed))
        else:
            query_packed2 = query_packed
            key_packed2 = key_packed
            value_packed2 = value_packed

        # tweak non-causal tokens to see if they are properly masked
        set_und_seq(query_packed2, 2 * get_und_seq(query_packed))
        set_und_seq(key_packed2, 2 * get_und_seq(key_packed))
        set_und_seq(value_packed2, 2 * get_und_seq(value_packed))

        return query_packed2, key_packed2, value_packed2

    make_pack_1 = IMPL_TO_MAKE_PACK[impl_1]
    make_pack_2 = IMPL_TO_MAKE_PACK[impl_2]

    query_factored_1, key_factored_1, value_factored_1 = split_qkv(packed_qkv11, make_pack_1)
    query_factored_2, key_factored_2, value_factored_2 = split_qkv(packed_qkv21, make_pack_1)

    query_joint_1, key_joint_1, value_joint_1 = split_qkv(packed_qkv12, make_pack_2)
    query_joint_2, key_joint_2, value_joint_2 = split_qkv(packed_qkv22, make_pack_2)

    def compile(x):
        if use_compile:
            return torch.compile(x, fullgraph=True, backend="eager")
        else:
            return x

    class AttentionWrapper(torch.nn.Module):
        def __init__(self, attention_func, sdpa_func=None):
            super().__init__()
            self.attention_func = attention_func
            self.sdpa_func = sdpa_func

        def forward(self, *args, **kwargs):
            if self.sdpa_func is not None:
                kwargs["sdpa_func"] = self.sdpa_func
            return self.attention_func(*args, **kwargs)


    # between queries and key/values, since this enables the "don't care" mask, which enables
    # more attention backends in I4 attention.
    if query_factored_1["_causal_seq_offsets"].equal(key_factored_1["_causal_seq_offsets"]) and query_factored_1[
        "_causal_seq_offsets"
    ].equal(value_factored_1["_causal_seq_offsets"]):
        key_factored_1["_causal_seq_offsets"] = query_factored_1["_causal_seq_offsets"]
        value_factored_1["_causal_seq_offsets"] = query_factored_1["_causal_seq_offsets"]

    if query_joint_1["_causal_seq_offsets"].equal(key_joint_1["_causal_seq_offsets"]) and query_joint_1[
        "_causal_seq_offsets"
    ].equal(value_joint_1["_causal_seq_offsets"]):
        key_joint_1["_causal_seq_offsets"] = query_joint_1["_causal_seq_offsets"]
        value_joint_1["_causal_seq_offsets"] = query_joint_1["_causal_seq_offsets"]

    if query_factored_2["_causal_seq_offsets"].equal(key_factored_2["_causal_seq_offsets"]) and query_factored_2[
        "_causal_seq_offsets"
    ].equal(value_factored_2["_causal_seq_offsets"]):
        key_factored_2["_causal_seq_offsets"] = query_factored_2["_causal_seq_offsets"]
        value_factored_2["_causal_seq_offsets"] = query_factored_2["_causal_seq_offsets"]

    if query_joint_2["_causal_seq_offsets"].equal(key_joint_2["_causal_seq_offsets"]) and query_joint_2[
        "_causal_seq_offsets"
    ].equal(value_joint_2["_causal_seq_offsets"]):
        key_joint_2["_causal_seq_offsets"] = query_joint_2["_causal_seq_offsets"]
        value_joint_2["_causal_seq_offsets"] = query_joint_2["_causal_seq_offsets"]

    # Build a matching flex attention_meta for the given shapes
    kwargs_1 = {}
    kwargs_2 = {}
    if impl_1 == "flex" or impl_2 == "flex":
        packed_sequence = packed_qkv22 if impl_2 == "flex" else packed_qkv21
        _, attention_mask, _ = build_packed_sequence(
            "flex",
            packed_sequence=packed_sequence[:, :num_q_heads, :],
            attn_modes=attn_modes,
            split_lens=split_lens,
            sample_lens=sample_lens,
            packed_und_token_indexes=packed_und_idx_t,
            packed_gen_token_indexes=packed_gen_idx_t,
            num_heads=num_q_heads,
            head_dim=head_dim,
            num_layers=num_layers,
        )
        if impl_1 == "flex":
            kwargs_1["attention_mask"] = attention_mask
        elif impl_2 == "flex":
            kwargs_2["attention_mask"] = attention_mask

    # natten_metadata is a required argument, but setting it to None implements standard self attn.
    if impl_1 == "three_way":
        kwargs_1["natten_metadata"] = None
    elif impl_2 == "three_way":
        kwargs_2["natten_metadata"] = None

    output1_factored = compile(AttentionWrapper(fn_1))(
        query_factored_1,
        key_factored_1,
        value_factored_1,
        **kwargs_1,
    )
    torch.cuda.synchronize()
    output1_joint = compile(AttentionWrapper(fn_1))(
        query_joint_1,
        key_joint_1,
        value_joint_1,
        **kwargs_1,
    )
    torch.cuda.synchronize()

    output2_factored = compile(AttentionWrapper(fn_2))(
        query_factored_2,
        key_factored_2,
        value_factored_2,
        **kwargs_2,
    )
    torch.cuda.synchronize()
    output2_joint = compile(AttentionWrapper(fn_2))(
        query_joint_2,
        key_joint_2,
        value_joint_2,
        **kwargs_2,
    )
    torch.cuda.synchronize()

    # joint vs factored test. should be same
    torch.testing.assert_close(
        get_all_seq(output1_factored)[:real_len], get_all_seq(output1_joint)[:real_len], atol=atol_self, rtol=rtol_self
    )
    torch.testing.assert_close(
        get_all_seq(output2_factored)[:real_len], get_all_seq(output2_joint)[:real_len], atol=atol_self, rtol=rtol_self
    )

    # impl 1 vs impl 2. needs more tolerance
    torch.testing.assert_close(
        get_all_seq(output2_factored)[:real_len], get_all_seq(output1_factored)[:real_len], atol=atol_cmp, rtol=rtol_cmp
    )
    torch.testing.assert_close(
        get_all_seq(output2_joint)[:real_len], get_all_seq(output1_joint)[:real_len], atol=atol_cmp, rtol=rtol_cmp
    )

    if test_backward:
        get_all_seq(output1_joint)[:real_len].sum().backward()
        get_all_seq(output2_joint)[:real_len].sum().backward()
        get_all_seq(output1_factored)[:real_len].sum().backward()
        get_all_seq(output2_factored)[:real_len].sum().backward()

        # should be close but not necessarily exactly the same because of aggregation order in bwd
        torch.testing.assert_close(
            packed_qkv11.grad[:real_len], packed_qkv12.grad[:real_len], atol=atol_bwd_self, rtol=rtol_bwd_self
        )
        torch.testing.assert_close(
            packed_qkv21.grad[:real_len], packed_qkv22.grad[:real_len], atol=atol_bwd_self, rtol=rtol_bwd_self
        )

        # different attention implementations, needs more tolerance
        torch.testing.assert_close(
            packed_qkv11.grad[:real_len], packed_qkv21.grad[:real_len], atol=atol_bwd_cmp, rtol=rtol_bwd_cmp
        )


@pytest.mark.L0
@pytest.mark.skipif(not NATTEN_SUPPORTED, reason="NATTEN is not available, or too old.")
def test_two_way_attention_cmp_flex_attn():
    _test_attention_impls("two_way", "flex")


@pytest.mark.L0
@pytest.mark.skipif(not NATTEN_SUPPORTED, reason="NATTEN is not available, or too old.")
def test_two_way_attention_vs_three_way_attention():
    _test_attention_impls("two_way", "three_way")


@pytest.mark.L0
def test_decoder_layer_optimized_path_empty_und_tensor_shape():
    """Empty und tensors in the optimized AR path must be 2D, not 1D.

    In the optimized path (frame > 0, KV cache active), the decoder layer creates empty
    und tensors for all intermediate und variables.  These tensors are stored as
    ``causal_seq`` in the output FactoredSequencePack, and the *next* decoder layer
    calls ``get_und_seq(input)`` to retrieve them.  If they are 1D ``[0]``, a subsequent
    RMSNorm ``weight [H] * hidden_states [0]`` triggers:
        RuntimeError: The size of tensor a (H) must match tensor b (0) at non-singleton dim 0
    because broadcasting requires one dim to be 1, but H != 0.

    The fix is ``.new_empty(0, X.shape[-1])`` which yields 2D ``[0, H]``.
    """
    hidden_dim = 32
    device = torch.device("cpu")
    dtype = torch.float32

    # Old (buggy): torch.empty(0, ...) produces 1D [0]
    old_und = torch.empty(0, device=device, dtype=dtype)  # [0]
    assert old_und.shape == (0,), "sanity: old code creates 1D tensor"

    # Simulate RMSNorm: weight [H] * hidden_states [0]  → fails
    weight = torch.ones(hidden_dim, device=device, dtype=dtype)  # [H]
    with pytest.raises(RuntimeError):
        _ = weight * old_und  # [H] * [0] → dimension mismatch

    # New (fixed): .new_empty(0, H) produces 2D [0, H]
    ref = torch.randn(4, hidden_dim, device=device, dtype=dtype)  # [S_gen, H]
    new_und = ref.new_empty(0, ref.shape[-1])  # [0, H]
    assert new_und.shape == (0, hidden_dim), "fix: 2D tensor with correct hidden dim"

    # RMSNorm on 2D empty tensor succeeds (result is also [0, H])
    norm_out = weight * new_und  # [H] * [0, H] → [0, H]
    assert norm_out.shape == (0, hidden_dim)

    # Verify round-trip through FactoredSequencePack preserves 2D shape.
    # from_mode_splits(und, gen, meta) stores und as causal_seq; get_und_seq retrieves it.
    meta = {"causal_seq": new_und, "full_only_seq": ref}
    retrieved = get_und_seq(meta)  # type: ignore[arg-type]
    assert retrieved.shape == (0, hidden_dim), "get_und_seq must return 2D tensor"


if __name__ == "__main__":
    test_two_way_attention_cmp_flex_attn()
    test_two_way_attention_vs_three_way_attention()
