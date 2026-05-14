import torch
from torch.nn.attention.flex_attention import BlockMask, create_block_mask, flex_attention

from cosmos.model.attention import (
    attention,
    merge_attentions,
    multi_dimensional_attention_varlen,
)
from cosmos.model.attention.masks import CausalType
from cosmos.model.vfm.utils.memory import KVToStore, MemoryValue

flex_attention = torch.compile(flex_attention)


class SplitInfo:
    def __init__(
        self,
        split_lens: list[int],
        attn_modes: list[str],
        sample_lens: list[int],
        actual_len: int,
        is_three_way: bool = False,
        vision_token_shapes: list[tuple[int, int, int]] | None = None,
        action_token_shapes: list[tuple[int, ...]] | None = None,
        num_action_tokens_per_supertoken: int = 0,
        null_action_supertokens: bool = False,
    ):
        """
        Actual len is the actual non-padded length of the packed sequence.
        It's used to trim split_lens, attn_modes and sample_lens, which were
        originally padded to max sequence length (likely for flex attention).
        """
        assert sum(sample_lens) == sum(split_lens), (
            f"Sum of new sample lens {sum(sample_lens)} is not equal to sum of new split lens {sum(split_lens)}"
        )

        max_causal_len = 0
        max_full_len = 0
        for split_len, attn_mode in zip(split_lens, attn_modes):
            if attn_mode == "causal":
                max_causal_len = max(max_causal_len, split_len)
            elif attn_mode == "full":
                max_full_len = max(max_full_len, split_len)

        self.max_causal_len = max_causal_len
        self.max_full_len = max_full_len
        self.max_sample_len = max(sample_lens)

        self.split_lens = split_lens
        self.attn_modes = attn_modes
        self.sample_lens = sample_lens

        self.is_three_way = is_three_way
        self.vision_token_shapes = vision_token_shapes
        self.action_token_shapes = action_token_shapes
        self.num_action_tokens_per_supertoken = num_action_tokens_per_supertoken
        self.null_action_supertokens = null_action_supertokens


AttentionMaskType = BlockMask | SplitInfo


_dotproduct_attention_cache = {}


from cosmos.data.vfm.sequence_packing import (
    FactoredSequencePack,
    JointSequencePack,
    create_sparse_mask,
    factored_from_joint_sequence,
    from_joint,
    from_mode_splits,
    generate_natten_metadata,
    generate_temporal_causal_natten_metadata,
    get_all_seq,
    get_causal_seq,
    get_full_only_seq,
    joint_from_joint_sequence,
)


def two_way_attention(
    packed_query_states: FactoredSequencePack | JointSequencePack,
    packed_key_states: FactoredSequencePack | JointSequencePack,
    packed_value_states: FactoredSequencePack | JointSequencePack,
):
    """
    Performs two-way attention with causal and full attention.
    """

    causal_q, causal_q_offsets = get_causal_seq(packed_query_states)
    causal_k, causal_k_offsets = get_causal_seq(packed_key_states)
    causal_v, _ = get_causal_seq(packed_value_states)
    full_q, full_q_offsets = get_full_only_seq(packed_query_states)

    sample_offsets = packed_query_states["sample_offsets"]

    use_dont_care_mask = causal_q_offsets is causal_k_offsets


    causal_res = attention(
        causal_q.unsqueeze(0),  # [1,N_und,heads,head_dim]
        causal_k.unsqueeze(0),  # [1,N_und,heads,head_dim]
        causal_v.unsqueeze(0),  # [1,N_und,heads,head_dim]
        cumulative_seqlen_Q=causal_q_offsets,
        cumulative_seqlen_KV=causal_k_offsets,
        max_seqlen_Q=packed_query_states["max_causal_len"],
        max_seqlen_KV=packed_query_states["max_causal_len"],
        is_causal=True,
        causal_type=CausalType.DontCare if use_dont_care_mask else CausalType.TopLeft,
    )  # [1,N_und,heads,head_dim]

    # [1,N_und,heads,head_dim] -> [N_und,heads,head_dim] -> [N_und,heads*head_dim]
    causal_out = causal_res.squeeze(0).flatten(-2, -1)  # type: ignore  # [N_und,heads*head_dim]

    full_res = attention(
        full_q.unsqueeze(0),  # [1,N_full,heads,head_dim]
        get_all_seq(packed_key_states).unsqueeze(0),  # [1,N_all,heads,head_dim]
        get_all_seq(packed_value_states).unsqueeze(0),  # [1,N_all,heads,head_dim]
        cumulative_seqlen_Q=full_q_offsets,
        cumulative_seqlen_KV=sample_offsets,
        max_seqlen_Q=packed_query_states["max_full_len"],
        max_seqlen_KV=packed_query_states["max_sample_len"],
    )  # [1,N_full,heads,head_dim]

    # [1,N_full,heads,head_dim] -> [N_full,heads,head_dim] -> [N_full,heads*head_dim]
    full_out = full_res.squeeze(0).flatten(-2, -1)  # type: ignore  # [N_full,heads*head_dim]

    out_all = from_mode_splits(causal_out, full_out, packed_query_states)
    return out_all


def three_way_attention(
    packed_query_states: FactoredSequencePack | JointSequencePack,
    packed_key_states: FactoredSequencePack | JointSequencePack,
    packed_value_states: FactoredSequencePack | JointSequencePack,
    natten_metadata: dict | None,
    attention_meta: SplitInfo | None = None,
):
    """
    Performs three-way attention, with understanding and generations attentions fully decomposed,
    and allows sparsity / multi-dimensional masking in the generation tower.

    When attention_meta is provided with null_action_supertokens=True, zeros V for the first
    num_action_tokens_per_supertoken tokens of each sample's GEN sequence (null action
    supertokens for temporal causal training). The metadata encodes is_causal=(True, False):
    causal across T supertokens, full within each supertoken S.

    NOTE: the three-way decomposition is only done so we can handle sparsity in the gen tower,
    but a KEY assumption is that the "full" tokens all correspond to the same modality!
    We should be careful when extending this to beyond t2i and t2v.
    """

    causal_q, causal_q_offsets = get_causal_seq(packed_query_states)
    causal_k, causal_k_offsets = get_causal_seq(packed_key_states)
    causal_v, _ = get_causal_seq(packed_value_states)
    full_q, full_q_offsets = get_full_only_seq(packed_query_states)
    full_k, full_k_offsets = get_full_only_seq(packed_key_states)
    full_v, _ = get_full_only_seq(packed_value_states)

    sample_offsets = packed_query_states["sample_offsets"]

    if attention_meta is not None and attention_meta.null_action_supertokens:
        # Zero V for the first num_action_tokens_per_supertoken tokens of each
        # sample's GEN sequence (null action supertokens at t=0).
        # out_i = Σ_j softmax(QKᵀ/√d)_j · V_j — terms with V_j=0 contribute exactly 0 to the output,
        # regardless of attention weights. Softmax mass is still allocated to these positions (not
        # redistributed), so this differs from hard key masking, but the output contribution is 0.
        full_v = full_v.clone()
        starts = full_q_offsets[:-1].long()  # [B]
        null_positions = (
            starts.unsqueeze(1) + torch.arange(attention_meta.num_action_tokens_per_supertoken, device=starts.device)
        ).reshape(-1)
        full_v[null_positions] = 0

    use_dont_care_mask = causal_q_offsets is causal_k_offsets


    causal_res = attention(
        causal_q.unsqueeze(0),  # [1,N_und,heads,head_dim]
        causal_k.unsqueeze(0),  # [1,N_und,heads,head_dim]
        causal_v.unsqueeze(0),  # [1,N_und,heads,head_dim]
        cumulative_seqlen_Q=causal_q_offsets,
        cumulative_seqlen_KV=causal_k_offsets,
        max_seqlen_Q=packed_query_states["max_causal_len"],
        max_seqlen_KV=packed_query_states["max_causal_len"],
        is_causal=True,
        causal_type=CausalType.DontCare if use_dont_care_mask else CausalType.TopLeft,
    )  # [1,N_und,heads,head_dim]
    # [1,N_und,heads,head_dim] -> [N_und,heads,head_dim] -> [N_und,heads*head_dim]
    causal_out = causal_res.squeeze(0).flatten(-2, -1)  # type: ignore  # [N_und,heads*head_dim]

    # If there's no metadata, it's a dense layer
    if natten_metadata is None:
        full_sa, full_sa_lse = attention(
            full_q.unsqueeze(0),  # [1,N_full,heads,head_dim]
            full_k.unsqueeze(0),  # [1,N_full,heads,head_dim]
            full_v.unsqueeze(0),  # [1,N_full,heads,head_dim]
            cumulative_seqlen_Q=full_q_offsets,
            cumulative_seqlen_KV=full_k_offsets,
            max_seqlen_Q=packed_query_states["max_full_len"],
            max_seqlen_KV=packed_query_states["max_full_len"],
            return_lse=True,
        )  # full_sa: [1,N_full,heads,head_dim], full_sa_lse: [1,N_full,heads]
    else:
        assert natten_metadata is not None
        full_sa, full_sa_lse = multi_dimensional_attention_varlen(
            full_q.unsqueeze(0),  # [1,N_full,heads,head_dim]
            full_k.unsqueeze(0),  # [1,N_full,heads,head_dim]
            full_v.unsqueeze(0),  # [1,N_full,heads,head_dim]
            metadata=natten_metadata,
            return_lse=True,
        )  # full_sa: [1,N_full,heads,head_dim], full_sa_lse: [1,N_full,heads]

    full_ca, full_ca_lse = attention(
        full_q.unsqueeze(0),  # [1,N_full,heads,head_dim]
        causal_k.unsqueeze(0),  # [1,N_und,heads,head_dim]
        causal_v.unsqueeze(0),  # [1,N_und,heads,head_dim]
        cumulative_seqlen_Q=full_q_offsets,
        cumulative_seqlen_KV=causal_k_offsets,
        max_seqlen_Q=packed_query_states["max_full_len"],
        max_seqlen_KV=packed_query_states["max_causal_len"],
        return_lse=True,
    )  # full_ca: [1,N_full,heads,head_dim], full_ca_lse: [1,N_full,heads]

    assert full_sa.shape == full_ca.shape
    full_res, _ = merge_attentions(
        outputs=[full_sa, full_ca], lse_tensors=[full_sa_lse, full_ca_lse], torch_compile=False
    )  # [1,N_full,heads,head_dim]

    # [1,N_full,heads,head_dim] -> [N_full,heads,head_dim] -> [N_full,heads*head_dim]
    full_out = full_res.squeeze(0).flatten(-2, -1)  # type: ignore  # [N_full,heads*head_dim]

    out_all = from_mode_splits(causal_out, full_out, packed_query_states)
    return out_all


def pad_sequence(tensor, pad_size):
    """
    Pad a tensor along the second-to-last dimension.

    Args:
        tensor: Input tensor to pad
        pad_size: Number of padding elements to add

    Returns:
        Padded tensor with zeros added along dim=-2
    """
    if pad_size <= 0:
        return tensor
    pad_shape = list(tensor.shape)
    pad_shape[-2] = pad_size
    padding = torch.zeros(pad_shape, dtype=tensor.dtype, device=tensor.device)
    return torch.cat([tensor, padding], dim=-2)  # [...,S+pad_size,...]


def block_flex_attention(
    packed_query_states: FactoredSequencePack | JointSequencePack,
    packed_key_states: FactoredSequencePack | JointSequencePack,
    packed_value_states: FactoredSequencePack | JointSequencePack,
    attention_mask: BlockMask,
    block_size: int | None = None,
):
    packed_queries = get_all_seq(packed_query_states)  # [N,heads,head_dim]
    packed_keys = get_all_seq(packed_key_states)  # [N,heads,head_dim]
    packed_values = get_all_seq(packed_value_states)  # [N,heads,head_dim]
    max_num_tokens = packed_query_states["max_num_tokens"]

    num_attention_heads = packed_queries.shape[1]
    head_dim = packed_queries.shape[2]

    # Handle block mask attention with flex_attention
    pad_size = max_num_tokens - packed_queries.shape[0]
    packed_queries_padded = pad_sequence(packed_queries.permute(1, 0, 2), pad_size)  # [heads,max_num_tokens,head_dim]
    packed_keys_padded = pad_sequence(packed_keys.permute(1, 0, 2), pad_size)  # [heads,max_num_tokens,head_dim]
    packed_values_padded = pad_sequence(packed_values.permute(1, 0, 2), pad_size)  # [heads,max_num_tokens,head_dim]

    packed_attn_output = flex_attention(
        packed_queries_padded.unsqueeze(0),  # [1,heads,max_num_tokens,head_dim]
        packed_keys_padded.unsqueeze(0),  # [1,heads,max_num_tokens,head_dim]
        packed_values_padded.unsqueeze(0),  # [1,heads,max_num_tokens,head_dim]
        enable_gqa=True,
        block_mask=attention_mask,
    )  # [1,heads,max_num_tokens,head_dim]
    assert isinstance(packed_attn_output, torch.Tensor)

    end_index = packed_attn_output.shape[2] - pad_size
    packed_attn_output = packed_attn_output[0, :, :end_index, :]  # [heads,N,head_dim]
    packed_attn_output = packed_attn_output.transpose(0, 1).reshape(
        -1, num_attention_heads * head_dim
    )  # [N,heads*head_dim]

    return from_joint(packed_attn_output, packed_query_states)


def dispatch_attention(
    packed_query_states: FactoredSequencePack | JointSequencePack,
    packed_key_states: FactoredSequencePack | JointSequencePack,
    packed_value_states: FactoredSequencePack | JointSequencePack,
    attention_mask: BlockMask | SplitInfo,
    natten_metadata: dict | None = None,
    memory_value: MemoryValue | None = None,
) -> tuple[FactoredSequencePack | JointSequencePack, KVToStore | None]:
    assert memory_value is None, "Base dispatch_attention does not handle MemoryValue"
    if isinstance(attention_mask, SplitInfo) and attention_mask.is_three_way:
        output = three_way_attention(
            packed_query_states,
            packed_key_states,
            packed_value_states,
            natten_metadata=natten_metadata,
            attention_meta=attention_mask,
        )
    elif isinstance(attention_mask, SplitInfo):
        output = two_way_attention(packed_query_states, packed_key_states, packed_value_states)
    else:
        output = block_flex_attention(packed_query_states, packed_key_states, packed_value_states, attention_mask)
    return output, None


def build_packed_sequence(
    joint_attn_implementation: str,
    *,
    packed_sequence: torch.Tensor,
    attn_modes: list[str],
    split_lens: list[int],
    sample_lens: list[int],
    packed_und_token_indexes: torch.LongTensor,
    packed_gen_token_indexes: torch.LongTensor,
    num_heads: int,
    head_dim: int,
    num_layers: int,
    token_shapes: list[tuple[int, int, int]] | None = None,
    natten_parameter_list: list | None = None,
    block_size: int = 128,
    is_image_batch: bool = False,
    cp_world_size: int = 1,
    video_temporal_causal: bool = False,
    use_rolling_kv_cache: bool = False,
    vision_token_shapes: list[tuple[int, int, int]] | None = None,
    action_token_shapes: list[tuple[int, ...]] | None = None,
    num_action_tokens_per_supertoken: int = 0,
    null_action_supertokens: bool = False,
    pad_for_cuda_graphs: bool = False,
) -> tuple[FactoredSequencePack | JointSequencePack, AttentionMaskType, list | None]:
    """
    Build the model input pack and attention meta for joint attention.
    Returns a tuple: (input_pack, attention_meta).
    """
    device = packed_sequence.device
    natten_metadata_list = None
    if joint_attn_implementation == "flex":
        sparse_mask = create_sparse_mask(sample_lens, split_lens, attn_modes, device)
        seqlen = sum(sample_lens)
        attention_meta = create_block_mask(
            sparse_mask,
            B=1,
            H=num_heads,
            Q_LEN=seqlen,
            KV_LEN=seqlen,
            device=device,
            BLOCK_SIZE=block_size,
            _compile=True,
        )
        make_pack = joint_from_joint_sequence
    elif joint_attn_implementation == "two_way":
        attention_meta = SplitInfo(
            split_lens=split_lens,
            attn_modes=attn_modes,
            sample_lens=sample_lens,
            actual_len=int(packed_sequence.shape[0]),
        )
        make_pack = factored_from_joint_sequence
    elif joint_attn_implementation == "three_way":
        attention_meta = SplitInfo(
            split_lens=split_lens,
            attn_modes=attn_modes,
            sample_lens=sample_lens,
            actual_len=int(packed_sequence.shape[0]),
            is_three_way=True,
            vision_token_shapes=vision_token_shapes,
            action_token_shapes=action_token_shapes,
            num_action_tokens_per_supertoken=num_action_tokens_per_supertoken,
            null_action_supertokens=null_action_supertokens,
        )
        make_pack = factored_from_joint_sequence
        # The rolling KV-cache path implements temporal causality in
        # three_way_attention_with_kv_cache; skip NATTEN metadata.
        if not use_rolling_kv_cache:
            # Temporal causal: encode (T, S) supertoken layout; spatial NATTEN: encode (H, W) layout.
            if video_temporal_causal:
                natten_metadata_list = generate_temporal_causal_natten_metadata(
                    vision_token_shapes=vision_token_shapes,
                    num_action_tokens_per_supertoken=num_action_tokens_per_supertoken,
                    num_layers=num_layers,
                    head_dim=head_dim,
                    device=device,
                    dtype=packed_sequence.dtype,
                    requires_grad=packed_sequence.requires_grad,
                )
            else:
                natten_metadata_list = generate_natten_metadata(
                    token_shapes=token_shapes,
                    head_dim=head_dim,
                    num_layers=num_layers,
                    device=device,
                    dtype=packed_sequence.dtype,
                    requires_grad=packed_sequence.requires_grad,
                    natten_parameter_list=natten_parameter_list,
                )
    else:
        raise ValueError(
            f"Invalid joint_attn_implementation: {joint_attn_implementation}. "
            "Must be 'two_way', 'three_way', or 'flex'."
        )

    input_pack = make_pack(
        packed_sequence=packed_sequence,
        attn_modes=attn_modes,
        split_lens=split_lens,
        sample_lens=sample_lens,
        packed_und_token_indexes=packed_und_token_indexes.to(device),
        packed_gen_token_indexes=packed_gen_token_indexes.to(device),
        is_image_batch=is_image_batch,
        cp_world_size=cp_world_size,
        pad_for_cuda_graphs=pad_for_cuda_graphs,
    )
    # Not needed anymore, can cause recompilations.
    input_pack.pop("split_lens", None)
    input_pack.pop("attn_modes", None)
    return input_pack, attention_meta, natten_metadata_list
