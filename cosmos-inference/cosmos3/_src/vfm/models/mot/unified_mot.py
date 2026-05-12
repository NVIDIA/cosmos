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
import json
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn
from transformers.utils import ModelOutput

from cosmos3._src.imaginaire.utils import log
from cosmos3._src.vfm.datasets.sequence_packing import (
    FactoredSequencePack,
    from_joint,
    from_und_gen_splits,
    get_device_and_dtype,
    get_gen_seq,
    get_und_seq,
    set_gen_seq,
    set_und_seq,
    zeros_like,
)
from cosmos3._src.vfm.models.mot.attention import (
    AttentionMaskType,
    dispatch_attention,
)
from cosmos3._src.vfm.models.utils.memory import KVToStore, MemoryState, MemoryValue

# Nemotron 3 Dense VL imports
from cosmos3._src.vfm.models.vlm.nemotron_3_dense_vl.configuration_nemotron_3_dense_vl import (
    Nemotron3DenseVLTextConfig as _Nemotron3DenseVLTextConfig,
)
from cosmos3._src.vfm.models.vlm.nemotron_3_dense_vl.nemotron_3_dense_vl import (
    MultiModalRotaryEmbedding,
    Nemotron3DenseVLMLP,
    Nemotron3DenseVLPreTrainedModel,
    Nemotron3DenseVLRMSNorm,
    apply_rotary_pos_emb_partial,
)

# Qwen3-VL imports
from cosmos3._src.vfm.models.vlm.qwen3_vl.configuration_qwen3_vl import (
    Qwen3VLTextConfig as _Qwen3VLTextConfig,
)
from cosmos3._src.vfm.models.vlm.qwen3_vl.qwen3_vl import (
    Qwen3VLPreTrainedModel,
    Qwen3VLTextMLP,
    Qwen3VLTextRMSNorm,
    Qwen3VLTextRotaryEmbedding,
)
from cosmos3._src.vfm.models.vlm.qwen3_vl.qwen3_vl import (
    apply_rotary_pos_emb as qwen3_vl_apply_rotary_pos_emb,
)

# Qwen3-VL-MoE imports
from cosmos3._src.vfm.models.vlm.qwen3_vl_moe.configuration_qwen3_vl_moe import (
    Qwen3VLMoeTextConfig as _Qwen3VLMoeTextConfig,
)
from cosmos3._src.vfm.models.vlm.qwen3_vl_moe.qwen3_vl_moe import (
    LBLMetadata,
    Qwen3VLMoePreTrainedModel,
    Qwen3VLMoeTextMLP,
    Qwen3VLMoeTextRMSNorm,
    Qwen3VLMoeTextRotaryEmbedding,
    Qwen3VLMoeTextSparseMoeBlock,
)

# Torch optimization settings
torch._dynamo.config.cache_size_limit = 512
torch._dynamo.config.accumulated_cache_size_limit = 4096

# -----------------------------------------------------------------------------
# Unified MoT (Mixture of Transformers) implementation supporting:
#   - Qwen3-VL Dense, Qwen3-VL MoE, and Nemotron 3 Dense VL
#
# Shared components:
#   - PackedAttentionMoT (config-driven QK norm and RoPE)
#   - MoTDecoderLayer (used by all variants)
#   - _impl_* (shared init/forward)
#
# Variant-specific wrapper classes are needed for different PreTrainedModel bases.
# Sub-layer classes (MLP, RMSNorm, RotaryEmbedding, RoPE fn) are selected via LayerTypes.
# -----------------------------------------------------------------------------


class LayerTypes:
    def __init__(self, variant: str):
        self.variant = variant
        if variant == "qwen3_vl_moe":
            self.mlp = Qwen3VLMoeTextMLP
            self.rms_norm = Qwen3VLMoeTextRMSNorm
            self.rotary_embedding = Qwen3VLMoeTextRotaryEmbedding
            self.apply_rotary_pos_emb = qwen3_vl_apply_rotary_pos_emb
        elif variant == "nemotron_dense":
            self.mlp = Nemotron3DenseVLMLP
            self.rms_norm = Nemotron3DenseVLRMSNorm
            self.rotary_embedding = MultiModalRotaryEmbedding
            self.apply_rotary_pos_emb = apply_rotary_pos_emb_partial
        elif variant == "qwen3_vl_dense":
            self.mlp = Qwen3VLTextMLP
            self.rms_norm = Qwen3VLTextRMSNorm
            self.rotary_embedding = Qwen3VLTextRotaryEmbedding
            self.apply_rotary_pos_emb = qwen3_vl_apply_rotary_pos_emb
        else:
            raise ValueError(f"Unknown LayerTypes variant: {variant!r}")

    @property
    def is_moe(self) -> bool:
        return self.variant == "qwen3_vl_moe"


class NaiveCache:
    def __init__(self, num_layers):
        self.key_cache = {k: None for k in range(num_layers)}
        self.value_cache = {k: None for k in range(num_layers)}

    @property
    def num_layers(self):
        return len(self.key_cache)

    @property
    def seq_lens(self):
        if self.key_cache[0] is not None:
            return self.key_cache[0].shape[0]
        else:
            return 0


@dataclass
class BaseOutputWithPast(ModelOutput):
    packed_query_sequence: torch.FloatTensor = None
    past_key_values: Optional[NaiveCache] = None


# Qwen3-VL MoT (Mixture of Tokens) implementation
# Combines Qwen3-VL vision-language capabilities with MoT dual-pathway architecture


class Qwen3VLTextConfig(_Qwen3VLTextConfig):
    r"""
    Qwen3VLTextConfig with MoT-specific parameters.
    Extends Qwen3VLTextConfig for text component MoT support with comprehensive configuration.
    """

    def __init__(
        self,
        # MoT-specific parameters with comprehensive defaults
        qk_norm_for_text: bool = False,  # Whether to apply QK norm in the understanding (text) pathway
        qk_norm_for_diffusion: bool = True,  # Whether to apply QK norm in the generation (diffusion) pathway
        freeze_und: bool = False,  # Freeze understanding pathway
        layer_module: str = "MoTDecoderLayer",
        tie_word_embeddings: bool = True,
        **kwargs,
    ):
        # Store MoT-specific parameters
        self.qk_norm_for_text = qk_norm_for_text
        self.qk_norm_for_diffusion = qk_norm_for_diffusion
        self.freeze_und = freeze_und
        self.layer_module = layer_module
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)

    @classmethod
    def from_json_file(cls, json_file):
        """
        Enhanced from_json_file that handles both nested and flat configs.

        For nested configs (with text_config section), extracts the text_config.
        For flat configs, loads directly.
        """

        # Load the raw JSON
        with open(json_file, encoding="utf-8") as reader:
            config_dict = json.load(reader)

        # Check if this is a nested config with text_config section
        if "text_config" in config_dict and isinstance(config_dict["text_config"], dict):
            # Extract the text_config section for nested configs
            log.debug("Detected nested config, extracting text_config section")
            config_dict = config_dict["text_config"]
        else:
            # Use the config as-is for flat configs
            log.debug("Detected flat config, using directly")

        # Create config from the (potentially extracted) dict
        return cls(**config_dict)


# Qwen3-VL-MoE MoT (Mixture of Tokens) implementation
# Combines Qwen3-VL-MoE vision-language capabilities with MoT dual-pathway architecture


class Qwen3VLMoeTextConfig(_Qwen3VLMoeTextConfig):
    r"""
    Qwen3VLMoeTextConfig with MoT-specific parameters.
    Extends Qwen3VLMoeTextConfig for text component MoT support with comprehensive configuration.
    """

    def __init__(
        self,
        # MoT-specific parameters with comprehensive defaults
        qk_norm_for_text: bool = False,  # Whether to apply QK norm in the understanding (text) pathway
        qk_norm_for_diffusion: bool = True,  # Whether to apply QK norm in the generation (diffusion) pathway
        freeze_und: bool = False,  # Freeze understanding pathway
        layer_module: str = "MoTDecoderLayer",
        tie_word_embeddings: bool = True,
        **kwargs,
    ):
        # Store MoT-specific parameters
        self.qk_norm_for_text = qk_norm_for_text
        self.qk_norm_for_diffusion = qk_norm_for_diffusion
        self.freeze_und = freeze_und
        self.layer_module = layer_module
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)

    @classmethod
    def from_json_file(cls, json_file):
        """
        Enhanced from_json_file that handles both nested and flat configs.
        For nested configs (with text_config section), extracts the text_config.
        For flat configs, loads directly.
        """

        # Load the raw JSON
        with open(json_file, encoding="utf-8") as reader:
            config_dict = json.load(reader)

        # Check if this is a nested config with text_config section
        if "text_config" in config_dict and isinstance(config_dict["text_config"], dict):
            # Extract the text_config section for nested configs
            log.debug("Detected nested config, extracting text_config section")
            config_dict = config_dict["text_config"]
        else:
            # Use the config as-is for flat configs
            log.debug("Detected flat config, using directly")

        # Create config from the (potentially extracted) dict
        return cls(**config_dict)


# Nemotron 3 Dense VL MoT config

_NEMOTRON_MOT_TEXT_CONFIG_KEYS = {
    "vocab_size",
    "tie_word_embeddings",
    "hidden_size",
    "intermediate_size",
    "num_hidden_layers",
    "num_attention_heads",
    "head_dim",
    "num_key_value_heads",
    "mlp_hidden_act",
    "attention_bias",
    "mlp_bias",
    "initializer_range",
    "layer_norm_epsilon",
    "residual_in_fp32",
    "use_cache",
    "num_logits_to_keep",
    "pad_token_id",
    "bos_token_id",
    "eos_token_id",
    "sliding_window",
    "max_position_embeddings",
    "attention_dropout",
    "hidden_dropout",
    "enable_rope",
    "rope_scaling",
    "rope_theta",
    "enable_mrope",
    "mrope_section",
    "torch_dtype",
}


class Nemotron3DenseVLTextConfig(_Nemotron3DenseVLTextConfig):
    """MoT-enabled config for the Nemotron 3 Dense VL text backbone.

    Extends the upstream ``Nemotron3DenseVLTextConfig`` with MoT-specific
    fields (per-pathway QK normalisation, freeze control, decoder layer class).
    Supports both the VLM nested config and the flat LLM config format.
    """

    def __init__(
        self,
        qk_norm_for_text: bool = False,
        qk_norm_for_diffusion: bool = True,
        freeze_und: bool = False,
        layer_module: str = "MoTDecoderLayer",
        tie_word_embeddings: bool = False,
        **kwargs,
    ) -> None:
        self.qk_norm_for_text = qk_norm_for_text
        self.qk_norm_for_diffusion = qk_norm_for_diffusion
        self.freeze_und = freeze_und
        self.layer_module = layer_module
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)

    @classmethod
    def from_json_file(cls, json_file: str) -> "Nemotron3DenseVLTextConfig":
        """Load config from a JSON file, handling both VLM nested and flat LLM formats."""
        with open(json_file, encoding="utf-8") as reader:
            config_dict = json.load(reader)
        if "text_config" in config_dict and isinstance(config_dict["text_config"], dict):
            log.debug("Detected nested config, extracting text_config section")
            config_dict = dict(config_dict["text_config"])
        else:
            log.debug("Detected flat config, using directly")
        if config_dict.get("num_hidden_layers") == 56:
            # Upstream VLM stores attention and MLP as separate alternating blocks (56 total);
            # MoT combines both into standard transformer layers (28 total).
            config_dict = {**config_dict, "num_hidden_layers": 28}
        filtered = {k: v for k, v in config_dict.items() if k in _NEMOTRON_MOT_TEXT_CONFIG_KEYS}
        return cls(**filtered)


# -----------------------------------------------------------------------------
# Common layers between Qwen3VL Dense, MoE, and Nemotron 3 Dense VL models
# -----------------------------------------------------------------------------


class PackedAttentionMoT(nn.Module):
    """
    Dual-pathway packed attention for MoT architectures.
    Implements understanding and generation pathways with separate projections.

    Used for Qwen3VL (Dense), Qwen3VL-MoE, and Nemotron 3 Dense VL variants.
    QK normalisation and RoPE function are selected via ``layer_types`` and config
    attributes (``qk_norm_for_text`` / ``qk_norm_for_diffusion``).
    """

    def __init__(self, config, layer_idx: int, layer_types: LayerTypes):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout

        eps = config.rms_norm_eps

        # Understanding pathway projections
        self.q_proj = nn.Linear(self.hidden_size, self.num_attention_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_attention_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)

        # Understanding pathway QK norm
        if config.qk_norm_for_text:
            self.q_norm = layer_types.rms_norm(self.head_dim, eps=eps)
            self.k_norm = layer_types.rms_norm(self.head_dim, eps=eps)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

        # Generation pathway QK norm
        if config.qk_norm_for_diffusion:
            self.q_norm_moe_gen = layer_types.rms_norm(self.head_dim, eps=eps)
            self.k_norm_moe_gen = layer_types.rms_norm(self.head_dim, eps=eps)
        else:
            self.q_norm_moe_gen = nn.Identity()
            self.k_norm_moe_gen = nn.Identity()

        # Generation pathway linear projections
        self.q_proj_moe_gen = nn.Linear(
            self.hidden_size, self.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj_moe_gen = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj_moe_gen = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj_moe_gen = nn.Linear(
            self.num_attention_heads * self.head_dim, self.hidden_size, bias=config.attention_bias
        )

        self._apply_rotary_pos_emb = layer_types.apply_rotary_pos_emb
        self.dispatch_attention_fn = dispatch_attention
        self.cp_mesh = None

    def forward(
        self,
        pack: FactoredSequencePack,
        attention_mask: AttentionMaskType,
        packed_position_embeddings: Tuple[FactoredSequencePack, FactoredSequencePack],
        natten_metadata: dict | None = None,
        memory_value: MemoryValue | None = None,
    ) -> tuple[FactoredSequencePack, KVToStore | None]:
        """Forward pass with optional memory-augmented attention.

        When ``memory_value`` is provided, ``dispatch_attention_fn`` routes to
        the appropriate attention kernel (e.g. three-way KV-cache attention
        for training, or AR inference concat + dense attention).

        ``kv_to_store`` is produced when ``memory_value`` is present:
        ``(gen_k, gen_v, und_k, und_v)`` for the caller to write back via
        ``MemoryState.write_for_layer()``.  The tensors are passed with
        gradients attached; each ``MemoryState`` decides whether to detach
        (e.g. for truncated BPTT) or keep gradients (e.g. teacher forcing).

        Args:
            pack: Packed sequence with und/gen tokens
            attention_mask: Attention mask (BlockMask or SplitInfo)
            packed_position_embeddings: RoPE embeddings (cos, sin)
            natten_metadata: Optional NATTEN metadata for neighborhood attention.
            memory_value: Optional read-only tensor container for memory-augmented attention.
        """

        q_und_in = self.q_proj(get_und_seq(pack))  # [N_und,num_heads*head_dim]
        q_gen_in = self.q_proj_moe_gen(get_gen_seq(pack))  # [N_gen,num_heads*head_dim]

        k_und_in = self.k_proj(get_und_seq(pack))  # [N_und,num_kv_heads*head_dim]
        k_gen_in = self.k_proj_moe_gen(get_gen_seq(pack))  # [N_gen,num_kv_heads*head_dim]

        v_und_in = self.v_proj(get_und_seq(pack))  # [N_und,num_kv_heads*head_dim]
        v_gen_in = self.v_proj_moe_gen(get_gen_seq(pack))  # [N_gen,num_kv_heads*head_dim]

        q_und = q_und_in.view(-1, self.num_attention_heads, self.head_dim)  # [N_und,num_heads,head_dim]
        k_und = k_und_in.view(-1, self.num_key_value_heads, self.head_dim)  # [N_und,num_kv_heads,head_dim]
        v_und = v_und_in.view(-1, self.num_key_value_heads, self.head_dim)  # [N_und,num_kv_heads,head_dim]

        q_gen = q_gen_in.view(-1, self.num_attention_heads, self.head_dim)  # [N_gen,num_heads,head_dim]
        k_gen = k_gen_in.view(-1, self.num_key_value_heads, self.head_dim)  # [N_gen,num_kv_heads,head_dim]
        v_gen = v_gen_in.view(-1, self.num_key_value_heads, self.head_dim)  # [N_gen,num_kv_heads,head_dim]

        q_und = self.q_norm(q_und)  # [N_und,num_heads,head_dim]
        k_und = self.k_norm(k_und)  # [N_und,num_kv_heads,head_dim]

        q_gen = self.q_norm_moe_gen(q_gen)  # [N_gen,num_heads,head_dim]
        k_gen = self.k_norm_moe_gen(k_gen)  # [N_gen,num_kv_heads,head_dim]

        if self.config.freeze_und:
            q_und = q_und.detach()
            k_und = k_und.detach()
            v_und = v_und.detach()

        packed_cos = packed_position_embeddings[0]
        packed_sin = packed_position_embeddings[1]

        q_und_, k_und_ = self._apply_rotary_pos_emb(
            q_und,
            k_und,
            get_und_seq(packed_cos),
            get_und_seq(packed_sin),
            unsqueeze_dim=1,
        )  # q_und_: [N_und,num_heads,head_dim], k_und_: [N_und,num_kv_heads,head_dim]
        q_gen_, k_gen_ = self._apply_rotary_pos_emb(
            q_gen,
            k_gen,
            get_gen_seq(packed_cos),
            get_gen_seq(packed_sin),
            unsqueeze_dim=1,
        )  # q_gen_: [N_gen,num_heads,head_dim], k_gen_: [N_gen,num_kv_heads,head_dim]

        packed_query_states_ = from_und_gen_splits(q_und_, q_gen_, pack)  # [N_und+N_gen,num_heads,head_dim]
        packed_key_states_ = from_und_gen_splits(k_und_, k_gen_, pack)  # [N_und+N_gen,num_kv_heads,head_dim]
        packed_value_states_ = from_und_gen_splits(v_und, v_gen, pack)  # [N_und+N_gen,num_kv_heads,head_dim]

        packed_attn_output, kv_to_store = self.dispatch_attention_fn(
            packed_query_states_,
            packed_key_states_,
            packed_value_states_,
            attention_mask,
            natten_metadata=natten_metadata,
            memory_value=memory_value,
        )

        # Produce kv_to_store for MemoryState.write_for_layer() when the
        # dispatch didn't already provide one (e.g. standard or AR frame-0
        # non-CP paths).  CP dispatch returns head-sharded kv_to_store
        # directly, so kv_to_store is already non-None in that case.
        #
        # Gradient detach is NOT done here; each MemoryState.write_for_layer()
        # decides its own gradient policy (e.g. detach for truncated BPTT,
        # keep gradients for teacher forcing).
        if memory_value is not None and kv_to_store is None:
            und_len = pack["_num_causal_tokens"]
            gen_len = pack["_num_full_tokens"]
            kv_to_store = (
                k_gen_[:gen_len].unsqueeze(0),
                v_gen[:gen_len].unsqueeze(0),
                k_und_[:und_len].unsqueeze(0),
                v_und[:und_len].unsqueeze(0),
            )

        # Apply projections directly to get final results
        und_seq = self.o_proj(get_und_seq(packed_attn_output))  # [N_und,hidden_size]
        gen_seq = self.o_proj_moe_gen(get_gen_seq(packed_attn_output))  # [N_gen,hidden_size]
        return from_und_gen_splits(und_seq, gen_seq, pack), kv_to_store  # [N_und+N_gen,hidden_size]


def _impl_init(
    self, config: Qwen3VLTextConfig | Qwen3VLMoeTextConfig | Nemotron3DenseVLTextConfig, layer_types: LayerTypes
):
    """
    Common implementation for Qwen3VLTextModel, Qwen3VLMoeTextModel, and Nemotron3DenseVLTextModel __init__.
    """
    self.padding_idx = config.pad_token_id
    self.vocab_size = config.vocab_size
    assert "Mo" in config.layer_module, "Only MoT layers are supported"

    # Text configuration for decoder layers

    # Embeddings from Qwen3VL base
    self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)

    self.layers = nn.ModuleList(
        [MoTDecoderLayer(config, layer_idx, layer_types) for layer_idx in range(config.num_hidden_layers)]
    )

    # Layer norm and rotary embeddings (text-only optimized)
    self.norm = layer_types.rms_norm(config.hidden_size, eps=config.rms_norm_eps)

    # Pathway-specific normalization
    self.norm_moe_gen = layer_types.rms_norm(config.hidden_size, eps=config.rms_norm_eps)

    # Rotary embedding (text-only optimized)
    self.rotary_emb = layer_types.rotary_embedding(config)

    # Initialize weights and apply final processing
    self.post_init()


def _impl_init_taylorseer(self, cache_dic=None, current=None):
    """
    Initialize TaylorSeer acceleration attributes.
    Common implementation for Qwen3VLTextModel.init_taylorseer and Qwen3VLMoeTextModel.init_taylorseer
    """
    self.cache_dic = cache_dic or {}
    self.current = current or {
        "step": 0,
        "type": "full",
        "stream": "layers_stream",
        "layer": 0,
        "module": "total",
        "activated_steps": [0],
    }
    # Enable TaylorSeer flag
    self.enable_taylorseer = True


def _impl_forward(
    self,
    pack: FactoredSequencePack,
    attention_mask,
    position_ids: torch.Tensor,
    natten_metadata_list: list | None = None,
    memory: MemoryState | None = None,
) -> tuple[FactoredSequencePack, dict[str, LBLMetadata]]:
    """
    Training forward pass - Attempted port from qwen2_mot
    Common implementation for Qwen3VLTextModel.forward_train and Qwen3VLMoeTextModel.forward_train

    Args:
        pack: Packed sequence
        attention_mask: Attention mask
        position_ids: Position IDs
        natten_metadata_list: Optional per-layer NATTEN metadata.
        memory: Optional MemoryState for persistent memory across forward passes.
    """

    # Create position embeddings (Qwen3 style) - squeeze once at model level
    # tensor below is only used for its dtype and device
    device, dtype = get_device_and_dtype(pack)
    _meta_tensor = torch.tensor([], dtype=dtype, device=device)
    cos, sin = self.rotary_emb(
        _meta_tensor, position_ids=position_ids.unsqueeze(0) if position_ids.ndim == 1 else position_ids.unsqueeze(1)
    )  # if ndim == 2, then the mrope position_ids is (3, seq_len), we need to put batch dimension in the middle to make it compatible with the rotary_emb
    # cos, sin: [1,N,head_dim] (1D pos_ids) or [3,1,N,head_dim] (mrope pos_ids)
    cos = cos.squeeze(0)  # [N,head_dim] or [3,N,head_dim]
    sin = sin.squeeze(0)  # [N,head_dim] or [3,N,head_dim]
    position_embeddings = (
        from_joint(cos, pack),
        from_joint(sin, pack),
    )

    # Tracking the load balancing loss across all layers. For dense models, lbl_metadata_all
    # will be a dictionary with empty lists for each pathway. For MoE models, the lists
    # for each pathway will be populated with the load balancing loss metadata for each layer.
    lbl_metadata_all = dict(und=[], gen=[])

    hidden_states = pack

    # --- MemoryState: per-step init (outside compile) ---
    if memory is not None:
        memory.init(hidden_states, device)

    # Derive gen_only once (outside compile) if using MemoryState
    memory_gen_only = memory.is_gen_only() if memory is not None else False

    for i, decoder_layer in enumerate(self.layers):
        # MemoryState: produce read-only MemoryValue for this layer (outside compile)
        memory_value = memory.read_for_layer(i) if memory is not None else None

        hidden_states, lbl_metadata_dict, kv_to_store = decoder_layer(
            hidden_states,
            attention_mask,
            position_embeddings,
            natten_metadata=None if natten_metadata_list is None else natten_metadata_list[i],
            memory_value=memory_value,
            gen_only=memory_gen_only,
        )

        # MemoryState: store K/V produced by this layer (outside compile)
        if kv_to_store is not None and memory is not None:
            memory.write_for_layer(i, kv_to_store)

        for pathway, lbl_metadata in lbl_metadata_dict.items():
            lbl_metadata_all[pathway].append(lbl_metadata)

    # Compute the load balancing loss across all layers. For dense models, final_lbl_metadata
    # will be an empty dictionary. For MoE models, it will be a dictionary with the stacked
    # load balancing loss metadata for each pathway.
    final_lbl_metadata: dict[str, LBLMetadata] = dict()
    for pathway, lbl_metadata_list in lbl_metadata_all.items():
        if len(lbl_metadata_list) > 0:
            num_tokens_per_expert = torch.stack(
                [lbl_metadata.num_tokens_per_expert for lbl_metadata in lbl_metadata_list]
            )  # [num_layers,num_experts]
            num_tokens = torch.stack([lbl_metadata.num_tokens for lbl_metadata in lbl_metadata_list])  # [num_layers]
            mean_router_prob_per_expert = torch.stack(
                [lbl_metadata.mean_router_prob_per_expert for lbl_metadata in lbl_metadata_list]
            )  # [num_layers,num_experts]
            final_lbl_metadata[pathway] = LBLMetadata(
                num_tokens_per_expert=num_tokens_per_expert,
                num_tokens=num_tokens,
                mean_router_prob_per_expert=mean_router_prob_per_expert,
            )

    hidden_states_out = zeros_like(hidden_states)
    set_und_seq(hidden_states_out, self.norm(get_und_seq(hidden_states)))  # [N_und,hidden_size]
    set_gen_seq(hidden_states_out, self.norm_moe_gen(get_gen_seq(hidden_states)))  # [N_gen,hidden_size]

    return hidden_states_out, final_lbl_metadata


def _run_mlp(
    mlp: torch.nn.Module,
    input: torch.Tensor,
) -> tuple[torch.Tensor, LBLMetadata | None]:
    if isinstance(mlp, Qwen3VLMoeTextSparseMoeBlock):
        (
            output_tensor,
            lbl_metadata,
        ) = mlp(input)
    else:
        output_tensor = mlp(input)
        lbl_metadata = None
    return output_tensor, lbl_metadata


class MoTDecoderLayer(nn.Module):
    """
    Unified MoT (Mixture of Transformers) decoder layer.
    Features dual-pathway attention for understanding vs generation.

    This is used for both Dense and MoE models.
    """

    def __init__(
        self,
        config: Qwen3VLTextConfig | Qwen3VLMoeTextConfig | Nemotron3DenseVLTextConfig,
        layer_idx: int,
        layer_types: LayerTypes,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.freeze_und = config.freeze_und
        self.self_attn = PackedAttentionMoT(config, layer_idx, layer_types)

        if (
            hasattr(config, "mlp_only_layers")
            and (layer_idx not in config.mlp_only_layers)
            and (config.num_experts > 0 and (layer_idx + 1) % config.decoder_sparse_step == 0)
        ):
            self.mlp = Qwen3VLMoeTextSparseMoeBlock(config)
            self.mlp_moe_gen = Qwen3VLMoeTextSparseMoeBlock(config)
        else:
            self.mlp = layer_types.mlp(config)
            self.mlp_moe_gen = layer_types.mlp(config)

        self.input_layernorm = layer_types.rms_norm(config.hidden_size, eps=config.rms_norm_eps)
        self.input_layernorm_moe_gen = layer_types.rms_norm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = layer_types.rms_norm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm_moe_gen = layer_types.rms_norm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input: FactoredSequencePack,
        attention_mask,
        packed_position_embeddings: Tuple[FactoredSequencePack, FactoredSequencePack],
        natten_metadata: dict | None = None,
        memory_value: MemoryValue | None = None,
        gen_only: bool = False,
    ) -> tuple[FactoredSequencePack, dict[str, LBLMetadata], KVToStore | None]:
        """Forward pass with MoT routing and optional memory-augmented attention.

        Returns a 3-tuple: ``(hidden_states, lbl_metadata_dict, kv_to_store)``.
        ``kv_to_store`` is non-None when ``memory_value`` is provided,
        containing ``(gen_k, gen_v, und_k, und_v)`` to be written back by
        ``MemoryState.write_for_layer()`` outside the ``torch.compile``
        boundary.

        Args:
            input: Packed sequence with und/gen tokens
            attention_mask: Attention mask
            packed_position_embeddings: RoPE embeddings (cos, sin)
            natten_metadata: Optional NATTEN metadata for neighborhood attention.
            memory_value: Read-only tensor container from MemoryState.read_for_layer().
            gen_only: When True, skip the understanding pathway (und K/V come from cache).
        """
        # Pre-Attention layernorm
        pack_norm_out = from_und_gen_splits(
            self.input_layernorm(get_und_seq(input)),  # [N_und,hidden_size]
            self.input_layernorm_moe_gen(get_gen_seq(input)),  # [N_gen,hidden_size]
            input,
        )  # [N_und+N_gen,hidden_size]

        # Self Attention + Residual
        kv_to_store: KVToStore | None = None
        if gen_only:
            assert natten_metadata is None
            # gen_only: skip und, compute gen tokens only (und K/V come from cache)
            _gen_norm = get_gen_seq(pack_norm_out)
            gen_pack = from_und_gen_splits(
                _gen_norm.new_empty(0, _gen_norm.shape[-1]),
                _gen_norm,
                pack_norm_out,
            )

            # Build position embeddings whose und length matches gen_pack's
            # und length (always 0).  Required when the outer pack carries
            # a padded causal_seq (``pad_for_cuda_graphs=True``): without
            # this, the und RoPE inside ``PackedAttentionMoT.forward``
            # would broadcast cos/sin of shape ``(MAX_CAUSAL_LEN, head_dim)``
            # onto a length-0 ``q_und`` / ``k_und`` and crash.  When the
            # outer pack is unpadded (eager AR path), the und cos/sin
            # already have length 0 and this slice is a no-op.
            _cos, _sin = packed_position_embeddings
            _empty_cos_und = get_und_seq(_cos)[:0]
            _empty_sin_und = get_und_seq(_sin)[:0]
            gen_position_embeddings = (
                from_und_gen_splits(_empty_cos_und, get_gen_seq(_cos), _cos),
                from_und_gen_splits(_empty_sin_und, get_gen_seq(_sin), _sin),
            )

            pack_attn_out, kv_to_store = self.self_attn(
                gen_pack,
                attention_mask,
                gen_position_embeddings,
                natten_metadata=natten_metadata,
                memory_value=memory_value,
            )
            gen_attn_out = get_gen_seq(pack_attn_out)
            residual_und = gen_attn_out.new_empty(0, gen_attn_out.shape[-1])
            residual_gen = get_gen_seq(input) + gen_attn_out
        else:
            # STANDARD PATH: Process both und and gen tokens
            pack_attn_out, kv_to_store = self.self_attn(
                pack_norm_out,
                attention_mask,
                packed_position_embeddings,
                natten_metadata=natten_metadata,
                memory_value=memory_value,
            )
            residual_und = get_und_seq(input) + get_und_seq(pack_attn_out)  # [N_und,hidden_size]
            residual_gen = get_gen_seq(input) + get_gen_seq(pack_attn_out)  # [N_gen,hidden_size]

        # Pre-MLP layernorm and processing
        lbl_metadata_dict: dict[str, LBLMetadata] = dict()

        if gen_only:
            # gen_only: skip und, compute gen tokens only
            ln_out_und = residual_gen.new_empty(0, residual_gen.shape[-1])
            ln_out_gen = self.post_attention_layernorm_moe_gen(residual_gen)

            # UNPAD MLP INPUT (gen only)
            gen_len = pack_attn_out["_num_full_tokens"]
            ln_out_gen_unpadded = ln_out_gen[:gen_len]  # [N_gen_unpadded,hidden_size]

            # Run MLP (gen only)
            mlp_out_gen_unpadded, lbl_metadata_gen = _run_mlp(self.mlp_moe_gen, ln_out_gen_unpadded)
            # mlp_out_gen_unpadded: [N_gen_unpadded,hidden_size]

            # PAD MLP OUTPUT (gen only)
            mlp_out_gen = torch.cat([mlp_out_gen_unpadded, ln_out_gen[gen_len:]], dim=0)  # [N_gen,hidden_size]

            # Build metadata dict (no und metadata in optimized path)
            if lbl_metadata_gen is not None:
                lbl_metadata_dict["gen"] = lbl_metadata_gen

            # Final output with residual (gen only)
            mlp_out_und_seq = residual_gen.new_empty(0, residual_gen.shape[-1])
            mlp_out_gen_seq = residual_gen + mlp_out_gen
        else:
            # STANDARD PATH: Process both und and gen tokens
            ln_out_und = self.post_attention_layernorm(residual_und)  # [N_und,hidden_size]
            ln_out_gen = self.post_attention_layernorm_moe_gen(residual_gen)  # [N_gen,hidden_size]

            # UNPAD MLP INPUT ===============

            #       artificial expert inbalance due to routing padding tokens.
            gen_len = pack_attn_out["_num_full_tokens"]
            und_len = pack_attn_out["_num_causal_tokens"]
            ln_out_und_unpadded = ln_out_und[:und_len]  # [N_und_unpadded,hidden_size]
            ln_out_gen_unpadded = ln_out_gen[:gen_len]  # [N_gen_unpadded,hidden_size]

            mlp_out_und_unpadded, lbl_metadata_und = _run_mlp(self.mlp, ln_out_und_unpadded)
            # mlp_out_und_unpadded: [N_und_unpadded,hidden_size]
            mlp_out_gen_unpadded, lbl_metadata_gen = _run_mlp(self.mlp_moe_gen, ln_out_gen_unpadded)
            # mlp_out_gen_unpadded: [N_gen_unpadded,hidden_size]

            # PAD MLP OUTPUT ===============
            mlp_out_und = torch.cat([mlp_out_und_unpadded, ln_out_und[und_len:]], dim=0)  # [N_und,hidden_size]
            mlp_out_gen = torch.cat([mlp_out_gen_unpadded, ln_out_gen[gen_len:]], dim=0)  # [N_gen,hidden_size]

            if lbl_metadata_und is not None:
                lbl_metadata_dict["und"] = lbl_metadata_und
            if lbl_metadata_gen is not None:
                lbl_metadata_dict["gen"] = lbl_metadata_gen

            mlp_out_und_seq = residual_und + mlp_out_und  # [N_und,hidden_size]
            mlp_out_gen_seq = residual_gen + mlp_out_gen  # [N_gen,hidden_size]

        return from_und_gen_splits(mlp_out_und_seq, mlp_out_gen_seq, input), lbl_metadata_dict, kv_to_store


# Backward-compat alias: serialized checkpoint configs reference the old name.
Qwen3VLTextMoTDecoderLayer = MoTDecoderLayer


class Qwen3VLTextModel(Qwen3VLPreTrainedModel):
    """
    Qwen3VL text model for MoT with dense MLPs.
    This is a wrapper around the _impl_forward defined above,
    specialized for dense models.
    """

    def __init__(self, config: Qwen3VLMoeTextConfig):
        super().__init__(config)
        _impl_init(self, config, layer_types=LayerTypes("qwen3_vl_dense"))

    def init_taylorseer(self, cache_dic=None, current=None):
        _impl_init_taylorseer(self, cache_dic=cache_dic, current=current)

    def forward(self, *args, **kwargs):
        return _impl_forward(self, *args, **kwargs)


class Qwen3VLMoeTextModel(Qwen3VLMoePreTrainedModel):
    """
    Qwen3VL text model for MoT with MoE MLPs.
    This is a wrapper around the _impl_* helpers defined above,
    specialized for MoE models.
    """

    def __init__(self, config: Qwen3VLMoeTextConfig):
        super().__init__(config)
        _impl_init(self, config, layer_types=LayerTypes("qwen3_vl_moe"))

    def init_taylorseer(self, cache_dic=None, current=None):
        _impl_init_taylorseer(self, cache_dic=cache_dic, current=current)

    def forward(self, *args, **kwargs):
        return _impl_forward(self, *args, **kwargs)


class Qwen3VLTextForCausalLM(Qwen3VLPreTrainedModel):
    """
    Qwen3VL text causal language model for MoT.
    This variant is used for dense-only MLP models.
    """

    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: Qwen3VLTextConfig):
        super().__init__(config)
        self.model = Qwen3VLTextModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def init_moe(self) -> None:
        """Initialize MoE/MoT weights by copying understanding to generation pathway."""
        state_dict = self.state_dict()
        for name, param in self.named_parameters():
            if "moe_gen" in name:
                original_name = name.replace("_moe_gen", "").replace("_checkpoint_wrapped_module.", "")
                if original_name in state_dict:
                    param.data.copy_(state_dict[original_name].data)
                else:
                    raise ValueError(f"Could not find {original_name} in state_dict for initialization of {name}")

    def forward(
        self,
        pack: FactoredSequencePack,
        attention_mask,
        position_ids: torch.Tensor,
        natten_metadata_list: list | None = None,
        memory: MemoryState | None = None,
    ) -> tuple[FactoredSequencePack, dict[str, LBLMetadata]]:
        """Training forward pass - simplified to match qwen3_mot"""
        outputs = self.model(
            pack=pack,
            attention_mask=attention_mask,
            position_ids=position_ids,
            natten_metadata_list=natten_metadata_list,
            memory=memory,
        )
        return outputs


class Qwen3VLMoeTextForCausalLM(Qwen3VLMoePreTrainedModel):
    """
    Qwen3VL text causal language model for MoT with MoE on the generation pathway.
    This variant is used for MoE MLP models.
    """

    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: Qwen3VLMoeTextConfig):
        super().__init__(config)
        self.model = Qwen3VLMoeTextModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def init_moe(self) -> None:
        """Initialize MoE/MoT weights by copying understanding to generation pathway."""
        state_dict = self.state_dict()
        for name, param in self.named_parameters():
            if "moe_gen" in name:
                original_name = name.replace("_moe_gen", "").replace("_checkpoint_wrapped_module.", "")
                if original_name in state_dict:
                    param.data.copy_(state_dict[original_name].data)
                else:
                    raise ValueError(f"Could not find {original_name} in state_dict for initialization of {name}")

    def forward(
        self,
        pack: FactoredSequencePack,
        attention_mask,
        position_ids: torch.Tensor,
        natten_metadata_list: list | None = None,
        memory: MemoryState | None = None,
    ) -> tuple[FactoredSequencePack, dict[str, torch.Tensor]]:
        """Training forward pass - simplified to match qwen3_mot"""

        outputs = self.model(
            pack=pack,
            attention_mask=attention_mask,
            position_ids=position_ids,
            natten_metadata_list=natten_metadata_list,
            memory=memory,
        )

        return outputs


# -----------------------------------------------------------------------------
# Nemotron 3 Dense VL MoT model wrappers
# -----------------------------------------------------------------------------


class Nemotron3DenseVLTextModel(Nemotron3DenseVLPreTrainedModel):
    """Nemotron 3 Dense VL text model adapted for MoT training."""

    def __init__(self, config: Nemotron3DenseVLTextConfig) -> None:
        super().__init__(config)
        _impl_init(self, config, layer_types=LayerTypes("nemotron_dense"))

    def init_taylorseer(self, cache_dic=None, current=None) -> None:
        _impl_init_taylorseer(self, cache_dic=cache_dic, current=current)

    def forward(self, *args, **kwargs):
        return _impl_forward(self, *args, **kwargs)


class Nemotron3DenseVLTextForCausalLM(Nemotron3DenseVLPreTrainedModel):
    """Causal LM head on top of the Nemotron 3 Dense VL MoT text model."""

    _tied_weights_keys: list[str] = []

    def __init__(self, config: Nemotron3DenseVLTextConfig) -> None:
        super().__init__(config)
        self.model = Nemotron3DenseVLTextModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def init_moe(self) -> None:
        """Copy understanding-pathway weights into the generation-pathway parameters."""
        state_dict = self.state_dict()
        for name, param in self.named_parameters():
            if "moe_gen" not in name:
                continue
            original_name = name.replace("_moe_gen", "").replace("_checkpoint_wrapped_module.", "")
            if original_name in state_dict:
                param.data.copy_(state_dict[original_name].data)
            elif any(norm_key in original_name for norm_key in ("q_norm", "k_norm")):
                # qk_norm_for_text=False → q_norm/k_norm are nn.Identity() with no parameters;
                # the moe_gen counterpart (q_norm_moe_gen) is a real RMSNorm, so skip init here.
                pass
            else:
                raise ValueError(f"Could not find {original_name} in state_dict for initialization of {name}")

    def forward(
        self,
        pack: FactoredSequencePack,
        attention_mask,
        position_ids: torch.Tensor,
        natten_metadata_list: list | None = None,
        memory: MemoryState | None = None,
    ) -> tuple[FactoredSequencePack, dict[str, LBLMetadata]]:
        return self.model(
            pack=pack,
            attention_mask=attention_mask,
            position_ids=position_ids,
            natten_metadata_list=natten_metadata_list,
            memory=memory,
        )
