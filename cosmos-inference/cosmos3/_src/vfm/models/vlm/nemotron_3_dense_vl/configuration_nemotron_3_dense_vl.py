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

"""Configuration for Nemotron 3 Dense VL text backbone (paired hybrid blocks -> standard layers)."""

from transformers.configuration_utils import PretrainedConfig


class Nemotron3DenseVLTextConfig(PretrainedConfig):
    """Text config for Nemotron-H style language model after pairing attn+MLP blocks (28 effective layers)."""

    model_type = "nemotron_3_dense_vl_text"

    def __init__(
        self,
        vocab_size: int = 131072,
        tie_word_embeddings: bool = False,
        hidden_size: int = 2048,
        intermediate_size: int = 9216,
        num_hidden_layers: int = 28,
        num_attention_heads: int = 16,
        head_dim: int = 128,
        num_key_value_heads: int = 8,
        mlp_hidden_act: str = "relu2",
        attention_bias: bool = False,
        mlp_bias: bool = False,
        initializer_range: float = 0.02,
        layer_norm_epsilon: float = 1e-5,
        residual_in_fp32: bool = False,
        use_cache: bool = True,
        num_logits_to_keep: int = 1,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 11,
        sliding_window: int | None = None,
        max_position_embeddings: int = 131072,
        attention_dropout: float = 0.0,
        hidden_dropout: float = 0.0,
        enable_rope: bool = True,
        rope_scaling: dict | None = None,
        rope_theta: float = 100_000_000.0,
        enable_mrope: bool = True,
        mrope_section: list[int] | None = None,
        torch_dtype: str = "bfloat16",
        **kwargs,
    ) -> None:
        self.vocab_size = vocab_size
        self.tie_word_embeddings = tie_word_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.num_key_value_heads = num_key_value_heads
        self.mlp_hidden_act = mlp_hidden_act
        self.attention_bias = attention_bias
        self.mlp_bias = mlp_bias
        self.initializer_range = initializer_range
        self.layer_norm_epsilon = layer_norm_epsilon
        self.residual_in_fp32 = residual_in_fp32
        self.use_cache = use_cache
        self.num_logits_to_keep = num_logits_to_keep
        self.sliding_window = sliding_window
        self.max_position_embeddings = max_position_embeddings
        self.attention_dropout = attention_dropout
        self.hidden_dropout = hidden_dropout
        self.rope_scaling = rope_scaling
        self.rope_theta = rope_theta
        self.enable_rope = enable_rope
        self.enable_mrope = enable_mrope
        self.mrope_section = mrope_section if mrope_section is not None else [24, 20, 20]
        self.torch_dtype = torch_dtype
        self._attn_implementation = kwargs.pop("_attn_implementation", "eager")
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    @property
    def rms_norm_eps(self) -> float:
        """Alias for Qwen-style MoT code paths."""
        return self.layer_norm_epsilon
