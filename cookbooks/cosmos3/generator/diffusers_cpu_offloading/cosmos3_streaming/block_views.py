from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
from torch import nn

from .attention.base import CosmosAttentionBackend

type RotaryPair = tuple[torch.Tensor, torch.Tensor]
type PrefixKV = tuple[torch.Tensor, torch.Tensor]
MEMORY_SERIAL_MLP_CHUNK_TOKENS = 8_192


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rotary(x: torch.Tensor, rotary: RotaryPair) -> torch.Tensor:
    cos, sin = rotary
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return x * cos + _rotate_half(x) * sin


class UnderstandingBlockView(nn.Module):
    """Understanding-only execution view over one original Cosmos decoder layer."""

    def __init__(self, layer: nn.Module, backend: CosmosAttentionBackend) -> None:
        super().__init__()
        attention = layer.self_attn

        # Register only understanding-side modules. Registering the complete
        # attention module would make its generation projections part of this
        # offload group as well.
        self.input_layernorm = layer.input_layernorm
        self.post_attention_layernorm = layer.post_attention_layernorm
        self.mlp = layer.mlp
        self.to_q = attention.to_q
        self.to_k = attention.to_k
        self.to_v = attention.to_v
        self.to_out = attention.to_out
        self.norm_q = attention.norm_q
        self.norm_k = attention.norm_k
        self.k_norm_und_for_gen = attention.k_norm_und_for_gen

        self.num_attention_heads = attention.num_attention_heads
        self.num_key_value_heads = attention.num_key_value_heads
        self.head_dim = attention.head_dim
        self._backend = backend

    def _one_branch(self, hidden: torch.Tensor, rotary: RotaryPair) -> tuple[torch.Tensor, PrefixKV]:
        normalized = self.input_layernorm(hidden)
        q = self.to_q(normalized).view(-1, self.num_attention_heads, self.head_dim)
        k = self.to_k(normalized).view(-1, self.num_key_value_heads, self.head_dim)
        v = self.to_v(normalized).view(-1, self.num_key_value_heads, self.head_dim)

        q = self.norm_q(q)
        k = self.norm_k(k)
        k_for_generation = self.k_norm_und_for_gen(k) if self.k_norm_und_for_gen is not None else k

        q = apply_rotary(q, rotary)
        k = apply_rotary(k, rotary)
        k_for_generation = apply_rotary(k_for_generation, rotary)

        attention_output = self._backend(q, k, v, is_causal=True).flatten(-2, -1)
        residual = hidden + self.to_out(attention_output)
        output = residual + self.mlp(self.post_attention_layernorm(residual))
        return output, (k_for_generation, v)

    def forward(
        self,
        conditional_hidden: torch.Tensor,
        conditional_rotary: RotaryPair,
        unconditional_hidden: torch.Tensor | None = None,
        unconditional_rotary: RotaryPair | None = None,
    ) -> tuple[
        torch.Tensor,
        PrefixKV,
        torch.Tensor | None,
        PrefixKV | None,
    ]:
        conditional_hidden, conditional_prefix = self._one_branch(conditional_hidden, conditional_rotary)

        unconditional_prefix = None
        if unconditional_hidden is not None:
            if unconditional_rotary is None:
                raise ValueError("unconditional_rotary is required with unconditional_hidden")
            unconditional_hidden, unconditional_prefix = self._one_branch(
                unconditional_hidden, unconditional_rotary
            )

        return conditional_hidden, conditional_prefix, unconditional_hidden, unconditional_prefix


class GenerationBlockView(nn.Module):
    """Generation-only execution view over one original Cosmos decoder layer."""

    def __init__(self, layer: nn.Module, backend: CosmosAttentionBackend) -> None:
        super().__init__()
        attention = layer.self_attn

        self.input_layernorm = layer.input_layernorm_moe_gen
        self.post_attention_layernorm = layer.post_attention_layernorm_moe_gen
        self.mlp = layer.mlp_moe_gen
        self.to_q = attention.add_q_proj
        self.to_k = attention.add_k_proj
        self.to_v = attention.add_v_proj
        self.to_out = attention.to_add_out
        self.norm_q = attention.norm_added_q
        self.norm_k = attention.norm_added_k

        self.num_attention_heads = attention.num_attention_heads
        self.num_key_value_heads = attention.num_key_value_heads
        self.head_dim = attention.head_dim
        self._backend = backend

    def _one_branch(
        self,
        hidden: torch.Tensor,
        rotary: RotaryPair,
        understanding_prefix: PrefixKV,
        *,
        mlp_chunk_size: int | None = None,
    ) -> torch.Tensor:
        normalized = self.input_layernorm(hidden)
        q = self.to_q(normalized).view(-1, self.num_attention_heads, self.head_dim)
        k = self.to_k(normalized).view(-1, self.num_key_value_heads, self.head_dim)
        v = self.to_v(normalized).view(-1, self.num_key_value_heads, self.head_dim)

        q = apply_rotary(self.norm_q(q), rotary)
        k = apply_rotary(self.norm_k(k), rotary)

        prefix_k, prefix_v = understanding_prefix
        all_k = torch.cat((prefix_k, k), dim=0)
        all_v = torch.cat((prefix_v, v), dim=0)
        attention_output = self._backend(q, all_k, all_v, is_causal=False).flatten(-2, -1)

        residual = hidden + self.to_out(attention_output)
        normalized = self.post_attention_layernorm(residual)
        if mlp_chunk_size is None or normalized.shape[0] <= mlp_chunk_size:
            return residual + self.mlp(normalized)

        # The unchunked 720p Transfer MLP needs a 1.31 GiB contiguous
        # intermediate. Process token rows independently and accumulate into
        # the existing residual allocation. Linear and pointwise MLP
        # operations do not mix information across the token dimension.
        for start in range(0, normalized.shape[0], mlp_chunk_size):
            end = min(start + mlp_chunk_size, normalized.shape[0])
            residual[start:end].add_(self.mlp(normalized[start:end]))
        return residual

    def forward(
        self,
        conditional_hidden: torch.Tensor,
        conditional_rotary: RotaryPair,
        conditional_prefix: PrefixKV,
        unconditional_hidden: torch.Tensor | None = None,
        unconditional_rotary: RotaryPair | None = None,
        unconditional_prefix: PrefixKV | None = None,
        extra_hiddens: list[torch.Tensor] | None = None,
        extra_rotaries: list[RotaryPair] | None = None,
        extra_prefixes: list[PrefixKV] | None = None,
        stage_branches: bool = False,
        execution_device: torch.device | None = None,
        offload_device: torch.device | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None] | tuple[torch.Tensor, torch.Tensor | None, list[torch.Tensor]]:
        def run_branch(hidden: torch.Tensor, rotary: RotaryPair, prefix: PrefixKV) -> torch.Tensor:
            if not stage_branches:
                return self._one_branch(hidden, rotary, prefix)
            if execution_device is None or offload_device is None:
                raise ValueError("execution_device and offload_device are required when staging branches")

            device_hidden = hidden.to(
                execution_device,
                non_blocking=hidden.device.type == "cpu" and hidden.is_pinned(),
            )
            device_output = self._one_branch(
                device_hidden,
                rotary,
                prefix,
                mlp_chunk_size=MEMORY_SERIAL_MLP_CHUNK_TOKENS,
            )
            if hidden.device == offload_device and hidden.shape == device_output.shape and hidden.dtype == device_output.dtype:
                hidden.copy_(device_output, non_blocking=hidden.device.type == "cpu" and hidden.is_pinned())
                return hidden
            return device_output.to(offload_device)

        conditional_hidden = run_branch(conditional_hidden, conditional_rotary, conditional_prefix)

        if unconditional_hidden is not None:
            if unconditional_rotary is None or unconditional_prefix is None:
                raise ValueError("unconditional rotary and prefix are required with unconditional_hidden")
            unconditional_hidden = run_branch(
                unconditional_hidden,
                unconditional_rotary,
                unconditional_prefix,
            )
        if extra_hiddens is not None:
            if extra_rotaries is None or extra_prefixes is None:
                raise ValueError("extra rotary embeddings and prefixes are required with extra branches")
            if not (len(extra_hiddens) == len(extra_rotaries) == len(extra_prefixes)):
                raise ValueError("extra generation branch inputs must have the same length")
            extra_hiddens = [
                run_branch(hidden, rotary, prefix)
                for hidden, rotary, prefix in zip(extra_hiddens, extra_rotaries, extra_prefixes)
            ]
            return conditional_hidden, unconditional_hidden, extra_hiddens
        return conditional_hidden, unconditional_hidden


class UnderstandingStack(nn.Module):
    """Runs both CFG understanding branches once and returns per-layer K/V."""

    def __init__(
        self,
        layers: Sequence[nn.Module],
        backend: CosmosAttentionBackend,
        block_callback: Callable[[str, int], None] | None = None,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(UnderstandingBlockView(layer, backend) for layer in layers)
        self._block_callback = block_callback
        # Diffusers streamed group offload needs an unmatched root group to own
        # the lazy-prefetch trace. Identity is intentionally parameter-free.
        self._prefetch_anchor = nn.Identity()

    def forward(
        self,
        conditional_hidden: torch.Tensor,
        conditional_rotary: RotaryPair,
        unconditional_hidden: torch.Tensor | None = None,
        unconditional_rotary: RotaryPair | None = None,
    ) -> tuple[list[PrefixKV], list[PrefixKV] | None]:
        conditional_cache: list[PrefixKV] = []
        unconditional_cache: list[PrefixKV] | None = [] if unconditional_hidden is not None else None

        for index, block in enumerate(self.blocks):
            conditional_hidden, conditional_prefix, unconditional_hidden, unconditional_prefix = block(
                conditional_hidden,
                conditional_rotary,
                unconditional_hidden,
                unconditional_rotary,
            )
            conditional_cache.append(conditional_prefix)
            if self._block_callback is not None:
                self._block_callback("understanding", index)
            if unconditional_cache is not None:
                if unconditional_prefix is None:
                    raise RuntimeError("understanding block did not return an unconditional prefix")
                unconditional_cache.append(unconditional_prefix)

        return conditional_cache, unconditional_cache


class GenerationStack(nn.Module):
    """Runs both CFG generation branches with one onload per decoder block."""

    def __init__(
        self,
        layers: Sequence[nn.Module],
        backend: CosmosAttentionBackend,
        block_callback: Callable[[str, int], None] | None = None,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(GenerationBlockView(layer, backend) for layer in layers)
        self._block_callback = block_callback
        self._prefetch_anchor = nn.Identity()

    def forward(
        self,
        conditional_hidden: torch.Tensor,
        conditional_rotary: RotaryPair,
        conditional_cache: Sequence[PrefixKV],
        unconditional_hidden: torch.Tensor | None = None,
        unconditional_rotary: RotaryPair | None = None,
        unconditional_cache: Sequence[PrefixKV] | None = None,
        extra_hiddens: list[torch.Tensor] | None = None,
        extra_rotaries: list[RotaryPair] | None = None,
        extra_caches: list[Sequence[PrefixKV]] | None = None,
        stage_branches: bool = False,
        execution_device: torch.device | None = None,
        offload_device: torch.device | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None] | tuple[torch.Tensor, torch.Tensor | None, list[torch.Tensor]]:
        if len(conditional_cache) != len(self.blocks):
            raise ValueError("conditional K/V cache does not match the decoder depth")
        if unconditional_hidden is not None and (
            unconditional_cache is None or len(unconditional_cache) != len(self.blocks)
        ):
            raise ValueError("unconditional K/V cache does not match the decoder depth")
        if extra_hiddens is not None:
            if extra_rotaries is None or extra_caches is None:
                raise ValueError("extra rotary embeddings and K/V caches are required with extra branches")
            if not (len(extra_hiddens) == len(extra_rotaries) == len(extra_caches)):
                raise ValueError("extra generation branch inputs must have the same length")
            if any(len(cache) != len(self.blocks) for cache in extra_caches):
                raise ValueError("an extra K/V cache does not match the decoder depth")

        for index, block in enumerate(self.blocks):
            unconditional_prefix = None if unconditional_cache is None else unconditional_cache[index]
            if extra_hiddens is None:
                conditional_hidden, unconditional_hidden = block(
                    conditional_hidden,
                    conditional_rotary,
                    conditional_cache[index],
                    unconditional_hidden,
                    unconditional_rotary,
                    unconditional_prefix,
                    stage_branches=stage_branches,
                    execution_device=execution_device,
                    offload_device=offload_device,
                )
            else:
                conditional_hidden, unconditional_hidden, extra_hiddens = block(
                    conditional_hidden,
                    conditional_rotary,
                    conditional_cache[index],
                    unconditional_hidden,
                    unconditional_rotary,
                    unconditional_prefix,
                    extra_hiddens,
                    extra_rotaries,
                    [cache[index] for cache in extra_caches],
                    stage_branches=stage_branches,
                    execution_device=execution_device,
                    offload_device=offload_device,
                )
            if self._block_callback is not None:
                self._block_callback("generation", index)
        if extra_hiddens is not None:
            return conditional_hidden, unconditional_hidden, extra_hiddens
        return conditional_hidden, unconditional_hidden
