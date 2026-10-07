from __future__ import annotations

import torch
import torch.nn.functional as F


class TorchSdpaBackend:
    """BF16 reference backend using PyTorch scaled dot-product attention."""

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        is_causal: bool,
    ) -> torch.Tensor:
        if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
            raise ValueError("q, k, and v must use [tokens, heads, dim] layout")
        if k.shape != v.shape:
            raise ValueError(f"k and v shapes must match, got {tuple(k.shape)} and {tuple(v.shape)}")
        if q.shape[-1] != k.shape[-1]:
            raise ValueError("q and k head dimensions must match")
        if q.shape[1] % k.shape[1] != 0:
            raise ValueError("query head count must be divisible by KV head count")

        # torch SDPA consumes [batch, heads, tokens, dim].
        q_bhsd = q.transpose(0, 1).unsqueeze(0)
        k_bhsd = k.transpose(0, 1).unsqueeze(0)
        v_bhsd = v.transpose(0, 1).unsqueeze(0)

        try:
            output = F.scaled_dot_product_attention(
                q_bhsd,
                k_bhsd,
                v_bhsd,
                is_causal=is_causal,
                enable_gqa=q.shape[1] != k.shape[1],
            )
        except TypeError:
            # Compatibility fallback for PyTorch releases without enable_gqa.
            groups = q.shape[1] // k.shape[1]
            if groups != 1:
                k_bhsd = k_bhsd.repeat_interleave(groups, dim=1)
                v_bhsd = v_bhsd.repeat_interleave(groups, dim=1)
            output = F.scaled_dot_product_attention(
                q_bhsd,
                k_bhsd,
                v_bhsd,
                is_causal=is_causal,
            )

        return output.squeeze(0).transpose(0, 1).contiguous()
