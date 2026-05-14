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

"""FLOPs estimation for the Wan 2.2 VAE encoder (Encoder3d)."""

from decimal import Decimal


def compute_wan_vae_encoder_flops(
    B: int | Decimal,
    T: int,
    H: int,
    W: int,
    *,
    dim: int = 160,
    z_dim: int = 48,
    dim_mult: list[int] | None = None,
    num_res_blocks: int = 2,
    temperal_downsample: list[bool] | None = None,
) -> Decimal:
    """Compute forward-pass FLOPs for the Wan 2.2 VAE encoder (Encoder3d).

    The encoder converts a pixel-space video [B, 3, T, H, W] into a latent
    [B, z_dim, T//4, H//16, W//16].  It is frozen during training so only
    forward-pass FLOPs are counted (no backward).

    The architecture: patchify(2) -> conv1 -> 4 downsample stages (each with
    ``num_res_blocks`` residual blocks + optional spatial/temporal downsample)
    -> middle block (ResBlock + single-head spatial attention + ResBlock)
    -> head (RMSNorm + SiLU + conv) -> pointwise 1x1 conv.

    Args:
        B: Batch size.
        T: Number of pixel-space temporal frames.
        H: Pixel-space height (must be divisible by 16).
        W: Pixel-space width (must be divisible by 16).
        dim: Base channel dimension of the encoder (default 160).
        z_dim: Latent channel dimension (default 48, encoder outputs 2*z_dim).
        dim_mult: Channel multiplier per stage (default [1, 2, 4, 4]).
        num_res_blocks: Residual blocks per downsample stage (default 2).
        temperal_downsample: Per-stage temporal downsampling flags (default
            [False, True, True]).

    Returns:
        Total forward-pass FLOPs as a Decimal.
    """
    if dim_mult is None:
        dim_mult = [1, 2, 4, 4]
    if temperal_downsample is None:
        temperal_downsample = [False, True, True]

    B = int(B)
    flops = Decimal(0)

    def _causalconv3d_flops(c_in: int, c_out: int, kt: int, kh: int, kw: int, bt: int, bh: int, bw: int) -> int:
        return 2 * c_out * c_in * kt * kh * kw * B * bt * bh * bw

    def _resblock_flops(in_dim: int, out_dim: int, bt: int, bh: int, bw: int) -> int:
        vol = B * bt * bh * bw
        f = 0
        f += 5 * in_dim * vol  # RMS_norm(in_dim)
        f += 2 * out_dim * in_dim * 27 * vol  # CausalConv3d(in_dim, out_dim, 3)
        f += 5 * out_dim * vol  # RMS_norm(out_dim)
        f += 2 * out_dim * out_dim * 27 * vol  # CausalConv3d(out_dim, out_dim, 3)
        if in_dim != out_dim:
            f += 2 * out_dim * in_dim * vol  # shortcut CausalConv3d(in_dim, out_dim, 1)
        return f

    def _attnblock_flops(d: int, bt: int, bh: int, bw: int) -> int:
        vol = B * bt * bh * bw
        seq = bh * bw
        f = 0
        f += 5 * d * vol  # RMS_norm
        f += 2 * (d * 3) * d * vol  # to_qkv Conv2d(d, 3d, 1)
        f += 4 * B * bt * seq * seq * d  # QK^T + Attn*V
        f += 2 * d * d * vol  # proj Conv2d(d, d, 1)
        return f

    # After patchify(patch_size=2): [B, 12, T, H/2, W/2]
    t, h, w = T, H // 2, W // 2

    # conv1: CausalConv3d(12, dims[0], 3)
    dims = [dim * u for u in [1] + dim_mult]  # [160, 160, 320, 640, 640]
    flops += _causalconv3d_flops(12, dims[0], 3, 3, 3, t, h, w)

    # Downsample stages
    for i, (in_d, out_d) in enumerate(zip(dims[:-1], dims[1:])):
        t_down = temperal_downsample[i] if i < len(temperal_downsample) else False
        down_flag = i != len(dim_mult) - 1

        cur_in = in_d
        for _ in range(num_res_blocks):
            flops += _resblock_flops(cur_in, out_d, t, h, w)
            cur_in = out_d

        if down_flag:
            if t_down:
                h_new, w_new = h // 2, w // 2
                flops += 2 * out_d * out_d * 9 * B * t * h_new * w_new  # spatial conv2d
                t_new = t // 2
                flops += 2 * out_d * out_d * 3 * B * t_new * h_new * w_new  # temporal conv3d(3,1,1)
                t, h, w = t_new, h_new, w_new
            else:
                h_new, w_new = h // 2, w // 2
                flops += 2 * out_d * out_d * 9 * B * t * h_new * w_new
                h, w = h_new, w_new

    # Middle block: ResBlock + AttentionBlock + ResBlock
    mid_dim = dims[-1]
    flops += _resblock_flops(mid_dim, mid_dim, t, h, w)
    flops += _attnblock_flops(mid_dim, t, h, w)
    flops += _resblock_flops(mid_dim, mid_dim, t, h, w)

    # Head: RMS_norm + SiLU + CausalConv3d(mid_dim, z_dim*2, 3)
    enc_out_dim = z_dim * 2
    flops += 5 * mid_dim * B * t * h * w  # RMS_norm
    flops += _causalconv3d_flops(mid_dim, enc_out_dim, 3, 3, 3, t, h, w)

    # WanVAE_.conv1: CausalConv3d(z_dim*2, z_dim*2, 1) — pointwise 1x1
    flops += _causalconv3d_flops(enc_out_dim, enc_out_dim, 1, 1, 1, t, h, w)

    return Decimal(flops)
