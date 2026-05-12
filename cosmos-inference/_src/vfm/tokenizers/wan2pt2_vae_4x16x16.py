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

# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.

import os
import time
from collections.abc import Callable, Generator, Mapping, Sequence
from contextlib import contextmanager, nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from cosmos3._src.imaginaire.flags import DEVICE, INTERNAL, TRAINING
from cosmos3._src.imaginaire.utils import log
from cosmos3._src.imaginaire.utils.distributed import get_rank, sync_model_states
from cosmos3._src.imaginaire.utils.easy_io import easy_io
from cosmos3._src.vfm.datasets.utils import VIDEO_RES_SIZE_INFO
from cosmos3._src.vfm.tokenizers.interface import VideoTokenizerInterface
from cosmos3._src.vfm.utils.data_utils import get_vision_data_resolution

# For sequential decoding, CACHE_T is the number of frames to cache.
CACHE_T = 2


def _contiguous_clone(t: torch.Tensor) -> torch.Tensor:
    """Return a contiguous copy of *t* using exactly one allocation.

    When *t* is already contiguous, ``.contiguous()`` would be a no-op that
    returns the *same* tensor (sharing storage), so we need ``.clone()``.
    When *t* is non-contiguous, ``.contiguous()`` already allocates a fresh
    tensor with independent storage — no extra ``.clone()`` needed.
    """
    if t.is_contiguous():
        return t.clone()
    return t.contiguous()


def _update_cache_and_apply(
    x: torch.Tensor,
    layer: "CausalConv3d",
    feat_cache: list,
    feat_idx: list[int],
) -> torch.Tensor:
    """Apply a CausalConv3d with temporal cache management.

    Saves the last CACHE_T frames of ``x`` as the new cache entry and,
    when the current chunk has fewer than 2 frames, prepends the last
    cached frame so the cache always spans 2 frames.

    Note that feat_idx is a list with a single element, which stores
    the index of the current CausalConv3d layer. List is used here so
    feat_idx can be mutated in place, and the caller can pass in a reference
    to the list.
    """
    idx = feat_idx[0]
    cache_x = _contiguous_clone(x[:, :, -CACHE_T:, :, :])
    if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
        cache_x = torch.cat(
            [
                feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device),
                cache_x,
            ],
            dim=2,
        )  # [B,C,2,H,W]
    x = layer(x, feat_cache[idx])
    feat_cache[idx] = cache_x
    feat_idx[0] += 1
    return x


class CausalConv3d(nn.Conv3d):
    """
    Causal 3d convolution.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._padding = (
            self.padding[2],
            self.padding[2],
            self.padding[1],
            self.padding[1],
            2 * self.padding[0],
            0,
        )
        self.padding = (0, 0, 0)

    def forward(self, x, cache_x=None):  # x: [B,C,T,H,W]
        padding = list(self._padding)
        if cache_x is not None and self._padding[4] > 0:
            cache_x = cache_x.to(x.device)
            x = torch.cat([cache_x, x], dim=2)  # [B,C,T+cache_T,H,W]
            padding[4] -= cache_x.shape[2]
        x = F.pad(x, padding)  # [B,C,T_padded,H_padded,W_padded]

        return super().forward(x)  # [B,out_C,T_out,H_out,W_out]


class RMS_norm(nn.Module):
    def __init__(self, dim, channel_first=True, images=True, bias=False):
        super().__init__()
        broadcastable_dims = (1, 1, 1) if not images else (1, 1)
        shape = (dim, *broadcastable_dims) if channel_first else (dim,)

        self.channel_first = channel_first
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else 0.0

    def forward(self, x):
        return F.normalize(x, dim=(1 if self.channel_first else -1)) * self.scale * self.gamma + self.bias


class Upsample(nn.Upsample):
    def forward(self, x):
        """
        Fix bfloat16 support for nearest neighbor interpolation.
        """
        return super().forward(x.float()).type_as(x)


class Resample(nn.Module):
    def __init__(self, dim, mode):
        assert mode in (
            "none",
            "upsample2d",
            "upsample3d",
            "downsample2d",
            "downsample3d",
        )
        super().__init__()
        self.dim = dim
        self.mode = mode

        # layers
        if mode == "upsample2d":
            self.resample = nn.Sequential(
                Upsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
                nn.Conv2d(dim, dim, 3, padding=1),
            )
        elif mode == "upsample3d":
            self.resample = nn.Sequential(
                Upsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
                nn.Conv2d(dim, dim, 3, padding=1),
            )
            self.time_conv = CausalConv3d(dim, dim * 2, (3, 1, 1), padding=(1, 0, 0))
        elif mode == "downsample2d":
            self.resample = nn.Sequential(nn.ZeroPad2d((0, 1, 0, 1)), nn.Conv2d(dim, dim, 3, stride=(2, 2)))
        elif mode == "downsample3d":
            self.resample = nn.Sequential(nn.ZeroPad2d((0, 1, 0, 1)), nn.Conv2d(dim, dim, 3, stride=(2, 2)))
            self.time_conv = CausalConv3d(dim, dim, (3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0))
        else:
            self.resample = nn.Identity()

    def forward(self, x, feat_cache=None, feat_idx=[0]):  # x: [B,C,T,H,W]
        b, c, t, h, w = x.size()
        if self.mode == "upsample3d":
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    # First frame: skip time_conv, seed cache with zeros so the next call sees a real tensor
                    feat_cache[idx] = torch.zeros(b, c, CACHE_T, h, w, device=x.device, dtype=x.dtype)  # [B,C,2,H,W]
                    feat_idx[0] += 1
                else:
                    cache_x = _contiguous_clone(x[:, :, -CACHE_T:, :, :])  # [B,C,<=2,H,W]
                    if cache_x.shape[2] < 2:
                        cache_x = torch.cat(
                            [
                                feat_cache[idx][:, :, -1, :, :].unsqueeze(2),
                                cache_x,
                            ],
                            dim=2,
                        )  # [B,C,2,H,W]
                    x = self.time_conv(x, feat_cache[idx])  # [B,C*2,T,H,W]
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1
                    x = x.reshape(b, 2, c, t, h, w)  # [B,2,C,T,H,W]
                    x = torch.stack((x[:, 0, :, :, :, :], x[:, 1, :, :, :, :]), 3)  # [B,C,T,2,H,W]
                    x = x.reshape(b, c, t * 2, h, w)  # [B,C,T*2,H,W]
        t = x.shape[2]
        x = rearrange(x, "b c t h w -> (b t) c h w")  # [B*T,C,H,W]
        x = self.resample(x)  # [B*T,C_out,H_out,W_out]
        x = rearrange(x, "(b t) c h w -> b c t h w", t=t)  # [B,C_out,T,H_out,W_out]

        if self.mode == "downsample3d":
            # Important for torch.compile: when we're *not* doing streaming/cache-based inference
            # (feat_cache is None), we still need to apply the temporal downsample conv.
            if feat_cache is None:
                # `time_conv` has kernel (3,1,1), stride 2 in time, and no internal temporal padding.
                # In the streaming path, we effectively provide left temporal context via cached frames.
                # For the non-streaming path, pad 2 frames on the left so:
                # - the conv is always valid (T>=3)
                # - the output temporal length matches the shortcut path's ceil(T/2) behavior
                x = F.pad(x, (0, 0, 0, 0, 2, 0))  # [B,C,T+2,H_out,W_out]
                x = self.time_conv(x)  # [B,C,T//2+1,H_out,W_out]
            else:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    # First call for this layer in a streaming/windowed pass.
                    # The baseline streaming path primes caches with a single-frame chunk (T==1),
                    # where skipping time_conv preserves both correctness and shape alignment.
                    #
                    # If this is ever called with T>1 (non-standard chunking), fall back to a padded
                    # time_conv so the main path stays compatible with the shortcut path.
                    if x.shape[2] == 1:
                        feat_cache[idx] = _contiguous_clone(x)
                    else:
                        cache_x = _contiguous_clone(x[:, :, -1:, :, :])  # [B,C,1,H_out,W_out]
                        x_in = F.pad(x, (0, 0, 0, 0, 2, 0))  # [B,C,T+2,H_out,W_out]
                        x = self.time_conv(x_in)  # [B,C,T//2+1,H_out,W_out]
                        feat_cache[idx] = cache_x
                    feat_idx[0] += 1
                else:
                    cache_x = _contiguous_clone(x[:, :, -1:, :, :])  # [B,C,1,H_out,W_out]
                    x_cat = torch.cat([feat_cache[idx][:, :, -1:, :, :], x], 2)  # [B,C,T+1,H_out,W_out]
                    t_cat = x_cat.shape[2]
                    if t_cat < 3:
                        x_cat = F.pad(x_cat, (0, 0, 0, 0, 3 - t_cat, 0))  # [B,C,3,H_out,W_out]
                    x = self.time_conv(x_cat)  # [B,C,T//2+1,H_out,W_out]
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.0):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        # layers
        self.residual = nn.Sequential(
            RMS_norm(in_dim, images=False),
            nn.SiLU(),
            CausalConv3d(in_dim, out_dim, 3, padding=1),
            RMS_norm(out_dim, images=False),
            nn.SiLU(),
            nn.Dropout(dropout),
            CausalConv3d(out_dim, out_dim, 3, padding=1),
        )
        self.shortcut = CausalConv3d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        h = self.shortcut(x)
        for layer in self.residual:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                x = _update_cache_and_apply(x, layer, feat_cache, feat_idx)
            else:
                x = layer(x)
        return x + h


class AttentionBlock(nn.Module):
    """
    Causal self-attention with a single head.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        # layers
        self.norm = RMS_norm(dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

        # zero out the last layer params
        nn.init.zeros_(self.proj.weight)

    def forward(self, x):  # x: [B,C,T,H,W]
        identity = x
        b, c, t, h, w = x.size()
        x = rearrange(x, "b c t h w -> (b t) c h w")  # [B*T,C,H,W]
        x = self.norm(x)  # [B*T,C,H,W]
        # compute query, key, value
        q, k, v = self.to_qkv(x).reshape(b * t, 1, c * 3, -1).permute(0, 1, 3, 2).contiguous().chunk(3, dim=-1)
        # q,k,v: [B*T,1,H*W,C]

        # apply attention
        x = F.scaled_dot_product_attention(
            q,
            k,
            v,
        )  # [B*T,1,H*W,C]
        x = x.squeeze(1).permute(0, 2, 1).contiguous().reshape(b * t, c, h, w)  # [B*T,C,H,W]

        # output
        x = self.proj(x)  # [B*T,C,H,W]
        x = rearrange(x, "(b t) c h w-> b c t h w", t=t)  # [B,C,T,H,W]
        return x + identity  # [B,C,T,H,W]


def patchify(x, patch_size):  # x: [B,C,H,W] or [B,C,T,H,W] -> [B,C*p^2,H//p,W//p] or [B,C*p^2,T,H//p,W//p]
    if patch_size == 1:
        return x
    # Fast path: patch_size==2 is the only one used in this tokenizer.
    # Implement it with pure view/permute/reshape to be maximally torch.compile-friendly.
    if patch_size == 2:
        if x.dim() == 4:
            b, c, h, w = x.shape
            x = x.view(b, c, h // 2, 2, w // 2, 2)  # [B,C,H//2,2,W//2,2]
            x = x.permute(0, 1, 5, 3, 2, 4).contiguous()  # [B,C,2,2,H//2,W//2]
            return x.view(b, c * 4, h // 2, w // 2)  # [B,C*4,H//2,W//2]
        if x.dim() == 5:
            b, c, f, h, w = x.shape
            x = x.view(b, c, f, h // 2, 2, w // 2, 2)  # [B,C,T,H//2,2,W//2,2]
            x = x.permute(0, 1, 6, 4, 2, 3, 5).contiguous()  # [B,C,2,2,T,H//2,W//2]
            return x.view(b, c * 4, f, h // 2, w // 2)  # [B,C*4,T,H//2,W//2]
    if x.dim() == 4:
        x = rearrange(x, "b c (h q) (w r) -> b (c r q) h w", q=patch_size, r=patch_size)  # [B,C*p^2,H//p,W//p]
    elif x.dim() == 5:
        x = rearrange(
            x,
            "b c f (h q) (w r) -> b (c r q) f h w",
            q=patch_size,
            r=patch_size,
        )  # [B,C*p^2,T,H//p,W//p]
    else:
        raise ValueError(f"Invalid input shape: {x.shape}")

    return x


def unpatchify(x, patch_size):  # x: [B,C*p^2,H,W] or [B,C*p^2,T,H,W] -> [B,C,H*p,W*p] or [B,C,T,H*p,W*p]
    if patch_size == 1:
        return x

    if x.dim() == 4:
        x = rearrange(x, "b (c r q) h w -> b c (h q) (w r)", q=patch_size, r=patch_size)  # [B,C,H*p,W*p]
    elif x.dim() == 5:
        x = rearrange(
            x,
            "b (c r q) f h w -> b c f (h q) (w r)",
            q=patch_size,
            r=patch_size,
        )  # [B,C,T,H*p,W*p]
    return x


class AvgDown3D(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        factor_t,
        factor_s=1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.factor_t = factor_t
        self.factor_s = factor_s
        self.factor = self.factor_t * self.factor_s * self.factor_s

        assert in_channels * self.factor % out_channels == 0
        self.group_size = in_channels * self.factor // out_channels

    def forward(
        self, x: torch.Tensor
    ) -> torch.Tensor:  # x: [B,C,T,H,W] -> [B,out_channels,T//factor_t,H//factor_s,W//factor_s]
        pad_t = (self.factor_t - x.shape[2] % self.factor_t) % self.factor_t
        pad = (0, 0, 0, 0, pad_t, 0)
        x = F.pad(x, pad)  # [B,C,T_padded,H,W]
        B, C, T, H, W = x.shape
        x = x.view(
            B,
            C,
            T // self.factor_t,
            self.factor_t,
            H // self.factor_s,
            self.factor_s,
            W // self.factor_s,
            self.factor_s,
        )  # [B,C,T//ft,ft,H//fs,fs,W//fs,fs]
        x = x.permute(0, 1, 3, 5, 7, 2, 4, 6).contiguous()  # [B,C,ft,fs,fs,T//ft,H//fs,W//fs]
        x = x.view(
            B,
            C * self.factor,
            T // self.factor_t,
            H // self.factor_s,
            W // self.factor_s,
        )  # [B,C*factor,T//ft,H//fs,W//fs]
        x = x.view(
            B,
            self.out_channels,
            self.group_size,
            T // self.factor_t,
            H // self.factor_s,
            W // self.factor_s,
        )  # [B,out_channels,group_size,T//ft,H//fs,W//fs]
        x = x.mean(dim=2)  # [B,out_channels,T//ft,H//fs,W//fs]
        return x


class DupUp3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        factor_t,
        factor_s=1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.factor_t = factor_t
        self.factor_s = factor_s
        self.factor = self.factor_t * self.factor_s * self.factor_s

        assert out_channels * self.factor % in_channels == 0
        self.repeats = out_channels * self.factor // in_channels

    def forward(
        self, x: torch.Tensor, first_chunk=False
    ) -> torch.Tensor:  # x: [B,in_channels,T,H,W] -> [B,out_channels,T*factor_t,H*factor_s,W*factor_s]
        x = x.repeat_interleave(self.repeats, dim=1)  # [B,in_channels*repeats,T,H,W]
        x = x.view(
            x.size(0),
            self.out_channels,
            self.factor_t,
            self.factor_s,
            self.factor_s,
            x.size(2),
            x.size(3),
            x.size(4),
        )  # [B,out_channels,ft,fs,fs,T,H,W]
        x = x.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()  # [B,out_channels,T,ft,H,fs,W,fs]
        x = x.view(
            x.size(0),
            self.out_channels,
            x.size(2) * self.factor_t,
            x.size(4) * self.factor_s,
            x.size(6) * self.factor_s,
        )  # [B,out_channels,T*ft,H*fs,W*fs]
        if first_chunk:
            x = x[:, :, self.factor_t - 1 :, :, :]  # [B,out_channels,T*ft-ft+1,H*fs,W*fs]
        return x


class Down_ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout, mult, temperal_downsample=False, down_flag=False):
        super().__init__()

        # Shortcut path with downsample
        self.avg_shortcut = AvgDown3D(
            in_dim,
            out_dim,
            factor_t=2 if temperal_downsample else 1,
            factor_s=2 if down_flag else 1,
        )

        # Main path with residual blocks and downsample
        downsamples = []
        for _ in range(mult):
            downsamples.append(ResidualBlock(in_dim, out_dim, dropout))
            in_dim = out_dim

        # Add the final downsample block
        if down_flag:
            mode = "downsample3d" if temperal_downsample else "downsample2d"
            downsamples.append(Resample(out_dim, mode=mode))

        self.downsamples = nn.Sequential(*downsamples)

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        # Avoid cloning the full activation (extra kernel + bandwidth).
        # None of the downstream modules are in-place, so taking the shortcut first is safe.
        x_shortcut = self.avg_shortcut(x)
        for module in self.downsamples:
            x = module(x, feat_cache, feat_idx)

        return x + x_shortcut


class Up_ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout, mult, temperal_upsample=False, up_flag=False):
        super().__init__()
        # Shortcut path with upsample
        if up_flag:
            self.avg_shortcut = DupUp3D(
                in_dim,
                out_dim,
                factor_t=2 if temperal_upsample else 1,
                factor_s=2 if up_flag else 1,
            )
        else:
            self.avg_shortcut = None

        # Main path with residual blocks and upsample
        upsamples = []
        for _ in range(mult):
            upsamples.append(ResidualBlock(in_dim, out_dim, dropout))
            in_dim = out_dim

        # Add the final upsample block
        if up_flag:
            mode = "upsample3d" if temperal_upsample else "upsample2d"
            upsamples.append(Resample(out_dim, mode=mode))

        self.upsamples = nn.Sequential(*upsamples)

    def forward(self, x, feat_cache=None, feat_idx=[0], first_chunk=False):
        x_shortcut = self.avg_shortcut(x, first_chunk) if self.avg_shortcut is not None else None
        for module in self.upsamples:
            x = module(x, feat_cache, feat_idx)
        if x_shortcut is not None:
            return x + x_shortcut
        return x


class Encoder3d(nn.Module):
    def __init__(
        self,
        dim=128,
        z_dim=4,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_downsample=[False, True, True],
        dropout=0.0,
    ):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_downsample = temperal_downsample

        # dimensions
        dims = [dim * u for u in [1] + dim_mult]
        scale = 1.0

        # init block
        self.conv1 = CausalConv3d(12, dims[0], 3, padding=1)

        # downsample blocks
        downsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            t_down_flag = temperal_downsample[i] if i < len(temperal_downsample) else False
            downsamples.append(
                Down_ResidualBlock(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    dropout=dropout,
                    mult=num_res_blocks,
                    temperal_downsample=t_down_flag,
                    down_flag=i != len(dim_mult) - 1,
                )
            )
            scale /= 2.0
        self.downsamples = nn.Sequential(*downsamples)

        # middle blocks
        self.middle = nn.Sequential(
            ResidualBlock(out_dim, out_dim, dropout),
            AttentionBlock(out_dim),
            ResidualBlock(out_dim, out_dim, dropout),
        )

        # output blocks
        self.head = nn.Sequential(
            RMS_norm(out_dim, images=False),
            nn.SiLU(),
            CausalConv3d(out_dim, z_dim, 3, padding=1),
        )

    def forward(self, x, feat_cache=None):  # x: [B,12,T,H//2,W//2] -> [B,z_dim,T//4,H//16,W//16]
        feat_idx = [0]

        if feat_cache is not None:
            x = _update_cache_and_apply(x, self.conv1, feat_cache, feat_idx)  # [B,dim,T,H//2,W//2]
        else:
            x = self.conv1(x)  # [B,dim,T,H//2,W//2]

        # downsamples
        for layer in self.downsamples:
            if feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)
        # x: [B,dim*dim_mult[-1],T//4,H//16,W//16]

        # middle
        for layer in self.middle:
            if isinstance(layer, ResidualBlock) and feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)
        # x: [B,dim*dim_mult[-1],T//4,H//16,W//16]

        # head
        for layer in self.head:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                x = _update_cache_and_apply(x, layer, feat_cache, feat_idx)
            else:
                x = layer(x)

        return x  # [B,z_dim,T//4,H//16,W//16]


class Decoder3d(nn.Module):
    def __init__(
        self,
        dim=128,
        z_dim=4,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_upsample=[True, True, False],
        dropout=0.0,
    ):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_upsample = temperal_upsample

        # dimensions
        dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]

        # init block
        self.conv1 = CausalConv3d(z_dim, dims[0], 3, padding=1)

        # middle blocks
        self.middle = nn.Sequential(
            ResidualBlock(dims[0], dims[0], dropout),
            AttentionBlock(dims[0]),
            ResidualBlock(dims[0], dims[0], dropout),
        )

        # upsample blocks
        upsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            t_up_flag = temperal_upsample[i] if i < len(temperal_upsample) else False
            upsamples.append(
                Up_ResidualBlock(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    dropout=dropout,
                    mult=num_res_blocks + 1,
                    temperal_upsample=t_up_flag,
                    up_flag=i != len(dim_mult) - 1,
                )
            )
        self.upsamples = nn.Sequential(*upsamples)

        # output blocks
        self.head = nn.Sequential(
            RMS_norm(out_dim, images=False),
            nn.SiLU(),
            CausalConv3d(out_dim, 12, 3, padding=1),
        )

    def forward(self, x, feat_cache=None, first_chunk=False):  # x: [B,z_dim,T,H,W] -> [B,12,T*4,H*8,W*8]
        feat_idx = [0]

        if feat_cache is not None:
            x = _update_cache_and_apply(x, self.conv1, feat_cache, feat_idx)  # [B,dim*dim_mult[-1],T,H,W]
        else:
            x = self.conv1(x)  # [B,dim*dim_mult[-1],T,H,W]

        for layer in self.middle:
            if isinstance(layer, ResidualBlock) and feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)
        # x: [B,dim*dim_mult[-1],T,H,W]

        # upsamples
        for layer in self.upsamples:
            if feat_cache is not None:
                x = layer(x, feat_cache, feat_idx, first_chunk)
            else:
                x = layer(x)
        # x: [B,dec_dim,T*4,H*8,W*8]

        # head
        for layer in self.head:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                x = _update_cache_and_apply(x, layer, feat_cache, feat_idx)
            else:
                x = layer(x)
        return x  # [B,12,T*4,H*8,W*8]


def count_conv3d(model: nn.Module) -> int:
    return sum(1 for m in model.modules() if isinstance(m, CausalConv3d))


class WanVAE_(nn.Module):
    def __init__(
        self,
        dim=160,
        dec_dim=256,
        z_dim=48,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_downsample=[False, True, True],
        dropout=0.0,
        temporal_window: int | Mapping[str, int] = 4,
        encode_exact_durations: list[int] | None = None,
    ):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_downsample = temperal_downsample
        self.temperal_upsample = temperal_downsample[::-1]
        self.temporal_window = temporal_window
        self._encode_exact_durations: set[int] = set(encode_exact_durations or [])

        # modules
        self.encoder = Encoder3d(
            dim,
            z_dim * 2,
            dim_mult,
            num_res_blocks,
            attn_scales,
            self.temperal_downsample,
            dropout,
        )
        self.conv1 = CausalConv3d(z_dim * 2, z_dim * 2, 1)
        self.conv2 = CausalConv3d(z_dim, z_dim, 1)
        self.decoder = Decoder3d(
            dec_dim,
            z_dim,
            dim_mult,
            num_res_blocks,
            attn_scales,
            self.temperal_upsample,
            dropout,
        )

        self._enc_conv_num = count_conv3d(self.encoder)
        self._dec_conv_num = count_conv3d(self.decoder)
        self._dec_cache: list[torch.Tensor | None] = self._new_dec_cache()

    def _new_enc_cache(self) -> list:
        """Fresh per-layer cache for the encoder (one slot per CausalConv3d)."""
        return [None] * self._enc_conv_num

    def _new_dec_cache(self) -> list:
        """Fresh per-layer cache for the decoder (one slot per CausalConv3d)."""
        return [None] * self._dec_conv_num

    def forward(self, x, scale):
        mu = self.encode(x, scale)
        x_recon = self.decode(mu, scale, clear_decoder_cache=True)
        return x_recon, mu

    def _normalize_latent(self, z: torch.Tensor, scale: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Normalize the latent."""
        assert len(scale) == 2, "scale must be a tuple with two tensors"
        s0 = scale[0].view(1, self.z_dim, 1, 1, 1)
        s1 = scale[1].view(1, self.z_dim, 1, 1, 1)
        return (z - s0) * s1  # [B,z_dim,T,H,W]

    def _denormalize_latent(self, z: torch.Tensor, scale: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Invert the normalization applied by _encode_features_to_mu."""
        assert len(scale) == 2, "scale must be a tuple with two tensors"
        s0 = scale[0].view(1, self.z_dim, 1, 1, 1)
        s1 = scale[1].view(1, self.z_dim, 1, 1, 1)
        return z / s1 + s0  # [B,z_dim,T,H,W]

    def _encode_chunk_impl(
        self,
        x_chunk: torch.Tensor,
        feat_cache: list[torch.Tensor | None],
        scale: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, list[torch.Tensor | None]]:
        """Run the encoder on one temporal chunk and normalize the output.

        Defined as an instance method (not a closure inside ``encode``) so
        that ``_ChunkEncodeForAOT`` can wrap it for ``torch.export``.

        Note: Since ``feat_cache`` is mutated in-place by the encoder (each
        ``CausalConv3d`` layer overwrites its slot), we pass a shallow copy
        to preserve the original cache for compilation.
        """
        feat_cache = list(feat_cache)

        assert all(c is None or c.is_contiguous() for c in feat_cache)
        assert x_chunk.is_contiguous()

        out = self.encoder(x_chunk, feat_cache=feat_cache)

        assert out.is_contiguous()
        assert all(c is None or c.is_contiguous() for c in feat_cache)

        # Project encoder features through conv1, split to mu/log_var, and normalize.
        mu, _log_var = self.conv1(out).chunk(2, dim=1)
        return self._normalize_latent(mu, scale), feat_cache

    def encode(self, x: torch.Tensor, scale: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Chunked causal encoding that converts pixel-space video to latent space.

        The Wan 2.2 VAE encoder uses causal 3D convolutions, which means each
        ``CausalConv3d`` layer needs temporal context from previous frames.  For
        long videos, running the full sequence at once would create intermediate
        tensors of shape ``(B*T, C, H, W)`` that can exceed Triton's int32
        indexing limit.  Instead, we split the video into fixed-size temporal
        chunks and maintain a per-layer feature cache (``feat_cache``) so each
        chunk can access the causal context from previous chunks.

        **Encoding strategy:**

        1. The first frame is encoded alone (the "key-frame prime") to seed
           the causal caches with initial state.
        2. Subsequent frames are processed in chunks of ``temporal_window`` (e.g.
           68 frames).  Each chunk reads from and writes to the shared
           ``feat_cache`` list, which stores the last ``CACHE_T=2`` frames of
           activations per ``CausalConv3d`` layer.
        3. The per-chunk latents are concatenated along the temporal axis.

        **AOT compilation:**

        When ``_aot_chunk_fns`` has been installed (via
        :meth:`Wan2pt2VAEInterface.compile_encode`),
        each chunk is dispatched to a pre-compiled ``.pt2`` function keyed by
        ``(T_chunk, H_patch, W_patch, cache_t)``.  Padding ensures every chunk
        (except possibly the last, handled by ``should_pad``) has exactly
        ``temporal_window`` frames, keeping the set of compiled input shapes small.

        Args:
            x: Pixel-space video tensor of shape ``[B, 3, T, H, W]``.
                ``T`` must satisfy ``T == 1`` or ``(T - 1) % 4 == 0`` (the
                4x temporal compression constraint).  Use ``pad_video_batch``
                to pad to a valid length before calling.
            scale: Tuple of ``(mean, inv_std)`` tensors, each of shape
                ``[z_dim]``.  Used to normalize the latent: ``(z - mean) * inv_std``.

        Returns:
            Normalized latent tensor of shape ``[B, z_dim, T_latent, H//16, W//16]``
            where ``T_latent = 1 + (T - 1) // 4``.  Always corresponds to the
            *original* (unpadded) input length, even when internal padding was applied.
        """
        T, H, W = x.shape[2], x.shape[3], x.shape[4]

        # ``temporal_window`` can be a per-resolution mapping (e.g.
        # {"256": 68, "480": 32, "720": 16}) from a Hydra/OmegaConf config,
        # which arrives as a ``DictConfig`` (not a plain ``dict``).
        # Using ``Mapping`` catches both.
        if isinstance(self.temporal_window, Mapping):
            resolution = get_vision_data_resolution((H, W))
            temporal_window = self.temporal_window[resolution]
        else:
            temporal_window = self.temporal_window

        assert T == 1 or (T - 1) % 4 == 0, (
            f"Input temporal length must be 4n+1 (got {T}). "
            "Use pad_video_batch to pad before encoding, check wan2pt2_vae_4x16x16_test on how to use it."
        )

        # The 4x temporal compression maps T pixel frames → ceil-like latent frames.
        # For T=1 (single image), latent_T=1.  For T=4n+1, latent_T=n+1.
        latent_T = 1 + (T - 1) // 4

        # Certain short-clip durations (e.g. robotics datasets with T=17) can be
        # encoded at their exact length, avoiding the overhead of padding to the
        # next chunk boundary.  All other lengths are padded so that each chunk
        # has exactly ``temporal_window`` frames, giving the compiled function a
        # fixed input shape per {resolution, aspect_ratio} bucket.
        should_pad = T not in self._encode_exact_durations

        if should_pad:
            # Pad T to ``1 + k * temporal_window`` so that after removing the 1-frame
            # prime, the remaining frames divide evenly into ``temporal_window``-sized chunks.
            T = 1 + ((T - 1 + temporal_window - 1) // temporal_window) * temporal_window
            x = F.pad(x, (0, 0, 0, 0, 0, T - x.shape[2]))

        # One cache slot per CausalConv3d layer in the encoder, initially all None.
        enc_cache = self._new_enc_cache()

        # Patchify merges each 2×2 spatial patch into the channel dim:
        # [B, 3, T, H, W] → [B, 12, T, H//2, W//2].
        x = patchify(x, patch_size=2)

        aot_chunk_fns: dict | None = getattr(self, "_aot_chunk_fns", None)
        H_patch, W_patch = x.shape[3], x.shape[4]

        def _run_chunk(
            x_chunk: torch.Tensor,
            feat_cache: list[torch.Tensor | None],
        ) -> tuple[torch.Tensor, list[torch.Tensor | None]]:
            """Encode one chunk, through the AOT path or eager fallback.

            If AOT-compiled chunk functions are installed, the function is
            looked up by ``(T_chunk, H_patch, W_patch, cache_t)`` where
            ``cache_t`` is the minimum temporal extent of the cache tensors
            (0 = all-None prime, 1 = post-prime, 2 = steady state).

            Both full-size and remainder chunks (from ``encode_exact_durations``)
            are dispatched through the AOT path when a matching compiled
            function exists; uncompiled shapes fall back to eager.
            """
            is_prime = feat_cache[0] is None

            # Ensure contiguity so AOT-compiled functions receive tensors
            # with deterministic strides regardless of the source slice.
            x_chunk = x_chunk.contiguous()

            if aot_chunk_fns is not None:
                cache_t = 0 if is_prime else feat_cache[0].shape[2]
                aot_key = (x_chunk.shape[2], H_patch, W_patch, cache_t)
                aot_fn = aot_chunk_fns.get(aot_key)

                if aot_fn is not None:
                    return aot_fn(x_chunk, feat_cache)

            return self._encode_chunk_impl(x_chunk, feat_cache, scale)

        # --- Chunked encoding loop ---
        # Chunk 0: single-frame "key-frame prime" to seed all causal caches.
        out, enc_cache = _run_chunk(x[:, :, :1], feat_cache=enc_cache)
        outs = [out]

        # Chunks 1..N: process the remaining frames in fixed-size windows.
        for start in range(1, T, temporal_window):
            x_chunk = x[:, :, start : start + temporal_window]
            out, enc_cache = _run_chunk(x_chunk, feat_cache=enc_cache)
            outs.append(out)

        final_out = torch.cat(outs, dim=2) if len(outs) > 1 else outs[0]

        # If we padded the input, trim the latent back to the original length.
        if should_pad:
            final_out = final_out[:, :, :latent_T]
        return final_out

    def decode(
        self,
        z,
        scale,
        clear_decoder_cache: bool,
    ):  # z: [B,z_dim,T_latent,H_latent,W_latent] -> [B,3,T,H,W]
        if clear_decoder_cache:
            self._dec_cache = self._new_dec_cache()

        z = self._denormalize_latent(z, scale)
        x = self.conv2(z)  # [B,z_dim,T_latent,H_latent,W_latent]

        parts = []
        for i in range(x.shape[2]):
            first_chunk = (i == 0) and all(c is None for c in self._dec_cache)
            parts.append(
                self.decoder(
                    x[:, :, i : i + 1],
                    feat_cache=self._dec_cache,
                    first_chunk=first_chunk,
                )
            )

        if clear_decoder_cache:
            self._dec_cache = self._new_dec_cache()

        decoded = unpatchify(torch.cat(parts, dim=2), patch_size=2)  # [B,3,T,H,W]
        return decoded  # [B,3,T,H,W]

    def clear_decoder_cache(self) -> None:
        self._dec_cache = self._new_dec_cache()


def _video_vae(
    pretrained_path=None,
    device="cpu",
    object_store_credential_path_pretrained="",
    temporal_window: int | Mapping[str, int] = 4,
    encode_exact_durations: list[int] | None = None,
):
    """
    Autoencoder3d adapted from Wan 2.2.
    """
    # init model
    with torch.device("meta"):
        model = WanVAE_(
            temporal_window=temporal_window,
            encode_exact_durations=encode_exact_durations,
        )
    if not TRAINING:
        model.to_empty(device=device)

    if pretrained_path is None:
        model.to_empty(device=device)
    else:
        if get_rank() == 0:
            if not INTERNAL:
                from cosmos3._src.imaginaire.utils.checkpoint_db import download_checkpoint_v2

                pretrained_path = download_checkpoint_v2(pretrained_path)
            if pretrained_path.startswith("s3://"):
                backend_args = {
                    "backend": "s3",
                    "s3_credential_path": object_store_credential_path_pretrained,
                }
            else:
                backend_args = None

            ckpt = easy_io.load(
                pretrained_path,
                backend_args=backend_args,
                map_location=device,
            )

            # load checkpoint
            log.info(f"loading {pretrained_path}")
            model.load_state_dict(ckpt, assign=TRAINING)
        else:
            model.to_empty(device=device)
    sync_model_states(model)

    return model


class WanVAE:
    def __init__(
        self,
        z_dim=48,
        vae_pth="",
        object_store_credential_path_pretrained="",
        dtype=torch.bfloat16,
        device=DEVICE,
        is_amp=True,
        temporal_window: int | Mapping[str, int] = 4,
        encode_exact_durations: list[int] | None = None,
    ):
        self.dtype = dtype
        self.device = device

        # Wan 2.2 mean and std values (48 dimensions)
        mean = [
            -0.2289,
            -0.0052,
            -0.1323,
            -0.2339,
            -0.2799,
            0.0174,
            0.1838,
            0.1557,
            -0.1382,
            0.0542,
            0.2813,
            0.0891,
            0.1570,
            -0.0098,
            0.0375,
            -0.1825,
            -0.2246,
            -0.1207,
            -0.0698,
            0.5109,
            0.2665,
            -0.2108,
            -0.2158,
            0.2502,
            -0.2055,
            -0.0322,
            0.1109,
            0.1567,
            -0.0729,
            0.0899,
            -0.2799,
            -0.1230,
            -0.0313,
            -0.1649,
            0.0117,
            0.0723,
            -0.2839,
            -0.2083,
            -0.0520,
            0.3748,
            0.0152,
            0.1957,
            0.1433,
            -0.2944,
            0.3573,
            -0.0548,
            -0.1681,
            -0.0667,
        ]
        std = [
            0.4765,
            1.0364,
            0.4514,
            1.1677,
            0.5313,
            0.4990,
            0.4818,
            0.5013,
            0.8158,
            1.0344,
            0.5894,
            1.0901,
            0.6885,
            0.6165,
            0.8454,
            0.4978,
            0.5759,
            0.3523,
            0.7135,
            0.6804,
            0.5833,
            1.4146,
            0.8986,
            0.5659,
            0.7069,
            0.5338,
            0.4889,
            0.4917,
            0.4069,
            0.4999,
            0.6866,
            0.4093,
            0.5709,
            0.6065,
            0.6415,
            0.4944,
            0.5726,
            1.2042,
            0.5458,
            1.6887,
            0.3971,
            1.0600,
            0.3943,
            0.5537,
            0.5444,
            0.4089,
            0.7468,
            0.7744,
        ]

        mean = torch.tensor(mean, dtype=dtype, device=device)  # [z_dim]
        std = torch.tensor(std, dtype=dtype, device=device)  # [z_dim]
        self.scale = (mean, 1.0 / std)

        # init model
        self.model = _video_vae(
            pretrained_path=vae_pth,
            object_store_credential_path_pretrained=object_store_credential_path_pretrained,
            device=device,
            temporal_window=temporal_window,
            encode_exact_durations=encode_exact_durations,
        )

        self.model = self.model.eval().requires_grad_(False)
        self.is_amp = is_amp
        if not is_amp:
            self.model = self.model.to(dtype=dtype)
            self.context = nullcontext()
        else:
            self.context = torch.amp.autocast("cuda", dtype=dtype)

    def count_param(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    @torch.no_grad()
    def encode(self, videos: torch.Tensor) -> torch.Tensor:
        """Encode a batch of videos.

        AOT-compiled chunk functions (if installed on ``self.model`` via
        :meth:`Wan2pt2VAEInterface.compile_encode`)
        are dispatched from inside ``WanVAE_.encode`` at the per-chunk level,
        preserving the chunked encoding loop and temporal padding logic.

        Args:
            videos: Tensor of shape ``[B, C, T, H, W]``.

        Returns:
            Tensor of shape ``[B, z_dim, T//4, H//16, W//16]``.
        """
        in_dtype = videos.dtype
        with self.context:
            if not self.is_amp:
                videos = videos.to(self.dtype)
            latent = self.model.encode(videos, self.scale)
        latent = latent.to(in_dtype)
        return latent

    @torch.no_grad()
    def decode(self, zs: torch.Tensor, clear_decoder_cache: bool = True) -> torch.Tensor:
        """Decode a batch of latent tensors.

        Args:
            zs: Tensor of shape ``[B, z_dim, T, H, W]``.
            clear_decoder_cache: Whether to clear the decoder cache between decode calls.

        Returns:
            Tensor of shape ``[B, C, T, H, W]``.
        """
        in_dtype = zs.dtype
        with self.context:
            if not self.is_amp:
                zs = zs.to(self.dtype)
            video_recon = self.model.decode(zs, self.scale, clear_decoder_cache)
        video_recon = video_recon.to(in_dtype)
        return video_recon


# ---------------------------------------------------------------------------
# AOT compilation helpers
# ---------------------------------------------------------------------------

_ShapeInfo = tuple[int, int, int]  # (chunk_frames, H_patch, W_patch)
_AOTChunkKey = tuple[int, int, int, int]  # (T_chunk, H_patch, W_patch, cache_t)


class _ChunkEncodeForAOT(torch.nn.Module):
    """Wrapper around ``WanVAE_._encode_chunk_impl`` for ``torch.export``.

    Absorbs the ``(mean, inv_std)`` scale as registered buffers so the
    exported signature is just ``(x_chunk, feat_cache)``.  A separate
    wrapper instance (and export) is created per ``cache_t`` because the
    ``None``-pattern in ``feat_cache`` differs between ``cache_t=0``
    (all ``None``) and ``cache_t>=1`` (some tensors, some ``None``).
    """

    def __init__(
        self,
        vae_model: torch.nn.Module,
        scale_mean: torch.Tensor,
        scale_inv_std: torch.Tensor,
    ) -> None:
        super().__init__()
        self.vae = vae_model
        self.register_buffer("scale_mean", scale_mean.clone())
        self.register_buffer("scale_inv_std", scale_inv_std.clone())

    def forward(
        self,
        x_chunk: torch.Tensor,
        feat_cache: list[torch.Tensor | None],
    ) -> tuple[torch.Tensor, list[torch.Tensor | None]]:
        return self.vae._encode_chunk_impl(
            x_chunk,
            feat_cache,
            (self.scale_mean, self.scale_inv_std),
        )


def _collect_warmup_shapes(
    tokenizer: "Wan2pt2VAEInterface",
    warmup_resolutions: Sequence[str],
    aspect_ratio: str | None = None,
) -> list[_ShapeInfo]:
    """Return ``[(chunk_frames, H_patch, W_patch), ...]`` for all warmup shapes.

    Expands *warmup_resolutions* into concrete spatial shapes using
    ``VIDEO_RES_SIZE_INFO``.  Each resolution may have multiple aspect
    ratios (e.g. ``"16,9"``, ``"9,16"``, ``"1,1"``); optionally filtered
    to a single ratio via *aspect_ratio*.  ``chunk_frames`` is looked up
    from the tokenizer (scalar or per-resolution dict).  Spatial
    dimensions are halved to account for patchify (``patch_size=2``).
    """
    all_shapes: list[_ShapeInfo] = []
    for res_key in warmup_resolutions:
        if res_key not in VIDEO_RES_SIZE_INFO:
            raise ValueError(f"Resolution {res_key} not found in VIDEO_RES_SIZE_INFO")

        if isinstance(tokenizer.encode_chunk_frames, Mapping):
            if res_key not in tokenizer.encode_chunk_frames:
                raise ValueError(f"Resolution {res_key} not found in tokenizer.encode_chunk_frames")

        res_dict = VIDEO_RES_SIZE_INFO[res_key]
        if aspect_ratio is not None:
            if aspect_ratio not in res_dict:
                raise ValueError(f"Aspect ratio {aspect_ratio} not found in resolution {res_key}")
            res_dict = {aspect_ratio: res_dict[aspect_ratio]}

        for H, W in res_dict.values():
            if isinstance(tokenizer.encode_chunk_frames, Mapping):
                chunk_frames = tokenizer.encode_chunk_frames[res_key]
            else:
                chunk_frames = tokenizer.encode_chunk_frames

            H_patch, W_patch = H // 2, W // 2
            all_shapes.append((chunk_frames, H_patch, W_patch))
    return all_shapes


class Wan2pt2VAEInterface(VideoTokenizerInterface):
    def __init__(
        self,
        bucket_name: str = "",
        object_store_credential_path_pretrained: str = "",
        vae_path: str = "",
        chunk_duration: int = 93,
        keep_decoder_cache: bool = False,
        use_streaming_encode: bool = False,
        # Granularity of the encoding chunks. Larger values result in higher TensorCore utilization,
        # and lower values result in lower memory usage. To optimize for speed and memory usage,
        # use a dictionary of chunk frames, one for each resolution. If a single integer is provided,
        # it will be used for all resolutions.
        encode_chunk_frames: int | Mapping[str, int] = 4,
        # Exact frame durations that get encoded without padding. Useful for short-clip datasets
        # (e.g. robotics) where the standard bucketing would inflate the input by many multiples
        # (e.g. 17 frames → 69 with encode_chunk_frames=68). Must be a list of integers.
        encode_exact_durations: list[int] | None = None,
        # Compression factors for spatial and temporal dimensions (4x16x16 tokenizer).
        spatial_compression_factor: int = 16,
        temporal_compression_factor: int = 4,
        # Deprecated arguments. This is kept for backwards compatibility
        # with older configurations.
        temporal_window: int | None = None,
        encode_bucket_multiple: int | None = None,
    ):
        # Remove temporal_window and encode_bucket_multiple once they have been
        # removed from the uploaded HuggingFace checkpoint.
        if temporal_window is not None:
            log.warning("temporal_window is deprecated; remove it.")
        del temporal_window

        if encode_bucket_multiple is not None:
            log.warning("encode_bucket_multiple is deprecated; remove it.")
        del encode_bucket_multiple

        # Remove special handling for encode_chunk_frames once the uploaded
        # HuggingFace checkpoint has been updated to use a dictionary of chunk frames,
        # one for each resolution.
        if isinstance(encode_chunk_frames, int):
            encode_chunk_frames = {"256": 68, "480": 24, "720": 12}
        assert isinstance(encode_chunk_frames, Mapping)

        assert all(c % 4 == 0 for c in encode_chunk_frames.values()), "encode_chunk_frames must be a multiple of 4"

        self.chunk_duration = chunk_duration

        vae_path_full = f"s3://{bucket_name}/{vae_path}"
        self.model = WanVAE(
            dtype=torch.bfloat16,
            is_amp=False,
            vae_pth=vae_path_full,
            object_store_credential_path_pretrained=object_store_credential_path_pretrained,
            temporal_window=encode_chunk_frames,
            encode_exact_durations=encode_exact_durations,
        )
        self.encode_chunk_frames = encode_chunk_frames
        self.encode_exact_durations = encode_exact_durations

        # When True, keep the decoder cache between decode calls.
        self._keep_decoder_cache = keep_decoder_cache

        # When True, always use the streaming/chunked encode path (correct but typically slower).
        self.use_streaming_encode = use_streaming_encode

        self._spatial_compression_factor = spatial_compression_factor
        self._temporal_compression_factor = temporal_compression_factor

    @property
    def dtype(self) -> torch.dtype:
        return self.model.dtype

    def reset_dtype(self) -> None:
        pass

    @contextmanager
    def use_cached_decoder(self) -> Generator[None, None, None]:
        """Enable decoder-cache reuse for sequential decode calls."""
        self.model.model.clear_decoder_cache()
        self._keep_decoder_cache = True
        try:
            yield
        finally:
            self.model.model.clear_decoder_cache()
            self._keep_decoder_cache = False

    def encode(self, state: torch.Tensor) -> torch.Tensor:
        """Encode a batch of videos.

        Args:
            state: Tensor of shape ``[B, C, T, H, W]``.

        Returns:
            Tensor of shape ``[B, z_dim, T//4, H//16, W//16]``.
        """
        return self.model.encode(state)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode a batch of latent tensors.

        Args:
            latent: Tensor of shape ``[B, z_dim, T, H, W]``.

        Returns:
            Tensor of shape ``[B, C, T, H, W]``.
        """
        return self.model.decode(
            latent,
            clear_decoder_cache=not self._keep_decoder_cache,
        )  # [B,3,T,H,W]

    @torch.no_grad()
    def compile_encode(
        self,
        warmup_resolutions: Sequence[str],
        output_dir: str,
        aspect_ratio: str | None = None,
    ) -> None:
        """AOT-compile the tokenizer's chunk-level encode for every resolution.

        Compiles ``WanVAE_._encode_chunk_impl`` for each
        ``(resolution, aspect_ratio, cache_t)`` variant, producing ``.pt2``
        packages that are loaded on all ranks for zero-overhead dispatch
        during training.

        **Variant enumeration** — for each resolution, three standard
        ``cache_t`` variants (prime, post-prime, steady-state) are compiled.
        When ``encode_exact_durations`` is configured, additional remainder
        variants are appended.

        **Distribution** — individual variants are assigned round-robin
        across ranks.  Reference caches are built lazily.

        **Shared weights** — packages are compiled with
        ``package_constants_in_so=False``.  After loading, each runner
        receives the same encoder weights via
        ``load_constants(user_managed=True)``.

        Compiled functions are installed as ``self.model.model._aot_chunk_fns``
        for dispatch by ``_run_chunk`` inside ``WanVAE_.encode``.

        Args:
            warmup_resolutions: Resolution keys (e.g. ``["256", "480", "720"]``).
            output_dir: Root directory under which compiled ``.pt2`` packages
                are written (an ``aot_tokenizer/`` subdirectory will be
                created).  Typically the job's local output path
                (``config.job.path_local``).
            aspect_ratio: If given, only compile this single aspect ratio per
                resolution instead of all available ratios.
        """
        import torch._inductor
        import torch.distributed as dist

        log.info(f"AOT chunk-level warmup for resolutions: {warmup_resolutions}", rank0_only=False)
        start_time = time.time()

        save_dir = os.path.join(output_dir, "aot_tokenizer")

        all_shapes = _collect_warmup_shapes(self, warmup_resolutions, aspect_ratio)

        is_distributed = dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1
        rank = dist.get_rank() if is_distributed else 0
        world_size = dist.get_world_size() if is_distributed else 1

        wanvae = self.model  # WanVAE (plain class)
        wanvae_model = wanvae.model  # WanVAE_ (nn.Module)
        scale = wanvae.scale  # (mean, 1/std)
        n_cache_slots = wanvae_model._enc_conv_num

        if rank == 0:
            log.info(f"Saving AOT compiled packages to {save_dir}")
            os.makedirs(save_dir, exist_ok=True)
        if is_distributed:
            dist.barrier()

        # -- Helper functions --------------------------------------------------

        def _rand_cache(cache: list[torch.Tensor | None]) -> list[torch.Tensor | None]:
            return [torch.rand_like(c) if c is not None else None for c in cache]

        def _rand_input(t: int, h: int, w: int) -> torch.Tensor:
            return torch.rand((1, 12, t, h, w), dtype=torch.bfloat16, device="cuda")

        def _compile_variant(
            wrapper: _ChunkEncodeForAOT,
            aot_key: _AOTChunkKey,
            ref_cache: list[torch.Tensor | None],
        ) -> str | None:
            """Export + compile one variant, returning the .pt2 path or None."""
            t_chunk, H_patch, W_patch, cache_t = aot_key
            pkg_name = f"chunk_ct{cache_t}_{t_chunk}f_{H_patch}x{W_patch}.pt2"
            pkg_path = os.path.join(save_dir, pkg_name)

            if os.path.exists(pkg_path):
                log.info(f"Rank {rank}: reusing cached {pkg_name}", rank0_only=False)
                return pkg_path

            t0 = time.time()
            try:
                exported = torch.export.export(
                    wrapper,
                    (_rand_input(t_chunk, H_patch, W_patch), _rand_cache(ref_cache)),
                    strict=False,
                )
                torch._inductor.aoti_compile_and_package(
                    exported,
                    package_path=pkg_path,
                    inductor_configs={"aot_inductor.package_constants_in_so": False},
                )
                log.info(
                    f"Rank {rank}: AOT compiled cache_t={cache_t} "
                    f"{t_chunk}f {H_patch}x{W_patch} in {time.time() - t0:.1f}s",
                    rank0_only=False,
                )
                return pkg_path
            except Exception as e:
                log.warning(
                    f"Rank {rank}: AOT compile failed for cache_t={cache_t} {t_chunk}f {H_patch}x{W_patch}: {e}",
                    rank0_only=False,
                )
                return None

        # -- Enumerate all variant keys and distribute across ranks ------------

        all_variant_keys: list[tuple[_AOTChunkKey, _ShapeInfo]] = []
        seen_keys: set[_AOTChunkKey] = set()
        for chunk_frames, H_patch, W_patch in all_shapes:
            for cache_t in (0, 1, 2):
                t_chunk = 1 if cache_t == 0 else chunk_frames
                aot_key: _AOTChunkKey = (t_chunk, H_patch, W_patch, cache_t)

                assert aot_key not in seen_keys, f"Duplicate AOT key: {aot_key}"
                seen_keys.add(aot_key)
                all_variant_keys.append((aot_key, (chunk_frames, H_patch, W_patch)))

            for T in sorted(self.encode_exact_durations or []):
                remaining = T - 1
                if remaining <= 0:
                    continue
                remainder = remaining % chunk_frames
                if remainder == 0:
                    continue
                n_full = remaining // chunk_frames
                cache_t = 1 if n_full == 0 else 2
                aot_key = (remainder, H_patch, W_patch, cache_t)

                if aot_key not in seen_keys:
                    seen_keys.add(aot_key)
                    all_variant_keys.append((aot_key, (chunk_frames, H_patch, W_patch)))

        my_variant_keys = [v for i, v in enumerate(all_variant_keys) if i % world_size == rank]
        log.info(
            f"Rank {rank}: assigned {len(my_variant_keys)}/{len(all_variant_keys)} variants (world_size={world_size})",
            rank0_only=False,
        )

        # -- Build reference caches lazily, only for this rank's shapes --------

        wrapper = _ChunkEncodeForAOT(wanvae_model, scale[0], scale[1])
        wrapper.eval()

        def _get_ref_caches(
            chunk_frames: int,
            H_patch: int,
            W_patch: int,
        ) -> dict[int, list[torch.Tensor | None]]:
            cache_ct0: list[torch.Tensor | None] = [None] * n_cache_slots
            _, cache_ct1 = wanvae_model._encode_chunk_impl(
                _rand_input(1, H_patch, W_patch),
                list(cache_ct0),
                scale,
            )
            _, cache_ct2 = wanvae_model._encode_chunk_impl(
                _rand_input(chunk_frames, H_patch, W_patch),
                list(cache_ct1),
                scale,
            )
            return {0: cache_ct0, 1: cache_ct1, 2: cache_ct2}

        ref_cache_map: dict[_ShapeInfo, dict[int, list[torch.Tensor | None]]] = {}

        my_pkg_paths: dict[_AOTChunkKey, str] = {}
        for aot_key, shape_info in my_variant_keys:
            cache_t = aot_key[3]
            if shape_info not in ref_cache_map:
                ref_cache_map[shape_info] = _get_ref_caches(*shape_info)
            ref_cache = ref_cache_map[shape_info][cache_t]
            pkg_path = _compile_variant(wrapper, aot_key, ref_cache)
            if pkg_path is not None:
                my_pkg_paths[aot_key] = pkg_path

        # -- Gather .pt2 paths from every rank so all ranks can load all variants.
        if is_distributed:
            gathered: list[dict[_AOTChunkKey, str] | None] = [None] * world_size
            dist.all_gather_object(gathered, my_pkg_paths)
            pkg_paths: dict[_AOTChunkKey, str] = {}
            for rank_paths in gathered:
                if rank_paths:
                    pkg_paths.update(rank_paths)
            dist.barrier()
        else:
            pkg_paths = my_pkg_paths

        # -- Load every .pt2 package and bind to the existing encoder weights. --
        device_index = torch.cuda.current_device()
        state_dict = wrapper.state_dict()

        loaded_fns: dict[_AOTChunkKey, Callable] = {}
        for key, pkg_path in pkg_paths.items():
            try:
                fn = torch._inductor.aoti_load_package(pkg_path, device_index=device_index)

                required_keys = set(fn.get_constant_fqns())
                constants_map = {k: v for k, v in state_dict.items() if k in required_keys}
                fn.load_constants(constants_map, check_full_update=True, user_managed=True)

                loaded_fns[key] = fn
            except Exception as e:
                log.warning(
                    f"Rank {rank}: failed to load {pkg_path}: {e}",
                    rank0_only=False,
                )

        wanvae_model._aot_chunk_fns = loaded_fns

        log.info(
            f"Rank {rank}: AOT compiled {len(my_pkg_paths)}, "
            f"loaded {len(loaded_fns)}/{len(all_variant_keys)} chunk variants, "
            f"time: {time.time() - start_time:.2f}s",
            rank0_only=False,
        )

        # Clean up .pt2 files so stale packages don't persist across restarts.
        if is_distributed:
            dist.barrier()
        if rank == 0:
            import shutil

            try:
                shutil.rmtree(save_dir)
                log.info(f"Cleaned up AOT cache dir: {save_dir}")
            except OSError as e:
                log.warning(f"Failed to clean AOT cache dir {save_dir}: {e}")

        if not loaded_fns:
            raise RuntimeError("AOT compilation produced no loadable functions")

    def get_latent_num_frames(self, num_pixel_frames: int) -> int:
        return 1 + (num_pixel_frames - 1) // 4

    def get_pixel_num_frames(self, num_latent_frames: int) -> int:
        return (num_latent_frames - 1) * 4 + 1

    @property
    def spatial_compression_factor(self) -> int:
        return self._spatial_compression_factor

    @property
    def temporal_compression_factor(self) -> int:
        return self._temporal_compression_factor

    @property
    def pixel_chunk_duration(self) -> int:
        return self.chunk_duration

    @property
    def latent_chunk_duration(self) -> int:
        return self.get_latent_num_frames(self.chunk_duration)

    @property
    def latent_ch(self) -> int:
        return 48

    @property
    def spatial_resolution(self) -> int:
        return 512

    @property
    def name(self) -> str:
        return "wan2pt2_tokenizer"
