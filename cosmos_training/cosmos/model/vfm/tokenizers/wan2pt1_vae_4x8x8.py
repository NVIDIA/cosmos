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

from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from cosmos.utils import log
from cosmos.utils.distributed import get_rank, sync_model_states
from cosmos.utils.easy_io import easy_io
from cosmos.model.vfm.tokenizers.interface import VideoTokenizerInterface

# For sequential decoding, CACHE_T is the number of frames to cache.
CACHE_T = 2


class CausalConv3d(nn.Conv3d):
    """
    Causal 3d convolusion.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._padding = (self.padding[2], self.padding[2], self.padding[1], self.padding[1], 2 * self.padding[0], 0)
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
        assert mode in ("none", "upsample2d", "upsample3d", "downsample2d", "downsample3d")
        super().__init__()
        self.dim = dim
        self.mode = mode

        # layers
        if mode == "upsample2d":
            self.resample = nn.Sequential(
                Upsample(scale_factor=(2.0, 2.0), mode="nearest-exact"), nn.Conv2d(dim, dim // 2, 3, padding=1)
            )
        elif mode == "upsample3d":
            self.resample = nn.Sequential(
                Upsample(scale_factor=(2.0, 2.0), mode="nearest-exact"), nn.Conv2d(dim, dim // 2, 3, padding=1)
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
                    feat_cache[idx] = "Rep"
                    feat_idx[0] += 1
                else:
                    cache_x = x[:, :, -CACHE_T:, :, :].clone()
                    if cache_x.shape[2] < 2 and feat_cache[idx] is not None and feat_cache[idx] != "Rep":
                        # cache last frame of last two chunk
                        cache_x = torch.cat(
                            [feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2
                        )  # [B,C,2,H,W]
                    if cache_x.shape[2] < 2 and feat_cache[idx] is not None and feat_cache[idx] == "Rep":
                        cache_x = torch.cat(
                            [torch.zeros_like(cache_x).to(cache_x.device), cache_x], dim=2
                        )  # [B,C,2,H,W]
                    if feat_cache[idx] == "Rep":
                        x = self.time_conv(x)  # [B,C*2,T,H,W]
                    else:
                        x = self.time_conv(x, feat_cache[idx])  # [B,C*2,T,H,W]
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1

                    x = x.reshape(b, 2, c, t, h, w)  # [B,2,C,T,H,W]
                    x = torch.stack((x[:, 0, :, :, :, :], x[:, 1, :, :, :, :]), 3)  # [B,C,T,2,H,W]
                    x = x.reshape(b, c, t * 2, h, w)  # [B,C,T*2,H,W]
        t = x.shape[2]
        x = rearrange(x, "b c t h w -> (b t) c h w")  # [B*T,C,H,W]
        x = self.resample(x)  # [B*T,C//2,H*2,W*2] for upsample2d/3d
        x = rearrange(x, "(b t) c h w -> b c t h w", t=t)  # [B,C_out,T,H_out,W_out]

        if self.mode == "downsample3d":
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = x.clone()
                    feat_idx[0] += 1
                else:
                    cache_x = x[:, :, -1:, :, :].clone()
                    # if cache_x.shape[2] < 2 and feat_cache[idx] is not None and feat_cache[idx]!='Rep':
                    #     # cache last frame of last two chunk
                    #     cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)

                    x = self.time_conv(torch.cat([feat_cache[idx][:, :, -1:, :, :], x], 2))  # [B,C,T//2,H_out,W_out]
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1
        return x

    def init_weight(self, conv):
        conv_weight = conv.weight
        nn.init.zeros_(conv_weight)
        c1, c2, t, h, w = conv_weight.size()
        one_matrix = torch.eye(c1, c2)
        init_matrix = one_matrix
        nn.init.zeros_(conv_weight)
        # conv_weight.data[:,:,-1,1,1] = init_matrix * 0.5
        conv_weight.data[:, :, 1, 0, 0] = init_matrix  # * 0.5
        conv.weight.data.copy_(conv_weight)
        nn.init.zeros_(conv.bias.data)

    def init_weight2(self, conv):
        conv_weight = conv.weight.data
        nn.init.zeros_(conv_weight)
        c1, c2, t, h, w = conv_weight.size()
        init_matrix = torch.eye(c1 // 2, c2)
        # init_matrix = repeat(init_matrix, 'o ... -> (o 2) ...').permute(1,0,2).contiguous().reshape(c1,c2)
        conv_weight[: c1 // 2, :, -1, 0, 0] = init_matrix
        conv_weight[c1 // 2 :, :, -1, 0, 0] = init_matrix
        conv.weight.data.copy_(conv_weight)
        nn.init.zeros_(conv.bias.data)


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
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    # cache last frame of last two chunk
                    cache_x = torch.cat(
                        [feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2
                    )  # [B,C,2,H,W]
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
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
        x = x.squeeze(1).permute(0, 2, 1).reshape(b * t, c, h, w)  # [B*T,C,H,W]

        # output
        x = self.proj(x)  # [B*T,C,H,W]
        x = rearrange(x, "(b t) c h w-> b c t h w", t=t)  # [B,C,T,H,W]
        return x + identity  # [B,C,T,H,W]


class Encoder3d(nn.Module):
    def __init__(
        self,
        dim=128,
        z_dim=4,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_downsample=[True, True, False],
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
        self.conv1 = CausalConv3d(3, dims[0], 3, padding=1)

        # downsample blocks
        downsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            # residual (+attention) blocks
            for _ in range(num_res_blocks):
                downsamples.append(ResidualBlock(in_dim, out_dim, dropout))
                if scale in attn_scales:
                    downsamples.append(AttentionBlock(out_dim))
                in_dim = out_dim

            # downsample block
            if i != len(dim_mult) - 1:
                mode = "downsample3d" if temperal_downsample[i] else "downsample2d"
                downsamples.append(Resample(out_dim, mode=mode))
                scale /= 2.0
        self.downsamples = nn.Sequential(*downsamples)

        # middle blocks
        self.middle = nn.Sequential(
            ResidualBlock(out_dim, out_dim, dropout), AttentionBlock(out_dim), ResidualBlock(out_dim, out_dim, dropout)
        )

        # output blocks
        self.head = nn.Sequential(
            RMS_norm(out_dim, images=False), nn.SiLU(), CausalConv3d(out_dim, z_dim, 3, padding=1)
        )

    def forward(self, x, feat_cache=None, feat_idx=[0]):  # x: [B,3,T,H,W] -> [B,z_dim,T//4,H//8,W//8]
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                # cache last frame of last two chunk
                cache_x = torch.cat(
                    [feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2
                )  # [B,C,2,H,W]
            x = self.conv1(x, feat_cache[idx])  # [B,dim,T,H,W]
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)  # [B,dim,T,H,W]

        # downsamples
        for layer in self.downsamples:
            if feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)
        # x: [B,dim*dim_mult[-1],T//4,H//8,W//8]

        # middle
        for layer in self.middle:
            if isinstance(layer, ResidualBlock) and feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)
        # x: [B,dim*dim_mult[-1],T//4,H//8,W//8]

        # head
        for layer in self.head:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    # cache last frame of last two chunk
                    cache_x = torch.cat(
                        [feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2
                    )  # [B,C,2,H,W]
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)
        return x  # [B,z_dim,T//4,H//8,W//8]


class Decoder3d(nn.Module):
    def __init__(
        self,
        dim=128,
        z_dim=4,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_upsample=[False, True, True],
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
        scale = 1.0 / 2 ** (len(dim_mult) - 2)

        # init block
        self.conv1 = CausalConv3d(z_dim, dims[0], 3, padding=1)

        # middle blocks
        self.middle = nn.Sequential(
            ResidualBlock(dims[0], dims[0], dropout), AttentionBlock(dims[0]), ResidualBlock(dims[0], dims[0], dropout)
        )

        # upsample blocks
        upsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            # residual (+attention) blocks
            if i == 1 or i == 2 or i == 3:
                in_dim = in_dim // 2
            for _ in range(num_res_blocks + 1):
                upsamples.append(ResidualBlock(in_dim, out_dim, dropout))
                if scale in attn_scales:
                    upsamples.append(AttentionBlock(out_dim))
                in_dim = out_dim

            # upsample block
            if i != len(dim_mult) - 1:
                mode = "upsample3d" if temperal_upsample[i] else "upsample2d"
                upsamples.append(Resample(out_dim, mode=mode))
                scale *= 2.0
        self.upsamples = nn.Sequential(*upsamples)

        # output blocks
        self.head = nn.Sequential(RMS_norm(out_dim, images=False), nn.SiLU(), CausalConv3d(out_dim, 3, 3, padding=1))

    def forward(self, x, feat_cache=None, feat_idx=[0]):  # x: [B,z_dim,T,H,W] -> [B,3,T*4,H*8,W*8]
        # conv1
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                # cache last frame of last two chunk
                cache_x = torch.cat(
                    [feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2
                )  # [B,C,2,H,W]
            x = self.conv1(x, feat_cache[idx])  # [B,dim*dim_mult[-1],T,H,W]
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)  # [B,dim*dim_mult[-1],T,H,W]

        # middle
        for layer in self.middle:
            if isinstance(layer, ResidualBlock) and feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)
        # x: [B,dim*dim_mult[-1],T,H,W]

        # upsamples
        for layer in self.upsamples:
            if feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)
        # x: [B,dim,T*4,H*8,W*8]

        # head
        for layer in self.head:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    # cache last frame of last two chunk
                    cache_x = torch.cat(
                        [feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2
                    )  # [B,C,2,H,W]
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)
        return x  # [B,3,T*4,H*8,W*8]


def count_conv3d(model):
    count = 0
    for m in model.modules():
        if isinstance(m, CausalConv3d):
            count += 1
    return count


class WanVAE_(nn.Module):
    def __init__(
        self,
        dim=128,
        z_dim=4,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_downsample=[True, True, False],
        dropout=0.0,
        temporal_window=4,
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
        # modules
        self.encoder = Encoder3d(
            dim, z_dim * 2, dim_mult, num_res_blocks, attn_scales, self.temperal_downsample, dropout
        )
        self.conv1 = CausalConv3d(z_dim * 2, z_dim * 2, 1)
        self.conv2 = CausalConv3d(z_dim, z_dim, 1)
        self.decoder = Decoder3d(dim, z_dim, dim_mult, num_res_blocks, attn_scales, self.temperal_upsample, dropout)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decode(z)
        return x_recon, mu, log_var

    def encode(self, x, scale, clear_encoder_cache=True):  # x: [B,3,T,H,W] -> [B,z_dim,T//4,H//8,W//8]
        t = x.shape[2]
        assert t == 1 or (t - 1) % 4 == 0, (
            f"Input temporal length must be 4n+1 (got {t}). "
            "Use pad_video_batch to pad before encoding, check wan2pt1_vae_4x8x8_test on how to use it."
        )
        if clear_encoder_cache:
            self.clear_cache()
        # cache
        iter_ = 1 + (t - 1) // self.temporal_window
        # 对encode输入的x，按时间拆分为1、self.temporal_stride、self.temporal_stride、self.temporal_window....
        for i in range(iter_):
            self._enc_conv_idx = [0]
            if i == 0:
                out = self.encoder(x[:, :, :1, :, :], feat_cache=self._enc_feat_map, feat_idx=self._enc_conv_idx)
                # out: [B,z_dim*2,1,H//8,W//8]
            else:
                out_ = self.encoder(
                    x[:, :, 1 + self.temporal_window * (i - 1) : 1 + self.temporal_window * i, :, :],
                    feat_cache=self._enc_feat_map,
                    feat_idx=self._enc_conv_idx,
                )  # [B,z_dim*2,temporal_window//4,H//8,W//8]
                out = torch.cat([out, out_], 2)  # [B,z_dim*2,T_latent_so_far,H//8,W//8]
        if (t - 1) % self.temporal_window:
            self._enc_conv_idx = [0]
            out_ = self.encoder(
                x[:, :, 1 + self.temporal_window * (iter_ - 1) :, :, :],
                feat_cache=self._enc_feat_map,
                feat_idx=self._enc_conv_idx,
            )
            out = torch.cat([out, out_], 2)  # [B,z_dim*2,T//4,H//8,W//8]
        mu, log_var = self.conv1(out).chunk(2, dim=1)  # mu,log_var: [B,z_dim,T//4,H//8,W//8]
        if isinstance(scale[0], torch.Tensor):
            mu = (mu - scale[0].view(1, self.z_dim, 1, 1, 1)) * scale[1].view(
                1, self.z_dim, 1, 1, 1
            )  # [B,z_dim,T//4,H//8,W//8]
        else:
            mu = (mu - scale[0]) * scale[1]  # [B,z_dim,T//4,H//8,W//8]
        if clear_encoder_cache:
            self.clear_cache()
        return mu  # [B,z_dim,T//4,H//8,W//8]

    def decode(self, z, scale, clear_decoder_cache=True):  # z: [B,z_dim,T_latent,H_latent,W_latent] -> [B,3,T,H,W]
        if clear_decoder_cache:
            self.clear_cache()
        # z: [B,z_dim,T_latent,H_latent,W_latent]
        if isinstance(scale[0], torch.Tensor):
            z = z / scale[1].view(1, self.z_dim, 1, 1, 1) + scale[0].view(
                1, self.z_dim, 1, 1, 1
            )  # [B,z_dim,T_latent,H_latent,W_latent]
        else:
            z = z / scale[1] + scale[0]  # [B,z_dim,T_latent,H_latent,W_latent]
        iter_ = z.shape[2]
        x = self.conv2(z)  # [B,z_dim,T_latent,H_latent,W_latent]
        for i in range(iter_):
            self._conv_idx = [0]
            if i == 0:
                out = self.decoder(x[:, :, i : i + 1, :, :], feat_cache=self._feat_map, feat_idx=self._conv_idx)
                # out: [B,3,T_chunk,H,W]
            else:
                out_ = self.decoder(x[:, :, i : i + 1, :, :], feat_cache=self._feat_map, feat_idx=self._conv_idx)
                out = torch.cat([out, out_], 2)  # [B,3,T_decoded_so_far,H,W]
        if clear_decoder_cache:
            self.clear_cache()
        return out  # [B,3,T,H,W]

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)  # [B,z_dim,T,H,W]
        eps = torch.randn_like(std)  # [B,z_dim,T,H,W]
        return eps * std + mu  # [B,z_dim,T,H,W]

    def sample(self, imgs, deterministic=False):
        mu, log_var = self.encode(imgs)
        if deterministic:
            return mu
        std = torch.exp(0.5 * log_var.clamp(-30.0, 20.0))  # [B,z_dim,T,H,W]
        return mu + std * torch.randn_like(std)  # [B,z_dim,T,H,W]

    def clear_cache(self):
        self._conv_num = count_conv3d(self.decoder)
        self._conv_idx = [0]
        self._feat_map = [None] * self._conv_num
        # cache encode
        self._enc_conv_num = count_conv3d(self.encoder)
        self._enc_conv_idx = [0]
        self._enc_feat_map = [None] * self._enc_conv_num


def _video_vae(
    pretrained_path=None,
    z_dim=None,
    device="cpu",
    object_store_credential_path_pretrained: str = "",
    temporal_window: int = 4,
):
    """
    Autoencoder3d adapted from Stable Diffusion 1.x, 2.x and XL.
    """
    # params
    cfg = dict(
        dim=96,
        z_dim=z_dim,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_downsample=[False, True, True],
        dropout=0.0,
        temporal_window=temporal_window,
    )

    # init model
    with torch.device("meta"):
        model = WanVAE_(**cfg)

    if pretrained_path is None:
        model.to_empty(device=device)
    else:
        if get_rank() == 0:
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
            model.load_state_dict(ckpt, assign=True)
        else:
            model.to_empty(device=device)
    sync_model_states(model)

    return (
        model,
        torch.zeros(1, 1, 1, 1, 1, device=device),  # img_mean: [1,1,1,1,1]
        torch.ones(1, 1, 1, 1, 1, device=device),  # img_std:  [1,1,1,1,1]
        torch.zeros(1, 1, 50, 1, 1, device=device),  # video_mean: [1,1,50,1,1]
        torch.ones(1, 1, 50, 1, 1, device=device),  # video_std:  [1,1,50,1,1]
    )


class WanVAE:
    def __init__(
        self,
        z_dim=16,
        vae_pth="",
        object_store_credential_path_pretrained: str = "",
        dtype=torch.bfloat16,
        device="cuda",
        is_amp=True,
        temporal_window: int = 4,
        use_channels_last_memory_format: bool = False,
    ):
        self.dtype = dtype
        self.device = device
        self.temporal_window = temporal_window

        mean = [
            -0.7571,
            -0.7089,
            -0.9113,
            0.1075,
            -0.1745,
            0.9653,
            -0.1517,
            1.5508,
            0.4134,
            -0.0715,
            0.5517,
            -0.3632,
            -0.1922,
            -0.9497,
            0.2503,
            -0.2921,
        ]
        std = [
            2.8184,
            1.4541,
            2.3275,
            2.6558,
            1.2196,
            1.7708,
            2.6052,
            2.0743,
            3.2687,
            2.1526,
            2.8652,
            1.5579,
            1.6382,
            1.1253,
            2.8251,
            1.9160,
        ]
        self.mean = torch.tensor(mean, dtype=dtype, device=device)  # [z_dim]
        self.std = torch.tensor(std, dtype=dtype, device=device)  # [z_dim]
        self.scale = [self.mean, 1.0 / self.std]

        # init model
        self.model, self.img_mean, self.img_std, self.video_mean, self.video_std = _video_vae(
            pretrained_path=vae_pth,
            z_dim=z_dim,
            object_store_credential_path_pretrained=object_store_credential_path_pretrained,
            device=device,
            temporal_window=temporal_window,
        )
        self.model = self.model.eval().requires_grad_(False)
        self.is_amp = is_amp
        if not is_amp:
            self.model = self.model.to(dtype=dtype)
            self.context = nullcontext()
        else:
            self.context = torch.amp.autocast("cuda", dtype=dtype)

        if use_channels_last_memory_format:
            for _, module in self.model.encoder.named_modules():
                if hasattr(module, "weight"):
                    shape = module.weight.shape
                    if len(shape) == 4:
                        module.to(memory_format=torch.channels_last)
                    elif len(shape) == 5:
                        module.to(memory_format=torch.channels_last_3d)

    def count_param(self):
        return sum(p.numel() for p in self.model.parameters())

    @torch.no_grad()
    def encode(self, videos, clear_encoder_cache=True):  # videos: [B,3,T,H,W] -> [B,z_dim,T//4,H//8,W//8]
        """
        videos: A list of videos each with shape [C, T, H, W].
        """
        in_dtype = videos.dtype
        with self.context:
            if not self.is_amp:
                videos = videos.to(self.dtype)
            latent = self.model.encode(videos, self.scale, clear_encoder_cache)  # [B,z_dim,T//4,H//8,W//8]
        latent = latent.to(in_dtype)
        return latent.contiguous()  # [B,z_dim,T//4,H//8,W//8]

    @torch.no_grad()
    def decode(self, zs, clear_decoder_cache=True):  # zs: [B,z_dim,T_latent,H_latent,W_latent] -> [B,3,T,H,W]
        in_dtype = zs.dtype
        with self.context:
            if not self.is_amp:
                zs = zs.to(self.dtype)
            video_recon = self.model.decode(zs, self.scale, clear_decoder_cache)  # [B,3,T,H,W]
        video_recon = video_recon.to(in_dtype)
        return video_recon  # [B,3,T,H,W]


class Wan2pt1VAEInterface(VideoTokenizerInterface):
    def __init__(
        self,
        bucket_name: str = "",
        object_store_credential_path_pretrained: str = "",
        vae_path: str = "",
        chunk_duration: int = 81,
        temporal_window: int = 4,
        keep_decoder_cache: bool = False,
        keep_encoder_cache: bool = False,
        use_channels_last_memory_format: bool = False,
        spatial_compression_factor: int = 8,
        temporal_compression_factor: int = 4,
    ):
        vae_path_full = f"s3://{bucket_name}/{vae_path}"
        self.keep_decoder_cache = keep_decoder_cache
        self.keep_encoder_cache = keep_encoder_cache
        self.model = WanVAE(
            dtype=torch.bfloat16,
            is_amp=False,
            vae_pth=vae_path_full,
            object_store_credential_path_pretrained=object_store_credential_path_pretrained,
            temporal_window=temporal_window,
            use_channels_last_memory_format=use_channels_last_memory_format,
        )
        self.chunk_duration = chunk_duration

    @property
    def dtype(self):
        return self.model.dtype

    def reset_dtype(self):
        pass

    def clear_cache(self):
        """Clear the feature cache for both encoder and decoder."""
        self.model.model.clear_cache()

    def encode(self, state: torch.Tensor) -> torch.Tensor:  # state: [B,3,T,H,W] -> [B,C,T//4,H//8,W//8]
        latents = self.model.encode(state, clear_encoder_cache=not self.keep_encoder_cache)  # [B,C,T//4,H//8,W//8]
        num_frames = latents.shape[2]
        if num_frames == 1:
            return (latents - self.model.img_mean.type_as(latents)) / self.model.img_std.type_as(
                latents
            )  # [B,C,1,H//8,W//8]
        else:
            return (latents - self.model.video_mean[:, :, :num_frames].type_as(latents)) / self.model.video_std[
                :, :, :num_frames
            ].type_as(latents)  # [B,C,T//4,H//8,W//8]

    def decode(self, latent: torch.Tensor) -> torch.Tensor:  # latent: [B,C,T_latent,H_latent,W_latent] -> [B,3,T,H,W]
        num_frames = latent.shape[2]
        if num_frames == 1:
            return self.model.decode(
                (latent * self.model.img_std.type_as(latent)) + self.model.img_mean.type_as(latent),
                clear_decoder_cache=not self.keep_decoder_cache,
            )  # [B,3,1,H,W]
        else:
            return self.model.decode(
                (latent * self.model.video_std[:, :, :num_frames].type_as(latent))
                + self.model.video_mean[:, :, :num_frames].type_as(latent),
                clear_decoder_cache=not self.keep_decoder_cache,
            )  # [B,3,T,H,W]

    def get_latent_num_frames(self, num_pixel_frames: int) -> int:
        return 1 + (num_pixel_frames - 1) // 4

    def get_pixel_num_frames(self, num_latent_frames: int) -> int:
        return (num_latent_frames - 1) * 4 + 1

    @property
    def spatial_compression_factor(self):
        return 8

    @property
    def temporal_compression_factor(self):
        return 4

    @property
    def pixel_chunk_duration(self):
        return self.chunk_duration

    @property
    def latent_chunk_duration(self):
        return self.get_latent_num_frames(self.chunk_duration)

    @property
    def latent_ch(self):
        return 16

    @property
    def spatial_resolution(self):
        return 512

    @property
    def name(self):
        return "wan2pt1_tokenizer"
