# Copyright 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import collections
from typing import Any, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_same_padding(kernel_size: int | tuple[int, ...]) -> int | tuple[int, ...]:
    if isinstance(kernel_size, (tuple, list)):
        return tuple([get_same_padding(ks) for ks in kernel_size])
    else:
        assert kernel_size % 2 > 0, "kernel size should be odd number"
        return kernel_size // 2


def get_submodule_weights(weights: collections.OrderedDict, prefix: str):
    submodule_weights = collections.OrderedDict()
    len_prefix = len(prefix)
    for key, weight in weights.items():
        if key.startswith(prefix):
            submodule_weights[key[len_prefix:]] = weight
    return submodule_weights


def val2list(x: list | tuple | Any, repeat_time=1) -> list:
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x for _ in range(repeat_time)]


def val2tuple(x: list | tuple | Any, min_len: int = 1, idx_repeat: int = -1) -> tuple:
    x = val2list(x)

    # repeat elements if necessary
    if len(x) > 0:
        x[idx_repeat:idx_repeat] = [x[idx_repeat] for _ in range(min_len - len(x))]

    return tuple(x)


REGISTERED_ACT_DICT: dict[str, type] = {
    "silu": nn.SiLU,
}


def build_act(name: Optional[str]) -> Optional[nn.Module]:
    if name in REGISTERED_ACT_DICT:
        act_cls = REGISTERED_ACT_DICT[name]
        return act_cls()
    else:
        return None


class TritonRMSNorm2d(nn.LayerNorm):
    def zero_out(self):
        nn.init.constant_(self.weight, 0)
        nn.init.constant_(self.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from cosmos.model.vfm.tokenizers.dc_ae.dc_ae_v_triton_rms_norm import TritonRMSNorm2dFunc

        if not torch.compiler.is_compiling():
            input_numel = x.numel()
            if input_numel >= 1 << 31:
                num_chunks = (input_numel - 1) // (1 << 31) + 1
                output = []
                for x_chunk in x.chunk(num_chunks, dim=2):
                    output.append(TritonRMSNorm2dFunc.apply(x_chunk.contiguous(), self.weight, self.bias, self.eps))
                output = torch.cat(output, dim=2)
                return output
        return TritonRMSNorm2dFunc.apply(x.contiguous(), self.weight, self.bias, self.eps)


# register normalization function here
REGISTERED_NORM_DICT: dict[str, type] = {
    "trms2d": TritonRMSNorm2d,
}


def build_norm(name: Optional[str] = "bn2d", num_features=None, **kwargs) -> Optional[nn.Module]:
    if name in ["trms2d"]:
        kwargs["normalized_shape"] = num_features
    else:
        kwargs["num_features"] = num_features
    if name in REGISTERED_NORM_DICT:
        norm_cls = REGISTERED_NORM_DICT[name]
        return norm_cls(**kwargs)
    else:
        return None


class IdentityLayer(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class OpSequential(nn.Module):
    def __init__(self, op_list: list[Optional[nn.Module]]):
        super(OpSequential, self).__init__()
        valid_op_list = []
        for op in op_list:
            if op is not None:
                valid_op_list.append(op)
        self.op_list = nn.ModuleList(valid_op_list)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for op in self.op_list:
            x = op(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(
        self,
        main: Optional[nn.Module],
        shortcut: Optional[nn.Module],
        post_act=None,
        pre_norm: Optional[nn.Module] = None,
    ):
        super(ResidualBlock, self).__init__()

        self.pre_norm = pre_norm
        self.main = main
        self.shortcut = shortcut
        self.post_act = build_act(post_act)

    def forward_main(self, x: torch.Tensor) -> torch.Tensor:
        if self.pre_norm is None:
            return self.main(x)
        else:
            return self.main(self.pre_norm(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.main is None:
            res = x
        elif self.shortcut is None:
            res = self.forward_main(x)
        else:
            res = self.forward_main(x) + self.shortcut(x)
            if self.post_act:
                res = self.post_act(res)
        return res


def conv3d_split_channel(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    stride: int | Sequence[int],
    padding: int | Sequence[int],
    dilation: int | Sequence[int],
    num_in_channel_chunks: int,
    num_out_channel_chunks: int,
) -> torch.Tensor:
    out_channels, in_channels = weight.shape[0], weight.shape[1]
    assert in_channels % num_in_channel_chunks == 0 and out_channels % num_out_channel_chunks == 0
    in_channels_per_split = in_channels // num_in_channel_chunks
    out_channels_per_split = out_channels // num_out_channel_chunks

    output = []
    for i in range(num_out_channel_chunks):
        out_channels_start, out_channels_end = i * out_channels_per_split, (i + 1) * out_channels_per_split
        output_i = 0
        for j in range(num_in_channel_chunks):
            in_channels_start, in_channels_end = j * in_channels_per_split, (j + 1) * in_channels_per_split
            x_j = x[:, in_channels_start:in_channels_end]
            weight_j = weight[out_channels_start:out_channels_end, in_channels_start:in_channels_end]
            output_i = output_i + F.conv3d(x_j, weight_j, stride=stride, padding=padding, dilation=dilation, groups=1)
        output.append(output_i)
    output = torch.cat(output, dim=1)
    if bias is not None:
        output = output + bias[:, None, None, None]
    return output


def custom_conv3d(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    stride: Sequence[int],
    padding: int | Sequence[int],
    dilation: int | Sequence[int],
    groups: int,
) -> torch.Tensor:
    input_sample_numel = input[0].numel()
    output_sample_numel = (
        weight.shape[0] * (input.shape[2] // stride[0]) * (input.shape[3] // stride[1]) * (input.shape[4] // stride[2])
    )

    if (input_sample_numel >= 1 << 31 or output_sample_numel >= 1 << 31) and groups == 1:
        num_in_channel_chunks, num_out_channel_chunks = 1, 1
        while input_sample_numel // num_in_channel_chunks >= 1 << 31:
            num_in_channel_chunks *= 2
        while output_sample_numel // num_out_channel_chunks >= 1 << 31:
            num_out_channel_chunks *= 2
        # print(f"num_in_channel_chunks {num_in_channel_chunks}, num_out_channel_chunks {num_out_channel_chunks}")
        output = conv3d_split_channel(
            input, weight, bias, stride, padding, dilation, num_in_channel_chunks, num_out_channel_chunks
        )
        return output
    else:
        return F.conv3d(input, weight, bias, stride, padding, dilation, groups)


class CustomConv3d(nn.Conv3d):
    def _conv_forward(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        padding: Optional[tuple[int, ...]] = None,
    ):
        assert self.padding_mode == "zeros"
        return custom_conv3d(input, weight, bias, self.stride, padding or self.padding, self.dilation, self.groups)

    def forward(self, input: torch.Tensor, padding: Optional[tuple[int, ...]] = None) -> torch.Tensor:
        return self._conv_forward(input, self.weight, self.bias, padding)


class ConvLayer3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, ...] = 3,
        stride: int | tuple[int, ...] = 1,
        groups: int = 1,
        use_bias: bool = False,
        norm: Optional[str] = "bn2d",
        act_func: Optional[str] = "relu",
        zero_out: bool = False,
        spatial_padding_mode: str = "zeros",
        temporal_padding_mode: str = "zeros",
        causal: bool = False,
        causal_chunk_length: Optional[int] = None,
    ):
        super().__init__()
        kernel_size = val2tuple(kernel_size, 3)
        stride = val2tuple(stride, 3)
        padding = get_same_padding(kernel_size)
        self.causal = causal
        self.causal_chunk_length = causal_chunk_length
        self.spatial_padding_mode = spatial_padding_mode
        self.temporal_padding_mode = temporal_padding_mode
        if causal:
            self.custom_padding = (0, 0, 0, 0, 2 * padding[0], 0)
            padding = (0, padding[1], padding[2])
            self.custom_padding_mode = "constant" if temporal_padding_mode == "zeros" else temporal_padding_mode
        elif causal_chunk_length is not None:
            assert spatial_padding_mode == "zeros"
            self.custom_padding = None
            self.custom_padding_mode = None
        elif spatial_padding_mode != temporal_padding_mode:
            self.custom_padding = (0, 0, 0, 0, padding[0], padding[0])
            padding = (0, padding[1], padding[2])
            self.custom_padding_mode = "constant" if temporal_padding_mode == "zeros" else temporal_padding_mode
        else:
            self.custom_padding = None
            self.custom_padding_mode = None
        self.conv = CustomConv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=use_bias,
            padding_mode=spatial_padding_mode,
        )
        self.norm = build_norm(norm, num_features=out_channels)
        self.act = build_act(act_func)

        self.zero_out = zero_out
        if zero_out:
            if self.norm:
                self.norm.zero_out()
            else:
                nn.init.constant_(self.conv.weight, 0)
                nn.init.constant_(self.conv.bias, 0)

    def load_state_dict_from_2d(self, state_dict: dict[str, torch.Tensor], method: str = "zero_pad"):
        if method == "zero_pad":
            nn.init.constant_(self.conv.weight, 0)
            if self.causal:
                self.conv.weight.data[:, :, -1] = state_dict["conv.weight"]
            else:
                self.conv.weight.data[:, :, self.conv.weight.data.shape[2] // 2] = state_dict["conv.weight"]
        elif method == "split":
            self.conv.weight.data.copy_(state_dict["conv.weight"][:, :, None] / self.conv.weight.shape[2])
        else:
            raise ValueError(f"init method {method} is not supported")
        if self.conv.bias is not None:
            nn.init.constant_(self.conv.bias, 0)
            self.conv.bias.data = state_dict["conv.bias"]
        if self.norm:
            self.norm.load_state_dict(get_submodule_weights(state_dict, "norm."))
        if self.act:
            self.act.load_state_dict(get_submodule_weights(state_dict, "act."))

    def forward(
        self,
        x: torch.Tensor,
        feature_cache: list[torch.Tensor | None] | None = None,
        feat_idx: list[int] | None = None,
    ) -> tuple[torch.Tensor, None]:
        if self.custom_padding is not None:
            x = F.pad(x, self.custom_padding, mode=self.custom_padding_mode)

        if self.causal_chunk_length is not None:

            B, C, T, H, W = x.shape
            assert T % self.causal_chunk_length == 0
            assert self.conv.stride[0] == 1
            x = x.reshape(B, C, T // self.causal_chunk_length, self.causal_chunk_length, H, W).transpose(
                1, 2
            )  # (B, T // self.causal_chunk_length, C, self.causal_chunk_length, H, W)

            if feature_cache is not None:
                idx = feat_idx[0]
                first_left_pad = feature_cache[idx]
                feature_cache[idx] = x[:, -1:, :, -self.conv.padding[0] :].clone().detach()
                feat_idx[0] += 1
            else:
                first_left_pad = None
            if first_left_pad is None:
                if self.temporal_padding_mode == "zeros":
                    first_left_pad = torch.zeros((B, 1, C, self.conv.padding[0], H, W), dtype=x.dtype, device=x.device)
                elif self.temporal_padding_mode == "replicate":
                    first_left_pad = x[:, :1, :, :1, :, :].repeat((1, 1, 1, self.conv.padding[0], 1, 1))
                else:
                    raise ValueError(f"temporal padding mode {self.temporal_padding_mode} is not supported")
            else:
                assert (
                    first_left_pad.shape[0] == B
                    and first_left_pad.shape[1] == 1
                    and first_left_pad.shape[2] == C
                    and first_left_pad.shape[3] <= self.conv.padding[0]
                    and first_left_pad.shape[4] == H
                    and first_left_pad.shape[5] == W
                )
                if first_left_pad.shape[3] < self.conv.padding[0]:
                    assert self.temporal_padding_mode == "zeros"
                    first_left_pad = torch.cat(
                        [
                            torch.zeros(
                                (B, 1, C, self.conv.padding[0] - first_left_pad.shape[3], H, W),
                                dtype=x.dtype,
                                device=x.device,
                            ),
                            first_left_pad,
                        ],
                        dim=3,
                    )  # (B, 1, C, self.conv.padding[0], H, W)

            left_pad = torch.cat(
                [first_left_pad, x[:, :-1, :, -self.conv.padding[0] :]], dim=1
            )  # (B, T // self.causal_chunk_length, C, self.conv.padding[0], H, W)
            if self.temporal_padding_mode == "zeros":
                right_pad = torch.zeros(
                    (B, T // self.causal_chunk_length, C, self.conv.padding[0], H, W), dtype=x.dtype, device=x.device
                )  # (B, T // self.causal_chunk_length, C, self.conv.padding[0], H, W)
            elif self.temporal_padding_mode == "replicate":
                right_pad = x[:, :, :, -1:].repeat(
                    (1, 1, 1, self.conv.padding[0], 1, 1)
                )  # (B, T // self.causal_chunk_length, C, self.conv.padding[0], H, W)
            else:
                raise ValueError(f"temporal padding mode {self.temporal_padding_mode} is not supported")
            x = torch.cat(
                [left_pad, x, right_pad], dim=3
            )  # (B, T // self.causal_chunk_length, C, self.causal_chunk_length + 2 * self.conv.padding[0], H, W)
            x = x.reshape(
                B * (T // self.causal_chunk_length), C, self.causal_chunk_length + 2 * self.conv.padding[0], H, W
            )
            x = self.conv(
                x, (0, self.conv.padding[1], self.conv.padding[2])
            )  # (B * (T // self.causal_chunk_length), C, self.causal_chunk_length, H, W)
            x = (
                x.reshape(B, T // self.causal_chunk_length, -1, self.causal_chunk_length, H, W)
                .transpose(1, 2)
                .reshape(B, -1, T, H, W)
            )  # (B, C, T // self.causal_chunk_length, self.causal_chunk_length, H, W)
        elif self.causal:
            if feature_cache is not None:
                idx = feat_idx[0]
                cache = x[:, :, -self.custom_padding[4] :].clone().detach()
                if feature_cache[idx] is not None:
                    x[:, :, : self.custom_padding[4]] = feature_cache[idx]
                feature_cache[idx] = cache
                feat_idx[0] += 1
            x = self.conv(x)
        else:
            x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.act:
            x = self.act(x)
        return x, None

    def __repr__(self):
        _str = f"{self.__class__.__name__}(\n  (conv): {self.conv}\n"
        if self.norm:
            _str += f"  (norm): {self.norm}\n"
        if self.act:
            _str += f"  (act): {self.act}\n"
        _str += f"  zero_out={self.zero_out}\n"
        _str += f"  causal={self.causal}\n"
        _str += f"  causal_chunk_length={self.causal_chunk_length}\n"
        _str += f")"
        return _str


class ResBlock3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int, int] = 3,
        stride: int | tuple[int, int, int] = 1,
        mid_channels: Optional[int] = None,
        expand_ratio: float = 1,
        use_bias: bool | tuple[bool, bool] = False,
        norm: str | tuple[Optional[str], Optional[str]] = ("bn2d", "bn2d"),
        act_func: str | tuple[Optional[str], Optional[str]] = ("relu6", None),
        zero_out: bool = False,
        spatial_padding_mode: str = "zeros",
        temporal_padding_mode: str = "zeros",
        causal: bool = False,
        causal_chunk_length: Optional[int] = None,
    ):
        super().__init__()
        use_bias = val2tuple(use_bias, 2)
        norm = val2tuple(norm, 2)
        act_func = val2tuple(act_func, 2)

        mid_channels = round(in_channels * expand_ratio) if mid_channels is None else mid_channels

        self.conv1 = ConvLayer3d(
            in_channels,
            mid_channels,
            kernel_size,
            stride,
            use_bias=use_bias[0],
            norm=norm[0],
            act_func=act_func[0],
            spatial_padding_mode=spatial_padding_mode,
            temporal_padding_mode=temporal_padding_mode,
            causal=causal,
            causal_chunk_length=causal_chunk_length,
        )
        self.conv2 = ConvLayer3d(
            mid_channels,
            out_channels,
            kernel_size,
            1,
            use_bias=use_bias[1],
            norm=norm[1],
            act_func=act_func[1],
            zero_out=zero_out,
            spatial_padding_mode=spatial_padding_mode,
            temporal_padding_mode=temporal_padding_mode,
            causal=causal,
            causal_chunk_length=causal_chunk_length,
        )

    def load_state_dict_from_2d(self, state_dict: dict[str, torch.Tensor], method: str):
        self.conv1.load_state_dict_from_2d(get_submodule_weights(state_dict, "conv1."), method)
        self.conv2.load_state_dict_from_2d(get_submodule_weights(state_dict, "conv2."), method)

    def forward(
        self,
        x: torch.Tensor,
        feature_cache: list[torch.Tensor | None] | None = None,
        feat_idx: list[int] | None = None,
    ) -> tuple[torch.Tensor, None]:
        x, _ = self.conv1(x, feature_cache, feat_idx)
        x, _ = self.conv2(x, feature_cache, feat_idx)
        return x, None


def pixel_unshuffle_3d(x: torch.Tensor, spatial_factor: int, temporal_factor: int) -> torch.Tensor:
    # x: (B, C, T, H, W)
    B, C, T, H, W = x.shape
    assert T % temporal_factor == 0 and W % spatial_factor == 0 and H % spatial_factor == 0, (
        f"{T=}, {W=}, {H=}, {spatial_factor=}, {temporal_factor=} are not supported"
    )
    x = (
        x.reshape(
            (
                B,
                C,
                T // temporal_factor,
                temporal_factor,
                H // spatial_factor,
                spatial_factor,
                W // spatial_factor,
                spatial_factor,
            )
        )
        .permute(0, 1, 3, 5, 7, 2, 4, 6)
        .reshape(
            B, C * temporal_factor * spatial_factor**2, T // temporal_factor, H // spatial_factor, W // spatial_factor
        )
    )
    return x


def pixel_shuffle_3d(x: torch.Tensor, spatial_factor: int, temporal_factor: int) -> torch.Tensor:
    # x: (B, C, T, H, W)
    B, C, T, H, W = x.shape
    assert C % (temporal_factor * spatial_factor**2) == 0
    x = (
        x.reshape(
            (B, C // temporal_factor // spatial_factor**2, temporal_factor, spatial_factor, spatial_factor, T, H, W)
        )
        .permute(0, 1, 5, 2, 6, 3, 7, 4)
        .reshape(
            B, C // temporal_factor // spatial_factor**2, T * temporal_factor, H * spatial_factor, W * spatial_factor
        )
    )
    return x


class ConvPixelUnshuffleDownSampleLayer3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int],
        spatial_factor: int,
        temporal_factor: int,
        spatial_padding_mode: str = "zeros",
        temporal_padding_mode: str = "zeros",
        zero_out: bool = False,
        causal: bool = False,
        causal_chunk_length: Optional[int] = None,
    ):
        super().__init__()
        self.spatial_factor = spatial_factor
        self.temporal_factor = temporal_factor
        out_ratio = spatial_factor**2 * temporal_factor
        assert out_channels % out_ratio == 0
        self.conv = ConvLayer3d(
            in_channels=in_channels,
            out_channels=out_channels // out_ratio,
            kernel_size=kernel_size,
            use_bias=True,
            norm=None,
            act_func=None,
            spatial_padding_mode=spatial_padding_mode,
            temporal_padding_mode=temporal_padding_mode,
            zero_out=zero_out,
            causal=causal,
            causal_chunk_length=causal_chunk_length,
        )

    def load_state_dict_from_2d(self, state_dict: dict[str, torch.Tensor], method: str):
        self.conv.load_state_dict_from_2d(get_submodule_weights(state_dict, "conv."), method)

    def forward(
        self,
        x: torch.Tensor,
        feature_cache: list[torch.Tensor | None] | None = None,
        feat_idx: list[int] | None = None,
    ) -> tuple[torch.Tensor, None]:
        x, _ = self.conv(x, feature_cache, feat_idx)
        x = pixel_unshuffle_3d(x, self.spatial_factor, self.temporal_factor)
        x = x.to(memory_format=torch.channels_last_3d)
        return x, None


class PixelUnshuffleChannelAveragingDownSampleLayer3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        spatial_factor: int,
        temporal_factor: int,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.spatial_factor = spatial_factor
        self.temporal_factor = temporal_factor
        assert in_channels * spatial_factor**2 * temporal_factor % out_channels == 0
        self.group_size = in_channels * spatial_factor**2 * temporal_factor // out_channels

    def load_state_dict_from_2d(self, state_dict: dict[str, torch.Tensor], method: str):
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = pixel_unshuffle_3d(x, self.spatial_factor, self.temporal_factor)
        B, C, T, H, W = x.shape
        x = x.view(B, self.out_channels, self.group_size, T, H, W)
        x = x.mean(dim=2)
        return x.to(memory_format=torch.channels_last_3d)


class ConvPixelShuffleUpSampleLayer3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int],
        spatial_factor: int,
        temporal_factor: int,
        spatial_padding_mode: str = "zeros",
        temporal_padding_mode: str = "zeros",
        zero_out: bool = False,
        causal: bool = False,
        causal_chunk_length: Optional[int] = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.spatial_factor = spatial_factor
        self.temporal_factor = temporal_factor
        out_ratio = spatial_factor**2 * temporal_factor
        self.conv = ConvLayer3d(
            in_channels=in_channels,
            out_channels=out_channels * out_ratio,
            kernel_size=kernel_size,
            use_bias=True,
            norm=None,
            act_func=None,
            spatial_padding_mode=spatial_padding_mode,
            temporal_padding_mode=temporal_padding_mode,
            zero_out=zero_out,
            causal=causal,
            causal_chunk_length=causal_chunk_length,
        )

    def load_state_dict_from_2d(self, state_dict: dict[str, torch.Tensor], method: str):
        self.conv.load_state_dict_from_2d(get_submodule_weights(state_dict, "conv."), method)

    def forward(
        self,
        x: torch.Tensor,
        feature_cache: list[torch.Tensor | None] | None = None,
        feat_idx: list[int] | None = None,
    ) -> tuple[torch.Tensor, None]:
        x, _ = self.conv(x, feature_cache, feat_idx)
        x = pixel_shuffle_3d(x, self.spatial_factor, self.temporal_factor)
        return x, None


class ChannelDuplicatingPixelShuffleUpSampleLayer3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        spatial_factor: int,
        temporal_factor: int,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.spatial_factor = spatial_factor
        self.temporal_factor = temporal_factor
        assert out_channels * spatial_factor**2 * temporal_factor % in_channels == 0
        self.repeats = out_channels * spatial_factor**2 * temporal_factor // in_channels

    def load_state_dict_from_2d(self, state_dict: dict[str, torch.Tensor], method: str):
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.repeat_interleave(self.repeats, dim=1)
        x = pixel_shuffle_3d(x, self.spatial_factor, self.temporal_factor)
        return x


class ResidualBlock3d(ResidualBlock):
    def load_state_dict_from_2d(self, state_dict: dict[str, torch.Tensor], method: str):
        self.main.load_state_dict_from_2d(get_submodule_weights(state_dict, f"main."), method)
        if isinstance(self.shortcut, (IdentityLayer,)):
            pass
        else:
            self.shortcut.load_state_dict_from_2d(get_submodule_weights(state_dict, f"shortcut."), method)

    def forward_main(
        self,
        x: torch.Tensor,
        feature_cache: list[torch.Tensor | None] | None = None,
        feat_idx: list[int] | None = None,
    ) -> tuple[torch.Tensor, None]:
        if self.pre_norm is None:
            return self.main(x, feature_cache, feat_idx)
        else:
            return self.main(self.pre_norm(x), feature_cache, feat_idx)

    def forward(
        self,
        x: torch.Tensor,
        feature_cache: list[torch.Tensor | None] | None = None,
        feat_idx: list[int] | None = None,
    ) -> tuple[torch.Tensor, None]:
        if self.main is None:
            res = x
        elif self.shortcut is None:
            res, _ = self.forward_main(x, feature_cache, feat_idx)
        else:
            res_main, _ = self.forward_main(x, feature_cache, feat_idx)
            res_shortcut = self.shortcut(x)
            res = res_main + res_shortcut
            if self.post_act:
                res = self.post_act(res)
        return res, None


class OpSequential3d(OpSequential):
    def load_state_dict_from_2d(self, state_dict: dict[str, torch.Tensor], method: str):
        for i, op in enumerate(self.op_list):
            if isinstance(op, (TritonRMSNorm2d, nn.SiLU)):
                op.load_state_dict(get_submodule_weights(state_dict, f"op_list.{i}."))
            else:
                op.load_state_dict_from_2d(get_submodule_weights(state_dict, f"op_list.{i}."), method)

    def forward(
        self,
        x: torch.Tensor,
        feature_cache: list[torch.Tensor | None] | None = None,
        feat_idx: list[int] | None = None,
    ) -> tuple[torch.Tensor, None]:
        for i, op in enumerate(self.op_list):
            if isinstance(op, torch.distributed.algorithms._checkpoint.checkpoint_wrapper.CheckpointWrapper):
                op_class = type(op._checkpoint_wrapped_module)
            else:
                op_class = type(op)
            if issubclass(
                op_class,
                (
                    ConvLayer3d,
                    ResBlock3d,
                    ResidualBlock3d,
                    ConvPixelUnshuffleDownSampleLayer3d,
                    ConvPixelShuffleUpSampleLayer3d,
                ),
            ):
                x, _ = op(x, feature_cache, feat_idx)
            elif issubclass(op_class, (TritonRMSNorm2d, nn.SiLU)):
                x = op(x)
            else:
                raise ValueError(f"Unsupported op class: {op_class}")
        return x, None


class CompilableRMSNorm2d(nn.Module):
    """Pure PyTorch RMSNorm over the channel dimension -- torch.compile-friendly
    replacement for TritonRMSNorm2d. State-dict compatible (same weight/bias shapes).
    """

    def __init__(self, normalized_shape: int | tuple[int, ...], eps: float = 1e-5, **kwargs):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(self.normalized_shape))
        self.bias = nn.Parameter(torch.zeros(self.normalized_shape))

    def zero_out(self) -> None:
        nn.init.constant_(self.weight, 0)
        nn.init.constant_(self.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=1, keepdim=True)
        x = x * torch.rsqrt(rms + self.eps)
        shape = [1, -1] + [1] * (x.ndim - 2)
        return x * self.weight.view(shape) + self.bias.view(shape)


_CACHE_ARG_OP_TYPES = (
    ConvLayer3d,
    ResBlock3d,
    ResidualBlock3d,
    ConvPixelUnshuffleDownSampleLayer3d,
    ConvPixelShuffleUpSampleLayer3d,
)


class CompilableOpSequential3d(nn.Module):
    """OpSequential3d without isinstance dispatch in forward -- torch.compile-friendly.
    Op types are classified once at construction time.
    """

    _takes_cache_args: tuple[bool, ...]

    @staticmethod
    def _get_effective_op_class(op: nn.Module) -> type[nn.Module]:
        if isinstance(op, torch.distributed.algorithms._checkpoint.checkpoint_wrapper.CheckpointWrapper):
            return type(op._checkpoint_wrapped_module)
        return type(op)

    @classmethod
    def _op_takes_cache_args(cls, op: nn.Module) -> bool:
        return issubclass(cls._get_effective_op_class(op), _CACHE_ARG_OP_TYPES)

    def __init__(self, op_list: list[Optional[nn.Module]]):
        super().__init__()
        valid_ops = [op for op in op_list if op is not None]
        self.op_list = nn.ModuleList(valid_ops)
        self._takes_cache_args = tuple(self._op_takes_cache_args(op) for op in valid_ops)

    @classmethod
    def from_op_sequential_3d(cls, seq: OpSequential3d) -> "CompilableOpSequential3d":
        """Create from an existing OpSequential3d, sharing its op_list."""
        instance = cls.__new__(cls)
        nn.Module.__init__(instance)
        instance.op_list = seq.op_list
        instance._takes_cache_args = tuple(cls._op_takes_cache_args(op) for op in seq.op_list)
        return instance

    def load_state_dict_from_2d(self, state_dict: dict[str, torch.Tensor], method: str) -> None:
        for i, op in enumerate(self.op_list):
            if isinstance(op, (TritonRMSNorm2d, CompilableRMSNorm2d, nn.SiLU)):
                op.load_state_dict(get_submodule_weights(state_dict, f"op_list.{i}."))
            else:
                op.load_state_dict_from_2d(get_submodule_weights(state_dict, f"op_list.{i}."), method)

    def forward(
        self,
        x: torch.Tensor,
        feature_cache: list[torch.Tensor | None] | None = None,
        feat_idx: list[int] | None = None,
    ) -> tuple[torch.Tensor, None]:
        for op, takes_cache in zip(self.op_list, self._takes_cache_args):
            if takes_cache:
                x, _ = op(x, feature_cache, feat_idx)
            else:
                x = op(x)
        return x, None
