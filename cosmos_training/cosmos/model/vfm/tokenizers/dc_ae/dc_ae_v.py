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

from dataclasses import dataclass, field
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import MISSING
from tqdm import tqdm

from cosmos.model.vfm.tokenizers.dc_ae.dc_ae_v_ops import (
    ChannelDuplicatingPixelShuffleUpSampleLayer3d,
    CompilableOpSequential3d,
    CompilableRMSNorm2d,
    ConvLayer3d,
    ConvPixelShuffleUpSampleLayer3d,
    ConvPixelUnshuffleDownSampleLayer3d,
    CustomConv3d,
    IdentityLayer,
    OpSequential3d,
    PixelUnshuffleChannelAveragingDownSampleLayer3d,
    ResBlock3d,
    ResidualBlock3d,
    TritonRMSNorm2d,
    build_act,
    build_norm,
)


@dataclass
class BlockConfig:
    block_name: str = MISSING
    spatial_kernel_size: int = 3
    temporal_kernel_size: int = 1
    causal_chunk_length: Optional[int] = None
    spatial_padding_mode: Optional[str] = None
    temporal_padding_mode: Optional[str] = None


@dataclass
class SampleBlockConfig(BlockConfig):
    spatial_factor: int = 2
    temporal_factor: int = 1


@dataclass
class DCAEVEncoderConfig:
    in_channels: int = MISSING
    latent_channels: int = MISSING

    project_in_block_type: Any = field(
        default_factory=lambda: SampleBlockConfig(
            block_name="ConvPixelUnshuffle",
            spatial_factor=2,
            temporal_factor=1,
            spatial_kernel_size=3,
            temporal_kernel_size=1,
        )
    )
    width_list: tuple[int, ...] = (128, 256, 512, 512, 1024, 1024)
    depth_list: tuple[int, ...] = (2, 2, 2, 2, 2, 2)
    block_type: Any = field(
        default_factory=lambda: BlockConfig(block_name="ResBlock3d", spatial_kernel_size=3, temporal_kernel_size=1)
    )
    norm: Any = "trms2d"
    act: str = "silu"
    downsample_block_type: Any = field(
        default_factory=lambda: SampleBlockConfig(
            block_name="ConvPixelUnshuffle",
            spatial_factor=2,
            temporal_factor=1,
            spatial_kernel_size=3,
            temporal_kernel_size=1,
        )
    )
    downsample_shortcut: Optional[str] = "averaging"
    project_out_block_type: Any = field(
        default_factory=lambda: BlockConfig(block_name="ConvLayer3d", spatial_kernel_size=3, temporal_kernel_size=1)
    )

    zero_out: bool = MISSING


@dataclass
class DCAEVDecoderConfig:
    in_channels: int = MISSING
    latent_channels: int = MISSING

    project_in_block_type: Any = field(
        default_factory=lambda: BlockConfig(block_name="ConvLayer3d", spatial_kernel_size=3, temporal_kernel_size=1)
    )

    width_list: tuple[int, ...] = (128, 256, 512, 512, 1024, 1024)
    depth_list: tuple[int, ...] = (2, 2, 2, 2, 2, 2)
    block_type: Any = field(
        default_factory=lambda: BlockConfig(block_name="ResBlock3d", spatial_kernel_size=3, temporal_kernel_size=1)
    )
    norm: Any = "trms2d"
    act: Any = "silu"
    upsample_block_type: Any = field(
        default_factory=lambda: SampleBlockConfig(
            block_name="ConvPixelShuffle",
            spatial_factor=2,
            temporal_factor=1,
            spatial_kernel_size=3,
            temporal_kernel_size=1,
        )
    )
    upsample_shortcut: str = "duplicating"
    project_out_block_type: Any = field(
        default_factory=lambda: SampleBlockConfig(
            block_name="ConvPixelShuffle",
            spatial_factor=2,
            temporal_factor=1,
            spatial_kernel_size=3,
            temporal_kernel_size=1,
        )
    )
    out_norm: str = "trms2d"
    out_act: str = "silu"

    zero_out: bool = MISSING


@dataclass
class DCAEVConfig:
    in_channels: int = 3
    latent_channels: int = 32
    encoder: DCAEVEncoderConfig = field(
        default_factory=lambda: DCAEVEncoderConfig(
            in_channels="${..in_channels}",
            latent_channels="${..latent_channels}",
            zero_out="${..zero_out}",
        )
    )
    decoder: DCAEVDecoderConfig = field(
        default_factory=lambda: DCAEVDecoderConfig(
            in_channels="${..in_channels}",
            latent_channels="${..latent_channels}",
            zero_out="${..zero_out}",
        )
    )

    num_pad_frames: int = 0
    temporal_remainder: int = 0

    pretrained_path: Optional[str] = None
    pretrained_source: str = "dc-ae-v"
    pretrained_ema: bool = True
    zero_out: bool = False
    use_feature_cache: bool = False

    encode_temporal_tile_size: Optional[int] = None
    encode_temporal_tile_latent_size: Optional[int] = None
    decode_temporal_tile_size: Optional[int] = None
    decode_temporal_tile_latent_size: Optional[int] = None
    encode_temporal_tile_overlap_factor: float = 0.0
    decode_temporal_tile_overlap_factor: float = 0.0

    spatial_tile_size: Optional[int] = None
    spatial_tile_overlap_factor: float = 0.25

    scaling_factor: float = MISSING

    compilable: bool = False

    verbose: bool = False


def build_downsample_block(
    block_type: SampleBlockConfig, in_channels: int, out_channels: int, shortcut: Optional[str], zero_out: bool = False
) -> nn.Module:
    block_name = block_type.block_name
    kernel_size = (block_type.temporal_kernel_size, block_type.spatial_kernel_size, block_type.spatial_kernel_size)
    kwargs = {}
    if block_type.spatial_padding_mode is not None:
        kwargs["spatial_padding_mode"] = block_type.spatial_padding_mode
        kwargs["temporal_padding_mode"] = block_type.temporal_padding_mode
    if block_name in ["ConvPixelUnshuffle", "CausalConvPixelUnshuffle"]:
        block = ConvPixelUnshuffleDownSampleLayer3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            spatial_factor=block_type.spatial_factor,
            temporal_factor=block_type.temporal_factor,
            zero_out=zero_out,
            causal=block_name == "CausalConvPixelUnshuffle",
            **kwargs,
        )
    elif block_name == "ChunkCausalConvPixelUnshuffle":
        block = ConvPixelUnshuffleDownSampleLayer3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            spatial_factor=block_type.spatial_factor,
            temporal_factor=block_type.temporal_factor,
            causal_chunk_length=block_type.causal_chunk_length,
            **kwargs,
        )
    else:
        raise ValueError(f"block_name {block_name} is not supported for downsampling")
    if shortcut is None:
        pass
    elif shortcut == "averaging":
        shortcut_block = PixelUnshuffleChannelAveragingDownSampleLayer3d(
            in_channels=in_channels,
            out_channels=out_channels,
            spatial_factor=block_type.spatial_factor,
            temporal_factor=block_type.temporal_factor,
        )
        block = ResidualBlock3d(block, shortcut_block)
    else:
        raise ValueError(f"shortcut {shortcut} is not supported for downsample")
    return block


def build_upsample_block(
    block_type: SampleBlockConfig, in_channels: int, out_channels: int, shortcut: Optional[str], zero_out: bool = False
) -> nn.Module:
    block_name = block_type.block_name
    kernel_size = (block_type.temporal_kernel_size, block_type.spatial_kernel_size, block_type.spatial_kernel_size)
    kwargs = {}
    if block_type.spatial_padding_mode is not None:
        kwargs["spatial_padding_mode"] = block_type.spatial_padding_mode
        kwargs["temporal_padding_mode"] = block_type.temporal_padding_mode
    if block_name in ["ConvPixelShuffle", "CausalConvPixelShuffle"]:
        block = ConvPixelShuffleUpSampleLayer3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            spatial_factor=block_type.spatial_factor,
            temporal_factor=block_type.temporal_factor,
            zero_out=zero_out,
            causal=block_name == "CausalConvPixelShuffle",
            **kwargs,
        )
    elif block_name in ["ChunkCausalConvPixelShuffle"]:
        block = ConvPixelShuffleUpSampleLayer3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            spatial_factor=block_type.spatial_factor,
            temporal_factor=block_type.temporal_factor,
            zero_out=zero_out,
            causal_chunk_length=block_type.causal_chunk_length,
            **kwargs,
        )
    else:
        raise ValueError(f"block_name {block_name} is not supported for upsampling")
    if shortcut is None:
        pass
    elif shortcut == "duplicating":
        shortcut_block = ChannelDuplicatingPixelShuffleUpSampleLayer3d(
            in_channels=in_channels,
            out_channels=out_channels,
            spatial_factor=block_type.spatial_factor,
            temporal_factor=block_type.temporal_factor,
        )
        block = ResidualBlock3d(block, shortcut_block)
    else:
        raise ValueError(f"shortcut {shortcut} is not supported for upsample")
    return block


def build_block(
    block_type: BlockConfig, channels: int, norm: Optional[str], act: Optional[str], zero_out: bool
) -> nn.Module:
    block_name = block_type.block_name
    kernel_size = (block_type.temporal_kernel_size, block_type.spatial_kernel_size, block_type.spatial_kernel_size)
    kwargs = {}
    if block_type.spatial_padding_mode is not None:
        kwargs["spatial_padding_mode"] = block_type.spatial_padding_mode
        kwargs["temporal_padding_mode"] = block_type.temporal_padding_mode
    if block_name in ["ResBlock3d", "CausalResBlock3d"]:
        main_block = ResBlock3d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            stride=1,
            use_bias=(True, False),
            norm=(None, norm),
            act_func=(act, None),
            zero_out=zero_out,
            causal=block_name == "CausalResBlock3d",
            **kwargs,
        )
        block = ResidualBlock3d(main_block, IdentityLayer())
    elif block_name in ["ChunkCausalResBlock3d"]:
        main_block = ResBlock3d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            stride=1,
            use_bias=(True, False),
            norm=(None, norm),
            act_func=(act, None),
            zero_out=zero_out,
            causal_chunk_length=block_type.causal_chunk_length,
            **kwargs,
        )
        block = ResidualBlock3d(main_block, IdentityLayer())
    else:
        raise ValueError(f"block_name {block_name} is not supported")
    return block


def build_stage_main(
    width: int, depth: int, block_type: BlockConfig | list[BlockConfig], norm: str, act: str, zero_out: bool = False
) -> list[nn.Module]:
    assert isinstance(block_type, BlockConfig) or (isinstance(block_type, list) and depth == len(block_type))
    stage = []
    for d in range(depth):
        current_block_type = block_type[d] if isinstance(block_type, list) else block_type
        block = build_block(
            block_type=current_block_type,
            channels=width,
            norm=norm,
            act=act,
            zero_out=zero_out,
        )
        stage.append(block)
    return stage


def build_encoder_project_in_block(block_type: SampleBlockConfig, in_channels: int, out_channels: int):
    block = build_downsample_block(
        block_type=block_type, in_channels=in_channels, out_channels=out_channels, shortcut=None
    )
    return block


def build_encoder_project_out_block(block_type: BlockConfig, in_channels: int, out_channels: int):
    block_name = block_type.block_name
    kernel_size = (block_type.temporal_kernel_size, block_type.spatial_kernel_size, block_type.spatial_kernel_size)
    kwargs = {}
    if block_type.spatial_padding_mode is not None:
        kwargs["spatial_padding_mode"] = block_type.spatial_padding_mode
        kwargs["temporal_padding_mode"] = block_type.temporal_padding_mode
    if block_name in ["ConvLayer3d", "CausalConvLayer3d"]:
        block = ConvLayer3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            use_bias=True,
            norm=None,
            act_func=None,
            causal=block_name == "CausalConvLayer3d",
            **kwargs,
        )
    elif block_name in ["ChunkCausalConvLayer3d"]:
        block = ConvLayer3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            use_bias=True,
            norm=None,
            act_func=None,
            causal_chunk_length=block_type.causal_chunk_length,
            **kwargs,
        )
    else:
        raise ValueError(f"encoder project out block name {block_name} is not supported")
    return block


def build_decoder_project_in_block(block_type: BlockConfig, in_channels: int, out_channels: int):
    block_name = block_type.block_name
    kernel_size = (block_type.temporal_kernel_size, block_type.spatial_kernel_size, block_type.spatial_kernel_size)
    kwargs = {}
    if block_type.spatial_padding_mode is not None:
        kwargs["spatial_padding_mode"] = block_type.spatial_padding_mode
        kwargs["temporal_padding_mode"] = block_type.temporal_padding_mode
    if block_name in ["ConvLayer3d", "CausalConvLayer3d"]:
        block = ConvLayer3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            use_bias=True,
            norm=None,
            act_func=None,
            causal=block_name == "CausalConvLayer3d",
            **kwargs,
        )
    elif block_name in ["ChunkCausalConvLayer3d"]:
        block = ConvLayer3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            use_bias=True,
            norm=None,
            act_func=None,
            causal_chunk_length=block_type.causal_chunk_length,
            **kwargs,
        )
    else:
        raise ValueError(f"decoder project in block name {block_name} is not supported")
    return block


def build_decoder_project_out_block(
    block_type: SampleBlockConfig, in_channels: int, out_channels: int, norm: Optional[str], act: Optional[str]
):
    layers: list[nn.Module] = [
        build_norm(norm, in_channels),
        build_act(act),
        build_upsample_block(block_type=block_type, in_channels=in_channels, out_channels=out_channels, shortcut=None),
    ]
    return OpSequential3d(layers)


class DCAEVEncoder(nn.Module):
    def __init__(self, cfg: DCAEVEncoderConfig):
        super().__init__()
        self.cfg = cfg

        start_stage = 0
        while cfg.depth_list[start_stage] == 0:
            start_stage += 1
        self.project_in = build_encoder_project_in_block(
            block_type=cfg.project_in_block_type,
            in_channels=cfg.in_channels,
            out_channels=cfg.width_list[start_stage],
        )

        num_stages = len(cfg.width_list)
        self.num_stages = num_stages
        assert len(cfg.depth_list) == num_stages
        assert len(cfg.width_list) == num_stages
        assert isinstance(cfg.block_type, BlockConfig) or (
            isinstance(cfg.block_type, list) and len(cfg.block_type) == num_stages
        )
        assert isinstance(cfg.norm, str) or (isinstance(cfg.norm, list) and len(cfg.norm) == num_stages)
        assert isinstance(cfg.downsample_block_type, SampleBlockConfig) or (
            isinstance(cfg.downsample_block_type, list) and len(cfg.downsample_block_type) == num_stages - 1
        )

        self.stages: list[OpSequential3d] = []
        for stage_id, (width, depth) in enumerate(zip(cfg.width_list, cfg.depth_list)):
            block_type = cfg.block_type[stage_id] if isinstance(cfg.block_type, list) else cfg.block_type
            norm = cfg.norm[stage_id] if isinstance(cfg.norm, list) else cfg.norm
            stage = build_stage_main(
                width=width,
                depth=depth,
                block_type=block_type,
                norm=norm,
                act=cfg.act,
                zero_out=cfg.zero_out,
            )
            if stage_id < num_stages - 1 and depth > 0:
                downsample_block_type = (
                    cfg.downsample_block_type[stage_id]
                    if isinstance(cfg.downsample_block_type, list)
                    else cfg.downsample_block_type
                )
                downsample_block = build_downsample_block(
                    block_type=downsample_block_type,
                    in_channels=width,
                    out_channels=cfg.width_list[stage_id + 1],
                    shortcut=cfg.downsample_shortcut,
                    zero_out=cfg.zero_out,
                )
                stage.append(downsample_block)
            self.stages.append(OpSequential3d(stage))
        self.stages = nn.ModuleList(self.stages)

        self.project_out = build_encoder_project_out_block(
            block_type=cfg.project_out_block_type,
            in_channels=cfg.width_list[-1],
            out_channels=cfg.latent_channels,
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        feature_cache: list[torch.Tensor | None] | None = None,
        feat_idx: list[int] | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        x, _ = self.project_in(x, feature_cache, feat_idx)
        for stage in self.stages:
            if len(stage.op_list) == 0:
                continue
            x, _ = stage(x, feature_cache, feat_idx)
        x, _ = self.project_out(x, feature_cache, feat_idx)
        return x, {}


class DCAEVDecoder(nn.Module):
    def __init__(self, cfg: DCAEVDecoderConfig):
        super().__init__()
        self.cfg = cfg

        self.project_in = build_decoder_project_in_block(
            block_type=cfg.project_in_block_type,
            in_channels=cfg.latent_channels,
            out_channels=cfg.width_list[-1],
        )

        num_stages = len(cfg.width_list)
        self.num_stages = num_stages
        assert len(cfg.depth_list) == num_stages
        assert len(cfg.width_list) == num_stages
        assert isinstance(cfg.block_type, BlockConfig) or (
            isinstance(cfg.block_type, list) and len(cfg.block_type) == num_stages
        )
        assert isinstance(cfg.norm, str) or (isinstance(cfg.norm, list) and len(cfg.norm) == num_stages)
        assert isinstance(cfg.act, str) or (isinstance(cfg.act, list) and len(cfg.act) == num_stages)
        assert isinstance(cfg.upsample_block_type, SampleBlockConfig) or (
            isinstance(cfg.upsample_block_type, list) and len(cfg.upsample_block_type) == num_stages - 1
        )
        self.stages: list[OpSequential3d] = []
        self.spatial_compression_ratio = 1
        self.temporal_compression_ratio = 1
        for stage_id, (width, depth) in reversed(list(enumerate(zip(cfg.width_list, cfg.depth_list)))):
            stage = []
            if stage_id < num_stages - 1 and depth > 0:
                upsample_block_type = (
                    cfg.upsample_block_type[stage_id]
                    if isinstance(cfg.upsample_block_type, list)
                    else cfg.upsample_block_type
                )
                upsample_block = build_upsample_block(
                    block_type=upsample_block_type,
                    in_channels=cfg.width_list[stage_id + 1],
                    out_channels=width,
                    shortcut=cfg.upsample_shortcut,
                    zero_out=cfg.zero_out,
                )
                stage.append(upsample_block)
                self.spatial_compression_ratio *= upsample_block_type.spatial_factor
                self.temporal_compression_ratio *= upsample_block_type.temporal_factor

            block_type = cfg.block_type[stage_id] if isinstance(cfg.block_type, list) else cfg.block_type
            norm = cfg.norm[stage_id] if isinstance(cfg.norm, list) else cfg.norm
            act = cfg.act[stage_id] if isinstance(cfg.act, list) else cfg.act
            stage.extend(
                build_stage_main(
                    width=width, depth=depth, block_type=block_type, norm=norm, act=act, zero_out=cfg.zero_out
                )
            )
            self.stages.insert(0, OpSequential3d(stage))
        self.stages = nn.ModuleList(self.stages)

        start_stage = 0
        while cfg.depth_list[start_stage] == 0:
            start_stage += 1
        self.project_out = build_decoder_project_out_block(
            block_type=cfg.project_out_block_type,
            in_channels=cfg.width_list[start_stage],
            out_channels=cfg.in_channels,
            norm=cfg.out_norm,
            act=cfg.out_act,
        )
        self.spatial_compression_ratio *= cfg.project_out_block_type.spatial_factor
        self.temporal_compression_ratio *= cfg.project_out_block_type.temporal_factor

    def forward(
        self,
        x: torch.Tensor,
        feature_cache: list[torch.Tensor | None] | None = None,
        feat_idx: list[int] | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        x, _ = self.project_in(x, feature_cache, feat_idx)
        for stage_id, stage in reversed(list(enumerate(self.stages))):
            if len(stage.op_list) == 0:
                continue
            x, _ = stage(x, feature_cache, feat_idx)
        x, _ = self.project_out(x, feature_cache, feat_idx)
        return x, {}


def _replace_with_compilable_ops(module: nn.Module) -> None:
    """Recursively replace compile-unfriendly ops throughout *module*:
    - TritonRMSNorm2d   -> CompilableRMSNorm2d  (pure PyTorch, no Triton kernel)
    - OpSequential3d    -> CompilableOpSequential3d (no isinstance dispatch)
    - CustomConv3d      -> torch.nn.Conv3d
    """
    for name, child in list(module.named_children()):
        if isinstance(child, TritonRMSNorm2d):
            compilable = CompilableRMSNorm2d(child.normalized_shape, eps=child.eps)
            compilable.weight = child.weight
            compilable.bias = child.bias
            setattr(module, name, compilable)
        elif isinstance(child, CustomConv3d):
            compilable_conv = torch.nn.Conv3d(
                child.in_channels,
                child.out_channels,
                child.kernel_size,
                stride=child.stride,
                padding=child.padding,
                dilation=child.dilation,
                groups=child.groups,
                bias=child.bias is not None,
                padding_mode=child.padding_mode,
            )
            compilable_conv.weight = child.weight
            if child.bias is not None:
                compilable_conv.bias = child.bias
            setattr(module, name, compilable_conv)
        elif isinstance(child, OpSequential3d) and not isinstance(child, CompilableOpSequential3d):
            compilable_seq = CompilableOpSequential3d.from_op_sequential_3d(child)
            setattr(module, name, compilable_seq)
            _replace_with_compilable_ops(compilable_seq)
        else:
            _replace_with_compilable_ops(child)


class CompilableDCAEVEncoder(DCAEVEncoder):
    """DCAEVEncoder with compile-friendly ops."""

    def __init__(self, cfg: DCAEVEncoderConfig):
        super().__init__(cfg)
        _replace_with_compilable_ops(self)

    def forward(
        self,
        x: torch.Tensor,
        *,
        feature_cache: list[torch.Tensor | None] | None = None,
        feat_idx: list[int] | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        x = x.to(memory_format=torch.channels_last_3d)
        x, _ = self.project_in(x, feature_cache, feat_idx)
        for stage in self.stages:
            if len(stage.op_list) == 0:
                continue
            x, _ = stage(x, feature_cache, feat_idx)
        x, _ = self.project_out(x, feature_cache, feat_idx)
        return x, {}


class CompilableDCAEVDecoder(DCAEVDecoder):
    """DCAEVDecoder with compile-friendly ops."""

    def __init__(self, cfg: DCAEVDecoderConfig):
        super().__init__(cfg)
        _replace_with_compilable_ops(self)


def _count_causal_convs(model: nn.Module) -> int:
    """Count ConvLayer3d instances with causal or causal_chunk_length set.
    Used to pre-allocate the flat feature cache list."""
    count = 0
    for m in model.modules():
        if isinstance(m, ConvLayer3d) and (m.causal or m.causal_chunk_length is not None):
            count += 1
    return count


def _build_encoder_feature_cache(
    encoder: nn.Module,
    batch_size: int,
    height: int,
    width: int,
    dtype: torch.dtype,
    device: torch.device,
) -> list[torch.Tensor]:
    """Pre-allocate zero-filled feature cache for a causal encoder.

    Walks the encoder in forward order, creating a ``(B, C_in, pad, H, W)``
    zero tensor for every causal ``ConvLayer3d`` and shrinking H/W whenever
    a ``ConvPixelUnshuffleDownSampleLayer3d`` is encountered.
    """
    cache: list[torch.Tensor] = []
    h = height
    w = width

    def _visit(module: nn.Module) -> None:
        nonlocal h, w
        if isinstance(module, ConvPixelUnshuffleDownSampleLayer3d):
            _visit(module.conv)
            h //= module.spatial_factor
            w //= module.spatial_factor
        elif isinstance(module, ConvLayer3d):
            if module.causal:
                cache.append(
                    torch.zeros(
                        batch_size,
                        module.conv.in_channels,
                        module.custom_padding[4],
                        h,
                        w,
                        dtype=dtype,
                        device=device,
                    )
                )
        elif isinstance(module, ResBlock3d):
            _visit(module.conv1)
            _visit(module.conv2)
        elif isinstance(module, ResidualBlock3d):
            if module.main is not None:
                _visit(module.main)
        elif isinstance(module, (OpSequential3d, CompilableOpSequential3d)):
            for op in module.op_list:
                _visit(op)
        else:
            raise ValueError(f"Unsupported module: {type(module)}")

    _visit(encoder.project_in)
    for stage in encoder.stages:
        _visit(stage)
    _visit(encoder.project_out)

    return cache


class DCAEV(nn.Module):
    def __init__(self, cfg: DCAEVConfig):
        super().__init__()
        self.cfg = cfg
        if cfg.compilable:
            self.encoder = CompilableDCAEVEncoder(cfg.encoder)
            self.decoder = DCAEVDecoder(cfg.decoder)
        else:
            self.encoder = DCAEVEncoder(cfg.encoder)
            self.decoder = DCAEVDecoder(cfg.decoder)

        if cfg.pretrained_path is not None:
            self.load_model()

    def load_model(self):
        if self.cfg.pretrained_source == "dc-ae-v-fsdp":
            checkpoint = torch.load(self.cfg.pretrained_path, map_location="cpu", weights_only=True)
            if self.cfg.pretrained_ema and "ema_model_state_dict" in checkpoint:
                state_dict = checkpoint["ema_model_state_dict"]
                state_dict = state_dict[list(state_dict)[0]]
            else:
                state_dict = checkpoint["model_state_dict"]
            self.load_state_dict(state_dict)
        else:
            raise NotImplementedError

    def blend_t(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
        blend_extent = min(a.shape[-3], b.shape[-3], blend_extent)
        for x in range(blend_extent):
            blend_ratio = x / blend_extent
            b[:, :, x, :, :] = a[:, :, -blend_extent + x, :, :] * (1 - blend_ratio) + b[:, :, x, :, :] * blend_ratio
        return b

    def blend_w(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
        blend_extent = min(a.shape[-2], b.shape[-2], blend_extent)
        for y in range(blend_extent):
            b[..., y, :] = a[..., -blend_extent + y, :] * (1 - y / blend_extent) + b[..., y, :] * (y / blend_extent)
        return b

    def blend_h(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
        blend_extent = min(a.shape[-1], b.shape[-1], blend_extent)
        for x in range(blend_extent):
            b[..., x] = a[..., -blend_extent + x] * (1 - x / blend_extent) + b[..., x] * (x / blend_extent)
        return b

    def temporal_tiled_encode(self, x: torch.Tensor) -> torch.Tensor:
        overlap_size = int(self.cfg.encode_temporal_tile_size * (1 - self.cfg.encode_temporal_tile_overlap_factor))
        blend_extent = int(self.cfg.encode_temporal_tile_latent_size * self.cfg.encode_temporal_tile_overlap_factor)
        t_limit = self.cfg.encode_temporal_tile_latent_size - blend_extent

        if self.cfg.use_feature_cache:
            feature_cache: list[torch.Tensor] | None = _build_encoder_feature_cache(
                self.encoder,
                batch_size=x.shape[0],
                height=x.shape[3],
                width=x.shape[4],
                dtype=x.dtype,
                device=x.device,
            )
            feat_idx: list[int] | None = [0]
        else:
            feature_cache = None
            feat_idx = None

        # Split the video into tiles and encode them separately.
        # For compiled tokenizer, pad the last tile to full size so the compiled
        # encoder always sees the same shape (avoids recompilations).
        # Otherwise only pad to multiple of compression_factor to avoid errors in unshuffle layer.
        tile_size = self.cfg.encode_temporal_tile_size
        compression_factor = self.decoder.temporal_compression_ratio

        row = []
        for i in tqdm(range(0, x.shape[2], overlap_size), desc="Tiled Encode", disable=not self.cfg.verbose):
            tile = x[:, :, i : i + tile_size, :, :]
            actual_t = tile.shape[2]
            remove_padding = False
            if actual_t < tile_size and self.cfg.compilable:
                tile = F.pad(tile, (0, 0, 0, 0, 0, tile_size - actual_t))
                remove_padding = True
            assert tile.numel() < 1 << 31, "Tile size exceeds the int32 limit (torch compile and/or cudnn indexing)"

            if feat_idx is not None:
                feat_idx[0] = 0
            if feature_cache is not None and self.cfg.compilable:
                feature_cache = [f.clone() if f is not None else None for f in feature_cache]
            tile = self.encoder(tile, feature_cache=feature_cache, feat_idx=feat_idx)[0].clone()
            if remove_padding:
                valid_latent_t = (actual_t + compression_factor - 1) // compression_factor
                tile = tile[:, :, :valid_latent_t, :, :]
            row.append(tile)

        result_row = []
        for i, tile in enumerate(row):
            if i > 0:
                tile = self.blend_t(row[i - 1], tile, blend_extent)
            result_row.append(tile[:, :, :t_limit, :, :])

        return torch.cat(result_row, dim=2)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if self.cfg.num_pad_frames > 0:
            x = F.pad(x, (0, 0, 0, 0, self.cfg.num_pad_frames, 0), mode="replicate")
        if self.cfg.spatial_tile_size is not None:
            raise NotImplementedError("Spatial tiling is not supported for DCAEV")
        elif self.cfg.encode_temporal_tile_size is not None:
            x = self.temporal_tiled_encode(x)
        else:
            x, _ = self.encoder(x)
        return x * self.cfg.scaling_factor

    def temporal_tiled_decode(self, z: torch.Tensor) -> torch.Tensor:
        overlap_size = int(
            self.cfg.decode_temporal_tile_latent_size * (1 - self.cfg.decode_temporal_tile_overlap_factor)
        )
        blend_extent = int(self.cfg.decode_temporal_tile_size * self.cfg.decode_temporal_tile_overlap_factor)
        t_limit = self.cfg.decode_temporal_tile_size - blend_extent

        if self.cfg.use_feature_cache:
            feature_cache: list[torch.Tensor | None] | None = [None] * _count_causal_convs(self.decoder)
            feat_idx: list[int] | None = [0]
        else:
            feature_cache = None
            feat_idx = None

        row = []
        for i in tqdm(range(0, z.shape[2], overlap_size), desc="Tiled Decode", disable=not self.cfg.verbose):
            tile = z[:, :, i : i + self.cfg.decode_temporal_tile_latent_size, :, :]
            if feat_idx is not None:
                feat_idx[0] = 0
            decoded, _ = self.decoder(tile, feature_cache=feature_cache, feat_idx=feat_idx)
            row.append(decoded.clone())
        result_row = []
        for i, tile in enumerate(row):
            if i > 0:
                tile = self.blend_t(row[i - 1], tile, blend_extent)
            result_row.append(tile[:, :, :t_limit, :, :])

        return torch.cat(result_row, dim=2)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        z = z / self.cfg.scaling_factor
        if self.cfg.spatial_tile_size is not None:
            raise NotImplementedError("Spatial tiling is not supported for DCAEV")
        elif self.cfg.decode_temporal_tile_size is not None:
            z = self.temporal_tiled_decode(z)
        else:
            z, _ = self.decoder(z)
        if self.cfg.num_pad_frames > 0:
            z = z[:, :, self.cfg.num_pad_frames :, :, :]
        return z

    @torch.no_grad()
    def reconstruct_image(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """
        x: (B, 3, H, W) [-1, 1]
        """
        x = x.unsqueeze(2)
        if self.cfg.num_pad_frames == 0:
            x = x.repeat(1, 1, self.decoder.temporal_compression_ratio, 1, 1)
        elif self.cfg.num_pad_frames == self.decoder.temporal_compression_ratio - 1:
            pass
        else:
            raise ValueError(
                f"num_pad_frames {self.cfg.num_pad_frames} and temporal_compression_ratio {self.decoder.temporal_compression_ratio} is not supported for image reconsruction"
            )
        z = self.encode(x)
        y = self.decode(z)
        return y[:, :, 0], {"latent": z}


def dc_ae_v_f32t4_encoder_causal_decoder_chunk_causal_4(
    name: str,
    pretrained_path: Optional[str],
) -> DCAEVConfig:
    if name in [
        "dcae4x32x32_c64_t120_256p_fps_all_encoder_causal_decoder_chunk_causal_4_nogan_cosmos_pad_7_v0.1",
    ]:
        latent_channels, num_pad_frames, temporal_remainder, scaling_factor = 64, 7, 1, 0.7103
        encoder_width_list = [128, 256, 512, 512, 1024, 1024, 1024]
    elif name in [
        "dcae4x32x32_c32_t120_256p_fps_all_encoder_causal_decoder_chunk_causal_4_nogan_cosmos_pad_11_v0.1",
    ]:
        latent_channels, num_pad_frames, temporal_remainder, scaling_factor = 32, 11, 1, 0.6774
        encoder_width_list = [128, 256, 512, 512, 1024, 1024, 1024]
    elif name in [
        "dcae4x32x32_c64_t120_256p_fps_all_encoder_causal_decoder_chunk_causal_4_nogan_cosmos_pad_7_v0.2",
    ]:
        latent_channels, num_pad_frames, temporal_remainder, scaling_factor = 64, 7, 1, 0.7103
        encoder_width_list = [0, 64, 128, 512, 1024, 1024, 1024]
    else:
        raise ValueError(f"model {name} is not supported")

    def causal_downsample(sf, tf):
        return SampleBlockConfig(
            block_name="CausalConvPixelUnshuffle",
            spatial_factor=sf,
            temporal_factor=tf,
            spatial_kernel_size=3,
            temporal_kernel_size=3,
        )

    def chunk_causal_upsample(sf, tf, cl):
        return SampleBlockConfig(
            block_name="ChunkCausalConvPixelShuffle",
            spatial_factor=sf,
            temporal_factor=tf,
            spatial_kernel_size=3,
            temporal_kernel_size=3,
            causal_chunk_length=cl,
        )

    cfg = DCAEVConfig(
        latent_channels=latent_channels,
        use_feature_cache=True,
        encode_temporal_tile_size=16,
        encode_temporal_tile_latent_size=4,
        decode_temporal_tile_size=16,
        decode_temporal_tile_latent_size=4,
        num_pad_frames=num_pad_frames,
        temporal_remainder=temporal_remainder,
        scaling_factor=scaling_factor,
        pretrained_source="dc-ae-v-fsdp",
        pretrained_path=pretrained_path,
        encoder=DCAEVEncoderConfig(
            in_channels=3,
            latent_channels=latent_channels,
            zero_out=False,
            project_in_block_type=SampleBlockConfig(
                block_name="CausalConvPixelUnshuffle",
                spatial_factor=2,
                temporal_factor=1,
                spatial_kernel_size=3,
                temporal_kernel_size=3,
            ),
            depth_list=(0, 5, 10, 4, 4, 4, 4),
            width_list=tuple(encoder_width_list),
            block_type=BlockConfig(
                block_name="CausalResBlock3d",
                spatial_kernel_size=3,
                temporal_kernel_size=3,
            ),
            downsample_block_type=[
                causal_downsample(2, 1),
                causal_downsample(2, 1),
                causal_downsample(2, 1),
                causal_downsample(2, 1),
                causal_downsample(2, 1),
                causal_downsample(1, 4),
            ],
            project_out_block_type=BlockConfig(
                block_name="CausalConvLayer3d",
                spatial_kernel_size=3,
                temporal_kernel_size=3,
            ),
        ),
        decoder=DCAEVDecoderConfig(
            in_channels=3,
            latent_channels=latent_channels,
            zero_out=False,
            depth_list=(0, 5, 10, 4, 4, 4, 4),
            width_list=(128, 256, 512, 512, 1024, 1024, 1024),
            project_in_block_type=BlockConfig(
                block_name="ChunkCausalConvLayer3d",
                spatial_kernel_size=3,
                temporal_kernel_size=3,
                causal_chunk_length=1,
            ),
            block_type=[
                BlockConfig(
                    block_name="ChunkCausalResBlock3d",
                    spatial_kernel_size=3,
                    temporal_kernel_size=3,
                    causal_chunk_length=4,
                ),
                BlockConfig(
                    block_name="ChunkCausalResBlock3d",
                    spatial_kernel_size=3,
                    temporal_kernel_size=3,
                    causal_chunk_length=4,
                ),
                BlockConfig(
                    block_name="ChunkCausalResBlock3d",
                    spatial_kernel_size=3,
                    temporal_kernel_size=3,
                    causal_chunk_length=4,
                ),
                BlockConfig(
                    block_name="ChunkCausalResBlock3d",
                    spatial_kernel_size=3,
                    temporal_kernel_size=3,
                    causal_chunk_length=4,
                ),
                BlockConfig(
                    block_name="ChunkCausalResBlock3d",
                    spatial_kernel_size=3,
                    temporal_kernel_size=3,
                    causal_chunk_length=4,
                ),
                BlockConfig(
                    block_name="ChunkCausalResBlock3d",
                    spatial_kernel_size=3,
                    temporal_kernel_size=3,
                    causal_chunk_length=4,
                ),
                BlockConfig(
                    block_name="ChunkCausalResBlock3d",
                    spatial_kernel_size=3,
                    temporal_kernel_size=3,
                    causal_chunk_length=1,
                ),
            ],
            upsample_block_type=[
                chunk_causal_upsample(2, 1, 4),
                chunk_causal_upsample(2, 1, 4),
                chunk_causal_upsample(2, 1, 4),
                chunk_causal_upsample(2, 1, 4),
                chunk_causal_upsample(2, 1, 4),
                chunk_causal_upsample(1, 4, 1),
            ],
            project_out_block_type=SampleBlockConfig(
                block_name="ChunkCausalConvPixelShuffle",
                spatial_factor=2,
                temporal_factor=1,
                spatial_kernel_size=3,
                temporal_kernel_size=3,
                causal_chunk_length=4,
            ),
        ),
    )
    return cfg
