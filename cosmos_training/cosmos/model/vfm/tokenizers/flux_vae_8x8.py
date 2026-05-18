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

from dataclasses import dataclass
from typing import Optional

import safetensors.torch
import torch
from einops import rearrange
from torch import Tensor, nn

from cosmos.utils.easy_io import easy_io
from cosmos.model.vfm.tokenizers.interface import VideoTokenizerInterface


@dataclass
class FluxAEParams:
    resolution: int
    in_channels: int
    downsample: int
    ch: int
    out_ch: int
    ch_mult: list[int]
    num_res_blocks: int
    z_channels: int
    scale_factor: float
    shift_factor: float


def swish(x: Tensor) -> Tensor:
    return x * torch.sigmoid(x)


class AttnBlock(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels

        self.norm = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)

        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def attention(self, h_: Tensor) -> Tensor:  # h_: [B,C,H,W]
        h_ = self.norm(h_)  # [B,C,H,W]
        q = self.q(h_)  # [B,C,H,W]
        k = self.k(h_)  # [B,C,H,W]
        v = self.v(h_)  # [B,C,H,W]

        b, c, h, w = q.shape
        q = rearrange(q, "b c h w -> b 1 (h w) c").contiguous()  # [B,1,H*W,C]
        k = rearrange(k, "b c h w -> b 1 (h w) c").contiguous()  # [B,1,H*W,C]
        v = rearrange(v, "b c h w -> b 1 (h w) c").contiguous()  # [B,1,H*W,C]
        h_ = nn.functional.scaled_dot_product_attention(q, k, v)  # [B,1,H*W,C]

        return rearrange(h_, "b 1 (h w) c -> b c h w", h=h, w=w, c=c, b=b)  # [B,C,H,W]

    def forward(self, x: Tensor) -> Tensor:
        return x + self.proj_out(self.attention(x))


class ResnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=out_channels, eps=1e-6, affine=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if self.in_channels != self.out_channels:
            self.nin_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = swish(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = swish(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)

        return x + h


class Downsample(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        # no asymmetric padding in torch conv, must do it ourselves
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x: Tensor):  # x: [B,C,H,W] -> [B,C,H//2,W//2]
        pad = (0, 1, 0, 1)
        x = nn.functional.pad(x, pad, mode="constant", value=0)  # [B,C,H+1,W+1]
        x = self.conv(x)  # [B,C,H//2,W//2]
        return x


class Upsample(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: Tensor):  # x: [B,C,H,W] -> [B,C,H*2,W*2]
        x = nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")  # [B,C,H*2,W*2]
        x = self.conv(x)  # [B,C,H*2,W*2]
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        resolution: int,
        in_channels: int,
        ch: int,
        ch_mult: list[int],
        num_res_blocks: int,
        z_channels: int,
    ):
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        # downsampling
        self.conv_in = nn.Conv2d(in_channels, self.ch, kernel_size=3, stride=1, padding=1)

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        block_in = self.ch
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out))
                block_in = block_out
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in)

        # end
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(block_in, 2 * z_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: Tensor) -> Tensor:  # x: [B,in_channels,H,W] -> [B,2*z_channels,H//8,W//8]
        # downsampling
        hs = [self.conv_in(x)]  # hs[0]: [B,ch,H,W]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1])
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))
        # h: [B,ch*ch_mult[-1],H//8,W//8]

        # middle
        h = hs[-1]
        h = self.mid.block_1(h)  # [B,ch*ch_mult[-1],H//8,W//8]
        h = self.mid.attn_1(h)  # [B,ch*ch_mult[-1],H//8,W//8]
        h = self.mid.block_2(h)  # [B,ch*ch_mult[-1],H//8,W//8]
        # end
        h = self.norm_out(h)
        h = swish(h)
        h = self.conv_out(h)  # [B,2*z_channels,H//8,W//8]
        return h


class Decoder(nn.Module):
    def __init__(
        self,
        ch: int,
        out_ch: int,
        ch_mult: list[int],
        num_res_blocks: int,
        in_channels: int,
        resolution: int,
        z_channels: int,
    ):
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.ffactor = 2 ** (self.num_resolutions - 1)

        # compute in_ch_mult, block_in and curr_res at lowest res
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)

        # z to block_in
        self.conv_in = nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out))
                block_in = block_out
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, z: Tensor) -> Tensor:  # z: [B,z_channels,H,W] -> [B,out_ch,H*8,W*8]
        # z to block_in
        h = self.conv_in(z)  # [B,ch*ch_mult[-1],H,W]

        # middle
        h = self.mid.block_1(h)  # [B,ch*ch_mult[-1],H,W]
        h = self.mid.attn_1(h)  # [B,ch*ch_mult[-1],H,W]
        h = self.mid.block_2(h)  # [B,ch*ch_mult[-1],H,W]

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)
        # h: [B,ch,H*8,W*8]

        # end
        h = self.norm_out(h)
        h = swish(h)
        h = self.conv_out(h)  # [B,out_ch,H*8,W*8]
        return h


class DiagonalGaussian(nn.Module):
    def __init__(self, sample: bool = True, chunk_dim: int = 1):
        super().__init__()
        self.sample = sample
        self.chunk_dim = chunk_dim

    def forward(self, z: Tensor) -> Tensor:  # z: [B,2*z_channels,...] -> [B,z_channels,...]
        mean, logvar = torch.chunk(z, 2, dim=self.chunk_dim)  # mean,logvar: [B,z_channels,...]
        if self.sample:
            std = torch.exp(0.5 * logvar)  # [B,z_channels,...]
            return mean + std * torch.randn_like(mean)  # [B,z_channels,...]
        else:
            return mean  # [B,z_channels,...]


class FluxVAE(nn.Module):
    def __init__(self, params: FluxAEParams):
        super().__init__()
        self.encoder = Encoder(
            resolution=params.resolution,
            in_channels=params.in_channels,
            ch=params.ch,
            ch_mult=params.ch_mult,
            num_res_blocks=params.num_res_blocks,
            z_channels=params.z_channels,
        )
        self.decoder = Decoder(
            resolution=params.resolution,
            in_channels=params.in_channels,
            ch=params.ch,
            out_ch=params.out_ch,
            ch_mult=params.ch_mult,
            num_res_blocks=params.num_res_blocks,
            z_channels=params.z_channels,
        )
        self.reg = DiagonalGaussian()

        self.scale_factor = params.scale_factor
        self.shift_factor = params.shift_factor

    def encode(self, x: Tensor) -> Tensor:  # x: [B,in_channels,H,W] -> [B,z_channels,H//8,W//8]
        z = self.reg(self.encoder(x))  # [B,z_channels,H//8,W//8]
        z = self.scale_factor * (z - self.shift_factor)  # [B,z_channels,H//8,W//8]
        return z

    def decode(self, z: Tensor) -> Tensor:  # z: [B,z_channels,H,W] -> [B,out_ch,H*8,W*8]
        z = z / self.scale_factor + self.shift_factor  # [B,z_channels,H,W]
        return self.decoder(z)  # [B,out_ch,H*8,W*8]

    def forward(self, x: Tensor) -> Tensor:
        return self.decode(self.encode(x))


def print_load_warning(missing: list[str], unexpected: list[str]) -> None:
    if len(missing) > 0 and len(unexpected) > 0:
        print(f"Got {len(missing)} missing keys:\n\t" + "\n\t".join(missing))
        print("\n" + "-" * 79 + "\n")
        print(f"Got {len(unexpected)} unexpected keys:\n\t" + "\n\t".join(unexpected))
    elif len(missing) > 0:
        print(f"Got {len(missing)} missing keys:\n\t" + "\n\t".join(missing))
    elif len(unexpected) > 0:
        print(f"Got {len(unexpected)} unexpected keys:\n\t" + "\n\t".join(unexpected))


def load_ae(local_path: str, backend_args: Optional[dict] = None) -> tuple[FluxVAE, FluxAEParams]:
    ae_params = FluxAEParams(
        resolution=256,
        in_channels=3,
        downsample=8,
        ch=128,
        out_ch=3,
        ch_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        z_channels=16,
        scale_factor=0.3611,
        shift_factor=0.1159,
    )

    # Loading the autoencoder
    ae = FluxVAE(ae_params)

    if local_path is not None and local_path != "":
        # We use safetensors.torch.load_file to load the checkpoint from Hugging Face.
        # For Cosmos checkpoints, they are stored as distributed checkpoints.
        # Handle S3 paths using easy_io
        if local_path.startswith("s3://"):
            byte_stream = easy_io.load(local_path, backend_args=backend_args, file_format="byte")
            sd = safetensors.torch.load(byte_stream)
        else:
            sd = safetensors.torch.load_file(local_path)
        missing, unexpected = ae.load_state_dict(sd, strict=False, assign=True)
        print_load_warning(missing, unexpected)
    return ae, ae_params


class FluxVAEInterface(VideoTokenizerInterface):
    """Flux VAE interface for image tokenization."""

    def __init__(
        self,
        bucket_name: str = "",
        object_store_credential_path_pretrained: str = "",
        vae_path: Optional[str] = None,
        chunk_duration: int = 1,
        spatial_compression_factor: int = 8,
        temporal_compression_factor: int = 1,
    ):
        super().__init__(object_store_credential_path_pretrained=object_store_credential_path_pretrained)

        # Load the Flux VAE model, passing backend_args for S3 support
        vae_path_full = f"s3://{bucket_name}/{vae_path}"
        self.model, self.params = load_ae(vae_path_full or "", backend_args=self.backend_args)
        self.model.eval()

        # Set device and dtype
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self._dtype = torch.bfloat16
        self.model = self.model.to(self._dtype)

        self.chunk_duration = chunk_duration

        # Flux VAE is image-only, so temporal compression is 1
        self.temporal_compression = 1
        self.spatial_compression = 8  # 256/32 = 8

    @property
    def dtype(self):
        return self._dtype

    def reset_dtype(self):
        """Reset the dtype of the model."""
        self.model = self.model.to(self._dtype)

    def encode(self, state: torch.Tensor) -> torch.Tensor:
        """Encode input tensor to latent space."""
        with torch.no_grad():
            # Ensure input is in the right format (B, C, H, W)
            if state.dim() == 5:  # Video input (B, C, T, H, W)
                # For video, we'll process frame by frame
                batch_size, channels, time, height, width = state.shape
                state = state.permute(0, 2, 1, 3, 4).contiguous().view(-1, channels, height, width)  # [B*T,C,H,W]
                latents = self.model.encode(state)  # [B*T,C,H//8,W//8]
                # Reshape back to video format
                latent_channels = latents.shape[1]
                latent_height = latents.shape[2]
                latent_width = latents.shape[3]
                latents = latents.view(
                    batch_size, time, latent_channels, latent_height, latent_width
                )  # [B,T,C,H//8,W//8]
                latents = latents.permute(0, 2, 1, 3, 4).contiguous()  # [B,C,T,H//8,W//8]
            else:  # Image input (B, C, H, W)
                latents = self.model.encode(state)  # [B,C,H//8,W//8]

            return latents

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent tensor to pixel space."""
        with torch.no_grad():
            if latent.dim() == 5:  # Video latent (B, C, T, H, W)
                batch_size, channels, time, height, width = latent.shape
                latent = latent.permute(0, 2, 1, 3, 4).contiguous().view(-1, channels, height, width)  # [B*T,C,H,W]
                decoded = self.model.decode(latent)  # [B*T,3,H*8,W*8]
                # Reshape back to video format
                pixel_channels = decoded.shape[1]
                pixel_height = decoded.shape[2]
                pixel_width = decoded.shape[3]
                decoded = decoded.view(batch_size, time, pixel_channels, pixel_height, pixel_width)  # [B,T,3,H*8,W*8]
                decoded = decoded.permute(0, 2, 1, 3, 4).contiguous()  # [B,3,T,H*8,W*8]
            else:  # Image latent (B, C, H, W)
                decoded = self.model.decode(latent)  # [B,3,H*8,W*8]

            return decoded

    def get_latent_num_frames(self, num_pixel_frames: int) -> int:
        """Get number of latent frames from pixel frames."""
        return num_pixel_frames  # Flux VAE doesn't compress temporally

    def get_pixel_num_frames(self, num_latent_frames: int) -> int:
        """Get number of pixel frames from latent frames."""
        return num_latent_frames  # Flux VAE doesn't compress temporally

    @property
    def spatial_compression_factor(self):
        """Spatial compression factor."""
        return self.spatial_compression

    @property
    def temporal_compression_factor(self):
        """Temporal compression factor."""
        return self.temporal_compression

    @property
    def spatial_resolution(self):
        """Spatial resolution."""
        return 256  # Flux VAE default resolution

    @property
    def pixel_chunk_duration(self):
        """Pixel chunk duration."""
        return self.chunk_duration

    @property
    def latent_chunk_duration(self):
        """Latent chunk duration."""
        return self.get_latent_num_frames(self.chunk_duration)

    @property
    def latent_ch(self):
        """Number of latent channels."""
        return 16  # From FluxAEParams.z_channels

    @property
    def name(self):
        """Name of the tokenizer."""
        return "flux_tokenizer"

    def count_param(self):
        """Count the number of parameters in the model."""
        return sum(p.numel() for p in self.model.parameters())
