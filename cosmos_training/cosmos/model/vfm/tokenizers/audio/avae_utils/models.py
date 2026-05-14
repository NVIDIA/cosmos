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
# Adapted from https://github.com/jik876/hifi-gan under the MIT license.

"""AVAE Models.

This file contains only the models needed for the spec_convnext encoder +
oobleck decoder + vae configuration.
"""

import math
from functools import partial
from typing import Any, Callable

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.utils import remove_weight_norm
from torch.nn.utils.parametrize import remove_parametrizations

from .env import AttrDict
from .modules import ConvNeXtBlock, OobleckDecoderBlock, WNConv1d, get_activation

# for causal models we use encodec modules
from .modules_encodec import SConv1d


def load_generator(model_type: str, h: AttrDict, device: torch.device | str) -> nn.Module:
    """
    Load generator model based on model_type.

    Cleaned version only supports 'autoencoder_v2' type.
    """
    print(f"model_type is {model_type}")

    if model_type in ["autoencoder_v2"]:
        generator = LatentAutoEncoderV2(h).to(device)
        print("Autoencoder params: {}".format(sum(p.numel() for p in generator.parameters())))
        print("Encoder params: {}".format(sum(p.numel() for p in generator.encoder.parameters())))
        if generator.decoder is not None:
            print("Decoder params: {}".format(sum(p.numel() for p in generator.decoder.parameters())))
    else:
        raise NotImplementedError(
            f"Model type '{model_type}' not supported in cleaned AVAE. Only 'autoencoder_v2' is supported."
        )

    return generator


class TrimPadding(nn.Module):
    """
    Used for causal convolution support of a conv layer wrapped with nn.Sequential
    """

    def __init__(self: "TrimPadding", padding: int) -> None:
        super().__init__()
        self.padding = padding

    def forward(self: "TrimPadding", x: torch.Tensor) -> torch.Tensor:
        return x[:, :, : -self.padding]  # [B,C,T-padding]


class SpectrogramConvNeXtEncoder(nn.Module):
    """
    Spectrogram Encoder with ConvNeXtBlocks

    This encoder processes input waveforms by converting them into spectrograms
    (magnitude and phase concatenated along the channel dimension) and encodes them
    using a sequence of ConvNeXtBlocks and downsampling layers.

    Args (mapped from h):
        in_channels (int): Number of input audio channels (1 for mono, 2 for stereo).
        channels (int): Base number of channels for the encoder.
        latent_dim (int): Dimensionality of the final latent representation.
        c_mults (List[int]): Channel multipliers at each depth of the encoder.
        strides (List[int]): Downsampling strides for each depth.
        num_blocks (int): Number of ConvNeXtBlocks to stack per depth.
        identity_init (bool): Whether to initialize the 1x1 convs in residual paths as zeros.
        n_fft (int): Number of FFT points for spectrogram computation.
        hop_length (int): Hop length for the STFT.
        use_snake (bool): Whether to use Snake activation in ConvNeXtBlocks.
        causal (bool): If True, uses causal convolutions.
        padding_mode (str): Padding mode for convolutions (default: 'zeros').

    Inputs:
        x (torch.Tensor): Input waveform tensor of shape `[batch, in_channels, time]`.

    Outputs:
        torch.Tensor: Encoded representation of shape `[batch, time_out, latent_dim]`.

    Forward Pass:
        - Converts waveform input into spectrograms (concatenates magnitude and phase).
        - Processes the spectrogram through stacked ConvNeXtBlocks and downsampling layers.
        - Outputs the final latent representation of specified dimensionality.

    Example:
        encoder = SpectrogramConvNeXtEncoder(
            in_channels=2, channels=256, latent_dim=128, c_mults=[1, 2, 4], strides=[4, 4, 8]
        )
        waveform = torch.randn(8, 2, 65536)  # [batch, channels, time]
        encoded = encoder(waveform)  # Output: [8, time_out, 128]

    NOTE: output is in [B, T, C] to be consistent with other encoders
    """

    def __init__(self: "SpectrogramConvNeXtEncoder", h: AttrDict, **kwargs: Any) -> None:
        super().__init__()

        # Handle any deprecated or unused kwargs
        for key in kwargs:
            msg = f"[WARNING (SpectrogramConvNeXtEncoder)]: '{key}' is an unsupported argument and has been ignored."
            print(msg)

        self.in_channels = h.input_channels
        if getattr(h, "stereo", False):
            self.in_channels *= 2

        # if "enc_latent_dim" is found in v2 config, set it as latent_dim
        if hasattr(h, "enc_latent_dim"):
            self.latent_dim = h.enc_latent_dim
        else:
            # if not found, fallback to v1 logic
            self.latent_dim = h.vocoder_input_dim
            if h.model_type == "vae":
                self.latent_dim *= 2

        self.channels = h.enc_dim

        self.c_mults = h.enc_c_mults
        self.strides = h.enc_strides
        self.num_blocks = h.enc_num_blocks
        self.identity_init = h.enc_identity_init
        self.causal = h.causal
        self.padding_mode = h.padding_mode

        self.use_snake = h.enc_use_snake

        # Basic checks
        assert len(self.c_mults) == len(self.strides), (
            f"The length of c_mults and strides must match. Got {len(self.c_mults)} vs {len(self.strides)}."
        )

        # Spectrogram function
        self.n_fft = h.enc_n_fft
        self.hop_length = h.enc_hop_length
        self.spectrogram_fn = partial(
            self.spectrogram,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window_fn=torch.hann_window,
        )

        # ---------------------------------------------------------------------
        # 1) Initial projection (similar to the first_conv in OobleckEncoder),
        #    but here we typically use a 1x1 conv for a "spectrogram style" input.
        # ---------------------------------------------------------------------
        layers = []
        layers.append(
            WNConv1d((self.n_fft + 2) * self.in_channels, self.c_mults[0] * self.channels, kernel_size=1, bias=False)
        )

        # ---------------------------------------------------------------------
        # 2) Stages: For each i in range(len(c_mults)):
        #       - Stack num_blocks of ConvNeXtBlock
        #       - Downsample via stride convolution
        # ---------------------------------------------------------------------
        for i in range(len(self.c_mults)):
            dim_in = self.c_mults[i] * self.channels
            # Determine output dimension for the block
            if i < len(self.c_mults) - 1:  # If not the last block
                dim_out = self.c_mults[i + 1] * self.channels
            else:  # For the last block, dim_out is c_mults[-1] * channels
                dim_out = self.c_mults[-1] * self.channels
            ds_rate = self.strides[i]

            # (a) Repeated ConvNeXtBlocks
            for _ in range(self.num_blocks):
                layers.append(
                    ConvNeXtBlock(
                        dim=dim_in,
                        intermediate_dim=dim_in * 4,
                        identity_init=self.identity_init,
                        use_snake=self.use_snake,
                        causal=self.causal,
                    )
                )

            # (b) Downsampling convolution
            layers.append(self._create_downsample_layer(dim_in, dim_out, ds_rate, self.causal, self.padding_mode))

        # ---------------------------------------------------------------------
        # 3) Final projection from the last channel dimension to latent_dim.
        # ---------------------------------------------------------------------
        layers.append(WNConv1d(self.c_mults[-1] * self.channels, self.latent_dim, kernel_size=1, bias=False))

        self.layers = nn.Sequential(*layers)

    def spectrogram(
        self: "SpectrogramConvNeXtEncoder",
        wav: Tensor,
        n_fft: int,
        hop_length: int,
        win_length: int,
        window_fn: Callable[[int], torch.Tensor] = torch.hann_window,
    ) -> Tensor:
        """
        wav: [B_ch,T_audio] where B_ch = batch * channels (channel folded into batch)
        returns: [B_ch,n_fft//2+1,T_frames] complex
        """
        pad_size_l = (n_fft - hop_length) // 2
        pad_size_r = (n_fft - hop_length) - pad_size_l
        with torch.autocast(device_type=wav.device.type, enabled=False):
            wav = F.pad(wav, (pad_size_l, pad_size_r)).float()  # [B_ch,T_audio+pad]
            spec = torch.stft(
                wav,
                n_fft,
                hop_length=hop_length,
                win_length=win_length,
                window=window_fn(win_length).to(wav),
                center=False,
                normalized=False,
                onesided=True,
                return_complex=True,
            )  # [B_ch,n_fft//2+1,T_frames]
        return spec  # [B_ch,n_fft//2+1,T_frames]

    def _create_downsample_layer(
        self: "SpectrogramConvNeXtEncoder",
        in_channels: int,
        out_channels: int,
        stride: int,
        causal: bool,
        padding_mode: str,
    ) -> nn.Module:
        if (
            causal
        ):  # use EnCodec's SConv1d for convenience without reinventing the wheels. padding_mode is reflect by default
            downsample_layer = SConv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=2 * stride,
                stride=stride,
                causal=True,
                norm="weight_norm",
            )
        else:  # original non-causal implmentation
            downsample_layer = WNConv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
                padding_mode=padding_mode,
            )
        return downsample_layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B,C,T_audio] waveform (mono: C=1, stereo: C=2)

        Returns:
            [B,T_latent,latent_dim]
        """

        # Handle stereo input by merging channel dim into batch dim
        batch, channels, length = x.shape
        if channels > 1:  # Stereo case
            x = x.reshape(batch * channels, 1, length)  # [B*C,1,T_audio] (channel folded into batch)

        # Compute the spectrogram
        with torch.autocast(device_type=x.device.type, enabled=False):
            spec = self.spectrogram_fn(x.float().squeeze(1))  # [B*C,n_fft//2+1,T_frames] complex
            mag, ph = torch.view_as_real(spec).chunk(2, dim=-1)  # each [B*C,n_fft//2+1,T_frames,1]
            spectrogram = torch.cat([mag, ph], dim=1).squeeze(-1)  # [B*C,n_fft+2,T_frames]

        # Cast spectrogram back to original dtype
        spectrogram = spectrogram.to(x.dtype)  # [B*C,n_fft+2,T_frames]

        # Restore stereo structure if needed
        if channels > 1:  # Stereo case
            freq = spectrogram.shape[1]  # Get the frequency dimension
            spectrogram = spectrogram.reshape(
                batch, channels * freq, *spectrogram.shape[2:]
            )  # [B,(n_fft+2)*C,T_frames]

        # forward pass the encoder
        output = self.layers(spectrogram)  # [B,latent_dim,T_latent]

        return output.transpose(1, 2)  # [B,T_latent,latent_dim]

    def remove_weight_norm(self: "SpectrogramConvNeXtEncoder") -> None:
        print("INFO: Removing all weight norm from SpectrogramConvNeXtEncoder")
        for module in self.modules():
            if hasattr(module, "parametrizations"):  # for new WN implementation using parameterizations
                try:
                    remove_parametrizations(module, "weight")
                except ValueError:
                    print(
                        f"[WARNING] No weight norm found in {module} with parameterizations. You can ignore this if you know that this module does not apply weight norm."
                    )
            elif hasattr(module, "weight"):
                try:
                    remove_weight_norm(module)
                except ValueError:
                    pass


class OobleckDecoder(nn.Module):
    """
    Oobleck Decoder for audio synthesis.

    Decodes latent representations into audio waveforms using
    upsampling blocks with optional Snake activation and anti-aliasing.
    """

    def __init__(
        self: "OobleckDecoder",
        h: AttrDict,
    ) -> None:
        super().__init__()

        self.h = h

        latent_dim = self.h.vocoder_input_dim

        out_channels = self.h.input_channels
        if getattr(h, "stereo", False):
            out_channels *= 2

        channels = self.h.dec_dim
        c_mults = self.h.dec_c_mults
        strides = self.h.dec_strides
        use_snake = self.h.dec_use_snake
        use_nearest_upsample = self.h.dec_use_nearest_upsample
        antialias_activation = self.h.dec_anti_aliasing
        causal = self.h.causal
        final_tanh = self.h.dec_use_tanh_at_final
        padding_mode = self.h.padding_mode

        c_mults = [1, *c_mults]

        self.depth = len(c_mults)

        # Padding for the first convolution layer
        self.first_padding = 6 if causal else 3
        first_conv = WNConv1d(
            in_channels=latent_dim,
            out_channels=c_mults[-1] * channels,
            kernel_size=7,
            padding=self.first_padding,
            padding_mode=padding_mode,
        )

        if causal:
            first_conv = nn.Sequential(first_conv, TrimPadding(self.first_padding))

        layers = [first_conv]

        for i in range(self.depth - 1, 0, -1):
            layers += [
                OobleckDecoderBlock(
                    in_channels=c_mults[i] * channels,
                    out_channels=c_mults[i - 1] * channels,
                    stride=strides[i - 1],
                    use_snake=use_snake,
                    antialias_activation=antialias_activation,
                    use_nearest_upsample=use_nearest_upsample,
                    causal=causal,
                    padding_mode=padding_mode,
                )
            ]

        # Padding for the final convolution layer
        self.final_padding = 6 if causal else 3
        final_conv = WNConv1d(
            in_channels=c_mults[0] * channels,
            out_channels=out_channels,
            kernel_size=7,
            padding=self.final_padding,
            padding_mode=padding_mode,
            bias=False,
        )

        if causal:
            final_conv = nn.Sequential(final_conv, TrimPadding(self.final_padding))

        layers += [
            get_activation(
                "snake" if use_snake else "elu", antialias=antialias_activation, channels=c_mults[0] * channels
            ),
            final_conv,
            nn.Tanh() if final_tanh else nn.Identity(),
        ]

        self.layers = nn.Sequential(*layers)

    def forward(self: "OobleckDecoder", x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B,latent_dim,T_latent]

        Returns:
            [B,C,T_audio]
        """
        x = self.layers(x)  # [B,C,T_audio]
        return x  # [B,C,T_audio]

    def remove_weight_norm(self: "OobleckDecoder") -> None:
        print("INFO: Removing all weight norm from OobleckDecoder")
        for module in self.modules():
            if hasattr(module, "parametrizations"):  # for new WN implementation using parameterizations
                try:
                    remove_parametrizations(module, "weight")
                except ValueError:
                    msg = (
                        f"[WARNING] No weight norm found in {module} with parameterizations. "
                        "You can ignore this if you know that this module does not apply weight norm."
                    )
                    print(msg)
            elif hasattr(module, "weight"):
                try:
                    remove_weight_norm(module)
                except ValueError:
                    pass


class LatentAutoEncoderV2(nn.Module):
    """
    A Latent AutoEncoder class with cleaner implementation to generalize using bottleneck.py

    Attributes:
        h: Configuration object containing model hyperparameters.
        encoder (nn.Module): The encoder module based on configuration.
        bottleneck (Bottleneck): Bottleneck module from bottleneck.py.
        decoder (nn.Module): The decoder module based on configuration.
    """

    def __init__(self: "LatentAutoEncoderV2", h: AttrDict) -> None:
        super().__init__()
        self.h = h

        # Set up basic model properties
        self.stereo = getattr(self.h, "stereo", False)

        # Determine input type
        self.input_type = None
        if getattr(self.h, "use_wav_as_input", False):
            print("INFO: Encoder's input feature is waveform")
            self.input_type = "waveform"
            self.h.input_channels = 1
        elif getattr(self.h, "use_linear_spec_as_input", False):
            print("INFO: Encoder's input feature is linear")
            self.input_type = "linear"
            self.h.input_channels = self.h.num_linears
        elif getattr(self.h, "use_discrete_code_as_input", False):
            print("INFO: Encoder's input feature is discrete_code")
            self.input_type = "discrete_code"
            self.h.input_channels = 1
        else:
            print("INFO: Encoder's input feature is mel")
            self.input_type = "mel"
            self.h.input_channels = self.h.num_mels

        # hop_size defines the down/up sampling factor of the autoencoder
        self.hop_size = self.h.hop_size

        # Initialize encoder
        self.enc_type = getattr(self.h, "enc_type", "convnext")
        print(f"INFO: Using {self.enc_type} as encoder")

        # Define encoder (only spec_convnext supported in cleaned version)
        if self.enc_type == "spec_convnext":
            self.encoder = SpectrogramConvNeXtEncoder(self.h)
        else:
            raise NotImplementedError(
                f"Encoder type '{self.enc_type}' not supported in cleaned AVAE. Only 'spec_convnext' is supported."
            )

        # Initialize encoder projector (Identity for spec_convnext)
        self.encoder_proj = nn.Identity()

        # Initialize bottleneck from config
        from .bottlenecks import create_bottleneck_from_config

        if hasattr(self.h, "bottleneck"):
            self.bottleneck = create_bottleneck_from_config(self.h.bottleneck)
            print(f"INFO: Created bottleneck of type {self.h.bottleneck['type']}")
        else:
            raise ValueError("Bottleneck configuration must be specified")

        # Check for encoder-only mode
        self.encoder_only = getattr(self.h, "encoder_only", False)

        if not self.encoder_only:
            # Initialize decoder
            self.dec_type = getattr(self.h, "dec_type", "oobleck")
            print(f"INFO: Using {self.dec_type} as decoder")
            if self.dec_type == "oobleck":
                self.decoder = OobleckDecoder(self.h)
            else:
                raise NotImplementedError(
                    f"Decoder type '{self.dec_type}' not supported in cleaned AVAE. Only 'oobleck' is supported."
                )
        else:
            # Skip decoder initialization
            self.decoder = None
            print("INFO: Running in encoder-only mode, decoder is set to None")

        # Whether to freeze encoder
        self.freeze_encoder = getattr(self.h, "freeze_encoder", False)
        if self.freeze_encoder:
            print("WARNING: freeze_encoder set to true. The encoder will not be updated during training!")
            for param in self.encoder.parameters():
                param.requires_grad = False

    def calculate_latent_lengths(self: "LatentAutoEncoderV2", audio_lengths: torch.Tensor) -> torch.Tensor:
        """
        Calculates the latent lengths given the original audio lengths.

        Args:
            audio_lengths (torch.Tensor): A tensor of shape [B] containing the lengths of the original audio samples.

        Returns:
            torch.Tensor: A tensor of shape [B] containing the corresponding latent lengths.
        """
        if self.input_type == "waveform":
            # The latent length is the audio length divided by the hop_size
            latent_lengths = torch.ceil(audio_lengths.float() / self.hop_size).long()  # [B]
        else:
            # The latent length is same as audio_lengths
            latent_lengths = audio_lengths  # [B]

        return latent_lengths

    def forward(self: "LatentAutoEncoderV2", x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor to the model with shape [B,C,T_audio].

        Returns:
            dict[str, torch.Tensor]: Dictionary of output tensors including:
                - encoder_out: Raw encoder output
                - latent: Bottleneck latent representation
                - decoder_out: Decoded output (if decoder exists)
                - Additional outputs specific to the bottleneck type
        """
        return_dict = {}

        # Encoder
        encoder_out = self.encoder(x)  # [B,T_latent,enc_latent_dim]
        encoder_out_proj = self.encoder_proj(encoder_out)  # [B,T_latent,enc_latent_dim]

        # Apply bottleneck after reshaping to [B, C, T] again
        latent, bottleneck_enc_info = self.bottleneck.encode(
            encoder_out_proj.transpose(1, 2),
            return_info=True,  # transpose: [B,enc_latent_dim,T_latent]
        )  # [B,C,T_latent]

        # Update return dictionary
        return_dict.update(
            {"encoder_out": encoder_out.transpose(1, 2), "latent": latent}  # encoder_out: [B,enc_latent_dim,T_latent]
        )
        # Add bottleneck-specific info to return dict
        for k, v in bottleneck_enc_info.items():
            return_dict[k] = v

        # Decode (if decoder exists)
        if self.decoder is not None:
            # Apply bottleneck decode
            decoded_latent, bottleneck_dec_info = self.bottleneck.decode(latent, return_info=True)  # [B,C,T_latent]
            # Apply decoder
            decoder_out = self.decoder(decoded_latent)  # [B,C,T_audio]

            # Update return dictionary
            return_dict["decoder_out"] = decoder_out  # [B,C,T_audio]
            # Add bottleneck-specific info to return dict
            for k, v in bottleneck_dec_info.items():
                return_dict[k] = v

        return return_dict

    def encode(self: "LatentAutoEncoderV2", x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Encodes input x into latent representation using encoder and bottleneck.

        Args:
            x (torch.Tensor): Input tensor with shape [B, C, T].

        Returns:
            dict[str, torch.Tensor]: Dictionary containing:
                - latent: Bottleneck latent representation
                - Additional outputs specific to the bottleneck type
        """
        encoder_out = self.encoder(x)  # [B,T_latent,enc_latent_dim]
        encoder_out_proj = self.encoder_proj(encoder_out)  # [B,T_latent,enc_latent_dim]
        latent, bottleneck_info = self.bottleneck.encode(
            encoder_out_proj.transpose(1, 2),
            return_info=True,  # transpose: [B,enc_latent_dim,T_latent]
        )  # [B,C,T_latent]

        return_dict = {"latent": latent}  # latent: [B,C,T_latent]
        # Add bottleneck-specific info to return dict
        for k, v in bottleneck_info.items():
            return_dict[k] = v

        return return_dict

    def decode(self: "LatentAutoEncoderV2", latent: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Decodes continuous latent representation into output using bottleneck and decoder.

        Args:
            latent (torch.Tensor): continuous latent representation with shape [B, C, T].

        Returns:
            dict[str, torch.Tensor]: Dictionary containing:
                - decoder_out: The output from the decoder
                - Additional outputs from the bottleneck decode process
        """
        # Apply bottleneck decode
        decoded_latent, bottleneck_info = self.bottleneck.decode(latent, return_info=True)  # [B,C,T_latent]

        # Apply decoder
        decoder_out = self.decoder(decoded_latent)  # [B,C,T_audio]

        return_dict = {"decoder_out": decoder_out}  # decoder_out: [B,C,T_audio]
        # Add bottleneck-specific info to return dict
        for k, v in bottleneck_info.items():
            return_dict[k] = v

        return return_dict

    def remove_weight_norm(self: "LatentAutoEncoderV2") -> None:
        """Remove weight normalization from all components."""
        self.encoder.remove_weight_norm()
        if self.decoder is not None:
            self.decoder.remove_weight_norm()
