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

import os

import pytest
import soundfile as sf
import torch
import torch.nn.functional as F
from torchcodec.decoders import AudioDecoder

from cosmos.utils.easy_io import easy_io
from cosmos.utils.helper_test import RunIf
from configs.base.defaults.cluster import GCPIADGB200Config as CLUSTER_CONFIG
from configs.base.defaults.tokenizer import PRETRAINED_TOKENIZER_AVAE_PTH
from configs.base.defaults.unittest import (
    AVAE_RECONSTRUCTION_AUDIO_PATH,
    UNITTEST_CONFIG,
)
from cosmos.model.vfm.tokenizers.audio.avae import AVAEInterface, AVAEModel

"""
Usage:
    export RUN_SKIPPED_TEST_LOCALLY=1
    pytest -s cosmos/model/vfm/tokenizers/audio/avae_test.py
"""

AVAE_CHECKPOINT_PATH_PDX = f"s3://{CLUSTER_CONFIG.object_store_bucket_pretrained}/{PRETRAINED_TOKENIZER_AVAE_PTH}"
CKPT_CREDENTIALS_PATH = CLUSTER_CONFIG.object_store_credential_pretrained


@pytest.mark.L0
@pytest.mark.skipif(os.getenv("RUN_SKIPPED_TEST_LOCALLY") != "1", reason="local_test_only")
def test_avae_model_basic():
    """Test AVAEModel basic encoding and decoding."""
    model = AVAEModel(
        vae_pth="",  # No checkpoint for basic test
        sample_rate=44100,
        audio_channels=2,
        io_channels=64,
        hop_size=2048,
        device="cuda",
    )

    # Count parameters
    param_count = model.count_param()
    print(f"AVAE Parameters: {param_count / 1e6:.2f}M")

    # Test encoding and decoding
    audio = torch.randn(1, 2, 88200).cuda()  # [B,C,T] — 2 seconds at 44.1kHz
    latents = model.encode(audio)
    print(f"Audio shape: {audio.shape} -> Latent shape: {latents.shape}")

    audio_recon = model.decode(latents)
    print(f"Reconstructed audio shape: {audio_recon.shape}")

    assert audio_recon.shape[1] == 2, f"Expected 2 audio channels, got {audio_recon.shape[1]}"
    assert audio_recon.min() >= -1.0 and audio_recon.max() <= 1.0, "Audio not in valid range [-1, 1]"


@pytest.mark.L0
@pytest.mark.skipif(os.getenv("RUN_SKIPPED_TEST_LOCALLY") != "1", reason="local_test_only")
def test_avae_interface():
    """Test AVAEInterface implementation."""
    tokenizer = AVAEInterface(
        bucket_name="",
        avae_path="",
        object_store_credential_path_pretrained=None,
        sample_rate=44100,
        audio_channels=2,
        io_channels=64,
        hop_size=2048,
    )

    # Test interface properties
    assert tokenizer.sample_rate == 44100
    assert tokenizer.audio_channels == 2
    assert tokenizer.latent_ch == 64
    assert tokenizer.temporal_compression_factor == 2048
    assert tokenizer.name == "avae_tokenizer"
    assert not tokenizer.is_causal

    print(f"Sample rate: {tokenizer.sample_rate} Hz")
    print(f"Audio channels: {tokenizer.audio_channels}")
    print(f"Latent channels: {tokenizer.latent_ch}")
    print(f"Temporal compression: {tokenizer.temporal_compression_factor}x")

    # Test conversion methods
    num_audio_samples = 88200
    num_latent_samples = tokenizer.get_latent_num_samples(num_audio_samples)
    assert num_latent_samples == num_audio_samples // 2048
    print(f"Audio samples {num_audio_samples} -> Latent samples {num_latent_samples}")

    reconstructed_samples = tokenizer.get_audio_num_samples(num_latent_samples)
    assert reconstructed_samples == num_latent_samples * 2048
    print(f"Latent samples {num_latent_samples} -> Audio samples {reconstructed_samples}")

    # Test encode/decode
    audio = torch.randn(1, 2, 44100).cuda()  # [B,C,T]
    latents = tokenizer.encode(audio)
    print(f"Encode: {audio.shape} -> {latents.shape}")

    audio_recon = tokenizer.decode(latents)
    print(f"Decode: {latents.shape} -> {audio_recon.shape}")

    # Test reset_dtype
    tokenizer.reset_dtype()
    assert tokenizer.dtype == torch.bfloat16

    # Test count_param
    param_count = tokenizer.count_param()
    print(f"Total parameters: {param_count / 1e6:.2f}M")


@pytest.mark.L0
@pytest.mark.skipif(os.getenv("RUN_SKIPPED_TEST_LOCALLY") != "1", reason="local_test_only")
def test_avae_batch_processing():
    """Test AVAE with different batch sizes."""
    model = AVAEModel(
        vae_pth="",
        sample_rate=44100,
        device="cuda",
    )

    for batch_size in [1, 2, 4]:
        audio = torch.randn(batch_size, 2, 88200).cuda()  # [B,C,T]
        latents = model.encode(audio)
        print(f"Batch {batch_size}: {audio.shape} -> {latents.shape}")

        audio_recon = model.decode(latents)
        print(f"Batch {batch_size} reconstructed: {audio_recon.shape}")

        assert audio_recon.shape[0] == batch_size


@pytest.mark.L0
@RunIf(
    requires_file=[
        UNITTEST_CONFIG.object_store_credential_data,
    ],
    requires_package="torchcodec",
)
@pytest.mark.skipif(os.getenv("RUN_SKIPPED_TEST_LOCALLY") != "1", reason="local_test_only")
def test_avae_with_s3_audio():
    """Test AVAE with an audio file from S3."""
    # Load audio from S3
    audio_path_s3 = os.path.join(f"s3://{UNITTEST_CONFIG.object_store_bucket_data}", AVAE_RECONSTRUCTION_AUDIO_PATH)
    print(f"\nLoading audio from {audio_path_s3}...")

    try:
        audio_bytes = easy_io.load(
            audio_path_s3,
            backend_args={
                "backend": "s3",
                "s3_credential_path": UNITTEST_CONFIG.object_store_credential_data,
            },
            file_format="byte",
        )

        # Decode audio from bytes with torchcodec (no torchaudio dependency).
        audio_decoder = AudioDecoder(audio_bytes)
        samples = audio_decoder.get_all_samples()
        audio_tensor = samples.data  # (C, N)
        sr = audio_decoder.metadata.sample_rate

        # Defensive shape normalization: torchcodec generally returns (C, N).
        if audio_tensor.ndim == 1:
            audio_tensor = audio_tensor.unsqueeze(0)

        # Resample with torch only if needed.
        # Input: [C, N] -> [1, C, N] for interpolate.
        if sr != 44100:
            target_len = int(round(audio_tensor.shape[-1] * 44100 / float(sr)))
            audio_tensor = audio_tensor.unsqueeze(0).to(torch.float32)  # [1,C,N]
            audio_tensor = F.interpolate(
                audio_tensor, size=target_len, mode="linear", align_corners=False
            )  # [1,C,N_resampled]
            audio_tensor = audio_tensor.squeeze(0)  # [C,N_resampled]
            sr = 44100

        print(f"Loaded audio: {audio_tensor.shape}, sr={sr}")

    except Exception as e:
        pytest.fail(f"Failed to load audio from S3: {e}")

    # Initialize model
    model = AVAEModel(
        vae_pth=AVAE_CHECKPOINT_PATH_PDX,
        object_store_credential_path_pretrained=CKPT_CREDENTIALS_PATH,
        sample_rate=44100,
        audio_channels=2,
        io_channels=64,
        hop_size=2048,
        device="cuda",
        dtype=torch.bfloat16,
    )

    # Normalize and prepare
    if audio_tensor.abs().max() > 1.0:
        audio_tensor = audio_tensor / audio_tensor.abs().max()

    # Convert to [B, C, T] and move to GPU.
    audio_tensor = audio_tensor.unsqueeze(0).cuda()  # [1,C,T]

    # Ensure stereo
    if audio_tensor.shape[1] == 1:
        audio_tensor = audio_tensor.repeat(1, 2, 1)
    elif audio_tensor.shape[1] > 2:
        audio_tensor = audio_tensor[:, :2, :]

    # Encode and decode
    latents = model.encode(audio_tensor)
    print(f"Latent shape: {latents.shape}")
    print(f"Latent stats - mean: {latents.mean().item():.4f}, std: {latents.std().item():.4f}")

    audio_recon = model.decode(latents)
    print(f"Reconstructed audio shape: {audio_recon.shape}")

    # Save reconstruction
    output_path = os.path.expanduser("logs/test_audio_recon.wav")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save with soundfile to avoid a torchaudio dependency.
    # soundfile expects (num_frames, num_channels) on CPU as a numpy array.
    audio_np = audio_recon[0].detach().cpu().float().transpose(0, 1).numpy()  # [T,C]
    sf.write(output_path, audio_np, samplerate=44100)
    print(f"Saved reconstruction to {output_path}")

    assert torch.isfinite(audio_recon).all(), "Reconstructed audio contains NaN/Inf"
