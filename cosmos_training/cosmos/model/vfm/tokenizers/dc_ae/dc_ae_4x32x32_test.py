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

import numpy as np
import pytest
import torch

from cosmos.utils.easy_io import easy_io
from cosmos.utils.helper_test import RunIf
from configs.base.defaults.cluster import GCPIADGB200Config as CLUSTER_CONFIG
from configs.base.defaults.tokenizer import PRETRAINED_TOKENIZER_DCAE_PTH
from configs.base.defaults.unittest import TOKENIZER_RECONSTRUCTION_VIDEO_PATH, UNITTEST_CONFIG
from cosmos.model.vfm.tokenizers.dc_ae.dc_ae_4x32x32 import DEFAULT_MODEL_NAME, DCAE4x32x32Interface
from cosmos.model.vfm.tokenizers.dc_ae.dc_ae_v import DCAEV, dc_ae_v_f32t4_encoder_causal_decoder_chunk_causal_4
from cosmos.model.vfm.tokenizers.unittest_utils import (
    numpy2tensor,
    pad_video_batch,
    tensor2numpy,
    unpad_video_batch,
)

"""
Usage:
    RUN_SKIPPED_TEST_LOCALLY=1 pytest -s cosmos/model/vfm/tokenizers/dc_ae/dc_ae_4x32x32_test.py
    RUN_SKIPPED_TEST_LOCALLY=1 pytest -s cosmos/model/vfm/tokenizers/dc_ae/dc_ae_4x32x32_test.py -k test_dc_ae_local_checkpoint
"""

VAE_PATH = PRETRAINED_TOKENIZER_DCAE_PTH
LOCAL_CHECKPOINT = os.path.expanduser(
    "~/work/imaginaire4/logs/cosmos_4x32x32_0211/checkpoints/"
    "dcae4x32x32_c64_t120_256p_fps_all_encoder_causal_decoder_chunk_causal_4_nogan_cosmos_pad_7_v0.1.pt"
)


def _make_dc_ae_from_s3():
    return DCAE4x32x32Interface(
        bucket_name=CLUSTER_CONFIG.object_store_bucket_pretrained,
        object_store_credential_path_pretrained=CLUSTER_CONFIG.object_store_credential_pretrained,
        vae_path=VAE_PATH,
    )


def _make_dc_ae_from_local(checkpoint_path=LOCAL_CHECKPOINT):
    """Load DCAE directly from a local checkpoint (no S3, no distributed sync)."""
    cfg = dc_ae_v_f32t4_encoder_causal_decoder_chunk_causal_4(DEFAULT_MODEL_NAME, pretrained_path=None)
    with torch.device("meta"):
        model = DCAEV(cfg)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model_state_dict"]
    model.load_state_dict(state_dict, assign=True)
    model.eval().requires_grad_(False).to(dtype=torch.bfloat16, device="cuda")
    return model, cfg


@pytest.mark.L0
@pytest.mark.skipif(not os.path.exists(LOCAL_CHECKPOINT), reason=f"local checkpoint not found: {LOCAL_CHECKPOINT}")
@pytest.mark.skipif(os.getenv("RUN_SKIPPED_TEST_LOCALLY") != "1", reason="local_test_only")
def test_dc_ae_local_checkpoint():
    """Test loading from local checkpoint and encode/decode with various temporal lengths."""
    model, cfg = _make_dc_ae_from_local()
    print(f"\n[DCAE Local] Loaded from: {LOCAL_CHECKPOINT}")
    print(
        f"  latent_channels={cfg.latent_channels}, num_pad_frames={cfg.num_pad_frames}, scaling_factor={cfg.scaling_factor}"
    )

    H, W = 512, 512
    for T in [1, 25, 81]:
        print(f"\n[DCAE Local] Testing with T={T} frames")
        video = torch.randn(1, 3, T, H, W, device="cuda", dtype=torch.bfloat16)
        latents = model.encode(video)
        expected_latent_t = (T + cfg.num_pad_frames) // 4
        print(f"  Input: {video.shape} -> Latent: {latents.shape} (expected latent_t={expected_latent_t})")
        assert latents.shape == (1, cfg.latent_channels, expected_latent_t, H // 32, W // 32)

        video_recon = model.decode(latents)
        print(f"  Reconstructed: {video_recon.shape}")
        assert video_recon.shape == (1, 3, T, H, W)


@pytest.mark.L0
@pytest.mark.skipif(not os.path.exists(LOCAL_CHECKPOINT), reason=f"local checkpoint not found: {LOCAL_CHECKPOINT}")
@pytest.mark.skipif(os.getenv("RUN_SKIPPED_TEST_LOCALLY") != "1", reason="local_test_only")
def test_dc_ae_local_checkpoint_keys():
    """Verify checkpoint structure and that all keys match the model."""
    checkpoint = torch.load(LOCAL_CHECKPOINT, map_location="cpu", weights_only=False)

    # Verify checkpoint structure
    assert "model_state_dict" in checkpoint, f"Expected 'model_state_dict', got keys: {list(checkpoint.keys())}"
    has_ema = "ema_model_state_dict" in checkpoint
    print(f"\n[DCAE Keys] Checkpoint keys: {list(checkpoint.keys())}")
    print(f"  Has EMA: {has_ema}")
    print(f"  model_state_dict: {len(checkpoint['model_state_dict'])} parameters")

    # Verify keys match model
    cfg = dc_ae_v_f32t4_encoder_causal_decoder_chunk_causal_4(DEFAULT_MODEL_NAME, pretrained_path=None)
    with torch.device("meta"):
        model = DCAEV(cfg)

    model_keys = set(model.state_dict().keys())
    ckpt_keys = set(checkpoint["model_state_dict"].keys())
    missing = model_keys - ckpt_keys
    extra = ckpt_keys - model_keys

    assert not missing, f"Missing keys in checkpoint: {missing}"
    assert not extra, f"Extra keys in checkpoint: {extra}"
    print(f"  All {len(model_keys)} keys match.")


@pytest.mark.L0
@RunIf(requires_file=[CLUSTER_CONFIG.object_store_credential_pretrained])
@pytest.mark.skipif(os.getenv("RUN_SKIPPED_TEST_LOCALLY") != "1", reason="local_test_only")
def test_dc_ae_encode_decode():
    """Test encode and decode with random input at various temporal lengths (S3 loading)."""
    model = _make_dc_ae_from_s3()
    H, W = 512, 512
    # num_pad_frames=7, temporal_compression=4, so valid T values:
    # T=1 -> pad to 1+7=8 -> latent_t=8//4=2
    # T=25 -> pad to 25+7=32 -> latent_t=32//4=8
    # T=81 -> pad to 81+7=88 -> latent_t=88//4=22
    for T in [1, 25, 81]:
        print(f"\n[DCAE] Testing with T={T} frames")
        video = torch.randn(1, 3, T, H, W, device="cuda", dtype=torch.bfloat16)
        latents = model.encode(video)
        expected_latent_t = model.get_latent_num_frames(T)
        print(f"  Input: {video.shape} -> Latent: {latents.shape} (expected latent_t={expected_latent_t})")
        assert latents.shape[0] == 1
        assert latents.shape[1] == model.latent_ch  # 64
        assert latents.shape[2] == expected_latent_t
        assert latents.shape[3] == H // model.spatial_compression_factor
        assert latents.shape[4] == W // model.spatial_compression_factor

        video_recon = model.decode(latents)
        print(f"  Reconstructed: {video_recon.shape}")
        assert video_recon.shape == (1, 3, T, H, W)


@pytest.mark.L0
@RunIf(requires_file=[CLUSTER_CONFIG.object_store_credential_pretrained])
@pytest.mark.skipif(os.getenv("RUN_SKIPPED_TEST_LOCALLY") != "1", reason="local_test_only")
def test_dc_ae_properties():
    """Test that interface properties return expected values."""
    model = _make_dc_ae_from_s3()
    assert model.spatial_compression_factor == 32
    assert model.temporal_compression_factor == 4
    assert model.latent_ch == 64
    assert model.spatial_resolution == 512
    assert model.pixel_chunk_duration == 81
    assert model.name == "dc_ae_4x32x32_tokenizer"

    # Test frame count conversions
    # With num_pad_frames=7: latent_t = (T + 7) // 4
    assert model.get_latent_num_frames(1) == 2  # (1+7)//4 = 2
    assert model.get_latent_num_frames(81) == 22  # (81+7)//4 = 22
    # Inverse: pixel_t = latent_t * 4 - 7
    assert model.get_pixel_num_frames(2) == 1  # 2*4-7 = 1
    assert model.get_pixel_num_frames(22) == 81  # 22*4-7 = 81
    assert model.latent_chunk_duration == model.get_latent_num_frames(81)
    print(f"\n[DCAE Properties] All properties verified.")


"""
RUN_SKIPPED_TEST_LOCALLY=1 pytest -s cosmos/model/vfm/tokenizers/dc_ae/dc_ae_4x32x32_test.py -k test_dc_ae_local_video
"""


@pytest.mark.L0
@RunIf(
    requires_file=[
        CLUSTER_CONFIG.object_store_credential_pretrained,
        UNITTEST_CONFIG.object_store_credential_data,
    ]
)
@pytest.mark.skipif(os.getenv("RUN_SKIPPED_TEST_LOCALLY") != "1", reason="local_test_only")
def test_dc_ae_local_video():
    """Test encode/decode with a real video from S3."""
    model = _make_dc_ae_from_s3()

    # Load video as numpy array (T, H, W, C) in range [0, 255]
    video_in_numpy = easy_io.load(
        os.path.join(f"s3://{UNITTEST_CONFIG.object_store_bucket_data}", TOKENIZER_RECONSTRUCTION_VIDEO_PATH),
        backend_args={
            "backend": "s3",
            "s3_credential_path": UNITTEST_CONFIG.object_store_credential_data,
        },
    )[0][:101]

    # Pad video to meet stride alignment requirements
    padded_video_batch, crop_region = pad_video_batch(
        video_in_numpy[np.newaxis, ...],  # Add batch dimension
        temporal_align=4,  # Temporal compression factor
        spatial_align=32,  # Spatial compression factor for dc_ae
        causal_mode=True,
        only_pad_end=True,
    )

    # Convert to tensor format (B, C, T, H, W) in range [-1, 1]
    video_tensor = numpy2tensor(padded_video_batch)

    # Encode and decode
    print(f"\n[DCAE Local Video] Input tensor shape: {video_tensor.shape}")
    latents = model.encode(video_tensor)
    print(f"[DCAE Local Video] Latent shape: {latents.shape}")
    print(f"[DCAE Local Video] Latent statistics: mean={latents.mean():.4f}, std={latents.std():.4f}")
    video_recon = model.decode(latents)
    print(f"[DCAE Local Video] Reconstructed shape: {video_recon.shape}")

    # Convert back to numpy and unpad
    video_recon_numpy = tensor2numpy(video_recon)
    video_recon_unpadded = unpad_video_batch(video_recon_numpy, crop_region)

    # Save reconstruction
    output_path = os.path.expanduser("logs/dc_ae_recon.mp4")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    easy_io.dump(video_recon_unpadded[0].astype("uint8"), output_path)
    print(f"[DCAE Local Video] Saved reconstruction to: {output_path}")
