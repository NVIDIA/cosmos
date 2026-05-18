# -----------------------------------------------------------------------------
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# -----------------------------------------------------------------------------

"""
Tests for UniAE S1 tokenizer (4x16x16).

Usage:
    # Basic encode/decode test with random data
    CUDA_VISIBLE_DEVICES=0 RUN_SKIPPED_TEST_LOCALLY=1 pytest -s cosmos/model/vfm/tokenizers/uniae/noncausal_4x16x16_test.py -k test_uniae_s1

    # Full reconstruction test with real video (saves uniae_recon.mp4)
    CUDA_VISIBLE_DEVICES=0 RUN_SKIPPED_TEST_LOCALLY=1 pytest -s cosmos/model/vfm/tokenizers/uniae/noncausal_4x16x16_test.py -k test_local_video

Note: On this machine, CUDA device 0 = RTX 6000 Ada (48GB), device 1 = T400 (2GB).
      Always use CUDA_VISIBLE_DEVICES=0 for the RTX 6000.
"""

import os

import numpy as np
import pytest
import torch

from cosmos.utils.easy_io import easy_io
from cosmos.utils.helper_test import RunIf
from configs.base.defaults.cluster import GCPIADGB200Config as CLUSTER_CONFIG
from configs.base.defaults.unittest import TOKENIZER_RECONSTRUCTION_VIDEO_PATH, UNITTEST_CONFIG
from cosmos.model.vfm.tokenizers.uniae.noncausal_4x16x16 import UniAEVAE
from cosmos.model.vfm.tokenizers.unittest_utils import (
    numpy2tensor,
    pad_video_batch,
    tensor2numpy,
    unpad_video_batch,
)

UNIAE_S1_PATH = (
    "s3://nv-00-10206-checkpoint/pretrained/tokenizers/video/cosmos/"
    "uniae4x16x16_c48_t8to24_64to512p_fps_all_encoder_noncausal_decoder_noncausal_nogan_best_s1.pt"
)


@pytest.mark.L0
@pytest.mark.skipif(os.getenv("RUN_SKIPPED_TEST_LOCALLY") != "1", reason="local_test_only")
def test_uniae_s1():
    """Basic encode/decode test with random data."""
    vae = UniAEVAE(
        vae_pth=UNIAE_S1_PATH,
        object_store_credential_path_pretrained=CLUSTER_CONFIG.object_store_credential_pretrained,
        device="cuda",
    )
    print(f"\n[UniAE S1] Model parameters: {vae.count_param() / 1e6:.2f}M")

    H, W = 256, 256
    for T in [4, 16, 32]:
        print(f"\n[UniAE S1] Testing with T={T} frames, H={H}, W={W}")
        video = torch.randn(1, 3, T, H, W, device="cuda")
        latents = vae.encode(video)
        print(f"  Input video shape: {video.shape} -> Latent shape: {latents.shape}")
        print(f"  Latent stats: mean={latents.mean():.4f}, std={latents.std():.4f}")
        video_recon = vae.decode(latents)
        print(f"  Reconstructed video shape: {video_recon.shape}")

        # Verify shapes
        expected_T_latent = T // 4
        expected_H_latent = H // 16
        expected_W_latent = W // 16
        assert latents.shape == (1, 48, expected_T_latent, expected_H_latent, expected_W_latent), (
            f"Expected latent shape (1, 48, {expected_T_latent}, {expected_H_latent}, {expected_W_latent}), "
            f"got {latents.shape}"
        )


@pytest.mark.L0
@RunIf(
    requires_file=[
        CLUSTER_CONFIG.object_store_credential_pretrained,
        UNITTEST_CONFIG.object_store_credential_data,
    ]
)
@pytest.mark.skipif(os.getenv("RUN_SKIPPED_TEST_LOCALLY") != "1", reason="local_test_only")
def test_local_video():
    """Full reconstruction test with a real video — saves output to logs/uniae_recon.mp4."""
    vae = UniAEVAE(
        vae_pth=UNIAE_S1_PATH,
        object_store_credential_path_pretrained=CLUSTER_CONFIG.object_store_credential_pretrained,
        device="cuda",
        dtype=torch.bfloat16,
    )

    # Load video as numpy array (T, H, W, C) in range [0, 255]
    video_in_numpy = easy_io.load(
        os.path.join(f"s3://{UNITTEST_CONFIG.object_store_bucket_data}", TOKENIZER_RECONSTRUCTION_VIDEO_PATH),
        backend_args={
            "backend": "s3",
            "s3_credential_path": UNITTEST_CONFIG.object_store_credential_data,
        },
    )[0][:32]  # Take 32 frames (divisible by 4)

    # Pad video to meet stride alignment requirements
    padded_video_batch, crop_region = pad_video_batch(
        video_in_numpy[np.newaxis, ...],  # Add batch dimension
        temporal_align=4,  # Temporal compression factor
        spatial_align=16,  # Spatial compression factor
        causal_mode=False,  # UniAE is non-causal
        only_pad_end=True,
    )

    # Convert to tensor format (B, C, T, H, W) in range [-1, 1]
    video_tensor = numpy2tensor(padded_video_batch)

    # Encode and decode
    print(f"\n[UniAE S1 Local Video] Input tensor shape: {video_tensor.shape}")
    latents = vae.encode(video_tensor)
    print(f"[UniAE S1 Local Video] Latent shape: {latents.shape}")
    print(f"[UniAE S1 Local Video] Latent statistics: mean={latents.mean():.4f}, std={latents.std():.4f}")
    video_recon = vae.decode(latents)
    print(f"[UniAE S1 Local Video] Reconstructed shape: {video_recon.shape}")

    # Convert back to numpy and unpad
    video_recon_numpy = tensor2numpy(video_recon)
    video_recon_unpadded = unpad_video_batch(video_recon_numpy, crop_region)

    # Compute PSNR
    gt = video_in_numpy[: video_recon_unpadded.shape[1]].astype(np.float32)
    recon = video_recon_unpadded[0].astype(np.float32)
    mse = np.mean((gt - recon) ** 2)
    psnr = 10 * np.log10(255**2 / max(mse, 1e-10))
    print(f"[UniAE S1 Local Video] PSNR: {psnr:.2f} dB")

    # Save reconstruction
    output_path = os.path.expanduser("logs/uniae_recon.mp4")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    easy_io.dump(video_recon_unpadded[0].astype("uint8"), output_path)
    print(f"[UniAE S1 Local Video] Saved reconstruction to: {output_path}")


"""
Usage:
    CUDA_VISIBLE_DEVICES=0 RUN_SKIPPED_TEST_LOCALLY=1 pytest -s cosmos/model/vfm/tokenizers/uniae/noncausal_4x16x16_test.py -k test_local_image
"""


@pytest.mark.L0
@RunIf(
    requires_file=[
        CLUSTER_CONFIG.object_store_credential_pretrained,
        UNITTEST_CONFIG.object_store_credential_data,
    ]
)
@pytest.mark.skipif(os.getenv("RUN_SKIPPED_TEST_LOCALLY") != "1", reason="local_test_only")
def test_local_image():
    """Image reconstruction test — repeats image to 4 frames for non-causal tokenizer."""
    from PIL import Image

    vae = UniAEVAE(
        vae_pth=UNIAE_S1_PATH,
        object_store_credential_path_pretrained=CLUSTER_CONFIG.object_store_credential_pretrained,
        device="cuda",
        dtype=torch.bfloat16,
    )

    # Load first frame of test video as image
    video_in_numpy = easy_io.load(
        os.path.join(f"s3://{UNITTEST_CONFIG.object_store_bucket_data}", TOKENIZER_RECONSTRUCTION_VIDEO_PATH),
        backend_args={
            "backend": "s3",
            "s3_credential_path": UNITTEST_CONFIG.object_store_credential_data,
        },
    )[0][0]  # First frame: (H, W, C) in [0, 255]

    H, W, C = video_in_numpy.shape
    print(f"\n[UniAE S1 Image] Original image shape: ({H}, {W}, {C})")

    # Pad spatial dimensions to be divisible by 16
    pad_h = (16 - H % 16) % 16
    pad_w = (16 - W % 16) % 16
    if pad_h > 0 or pad_w > 0:
        video_in_numpy = np.pad(video_in_numpy, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
    H_padded, W_padded = video_in_numpy.shape[:2]

    # Convert to tensor [-1, 1] as single image (encode handles repeat internally)
    image_tensor = torch.from_numpy(video_in_numpy).float().permute(2, 0, 1) / 127.5 - 1.0  # (C, H, W)
    image_batch = image_tensor.unsqueeze(0).cuda()  # (1, C, H, W)

    print(f"[UniAE S1 Image] Input tensor shape: {image_batch.shape}")

    # Encode and decode (encode handles repeat to 4 frames internally)
    latents = vae.encode(image_batch)
    print(f"[UniAE S1 Image] Latent shape: {latents.shape}")
    print(f"[UniAE S1 Image] Latent statistics: mean={latents.mean():.4f}, std={latents.std():.4f}")
    video_recon = vae.decode(latents)
    print(f"[UniAE S1 Image] Reconstructed shape: {video_recon.shape}")

    # Take the first frame as reconstructed image
    recon_image = video_recon[0, :, 0].clamp(-1, 1)  # (C, H, W)
    recon_numpy = ((recon_image.float().cpu().permute(1, 2, 0).numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)

    # Crop back to original size
    recon_numpy = recon_numpy[:H, :W]

    # Compute PSNR against original (unpadded)
    gt = video_in_numpy[:H, :W].astype(np.float32)
    recon_f = recon_numpy.astype(np.float32)
    mse = np.mean((gt - recon_f) ** 2)
    psnr = 10 * np.log10(255**2 / max(mse, 1e-10))
    print(f"[UniAE S1 Image] PSNR: {psnr:.2f} dB")

    # Save original and reconstruction side by side
    output_path = os.path.expanduser("logs/uniae_image_recon.png")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    orig_img = Image.fromarray(video_in_numpy[:H, :W])
    recon_img = Image.fromarray(recon_numpy)
    side_by_side = Image.new("RGB", (W * 2 + 10, H))
    side_by_side.paste(orig_img, (0, 0))
    side_by_side.paste(recon_img, (W + 10, 0))
    side_by_side.save(output_path)
    print(f"[UniAE S1 Image] Saved side-by-side to: {output_path}")
