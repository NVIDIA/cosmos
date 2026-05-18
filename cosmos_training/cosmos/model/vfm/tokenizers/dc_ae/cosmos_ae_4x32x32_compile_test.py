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

import math
import os
import time

import numpy as np
import pytest
import torch
import torch.distributed.algorithms._checkpoint.checkpoint_wrapper
import torch.nn.functional as F

from cosmos.utils.easy_io import easy_io
from cosmos.utils.helper_test import RunIf
from configs.base.defaults.cluster import GCPIADGB200Config as CLUSTER_CONFIG
from configs.base.defaults.tokenizer import PRETRAINED_TOKENIZER_DCAE_PTH
from cosmos.model.vfm.tokenizers.dc_ae.dc_ae_4x32x32 import DCAE4x32x32Interface
from cosmos.model.vfm.tokenizers.unittest_utils import (
    numpy2tensor,
    pad_video_batch,
    tensor2numpy,
    unpad_video_batch,
)

"""
Usage:
    RUN_SKIPPED_TEST_LOCALLY=1 pytest -s cosmos/model/vfm/tokenizers/dc_ae/cosmos_ae_4x32x32_compile_test.py

"""

VAE_PATH = PRETRAINED_TOKENIZER_DCAE_PTH


def _make_cosmos_ae_from_s3(encoder_width_list):
    return DCAE4x32x32Interface(
        bucket_name=CLUSTER_CONFIG.object_store_bucket_pretrained,
        object_store_credential_path_pretrained=CLUSTER_CONFIG.object_store_credential_pretrained,
        vae_path=VAE_PATH,
        compilable=False,
    )


def _make_cosmos_ae_from_s3_compiled(encoder_width_list):
    return DCAE4x32x32Interface(
        bucket_name=CLUSTER_CONFIG.object_store_bucket_pretrained,
        object_store_credential_path_pretrained=CLUSTER_CONFIG.object_store_credential_pretrained,
        vae_path=VAE_PATH,
        compilable=True,
    )


def benchmark(func, x, num_warmups, num_runs):
    times = []
    for _ in range(num_warmups):
        _ = func(x)
        torch.cuda.synchronize()

    # torch.cuda.cudart().cudaProfilerStart()
    for _ in range(num_runs):
        time_start = time.perf_counter_ns()
        # torch.cuda.nvtx.range_push("encode")
        _ = func(x)
        torch.cuda.synchronize()
        # torch.cuda.nvtx.range_pop()
        time_end = time.perf_counter_ns()
        time_taken = time_end - time_start
        times.append(time_taken)
    # torch.cuda.cudart().cudaProfilerStop()
    return times


# @pytest.mark.L0
# @pytest.mark.skipif(torch.cuda.is_available() is False, reason="requires CUDA for torch.compile")
# @RunIf(requires_file=[CLUSTER_CONFIG.object_store_credential_pretrained])
# @pytest.mark.skipif(os.getenv("RUN_SKIPPED_TEST_LOCALLY") != "1", reason="local_test_only (torch.compile can be slow)")
# @torch.no_grad()
# def test_cosmos_ae_encode_compiled_matches_baseline():
#     torch.manual_seed(0)

#     device = torch.device("cuda")
#     dtype = torch.bfloat16
#     model = _make_cosmos_ae_from_s3()


#     num_warmups = 3
#     num_runs = 3

#     Ts = [93]

#     for T in Ts:
#         B, C, H, W = 1, 3, 640, 640
#         x = torch.randn(B, C, T, H, W, device=device, dtype=dtype)

#         baseline_times = benchmark(model.encode, x, num_warmups, num_runs)
#         print(f"Time taken baseline: {np.mean(baseline_times) / 1e6} ms")

#         model.model = torch.compile(model.model.encoder)
#         compiled_times = benchmark(model.encode, x, num_warmups, num_runs)
#         print(f"Time taken compiled: {np.mean(compiled_times) / 1e6} ms")


#         # Compiled path can fuse ops differently; allow a slightly looser tolerance.
#         # torch.testing.assert_close(out[:, :, :latent_t], baseline, rtol=1e-3, atol=1e-1)


def compute_psnr(original: torch.Tensor, reconstructed: torch.Tensor) -> float:
    """Compute PSNR between two tensors in [-1, 1] range.

    Args:
        original: (B, C, T, H, W) in [-1, 1]
        reconstructed: (B, C, T, H, W) in [-1, 1]

    Returns:
        PSNR in dB
    """
    # Convert from [-1, 1] to [0, 1]
    orig = (original.float() + 1.0) / 2.0
    recon = (reconstructed.float() + 1.0) / 2.0

    mse = F.mse_loss(recon, orig).item()
    if mse < 1e-10:
        return float("inf")
    return 10.0 * math.log10(1.0 / mse)


@pytest.mark.L0
@pytest.mark.skipif(torch.cuda.is_available() is False, reason="requires CUDA for torch.compile")
@RunIf(requires_file=[CLUSTER_CONFIG.object_store_credential_pretrained])
@pytest.mark.skipif(os.getenv("RUN_SKIPPED_TEST_LOCALLY") != "1", reason="local_test_only (torch.compile can be slow)")
@torch.inference_mode()
def test_cosmos_ae_encode_batched_matches_baseline():
    torch.manual_seed(0)
    torch._dynamo.config.cache_size_limit = 300

    num_warmups = 3
    num_runs = 3

    Ts = [1, 93, 301, 1001]
    # Ts = [93]
    encoder_width_list_list = [
        None,
        #    [0,128,256,512,1024,1024,1024],
        #    [0,64,128,512,1024,1024,1024],
        #    [0,64,128,256,1024,1024,1024],
        #    [0,64,128,256,512,1024,1024]
    ]
    for encoder_width_list in encoder_width_list_list:
        print(f"encoder_width_list={encoder_width_list}")
        for T in Ts:
            B, C, H, W = 1, 3, 256, 256
            x = (torch.randn(T, H, W, C) * 255).clip(0, 255).to(torch.uint8)
            x = numpy2tensor(x.cpu().numpy()).transpose(0, 1).unsqueeze(0)

            model = _make_cosmos_ae_from_s3(encoder_width_list)
            baseline_times = benchmark(model.encode, x, num_warmups, num_runs)
            baseline = model.encode(x)
            print(f"[baseline, T={T}] mean={np.mean(baseline_times) / 1e6} ms, std={np.std(baseline_times) / 1e6} ms")
            baseline_decoded = model.decode(baseline)

            # decode_baseline_times = benchmark(model.decode, baseline, num_warmups, num_runs)
            # print(f"Time taken decode baseline: {np.mean(decode_baseline_times) / 1e6} ms")

            torch.backends.cudnn.benchmark = True
            model = _make_cosmos_ae_from_s3_compiled(encoder_width_list)
            model.model.cfg.use_feature_cache = True

            model.model.encoder = model.model.encoder.to(memory_format=torch.channels_last_3d)
            # scale = 1
            # model.model.cfg.encode_temporal_tile_size = 16 * scale
            # model.model.cfg.encode_temporal_tile_latent_size = 4 * scale

            model.model.encoder = torch.compile(model.model.encoder, fullgraph=True, mode="max-autotune")
            batched_times = benchmark(model.encode, x, num_warmups, num_runs)
            print(f"[compiled, T={T}] mean={np.mean(batched_times) / 1e6} ms, std={np.std(batched_times) / 1e6} ms")
            batched = model.encode(x)
            batched_decoded = model.decode(batched)
            print(f"{baseline_decoded.shape=}, {batched_decoded.shape=}")
            assert baseline.shape == batched.shape

            psnr_baseline = compute_psnr(x, baseline_decoded[:, :, :T])
            psnr_batched = compute_psnr(x, batched_decoded[:, :, :T])
            print(f"PSNR baseline: {psnr_baseline:.2f} dB")
            print(f"PSNR batched: {psnr_batched:.2f} dB")
            torch.testing.assert_close(psnr_batched, psnr_baseline, rtol=5e-5, atol=5e-4)
            # # # torch.testing.assert_close(baseline_decoded, batched_decoded, rtol=1e-3, atol=1e-1)
            mask = (baseline - batched).abs() > 1e-1
            print(f"Percentage of outliers: {mask.sum() / baseline.numel() * 100:.2f}%")
            # Allow 2% outliers
            assert mask.sum() <= 0.02 * baseline.numel(), (
                f"Percentage of outliers is too high: {mask.sum() / baseline.numel() * 100:.2f}%"
            )
            torch.testing.assert_close(batched[~mask], baseline[~mask], rtol=1e-3, atol=1e-1)


@pytest.mark.L0
@RunIf(
    requires_file=[
        CLUSTER_CONFIG.object_store_credential_pretrained,
        # UNITTEST_CONFIG.object_store_credential_data,
    ]
)
@pytest.mark.skipif(os.getenv("RUN_SKIPPED_TEST_LOCALLY") != "1", reason="local_test_only")
def test_cosmos_ae_local_video_psnt_compile():
    """Test encode/decode with a real video from S3."""
    model = _make_cosmos_ae_from_s3(None)
    scale = 4
    model.model.cfg.encode_temporal_tile_size = scale * 16
    model.model.cfg.decode_temporal_tile_size = scale * 16
    model.model.cfg.encode_temporal_tile_latent_size = scale * 4
    model.model.cfg.decode_temporal_tile_latent_size = scale * 4
    # Load video as numpy array (T, H, W, C) in range [0, 255]
    video_in_numpy = easy_io.load(
        "results/reference/0.mp4",
        backend_args={
            # "backend": "s3",
            # "s3_credential_path": UNITTEST_CONFIG.object_store_credential_data,
        },
    )[0][:93]

    # Pad video to meet stride alignment requirements
    padded_video_batch, crop_region = pad_video_batch(
        video_in_numpy[np.newaxis, ...],  # Add batch dimension
        temporal_align=4,  # Temporal compression factor
        spatial_align=32,  # Spatial compression factor for cosmos_ae
        causal_mode=True,
        only_pad_end=True,
    )

    # Convert to tensor format (B, C, T, H, W) in range [-1, 1]
    video_tensor = numpy2tensor(padded_video_batch)

    # Encode and decode
    print(f"\n[CosmosAE Local Video] Input tensor shape: {video_tensor.shape}")
    latents = model.encode(video_tensor)
    print(f"[CosmosAE Local Video] Latent shape: {latents.shape}")
    print(f"[CosmosAE Local Video] Latent statistics: mean={latents.mean():.4f}, std={latents.std():.4f}")
    video_recon = model.decode(latents)
    print(f"[CosmosAE Local Video] Reconstructed shape: {video_recon.shape}")

    # Convert back to numpy and unpad
    video_recon_numpy = tensor2numpy(video_recon)
    video_recon_unpadded = unpad_video_batch(video_recon_numpy, crop_region)

    output_path = os.path.expanduser(f"logs/cosmos_ae_recon_0_baseline_scale_{scale}.mp4")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    easy_io.dump(video_recon_unpadded[0].astype("uint8"), output_path)

    psnd_baseline = compute_psnr(video_tensor, numpy2tensor(video_recon_unpadded))

    torch.backends.cudnn.benchmark = True
    model = _make_cosmos_ae_from_s3_compiled(None)
    model.model.cfg.use_feature_cache = True
    # print(model.model.encoder)
    model.model.encoder = model.model.encoder.to(memory_format=torch.channels_last_3d)
    model.model.cfg.batch_tiles = True
    model.model.cfg.encode_temporal_tile_size = scale * 16
    model.model.cfg.decode_temporal_tile_size = scale * 16
    model.model.cfg.encode_temporal_tile_latent_size = scale * 4
    model.model.cfg.decode_temporal_tile_latent_size = scale * 4
    # model.model.cfg.mini_batch_size = 2
    model.model.encoder = torch.compile(model.model.encoder, fullgraph=True, mode="max-autotune")
    for i in range(10):
        _ = model.encode(video_tensor)
        torch.cuda.synchronize()
    latents = model.encode(video_tensor)
    video_recon = model.decode(latents)
    video_recon_numpy = tensor2numpy(video_recon)
    video_recon_unpadded = unpad_video_batch(video_recon_numpy, crop_region)

    psnr_batched = compute_psnr(video_tensor, numpy2tensor(video_recon_unpadded))

    output_path = os.path.expanduser(f"logs/cosmos_ae_recon_0_compiled_scale_{scale}.mp4")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    easy_io.dump(video_recon_unpadded[0].astype("uint8"), output_path)

    print(f"PSNR batched: {psnr_batched:.2f} dB")
    print(f"PSNR baseline: {psnd_baseline:.2f} dB")
    torch.testing.assert_close(psnr_batched, psnd_baseline, rtol=1e-3, atol=1e-1)
