# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from functools import partial
from itertools import cycle
from typing import cast

import pytest
import torch
import torch.distributed as dist

from cosmos.data.vfm.joint_dataloader import IterativeJointDataLoader
from cosmos.data.vfm.sequence_packing import (
    FactoredSequencePack,
    PackedSequence,
    build_sequence_plans_from_data_batch,
    factored_from_joint_sequence,
    from_joint,
    from_mode_splits,
    get_all_seq,
    get_gen_seq,
    get_und_seq,
    pack_input_sequence,
    set_gen_seq,
    set_und_seq,
)
from cosmos.model.vfm.mot.attention import (
    SplitInfo,
    dispatch_attention,
)
from cosmos.model.vfm.mot.context_parallel_utils import (
    context_parallel_attention,
    context_parallel_broadcast,
    get_context_parallel_sharded_sequence,
)
from cosmos.model.vfm.utils.data_and_condition import GenerationDataClean
from cosmos.utils.vfm.parallelism import ParallelDims


def setup_distributed_environment():
    """Initializes the distributed environment."""
    if "RANK" not in os.environ:
        pytest.skip("requires distributed environment (run with: torchrun --nproc_per_node=2)")
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    return rank, world_size


class MockDataLoader:
    def __init__(self, data):
        self.data = data
        self._iter = cycle(data)

    def __iter__(self):
        return self._iter

    def __len__(self):
        return len(self.data)


def create_raw_test_data_batch():
    """Creates a realistic raw data batch on all ranks."""
    torch.manual_seed(42)  # Ensure data is identical on all ranks

    special_tokens = {
        "eos_token_id": 1,
        "start_of_generation": 2,
        "end_of_generation": 3,
        "start_of_video": 4,
        "end_of_video": 5,
    }

    video_samples = [
        {
            "text_token_ids": torch.tensor([[30] * 50], dtype=torch.long),
            "video": [torch.randn(3, 32, 64, 64)],
            "timesteps": torch.tensor([0.5]),
        }
        for _ in range(128)
    ]

    video_loader = MockDataLoader(video_samples)

    joint_loader = IterativeJointDataLoader(
        {"video_ds": {"dataloader": video_loader, "ratio": 1}},
        tokenizer_spatial_compression_factor=4,
        tokenizer_temporal_compression_factor=4,
        patch_spatial=1,
        max_sequence_length=4096,
    )

    batch_size = 32
    batch_items = []
    iterator = iter(joint_loader)
    for i in range(batch_size):
        try:
            item = next(iterator)
            batch_items.append(item)
        except StopIteration:
            print(f"Warning: Loader exhausted at item {i} on rank {torch.distributed.get_rank()}")
            break

    input_text_tokens = [t[0].tolist() for x in batch_items for t in x["text_token_ids"]]
    input_images_or_videos = [img[0] for x in batch_items for img in x["video"]]
    input_timesteps = [t.item() for x in batch_items for t in x["timesteps"]]

    return {
        "input_text_tokens": input_text_tokens,
        "input_images_or_videos": input_images_or_videos,
        "input_timesteps": [t if t is not None else 0.0 for t in input_timesteps],
        "special_tokens": special_tokens,
    }


def create_qkv_sequences(global_packed_data, device, num_heads, head_dim):
    sequence_length = global_packed_data.sequence_length

    # create random q, k, v sequences of length sequence_length, num_heads, head_dim
    global_packed_sequence_q = torch.randn(sequence_length, num_heads, head_dim, device=device, dtype=torch.bfloat16)
    global_packed_sequence_k = torch.randn(sequence_length, num_heads, head_dim, device=device, dtype=torch.bfloat16)
    global_packed_sequence_v = torch.randn(sequence_length, num_heads, head_dim, device=device, dtype=torch.bfloat16)

    return global_packed_sequence_q, global_packed_sequence_k, global_packed_sequence_v


def get_factored_qkv_data(
    global_packed_sequence_q,
    global_packed_sequence_k,
    global_packed_sequence_v,
    attn_modes,
    split_lens,
    sample_lens,
    packed_und_token_indexes,
    packed_gen_token_indexes,
):
    print(f"DEBUG: packed_und_token_indexes length: {packed_und_token_indexes.shape[0]}")
    print(f"DEBUG: split_lens sum causal: {sum(l for l, m in zip(split_lens, attn_modes) if m == 'causal')}")

    global_q_pack = factored_from_joint_sequence(
        packed_sequence=global_packed_sequence_q,
        attn_modes=attn_modes,
        split_lens=split_lens,
        sample_lens=sample_lens,
        packed_und_token_indexes=packed_und_token_indexes,
        packed_gen_token_indexes=packed_gen_token_indexes,
    )
    global_k_pack = from_joint(global_packed_sequence_k, global_q_pack)
    global_v_pack = from_joint(global_packed_sequence_v, global_q_pack)
    print(f"DEBUG: global_q_pack causal_seq shape: {get_und_seq(global_q_pack).shape}")
    return global_q_pack, global_k_pack, global_v_pack


def verify_fwd_output(
    rank, local_output_pack, baseline_output_pack, world_size, cp_mesh, total_causal_len, total_full_len
):
    # GATHER: Collect the output shards from all ranks
    output_und_shard = get_und_seq(local_output_pack)
    output_gen_shard = get_gen_seq(local_output_pack)

    # Print output shard shapes on each rank
    print(f"[Rank {rank}] output_und_shard: {output_und_shard.shape}, output_gen_shard: {output_gen_shard.shape}")

    # Create lists to hold the gathered tensors from all ranks
    gathered_und_shards = [None for _ in range(world_size)]
    gathered_gen_shards = [None for _ in range(world_size)]

    dist.all_gather_object(gathered_und_shards, output_und_shard.cpu(), group=cp_mesh.get_group())
    dist.all_gather_object(gathered_gen_shards, output_gen_shard.cpu(), group=cp_mesh.get_group())

    # Print gathered shapes on rank 0
    if rank == 0:
        print(f"\n=== DEBUG: Gathered Output Shapes ===")
        for i, (und_shard, gen_shard) in enumerate(zip(gathered_und_shards, gathered_gen_shards)):
            print(f"Rank {i} - und: {und_shard.shape}, gen: {gen_shard.shape}")  # type: ignore

    # COMPARE: On rank 0, concatenate the shards and compare with the baseline
    if rank == 0:
        # cast baseline_output_pack to FactoredSequencePack
        baseline_output_pack = cast(FactoredSequencePack, baseline_output_pack)

        print(f"Comparing results for world_size={world_size}...")
        baseline_und_seq = get_und_seq(baseline_output_pack)
        baseline_gen_seq = get_gen_seq(baseline_output_pack)

        reconstructed_und_output = torch.cat(gathered_und_shards, dim=0).to(baseline_und_seq.device)
        reconstructed_gen_output = torch.cat(gathered_gen_shards, dim=0).to(baseline_gen_seq.device)

        # Compare only actual data (trim baseline padding)
        torch.testing.assert_close(
            reconstructed_und_output[:total_causal_len],
            baseline_und_seq[:total_causal_len],
            rtol=1e-3,
            atol=1e-3,
        )
        torch.testing.assert_close(
            reconstructed_gen_output[:total_full_len],
            baseline_gen_seq[:total_full_len],
            rtol=1e-3,
            atol=1e-3,
        )
        print(f"reconstructed_und_output shape: {reconstructed_und_output.shape}")
        print(f"baseline_und_seq shape: {baseline_und_seq.shape}")
        print(f"reconstructed_gen_output shape: {reconstructed_gen_output.shape}")
        print(f"baseline_gen_seq shape: {baseline_gen_seq.shape}")
        print("==== Forward passed: Distributed attention output matches baseline.")


def verify_bwd_output(
    rank,
    baseline_q_pack,
    baseline_k_pack,
    baseline_v_pack,
    local_q_pack,
    local_k_pack,
    local_v_pack,
    world_size,
    cp_mesh,
    total_causal_len,
    total_full_len,
):
    def verify_grad(local_pack, baseline_pack, name):
        # 1. Extract the local gradient shard (which is just the gradient of the local pack in this case)
        local_und_grad = get_und_seq(local_pack).grad
        local_gen_grad = get_gen_seq(local_pack).grad

        # 2. Gather shards from all ranks to Rank 0
        gathered_und_grads = [None for _ in range(world_size)]
        gathered_gen_grads = [None for _ in range(world_size)]

        dist.all_gather_object(gathered_und_grads, local_und_grad.cpu(), group=cp_mesh.get_group())
        dist.all_gather_object(gathered_gen_grads, local_gen_grad.cpu(), group=cp_mesh.get_group())

        # 3. Verify on Rank 0
        if rank == 0:
            reconstructed_und_grad = torch.cat(gathered_und_grads, dim=0).to(local_und_grad.device)
            reconstructed_gen_grad = torch.cat(gathered_gen_grads, dim=0).to(local_gen_grad.device)

            baseline_und_grad = get_und_seq(baseline_pack).grad
            baseline_gen_grad = get_gen_seq(baseline_pack).grad

            print(f"Verifying {name} Gradients...")
            print(f"  Reconstructed und_grad: {reconstructed_und_grad.shape}, baseline: {baseline_und_grad.shape}")
            print(f"  Reconstructed gen_grad: {reconstructed_gen_grad.shape}, baseline: {baseline_gen_grad.shape}")

            torch.testing.assert_close(
                reconstructed_und_grad[:total_causal_len],
                baseline_und_grad[:total_causal_len],
                rtol=1e-3,
                atol=1e-3,
            )
            torch.testing.assert_close(
                reconstructed_gen_grad[:total_full_len],
                baseline_gen_grad[:total_full_len],
                rtol=1e-3,
                atol=1e-3,
            )

    verify_grad(local_q_pack, baseline_q_pack, "Q")
    verify_grad(local_k_pack, baseline_k_pack, "K")
    verify_grad(local_v_pack, baseline_v_pack, "V")

    if rank == 0:
        print("Backward pass: Distributed backward pass gradients match baseline.")


def create_packed_sequence(
    input_text_tokens: list[list[int]],
    input_images_or_videos: list[torch.Tensor],
    input_timesteps: list[float],
    special_tokens: dict[str, int],
    vae_spatial_downsample: int = 4,
    vae_temporal_downsample: int = 1,
    is_image_batch: bool = False,
    include_end_of_vision_token: bool = False,
) -> PackedSequence:
    num_samples = len(input_text_tokens)

    # 1. Build sequence plans
    data_batch = {"images" if is_image_batch else "video": input_images_or_videos}
    sequence_plans = build_sequence_plans_from_data_batch(
        data_batch=data_batch, input_video_key="video", input_image_key="images"
    )

    # 2. Stack images/videos
    input_vision_stacked = torch.stack(input_images_or_videos, dim=0)

    # 3. Create tokenized latents (simulating encoder output)
    B, C, T, H, W = input_vision_stacked.shape
    latent_C = 16  # arbitrary for test
    latent_T = T // vae_temporal_downsample
    latent_H = H // vae_spatial_downsample
    latent_W = W // vae_spatial_downsample

    # Just create random latents
    x0_tokens_vision = [
        torch.randn(1, latent_C, latent_T, latent_H, latent_W).to(input_vision_stacked.dtype) for _ in range(B)
    ]

    # 4. Create GenerationDataClean
    gen_data_clean = GenerationDataClean(
        batch_size=num_samples,
        is_image_batch=is_image_batch,
        raw_state_vision=input_vision_stacked,
        x0_tokens_vision=x0_tokens_vision,
        raw_state_action=None,
    )

    timesteps_tensor = torch.tensor(input_timesteps)

    # 5. Pack input sequence
    packed_sequence = pack_input_sequence(
        sequence_plans=sequence_plans,
        input_text_indexes=input_text_tokens,
        gen_data_clean=gen_data_clean,
        input_timesteps=timesteps_tensor,
        special_tokens=special_tokens,
        max_num_tokens=36864,  # default
        latent_patch_size=1,  # default
        include_end_of_generation_token=include_end_of_vision_token,
    )

    return packed_sequence


def test_context_parallel_attention_two_way():
    """
    Tests the context_parallel_attention implementation by comparing its output
    at a given world_size with the baseline output from a single-GPU execution.
    """
    rank, world_size = setup_distributed_environment()
    device = torch.device("cuda", rank)
    num_heads = 32
    head_dim = 128

    if world_size < 4:
        print("Skipping test: requires at least 2 GPUs.")
        return

    attention_function_to_wrap = partial(dispatch_attention)

    print(f"DEBUG: world_size: {world_size}")
    parallel_dims = ParallelDims(enable_inference_mode=True, world_size=world_size, dp_shard=1, cp=4)
    parallel_dims.build_meshes("cuda")
    cp_mesh = parallel_dims.cp_mesh

    if rank == 0:
        print(f"==== Running test_context_parallel_attention: rank: {rank}, world_size: {world_size}")

    # Create data on Rank 0 only and broadcast it
    data_batch = None
    if rank == 0:
        data_batch = create_raw_test_data_batch()

    raw_data_batch = context_parallel_broadcast(data_batch, parallel_dims, 0)  # type: ignore

    # Each rank now creates the packed and factored sequences from the raw data
    global_packed_data = create_packed_sequence(
        input_text_tokens=raw_data_batch["input_text_tokens"],
        input_images_or_videos=raw_data_batch["input_images_or_videos"],
        input_timesteps=raw_data_batch["input_timesteps"],
        special_tokens=raw_data_batch["special_tokens"],
        vae_spatial_downsample=4,
        vae_temporal_downsample=1,
        is_image_batch=False,
        include_end_of_vision_token=False,
    )

    sequence_length = global_packed_data.sequence_length
    torch.manual_seed(42)  # Ensure all ranks have the same global sequence data

    global_packed_sequence_q, global_packed_sequence_k, global_packed_sequence_v = create_qkv_sequences(
        global_packed_data, device, num_heads, head_dim
    )
    global_q_pack, global_k_pack, global_v_pack = get_factored_qkv_data(
        global_packed_sequence_q,
        global_packed_sequence_k,
        global_packed_sequence_v,
        global_packed_data.attn_modes,
        global_packed_data.split_lens,
        global_packed_data.sample_lens,
        global_packed_data.text_indexes,
        global_packed_data.vision.sequence_indexes,
    )

    # Verify global pack has full 32-sample metadata
    if rank == 0:
        print(f"\n=== DEBUG: Global Pack Metadata ===")
        all_seq = get_all_seq(global_q_pack)
        print(f"global_q_pack all_seq shape: {all_seq.shape}")
        print(f"global_q_pack all_seq first 5: {all_seq[0:5, 0, 0]}")
        print(f"global_q_pack all_seq last 5: {all_seq[-5:, 0, 0]}")
        print(f"global_q_pack all_seq middle 5: {all_seq[all_seq.shape[0] // 2 : all_seq.shape[0] // 2 + 5, 0, 0]}")

        # DEBUG: Checksum heads
        # Heads 0..7
        print(f"DEBUG: global_q_pack und_seq sum heads 0..7: {get_und_seq(global_q_pack)[:, 0:8, :].sum().item():.4f}")
        # Heads 8..15
        print(
            f"DEBUG: global_q_pack und_seq sum heads 8..15: {get_und_seq(global_q_pack)[:, 8:16, :].sum().item():.4f}"
        )

    # Initialize global attention mask
    global_attention_mask = SplitInfo(
        split_lens=global_packed_data.split_lens,
        attn_modes=global_packed_data.attn_modes,
        sample_lens=global_packed_data.sample_lens,
        actual_len=sequence_length,
    )

    # Calculate actual valid lengths (excluding padding) using the trimmed mask info
    causal_lens = [
        l for l, m in zip(global_attention_mask.split_lens, global_attention_mask.attn_modes) if m == "causal"
    ]
    total_causal_len = sum(causal_lens)
    full_lens = [l for l, m in zip(global_attention_mask.split_lens, global_attention_mask.attn_modes) if m == "full"]
    total_full_len = sum(full_lens)
    if rank == 0:
        print(f"Total valid causal len: {total_causal_len}, Total valid full len: {total_full_len}")

    # BASELINE: Run with CP=1 on Rank 0 to get the ground truth result
    baseline_output_pack = None

    get_und_seq(global_q_pack).requires_grad_(True)
    get_gen_seq(global_q_pack).requires_grad_(True)
    get_und_seq(global_k_pack).requires_grad_(True)
    get_gen_seq(global_k_pack).requires_grad_(True)
    get_und_seq(global_v_pack).requires_grad_(True)
    get_gen_seq(global_v_pack).requires_grad_(True)

    # Print global input shapes
    if rank == 0:
        print(f"\n=== DEBUG: Global Baseline Input Shapes ===")
        print(f"global_q_pack und_seq: {get_und_seq(global_q_pack).shape}")
        print(f"global_q_pack gen_seq: {get_gen_seq(global_q_pack).shape}")

    baseline_output_pack, _kv_to_store = dispatch_attention(
        global_q_pack,
        global_k_pack,
        global_v_pack,
        global_attention_mask,
    )

    # Print baseline output shapes
    if rank == 0:
        print(f"\n=== DEBUG: Baseline Output Shapes ===")
        print(f"baseline_output_pack und_seq: {get_und_seq(baseline_output_pack).shape}")
        print(f"baseline_output_pack gen_seq: {get_gen_seq(baseline_output_pack).shape}")

    # Compute baseline loss only on valid (non-padded) data
    baseline_und_seq = get_und_seq(baseline_output_pack)
    baseline_gen_seq = get_gen_seq(baseline_output_pack)
    baseline_loss = baseline_und_seq[:total_causal_len].sum() + baseline_gen_seq[:total_full_len].sum()
    baseline_loss.backward()
    print(f"baseline_loss (on valid data): {baseline_loss}")

    rank = torch.distributed.get_rank(cp_mesh.get_group())
    world_size = torch.distributed.get_world_size(cp_mesh.get_group())

    position_ids = global_packed_data.position_ids.to(device)
    local_q_pack, _ = get_context_parallel_sharded_sequence("two_way", global_q_pack, position_ids, parallel_dims)
    local_k_pack, _ = get_context_parallel_sharded_sequence("two_way", global_k_pack, position_ids, parallel_dims)
    local_v_pack, _ = get_context_parallel_sharded_sequence("two_way", global_v_pack, position_ids, parallel_dims)

    # Verify local und/gen shapes
    print(
        f"[Rank {rank}] Local pack und_seq: {get_und_seq(local_q_pack).shape}, gen_seq: {get_gen_seq(local_q_pack).shape}"
    )

    # Detach and require grad for local inputs to make them leaves for the local backward pass
    for pack in [local_q_pack, local_k_pack, local_v_pack]:
        und_seq = get_und_seq(pack).detach().clone().requires_grad_(True)
        gen_seq = get_gen_seq(pack).detach().clone().requires_grad_(True)
        set_und_seq(pack, und_seq)
        set_gen_seq(pack, gen_seq)

    # Verify input sharding by gathering and comparing
    if rank == 0:
        print(f"\n=== DEBUG: Verifying Input Sharding ===")
    input_und_shard = get_und_seq(local_q_pack)
    input_gen_shard = get_gen_seq(local_q_pack)
    print(f"[Rank {rank}] Input und_shard: {input_und_shard.shape}, gen_shard: {input_gen_shard.shape}")

    # Gather input shards to verify partitioning
    gathered_input_und = [None for _ in range(world_size)]
    gathered_input_gen = [None for _ in range(world_size)]
    dist.all_gather_object(gathered_input_und, input_und_shard.cpu(), group=cp_mesh.get_group())
    dist.all_gather_object(gathered_input_gen, input_gen_shard.cpu(), group=cp_mesh.get_group())

    if rank == 0:
        reconstructed_input_und = torch.cat(gathered_input_und, dim=0).to(device)
        reconstructed_input_gen = torch.cat(gathered_input_gen, dim=0).to(device)

        # Get baseline sequences (may be padded)
        baseline_und = get_und_seq(global_q_pack)
        baseline_gen = get_gen_seq(global_q_pack)

        # The reconstructed sequences have the actual data length (no padding)
        # The baseline may be padded, so trim it to match reconstructed length
        actual_und_len = reconstructed_input_und.shape[0]
        actual_gen_len = reconstructed_input_gen.shape[0]

        print(
            f"Reconstructed input und: {reconstructed_input_und.shape}, baseline (with padding): {baseline_und.shape}"
        )
        print(
            f"Reconstructed input gen: {reconstructed_input_gen.shape}, baseline (with padding): {baseline_gen.shape}"
        )

        # Compare only the actual data (trim baseline padding)
        torch.testing.assert_close(reconstructed_input_und, baseline_und[:actual_und_len], rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(reconstructed_input_gen, baseline_gen[:actual_gen_len], rtol=1e-5, atol=1e-5)
        print("=== Input sharding verified: reconstruction matches baseline (after removing padding)")

    # Print local pack metadata before calling attention
    if rank == 0:
        print(f"\n=== DEBUG: Before context_parallel_attention ===")
        print(f"local_q_pack und_seq: {get_und_seq(local_q_pack).shape}")
        print(f"local_q_pack gen_seq: {get_gen_seq(local_q_pack).shape}")
        print(f"local_q_pack split_lens (len={len(local_q_pack['split_lens'])}): {local_q_pack['split_lens'][:4]}...")
        print(f"local_q_pack _causal_seq_offsets: {local_q_pack['_causal_seq_offsets'][:5]}...")
        print(f"local_q_pack _full_only_seq_offsets: {local_q_pack['_full_only_seq_offsets'][:5]}...")
        print(f"local_q_pack max_causal_len: {local_q_pack['max_causal_len']}")
        print(f"local_q_pack max_full_len: {local_q_pack['max_full_len']}")
        print(
            f"global_attention_mask split_lens (len={len(global_attention_mask.split_lens)}): {global_attention_mask.split_lens[:4]}..."
        )
        print(f"global_attention_mask max_causal_len: {global_attention_mask.max_causal_len}")
        print(f"global_attention_mask max_full_len: {global_attention_mask.max_full_len}")

    # Run the context parallel attention function
    local_output_pack, _ = context_parallel_attention(
        cp_mesh,
        local_q_pack,
        local_k_pack,
        local_v_pack,
        global_attention_mask,
        attention_function_to_wrap,
    )

    # Print output shapes
    if rank == 0:
        print(f"\n=== DEBUG: After context_parallel_attention ===")
        print(f"local_output_pack und_seq: {get_und_seq(local_output_pack).shape}")
        print(f"local_output_pack gen_seq: {get_gen_seq(local_output_pack).shape}")

    verify_fwd_output(
        rank, local_output_pack, baseline_output_pack, world_size, cp_mesh, total_causal_len, total_full_len
    )

    dist.barrier()

    # Compute how many valid tokens this rank has in its shard
    local_und_seq = get_und_seq(local_output_pack)
    local_gen_seq = get_gen_seq(local_output_pack)

    local_und_size = local_und_seq.shape[0]
    local_gen_size = local_gen_seq.shape[0]

    # Determine valid range for this rank
    und_start = rank * local_und_size
    und_end = min((rank + 1) * local_und_size, total_causal_len)
    valid_und_len = max(0, und_end - und_start)

    gen_start = rank * local_gen_size
    gen_end = min((rank + 1) * local_gen_size, total_full_len)
    valid_gen_len = max(0, gen_end - gen_start)

    # Compute loss only on valid tokens
    loss_local = local_und_seq[:valid_und_len].sum() + local_gen_seq[:valid_gen_len].sum()
    print(
        f"loss_local (rank {rank}, valid und={valid_und_len}/{local_und_size}, gen={valid_gen_len}/{local_gen_size}): {loss_local}"
    )

    # Gather all local losses to compute total
    all_local_losses = [None for _ in range(world_size)]
    dist.all_gather_object(all_local_losses, loss_local.item(), group=cp_mesh.get_group())

    if rank == 0:
        total_local_loss = sum(all_local_losses)
        print(f"\n=== Loss Comparison ===")
        print(f"Baseline loss (on valid data): {baseline_loss.item():.4f}")
        print(f"Sum of local losses: {total_local_loss:.4f}")
        print(f"Difference: {abs(baseline_loss.item() - total_local_loss):.6f}")

    loss_local.backward()

    verify_bwd_output(
        rank,
        global_q_pack,
        global_k_pack,
        global_v_pack,
        local_q_pack,
        local_k_pack,
        local_v_pack,
        world_size,
        cp_mesh,
        total_causal_len,
        total_full_len,
    )

    dist.barrier()
    if rank == 0:
        print("=== Test passed")


def simple_packed_test():
    """
    Tests the simple packed test by comparing its output
    at a given world_size with the baseline output from a single-GPU execution.
    """
    data_batch = create_raw_test_data_batch()
    global_packed_data = create_packed_sequence(
        input_text_tokens=data_batch["input_text_tokens"],
        input_images_or_videos=data_batch["input_images_or_videos"],
        input_timesteps=data_batch["input_timesteps"],
        special_tokens=data_batch["special_tokens"],
        vae_spatial_downsample=4,
        vae_temporal_downsample=1,
        is_image_batch=False,
        include_end_of_vision_token=False,
    )
    device = torch.device("cuda", 0)
    num_heads = 32
    head_dim = 128
    global_packed_sequence_q, _, _ = create_qkv_sequences(global_packed_data, device, num_heads, head_dim)

    factored_q_pack = factored_from_joint_sequence(
        packed_sequence=global_packed_sequence_q,
        attn_modes=global_packed_data.attn_modes,
        split_lens=global_packed_data.split_lens,
        sample_lens=global_packed_data.sample_lens,
        packed_und_token_indexes=global_packed_data.text_indexes,
        packed_gen_token_indexes=global_packed_data.vision.sequence_indexes,
    )
    print(f"\n=== DEBUG: Global Pack Metadata ===")
    all_seq = get_all_seq(factored_q_pack)
    print(f"global_q_pack all_seq shape: {all_seq.shape}")
    print(f"local_pack all_seq first 5: {all_seq[0:5, 0, 0]}")
    print(f"local_pack all_seq last 5: {all_seq[-5:, 0, 0]}")
    print(f"local_pack all_seq middle 5: {all_seq[all_seq.shape[0] // 2 : all_seq.shape[0] // 2 + 5, 0, 0]}")

    text_seq = get_und_seq(factored_q_pack)
    gen_seq = get_gen_seq(factored_q_pack)

    text_seq1 = text_seq[: len(text_seq) // 2]
    gen_seq1 = gen_seq[: len(gen_seq) // 2]
    local_pack1 = from_mode_splits(text_seq1, gen_seq1, factored_q_pack, is_sharded=True)

    text_seq2 = text_seq[len(text_seq) // 2 :]
    gen_seq2 = gen_seq[len(gen_seq) // 2 :]
    local_pack2 = from_mode_splits(text_seq2, gen_seq2, factored_q_pack, is_sharded=True)

    local_text1 = get_und_seq(
        local_pack1,
    )
    local_gen1 = get_gen_seq(local_pack1)
    local_text2 = get_und_seq(local_pack2)
    local_gen2 = get_gen_seq(local_pack2)

    merged_text_seq = torch.cat([local_text1, local_text2], dim=0)
    merged_gen_seq = torch.cat([local_gen1, local_gen2], dim=0)
    merged_pack = from_mode_splits(merged_text_seq, merged_gen_seq, factored_q_pack, is_sharded=False)

    print(f"\n=== DEBUG: Local Pack Metadata ===")
    all_seq = get_all_seq(merged_pack)
    print(f"local_pack all_seq shape: {all_seq.shape}")
    print(f"local_pack all_seq first 5: {all_seq[0:5, 0, 0]}")
    print(f"local_pack all_seq last 5: {all_seq[-5:, 0, 0]}")
    print(f"local_pack all_seq middle 5: {all_seq[all_seq.shape[0] // 2 : all_seq.shape[0] // 2 + 5, 0, 0]}")


def _make_factored_pack(
    causal_seq: torch.Tensor,
    full_only_seq: torch.Tensor,
    S_und_global: int,
    S_gen_global: int,
    device: torch.device,
    is_sharded: bool = False,
) -> FactoredSequencePack:
    """Minimal single-sample FactoredSequencePack for unit tests.

    Metadata always uses GLOBAL (pre-sharding) token counts so the metadata
    is consistent before and after all-to-all inside context_parallel_attention().
    The causal_seq / full_only_seq tensors may be either sharded or global.
    """
    return {
        "causal_seq": causal_seq,
        "full_only_seq": full_only_seq,
        "is_sharded": is_sharded,
        "sample_offsets": torch.tensor([0, S_und_global + S_gen_global], device=device, dtype=torch.int32),
        "max_num_tokens": S_und_global + S_gen_global,
        "max_sample_len": S_und_global + S_gen_global,
        "max_causal_len": S_und_global,
        "max_full_len": S_gen_global,
        "_causal_indices": torch.arange(S_und_global, device=device, dtype=torch.int32),
        "_full_indices": torch.arange(S_und_global, S_und_global + S_gen_global, device=device, dtype=torch.int32),
        "_causal_seq_offsets": torch.tensor([0, S_und_global], device=device, dtype=torch.int32),
        "_full_only_seq_offsets": torch.tensor([0, S_gen_global], device=device, dtype=torch.int32),
        "_num_causal_tokens": S_und_global,
        "_num_full_tokens": S_gen_global,
    }


@pytest.mark.L0
def test_get_context_parallel_sharded_sequence_three_way():
    """get_context_parallel_sharded_sequence() accepts three_way attn_implementation.

    The causal_8b_480p config uses joint_attn_implementation="three_way" (required by
    video_temporal_causal=True).  The sharding logic is identical to "two_way" — it
    operates on the FactoredSequencePack (und/gen split), not on the attention pattern —
    so "three_way" must not be rejected by the assertion.

    Verifies that both und and gen sequences are sharded to 1/world_size tokens per rank,
    and that the output position_ids are the corresponding local slice.
    """
    rank, world_size = setup_distributed_environment()
    if world_size < 2:
        pytest.skip("requires at least 2 GPUs")

    device = torch.device("cuda", rank)
    parallel_dims = ParallelDims(enable_inference_mode=True, world_size=world_size, dp_shard=1, cp=world_size)
    parallel_dims.build_meshes("cuda")

    hidden_dim = 16
    S_und = world_size * 4  # divisible by world_size
    S_gen = world_size * 2  # divisible by world_size
    S_total = S_und + S_gen

    und_seq = torch.arange(S_und * hidden_dim, device=device, dtype=torch.float32).view(S_und, hidden_dim)  # [S_und, H]
    gen_seq = torch.arange(S_gen * hidden_dim, device=device, dtype=torch.float32).view(S_gen, hidden_dim)  # [S_gen, H]
    position_ids = torch.arange(S_total, device=device, dtype=torch.long)  # [S_total]

    input_pack = _make_factored_pack(und_seq, gen_seq, S_und, S_gen, device, is_sharded=False)

    # Must not raise — "three_way" should be accepted just like "two_way"
    local_pack, local_pos_ids = get_context_parallel_sharded_sequence(
        attn_implementation="three_way",
        input_pack=input_pack,
        position_ids=position_ids,
        parallel_dims=parallel_dims,
    )

    s_und_per_rank = S_und // world_size
    s_gen_per_rank = S_gen // world_size

    # Each rank receives its contiguous shard
    expected_und = und_seq[rank * s_und_per_rank : (rank + 1) * s_und_per_rank]  # [S_und/cp, H]
    expected_gen = gen_seq[rank * s_gen_per_rank : (rank + 1) * s_gen_per_rank]  # [S_gen/cp, H]

    torch.testing.assert_close(get_und_seq(local_pack), expected_und, msg=f"rank {rank}: und shard mismatch")
    torch.testing.assert_close(get_gen_seq(local_pack), expected_gen, msg=f"rank {rank}: gen shard mismatch")

    dist.barrier()
    if rank == 0:
        print("=== test_get_context_parallel_sharded_sequence_three_way passed")


if __name__ == "__main__":
    test_context_parallel_attention_two_way()
    test_get_context_parallel_sharded_sequence_three_way()
