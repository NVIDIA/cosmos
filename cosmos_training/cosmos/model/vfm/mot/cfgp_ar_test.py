# -----------------------------------------------------------------------------
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# -----------------------------------------------------------------------------

"""Multi-rank tests for CFGP (CFG Parallelism) in AR inference.

Run with:
    torchrun --nproc_per_node=2 -m pytest cosmos/model/vfm/mot/cfgp_ar_test.py -v -m L0
"""

import os

import pytest
import torch
import torch.distributed as dist

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


@pytest.mark.L0
def test_cfgp_velocity_exchange_correctness():
    """P2P velocity exchange produces the correct CFG-blended output on both ranks.

    Rank 0 holds v_cond, rank 1 holds v_uncond.  After batch_isend_irecv the
    CFG blend ``v_uncond + guidance * (v_cond - v_uncond)`` must be identical
    on both ranks and match the analytic expectation.
    """
    rank, world_size = setup_distributed_environment()
    if world_size < 2:
        pytest.skip("requires at least 2 GPUs")

    device = torch.device("cuda", rank)
    parallel_dims = ParallelDims(enable_inference_mode=True, world_size=world_size, dp_shard=1, cfgp=2)
    parallel_dims.build_meshes("cuda")
    cfgp_mesh = parallel_dims.cfgp_mesh
    cfgp_group = cfgp_mesh.get_group()
    cfgp_rank = parallel_dims.cfgp_rank
    cfgp_size = parallel_dims.cfgp_size

    # Known velocities: rank 0 = cond, rank 1 = uncond
    v_cond_val = 3.0
    v_uncond_val = 1.0
    guidance = 7.5
    shape = (1, 8)

    v = (
        torch.full(shape, v_cond_val, device=device, dtype=torch.float32)
        if cfgp_rank == 0
        else torch.full(shape, v_uncond_val, device=device, dtype=torch.float32)
    )  # [1,8]
    other_v = torch.empty_like(v)  # [1,8]

    cfgp_peer = (cfgp_rank + 1) % cfgp_size
    reqs = dist.batch_isend_irecv(
        [
            dist.P2POp(op=dist.isend, tensor=v, group_peer=cfgp_peer, group=cfgp_group),
            dist.P2POp(op=dist.irecv, tensor=other_v, group_peer=cfgp_peer, group=cfgp_group),
        ]
    )
    for req in reqs:
        req.wait()

    velocity_cond = v if cfgp_rank == 0 else other_v  # [1,8]
    velocity_uncond = other_v if cfgp_rank == 0 else v  # [1,8]
    velocity_pred = velocity_uncond + guidance * (velocity_cond - velocity_uncond)  # [1,8]

    expected_val = v_uncond_val + guidance * (v_cond_val - v_uncond_val)
    expected = torch.full(shape, expected_val, device=device, dtype=torch.float32)  # [1,8]
    torch.testing.assert_close(velocity_pred, expected, msg=f"rank {cfgp_rank}: CFG blend mismatch")

    dist.barrier()
    if rank == 0:
        print("=== test_cfgp_velocity_exchange_correctness passed")


@pytest.mark.L0
def test_cfgp_ar_denoising_loop_equivalence():
    """cfgp=2 multi-step denoising produces bit-exact results vs the sequential cfgp=1 path.

    The CFGP path calls velocity_fn once per denoising step; each call:
      1. Each rank computes its own branch (cond or uncond) as a deterministic function of (x, t)
      2. P2P exchange so both ranks hold v_cond and v_uncond
      3. CFG blend: v_pred = v_uncond + guidance * (v_cond - v_uncond)

    The sequential path computes both branches on one rank and blends identically.

    Because both paths perform the same arithmetic on the same values, the final
    denoised tensor must be bit-exact (rtol=0, atol=0) — no floating-point reordering
    occurs since the exchange merely moves tensors, not reduces them.
    """
    rank, world_size = setup_distributed_environment()
    if world_size < 2:
        pytest.skip("requires at least 2 GPUs")

    device = torch.device("cuda", rank)
    parallel_dims = ParallelDims(enable_inference_mode=True, world_size=world_size, dp_shard=1, cfgp=2)
    parallel_dims.build_meshes("cuda")
    cfgp_mesh = parallel_dims.cfgp_mesh
    cfgp_group = cfgp_mesh.get_group()
    cfgp_rank = parallel_dims.cfgp_rank
    cfgp_size = parallel_dims.cfgp_size

    guidance = 7.5
    latent_dim = 32
    num_steps = 5

    # Fixed linear velocity functions: v depends on current state x and timestep t.
    # Both ranks use the same cond/uncond weights (deterministic, seeded).
    torch.manual_seed(42)
    w_cond = torch.randn(latent_dim, device=device)  # [D]
    b_cond = torch.randn(latent_dim, device=device)  # [D]
    w_uncond = torch.randn(latent_dim, device=device)  # [D]
    b_uncond = torch.randn(latent_dim, device=device)  # [D]

    def v_cond_fn(x: torch.Tensor, t: float) -> torch.Tensor:
        return w_cond * x + b_cond * t  # [D]

    def v_uncond_fn(x: torch.Tensor, t: float) -> torch.Tensor:
        return w_uncond * x + b_uncond * t  # [D]

    # ── CFGP=2 path ──────────────────────────────────────────────────────────
    torch.manual_seed(0)
    x = torch.randn(latent_dim, device=device)  # [D] — same on both ranks

    for step in range(num_steps):
        t = 1.0 - step / num_steps
        v = v_cond_fn(x, t).contiguous() if cfgp_rank == 0 else v_uncond_fn(x, t).contiguous()
        other_v = torch.empty_like(v)
        cfgp_peer = (cfgp_rank + 1) % cfgp_size
        reqs = dist.batch_isend_irecv(
            [
                dist.P2POp(op=dist.isend, tensor=v, group_peer=cfgp_peer, group=cfgp_group),
                dist.P2POp(op=dist.irecv, tensor=other_v, group_peer=cfgp_peer, group=cfgp_group),
            ]
        )
        for req in reqs:
            req.wait()
        v_cond = v if cfgp_rank == 0 else other_v
        v_uncond = other_v if cfgp_rank == 0 else v
        v_pred = v_uncond + guidance * (v_cond - v_uncond)  # [D]
        x = x - v_pred / num_steps  # Euler step

    x_cfgp = x  # [D] — final denoised latent under CFGP

    # ── Sequential reference (cfgp=1): computed only on rank 0, then broadcast ──
    torch.manual_seed(0)
    x_seq = torch.randn(latent_dim, device=device)  # [D]

    if rank == 0:
        for step in range(num_steps):
            t = 1.0 - step / num_steps
            v_pred = v_uncond_fn(x_seq, t) + guidance * (v_cond_fn(x_seq, t) - v_uncond_fn(x_seq, t))
            x_seq = x_seq - v_pred / num_steps
    dist.broadcast(x_seq, src=0, group=cfgp_group)

    # Bit-exact: no floating-point reordering in the CFGP path (P2P is just a copy)
    torch.testing.assert_close(
        x_cfgp,
        x_seq,
        rtol=0,
        atol=0,
        msg=f"rank {cfgp_rank}: cfgp=2 denoised result differs from sequential reference",
    )

    dist.barrier()
    if rank == 0:
        print("=== test_cfgp_ar_denoising_loop_equivalence passed")


@pytest.mark.L0
def test_cfgp_seed_broadcast():
    """_broadcast_seed() aligns seeds across the CFGP group.

    Both ranks generate the same noise tensor when using the broadcast seed,
    even though they started with different local seeds.
    """
    rank, world_size = setup_distributed_environment()
    if world_size < 2:
        pytest.skip("requires at least 2 GPUs")

    device = torch.device("cuda", rank)
    parallel_dims = ParallelDims(enable_inference_mode=True, world_size=world_size, dp_shard=1, cfgp=2)
    parallel_dims.build_meshes("cuda")
    cfgp_mesh = parallel_dims.cfgp_mesh
    cfgp_group = cfgp_mesh.get_group()
    cfgp_rank = parallel_dims.cfgp_rank
    cfgp_size = parallel_dims.cfgp_size

    from cosmos.model.vfm.omni_mot_model import _broadcast_seed

    # Ranks start with different local seeds
    local_seed = 42 if cfgp_rank == 0 else 99
    broadcast_seed = _broadcast_seed([local_seed], cfgp_group, cfgp_rank)[0]

    # Both ranks should now use seed 42 (broadcasted from rank 0)
    ref = torch.zeros(1, 16, device=device)
    noise = torch.empty_like(ref).normal_(generator=torch.Generator(device=device).manual_seed(broadcast_seed))
    # [1,16]

    # Gather noise from both ranks; verify they're identical
    gathered = [torch.empty_like(noise) for _ in range(cfgp_size)]
    dist.all_gather(gathered, noise, group=cfgp_group)

    torch.testing.assert_close(
        gathered[0], gathered[1], msg="Noise tensors differ across CFGP ranks after seed broadcast"
    )

    dist.barrier()
    if rank == 0:
        print("=== test_cfgp_seed_broadcast passed")
