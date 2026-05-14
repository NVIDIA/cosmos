"""Speed benchmark for Qwen3VLMoeTextExpertsGroupedMm.

Usage:
    # Default benchmark (forward only, compiled, bf16):
    python -m cosmos.model.vfm.vlm.qwen3_vl_moe.moe_bench

    # Forward + backward:
    python -m cosmos.model.vfm.vlm.qwen3_vl_moe.moe_bench --backward

    # Compare grouped_mm vs naive:
    python -m cosmos.model.vfm.vlm.qwen3_vl_moe.moe_bench --compare

    # Disable torch.compile:
    python -m cosmos.model.vfm.vlm.qwen3_vl_moe.moe_bench --no-compile

    # Custom sweep:
    python -m cosmos.model.vfm.vlm.qwen3_vl_moe.moe_bench --backward \
        --hidden-size 4096 \
        --moe-intermediate-size 1536

    # Capture a torch profiler trace (Chrome trace JSON):
    python -m cosmos.model.vfm.vlm.qwen3_vl_moe.moe_bench --profile

    # Profile to a custom directory:
    python -m cosmos.model.vfm.vlm.qwen3_vl_moe.moe_bench --profile --profile-dir ./my_traces

    # All options:
    python -m cosmos.model.vfm.vlm.qwen3_vl_moe.moe_bench --help
"""

import itertools
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import torch
import tyro

from cosmos.model.vfm.vlm.qwen3_vl_moe.configuration_qwen3_vl_moe import (
    Qwen3VLMoeTextConfig,
)
from cosmos.model.vfm.vlm.qwen3_vl_moe.moe import (
    Qwen3VLMoeTextExpertsGroupedMm,
    Qwen3VLMoeTextExpertsNaive,
)


@dataclass
class BenchConfig:
    """Benchmark Qwen3VLMoeTextExpertsGroupedMm."""

    num_tokens: list[int] = field(default_factory=lambda: [16384, 32768])
    num_experts: list[int] = field(default_factory=lambda: [128])
    top_k: list[int] = field(default_factory=lambda: [8])
    hidden_size: list[int] = field(default_factory=lambda: [2048])
    moe_intermediate_size: list[int] = field(default_factory=lambda: [768])
    num_warmup: int = 10
    num_iters: int = 100
    backward: bool = False
    """Also benchmark backward pass."""
    compare: bool = False
    """Compare grouped_mm vs naive."""
    compile: bool = True
    """Wrap module with torch.compile before benchmarking."""
    profile: bool = False
    """Capture a torch profiler trace after benchmarking."""
    profile_dir: str = "./profiles"
    """Directory to write Chrome trace JSON files."""
    dtype: Literal["bf16", "fp32"] = "bf16"


@dataclass
class BenchResult:
    num_tokens: int
    num_experts: int
    top_k: int
    hidden_size: int
    moe_intermediate_size: int
    fwd_ms: float
    bwd_ms: float
    peak_mem_mb: float


def _make_inputs(
    num_tokens: int,
    config: Qwen3VLMoeTextConfig,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    hidden_states = torch.randn(num_tokens, config.hidden_size, device=device, dtype=dtype)

    expert_indices = torch.stack(
        [torch.randperm(config.num_experts, device=device)[: config.num_experts_per_tok] for _ in range(num_tokens)]
    ).to(torch.int64)

    topk_scores = torch.rand(num_tokens, config.num_experts_per_tok, device=device, dtype=dtype)
    topk_scores = topk_scores / topk_scores.sum(dim=-1, keepdim=True)

    num_tokens_per_expert = torch.zeros(config.num_experts, dtype=torch.int32, device=device)
    for idx in expert_indices.view(-1):
        num_tokens_per_expert[idx] += 1

    return hidden_states, topk_scores, expert_indices, num_tokens_per_expert


def bench_forward(
    module: torch.nn.Module,
    hidden_states: torch.Tensor,
    topk_scores: torch.Tensor,
    expert_indices: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
    num_warmup: int = 20,
    num_iters: int = 100,
) -> float:
    for _ in range(num_warmup):
        with torch.no_grad():
            module(hidden_states, topk_scores, expert_indices, num_tokens_per_expert)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(num_iters):
        with torch.no_grad():
            module(hidden_states, topk_scores, expert_indices, num_tokens_per_expert)
    end.record()
    torch.cuda.synchronize()

    return start.elapsed_time(end) / num_iters


def bench_backward(
    module: torch.nn.Module,
    hidden_states: torch.Tensor,
    topk_scores: torch.Tensor,
    expert_indices: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
    num_warmup: int = 20,
    num_iters: int = 100,
) -> float:
    for _ in range(num_warmup):
        h = hidden_states.detach().requires_grad_(True)
        out = module(h, topk_scores, expert_indices, num_tokens_per_expert)
        out.sum().backward()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(num_iters):
        h = hidden_states.detach().requires_grad_(True)
        out = module(h, topk_scores, expert_indices, num_tokens_per_expert)
        out.sum().backward()
    end.record()
    torch.cuda.synchronize()

    return start.elapsed_time(end) / num_iters


def profile_run(
    module: torch.nn.Module,
    hidden_states: torch.Tensor,
    topk_scores: torch.Tensor,
    expert_indices: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
    output_path: str,
    include_backward: bool = False,
    num_warmup: int = 5,
    num_active: int = 3,
) -> None:
    """Run a few iterations under the torch profiler and export a Chrome trace."""

    def _step() -> None:
        if include_backward:
            h = hidden_states.detach().requires_grad_(True)
            out = module(h, topk_scores, expert_indices, num_tokens_per_expert)
            out.sum().backward()
        else:
            with torch.no_grad():
                module(hidden_states, topk_scores, expert_indices, num_tokens_per_expert)

    for _ in range(num_warmup):
        _step()
    torch.cuda.synchronize()

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        with_stack=True,
    ) as prof:
        for _ in range(num_active):
            _step()
        torch.cuda.synchronize()

    prof.export_chrome_trace(output_path)
    print(f"\nProfile trace saved to {output_path}")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=25))


def run_single(
    num_tokens: int,
    num_experts: int,
    top_k: int,
    hidden_size: int,
    moe_intermediate_size: int,
    include_backward: bool,
    num_warmup: int,
    num_iters: int,
    use_compile: bool,
    dtype: torch.dtype = torch.bfloat16,
    trace_path: str | None = None,
) -> BenchResult:
    device = torch.device("cuda")
    config = Qwen3VLMoeTextConfig(
        hidden_size=hidden_size,
        moe_intermediate_size=moe_intermediate_size,
        num_experts=num_experts,
        num_experts_per_tok=top_k,
        hidden_act="silu",
    )
    module = Qwen3VLMoeTextExpertsGroupedMm(config).to(device=device, dtype=dtype)
    module.init_weights(device)
    if use_compile:
        module = torch.compile(module, fullgraph=True, dynamic=True)

    hidden_states, topk_scores, expert_indices, num_tokens_per_expert = _make_inputs(num_tokens, config, device, dtype)

    torch.cuda.reset_peak_memory_stats(device)

    fwd_ms = bench_forward(
        module,
        hidden_states,
        topk_scores,
        expert_indices,
        num_tokens_per_expert,
        num_warmup=num_warmup,
        num_iters=num_iters,
    )

    bwd_ms = 0.0
    if include_backward:
        bwd_ms = bench_backward(
            module,
            hidden_states,
            topk_scores,
            expert_indices,
            num_tokens_per_expert,
            num_warmup=num_warmup,
            num_iters=num_iters,
        )

    peak_mem_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)

    if trace_path is not None:
        profile_run(
            module,
            hidden_states,
            topk_scores,
            expert_indices,
            num_tokens_per_expert,
            output_path=trace_path,
            include_backward=include_backward,
        )

    return BenchResult(
        num_tokens=num_tokens,
        num_experts=num_experts,
        top_k=top_k,
        hidden_size=hidden_size,
        moe_intermediate_size=moe_intermediate_size,
        fwd_ms=fwd_ms,
        bwd_ms=bwd_ms,
        peak_mem_mb=peak_mem_mb,
    )


def run_comparison(
    num_tokens: int,
    config: Qwen3VLMoeTextConfig,
    num_warmup: int,
    num_iters: int,
    use_compile: bool,
    dtype: torch.dtype = torch.bfloat16,
    trace_dir: str | None = None,
) -> None:
    """Run grouped_mm vs naive side-by-side and report speedup."""
    device = torch.device("cuda")

    naive = Qwen3VLMoeTextExpertsNaive(config).to(device=device, dtype=dtype)
    grouped = Qwen3VLMoeTextExpertsGroupedMm(config).to(device=device, dtype=dtype)
    naive.init_weights(device)
    grouped.load_state_dict(naive.state_dict())
    if use_compile:
        grouped = torch.compile(grouped, fullgraph=True, dynamic=False)

    hidden_states, topk_scores, expert_indices, num_tokens_per_expert = _make_inputs(num_tokens, config, device, dtype)

    naive_ms = bench_forward(
        naive,
        hidden_states,
        topk_scores,
        expert_indices,
        num_tokens_per_expert,
        num_warmup=num_warmup,
        num_iters=num_iters,
    )
    grouped_ms = bench_forward(
        grouped,
        hidden_states,
        topk_scores,
        expert_indices,
        num_tokens_per_expert,
        num_warmup=num_warmup,
        num_iters=num_iters,
    )

    with torch.no_grad():
        out_naive = naive(hidden_states, topk_scores, expert_indices, num_tokens_per_expert)
        out_grouped = grouped(hidden_states, topk_scores, expert_indices, num_tokens_per_expert)
    rel_err = (out_naive - out_grouped).norm() / out_naive.norm()

    print(f"  naive:     {naive_ms:8.3f} ms")
    print(f"  grouped:   {grouped_ms:8.3f} ms")
    print(f"  speedup:   {naive_ms / grouped_ms:8.2f}x")
    print(f"  rel error: {rel_err.item():.6e}")

    if trace_dir is not None:
        tag = (
            f"T{num_tokens}_E{config.num_experts}_K{config.num_experts_per_tok}"
            f"_H{config.hidden_size}_I{config.moe_intermediate_size}"
        )
        for name, mod in [("naive", naive), ("grouped", grouped)]:
            path = os.path.join(trace_dir, f"compare_{name}_{tag}.json")
            profile_run(
                mod,
                hidden_states,
                topk_scores,
                expert_indices,
                num_tokens_per_expert,
                output_path=path,
            )


def main(args: BenchConfig) -> None:
    dtype_map = {"bf16": torch.bfloat16, "fp32": torch.float32}
    dtype = dtype_map[args.dtype]

    profile_dir: str | None = None
    if args.profile:
        profile_dir = args.profile_dir
        Path(profile_dir).mkdir(parents=True, exist_ok=True)

    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu_name}")
    print(f"dtype: {args.dtype}, compile: {args.compile}")
    print(f"warmup: {args.num_warmup}, iters: {args.num_iters}")
    if profile_dir:
        print(f"profile dir: {profile_dir}")
    print()

    if args.compare:
        for num_tokens, num_experts, top_k, hidden_size, moe_intermediate_size in itertools.product(
            args.num_tokens,
            args.num_experts,
            args.top_k,
            args.hidden_size,
            args.moe_intermediate_size,
        ):
            config = Qwen3VLMoeTextConfig(
                hidden_size=hidden_size,
                moe_intermediate_size=moe_intermediate_size,
                num_experts=num_experts,
                num_experts_per_tok=top_k,
                hidden_act="silu",
            )
            header = (
                f"tokens={num_tokens}  experts={num_experts}  top_k={top_k}  "
                f"hidden={hidden_size}  intermediate={moe_intermediate_size}"
            )
            print(header)
            run_comparison(
                num_tokens=num_tokens,
                config=config,
                num_warmup=args.num_warmup,
                num_iters=args.num_iters,
                use_compile=args.compile,
                dtype=dtype,
                trace_dir=profile_dir,
            )
            print()
        return

    header = (
        f"{'tokens':>8}  {'experts':>7}  {'top_k':>5}  {'hidden':>6}  "
        f"{'interm':>6}  {'fwd_ms':>8}  {'bwd_ms':>8}  {'peak_MB':>9}"
    )
    print(header)
    print("-" * len(header))

    for num_tokens, num_experts, top_k, hidden_size, moe_intermediate_size in itertools.product(
        args.num_tokens,
        args.num_experts,
        args.top_k,
        args.hidden_size,
        args.moe_intermediate_size,
    ):
        trace_path = None
        if profile_dir is not None:
            mode = "fwd_bwd" if args.backward else "fwd"
            tag = f"T{num_tokens}_E{num_experts}_K{top_k}_H{hidden_size}_I{moe_intermediate_size}"
            trace_path = os.path.join(profile_dir, f"{mode}_{tag}.json")
        result = run_single(
            num_tokens=num_tokens,
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            moe_intermediate_size=moe_intermediate_size,
            include_backward=args.backward,
            num_warmup=args.num_warmup,
            num_iters=args.num_iters,
            use_compile=args.compile,
            dtype=dtype,
            trace_path=trace_path,
        )
        bwd_str = f"{result.bwd_ms:8.3f}" if args.backward else "     N/A"
        print(
            f"{result.num_tokens:>8}  {result.num_experts:>7}  {result.top_k:>5}  "
            f"{result.hidden_size:>6}  {result.moe_intermediate_size:>6}  "
            f"{result.fwd_ms:>8.3f}  {bwd_str}  {result.peak_mem_mb:>9.1f}"
        )


if __name__ == "__main__":
    main(tyro.cli(BenchConfig))
