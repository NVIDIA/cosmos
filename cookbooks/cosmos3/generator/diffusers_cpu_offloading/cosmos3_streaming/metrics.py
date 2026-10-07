from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class RuntimeStats:
    """Low-overhead counters for one completed or active request.

    Byte counters cover decoder-block parameters only. CUDA/Nsight profiling
    remains the source of truth for physical PCIe traffic.
    """

    understanding_prefills: int = 0
    denoising_calls: int = 0
    understanding_weight_bytes: int = 0
    generation_weight_bytes: int = 0
    estimated_block_h2d_bytes: int = 0

    @property
    def estimated_block_h2d_gib(self) -> float:
        return self.estimated_block_h2d_bytes / 2**30
