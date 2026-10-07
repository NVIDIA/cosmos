from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(frozen=True, slots=True)
class StreamingConfig:
    """Runtime policy for Cosmos 3 split execution.

    The model weights remain in their checkpoint dtype. ``device`` and
    ``offload_device`` control placement only; this class does not enable model
    weight quantization.
    """

    device: str = "cuda"
    offload_device: str = "cpu"
    enable_group_offload: bool = True
    use_stream: bool = True
    record_stream: bool = False
    low_cpu_mem_usage: bool = True
    vae_leaf_offload: bool = True
    memory_profile: Literal["auto", "performance", "low_memory"] = "auto"
    attention_backend: str = "cosmos3_streaming.attention.sdpa:TorchSdpaBackend"
    attention_backend_kwargs: dict[str, Any] = field(default_factory=dict)
    empty_cuda_cache_on_end: bool = False
    memory_trace_path: str | None = None
    memory_trace_blocks: bool = False
    memory_trace_synchronize: bool = False

    def __post_init__(self) -> None:
        if not self.device:
            raise ValueError("device must be non-empty")
        if self.offload_device != "cpu":
            raise ValueError("the first implementation only supports CPU weight storage")
        if self.record_stream and not self.use_stream:
            raise ValueError("record_stream=True requires use_stream=True")
        if ":" not in self.attention_backend:
            raise ValueError("attention_backend must use 'module.path:ClassName' syntax")
        if self.memory_trace_blocks and self.memory_trace_path is None:
            raise ValueError("memory_trace_blocks=True requires memory_trace_path")
        if self.memory_profile not in ("auto", "performance", "low_memory"):
            raise ValueError(f"unsupported memory profile: {self.memory_profile}")
