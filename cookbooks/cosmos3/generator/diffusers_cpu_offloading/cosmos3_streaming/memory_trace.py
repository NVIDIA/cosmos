from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch


class CudaMemoryTrace:
    """Interval and cumulative CUDA-memory telemetry for one pipeline request.

    Each checkpoint closes the interval that began at the preceding checkpoint.
    PyTorch's allocator peak is reset after the sample, so ``interval_peak_*``
    describes that interval rather than the whole process. ``device_used_*``
    comes from the CUDA driver and therefore includes CUDA context and non-PyTorch
    allocations as well as this process's allocator state.
    """

    def __init__(self, path: str | Path, device: torch.device, *, synchronize: bool = False) -> None:
        self.path = Path(path)
        self.device = device
        self.synchronize = synchronize
        self._started_at = 0.0
        self._sequence = 0
        self._global_peak_allocated = 0
        self._global_peak_reserved = 0
        self._global_peak_device_used = 0
        self._active = False

    def begin(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("")
        self._started_at = time.perf_counter()
        self._sequence = 0
        self._global_peak_allocated = 0
        self._global_peak_reserved = 0
        self._global_peak_device_used = 0
        self._active = True
        if self.device.type == "cuda":
            with torch.cuda.device(self.device):
                torch.cuda.reset_peak_memory_stats()
        self.checkpoint("request.begin")

    def checkpoint(self, event: str, **fields: Any) -> None:
        if not self._active:
            return

        sample: dict[str, Any] = {
            "sequence": self._sequence,
            "elapsed_seconds": time.perf_counter() - self._started_at,
            "event": event,
            **fields,
        }
        if self.device.type == "cuda":
            if self.synchronize:
                torch.cuda.synchronize(self.device)
            with torch.cuda.device(self.device):
                allocated = torch.cuda.memory_allocated()
                reserved = torch.cuda.memory_reserved()
                interval_peak_allocated = torch.cuda.max_memory_allocated()
                interval_peak_reserved = torch.cuda.max_memory_reserved()
                free, total = torch.cuda.mem_get_info()
                device_used = total - free
                torch.cuda.reset_peak_memory_stats()

            self._global_peak_allocated = max(self._global_peak_allocated, interval_peak_allocated)
            self._global_peak_reserved = max(self._global_peak_reserved, interval_peak_reserved)
            self._global_peak_device_used = max(self._global_peak_device_used, device_used)
            sample.update(
                {
                    "allocated_bytes": allocated,
                    "reserved_bytes": reserved,
                    "interval_peak_allocated_bytes": interval_peak_allocated,
                    "interval_peak_reserved_bytes": interval_peak_reserved,
                    "device_used_bytes": device_used,
                    "device_total_bytes": total,
                    "global_peak_allocated_bytes": self._global_peak_allocated,
                    "global_peak_reserved_bytes": self._global_peak_reserved,
                    "global_peak_device_used_bytes": self._global_peak_device_used,
                }
            )

        with self.path.open("a") as handle:
            handle.write(json.dumps(sample, sort_keys=True) + "\n")
        self._sequence += 1

    def end(self) -> None:
        if not self._active:
            return
        self.checkpoint("request.end")
        self._active = False
