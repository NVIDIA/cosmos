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

"""Lightweight CPU memory-profiling helpers.

Only depends on ``os``, ``sys``, and ``psutil`` so it can be imported safely
from dataset modules without pulling in heavy dependencies.

Enable per-stage logging by setting the ``MEMORY_PROFILE`` env var::

    MEMORY_PROFILE=1 torchrun ...
"""

import contextlib
import gc
import logging
import os
import sys
from collections.abc import Callable, Iterator

import psutil

_log = logging.getLogger(__name__)


def memprofile_enabled() -> bool:
    """Return ``True`` when the ``MEMORY_PROFILE`` env var is truthy."""
    return os.environ.get("MEMORY_PROFILE", "").strip() not in ("", "0", "false")


def fmt_mb(mb: float) -> str:
    """Format a MiB value as a human-readable string (MiB or GiB)."""
    if mb >= 1024:
        return f"{mb / 1024:.2f} GiB"
    return f"{mb:.1f} MiB"


@contextlib.contextmanager
def rss_tracker(
    label: str,
    *,
    enabled: bool | None = None,
    extras_fn: Callable[[], list[str]] | None = None,
    after_fn: Callable[[], None] | None = None,
) -> Iterator[None]:
    """Track RSS delta across a block.  No-op when profiling is disabled.

    When *enabled* is ``False`` (or ``None`` and ``MEMORY_PROFILE`` is unset)
    the context manager yields immediately with zero overhead -- no
    ``gc.collect()`` and no ``psutil`` calls.

    Args:
        label: Human-readable description included in the log line.
        enabled: Explicit toggle.  When ``None``, falls back to
            ``memprofile_enabled()`` (i.e. the ``MEMORY_PROFILE`` env var).
        extras_fn: Optional callback invoked *after* the measured block.
            Each returned string is logged as a supplementary detail line.
        after_fn: Optional side-effect callback invoked after logging.
            Use for actions that should only run when profiling is active
            (e.g. detailed worker memory breakdowns).
    """
    if enabled is None:
        enabled = memprofile_enabled()
    if not enabled:
        yield
        return
    gc.collect()
    rss_before = get_rss_mb()
    yield
    gc.collect()
    rss_after = get_rss_mb()
    _log.debug(
        "[MEMPROFILE] %s | RSS: %s (delta: +%s)",
        label,
        fmt_mb(rss_after),
        fmt_mb(rss_after - rss_before),
    )
    if extras_fn is not None:
        for line in extras_fn():
            _log.debug("[MEMPROFILE]   %s", line)
    if after_fn is not None:
        after_fn()


def get_rss_mb() -> float:
    """Return the current process RSS in MiB."""
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


def get_process_tree_rss_mb() -> float:
    """Return RSS of the current process + all children in MiB."""
    proc = psutil.Process(os.getpid())
    total = proc.memory_info().rss
    for child in proc.children(recursive=True):
        try:
            total += child.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return total / (1024 * 1024)


def get_worker_memory_breakdown() -> list[tuple[int, float]]:
    """Return a list of ``(pid, rss_mib)`` for each child process."""
    proc = psutil.Process(os.getpid())
    result: list[tuple[int, float]] = []
    for child in proc.children(recursive=True):
        try:
            rss_mb = child.memory_info().rss / (1024 * 1024)
            result.append((child.pid, rss_mb))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return result


def get_worker_memory_detailed() -> list[dict[str, float]]:
    """Return RSS, USS (Unique Set Size), and PSS for each child process.

    USS is the memory *unique* to a process -- not shared with any other.
    It directly measures CoW-duplicated pages plus worker-only allocations.

    PSS counts shared pages proportionally (shared_page / num_sharers).

    Returns list of dicts with keys: ``pid``, ``rss``, ``uss``, ``pss`` (all in MiB).
    Falls back to RSS-only if ``memory_full_info()`` is unavailable.
    """
    proc = psutil.Process(os.getpid())
    result: list[dict[str, float]] = []
    for child in proc.children(recursive=True):
        try:
            full = child.memory_full_info()
            result.append(
                {
                    "pid": float(child.pid),
                    "rss": full.rss / (1024 * 1024),
                    "uss": full.uss / (1024 * 1024),
                    "pss": full.pss / (1024 * 1024),
                }
            )
        except (psutil.NoSuchProcess, psutil.AccessDenied, AttributeError):
            try:
                rss_mb = child.memory_info().rss / (1024 * 1024)
                result.append(
                    {
                        "pid": float(child.pid),
                        "rss": rss_mb,
                        "uss": -1.0,
                        "pss": -1.0,
                    }
                )
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
    return result


def get_uss_mb() -> float:
    """Return USS (Unique Set Size) of the current process in MiB.

    Falls back to RSS if ``memory_full_info()`` is unavailable.
    """
    proc = psutil.Process(os.getpid())
    try:
        return proc.memory_full_info().uss / (1024 * 1024)
    except (AttributeError, psutil.AccessDenied):
        return proc.memory_info().rss / (1024 * 1024)


def log_worker_memory_breakdown(dataset: object) -> None:
    """Log a detailed memory breakdown from inside a dataloader worker.

    Designed to be called periodically from ``__getitem__`` when
    ``MEMORY_PROFILE=1``.  Inspects the dataset's internal state to
    report how many ``LeRobotDataset`` instances are loaded, HuggingFace
    Arrow table sizes, and the LeRobot ``VideoDecoderCache`` size.

    Args:
        dataset: A ``BaseActionLeRobotDataset`` instance (or compatible).
    """
    import gc
    import logging

    pid = os.getpid()
    rss = get_rss_mb()
    uss = get_uss_mb()
    logger = logging.getLogger(f"memprofile.worker.{pid}")

    logger.warning(f"[WORKER {pid}] RSS={fmt_mb(rss)} USS={fmt_mb(uss)}")

    # --- LeRobotDataset instances ---
    datasets_list = getattr(dataset, "_datasets", [])
    loaded_count = sum(1 for ds in datasets_list if ds is not None)
    total_count = len(datasets_list)
    logger.warning(f"[WORKER {pid}]   LeRobotDataset: {loaded_count}/{total_count} loaded")

    total_arrow_bytes = 0
    total_hf_rows = 0
    for i, ds in enumerate(datasets_list):
        if ds is None:
            continue
        hf_ds = getattr(ds, "hf_dataset", None)
        if hf_ds is None:
            logger.warning(f"[WORKER {pid}]     ds[{i}]: hf_dataset not yet loaded")
            continue

        num_rows = len(hf_ds)
        total_hf_rows += num_rows

        arrow_bytes = 0
        data_table = getattr(hf_ds, "_data", None)
        if data_table is not None and hasattr(data_table, "nbytes"):
            arrow_bytes = data_table.nbytes
            total_arrow_bytes += arrow_bytes

        logger.warning(f"[WORKER {pid}]     ds[{i}]: rows={num_rows}, arrow={fmt_mb(arrow_bytes / (1024 * 1024))}")

    if loaded_count > 0:
        logger.warning(
            f"[WORKER {pid}]   Total HF rows={total_hf_rows}, total arrow={fmt_mb(total_arrow_bytes / (1024 * 1024))}"
        )

    # --- VideoDecoderCache ---
    try:
        from lerobot.datasets.video_utils import _default_decoder_cache

        cache_size = _default_decoder_cache.size()
        logger.warning(f"[WORKER {pid}]   VideoDecoderCache entries: {cache_size}")
    except Exception:
        pass

    # --- GC stats ---
    gc_counts = gc.get_count()
    all_objects = len(gc.get_objects())
    logger.warning(f"[WORKER {pid}]   GC counts={gc_counts}, tracked objects={all_objects}")


def deep_size(obj: object, seen: set | None = None) -> int:
    """Approximate deep memory size in bytes for nested Python containers.

    Recursively walks ``dict``, ``list``, ``tuple``, ``set``, and ``frozenset``.
    Does **not** follow arbitrary object attributes.
    """
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    seen.add(obj_id)
    size = sys.getsizeof(obj)
    if isinstance(obj, dict):
        for k, v in obj.items():
            size += deep_size(k, seen) + deep_size(v, seen)
    elif isinstance(obj, (list, tuple, set, frozenset)):
        for item in obj:
            size += deep_size(item, seen)
    return size
