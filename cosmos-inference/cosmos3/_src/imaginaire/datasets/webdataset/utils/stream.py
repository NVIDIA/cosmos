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


# PBSS
import atexit
import json
import os
import sys
import threading
import time
import weakref
from dataclasses import dataclass, field
from http.client import IncompleteRead
from pathlib import Path
from typing import Optional

import boto3
from botocore.exceptions import (
    ClientError,
    ConnectionClosedError,
    EndpointConnectionError,
    ResponseStreamingError,
)
from botocore.exceptions import (
    ReadTimeoutError as BotocoreReadTimeoutError,
)
from urllib3.exceptions import ProtocolError as URLLib3ProtocolError
from urllib3.exceptions import ReadTimeoutError as URLLib3ReadTimeoutError
from urllib3.exceptions import SSLError as URLLib3SSLError

from cosmos3._src.imaginaire.utils import log

# Public API - only these should be imported from this module
__all__ = [
    "RetryingStream",  # Main class for S3 streaming with retries
    "ENABLE_RETRY_STATS",  # Flag to enable/disable statistics (used in tests/benchmarks)
    "RETRY_STATS_LOG_INTERVAL",  # Interval in seconds between periodic statistics logs
    "ENABLE_THROUGHPUT_STATS",  # Flag to enable/disable throughput statistics
    "THROUGHPUT_STATS_LOG_INTERVAL",  # Interval between periodic throughput statistics logs
    "ENABLE_STREAM_WANDB",  # Flag to enable/disable IPC file writes for wandb metrics
    "WATCHDOG_ENABLED",  # Enable/disable watchdog reconnects
    "WATCHDOG_MIN_THROUGHPUT_MBPS",  # Minimum throughput (MB/s) before watchdog reconnects
    "RETRYABLE_EXCEPTIONS",  # Tuple of exceptions that trigger retries
    "collect_throughput_ipc_stats",  # Main-process reader for worker throughput IPC files, for wandb logging
]

# Flag to enable/disable statistics gathering (for performance testing)
# Set to False to disable all statistics overhead for maximum performance.
# When disabled, no thread-local tracking occurs and no logs are generated.
#
# Usage for benchmarking:
#   import cosmos3._src.imaginaire.datasets.webdataset.utils.stream as stream_module
#   stream_module.ENABLE_RETRY_STATS = False  # Disable stats
#   # ... run benchmark ...
#   stream_module.ENABLE_RETRY_STATS = True   # Re-enable
ENABLE_RETRY_STATS = False

# Interval in seconds between periodic retry statistics logs
# Default is 300 seconds (5 minutes). Set to a lower value for more frequent logging
# or a higher value to reduce log verbosity.
#
# Usage:
#   import cosmos3._src.imaginaire.datasets.webdataset.utils.stream as stream_module
#   stream_module.RETRY_STATS_LOG_INTERVAL = 600  # Log every 10 minutes
RETRY_STATS_LOG_INTERVAL = 300.0  # 5 minutes

# Flag to enable/disable throughput log messages (verbose per-worker logs).
# Does NOT affect IPC file writes (controlled by ENABLE_STREAM_WANDB).
#
# Usage:
#   import cosmos3._src.imaginaire.datasets.webdataset.utils.stream as stream_module
#   stream_module.ENABLE_THROUGHPUT_STATS = False
ENABLE_THROUGHPUT_STATS = False

# Interval in seconds between periodic throughput statistics logs.
# Independent from retry stats log interval.
#
# Usage:
#   import cosmos3._src.imaginaire.datasets.webdataset.utils.stream as stream_module
#   stream_module.THROUGHPUT_STATS_LOG_INTERVAL = 600  # Log every 10 minutes
THROUGHPUT_STATS_LOG_INTERVAL = 300.0  # 5 minutes

# Enable/disable IPC file writes for cross-worker metrics aggregation (wandb).
# This controls whether workers write cumulative stats to /tmp/throughput_stats/
# for the main process to collect and log to wandb. Independent from verbose
# log messages (ENABLE_THROUGHPUT_STATS).
#
# Env var: export ENABLE_STREAM_WANDB=0  (to disable; default enabled)
ENABLE_STREAM_WANDB = os.environ.get("ENABLE_STREAM_WANDB", "1") != "0"


# Enable/disable the throughput watchdog (reconnects on sustained low throughput).
# Default: True (enabled)
#
# Env var: export RETRYING_STREAM_WATCHDOG=0
# Python:  import cosmos3._src.imaginaire.datasets.webdataset.utils.stream as stream_module
#          stream_module.WATCHDOG_ENABLED = False
WATCHDOG_ENABLED = os.environ.get("RETRYING_STREAM_WATCHDOG", "1") != "0"

# Minimum throughput (MB/s) before the watchdog triggers a reconnect.
# Default: 10.0 MB/s
#
# Env var: export RETRYING_STREAM_WATCHDOG_MIN_MBPS=50.0
# Python:  import cosmos3._src.imaginaire.datasets.webdataset.utils.stream as stream_module
#          stream_module.WATCHDOG_MIN_THROUGHPUT_MBPS = 50.0
WATCHDOG_MIN_THROUGHPUT_MBPS = float(os.environ.get("RETRYING_STREAM_WATCHDOG_MIN_MBPS", "50.0"))


@dataclass
class GlobalRetryStatistics:
    """Per-process statistics aggregator for S3 retry operations.

    Aggregates statistics across all threads within this process (e.g., a DataLoader worker).
    Each process maintains its own independent statistics - no cross-process communication.
    In distributed training with DataLoader workers:
    - Each rank's main process has its own _global_retry_stats instance
    - Each DataLoader worker process (spawned via multiprocessing) has its own instance
    - Statistics are isolated per-process, ensuring accurate tracking without interference

    Uses WeakSet to track active instances - automatically handles cleanup
    even if threads die or exceptions occur during construction.

    Tracks both per-thread and cumulative statistics:
    - registered_threads: Per-thread counters (including PID and thread ID) for detailed breakdown
    - cumulative_*: Process-local cumulative counters (never reset, for atexit log)

    Statistics terminology:
    - operations_started: Number of S3 operations initiated (read/get_length/get_stream calls)
    - failed_operations: Operations that failed at least once and required retry
    - total_attempts: Sum of all attempts (initial + retries)

    Note: operations_started counts how many operations we started (each gets 1 initial attempt).
          total_attempts >= operations_started because failed operations retry multiple times.

    Thread safety:
    - Per-thread counters are lock-free (threading.local() ensures isolation)
    - Cumulative counters use a lock because += is not atomic in Python
    - Lock overhead is negligible (< 0.1% from benchmarks)
    - Single-threaded case: Lock acquisition is uncontended (instant, no blocking)
    - Multi-threaded case: Lock contention is minimal (only during retries, which are rare)
    """

    registered_threads: dict[int, dict[str, int]] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)  # Protects cumulative counters
    last_log_time: float = field(default_factory=time.time)
    active_instances: weakref.WeakSet = field(default_factory=weakref.WeakSet)  # Tracks active RetryingStream instances
    rank: int | None = None  # Lazily initialized rank ID (None = not yet initialized)
    pid: int | None = None  # Lazily initialized process ID (None = not yet initialized, cached to avoid OS calls)
    registered_pids: set[int] = field(
        default_factory=set
    )  # PIDs that have registered atexit handlers (for multiprocessing support)

    # Cumulative counters (never reset, for atexit final log)
    # These require the lock because += is not atomic (3 bytecode operations: LOAD, ADD, STORE)
    # Without the lock, concurrent increments can cause lost updates
    cumulative_operations_started: int = 0  # Number of operations initiated
    cumulative_failed_operations: int = 0  # Operations that failed and required retry
    cumulative_attempts: int = 0  # Sum of all attempts (initial + retries)

    def get_rank(self) -> int:
        """Get rank with lazy initialization.

        Rank is captured on first access, not at module import time.
        This ensures torch.distributed is initialized before we try to read it.

        Falls back to RANK environment variable if torch.distributed is not initialized.
        This handles DataLoader worker processes which inherit RANK from parent but
        don't have torch.distributed initialized.

        Returns:
            The rank ID (0 if distributed not available/initialized and no RANK env var)
        """
        if self.rank is None:
            try:
                import torch.distributed as dist

                if dist.is_available() and dist.is_initialized():
                    self.rank = dist.get_rank()
                else:
                    # Fallback to RANK environment variable (for DataLoader workers)
                    self.rank = int(os.environ.get("RANK", "0"))
            except Exception:
                self.rank = 0  # Fallback if distributed not available
        return self.rank

    def get_pid(self) -> int:
        """Get process ID with lazy caching (avoids repeated OS calls).

        Returns:
            The current process ID (cached after first call)
        """
        if self.pid is None:
            self.pid = os.getpid()
        return self.pid


# Conditionally create per-process statistics objects only if stats are enabled
# Each process maintains independent statistics (no cross-process communication)
if ENABLE_RETRY_STATS:
    _global_retry_stats = GlobalRetryStatistics()
    _thread_local_stats = threading.local()

    # Rank is lazily initialized on first access via get_rank()
    # This ensures torch.distributed is initialized before we try to read it

    # Note: atexit handler registration is now done lazily per-process in _get_thread_stats()
    # This ensures each DataLoader worker process (spawned via multiprocessing) automatically
    # registers its own atexit handler, making multiprocessing support transparent
else:
    _global_retry_stats = None  # type: ignore
    _thread_local_stats = None  # type: ignore


@dataclass
class GlobalThroughputStatistics:
    """Per-process statistics aggregator for shard throughput and watchdog events.

    Same aggregation pattern as GlobalRetryStatistics: each DataLoader worker process
    maintains its own independent instance. No cross-process communication.

    Periodic logs show interval values (delta since last log). The atexit final
    log shows lifetime cumulative totals. IPC files carry cumulative snapshots
    (deltas computed in collect_throughput_ipc_stats).

    Thread safety: Same lock pattern as GlobalRetryStatistics
    """

    lock: threading.Lock = field(default_factory=threading.Lock)
    last_log_time: float = field(default_factory=time.time)
    rank: int | None = None
    pid: int | None = None
    registered_pids: set[int] = field(default_factory=set)

    cumulative_bytes_read: int = 0  # log.warning + wandb (via IPC)
    cumulative_total_read_time: float = 0.0  # log.warning + wandb (via IPC)
    cumulative_watchdog_reconnects: int = 0  # log.warning + wandb (via IPC)

    _prev_bytes: int = 0
    _prev_read_time: float = 0.0
    _prev_watchdog: int = 0

    def get_rank(self) -> int:
        """Get rank with lazy initialization (same logic as GlobalRetryStatistics.get_rank)."""
        if self.rank is None:
            try:
                import torch.distributed as dist

                if dist.is_available() and dist.is_initialized():
                    self.rank = dist.get_rank()
                else:
                    self.rank = int(os.environ.get("RANK", "0"))
            except Exception:
                self.rank = 0
        return self.rank

    def get_pid(self) -> int:
        """Get process ID with lazy caching (avoids repeated OS calls)."""
        if self.pid is None:
            self.pid = os.getpid()
        return self.pid


if ENABLE_THROUGHPUT_STATS or ENABLE_STREAM_WANDB:
    _global_throughput_stats = GlobalThroughputStatistics()
else:
    _global_throughput_stats = None  # type: ignore

# Exceptions that should trigger retries for S3 streaming operations
RETRYABLE_EXCEPTIONS = (
    URLLib3ReadTimeoutError,
    URLLib3ProtocolError,
    URLLib3SSLError,
    IncompleteRead,
    IOError,
    ResponseStreamingError,
    ConnectionClosedError,
    BotocoreReadTimeoutError,
)


def _get_thread_stats() -> dict[str, int]:
    """Get or initialize thread-local statistics (lock-free).

    Lazily registers atexit handler on first call per process, making multiprocessing
    support transparent (each DataLoader worker process automatically gets its own handler).

    Performance optimizations:
    - PID cached in GlobalRetryStatistics.pid (avoids repeated os.getpid() syscalls)
    - Thread ID cached in thread-local counters (constant per thread lifetime)
    - Rank cached in GlobalRetryStatistics.rank (avoids repeated torch.distributed calls)

    Returns:
        Dictionary with thread-local counters for operations and retries.
        Returns empty dict if statistics are disabled.
    """
    # No-op if statistics are disabled (for performance)
    if not ENABLE_RETRY_STATS or _thread_local_stats is None:
        return {}  # Return empty dict as no-op

    if not hasattr(_thread_local_stats, "counters"):
        # Cache PID and thread ID (constant for the lifetime of this thread)
        pid = _global_retry_stats.get_pid()  # Cached to avoid repeated os.getpid() syscalls
        thread_id = threading.get_ident()  # Already fast, but cached for consistency

        counters = {
            "pid": pid,  # Process ID (distinguishes DataLoader workers)
            "thread_id": thread_id,  # Thread ID within this process
            "operations_started": 0,  # Number of S3 operations initiated (read/get_length/get_stream)
            "failed_operations": 0,  # Operations that failed at least once and required retry
            "total_attempts": 0,  # Sum of all attempts (initial + retries)
        }
        _thread_local_stats.counters = counters

        # Register this thread's stats for aggregation (only once per thread)
        with _global_retry_stats.lock:
            _global_retry_stats.registered_threads[thread_id] = counters

            # Lazily register atexit handler once per process (not per thread)
            # This provides best-effort final statistics logging when processes exit normally.
            # Note: atexit is unreliable in multiprocessing.Process (known Python limitation),
            # so tests/critical paths should explicitly call _log_retry_stats_internal(force=True).
            if pid not in _global_retry_stats.registered_pids:
                _global_retry_stats.registered_pids.add(pid)

                # Register atexit handler with proper error handling
                def _atexit_handler():
                    try:
                        if _global_retry_stats:
                            _log_retry_stats_internal(force=True)
                            # Flush output to ensure atexit logs are captured
                            try:
                                sys.stdout.flush()
                                sys.stderr.flush()
                            except Exception:
                                pass
                    except Exception as e:
                        # Fallback: try to print error if logging infrastructure is torn down
                        try:
                            print(f"[PID {os.getpid()}] atexit handler error: {e}", flush=True)
                        except Exception:
                            pass  # Silently fail if stdout is closed

                atexit.register(_atexit_handler)

    return _thread_local_stats.counters


def _log_retry_stats_internal(force: bool = False) -> None:
    """Internal function to log retry statistics with per-thread breakdown and process-local totals.

    Statistics are aggregated across all threads within this process only.
    Each process logs independently - no cross-process communication for zero overhead.

    Args:
        force: If True, log cumulative lifetime stats (for atexit).
               If False, log periodic snapshot of current stats (counters keep accumulating).
    """
    # No-op if statistics are disabled (for performance)
    if not ENABLE_RETRY_STATS or _global_retry_stats is None:
        return

    current_time = time.time()

    # Quick check without lock (small race condition is acceptable here)
    if not force and current_time - _global_retry_stats.last_log_time < RETRY_STATS_LOG_INTERVAL:
        return

    # Now acquire lock to read stats
    with _global_retry_stats.lock:
        # Double-check pattern for periodic logs (skip if time hasn't elapsed)
        if not force and current_time - _global_retry_stats.last_log_time < RETRY_STATS_LOG_INTERVAL:
            return

        # Get cumulative stats (for final log) or aggregate per-thread stats (for periodic)
        if force:
            # Final log: use cumulative counters (guaranteed monotonic)
            total_ops = _global_retry_stats.cumulative_operations_started
            failed_ops = _global_retry_stats.cumulative_failed_operations
            total_attempts = _global_retry_stats.cumulative_attempts
            per_thread_stats = None  # Not needed for final log
        else:
            # Periodic log: aggregate per-thread stats (snapshot, not cumulative)
            # Note: We track per-thread stats internally for correctness (handles rare multi-threaded
            # cases and ensures accurate aggregation), but only log the per-process cumulative totals.
            # In typical usage, each DataLoader worker process has a single thread doing I/O.
            per_thread_stats = {}
            total_ops = 0  # Total operations started across all threads
            failed_ops = 0  # Failed operations across all threads
            total_attempts = 0  # Total attempts across all threads

            for thread_id, thread_stats in _global_retry_stats.registered_threads.items():
                pid = thread_stats["pid"]  # Process ID (identifies DataLoader worker)
                ops = thread_stats["operations_started"]  # S3 operations started in this thread
                failed = thread_stats["failed_operations"]  # Operations that failed in this thread
                attempts = thread_stats["total_attempts"]  # All attempts (initial + retries) in this thread

                per_thread_stats[thread_id] = {
                    "pid": pid,
                    "thread_id": thread_id,
                    "operations_started": ops,
                    "failed_operations": failed,
                    "total_attempts": attempts,
                }

                # Aggregate across all threads
                total_ops += ops
                failed_ops += failed
                total_attempts += attempts

        if total_ops > 0:
            failure_percentage = (failed_ops / total_ops) * 100
            avg_attempts_per_op = total_attempts / total_ops

            prefix = "[RetryingStream Stats - Final]" if force else "[RetryingStream Stats]"
            # Include rank and PID in message (lazily cached to avoid repeated OS calls)
            rank = _global_retry_stats.get_rank()
            pid = _global_retry_stats.get_pid()
            message = (
                f"{prefix} [Rank {rank}] [PID {pid}] PROCESS-LOCAL: {total_ops} total operations, "
                f"{failed_ops} failed operations ({failure_percentage:.1f}%), "
                f"avg {avg_attempts_per_op:.2f} attempts/operation"
            )

            # Always use logging infrastructure (with fallback for atexit edge cases)
            try:
                # Only log the cumulative per-process summary
                # (Per-thread stats are still tracked internally for accuracy, just not printed)
                log.warning(message, rank0_only=False)
            except Exception:
                # Fallback to print if logging is torn down (rare edge case during atexit)
                try:
                    print(f"WARNING: {message}", flush=True)
                except Exception:
                    pass  # Silently fail if stdout is also closed (multiprocessing edge case)

            # Update last log time (only for periodic logs, not final)
            if not force:
                _global_retry_stats.last_log_time = current_time


def _maybe_log_retry_stats() -> None:
    """Log process-local retry statistics if RETRY_STATS_LOG_INTERVAL seconds have elapsed since last log.

    Each process logs independently - no cross-process communication.
    The log interval is configurable via the RETRY_STATS_LOG_INTERVAL module variable (default: 300 seconds).
    """
    if not ENABLE_RETRY_STATS:
        return
    _log_retry_stats_internal(force=False)


# Throughput statistics helpers (mirrors the retry statistics helpers above)


def _register_throughput_atexit() -> None:
    """Lazily register atexit handler once per process for final throughput log and IPC flush."""
    if _global_throughput_stats is None:
        return

    pid = _global_throughput_stats.get_pid()
    with _global_throughput_stats.lock:
        if pid not in _global_throughput_stats.registered_pids:
            _global_throughput_stats.registered_pids.add(pid)

            def _atexit_handler() -> None:
                try:
                    if _global_throughput_stats:
                        _log_throughput_stats_internal(force=True)
                        try:
                            sys.stdout.flush()
                            sys.stderr.flush()
                        except Exception:
                            pass
                except Exception as e:
                    try:
                        print(f"[PID {os.getpid()}] throughput atexit error: {e}", flush=True)
                    except Exception:
                        pass

            atexit.register(_atexit_handler)


def _log_throughput_stats_internal(force: bool = False) -> None:
    """Log throughput statistics with process-local totals and write IPC files.

    - force=False: periodic log (if ENABLE_THROUGHPUT_STATS) + IPC write (if ENABLE_STREAM_WANDB)
    - force=True: cumulative lifetime stats for atexit + final IPC write
    """
    if _global_throughput_stats is None:
        return

    current_time = time.time()

    if not force and current_time - _global_throughput_stats.last_log_time < THROUGHPUT_STATS_LOG_INTERVAL:
        return

    with _global_throughput_stats.lock:
        if not force and current_time - _global_throughput_stats.last_log_time < THROUGHPUT_STATS_LOG_INTERVAL:
            return

        s = _global_throughput_stats
        if force:
            bytes_read = s.cumulative_bytes_read
            read_time = s.cumulative_total_read_time
            watchdog = s.cumulative_watchdog_reconnects
        else:
            bytes_read = s.cumulative_bytes_read - s._prev_bytes
            read_time = s.cumulative_total_read_time - s._prev_read_time
            watchdog = s.cumulative_watchdog_reconnects - s._prev_watchdog

            s._prev_bytes = s.cumulative_bytes_read
            s._prev_read_time = s.cumulative_total_read_time
            s._prev_watchdog = s.cumulative_watchdog_reconnects

        if bytes_read > 0:
            if ENABLE_THROUGHPUT_STATS:
                mb_read = bytes_read / (1024**2)
                avg_mbps = mb_read / read_time if read_time > 0 else 0

                prefix = "[Throughput Stats - Final]" if force else "[Throughput Stats]"
                rank = _global_throughput_stats.get_rank()
                pid = _global_throughput_stats.get_pid()
                watchdog_part = f", {watchdog} watchdog reconnects" if WATCHDOG_ENABLED else ""
                message = (
                    f"{prefix} [Rank {rank}] [PID {pid}] PROCESS-LOCAL: "
                    f"{mb_read:.2f}MB in {read_time:.3f}s "
                    f"({avg_mbps:.1f}MB/s avg){watchdog_part}"
                )

                try:
                    log.warning(message, rank0_only=False)
                except Exception:
                    try:
                        print(f"WARNING: {message}", flush=True)
                    except Exception:
                        pass

            if ENABLE_STREAM_WANDB:
                _write_throughput_ipc()

            if not force:
                _global_throughput_stats.last_log_time = current_time


def _maybe_log_throughput_stats() -> None:
    """Log process-local throughput statistics and/or write IPC files if interval has elapsed."""
    if not ENABLE_THROUGHPUT_STATS and not ENABLE_STREAM_WANDB:
        return
    _log_throughput_stats_internal(force=False)


def _write_throughput_ipc() -> None:
    """Write cumulative throughput snapshot to a per-worker IPC file (for wandb logging)."""
    if _global_throughput_stats is None:
        return
    try:
        s = _global_throughput_stats
        rank = s.get_rank()
        ipc_dir = Path(f"/tmp/throughput_stats/rank_{rank}")
        ipc_dir.mkdir(parents=True, exist_ok=True)
        filepath = ipc_dir / f"worker_{os.getpid()}.json"
        tmp = filepath.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(
                {
                    "bytes": s.cumulative_bytes_read,
                    "read_time": s.cumulative_total_read_time,
                    "watchdog": s.cumulative_watchdog_reconnects,
                    "ts": time.time(),
                },
                f,
            )
        tmp.rename(filepath)
    except Exception:
        pass


def _is_pid_alive(pid: int) -> bool:
    """Check if a process with the given PID is still running (zero-cost signal 0)."""
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


_ipc_prev_per_file: dict[str, dict[str, float]] = {}


def collect_throughput_ipc_stats(rank: int | None = None) -> dict[str, float]:
    """Read per-worker IPC files and return accurate interval deltas for wandb.

    Deltas are tracked per file so that workers appearing (spawn) or
    disappearing (death/respawn with persistent_workers=False) never corrupt
    other workers' accounting.

    Dead-worker files are read first (to capture their final cumulative delta),
    then deleted. Workers do NOT delete their own files on exit — they only
    write a final flush via atexit. This avoids losing the last interval's data.

    Returns {"MBps": ..., "watchdog_reconnects": ...} if ENABLE_STREAM_WANDB is True, otherwise {}.

    Called by DataloadingMonitor callback once per logging window.
    """
    if rank is None:
        rank = int(os.environ.get("RANK", "0"))
    ipc_dir = Path(f"/tmp/throughput_stats/rank_{rank}")
    if not ipc_dir.exists():
        return {}

    total_d_bytes = 0
    total_d_time = 0.0
    total_d_watchdog = 0
    seen: set[str] = set()

    for filepath in ipc_dir.glob("worker_*.json"):
        try:
            pid = int(filepath.stem.split("_", 1)[1])
        except (ValueError, IndexError):
            continue

        alive = _is_pid_alive(pid)

        try:
            with open(filepath) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            if not alive:
                try:
                    filepath.unlink(missing_ok=True)
                except OSError:
                    pass
            continue

        fname = filepath.name
        seen.add(fname)

        cur = {
            "bytes": data.get("bytes", 0),
            "read_time": data.get("read_time", 0.0),
            "watchdog": data.get("watchdog", 0),
        }
        prev = _ipc_prev_per_file.get(fname, {"bytes": 0, "read_time": 0.0, "watchdog": 0})

        d_b = cur["bytes"] - prev["bytes"]
        d_t = cur["read_time"] - prev["read_time"]
        d_w = cur["watchdog"] - prev["watchdog"]

        if d_b < 0 or d_t < 0 or d_w < 0:
            log.info(
                f"[Stream IPC] PID reuse detected for {fname}, treating as fresh worker",
                rank0_only=False,
            )
            d_b, d_t, d_w = cur["bytes"], cur["read_time"], cur["watchdog"]

        total_d_bytes += d_b
        total_d_time += d_t
        total_d_watchdog += d_w
        _ipc_prev_per_file[fname] = cur

        if not alive:
            try:
                filepath.unlink(missing_ok=True)
                log.info(
                    f"[Stream IPC] Read final stats and removed file for dead worker PID {pid}: {fname}",
                    rank0_only=False,
                )
            except OSError:
                pass

    for fname in list(_ipc_prev_per_file):
        if fname not in seen:
            log.info(f"[Stream IPC] Purging stale tracking entry: {fname}", rank0_only=False)
            del _ipc_prev_per_file[fname]

    result: dict[str, float] = {
        "MBps": (total_d_bytes / (1024**2)) / total_d_time if total_d_time > 0 else 0,
    }
    if WATCHDOG_ENABLED:
        result["watchdog_reconnects"] = float(total_d_watchdog)
    return result


@dataclass
class WatchdogConfig:
    """Configuration for the throughput watchdog that resets stream connections with sustained low throughput.

    Attributes:
        enabled: Master switch. Controlled by env var ``RETRYING_STREAM_WATCHDOG``
            (``"0"`` to disable; default enabled).
        min_throughput_mbps: Sustained throughput threshold in MB/s.  If the moving
            window average drops below this, the connection is reset.  Controlled by env var ``RETRYING_STREAM_WATCHDOG_MIN_MBPS`` (default ``50.0``).
        min_window_seconds: Minimum accumulated read time (seconds) in the current
            window before a throughput check is meaningful. Prevents premature resets.
        check_interval: Number of ``read()`` calls between throughput checks to avoid checking overhead.
    """

    enabled: bool = WATCHDOG_ENABLED
    min_throughput_mbps: float = WATCHDOG_MIN_THROUGHPUT_MBPS
    min_window_seconds: float = 5.0
    check_interval: int = 50


class RetryingStream:
    def __init__(self, client: boto3.client, bucket: str, key: str, retries: int = 10):  # type: ignore
        r"""Class for loading data in a streaming fashion.
        Args:
            client (boto3.client): Boto3 client
            bucket (str): Bucket where data is stored
            key (str): Key to read
            retries (int): Number of retries
        """
        self.client = client
        self.bucket = bucket
        self.key = key
        self.retries = retries
        self.name = f"{bucket}/{key}"

        # Cache stats flag as instance variable to avoid module lookup overhead
        self._enable_retry_stats = ENABLE_RETRY_STATS
        self._enable_throughput_stats = ENABLE_THROUGHPUT_STATS
        self._enable_tracking = ENABLE_THROUGHPUT_STATS or ENABLE_STREAM_WANDB

        # Get content length (with retries for transient failures)
        self.content_size = self._retry_operation(
            operation=self.get_length,
            operation_name="get_length",
            max_attempts=self.retries,
        )

        # Get initial stream (with retries for transient failures)
        self.stream, _ = self._retry_operation(
            operation=self.get_stream,
            operation_name="get_stream",
            max_attempts=self.retries,
        )

        self._amount_read = 0

        # Per-shard read timing (accumulated across all read() calls)
        self._stream_read_time = 0.0

        self._watchdog = WatchdogConfig()
        self._read_count = 0
        self._window_start_read_time: float = 0.0
        self._window_start_bytes: int = 0

        if self._enable_retry_stats:
            with _global_retry_stats.lock:
                _global_retry_stats.active_instances.add(self)

        if self._enable_tracking:
            _register_throughput_atexit()

    def __del__(self) -> None:
        r"""Destructor for cleanup.

        Note: WeakSet automatically removes dead references, so no manual cleanup needed.
        Final statistics are logged by the atexit handler when the program exits.
        """
        # WeakSet handles cleanup automatically - no action needed
        # Final stats logging happens via atexit handler, not destructor
        pass

    def _watchdog_reset_stream_if_low_throughput(self, new_position: int) -> None:
        """Reset the stream connection if sustained throughput drops below a threshold.

        Cloud object-storage backends (especially GCS) occasionally serve individual connections at far below their healthy capacity (tail latency problem).

        Because the DataLoader blocks on the slowest worker, a single degraded connection can
        bottleneck the entire training step, observed as `dataloading spikes` in the training charts.

        This mitigation abandons the slow connection and opens a fresh one from the byte offset where the previous stream left off.
        This is proven not to lose bytes (reconnection continues from where the previous stream left off), and doesn't introduce overhead.

        The check runs every `WatchdogConfig.check_interval` read() calls.
        It computes a moving-window throughput (bytes read / accumulated read time) and compares it against `WatchdogConfig.min_throughput_mbps`.
        A minimum window read time (`WatchdogConfig.min_window_seconds`) prevents premature resets. After a reset, the window counters restart from the current position.

        When disabled (`RETRYING_STREAM_WATCHDOG=0`), this method returns immediately on the first check.
        """
        wd = self._watchdog
        if (
            not wd.enabled
            or self._read_count % wd.check_interval != 0
            or self._read_count == 0
            or new_position >= self.content_size
        ):
            return

        window_read_time = self._stream_read_time - self._window_start_read_time
        window_bytes = new_position - self._window_start_bytes
        if window_read_time <= wd.min_window_seconds or window_bytes <= 0:
            return

        throughput_mbps = (window_bytes / (1024 * 1024)) / window_read_time
        if throughput_mbps >= wd.min_throughput_mbps:
            return

        if self._enable_tracking:
            with _global_throughput_stats.lock:
                _global_throughput_stats.cumulative_watchdog_reconnects += 1

        if self._enable_throughput_stats:
            rank = _global_throughput_stats.get_rank()
            pid = _global_throughput_stats.get_pid()
            log.warning(
                f"[Throughput Watchdog] [Rank {rank}] [PID {pid}] reconnecting: "
                f"{throughput_mbps:.1f}MB/s < {wd.min_throughput_mbps}MB/s, "
                f"read_time {window_read_time:.1f}s, "
                f"{self.name} @ {new_position}/{self.content_size}",
                rank0_only=False,
            )

        try:
            self.stream, _ = self.get_stream(new_position)
        except (EndpointConnectionError, ClientError) as e:
            log.warning(
                f"[Throughput Watchdog] reconnect failed: {e} {self.name}",
                rank0_only=False,
            )
        self._window_start_read_time = self._stream_read_time
        self._window_start_bytes = new_position

    @staticmethod
    def _exponential_backoff_sleep(attempt: int) -> None:
        r"""Sleep with exponential backoff based on attempt number.

        Args:
            attempt: Zero-indexed attempt number (0 for first retry)
        """
        time.sleep(0.5 * 2**attempt)

    def _retry_operation(self, operation, operation_name: str, max_attempts: int = 3):
        r"""Retry an operation with exponential backoff for transient failures.

        Args:
            operation: Callable to execute
            operation_name: Name of operation for logging
            max_attempts: Maximum number of attempts

        Returns:
            Result of the operation

        Raises:
            Exception from the operation if all retries fail
        """
        # Track this operation in both thread-local and cumulative statistics
        if self._enable_retry_stats:
            _maybe_log_retry_stats()  # Check if periodic log is due

            # Track this operation (lock-free thread-local counters)
            stats = _get_thread_stats()
            stats["operations_started"] += 1  # Count this S3 operation being started
            stats["total_attempts"] += 1  # Count the initial attempt

            # Also update cumulative counters (requires lock because += is not atomic)
            # Lock overhead is negligible: uncontended in single-threaded case, minimal contention in multi-threaded
            with _global_retry_stats.lock:
                _global_retry_stats.cumulative_operations_started += 1
                _global_retry_stats.cumulative_attempts += 1
        else:
            stats = None

        # Include EndpointConnectionError for initialization operations
        init_retryable = RETRYABLE_EXCEPTIONS + (EndpointConnectionError,)

        operation_had_retry = False  # Track if this operation failed at least once
        for attempt in range(max_attempts):
            try:
                return operation()
            except init_retryable as e:
                if attempt == max_attempts - 1:  # Last attempt
                    raise

                # Track retry statistics
                if stats is not None:
                    # Mark this operation as failed (only once per operation, lock-free)
                    if not operation_had_retry:
                        stats["failed_operations"] += 1  # This operation failed at least once
                        operation_had_retry = True
                        # Also update cumulative counter (lock needed because += is not atomic)
                        with _global_retry_stats.lock:
                            _global_retry_stats.cumulative_failed_operations += 1

                    # Count this retry attempt (lock-free)
                    stats["total_attempts"] += 1  # Each retry is an additional attempt

                    # Also update cumulative counter (lock needed because += is not atomic)
                    with _global_retry_stats.lock:
                        _global_retry_stats.cumulative_attempts += 1

                # Only log retries after the first one (attempt >= 1)
                if attempt >= 1:
                    log.warning(
                        f"Transient error in {operation_name} for {self.name} "
                        f"(attempt {attempt + 1}/{max_attempts}): {type(e).__name__}: {e}",
                        rank0_only=False,
                    )
                self._exponential_backoff_sleep(attempt)

    def get_length(self) -> int:
        r"""Function for obtaining length of the bytestream"""
        head_obj = self.client.head_object(Bucket=self.bucket, Key=self.key)
        length = int(head_obj["ContentLength"])
        return length

    def get_stream(self, start_range: int = 0, end_range: Optional[int] = None):
        r"""Function for getting stream in a range
        Args:
            start_range (int): Start index for stream
            end_range (int): End index for stream
        Returns:
            stream (bytes): Stream of data being read
            content_size (int): Length of the bytestream read
        """
        extra_args = {}
        if start_range != 0 or end_range is not None:
            # End range in S3 is inclusive
            end_str = "" if end_range is None else str(end_range - 1)
            extra_args["Range"] = f"bytes={start_range}-{end_str}"

        response = self.client.get_object(Bucket=self.bucket, Key=self.key, **extra_args)

        # FIX: Use the public 'Body' property (StreamingBody)
        # It implements .read() and handles internal resource management
        return response["Body"], int(response["ContentLength"])

    def read(self, amt: Optional[int] = None) -> bytes:
        r"""Function for reading data from the stream
        Args:
            amt (int): Amount of data to read
        Returns:
            chunk (bytes): Data read from the stream
        """
        # Track this operation in both thread-local and cumulative statistics
        if self._enable_retry_stats:
            _maybe_log_retry_stats()  # Check if periodic log is due

            # Track this read operation (lock-free thread-local counters)
            stats = _get_thread_stats()
            stats["operations_started"] += 1  # Count this read() call being started
            stats["total_attempts"] += 1  # Count the initial attempt

            # Also update cumulative counters (requires lock)
            with _global_retry_stats.lock:
                _global_retry_stats.cumulative_operations_started += 1
                _global_retry_stats.cumulative_attempts += 1
        else:
            stats = None

        operation_had_retry = False  # Track if this read() failed at least once
        for cur_retry_idx in range(self.retries):
            try:
                t_read_start = time.monotonic()
                chunk = self.stream.read(amt)
                read_dur = time.monotonic() - t_read_start
                self._stream_read_time += read_dur  # always: used by watchdog
                if self._enable_tracking:
                    with _global_throughput_stats.lock:
                        _global_throughput_stats.cumulative_bytes_read += len(chunk)
                        _global_throughput_stats.cumulative_total_read_time += read_dur
                self._read_count += 1
                # Check for unexpected end of stream
                if amt is not None and amt > 0 and len(chunk) == 0 and self._amount_read != self.content_size:
                    raise IOError("Premature end of stream detected.")

                # Throughput watchdog
                # Periodically check if sustained throughput is too low.
                # If so, abandon the slow connection and open a fresh one from where we left off.
                new_position = self._amount_read + len(chunk)
                self._watchdog_reset_stream_if_low_throughput(new_position)

                # Success: Update pointer and return
                self._amount_read += len(chunk)
                if self._enable_tracking:
                    _maybe_log_throughput_stats()
                return chunk

            except RETRYABLE_EXCEPTIONS as e:
                self._stream_read_time += time.monotonic() - t_read_start
                # Track retry statistics
                if stats is not None:
                    # Mark this operation as failed (only once per operation, lock-free)
                    if not operation_had_retry:
                        stats["failed_operations"] += 1  # This operation failed at least once
                        operation_had_retry = True
                        # Also update cumulative counter (lock needed because += is not atomic)
                        with _global_retry_stats.lock:
                            _global_retry_stats.cumulative_failed_operations += 1

                    # Count this retry attempt (lock-free)
                    stats["total_attempts"] += 1  # Each retry is an additional attempt

                    # Also update cumulative counter (lock needed because += is not atomic)
                    with _global_retry_stats.lock:
                        _global_retry_stats.cumulative_attempts += 1

                # Only log retries after the first one (cur_retry_idx >= 1)
                if cur_retry_idx >= 1:
                    log.warning(
                        f"[read] {type(e).__name__}: {e} {self.name} retry: {cur_retry_idx + 1}/{self.retries}",
                        rank0_only=False,
                    )

                if cur_retry_idx == self.retries - 1:
                    raise  # Re-raise the last exception if all retries fail

                # Exponential backoff: 0.5s, 1s, 2s, 4s, 8s...
                self._exponential_backoff_sleep(cur_retry_idx)

                try:
                    # Close the old stream to prevent resource leaks
                    if hasattr(self.stream, "close"):
                        self.stream.close()
                    # Re-establish the stream from the last successful byte
                    self.stream, _ = self.get_stream(self._amount_read)
                except RETRYABLE_EXCEPTIONS + (EndpointConnectionError,) as e_conn:
                    # Only log reconnection failures after the first retry
                    if cur_retry_idx >= 1:
                        log.warning(
                            f"Failed to reconnect on attempt {cur_retry_idx + 1}/{self.retries}: "
                            f"{type(e_conn).__name__}: {e_conn}",
                            rank0_only=False,
                        )
                    # Loop continues, will retry the entire read operation (including get_stream) next iteration
                    # Note: self.stream may be in a bad state, but we'll create a fresh one on next iteration

        return b""  # Should theoretically not reach here due to the raise
