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

from __future__ import annotations

import contextlib
import threading
import time
from collections.abc import Iterator
from typing import Any, Optional, Union

from cosmos.utils import log


class OperationWatchdog:
    """A watchdog that monitors operations for hangs and collects performance statistics.

    This class provides a mechanism to detect when operations take longer than expected,
    which can help identify potential deadlocks or performance issues. It also collects
    statistics about operations to help identify bottlenecks.

    Attributes:
        warning_threshold: Time in seconds before warning about potential hangs.
        check_interval: Time in seconds between checks for hung operations.
        verbose_interval: Time in seconds between verbose logging.
    """

    def __init__(
        self,
        warning_threshold: int = 600,
        check_interval: int = 30,
        verbose_interval: int = -1,
        name: str = "OperationWatchdog",
    ) -> None:
        """Initialize the watchdog.

        Args:
            warning_threshold: Time in seconds before warning about potential hangs.
                Defaults to 600 (10 minutes).
            check_interval: Time in seconds between checks. Defaults to 30.
            verbose_interval: Time in seconds between verbose logging. Defaults to -1 (disabled).
        """
        self._warning_threshold = warning_threshold
        self._check_interval = check_interval
        self._verbose_interval = verbose_interval
        self._name = name
        self._ops: dict[str, dict[str, Any]] = {}  # Active operations
        self._stats: dict[str, dict[str, Union[int, float]]] = {}  # Operation statistics
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # Auto-start the monitoring thread
        self.start()

    def start(self) -> None:
        """Start the watchdog monitoring thread.

        If the thread is already running, this method does nothing.
        """
        if self._thread is None or not self._thread.is_alive():
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._monitor_loop, daemon=True, name=f"{self._name}_monitor_thread")
            self._thread.start()
            log.debug(f"[{self._name}] Watchdog monitoring thread started")

    def stop(self) -> None:
        """Stop the watchdog monitoring thread.

        This method is typically called when shutting down the application.
        """
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
            if self._thread.is_alive():
                log.warning(f"[{self._name}] Watchdog thread did not terminate within timeout", rank0_only=False)
            else:
                log.debug(f"[{self._name}] Watchdog monitoring thread stopped", rank0_only=False)

    @contextlib.contextmanager
    def watch(self, operation_name: str, description: str = "", verbose_first_n: int = -1) -> Iterator[None]:
        """Context manager for monitoring an operation.

        This is the primary interface for using the watchdog. It automatically
        tracks the start and end time of the operation and updates statistics.

        Args:
            operation_name: Name/type of the operation to monitor.
            description: Optional description providing more context.
            verbose_first_n: If positive, print verbose logs for first N operations for operation_name.

        Yields:
            None

        Example:
            with watchdog.watch("data_fetch", "Fetching user data"):
                data = fetch_user_data(user_id)
        """
        # Create unique ID for this specific operation instance
        op_id = f"{operation_name}_{int(time.time() * 1000)}"
        start_time = time.time()

        # Register operation
        with self._lock:
            self._ops[op_id] = {
                "name": operation_name,
                "desc": description or operation_name,
                "start": start_time,
                "update": start_time,
                "warned": False,
            }

        try:
            # Yield control back to the with-block
            yield
        finally:
            # Calculate duration and remove from active operations
            duration = time.time() - start_time
            if duration > self._warning_threshold:
                log.warning(
                    f"[{self._name}] Operation id: {op_id}, name: '{operation_name}' took {duration:.2f}s",
                    rank0_only=False,
                )

            with self._lock:
                # Remove from active operations
                if op_id in self._ops:
                    del self._ops[op_id]

                # Update statistics
                if operation_name not in self._stats:
                    self._stats[operation_name] = {"count": 0, "total_time": 0.0, "max_time": 0.0, "last_time": 0.0}

                stats = self._stats[operation_name]
                stats["count"] += 1
                stats["total_time"] += duration
                stats["max_time"] = max(stats["max_time"], duration)
                stats["last_time"] = duration

            if verbose_first_n > 0 and stats["count"] <= verbose_first_n:
                avg_time = stats["total_time"] / stats["count"]
                log.info(
                    f"[{self._name}] name: '{operation_name}', count {stats['count']} / {verbose_first_n} took {duration:.2f}s. avg {avg_time:.2f}s, max {stats['max_time']:.2f}s, last {stats['last_time']:.2f}s",
                    rank0_only=False,
                )

    def heartbeat(self, operation_name: str) -> None:
        """Send a heartbeat for all operations of a given type.

        Use this inside long operations to prevent false warnings.

        Args:
            operation_name: The operation type to update.
        """
        current_time = time.time()
        with self._lock:
            for _op_id, op in self._ops.items():
                if op["name"] == operation_name:
                    op["update"] = current_time
                    op["warned"] = False

    def get_stats(self, operation_name: Optional[str] = None) -> dict[str, Any]:
        """Get statistics for operations.

        Args:
            operation_name: Get stats for specific operation, or None for all.

        Returns:
            Dictionary of operation statistics. For each operation, includes:
            - count: Number of completed operations
            - total_time: Total time spent in this operation type
            - max_time: Maximum time spent in a single operation
            - last_time: Time spent in the most recent operation
            - avg_time: Average time per operation (if count > 0)
        """
        with self._lock:
            if operation_name:
                if operation_name in self._stats:
                    stats = self._stats[operation_name].copy()
                    if stats["count"] > 0:
                        stats["avg_time"] = stats["total_time"] / stats["count"]
                    return stats
                return {}

            # Return all stats
            result = {}
            for name, stats in self._stats.items():
                result[name] = stats.copy()
                if stats["count"] > 0:
                    result[name]["avg_time"] = stats["total_time"] / stats["count"]
            return result

    def print_stats(self) -> None:
        """Print statistics for all operations.

        This is a convenience method that logs statistics at INFO level.
        """
        stats = self.get_stats()
        if not stats:
            log.info(f"[{self._name}] No operation statistics available", rank0_only=False)
            return

        log.info(f"[{self._name}] Operation Statistics:", rank0_only=False)

        for name, s in stats.items():
            if s["count"] > 0:
                avg = s["total_time"] / s["count"]
                log.info(
                    f"[{self._name}]  {name}: count={s['count']}, "
                    f"avg={avg:.2f}s, max={s['max_time']:.2f}s, "
                    f"last={s['last_time']:.2f}s",
                    rank0_only=False,
                )

    def list_active_operations(self) -> dict[str, Any]:
        """List all currently active operations.

        Returns:
            Dictionary mapping operation IDs to information about active operations.
        """
        with self._lock:
            current_time = time.time()
            result = {}

            for op_id, op in self._ops.items():
                result[op_id] = {
                    "name": op["name"],
                    "description": op["desc"],
                    "running_time": current_time - op["start"],
                    "time_since_update": current_time - op["update"],
                }

            return result

    def reset_stats(self) -> None:
        """Reset all operation statistics.

        This clears all accumulated statistics but does not affect active operations.
        """
        with self._lock:
            self._stats.clear()

    def _monitor_loop(self) -> None:
        """Monitor registered operations for hangs.

        This is an internal method that runs in a separate thread.
        """
        last_verbose_time = 0
        while not self._stop_event.is_set():
            now = time.time()

            with self._lock:
                for _, op in list(self._ops.items()):
                    # Check if operation is hung
                    elapsed = now - op["update"]
                    if elapsed > self._warning_threshold and not op["warned"]:
                        total_time = now - op["start"]
                        log.warning(
                            f"[{self._name}] POTENTIAL HANG: '{op['name']}' ({op['desc']}) "
                            f"has been running for {total_time:.1f}s total, "
                            f"with {elapsed:.1f}s since last update",
                            rank0_only=False,
                        )
                        op["warned"] = True

            # Sleep between checks
            if self._verbose_interval > 0 and now - last_verbose_time > self._verbose_interval:
                self.print_stats()
                last_verbose_time = now
            self._stop_event.wait(self._check_interval)
