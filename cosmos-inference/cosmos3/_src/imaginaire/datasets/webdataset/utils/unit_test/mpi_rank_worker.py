#!/usr/bin/env python
# -----------------------------------------------------------------------------
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# -----------------------------------------------------------------------------

"""Worker script for MPI/torchrun multi-rank statistics testing.

This script is launched by torchrun or mpirun with multiple ranks.
Each rank spawns multiple worker processes to simulate DataLoader workers.
Each worker independently tests RetryingStream statistics tracking.

Dependencies:
    - torch.distributed (required)
    - mpi4py (optional, only needed for mpirun testing)

Usage with torchrun (no extra dependencies):
    torchrun --nproc_per_node=10 mpi_rank_worker.py

Usage with mpirun (requires mpi4py):
    uv pip install mpi4py  # Only if using mpirun
    mpirun -np 10 --oversubscribe python mpi_rank_worker.py

Environment variables:
    SIMULATE_FAILURE_RANKS: Comma-separated list of ranks to kill (e.g., "2,5,7")
    SKIP_BARRIER: If set to "1", skips the final barrier (for failure tests)
    NUM_WORKERS_PER_RANK: Number of worker processes per rank (default: 3, simulates DataLoader workers)
"""

import multiprocessing
import os
import random
import sys
from http.client import IncompleteRead
from unittest.mock import MagicMock

import cosmos3._src.imaginaire.datasets.webdataset.utils.stream as stream_module
from cosmos3._src.imaginaire.datasets.webdataset.utils.stream import RetryingStream
from cosmos3._src.imaginaire.utils import log

# Configure faster logging interval for tests (10 seconds instead of 5 minutes)
stream_module.RETRY_STATS_LOG_INTERVAL = 10.0


def init_distributed():
    """Initialize distributed environment (torchrun or mpirun).

    Returns:
        tuple: (rank, world_size, launcher_type)
        launcher_type: 'torchrun' or 'mpirun'

    Raises:
        RuntimeError: If neither torchrun nor mpirun is available
    """
    import torch.distributed as dist

    # Try torchrun first
    if "TORCHELASTIC_RUN_ID" in os.environ or ("RANK" in os.environ and "WORLD_SIZE" in os.environ):
        # Initialize PyTorch distributed
        if not dist.is_initialized():
            dist.init_process_group(backend="gloo")  # Use gloo for CPU testing

        rank = dist.get_rank()
        world_size = dist.get_world_size()
        return rank, world_size, "torchrun"

    # Try mpirun
    if "OMPI_COMM_WORLD_RANK" in os.environ or "PMI_RANK" in os.environ:
        try:
            from mpi4py import MPI  # type: ignore  # Optional dependency for MPI testing
        except ImportError as e:
            raise RuntimeError(
                "mpirun detected but mpi4py is not installed!\n"
                "Install with: uv pip install mpi4py\n"
                "Or use torchrun instead:\n"
                "  torchrun --nproc_per_node=N mpi_rank_worker.py"
            ) from e

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        world_size = comm.Get_size()

        # Initialize torch.distributed using MPI backend so log module can detect rank
        if not dist.is_initialized():
            # Set environment variables for torch.distributed
            os.environ["RANK"] = str(rank)
            os.environ["WORLD_SIZE"] = str(world_size)
            os.environ["MASTER_ADDR"] = "127.0.0.1"
            os.environ["MASTER_PORT"] = "29500"

            # Initialize with gloo backend (MPI backend requires special build)
            dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)

        return rank, world_size, "mpirun"

    # Neither launcher detected
    raise RuntimeError(
        "Neither torchrun nor mpirun detected!\n"
        "This script must be launched with:\n"
        "  torchrun --nproc_per_node=N mpi_rank_worker.py\n"
        "  OR\n"
        "  mpirun -np N python mpi_rank_worker.py"
    )


def worker_process(
    worker_id: int,
    rank: int,
    world_size: int,
    num_operations: int,
    failure_interval: int,
    should_fail: bool,
    result_queue: multiprocessing.Queue,
) -> None:
    """Worker process function that simulates a DataLoader worker.

    Each worker process:
    1. Has its own PID and independent _global_retry_stats instance
    2. Performs a different number of operations (worker-specific workload)
    3. Uses worker-specific failure patterns
    4. Logs its own statistics independently

    Args:
        worker_id: Unique worker ID within this rank (0, 1, 2, ...)
        rank: Distributed rank this worker belongs to
        world_size: Total number of distributed ranks
        num_operations: Base number of operations (will be varied per worker)
        failure_interval: How often operations fail
        should_fail: Whether this worker should simulate a crash
        result_queue: Queue to report results back to parent
    """
    try:
        # Initialize logging in worker process
        # Note: Worker processes spawned via multiprocessing.Process don't have torch.distributed
        # initialized, which is the same behavior as real PyTorch DataLoader workers.
        # They inherit the RANK environment variable from their parent, which RetryingStream's
        # get_rank() will use as a fallback. This ensures statistics log with the correct rank.
        log.init_loguru_stdout()

        # Enable statistics
        stream_module.ENABLE_RETRY_STATS = True

        # Note: atexit handler registration is now automatic (handled lazily in _get_thread_stats)
        # Each worker process automatically registers its own handler on first RetryingStream use

        # Each worker does a different amount of work to simulate real workload imbalance
        # Worker 0: 100% of base ops, Worker 1: 80%, Worker 2: 120%, Worker 3: 60%
        workload_multipliers = [1.0, 0.8, 1.2, 0.6, 1.1, 0.9]
        multiplier = workload_multipliers[worker_id % len(workload_multipliers)]
        worker_num_operations = int(num_operations * multiplier)

        # Each worker has a slightly different failure pattern
        worker_failure_interval = failure_interval + worker_id  # Offset by worker_id

        print(
            f"Rank {rank} Worker {worker_id} (PID={os.getpid()}): "
            f"{worker_num_operations} ops, fail every {worker_failure_interval}th",
            flush=True,
        )

        # Calculate expected stats for this worker
        expected_init_ops = worker_num_operations * 2
        expected_read_ops = worker_num_operations
        expected_total_ops = expected_init_ops + expected_read_ops

        # Count failures
        if worker_num_operations > 0:
            expected_failed_reads = (worker_num_operations - 1) // worker_failure_interval + 1
        else:
            expected_failed_reads = 0
        expected_total_failed = expected_failed_reads
        expected_total_attempts = expected_total_ops + expected_failed_reads

        # Setup mock S3 client
        client = MagicMock()
        test_data = b"X" * 1024
        client.head_object.return_value = {"ContentLength": str(len(test_data))}

        # Simulate crash at random point if requested
        failure_point = random.randint(30, 70) if should_fail else None
        if should_fail:
            print(
                f"⚠️  Rank {rank} Worker {worker_id}: WILL CRASH at operation {failure_point}/{worker_num_operations}",
                flush=True,
            )

        # Perform operations
        for i in range(worker_num_operations):
            if should_fail and i == failure_point:
                print(f"💥 Rank {rank} Worker {worker_id}: SIMULATING CRASH (killed at operation {i})", flush=True)
                sys.stdout.flush()
                sys.stderr.flush()
                os._exit(1)

            mock_body = MagicMock()

            # Fail based on worker-specific interval
            if i % worker_failure_interval == 0:
                mock_body.read.side_effect = [IncompleteRead(b"partial"), test_data]
            else:
                mock_body.read.return_value = test_data

            client.get_object.return_value = {"Body": mock_body, "ContentLength": len(test_data)}

            stream = RetryingStream(client, f"bucket-rank{rank}-w{worker_id}", f"file-{i}.tar", retries=5)
            try:
                _ = stream.read(1024)
            except Exception as e:
                print(f"Rank {rank} Worker {worker_id}: ERROR: {e}", flush=True)
                result_queue.put({"worker_id": worker_id, "success": False, "error": str(e)})
                sys.stdout.flush()
                sys.stderr.flush()
                sys.exit(1)

        # Force stats logging for this worker process
        with stream_module._global_retry_stats.lock:
            stream_module._global_retry_stats.last_log_time = 0
        stream_module._log_retry_stats_internal(force=False)

        # Get and verify cumulative stats
        with stream_module._global_retry_stats.lock:
            actual_ops_started = stream_module._global_retry_stats.cumulative_operations_started
            actual_failed_ops = stream_module._global_retry_stats.cumulative_failed_operations
            actual_total_attempts = stream_module._global_retry_stats.cumulative_attempts

            # Verify silently unless there's a mismatch

            # Verify stats
            success = (
                actual_ops_started == expected_total_ops
                and actual_failed_ops == expected_total_failed
                and actual_total_attempts == expected_total_attempts
            )

            if not success:
                print(f"❌ Rank {rank} Worker {worker_id}: Statistics mismatch!", flush=True)
                result_queue.put({"worker_id": worker_id, "success": False, "error": "stats_mismatch"})
                # Log final statistics even on failure
                stream_module._log_retry_stats_internal(force=True)
                sys.stdout.flush()
                sys.stderr.flush()
                sys.exit(1)

        print(f"✅ Rank {rank} Worker {worker_id}: Verified", flush=True)
        result_queue.put({"worker_id": worker_id, "success": True})

        # Explicitly log final statistics before exit
        # Note: atexit handlers are unreliable in multiprocessing.Process (even with sys.exit(0))
        # This is a known Python limitation, so we explicitly call the final log
        stream_module._log_retry_stats_internal(force=True)

        # Explicitly flush all output streams before exiting
        sys.stdout.flush()
        sys.stderr.flush()

        # Exit cleanly
        sys.exit(0)

    except Exception as e:
        print(f"❌ Rank {rank} Worker {worker_id}: Unexpected error: {e}", flush=True)
        result_queue.put({"worker_id": worker_id, "success": False, "error": str(e)})
        # Log final statistics even on error
        try:
            stream_module._log_retry_stats_internal(force=True)
        except Exception:
            pass  # Don't let logging errors mask the original error
        sys.stdout.flush()
        sys.stderr.flush()
        sys.exit(1)


def main():
    """Main function: spawns multiple worker processes per rank to simulate DataLoader workers."""
    # Initialize distributed environment (torchrun or mpirun)
    rank, world_size, launcher = init_distributed()

    # Get configuration from environment
    simulate_failure_ranks = os.environ.get("SIMULATE_FAILURE_RANKS", "")
    should_fail = str(rank) in simulate_failure_ranks.split(",") if simulate_failure_ranks else False
    skip_barrier = os.environ.get("SKIP_BARRIER", "0") == "1"
    num_workers = int(os.environ.get("NUM_WORKERS_PER_RANK", "3"))  # Default: 3 workers per rank

    try:
        # Initialize logging in main process
        log.init_loguru_stdout()

        # Base operations per worker
        num_operations = 50

        # Rank-specific failure pattern
        failure_intervals = [10, 7, 5, 4, 3, 3, 2, 2, 2, 2]
        failure_interval = failure_intervals[rank % len(failure_intervals)]

        print(
            f"Rank {rank}/{world_size} ({launcher}): Starting {num_workers} workers",
            flush=True,
        )

        # Create queue for worker results
        result_queue = multiprocessing.Queue()

        # Spawn worker processes (simulating DataLoader workers)
        workers = []
        for worker_id in range(num_workers):
            # Only simulate failure in the first worker of a failing rank
            worker_should_fail = should_fail and worker_id == 0

            p = multiprocessing.Process(
                target=worker_process,
                args=(
                    worker_id,
                    rank,
                    world_size,
                    num_operations,
                    failure_interval,
                    worker_should_fail,
                    result_queue,
                ),
            )
            p.start()
            workers.append(p)

        # Wait for all workers to complete
        all_success = True
        for p in workers:
            p.join()
            if p.exitcode != 0:
                print(f"❌ Rank {rank}: Worker with PID {p.pid} failed with exit code {p.exitcode}", flush=True)
                all_success = False

        # Collect results from queue
        worker_results = []
        while not result_queue.empty():
            worker_results.append(result_queue.get())

        # Verify all workers succeeded
        success_count = sum(1 for r in worker_results if r.get("success", False))

        if not all_success or success_count != num_workers:
            print(f"❌ Rank {rank}: {success_count}/{num_workers} workers succeeded", flush=True)
            sys.exit(1)

        print(f"✅ Rank {rank}: All {num_workers} workers verified", flush=True)

        # Synchronize all ranks before exit (silent)
        if not skip_barrier:
            try:
                import torch.distributed as dist

                if dist.is_initialized():
                    dist.barrier()
            except Exception:
                pass  # Ignore barrier failures

    finally:
        # Cleanup distributed environment
        if launcher == "torchrun":
            try:
                import torch.distributed as dist

                if dist.is_initialized():
                    dist.destroy_process_group()
            except (ImportError, Exception):
                pass
        # mpi4py calls MPI.Finalize() automatically at exit, no cleanup needed


if __name__ == "__main__":
    main()
