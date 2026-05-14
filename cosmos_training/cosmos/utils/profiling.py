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

import contextlib
import os
import time

import torch

from cosmos.utils import distributed, log
from cosmos.utils.easy_io import easy_io

# (qsh 2024-11-23)  credits
# https://github.com/pytorch/torchtitan/blob/main/torchtitan/profiling.py

# how much memory allocation/free ops to record in memory snapshots
MEMORY_SNAPSHOT_MAX_ENTRIES = 100000


@contextlib.contextmanager
def maybe_enable_profiling(config, *, global_step: int = 0):
    # get user defined profiler settings
    enable_profiling = config.trainer.profiling.enable_profiling
    profile_freq = config.trainer.profiling.profile_freq

    if enable_profiling:
        trace_dir = os.path.join(config.job.path_local, "torch_trace")
        if distributed.get_rank() == 0:
            os.makedirs(trace_dir, exist_ok=True)

        rank = distributed.get_rank()

        def trace_handler(prof):
            curr_trace_dir_name = "iteration_" + str(prof.step_num)
            curr_trace_dir = os.path.join(trace_dir, curr_trace_dir_name)
            if not os.path.exists(curr_trace_dir):
                os.makedirs(curr_trace_dir, exist_ok=True)

            log.info(f"Dumping traces at step {prof.step_num}")
            begin = time.monotonic()
            if rank in config.trainer.profiling.target_ranks:
                prof.export_chrome_trace(f"{curr_trace_dir}/rank{rank}_trace.json.gz")
            log.info(f"Finished dumping traces in {time.monotonic() - begin:.2f} seconds")

        log.info(f"Profiling active. Traces will be saved at {trace_dir}")

        if not os.path.exists(trace_dir):
            os.makedirs(trace_dir, exist_ok=True)

        warmup, active = config.trainer.profiling.profile_warmup, 1
        wait = profile_freq - (active + warmup)
        assert wait >= 0, "profile_freq must be greater than or equal to warmup + active"

        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(wait=wait, warmup=warmup, active=active),
            on_trace_ready=trace_handler,
            record_shapes=config.trainer.profiling.record_shape,
            profile_memory=config.trainer.profiling.profile_memory,
            with_stack=config.trainer.profiling.with_stack,
            with_modules=config.trainer.profiling.with_modules,
        ) as torch_profiler:
            torch_profiler.step_num = global_step
            yield torch_profiler
    else:
        torch_profiler = contextlib.nullcontext()
        yield None


@contextlib.contextmanager
def maybe_enable_memory_snapshot(config, *, global_step: int = 0):
    enable_snapshot = config.trainer.profiling.enable_memory_snapshot
    if enable_snapshot:
        if config.trainer.profiling.save_s3:
            snapshot_dir = "s3://rundir"
        else:
            snapshot_dir = os.path.join(config.job.path_local, "memory_snapshot")
            if distributed.get_rank() == 0:
                os.makedirs(snapshot_dir, exist_ok=True)

        rank = torch.distributed.get_rank()

        class MemoryProfiler:
            def __init__(self, step_num: int, freq: int):
                torch.cuda.memory._record_memory_history(max_entries=MEMORY_SNAPSHOT_MAX_ENTRIES)
                # when resume training, we start from the last step
                self.step_num = step_num
                self.freq = freq

            def step(self, exit_ctx: bool = False):
                self.step_num += 1
                if not exit_ctx and self.step_num % self.freq != 0:
                    return
                if not exit_ctx:
                    curr_step = self.step_num
                    dir_name = f"iteration_{curr_step}"
                else:
                    # dump as iteration_0_exit if OOM at iter 1
                    curr_step = self.step_num - 1
                    dir_name = f"iteration_{curr_step}_exit"
                curr_snapshot_dir = os.path.join(snapshot_dir, dir_name)
                if not config.trainer.profiling.save_s3 and not os.path.exists(curr_snapshot_dir):
                    os.makedirs(curr_snapshot_dir, exist_ok=True)
                log.info(f"Dumping memory snapshot at step {curr_step}")
                begin = time.monotonic()

                if rank in config.trainer.profiling.target_ranks:
                    easy_io.dump(
                        torch.cuda.memory._snapshot(),
                        f"{curr_snapshot_dir}/rank{rank}_memory_snapshot.pickle",
                    )
                log.info(f"Finished dumping memory snapshot in {time.monotonic() - begin:.2f} seconds")

        log.info(f"Memory profiler active. Snapshot will be saved at {snapshot_dir}")
        profiler = MemoryProfiler(global_step, config.trainer.profiling.profile_freq)
        try:
            yield profiler
        except torch.cuda.OutOfMemoryError as e:
            profiler.step(exit_ctx=True)
    else:
        yield None


@contextlib.contextmanager
def maybe_enable_nsys_profiling(config, *, global_step: int = 0):
    """Context manager for Nsight Systems profiling via cudaProfilerStart/Stop.

    Usage: launch training with
        nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop python ...
    and set trainer.profiling.enable_nsys=true, profile_freq=<iter>.

    Reuses the torch-profile flags (profile_freq, target_ranks, profile_warmup).
    The profiler is started `profile_warmup` iterations before the target and
    stopped right after it.
    """
    enable_nsys = config.trainer.profiling.enable_nsys
    if not enable_nsys:
        yield None
        return

    rank = distributed.get_rank()
    target_ranks = config.trainer.profiling.target_ranks
    freq = config.trainer.profiling.profile_freq
    warmup = config.trainer.profiling.profile_warmup

    active_iter = freq - 1  # profile_freq=5001 profiles iter 5000
    start_iter = max(0, active_iter - warmup)

    class NsysProfiler:
        def __init__(self, step_num: int):
            self.step_num = step_num
            self._profiling = False

        def step(self):
            self.step_num += 1
            if rank not in target_ranks:
                return
            if self.step_num == start_iter and not self._profiling:
                log.info(f"[Nsys] Starting CUDA profiler at iter {self.step_num} (active iter: {active_iter})")
                torch.cuda.cudart().cudaProfilerStart()
                self._profiling = True
            if self.step_num == active_iter + 1 and self._profiling:
                torch.cuda.cudart().cudaProfilerStop()
                self._profiling = False
                log.info(f"[Nsys] Stopped CUDA profiler at iter {self.step_num}")

    log.info(f"[Nsys] Profiling enabled. Will capture iter {start_iter}-{active_iter} on ranks {target_ranks}")
    profiler = NsysProfiler(global_step)
    try:
        yield profiler
    finally:
        if profiler._profiling:
            torch.cuda.cudart().cudaProfilerStop()
