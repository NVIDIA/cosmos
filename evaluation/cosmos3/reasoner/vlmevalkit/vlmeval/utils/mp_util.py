import multiprocessing
import os
import os.path as osp
import time
from pathlib import Path
from typing import Callable, Iterable

from ..smp import dump, load

# Set start method to spawn for general stability and compatibility with s3fs package
multiprocessing.set_start_method('forkserver', force=True)


def cpu_count():
    # Handle K8s LXCFS setting.
    period = Path('/sys/fs/cgroup/cpu/cpu.cfs_period_us')
    quota = Path('/sys/fs/cgroup/cpu/cpu.cfs_quota_us')
    try:
        if period.exists() and quota.exists():
            return int(quota.read_text()) // int(period.read_text())
    except Exception:
        pass
    return os.cpu_count()


def track_progress_rich(
        func: Callable,
        tasks: Iterable = tuple(),
        nproc: int = 1,
        save=None,
        keys=None,
        use_process=False,
        **kwargs) -> list:

    from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
    if use_process:
        PoolExecutor = ProcessPoolExecutor
    else:
        PoolExecutor = ThreadPoolExecutor

    from tqdm import tqdm
    if save is not None:
        assert osp.exists(osp.dirname(save)) or osp.dirname(save) == ''
        if not osp.exists(save):
            dump({}, save)
    if keys is not None:
        assert len(keys) == len(tasks)
    if not callable(func):
        raise TypeError('func must be a callable object')
    if not isinstance(tasks, Iterable):
        raise TypeError(
            f'tasks must be an iterable object, but got {type(tasks)}')
    assert nproc > 0, 'nproc must be a positive number'
    res = load(save) if save is not None else {}
    results = [None for _ in range(len(tasks))]

    with PoolExecutor(max_workers=nproc) as executor:
        futures = []

        for inputs in tasks:
            if not isinstance(inputs, (tuple, list, dict)):
                inputs = (inputs, )
            if isinstance(inputs, dict):
                future = executor.submit(func, **inputs)
            else:
                future = executor.submit(func, *inputs)
            futures.append(future)

        unfinished = set(range(len(tasks)))
        pbar = tqdm(total=len(unfinished))
        while len(unfinished):
            new_finished = set()
            for idx in unfinished:
                if futures[idx].done():
                    results[idx] = futures[idx].result()
                    new_finished.add(idx)
                    if keys is not None:
                        res[keys[idx]] = results[idx]
            if len(new_finished):
                if save is not None:
                    dump(res, save)
                pbar.update(len(new_finished))
                for k in new_finished:
                    unfinished.remove(k)
            time.sleep(0.1)
        pbar.close()

    if save is not None:
        dump(res, save)
    return results


def _worker_wrapper(args):
    """Wrapper to unpack arguments for pool.imap_unordered."""
    func, idx, inputs = args
    if isinstance(inputs, dict):
        return idx, func(**inputs)
    else:
        return idx, func(*inputs)


def track_progress_rich_new(
        func: Callable,
        tasks: Iterable = tuple(),
        nproc: int = 1,
        save=None,
        keys=None,
        **kwargs) -> list:

    from tqdm import tqdm
    if save is not None:
        assert osp.exists(osp.dirname(save)) or osp.dirname(save) == ''
        if not osp.exists(save):
            dump({}, save)
    if keys is not None:
        assert len(keys) == len(tasks)
    if not callable(func):
        raise TypeError('func must be a callable object')
    if not isinstance(tasks, Iterable):
        raise TypeError(
            f'tasks must be an iterable object, but got {type(tasks)}')
    assert nproc > 0, 'nproc must be a positive number'
    res = load(save) if save is not None else {}
    results = [None for _ in range(len(tasks))]

    # Prepare tasks with index for tracking
    def task_generator():
        for idx, inputs in enumerate(tasks):
            if not isinstance(inputs, (tuple, list, dict)):
                inputs = (inputs, )
            yield (func, idx, inputs)

    print(f'Using {nproc} processes...')
    with multiprocessing.Pool(processes=nproc) as pool:
        pbar = tqdm(total=len(tasks))
        chunksize = max(1, min(len(tasks) // nproc, 8))
        for idx, result in pool.imap_unordered(
                _worker_wrapper, task_generator(), chunksize=chunksize):
            results[idx] = result
            if keys is not None:
                res[keys[idx]] = result
            pbar.update(1)
        pbar.close()

    if save is not None:
        dump(res, save)
    return results
