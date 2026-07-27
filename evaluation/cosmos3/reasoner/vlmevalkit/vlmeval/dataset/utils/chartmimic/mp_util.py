import os
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Callable, Iterable

from tqdm import tqdm

from vlmeval.smp import dump, get_logger, load

logger = get_logger(__name__)


def track_progress_rich_new(
    func: Callable,
    tasks: Iterable = tuple(),
    nproc: int = 1,
    save=None,
    keys=None,
    **kwargs
) -> list:
    """
    Parallel execution with progress tracking and safe interim saving.
    """
    # Prepare persistent storage
    if save:
        os.makedirs(os.path.dirname(save), exist_ok=True)
        if not os.path.exists(save):
            dump({}, save)
        res = load(save)
    else:
        res = {}

    results = [None] * len(tasks)
    future_to_idx = {}

    # Use process pool to bypass GIL for CPU-bound tasks
    with ProcessPoolExecutor(max_workers=nproc) as executor:
        for idx, inp in enumerate(tasks):
            # Support dict, tuple/list, or single-value tasks
            if isinstance(inp, dict):
                future = executor.submit(func, **inp)
            elif isinstance(inp, (list, tuple)):
                future = executor.submit(func, *inp)
            else:
                future = executor.submit(func, inp)
            future_to_idx[future] = idx

        # Display progress bar as tasks complete
        with tqdm(total=len(tasks)) as pbar:
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                key = keys[idx] if keys else None
                try:
                    result = future.result()
                except Exception as e:
                    exc_type = type(e).__name__
                    err_msg = f"[{exc_type}] Exception in task {key or idx}: {str(e)}"
                    logger.error(err_msg)
                    logger.error("Full traceback:")
                    logger.error(traceback.format_exc())

                    # Optional: attach traceback to result for downstream
                    # reference
                    result = getattr(e, 'result', (-1, {
                        'msg': err_msg,
                        'traceback': traceback.format_exc(),
                    }))

                results[idx] = result
                # Update persistent results
                if keys and key is not None:
                    res[key] = result
                if save:
                    dump(res, save)  # save after each task

                pbar.update(1)

    return results
