import asyncio
import json
import math
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass, field
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Literal

import pandas as pd
from tabulate import tabulate

from vlmeval.smp import dump, get_logger, load
from vlmeval.smp.log import setup_subprocess_logger

logger = get_logger(__name__)

FAIL_MSG = 'Failed to obtain answer via API.'
DatasetType = Literal["image", "video", "mt"]


def _eval_subprocess_target(
    dataset_obj,
    result_file: str,
    judge_kwargs: dict,
    log_file: str,
):
    """Evaluate function in child processes."""
    setup_subprocess_logger(log_file)
    logger.info(f"🔔 [Eval Start] {dataset_obj.dataset_name}")

    try:
        with open(log_file, 'a') as f:
            with redirect_stdout(f), redirect_stderr(f):
                result = dataset_obj.evaluate(result_file, **judge_kwargs)

        if isinstance(result, pd.DataFrame):
            result = result.to_dict()

        return {'success': True, 'result': result, 'error': None}

    except Exception as e:
        import traceback
        error_msg = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        print(f"\n❌ EVALUATION ERROR: {error_msg}")

        return {'success': False, 'result': None, 'error': error_msg}


def chat_mt(model, messages: List[dict], dataset_name: str) -> List[str]:
    """Infer function for multi-turn dataset."""
    assert len(messages) % 2 == 0, "Messages should be pairs of user and assistant"
    nturn = len(messages) // 2
    utter_stack = []
    predictions = []

    for i in range(nturn):
        utter = messages[2 * i]
        utter_stack.append(utter)
        try:
            resp = model.chat(utter_stack, dataset=dataset_name)
            utter_stack.append(dict(role='assistant', content=resp))
        except Exception as e:
            resp = FAIL_MSG + str(e)
            utter_stack.append(dict(role='assistant', content=resp))
        predictions.append(resp)

    return predictions

# ==========================================
# Core Data Structures
# ==========================================


class EvalStatus(Enum):
    Pending = 1
    Running = 2
    Done = 3
    Error = 4
    Skipped = 5


@dataclass
class InferenceTask:
    """Data for a single sample to inference."""
    sample_index: str
    prompt_struct: Any     # Constructed prompt

    dataset_name: str      # Dataset name (for naming task)
    model_name: str        # Model name (for naming task)
    dataset_type: DatasetType = "image"


@dataclass
class DatasetConfig:
    """Dataset Config and Status."""
    dataset_name: str
    dataset_obj: Any  # The constructed dataset.

    model_name: str
    model_obj: Any  # The inference model.

    work_dir: str
    result_file: str
    judge_kwargs: dict  # judge model parameters.
    verbose: bool = False

    dataset_type: DatasetType = "image"

    # Configuration for Video dataset
    # Whether to pass raw video file to model. If None, follow model.
    video_llm: bool | None = None

    # Runtime state
    total_samples: int = 0
    processed: int = 0
    eval_status: EvalStatus = EvalStatus.Pending
    eval_start_time: float = 0.0
    eval_duration: float = 0.0

    # Inference timing statistics
    infer_start_time: float = 0.
    infer_end_time: float = 0.
    infer_total_time: float = 0.
    infer_count: int = 0

    # For final result assembly
    results_dict: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.total_samples = len(self.dataset_obj.data)


# ==========================================
# Main Pipeline Class
# ==========================================


class APIEvalPipeline:

    def __init__(
        self,
        dataset_configs: List[DatasetConfig],
        concurrency: int = 32,
        monitor_interval: int = 30,
        run_infer: bool = True,
        run_eval: bool = True,
        debug: bool = False,
    ):
        """
        Args:
            dataset_configs: Dataset configurations.
            concurrency: The concurrency to inference.
            monitor_interval: The internal to print status summary.
            run_infer: Whether to inference.
            run_eval: Whether to eval.
            debug: Debug mode (Evaluate in the main process).
        """
        self.dataset_configs = dataset_configs
        self.concurrency = concurrency
        self.monitor_interval = monitor_interval
        self.run_infer = run_infer
        self.run_eval = run_eval
        self.debug = debug
        self.all_infer_done = False

        self.infer_executor = ThreadPoolExecutor(max_workers=concurrency)
        self.eval_executor = ProcessPoolExecutor(max_workers=4)
        self.producer_executor = ProcessPoolExecutor(max_workers=1)
        # The inference tasks queue (Prefetch 20% data).
        self.queue = asyncio.Queue(maxsize=int(concurrency * 1.2))
        self.states: Dict[str, DatasetConfig] = {
            cfg.dataset_name: cfg for cfg in dataset_configs
        }
        # File locks to save checkpoints threading-safety.
        self.file_locks: Dict[str, threading.Lock] = {
            cfg.dataset_name: threading.Lock() for cfg in dataset_configs
        }

        # Runtime status
        self.start_time = 0.0
        self.active_workers = 0
        self.total_tasks_generated = 0

        # Make work dirs
        for cfg in dataset_configs:
            Path(cfg.work_dir).mkdir(parents=True, exist_ok=True)
            if not self.run_eval:
                cfg.eval_status = EvalStatus.Skipped

    def _release_dataset_memory(self, cfg: DatasetConfig):
        """Release dataset after evaluation."""
        import gc
        dataset_name = cfg.dataset_name

        if hasattr(cfg.dataset_obj, 'data') and cfg.dataset_obj.data is not None:
            try:
                cfg.dataset_obj.data = None
                gc.collect()
                logger.info(f"🧹 [{dataset_name}] Memory released.")
            except Exception as e:
                logger.warning(f"   [{dataset_name}] Failed to release dataset memory: {e}")

    def _shutdown_executors(self):
        """Shutdown all Executor."""
        try:
            self.infer_executor.shutdown(wait=False)
            logger.debug("Shutdown infer_executor")

            self.eval_executor.shutdown(wait=False)
            logger.debug("Shutdown eval_executor")

            self.producer_executor.shutdown(wait=False)
            logger.debug("Shutdown producer_executor")

            logger.info("🧹 All executors shutdown")

        except Exception as e:
            logger.warning(f"Failed to shutdown executors: {e}")

    def _get_checkpoint_file(self, dataset_name: str) -> Path:
        cfg = self.states[dataset_name]
        return Path(cfg.work_dir) / f"{cfg.model_name}_{dataset_name}_checkpoint.pkl"

    def _load_checkpoint(self, dataset_name: str) -> Dict[str, Any]:
        """Load finished inference result from previous runs."""
        cfg = self.states[dataset_name]
        results = {}

        # 1. Try to load checkpoint at first.
        checkpoint_file = self._get_checkpoint_file(dataset_name)
        if checkpoint_file.exists():
            try:
                results = load(str(checkpoint_file))
                logger.info(f"   [{dataset_name}] Loaded {len(results)} results from checkpoint")
            except Exception as e:
                logger.warning(f"   [{dataset_name}] Failed to load checkpoint: {e}")

        # Try to load from the result file.
        result_path = Path(cfg.result_file)
        if result_path.exists():
            try:
                data = load(str(result_path))
                if isinstance(data, pd.DataFrame):
                    existing_results = {
                        str(idx): pred
                        for idx, pred in zip(data['index'], data['prediction'])
                        if FAIL_MSG not in str(pred)
                    }
                    results.update(existing_results)
                    logger.info(f"   [{dataset_name}] Loaded {len(existing_results)} "
                                "results from result file")
            except Exception as e:
                logger.warning(f"   [{dataset_name}] Failed to load result file: {e}")

        return results

    def _save_checkpoint(self, dataset_name: str, result: dict):
        """Save results to checkpoint file threading-safety."""
        checkpoint_file = self._get_checkpoint_file(dataset_name)

        with self.file_locks[dataset_name]:
            if checkpoint_file.exists():
                results = load(str(checkpoint_file))
            else:
                results = {}

            results[result['index']] = result['prediction']

            dump(results, str(checkpoint_file))

    def _save_final_result(self, dataset_name: str) -> bool:
        """Save the final inference result file."""
        cfg = self.states[dataset_name]

        with self.file_locks[dataset_name]:
            if 'image' in cfg.dataset_obj.data:
                dataset_data = cfg.dataset_obj.data.drop('image', axis=1)
            else:
                dataset_data = cfg.dataset_obj.data.copy()

            missing_indices = []
            for idx in dataset_data['index']:
                idx_str = str(idx)
                if idx_str not in cfg.results_dict:
                    missing_indices.append(idx_str)

            if missing_indices:
                logger.warning(
                    f"   [{dataset_name}] Missing results for {len(missing_indices)} samples: "
                    f"{missing_indices[:5]}{'...' if len(missing_indices) > 5 else ''}"
                )
                return False

            predictions = [cfg.results_dict[str(idx)] for idx in dataset_data['index']]
            dataset_data['prediction'] = predictions

            dump(dataset_data, cfg.result_file)
            logger.info(f"   [{dataset_name}] Saved final results to {cfg.result_file}")

            # Delete checkpoint file.
            checkpoint_file = self._get_checkpoint_file(dataset_name)
            if checkpoint_file.exists():
                checkpoint_file.unlink()

            return True

    async def _producer(self):
        """Genearte all sampels to inference."""
        logger.info("📦 Initializing tasks and checking checkpoints...")

        for cfg in self.dataset_configs:
            dataset_name = cfg.dataset_name
            dataset_type = cfg.dataset_type

            logger.info(
                f"   [{dataset_name}] Start build prompt [type={dataset_type}]"
            )
            # 1. Try to load checkpoint
            existing_results = self._load_checkpoint(dataset_name)
            cfg.results_dict = existing_results
            cfg.processed = len(existing_results)
            cfg.total_samples = len(cfg.dataset_obj.data)

            # 2. Check whether to need infer.
            if not self.run_infer or cfg.processed == cfg.total_samples:
                if cfg.processed < cfg.total_samples:
                    logger.warning(f"   [{dataset_name}] is incompleted "
                                   f"({cfg.processed}/{cfg.total_samples}). "
                                   "The evaluation may be inaccurate.")
                else:
                    logger.info(f"   [{dataset_name}] All samples completed "
                                f"({cfg.processed}/{cfg.total_samples}). "
                                "Will trigger eval directly.")
                # Save result file if not exists.
                if not Path(cfg.result_file).exists():
                    self._save_final_result(dataset_name)
                # Trigger evaluation.
                asyncio.create_task(self._trigger_eval(dataset_name))
                continue

            # 3. Dispatch according to dataset type.
            if dataset_type == "video":
                tasks_generated = await self._produce_video_tasks(cfg, existing_results)
            elif dataset_type == "mt":
                tasks_generated = await self._produce_mt_tasks(cfg, existing_results)
            else:  # image
                tasks_generated = await self._produce_image_tasks(cfg, existing_results)

            # Skip evaluation if no prompt is built.
            if tasks_generated == 0:
                cfg.eval_status = EvalStatus.Skipped

            self.total_tasks_generated += tasks_generated
            logger.info(
                f"   [{dataset_name}] Generated {tasks_generated} tasks "
                f"(Skipped {cfg.processed}) [type={dataset_type}]"
            )

        # Stop sign
        if self.run_infer:
            for _ in range(self.concurrency):
                await self.queue.put(None)

        logger.info(f"🚀 Queue ready. Total pending tasks: {self.total_tasks_generated}")

    async def _produce_image_tasks(self, cfg: DatasetConfig, existing_results: dict) -> int:
        """Produce image dataset inference task."""
        model = cfg.model_obj
        dataset = cfg.dataset_obj
        dataset_name = cfg.dataset_name

        if hasattr(model, 'set_dump_image'):
            model.set_dump_image(dataset.dump_image)

        use_custom_prompt = (
            hasattr(model, 'use_custom_prompt')
            and model.use_custom_prompt(dataset_name)
        )
        if use_custom_prompt:
            logger.info(f"   [{dataset_name}] Using model custom prompt")
        else:
            logger.info(f"   [{dataset_name}] Using vanilla dataset prompt")

        tasks_generated = 0
        for i in range(len(dataset)):
            item = dataset[i]
            idx_str = str(item['index'])

            if idx_str in existing_results:
                continue

            try:
                if use_custom_prompt:
                    prompt_struct = model.build_prompt(item, dataset=dataset_name)
                else:
                    prompt_struct = dataset.build_prompt(item)
            except Exception as e:
                import traceback
                logger.error(f"   [{dataset_name}] Failed to build prompt "
                             f"for sample {idx_str}: {repr(e)}")
                logger.debug(traceback.format_exception(e))
                # Skip dataset if has fatal sample.
                return 0

            task = InferenceTask(
                dataset_name=dataset_name,
                model_name=cfg.model_name,
                sample_index=idx_str,
                prompt_struct=prompt_struct,
                dataset_type="image"
            )
            await self.queue.put(task)
            tasks_generated += 1

        return tasks_generated

    async def _produce_video_tasks(self, cfg: DatasetConfig, existing_results: dict) -> int:
        """Produce video dataset inference task."""
        loop = asyncio.get_running_loop()
        model = cfg.model_obj
        dataset = cfg.dataset_obj
        dataset_name = cfg.dataset_name

        if cfg.video_llm is not None:
            video_llm = cfg.video_llm
        else:
            video_llm = getattr(model, 'VIDEO_LLM', False)

        logger.info(f"   [{dataset_name}] Video mode: video_llm={video_llm}")

        tasks_generated = 0
        for i in range(len(dataset)):
            item = dataset[i]
            idx_str = str(item['index'])

            if idx_str in existing_results:
                continue

            try:
                # Use process executor to produce video sample since it's heavy.
                prompt_struct = await loop.run_in_executor(
                    self.producer_executor,
                    partial(dataset.build_prompt, i, video_llm=video_llm),
                )
                if prompt_struct is None:
                    continue
            except Exception as e:
                import traceback
                logger.error(f"   [{dataset_name}] Failed to build prompt "
                             f"for sample {idx_str}: {repr(e)}")
                logger.debug(traceback.format_exception(e))
                # Skip dataset if has fatal sample.
                return 0

            task = InferenceTask(
                dataset_name=dataset_name,
                model_name=cfg.model_name,
                sample_index=idx_str,
                prompt_struct=prompt_struct,
                dataset_type="video"
            )
            await self.queue.put(task)
            tasks_generated += 1

        return tasks_generated

    async def _produce_mt_tasks(self, cfg: DatasetConfig, existing_results: dict) -> int:
        """Produce multi-turns dataset inference task."""
        dataset = cfg.dataset_obj
        dataset_name = cfg.dataset_name

        logger.info(f"   [{dataset_name}] Multi-turn dialogue mode")

        tasks_generated = 0
        for i in range(len(dataset)):
            item = dataset[i]
            idx_str = str(item['index'])

            if idx_str in existing_results:
                continue

            try:
                prompt_struct = dataset.build_prompt(item)
            except Exception as e:
                import traceback
                logger.error(f"   [{dataset_name}] Failed to build prompt "
                             f"for sample {idx_str}: {repr(e)}")
                logger.debug(traceback.format_exception(e))
                # Skip dataset if has fatal sample.
                return 0

            task = InferenceTask(
                dataset_name=dataset_name,
                model_name=cfg.model_name,
                sample_index=idx_str,
                prompt_struct=prompt_struct,
                dataset_type="mt"
            )
            await self.queue.put(task)
            tasks_generated += 1

        return tasks_generated

    async def _worker(self):
        """Worker to inference every infernece tasks."""
        loop = asyncio.get_running_loop()

        while True:
            task = await self.queue.get()
            if task is None:
                self.queue.task_done()
                break

            self.active_workers += 1
            cfg = self.states[task.dataset_name]
            try:
                model = cfg.model_obj
                start_time = time.time()
                if not cfg.infer_start_time:
                    cfg.infer_start_time = start_time

                # Select inference style according ot dataset type.
                if task.dataset_type == "mt":
                    output = await loop.run_in_executor(
                        self.infer_executor,
                        partial(chat_mt,
                                model=model,
                                messages=task.prompt_struct,
                                dataset_name=task.dataset_name),
                    )
                else:
                    def inference_call():
                        return model.generate(
                            message=task.prompt_struct,
                            dataset=task.dataset_name
                        )

                    output = await loop.run_in_executor(self.infer_executor, inference_call)

                inference_time = time.time() - start_time

                # Save checkpoint
                result_item = {
                    "index": task.sample_index,
                    "prediction": output
                }
                self._save_checkpoint(task.dataset_name, result_item)

                # Update status
                cfg.results_dict[task.sample_index] = output
                cfg.infer_end_time = time.time()
                cfg.infer_total_time += inference_time
                cfg.infer_count += 1
                cfg.processed += 1

                if cfg.verbose:
                    output_preview = str(output) if output else ""
                    if len(output_preview) > 100:
                        output_preview = output_preview[:100] + '...'
                    logger.info(
                        f"[{task.dataset_name}] Sample {task.sample_index}: "
                        f"{output_preview} (took {inference_time:.2f}s)")

                # Trigger evaluation if all tasks of the dataset is done.
                if cfg.processed == cfg.total_samples and cfg.eval_status == EvalStatus.Pending:
                    if self._save_final_result(task.dataset_name):
                        asyncio.create_task(self._trigger_eval(task.dataset_name))

            except Exception as e:
                logger.error(
                    f"❌ Worker error on {task.dataset_name}/{task.sample_index}: {e}",
                    exc_info=True
                )
            finally:
                self.active_workers -= 1
                self.queue.task_done()

    async def _trigger_eval(self, dataset_name: str):
        """Evaluate the specified dataset."""
        cfg = self.states[dataset_name]

        # Avoid multiple trigger.
        if cfg.eval_status != EvalStatus.Pending:
            return

        cfg.eval_status = EvalStatus.Running
        cfg.eval_start_time = time.time()

        # Create evaluation log.
        eval_log_dir = Path(cfg.work_dir) / 'eval_logs'
        eval_log_dir.mkdir(parents=True, exist_ok=True)
        eval_log_path = str(eval_log_dir / f"{cfg.model_name}_{dataset_name}_eval.log")
        logger.info(f"    [{dataset_name}] Trigger evaluation.")

        if self.debug:
            await self._run_eval_in_main_process(dataset_name)
        else:
            await self._run_eval_in_subprocess(dataset_name, eval_log_path)

    async def _run_eval_in_main_process(self, dataset_name: str):
        cfg = self.states[dataset_name]
        logger.info(f"🔔 [Eval Start - Debug Mode] {dataset_name}")

        def run_eval():
            try:
                result = cfg.dataset_obj.evaluate(cfg.result_file, **cfg.judge_kwargs)
                return {'success': True, 'result': result, 'error': None}

            except Exception as e:
                import traceback
                error_msg = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
                return {'success': False, 'result': None, 'error': error_msg}

        try:
            eval_result = run_eval()
            self._handle_eval_result(dataset_name, eval_result)

        except Exception as e:
            cfg.eval_status = EvalStatus.Error
            cfg.eval_duration = time.time() - cfg.eval_start_time
            logger.error(f"❌ [Eval Failed] {dataset_name}: {e}")

    async def _run_eval_in_subprocess(self, dataset_name: str, eval_log_path: str):
        cfg = self.states[dataset_name]

        loop = asyncio.get_running_loop()

        try:
            eval_result = await loop.run_in_executor(
                self.eval_executor,
                _eval_subprocess_target,
                cfg.dataset_obj,
                cfg.result_file,
                cfg.judge_kwargs,
                eval_log_path,
            )
            self._handle_eval_result(dataset_name, eval_result, eval_log_path)

        except Exception as e:
            cfg.eval_status = EvalStatus.Error
            cfg.eval_duration = time.time() - cfg.eval_start_time
            logger.error(
                f"❌ [Eval Failed] {dataset_name}: {e}\n"
                f"   Check log file: {eval_log_path}"
            )

    def _handle_eval_result(self,
                            dataset_name: str,
                            eval_result: dict | pd.DataFrame,
                            eval_log_path: str | None = None):
        """After evaluation"""
        cfg = self.states[dataset_name]

        if eval_result['success']:
            cfg.eval_status = EvalStatus.Done
            cfg.eval_duration = time.time() - cfg.eval_start_time
            logger.info(
                f"✅ [Eval Finish] {dataset_name} (Took {cfg.eval_duration:.1f}s)"
            )

            # Print evaluation result.
            result = eval_result['result']
            if isinstance(result, dict):
                logger.info(f"   Results: {json.dumps(result, indent=2, default=str)}")
            else:
                logger.info(f"   Results:\n{result}")
        else:
            cfg.eval_status = EvalStatus.Error
            cfg.eval_duration = time.time() - cfg.eval_start_time
            logger.error(
                f"❌ [Eval Failed] {dataset_name}: {eval_result['error']}\n"
                f"   Check log file: {eval_log_path}"
            )

        # Release dataset data after evaluation.
        self._release_dataset_memory(cfg)

    async def _monitor_loop(self):
        """monitor task progress in loop."""
        last_log = 0
        while True:
            # Exit if all infer tasks done & all eval tasks done.
            all_eval_done = all(
                cfg.eval_status in [EvalStatus.Done, EvalStatus.Error, EvalStatus.Skipped]
                for cfg in self.states.values()
            )

            if self.all_infer_done and all_eval_done:
                break

            if time.time() - last_log > self.monitor_interval:
                if not self.all_infer_done:
                    logger.info(f'Infer task queue: {self.queue.qsize()}')
                    logger.info(f'Active workers: {self.active_workers}')
                if not all_eval_done:
                    logger.info(', '.join(f'{cfg.dataset_name}: {cfg.eval_status.name}'
                                          for cfg in self.states.values()))
                self._log_snapshot()
                last_log = time.time()

            await asyncio.sleep(1)

    def _log_snapshot(self):
        elapsed = time.time() - self.start_time
        lines = []

        total_processed = 0
        total_samples = 0

        for cfg in self.states.values():
            total_processed += cfg.processed
            total_samples += cfg.total_samples

            if cfg.total_samples > 0:
                infer_pct = cfg.processed / cfg.total_samples
                infer_pct = math.floor(infer_pct * 1000) / 10
            else:
                infer_pct = 0
            infer_str = f"{infer_pct:>5.1f}% ({cfg.processed:>4}/{cfg.total_samples:<4})"

            # Inference time
            if cfg.infer_count > 0:
                if cfg.processed == cfg.total_samples:
                    infer_time = cfg.infer_end_time - cfg.infer_start_time
                else:
                    infer_time = time.time() - cfg.infer_start_time
                infer_str += f" ({infer_time:.0f}s)"

            # Evaluate status
            eval_str = cfg.eval_status.name
            if cfg.eval_status == EvalStatus.Running:
                dur = time.time() - cfg.eval_start_time
                eval_str = f"Running ({dur:.0f}s)"
            elif cfg.eval_status == EvalStatus.Done:
                eval_str = f"Done ({cfg.eval_duration:.1f}s)"

            # Format output
            name_str = cfg.dataset_name[:20].ljust(20)
            line = [name_str, f"Infer: {infer_str} | Eval: {eval_str:<20}"]
            lines.append(line)

        # Global progress.
        global_pct = (total_processed / total_samples * 100) if total_samples > 0 else 0
        global_str = f"Infer: {global_pct:>5.1f}% ({total_processed:>4}/{total_samples:<4})"

        table = tabulate(
            lines, [f"Elapsed: {elapsed:.0f}s", global_str], tablefmt='simple_outline')
        logger.info('\n' + table)

    async def run(self):
        self.start_time = time.time()

        logger.info("  API Inference & Evaluation Pipeline")
        logger.info(f"  Datasets: {len(self.dataset_configs)}")
        logger.info(f"  Concurrency: {self.concurrency}")
        logger.info(f"  Infer: {self.run_infer}")
        logger.info(f"  Eval: {self.run_eval}")

        # Start producer
        producer_task = asyncio.create_task(self._producer())

        # Start monitor
        monitor_task = asyncio.create_task(self._monitor_loop())
        try:
            # Start consumer
            if self.run_infer:
                workers = [
                    asyncio.create_task(self._worker())
                    for _ in range(self.concurrency)
                ]
                await producer_task
                self.all_infer_done = True
                await asyncio.gather(*workers)
                logger.info("🎉 All inference tasks finished. Waiting for pending evaluations...")
            else:
                await producer_task
                self.all_infer_done = True
                logger.info("📊 Eval mode only. Skipping inference...")

            await monitor_task

            # Final report
            self._log_snapshot()
            total_time = time.time() - self.start_time
            logger.info(f"  🏁 Pipeline Completed in {total_time:.1f}s")
        finally:
            # Clean resources
            self._shutdown_executors()
