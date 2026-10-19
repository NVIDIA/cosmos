#!/usr/bin/env python3
"""Run PAIBench AMT motion smoothness without a distributed process group."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import queue
import sys
import traceback
from pathlib import Path
from statistics import fmean


RUNNER_VERSION = 2
# Quick second attempt before recording a video as permanently errored; AMT
# scoring failures (decoder hiccups, transient CUDA errors) are often transient.
VIDEO_RETRIES = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluator-root", type=Path, required=True)
    parser.add_argument("--videos-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--result-file", type=Path, required=True)
    parser.add_argument("--task", choices=("t2v", "i2v"), required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def read_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = json.loads(path.read_text())
    if not isinstance(records, list):
        raise TypeError(f"expected a list in {path}")
    return records


def write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def discover_videos(videos_dir: Path) -> list[Path]:
    if not videos_dir.is_dir():
        raise FileNotFoundError(f"organized videos directory is missing: {videos_dir}")
    videos = sorted(
        (path.absolute() for path in videos_dir.iterdir() if path.suffix.lower() == ".mp4" and path.is_file()),
        key=lambda path: path.name,
    )
    if not videos:
        raise ValueError(f"no MP4 videos found in {videos_dir}")
    names = [path.name for path in videos]
    if len(names) != len(set(names)):
        raise ValueError(f"duplicate video names found in {videos_dir}")
    return videos


def file_identity(path: Path) -> dict:
    stat = path.stat()
    return {
        "path": str(path),
        "resolved_path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def build_manifest(
    *,
    task: str,
    model_name: str,
    videos: list[Path],
    evaluator_root: Path,
    config: Path,
    checkpoint: Path,
) -> dict:
    return {
        "runner_version": RUNNER_VERSION,
        "task": task,
        "model_name": model_name,
        "evaluator_root": str(evaluator_root),
        "config": file_identity(config),
        "checkpoint": file_identity(checkpoint),
        "videos": [file_identity(path) for path in videos],
    }


def prepare_manifest(output_dir: Path, result_file: Path, manifest: dict) -> None:
    manifest_path = output_dir / "manifest.json"
    worker_files = sorted(output_dir.glob("worker_*.json"))
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous != manifest:
            raise RuntimeError(
                f"manifest mismatch in {output_dir}; use a new output directory "
                "or restore the original inputs"
            )
        return
    if worker_files:
        raise RuntimeError(f"worker checkpoints exist without a manifest in {output_dir}")
    if result_file.exists():
        raise RuntimeError(f"result file exists without a matching manifest: {result_file}")
    write_json_atomic(manifest_path, manifest)


def collect_saved_records(output_dir: Path) -> dict[str, dict]:
    by_video: dict[str, dict] = {}
    for path in sorted(output_dir.glob("worker_*.json")):
        for record in read_records(path):
            video_path = str(Path(record["video_path"]).absolute())
            normalized = {
                "video_path": video_path,
                "video_results": float(record["video_results"]),
            }
            if video_path in by_video:
                raise ValueError(f"duplicate saved score for {video_path}")
            by_video[video_path] = normalized
    return by_video


def worker_main(
    worker_id: int,
    physical_gpu: str,
    evaluator_root: str,
    config: str,
    checkpoint: str,
    output_dir: str,
    task_queue: mp.Queue,
    status_queue: mp.Queue,
) -> None:
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = physical_gpu
        os.environ["PYTHONNOUSERSITE"] = "1"
        sys.path.insert(0, evaluator_root)

        from pbench.motion_smoothness import MotionSmoothness

        output_path = Path(output_dir) / f"worker_{worker_id:02d}.json"
        records = read_records(output_path)
        motion = MotionSmoothness(config, checkpoint, "cuda")
        status_queue.put(("ready", worker_id, physical_gpu, ""))

        while True:
            try:
                video_path = task_queue.get(timeout=5)
            except queue.Empty:
                continue
            if video_path is None:
                break
            final_error = ""
            for _attempt in range(1 + VIDEO_RETRIES):
                try:
                    score = float(motion.motion_score(video_path))
                except Exception:
                    final_error = traceback.format_exc()
                    continue
                records.append({"video_path": video_path, "video_results": score})
                write_json_atomic(output_path, records)
                status_queue.put(("done", worker_id, video_path, score))
                break
            else:
                status_queue.put(("error", worker_id, video_path, final_error))
        status_queue.put(("exit", worker_id, physical_gpu, ""))
    except Exception:
        status_queue.put(("fatal", worker_id, physical_gpu, traceback.format_exc()))
        raise


def write_merged_result(
    *,
    output_dir: Path,
    result_file: Path,
    videos: list[str],
    require_complete: bool = True,
) -> tuple[float, list[str]]:
    saved = collect_saved_records(output_dir)
    target = set(videos)
    unexpected = set(saved) - target
    if unexpected:
        raise RuntimeError(f"found {len(unexpected)} results outside the input manifest")
    missing = [video_path for video_path in videos if video_path not in saved]
    if missing and require_complete:
        raise RuntimeError(f"missing {len(missing)} motion-smoothness results")
    scored = [video_path for video_path in videos if video_path in saved]
    if not scored:
        raise RuntimeError("no motion-smoothness results were computed")
    ordered = [saved[video_path] for video_path in scored]
    mean_score = fmean(record["video_results"] for record in ordered)
    payload: dict[str, object] = {"motion_smoothness": [mean_score, ordered]}
    if missing:
        payload["missing_videos"] = missing
    write_json_atomic(result_file, payload)
    return mean_score, missing


def main() -> None:
    args = parse_args()
    evaluator_root = args.evaluator_root.resolve()
    videos_dir = args.videos_dir.resolve()
    config = args.config.resolve()
    checkpoint = args.checkpoint.resolve()
    output_dir = args.output_dir.resolve()
    result_file = args.result_file.resolve()

    for label, path in (
        ("evaluator root", evaluator_root),
        ("AMT config", config),
        ("AMT checkpoint", checkpoint),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{label} is missing: {path}")
    if not args.model_name or Path(args.model_name).name != args.model_name:
        raise ValueError("--model-name must be one path component")
    if args.expected_count <= 0:
        raise ValueError("--expected-count must be positive")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")

    gpu_ids = [item.strip() for item in args.gpus.split(",") if item.strip()]
    if not gpu_ids:
        raise ValueError("--gpus must select at least one GPU")
    if len(gpu_ids) != len(set(gpu_ids)):
        raise ValueError("--gpus contains duplicate GPU identifiers")

    video_paths = discover_videos(videos_dir)
    if args.limit is not None:
        video_paths = video_paths[: args.limit]
    if len(video_paths) != args.expected_count:
        raise ValueError(
            f"expected {args.expected_count} videos, found {len(video_paths)} "
            "after applying --limit"
        )
    videos = [str(path) for path in video_paths]

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(
        task=args.task,
        model_name=args.model_name,
        videos=video_paths,
        evaluator_root=evaluator_root,
        config=config,
        checkpoint=checkpoint,
    )
    prepare_manifest(output_dir, result_file, manifest)

    saved = collect_saved_records(output_dir)
    target = set(videos)
    unexpected = set(saved) - target
    if unexpected:
        raise ValueError(f"output directory contains {len(unexpected)} unexpected videos")
    pending = [video_path for video_path in videos if video_path not in saved]

    print(
        f"motion_smoothness task={args.task} model={args.model_name} "
        f"target={len(videos)} saved={len(saved)} pending={len(pending)} "
        f"workers={len(gpu_ids)}",
        flush=True,
    )

    run_failed = False
    if pending:
        context = mp.get_context("spawn")
        task_queue = context.Queue()
        status_queue = context.Queue()
        workers: list[mp.Process] = []
        try:
            for worker_id, gpu_id in enumerate(gpu_ids):
                process = context.Process(
                    target=worker_main,
                    args=(
                        worker_id,
                        gpu_id,
                        str(evaluator_root),
                        str(config),
                        str(checkpoint),
                        str(output_dir),
                        task_queue,
                        status_queue,
                    ),
                )
                process.start()
                workers.append(process)

            # Start consumers before filling the queue so large evaluations
            # cannot fill the queue pipe before workers begin consuming tasks.
            for video_path in pending:
                task_queue.put(video_path)
            for _ in gpu_ids:
                task_queue.put(None)

            completed = len(saved)
            finished_workers: set[int] = set()
            failures: list[tuple] = []
            while len(finished_workers) < len(workers):
                try:
                    message = status_queue.get(timeout=15)
                except queue.Empty:
                    for worker_id, process in enumerate(workers):
                        if worker_id not in finished_workers and not process.is_alive():
                            finished_workers.add(worker_id)
                            failures.append(
                                ("fatal", worker_id, process.exitcode, "worker exited without status")
                            )
                    continue

                kind, worker_id, subject, detail = message
                if kind == "ready":
                    print(f"worker={worker_id} gpu={subject} ready", flush=True)
                elif kind == "done":
                    completed += 1
                    print(
                        f"completed={completed}/{len(videos)} worker={worker_id} "
                        f"score={float(detail):.10f} video={subject}",
                        flush=True,
                    )
                elif kind in {"error", "fatal"}:
                    failures.append(message)
                    print(f"{kind} worker={worker_id} subject={subject}\n{detail}", flush=True)
                    if kind == "fatal":
                        finished_workers.add(worker_id)
                elif kind == "exit":
                    finished_workers.add(worker_id)

            for process in workers:
                process.join()
            bad_exit_codes = [process.exitcode for process in workers if process.exitcode != 0]
            run_failed = bool(bad_exit_codes or failures)
            if run_failed:
                print(
                    f"workers finished with failures={len(failures)} "
                    f"exit_codes={bad_exit_codes}; merging partial results",
                    flush=True,
                )
        except BaseException:
            for process in workers:
                if process.is_alive():
                    process.terminate()
            for process in workers:
                process.join()
            raise
        finally:
            task_queue.cancel_join_thread()
            status_queue.cancel_join_thread()
            task_queue.close()
            status_queue.close()

    # Persist whatever scored cleanly even when workers failed mid-run; the
    # missing entries stay discoverable via the result file's "missing_videos".
    mean_score, missing = write_merged_result(
        output_dir=output_dir,
        result_file=result_file,
        videos=videos,
        require_complete=not run_failed,
    )
    if run_failed:
        print(
            f"INCOMPLETE run: scored={len(videos) - len(missing)}/{len(videos)}; "
            f"partial results: {result_file} -- re-invoke the runner to rescore missing videos",
            flush=True,
        )
        sys.exit(1)
    print(f"motion_smoothness {mean_score:.12f}", flush=True)
    print(f"saved {result_file}", flush=True)


if __name__ == "__main__":
    main()
