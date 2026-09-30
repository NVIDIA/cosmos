from __future__ import annotations

import json
import multiprocessing as mp
import queue
import sys
import types
from pathlib import Path

import pytest

MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))

from run_motion_smoothness_sharded import (  # noqa: E402
    collect_saved_records,
    worker_main,
    write_json_atomic,
    write_merged_result,
)


@pytest.mark.parametrize("bad_val", [float("nan"), float("inf"), float("-inf")])
def test_write_json_atomic_rejects_nan_and_inf(tmp_path: Path, bad_val: float) -> None:
    target = tmp_path / "result.json"
    with pytest.raises(
        ValueError, match="Out of range float values are not JSON compliant"
    ):
        write_json_atomic(target, {"score": bad_val})
    assert not target.exists()


def test_write_json_atomic_accepts_finite(tmp_path: Path) -> None:
    target = tmp_path / "result.json"
    write_json_atomic(target, {"score": 1.25, "mean": 0.0})
    assert target.exists()
    loaded = json.loads(target.read_text())
    assert loaded == {"score": 1.25, "mean": 0.0}


def test_collect_saved_records_accepts_finite(tmp_path: Path) -> None:
    worker_file = tmp_path / "worker_00.json"
    records = [
        {"video_path": "/path/video1.mp4", "video_results": 1.5},
        {"video_path": "/path/video2.mp4", "video_results": 0.0},
    ]
    write_json_atomic(worker_file, records)

    saved = collect_saved_records(tmp_path)
    assert len(saved) == 2
    assert saved["/path/video1.mp4"]["video_results"] == 1.5
    assert saved["/path/video2.mp4"]["video_results"] == 0.0


@pytest.mark.parametrize(
    "raw_json_value",
    ["NaN", "Infinity", "-Infinity"],
)
def test_collect_saved_records_rejects_non_finite(
    tmp_path: Path, raw_json_value: str
) -> None:
    worker_file = tmp_path / "worker_00.json"
    worker_file.write_text(
        f'[{{"video_path": "/path/video.mp4", "video_results": {raw_json_value}}}]'
    )

    with pytest.raises(ValueError, match="non-finite saved score"):
        collect_saved_records(tmp_path)


@pytest.fixture
def mock_evaluator(tmp_path: Path):
    """Create a mock pbench.motion_smoothness evaluator tree."""
    evaluator_root = tmp_path / "evaluator"
    pbench_dir = evaluator_root / "pbench"
    pbench_dir.mkdir(parents=True)
    (pbench_dir / "__init__.py").write_text("")
    return evaluator_root


def setup_mock_motion_smoothness(evaluator_root: Path, score_fn) -> None:
    pbench_module = types.ModuleType("pbench")
    motion_module = types.ModuleType("pbench.motion_smoothness")

    class MockMotionSmoothness:
        def __init__(self, config: str, checkpoint: str, device: str) -> None:
            self.config = config
            self.checkpoint = checkpoint
            self.device = device

        def motion_score(self, video_path: str) -> float:
            return score_fn(video_path)

    motion_module.MotionSmoothness = MockMotionSmoothness
    pbench_module.motion_smoothness = motion_module
    sys.modules["pbench"] = pbench_module
    sys.modules["pbench.motion_smoothness"] = motion_module


def teardown_mock_motion_smoothness() -> None:
    sys.modules.pop("pbench.motion_smoothness", None)
    sys.modules.pop("pbench", None)


def drain_queue(q: mp.Queue, expected_count: int, timeout: float = 2.0) -> list[tuple]:
    items = []
    for _ in range(expected_count):
        try:
            items.append(q.get(timeout=timeout))
        except queue.Empty:
            break
    return items


def test_worker_main_finite_score(mock_evaluator: Path, tmp_path: Path) -> None:
    setup_mock_motion_smoothness(mock_evaluator, lambda _: 2.5)
    try:
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        task_queue = mp.Queue()
        status_queue = mp.Queue()

        task_queue.put("/data/video_a.mp4")
        task_queue.put(None)

        worker_main(
            worker_id=0,
            physical_gpu="0",
            evaluator_root=str(mock_evaluator),
            config="mock_config.yaml",
            checkpoint="mock_ckpt.pt",
            output_dir=str(output_dir),
            task_queue=task_queue,
            status_queue=status_queue,
        )

        worker_file = output_dir / "worker_00.json"
        assert worker_file.exists()
        saved = json.loads(worker_file.read_text())
        assert len(saved) == 1
        assert saved[0] == {"video_path": "/data/video_a.mp4", "video_results": 2.5}

        messages = drain_queue(status_queue, expected_count=3)
        assert len(messages) == 3
        assert messages[0] == ("ready", 0, "0", "")
        assert messages[1] == ("done", 0, "/data/video_a.mp4", 2.5)
        assert messages[2] == ("exit", 0, "0", "")
    finally:
        teardown_mock_motion_smoothness()


@pytest.mark.parametrize("bad_score", [float("nan"), float("inf"), float("-inf")])
def test_worker_main_rejects_non_finite_scores(
    mock_evaluator: Path, tmp_path: Path, bad_score: float
) -> None:
    setup_mock_motion_smoothness(mock_evaluator, lambda _: bad_score)
    try:
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        task_queue = mp.Queue()
        status_queue = mp.Queue()

        task_queue.put("/data/video_bad.mp4")
        task_queue.put(None)

        worker_main(
            worker_id=0,
            physical_gpu="0",
            evaluator_root=str(mock_evaluator),
            config="mock_config.yaml",
            checkpoint="mock_ckpt.pt",
            output_dir=str(output_dir),
            task_queue=task_queue,
            status_queue=status_queue,
        )

        worker_file = output_dir / "worker_00.json"
        # Worker file should either not exist or have no records
        if worker_file.exists():
            records = json.loads(worker_file.read_text())
            assert len(records) == 0

        messages = drain_queue(status_queue, expected_count=3)
        kinds = [m[0] for m in messages]
        assert "ready" in kinds
        assert "error" in kinds
        assert "done" not in kinds
        assert "exit" in kinds

        error_msg = [m for m in messages if m[0] == "error"][0]
        assert error_msg[1] == 0
        assert error_msg[2] == "/data/video_bad.mp4"
        assert "ValueError: motion score is non-finite" in error_msg[3]
    finally:
        teardown_mock_motion_smoothness()


def test_write_merged_result_finite(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    result_file = tmp_path / "results" / "merged.json"

    video1 = str((tmp_path / "v1.mp4").resolve())
    video2 = str((tmp_path / "v2.mp4").resolve())

    write_json_atomic(
        output_dir / "worker_00.json",
        [{"video_path": video1, "video_results": 1.0}],
    )
    write_json_atomic(
        output_dir / "worker_01.json",
        [{"video_path": video2, "video_results": 2.0}],
    )

    mean = write_merged_result(
        output_dir=output_dir,
        result_file=result_file,
        videos=[video1, video2],
    )

    assert mean == 1.5
    assert result_file.exists()
    content = json.loads(result_file.read_text())
    assert content["motion_smoothness"][0] == 1.5
    assert len(content["motion_smoothness"][1]) == 2


def test_write_merged_result_rejects_non_finite_record(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    result_file = tmp_path / "results" / "merged.json"

    video1 = str((tmp_path / "v1.mp4").resolve())
    (output_dir / "worker_00.json").write_text(
        f'[{{"video_path": "{video1}", "video_results": NaN}}]'
    )

    with pytest.raises(ValueError, match="non-finite saved score"):
        write_merged_result(
            output_dir=output_dir,
            result_file=result_file,
            videos=[video1],
        )
    assert not result_file.exists()


def test_resume_retries_after_non_finite_error(
    mock_evaluator: Path, tmp_path: Path
) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    result_file = tmp_path / "results" / "merged.json"

    video1 = str((tmp_path / "v1.mp4").resolve())
    video2 = str((tmp_path / "v2.mp4").resolve())
    all_videos = [video1, video2]

    # Run 1: video1 succeeds (2.0), video2 yields NaN
    scores_run1 = {video1: 2.0, video2: float("nan")}
    setup_mock_motion_smoothness(mock_evaluator, lambda path: scores_run1[path])
    try:
        tq1 = mp.Queue()
        sq1 = mp.Queue()
        for v in all_videos:
            tq1.put(v)
        tq1.put(None)

        worker_main(
            worker_id=0,
            physical_gpu="0",
            evaluator_root=str(mock_evaluator),
            config="mock_config.yaml",
            checkpoint="mock_ckpt.pt",
            output_dir=str(output_dir),
            task_queue=tq1,
            status_queue=sq1,
        )

        saved = collect_saved_records(output_dir)
        assert len(saved) == 1
        assert video1 in saved
        assert video2 not in saved
    finally:
        teardown_mock_motion_smoothness()

    # Run 2 (resume): video1 is skipped because it is saved; video2 is retried and succeeds (1.0)
    pending = [v for v in all_videos if v not in saved]
    assert pending == [video2]

    scores_run2 = {video2: 1.0}
    setup_mock_motion_smoothness(mock_evaluator, lambda path: scores_run2[path])
    try:
        tq2 = mp.Queue()
        sq2 = mp.Queue()
        for v in pending:
            tq2.put(v)
        tq2.put(None)

        worker_main(
            worker_id=0,
            physical_gpu="0",
            evaluator_root=str(mock_evaluator),
            config="mock_config.yaml",
            checkpoint="mock_ckpt.pt",
            output_dir=str(output_dir),
            task_queue=tq2,
            status_queue=sq2,
        )

        saved_after = collect_saved_records(output_dir)
        assert len(saved_after) == 2
        assert saved_after[video1]["video_results"] == 2.0
        assert saved_after[video2]["video_results"] == 1.0

        mean = write_merged_result(
            output_dir=output_dir,
            result_file=result_file,
            videos=all_videos,
        )
        assert mean == 1.5
        assert result_file.exists()
    finally:
        teardown_mock_motion_smoothness()
