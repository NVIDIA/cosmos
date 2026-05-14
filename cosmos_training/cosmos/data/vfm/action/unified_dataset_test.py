from __future__ import annotations

import itertools
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch.utils.data import IterableDataset

from cosmos.data.vfm.action.unified_dataset import ActionUnifiedIterableDataset
from cosmos.model.vfm.omni_mot_model_test import _maybe_init_distributed

"""
pytest -v -s cosmos/data/vfm/action/unified_dataset_test.py --L0
pytest -v -s cosmos/data/vfm/action/unified_dataset_test.py::test_hare_niemeyer --L0
pytest -v -s cosmos/data/vfm/action/unified_dataset_test.py::test_assign_worker --L0
pytest -v -s cosmos/data/vfm/action/unified_dataset_test.py::test_iter --L0
pytest -v -s cosmos/data/vfm/action/unified_dataset_test.py::test_backward_compat --L0
pytest -v -s cosmos/data/vfm/action/unified_dataset_test.py::test_worker_init_fn --L0
pytest -v -s cosmos/data/vfm/action/unified_dataset_test.py::test_register_sources --L0
torchrun --nproc_per_node=4 --standalone -m pytest -v -s cosmos/data/vfm/action/unified_dataset_test.py::test_distributed_4rank_shard_assignment --L1
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
# Dataset entries are dicts with "name", "dataset", "ratio", and optionally
# "resolution". Resolution is always in the dict (use None when not needed);
# it is not passed to the underlying dataset or to wrap_dataset.


def _make_sample(
    video_shape: tuple[int, ...] = (3, 17, 256, 256),
    action_dim: int = 10,
    mode: str = "policy",
) -> dict:
    """Minimal sample dict compatible with ActionTransformPipeline."""
    action_length = video_shape[1] - 1
    return {
        "video": torch.randint(0, 256, video_shape, dtype=torch.uint8),
        "action": torch.randn(action_length, action_dim),
        "ai_caption": "A test caption.",
        "mode": mode,
        "domain_id": torch.tensor(0, dtype=torch.long),
    }


class _TaggedIterableDataset(IterableDataset):
    """Iterable dataset that tags samples with a ``dataset_tag``."""

    def __init__(self, tag: str, length: int = 50) -> None:
        self._tag = tag
        self._length = length

    def __len__(self) -> int:
        return self._length

    def __iter__(self):
        while True:
            sample = _make_sample()
            sample["dataset_tag"] = self._tag
            yield sample


class _FiniteIterableDataset(IterableDataset):
    """Iterable dataset that yields a fixed number of samples then stops."""

    def __init__(self, tag: str, count: int = 3) -> None:
        self._tag = tag
        self._count = count

    def __len__(self) -> int:
        return self._count

    def __iter__(self):
        for _ in range(self._count):
            sample = _make_sample()
            sample["dataset_tag"] = self._tag
            yield sample


class _ShardableDataset(IterableDataset):
    """Dataset with ``_all_shard_roots`` and ``_register_sources`` for testing."""

    def __init__(self, tag: str, num_shards: int = 10) -> None:
        self._tag = tag
        self._all_shard_roots = [{"shard_id": i} for i in range(num_shards)]
        self._registered_shards: list[int] = []

    def _register_sources(self, indices: list[int] | None = None) -> None:
        if indices is None:
            indices = list(range(len(self._all_shard_roots)))
        self._registered_shards = indices

    def __iter__(self):
        while True:
            sample = _make_sample()
            sample["dataset_tag"] = self._tag
            sample["registered_shards"] = list(self._registered_shards)
            yield sample


def _identity_transform(data_dict: dict, resolution: str | None = None) -> dict:
    return data_dict


def _make_unified(
    datasets: list[dict[str, Any]],
    shard_across_workers: bool = True,
) -> ActionUnifiedIterableDataset:
    transform = MagicMock(side_effect=_identity_transform)
    return ActionUnifiedIterableDataset(
        datasets=datasets,
        transform=transform,
        shard_across_workers=shard_across_workers,
    )


# ---------------------------------------------------------------------------
# Hare-Niemeyer rank allocation
# ---------------------------------------------------------------------------


@pytest.mark.L0
def test_hare_niemeyer_basic_proportional():
    """4 datasets, 8 ranks with ratios [4,2,1,1].

    Hare-Niemeyer with 1-rank guarantee:
    - Each dataset gets 1 guaranteed rank, leaving 4 to distribute.
    - Remaining 4 allocated proportionally: A=2.0, B=1.0, C=0.5, D=0.5
    - Floors: [2,1,0,0], leftover=1 goes to C (highest remainder).
    - Final: [3,2,2,1].
    """
    datasets = [
        {"name": "A", "ratio": 4},
        {"name": "B", "ratio": 2},
        {"name": "C", "ratio": 1},
        {"name": "D", "ratio": 1},
    ]
    ranges = ActionUnifiedIterableDataset._compute_rank_ranges(datasets, 8)
    counts = [end - start for start, end in ranges]
    print(f"  Counts: {counts}")
    assert counts == [3, 2, 2, 1]
    assert ranges[0] == (0, 3)
    assert ranges[1] == (3, 5)
    assert ranges[2] == (5, 7)
    assert ranges[3] == (7, 8)
    print("  Success: basic proportional allocation correct")


@pytest.mark.L0
def test_hare_niemeyer_minimum_one_rank():
    """Even with very skewed ratios, each dataset gets at least 1 rank."""
    datasets = [
        {"name": "big", "ratio": 100},
        {"name": "tiny", "ratio": 1},
    ]
    ranges = ActionUnifiedIterableDataset._compute_rank_ranges(datasets, 3)
    counts = [end - start for start, end in ranges]
    print(f"  Counts: {counts}")
    assert all(c >= 1 for c in counts), f"All datasets must get >= 1 rank: {counts}"
    assert sum(counts) == 3
    print("  Success: minimum 1-rank guarantee holds")


@pytest.mark.L0
def test_hare_niemeyer_insufficient_ranks():
    """world_size < num_datasets should raise ValueError."""
    datasets = [{"name": f"ds_{i}", "ratio": 1} for i in range(5)]
    with pytest.raises(ValueError, match="must be >= number of datasets"):
        ActionUnifiedIterableDataset._compute_rank_ranges(datasets, 3)
    print("  Success: insufficient ranks raises ValueError")


@pytest.mark.L0
def test_hare_niemeyer_total_matches_world_size():
    """Sum of allocated ranks always equals world_size."""
    for world_size in [4, 8, 16, 32, 64]:
        datasets = [
            {"name": "A", "ratio": 5},
            {"name": "B", "ratio": 3},
            {"name": "C", "ratio": 1},
        ]
        ranges = ActionUnifiedIterableDataset._compute_rank_ranges(datasets, world_size)
        total = sum(end - start for start, end in ranges)
        assert total == world_size, f"world_size={world_size}: total={total}"
        print(f"  world_size={world_size}: total={total} OK")
    print("  Success: totals match for all world sizes")


@pytest.mark.L0
def test_hare_niemeyer_contiguous_non_overlapping():
    """Ranges are contiguous and non-overlapping."""
    datasets = [
        {"name": "A", "ratio": 4},
        {"name": "B", "ratio": 2},
        {"name": "C", "ratio": 1},
    ]
    ranges = ActionUnifiedIterableDataset._compute_rank_ranges(datasets, 10)
    print(f"  Ranges: {ranges}")
    for i in range(1, len(ranges)):
        assert ranges[i][0] == ranges[i - 1][1], f"Gap between range {i - 1} and {i}"
    print("  Success: ranges are contiguous")


# ---------------------------------------------------------------------------
# assign_worker
# ---------------------------------------------------------------------------


@pytest.mark.L0
def test_assign_worker_dataset_assignment():
    """Each rank is assigned to the correct dataset family."""
    ds_a = _ShardableDataset("A", num_shards=20)
    ds_b = _ShardableDataset("B", num_shards=5)
    datasets = [
        {"name": "A", "dataset": ds_a, "ratio": 3, "resolution": None},
        {"name": "B", "dataset": ds_b, "ratio": 1, "resolution": None},
    ]

    # 4 ranks, ratios [3,1]: guarantee 1 each, remaining 2 -> A=1.5 B=0.5
    # floors [1,0], leftover 1 -> A. Final: [2,2] -> A=[0,1], B=[2,3]
    print("  Testing rank 0 -> dataset A...")
    unified = _make_unified(datasets)
    unified.assign_worker(worker_id=0, num_workers=2, rank=0, world_size=4)
    assert unified._dataset is ds_a

    print("  Testing rank 3 -> dataset B...")
    unified2 = _make_unified(
        [
            {"name": "A", "dataset": _ShardableDataset("A", num_shards=20), "ratio": 3, "resolution": None},
            {"name": "B", "dataset": ds_b, "ratio": 1, "resolution": None},
        ]
    )
    unified2.assign_worker(worker_id=0, num_workers=2, rank=3, world_size=4)
    assert unified2._dataset is ds_b
    print("  Success: rank-to-dataset assignment correct")


@pytest.mark.L0
def test_assign_worker_shard_round_robin():
    """Shards are distributed round-robin across the family's workers."""
    print("  1 rank, 4 workers, 10 shards...")
    ds = _ShardableDataset("A", num_shards=10)
    datasets = [{"name": "A", "dataset": ds, "ratio": 1, "resolution": None}]
    unified = _make_unified(datasets)

    # Worker 0 -> shards [0, 4, 8]
    unified.assign_worker(worker_id=0, num_workers=4, rank=0, world_size=1)
    print(f"  Worker 0 shards: {ds._registered_shards}")
    assert ds._registered_shards == [0, 4, 8]

    # Worker 1 -> shards [1, 5, 9]
    ds2 = _ShardableDataset("A", num_shards=10)
    datasets2 = [{"name": "A", "dataset": ds2, "ratio": 1, "resolution": None}]
    unified2 = _make_unified(datasets2)
    unified2.assign_worker(worker_id=1, num_workers=4, rank=0, world_size=1)
    print(f"  Worker 1 shards: {ds2._registered_shards}")
    assert ds2._registered_shards == [1, 5, 9]
    print("  Success: round-robin shard distribution correct")


@pytest.mark.L0
def test_assign_worker_shard_wrap_around():
    """When workers > shards, workers wrap around."""
    ds = _ShardableDataset("A", num_shards=3)
    datasets = [{"name": "A", "dataset": ds, "ratio": 1, "resolution": None}]
    unified = _make_unified(datasets)

    # Worker 5 of 8 -> family_worker_id = 5, range(5, 3, 8) empty -> fallback [5 % 3] = [2]
    unified.assign_worker(worker_id=5, num_workers=8, rank=0, world_size=1)
    print(f"  Worker 5 shards: {ds._registered_shards}")
    assert ds._registered_shards == [2]
    print("  Success: wrap-around fallback correct")


@pytest.mark.L0
def test_assign_worker_no_shard_roots():
    """Datasets without _all_shard_roots skip shard distribution (Case C).

    Most datasets (LIBERO, PushT, CameraDataset, AVDataset) don't populate
    _all_shard_roots. They are still assigned to a rank family via
    Hare-Niemeyer, but every worker in the family iterates the full dataset
    with different RNG seeds — no round-robin shard splitting occurs.
    See dataloader.md "Case C" for details.
    """
    ds = _TaggedIterableDataset("plain")
    datasets = [{"name": "plain", "dataset": ds, "ratio": 1, "resolution": None}]
    unified = _make_unified(datasets)

    unified.assign_worker(worker_id=0, num_workers=4, rank=0, world_size=1)
    assert unified._dataset is ds
    assert not hasattr(ds, "_registered_shards")
    print("  Success: no shard roots -> no registration")


@pytest.mark.L0
def test_assign_worker_unsharded_mode():
    """shard_across_workers=False: every worker loads all datasets (default).

    When shard_across_workers=False (the default), Hare-Niemeyer rank
    assignment is skipped entirely. Every worker on every rank sees all
    datasets, and __iter__ uses weighted random selection based on the
    ratio values. This is the safe default that works without any dataset
    changes — opt in to shard_across_workers=True for large sharded
    datasets like AgiBotWorld.
    """
    ds_a = _ShardableDataset("A", num_shards=10)
    ds_b = _ShardableDataset("B", num_shards=5)
    datasets = [
        {"name": "A", "dataset": ds_a, "ratio": 1},
        {"name": "B", "dataset": ds_b, "ratio": 1},
    ]
    unified = _make_unified(datasets, shard_across_workers=False)

    unified.assign_worker(worker_id=0, num_workers=4, rank=0, world_size=4)

    assert unified._dataset is None
    assert ds_a._registered_shards == list(range(10))
    assert ds_b._registered_shards == list(range(5))
    print("  Success: unsharded mode registers all shards")


@pytest.mark.L0
def test_assign_worker_multi_rank_family():
    """Shards distribute across workers on multiple ranks of the same family."""
    print("  2 ranks, 4 workers each, 20 shards -> family_total_workers = 8")

    # Rank 0, worker 2 -> family_worker_id = 0*4+2 = 2 -> shards [2, 10, 18]
    ds = _ShardableDataset("A", num_shards=20)
    datasets = [{"name": "A", "dataset": ds, "ratio": 1, "resolution": None}]
    unified = _make_unified(datasets)
    unified.assign_worker(worker_id=2, num_workers=4, rank=0, world_size=2)
    print(f"  Rank 0, Worker 2 shards: {ds._registered_shards}")
    assert ds._registered_shards == [2, 10, 18]

    # Rank 1, worker 1 -> family_worker_id = 1*4+1 = 5 -> shards [5, 13]
    ds2 = _ShardableDataset("A", num_shards=20)
    datasets2 = [{"name": "A", "dataset": ds2, "ratio": 1, "resolution": None}]
    unified2 = _make_unified(datasets2)
    unified2.assign_worker(worker_id=1, num_workers=4, rank=1, world_size=2)
    print(f"  Rank 1, Worker 1 shards: {ds2._registered_shards}")
    assert ds2._registered_shards == [5, 13]
    print("  Success: multi-rank family distribution correct")


# ---------------------------------------------------------------------------
# create_action_worker_init_fn
# ---------------------------------------------------------------------------


@pytest.mark.L0
def test_worker_init_fn_calls_assign_worker():
    """worker_init_fn calls dataset.assign_worker with correct args."""
    from cosmos.data.vfm.action.dataloaders import create_action_worker_init_fn

    seed = 42

    mock_dataset = MagicMock(spec=ActionUnifiedIterableDataset)
    mock_info = MagicMock()
    mock_info.dataset = mock_dataset
    mock_info.num_workers = 8

    with patch("cosmos.data.vfm.action.dataloaders.distributed") as mock_dist:
        mock_dist.get_rank.return_value = 3
        mock_dist.get_world_size.return_value = 8
        fn = create_action_worker_init_fn(seed)
        with patch("torch.utils.data.get_worker_info", return_value=mock_info):
            fn(worker_id=5)

    mock_dataset.assign_worker.assert_called_once_with(5, 8, 3, 8)
    print("  Success: assign_worker called with (worker_id=5, num_workers=8, rank=3, world_size=8)")


# ---------------------------------------------------------------------------
# BaseActionLeRobotDataset._register_sources
# ---------------------------------------------------------------------------


@pytest.mark.L0
def test_register_sources_all():
    """_register_sources() with no args registers all shard roots."""
    from cosmos.data.vfm.action.cosmos3_action_lerobot import BaseActionLeRobotDataset

    ds = MagicMock(spec=BaseActionLeRobotDataset)
    ds._all_shard_roots = ["/data/a", "/data/b", "/data/c"]
    ds._delta_timestamps = {"obs": [0.0, 0.1]}
    ds._tolerance_s = 0.01
    ds._enable_fast_init = False
    BaseActionLeRobotDataset._register_sources(ds, indices=None)
    print(f"  _register_source called {ds._register_source.call_count} times")
    assert ds._register_source.call_count == 3
    print("  Success: all 3 shard roots registered")


@pytest.mark.L0
def test_register_sources_subset():
    """_register_sources(indices=[1]) registers only the specified shard."""
    from cosmos.data.vfm.action.cosmos3_action_lerobot import BaseActionLeRobotDataset

    ds = MagicMock(spec=BaseActionLeRobotDataset)
    ds._all_shard_roots = ["/data/a", "/data/b"]
    ds._delta_timestamps = {"obs": [0.0]}
    ds._tolerance_s = 0.01
    ds._enable_fast_init = False
    BaseActionLeRobotDataset._register_sources(ds, indices=[1])
    ds._register_source.assert_called_once_with(
        root="/data/b",
        delta_timestamps={"obs": [0.0]},
        tolerance_s=0.01,
        dataset_label="b",
        prefetched_meta=None,
    )
    print("  Success: only shard [1] registered")


@pytest.mark.L0
def test_register_sources_empty():
    """_register_sources() on empty _all_shard_roots is a no-op."""
    from cosmos.data.vfm.action.cosmos3_action_lerobot import BaseActionLeRobotDataset

    ds = MagicMock(spec=BaseActionLeRobotDataset)
    ds._all_shard_roots = []
    BaseActionLeRobotDataset._register_sources(ds, indices=None)
    ds._register_source.assert_not_called()
    print("  Success: empty shard roots -> no calls")


# ---------------------------------------------------------------------------
# Distributed: 4-rank end-to-end test
# ---------------------------------------------------------------------------


@pytest.mark.L1
def test_distributed_4rank_shard_assignment():
    """End-to-end test: 4 ranks, 2 datasets, 2 workers per rank via InfiniteDataLoader.

    Verifies that across 4 real distributed ranks:
    - Hare-Niemeyer assigns ranks to the correct dataset family
    - assign_worker is called by create_action_worker_init_fn inside DataLoader workers
    - Each rank's workers only yield samples from the assigned dataset
    - Shardable datasets get round-robin shard distribution

    Run with:
        torchrun --nproc_per_node=4 --standalone -m pytest -v -s \\
            cosmos/data/vfm/action/unified_dataset_test.py::test_distributed_4rank_shard_assignment --L1
    """
    _maybe_init_distributed()

    world_size = torch.distributed.get_world_size()
    if world_size != 4:
        pytest.skip(f"This test requires exactly 4 ranks (got world_size={world_size})")

    rank = torch.distributed.get_rank()

    from cosmos.data.vfm.action.dataloaders import InfiniteDataLoader, create_action_worker_init_fn

    # Dataset A: shardable with 20 shards, ratio=3
    # Dataset B: plain iterable (no shards), ratio=1
    # Hare-Niemeyer with 4 ranks, ratios [3,1]:
    #   guarantee 1 each -> remaining 2 -> A=1.5 B=0.5
    #   floors [1,0], leftover 1 -> A (tied remainder, lower index wins)
    #   Final: A=3 ranks [0,1,2], B=1 rank [3]
    ds_a = _ShardableDataset("A", num_shards=20)
    ds_b = _TaggedIterableDataset("B", length=100)

    datasets = [
        {"name": "A", "dataset": ds_a, "ratio": 3, "resolution": None},
        {"name": "B", "dataset": ds_b, "ratio": 1, "resolution": None},
    ]

    transform = MagicMock(side_effect=_identity_transform)
    unified = ActionUnifiedIterableDataset(
        datasets=datasets,
        transform=transform,
        shard_across_workers=True,
    )

    num_workers = 2
    loader = InfiniteDataLoader(
        dataset=unified,
        batch_size=1,
        num_workers=num_workers,
        worker_init_fn=create_action_worker_init_fn(seed=42),
    )

    # Consume samples and verify dataset assignment
    n_samples = 20
    samples = list(itertools.islice(loader, n_samples))
    assert len(samples) == n_samples, f"[Rank {rank}] Expected {n_samples} batches, got {len(samples)}"

    # Each sample is a collated batch dict; extract the dataset_tag
    tags = set()
    for batch in samples:
        if "dataset_tag" in batch:
            tag = batch["dataset_tag"]
            if isinstance(tag, list):
                tags.update(tag)
            else:
                tags.add(tag)

    # Ranks 0-2 -> dataset A, rank 3 -> dataset B
    if rank in (0, 1, 2):
        expected_tag = "A"
    else:
        expected_tag = "B"

    print(f"[Rank {rank}] Tags seen: {tags}, expected: {{{expected_tag}}}")
    assert tags == {expected_tag}, f"[Rank {rank}] Expected only '{expected_tag}' samples, got tags: {tags}"

    # For shardable dataset A (ranks 0-2): verify shards were distributed
    if rank in (0, 1, 2):
        # After workers ran, ds_a._registered_shards should have been set
        # by _register_sources in the worker processes. Since workers are
        # separate processes, we can't inspect ds_a directly. Instead we
        # check that the "registered_shards" key in the yielded samples
        # contains a non-empty subset.
        for batch in samples:
            if "registered_shards" in batch:
                shards = batch["registered_shards"]
                if isinstance(shards, torch.Tensor):
                    shards = shards.tolist()
                elif isinstance(shards, list) and len(shards) > 0 and isinstance(shards[0], torch.Tensor):
                    shards = [s.item() for s in shards]
                print(f"[Rank {rank}] Sample registered_shards: {shards}")
                assert len(shards) > 0, f"[Rank {rank}] Expected non-empty registered_shards"
                assert all(0 <= s < 20 for s in shards), f"[Rank {rank}] Shard indices out of range: {shards}"
                break

    # Barrier to ensure all ranks completed
    torch.distributed.barrier()
    print(f"[Rank {rank}] Success: distributed shard assignment test passed!")
