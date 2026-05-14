from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from cosmos.data.vfm.action.action_normalization import load_action_stats
from cosmos.data.vfm.action.cosmos3_action_lerobot import (
    ActionNormalization,
    BaseActionLeRobotDataset,
)


class _StatsOnlyDataset(BaseActionLeRobotDataset):
    """Minimal shell for testing BaseActionLeRobotDataset normalization helpers."""

    def __init__(self, stats_path: Path, action_normalization: ActionNormalization | None) -> None:
        self._stats_path = stats_path
        self._action_normalization = action_normalization
        self._norm_stats: dict[str, torch.Tensor] | None = None

    def _normalizer_path(self) -> Path:
        return self._stats_path


def _write_stats(tmp_path: Path) -> Path:
    stats_path = tmp_path / "stats.json"
    payload = {
        "global": {
            "mean": [0.0, 0.0],
            "std": [1.0, 1.0],
            "min": [0.0, -1.0],
            "max": [10.0, 1.0],
            "q01": [0.0, -1.0],
            "q99": [10.0, 1.0],
        },
        "global_raw": {
            "mean": [0.0, 0.0],
            "std": [1.0, 1.0],
            "min": [0.0, 0.0],
            "max": [10.0, 100.0],
            "q01": [0.0, 0.0],
            "q99": [10.0, 100.0],
        },
    }
    stats_path.write_text(json.dumps(payload))
    return stats_path


@pytest.mark.L0
def test_load_action_stats_selects_global_raw_block(tmp_path: Path) -> None:
    stats_path = _write_stats(tmp_path)

    stats = load_action_stats(str(stats_path), stats_key="global_raw")

    assert stats["q99"].tolist() == [10.0, 100.0]


@pytest.mark.L0
def test_quantile_rot_uses_global_raw_stats(tmp_path: Path) -> None:
    stats_path = _write_stats(tmp_path)
    action = torch.tensor([[5.0, 50.0]], dtype=torch.float32)  # [T,D]

    quantile_dataset = _StatsOnlyDataset(stats_path, "quantile")
    quantile_rot_dataset = _StatsOnlyDataset(stats_path, "quantile_rot")

    quantile_action = quantile_dataset._normalize_action(action)  # [T,D]
    quantile_rot_action = quantile_rot_dataset._normalize_action(action)  # [T,D]
    expected_quantile = torch.tensor([[0.0, 1.0]], dtype=torch.float32)  # [T,D]
    expected_quantile_rot = torch.tensor([[0.0, 0.0]], dtype=torch.float32)  # [T,D]

    torch.testing.assert_close(quantile_action, expected_quantile)
    torch.testing.assert_close(quantile_rot_action, expected_quantile_rot)
