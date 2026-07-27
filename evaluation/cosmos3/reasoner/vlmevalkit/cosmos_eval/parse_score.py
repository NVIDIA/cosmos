# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Standalone score reporter for the cosmos_eval kit.

Reports the headline Overall score (0-100) plus every native sub-score key from a
`run.py --save-eval-results` evaluation output. No database, no parent
`vlmeval-metric` dependency — stdlib only.

This file is a VERBATIM VENDORED PORT of two internal modules, assembled by the
maintainer generator so the public scores match the internal pipeline exactly:

  * `vlmeval_metric/score_parser.py`  — all format detectors (A/B/C/D) + every
    per-benchmark override (`extract_overall_score`, `parse_scores`, ...).
  * `vlmeval_metric/eval_data.py`     — the `*.dict.eval.json` / `*.df.eval.json` loader.

Do not hand-edit the ported sections; regenerate. A maintainer CI gate runs this
port and the real modules on the same eval JSONs and asserts identical output.

Score formats handled (see `parse_scores`):
  A: flat scalar dict   B: multi-split arrays   C: indexed lists   D: simple numeric
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ===========================================================================
# Vendored verbatim from vlmeval_metric/eval_data.py
# ===========================================================================

@dataclass
class EvalSummary:
    """Evaluation summary with optional column metadata."""

    data: dict[str, Any]
    columns: list[str] = field(default_factory=list)


def load_eval_summary(filepath: Path) -> dict[str, Any]:
    """Load eval summary — backward compatible wrapper."""
    return load_eval_summary_with_metadata(filepath).data


def load_eval_summary_with_metadata(filepath: Path) -> EvalSummary:
    """Load eval summary preserving column names for .df.eval.json files.

    For .dict.eval.json: columns will be empty (keys are already column names).
    For .df.eval.json: columns contains the DataFrame column headers.
    """
    doc = json.loads(filepath.read_text("utf-8"))

    if filepath.name.endswith(".dict.eval.json"):
        return EvalSummary(data=doc)

    if not filepath.name.endswith(".df.eval.json"):
        raise ValueError("expected filepath to end with known suffix")

    columns = doc["columns"]
    index = [str(k) for k in doc["index"]]
    data = doc["data"]

    if len(columns) == 1:
        data = [x[0] for x in data]

    return EvalSummary(
        data=dict(zip(index, data, strict=False)),
        columns=[str(c) for c in columns],
    )


# ===========================================================================
# Vendored verbatim from vlmeval_metric/score_parser.py
# ===========================================================================

# Column names to search for the "overall" metric, in priority order.
ACCURACY_KEYS = [
    "CV-Bench Accuracy",
    "Overall",
    "overall",
    "English Overall Score",
    "Accuracy",
    "accuracy",
    "acc",
    "Acc",
]


def normalize_score(value: float) -> float:
    """Normalize score to 0-100 range.

    VLMEvalKit is inconsistent: some datasets store 0-1 (AI2D, MMBench),
    others store 0-100 (CountBenchQA, HallusionBench).
    Heuristic: values <= 1.0 are 0-1 scale, multiply by 100.
    """
    return value * 100 if value <= 1.0 else value


def _is_format_c(scores_dict: dict) -> bool:
    """Check if all non-metadata keys are numeric strings with list values."""
    non_meta_keys = [k for k in scores_dict if k != "__columns__"]
    if not non_meta_keys:
        return False
    if any(not k.isdigit() for k in non_meta_keys):
        return False
    return isinstance(scores_dict[non_meta_keys[0]], list)


def _detect_format(scores_dict: dict) -> str:
    """Detect which format the scores dict is in.

    Returns one of: "A", "B", "C", "D", "empty", "unknown".
    """
    if not scores_dict:
        return "empty"

    if _is_format_c(scores_dict):
        return "C"

    for key in ACCURACY_KEYS:
        if key in scores_dict and isinstance(scores_dict[key], list):
            return "B"

    for key in ACCURACY_KEYS:
        if key in scores_dict and isinstance(scores_dict[key], int | float):
            return "A"

    for key, value in scores_dict.items():
        if key != "__columns__" and isinstance(value, int | float):
            return "D"

    return "unknown"


def _parse_format_a(scores_dict: dict) -> dict[str, str]:
    """Parse flat dict with scalar values."""
    for key in ACCURACY_KEYS:
        if key in scores_dict and isinstance(scores_dict[key], int | float):
            score = normalize_score(scores_dict[key])
            return {"overall": f"{score:.2f}"}
    return {}


def _parse_format_b(scores_dict: dict) -> dict[str, str]:
    """Parse dict with array values (multi-split)."""
    result: dict[str, str] = {}
    splits = scores_dict.get("split", [])
    if not isinstance(splits, list):
        splits = [splits]

    for key in ACCURACY_KEYS:
        if key not in scores_dict:
            continue
        values = scores_dict[key]
        if not isinstance(values, list):
            continue

        if len(splits) == len(values):
            for split_name, val in zip(splits, values, strict=False):
                if isinstance(val, int | float):
                    score = normalize_score(val)
                    result[str(split_name)] = f"{score:.2f}"
        elif len(values) == 1 and isinstance(values[0], int | float):
            score = normalize_score(values[0])
            result["overall"] = f"{score:.2f}"

        if result:
            return result

    return {}


def _extract_accuracy_by_columns(values: list, columns: list[str]) -> float | None:
    """Column-aware accuracy extraction.

    columns[0] is the label/Category column; columns[1:] correspond to values[0:].
    Finds the LAST column whose name contains an accuracy keyword.
    """
    acc_col_names = {"accuracy", "acc", "overall", "iou"}
    data_columns = columns[1:]
    last_match: float | None = None
    for j, col in enumerate(data_columns):
        col_lower = str(col).lower()
        if any(name in col_lower for name in acc_col_names):
            if j < len(values) and isinstance(values[j], int | float):
                last_match = values[j]
    if last_match is not None:
        return last_match
    for j in range(len(values) - 1, -1, -1):
        if isinstance(values[j], int | float):
            return values[j]
    return None


def _extract_accuracy_by_heuristic(values: list) -> float | None:
    """Heuristic accuracy extraction when no column names available."""
    numeric = [v for v in values if isinstance(v, int | float)]
    if not numeric:
        return None
    if len(numeric) == 1:
        return numeric[0]

    small_values = [v for v in numeric if v <= 100]
    if small_values and len(small_values) < len(numeric):
        return small_values[-1]

    return numeric[0]


def _extract_accuracy_from_row(values: list, columns: list[str] | None = None) -> float | None:
    """Extract the accuracy value from a data row."""
    if columns is not None and len(columns) > 1:
        return _extract_accuracy_by_columns(values, columns)
    return _extract_accuracy_by_heuristic(values)


def _iter_format_c_rows(scores_dict: dict) -> list[tuple[str, list]]:
    """Iterate Format C rows in sorted order, skipping metadata."""
    rows = []
    for key, value in sorted(scores_dict.items(), key=lambda x: x[0]):
        if key == "__columns__":
            continue
        if isinstance(value, list) and len(value) >= 2:
            rows.append((key, value))
    return rows


def _get_column_list(scores_dict: dict) -> list[str] | None:
    """Extract __columns__ metadata if present."""
    columns = scores_dict.get("__columns__")
    if isinstance(columns, list) and columns:
        return [str(c) for c in columns]
    return None


def _parse_format_c(scores_dict: dict) -> dict[str, str]:
    """Parse indexed lists format."""
    col_list = _get_column_list(scores_dict)
    rows = _iter_format_c_rows(scores_dict)

    for _key, value in rows:
        label = value[0]
        if not isinstance(label, str):
            continue
        if label.lower() not in ("overall", "cv-bench accuracy"):
            continue
        acc = _extract_accuracy_from_row(value[1:], col_list)
        if acc is not None:
            return {"overall": f"{normalize_score(acc):.2f}"}

    for _key, value in rows:
        label = value[0] if isinstance(value[0], str) else "unknown"
        acc = _extract_accuracy_from_row(value[1:], col_list)
        if acc is not None:
            clean_label = label.lower().replace(" ", "_")
            return {clean_label: f"{normalize_score(acc):.2f}"}

    return {}


def parse_scores(scores_dict: dict[str, Any]) -> dict[str, str]:
    """Parse scores dict and extract metrics.

    Returns:
        Dictionary mapping metric names to formatted score strings
        (0-100 scale, 2 decimal places).
    """
    fmt = _detect_format(scores_dict)

    if fmt == "A":
        return _parse_format_a(scores_dict)
    if fmt == "B":
        return _parse_format_b(scores_dict)
    if fmt == "C":
        return _parse_format_c(scores_dict)
    if fmt == "D":
        for value in scores_dict.values():
            if isinstance(value, int | float):
                score = normalize_score(value)
                return {"overall": f"{score:.2f}"}

    return {}


# Dataset-specific score extraction overrides.
# Each entry maps dataset_name to a callable that extracts the overall score
# from the raw scores dict. Only needed for benchmarks where the generic
# parse_scores() logic doesn't pick the right metric.
SCORE_OVERRIDES: dict[str, Callable[[dict[str, Any]], float | None]] = {
    # MMMU_DEV_VAL: multi-split, use validation split
    "MMMU_DEV_VAL": lambda d: _extract_split_score(d, "validation"),
    # Anomaly-detection-style binary classification (sklearn classification_report
    # shape) — class distributions are heavily skewed, so macro-F1 is the honest
    # summary; accuracy/micro-F1 is flattered by the majority class.
    "TailgatingVerification": lambda d: _extract_key_score(d, "macro avg--f1-score"),
    "LVEventVerification": lambda d: _extract_key_score(d, "macro avg--f1-score"),
    "MetropolisEventVerification": lambda d: _extract_key_score(d, "macro avg--f1-score"),
    "VANTAGE_EventVerification": lambda d: _extract_key_score(d, "macro avg--f1-score"),
    "WarehouseNearMiss": lambda d: _extract_key_score(d, "macro avg--f1-score"),
    "ITSCollision": lambda d: _extract_key_score(d, "macro avg--f1-score"),
    "MetropolisDVC": lambda d: _extract_nested_score(d, "overall", "SODA_c"),
    "VANTAGE_DVC": lambda d: _extract_nested_score(d, "overall", "SODA_c"),
    # MVBench: overall is [correct, total, "pct%"], extract percentage
    "MVBench": lambda d: _extract_pct_from_list(d, "overall"),
    # Video-MME: nested {"overall": {"overall": "0.606"}}, use overall.overall
    "Video-MME": lambda d: _extract_nested_score(d, "overall", "overall"),
    # RefCOCO: Format C with columns [Split, Precision@1, Average IoU, Samples],
    # macro-average row labeled "Average". Extract Precision@1 from that row.
    "RefCOCO": lambda d: _extract_refcoco_precision(d),
    # IFBench: Format C with columns [strict, loose], single row.
    # Overall = average of strict and loose, normalized from 0-1 to 0-100.
    "IFBench": lambda d: _extract_ifbench_avg(d),
    # VideoPhy2: correlation score
    "VideoPhy2": lambda d: _extract_key_score(d, "Correlation"),
    # CausalVQA: average of unpaired and paired accuracy
    "CausalVQA": lambda d: _extract_avg_keys(d, "Unpaired Accuracy", "Paired Accuracy"),
    # MVPBench: average of single and pair accuracy
    "MVPBench": lambda d: _extract_avg_keys(d, "Single Accuracy", "Pair Accuracy"),
    # MetropolisTemporal: nested {"overall": {"iou": 0.45, ...}}
    "MetropolisTemporal": lambda d: _extract_nested_score(d, "overall", "iou"),
    "VANTAGE_Temporal": lambda d: _extract_nested_score(d, "overall", "iou"),
    # MetropolisVQA: general-purpose VQA, classes roughly balanced — accuracy
    # is the right summary (unlike the anomaly-detection benchmarks above).
    "MetropolisVQA": lambda d: _extract_key_score(d, "accuracy"),
    "VANTAGE_VQA": lambda d: _extract_key_score(d, "accuracy"),
    "Metropolis2DGrounding": lambda d: _extract_key_score(d, "Mean_IoU"),
    "VANTAGE_2DGrounding": lambda d: _extract_key_score(d, "Mean_IoU"),
    # ThreeDAVGroundingBench: the harness writes a transposed 1-row DF as a flat
    # dict — metric names are top-level keys (e.g. {"IoU Accuracy": 0.086, ...,
    # "__columns__": ["0"]}). `_extract_key_score` reads the headline directly.
    # Fixture: tests/fixtures/threedavgrounding_real_scores.json.
    "ThreeDAVGroundingBench": lambda d: _extract_key_score(d, "IoU Accuracy"),
    "Astro2DBench": lambda d: _extract_key_score(d, "f1"),
    "VANTAGE_Astro2D": lambda d: _extract_key_score(d, "f1"),
    "VANTAGE_SOT": lambda d: _extract_key_score(d, "Overall"),
    # WarehouseSpatialAI: custom key not in ACCURACY_KEYS
    "WarehouseSpatialAI": lambda d: _extract_key_score(d, "Overall_acc"),
    # LingoQA: benchmark_score (0-1 scale)
    "LingoQA": lambda d: _extract_key_score(d, "benchmark_score"),
    # AVSpecial*Bench: "Overall Accuracy" key (not in ACCURACY_KEYS)
    "AVSpecialCollisionBench": lambda d: _extract_key_score(d, "Overall Accuracy"),
    "AVSpecialStopBehaviorBench": lambda d: _extract_key_score(d, "Overall Accuracy"),
    # AVSpecialEnvironmentBench: pooled accuracy on 0-1 scale (normalize_score rescales to 0-100).
    "AVSpecialEnvironmentBench": lambda d: _extract_key_score(d, "accuracy"),
    # AVSpecialOODReasoningBench: mean Lingo-Judge sigmoid prob on 0-1 scale.
    "AVSpecialOODReasoningBench": lambda d: _extract_key_score(d, "lingo_judge_mean"),
    # AVPromptFollowingBench: sample-level success rate on 0-1 scale (normalize_score rescales to 0-100).
    "AVPromptFollowingBench": lambda d: _extract_key_score(d, "success_rate"),
    # AETCBench: harness writes a transposed 1-row DF as a flat dict — metric
    # names are top-level keys (e.g. {"bcq_accuracy": ..., "weighted_mean": ...,
    # "__columns__": ["0"]}). Pick "weighted_mean" as the headline.
    "AETCBench_all": lambda d: _extract_key_score(d, "weighted_mean"),
    # LVS_ai_hallucination: aggregate.avg_factual_accuracy on a 0-10 scale → 0-100.
    "LVS_ai_hallucination": lambda d: _extract_lvs_ai_hallucination_score(d),
    # CameraBench: aggregate.overall_f1 on a 0-1 scale → 0-100.
    "CameraBench": lambda d: _extract_camera_bench_score(d),
    # Cosmos-CAB-Video (both General and Camera variants): aggregate.overall_score on
    # a 0-1 scale → 0-100. General = F1(precision, recall); Camera = mean of three macro-F1.
    "Cosmos-CAB-Video_General": lambda d: _extract_cosmos_cab_score(d),
    "Cosmos-CAB-Video_Camera": lambda d: _extract_cosmos_cab_score(d),
    # Cosmos-CAB-Image: aggregate.overall_score = F1(precision, recall) on 0-1 → 0-100.
    "Cosmos-CAB-Image": lambda d: _extract_cosmos_cab_score(d),
    # LocateAnythingBench (both Box and Point variants): aggregate.overall_score on
    # a 0-1 scale (mean of per-dataset F1 across 7 anchor datasets), rescale to 0-100.
    "LocateAnythingBench-Box":   lambda d: _extract_locate_anything_bench_score(d),
    "LocateAnythingBench-Point": lambda d: _extract_locate_anything_bench_score(d),
    # ODinW13: evaluate() returns a wide DataFrame (rows = ['Overall', *13 datasets],
    # cols = ['mAP', 'mAP_50']) on the natural 0-1 mAP scale. Headline = average mAP at
    # scores['Overall'][0]; the override applies the single 0-1 -> 0-100 conversion.
    "ODinW13": lambda d: _extract_odinw_score(d),
}

# Prefix-based overrides: dataset names starting with these prefixes use the
# corresponding override. Allows e.g. MVBench_8frame, MVBench_64frame to share logic.
SCORE_OVERRIDE_PREFIXES: list[tuple[str, Callable[[dict[str, Any]], float | None]]] = [
    ("MVBench", SCORE_OVERRIDES["MVBench"]),
    ("Video-MME", SCORE_OVERRIDES["Video-MME"]),
    ("tailgating_", SCORE_OVERRIDES["TailgatingVerification"]),
    ("CausalVQA", SCORE_OVERRIDES["CausalVQA"]),
    ("MetropolisTemporal", SCORE_OVERRIDES["MetropolisTemporal"]),
    ("MetropolisVQA", SCORE_OVERRIDES["MetropolisVQA"]),
    ("AETCBench", SCORE_OVERRIDES["AETCBench_all"]),
    ("LVS_ai_hallucination", SCORE_OVERRIDES["LVS_ai_hallucination"]),
    ("CameraBench", SCORE_OVERRIDES["CameraBench"]),
    ("Cosmos-CAB-Video_", SCORE_OVERRIDES["Cosmos-CAB-Video_General"]),
    ("Cosmos-CAB-Image", SCORE_OVERRIDES["Cosmos-CAB-Image"]),
    ("LocateAnythingBench", SCORE_OVERRIDES["LocateAnythingBench-Box"]),
]


def _extract_key_score(scores_dict: dict[str, Any], key: str) -> float | None:
    """Extract a score by direct key lookup from flattened eval dict."""
    val = scores_dict.get(key)
    if isinstance(val, int | float):
        return normalize_score(val)
    return None


def _extract_df_split_column(scores_dict: dict[str, Any], column_name: str) -> float | None:
    """Extract a scalar from a one-row .df.eval.json by column name via __columns__."""
    columns = _get_column_list(scores_dict)
    if not columns or column_name not in columns:
        return None
    idx = columns.index(column_name)
    rows = _iter_format_c_rows(scores_dict)
    if not rows:
        return None
    _, row = rows[0]
    if idx >= len(row):
        return None
    val = row[idx]
    if isinstance(val, int | float):
        return normalize_score(val)
    return None


def _extract_avg_keys(scores_dict: dict[str, Any], *keys: str) -> float | None:
    """Average multiple keys and normalize to 0-100."""
    vals = [scores_dict[k] for k in keys if isinstance(scores_dict.get(k), int | float)]
    if len(vals) != len(keys):
        return None
    return normalize_score(sum(vals) / len(vals))


def _extract_split_score(scores_dict: dict[str, Any], split_name: str) -> float | None:
    """Extract score for a specific split from Format B (multi-split) data."""
    parsed = parse_scores(scores_dict)
    val = parsed.get(split_name)
    return float(val) if val else None


def _extract_pct_from_list(scores_dict: dict[str, Any], key: str) -> float | None:
    """Extract percentage from [correct, total, "pct%"] format like MVBench."""
    val = scores_dict.get(key)
    if not isinstance(val, list) or len(val) < 3:
        return None
    pct_str = val[2]
    if isinstance(pct_str, str) and pct_str.endswith("%"):
        try:
            return float(pct_str.rstrip("%"))
        except ValueError:
            return None
    # Fallback: compute from correct/total
    if isinstance(val[0], int | float) and isinstance(val[1], int | float) and val[1] > 0:
        return val[0] / val[1] * 100
    return None


def _extract_nested_score(scores_dict: dict[str, Any], outer_key: str, inner_key: str) -> float | None:
    """Extract a value from a nested dict like {"overall": {"mIoU": 0.5}} or {"overall": {"overall": "0.606"}}."""
    outer = scores_dict.get(outer_key)
    if isinstance(outer, dict):
        val = outer.get(inner_key)
        if isinstance(val, int | float):
            return normalize_score(val)
        if isinstance(val, str):
            try:
                return normalize_score(float(val))
            except ValueError:
                return None
    return None


def _extract_refcoco_precision(scores_dict: dict[str, Any]) -> float | None:
    """Extract Precision@1 from RefCOCO's macro-average row.

    RefCOCO outputs Format C with columns ["Split", "Precision@1", "Average IoU", "Samples"].
    The last row (label "Average") contains the macro-average across all 8 splits.
    """
    columns = _get_column_list(scores_dict)
    rows = _iter_format_c_rows(scores_dict)

    # Find the column index for "Precision@1"
    precision_idx: int | None = None
    if columns and len(columns) > 1:
        data_columns = columns[1:]  # columns[0] is the label column
        for j, col in enumerate(data_columns):
            if "precision" in col.lower():
                precision_idx = j
                break

    for _key, value in rows:
        label = value[0]
        if not isinstance(label, str) or label.lower() != "average":
            continue
        data = value[1:]
        if precision_idx is not None and precision_idx < len(data):
            val = data[precision_idx]
            if isinstance(val, int | float):
                return float(val)  # already 0-100 scale
        # Fallback: first numeric value in the row
        for v in data:
            if isinstance(v, int | float):
                return float(v)

    return None


def _extract_ifbench_avg(scores_dict: dict[str, Any]) -> float | None:
    """Extract average of strict and loose accuracy from IFBench.

    IFBench returns a 1-row, 2-column DataFrame which run.py transposes,
    producing: {"strict": scalar, "loose": scalar, "__columns__": [0]}.
    Overall = (strict + loose) / 2, normalized from 0-1 to 0-100.
    """
    strict = scores_dict.get("strict")
    loose = scores_dict.get("loose")
    if isinstance(strict, int | float) and isinstance(loose, int | float):
        return normalize_score((strict + loose) / 2)
    return None


def _extract_lvs_ai_hallucination_score(scores_dict: dict[str, Any]) -> float | None:
    """Extract aggregate.avg_factual_accuracy (0-10 scale) and rescale to 0-100."""
    aggregate = scores_dict.get("aggregate")
    if not isinstance(aggregate, dict):
        return None
    raw = aggregate.get("avg_factual_accuracy")
    if not isinstance(raw, int | float):
        return None
    return float(raw) * 10.0


def _extract_camera_bench_score(scores_dict: dict[str, Any]) -> float | None:
    """Extract the headline overall_f1 (0-100) for CameraBench from the wide
    df.eval.json shape: 'Overall' row, columns [precision, recall, f1, accuracy].
    """
    overall_row = scores_dict.get("Overall")
    columns = scores_dict.get("__columns__")
    if not isinstance(overall_row, list) or not isinstance(columns, list):
        return None
    try:
        idx = columns.index("f1")
    except ValueError:
        return None
    if idx >= len(overall_row):
        return None
    val = overall_row[idx]
    return float(val) if isinstance(val, int | float) else None


def _extract_cosmos_cab_score(scores_dict: dict[str, Any]) -> float | None:
    """Extract the headline f1 (0-100) for Cosmos-CAB benchmarks from the wide
    df.eval.json shape: 'Overall' row, columns starting with 'f1'.
    """
    overall_row = scores_dict.get("Overall")
    columns = scores_dict.get("__columns__")
    if not isinstance(overall_row, list) or not isinstance(columns, list):
        return None
    try:
        idx = columns.index("f1")
    except ValueError:
        return None
    if idx >= len(overall_row):
        return None
    val = overall_row[idx]
    return float(val) if isinstance(val, int | float) else None


def _extract_locate_anything_bench_score(scores_dict: dict[str, Any]) -> float | None:
    """Extract the headline overall_score (0-100) for LocateAnythingBench.

    `evaluate()` returns a wide DataFrame: rows = ['Overall', *7 datasets];
    cols start with the headline metric (avg_f1 for Box, f1 for Point).
    After `eval_data.load_eval_summary_with_metadata`, the 'Overall' row
    becomes a top-level list whose first element is the headline score
    (already on the 0-100 scale).
    """
    overall_row = scores_dict.get("Overall")
    if isinstance(overall_row, list) and overall_row and isinstance(overall_row[0], int | float):
        return float(overall_row[0])
    return None


def _extract_odinw_score(scores_dict: dict[str, Any]) -> float | None:
    """Extract the headline average mAP for ODinW13, converted to 0-100.

    `evaluate()` returns a wide DataFrame: rows = ['Overall', *13 datasets];
    cols = ['mAP', 'mAP_50'], values on the natural 0-1 mAP scale. After
    `load_eval_summary_with_metadata`, the 'Overall' row becomes a top-level list
    whose first element is the average mAP (0-1). `normalize_score` applies the
    single 0-1 -> 0-100 conversion; since mAP is always in [0,1] this is
    unambiguous and cannot double-scale.
    """
    overall_row = scores_dict.get("Overall")
    if isinstance(overall_row, list) and overall_row and isinstance(overall_row[0], int | float):
        return normalize_score(float(overall_row[0]))
    return None


def extract_overall_score(scores_dict: dict[str, Any], dataset_name: str = "") -> float:
    """Extract a single overall score from VLMEvalKit output.

    Uses dataset-specific overrides when available, otherwise falls back to
    generic parse_scores() logic. Returns the score on the 0-100 scale, or
    0.0 if extraction fails.
    """
    # Check dataset-specific override first (exact match, then prefix match)
    override = SCORE_OVERRIDES.get(dataset_name)
    if override is None:
        for prefix, fn in SCORE_OVERRIDE_PREFIXES:
            if dataset_name.startswith(prefix):
                override = fn
                break
    if override is not None:
        result = override(scores_dict)
        if result is not None:
            return result

    # Generic fallback
    parsed = parse_scores(scores_dict)
    if not parsed:
        return 0.0
    if "overall" in parsed:
        try:
            return float(parsed["overall"])
        except (ValueError, TypeError):
            pass
    score_str = next(iter(parsed.values()))
    try:
        return float(score_str)
    except (ValueError, TypeError):
        return 0.0


# --- RynnScale harness adapter (layer-1: native <suite>.json -> standardized scores) ---
# The ``rynnscale`` backend emits a native ``<suite>.json`` ({"metrics": {...}}) whose keys
# use RynnScale's own vocabulary ("Object Cognition", "traj", ...). These functions map that
# into standardized 0-100 component scores; the result then flows through the common
# ``extract_overall_score`` like every other backend. Ported verbatim from
# rynnscale-metric/rynnscale_metric/score_parser.py for score parity. NOTE: RynnScale always
# reports 0-1, so this uses an unconditional ``round(value*100, 2)`` — deliberately distinct
# from the heuristic ``normalize_score`` above (which serves VLMEvalKit's mixed 0-1/0-100 storage).

RYNNSCALE_BENCHMARK_REGISTRY: dict[str, dict[str, Any]] = {
    "RynnBrainCog": {
        "output_file": "RynnBrainCog.json",
        "components": {
            "RynnBrain-Object": "Object Cognition",
            "RynnBrain-Spatial": "Spatial Cognition",
        },
    },
    "RynnBrainLoc": {
        "output_file": "RynnBrainLoc.json",
        "components": {
            "RynnBrain-Grounding": "Object Referring",
            "RynnBrain-Area": "area",
            "RynnBrain-Affordance": "affordance",
            "RynnBrain-Trajectory": "traj",
        },
    },
}


def extract_rynnscale_suite_scores(metrics: dict[str, Any], suite_name: str) -> dict[str, float]:
    """Extract component scores for a RynnScale suite, normalized to 0-100.

    Returns a dict with component names as keys plus an "Overall" key for the average.
    """
    registry = RYNNSCALE_BENCHMARK_REGISTRY.get(suite_name)
    if not registry:
        return {}

    components = registry["components"]
    scores: dict[str, float] = {}
    values = []

    for component_name, output_key in components.items():
        if output_key in metrics:
            normalized = round(metrics[output_key] * 100.0, 2)  # RynnScale is always 0-1; see note above
            scores[component_name] = normalized
            values.append(normalized)

    if values:
        scores["Overall"] = round(sum(values) / len(values), 2)

    return scores


def extract_rynnscale_scores(
    raw_metrics: dict[str, dict[str, Any]],
    benchmarks: list[str],
) -> dict[str, dict[str, float]]:
    """Extract scores for all requested RynnScale suites.

    Args:
        raw_metrics: suite name -> full metrics dict from the RynnScale ``<suite>.json``.
        benchmarks: suite names to extract (e.g. ["RynnBrainCog", "RynnBrainLoc"]).

    Returns:
        suite name -> {component: score_0_100, "Overall": avg_0_100}.
    """
    result: dict[str, dict[str, float]] = {}
    for suite in benchmarks:
        if suite in raw_metrics:
            result[suite] = extract_rynnscale_suite_scores(raw_metrics[suite], suite)
    return result


# ===========================================================================
# cosmos_eval CLI: locate the eval output, report Overall + native sub-scores.
# (Mirrors vlmeval_run._finalize_scores: load summary, inject __columns__,
#  then extract_overall_score keyed on the benchmark name.)
# ===========================================================================

# Metadata keys injected alongside real scores; never reported as sub-scores.
_META_PREFIX = "__"


def find_eval_output(work_dir: Path) -> Path | None:
    """Newest `*/*/*.eval.json` under a run.py work-dir (matches the internal finder)."""
    files = list(Path(work_dir).glob("*/*/*.eval.json"))
    return max(files, key=lambda p: p.stat().st_mtime, default=None)


def load_scores(eval_path: Path) -> dict[str, Any]:
    """Load an eval output into the scores dict the parser expects (with __columns__)."""
    summary = load_eval_summary_with_metadata(Path(eval_path))
    scores = summary.data
    if summary.columns:
        scores["__columns__"] = summary.columns
    return scores


def _infer_dataset_name(eval_path: Path) -> str:
    """Best-effort benchmark name from a `<model>_<dataset>.{dict,df}.eval.json` filename.

    Used only when --dataset is omitted; the launcher always passes it explicitly
    (the per-benchmark score override is keyed on this name).
    """
    name = eval_path.name
    for suffix in (".dict.eval.json", ".df.eval.json"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return name.split("_", 1)[1] if "_" in name else name


def report(
    work_dir: str | None = None,
    eval_json: str | None = None,
    dataset_name: str = "",
) -> dict[str, Any]:
    """Resolve the eval output and return {eval_json, dataset, overall, subscores}."""
    eval_path = Path(eval_json) if eval_json else find_eval_output(Path(work_dir))
    if eval_path is None:
        return {"eval_json": None, "dataset": dataset_name, "overall": None, "subscores": {}}
    scores = load_scores(eval_path)
    dataset = dataset_name or _infer_dataset_name(eval_path)
    overall = extract_overall_score(scores, dataset_name=dataset)
    subscores = {
        k: v
        for k, v in scores.items()
        if not str(k).startswith(_META_PREFIX) and isinstance(v, (int, float)) and not isinstance(v, bool)
    }
    return {"eval_json": str(eval_path), "dataset": dataset, "overall": overall, "subscores": subscores}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Report Overall + sub-scores from a run.py eval output.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--work-dir", help="run.py output dir; the newest */*/*.eval.json is used")
    src.add_argument("--eval-json", help="explicit path to a *.eval.json")
    ap.add_argument("--dataset", default="", help="benchmark name (drives the per-benchmark score override)")
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = ap.parse_args(argv)

    r = report(work_dir=args.work_dir, eval_json=args.eval_json, dataset_name=args.dataset)
    if args.json:
        print(json.dumps(r))
        return
    if r["eval_json"] is None:
        print("no eval output found", file=sys.stderr)
        raise SystemExit(1)
    print(f"{r['dataset']}  Overall: {r['overall']:.2f}")
    for k, v in sorted(r["subscores"].items()):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
