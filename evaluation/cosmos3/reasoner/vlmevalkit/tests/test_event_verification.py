# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the real evaluator and file I/O without loading unrelated model registries."""

import importlib
import importlib.util
import sys
import types
from pathlib import Path

import pandas as pd
import pytest
from sklearn.metrics import classification_report


@pytest.fixture(scope="module")
def evaluator_modules():
    root = Path(__file__).resolve().parents[1]
    saved = {name: module for name, module in sys.modules.items()
             if name == "vlmeval" or name.startswith("vlmeval.")}
    # The package initializers eagerly import every model/dataset and change SSL
    # defaults. Resolve the real dataset, base class and I/O modules directly.
    for name, path in (("vlmeval", root / "vlmeval"),
                       ("vlmeval.dataset", root / "vlmeval/dataset")):
        package = types.ModuleType(name)
        package.__path__ = [str(path)]
        sys.modules[name] = package
    reporter_name = "event_verification_score_reporter"
    previous_reporter = sys.modules.get(reporter_name)
    try:
        evaluator = importlib.import_module("vlmeval.dataset.metropolis_event_verification")
        spec = importlib.util.spec_from_file_location(reporter_name, root / "cosmos_eval/parse_score.py")
        reporter = importlib.util.module_from_spec(spec)
        sys.modules[reporter_name] = reporter
        spec.loader.exec_module(reporter)
        yield evaluator, reporter
    finally:
        for name in list(sys.modules):
            if name == "vlmeval" or name.startswith("vlmeval."):
                del sys.modules[name]
        sys.modules.update(saved)
        if previous_reporter is None:
            sys.modules.pop(reporter_name, None)
        else:
            sys.modules[reporter_name] = previous_reporter


def evaluate_rows(tmp_path, modules, truth, predictions):
    evaluator, _ = modules
    # Dataset construction downloads videos; evaluation itself needs annotations only.
    dataset = object.__new__(evaluator.MetropolisEventVerification)
    dataset.data = pd.DataFrame({"index": [f"sample-{i}" for i in range(len(truth))], "answer": truth})
    source = tmp_path / "predictions.csv"
    pd.DataFrame({"index": [f"sample-{i}" for i in range(len(predictions))], "prediction": predictions}).to_csv(source, index=False)
    before = source.read_bytes()
    result = dataset.evaluate(str(source)).iloc[0].to_dict()
    assert source.read_bytes() == before
    saved = pd.read_csv(tmp_path / "predictions_acc.csv").iloc[0].to_dict()
    assert saved["macro_f1"] == pytest.approx(result["macro_f1"])
    return result


@pytest.mark.parametrize("truth,predictions,expected,valid", [
    (["yes", "yes", "yes"], ["yes", "uncertain", "?"], 0.5, 1),
    (["yes", "no", "no"], ["yes", "uncertain", "?"], 0.5, 1),
    (["yes", "no", "yes", "no"], ["yes", "no", "?", "?"], 2 / 3, 2),
    (["yes", "no"], ["?", "uncertain"], 0.0, 0),
    (["yes"], [None], 0.0, 0),
    (["no"], [""], 0.0, 0),
    (["yes", "no"], ["YES", "No."], 1.0, 2),
    (["yes", "no"], ["no", "yes"], 0.0, 2),
])
def test_abstentions_remain_in_scored_support(tmp_path, evaluator_modules, truth, predictions, expected, valid):
    result = evaluate_rows(tmp_path, evaluator_modules, truth, predictions)
    assert result["macro_f1"] == pytest.approx(expected)
    assert result["macro avg--f1-score"] == pytest.approx(expected)
    assert result["macro avg--support"] == len(truth)
    assert result["Valid Predictions"] == valid
    assert result["Total Samples"] == len(truth)
    assert not any(key.startswith("__invalid__") for key in result)
    _, reporter = evaluator_modules
    assert reporter.extract_overall_score(result, "VANTAGE_EventVerification") == pytest.approx(100 * expected)


@pytest.mark.parametrize("truth,predictions", [
    (["yes", "yes"], ["yes", "yes"]),
    (["no", "no"], ["no", "no"]),
    (["yes", "no", "yes"], ["yes", "yes", "no"]),
    (["yes", "yes"], ["no", "no"]),
])
def test_fully_valid_reports_preserve_existing_metrics(tmp_path, evaluator_modules, truth, predictions):
    result = evaluate_rows(tmp_path, evaluator_modules, truth, predictions)
    evaluator, _ = evaluator_modules
    expected = evaluator.flatten_dict(classification_report(truth, predictions, output_dict=True, zero_division=0))
    for key, value in expected.items():
        assert result[key] == pytest.approx(value)


def test_missing_ground_truth_retains_existing_exclusion(tmp_path, evaluator_modules):
    evaluator, _ = evaluator_modules
    dataset = object.__new__(evaluator.MetropolisEventVerification)
    dataset.data = pd.DataFrame({"index": [0], "answer": ["yes"]})
    path = tmp_path / "unmatched.csv"
    pd.DataFrame({"index": [0, 9], "prediction": ["yes", "?"], "answer": ["yes", None]}).to_csv(path, index=False)
    result = dataset.evaluate(str(path)).iloc[0]
    assert result["macro_f1"] == 1.0
    assert result["Valid Predictions"] == 1
    assert result["Total Samples"] == 2


def test_empty_input_remains_unscored(tmp_path, evaluator_modules):
    evaluator, _ = evaluator_modules
    dataset = object.__new__(evaluator.MetropolisEventVerification)
    dataset.data = pd.DataFrame(columns=["index", "answer"])
    path = tmp_path / "empty.csv"
    pd.DataFrame(columns=["index", "prediction"]).to_csv(path, index=False)
    result = dataset.evaluate(str(path)).iloc[0]
    assert pd.isna(result["macro_f1"])
    assert result["Valid Predictions"] == result["Total Samples"] == 0
