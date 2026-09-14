# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Offline tests for the batch runner (no subprocess spawn, no NIM)."""

from __future__ import annotations

import pytest

import request_builders as builders
import run_all_examples as runner


@pytest.fixture
def valid_generator_metadata(fake_server):
    fake_server(
        "GET",
        "http://nim/v1/metadata",
        json_payload={
            "model_type": "generator",
            "inference_endpoint": "/v1/infer",
            "selectedModelProfileId": "p1",
            "model_variant": "nano",
        },
    )


def test_registry_contents():
    ids = [run["id"] for run in runner.RUNS]
    assert "t2v" in ids and "reasoner" in ids
    assert ids == [
        "t2v", "t2i", "i2v", "i2v_4step", "t2i_4step",
        "v2v", "transfer", "action", "reasoner",
    ]


def test_resolve_run():
    assert runner.resolve_run("t2v")["runtime"] == "generator"
    assert runner.resolve_run("reasoner")["runtime"] == "reasoner"
    assert runner.resolve_run("nope") is None


def test_preflight_ok_with_assets(valid_generator_metadata):
    selected = [runner.resolve_run("t2v")]
    results, runtime_ok = runner.preflight("http://nim", selected, quiet=True)
    assert runtime_ok is True
    assert results == []


def test_preflight_skips_all_on_no_runtime():
    # No endpoint reachable -> every run skipped, runtime_ok False.
    selected = [runner.resolve_run("v2v"), runner.resolve_run("reasoner")]
    results, runtime_ok = runner.preflight("http://invalid-nim", selected, quiet=True)
    assert runtime_ok is False
    assert len(results) == 2
    assert all(r.status == "skip" for r in results)


def test_preflight_skips_on_missing_asset(valid_generator_metadata, monkeypatch):
    # Fake metadata present but a required asset is gone -> that run skipped.
    def fake_check(rid):
        return ["/nonexistent/prompt.json"]

    monkeypatch.setattr(builders, "check_assets", fake_check)
    selected = [runner.resolve_run("t2v")]
    results, runtime_ok = runner.preflight("http://nim", selected, quiet=True)
    assert runtime_ok is True
    assert len(results) == 1
    assert results[0].rid == "t2v"
    assert results[0].status == "skip"
    assert "missing assets" in results[0].detail


def test_preflight_skips_variant_mismatch(fake_server):
    """A Generator serving super-i2v-4step skips runs needing other variants,
    but passes one whose allowed_variants contains the served variant."""
    fake_server(
        "GET",
        "http://nim/v1/metadata",
        json_payload={
            "model_type": "generator",
            "inference_endpoint": "/v1/infer",
            "selectedModelProfileId": "p1",
            "model_variant": "super-i2v-4step",
        },
    )
    # t2v allows (nano, super) -> skip. i2v_4step allows (super-i2v-4step,) -> ok.
    selected = [runner.resolve_run("t2v"), runner.resolve_run("i2v_4step")]
    results, runtime_ok = runner.preflight("http://nim", selected, quiet=True)
    assert runtime_ok is True
    by_id = {r.rid: r for r in results}
    assert "t2v" in by_id and by_id["t2v"].status == "skip"
    assert "served variant" in by_id["t2v"].detail
    assert "i2v_4step" not in by_id  # not skipped -> fine to run


def test_summary_lines_reports_totals():
    results = [
        runner.RunResult("t2v", "pass"),
        runner.RunResult("i2v", "fail", detail="exit 1"),
        runner.RunResult("reasoner", "skip", detail="no runtime"),
    ]
    lines = runner.summary_lines(results)
    assert any("passed: 1" in line for line in lines)
    assert any("failed: 1" in line for line in lines)
    assert any("skipped: 1" in line for line in lines)


def test_run_all_skips_when_no_runtime(fake_server):
    # No metadata registered -> connection error -> exit 3, all skipped.
    selected = [runner.resolve_run(rid) for rid in ("t2v", "reasoner")]
    results, code = runner.run_all("http://nim-down", selected=selected, timeout=5, verbose=False, quiet=True)
    assert code == 3
    assert all(r.status == "skip" for r in results)


def test_run_all_executes_and_passes(valid_generator_metadata, monkeypatch):
    # Stub the subprocess: simulate a script that exits 0.
    def fake_run(*args, **kwargs):
        class _P:
            returncode = 0
            stdout = "saved video to x.mp4\n"
            stderr = ""

        return _P()

    monkeypatch.setattr(runner, "run_in_subprocess", fake_run)
    selected = [runner.resolve_run("t2v")]
    results, code = runner.run_all("http://nim", selected=selected, timeout=5, verbose=False, quiet=True)
    assert code == 0
    assert results[0].status == "pass"


def test_run_all_exit_1_on_failure(valid_generator_metadata, monkeypatch):
    class _P:
        returncode = 1
        stdout = ""
        stderr = "boom"

    monkeypatch.setattr(runner, "run_in_subprocess", lambda *a, **k: _P())
    selected = [runner.resolve_run("t2v")]
    results, code = runner.run_all("http://nim", selected=selected, timeout=5, verbose=False, quiet=True)
    assert code == 1
    assert results[0].status == "fail"


def test_run_all_parallel_preserves_order(valid_generator_metadata, monkeypatch):
    """With parallel=True the summary keeps run order and all pass."""
    def fake_run(run, **kwargs):
        class _P:
            returncode = 0
            stdout = ""
            stderr = ""

        return _P()

    monkeypatch.setattr(runner, "run_in_subprocess", fake_run)
    selected = [runner.resolve_run(rid) for rid in ("t2v", "v2v", "transfer")]
    results, code = runner.run_all(
        "http://nim", selected=selected, timeout=5, verbose=False, quiet=True, parallel=True
    )
    assert code == 0
    assert [r.rid for r in results] == ["t2v", "v2v", "transfer"]
    assert all(r.status == "pass" for r in results)


def test_manifest_loads():
    """The manifest (runs.yaml) drives the registry."""
    runs = runner.load_runs()
    assert len(runs) == 9
    assert runs[0]["id"] == "t2v"
    assert runs[-1]["id"] == "reasoner"