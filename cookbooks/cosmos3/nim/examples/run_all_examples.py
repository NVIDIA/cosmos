# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Run the Cosmos3 NIM example clients in sequence against a NIM endpoint.

This runner executes each example script as a subprocess (``sys.executable -m
<module>``) so that a failing example cannot take down the batch, and so the
runner itself requires no NIM/GPU to import or to test. It:

- validates the served runtime + selected Generator profile once up front,
- pre-flights the local assets each example needs,
- runs each example and records PASS / SKIP / FAIL,
- prints a summary and returns a meaningful exit code.

Exit codes:
  0  every run passed
  1  one or more runs failed
  2  usage / check failure (bad flags, no compatible runtime)
  3  nothing ran (no example was selected or skippable)
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

# Importable from the same directory as this file (no third-party imports).
import common
import request_builders as builders
import settings

HERE = Path(__file__).resolve().parent
MANIFEST_PATH = HERE / "runs.yaml"

# Built-in fallback registry used when `runs.yaml` is missing or empty.
# ``allowed_variants`` limits which served Generator profiles an example works
# with; ``--check`` compares the served variant against it.
_BUILTIN_RUNS: list[dict] = [
    {"id": "t2v", "module": "t2v", "runtime": "generator", "allowed_variants": ("nano", "super")},
    {"id": "t2i", "module": "t2i", "runtime": "generator", "allowed_variants": ("nano", "super", "super-t2i")},
    {"id": "i2v", "module": "i2v", "runtime": "generator", "allowed_variants": ("nano", "super", "super-i2v")},
    {"id": "i2v_4step", "module": "i2v_4step", "runtime": "generator", "allowed_variants": ("super-i2v-4step",)},
    {"id": "t2i_4step", "module": "t2i_4step", "runtime": "generator", "allowed_variants": ("super-t2i-4step",)},
    {"id": "v2v", "module": "v2v", "runtime": "generator", "allowed_variants": ("nano", "super")},
    {"id": "transfer", "module": "transfer", "runtime": "generator", "allowed_variants": ("nano", "super")},
    {"id": "action", "module": "action", "runtime": "generator", "allowed_variants": ("nano", "super")},
    {"id": "reasoner", "module": "reasoner", "runtime": "reasoner", "allowed_variants": None},
]


def load_runs() -> list[dict]:
    """Load the run registry from ``runs.yaml``, falling back to the built-in list.

    A missing/empty manifest yields the built-in registry; malformed YAML is a
    hard error (a config a user wrote should not be silently ignored).
    """
    if not MANIFEST_PATH.is_file():
        return [dict(run) for run in _BUILTIN_RUNS]
    try:
        import yaml
    except ImportError:
        return [dict(run) for run in _BUILTIN_RUNS]

    doc = yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))
    if not doc or not doc.get("runs"):
        return [dict(run) for run in _BUILTIN_RUNS]
    entries = doc["runs"]
    if not isinstance(entries, list):
        raise ValueError("runs.yaml 'runs' must be a list")
    runs: list[dict] = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("every runs.yaml entry must be a mapping")
        rid = entry.get("id")
        if not isinstance(rid, str) or not rid:
            raise ValueError("every runs.yaml entry needs a non-empty 'id'")
        runs.append(
            {
                "id": rid,
                "module": entry.get("module", rid),
                "runtime": entry.get("runtime", "generator"),
                "allowed_variants": tuple(entry.get("allowed_variants") or ()),
            }
        )
    return runs


RUNS: list[dict] = load_runs()

DEFAULT_TIMEOUT = 1800
# Transfer requests can take longer (the script itself uses a 3600 s HTTP timeout).
LONGER_TIMEOUT_IDS = {"transfer": 3600}


def _sys_path() -> list[str]:
    return [str(HERE), *sys.path]


def resolve_run(rid: str) -> dict | None:
    for run in RUNS:
        if run["id"] == rid:
            return run
    return None


def run_in_subprocess(
    run: dict,
    *,
    timeout: int,
    nim_url: str,
    extra_args: list[str] | None = None,
) -> subprocess.CompletedProcess:
    """Execute one example script as ``python -m <module> [extra_args]``.

    ``NIM_URL`` is exported into the subprocess environment so the example
    script targets the same endpoint the runner preflights.
    """
    cmd = [sys.executable, "-m", run["module"], *(extra_args or [])]
    env = {**os.environ, "PYTHONPATH": ":".join(_sys_path()), "NIM_URL": nim_url}
    return subprocess.run(
        cmd,
        cwd=str(HERE),
        env=env,
        timeout=timeout,
        capture_output=True,
        text=True,
    )


@dataclass
class RunResult:
    rid: str
    status: str  # "pass" | "skip" | "fail"
    detail: str = ""
    output: str = field(default="", repr=False)


def check_runtime(nim_url: str) -> dict:
    """Return the /v1/metadata dict, raising if unreachable or malformed."""
    return common.require_runtime(
        nim_url,
        expected_runtime="generator",
        expected_endpoint="/v1/infer",
    )


def check_profile(nim_url: str, *, allowed_variants: tuple[str, ...]) -> dict:
    """Require a compatible Generator profile (raises if not)."""
    return common.require_generator_profile(nim_url, allowed_variants=allowed_variants)


def preflight_assets(run: dict) -> list[str]:
    """Return missing asset paths for a run (empty if OK)."""
    if run["id"] == "reasoner":
        # The reasoner validates its case catalog and asset paths at import
        # time; nothing extra to preflight here.
        return []
    return builders.check_assets(run["id"])


def preflight(
    nim_url: str,
    selected: list[dict],
    *,
    quiet: bool,
) -> tuple[list[RunResult], bool]:
    """Validate runtime/profile + local assets. Returns (skip_results, runtime_ok)."""
    skips: list[RunResult] = []

    # 1. Runtime must be a Generator (the reasoner runs through the same NIM
    #    image, but the runner only drives /v1/infer).
    served_variant: str | None = None
    try:
        metadata = check_runtime(nim_url)
        if any(run["runtime"] == "generator" for run in selected):
            allowed = tuple(
                sorted({v for run in selected for v in (run.get("allowed_variants") or ())})
            )
            check_profile(nim_url, allowed_variants=allowed)
            served_variant = metadata.get("model_variant")
    except Exception as exc:
        if not quiet:
            print(f"[SKIP] runtime/profile check failed: {exc}", file=sys.stderr)
        for run in selected:
            skips.append(RunResult(run["id"], "skip", detail="no compatible runtime"))
        return skips, False

    # 2. Per-run: asset presence + served-variant compatibility.
    for run in selected:
        detail = ""
        allowed = run.get("allowed_variants")
        if allowed and served_variant is not None and served_variant not in allowed:
            detail = f"served variant {served_variant!r} not in {tuple(allowed)}"
        if not detail:
            missing = preflight_assets(run)
            if missing:
                detail = f"missing assets: {', '.join(missing)}"
        if detail:
            skips.append(RunResult(run["id"], "skip", detail=detail))
            if not quiet:
                print(f"[SKIP] {run['id']}: {detail}", file=sys.stderr)
    return skips, True


def _run_one(
    run: dict,
    *,
    timeout: int,
    nim_url: str,
    verbose: bool,
    quiet: bool,
) -> RunResult:
    """Execute one example, returning its RunResult."""
    per_run_timeout = LONGER_TIMEOUT_IDS.get(run["id"], timeout)
    extra = ["--case", "all"] if run["id"] == "reasoner" else None
    try:
        proc = run_in_subprocess(
            run, timeout=per_run_timeout, nim_url=nim_url, extra_args=extra
        )
    except subprocess.TimeoutExpired:
        return RunResult(run["id"], "fail", detail=f"timeout after {per_run_timeout}s")
    ok = proc.returncode == 0
    if verbose and proc.stdout:
        print(f"--- {run['id']} stdout ---")
        print(proc.stdout.rstrip())
    if proc.stderr and not quiet:
        print(proc.stderr.rstrip(), file=sys.stderr)
    return RunResult(
        run["id"],
        "pass" if ok else "fail",
        detail=("" if ok else f"exit {proc.returncode}"),
        output=proc.stdout,
    )


def run_all(
    nim_url: str,
    *,
    selected: list[dict],
    timeout: int,
    verbose: bool,
    quiet: bool,
    parallel: bool = False,
) -> tuple[list[RunResult], int]:
    """Execute the selected example runs, returning (results, exit_code).

    With ``parallel=True``, independent runs execute concurrently (each as its
    own subprocess); the reasoner still runs as one ``--case all`` subprocess.
    """
    results, runtime_ok = preflight(nim_url, selected, quiet=quiet)
    if not runtime_ok:
        return results, 3

    runnable = [run for run in selected if not any(r.rid == run["id"] for r in results)]
    if parallel and len(runnable) > 1:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        results_by_rid: dict[str, RunResult] = {}
        with ThreadPoolExecutor(max_workers=len(runnable)) as pool:
            futures = {
                pool.submit(_run_one, run, timeout=timeout, nim_url=nim_url, verbose=verbose, quiet=quiet): run["id"]
                for run in runnable
            }
            for future in as_completed(futures):
                rid = futures[future]
                results_by_rid[rid] = future.result()
        # Preserve run order for a stable summary.
        for run in runnable:
            results.append(results_by_rid[run["id"]])
    else:
        for run in runnable:
            results.append(_run_one(run, timeout=timeout, nim_url=nim_url, verbose=verbose, quiet=quiet))

    exit_code = 1 if any(r.status == "fail" for r in results) else 0
    return results, exit_code


def summary_lines(results: list[RunResult]) -> list[str]:
    infos = " | ".join(f"{r.rid}={r.status}" for r in results) or "(nothing ran)"
    passed = sum(r.status == "pass" for r in results)
    skipped = sum(r.status == "skip" for r in results)
    failed = sum(r.status == "fail" for r in results)
    lines = [infos]
    lines.append(f"runs: {len(results)}, passed: {passed}, skipped: {skipped}, failed: {failed}")
    if failed:
        lines.append("failed: " + ", ".join(r.rid for r in results if r.status == "fail"))
    if skipped:
        lines.append("skipped: " + ", ".join(r.rid for r in results if r.status == "skip"))
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--nim-url",
        default=common.get_default_nim_url(),
        help="Base URL of the NIM (default: $NIM_URL or http://localhost:8000).",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print the run registry and exit.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate runtime/profile/assets without running inference.",
    )
    parser.add_argument(
        "--case",
        choices=[run["id"] for run in RUNS],
        help="Run only the named example.",
    )
    parser.add_argument(
        "--no-reasoner",
        action="store_true",
        help="Skip the reasoner example.",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help=(
            "Run independent examples concurrently (each as its own subprocess). "
            "Reasoner still runs as one --case all subprocess."
        ),
    )
    parser.add_argument(
        "--override",
        action="append",
        metavar="KEY=VALUE",
        default=[],
        help=(
            "Override a shared generation default for every run "
            "(e.g. --override seed=42). Repeatable."
        ),
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help=f"Per-run timeout in seconds (default: {DEFAULT_TIMEOUT}).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print each run's stdout after completion.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-run progress output.",
    )
    parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Summary output format.",
    )
    args = parser.parse_args()

    if args.list:
        for run in RUNS:
            print(f"{run['id']:12s} runtime={run['runtime']}")
        return

    nim_url = args.nim_url.rstrip("/")
    selected = [resolve_run(args.case)] if args.case else list(RUNS)
    if args.no_reasoner:
        selected = [run for run in selected if run["id"] != "reasoner"]
    if not selected:
        print("Nothing selected to run.", file=sys.stderr)
        raise SystemExit(3)

    # Parse --override KEY=VALUE into a JSON object exported to every subprocess
    # so the shared settings module applies it to all runs.
    overrides: dict = {}
    for item in args.override:
        key, separator, value = item.partition("=")
        if not separator:
            parser.error(f"--override must be KEY=VALUE, got {item!r}")
        overrides[key] = value
    if overrides:
        os.environ[settings.OVERRIDES_ENV] = json.dumps(overrides)

    if args.check:
        results, runtime_ok = preflight(nim_url, selected, quiet=args.quiet)
        # --check must not execute inference; report the preflight only.
        if args.format == "json":
            payload = {
                "nim_url": nim_url,
                "ok": runtime_ok,
                "results": [{"id": r.rid, "status": r.status, "detail": r.detail} for r in results],
            }
            print(json.dumps(payload, indent=2))
        else:
            for line in summary_lines(results if results else []):
                print(line)
            print("check: " + ("ok" if runtime_ok else "failed"))
        # Reuse exit-code conventions: no runtime => 3, else 0/1 from preflight.
        if not runtime_ok:
            raise SystemExit(3)
        raise SystemExit(1 if any(r.status == "fail" for r in results) else 0)

    results, exit_code = run_all(
        nim_url,
        selected=selected,
        timeout=args.timeout,
        verbose=args.verbose,
        quiet=args.quiet,
        parallel=args.parallel,
    )

    if args.format == "json":
        payload = {
            "nim_url": nim_url,
            "results": [{"id": r.rid, "status": r.status, "detail": r.detail} for r in results],
            "exit_code": exit_code,
        }
        out = common.default_output_dir(__file__) / "run_summary.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(payload, indent=2))
    else:
        for line in summary_lines(results):
            print(line)
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()