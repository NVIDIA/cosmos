#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Notebook/CLI orchestration; evaluation and preprocessing remain VLMEvalKit's."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
KIT = ROOT / "cosmos_eval"
DATASET_REVISION = "ad5297f645ba90830478a4c6a72a3a7ab077a2f7"


def manifest():
    return json.loads((KIT / "manifest.json").read_text())


def benchmarks(value):
    available = [key.split("/", 1)[1] for key in manifest() if "/VANTAGE_" in key]
    selected = available if value == "all" else value.split(",")
    unknown = set(selected) - set(available)
    if unknown:
        raise ValueError(f"Unknown VANTAGE benchmark(s): {', '.join(sorted(unknown))}")
    # run_all selects in manifest order; keep Astro2D's indexed output consistent.
    return [bench for bench in available if bench in selected]


def endpoint(wait=0):
    base = os.environ.get("COSMOS_API_BASE", "http://cosmos-endpoint:8080/v1/chat/completions")
    if not base.endswith("/chat/completions"):
        raise ValueError("COSMOS_API_BASE must be the complete /v1/chat/completions URL")
    os.environ["COSMOS_API_BASE"] = base
    url = base.removesuffix("/chat/completions") + "/models"
    key = os.environ.get("COSMOS_API_KEY", "EMPTY")
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}"})
    deadline = time.monotonic() + wait
    while True:
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                models = json.load(response)["data"]
            if not models:
                raise ValueError("The endpoint returned no served models")
            requested = os.environ.get("COSMOS_MODEL")
            ids = [model["id"] for model in models]
            if requested and requested not in ids:
                raise ValueError(f"COSMOS_MODEL {requested!r} is not served; available: {ids}")
            if not requested and len(ids) != 1:
                raise ValueError(f"Set COSMOS_MODEL to one of {ids}")
            os.environ["COSMOS_MODEL"] = requested or ids[0]
            print(f"Endpoint ready; model: {os.environ['COSMOS_MODEL']}", flush=True)
            return
        except urllib.error.HTTPError as exc:
            if exc.code not in (502, 503, 504):
                raise RuntimeError(f"Model endpoint returned HTTP {exc.code}") from exc
            if time.monotonic() >= deadline:
                raise RuntimeError(f"Model endpoint not ready after {wait}s (HTTP {exc.code})") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            if time.monotonic() >= deadline:
                raise RuntimeError(f"Model endpoint not reachable after {wait}s") from exc
        time.sleep(min(5, max(0, deadline - time.monotonic())))


def prepare(selected):
    # Use the same config composition and dataset constructors as run.py. Do not
    # transcode, re-sample, or invent a second dataset conversion convention.
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(KIT))
    import run_all
    from run import build_dataset_from_config

    layer = run_all.load_model_layer("cosmos")
    for bench in selected:
        cfg = run_all.build_config(layer, "metropolis", bench, render=False)
        dataset = build_dataset_from_config(cfg["data"], bench)
        if dataset is None or dataset.data.empty:
            raise RuntimeError(f"{bench}: no prepared dataset rows")
        print(f"{bench}: prepared {len(dataset.data)} questions", flush=True)


def run_command(command, **kwargs):
    subprocess.run(command, cwd=ROOT, check=True, **kwargs)


def prediction_files(work):
    # Only dated run directories, not the symlinks run.py makes at model level.
    return sorted(p for p in work.rglob("cosmos_*")
                  if p.parent.name.startswith("T") and p.suffix in {".tsv", ".xlsx", ".json"}
                  and p.stem in {f"cosmos_{bench}" for bench in benchmarks("all")})


def verify_predictions(work, bench):
    import pandas as pd

    files = [p for p in prediction_files(work) if p.stem == f"cosmos_{bench}"]
    if not files:
        # run.py catches dataset/model exceptions and may exit 0 without output.
        raise RuntimeError(f"{bench}: no predictions were written; inspect the run log")
    path = files[-1]
    if path.suffix == ".tsv":
        frame = pd.read_csv(path, sep="\t")
    elif path.suffix == ".json":
        frame = pd.read_json(path)
    else:
        frame = pd.read_excel(path)
    if "prediction" not in frame or frame.empty:
        raise RuntimeError(f"{bench}: empty or invalid prediction output")
    failed = frame["prediction"].isna() | frame["prediction"].astype(str).str.contains(
        "Failed to obtain answer via API.", regex=False
    )
    if failed.any():
        raise RuntimeError(f"{bench}: {int(failed.sum())} failed/empty predictions; inspect the run log")
    print(f"{bench}: {len(frame)} predictions verified", flush=True)


def run(selected, work, mode, smoke, dry_run, source):
    if smoke and selected != ["VANTAGE_VQA"]:
        raise ValueError("--smoke requires --benchmarks VANTAGE_VQA")
    if work.exists() and not dry_run:
        raise ValueError(f"Use a fresh --work-dir (already exists: {work}) to avoid stale results")
    if not dry_run:
        endpoint(wait=1800)
    if source == "public":
        import public_eval

        public_eval.run(selected, work, mode, smoke, dry_run)
        return
    configs = work / "_configs"
    run_command([sys.executable, str(KIT / "run_all.py"), "--benchmarks", ",".join(selected),
                 "--export-configs", str(configs)])
    if mode == "all" and not smoke:
        command = [sys.executable, str(KIT / "run_all.py"), "--benchmarks", ",".join(selected),
                   "--concurrency", "1", "--work-dir", str(work)]
        if dry_run:
            command.append("--dry-run")
        run_command(command)
        if not dry_run:
            for bench in selected:
                verify_predictions(work, bench)
            record_run(work, selected, mode, smoke)
        return
    # run_all.py only implements inference+scoring. Use its official composed
    # configs with run.py's native infer mode for public inputs / a short smoke.
    entries = manifest()
    for index, bench in enumerate(selected):
        path = configs / f"{bench}.json"
        cfg = json.loads(path.read_text())
        if smoke:
            cfg["data"][bench].update(limit=5, random_state=0)
        for model in cfg["model"].values():
            model["api_base"] = os.environ.get("COSMOS_API_BASE", "${COSMOS_API_BASE}")
            model["model"] = os.environ.get("COSMOS_MODEL", "${COSMOS_MODEL}")
        path.write_text(json.dumps(cfg, indent=2) + "\n")
        # Astro2D rewrites its dataset name throughout paths: use name-free dirs.
        out = work / f"bench_{index:02d}"
        flags = entries[f"metropolis/{bench}"]["run"]
        command = [sys.executable, str(ROOT / "run.py"), "--config", str(path),
                   "--mode", mode, "--work-dir", str(out),
                   "--api-nproc", str(flags["api_nproc"]),
                   "--judge-nproc", str(flags["judge_nproc"]), "--verbose", "--save-eval-results"]
        if dry_run:
            print(" ".join(command))
            continue
        out.mkdir(parents=True)
        with (out / "run.log").open("w") as log:
            run_command(command, stdout=log, stderr=subprocess.STDOUT)
        verify_predictions(out, bench)
    if not dry_run:
        record_run(work, selected, mode, smoke)


def record_run(work, selected, mode, smoke):
    provenance = {
        "cosmos_commit": os.environ.get("COSMOS_COMMIT", "unknown"),
        "model": os.environ["COSMOS_MODEL"], "benchmarks": selected,
        "source": "internal", "mode": mode, "smoke": smoke,
        "work_dirs": {
            bench: (f"_concat_{i:02d}" if bench == "VANTAGE_Astro2D" else bench)
            if mode == "all" and not smoke else f"bench_{i:02d}"
            for i, bench in enumerate(selected)
        },
    }
    (work / "run.json").write_text(json.dumps(provenance, indent=2) + "\n")
    packages = Path("/opt/evaluation-packages.txt")
    if packages.exists():
        shutil.copyfile(packages, work / "evaluation-packages.txt")


def results(work):
    sys.path.insert(0, str(KIT))
    import parse_score

    metadata = json.loads((work / "run.json").read_text())
    for bench, directory in metadata["work_dirs"].items():
        report = parse_score.report(work_dir=str(work / directory), dataset_name=bench)
        score = report["overall"]
        print(f"{bench:26} {score if score is not None else 'not locally scored'}")
    for path in prediction_files(work):
        print(f"Predictions: {path}")


def export(work):
    if not prediction_files(work):
        raise ValueError("No predictions found; run inference before exporting")
    archive = work.parent / f"{work.name}-predictions.tar.gz"
    # Exclude logs and configs: they can contain endpoint URLs or model responses.
    # This is an evidence bundle, NOT an undocumented leaderboard conversion.
    with tarfile.open(archive, "w:gz") as bundle:
        for path in prediction_files(work):
            bundle.add(path, arcname=str(path.relative_to(work)), recursive=False)
        for name in ("run.json", "evaluation-packages.txt"):
            if (work / name).exists():
                bundle.add(work / name, arcname=name, recursive=False)
    print(f"Prediction bundle: {archive}")
    print("Internal evidence only: may contain private ground truth. Not a public leaderboard submission.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("check", help="Wait for /v1/models and verify the requested model")
    check.add_argument("--wait", type=int, default=1800)
    prep = sub.add_parser("prepare", help="Prepare official public or internal datasets")
    prep.add_argument("--source", choices=["public", "internal"], default="public")
    prep.add_argument("--benchmarks", default="VANTAGE_VQA")
    prep.add_argument("--revision", default=DATASET_REVISION)
    runner = sub.add_parser("run", help="Run the selected public or NVIDIA-internal evaluation harness")
    runner.add_argument("--benchmarks", default="VANTAGE_VQA")
    runner.add_argument("--source", choices=["public", "internal"], default="public")
    runner.add_argument("--mode", choices=["infer", "all"], default="infer")
    runner.add_argument("--smoke", action="store_true", help="Five seeded VQA samples, not a reference score")
    runner.add_argument("--dry-run", action="store_true")
    runner.add_argument("--work-dir", type=Path, required=True)
    for command in ("results", "export"):
        child = sub.add_parser(command)
        child.add_argument("--work-dir", type=Path, required=True)
        child.add_argument("--source", choices=["public", "internal"], default="public")
    args = parser.parse_args()
    if hasattr(args, "source"):
        data_root = Path(os.environ.get("LAUNCHABLE_DATA_ROOT", "/data"))
        os.environ["LMUData"] = str(data_root / args.source / "LMUData")
        if args.command in {"prepare", "run"} and not getattr(args, "dry_run", False):
            Path(os.environ["LMUData"]).mkdir(parents=True, exist_ok=True)
    if args.command == "check":
        endpoint(args.wait)
    elif args.command == "prepare":
        if args.source == "public":
            import public_eval

            public_eval.prepare(benchmarks(args.benchmarks), args.revision)
        else:
            prepare(benchmarks(args.benchmarks))
    elif args.command == "run":
        run(benchmarks(args.benchmarks), args.work_dir.resolve(), args.mode, args.smoke, args.dry_run, args.source)
    elif args.command == "results":
        if args.source == "public":
            import public_eval

            public_eval.results(args.work_dir.resolve())
        else:
            results(args.work_dir.resolve())
    else:
        if args.source == "public":
            import public_eval

            public_eval.export(args.work_dir.resolve())
        else:
            export(args.work_dir.resolve())


if __name__ == "__main__":
    main()
