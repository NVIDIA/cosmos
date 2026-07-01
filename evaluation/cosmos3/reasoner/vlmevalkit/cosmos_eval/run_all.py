# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Parallel launcher for the cosmos_eval kit.

Composes each benchmark's `run.py --config` JSON from the committed split
artifacts (`data/<domain>/<bench>.json` + `models/<family>.json`), runs the
stock vlmevalkit `run.py` for each in a bounded process pool, and reports the
headline score per benchmark via `parse_score.py`.

This is the END-USER entrypoint: stdlib only, no `vlmeval-metric` dependency.
The internal generator (`gen_oss_configs.py`) is what *produced* the artifacts;
this just consumes them with a dumb dict-merge — no routing/schema logic here.

Quickstart:

    export COSMOS_API_BASE=https://<endpoint>/v1/chat/completions
    export COSMOS_MODEL=<served-model-id>   COSMOS_API_KEY=<key>
    export OPENAI_API_BASE=<judge-endpoint> OPENAI_API_KEY=<judge-key>   # gpt-4o judge
    python cosmos_eval/run_all.py --model cosmos --concurrency 8 --work-dir ./out
      # runs every benchmark in the manifest

    # one/few:   --benchmarks VANTAGE_VQA,AETCBench_all
    # inspect:   --export-configs ./cfgs   (compose + write, no run)
    # tweak-run: --import-configs ./cfgs   (run from a dir of edited configs)
"""

from __future__ import annotations

import argparse
import json
import os
import string
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import parse_score  # sibling module in cosmos_eval/

HERE = Path(__file__).resolve().parent
VLMEVALKIT_ROOT = HERE.parent
RUN_PY = VLMEVALKIT_ROOT / "run.py"

# ConcatDataset benchmarks (e.g. Astro2D) evaluate per sub-dataset via
# `eval_file.replace(dataset_name, sub_name)`, which rewrites the name EVERYWHERE in the
# path — including the work-dir component. So they need a work-dir whose path does NOT
# contain the benchmark name, otherwise the sub-result targets a sibling dir that is never
# created (FileNotFoundError). These get a name-free (indexed) work-dir; all other benches
# keep the readable <work-dir>/<bench> layout.
_CONCAT_BENCHES = {"VANTAGE_Astro2D"}


# ---------------------------------------------------------------------------
# Kit loading + composition (the dumb dict-merge; mirrors gen_oss_configs.compose)
# ---------------------------------------------------------------------------


def load_model_layer(model: str) -> dict[str, Any]:
    path = HERE / "models" / f"{model}.json"
    if not path.exists():
        sys.exit(f"Error: no model layer at {path} (have: "
                 f"{', '.join(p.stem for p in sorted((HERE / 'models').glob('*.json')))})")
    return json.loads(path.read_text("utf-8"))


def load_manifest() -> dict[str, Any]:
    return json.loads((HERE / "manifest.json").read_text("utf-8"))


def load_data_conf(domain: str, bench: str) -> dict[str, Any]:
    return json.loads((HERE / "data" / domain / f"{bench}.json").read_text("utf-8"))


def _render_env(obj: Any) -> Any:
    """Substitute ${VAR} placeholders from the environment, recursively."""
    if isinstance(obj, str):
        return string.Template(obj).safe_substitute(os.environ)
    if isinstance(obj, dict):
        return {k: _render_env(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_render_env(v) for v in obj]
    return obj


def compose(data_conf: dict[str, Any], model_layer: dict[str, Any], bench: str) -> tuple[dict, dict]:
    """model_conf = {class} | defaults | benchmarks[bench]; dataset_conf = data | {model_family}.

    Placeholders are kept verbatim (env substitution is applied separately at write time).
    """
    model_conf = {"class": model_layer["class"]}
    model_conf.update(model_layer["defaults"])
    model_conf.update(model_layer["benchmarks"].get(bench, {}))
    dataset_conf = dict(data_conf)
    dataset_conf["model_family"] = model_layer["model_family"]
    return model_conf, dataset_conf


def build_config(model_layer: dict[str, Any], domain: str, bench: str, *, render: bool) -> dict[str, Any]:
    """The full `run.py --config` document for one benchmark."""
    model_conf, dataset_conf = compose(load_data_conf(domain, bench), model_layer, bench)
    if render:
        model_conf = _render_env(model_conf)
    return {"model": {model_layer["model_key"]: model_conf}, "data": {bench: dataset_conf}}


# ---------------------------------------------------------------------------
# Benchmark selection
# ---------------------------------------------------------------------------


def select_benches(
    manifest: dict[str, Any],
    *,
    domains: list[str] | None,
    benchmarks: list[str] | None,
) -> list[dict[str, Any]]:
    """Resolve the manifest into an ordered list of {key, bench, domain, run}.

    Runs every benchmark in the manifest, optionally narrowed by `domains` / `benchmarks`.
    """
    selected: list[dict[str, Any]] = []
    for key, entry in manifest.items():
        domain, bench = key.split("/", 1)
        if domains and domain not in domains:
            continue
        if benchmarks and bench not in benchmarks:
            continue
        selected.append({"key": key, "bench": bench, "domain": domain, "run": entry.get("run", {})})
    for i, it in enumerate(selected):
        it["idx"] = i  # stable index for name-free work-dirs (ConcatDataset benches)
    return selected


def _bench_workdir(work_root: Path, item: dict[str, Any]) -> Path:
    """Per-bench work-dir. ConcatDataset benches get a NAME-FREE dir (see _CONCAT_BENCHES)."""
    if item["bench"] in _CONCAT_BENCHES:
        return work_root / f"_concat_{item['idx']:02d}"
    return work_root / item["bench"]


# ---------------------------------------------------------------------------
# run.py invocation
# ---------------------------------------------------------------------------


def _run_cmd(config_path: Path, work_dir: Path, run_flags: dict[str, Any]) -> list[str]:
    """Build the stock `run.py --config` command (no --data/--model; cfg drives them)."""
    cmd = [sys.executable, str(RUN_PY), "--config", str(config_path), "--work-dir", str(work_dir)]
    cmd += ["--api-nproc", str(run_flags.get("api_nproc", 16))]
    cmd += ["--judge-nproc", str(run_flags.get("judge_nproc", 4))]
    if run_flags.get("judge"):
        cmd += ["--judge", str(run_flags["judge"])]
    if run_flags.get("judge_args"):
        cmd += ["--judge-args", str(run_flags["judge_args"])]
    cmd += ["--verbose", "--save-eval-results"]
    return cmd


def run_one(item: dict[str, Any], model_layer: dict[str, Any], *, work_root: Path,
            configs_dir: Path, import_dir: Path | None) -> dict[str, Any]:
    """Compose (or import) the config, run run.py, then parse the score. Returns a result row."""
    bench, domain = item["bench"], item["domain"]
    bench_out = _bench_workdir(work_root, item)
    bench_out.mkdir(parents=True, exist_ok=True)

    if import_dir is not None:
        config_path = import_dir / f"{bench}.json"
        if not config_path.exists():
            return {**item, "status": "error", "score": None, "detail": f"no imported config {config_path}"}
        # Imported configs may still carry ${...} placeholders; render into the work copy.
        rendered = _render_env(json.loads(config_path.read_text("utf-8")))
        config_path = configs_dir / f"{bench}.json"
        config_path.write_text(json.dumps(rendered, indent=2), "utf-8")
    else:
        cfg = build_config(model_layer, domain, bench, render=True)
        config_path = configs_dir / f"{bench}.json"
        config_path.write_text(json.dumps(cfg, indent=2), "utf-8")

    cmd = _run_cmd(config_path, bench_out, item["run"])
    log_path = bench_out / "run.log"
    with log_path.open("w") as log:
        proc = subprocess.run(cmd, cwd=str(VLMEVALKIT_ROOT), stdout=log,
                              stderr=subprocess.STDOUT, env=os.environ.copy())
    if proc.returncode != 0:
        return {**item, "status": "error", "score": None, "detail": f"run.py exit {proc.returncode}; see {log_path}"}

    rep = parse_score.report(work_dir=str(bench_out), dataset_name=bench)
    if rep["eval_json"] is None:
        return {**item, "status": "no-eval", "score": None, "detail": f"no eval output; see {log_path}"}
    return {**item, "status": "ok", "score": rep["overall"], "subscores": rep["subscores"]}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _csv(value: str | None) -> list[str] | None:
    return [x.strip() for x in value.split(",") if x.strip()] if value else None


def _print_summary(results: list[dict[str, Any]]) -> None:
    print("\n=== cosmos_eval summary ===")
    width = max([len(r["bench"]) for r in results] + [9])
    for r in sorted(results, key=lambda x: x["key"]):
        score = f"{r['score']:.2f}" if isinstance(r.get("score"), (int, float)) else "-"
        line = f"  {r['bench']:<{width}}  {r['status']:<8}  {score:>7}"
        if r["status"] != "ok":
            line += f"   {r.get('detail', '')}"
        print(line)
    ok = sum(1 for r in results if r["status"] == "ok")
    print(f"\n  {ok}/{len(results)} ok, {len(results) - ok} failed")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="cosmos", help="model layer (models/<name>.json); default cosmos")
    ap.add_argument("--domains", default=None, help="comma-separated domains (default: all in the manifest)")
    ap.add_argument("--benchmarks", default=None, help="comma-separated benchmark names (default: all in the manifest)")
    ap.add_argument("--concurrency", type=int, default=2, help="max concurrent run.py subprocesses")
    ap.add_argument("--work-dir", default="./cosmos_eval_out", help="run output root")
    ap.add_argument("--dry-run", action="store_true", help="print the run.py commands and exit")
    ap.add_argument("--export-configs", metavar="DIR", help="compose + write each config to DIR, then exit")
    ap.add_argument("--import-configs", metavar="DIR", help="run from a dir of pre-composed configs instead of composing")
    args = ap.parse_args(argv)

    model_layer = load_model_layer(args.model)
    manifest = load_manifest()

    selected = select_benches(manifest, domains=_csv(args.domains), benchmarks=_csv(args.benchmarks))
    if not selected:
        sys.exit("Error: no benchmarks selected (check --domains/--benchmarks).")

    # --export-configs: compose (placeholders intact) + write, then exit.
    if args.export_configs:
        out = Path(args.export_configs)
        out.mkdir(parents=True, exist_ok=True)
        for item in selected:
            cfg = build_config(model_layer, item["domain"], item["bench"], render=False)
            (out / f"{item['bench']}.json").write_text(json.dumps(cfg, indent=2), "utf-8")
        print(f"Exported {len(selected)} config(s) to {out} (placeholders intact).")
        return

    work_root = Path(args.work_dir)
    configs_dir = work_root / "_configs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    import_dir = Path(args.import_configs) if args.import_configs else None

    if args.dry_run:
        for item in selected:
            cfg_path = (import_dir or configs_dir) / f"{item['bench']}.json"
            print(" ".join(_run_cmd(cfg_path, work_root / item["bench"], item["run"])))
        print(f"\n[dry-run] {len(selected)} benchmark(s) selected.")
        return

    print(f"Running {len(selected)} benchmark(s) with concurrency {args.concurrency} -> {work_root}")
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
        futures = {
            pool.submit(run_one, item, model_layer, work_root=work_root,
                        configs_dir=configs_dir, import_dir=import_dir): item
            for item in selected
        }
        for fut in as_completed(futures):
            r = fut.result()
            results.append(r)
            mark = "ok" if r["status"] == "ok" else r["status"].upper()
            score = f"{r['score']:.2f}" if isinstance(r.get("score"), (int, float)) else "-"
            print(f"  [{mark}] {r['bench']}: {score}")

    _print_summary(results)
    if any(r["status"] != "ok" for r in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
