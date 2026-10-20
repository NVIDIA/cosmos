# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Public VANTAGE orchestration, isolated from the NVIDIA harness's imports.

All dataset/model execution, conversion, validation and packaging runs with the
pinned Clemson checkout's Python. This module itself needs only the stdlib.
Public annotations are inference inputs, never local scoring ground truth.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shlex
import string
import subprocess

ROOT = Path(__file__).resolve().parents[1]
KIT = ROOT / "cosmos_eval"
HARNESS_REVISION = "ec1ebdebb955332a9d7fdd9e98f39627b2822ff5"
DATASET_REVISION = "ad5297f645ba90830478a4c6a72a3a7ab077a2f7"
DATASET_REPO = "nvidia/PhysicalAI-VANTAGE-Bench"
# NVIDIA benchmark -> (official preparation task, public dataset, public class,
#                       official submission packager key).
TASKS = {
    "VANTAGE_2DGrounding": ("grounding", "VANTAGE_2DGrounding", "VANTAGE_2DGroundingDataset", "grounding"),
    "VANTAGE_2DPointing": ("pointing", "VANTAGE_2DPointing", "VANTAGE_2DPointing", "pointing"),
    "VANTAGE_Astro2D": ("astro2d", "Astro2D", "Astro2DDetectionDataset", "astro"),
    "VANTAGE_DVC": ("dvc", "VANTAGE_DVC", "VANTAGE_DVC", "dvc"),
    "VANTAGE_EventVerification": ("event_verification", "VANTAGE_EventVerification", "VANTAGE_EventVerification", "event_verification"),
    "VANTAGE_SOT": ("sot", "VANTAGE_SOT", "VANTAGE_SOT", "sot"),
    "VANTAGE_Temporal": ("temporal", "VANTAGE_Temporal", "VANTAGE_Temporal", "temporal"),
    "VANTAGE_VQA": ("vqa", "VANTAGE_VQA", "VANTAGE_VQA", "vqa"),
}


def _harness():
    return Path(os.environ.get("PUBLIC_VLMEVAL_ROOT", "/opt/vantage-public"))


def _python():
    return os.environ.get("PUBLIC_VLMEVAL_PYTHON", str(_harness() / ".venv/bin/python"))


def _lmu():
    return Path(os.environ.get("LMUData", "/data/public/LMUData"))


def _env():
    env = os.environ.copy()
    # Never import the NVIDIA checkout via inherited notebook/container settings.
    env["PYTHONPATH"] = str(_harness())
    env["LMUData"] = str(_lmu())
    env["PRED_FORMAT"] = env.get("PRED_FORMAT") or "tsv"
    # The public runner otherwise silently replaces the explicit --work-dir.
    env.pop("MMEVAL_ROOT", None)
    return env


def _check_harness():
    if not Path(_python()).is_file():
        raise RuntimeError(f"Public harness Python missing: {_python()}; rebuild the evaluation image")
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=_harness(), text=True,
    ).strip()
    if revision != HARNESS_REVISION:
        raise RuntimeError(f"Public harness must be pinned to {HARNESS_REVISION}; found {revision}")


def _execute(command, *, log=None):
    display = [str(part) for part in command]
    if len(display) > 2 and display[1] == "-c":
        display[2] = "<public preparation/verification helper>"
    print(shlex.join(display), flush=True)
    if log is None:
        subprocess.run(command, cwd=_harness(), env=_env(), check=True)
        return
    print(f"Harness stdout/stderr: {log}", flush=True)
    with Path(log).open("w") as stream:
        result = subprocess.run(command, cwd=_harness(), env=_env(), stdout=stream,
                                stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"Public harness exited {result.returncode}; see {log}")


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def _selected(selected):
    if not selected or len(set(selected)) != len(selected) or set(selected) - TASKS.keys():
        raise ValueError("Select distinct supported VANTAGE benchmarks")


# Import the official script's task patterns instead of maintaining a second
# download/layout convention. Markers are required by --local-source validation.
_PREPARE = r'''
import csv, importlib.util, json, os, subprocess, sys
from pathlib import Path
from huggingface_hub import snapshot_download
root, lmu, revision, selected_json, repo = sys.argv[1:]
root, lmu = Path(root), Path(lmu)
selected = json.loads(selected_json)
spec = importlib.util.spec_from_file_location("vantage_public_prep", root / "scripts/run_lmudata.py")
prep = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = prep
spec.loader.exec_module(prep)
patterns = ["README.md", "LICENSE.md"]
for task in selected.values():
    patterns.extend(prep.TASK_CONFIG[task]["hf_patterns"])
snapshot = snapshot_download(repo, repo_type="dataset", revision=revision,
                             allow_patterns=sorted(set(patterns)), token=os.environ.get("HF_TOKEN"))
state_path = lmu / "launchable-public.json"
state = json.loads(state_path.read_text()) if state_path.exists() else {"tasks": {}}
for bench in selected:
    previous = state["tasks"].get(bench, {}).get("dataset_revision")
    if previous and previous != revision:
        raise RuntimeError("Changing a prepared task revision requires a fresh LAUNCHABLE_DATA_ROOT; "
                           "refusing to mix cached public media revisions")
for bench in selected:
    state["tasks"].pop(bench, None)
state_path.write_text(json.dumps(state, indent=2) + "\n")
subprocess.run([sys.executable, str(root / "scripts/run_lmudata.py"),
                "--local-source", snapshot, "--tasks", ",".join(selected.values()),
                "--lmu-root", str(lmu), "--symlink", "--force"], check=True)
for bench, task in selected.items():
    target = lmu / "datasets" / prep.TASK_CONFIG[task]["lmu_name"]
    if task == "sot":
        source = prep._load_jsonl(Path(snapshot) / "data/tracking/sot_benchmark.jsonl")
        names = {item["seq_id"] for item in source}
        actual = {p.parent.name for p in target.glob("*/gt.json")}
        if actual != names:
            raise RuntimeError(f"SOT preparation incomplete: {len(names - actual)} missing and "
                               f"{len(actual - names)} unexpected sequences")
        count = len(names)
    elif task == "astro2d":
        count = sum(1 for p in (target / "images").iterdir() if p.is_file())
    elif task == "grounding":
        count = len(json.loads((target / "annotations.json").read_text()))
    else:
        with (target / prep.TASK_CONFIG[task]["index_file"]).open() as stream:
            count = sum(1 for _ in csv.DictReader(stream, delimiter="\t"))
    if not count:
        raise RuntimeError(f"{bench}: no prepared rows")
    state["tasks"][bench] = {"dataset_revision": revision, "snapshot": snapshot, "expected_rows": count}
state.update(dataset_repo=repo, harness_revision="ec1ebdebb955332a9d7fdd9e98f39627b2822ff5")
state_path.write_text(json.dumps(state, indent=2) + "\n")
'''


def prepare(selected: list[str], revision: str):
    """Materialize only selected task subtrees, using official preparation."""
    _selected(selected)
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Use an immutable 40-character HF dataset commit SHA")
    _check_harness()
    _lmu().mkdir(parents=True, exist_ok=True)
    _execute([_python(), "-c", _PREPARE, str(_harness()), str(_lmu()), revision,
              json.dumps({bench: TASKS[bench][0] for bench in selected}), DATASET_REPO])
    print(f"Public inputs prepared at {_lmu()} (only selected task subtrees downloaded).")
    print("SOT additionally requires accepted nvidia/PhysicalAI-SmartSpaces license access; "
          "grounding uses the official VisDrone downloader.")


def _render(value):
    if isinstance(value, str):
        return string.Template(value).safe_substitute(os.environ)
    if isinstance(value, dict):
        return {key: _render(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_render(item) for item in value]
    return value


def _config(bench, smoke):
    layer = json.loads((KIT / "models/cosmos.json").read_text())
    data = json.loads((KIT / "data/metropolis" / f"{bench}.json").read_text())
    model = {"class": layer["class"], **layer["defaults"], **layer["benchmarks"].get(bench, {})}
    data.update(dataset=TASKS[bench][1], **{"class": TASKS[bench][2]})
    # Public image constructors do not accept model_family; SOT does, and uses
    # it for xyxy coordinate order. The public run.py filters unsupported kwargs.
    if bench == "VANTAGE_SOT":
        data["model_family"] = layer["model_family"]
    if smoke:
        data.update(limit=5, random_state=0)
    return {"model": {layer["model_key"]: _render(model)}, "data": {TASKS[bench][1]: data}}


# Verify both native table and canonical emitter output: upstream dataset errors
# and canonical-emitter errors can otherwise leave a successful process status.
_VERIFY = r'''
import json, sys
from pathlib import Path
import pandas as pd
from run import build_dataset_from_config
from vlmeval.smp import load
sys.path.insert(0, str(Path.cwd() / "scripts"))
from package_submission import find_submission_files
config_path, out, dataset_name, task, smoke, report_path = sys.argv[1:]
out = Path(out)
cfg = json.loads(Path(config_path).read_text())
dataset = build_dataset_from_config(cfg["data"], dataset_name)
expected = dataset.data
if expected.empty or (smoke == "1" and len(expected) != 5):
    raise RuntimeError("Expected five smoke rows or a nonempty complete dataset")
state = json.loads((Path(__import__("os").environ["LMUData"]) / "launchable-public.json").read_text())
prepared_bench = "VANTAGE_Astro2D" if dataset_name == "Astro2D" else dataset_name
if smoke != "1" and len(expected) != state["tasks"][prepared_bench]["expected_rows"]:
    raise RuntimeError("Dataset row count changed since pinned preparation; refusing incomplete predictions")
source_tsv = Path(__import__("os").environ["LMUData"]) / "datasets" / dataset_name / (dataset_name + ".tsv")
if smoke != "1" and source_tsv.exists() and len(pd.read_csv(source_tsv, sep="\t")) != len(expected):
    raise RuntimeError("Dataset dropped prepared rows (usually missing media); refusing incomplete predictions")
candidates = {p.resolve() for suffix in ("tsv", "xlsx", "json")
              for p in out.rglob("cosmos_" + dataset_name + "." + suffix) if p.is_file()}
if len(candidates) != 1:
    raise RuntimeError("Expected exactly one native prediction table, found: " + str(sorted(map(str, candidates))))
prediction_path = candidates.pop()
frame = load(str(prediction_path))
if not isinstance(frame, pd.DataFrame):
    frame = pd.DataFrame(frame)
if "prediction" not in frame or len(frame) != len(expected):
    raise RuntimeError("Native prediction count/schema does not match the selected dataset")
def bad(value):
    text = str(value).strip()
    return (value is None or text.lower() in ("", "nan", "none", "null", "<na>")
            or "failed to obtain answer" in text.lower())
if frame["prediction"].map(bad).any():
    raise RuntimeError("Failed/null native predictions; inspect run.log")
if "index" not in frame or frame["index"].duplicated().any() or set(map(str, frame["index"])) != set(map(str, expected["index"])):
    raise RuntimeError("Prediction indexes do not match selected dataset indexes")
found = find_submission_files(out)
if set(found) != {task}:
    raise RuntimeError("Missing or unexpected official canonical submission files: " + str(found))
canonical = found[task]
records = [json.loads(line) for line in canonical.read_text().splitlines() if line.strip()]
if len(records) != len(expected) or len({r["id"] for r in records}) != len(expected):
    raise RuntimeError("Canonical submission count/unique IDs do not match predictions")
from vlmeval.dataset.utils.vantagebench.emit import _TASK_SPECS
canonical_ids = {_TASK_SPECS[task].id_resolver(row, dataset) for _, row in expected.iterrows()}
if None in canonical_ids or {r["id"] for r in records} != canonical_ids:
    raise RuntimeError("Canonical IDs differ from the official emitter's selected dataset IDs")
for record in records:
    answers = [c.get("value") for c in record.get("conversations", []) if c.get("from") == "assistant"]
    if not answers or any(bad(answer) for answer in answers):
        raise RuntimeError("Canonical submission contains failed/null predictions")
Path(report_path).write_text(json.dumps({"rows": len(frame), "prediction": str(prediction_path),
                                       "submission": str(canonical.resolve())}, indent=2) + "\n")
print(f"Verified {dataset_name}: {len(frame)} native predictions and canonical records")
'''


def _prepared(selected):
    state_path = _lmu() / "launchable-public.json"
    if not state_path.is_file():
        raise ValueError("Prepare the selected public tasks before running (no pinned preparation metadata found)")
    state = json.loads(state_path.read_text())
    missing = set(selected) - state.get("tasks", {}).keys()
    if missing:
        raise ValueError(f"Public tasks not prepared: {', '.join(sorted(missing))}")
    if state.get("harness_revision") != HARNESS_REVISION:
        raise ValueError("Public preparation used a different harness revision; prepare again")
    return {bench: state["tasks"][bench]["dataset_revision"] for bench in selected}


def run(selected: list[str], work: Path, mode: str, smoke: bool, dry_run: bool):
    """Run official public inference; never score missing public ground truth."""
    _selected(selected)
    work = Path(work).resolve()
    if smoke and (selected != ["VANTAGE_VQA"] or mode != "infer"):
        raise ValueError("Public smoke requires VANTAGE_VQA and --mode infer")
    if mode not in ("infer", "all"):
        raise ValueError("Public mode must be infer or all")
    if mode == "all":
        print("Public annotations omit scoring ground truth: --mode all runs inference only; score on the official portal.")
    revisions = {} if dry_run else _prepared(selected)
    if not dry_run:
        _check_harness()
    configs = work / "_configs"
    configs.mkdir(parents=True, exist_ok=True)
    entries = json.loads((KIT / "manifest.json").read_text())
    provenance = {
        "source": "public", "harness_repo": "https://github.com/Clemson-Capstone/VANTAGE-Bench",
        "harness_revision": HARNESS_REVISION, "dataset_repo": DATASET_REPO,
        "harness_patch": "public-harness.patch (TLS/credential safety; NVIDIA-aligned lossless video and vLLM kwargs)",
        "dataset_revisions": revisions, "cosmos_commit": os.environ.get("COSMOS_COMMIT", "unknown"),
        "benchmarks": selected, "model": os.environ.get("COSMOS_MODEL", "${COSMOS_MODEL}"),
        "requested_mode": mode, "effective_mode": "infer", "smoke": smoke,
        "subset": smoke, "dry_run": dry_run, "status": "planned" if dry_run else "running",
        "sampling_note": "Committed NVIDIA data/model settings retained where public constructors accept them; "
                         "public task classes, prompts, IDs and preparation differ. SOT uses public prepared 8-frame clips "
                         "(nframe is a compatibility argument). No parity claim or local score.",
        "artifacts": {},
    }
    _write_json(work / "run.json", provenance)
    for index, bench in enumerate(selected):
        dataset = TASKS[bench][1]
        config = _config(bench, smoke)
        path = configs / f"{bench}.json"
        _write_json(path, config)
        flags = entries[f"metropolis/{bench}"]["run"]
        out = work / f"bench_{index:02d}"
        command = [_python(), str(_harness() / "run.py"),
                   "--config", str(path), "--mode", "infer", "--work-dir", str(out),
                   "--lmudata-root", str(_lmu()), "--api-nproc", str(flags["api_nproc"]),
                   "--verbose"]
        if dry_run:
            print(shlex.join(command), flush=True)
            continue
        out.mkdir(parents=True)
        report = out / "predictions.json"
        try:
            _execute(command, log=out / "run.log")
            _execute([_python(), "-c", _VERIFY, str(path), str(out), dataset, TASKS[bench][3],
                      "1" if smoke else "0", str(report)], log=out / "verify.log")
            validate = [_python(), str(_harness() / "scripts/validate_submission.py"), "--work-dir", str(out)]
            if not smoke:
                validate += ["--lmu-root", str(_lmu())]
            _execute(validate, log=out / "validate.log")
            provenance["artifacts"][bench] = json.loads(report.read_text())
            # Safe model and sampling provenance: omit endpoint and credentials.
            model = dict(config["model"]["cosmos"])
            model.pop("api_base", None)
            model.pop("key", None)
            provenance.setdefault("configs", {})[bench] = {"model": model, "data": config["data"]}
        except Exception:
            provenance["status"] = "failed"
            _write_json(work / "run.json", provenance)
            raise
        _write_json(work / "run.json", provenance)
    if not dry_run:
        provenance["status"] = "complete"
        _write_json(work / "run.json", provenance)
        results(work)


def _run_state(work):
    path = Path(work) / "run.json"
    if not path.is_file():
        raise ValueError(f"No public run metadata at {path}")
    state = json.loads(path.read_text())
    if state.get("source") != "public":
        raise ValueError("This work directory is not a public run")
    return state


def results(work: Path):
    """Show verified prediction artifacts, not unavailable local scores."""
    state = _run_state(work)
    print(f"Public run: {state['status']}" + (" (five-question smoke subset)" if state["smoke"] else ""))
    for bench in state["benchmarks"]:
        artifact = state.get("artifacts", {}).get(bench)
        if artifact:
            print(f"{bench}: {artifact['rows']} verified predictions\n  Native: {artifact['prediction']}\n  Canonical: {artifact['submission']}")
        else:
            print(f"{bench}: no verified predictions; inspect bench_*/run.log and verify.log")
    print("No local public score: hidden labels are evaluated at https://vantage-bench.org/submit (up to 24 hours).")


# The official validator warns but does not fail partial pillars. Enforce that
# policy before calling the official validator and packager, without inventing a
# new archive format or a second filename-discovery implementation.
_COVERAGE = r'''
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / "scripts"))
from package_submission import find_submission_files, PILLAR_TASKS
work, expected_json = sys.argv[1:]
found = find_submission_files(Path(work))
if set(found) != set(json.loads(expected_json)):
    raise RuntimeError("Discovered task files do not match the verified run metadata")
for pillar, tasks in PILLAR_TASKS.items():
    covered = set(tasks) & found.keys()
    if covered and covered != set(tasks):
        raise RuntimeError(f"Partial pillar {pillar}: missing {sorted(set(tasks) - covered)}. Run the complete pillar before export.")
if not found:
    raise RuntimeError("No canonical predictions to package")
'''


def export(work: Path):
    """Validate and package complete public pillars with upstream tooling."""
    work = Path(work).resolve()
    state = _run_state(work)
    if state.get("smoke") or state.get("subset") or state.get("dry_run") or state.get("status") != "complete":
        raise ValueError("Only completed, full-task public runs can be exported; smoke/dry-run/subset runs cannot be submitted")
    _check_harness()
    _execute([_python(), "-c", _COVERAGE, str(work),
              json.dumps([TASKS[bench][3] for bench in state["benchmarks"]])])
    # Re-read predictions at export: a completed run.json is not proof that files
    # still exist or have not been edited. The public venv handles every format.
    for index, bench in enumerate(state["benchmarks"]):
        out = work / f"bench_{index:02d}"
        _execute([_python(), "-c", _VERIFY, str(work / "_configs" / f"{bench}.json"),
                  str(out), TASKS[bench][1], TASKS[bench][3], "0", str(out / "predictions.json")],
                 log=out / "export-verify.log")
    validator = [_python(), str(_harness() / "scripts/validate_submission.py")]
    _execute(validator + ["--work-dir", str(work), "--lmu-root", str(_lmu())], log=work / "export-validate.log")
    archive = work.parent / f"{work.name}-submission.tar.gz"
    _execute([_python(), str(_harness() / "scripts/package_submission.py"),
              "--work-dir", str(work), "--out", str(archive)], log=work / "package.log")
    _execute(validator + ["--archive", str(archive), "--lmu-root", str(_lmu())], log=work / "archive-validate.log")
    if not archive.is_file() or not archive.stat().st_size:
        raise RuntimeError(f"Official packager did not create {archive}; see {work / 'package.log'}")
    print(f"Official public submission: {archive}")
    print("Upload at https://vantage-bench.org/submit; scoring may take up to 24 hours. No local score is claimed.")
    return archive
