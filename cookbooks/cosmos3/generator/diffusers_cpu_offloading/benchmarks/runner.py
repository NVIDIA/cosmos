"""Run full BF16 or checkpoint-policy FP8 benchmarks with safety checking enabled.

Default: seven modalities, three resolutions, full steps, one GPU, fresh process
per case. See benchmarks/README.md for timing definitions and requirements.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
import urllib.request
from pathlib import Path

from fp8_benchmark_audit import AUDIT_SCHEMA_VERSION, AUDIT_TYPE

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
MANIFEST = Path(__file__).with_name("manifest.json")
MODALITIES = ("t2v", "i2v", "t2vs", "i2vs", "action", "v2v", "transfer")
RESOLUTIONS = ("240p", "480p", "720p")


def diffusers_installation():
    """Identify the installed Git revision without importing the GPU runtime."""
    distribution = importlib.metadata.distribution("diffusers")
    origin = json.loads(distribution.read_text("direct_url.json") or "{}")
    commit = origin.get("vcs_info", {}).get("commit_id")
    if not isinstance(commit, str) or len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
        raise ValueError(
            "Benchmarks require a Git installation of Diffusers with an immutable commit ID. "
            "Install this cookbook's dependencies from pyproject.toml; see benchmarks/README.md."
        )
    return {"diffusers_commit": commit, "diffusers_version": distribution.version}


def check_diffusers_installation(manifest):
    """Prevent workers from mixing revisions within a prepared benchmark."""
    installed = diffusers_installation()
    if any(manifest.get(key) != value for key, value in installed.items()):
        raise ValueError("Diffusers changed after benchmark preparation. Use a new output directory.")


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def fingerprint():
    paths = [
        *ROOT.glob("cosmos3_streaming/**/*.py"),
        *ROOT.glob("benchmarks/**/*.py"),
        *ROOT.glob("benchmarks/**/*.json"),
        ROOT / "pyproject.toml",
    ]
    files = {str(path.relative_to(ROOT)): digest(path) for path in sorted(paths) if path.is_file()}
    return {"sha256": hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(), "files": files}


def configure_cache(cache):
    cache = cache.resolve()
    paths = {
        "HF_HOME": cache / "huggingface",
        "HF_HUB_CACHE": cache / "huggingface/hub",
        "HUGGINGFACE_HUB_CACHE": cache / "huggingface/hub",
        "XDG_CACHE_HOME": cache,
        "TORCH_HOME": cache / "torch",
        "TRITON_CACHE_DIR": cache / "triton",
        "TORCHINDUCTOR_CACHE_DIR": cache / "inductor",
        "CUDA_CACHE_PATH": cache / "cuda",
        "TMPDIR": cache / "tmp",
        "XDG_CONFIG_HOME": cache / "config",
        "XDG_DATA_HOME": cache / "data",
    }
    for name, path in paths.items():
        path.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(path)
    # Fixed CPU threading is part of the benchmark recipe.
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "8"
    os.environ["PYTHONUNBUFFERED"] = "1"


def prepare(args, manifest):
    from huggingface_hub import snapshot_download

    assets = args.output_dir / "inputs"
    for item in manifest["assets"]:
        if not set(args.modalities).intersection(item["modalities"]):
            continue
        target = assets / item["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            temporary = target.with_suffix(target.suffix + ".download")
            if "bundled" in item:
                shutil.copyfile(MANIFEST.parent / item["bundled"], temporary)
            else:
                print(f"Downloading {item['path']}", flush=True)
                with urllib.request.urlopen(item["url"], timeout=120) as source, temporary.open("wb") as dest:
                    shutil.copyfileobj(source, dest)
            if digest(temporary) != item["sha256"]:
                raise ValueError(f"Asset checksum mismatch: {item['path']}")
            temporary.replace(target)
        if digest(target) != item["sha256"]:
            raise ValueError(f"Existing asset checksum mismatch: {target}")

    examples = Path(snapshot_download(**manifest["example_model"], allow_patterns=["assets/**"]))
    model = None
    if not args.preflight_only:
        model = Path(
            snapshot_download(
                repo_id=manifest["model"]["repo_id"],
                revision=manifest["model"]["revision"],
                allow_patterns=[
                    "*.json",
                    "*.txt",
                    "*.model",
                    "*.jinja",
                    "transformer/**",
                    "vae/**",
                    "vision_encoder/**",
                    "sound_tokenizer/**",
                    "text_tokenizer/**",
                    "scheduler/**",
                ],
            )
        )
    for safety in manifest["safety_models"]:
        snapshot = Path(snapshot_download(**safety))
        # Guardrail 0.3.1 requests main internally. Pin it in this benchmark's
        # dedicated cache; workers are offline, so it cannot drift during a run.
        refs = snapshot.parent.parent / "refs"
        refs.mkdir(exist_ok=True)
        (refs / "main").write_text(safety["revision"])
    model_hashes = {
        str(path.relative_to(model)): digest(path)
        for path in (sorted(model.rglob("*")) if model is not None else [])
        if path.is_file() and ".cache" not in path.relative_to(model).parts
    }
    prepared = {
        "model_path": str(model) if model is not None else None,
        "example_model_path": str(examples),
        "example_files_sha256": {
            str(path.relative_to(examples)): digest(path)
            for path in sorted((examples / "assets").rglob("*"))
            if path.is_file()
        },
        "manifest": manifest,
        "model_files_sha256": model_hashes,
        "weights_requested": not args.preflight_only,
    }
    write_json(args.output_dir / "prepared.json", prepared)
    return prepared


def resolve_model(manifest, previous=None):
    """Pin mutable Hub revisions once; a resumed run retains its original commit."""
    from huggingface_hub import HfApi

    requested = manifest["requested_model"]
    selected = dict(requested)
    if previous and previous["manifest"].get("requested_model") == requested:
        selected = dict(previous["manifest"]["model"])
    info = HfApi().model_info(selected["repo_id"], revision=selected["revision"])
    names = {item.rfilename for item in info.siblings}
    if "transformer/config.json" not in names or not any(
        name.startswith("transformer/") and name.endswith((".safetensors", ".bin", ".pt", ".pth")) for name in names
    ):
        raise ValueError(
            f"{selected['repo_id']}@{selected['revision']} does not contain a transformer checkpoint. "
            "Wait for the weights to be published or select a published --revision. "
            "No fallback to a different checkpoint or precision is performed."
        )
    selected["revision"] = info.sha
    return selected


def guardrails_enabled(manifest, modality):
    exception = manifest.get("transfer_without_guardrails", False)
    if exception and manifest.get("checkpoint_precision") not in ("bf16", "fp8"):
        raise ValueError("The guardrail exception requires BF16 or FP8 Transfer")
    return not (exception and modality == "transfer")


def preflight_allows_generation(row, enabled):
    # Explicit exceptions may bypass a rejection, never a checker/infrastructure error.
    return row["status"] == "passed" or (not enabled and row["status"] == "safety_rejected")


def passed_and_intact(report, safety_checker=True):
    checks = report.get("audit", {}).get("safety_checks", [])
    if safety_checker:
        safety_valid = all(
            any(event.get("kind") == kind and event.get("passed") is True for event in checks)
            for kind in ("text", "video")
        )
    else:
        precision = report.get("checkpoint_precision")
        safety_valid = (
            report.get("modality") == "transfer"
            and precision in ("bf16", "fp8")
            and report.get("audit", {}).get("guardrails") == f"disabled_by_request_for_{precision}_transfer"
            and (precision != "fp8" or report.get("audit", {}).get("policy_validation_passed") is True)
            and not checks
        )
    return (
        report.get("status") == "passed"
        and (
            report.get("checkpoint_precision") != "fp8"
            or (
                report.get("audit", {}).get("precision_dispatch") == "native_diffusers"
                and report.get("audit", {}).get("policy_validation_passed") is True
                and report.get("audit", {}).get("audit_type") == AUDIT_TYPE
                and report.get("audit", {}).get("schema_version") == AUDIT_SCHEMA_VERSION
            )
        )
        and report.get("safety_checker") is safety_checker
        and safety_valid
        and bool(report.get("artifacts"))
        and all(Path(item["path"]).is_file() and digest(item["path"]) == item["sha256"] for item in report["artifacts"])
    )


def summarize(root, cases, precision="fp8"):
    rows = []
    for modality, resolution in cases:
        report_path = root / modality / resolution / "result.json"
        report = json.loads(report_path.read_text()) if report_path.exists() else {}
        rows.append(
            {
                "modality": modality,
                "resolution": resolution,
                "precision": "BF16" if precision == "bf16" else "FP8 (checkpoint policy)",
                "fp8_policy": report.get("audit", {}).get("policy") if precision == "fp8" else None,
                "safety_checker": report.get("safety_checker"),
                "status": report.get("status", "pending"),
                "gpu": report.get("system", {}).get("gpu", ""),
                "output_width": report.get("outputs", {}).get("width", ""),
                "output_height": report.get("outputs", {}).get("height", ""),
                "frames": report.get("outputs", {}).get("frames", ""),
                **{
                    key: report.get("timing", {}).get(key, "")
                    for key in ("load_seconds", "pipeline_seconds", "export_seconds", "worker_seconds")
                },
            }
        )
    write_json(root / "summary.json", rows)
    with (root / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "| Modality | Resolution | Output | Frames | Pipeline (s) | Guardrails | Status |",
        "|---|---|---|---:|---:|---|---|",
    ]
    for row in rows:
        seconds = row["pipeline_seconds"]
        timing = f"{seconds:.3f}" if isinstance(seconds, (int, float)) else "N/A"
        safety = {True: "On", False: "Off", None: "Not recorded"}[row["safety_checker"]]
        lines.append(
            f"| {row['modality']} | {row['resolution']} | "
            f"{row['output_width']}x{row['output_height']} | {row['frames']} | {timing} | {safety} | {row['status']} |"
        )
    (root / "summary.md").write_text("\n".join(lines) + "\n")
    return rows


def worker(args):
    # Only workers import CUDA libraries. Downloads and matrix orchestration
    # never create a CUDA context and are excluded from pipeline latency.
    from dataclasses import asdict

    import torch
    from benchmark_safety import GuardrailDisabledTransferAudit, SafetyAudit, benchmark_prompt
    from diffusers.schedulers import UniPCMultistepScheduler
    from diffusers.utils import export_to_video
    from fp8_benchmark_audit import FP8PolicyAudit
    from modalities import RESOLUTIONS as SIZES
    from modalities import _prepare_call, _summarize_trace, _write_sound

    from cosmos3_streaming import StreamingConfig, build_pipeline

    modality, resolution = args.worker
    directory = args.output_dir / modality / resolution
    directory.mkdir(parents=True, exist_ok=True)
    prepared = json.loads((args.output_dir / "prepared.json").read_text())
    precision = prepared["manifest"].get("checkpoint_precision", "fp8")
    safety_checker = guardrails_enabled(prepared["manifest"], modality)
    trace = directory / "memory_trace.jsonl"
    report = {
        "status": "running",
        "modality": modality,
        "resolution": resolution,
        "full_generation": True,
        "checkpoint_precision": precision,
        "safety_checker": safety_checker,
        "timing": {},
        "artifacts": [],
    }
    audit = None
    if not safety_checker:
        report["safety_exception"] = f"Explicit {precision.upper()} Transfer run with guardrails off"
        report["prompt_preflight"] = next(
            row for row in json.loads((args.output_dir / "preflight.json").read_text()) if row["modality"] == modality
        )
    started = time.perf_counter()
    try:
        torch.cuda.set_device(args.device)
        properties = torch.cuda.get_device_properties(args.device)
        report["system"] = {
            "gpu": properties.name,
            "vram_bytes": properties.total_memory,
            "compute_capability": f"{properties.major}.{properties.minor}",
            "host": os.uname().nodename,
            "python": sys.version,
            "cuda": torch.version.cuda,
            "ram_bytes": os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"),
            "versions": {
                name: importlib.metadata.version(name)
                for name in (
                    ("torch", "diffusers", "transformers", "cosmos-guardrail")
                    + (("nvidia-modelopt",) if precision == "fp8" else ())
                )
            },
            "nvidia_smi": subprocess.check_output(
                [
                    "nvidia-smi",
                    (
                        "--query-gpu=name,uuid,memory.total,driver_version,"
                        "power.limit,pcie.link.gen.max,pcie.link.width.max"
                    ),
                    "--format=csv",
                ],
                text=True,
            ),
        }
        origin = json.loads(importlib.metadata.distribution("diffusers").read_text("direct_url.json") or "{}")
        report["diffusers_origin"] = origin
        check_diffusers_installation(prepared["manifest"])
        height, width = SIZES[resolution]
        inputs = args.output_dir / "inputs"
        call, report["inputs"], fps = _prepare_call(
            modality,
            Path(prepared["example_model_path"]),
            args.device,
            height,
            width,
            resolution,
            inputs / "v2v",
            inputs / "transfer",
            inputs / "action/bridge_20260501_0.mp4",
        )
        if call["prompt"] != benchmark_prompt(modality, prepared["example_model_path"], inputs):
            raise AssertionError("Generation prompt differs from the preflight prompt")
        report["settings"] = {
            key: value
            for key, value in call.items()
            if isinstance(value, (str, int, float, bool)) or key in ("output", "condition_frame_indexes_vision")
        }
        report["settings"]["seed"] = call["generator"].initial_seed()
        report["settings"]["flow_shift"] = 10.0
        report["settings"]["use_karras_sigmas"] = False
        if modality == "action":
            report["settings"]["action_resolution_tier"] = {"240p": 256, "480p": 480, "720p": 720}[resolution]
        config = StreamingConfig(device=args.device, memory_trace_path=str(trace), memory_trace_blocks=True)
        report["streaming_config"] = asdict(config)
        load_start = time.perf_counter()
        pipe = build_pipeline(
            prepared["model_path"],
            config=config,
            dtype=torch.bfloat16,
            checkpoint_precision=precision,
            local_files_only=True,
            enable_safety_checker=safety_checker,
        )
        pipe.scheduler = UniPCMultistepScheduler.from_config(
            pipe.scheduler.config, flow_shift=10.0, use_karras_sigmas=False
        )
        torch.cuda.synchronize(args.device)
        report["timing"]["load_seconds"] = time.perf_counter() - load_start
        if precision == "fp8":
            audit = FP8PolicyAudit(pipe, safety_checker=safety_checker)
        elif not safety_checker:
            audit = GuardrailDisabledTransferAudit(pipe)
        else:
            audit = SafetyAudit(pipe)
        write_json(directory / "result.json", report)
        print(f"Generating full {modality} {resolution} on {properties.name}", flush=True)
        generation_start = time.perf_counter()
        result = pipe(**call)
        torch.cuda.synchronize(args.device)
        report["timing"]["pipeline_seconds"] = time.perf_counter() - generation_start
        audit.validate(call["num_inference_steps"])
        videos = result["videos"] if isinstance(result, dict) else result
        expected_frames = 17 if modality == "action" else call["num_frames"]
        if len(videos) != expected_frames:
            raise AssertionError(f"Expected {expected_frames} frames, received {len(videos)}")
        output_width, output_height = videos[0].size
        if modality != "action" and (output_height, output_width) != (height, width):
            raise AssertionError("Output resolution does not match the requested resolution")
        export_start = time.perf_counter()
        video = directory / "video.mp4"
        export_to_video(videos, str(video), fps=fps, macro_block_size=1)
        outputs = {"frames": len(videos), "width": output_width, "height": output_height, "fps": fps}
        artifacts = [video]
        if modality in ("t2vs", "i2vs"):
            sound = directory / "audio.wav"
            outputs["sound"] = _write_sound(result, sound)
            artifacts.append(sound)
        if modality == "action":
            action = directory / "action.json"
            write_json(action, result["action"][0].float().cpu().tolist())
            artifacts.append(action)
        report["timing"]["export_seconds"] = time.perf_counter() - export_start
        report["outputs"] = outputs
        report["artifacts"] = [{"path": str(path.resolve()), "sha256": digest(path)} for path in artifacts]
        report["runtime_stats"] = asdict(pipe._streaming_runtime.stats)
        report["status"] = "passed"
    except Exception:  # noqa: BLE001 - persist any failed case and continue the matrix
        rejected = audit is not None and any(
            not event["passed"] and event.get("status") != "error" for event in audit.safety
        )
        report["status"] = "safety_rejected" if rejected else "failed"
        report["error"] = traceback.format_exc()
        # A failed/rejected case has no valid generation latency.
        report["timing"].pop("pipeline_seconds", None)
        print(report["error"], flush=True)
    finally:
        if audit is not None:
            report["audit"] = audit.report()
            audit.close()
        if trace.exists():
            report["memory"] = _summarize_trace(trace)
        report["timing"]["worker_seconds"] = time.perf_counter() - started
        write_json(directory / "result.json", report)
    return 0 if report["status"] == "passed" else 1


def main(precision="fp8"):
    parser = argparse.ArgumentParser(description=f"Full {precision.upper()} benchmark with safety checking by default")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, help="Dedicated benchmark cache (default: OUTPUT/cache)")
    parser.add_argument("--modalities", nargs="+", choices=MODALITIES, default=list(MODALITIES))
    parser.add_argument("--resolutions", nargs="+", choices=RESOLUTIONS, default=list(RESOLUTIONS))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Check every selected prompt before downloading generation weights or generating media",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--transfer-without-guardrails",
        action="store_true",
        help="Explicit Transfer-only exception; retain prompt rejection evidence but generate with guardrails off",
    )
    parser.add_argument("--revision", help="Override the checkpoint revision; resolved to an immutable Hub commit")
    parser.add_argument("--worker", nargs=2, metavar=("MODALITY", "RESOLUTION"), help=argparse.SUPPRESS)
    parser.add_argument("--preflight-worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.transfer_without_guardrails and args.modalities != ["transfer"]:
        parser.error("--transfer-without-guardrails requires --modalities transfer")
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configure_cache(args.cache_dir or args.output_dir / "cache")
    if args.worker:
        return worker(args)
    if args.preflight_worker:
        from benchmark_safety import run_preflight, write_preflight_markdown

        prepared = json.loads((args.output_dir / "prepared.json").read_text())
        try:
            check_diffusers_installation(prepared["manifest"])
            rows = run_preflight(
                prepared["example_model_path"],
                args.output_dir / "inputs",
                args.modalities,
                args.resolutions,
                args.device,
                prepared["manifest"]["checkpoint_precision"],
            )
        except Exception as exc:  # noqa: BLE001 - initialization failures must not disappear from the report
            rows = [
                {
                    "modality": modality,
                    "resolutions": args.resolutions,
                    "status": "preflight_error",
                    "stage": "preflight_initialization",
                    "generation_started": False,
                    "components": [],
                    "error": str(exc),
                }
                for modality in args.modalities
            ]
            traceback.print_exc()
        write_json(args.output_dir / "preflight.json", rows)
        write_preflight_markdown(args.output_dir / "preflight.md", rows)
        return 0  # Rejection/error statuses are in the report and handled by the parent.
    manifest = json.loads(MANIFEST.read_text())
    manifest.update(diffusers_installation())
    if precision == "bf16":
        manifest["model"] = manifest["bf16_model"]
        manifest["name"] = "cosmos3-nano-bf16-safety"
    if args.revision:
        manifest["model"]["revision"] = args.revision
    manifest["checkpoint_precision"] = precision
    if args.transfer_without_guardrails:
        manifest["transfer_without_guardrails"] = True
    recipe = args.output_dir / "recipe.json"
    previous = json.loads(recipe.read_text()) if recipe.exists() else None
    manifest["requested_model"] = dict(manifest["model"])
    if not args.preflight_only:
        manifest["model"] = resolve_model(manifest, previous)
    identity = {
        "source": fingerprint(),
        "preflight_only": args.preflight_only,
        "device": args.device,
        "manifest": manifest,
        "modalities": args.modalities,
        "resolutions": args.resolutions,
    }
    if recipe.exists():
        if previous != identity:
            raise ValueError("Recipe/source changed. Use a new output directory, never mix benchmark results")
        if not args.resume:
            raise ValueError("Output already initialized. Use --resume or a new output directory")
    write_json(recipe, identity)
    prepare(args, manifest)
    if args.prepare_only:
        print(f"Prepared checkpoint and inputs in {args.output_dir}")
        return 0
    cases = [(modality, resolution) for modality in args.modalities for resolution in args.resolutions]
    summarize(args.output_dir, cases, precision)
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    preflight_command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--output-dir",
        str(args.output_dir),
        "--cache-dir",
        os.environ["XDG_CACHE_HOME"],
        "--device",
        args.device,
        "--preflight-worker",
        "--modalities",
        *args.modalities,
        "--resolutions",
        *args.resolutions,
    ]
    print(f"Checking prompts before generation; log: {args.output_dir / 'preflight.log'}", flush=True)
    with (args.output_dir / "preflight.log").open("w") as log:
        subprocess.run(preflight_command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    preflight = {row["modality"]: row for row in json.loads((args.output_dir / "preflight.json").read_text())}
    if args.preflight_only:
        print((args.output_dir / "preflight.md").read_text(), flush=True)
        return 0 if all(row["status"] == "passed" for row in preflight.values()) else 1
    for modality, resolution in cases:
        safety_checker = guardrails_enabled(manifest, modality)
        directory = args.output_dir / modality / resolution
        result_path = directory / "result.json"
        if result_path.exists() and passed_and_intact(json.loads(result_path.read_text()), safety_checker):
            print(f"Verified completed case: {modality}/{resolution}", flush=True)
            continue
        if directory.exists():
            archive = args.output_dir / "attempts" / f"{modality}-{resolution}-{time.time_ns()}"
            archive.parent.mkdir(exist_ok=True)
            directory.rename(archive)
        directory.mkdir(parents=True)
        if not preflight_allows_generation(preflight[modality], safety_checker):
            write_json(
                result_path,
                {
                    "status": preflight[modality]["status"],
                    "modality": modality,
                    "resolution": resolution,
                    "checkpoint_precision": precision,
                    "safety_checker": True,
                    "generation_started": False,
                    "failure_stage": "prompt_preflight",
                    "preflight": preflight[modality],
                },
            )
            summarize(args.output_dir, cases, precision)
            continue
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--output-dir",
            str(args.output_dir),
            "--cache-dir",
            os.environ["XDG_CACHE_HOME"],
            "--device",
            args.device,
            "--worker",
            modality,
            resolution,
        ]
        print(f"Starting {modality}/{resolution}; log: {directory / 'run.log'}", flush=True)
        with (directory / "run.log").open("w") as log:
            completed = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
        if not result_path.exists() or json.loads(result_path.read_text()).get("status") == "running":
            write_json(result_path, {"status": "failed", "error": f"Worker exited {completed.returncode}; see run.log"})
        print(f"Finished {modality}/{resolution}: {json.loads(result_path.read_text())['status']}", flush=True)
        summarize(args.output_dir, cases, precision)
    rows = summarize(args.output_dir, cases, precision)
    write_json(
        args.output_dir / "completed.json",
        {"cases": len(rows), "passed": sum(row["status"] == "passed" for row in rows), "finished_at_unix": time.time()},
    )
    return 0 if all(row["status"] == "passed" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
