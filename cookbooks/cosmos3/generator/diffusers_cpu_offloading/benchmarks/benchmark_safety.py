"""Shared, mandatory safety auditing for BF16 and FP8 benchmarks."""

import hashlib
import json
import time
from pathlib import Path


def benchmark_prompt(modality, model, inputs):
    if modality == "action":
        return "Put the pot to the left of the purple item."
    if modality == "transfer":
        path = Path(inputs) / "transfer/edge/prompt.json"
    elif modality == "v2v":
        path = Path(inputs) / "v2v/input_prompt.json"
    else:
        name = {
            "t2v": "example_t2v_prompt.json",
            "t2vs": "example_t2vs_prompt.json",
            "i2v": "example_i2v_prompt.json",
            "i2vs": "example_i2v_prompt.json",
        }[modality]
        path = Path(model) / "assets" / name
    return json.dumps(json.loads(path.read_text()))


class GuardrailDisabledTransferAudit:
    """Explicit Transfer exception; never report a safety check as passed."""

    def __init__(self, pipe, precision="bf16"):
        if precision not in ("bf16", "fp8"):
            raise ValueError("Unsupported Transfer precision")
        self.precision = precision
        self.pipe = pipe
        self.safety = []
        self._check_disabled()

    def _check_disabled(self):
        if self.pipe.requires_safety_checker:
            raise AssertionError("Transfer exception requires guardrails to be explicitly disabled")

    def validate(self, steps):
        self._check_disabled()
        count = self.pipe._streaming_runtime.stats.denoising_calls
        if not count or count % steps:
            raise AssertionError(f"Incomplete denoising: {count} calls for {steps} steps")

    def report(self):
        return {"safety_checks": [], "guardrails": f"disabled_by_request_for_{self.precision}_transfer"}

    def close(self):
        pass


class SafetyAudit:
    """Observe the default checker, retaining its decisions and failing closed on errors."""

    def __init__(self, pipe):
        if not pipe.requires_safety_checker or pipe.safety_checker is None:
            raise AssertionError("Safety checking is mandatory for benchmark results")
        self.pipe = pipe
        self.safety = []
        self.components = []
        self.originals = []
        checker = pipe.safety_checker
        for model in checker.text_guardrail.safety_models:
            original = model.is_safe
            name = type(model).__name__
            self.originals.append((model, "is_safe", original))

            def checked_component(prompt, _original=original, _name=name):
                started = time.perf_counter()
                try:
                    passed, reason = _original(prompt)
                    # Qwen3Guard 0.3.1 returns True on internal errors. Such an
                    # error is not a successful safety evaluation for a benchmark.
                    if "unexpected error" in reason.lower():
                        raise RuntimeError(reason)
                except Exception as exc:
                    self.components.append(
                        {
                            "component": _name,
                            "status": "error",
                            "reason": str(exc),
                            "seconds": time.perf_counter() - started,
                        }
                    )
                    raise
                self.components.append(
                    {
                        "component": _name,
                        "status": "passed" if passed else "rejected",
                        "reason": reason,
                        "seconds": time.perf_counter() - started,
                    }
                )
                return passed, reason

            model.is_safe = checked_component
        for kind, name in (("text", "check_text_safety"), ("video", "check_video_safety")):
            original = getattr(checker, name)
            self.originals.append((checker, name, original))

            def checked(*args, _original=original, _kind=kind, **kwargs):
                started = time.perf_counter()
                try:
                    value = _original(*args, **kwargs)
                except Exception as exc:
                    self.safety.append({"kind": _kind, "passed": False, "status": "error", "reason": str(exc)})
                    raise
                event = {
                    "kind": _kind,
                    "passed": bool(value) if _kind == "text" else value is not None,
                    "seconds": time.perf_counter() - started,
                }
                self.safety.append(event)
                print(f"Safety: {event}", flush=True)
                return value

            setattr(checker, name, checked)

    def report(self):
        return {"safety_checks": self.safety, "prompt_components": self.components}

    def validate(self, steps=None):
        if steps is not None:
            count = self.pipe._streaming_runtime.stats.denoising_calls
            if not count or count % steps:
                raise AssertionError(f"Incomplete denoising: {count} calls for {steps} steps")
        for kind in ("text", "video"):
            events = [event for event in self.safety if event["kind"] == kind]
            if not events or not all(event["passed"] for event in events):
                raise AssertionError(f"Missing or rejected {kind} safety check")

    def close(self):
        for obj, name, original in reversed(self.originals):
            setattr(obj, name, original)


def run_preflight(model, inputs, modalities, resolutions, device, precision):
    from types import SimpleNamespace

    import torch
    from cosmos_guardrail.cosmos_guardrail import CosmosSafetyChecker

    # Load only the default guardrail models, never the Cosmos transformer,
    # vision encoder or VAE. Use the same placement as the pipeline's prompt check.
    checker = CosmosSafetyChecker()
    checker.to(device)  # cosmos-guardrail 0.3.1 .to() mutates in place and returns None
    audit = SafetyAudit(SimpleNamespace(requires_safety_checker=True, safety_checker=checker))
    rows = []
    try:
        for modality in modalities:
            prompt = benchmark_prompt(modality, model, inputs)
            audit.safety.clear()
            audit.components.clear()
            started = time.perf_counter()
            error = None
            try:
                with torch.inference_mode():
                    passed = checker.check_text_safety(prompt)
                status = "passed" if passed else "safety_rejected"
            except Exception as exc:  # noqa: BLE001 - record errors separately, never as passes
                status, error = "preflight_error", str(exc)
            rows.append(
                {
                    "modality": modality,
                    "resolutions": resolutions,
                    "precision": precision,
                    "status": status,
                    "stage": "before_generation",
                    "generation_started": False,
                    "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                    "seconds": time.perf_counter() - started,
                    "error": error,
                    "safety_checks": list(audit.safety),
                    "components": list(audit.components),
                }
            )
            print(f"Preflight {modality}: {status}", flush=True)
    finally:
        audit.close()
        checker.to("cpu")
    return rows


def write_preflight_markdown(path, rows):
    lines = [
        "# Prompt safety preflight",
        "",
        "These checks run before Cosmos model loading and generation. Output checks still run after generation.",
        "",
        "| Modality | Resolutions | Status | Rejecting/error component | Detail |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        failures = [event for event in row["components"] if event["status"] != "passed"]
        names = ", ".join(event["component"] for event in failures) or "None"
        reasons = []
        for event in failures:
            reason = event["reason"]
            # Keep large structured prompts out of the overview; full reasons
            # remain in JSON and logs. Retain the exact matching keyword.
            if "Match Word:" in reason:
                reason = reason.split("Prompt:", 1)[0] + reason[reason.rfind("Match Word:") :]
            reasons.append(reason.replace("|", "/").replace("\n", " "))
        detail = "; ".join(reasons) or row.get("error") or "Prompt checks passed"
        lines.append(f"| {row['modality']} | {', '.join(row['resolutions'])} | {row['status']} | {names} | {detail} |")
    Path(path).write_text("\n".join(lines) + "\n")
