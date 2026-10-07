"""Observe Diffusers' checkpoint-defined FP8 policy during full generation."""

from collections import Counter
from dataclasses import asdict

AUDIT_TYPE = "diffusers_fp8_policy"
AUDIT_SCHEMA_VERSION = 1


class FP8PolicyAudit:
    """Validate native precision selection without changing forwards or kernels."""

    def __init__(self, pipe, safety_checker=True):
        from benchmark_safety import GuardrailDisabledTransferAudit, SafetyAudit
        from diffusers.pipelines.cosmos import mixed_precision as mixed

        self.mixed = mixed
        self.pipe = pipe
        self.policy = mixed.Cosmos3MixedPrecisionConfig.resolve(pipe.transformer)
        self.groups = {
            id(module): group
            for name, module in pipe.transformer.named_modules()
            if (group := mixed._classify_cosmos3_linear(name)) is not None and mixed._is_modelopt_fp8_linear(module)
        }
        if set(self.groups.values()) != {"generation", "reasoner"}:
            raise AssertionError("Expected ModelOpt FP8 linears in both Diffusers precision groups")
        self.safety_audit = (
            SafetyAudit(pipe) if safety_checker else GuardrailDisabledTransferAudit(pipe, precision="fp8")
        )
        self.safety = self.safety_audit.safety
        self.policy_validation_passed = False
        self.selections = Counter()
        self.handles = []
        try:
            for module in pipe.transformer.modules():
                if id(module) in self.groups:
                    self.handles.append(module.register_forward_pre_hook(self._observe))
        except Exception:
            self.close()
            raise

    def _observe(self, module, _inputs):
        step = self.pipe._streaming_runtime.stats.denoising_calls - 1
        group = self.groups[id(module)]
        runtime = getattr(self.pipe.transformer, self.mixed._RUNTIME_ATTRIBUTE, None)
        active = runtime is not None and runtime.active
        if active and runtime.config != self.policy:
            raise AssertionError("Diffusers runtime policy differs from the checkpoint policy")
        if self.policy.enabled and not active:
            raise AssertionError("Diffusers mixed-precision runtime is inactive during generation")
        mode = "W8A16" if active and runtime.use_high_precision(group) else "W8A8"
        self.selections[step, group, mode] += 1

    def report(self):
        return {
            "audit_type": AUDIT_TYPE,
            "schema_version": AUDIT_SCHEMA_VERSION,
            "policy": asdict(self.policy),
            "precision_dispatch": "native_diffusers",
            "policy_validation_passed": self.policy_validation_passed,
            **self.safety_audit.report(),
            "observed_linears": [
                {"step": step, "group": group, "selected_precision": mode, "calls": count}
                for (step, group, mode), count in sorted(self.selections.items())
            ],
        }

    def validate(self, steps):
        self.policy_validation_passed = False
        total_steps = self.pipe._streaming_runtime.stats.denoising_calls
        if steps <= 0 or not total_steps or total_steps % steps:
            raise AssertionError(f"Incomplete denoising: {total_steps} calls for {steps} steps")
        for step in range(total_steps):
            if not sum(self.selections[step, "generation", mode] for mode in ("W8A8", "W8A16")):
                raise AssertionError(f"Missing FP8 generation observations at step {step}")
        if not any(group == "reasoner" for _, group, _ in self.selections):
            raise AssertionError("Missing FP8 reasoner observations")
        for (step, group, mode), count in self.selections.items():
            if not 0 <= step < total_steps:
                raise AssertionError(f"FP8 observation outside the denoising loop: {step}")
            high = self.policy.enabled and (
                self.policy.use_high_precision(step % steps, steps)
                if group == "generation"
                else self.policy.reasoner_policy == "high_precision"
            )
            expected = "W8A16" if high else "W8A8"
            if count and mode != expected:
                raise AssertionError(f"Diffusers selected {mode}, expected {expected} for {group} at step {step}")
        self.safety_audit.validate(steps)
        self.policy_validation_passed = True

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.safety_audit.close()
