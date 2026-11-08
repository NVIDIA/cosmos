# Benchmark coverage and validation notes

[Back to the inference benchmark index](../inference_benchmarks.md)

This update covers Cosmos3-Edge, Cosmos3-Nano, and Cosmos3-Super, plus their FP8 versions, for Generator and Reasoner measurements. Separate 4Step, Sim, Harmonizer, training, and competitive-model results are excluded. Existing separately identified Policy-DROID results remain distinct from base-model action results.

## Coverage added

- SGLang Generator measurements for Edge, Nano, and Super, including Nano/Super FP8, vision, audiovisual, transfer, and identified action embodiments.
- Nano/Super FP8 vLLM-Omni vision, audiovisual, and transfer measurements.
- Edge Diffusers datacenter measurements and additional validated PyTorch, vLLM-Omni, and TensorRT-LLM cells at the stated GPU counts.
- Edge FP8 vLLM-Omni request timings for datacenter, enterprise, and embedded platforms, with separate timing and workload definitions.
- FP8 Reasoner pages, additional BF16 Edge serving coverage, and separate NIM serving sections. FP8 and NVFP4 results are not combined.

The existing GPU-count column and resolution columns are retained. New Generator results follow the current 1-or-8 GPU selection; measurements available only at two GPUs are omitted. Previously published configurations retain their recorded GPU counts. Results measured at a different GPU count are never substituted into a cell. Runtime coverage is added only where measurements exist; an absent runtime or a blank cell does not establish lack of support.

The [anomaly review](benchmark_anomaly_review.md) records unresolved runtime/GPU reversals, source caveats, and remaining campaigns. This update does not establish complete coverage of every recent measurement. The audit corrected six misassigned Super FP8 transfer values, withheld 45 inconsistent Super FP8 enterprise Reasoner values, and added 193 omitted numeric cells: 100 NIM 2.0.0 entries (50 configurations shown on both Nano pages) and 93 enterprise Generator entries. All pre-existing numeric cells from before this PR remain unchanged.

## Numeric reconciliation

The Generator reconciliation found 3,216 previously published cells matching the available source values at display precision. Another 166 cells still lack an exact, complete source match in the audit. Those values are preserved rather than silently replaced.

For the existing Nano/Super BF16 Reasoner tables, 1,120 cells match. Five cells in the Nano/B300 workload with input 50, output 1, video 1 FPS, concurrency 256 disagree with a source spreadsheet: the request population is 511 in the current table and 512 in that spreadsheet, with corresponding small latency and throughput differences. The previously published values remain until the request-accounting difference is resolved. This is not treated as a confirmed regression or a transcription correction.

## Measurements awaiting clarification

| Area | Required clarification before further changes |
|---|---|
| Nano FP8 enterprise Generator | Resolve contradictory runtime identification before assigning these measurements to an engine. |
| Edge enterprise Diffusers | Verify generation/diffusion timing labels where the reported generation time is below diffusion time. |
| SGLang BF16 policy/robot | Verify the inconsistent action-resolution mapping before filling the policy table. |
| FP8 generic action samples | Record the embodiment before assigning values to AV, camera, or robot rows. |
| Super FP8 enterprise Reasoner | Withhold the entire vLLM block: output/concurrency labels conflict with request counts and output-token rates, and some rows have fractional counts or repeated unrelated values. Corrected request-level results are required. |
| Single-token Reasoner serving | Retain successful-request counts and disclose empty replies; obtain clean reruns where needed. |
| Campaign coverage | Reconcile the AMD and regional GPU Generator campaigns, separate NIM Reasoner profiles, and Edge Policy-DROID NIM workloads listed in the anomaly review. |
| Source completeness | Obtain full tables for missing Nano/Super datacenter and GB300 Reasoner cells, BF16 Edge workloads, and transfer measurements whose accessible table views are incomplete. |
| Reproduction metadata | Supply missing runtime versions, precise parallel splits, prompt/seed information, step budgets, warmup counts, and action payload definitions. |

No values are inferred for missing combinations. The remaining gaps are validation and source-completeness work, not evidence that a model or runtime is unsupported. See [measurement settings and reproduction limits](benchmark_methodology.md) before comparing engines.
