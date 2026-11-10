# Benchmark coverage and publication notes

[Back to the inference benchmark index](../inference_benchmarks.md)

The benchmark pages cover Cosmos3-Edge, Cosmos3-Nano, and Cosmos3-Super, plus their FP8 versions, for Generator and Reasoner measurements. Separate 4Step, Sim, Harmonizer, training, and competitive-model results are excluded. Separately identified Policy-DROID workloads remain distinct from base-model action results.

## Runtime and modality coverage

Generator tables include PyTorch, vLLM-Omni, Diffusers, SGLang, TensorRT-LLM, and NIM where validated measurements are available. Vision, audiovisual, transfer, and identified action workloads are kept separate. Reasoner tables distinguish vLLM, NIM, and eager Transformers timing protocols. FP8 and NVFP4 results are not combined.

Runtime presence does not imply every modality or configuration was measured. Some AMD and regional GPU campaigns, separate NIM Reasoner profiles, and Policy-DROID NIM workloads still need validation before publication. Measurements are added incrementally.

## Publication criteria

Publication is based on source reconciliation and workload validity. Source-verified results retain their measured performance, including differences among runtimes, GPUs, precisions, and resolutions. Results with unresolved source verification, request accounting, or workload definitions are left blank. Blank cells can also represent an unmeasured or unsuccessful configuration; they do not establish lack of support. No missing value is inferred from another runtime, GPU, precision, or resolution.

The existing GPU-count and resolution columns are retained. New Generator results follow the current 1-or-8 GPU selection; measurements available only at two GPUs are omitted. Previously published configurations retain their recorded GPU counts where validated. Results measured at a different GPU count are never substituted into a cell. Measurements at other canvas dimensions are not relabelled to fit an existing resolution tier.

See [measurement settings and reproduction limits](benchmark_methodology.md) for timing boundaries, precision recipes, workload budgets, and serving protocols. Compare only configurations whose recorded settings match.
