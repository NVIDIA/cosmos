# Reproduce Cosmos 3 Nano BF16 and FP8 benchmarks

These runners generate complete examples using BF16 or the FP8 checkpoint's
native Diffusers policy, with the default Cosmos safety checker enabled.
Each run records generation settings, the precision policy, timings, and outputs.

## Requirements

- Linux, Python 3.12 and an NVIDIA GPU supported by your CUDA-enabled
  PyTorch build, with a compatible NVIDIA driver. The target RTX 5070 has
  12 GB VRAM. We use a node configured with 128 GB system RAM; this is the test
  configuration, not a measured minimum RAM requirement.
- Allow at least 80 GB per precision of disk space for dependencies, checkpoint caches,
  compiler caches, and generated artifacts. A complete matrix can take hours.
- Access to `nvidia/Cosmos3-Nano` (including its `fp8` branch) and the safety-model repositories on Hugging
  Face. Accept any applicable model terms. For gated downloads, supply an
  `HF_TOKEN` through your usual secure environment setup. Do not put tokens
  into scripts, logs, or a public issue.
- Use an otherwise idle GPU. CPU speed, PCIe bandwidth, GPU power limits,
  driver version, and compiler-cache state can affect results.

## Run

From the Cosmos repository root, change into this cookbook:

```bash
cd cookbooks/cosmos3/generator/diffusers_cpu_offloading
```

Use your existing CUDA-enabled environment. Install the helper and benchmark
dependencies with `python -m pip install '.[benchmarks]'` for BF16, or
`python -m pip install '.[benchmarks,fp8]'` for FP8. An equivalent
`uv pip install` command works in an activated environment. No new environment
or `uv sync` is required. The installer provides the required dependencies,
and the runner checks the environment before generation.

The cookbook installs Diffusers from `main`. At the start of a benchmark, the
runner records the installed Git commit and package version in `recipe.json`.
Workers must use that revision, and resuming after a Diffusers upgrade requires
a new output directory. This keeps a matrix consistent without pinning every
user to one development commit. Compare reports from the same software revision
when evaluating performance differences.

```bash
python benchmarks/benchmark_bf16.py --output-dir artifacts/benchmark-bf16
python benchmarks/benchmark_fp8.py --output-dir artifacts/benchmark-fp8
```

No manual checkpoint download or tiling settings are needed. BF16 uses
`nvidia/Cosmos3-Nano`; FP8 uses the same repository with `revision="fp8"`.
The runner resolves the branch to an immutable commit for each new run and
retains that commit on resume. The runner also
downloads the example prompts and conditioning inputs it needs.

Subsequent runs reuse the dedicated cache under the output directory.
Use `--cache-dir /path/to/benchmark-cache` to share that cache between runs.
Use a **dedicated benchmark cache**, not your general Hugging Face cache:
the runner manages safety-model downloads there and runs generation offline.
Do not run concurrent preparations against the same cache.

To run just one complete example first:

```bash
python benchmarks/benchmark_bf16.py \
    --output-dir artifacts/bf16-t2v-480p \
    --modalities t2v --resolutions 480p
```

For FP8, use `benchmark_fp8.py` and a separate output directory with the same
modality and resolution arguments. Both commands use the full 35 denoising
steps and 189 frames. Safety checking cannot be disabled for this T2V example;
the explicit exception applies only to `--modalities transfer`.

You can prepare downloads before occupying a GPU by adding `--prepare-only`.
Then repeat the same command with `--resume` instead of `--prepare-only`.
To resume an interrupted matrix, add `--resume` to the original command. Only
successful cases with verified outputs are skipped. When a case is retried,
its prior artifacts are preserved under `attempts/`. Use a new output directory
if you change the code or benchmark settings.

## What the matrix runs

| Modality flag | Example | Steps | Frames | FPS | Seed | Guidance |
|---|---|---:|---:|---:|---:|---:|
| `t2v` | Text to video | 35 | 189 | 24 | 123 | 6 |
| `i2v` | Image to video | 35 | 189 | 24 | 1111 | 6 |
| `t2vs` | Text to video with sound | 35 | 189 | 24 | 0 | 6 |
| `i2vs` | Image to video with sound | 35 | 189 | 24 | 0 | 6 |
| `action` | Action policy with video output | 30 | 17 | 5 | 0 | 1 |
| `v2v` | Video-conditioned generation | 35 | 189 | 24 | 0 | 6 |
| `transfer` | Edge-conditioned transfer | 35 per chunk | 121 | 30 | 0 | 3 |

Each modality runs at `240p` (432x240), `480p` (848x480), and `720p`
(1280x720), in that order. Action uses its native resolution tiers 256, 480,
and 720 instead; the report records the actual output dimensions. Transfer
uses control guidance 1.5. All cases use UniPC, flow shift 10, no Karras sigmas,
and eight CPU compute threads. The adapter's automatic memory profile chooses
streaming and decode settings for the GPU; no tuning knobs are required.

For the tested Action input, the 240p tier produces 320x240 and both the 480p
and 720p tiers produce 640x480. The 720p-tier Action result is not a 1280x720
measurement.

Cases run sequentially, each in a fresh process. This prevents one modality's
resident state from carrying into the next. The compiler cache persists across
cases. Within Transfer, the low-memory adapter stages guidance branches
according to its automatic memory policy.

BF16 runs load the Nano checkpoint with `torch.bfloat16` and use BF16 PyTorch
SDPA attention. They do not use FP8 quantization or a mixed-precision schedule.

For FP8, Diffusers resolves the checkpoint's precision policy, including the first/last
step counts and the precision assigned to each module group. The runner does
not hardcode those choices. A checkpoint without a mixed-precision schedule
uses its base W8A8 forwards. Cached understanding K/V is rebuilt when precision
changes, and the schedule restarts for each Transfer chunk.

The FP8 audit uses read-only forward hooks to observe Diffusers' selected
precision during linear calls. It checks the selections against the resolved
checkpoint policy and requires observations for every denoising step. It does
not replace module forwards or execution functions, and it is not a hardware
kernel profiler. Reports contain the resolved policy and per-step observations.
Attention uses BF16 PyTorch SDPA.

The default checker in `cosmos-guardrail==0.3.1` uses
Blocklist and Qwen3Guard for prompts, and RetinaFace face processing for video
output. Its separate video-content classifier is not enabled by the package
default. The runner preserves that default rather than claiming additional
checks. A rejected prompt or output is recorded as `safety_rejected`; it does
not silently disable checking, change the prompt, or invent a latency.

## Read and compare results

The output directory contains:

```text
recipe.json                  # run configuration
prepared.json                # download preparation details
summary.csv                  # table for analysis
summary.json
summary.md                   # readable consolidated table
completed.json               # written only after the matrix finishes
t2v/240p/
    run.log
    result.json              # hardware, settings, timings, precision/safety audit
    video.mp4
    memory_trace.jsonl
...                          # other modality/resolution cases
```

Sound cases also produce `audio.wav`; Action produces `action.json`.

**Use `pipeline_seconds` to compare generation latency.** It measures the
complete pipeline call through CUDA completion: prompt checking, conditioning,
understanding, denoising, decoding, and output checking. Model loading, input
downloads, and MP4/WAV encoding are outside this timer. `load_seconds` and
`export_seconds` are recorded separately. `worker_seconds` covers the measured
worker body, including those stages and auditing, but excludes Python imports
and model/asset downloads.

These are instrumented, single-run measurements, not warmed medians. Compilation
that occurs during the pipeline call is included. For variability estimates,
repeat into a new output directory with the same `--cache-dir`, then compare
the same cases and note that compiler caches are warm. Compare actual
dimensions, frames, seeds, precision and safety settings, not resolution labels
alone. A timing match does not establish pixel-identical outputs across GPUs;
inspect the videos as part of quality validation.

The runner continues after failed cases and exits nonzero if any case did not
pass. `completed.json` means all requested cases were attempted, not that every
case passed. A `passed` result verifies execution, policy, safety calls, frame
count, and output geometry. It does not replace a human quality review.

Video inputs come from NVIDIA's public examples. V2V prompt JSONs are bundled
with this runner. Transfer uses the Cosmos cookbook prompt. If the default guardrail
rejects it, the runner records the rejection. The explicit exception below
allows Transfer generation without checks while retaining that preflight result.

## Prompt preflight and the Transfer exception

Check prompts without downloading generation weights:

```bash
python benchmarks/benchmark_bf16.py --output-dir artifacts/prompt-preflight \
    --preflight-only
```

Read `preflight.md` and `preflight.json` for the rejecting component and reason.
A rejection is distinct from a checker error. Use separate output directories
for preflight-only checks and generation. Both entry points
support preflight without downloading generation weights.

Safety checks remain enabled for normal runs. To benchmark Transfer with
guardrails off, use a separate output directory:

```bash
python benchmarks/benchmark_bf16.py --output-dir artifacts/bf16-transfer-no-guardrails \
    --modalities transfer --transfer-without-guardrails
python benchmarks/benchmark_fp8.py --output-dir artifacts/fp8-transfer-no-guardrails \
    --modalities transfer --transfer-without-guardrails
```

This option is rejected for any other modality selection. The prompt preflight
still runs and retains rejection evidence; checker errors are not bypassed.
Reports explicitly label checks as off. FP8 precision auditing is still required.
Use this exception only in accordance with the model's applicable terms.

No benchmark is launched during installation. The scripts do not provision
cluster nodes or include internal lab paths. Run the commands on the GPU whose
performance you intend to measure.
