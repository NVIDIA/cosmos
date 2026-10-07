# Cosmos3-Nano CPU-GPU weight streaming

Run the Cosmos3-Nano Diffusers modular pipeline with model weights stored in
CPU RAM and streamed to the GPU as needed. Computation stays on the GPU.
The adapter supports BF16 and the official ModelOpt FP8 checkpoint without
modifying installed Diffusers source files.

## Install into your existing environment

Start with the shared [Diffusers environment setup](../../README.md#diffusers),
or activate your existing CUDA-enabled environment. From the Cosmos repository
root, change into this cookbook and install the adapter and benchmark dependencies:

```bash
cd cookbooks/cosmos3/generator/diffusers_cpu_offloading
uv pip install '.[benchmarks]'
```

For FP8, install the optional dependency in the same environment:

```bash
uv pip install '.[benchmarks,fp8]'
```

Run all examples and benchmark commands from this cookbook directory.
The installer reads `pyproject.toml`; no `uv.lock` or `uv sync` is required.
You can use `python -m pip install` instead of `uv pip install`.
If you only need generation, install `.` for BF16 or `'.[fp8]'` for FP8.
Use `--no-deps` only when your environment already has all required dependencies.

The adapter requires Python 3.12 or newer. Measurements used Linux, Python 3.12,
and PyTorch 2.13/CUDA 13. This cookbook tracks Diffusers `main`; benchmark reports
record the installed commit, and upstream changes can affect compatibility and timings.
A BF16-capable NVIDIA GPU and substantial system RAM are required. The
consumer-GPU benchmark hosts had 128 GB RAM; that is a tested configuration,
not a measured minimum requirement for every workload.

Accept the model and safety-model access terms on Hugging Face before running.
Use your normal Hugging Face authentication. Checkpoints are downloaded on
first use and reused from the cache afterward. Set `HF_HOME` to a sufficiently
large data volume if necessary. A manual checkpoint download is not required.

## Generate video with BF16

```python
import torch
from diffusers.utils import export_to_video
from cosmos3_streaming import StreamingConfig, build_pipeline

pipe = build_pipeline(
    "nvidia/Cosmos3-Nano",
    config=StreamingConfig(device="cuda:0"),
    dtype=torch.bfloat16,
)

frames = pipe(
    prompt='{"scene":"A robot arm working in a bright kitchen"}',
    negative_prompt="",
    num_frames=189,
    height=480,
    width=848,
    num_inference_steps=35,
    guidance_scale=6.0,
    fps=24.0,
    generator=torch.Generator(device="cuda:0").manual_seed(0),
    output="videos",
)
export_to_video(frames, "video.mp4", fps=24)
```

Use the official model-card prompts and conditioning assets for comparison with
the model's examples. The short prompt above illustrates the API.

## Use the official FP8 checkpoint

```python
pipe = build_pipeline(
    "nvidia/Cosmos3-Nano",
    revision="fp8",
    checkpoint_precision="fp8",
    config=StreamingConfig(device="cuda:0"),
    dtype=torch.bfloat16,
)
```

FP8 uses the `fp8` branch of `nvidia/Cosmos3-Nano`. If you omit the model ID,
`checkpoint_precision="fp8"` selects that repository automatically. For this
repository, an omitted FP8 revision defaults to `fp8`; an explicit commit hash
or branch is preserved. Other repository IDs retain their own default branch.
BF16 continues to use `nvidia/Cosmos3-Nano` on its default branch.

Diffusers reads the precision policy from the checkpoint and selects the
precision for each module and denoising step. Attention uses BF16 PyTorch SDPA.
The adapter refreshes cached understanding K/V when native precision changes.

Native FP8 generation completed all 21 modality/resolution cases on RTX 5070,
with safety checking enabled except for explicitly labeled Transfer runs.
Use the benchmark runners below to validate performance and inspect outputs
on your hardware.

## Other generation modalities

Use the same `pipe` instance. Diffusers selects the generation workflow from
the call arguments.

| Modality | Additional call inputs |
| --- | --- |
| Image-to-video | `image=` with a conditioning image |
| Video-to-video | `video=` and the appropriate frame-conditioning arguments |
| T2V or I2V with sound | `enable_sound=True` |
| Action | `action=CosmosActionCondition(...)` |
| Transfer | `control_videos=` with the matching control prompt and settings |

For example, add a conditioning image to the video call:

```python
from diffusers.utils import load_image

image = load_image("conditioning.jpg")
frames = pipe(
    prompt='{"scene":"The scene continues with natural motion"}',
    negative_prompt="",
    image=image,
    num_frames=189,
    height=480,
    width=848,
    num_inference_steps=35,
    guidance_scale=6.0,
    output="videos",
)
```

See the
[Cosmos3-Nano model card](https://huggingface.co/nvidia/Cosmos3-Nano) for official
inputs and generation settings.

## Behavior and limitations

- The default `memory_profile="auto"` chooses prefetch, VAE tiling, and the
  constrained-memory Transfer strategy. Users do not need tile-size settings.
- Weight storage is in CPU RAM. Checkpoint files and the Hugging Face cache on
  disk are not an implementation of per-block disk streaming.
- Safety checking is enabled by default. Rejections are not silently bypassed.
  Any explicit use of `enable_safety_checker=False` remains the caller's
  responsibility and must follow the model's applicable terms.
- Calls on a single pipeline instance must be sequential, not concurrent.
  The instance can be reused for different supported workloads.
- The smaller GPU-memory footprint trades off against PCIe transfer time.
  These helpers are for inference, not training.
- Cosmos3-Nano is the supported model. Do not assume Cosmos3-Super or every
  action/control configuration has been validated.
- Runtime validation, failure cleanup, and optional diagnostic tracing remain
  in the implementation. Experimental NVFP4 kernels and external kernel build
  dependencies are not included.

## Reproduce benchmarks

Use the separate [BF16](benchmarks/benchmark_bf16.py) and
[FP8](benchmarks/benchmark_fp8.py) runners. See the
[benchmark guide](benchmarks/README.md) for setup, full modality/resolution
matrices, safety preflight, Transfer exceptions, and timing definitions.

```bash
python benchmarks/benchmark_bf16.py --output-dir artifacts/benchmark-bf16
python benchmarks/benchmark_fp8.py --output-dir artifacts/benchmark-fp8
```

These run complete examples at each selected resolution. FP8 uses the
checkpoint-defined Diffusers policy and `nvidia/Cosmos3-Nano` with `revision="fp8"`. Reports
include generation settings, the precision policy, safety settings, and timings.

## Source layout

```text
cosmos3_streaming/
    __init__.py       Public pipeline builders and configuration exports
    config.py         Runtime configuration
    modular.py        Diffusers modular-pipeline integration
    runtime.py        Conditioning cache and streamed execution
    block_views.py    Views of the original transformer blocks
    offloading.py     Group-offload lifecycle handling
    fp8.py            Official ModelOpt checkpoint restoration
    attention/        BF16 SDPA implementation and backend interface
    memory_trace.py   Optional diagnostic tracing
    metrics.py        Lightweight request counters
benchmarks/           BF16/FP8 reproduction scripts, example inputs, and guide
```

This cookbook includes the runtime adapter, BF16/FP8 benchmark runners, and small
benchmark input files. It does not include development tests, lab tooling, model
weights, or generated media. `pyproject.toml` supports installation directly from
this directory; no separately published package is required.
