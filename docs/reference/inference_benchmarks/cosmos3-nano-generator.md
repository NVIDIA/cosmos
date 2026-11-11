# Cosmos3-Nano Generator Inference Benchmarks

[Back to the inference benchmark index](../inference_benchmarks.md)

See [measurement settings and reproduction limits](benchmark_methodology.md) for runtime-specific timing, precision, and workload details.

These tables report **Cosmos3-Nano Generator** latency in seconds. Lower is better.

Every row reports one GPU configuration, named in the **GPUs** column, and all three resolutions in that row come from it. The configuration is one GPU when the model fits on a single device at its highest resolution, 720p, and eight GPUs when it does not. The **GPUs** column is authoritative: engines can use different counts when their memory requirements differ. Cosmos3-Nano rows use one GPU at 720p. Additional hardware is included only where that configuration was measured.

Empty cells have no publishable measurement at the stated configuration. They can reflect an unmeasured workload, an unsuccessful run, or a measurement awaiting validation; they do not establish lack of support.

## Table of Contents

- [Benchmark methodology](#benchmark-methodology)
- [Workload definitions](#workload-definitions)
- [Primary vision generation](#primary-vision-generation)
  - [Text-to-Video (t2v)](#text-to-video-t2v)
  - [Image-to-Video (i2v)](#image-to-video-i2v)
  - [Text-to-Image (t2i)](#text-to-image-t2i)
- [Additional audiovisual generation](#additional-audiovisual-generation)
  - [Video-to-Video (v2v)](#video-to-video-v2v)
  - [Text-to-Audio-and-Video (t2av)](#text-to-audio-and-video-t2av)
  - [Video-to-Audio-and-Video (v2av)](#video-to-audio-and-video-v2av)
  - [Image-to-Audio-and-Video (i2av)](#image-to-audio-and-video-i2av)
- [Transfer generation](#transfer-generation)
- [Action generation](#action-generation)
  - [Forward Dynamics — AV](#forward-dynamics--autonomous-vehicle-av)
  - [Forward Dynamics — Camera](#forward-dynamics--camera)
  - [Forward Dynamics — Robot](#forward-dynamics--robot)
  - [Inverse Dynamics — AV](#inverse-dynamics--autonomous-vehicle-av)
  - [Inverse Dynamics — Robot](#inverse-dynamics--robot)
  - [Policy — AV](#policy--autonomous-vehicle-av)
  - [Policy — Robot](#policy--robot)
  - [Policy — DROID](#policy--droid)

## Benchmark methodology

The primary t2v, i2v, and t2i tables cover PyTorch, vLLM-Omni, Diffusers, SGLang, TensorRT-LLM, and NIM where measurements are available. Their vLLM-Omni values come from a dedicated vLLM-Omni campaign; NIM values use the measured 2.1.0 latency profiles, with separately identified 2.0.0 enterprise measurements; the published PyTorch values use the previously measured campaign where validated. Those tables use BF16 precision (NIM uses its default precision, FP8), batch size 1, with prompt, seed, and sampler details retained only where documented. Recorded primary video workloads use 189 frames at 24 FPS. The separate SGLang tables do not retain a complete frame/step budget and software build alongside the measured values; see the reproduction limits before comparing them.

The additional audiovisual and action tables come from NVIDIA internal measurements. PyTorch values are average generation (sampling) latency from the native OSS path, using **CUDA Graphs disabled** and the **latency** automatic-sharding preset. vLLM-Omni values for text-to-audio-and-video (`t2av`/`t2vs`) and image-to-audio-and-video (`i2av`/`i2vs`) come from a separate vLLM-Omni audiovisual campaign. vLLM-Omni action values were measured with the `vllm/vllm-omni:cosmos3` image and the official action cookbook samples; `DIFFUSION_ATTENTION_BACKEND=TORCH_SDPA` was not set. Forward-dynamics cells are the mean of `av_forward`, `av_left`, and `av_right`; inverse-dynamics cells are the mean of `av_inverse_0` and `av_inverse_1`. That action sweep covers 1, 2, and 4 GPUs only, because its 8-GPU Ulysses runs failed a sequence-length divisibility check; those rows therefore report the single-GPU configuration. Policy-DROID is a separate checkpoint and was measured only at 480p on one GPU. That action sweep did not cover Super on H20, H100 NVL, or H100 80GB HBM3. Diffusers values come from separate campaigns: single-GPU runs (all modalities, and video-to-video), 4- and 8-GPU runs, and transfer. vLLM-Omni transfer values come from their own campaign. Where the multi-GPU Diffusers campaign offers several tensor- and context-parallel splits at the same GPU count, the fastest is published. Its 1-GPU numbers are not used, because that sweep ran a different denoising-step budget than the single-GPU reports. TensorRT-LLM values come from a TensorRT-LLM campaign that sweeps tensor-parallel, CFG-parallel, and Ulysses-parallel splits at each GPU count; the fastest split at the count a row reports is the one published. That campaign covers six GPUs, so the TensorRT-LLM rows for H100 NVL and H200 NVL are empty, and it includes no action workloads. Values are rounded to two decimal places.

These measurements establish the reported timing matrix but do not expose every prompt and action payload in this repository. The linked public recipes explain modality behavior and provide representative payloads; their example-specific frame counts and action chunk sizes should not be treated as the exact internal benchmark inputs.

## Workload definitions

| Workload | Input | Output |
|---|---|---|
| Video-to-video (`v2v`) | Text prompt and source video | Generated video |
| Text-to-audio-and-video (`t2av`) | Text prompt | Synchronized video and sound |
| Video-to-audio-and-video (`v2av`) | Text prompt and source video | Generated video with synchronized sound |
| Image-to-audio-and-video (`i2av`) | Text prompt and source image | Generated video with synchronized sound |
| Transfer video-to-video (`transfer`) | Text prompt, source video, and one control hint | Generated video steered by that hint |
| Forward dynamics | Initial visual observation and an action trajectory | Future-observation rollout video |
| Inverse dynamics | Observed video | Recovered action trajectory; some serving integrations also return video |
| Policy | Initial visual observation, instruction, and optional state | Predicted action trajectory and, for general Generator paths, a rollout video |
| Policy-DROID | Multiview wrist/exterior observations for the DROID embodiment | Predicted action chunk from the Policy-DROID checkpoint |

The benchmark uses `t2av`, `v2av`, and `i2av`; some public recipes call the same sound-producing modes `t2vs`, `v2vs`, and `i2vs`. See the [audiovisual cookbook](https://nvda.ws/4iGZxA5) for generation inputs and the [action cookbook](https://nvda.ws/487IcM4) for action representations and output contracts. vLLM-Omni request shapes are maintained in the [Nano recipe](https://github.com/vllm-project/vllm-omni/blob/main/recipes/cosmos3/Cosmos3-Nano.md). Public recipes cover more embodiments than these measurements; the action tables below intentionally use only its measured domain rows.

## Primary vision generation

### Text-to-Video (t2v)

| GPU | Engine | GPUs | 256p | 480p | 720p |
|---|---|:-:|---:|---:|---:|
| **RTX PRO 6000 Blackwell** | PyTorch | 1 | 13.95 | 180.81 | 786.37 |
|  | vLLM-Omni | 1 | 10.65 | 105.61 | 371.53 |
|  | Diffusers | 1 | 11.20 | 112.00 | 392.00 |
|  | TensorRT-LLM | 1 | 10.27 | 102.55 | 369.78 |
|  | NIM | 1 | 5.06 | 46.32 | 170.57 |
|  | SGLang | 1 | 10.45 | 104.21 | 371.54 |
| **H20** | PyTorch | 1 | 30.57 | 257.51 | 931.39 |
|  | vLLM-Omni | 1 | 28.58 | 256.97 | 929.81 |
|  | Diffusers | 1 | 30.20 | 258.00 | 926.00 |
|  | TensorRT-LLM | 1 | 26.92 | 251.79 | 908.90 |
|  | NIM | 1 | 11.78 | 86.02 | 282.28 |
|  | SGLang | 1 | 28.62 | 258.93 | 932.87 |
| **H100 NVL** | PyTorch | 1 | 10.03 | 84.12 | 297.27 |
|  | vLLM-Omni | 1 | 9.25 | 80.75 | 281.99 |
|  | Diffusers | 1 | 11.00 | 90.00 | 324.20 |
|  | TensorRT-LLM | — |  |  |  |
|  | NIM | — |  |  |  |
| **H200 NVL** | PyTorch | 1 | 8.17 | 69.79 | 244.39 |
|  | vLLM-Omni | 1 | 7.44 | 64.58 | 240.05 |
|  | Diffusers | 1 | 9.00 | 74.00 | 276.20 |
|  | TensorRT-LLM | — |  |  |  |
|  | NIM | — |  |  |  |
| **H100 80GB HBM3** | PyTorch | 1 | 7.61 | 59.83 | 207.78 |
|  | vLLM-Omni | 1 | 6.97 | 58.17 | 202.29 |
|  | Diffusers | 1 | 9.00 | 68.00 | 240.00 |
|  | TensorRT-LLM | 1 | 6.50 | 60.10 | 223.51 |
|  | NIM | 1 | 3.34 | 24.71 | 86.16 |
|  | SGLang | 1 | 6.46 | 55.05 | 195.89 |
| **H200 141GB HBM3** | PyTorch | 1 | 7.53 | 60.18 | 214.28 |
|  | vLLM-Omni | 1 | 6.79 | 58.14 | 208.36 |
|  | Diffusers | 1 | 9.00 | 67.00 | 239.60 |
|  | TensorRT-LLM | 1 | 6.42 | 60.57 | 225.06 |
|  | NIM | 1 | 3.15 | 24.62 | 85.55 |
|  | SGLang | 1 | 6.44 | 56.80 | 201.67 |
| **B200** | PyTorch | 1 | 4.56 | 33.20 | 114.85 |
|  | vLLM-Omni | 1 | 4.03 | 32.04 | 107.79 |
|  | Diffusers | 1 | 7.00 | 36.80 | 117.00 |
|  | TensorRT-LLM | 1 | 3.43 | 28.28 | 100.02 |
|  | NIM | 1 | 2.00 | 13.71 | 49.86 |
|  | SGLang | 1 | 3.66 | 32.38 | 122.12 |
| **B300** | PyTorch | — |  |  |  |
|  | vLLM-Omni | 1 | 4.46 | 32.18 | 102.10 |
|  | Diffusers | 1 | 6.40 | 39.32 | 121.74 |
|  | TensorRT-LLM | 1 | 3.31 | 26.52 | 91.15 |
|  | NIM | 1 | 1.98 | 13.10 | 45.76 |
|  | SGLang | 1 | 3.33 | 27.13 | 93.10 |
| **DGX Spark** | NIM | 1 | 36.78 | 358.61 | 1360.13 |
| **RTX PRO 5000 Blackwell 72GB** | NIM | 1 | 13.15 | 143.06 | 557.53 |

### Image-to-Video (i2v)

| GPU | Engine | GPUs | 256p | 480p | 720p |
|---|---|:-:|---:|---:|---:|
| **RTX PRO 6000 Blackwell** | PyTorch | 1 | 14.08 | 182.14 | 788.80 |
|  | vLLM-Omni | 1 | 11.04 | 107.77 | 376.44 |
|  | Diffusers | 1 | 12.00 | 112.00 | 397.00 |
|  | TensorRT-LLM | 1 | 10.77 | 105.03 | 374.89 |
|  | NIM | 1 | 4.90 | 46.43 | 171.34 |
|  | SGLang | 1 | 10.56 | 104.74 | 375.95 |
| **H20** | PyTorch | 1 | 31.36 | 257.10 | 933.07 |
|  | vLLM-Omni | 1 | 29.50 | 261.56 | 940.16 |
|  | Diffusers | 1 | 31.00 | 258.00 | 925.00 |
|  | TensorRT-LLM | 1 | 28.07 | 254.79 | 915.71 |
|  | NIM | 1 | 11.67 | 86.66 | 282.41 |
|  | SGLang | 1 | 28.82 | 259.62 | 935.20 |
| **H100 NVL** | PyTorch | 1 | 10.19 | 84.50 | 298.57 |
|  | vLLM-Omni | 1 | 9.62 | 82.61 | 286.19 |
|  | Diffusers | 1 | 11.00 | 91.00 | 325.20 |
|  | TensorRT-LLM | — |  |  |  |
|  | NIM | — |  |  |  |
| **H200 NVL** | PyTorch | 1 | 8.27 | 69.99 | 246.62 |
|  | vLLM-Omni | 1 | 7.83 | 66.39 | 243.52 |
|  | Diffusers | 1 | 9.00 | 74.00 | 275.20 |
|  | TensorRT-LLM | — |  |  |  |
|  | NIM | — |  |  |  |
| **H100 80GB HBM3** | PyTorch | 1 | 7.64 | 59.95 | 207.87 |
|  | vLLM-Omni | 1 | 7.37 | 59.77 | 205.97 |
|  | Diffusers | 1 | 9.00 | 68.00 | 239.80 |
|  | TensorRT-LLM | 1 | 6.90 | 63.24 | 226.15 |
|  | NIM | 1 | 3.39 | 24.65 | 86.60 |
|  | SGLang | 1 | 6.56 | 55.78 | 196.29 |
| **H200 141GB HBM3** | PyTorch | 1 | 7.65 | 60.51 | 214.80 |
|  | vLLM-Omni | 1 | 7.28 | 59.64 | 209.65 |
|  | Diffusers | 1 | 9.00 | 67.20 | 240.00 |
|  | TensorRT-LLM | 1 | 6.74 | 63.54 | 231.15 |
|  | NIM | 1 | 3.16 | 24.67 | 87.67 |
|  | SGLang | 1 | 6.53 | 56.95 | 202.15 |
| **B200** | PyTorch | 1 | 4.60 | 33.08 | 113.90 |
|  | vLLM-Omni | 1 | 4.33 | 33.09 | 110.08 |
|  | Diffusers | 1 | 7.00 | 37.00 | 116.00 |
|  | TensorRT-LLM | 1 | 3.70 | 29.33 | 102.35 |
|  | NIM | 1 | 1.95 | 13.76 | 49.16 |
|  | SGLang | 1 | 3.77 | 33.02 | 120.48 |
| **B300** | PyTorch | — |  |  |  |
|  | vLLM-Omni | 1 | 5.61 | 33.45 | 104.75 |
|  | Diffusers | 1 | 13.40 | 49.59 | 131.25 |
|  | TensorRT-LLM | 1 | 3.70 | 27.73 | 93.88 |
|  | NIM | 1 | 2.07 | 13.63 | 46.81 |
|  | SGLang | 1 | 3.43 | 27.30 | 93.22 |
| **DGX Spark** | NIM | 1 | 36.57 | 354.86 | 1350.29 |
| **RTX PRO 5000 Blackwell 72GB** | NIM | 1 | 13.24 | 143.25 | 557.87 |

### Text-to-Image (t2i)

| GPU | Engine | GPUs | 256p | 480p | 720p |
|---|---|:-:|---:|---:|---:|
| **RTX PRO 6000 Blackwell** | PyTorch | 1 | 2.99 | 4.51 | 7.12 |
|  | vLLM-Omni | 1 | 1.59 | 2.87 | 4.99 |
|  | Diffusers | 1 | 2.00 | 4.00 | 5.00 |
|  | TensorRT-LLM | 1 | 1.18 | 2.35 | 4.50 |
|  | NIM | 1 | 0.72 | 1.00 | 1.48 |
|  | SGLang | 1 | 1.39 | 1.85 | 3.33 |
| **H20** | PyTorch | 1 | 3.06 | 6.51 | 12.31 |
|  | vLLM-Omni | 1 | 1.73 | 4.92 | 10.73 |
|  | Diffusers | 1 | 3.00 | 6.00 | 10.00 |
|  | TensorRT-LLM | 1 | 1.94 | 5.55 | 11.56 |
|  | NIM | 1 | 1.09 | 1.61 | 2.80 |
|  | SGLang | 1 | 1.88 | 3.40 | 7.59 |
| **H100 NVL** | PyTorch | 1 | 2.77 | 2.83 | 4.21 |
|  | vLLM-Omni | 1 | 1.55 | 1.92 | 3.43 |
|  | Diffusers | 1 | 3.00 | 3.00 | 4.00 |
|  | TensorRT-LLM | — |  |  |  |
|  | NIM | — |  |  |  |
| **H200 NVL** | PyTorch | 1 | 2.75 | 2.85 | 3.58 |
|  | vLLM-Omni | 1 | 1.53 | 1.58 | 2.81 |
|  | Diffusers | 1 | 3.00 | 3.00 | 4.00 |
|  | TensorRT-LLM | — |  |  |  |
|  | NIM | — |  |  |  |
| **H100 80GB HBM3** | PyTorch | 1 | 3.01 | 3.01 | 3.45 |
|  | vLLM-Omni | 1 | 1.61 | 1.53 | 2.61 |
|  | Diffusers | 1 | 3.00 | 3.00 | 4.00 |
|  | TensorRT-LLM | 1 | 1.02 | 1.30 | 2.63 |
|  | NIM | 1 | 1.02 | 1.04 | 1.15 |
|  | SGLang | 1 | 1.69 | 1.76 | 1.75 |
| **H200 141GB HBM3** | PyTorch | 1 | 2.96 | 3.04 | 3.28 |
|  | vLLM-Omni | 1 | 1.57 | 1.52 | 2.60 |
|  | Diffusers | 1 | 3.00 | 3.00 | 4.00 |
|  | TensorRT-LLM | 1 | 1.05 | 1.30 | 2.60 |
|  | NIM | 1 | 1.02 | 1.05 | 1.08 |
|  | SGLang | 1 | 1.77 | 1.78 | 1.77 |
| **B200** | PyTorch | 1 | 2.68 | 2.75 | 2.87 |
|  | vLLM-Omni | 1 | 1.49 | 1.20 | 1.76 |
|  | Diffusers | 1 | 3.00 | 3.20 | 3.00 |
|  | TensorRT-LLM | 1 | 0.86 | 0.88 | 1.40 |
|  | NIM | 1 | 0.58 | 0.63 | 0.67 |
|  | SGLang | 1 | 1.82 | 1.91 | 1.82 |
| **B300** | PyTorch | — |  |  |  |
|  | vLLM-Omni | 1 | 1.97 | 1.81 | 2.34 |
|  | Diffusers | 1 | 3.86 | 3.86 | 3.99 |
|  | TensorRT-LLM | 1 | 1.36 | 1.30 | 1.40 |
|  | NIM | 1 | 0.72 | 0.84 | 0.87 |
|  | SGLang | 1 | 3.04 | 3.12 | 1.80 |
| **DGX Spark** | NIM | 1 | 2.34 | 3.87 | 7.47 |
| **RTX PRO 5000 Blackwell 72GB** | NIM | 1 | 1.63 | 2.08 | 3.45 |

<sub>Notes:
1. Measurements come from separate campaigns; identical prompt, seed, step budget, and timing boundaries are not established across engines. See the measurement settings before comparing rows.
2. Multi-GPU parallelism is runtime-specific and can split across tensor, CFG, context, Ring, or Ulysses dimensions; see the recorded settings.
3. vLLM-Omni numbers are for the upcoming public release in the vLLM-Omni repo and are subject to change; the H100 NVL values in particular are preliminary.
4. Diffusers numbers use the HuggingFace `diffusers` integration without custom CUDA graphs. Single-GPU and multi-GPU values come from separate campaigns.
5. PyTorch numbers report average generation (sampling) time from OSS inference benchmarking.
6. At 256p, multi-GPU configurations on B300 may underperform single-GPU because of small-workload tensor-parallel overhead, so single-GPU is the recommended deployment at this resolution.
7. NIM numbers use NIM 2.0.0 on DGX Spark and RTX PRO 5000, and NIM 2.1.0 (`nvcr.io/nim/nvidia/cosmos3:2.1.0`) on the other displayed platforms using latency profiles with FP8 precision, and report average generation time, excluding request processing and output encoding; step budgets and other workload differences still limit comparisons with other backends.</sub>

## Additional audiovisual generation

### Video-to-Video (v2v)

A text prompt and source video condition a generated continuation or transformation.

| GPU | Engine | GPUs | 256p | 480p | 720p |
|---|---|:-:|---:|---:|---:|
| **RTX PRO 6000 Blackwell** | PyTorch | 1 | 13.73 | 180.54 | 785.41 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 1 | 24.19 | 135.85 | 432.18 |
|  | TensorRT-LLM | 1 | 10.36 | 102.50 | 370.24 |
|  | SGLang | 1 | 13.26 | 109.65 | 381.71 |
| **H20** | PyTorch | 1 | 30.44 | 257.79 | 930.96 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 1 | 44.88 | 287.55 | 974.75 |
|  | TensorRT-LLM | 1 | 27.10 | 250.57 | 908.07 |
|  | SGLang | 1 | 32.27 | 267.21 | 938.34 |
| **H100 NVL** | PyTorch | 1 | 10.18 | 81.73 | 297.83 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 1 | 23.66 | 107.79 | 354.66 |
|  | TensorRT-LLM | — |  |  |  |
| **H200 NVL** | PyTorch | 1 | 8.10 | 69.70 | 250.33 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 1 | 21.40 | 96.52 | 313.13 |
|  | TensorRT-LLM | — |  |  |  |
| **H100 80GB HBM3** | PyTorch | 1 | 7.50 | 59.75 | 207.36 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 1 | 19.48 | 86.51 | 267.90 |
|  | TensorRT-LLM | 1 | 6.63 | 60.18 | 229.04 |
|  | SGLang | 1 | 9.21 | 60.48 | 200.71 |
| **H200 141GB HBM3** | PyTorch | 1 | 7.48 | 60.23 | 214.03 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 1 | 18.93 | 85.52 | 266.84 |
|  | TensorRT-LLM | 1 | 6.59 | 61.40 | 229.48 |
|  | SGLang | 1 | 9.33 | 61.36 | 206.40 |
| **B200** | PyTorch | 1 | 4.52 | 32.53 | 114.16 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 1 | 16.93 | 52.32 | 137.54 |
|  | TensorRT-LLM | 1 | 3.54 | 28.37 | 100.29 |
|  | SGLang | 1 | 6.24 | 36.83 | 125.59 |
| **B300** | PyTorch | — |  |  |  |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 1 | 16.14 | 50.74 | 131.11 |
|  | TensorRT-LLM | 1 | 3.44 | 26.63 | 91.40 |
|  | SGLang | 1 | 7.47 | 32.48 | 97.20 |
| **DGX Spark** | NIM | 1 | 37.20 | 360.62 | 1371.19 |
| **RTX PRO 5000 Blackwell 72GB** | NIM | 1 | 13.58 | 143.11 | 557.23 |

### Text-to-Audio-and-Video (t2av)

A text prompt produces synchronized video and sound.

| GPU | Engine | GPUs | 256p | 480p | 720p |
|---|---|:-:|---:|---:|---:|
| **RTX PRO 6000 Blackwell** | PyTorch | 1 | 14.13 | 180.98 | 794.74 |
|  | vLLM-Omni | 1 | 12.25 | 106.02 | 373.10 |
|  | Diffusers | 1 | 13.00 | 113.00 | 393.00 |
|  | TensorRT-LLM | 1 | 11.35 | 103.28 | 374.75 |
|  | SGLang | 1 | 12.01 | 105.59 | 377.00 |
| **H20** | PyTorch | 1 | 30.82 | 256.41 | 922.35 |
|  | vLLM-Omni | 1 | 29.43 | 260.84 | 937.93 |
|  | Diffusers | 1 | 32.00 | 263.00 | 932.60 |
|  | TensorRT-LLM | 1 | 29.38 | 254.88 | 913.33 |
|  | SGLang | 1 | 30.61 | 262.55 | 944.84 |
| **H100 NVL** | PyTorch | 1 | 9.99 | 82.32 | 297.88 |
|  | vLLM-Omni | 1 | 9.92 | 81.57 | 291.91 |
|  | Diffusers | 1 | 12.00 | 91.00 | 324.60 |
|  | TensorRT-LLM | — |  |  |  |
| **H200 NVL** | PyTorch | 1 | 8.20 | 69.88 | 250.46 |
|  | vLLM-Omni | 1 | 7.82 | 65.42 | 244.60 |
|  | Diffusers | 1 | 10.00 | 75.00 | 277.80 |
|  | TensorRT-LLM | — |  |  |  |
| **H100 80GB HBM3** | PyTorch | 1 | 7.60 | 59.09 | 207.87 |
|  | vLLM-Omni | 1 | 7.34 | 58.65 | 203.92 |
|  | Diffusers | 1 | 9.40 | 69.00 | 242.00 |
|  | TensorRT-LLM | 1 | 7.04 | 61.09 | 228.06 |
|  | SGLang | 1 | 7.02 | 56.29 | 197.76 |
| **H200 141GB HBM3** | PyTorch | 1 | 7.55 | 60.17 | 211.59 |
|  | vLLM-Omni | 1 | 7.13 | 58.27 | 209.29 |
|  | Diffusers | 1 | 9.00 | 69.00 | 237.00 |
|  | TensorRT-LLM | 1 | 6.93 | 60.77 | 233.06 |
|  | SGLang | 1 | 6.96 | 57.17 | 203.67 |
| **B200** | PyTorch | 1 | 4.63 | 32.60 | 113.89 |
|  | vLLM-Omni | 1 | 4.29 | 32.20 | 107.74 |
|  | Diffusers | 1 | 7.00 | 37.00 | 117.00 |
|  | TensorRT-LLM | 1 | 3.77 | 28.64 | 101.28 |
|  | SGLang | 1 | 4.01 | 32.93 | 123.37 |
| **B300** | PyTorch | — |  |  |  |
|  | vLLM-Omni | 1 | 5.59 | 33.15 | 104.92 |
|  | Diffusers | 1 | 6.46 | 39.66 | 122.33 |
|  | TensorRT-LLM | 1 | 3.65 | 26.83 | 91.85 |
|  | SGLang | 1 | 3.69 | 27.58 | 93.97 |

### Video-to-Audio-and-Video (v2av)

A text prompt and source video produce transformed video with synchronized sound.

| GPU | Engine | GPUs | 256p | 480p | 720p |
|---|---|:-:|---:|---:|---:|
| **RTX PRO 6000 Blackwell** | PyTorch | 1 | 13.73 | 180.72 | 786.11 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 1 | 26.30 | 135.47 | 429.61 |
|  | TensorRT-LLM | — |  |  |  |
| **H20** | PyTorch | 1 | 30.42 | 257.83 | 921.82 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 1 | 46.72 | 288.98 | 985.80 |
|  | TensorRT-LLM | — |  |  |  |
| **H100 NVL** | PyTorch | 1 | 9.91 | 81.96 | 297.51 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 1 | 22.95 | 110.93 | 354.02 |
|  | TensorRT-LLM | — |  |  |  |
| **H200 NVL** | PyTorch | 1 | 8.10 | 69.59 | 246.51 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 1 | 22.19 | 94.85 | 314.06 |
|  | TensorRT-LLM | — |  |  |  |
| **H100 80GB HBM3** | PyTorch | 1 | 7.49 | 59.86 | 207.55 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 1 | 21.73 | 88.42 | 268.93 |
|  | TensorRT-LLM | — |  |  |  |
| **H200 141GB HBM3** | PyTorch | 1 | 7.49 | 60.15 | 214.44 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 1 | 17.98 | 87.84 | 267.54 |
|  | TensorRT-LLM | — |  |  |  |
| **B200** | PyTorch | 1 | 4.51 | 32.62 | 113.72 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 1 | 15.64 | 50.57 | 138.15 |
|  | TensorRT-LLM | — |  |  |  |
| **B300** | PyTorch | — |  |  |  |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 1 | 15.93 | 51.87 | 131.52 |
|  | TensorRT-LLM | — |  |  |  |

### Image-to-Audio-and-Video (i2av)

A text prompt and source image produce video with synchronized sound.

| GPU | Engine | GPUs | 256p | 480p | 720p |
|---|---|:-:|---:|---:|---:|
| **RTX PRO 6000 Blackwell** | PyTorch | 1 | 14.08 | 181.64 | 787.81 |
|  | vLLM-Omni | 1 | 12.61 | 108.05 | 378.04 |
|  | Diffusers | 1 | 14.00 | 113.00 | 393.00 |
|  | TensorRT-LLM | 1 | 11.85 | 105.53 | 379.62 |
|  | SGLang | 1 | 12.07 | 105.72 | 376.87 |
| **H20** | PyTorch | 1 | 31.34 | 259.06 | 924.39 |
|  | vLLM-Omni | 1 | 30.36 | 265.37 | 948.09 |
|  | Diffusers | 1 | 32.00 | 263.00 | 934.80 |
|  | TensorRT-LLM | 1 | 29.95 | 259.43 | 924.85 |
|  | SGLang | 1 | 31.03 | 264.37 | 938.09 |
| **H100 NVL** | PyTorch | 1 | 10.45 | 84.37 | 298.61 |
|  | vLLM-Omni | 1 | 10.34 | 83.21 | 295.65 |
|  | Diffusers | 1 | 12.00 | 91.00 | 324.40 |
|  | TensorRT-LLM | — |  |  |  |
| **H200 NVL** | PyTorch | 1 | 8.29 | 71.20 | 248.35 |
|  | vLLM-Omni | 1 | 8.24 | 67.23 | 249.41 |
|  | Diffusers | 1 | 10.00 | 75.00 | 277.20 |
|  | TensorRT-LLM | — |  |  |  |
| **H100 80GB HBM3** | PyTorch | 1 | 7.73 | 59.13 | 207.93 |
|  | vLLM-Omni | 1 | 7.74 | 60.27 | 207.53 |
|  | Diffusers | 1 | 10.00 | 69.00 | 241.00 |
|  | TensorRT-LLM | 1 | 7.39 | 62.91 | 227.42 |
|  | SGLang | 1 | 7.03 | 56.41 | 197.71 |
| **H200 141GB HBM3** | PyTorch | 1 | 7.66 | 60.55 | 214.82 |
|  | vLLM-Omni | 1 | 7.55 | 59.73 | 212.73 |
|  | Diffusers | 1 | 9.00 | 68.00 | 241.00 |
|  | TensorRT-LLM | 1 | 7.37 | 62.87 | 231.61 |
|  | SGLang | 1 | 7.07 | 57.35 | 203.62 |
| **B200** | PyTorch | 1 | 4.62 | 32.82 | 113.85 |
|  | vLLM-Omni | 1 | 4.58 | 33.36 | 110.05 |
|  | Diffusers | 1 | 7.20 | 37.00 | 118.00 |
|  | TensorRT-LLM | 1 | 4.04 | 29.77 | 103.28 |
|  | SGLang | 1 | 4.11 | 33.51 | 123.66 |
| **B300** | PyTorch | — |  |  |  |
|  | vLLM-Omni | 1 | 6.01 | 34.39 | 108.59 |
|  | Diffusers | 1 | 15.35 | 50.19 | 131.92 |
|  | TensorRT-LLM | 1 | 4.03 | 28.10 | 94.57 |
|  | SGLang | 1 | 3.75 | 27.69 | 94.18 |

## Transfer generation

Transfer conditions a video-to-video generation on a structural control hint
extracted from the source video. Each control is reported separately because
the hint changes how much of the frame the model must synthesise.

| GPU | Transfer control | Engine | GPUs | 256p | 480p | 720p |
|---|---|---|:-:|---:|---:|---:|
| **RTX PRO 6000 Blackwell** | blur | PyTorch | 1 | 61.28 | 488.21 | 2066.63 |
|  |  | vLLM-Omni | 1 | 40.11 | 288.60 | 1061.44 |
|  |  | Diffusers | 1 | 30.95 | 217.40 | 735.43 |
|  |  | TensorRT-LLM | 1 | 25.62 | 249.50 | 935.86 |
|  |  | NIM | 1 |  |  |  |
|  |  | SGLang | 1 |  | 252.00 | 933.35 |
|  | depth | PyTorch | 1 | 57.55 | 506.88 | 2140.99 |
|  |  | vLLM-Omni | 1 | 31.56 | 291.76 | 1090.33 |
|  |  | Diffusers | 1 | 36.52 | 225.34 | 745.70 |
|  |  | TensorRT-LLM | 1 | 26.13 | 253.94 | 942.69 |
|  |  | NIM | 1 |  |  |  |
|  |  | SGLang | 1 |  | 253.91 | 943.03 |
|  | edge | PyTorch | 1 | 48.48 | 489.90 | 2102.40 |
|  |  | vLLM-Omni | 1 | 30.64 | 287.82 | 1080.82 |
|  |  | Diffusers | 1 | 31.25 | 217.90 | 735.35 |
|  |  | TensorRT-LLM | 1 | 25.67 | 250.54 | 938.79 |
|  |  | NIM | 1 |  |  |  |
|  |  | SGLang | 1 |  | 252.99 | 933.87 |
|  | multi_control | PyTorch | 1 | 60.58 | 879.86 | 4145.87 |
|  |  | vLLM-Omni | 1 | 73.36 | 1060.11 | 3988.93 |
|  |  | Diffusers | 1 |  |  |  |
|  |  | TensorRT-LLM | 1 | 35.07 | 423.61 | 1745.86 |
|  |  | NIM | 1 |  |  |  |
|  |  | SGLang | 1 |  | 420.96 | 1711.96 |
|  | seg | PyTorch | 1 | 51.45 | 492.85 | 2116.11 |
|  |  | vLLM-Omni | 1 | 31.50 | 289.01 | 1083.64 |
|  |  | Diffusers | 1 | 32.89 | 220.10 | 738.51 |
|  |  | TensorRT-LLM | 1 | 25.85 | 250.47 | 937.18 |
|  |  | NIM | 1 |  |  |  |
|  |  | SGLang | 1 |  | 251.17 | 935.51 |
|  | wsm | PyTorch | 1 | 46.48 | 376.67 | 1544.04 |
|  |  | vLLM-Omni | 1 |  |  |  |
|  |  | Diffusers | 1 | 29.84 | 176.06 | 565.32 |
|  |  | TensorRT-LLM | 1 | 12.95 | 114.91 | 407.62 |
|  |  | NIM | 1 |  |  |  |
|  |  | SGLang | 1 |  | 116.56 | 405.72 |
| **H20** | blur | PyTorch | 1 | 115.28 | 659.13 | 2315.28 |
|  |  | vLLM-Omni | 1 | 85.93 | 625.60 | 2280.10 |
|  |  | Diffusers | 1 | 73.61 | 491.22 | 1726.61 |
|  |  | TensorRT-LLM | 1 | 63.10 | 614.14 | 2308.16 |
|  |  | NIM | 1 |  |  |  |
|  |  | SGLang | 1 |  | 621.85 | 2317.09 |
|  | depth | PyTorch | 1 | 110.94 | 676.32 | 2390.43 |
|  |  | vLLM-Omni | 1 | 66.69 | 636.40 | 2367.37 |
|  |  | Diffusers | 1 | 86.84 | 510.15 | 1753.96 |
|  |  | TensorRT-LLM | 1 | 64.44 | 622.79 | 2328.65 |
|  |  | NIM | 1 |  |  |  |
|  |  | SGLang | 1 |  | 631.99 | 2361.39 |
|  | edge | PyTorch | 1 | 95.85 | 653.63 | 2379.49 |
|  |  | vLLM-Omni | 1 | 64.75 | 626.76 | 2321.00 |
|  |  | Diffusers | 1 | 74.68 | 492.78 | 1728.26 |
|  |  | TensorRT-LLM | 1 | 63.92 | 615.09 | 2310.55 |
|  |  | NIM | 1 |  |  |  |
|  |  | SGLang | 1 |  | 623.20 | 2316.19 |
|  | multi_control | PyTorch | 8 | 45.98 | 315.39 | 1177.07 |
|  |  | vLLM-Omni | 8 | 60.40 |  | 2622.35 |
|  |  | Diffusers | 8 |  |  |  |
|  |  | TensorRT-LLM | 8 | 17.62 | 148.26 | 558.95 |
|  |  | NIM | 8 |  |  |  |
|  |  | SGLang | 8 |  | 256.18 | 1028.45 |
|  | seg | PyTorch | 1 | 99.17 | 658.28 | 2382.09 |
|  |  | vLLM-Omni | 1 | 65.84 | 623.52 | 2341.06 |
|  |  | Diffusers | 1 | 77.50 | 496.77 | 1734.77 |
|  |  | TensorRT-LLM | 1 | 63.56 | 616.74 | 2314.38 |
|  |  | NIM | 1 |  |  |  |
|  |  | SGLang | 1 |  | 624.39 | 2346.04 |
|  | wsm | PyTorch | 1 | 91.06 | 528.65 | 1807.24 |
|  |  | vLLM-Omni | 1 |  |  |  |
|  |  | Diffusers | 1 | 70.42 | 398.67 | 1319.24 |
|  |  | TensorRT-LLM | 1 | 32.09 | 284.19 | 1001.59 |
|  |  | NIM | 1 |  |  |  |
|  |  | SGLang | 1 |  | 286.47 | 1016.23 |
| **H100 NVL** | blur | PyTorch | 1 | 37.53 | 202.37 | 710.92 |
|  |  | vLLM-Omni | 1 |  |  |  |
|  |  | Diffusers | 1 | 26.97 | 224.61 | 775.61 |
|  |  | TensorRT-LLM | 1 |  |  |  |
|  |  | NIM | 1 |  |  |  |
|  | depth | PyTorch | 1 | 36.00 | 210.05 | 736.95 |
|  |  | vLLM-Omni | 1 |  |  |  |
|  |  | Diffusers | 1 | 31.63 | 233.03 | 610.83 |
|  |  | TensorRT-LLM | 1 |  |  |  |
|  |  | NIM | 1 |  |  |  |
|  | edge | PyTorch | 1 | 40.55 | 268.07 | 952.43 |
|  |  | vLLM-Omni | 1 |  |  |  |
|  |  | Diffusers | 1 | 34.73 | 173.04 | 602.04 |
|  |  | TensorRT-LLM | 1 |  |  |  |
|  |  | NIM | 1 |  |  |  |
|  | multi_control | PyTorch | 1 | 36.65 | 437.89 | 1332.92 |
|  |  | vLLM-Omni | 1 |  |  |  |
|  |  | Diffusers | 1 |  |  |  |
|  |  | TensorRT-LLM | 1 |  |  |  |
|  |  | NIM | 1 |  |  |  |
|  | seg | PyTorch | 1 | 32.49 | 269.41 | 725.45 |
|  |  | vLLM-Omni | 1 |  |  |  |
|  |  | Diffusers | 1 | 28.55 | 174.78 | 777.58 |
|  |  | TensorRT-LLM | 1 |  |  |  |
|  |  | NIM | 1 |  |  |  |
|  | wsm | PyTorch | 1 | 38.82 | 164.02 | 548.03 |
|  |  | vLLM-Omni | 1 |  |  |  |
|  |  | Diffusers | 1 | 26.17 | 139.82 | 594.21 |
|  |  | TensorRT-LLM | 1 |  |  |  |
|  |  | NIM | 1 |  |  |  |
| **H100 80GB HBM3** | blur | PyTorch | 1 | 28.12 | 147.27 | 509.96 |
|  |  | vLLM-Omni | 1 | 20.40 | 133.27 | 476.11 |
|  |  | Diffusers | 1 | 21.06 | 131.86 | 454.46 |
|  |  | TensorRT-LLM | 1 | 15.20 | 149.57 | 575.59 |
|  |  | NIM | 1 |  |  |  |
|  |  | SGLang | 1 |  | 132.13 | 484.80 |
|  | depth | PyTorch | 1 | 26.98 | 152.75 | 534.58 |
|  |  | vLLM-Omni | 1 | 15.75 | 135.11 | 488.18 |
|  |  | Diffusers | 1 | 24.61 | 135.88 | 457.43 |
|  |  | TensorRT-LLM | 1 | 15.45 | 149.96 | 580.66 |
|  |  | NIM | 1 |  |  |  |
|  |  | SGLang | 1 |  | 134.31 | 489.15 |
|  | edge | PyTorch | 1 | 23.36 | 148.65 | 527.89 |
|  |  | vLLM-Omni | 1 | 15.33 | 133.00 | 483.75 |
|  |  | Diffusers | 1 | 21.28 | 132.06 | 447.24 |
|  |  | TensorRT-LLM | 1 | 15.28 | 148.99 | 572.91 |
|  |  | NIM | 1 |  |  |  |
|  |  | SGLang | 1 |  | 132.29 | 484.55 |
|  | multi_control | PyTorch | 1 | 27.44 | 243.48 | 976.27 |
|  |  | vLLM-Omni | 1 | 36.94 | 477.12 | 2122.18 |
|  |  | Diffusers | 1 |  |  |  |
|  |  | TensorRT-LLM | 1 | 21.50 | 254.29 | 1072.98 |
|  |  | NIM | 1 |  |  |  |
|  |  | SGLang | 1 |  | 222.00 | 888.45 |
|  | seg | PyTorch | 1 | 24.55 | 148.95 | 531.67 |
|  |  | vLLM-Omni | 1 | 16.04 | 133.82 | 484.97 |
|  |  | Diffusers | 1 | 22.54 | 133.24 | 456.55 |
|  |  | TensorRT-LLM | 1 | 15.42 | 145.46 | 576.57 |
|  |  | NIM | 1 |  |  |  |
|  |  | SGLang | 1 |  | 133.24 | 481.78 |
|  | wsm | PyTorch | 1 | 22.50 | 119.44 | 395.90 |
|  |  | vLLM-Omni | 1 |  |  |  |
|  |  | Diffusers | 1 | 20.25 | 106.79 | 342.79 |
|  |  | TensorRT-LLM | 1 | 7.96 | 68.97 | 248.27 |
|  |  | NIM | 1 |  |  |  |
|  |  | SGLang | 1 |  | 62.38 | 211.85 |
| **H200 NVL** | blur | PyTorch | 1 | 31.98 | 176.71 | 624.99 |
|  |  | vLLM-Omni | 1 |  |  |  |
|  |  | Diffusers | 1 | 22.82 | 152.13 | 535.83 |
|  |  | TensorRT-LLM | 1 |  |  |  |
|  |  | NIM | 1 |  |  |  |
|  | depth | PyTorch | 1 | 30.17 | 183.55 | 648.42 |
|  |  | vLLM-Omni | 1 |  |  |  |
|  |  | Diffusers | 1 | 26.80 | 156.98 | 541.71 |
|  |  | TensorRT-LLM | 1 |  |  |  |
|  |  | NIM | 1 |  |  |  |
|  | edge | PyTorch | 1 | 25.80 | 177.10 | 639.67 |
|  |  | vLLM-Omni | 1 |  |  |  |
|  |  | Diffusers | 1 | 23.06 | 152.54 | 533.87 |
|  |  | TensorRT-LLM | 1 |  |  |  |
|  |  | NIM | 1 |  |  |  |
|  | multi_control | PyTorch | 1 | 30.56 | 290.20 | 1175.02 |
|  |  | vLLM-Omni | 1 |  |  |  |
|  |  | Diffusers | 1 |  |  |  |
|  |  | TensorRT-LLM | 1 |  |  |  |
|  |  | NIM | 1 |  |  |  |
|  | seg | PyTorch | 1 | 27.52 | 178.83 | 640.28 |
|  |  | vLLM-Omni | 1 |  |  |  |
|  |  | Diffusers | 1 | 24.12 | 153.68 | 537.15 |
|  |  | TensorRT-LLM | 1 |  |  |  |
|  |  | NIM | 1 |  |  |  |
|  | wsm | PyTorch | 1 | 24.52 | 142.66 | 486.21 |
|  |  | vLLM-Omni | 1 |  |  |  |
|  |  | Diffusers | 1 | 21.90 | 123.08 | 407.57 |
|  |  | TensorRT-LLM | 1 |  |  |  |
|  |  | NIM | 1 |  |  |  |
| **H200 141GB HBM3** | blur | PyTorch | 1 | 28.31 | 151.51 | 527.08 |
|  |  | vLLM-Omni | 1 |  |  |  |
|  |  | Diffusers | 1 | 20.63 | 128.41 | 450.79 |
|  |  | TensorRT-LLM | 1 | 15.11 | 149.55 | 580.80 |
|  |  | NIM | 1 |  |  |  |
|  |  | SGLang | 1 |  | 135.54 | 495.17 |
|  | depth | PyTorch | 1 | 26.89 | 157.02 | 546.39 |
|  |  | vLLM-Omni | 1 |  |  |  |
|  |  | Diffusers | 1 | 24.01 | 134.51 | 457.91 |
|  |  | TensorRT-LLM | 1 | 15.42 | 152.64 | 587.03 |
|  |  | NIM | 1 |  |  |  |
|  |  | SGLang | 1 |  | 137.82 | 498.16 |
|  | edge | PyTorch | 1 | 23.16 | 151.57 | 538.67 |
|  |  | vLLM-Omni | 1 |  |  |  |
|  |  | Diffusers | 1 | 20.87 | 130.99 | 443.92 |
|  |  | TensorRT-LLM | 1 | 14.95 | 148.90 | 586.42 |
|  |  | NIM | 1 |  |  |  |
|  |  | SGLang | 1 |  | 135.81 | 494.98 |
|  | multi_control | PyTorch | 1 | 27.28 | 247.75 | 992.09 |
|  |  | vLLM-Omni | 1 |  |  |  |
|  |  | Diffusers | 1 |  |  |  |
|  |  | TensorRT-LLM | 1 | 21.65 | 259.30 | 1095.79 |
|  |  | NIM | 1 |  |  |  |
|  |  | SGLang | 1 |  | 225.14 | 915.75 |
|  | seg | PyTorch | 1 | 24.66 | 153.20 | 540.77 |
|  |  | vLLM-Omni | 1 |  |  |  |
|  |  | Diffusers | 1 | 21.84 | 130.12 | 453.51 |
|  |  | TensorRT-LLM | 1 | 15.31 | 148.26 | 583.48 |
|  |  | NIM | 1 |  |  |  |
|  |  | SGLang | 1 |  | 136.88 | 491.92 |
|  | wsm | PyTorch | 1 | 22.15 | 122.36 | 408.71 |
|  |  | vLLM-Omni | 1 |  |  |  |
|  |  | Diffusers | 1 | 19.65 | 106.00 | 344.09 |
|  |  | TensorRT-LLM | 1 | 7.90 | 68.71 | 252.43 |
|  |  | NIM | 1 |  |  |  |
|  |  | SGLang | 1 |  | 63.16 | 215.45 |
| **B200** | blur | PyTorch | 1 | 16.44 | 82.91 | 281.31 |
|  |  | vLLM-Omni | 1 | 13.26 | 72.95 | 257.39 |
|  |  | Diffusers | 1 | 12.69 | 70.78 | 226.51 |
|  |  | TensorRT-LLM | 1 | 7.73 | 68.21 | 252.69 |
|  |  | NIM | 1 |  |  |  |
|  |  | SGLang | 1 |  | 79.73 | 305.56 |
|  | depth | PyTorch | 1 | 15.85 | 85.37 | 293.55 |
|  |  | vLLM-Omni | 1 | 10.55 | 73.56 | 264.11 |
|  |  | Diffusers | 1 | 14.79 | 73.23 | 229.54 |
|  |  | TensorRT-LLM | 1 | 7.92 | 68.77 | 248.98 |
|  |  | NIM | 1 |  |  |  |
|  |  | SGLang | 1 |  | 80.01 | 304.42 |
|  | edge | PyTorch | 1 | 13.52 | 81.88 | 288.07 |
|  |  | vLLM-Omni | 1 | 10.35 | 74.00 | 261.85 |
|  |  | Diffusers | 1 | 12.76 | 71.04 | 223.24 |
|  |  | TensorRT-LLM | 1 | 7.78 | 68.29 | 248.51 |
|  |  | NIM | 1 |  |  |  |
|  |  | SGLang | 1 |  | 79.87 | 305.58 |
|  | multi_control | PyTorch | 1 | 15.95 | 133.97 | 536.09 |
|  |  | vLLM-Omni | 1 | 23.27 | 257.99 | 1111.03 |
|  |  | Diffusers | 1 |  |  |  |
|  |  | TensorRT-LLM | 1 | 10.72 | 112.58 | 455.44 |
|  |  | NIM | 1 |  |  |  |
|  |  | SGLang | 1 |  | 137.61 | 577.54 |
|  | seg | PyTorch | 1 | 15.07 | 83.15 | 289.21 |
|  |  | vLLM-Omni | 1 | 11.10 | 73.35 | 262.72 |
|  |  | Diffusers | 1 | 13.54 | 71.21 | 223.41 |
|  |  | TensorRT-LLM | 1 | 7.99 | 68.45 | 248.75 |
|  |  | NIM | 1 |  |  |  |
|  |  | SGLang | 1 |  | 79.69 | 302.30 |
|  | wsm | PyTorch | 1 | 13.43 | 66.99 | 220.43 |
|  |  | vLLM-Omni | 1 |  |  |  |
|  |  | Diffusers | 1 | 12.16 | 59.29 | 172.60 |
|  |  | TensorRT-LLM | 1 | 4.25 | 32.24 | 108.75 |
|  |  | NIM | 1 |  |  |  |
|  |  | SGLang | 1 |  | 37.33 | 132.65 |
| **B300** | blur | PyTorch | 1 |  |  |  |
|  |  | vLLM-Omni | 1 | 13.85 | 73.35 | 240.39 |
|  |  | Diffusers | 1 | 12.39 | 66.13 | 207.59 |
|  |  | TensorRT-LLM | 1 | 7.67 | 62.95 | 228.34 |
|  |  | NIM | 1 |  |  |  |
|  |  | SGLang | 1 |  | 65.08 | 226.48 |
|  | depth | PyTorch | 1 |  |  |  |
|  |  | vLLM-Omni | 1 | 11.11 | 71.59 | 247.16 |
|  |  | Diffusers | 1 | 14.26 | 70.00 | 211.12 |
|  |  | TensorRT-LLM | 1 | 7.67 | 63.68 | 226.26 |
|  |  | NIM | 1 |  |  |  |
|  |  | SGLang | 1 |  | 65.39 | 228.50 |
|  | edge | PyTorch | 1 |  |  |  |
|  |  | vLLM-Omni | 1 | 13.05 | 72.93 | 245.16 |
|  |  | Diffusers | 1 | 12.48 | 67.68 | 207.95 |
|  |  | TensorRT-LLM | 1 | 7.58 | 63.10 | 228.36 |
|  |  | NIM | 1 |  |  |  |
|  |  | SGLang | 1 |  | 64.17 | 230.10 |
|  | multi_control | PyTorch | 1 |  |  |  |
|  |  | vLLM-Omni | 1 | 26.79 | 243.97 | 1026.91 |
|  |  | Diffusers | 1 |  |  |  |
|  |  | TensorRT-LLM | 1 | 10.59 | 105.33 | 414.58 |
|  |  | NIM | 1 |  |  |  |
|  |  | SGLang | 1 |  | 108.87 | 410.34 |
|  | seg | PyTorch | 1 |  |  |  |
|  |  | vLLM-Omni | 1 | 14.38 | 71.85 | 246.04 |
|  |  | Diffusers | 1 | 13.43 | 68.85 | 210.33 |
|  |  | TensorRT-LLM | 1 | 7.64 | 64.13 | 228.37 |
|  |  | NIM | 1 |  |  |  |
|  |  | SGLang | 1 |  | 66.27 | 227.29 |
|  | wsm | PyTorch | 1 |  |  |  |
|  |  | vLLM-Omni | 1 |  |  |  |
|  |  | Diffusers | 1 | 11.95 | 55.41 | 162.28 |
|  |  | TensorRT-LLM | 1 | 4.23 | 30.44 | 101.17 |
|  |  | NIM | 1 |  |  |  |
|  |  | SGLang | 1 |  | 31.42 | 101.11 |
| **DGX Spark** | depth | NIM | 1 |  | 118.52 |  |
| **DGX Spark** | edge | NIM | 1 |  | 114.30 |  |
| **DGX Spark** | seg | NIM | 1 |  | 116.26 |  |
| **DGX Spark** | wsm | NIM | 1 |  | 71.25 |  |
| **RTX PRO 5000 Blackwell 72GB** | depth | NIM | 1 |  | 46.93 |  |
| **RTX PRO 5000 Blackwell 72GB** | edge | NIM | 1 |  | 45.56 |  |
| **RTX PRO 5000 Blackwell 72GB** | seg | NIM | 1 |  | 46.29 |  |
| **RTX PRO 5000 Blackwell 72GB** | wsm | NIM | 1 |  | 28.19 |  |

<sub>Transfer notes:
1. Values are generation latency in seconds; lower is better. Diffusers, vLLM-Omni, and TensorRT-LLM numbers come from separate campaigns; some Diffusers single-GPU modality values report the best run.
2. The **GPUs** column gives the number of GPUs behind every value in that row; check it before comparing engines.
3. PyTorch transfer results are included where measured. Published NIM transfer results use FP8 NIM 2.0.0 on DGX Spark and RTX PRO 5000 Blackwell 72GB; see the measurement settings for their workload definitions.
4. Control hints follow the vLLM-Omni `extra_params` names: `blur`, `depth`, `edge`, `multi_control`, `seg`, and `wsm`. `multi_control` combines several hints in one request.</sub>

## Action generation

Forward dynamics is reported separately for AV, camera, and robot inputs. Inverse dynamics and policy are reported for AV and robot because those are the only domain rows in the PyTorch action measurements; no camera row is inferred. For each domain, PyTorch is populated from that report. vLLM-Omni currently covers forward dynamics AV and inverse dynamics AV from the vLLM-Omni action measurements. Camera, robot-FD, robot-ID, policy-AV, and policy-robot vLLM-Omni rows remain reserved. **Policy — DROID** is a separate checkpoint (`nvidia/Cosmos3-Nano-Policy-DROID`) and is not mixed with the policy-robot table.

### Forward Dynamics — Autonomous Vehicle (AV)

| GPU | Engine | GPUs | 256p | 480p | 720p |
|---|---|:-:|---:|---:|---:|
| **RTX PRO 6000 Blackwell** | PyTorch | 1 | 16.87 | 16.86 | 16.85 |
|  | vLLM-Omni | 1 | 2.08 | 12.94 | 36.19 |
|  | Diffusers | 1 | 2.00 | 12.00 | 12.00 |
|  | TensorRT-LLM | — |  |  |  |
|  | SGLang | 1 | 1.69 | 11.21 | 32.68 |
| **H20** | PyTorch | 1 | 31.47 | 31.16 | 31.45 |
|  | vLLM-Omni | 1 | 4.82 | 31.36 | 87.55 |
|  | Diffusers | 1 | 5.00 | 28.00 | 28.20 |
|  | TensorRT-LLM | — |  |  |  |
|  | SGLang | 1 | 4.17 | 28.31 | 81.11 |
| **H100 NVL** | PyTorch | 1 | 10.32 | 10.66 | 10.64 |
|  | vLLM-Omni | 1 | 1.71 | 10.58 | 28.26 |
|  | Diffusers | 1 | 2.00 | 10.00 | 10.00 |
|  | TensorRT-LLM | — |  |  |  |
| **H200 NVL** | PyTorch | 1 | 8.30 | 8.31 | 8.33 |
|  | vLLM-Omni | 1 | 1.43 | 8.21 | 22.78 |
|  | Diffusers | 1 | 2.00 | 8.00 | 8.00 |
|  | TensorRT-LLM | — |  |  |  |
| **H100 80GB HBM3** | PyTorch | 1 | 7.80 | 7.87 | 7.79 |
|  | vLLM-Omni | 1 | 1.33 | 7.74 | 20.60 |
|  | Diffusers | 1 | 2.00 | 8.00 | 8.00 |
|  | TensorRT-LLM | — |  |  |  |
|  | SGLang | 1 | 1.10 | 6.55 | 18.05 |
| **H200 141GB HBM3** | PyTorch | 1 | 7.77 | 7.76 | 7.81 |
|  | vLLM-Omni | 1 | 1.33 | 7.72 | 20.65 |
|  | Diffusers | 1 | 2.00 | 8.00 | 8.00 |
|  | TensorRT-LLM | — |  |  |  |
|  | SGLang | 1 | 1.13 | 6.53 | 18.28 |
| **B200** | PyTorch | 1 | 4.75 | 4.75 | 4.73 |
|  | vLLM-Omni | 1 | 0.94 | 4.93 | 12.00 |
|  | Diffusers | 1 | 2.00 | 5.00 | 5.00 |
|  | TensorRT-LLM | — |  |  |  |
|  | SGLang | 1 | 0.77 | 3.65 | 10.46 |
| **B300** | PyTorch | — |  |  |  |
|  | vLLM-Omni | 1 | 1.09 | 4.86 | 11.70 |
|  | Diffusers | 1 | 20.87 | 24.49 | 26.44 |
|  | TensorRT-LLM | — |  |  |  |
|  | SGLang | 1 | 0.72 | 3.34 | 8.98 |
| **DGX Spark** | NIM | 1 | 5.98 | 41.33 | 119.42 |
| **RTX PRO 5000 Blackwell 72GB** | NIM | 1 | 2.24 | 15.28 | 46.28 |

### Forward Dynamics — Camera

| GPU | Engine | GPUs | 256p | 480p | 720p |
|---|---|:-:|---:|---:|---:|
| **RTX PRO 6000 Blackwell** | PyTorch | 1 | 16.82 | 16.82 | 16.81 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | — |  |  |  |
|  | TensorRT-LLM | — |  |  |  |
|  | SGLang | 1 | 1.70 | 11.23 | 32.72 |
| **H20** | PyTorch | 1 | 31.40 | 31.10 | 31.40 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | — |  |  |  |
|  | TensorRT-LLM | — |  |  |  |
|  | SGLang | 1 | 4.19 | 28.63 | 81.38 |
| **H100 NVL** | PyTorch | 1 | 10.61 | 10.32 | 10.57 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | — |  |  |  |
|  | TensorRT-LLM | — |  |  |  |
| **H200 NVL** | PyTorch | 1 | 8.24 | 8.26 | 8.26 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | — |  |  |  |
|  | TensorRT-LLM | — |  |  |  |
| **H100 80GB HBM3** | PyTorch | 1 | 7.78 | 7.83 | 7.75 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | — |  |  |  |
|  | TensorRT-LLM | — |  |  |  |
|  | SGLang | 1 | 1.15 | 6.57 | 18.04 |
| **H200 141GB HBM3** | PyTorch | 1 | 7.73 | 7.75 | 7.75 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | — |  |  |  |
|  | TensorRT-LLM | — |  |  |  |
|  | SGLang | 1 | 1.13 | 6.52 | 18.29 |
| **B200** | PyTorch | 1 | 4.70 | 4.74 | 4.71 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | — |  |  |  |
|  | TensorRT-LLM | — |  |  |  |
|  | SGLang | 1 | 0.83 | 3.69 | 10.34 |
| **B300** | PyTorch | — |  |  |  |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | — |  |  |  |
|  | TensorRT-LLM | — |  |  |  |
|  | SGLang | 1 | 0.83 | 3.36 | 9.01 |

### Forward Dynamics — Robot

| GPU | Engine | GPUs | 256p | 480p | 720p |
|---|---|:-:|---:|---:|---:|
| **RTX PRO 6000 Blackwell** | PyTorch | 1 | 3.74 | 3.73 | 3.72 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 1 | 2.00 | 3.00 | 3.00 |
|  | TensorRT-LLM | — |  |  |  |
|  | SGLang | 1 | 0.70 | 3.00 | 7.33 |
| **H20** | PyTorch | 1 | 7.00 | 7.01 | 7.01 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 1 | 3.00 | 7.00 | 7.00 |
|  | TensorRT-LLM | — |  |  |  |
|  | SGLang | 1 | 1.32 | 7.46 | 18.70 |
| **H100 NVL** | PyTorch | 1 | 2.49 | 2.50 | 2.55 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 1 | 2.00 | 3.00 | 3.00 |
|  | TensorRT-LLM | — |  |  |  |
| **H200 NVL** | PyTorch | 1 | 2.08 | 2.09 | 2.09 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 1 | 2.00 | 3.00 | 3.00 |
|  | TensorRT-LLM | — |  |  |  |
| **H100 80GB HBM3** | PyTorch | 1 | 2.00 | 2.01 | 1.98 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 1 | 2.00 | 3.00 | 3.00 |
|  | TensorRT-LLM | — |  |  |  |
|  | SGLang | 1 | 0.91 | 1.74 | 4.23 |
| **H200 141GB HBM3** | PyTorch | 1 | 1.93 | 1.93 | 1.93 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 1 | 2.00 | 2.00 | 2.00 |
|  | TensorRT-LLM | — |  |  |  |
|  | SGLang | 1 | 0.93 | 1.69 | 4.25 |
| **B200** | PyTorch | 1 | 1.28 | 1.30 | 1.31 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 1 | 2.00 | 2.00 | 2.00 |
|  | TensorRT-LLM | — |  |  |  |
|  | SGLang | 1 | 0.65 | 0.97 | 2.42 |
| **B300** | PyTorch | — |  |  |  |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 1 | 22.65 | 20.75 | 18.10 |
|  | TensorRT-LLM | — |  |  |  |
|  | SGLang | 1 | 0.59 | 1.14 | 2.22 |

### Inverse Dynamics — Autonomous Vehicle (AV)

| GPU | Engine | GPUs | 256p | 480p | 720p |
|---|---|:-:|---:|---:|---:|
| **RTX PRO 6000 Blackwell** | PyTorch | 1 | 16.74 | 16.75 | 16.75 |
|  | vLLM-Omni | 1 | 2.10 | 12.96 | 36.10 |
|  | Diffusers | 1 | 2.00 | 12.00 | 12.00 |
|  | TensorRT-LLM | — |  |  |  |
|  | SGLang | 1 | 10.58 | 10.57 | 10.57 |
| **H20** | PyTorch | 1 | 31.22 | 30.99 | 31.22 |
|  | vLLM-Omni | 1 | 4.84 | 31.36 | 87.50 |
|  | Diffusers | 1 | 5.00 | 28.00 | 28.00 |
|  | TensorRT-LLM | — |  |  |  |
|  | SGLang | 1 | 26.71 | 26.77 | 26.78 |
| **H100 NVL** | PyTorch | 1 | 10.47 | 10.45 | 10.44 |
|  | vLLM-Omni | 1 | 1.74 | 10.54 | 28.16 |
|  | Diffusers | 1 | 2.00 | 10.00 | 10.00 |
|  | TensorRT-LLM | — |  |  |  |
| **H200 NVL** | PyTorch | 1 | 8.12 | 8.13 | 8.10 |
|  | vLLM-Omni | 1 | 1.46 | 8.20 | 22.34 |
|  | Diffusers | 1 | 2.00 | 8.00 | 8.00 |
|  | TensorRT-LLM | — |  |  |  |
| **H100 80GB HBM3** | PyTorch | 1 | 7.62 | 7.63 | 7.68 |
|  | vLLM-Omni | 1 | 1.39 | 7.72 | 20.55 |
|  | Diffusers | 1 | 2.00 | 8.00 | 8.00 |
|  | TensorRT-LLM | — |  |  |  |
|  | SGLang | 1 | 6.33 | 6.21 | 6.32 |
| **H200 141GB HBM3** | PyTorch | 1 | 7.51 | 7.64 | 7.51 |
|  | vLLM-Omni | 1 | 1.36 | 7.69 | 20.65 |
|  | Diffusers | 1 | 2.00 | 8.00 | 8.00 |
|  | TensorRT-LLM | — |  |  |  |
|  | SGLang | 1 | 6.26 | 6.28 | 6.30 |
| **B200** | PyTorch | 1 | 4.60 | 4.63 | 4.58 |
|  | vLLM-Omni | 1 | 0.92 | 5.10 | 12.29 |
|  | Diffusers | 1 | 2.00 | 5.00 | 5.00 |
|  | TensorRT-LLM | — |  |  |  |
|  | SGLang | 1 | 3.70 | 3.69 | 3.68 |
| **B300** | PyTorch | — |  |  |  |
|  | vLLM-Omni | 1 | 1.36 | 5.75 | 11.89 |
|  | Diffusers | 1 | 8.76 | 17.79 | 17.99 |
|  | TensorRT-LLM | — |  |  |  |
|  | SGLang | 1 | 3.38 | 3.50 | 3.50 |
| **DGX Spark** | NIM | 1 | 6.94 | 45.62 | 128.84 |
| **RTX PRO 5000 Blackwell 72GB** | NIM | 1 | 3.11 | 17.24 | 49.90 |

### Inverse Dynamics — Robot

| GPU | Engine | GPUs | 256p | 480p | 720p |
|---|---|:-:|---:|---:|---:|
| **RTX PRO 6000 Blackwell** | PyTorch | 1 | 3.70 | 3.70 | 3.70 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 1 | 2.00 | 3.00 | 3.00 |
|  | TensorRT-LLM | — |  |  |  |
|  | SGLang | 1 | 3.18 | 3.17 | 3.18 |
| **H20** | PyTorch | 1 | 6.94 | 6.94 | 6.89 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 1 | 3.00 | 7.00 | 7.00 |
|  | TensorRT-LLM | — |  |  |  |
|  | SGLang | 1 | 7.54 | 7.55 | 7.54 |
| **H100 NVL** | PyTorch | 1 | 2.53 | 2.47 | 2.54 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 1 | 2.00 | 3.00 | 3.00 |
|  | TensorRT-LLM | — |  |  |  |
| **H200 NVL** | PyTorch | 1 | 2.05 | 2.05 | 2.05 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 1 | 2.00 | 3.00 | 3.00 |
|  | TensorRT-LLM | — |  |  |  |
| **H100 80GB HBM3** | PyTorch | 1 | 1.96 | 1.96 | 1.95 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 1 | 2.00 | 3.00 | 3.00 |
|  | TensorRT-LLM | — |  |  |  |
|  | SGLang | 1 | 1.95 | 1.96 | 2.00 |
| **H200 141GB HBM3** | PyTorch | 1 | 1.90 | 1.90 | 1.91 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 1 | 2.00 | 2.20 | 2.00 |
|  | TensorRT-LLM | — |  |  |  |
|  | SGLang | 1 | 1.96 | 1.93 | 1.93 |
| **B200** | PyTorch | 1 | 1.28 | 1.27 | 1.28 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 1 | 2.00 | 2.00 | 2.00 |
|  | TensorRT-LLM | — |  |  |  |
|  | SGLang | 1 | 1.27 | 1.22 | 1.24 |
| **B300** | PyTorch | — |  |  |  |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 1 | 9.28 | 12.40 | 10.86 |
|  | TensorRT-LLM | — |  |  |  |
|  | SGLang | 1 | 1.33 | 1.25 | 1.15 |

### Policy — Autonomous Vehicle (AV)

| GPU | Engine | GPUs | 256p | 480p | 720p |
|---|---|:-:|---:|---:|---:|
| **RTX PRO 6000 Blackwell** | PyTorch | 1 | 16.88 | 16.89 | 16.91 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 1 | 2.00 | 12.00 | 12.00 |
|  | TensorRT-LLM | — |  |  |  |
| **H20** | PyTorch | 1 | 31.51 | 31.51 | 31.50 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 1 | 5.00 | 28.60 | 28.40 |
|  | TensorRT-LLM | — |  |  |  |
| **H100 NVL** | PyTorch | 1 | 10.66 | 10.65 | 10.35 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 1 | 2.00 | 10.00 | 10.00 |
|  | TensorRT-LLM | — |  |  |  |
| **H200 NVL** | PyTorch | 1 | 8.31 | 8.32 | 8.31 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 1 | 2.00 | 8.00 | 8.00 |
|  | TensorRT-LLM | — |  |  |  |
| **H100 80GB HBM3** | PyTorch | 1 | 7.82 | 7.82 | 7.80 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 1 | 2.00 | 8.00 | 8.00 |
|  | TensorRT-LLM | — |  |  |  |
| **H200 141GB HBM3** | PyTorch | 1 | 7.76 | 7.69 | 7.66 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 1 | 2.00 | 8.00 | 8.00 |
|  | TensorRT-LLM | — |  |  |  |
| **B200** | PyTorch | 1 | 4.75 | 4.77 | 4.74 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 1 | 2.00 | 5.00 | 5.00 |
|  | TensorRT-LLM | — |  |  |  |
| **B300** | PyTorch | — |  |  |  |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 1 | 13.11 | 18.21 | 16.49 |
|  | TensorRT-LLM | — |  |  |  |
| **DGX Spark** | NIM | 1 | 6.06 | 41.52 | 119.82 |
| **RTX PRO 5000 Blackwell 72GB** | NIM | 1 | 2.28 | 15.34 | 46.40 |

### Policy — Robot

| GPU | Engine | GPUs | 256p | 480p | 720p |
|---|---|:-:|---:|---:|---:|
| **RTX PRO 6000 Blackwell** | PyTorch | 1 | 3.74 | 3.74 | 3.74 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 1 | 2.00 | 3.00 | 3.00 |
|  | TensorRT-LLM | — |  |  |  |
| **H20** | PyTorch | 1 | 6.95 | 7.00 | 7.01 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 1 | 3.00 | 7.00 | 7.00 |
|  | TensorRT-LLM | — |  |  |  |
| **H100 NVL** | PyTorch | 1 | 2.50 | 2.51 | 2.55 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 1 | 2.00 | 3.00 | 3.00 |
|  | TensorRT-LLM | — |  |  |  |
| **H200 NVL** | PyTorch | 1 | 2.11 | 2.09 | 2.09 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 1 | 2.00 | 3.00 | 3.00 |
|  | TensorRT-LLM | — |  |  |  |
| **H100 80GB HBM3** | PyTorch | 1 | 1.99 | 1.99 | 1.99 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 1 | 2.00 | 3.00 | 3.00 |
|  | TensorRT-LLM | — |  |  |  |
| **H200 141GB HBM3** | PyTorch | 1 | 1.93 | 1.93 | 1.94 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 1 | 2.00 | 2.00 | 2.20 |
|  | TensorRT-LLM | — |  |  |  |
| **B200** | PyTorch | 1 | 1.30 | 1.30 | 1.30 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 1 | 2.00 | 2.00 | 2.00 |
|  | TensorRT-LLM | — |  |  |  |
| **B300** | PyTorch | — |  |  |  |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 1 | 10.62 | 12.78 | 11.00 |
|  | TensorRT-LLM | — |  |  |  |

### Policy — DROID

These rows report **Cosmos3-Nano-Policy-DROID**, not the general Cosmos3-Nano checkpoint used in the other action tables. vLLM-Omni only supports this modality at **480p on one GPU** (multi-GPU is disabled, and resolution is fixed for the multiview wrist/exterior inputs). PyTorch and Diffusers cells are reserved.

| GPU | Engine | GPUs | 256p | 480p | 720p |
|---|---|:-:|---:|---:|---:|
| **RTX PRO 6000 Blackwell** | PyTorch | — |  |  |  |
|  | vLLM-Omni | 1 |  | 3.82 |  |
|  | Diffusers | — |  |  |  |
|  | TensorRT-LLM | — |  |  |  |
| **H20** | PyTorch | — |  |  |  |
|  | vLLM-Omni | 1 |  | 8.51 |  |
|  | Diffusers | — |  |  |  |
|  | TensorRT-LLM | — |  |  |  |
| **H100 NVL** | PyTorch | — |  |  |  |
|  | vLLM-Omni | 1 |  | 2.94 |  |
|  | Diffusers | — |  |  |  |
|  | TensorRT-LLM | — |  |  |  |
| **H200 NVL** | PyTorch | — |  |  |  |
|  | vLLM-Omni | 1 |  | 2.34 |  |
|  | Diffusers | — |  |  |  |
|  | TensorRT-LLM | — |  |  |  |
| **H100 80GB HBM3** | PyTorch | — |  |  |  |
|  | vLLM-Omni | 1 |  | 2.23 |  |
|  | Diffusers | — |  |  |  |
|  | TensorRT-LLM | — |  |  |  |
| **H200 141GB HBM3** | PyTorch | — |  |  |  |
|  | vLLM-Omni | 1 |  | 2.21 |  |
|  | Diffusers | — |  |  |  |
|  | TensorRT-LLM | — |  |  |  |
| **B200** | PyTorch | — |  |  |  |
|  | vLLM-Omni | 1 |  | 1.47 |  |
|  | Diffusers | — |  |  |  |
|  | TensorRT-LLM | — |  |  |  |
| **B300** | PyTorch | — |  |  |  |
|  | vLLM-Omni | 1 |  | 2.15 |  |
|  | Diffusers | — |  |  |  |
|  | TensorRT-LLM | — |  |  |  |

<sub>Additional-modality notes:
1. PyTorch values are average generation (sampling) latency in seconds; lower is better.
2. PyTorch values use `CUDA_GRAPH=No` and the `latency` automatic-sharding preset.
3. vLLM-Omni audiovisual values for `t2av`/`t2vs` and `i2av`/`i2vs` and vLLM-Omni action values come from separate vLLM-Omni campaigns.
4. Action vLLM-Omni rows report one GPU because the 8-GPU Ulysses runs failed sequence-length divisibility checks.
5. Policy-DROID vLLM-Omni uses the `nvidia/Cosmos3-Nano-Policy-DROID` checkpoint at 480p on one GPU only. It is reported in its own table and is not comparable to the PyTorch policy-robot row.
6. Diffusers rows are intentionally empty reservations for future benchmark campaigns.
7. The **GPUs** column gives the number of GPUs behind every value in that row.
8. Empty cells indicate unmeasured combinations, not unsupported combinations.</sub>
