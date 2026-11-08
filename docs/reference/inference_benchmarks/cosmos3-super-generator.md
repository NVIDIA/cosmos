# Cosmos3-Super Generator Inference Benchmarks

[Back to the inference benchmark index](../inference_benchmarks.md)

See [measurement settings and reproduction limits](benchmark_methodology.md) for runtime-specific timing, precision, and workload details.

These tables report **Cosmos3-Super Generator** latency in seconds. Lower is better.

Every row reports one GPU configuration, named in the **GPUs** column, and all three resolutions in that row come from it. The configuration is one GPU when the model fits on a single device at its highest resolution, 720p, and eight GPUs when it does not. The **GPUs** column is authoritative: engines can use different counts when their memory requirements differ. Primary PyTorch rows use eight GPUs on the H200 platforms while several other engines use one; SGLang uses eight on H200 141GB HBM3. Smaller-memory platforms also have eight-GPU rows. These configurations are measured setups rather than universal memory requirements.

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

The benchmark uses `t2av`, `v2av`, and `i2av`; some public recipes call the same sound-producing modes `t2vs`, `v2vs`, and `i2vs`. See the [audiovisual cookbook](https://nvda.ws/4zd0fw5) for generation inputs and the [action cookbook](https://nvda.ws/4dUeBsJ) for action representations and output contracts. vLLM-Omni request shapes are maintained in the [Super recipe](https://github.com/vllm-project/vllm-omni/blob/main/recipes/cosmos3/Cosmos3-Super.md). Public recipes cover more embodiments than these measurements; the action tables below intentionally use only its measured domain rows.

## Primary vision generation

### Text-to-Video (t2v)

| GPU | Engine | GPUs | 256p | 480p | 720p |
|---|---|:-:|---:|---:|---:|
| **RTX PRO 6000 Blackwell** | PyTorch | 8 |  | 118.90 | 427.16 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 8 |  |  | 437.57 |
|  | TensorRT-LLM | 8 |  | 85.99 | 254.55 |
|  | NIM | 8 | 4.15 | 27.18 | 91.15 |
|  | SGLang | 8 |  |  | 225.92 |
| **H20** | PyTorch | 8 | 27.72 | 152.46 | 492.41 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 8 | 26.66 | 161.49 | 517.37 |
|  | TensorRT-LLM | 8 | 15.89 | 131.10 | 443.47 |
|  | NIM | 8 | 6.79 |  |  |
|  | SGLang | 8 | 17.32 | 132.57 | 451.36 |
| **H100 NVL** | PyTorch | 8 |  | 64.14 | 186.19 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 8 | 24.21 | 108.70 | 322.28 |
|  | TensorRT-LLM | — |  |  |  |
|  | NIM | — |  |  |  |
| **H200 NVL** | PyTorch | 8 |  | 47.69 | 142.35 |
|  | vLLM-Omni | 1 | 27.54 | 252.33 | 911.49 |
|  | Diffusers | 1 | 33.00 | 286.80 | 1036.00 |
|  | TensorRT-LLM | — |  |  |  |
|  | NIM | — |  |  |  |
| **H100 80GB HBM3** | PyTorch | — |  |  |  |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 8 |  | 56.39 | 174.57 |
|  | TensorRT-LLM | 8 |  | 35.22 | 121.48 |
|  | NIM | — |  |  |  |
|  | SGLang | 8 |  | 32.25 | 113.48 |
| **H200 141GB HBM3** | PyTorch | 8 |  | 41.78 | 123.49 |
|  | vLLM-Omni | 1 | 25.61 | 219.11 | 769.63 |
|  | Diffusers | 1 | 31.00 | 251.60 | 886.20 |
|  | TensorRT-LLM | 1 | 24.21 | 234.08 | 853.44 |
|  | NIM | 1 | 9.75 |  |  |
|  | SGLang | 8 |  | 30.90 | 102.77 |
| **B200** | PyTorch | 1 | 14.66 | 114.38 | 407.50 |
|  | vLLM-Omni | 1 |  | 114.08 | 383.84 |
|  | Diffusers | 1 | 19.00 | 127.20 | 414.40 |
|  | TensorRT-LLM | 1 | 12.32 | 104.63 | 367.12 |
|  | NIM | 1 | 5.43 |  |  |
|  | SGLang | 1 | 13.07 | 121.68 | 445.94 |
| **B300** | PyTorch | — |  |  |  |
|  | vLLM-Omni | 1 |  | 109.03 | 366.66 |
|  | Diffusers | 1 | 17.73 | 125.70 | 398.94 |
|  | TensorRT-LLM | 1 | 11.85 | 97.66 | 333.28 |
|  | NIM | 1 | 5.42 |  |  |
|  | SGLang | 1 | 12.20 | 101.07 | 343.31 |

### Image-to-Video (i2v)

| GPU | Engine | GPUs | 256p | 480p | 720p |
|---|---|:-:|---:|---:|---:|
| **RTX PRO 6000 Blackwell** | PyTorch | 8 |  | 118.95 | 427.96 |
|  | vLLM-Omni | 8 |  |  |  |
|  | Diffusers | 8 |  |  |  |
|  | TensorRT-LLM | 8 |  | 88.56 |  |
|  | NIM | 8 | 4.01 | 27.39 | 91.65 |
|  | SGLang | 8 |  |  |  |
| **H20** | PyTorch | 8 | 28.22 | 153.19 | 491.93 |
|  | vLLM-Omni | 8 |  |  |  |
|  | Diffusers | 8 | 27.65 | 165.91 | 530.44 |
|  | TensorRT-LLM | 8 | 16.97 | 135.89 | 446.64 |
|  | NIM | 8 | 6.60 |  |  |
|  | SGLang | 8 | 17.44 | 132.63 | 451.96 |
| **H100 NVL** | PyTorch | 8 |  | 64.17 | 186.47 |
|  | vLLM-Omni | 8 |  |  |  |
|  | Diffusers | 8 |  | 112.36 | 325.56 |
|  | TensorRT-LLM | — |  |  |  |
|  | NIM | — |  |  |  |
| **H200 NVL** | PyTorch | 8 |  | 47.60 | 141.62 |
|  | vLLM-Omni | 1 | 27.90 | 254.29 | 915.05 |
|  | Diffusers | 1 | 33.00 | 287.20 | 1034.60 |
|  | TensorRT-LLM | — |  |  |  |
|  | NIM | — |  |  |  |
| **H100 80GB HBM3** | PyTorch | — |  |  |  |
|  | vLLM-Omni | 8 |  |  |  |
|  | Diffusers | 8 |  | 55.63 | 184.81 |
|  | TensorRT-LLM | 8 |  | 37.14 | 123.33 |
|  | NIM | — |  |  |  |
|  | SGLang | 8 |  | 31.90 | 113.18 |
| **H200 141GB HBM3** | PyTorch | 8 |  | 42.10 | 123.57 |
|  | vLLM-Omni | 1 | 25.47 | 220.70 | 766.33 |
|  | Diffusers | 1 | 31.00 | 249.20 | 879.20 |
|  | TensorRT-LLM | 1 | 24.46 | 230.02 |  |
|  | NIM | 1 | 9.27 |  |  |
|  | SGLang | 8 |  | 30.97 | 102.67 |
| **B200** | PyTorch | 1 | 14.71 | 112.40 | 397.31 |
|  | vLLM-Omni | 1 | 14.13 | 115.17 | 386.35 |
|  | Diffusers | 1 |  | 127.00 | 414.80 |
|  | TensorRT-LLM | 1 | 12.67 | 105.74 | 372.60 |
|  | NIM | 1 | 5.18 |  |  |
|  | SGLang | 1 | 13.22 | 120.24 | 453.23 |
| **B300** | PyTorch | — |  |  |  |
|  | vLLM-Omni | 1 | 14.14 | 111.42 | 368.73 |
|  | Diffusers | 1 |  | 132.23 | 410.81 |
|  | TensorRT-LLM | 1 |  | 98.90 | 336.20 |
|  | NIM | 1 | 5.27 |  |  |
|  | SGLang | 1 |  | 101.39 | 343.64 |

### Text-to-Image (t2i)

| GPU | Engine | GPUs | 256p | 480p | 720p |
|---|---|:-:|---:|---:|---:|
| **RTX PRO 6000 Blackwell** | PyTorch | 8 |  |  |  |
|  | vLLM-Omni | 8 |  |  |  |
|  | Diffusers | 8 |  |  |  |
|  | TensorRT-LLM | 8 |  |  |  |
|  | NIM | 8 | 1.52 | 1.64 | 1.61 |
|  | SGLang | 8 |  |  |  |
| **H20** | PyTorch | 8 |  |  |  |
|  | vLLM-Omni | 8 |  |  |  |
|  | Diffusers | 8 |  |  |  |
|  | TensorRT-LLM | 8 |  |  |  |
|  | NIM | 8 | 1.01 |  |  |
|  | SGLang | 8 |  |  |  |
| **H100 NVL** | PyTorch | 8 |  |  |  |
|  | vLLM-Omni | 8 |  |  |  |
|  | Diffusers | 8 |  |  |  |
|  | TensorRT-LLM | — |  |  |  |
|  | NIM | — |  |  |  |
| **H200 NVL** | PyTorch | 8 |  |  |  |
|  | vLLM-Omni | 1 | 2.73 | 6.28 | 11.02 |
|  | Diffusers | 1 | 5.00 | 8.00 | 12.00 |
|  | TensorRT-LLM | — |  |  |  |
|  | NIM | — |  |  |  |
| **H100 80GB HBM3** | PyTorch | — |  |  |  |
|  | vLLM-Omni | 8 |  |  |  |
|  | Diffusers | 8 |  |  |  |
|  | TensorRT-LLM | 8 |  |  |  |
|  | NIM | — |  |  |  |
|  | SGLang | 8 |  |  |  |
| **H200 141GB HBM3** | PyTorch | 8 |  |  |  |
|  | vLLM-Omni | 1 |  | 5.70 | 10.24 |
|  | Diffusers | 1 |  | 8.00 | 11.00 |
|  | TensorRT-LLM | 1 |  | 5.10 | 10.77 |
|  | NIM | 1 | 1.68 |  |  |
|  | SGLang | 8 |  |  |  |
| **B200** | PyTorch | 1 |  | 4.78 | 7.25 |
|  | vLLM-Omni | 1 |  |  |  |
|  | Diffusers | 1 |  |  | 8.00 |
|  | TensorRT-LLM | 1 |  |  | 5.72 |
|  | NIM | 1 |  |  |  |
|  | SGLang | 1 |  |  |  |
| **B300** | PyTorch | — |  |  |  |
|  | vLLM-Omni | 1 |  |  |  |
|  | Diffusers | 1 |  |  | 7.11 |
|  | TensorRT-LLM | 1 |  |  | 5.43 |
|  | NIM | 1 |  |  |  |
|  | SGLang | 1 |  |  |  |

<sub>Notes:
1. Measurements come from separate campaigns; identical prompt, seed, step budget, and timing boundaries are not established across engines. See the measurement settings before comparing rows.
2. Multi-GPU parallelism is runtime-specific and can split across tensor, CFG, context, Ring, or Ulysses dimensions; see the recorded settings.
3. vLLM-Omni numbers are for the upcoming public release in the vLLM-Omni repo and are subject to change.
4. Diffusers numbers use the HuggingFace `diffusers` integration without custom CUDA graphs. Single-GPU and multi-GPU values come from separate campaigns.
5. At 256p, multi-GPU configurations on B300 may underperform single-GPU because of small-workload tensor-parallel overhead, so single-GPU is the recommended deployment at this resolution.
6. PyTorch numbers report average generation (sampling) time from OSS inference benchmarking.
7. NIM numbers are from NIM 2.1.0 (`nvcr.io/nim/nvidia/cosmos3:2.1.0`) using latency profiles with FP8 precision, and report average generation time, excluding request processing and output encoding; step budgets and other workload differences still limit comparisons with other backends.
8. The 720p i2v cell for vLLM-Omni on H100 80GB HBM3 is empty because the multi-GPU runs for it returned times far below the 480p result, which is not physically plausible; the measurement is being rerun.
9. In the H200 NVL and H200 141GB HBM3 blocks, PyTorch reports eight GPUs, several engines report one, and SGLang reports eight on H200 141GB HBM3. Compare engines only after checking the count and workload.</sub>

## Additional audiovisual generation

### Video-to-Video (v2v)

A text prompt and source video condition a generated continuation or transformation.

| GPU | Engine | GPUs | 256p | 480p | 720p |
|---|---|:-:|---:|---:|---:|
| **RTX PRO 6000 Blackwell** | PyTorch | 8 |  | 118.92 | 427.19 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 8 |  |  |  |
|  | TensorRT-LLM | 8 |  | 86.53 |  |
|  | SGLang | 8 |  |  |  |
| **H20** | PyTorch | 8 | 27.98 | 153.02 | 491.75 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 8 | 28.18 | 167.95 | 532.68 |
|  | TensorRT-LLM | 8 | 16.67 | 131.96 | 444.75 |
|  | SGLang | 8 | 21.17 | 140.53 | 463.89 |
| **H100 NVL** | PyTorch | 8 |  | 63.84 | 183.64 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 8 | 25.01 | 112.47 | 348.31 |
|  | TensorRT-LLM | — |  |  |  |
| **H200 NVL** | PyTorch | 8 |  | 47.60 | 142.23 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 1 | 40.73 | 316.00 | 1093.16 |
|  | TensorRT-LLM | — |  |  |  |
| **H100 80GB HBM3** | PyTorch | — |  |  |  |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 8 |  | 57.74 | 188.43 |
|  | TensorRT-LLM | 8 |  | 36.06 | 122.50 |
|  | SGLang | 8 |  | 36.92 | 108.50 |
| **H200 141GB HBM3** | PyTorch | 8 | 11.76 | 41.85 | 123.09 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 1 | 41.17 | 269.44 | 916.41 |
|  | TensorRT-LLM | 1 | 24.19 | 230.98 |  |
|  | SGLang | 8 | 7.83 | 35.95 | 107.81 |
| **B200** | PyTorch | 1 | 14.30 | 111.49 | 395.97 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 1 |  | 142.61 | 437.27 |
|  | TensorRT-LLM | 1 |  | 104.79 | 370.21 |
|  | SGLang | 1 | 15.99 | 125.98 | 449.80 |
| **B300** | PyTorch | — |  |  |  |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 1 |  | 139.04 | 416.98 |
|  | TensorRT-LLM | 1 |  | 97.75 | 333.72 |
|  | SGLang | 1 | 16.21 | 106.91 | 347.54 |

### Text-to-Audio-and-Video (t2av)

A text prompt produces synchronized video and sound.

| GPU | Engine | GPUs | 256p | 480p | 720p |
|---|---|:-:|---:|---:|---:|
| **RTX PRO 6000 Blackwell** | PyTorch | 8 |  | 118.87 | 429.78 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 8 |  |  |  |
|  | TensorRT-LLM | 8 |  | 86.76 |  |
|  | SGLang | 8 |  |  |  |
| **H20** | PyTorch | 8 | 27.73 | 152.53 | 492.38 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 8 | 28.02 | 163.84 | 522.59 |
|  | TensorRT-LLM | 8 | 17.41 | 131.61 | 448.69 |
|  | SGLang | 8 | 18.89 | 132.99 | 457.49 |
| **H100 NVL** | PyTorch | 8 |  | 64.17 | 183.81 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 8 | 24.83 | 109.92 | 331.60 |
|  | TensorRT-LLM | — |  |  |  |
| **H200 NVL** | PyTorch | 8 |  | 47.67 | 142.06 |
|  | vLLM-Omni | 1 | 29.07 | 260.51 | 916.51 |
|  | Diffusers | 1 | 35.00 | 291.40 | 1034.20 |
|  | TensorRT-LLM | — |  |  |  |
| **H100 80GB HBM3** | PyTorch | — |  |  |  |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | — |  |  |  |
|  | TensorRT-LLM | — |  |  |  |
|  | SGLang | 8 | 5.04 | 32.79 | 114.59 |
| **H200 141GB HBM3** | PyTorch | 8 |  | 41.49 | 123.35 |
|  | vLLM-Omni | 1 | 27.21 | 221.67 | 763.74 |
|  | Diffusers | 1 | 33.00 | 253.40 | 874.60 |
|  | TensorRT-LLM | 1 | 25.89 | 229.12 |  |
|  | SGLang | 8 |  | 31.40 | 103.44 |
| **B200** | PyTorch | 1 | 14.52 | 112.99 | 395.20 |
|  | vLLM-Omni | 1 | 14.77 | 115.26 | 388.51 |
|  | Diffusers | 1 | 20.00 | 128.00 | 422.00 |
|  | TensorRT-LLM | 1 | 13.37 | 105.39 | 371.64 |
|  | SGLang | 1 | 14.25 | 121.67 | 451.00 |
| **B300** | PyTorch | — |  |  |  |
|  | vLLM-Omni | 1 | 15.26 | 113.34 | 372.74 |
|  | Diffusers | 1 | 18.94 | 126.86 | 402.37 |
|  | TensorRT-LLM | 1 | 12.76 | 98.59 | 335.38 |
|  | SGLang | 1 | 13.23 | 102.63 | 346.79 |

### Video-to-Audio-and-Video (v2av)

A text prompt and source video produce transformed video with synchronized sound.

| GPU | Engine | GPUs | 256p | 480p | 720p |
|---|---|:-:|---:|---:|---:|
| **RTX PRO 6000 Blackwell** | PyTorch | 8 |  | 118.88 | 428.64 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 8 |  | 159.85 | 548.30 |
|  | TensorRT-LLM | — |  |  |  |
| **H20** | PyTorch | 8 | 28.00 | 152.78 | 492.63 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 8 | 29.40 | 169.87 | 536.79 |
|  | TensorRT-LLM | — |  |  |  |
| **H100 NVL** | PyTorch | 8 |  | 64.69 | 183.62 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 8 | 26.59 | 116.36 | 338.13 |
|  | TensorRT-LLM | — |  |  |  |
| **H200 NVL** | PyTorch | 8 |  | 47.67 | 143.06 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 1 | 48.42 | 320.70 | 1093.26 |
|  | TensorRT-LLM | — |  |  |  |
| **H100 80GB HBM3** | PyTorch | — |  |  |  |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | — |  |  |  |
|  | TensorRT-LLM | — |  |  |  |
| **H200 141GB HBM3** | PyTorch | 8 | 11.74 | 41.59 | 123.06 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 1 | 43.23 | 270.19 | 909.99 |
|  | TensorRT-LLM | — |  |  |  |
| **B200** | PyTorch | 1 |  | 111.84 | 406.66 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 1 |  | 141.75 | 446.20 |
|  | TensorRT-LLM | — |  |  |  |
| **B300** | PyTorch | — |  |  |  |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 1 | 30.36 | 138.17 | 420.63 |
|  | TensorRT-LLM | — |  |  |  |

### Image-to-Audio-and-Video (i2av)

A text prompt and source image produce video with synchronized sound.

| GPU | Engine | GPUs | 256p | 480p | 720p |
|---|---|:-:|---:|---:|---:|
| **RTX PRO 6000 Blackwell** | PyTorch | 8 |  | 119.05 | 429.26 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 8 |  |  |  |
|  | TensorRT-LLM | 8 |  | 89.27 |  |
|  | SGLang | 8 |  |  |  |
| **H20** | PyTorch | 8 | 28.11 | 153.33 | 493.51 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 8 | 28.66 | 168.02 | 533.34 |
|  | TensorRT-LLM | 8 | 18.48 | 136.77 | 451.61 |
|  | SGLang | 8 | 19.04 | 133.38 | 457.18 |
| **H100 NVL** | PyTorch | 8 |  | 64.19 | 186.66 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 8 | 24.71 | 113.99 | 335.46 |
|  | TensorRT-LLM | — |  |  |  |
| **H200 NVL** | PyTorch | 8 |  | 47.65 | 142.18 |
|  | vLLM-Omni | 1 | 29.42 | 262.20 | 922.18 |
|  | Diffusers | 1 | 35.00 | 292.20 | 1037.00 |
|  | TensorRT-LLM | — |  |  |  |
| **H100 80GB HBM3** | PyTorch | — |  |  |  |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | — |  |  |  |
|  | TensorRT-LLM | — |  |  |  |
|  | SGLang | 8 | 5.14 | 32.70 | 114.75 |
| **H200 141GB HBM3** | PyTorch | 8 |  | 41.95 | 123.42 |
|  | vLLM-Omni | 1 | 27.58 | 220.57 | 774.15 |
|  | Diffusers | 1 | 33.00 | 256.00 | 880.40 |
|  | TensorRT-LLM | 1 | 26.41 | 233.74 |  |
|  | SGLang | 8 |  | 31.51 | 103.43 |
| **B200** | PyTorch | 1 | 14.84 | 113.32 | 407.30 |
|  | vLLM-Omni | 1 |  | 116.54 | 396.11 |
|  | Diffusers | 1 |  |  | 423.00 |
|  | TensorRT-LLM | 1 | 13.65 | 106.62 | 374.24 |
|  | SGLang | 1 | 14.57 | 121.49 | 454.14 |
| **B300** | PyTorch | — |  |  |  |
|  | vLLM-Omni | 1 |  | 114.02 | 373.17 |
|  | Diffusers | 1 |  |  | 414.22 |
|  | TensorRT-LLM | 1 |  | 99.94 | 338.12 |
|  | SGLang | 1 |  | 102.51 | 347.51 |

## Transfer generation

Transfer conditions a video-to-video generation on a structural control hint
extracted from the source video. Each control is reported separately because
the hint changes how much of the frame the model must synthesise.

| GPU | Transfer control | Engine | GPUs | 256p | 480p | 720p |
|---|---|---|:-:|---:|---:|---:|
| **RTX PRO 6000 Blackwell** | blur | PyTorch | 4 |  |  |  |
|  |  | vLLM-Omni | 4 |  |  |  |
|  |  | Diffusers | 4 |  |  |  |
|  |  | TensorRT-LLM | 4 |  |  |  |
|  |  | NIM | 4 |  |  |  |
|  |  | SGLang | 8 |  | 289.93 | 930.69 |
|  | depth | PyTorch | 4 |  |  |  |
|  |  | vLLM-Omni | 4 |  |  |  |
|  |  | Diffusers | 4 |  |  |  |
|  |  | TensorRT-LLM | 4 |  |  |  |
|  |  | NIM | 4 |  |  |  |
|  |  | SGLang | 8 |  | 292.48 | 932.27 |
|  | edge | PyTorch | 4 |  |  |  |
|  |  | vLLM-Omni | 4 |  |  |  |
|  |  | Diffusers | 4 |  |  |  |
|  |  | TensorRT-LLM | 4 |  |  |  |
|  |  | NIM | 4 |  |  |  |
|  |  | SGLang | 8 |  | 292.08 | 931.86 |
|  | multi_control | PyTorch | 4 |  |  |  |
|  |  | vLLM-Omni | 4 |  |  |  |
|  |  | Diffusers | 4 |  |  |  |
|  |  | TensorRT-LLM | 4 |  |  |  |
|  |  | NIM | 4 |  |  |  |
|  |  | SGLang | 8 |  | 495.39 | 1747.12 |
|  | seg | PyTorch | 4 |  |  |  |
|  |  | vLLM-Omni | 4 |  |  |  |
|  |  | Diffusers | 4 |  |  |  |
|  |  | TensorRT-LLM | 4 |  |  |  |
|  |  | NIM | 4 |  |  |  |
|  |  | SGLang | 8 |  | 293.07 | 933.66 |
|  | wsm | PyTorch | 4 |  |  |  |
|  |  | vLLM-Omni | 4 |  |  |  |
|  |  | Diffusers | 4 |  |  |  |
|  |  | TensorRT-LLM | 4 |  |  |  |
|  |  | NIM | 4 |  |  |  |
|  |  | SGLang | 8 |  | 117.26 | 363.28 |
| **H20** | blur | PyTorch | 4 |  |  |  |
|  |  | vLLM-Omni | 4 |  |  |  |
|  |  | Diffusers | 4 |  |  |  |
|  |  | TensorRT-LLM | 4 |  |  |  |
|  |  | NIM | 4 |  |  |  |
|  |  | SGLang | 8 |  | 531.24 | 1942.05 |
|  | depth | PyTorch | 4 |  |  |  |
|  |  | vLLM-Omni | 4 |  |  |  |
|  |  | Diffusers | 4 |  |  |  |
|  |  | TensorRT-LLM | 4 |  |  |  |
|  |  | NIM | 4 |  |  |  |
|  |  | SGLang | 8 |  | 538.33 | 1956.24 |
|  | edge | PyTorch | 4 |  |  |  |
|  |  | vLLM-Omni | 4 |  |  |  |
|  |  | Diffusers | 4 |  |  |  |
|  |  | TensorRT-LLM | 4 |  |  |  |
|  |  | NIM | 4 |  |  |  |
|  |  | SGLang | 8 |  | 532.48 | 1943.44 |
|  | multi_control | PyTorch | 4 |  |  |  |
|  |  | vLLM-Omni | 4 |  |  |  |
|  |  | Diffusers | 4 |  |  |  |
|  |  | TensorRT-LLM | 4 |  |  |  |
|  |  | NIM | 4 |  |  |  |
|  |  | SGLang | 8 |  | 955.81 | 3800.84 |
|  | seg | PyTorch | 4 |  |  |  |
|  |  | vLLM-Omni | 4 |  |  |  |
|  |  | Diffusers | 4 |  |  |  |
|  |  | TensorRT-LLM | 4 |  |  |  |
|  |  | NIM | 4 |  |  |  |
|  |  | SGLang | 8 |  | 534.44 | 1947.11 |
|  | wsm | PyTorch | 4 |  |  |  |
|  |  | vLLM-Omni | 4 |  |  |  |
|  |  | Diffusers | 4 |  |  |  |
|  |  | TensorRT-LLM | 4 |  |  |  |
|  |  | NIM | 4 |  |  |  |
|  |  | SGLang | 8 |  | 212.54 | 737.91 |
| **H100 NVL** | blur | PyTorch | — |  |  |  |
|  |  | vLLM-Omni | — |  |  |  |
|  |  | Diffusers | — |  |  |  |
|  |  | TensorRT-LLM | — |  |  |  |
|  |  | NIM | — |  |  |  |
|  | depth | PyTorch | — |  |  |  |
|  |  | vLLM-Omni | — |  |  |  |
|  |  | Diffusers | — |  |  |  |
|  |  | TensorRT-LLM | — |  |  |  |
|  |  | NIM | — |  |  |  |
|  | edge | PyTorch | — |  |  |  |
|  |  | vLLM-Omni | — |  |  |  |
|  |  | Diffusers | — |  |  |  |
|  |  | TensorRT-LLM | — |  |  |  |
|  |  | NIM | — |  |  |  |
|  | multi_control | PyTorch | — |  |  |  |
|  |  | vLLM-Omni | — |  |  |  |
|  |  | Diffusers | — |  |  |  |
|  |  | TensorRT-LLM | — |  |  |  |
|  |  | NIM | — |  |  |  |
|  | seg | PyTorch | — |  |  |  |
|  |  | vLLM-Omni | — |  |  |  |
|  |  | Diffusers | — |  |  |  |
|  |  | TensorRT-LLM | — |  |  |  |
|  |  | NIM | — |  |  |  |
|  | wsm | PyTorch | — |  |  |  |
|  |  | vLLM-Omni | — |  |  |  |
|  |  | Diffusers | — |  |  |  |
|  |  | TensorRT-LLM | — |  |  |  |
|  |  | NIM | — |  |  |  |
| **H100 80GB HBM3** | blur | PyTorch | 4 |  |  |  |
|  |  | vLLM-Omni | 4 |  |  |  |
|  |  | Diffusers | 4 |  |  |  |
|  |  | TensorRT-LLM | 4 | 23.08 | 169.42 | 604.47 |
|  |  | NIM | 4 |  |  |  |
|  |  | SGLang | 8 |  | 127.30 | 433.11 |
|  | depth | PyTorch | 4 |  |  |  |
|  |  | vLLM-Omni | 4 |  |  |  |
|  |  | Diffusers | 4 |  |  |  |
|  |  | TensorRT-LLM | 4 | 23.27 | 173.32 | 611.69 |
|  |  | NIM | 4 |  |  |  |
|  |  | SGLang | 8 |  | 128.48 | 439.88 |
|  | edge | PyTorch | 4 |  |  |  |
|  |  | vLLM-Omni | 4 |  |  |  |
|  |  | Diffusers | 4 |  |  |  |
|  |  | TensorRT-LLM | 4 | 23.04 | 170.63 | 603.31 |
|  |  | NIM | 4 |  |  |  |
|  |  | SGLang | 8 |  | 127.34 | 437.06 |
|  | multi_control | PyTorch | 4 |  |  |  |
|  |  | vLLM-Omni | 4 |  |  |  |
|  |  | Diffusers | 4 |  |  |  |
|  |  | TensorRT-LLM | 4 | 31.56 | 283.95 | 1092.62 |
|  |  | NIM | 4 |  |  |  |
|  |  | SGLang | 8 |  | 223.71 | 839.51 |
|  | seg | PyTorch | 4 |  |  |  |
|  |  | vLLM-Omni | 4 |  |  |  |
|  |  | Diffusers | 4 |  |  |  |
|  |  | TensorRT-LLM | 4 | 23.04 | 167.31 | 608.35 |
|  |  | NIM | 4 |  |  |  |
|  |  | SGLang | 8 |  | 126.94 | 438.26 |
|  | wsm | PyTorch | 4 |  |  |  |
|  |  | vLLM-Omni | 4 |  |  |  |
|  |  | Diffusers | 4 |  |  |  |
|  |  | TensorRT-LLM | 4 | 15.04 | 78.98 | 262.98 |
|  |  | NIM | 4 |  |  |  |
|  |  | SGLang | 8 |  | 50.03 | 168.60 |
| **H200 NVL** | blur | PyTorch | 1 |  |  |  |
|  |  | vLLM-Omni | 1 |  |  |  |
|  |  | Diffusers | 1 | 83.43 | 551.97 | 1935.10 |
|  |  | TensorRT-LLM | 1 |  |  |  |
|  |  | NIM | 1 |  |  |  |
|  | depth | PyTorch | 1 |  |  |  |
|  |  | vLLM-Omni | 1 |  |  |  |
|  |  | Diffusers | 1 | 98.52 | 572.97 | 1970.31 |
|  |  | TensorRT-LLM | 1 |  |  |  |
|  |  | NIM | 1 |  |  |  |
|  | edge | PyTorch | 1 |  |  |  |
|  |  | vLLM-Omni | 1 |  |  |  |
|  |  | Diffusers | 1 | 84.72 | 552.34 | 1940.58 |
|  |  | TensorRT-LLM | 1 |  |  |  |
|  |  | NIM | 1 |  |  |  |
|  | multi_control | PyTorch | 8 |  | 547.52 | 1259.88 |
|  |  | vLLM-Omni | 8 |  |  |  |
|  |  | Diffusers | 8 |  |  |  |
|  |  | TensorRT-LLM | 8 |  |  |  |
|  |  | NIM | 8 |  |  |  |
|  | seg | PyTorch | 1 |  |  |  |
|  |  | vLLM-Omni | 1 |  |  |  |
|  |  | Diffusers | 1 | 88.16 | 555.93 | 1950.38 |
|  |  | TensorRT-LLM | 1 |  |  |  |
|  |  | NIM | 1 |  |  |  |
|  | wsm | PyTorch | 1 |  |  |  |
|  |  | vLLM-Omni | 1 |  |  |  |
|  |  | Diffusers | 1 | 80.77 | 445.36 | 1470.27 |
|  |  | TensorRT-LLM | 1 |  |  |  |
|  |  | NIM | 1 |  |  |  |
| **H200 141GB HBM3** | blur | PyTorch | 1 |  |  |  |
|  |  | vLLM-Omni | 1 |  |  |  |
|  |  | Diffusers | 1 | 71.83 | 463.77 | 1593.34 |
|  |  | TensorRT-LLM | 1 | 57.89 | 580.64 | 2176.16 |
|  |  | NIM | 1 |  |  |  |
|  |  | SGLang | 8 |  | 123.70 | 431.24 |
|  | depth | PyTorch | 1 |  |  |  |
|  |  | vLLM-Omni | 1 |  |  |  |
|  |  | Diffusers | 1 | 84.88 | 481.84 | 1618.53 |
|  |  | TensorRT-LLM | 1 | 59.25 | 588.29 | 2193.25 |
|  |  | NIM | 1 |  |  |  |
|  |  | SGLang | 8 |  | 125.33 | 436.66 |
|  | edge | PyTorch | 1 |  |  |  |
|  |  | vLLM-Omni | 1 |  |  |  |
|  |  | Diffusers | 1 | 73.22 | 466.10 | 1594.90 |
|  |  | TensorRT-LLM | 1 | 58.54 | 580.81 | 2176.79 |
|  |  | NIM | 1 |  |  |  |
|  |  | SGLang | 8 |  | 123.91 | 430.80 |
|  | multi_control | PyTorch | 8 |  |  | 1021.43 |
|  |  | vLLM-Omni | 8 |  |  |  |
|  |  | Diffusers | 8 |  |  |  |
|  |  | TensorRT-LLM | 8 |  |  | 545.83 |
|  |  | NIM | 8 |  |  |  |
|  |  | SGLang | 8 |  | 217.73 | 826.71 |
|  | seg | PyTorch | 1 |  |  |  |
|  |  | vLLM-Omni | 1 |  |  |  |
|  |  | Diffusers | 1 | 77.09 | 476.15 | 1598.31 |
|  |  | TensorRT-LLM | 1 | 58.32 | 571.37 | 2166.54 |
|  |  | NIM | 1 |  |  |  |
|  |  | SGLang | 8 |  | 124.83 | 431.65 |
|  | wsm | PyTorch | 1 |  |  |  |
|  |  | vLLM-Omni | 1 |  |  |  |
|  |  | Diffusers | 1 |  | 382.72 | 1216.43 |
|  |  | TensorRT-LLM | 1 |  | 260.21 | 929.38 |
|  |  | NIM | 1 |  |  |  |
|  |  | SGLang | 8 |  | 50.53 | 166.76 |
| **B200** | blur | PyTorch | 1 | 58.11 | 298.06 | 1036.97 |
|  |  | vLLM-Omni | 1 |  |  |  |
|  |  | Diffusers | 1 | 42.14 | 241.28 | 789.92 |
|  |  | TensorRT-LLM | 1 | 30.02 | 258.30 | 939.00 |
|  |  | NIM | 1 |  |  |  |
|  |  | SGLang | 1 |  | 299.07 | 1131.59 |
|  | depth | PyTorch | 1 | 55.02 | 310.59 | 1069.76 |
|  |  | vLLM-Omni | 1 |  |  |  |
|  |  | Diffusers | 1 | 49.41 | 254.68 | 792.02 |
|  |  | TensorRT-LLM | 1 | 30.57 | 261.34 | 926.74 |
|  |  | NIM | 1 |  |  |  |
|  |  | SGLang | 1 |  | 300.86 | 1157.95 |
|  | edge | PyTorch | 1 | 47.64 | 297.04 | 1052.55 |
|  |  | vLLM-Omni | 1 |  |  |  |
|  |  | Diffusers | 1 | 43.57 | 241.17 | 774.34 |
|  |  | TensorRT-LLM | 1 | 29.67 | 258.50 | 919.21 |
|  |  | NIM | 1 |  |  |  |
|  |  | SGLang | 1 |  | 299.81 | 1146.37 |
|  | multi_control | PyTorch | 1 | 54.88 | 486.26 |  |
|  |  | vLLM-Omni | 1 |  |  |  |
|  |  | Diffusers | 1 |  |  |  |
|  |  | TensorRT-LLM | 1 | 40.45 |  |  |
|  |  | NIM | 1 |  |  |  |
|  |  | SGLang | 1 |  | 503.34 | 2122.28 |
|  | seg | PyTorch | 1 | 50.91 | 300.47 | 1055.66 |
|  |  | vLLM-Omni | 1 |  |  |  |
|  |  | Diffusers | 1 | 45.43 | 243.27 | 792.36 |
|  |  | TensorRT-LLM | 1 | 29.81 | 257.14 | 925.09 |
|  |  | NIM | 1 |  |  |  |
|  |  | SGLang | 1 |  | 295.67 | 1147.17 |
|  | wsm | PyTorch | 1 |  |  | 795.44 |
|  |  | vLLM-Omni | 1 |  |  |  |
|  |  | Diffusers | 1 |  | 198.30 | 604.37 |
|  |  | TensorRT-LLM | 1 |  |  | 403.95 |
|  |  | NIM | 1 |  |  |  |
|  |  | SGLang | 1 |  | 136.41 | 481.70 |
| **B300** | blur | PyTorch | 1 |  |  |  |
|  |  | vLLM-Omni | 1 |  |  |  |
|  |  | Diffusers | 1 | 39.64 | 226.08 | 720.04 |
|  |  | TensorRT-LLM | 1 | 28.84 | 239.51 | 836.14 |
|  |  | NIM | 1 |  |  |  |
|  |  | SGLang | 1 |  | 239.93 | 855.59 |
|  | depth | PyTorch | 1 |  |  |  |
|  |  | vLLM-Omni | 1 |  |  |  |
|  |  | Diffusers | 1 | 45.83 | 231.26 | 730.46 |
|  |  | TensorRT-LLM | 1 | 28.63 | 242.30 | 856.41 |
|  |  | NIM | 1 |  |  |  |
|  |  | SGLang | 1 |  | 243.00 | 848.57 |
|  | edge | PyTorch | 1 |  |  |  |
|  |  | vLLM-Omni | 1 |  |  |  |
|  |  | Diffusers | 1 | 39.99 | 226.42 | 719.35 |
|  |  | TensorRT-LLM | 1 | 28.70 | 243.59 | 835.37 |
|  |  | NIM | 1 |  |  |  |
|  |  | SGLang | 1 |  | 240.43 | 854.96 |
|  | multi_control | PyTorch | 1 |  |  |  |
|  |  | vLLM-Omni | 1 |  |  |  |
|  |  | Diffusers | 1 |  |  |  |
|  |  | TensorRT-LLM | 1 | 38.82 |  |  |
|  |  | NIM | 1 |  |  |  |
|  |  | SGLang | 1 |  |  |  |
|  | seg | PyTorch | 1 |  |  |  |
|  |  | vLLM-Omni | 1 |  |  |  |
|  |  | Diffusers | 1 | 41.64 | 228.18 | 720.63 |
|  |  | TensorRT-LLM | 1 | 28.66 | 239.26 | 834.49 |
|  |  | NIM | 1 |  |  |  |
|  |  | SGLang | 1 |  | 244.94 | 839.51 |
|  | wsm | PyTorch | 1 |  |  |  |
|  |  | vLLM-Omni | 1 |  |  |  |
|  |  | Diffusers | 1 |  | 186.76 | 555.47 |
|  |  | TensorRT-LLM | 1 |  | 112.87 | 372.96 |
|  |  | NIM | 1 |  |  |  |
|  |  | SGLang | 1 |  | 112.59 | 374.89 |

<sub>Transfer notes:
1. Values are generation latency in seconds; lower is better. Diffusers, vLLM-Omni, and TensorRT-LLM numbers come from separate campaigns; some Diffusers single-GPU modality values report the best run.
2. The **GPUs** column gives the number of GPUs behind every value in that row; check it before comparing engines.
3. PyTorch and NIM have no transfer coverage yet; those rows are unmeasured, not unsupported.
4. Control hints follow the vLLM-Omni `extra_params` names: `blur`, `depth`, `edge`, `multi_control`, `seg`, and `wsm`. `multi_control` combines several hints in one request.</sub>

## Action generation

Forward dynamics is reported separately for AV, camera, and robot inputs. Inverse dynamics and policy are reported for AV and robot because those are the only domain rows in the PyTorch action measurements; no camera row is inferred. For each domain, PyTorch is populated from that report. vLLM-Omni currently covers forward dynamics AV and inverse dynamics AV from the vLLM-Omni action measurements. Camera, robot-FD, robot-ID, policy-AV, and policy-robot vLLM-Omni rows remain reserved. **Policy — DROID** is a separate checkpoint and is not mixed with the policy-robot table; this Super page has no Policy-DROID measurements.

### Forward Dynamics — Autonomous Vehicle (AV)

| GPU | Engine | GPUs | 256p | 480p | 720p |
|---|---|:-:|---:|---:|---:|
| **RTX PRO 6000 Blackwell** | PyTorch | 8 |  |  |  |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 8 |  |  |  |
|  | TensorRT-LLM | — |  |  |  |
|  | SGLang | 8 |  |  |  |
| **H20** | PyTorch | 8 |  | 41.47 | 41.42 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 8 |  | 26.98 |  |
|  | TensorRT-LLM | — |  |  |  |
|  | SGLang | 8 |  | 32.12 |  |
| **H100 NVL** | PyTorch | 8 |  |  |  |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 8 |  | 19.29 | 20.05 |
|  | TensorRT-LLM | — |  |  |  |
| **H200 NVL** | PyTorch | 8 |  |  |  |
|  | vLLM-Omni | 1 |  | 28.78 | 82.08 |
|  | Diffusers | 1 |  | 53.19 | 51.89 |
|  | TensorRT-LLM | — |  |  |  |
| **H100 80GB HBM3** | PyTorch | — |  |  |  |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 8 |  | 12.89 | 13.45 |
|  | TensorRT-LLM | — |  |  |  |
|  | SGLang | 8 |  | 8.27 | 20.37 |
| **H200 141GB HBM3** | PyTorch | 8 |  | 14.41 | 14.41 |
|  | vLLM-Omni | 1 |  | 26.04 | 71.61 |
|  | Diffusers | 1 |  | 47.34 | 50.20 |
|  | TensorRT-LLM | — |  |  |  |
|  | SGLang | 8 |  | 8.10 | 19.73 |
| **B200** | PyTorch | 1 |  | 13.55 |  |
|  | vLLM-Omni | 1 |  | 14.59 |  |
|  | Diffusers | 1 |  |  |  |
|  | TensorRT-LLM | — |  |  |  |
|  | SGLang | 1 |  | 12.66 |  |
| **B300** | PyTorch | — |  |  |  |
|  | vLLM-Omni | 1 |  |  | 36.96 |
|  | Diffusers | 1 |  |  |  |
|  | TensorRT-LLM | — |  |  |  |
|  | SGLang | 1 |  |  | 31.31 |

### Forward Dynamics — Camera

| GPU | Engine | GPUs | 256p | 480p | 720p |
|---|---|:-:|---:|---:|---:|
| **RTX PRO 6000 Blackwell** | PyTorch | 8 |  |  | 59.55 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | — |  |  |  |
|  | TensorRT-LLM | — |  |  |  |
|  | SGLang | 8 |  |  | 64.29 |
| **H20** | PyTorch | 8 |  | 41.10 | 41.11 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | — |  |  |  |
|  | TensorRT-LLM | — |  |  |  |
|  | SGLang | 8 |  | 32.11 | 80.18 |
| **H100 NVL** | PyTorch | 8 |  |  |  |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | — |  |  |  |
|  | TensorRT-LLM | — |  |  |  |
| **H200 NVL** | PyTorch | 8 |  |  |  |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | — |  |  |  |
|  | TensorRT-LLM | — |  |  |  |
| **H100 80GB HBM3** | PyTorch | — |  |  |  |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | — |  |  |  |
|  | TensorRT-LLM | — |  |  |  |
|  | SGLang | 8 | 2.38 | 8.29 | 20.29 |
| **H200 141GB HBM3** | PyTorch | 8 |  | 14.30 | 14.29 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | — |  |  |  |
|  | TensorRT-LLM | — |  |  |  |
|  | SGLang | 8 |  | 8.12 | 19.74 |
| **B200** | PyTorch | 1 |  | 13.72 |  |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | — |  |  |  |
|  | TensorRT-LLM | — |  |  |  |
|  | SGLang | 1 |  | 12.63 |  |
| **B300** | PyTorch | — |  |  |  |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | — |  |  |  |
|  | TensorRT-LLM | — |  |  |  |
|  | SGLang | 1 | 1.90 | 11.72 | 31.47 |

### Forward Dynamics — Robot

| GPU | Engine | GPUs | 256p | 480p | 720p |
|---|---|:-:|---:|---:|---:|
| **RTX PRO 6000 Blackwell** | PyTorch | 8 |  |  |  |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 8 |  |  |  |
|  | TensorRT-LLM | — |  |  |  |
|  | SGLang | 8 |  |  |  |
| **H20** | PyTorch | 8 |  | 15.61 | 15.59 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 8 |  | 10.24 |  |
|  | TensorRT-LLM | — |  |  |  |
|  | SGLang | 8 |  | 8.90 |  |
| **H100 NVL** | PyTorch | 8 |  |  |  |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 8 | 8.03 | 8.49 | 8.61 |
|  | TensorRT-LLM | — |  |  |  |
| **H200 NVL** | PyTorch | 8 |  |  |  |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 1 | 22.21 | 28.53 | 27.35 |
|  | TensorRT-LLM | — |  |  |  |
| **H100 80GB HBM3** | PyTorch | — |  |  |  |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 8 |  |  | 9.25 |
|  | TensorRT-LLM | — |  |  |  |
|  | SGLang | 8 |  |  | 5.57 |
| **H200 141GB HBM3** | PyTorch | 8 |  |  | 8.70 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 1 | 22.02 | 26.45 | 26.38 |
|  | TensorRT-LLM | — |  |  |  |
|  | SGLang | 8 |  |  | 5.48 |
| **B200** | PyTorch | 1 |  | 3.49 |  |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 1 |  |  |  |
|  | TensorRT-LLM | — |  |  |  |
|  | SGLang | 1 |  | 3.55 |  |
| **B300** | PyTorch | — |  |  |  |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 1 |  |  |  |
|  | TensorRT-LLM | — |  |  |  |
|  | SGLang | 1 |  |  |  |

### Inverse Dynamics — Autonomous Vehicle (AV)

| GPU | Engine | GPUs | 256p | 480p | 720p |
|---|---|:-:|---:|---:|---:|
| **RTX PRO 6000 Blackwell** | PyTorch | 8 |  |  |  |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 8 |  |  |  |
|  | TensorRT-LLM | — |  |  |  |
|  | SGLang | 8 |  |  |  |
| **H20** | PyTorch | 8 |  | 40.96 | 41.06 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 8 |  | 26.66 | 28.46 |
|  | TensorRT-LLM | — |  |  |  |
|  | SGLang | 8 |  | 30.32 | 30.34 |
| **H100 NVL** | PyTorch | 8 |  |  |  |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 8 |  | 18.67 | 20.67 |
|  | TensorRT-LLM | — |  |  |  |
| **H200 NVL** | PyTorch | 8 |  |  |  |
|  | vLLM-Omni | 1 |  | 28.21 | 80.85 |
|  | Diffusers | 1 |  | 44.50 | 43.51 |
|  | TensorRT-LLM | — |  |  |  |
| **H100 80GB HBM3** | PyTorch | — |  |  |  |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 8 | 10.49 | 12.73 | 13.30 |
|  | TensorRT-LLM | — |  |  |  |
|  | SGLang | 8 | 8.11 | 8.07 | 8.07 |
| **H200 141GB HBM3** | PyTorch | 8 | 14.30 | 14.32 | 14.16 |
|  | vLLM-Omni | 1 |  | 25.99 | 71.45 |
|  | Diffusers | 1 |  | 39.75 | 41.71 |
|  | TensorRT-LLM | — |  |  |  |
|  | SGLang | 8 | 7.93 | 7.93 | 7.91 |
| **B200** | PyTorch | 1 |  | 13.37 |  |
|  | vLLM-Omni | 1 |  | 14.69 |  |
|  | Diffusers | 1 |  |  |  |
|  | TensorRT-LLM | — |  |  |  |
|  | SGLang | 1 |  | 12.89 |  |
| **B300** | PyTorch | — |  |  |  |
|  | vLLM-Omni | 1 |  | 14.44 |  |
|  | Diffusers | 1 |  |  |  |
|  | TensorRT-LLM | — |  |  |  |
|  | SGLang | 1 |  |  |  |

### Inverse Dynamics — Robot

| GPU | Engine | GPUs | 256p | 480p | 720p |
|---|---|:-:|---:|---:|---:|
| **RTX PRO 6000 Blackwell** | PyTorch | 8 |  |  |  |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 8 |  |  |  |
|  | TensorRT-LLM | — |  |  |  |
|  | SGLang | 8 |  |  |  |
| **H20** | PyTorch | 8 | 15.51 | 15.52 | 15.51 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 8 | 9.48 | 10.17 | 10.65 |
|  | TensorRT-LLM | — |  |  |  |
|  | SGLang | 8 | 9.01 | 9.01 | 9.01 |
| **H100 NVL** | PyTorch | 8 |  |  |  |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 8 | 7.92 | 8.46 | 8.79 |
|  | TensorRT-LLM | — |  |  |  |
| **H200 NVL** | PyTorch | 8 |  |  |  |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 1 | 12.09 | 16.87 | 18.74 |
|  | TensorRT-LLM | — |  |  |  |
| **H100 80GB HBM3** | PyTorch | — |  |  |  |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 8 |  |  |  |
|  | TensorRT-LLM | — |  |  |  |
|  | SGLang | 8 |  |  |  |
| **H200 141GB HBM3** | PyTorch | 8 |  |  |  |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 1 |  |  | 16.29 |
|  | TensorRT-LLM | — |  |  |  |
|  | SGLang | 8 |  |  |  |
| **B200** | PyTorch | 1 | 3.42 | 3.49 | 3.41 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 1 |  |  |  |
|  | TensorRT-LLM | — |  |  |  |
|  | SGLang | 1 | 3.85 | 3.80 | 3.78 |
| **B300** | PyTorch | — |  |  |  |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 1 |  |  |  |
|  | TensorRT-LLM | — |  |  |  |
|  | SGLang | 1 |  |  |  |

### Policy — Autonomous Vehicle (AV)

| GPU | Engine | GPUs | 256p | 480p | 720p |
|---|---|:-:|---:|---:|---:|
| **RTX PRO 6000 Blackwell** | PyTorch | 8 |  |  |  |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 8 |  |  |  |
|  | TensorRT-LLM | — |  |  |  |
| **H20** | PyTorch | 8 |  | 41.51 | 41.49 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 8 |  | 26.61 | 28.81 |
|  | TensorRT-LLM | — |  |  |  |
| **H100 NVL** | PyTorch | 8 |  |  |  |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 8 |  | 18.80 | 20.12 |
|  | TensorRT-LLM | — |  |  |  |
| **H200 NVL** | PyTorch | 8 |  |  |  |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 1 | 15.35 | 41.89 | 45.40 |
|  | TensorRT-LLM | — |  |  |  |
| **H100 80GB HBM3** | PyTorch | — |  |  |  |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 8 | 10.55 | 13.02 | 13.52 |
|  | TensorRT-LLM | — |  |  |  |
| **H200 141GB HBM3** | PyTorch | 8 | 14.54 | 14.41 | 14.52 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 1 | 14.71 | 41.30 | 40.35 |
|  | TensorRT-LLM | — |  |  |  |
| **B200** | PyTorch | 1 |  | 13.73 | 13.60 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 1 |  |  |  |
|  | TensorRT-LLM | — |  |  |  |
| **B300** | PyTorch | — |  |  |  |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 1 |  |  |  |
|  | TensorRT-LLM | — |  |  |  |

### Policy — Robot

| GPU | Engine | GPUs | 256p | 480p | 720p |
|---|---|:-:|---:|---:|---:|
| **RTX PRO 6000 Blackwell** | PyTorch | 8 |  |  |  |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 8 |  |  |  |
|  | TensorRT-LLM | — |  |  |  |
| **H20** | PyTorch | 8 | 15.60 | 15.60 | 15.62 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 8 | 9.03 | 10.19 | 10.73 |
|  | TensorRT-LLM | — |  |  |  |
| **H100 NVL** | PyTorch | 8 |  |  |  |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 8 | 8.05 | 8.60 | 8.89 |
|  | TensorRT-LLM | — |  |  |  |
| **H200 NVL** | PyTorch | 8 |  |  |  |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 1 | 13.82 | 16.92 | 17.03 |
|  | TensorRT-LLM | — |  |  |  |
| **H100 80GB HBM3** | PyTorch | — |  |  |  |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 8 | 8.71 | 9.09 | 9.12 |
|  | TensorRT-LLM | — |  |  |  |
| **H200 141GB HBM3** | PyTorch | 8 | 8.72 | 8.75 | 8.70 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 1 | 11.90 | 16.13 | 14.72 |
|  | TensorRT-LLM | — |  |  |  |
| **B200** | PyTorch | 1 | 3.48 | 3.49 | 3.48 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 1 |  |  |  |
|  | TensorRT-LLM | — |  |  |  |
| **B300** | PyTorch | — |  |  |  |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 1 |  |  |  |
|  | TensorRT-LLM | — |  |  |  |

### Policy — DROID

The vLLM-Omni action campaign measured Policy-DROID only for Cosmos3-Nano-Policy-DROID. Super has no Policy-DROID row in that sweep; this table is reserved.

| GPU | Engine | GPUs | 256p | 480p | 720p |
|---|---|:-:|---:|---:|---:|
| **RTX PRO 6000 Blackwell** | PyTorch | — |  |  |  |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | — |  |  |  |
|  | TensorRT-LLM | — |  |  |  |
| **H20** | PyTorch | — |  |  |  |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | — |  |  |  |
|  | TensorRT-LLM | — |  |  |  |
| **H100 NVL** | PyTorch | — |  |  |  |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | — |  |  |  |
|  | TensorRT-LLM | — |  |  |  |
| **H200 NVL** | PyTorch | — |  |  |  |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | — |  |  |  |
|  | TensorRT-LLM | — |  |  |  |
| **H100 80GB HBM3** | PyTorch | — |  |  |  |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | — |  |  |  |
|  | TensorRT-LLM | — |  |  |  |
| **H200 141GB HBM3** | PyTorch | — |  |  |  |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | — |  |  |  |
|  | TensorRT-LLM | — |  |  |  |
| **B200** | PyTorch | — |  |  |  |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | — |  |  |  |
|  | TensorRT-LLM | — |  |  |  |
| **B300** | PyTorch | — |  |  |  |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | — |  |  |  |
|  | TensorRT-LLM | — |  |  |  |

<sub>Additional-modality notes:
1. PyTorch values are average generation (sampling) latency in seconds; lower is better.
2. PyTorch values use `CUDA_GRAPH=No` and the `latency` automatic-sharding preset.
3. vLLM-Omni audiovisual values for `t2av`/`t2vs` and `i2av`/`i2vs` and vLLM-Omni action values come from separate vLLM-Omni campaigns.
4. Action vLLM-Omni rows report one GPU because the 8-GPU Ulysses runs failed sequence-length divisibility checks.
5. Policy-DROID is reported in its own table. The vLLM-Omni action campaign has no Super Policy-DROID measurement.
6. Diffusers rows are intentionally empty reservations for future benchmark campaigns.
7. The **GPUs** column gives the number of GPUs behind every value in that row.
8. Empty cells indicate unmeasured combinations, not unsupported combinations.</sub>
