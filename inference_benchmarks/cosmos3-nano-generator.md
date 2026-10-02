# Cosmos3-Nano Generator Inference Benchmarks

[Back to the inference benchmark index](../inference_benchmarks.md)

These tables report **Cosmos3-Nano Generator** latency in seconds. Lower is better.

Every row reports one GPU configuration, named in the **GPUs** column, and all three resolutions in that row come from it. The configuration is one GPU when the model fits on a single device at its highest resolution, 720p, and eight GPUs when it does not. Within one GPU's block every engine reports the same configuration, so a column can be read straight across the runtimes. Cosmos3-Nano fits on a single GPU at 720p, so all populated rows here report one GPU.

Empty cells mean that a run has not been completed for that GPU, engine, or resolution; they do not indicate that a combination is unsupported.

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

The primary t2v, i2v, and t2i tables cover PyTorch, vLLM-Omni, Diffusers, and NIM. Their vLLM-Omni values are from PBR `#307999`, **Cosmos3-Generator vLLM-Omni Inference Benchmarking (t2i, t2v, i2v)**; the PyTorch and NIM values preserve the previously published campaigns. Those tables use BF16 precision, batch size 1, and matched prompts, seeds, and sampler settings where documented. Video workloads follow the standard Cosmos3 generation profile of 189 frames at 24 FPS unless a resolution tier limits frame count.

The additional audiovisual and action tables come from three internal benchmark reports. PyTorch values are from PBR `#308197`, **Cosmos3-Generator OSS Inference Benchmarking 32B and 8B (189 frames)**: average generation (sampling) latency from the native OSS path, using **CUDA Graphs disabled** and the **latency** automatic-sharding preset. vLLM-Omni values for text-to-audio-and-video (`t2av`/`t2vs`) and image-to-audio-and-video (`i2av`/`i2vs`) are from PBR `#308195`. vLLM-Omni action values are from PBR `#308481`, **Cosmos3-Generator vLLM-Omni Inference Benchmarking (action)**, measured with the `vllm/vllm-omni:cosmos3` image and the official action cookbook samples; `DIFFUSION_ATTENTION_BACKEND=TORCH_SDPA` was not set. Forward-dynamics cells are the mean of `av_forward`, `av_left`, and `av_right`; inverse-dynamics cells are the mean of `av_inverse_0` and `av_inverse_1`. That action sweep covers 1, 2, and 4 GPUs only, because its 8-GPU Ulysses runs failed a sequence-length divisibility check; those rows therefore report the single-GPU configuration. Policy-DROID is a separate checkpoint and was measured only at 480p on one GPU. That action sweep did not cover Super on H20, H100 NVL, or H100 80GB HBM3. Diffusers values come from four further reports: PBR `#308202` (all modalities) and PBR `#308451` (video-to-video) on one GPU, PBR `#308918` for 4- and 8-GPU runs, and PBR `#308587` for transfer. vLLM-Omni transfer values are from PBR `#308574`. Where PBR `#308918` offers several tensor- and context-parallel splits at the same GPU count, the fastest is published. Its 1-GPU numbers are not used, because that sweep ran a different denoising-step budget than the single-GPU reports. Values are rounded to two decimal places.

These reports establish the reported timing matrix but do not expose every prompt and action payload in this repository. The linked public recipes explain modality behavior and provide representative payloads; their example-specific frame counts and action chunk sizes should not be treated as the exact internal benchmark inputs.

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

The PBR uses `t2av`, `v2av`, and `i2av`; some public recipes call the same sound-producing modes `t2vs`, `v2vs`, and `i2vs`. See the [audiovisual cookbook](../cookbooks/cosmos3/generator/audiovisual/README.md) for generation inputs and the [action cookbook](../cookbooks/cosmos3/generator/action/README.md) for action representations and output contracts. vLLM-Omni request shapes are maintained in the [Nano recipe](https://github.com/vllm-project/vllm-omni/blob/main/recipes/cosmos3/Cosmos3-Nano.md). Public recipes cover more embodiments than this PBR; the action tables below intentionally use only its measured domain rows.

## Primary vision generation

### Text-to-Video (t2v)

| GPU | Engine | GPUs | 256p | 480p | 720p |
|---|---|:-:|---:|---:|---:|
| **RTX PRO 6000 Blackwell** | PyTorch | 1 | 13.95 | 180.81 | 786.37 |
|  | vLLM-Omni | 1 | 10.65 | 105.61 | 371.53 |
|  | Diffusers | 1 | 11.20 | 112.00 | 392.00 |
|  | NIM | 1 | 7.93 | 82.69 | 318.69 |
| **H20** | PyTorch | 1 | 30.57 | 257.51 | 931.39 |
|  | vLLM-Omni | 1 | 28.58 | 256.97 | 929.81 |
|  | Diffusers | 1 | 30.20 | 258.00 | 926.00 |
|  | NIM | 1 | 17.81 | 192.07 | 771.37 |
| **H100 NVL** | PyTorch | 1 | 10.03 | 84.12 | 297.27 |
|  | vLLM-Omni | 1 | 9.25 | 80.75 | 281.99 |
|  | Diffusers | 1 | 11.00 | 90.00 | 324.20 |
|  | NIM | 1 | 7.09 | 68.01 | 267.73 |
| **H200 NVL** | PyTorch | 1 | 8.17 | 69.79 | 244.39 |
|  | vLLM-Omni | 1 | 7.44 | 64.58 | 240.05 |
|  | Diffusers | 1 | 9.00 | 74.00 | 276.20 |
|  | NIM | 1 | 5.87 | 57.10 | 229.63 |
| **H100 80GB HBM3** | PyTorch | 1 | 7.61 | 59.83 | 207.78 |
|  | vLLM-Omni | 1 | 6.97 | 58.17 | 202.29 |
|  | Diffusers | 1 | 9.00 | 68.00 | 240.00 |
|  | NIM | 1 | 5.72 | 51.73 | 199.46 |
| **H200 141GB HBM3** | PyTorch | 1 | 7.53 | 60.18 | 214.28 |
|  | vLLM-Omni | 1 | 6.79 | 58.14 | 208.36 |
|  | Diffusers | 1 | 9.00 | 67.00 | 239.60 |
|  | NIM | 1 | 5.72 | 51.90 | 200.63 |
| **B200** | PyTorch | 1 | 4.56 | 33.20 | 114.85 |
|  | vLLM-Omni | 1 | 4.03 | 32.04 | 107.79 |
|  | Diffusers | 1 | 7.00 | 36.80 | 117.00 |
|  | NIM | 1 | 3.72 | 26.68 | 93.33 |
| **B300** | PyTorch | — |  |  |  |
|  | vLLM-Omni | 1 | 4.46 | 32.18 | 102.10 |
|  | Diffusers | 1 | 6.40 | 39.32 | 121.74 |
|  | NIM | 1 | 4.51 | 27.49 | 90.39 |

### Image-to-Video (i2v)

| GPU | Engine | GPUs | 256p | 480p | 720p |
|---|---|:-:|---:|---:|---:|
| **RTX PRO 6000 Blackwell** | PyTorch | 1 | 14.08 | 182.14 | 788.80 |
|  | vLLM-Omni | 1 | 11.04 | 107.77 | 376.44 |
|  | Diffusers | 1 | 12.00 | 112.00 | 397.00 |
|  | NIM | 1 | 8.23 | 84.58 | 326.96 |
| **H20** | PyTorch | 1 | 31.36 | 257.10 | 933.07 |
|  | vLLM-Omni | 1 | 29.50 | 261.56 | 940.16 |
|  | Diffusers | 1 | 31.00 | 258.00 | 925.00 |
|  | NIM | 1 | 18.67 | 195.10 | 774.88 |
| **H100 NVL** | PyTorch | 1 | 10.19 | 84.50 | 298.57 |
|  | vLLM-Omni | 1 | 9.62 | 82.61 | 286.19 |
|  | Diffusers | 1 | 11.00 | 91.00 | 325.20 |
|  | NIM | 1 | 7.39 | 69.39 | 272.29 |
| **H200 NVL** | PyTorch | 1 | 8.27 | 69.99 | 246.62 |
|  | vLLM-Omni | 1 | 7.83 | 66.39 | 243.52 |
|  | Diffusers | 1 | 9.00 | 74.00 | 275.20 |
|  | NIM | 1 | 6.25 | 58.64 | 232.47 |
| **H100 80GB HBM3** | PyTorch | 1 | 7.64 | 59.95 | 207.87 |
|  | vLLM-Omni | 1 | 7.37 | 59.77 | 205.97 |
|  | Diffusers | 1 | 9.00 | 68.00 | 239.80 |
|  | NIM | 1 | 6.08 | 53.02 | 202.59 |
| **H200 141GB HBM3** | PyTorch | 1 | 7.65 | 60.51 | 214.80 |
|  | vLLM-Omni | 1 | 7.28 | 59.64 | 209.65 |
|  | Diffusers | 1 | 9.00 | 67.20 | 240.00 |
|  | NIM | 1 | 6.04 | 53.25 | 203.66 |
| **B200** | PyTorch | 1 | 4.60 | 33.08 | 113.90 |
|  | vLLM-Omni | 1 | 4.33 | 33.09 | 110.08 |
|  | Diffusers | 1 | 7.00 | 37.00 | 116.00 |
|  | NIM | 1 | 4.05 | 27.72 | 95.57 |
| **B300** | PyTorch | — |  |  |  |
|  | vLLM-Omni | 1 | 5.61 | 33.45 | 104.75 |
|  | Diffusers | 1 | 13.40 | 49.59 | 131.25 |
|  | NIM | 1 | 4.50 | 28.90 | 92.59 |

### Text-to-Image (t2i)

| GPU | Engine | GPUs | 256p | 480p | 720p |
|---|---|:-:|---:|---:|---:|
| **RTX PRO 6000 Blackwell** | PyTorch | 1 | 2.99 | 4.51 | 7.12 |
|  | vLLM-Omni | 1 | 1.59 | 2.87 | 4.99 |
|  | Diffusers | 1 | 2.00 | 4.00 | 5.00 |
| **H20** | PyTorch | 1 | 3.06 | 6.51 | 12.31 |
|  | vLLM-Omni | 1 | 1.73 | 4.92 | 10.73 |
|  | Diffusers | 1 | 3.00 | 6.00 | 10.00 |
| **H100 NVL** | PyTorch | 1 | 2.77 | 2.83 | 4.21 |
|  | vLLM-Omni | 1 | 1.55 | 1.92 | 3.43 |
|  | Diffusers | 1 | 3.00 | 3.00 | 4.00 |
| **H200 NVL** | PyTorch | 1 | 2.75 | 2.85 | 3.58 |
|  | vLLM-Omni | 1 | 1.53 | 1.58 | 2.81 |
|  | Diffusers | 1 | 3.00 | 3.00 | 4.00 |
| **H100 80GB HBM3** | PyTorch | 1 | 3.01 | 3.01 | 3.45 |
|  | vLLM-Omni | 1 | 1.61 | 1.53 | 2.61 |
|  | Diffusers | 1 | 3.00 | 3.00 | 4.00 |
| **H200 141GB HBM3** | PyTorch | 1 | 2.96 | 3.04 | 3.28 |
|  | vLLM-Omni | 1 | 1.57 | 1.52 | 2.60 |
|  | Diffusers | 1 | 3.00 | 3.00 | 4.00 |
| **B200** | PyTorch | 1 | 2.68 | 2.75 | 2.87 |
|  | vLLM-Omni | 1 | 1.49 | 1.20 | 1.76 |
|  | Diffusers | 1 | 3.00 | 3.20 | 3.00 |
| **B300** | PyTorch | — |  |  |  |
|  | vLLM-Omni | 1 | 1.97 | 1.81 | 2.34 |
|  | Diffusers | 1 | 3.86 | 3.86 | 3.99 |

<sub>Notes:
1. All times measured on identical workloads (same seed, sampler settings, prompt).
2. Multi-GPU configurations use tensor parallelism.
3. vLLM-Omni numbers are for the upcoming public release in the vLLM-Omni repo and are subject to change before GA; the H100 NVL values in particular are pre-release.
4. Diffusers numbers use the HuggingFace `diffusers` integration without custom CUDA graphs. Single-GPU values come from PBR `#308202` and PBR `#308451`; multi-GPU values from PBR `#308918`.
5. PyTorch numbers report average generation (sampling) time from OSS inference benchmarking.
6. At 256p, multi-GPU configurations on B300 may underperform single-GPU because of small-workload tensor-parallel overhead, so single-GPU is the recommended deployment at this resolution.
7. NIM numbers use latency profiles with FP8 precision and report end-to-end `Request Latency s`, including request processing, video generation, output encoding, and returning the response.</sub>

## Additional audiovisual generation

### Video-to-Video (v2v)

A text prompt and source video condition a generated continuation or transformation.

| GPU | Engine | GPUs | 256p | 480p | 720p |
|---|---|:-:|---:|---:|---:|
| **RTX PRO 6000 Blackwell** | PyTorch | 1 | 13.73 | 180.54 | 785.41 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 1 | 24.19 | 135.85 | 432.18 |
| **H20** | PyTorch | 1 | 30.44 | 257.79 | 930.96 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 1 | 44.88 | 287.55 | 974.75 |
| **H100 NVL** | PyTorch | 1 | 10.18 | 81.73 | 297.83 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 1 | 23.66 | 107.79 | 354.66 |
| **H200 NVL** | PyTorch | 1 | 8.10 | 69.70 | 250.33 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 1 | 21.40 | 96.52 | 313.13 |
| **H100 80GB HBM3** | PyTorch | 1 | 7.50 | 59.75 | 207.36 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 1 | 19.48 | 86.51 | 267.90 |
| **H200 141GB HBM3** | PyTorch | 1 | 7.48 | 60.23 | 214.03 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 1 | 18.93 | 85.52 | 266.84 |
| **B200** | PyTorch | 1 | 4.52 | 32.53 | 114.16 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 1 | 16.93 | 52.32 | 137.54 |
| **B300** | PyTorch | — |  |  |  |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 1 | 16.14 | 50.74 | 131.11 |

### Text-to-Audio-and-Video (t2av)

A text prompt produces synchronized video and sound.

| GPU | Engine | GPUs | 256p | 480p | 720p |
|---|---|:-:|---:|---:|---:|
| **RTX PRO 6000 Blackwell** | PyTorch | 1 | 14.13 | 180.98 | 794.74 |
|  | vLLM-Omni | 1 | 12.25 | 106.02 | 373.10 |
|  | Diffusers | 1 | 13.00 | 113.00 | 393.00 |
| **H20** | PyTorch | 1 | 30.82 | 256.41 | 922.35 |
|  | vLLM-Omni | 1 | 29.43 | 260.84 | 937.93 |
|  | Diffusers | 1 | 32.00 | 263.00 | 932.60 |
| **H100 NVL** | PyTorch | 1 | 9.99 | 82.32 | 297.88 |
|  | vLLM-Omni | 1 | 9.92 | 81.57 | 291.91 |
|  | Diffusers | 1 | 12.00 | 91.00 | 324.60 |
| **H200 NVL** | PyTorch | 1 | 8.20 | 69.88 | 250.46 |
|  | vLLM-Omni | 1 | 7.82 | 65.42 | 244.60 |
|  | Diffusers | 1 | 10.00 | 75.00 | 277.80 |
| **H100 80GB HBM3** | PyTorch | 1 | 7.60 | 59.09 | 207.87 |
|  | vLLM-Omni | 1 | 7.34 | 58.65 | 203.92 |
|  | Diffusers | 1 | 9.40 | 69.00 | 242.00 |
| **H200 141GB HBM3** | PyTorch | 1 | 7.55 | 60.17 | 211.59 |
|  | vLLM-Omni | 1 | 7.13 | 58.27 | 209.29 |
|  | Diffusers | 1 | 9.00 | 69.00 | 237.00 |
| **B200** | PyTorch | 1 | 4.63 | 32.60 | 113.89 |
|  | vLLM-Omni | 1 | 4.29 | 32.20 | 107.74 |
|  | Diffusers | 1 | 7.00 | 37.00 | 117.00 |
| **B300** | PyTorch | — |  |  |  |
|  | vLLM-Omni | 1 | 5.59 | 33.15 | 104.92 |
|  | Diffusers | 1 | 6.46 | 39.66 | 122.33 |

### Video-to-Audio-and-Video (v2av)

A text prompt and source video produce transformed video with synchronized sound.

| GPU | Engine | GPUs | 256p | 480p | 720p |
|---|---|:-:|---:|---:|---:|
| **RTX PRO 6000 Blackwell** | PyTorch | 1 | 13.73 | 180.72 | 786.11 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 1 | 26.30 | 135.47 | 429.61 |
| **H20** | PyTorch | 1 | 30.42 | 257.83 | 921.82 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 1 | 46.72 | 288.98 | 985.80 |
| **H100 NVL** | PyTorch | 1 | 9.91 | 81.96 | 297.51 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 1 | 22.95 | 110.93 | 354.02 |
| **H200 NVL** | PyTorch | 1 | 8.10 | 69.59 | 246.51 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 1 | 22.19 | 94.85 | 314.06 |
| **H100 80GB HBM3** | PyTorch | 1 | 7.49 | 59.86 | 207.55 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 1 | 21.73 | 88.42 | 268.93 |
| **H200 141GB HBM3** | PyTorch | 1 | 7.49 | 60.15 | 214.44 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 1 | 17.98 | 87.84 | 267.54 |
| **B200** | PyTorch | 1 | 4.51 | 32.62 | 113.72 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 1 | 15.64 | 50.57 | 138.15 |
| **B300** | PyTorch | — |  |  |  |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 1 | 15.93 | 51.87 | 131.52 |

### Image-to-Audio-and-Video (i2av)

A text prompt and source image produce video with synchronized sound.

| GPU | Engine | GPUs | 256p | 480p | 720p |
|---|---|:-:|---:|---:|---:|
| **RTX PRO 6000 Blackwell** | PyTorch | 1 | 14.08 | 181.64 | 787.81 |
|  | vLLM-Omni | 1 | 12.61 | 108.05 | 378.04 |
|  | Diffusers | 1 | 14.00 | 113.00 | 393.00 |
| **H20** | PyTorch | 1 | 31.34 | 259.06 | 924.39 |
|  | vLLM-Omni | 1 | 30.36 | 265.37 | 948.09 |
|  | Diffusers | 1 | 32.00 | 263.00 | 934.80 |
| **H100 NVL** | PyTorch | 1 | 10.45 | 84.37 | 298.61 |
|  | vLLM-Omni | 1 | 10.34 | 83.21 | 295.65 |
|  | Diffusers | 1 | 12.00 | 91.00 | 324.40 |
| **H200 NVL** | PyTorch | 1 | 8.29 | 71.20 | 248.35 |
|  | vLLM-Omni | 1 | 8.24 | 67.23 | 249.41 |
|  | Diffusers | 1 | 10.00 | 75.00 | 277.20 |
| **H100 80GB HBM3** | PyTorch | 1 | 7.73 | 59.13 | 207.93 |
|  | vLLM-Omni | 1 | 7.74 | 60.27 | 207.53 |
|  | Diffusers | 1 | 10.00 | 69.00 | 241.00 |
| **H200 141GB HBM3** | PyTorch | 1 | 7.66 | 60.55 | 214.82 |
|  | vLLM-Omni | 1 | 7.55 | 59.73 | 212.73 |
|  | Diffusers | 1 | 9.00 | 68.00 | 241.00 |
| **B200** | PyTorch | 1 | 4.62 | 32.82 | 113.85 |
|  | vLLM-Omni | 1 | 4.58 | 33.36 | 110.05 |
|  | Diffusers | 1 | 7.20 | 37.00 | 118.00 |
| **B300** | PyTorch | — |  |  |  |
|  | vLLM-Omni | 1 | 6.01 | 34.39 | 108.59 |
|  | Diffusers | 1 | 15.35 | 50.19 | 131.92 |

## Transfer generation

Transfer conditions a video-to-video generation on a structural control hint
extracted from the source video. Each control is reported separately because
the hint changes how much of the frame the model must synthesise.

| GPU | Transfer control | Engine | GPUs | 256p | 480p | 720p |
|---|---|---|:-:|---:|---:|---:|
| **RTX PRO 6000 Blackwell** | blur | Diffusers | 1 | 30.95 | 217.40 | 735.43 |
|  |  | vLLM-Omni | 1 | 40.11 | 288.60 | 1061.44 |
|  | depth | Diffusers | 1 | 36.52 | 225.34 | 745.70 |
|  |  | vLLM-Omni | 1 | 31.56 | 291.76 | 1090.33 |
|  | edge | Diffusers | 1 | 31.25 | 217.90 | 735.35 |
|  |  | vLLM-Omni | 1 | 30.64 | 287.82 | 1080.82 |
|  | multi_control | Diffusers | — |  |  |  |
|  |  | vLLM-Omni | 1 | 73.36 | 1060.11 | 3988.93 |
|  | seg | Diffusers | 1 | 32.89 | 220.10 | 738.51 |
|  |  | vLLM-Omni | 1 | 31.50 | 289.01 | 1083.64 |
|  | wsm | Diffusers | 1 | 29.84 | 176.06 | 565.32 |
|  |  | vLLM-Omni | 1 | 26.12 | 225.70 | 814.25 |
| **H20** | blur | Diffusers | 1 | 73.61 | 491.22 | 1726.61 |
|  |  | vLLM-Omni | 1 | 85.93 | 625.60 | 2280.10 |
|  | depth | Diffusers | 1 | 86.84 | 510.15 | 1753.96 |
|  |  | vLLM-Omni | 1 | 66.69 | 636.40 | 2367.37 |
|  | edge | Diffusers | 1 | 74.68 | 492.78 | 1728.26 |
|  |  | vLLM-Omni | 1 | 64.75 | 626.76 | 2321.00 |
|  | multi_control | Diffusers | — |  |  |  |
|  |  | vLLM-Omni | 8 | 60.40 |  | 2622.35 |
|  | seg | Diffusers | 1 | 77.50 | 496.77 | 1734.77 |
|  |  | vLLM-Omni | 1 | 65.84 | 623.52 | 2341.06 |
|  | wsm | Diffusers | 1 | 70.42 | 398.67 | 1319.24 |
|  |  | vLLM-Omni | 1 | 53.66 | 488.58 | 1753.23 |
| **H100 NVL** | blur | Diffusers | 1 | 26.97 | 224.61 | 775.61 |
|  |  | vLLM-Omni | 1 | 27.53 | 184.34 | 684.44 |
|  | depth | Diffusers | 1 | 31.63 | 233.03 | 610.83 |
|  |  | vLLM-Omni | 1 | 20.73 | 191.19 | 683.55 |
|  | edge | Diffusers | 1 | 34.73 | 173.04 | 602.04 |
|  |  | vLLM-Omni | 1 | 20.32 | 188.64 | 674.49 |
|  | multi_control | Diffusers | — |  |  |  |
|  |  | vLLM-Omni | 1 | 50.16 | 674.18 | 2999.04 |
|  | seg | Diffusers | 1 | 28.55 | 174.78 | 777.58 |
|  |  | vLLM-Omni | 1 | 20.87 | 190.35 | 680.37 |
|  | wsm | Diffusers | 1 | 26.17 | 139.82 | 594.21 |
|  |  | vLLM-Omni | 1 | 17.17 | 148.32 | 507.19 |
| **H100 80GB HBM3** | blur | Diffusers | 1 | 21.06 | 131.86 | 454.46 |
|  |  | vLLM-Omni | 1 | 20.40 | 133.27 | 476.11 |
|  | depth | Diffusers | 1 | 24.61 | 135.88 | 457.43 |
|  |  | vLLM-Omni | 1 | 15.75 | 135.11 | 488.18 |
|  | edge | Diffusers | 1 | 21.28 | 132.06 | 447.24 |
|  |  | vLLM-Omni | 1 | 15.33 | 133.00 | 483.75 |
|  | multi_control | Diffusers | — |  |  |  |
|  |  | vLLM-Omni | 1 | 36.94 | 477.12 | 2122.18 |
|  | seg | Diffusers | 1 | 22.54 | 133.24 | 456.55 |
|  |  | vLLM-Omni | 1 | 16.04 | 133.82 | 484.97 |
|  | wsm | Diffusers | 1 | 20.25 | 106.79 | 342.79 |
|  |  | vLLM-Omni | 1 | 13.03 | 105.20 | 365.05 |
| **H200 NVL** | blur | Diffusers | 1 | 22.82 | 152.13 | 535.83 |
|  |  | vLLM-Omni | 1 | 21.94 | 156.49 | 586.19 |
|  | depth | Diffusers | 1 | 26.80 | 156.98 | 541.71 |
|  |  | vLLM-Omni | 1 | 16.93 | 159.81 | 597.37 |
|  | edge | Diffusers | 1 | 23.06 | 152.54 | 533.87 |
|  |  | vLLM-Omni | 1 | 16.46 | 157.29 | 592.05 |
|  | multi_control | Diffusers | — |  |  |  |
|  |  | vLLM-Omni | 1 | 40.28 | 587.45 | 2792.29 |
|  | seg | Diffusers | 1 | 24.12 | 153.68 | 537.15 |
|  |  | vLLM-Omni | 1 | 17.13 | 157.81 | 594.45 |
|  | wsm | Diffusers | 1 | 21.90 | 123.08 | 407.57 |
|  |  | vLLM-Omni | 1 | 13.97 | 122.69 | 442.62 |
| **H200 141GB HBM3** | blur | Diffusers | 1 | 20.63 | 128.41 | 450.79 |
|  |  | vLLM-Omni | 1 | 20.78 | 134.95 | 492.24 |
|  | depth | Diffusers | 1 | 24.01 | 134.51 | 457.91 |
|  |  | vLLM-Omni | 1 | 16.00 | 138.25 | 497.13 |
|  | edge | Diffusers | 1 | 20.87 | 130.99 | 443.92 |
|  |  | vLLM-Omni | 1 | 15.26 | 136.41 | 492.75 |
|  | multi_control | Diffusers | — |  |  |  |
|  |  | vLLM-Omni | 1 | 38.09 | 500.91 | 2316.90 |
|  | seg | Diffusers | 1 | 21.84 | 130.12 | 453.51 |
|  |  | vLLM-Omni | 1 | 16.03 | 136.90 | 493.89 |
|  | wsm | Diffusers | 1 | 19.65 | 106.00 | 344.09 |
|  |  | vLLM-Omni | 1 | 12.95 | 109.39 | 371.24 |
| **B200** | blur | Diffusers | 1 | 12.69 | 70.78 | 226.51 |
|  |  | vLLM-Omni | 1 | 13.26 | 72.95 | 257.39 |
|  | depth | Diffusers | 1 | 14.79 | 73.23 | 229.54 |
|  |  | vLLM-Omni | 1 | 10.55 | 73.56 | 264.11 |
|  | edge | Diffusers | 1 | 12.76 | 71.04 | 223.24 |
|  |  | vLLM-Omni | 1 | 10.35 | 74.00 | 261.85 |
|  | multi_control | Diffusers | — |  |  |  |
|  |  | vLLM-Omni | 1 | 23.27 | 257.99 | 1111.03 |
|  | seg | Diffusers | 1 | 13.54 | 71.21 | 223.41 |
|  |  | vLLM-Omni | 1 | 11.10 | 73.35 | 262.72 |
|  | wsm | Diffusers | 1 | 12.16 | 59.29 | 172.60 |
|  |  | vLLM-Omni | 1 | 9.41 | 59.04 | 195.99 |
| **B300** | blur | Diffusers | 1 | 12.39 | 66.13 | 207.59 |
|  |  | vLLM-Omni | 1 | 13.85 | 73.35 | 240.39 |
|  | depth | Diffusers | 1 | 14.26 | 70.00 | 211.12 |
|  |  | vLLM-Omni | 1 | 11.11 | 71.59 | 247.16 |
|  | edge | Diffusers | 1 | 12.48 | 67.68 | 207.95 |
|  |  | vLLM-Omni | 1 | 13.05 | 72.93 | 245.16 |
|  | multi_control | Diffusers | — |  |  |  |
|  |  | vLLM-Omni | 1 | 26.79 | 243.97 | 1026.91 |
|  | seg | Diffusers | 1 | 13.43 | 68.85 | 210.33 |
|  |  | vLLM-Omni | 1 | 14.38 | 71.85 | 246.04 |
|  | wsm | Diffusers | 1 | 11.95 | 55.41 | 162.28 |
|  |  | vLLM-Omni | 1 | 10.06 | 58.97 | 185.08 |

<sub>Transfer notes:
1. Values are average generation latency in seconds; lower is better. Diffusers numbers come from PBR `#308587`, vLLM-Omni from PBR `#308574`.
2. The **GPUs** column gives the number of GPUs behind every value in that row.
3. PyTorch and NIM have no transfer coverage yet; those rows are unmeasured, not unsupported.
4. Control hints follow the vLLM-Omni `extra_params` names: `blur`, `depth`, `edge`, `multi_control`, `seg`, and `wsm`. `multi_control` combines several hints in one request.</sub>

## Action generation

Forward dynamics is reported separately for AV, camera, and robot inputs. Inverse dynamics and policy are reported for AV and robot because those are the only domain rows in PBR `#308197`; no camera row is inferred. For each domain, PyTorch is populated from that report. vLLM-Omni currently covers forward dynamics AV and inverse dynamics AV from PBR `#308481`. Camera, robot-FD, robot-ID, policy-AV, and policy-robot vLLM-Omni rows remain reserved. **Policy — DROID** is a separate checkpoint (`nvidia/Cosmos3-Nano-Policy-DROID`) and is not mixed with the policy-robot table.

### Forward Dynamics — Autonomous Vehicle (AV)

| GPU | Engine | GPUs | 256p | 480p | 720p |
|---|---|:-:|---:|---:|---:|
| **RTX PRO 6000 Blackwell** | PyTorch | 1 | 16.87 | 16.86 | 16.85 |
|  | vLLM-Omni | 1 | 2.08 | 12.94 | 36.19 |
|  | Diffusers | 1 | 2.00 | 12.00 | 12.00 |
| **H20** | PyTorch | 1 | 31.47 | 31.16 | 31.45 |
|  | vLLM-Omni | 1 | 4.82 | 31.36 | 87.55 |
|  | Diffusers | 1 | 5.00 | 28.00 | 28.20 |
| **H100 NVL** | PyTorch | 1 | 10.32 | 10.66 | 10.64 |
|  | vLLM-Omni | 1 | 1.71 | 10.58 | 28.26 |
|  | Diffusers | 1 | 2.00 | 10.00 | 10.00 |
| **H200 NVL** | PyTorch | 1 | 8.30 | 8.31 | 8.33 |
|  | vLLM-Omni | 1 | 1.43 | 8.21 | 22.78 |
|  | Diffusers | 1 | 2.00 | 8.00 | 8.00 |
| **H100 80GB HBM3** | PyTorch | 1 | 7.80 | 7.87 | 7.79 |
|  | vLLM-Omni | 1 | 1.33 | 7.74 | 20.60 |
|  | Diffusers | 1 | 2.00 | 8.00 | 8.00 |
| **H200 141GB HBM3** | PyTorch | 1 | 7.77 | 7.76 | 7.81 |
|  | vLLM-Omni | 1 | 1.33 | 7.72 | 20.65 |
|  | Diffusers | 1 | 2.00 | 8.00 | 8.00 |
| **B200** | PyTorch | 1 | 4.75 | 4.75 | 4.73 |
|  | vLLM-Omni | 1 | 0.94 | 4.93 | 12.00 |
|  | Diffusers | 1 | 2.00 | 5.00 | 5.00 |
| **B300** | PyTorch | — |  |  |  |
|  | vLLM-Omni | 1 | 1.09 | 4.86 | 11.70 |
|  | Diffusers | 1 | 20.87 | 24.49 | 26.44 |

### Forward Dynamics — Camera

| GPU | Engine | GPUs | 256p | 480p | 720p |
|---|---|:-:|---:|---:|---:|
| **RTX PRO 6000 Blackwell** | PyTorch | 1 | 16.82 | 16.82 | 16.81 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | — |  |  |  |
| **H20** | PyTorch | 1 | 31.40 | 31.10 | 31.40 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | — |  |  |  |
| **H100 NVL** | PyTorch | 1 | 10.61 | 10.32 | 10.57 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | — |  |  |  |
| **H200 NVL** | PyTorch | 1 | 8.24 | 8.26 | 8.26 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | — |  |  |  |
| **H100 80GB HBM3** | PyTorch | 1 | 7.78 | 7.83 | 7.75 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | — |  |  |  |
| **H200 141GB HBM3** | PyTorch | 1 | 7.73 | 7.75 | 7.75 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | — |  |  |  |
| **B200** | PyTorch | 1 | 4.70 | 4.74 | 4.71 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | — |  |  |  |
| **B300** | PyTorch | — |  |  |  |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | — |  |  |  |

### Forward Dynamics — Robot

| GPU | Engine | GPUs | 256p | 480p | 720p |
|---|---|:-:|---:|---:|---:|
| **RTX PRO 6000 Blackwell** | PyTorch | 1 | 3.74 | 3.73 | 3.72 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 1 | 2.00 | 3.00 | 3.00 |
| **H20** | PyTorch | 1 | 7.00 | 7.01 | 7.01 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 1 | 3.00 | 7.00 | 7.00 |
| **H100 NVL** | PyTorch | 1 | 2.49 | 2.50 | 2.55 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 1 | 2.00 | 3.00 | 3.00 |
| **H200 NVL** | PyTorch | 1 | 2.08 | 2.09 | 2.09 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 1 | 2.00 | 3.00 | 3.00 |
| **H100 80GB HBM3** | PyTorch | 1 | 2.00 | 2.01 | 1.98 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 1 | 2.00 | 3.00 | 3.00 |
| **H200 141GB HBM3** | PyTorch | 1 | 1.93 | 1.93 | 1.93 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 1 | 2.00 | 2.00 | 2.00 |
| **B200** | PyTorch | 1 | 1.28 | 1.30 | 1.31 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 1 | 2.00 | 2.00 | 2.00 |
| **B300** | PyTorch | — |  |  |  |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 1 | 22.65 | 20.75 | 18.10 |

### Inverse Dynamics — Autonomous Vehicle (AV)

| GPU | Engine | GPUs | 256p | 480p | 720p |
|---|---|:-:|---:|---:|---:|
| **RTX PRO 6000 Blackwell** | PyTorch | 1 | 16.74 | 16.75 | 16.75 |
|  | vLLM-Omni | 1 | 2.10 | 12.96 | 36.10 |
|  | Diffusers | 1 | 2.00 | 12.00 | 12.00 |
| **H20** | PyTorch | 1 | 31.22 | 30.99 | 31.22 |
|  | vLLM-Omni | 1 | 4.84 | 31.36 | 87.50 |
|  | Diffusers | 1 | 5.00 | 28.00 | 28.00 |
| **H100 NVL** | PyTorch | 1 | 10.47 | 10.45 | 10.44 |
|  | vLLM-Omni | 1 | 1.74 | 10.54 | 28.16 |
|  | Diffusers | 1 | 2.00 | 10.00 | 10.00 |
| **H200 NVL** | PyTorch | 1 | 8.12 | 8.13 | 8.10 |
|  | vLLM-Omni | 1 | 1.46 | 8.20 | 22.34 |
|  | Diffusers | 1 | 2.00 | 8.00 | 8.00 |
| **H100 80GB HBM3** | PyTorch | 1 | 7.62 | 7.63 | 7.68 |
|  | vLLM-Omni | 1 | 1.39 | 7.72 | 20.55 |
|  | Diffusers | 1 | 2.00 | 8.00 | 8.00 |
| **H200 141GB HBM3** | PyTorch | 1 | 7.51 | 7.64 | 7.51 |
|  | vLLM-Omni | 1 | 1.36 | 7.69 | 20.65 |
|  | Diffusers | 1 | 2.00 | 8.00 | 8.00 |
| **B200** | PyTorch | 1 | 4.60 | 4.63 | 4.58 |
|  | vLLM-Omni | 1 | 0.92 | 5.10 | 12.29 |
|  | Diffusers | 1 | 2.00 | 5.00 | 5.00 |
| **B300** | PyTorch | — |  |  |  |
|  | vLLM-Omni | 1 | 1.36 | 5.75 | 11.89 |
|  | Diffusers | 1 | 8.76 | 17.79 | 17.99 |

### Inverse Dynamics — Robot

| GPU | Engine | GPUs | 256p | 480p | 720p |
|---|---|:-:|---:|---:|---:|
| **RTX PRO 6000 Blackwell** | PyTorch | 1 | 3.70 | 3.70 | 3.70 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 1 | 2.00 | 3.00 | 3.00 |
| **H20** | PyTorch | 1 | 6.94 | 6.94 | 6.89 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 1 | 3.00 | 7.00 | 7.00 |
| **H100 NVL** | PyTorch | 1 | 2.53 | 2.47 | 2.54 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 1 | 2.00 | 3.00 | 3.00 |
| **H200 NVL** | PyTorch | 1 | 2.05 | 2.05 | 2.05 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 1 | 2.00 | 3.00 | 3.00 |
| **H100 80GB HBM3** | PyTorch | 1 | 1.96 | 1.96 | 1.95 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 1 | 2.00 | 3.00 | 3.00 |
| **H200 141GB HBM3** | PyTorch | 1 | 1.90 | 1.90 | 1.91 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 1 | 2.00 | 2.20 | 2.00 |
| **B200** | PyTorch | 1 | 1.28 | 1.27 | 1.28 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 1 | 2.00 | 2.00 | 2.00 |
| **B300** | PyTorch | — |  |  |  |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 1 | 9.28 | 12.40 | 10.86 |

### Policy — Autonomous Vehicle (AV)

| GPU | Engine | GPUs | 256p | 480p | 720p |
|---|---|:-:|---:|---:|---:|
| **RTX PRO 6000 Blackwell** | PyTorch | 1 | 16.88 | 16.89 | 16.91 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 1 | 2.00 | 12.00 | 12.00 |
| **H20** | PyTorch | 1 | 31.51 | 31.51 | 31.50 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 1 | 5.00 | 28.60 | 28.40 |
| **H100 NVL** | PyTorch | 1 | 10.66 | 10.65 | 10.35 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 1 | 2.00 | 10.00 | 10.00 |
| **H200 NVL** | PyTorch | 1 | 8.31 | 8.32 | 8.31 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 1 | 2.00 | 8.00 | 8.00 |
| **H100 80GB HBM3** | PyTorch | 1 | 7.82 | 7.82 | 7.80 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 1 | 2.00 | 8.00 | 8.00 |
| **H200 141GB HBM3** | PyTorch | 1 | 7.76 | 7.69 | 7.66 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 1 | 2.00 | 8.00 | 8.00 |
| **B200** | PyTorch | 1 | 4.75 | 4.77 | 4.74 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 1 | 2.00 | 5.00 | 5.00 |
| **B300** | PyTorch | — |  |  |  |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 1 | 13.11 | 18.21 | 16.49 |

### Policy — Robot

| GPU | Engine | GPUs | 256p | 480p | 720p |
|---|---|:-:|---:|---:|---:|
| **RTX PRO 6000 Blackwell** | PyTorch | 1 | 3.74 | 3.74 | 3.74 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 1 | 2.00 | 3.00 | 3.00 |
| **H20** | PyTorch | 1 | 6.95 | 7.00 | 7.01 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 1 | 3.00 | 7.00 | 7.00 |
| **H100 NVL** | PyTorch | 1 | 2.50 | 2.51 | 2.55 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 1 | 2.00 | 3.00 | 3.00 |
| **H200 NVL** | PyTorch | 1 | 2.11 | 2.09 | 2.09 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 1 | 2.00 | 3.00 | 3.00 |
| **H100 80GB HBM3** | PyTorch | 1 | 1.99 | 1.99 | 1.99 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 1 | 2.00 | 3.00 | 3.00 |
| **H200 141GB HBM3** | PyTorch | 1 | 1.93 | 1.93 | 1.94 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 1 | 2.00 | 2.00 | 2.20 |
| **B200** | PyTorch | 1 | 1.30 | 1.30 | 1.30 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 1 | 2.00 | 2.00 | 2.00 |
| **B300** | PyTorch | — |  |  |  |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 1 | 10.62 | 12.78 | 11.00 |

### Policy — DROID

These rows report **Cosmos3-Nano-Policy-DROID**, not the general Cosmos3-Nano checkpoint used in the other action tables. vLLM-Omni only supports this modality at **480p on one GPU** (multi-GPU is disabled, and resolution is fixed for the multiview wrist/exterior inputs). PyTorch and Diffusers cells are reserved.

| GPU | Engine | GPUs | 256p | 480p | 720p |
|---|---|:-:|---:|---:|---:|
| **RTX PRO 6000 Blackwell** | PyTorch | — |  |  |  |
|  | vLLM-Omni | 1 |  | 3.82 |  |
|  | Diffusers | — |  |  |  |
| **H20** | PyTorch | — |  |  |  |
|  | vLLM-Omni | 1 |  | 8.51 |  |
|  | Diffusers | — |  |  |  |
| **H100 NVL** | PyTorch | — |  |  |  |
|  | vLLM-Omni | 1 |  | 2.94 |  |
|  | Diffusers | — |  |  |  |
| **H200 NVL** | PyTorch | — |  |  |  |
|  | vLLM-Omni | 1 |  | 2.34 |  |
|  | Diffusers | — |  |  |  |
| **H100 80GB HBM3** | PyTorch | — |  |  |  |
|  | vLLM-Omni | 1 |  | 2.23 |  |
|  | Diffusers | — |  |  |  |
| **H200 141GB HBM3** | PyTorch | — |  |  |  |
|  | vLLM-Omni | 1 |  | 2.21 |  |
|  | Diffusers | — |  |  |  |
| **B200** | PyTorch | — |  |  |  |
|  | vLLM-Omni | 1 |  | 1.47 |  |
|  | Diffusers | — |  |  |  |
| **B300** | PyTorch | — |  |  |  |
|  | vLLM-Omni | 1 |  | 2.15 |  |
|  | Diffusers | — |  |  |  |

<sub>Additional-modality notes:
1. PyTorch values are average generation (sampling) latency in seconds from PBR `#308197`; lower is better.
2. PyTorch values use `CUDA_GRAPH=No` and the `latency` automatic-sharding preset.
3. vLLM-Omni audiovisual values for `t2av`/`t2vs` and `i2av`/`i2vs` are from PBR `#308195`. vLLM-Omni action values are from PBR `#308481`.
4. Action vLLM-Omni rows report one GPU because the 8-GPU Ulysses runs failed sequence-length divisibility checks.
5. Policy-DROID vLLM-Omni uses the `nvidia/Cosmos3-Nano-Policy-DROID` checkpoint at 480p on one GPU only. It is reported in its own table and is not comparable to the PyTorch policy-robot row.
6. Diffusers rows are intentionally empty reservations for future benchmark campaigns.
7. The **GPUs** column gives the number of GPUs behind every value in that row.
8. Empty cells indicate unmeasured combinations, not unsupported combinations.</sub>