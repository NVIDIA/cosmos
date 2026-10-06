# Cosmos3-Nano Generator Inference Benchmarks

[Back to the inference benchmark index](../inference_benchmarks.md)

These tables report **Cosmos3-Nano Generator** latency in seconds. Lower is better.

Each primary vision-generation row reports one GPU configuration, named in the **GPUs** column, and all three resolutions in that row come from it. The configuration is one GPU when the model fits on a single device at its highest resolution, 720p, and eight GPUs when it does not. Within one GPU's block every engine reports the same configuration, so a column can be read straight across the runtimes. Cosmos3-Nano fits on a single GPU at 720p, so its primary vision-generation rows report one GPU. Additional NIM GPU counts are listed separately.

Empty cells mean that a run has not been completed for that GPU, engine, or resolution; they do not indicate that a combination is unsupported.

## Table of Contents

- [Benchmark methodology](#benchmark-methodology)
- [Workload definitions](#workload-definitions)
- [Primary vision generation](#primary-vision-generation)
  - [Text-to-Video (t2v)](#text-to-video-t2v)
  - [Image-to-Video (i2v)](#image-to-video-i2v)
  - [Text-to-Image (t2i)](#text-to-image-t2i)
- [Additional NIM configurations](#additional-nim-configurations)
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

The primary t2v, i2v, and t2i tables cover PyTorch, vLLM-Omni, Diffusers, TensorRT-LLM, and NIM. Their vLLM-Omni values are from PBR `#307999`, **Cosmos3-Generator vLLM-Omni Inference Benchmarking (t2i, t2v, i2v)**; the PyTorch values preserve the previously published campaigns. NIM values report average generation time using FP8 latency profiles (see note 7). NIM FP8 output quality has been validated to be comparable to the BF16 baseline. OSS rows use BF16 precision, batch size 1, and matched prompts, seeds, and sampler settings where documented. Video workloads follow the standard Cosmos3 generation profile of 189 frames at 24 FPS unless a resolution tier limits frame count.

The additional audiovisual and action tables come from three internal benchmark reports. PyTorch values are from PBR `#308197`, **Cosmos3-Generator OSS Inference Benchmarking 32B and 8B (189 frames)**: average generation (sampling) latency from the native OSS path, using **CUDA Graphs disabled** and the **latency** automatic-sharding preset. vLLM-Omni values for text-to-audio-and-video (`t2av`/`t2vs`) and image-to-audio-and-video (`i2av`/`i2vs`) are from PBR `#308195`. vLLM-Omni action values are from PBR `#308481`, **Cosmos3-Generator vLLM-Omni Inference Benchmarking (action)**, measured with the `vllm/vllm-omni:cosmos3` image and the official action cookbook samples; `DIFFUSION_ATTENTION_BACKEND=TORCH_SDPA` was not set. Forward-dynamics cells are the mean of `av_forward`, `av_left`, and `av_right`; inverse-dynamics cells are the mean of `av_inverse_0` and `av_inverse_1`. That action sweep covers 1, 2, and 4 GPUs only, because its 8-GPU Ulysses runs failed a sequence-length divisibility check; those rows therefore report the single-GPU configuration. Policy-DROID is a separate checkpoint and was measured only at 480p on one GPU. That action sweep did not cover Super on H20, H100 NVL, or H100 80GB HBM3. Diffusers values come from four further reports: PBR `#308202` (all modalities) and PBR `#308451` (video-to-video) on one GPU, PBR `#308918` for 4- and 8-GPU runs, and PBR `#308587` for transfer. vLLM-Omni transfer values are from PBR `#308574`. Where PBR `#308918` offers several tensor- and context-parallel splits at the same GPU count, the fastest is published. Its 1-GPU numbers are not used, because that sweep ran a different denoising-step budget than the single-GPU reports. TensorRT-LLM values are from PBR `#308000`, **Cosmos3-Generator TRT-LLM Inference Benchmarking**, which sweeps tensor-parallel, CFG-parallel, and Ulysses-parallel splits at each GPU count; the fastest split at the count a row reports is the one published. That campaign covers six GPUs, so the TensorRT-LLM rows for H100 NVL and H200 NVL are empty, and it includes no action workloads. Values are rounded to two decimal places.

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
|  | TensorRT-LLM | 1 | 10.27 | 102.55 | 369.78 |
|  | NIM | 1 | 7.24 | 81.35 | 313.63 |
| **H20** | PyTorch | 1 | 30.57 | 257.51 | 931.39 |
|  | vLLM-Omni | 1 | 28.58 | 256.97 | 929.81 |
|  | Diffusers | 1 | 30.20 | 258.00 | 926.00 |
|  | TensorRT-LLM | 1 | 26.92 | 251.79 | 908.90 |
|  | NIM | 1 | 18.78 | 195.35 | 776.37 |
| **H100 NVL** | PyTorch | 1 | 10.03 | 84.12 | 297.27 |
|  | vLLM-Omni | 1 | 9.25 | 80.75 | 281.99 |
|  | Diffusers | 1 | 11.00 | 90.00 | 324.20 |
|  | TensorRT-LLM | — |  |  |  |
|  | NIM | 1 | 6.70 | 67.09 | 252.22 |
| **H200 NVL** | PyTorch | 1 | 8.17 | 69.79 | 244.39 |
|  | vLLM-Omni | 1 | 7.44 | 64.58 | 240.05 |
|  | Diffusers | 1 | 9.00 | 74.00 | 276.20 |
|  | TensorRT-LLM | — |  |  |  |
|  | NIM | 1 | 5.42 | 56.85 | 227.34 |
| **H100 80GB HBM3** | PyTorch | 1 | 7.61 | 59.83 | 207.78 |
|  | vLLM-Omni | 1 | 6.97 | 58.17 | 202.29 |
|  | Diffusers | 1 | 9.00 | 68.00 | 240.00 |
|  | TensorRT-LLM | 1 | 6.50 | 60.10 | 223.51 |
|  | NIM | 1 | 5.35 | 46.08 | 173.20 |
| **H200 141GB HBM3** | PyTorch | 1 | 7.53 | 60.18 | 214.28 |
|  | vLLM-Omni | 1 | 6.79 | 58.14 | 208.36 |
|  | Diffusers | 1 | 9.00 | 67.00 | 239.60 |
|  | TensorRT-LLM | 1 | 6.42 | 60.57 | 225.06 |
|  | NIM | 1 | 5.02 | 46.93 | 177.23 |
| **B200** | PyTorch | 1 | 4.56 | 33.20 | 114.85 |
|  | vLLM-Omni | 1 | 4.03 | 32.04 | 107.79 |
|  | Diffusers | 1 | 7.00 | 36.80 | 117.00 |
|  | TensorRT-LLM | 1 | 3.43 | 28.28 | 100.02 |
|  | NIM | 1 | 2.94 | 25.03 | 91.62 |
| **B300** | PyTorch | — |  |  |  |
|  | vLLM-Omni | 1 | 4.46 | 32.18 | 102.10 |
|  | Diffusers | 1 | 6.40 | 39.32 | 121.74 |
|  | TensorRT-LLM | 1 | 3.31 | 26.52 | 91.15 |
|  | NIM | 1 | 2.86 | 23.67 | 85.52 |

### Image-to-Video (i2v)

| GPU | Engine | GPUs | 256p | 480p | 720p |
|---|---|:-:|---:|---:|---:|
| **RTX PRO 6000 Blackwell** | PyTorch | 1 | 14.08 | 182.14 | 788.80 |
|  | vLLM-Omni | 1 | 11.04 | 107.77 | 376.44 |
|  | Diffusers | 1 | 12.00 | 112.00 | 397.00 |
|  | TensorRT-LLM | 1 | 10.77 | 105.03 | 374.89 |
|  | NIM | 1 | 7.27 | 81.50 | 313.64 |
| **H20** | PyTorch | 1 | 31.36 | 257.10 | 933.07 |
|  | vLLM-Omni | 1 | 29.50 | 261.56 | 940.16 |
|  | Diffusers | 1 | 31.00 | 258.00 | 925.00 |
|  | TensorRT-LLM | 1 | 28.07 | 254.79 | 915.71 |
|  | NIM | 1 | 18.69 | 195.87 | 776.48 |
| **H100 NVL** | PyTorch | 1 | 10.19 | 84.50 | 298.57 |
|  | vLLM-Omni | 1 | 9.62 | 82.61 | 286.19 |
|  | Diffusers | 1 | 11.00 | 91.00 | 325.20 |
|  | TensorRT-LLM | — |  |  |  |
|  | NIM | 1 | 6.76 | 67.50 | 251.21 |
| **H200 NVL** | PyTorch | 1 | 8.27 | 69.99 | 246.62 |
|  | vLLM-Omni | 1 | 7.83 | 66.39 | 243.52 |
|  | Diffusers | 1 | 9.00 | 74.00 | 275.20 |
|  | TensorRT-LLM | — |  |  |  |
|  | NIM | 1 | 5.43 | 57.02 | 226.12 |
| **H100 80GB HBM3** | PyTorch | 1 | 7.64 | 59.95 | 207.87 |
|  | vLLM-Omni | 1 | 7.37 | 59.77 | 205.97 |
|  | Diffusers | 1 | 9.00 | 68.00 | 239.80 |
|  | TensorRT-LLM | 1 | 6.90 | 63.24 | 226.15 |
|  | NIM | 1 | 5.37 | 46.34 | 173.14 |
| **H200 141GB HBM3** | PyTorch | 1 | 7.65 | 60.51 | 214.80 |
|  | vLLM-Omni | 1 | 7.28 | 59.64 | 209.65 |
|  | Diffusers | 1 | 9.00 | 67.20 | 240.00 |
|  | TensorRT-LLM | 1 | 6.74 | 63.54 | 231.15 |
|  | NIM | 1 | 5.07 | 46.95 | 177.57 |
| **B200** | PyTorch | 1 | 4.60 | 33.08 | 113.90 |
|  | vLLM-Omni | 1 | 4.33 | 33.09 | 110.08 |
|  | Diffusers | 1 | 7.00 | 37.00 | 116.00 |
|  | TensorRT-LLM | 1 | 3.70 | 29.33 | 102.35 |
|  | NIM | 1 | 2.98 | 25.59 | 91.65 |
| **B300** | PyTorch | — |  |  |  |
|  | vLLM-Omni | 1 | 5.61 | 33.45 | 104.75 |
|  | Diffusers | 1 | 13.40 | 49.59 | 131.25 |
|  | TensorRT-LLM | 1 | 3.70 | 27.73 | 93.88 |
|  | NIM | 1 | 2.88 | 23.74 | 85.50 |

### Text-to-Image (t2i)

| GPU | Engine | GPUs | 256p | 480p | 720p |
|---|---|:-:|---:|---:|---:|
| **RTX PRO 6000 Blackwell** | PyTorch | 1 | 2.99 | 4.51 | 7.12 |
|  | vLLM-Omni | 1 | 1.59 | 2.87 | 4.99 |
|  | Diffusers | 1 | 2.00 | 4.00 | 5.00 |
|  | TensorRT-LLM | 1 | 1.18 | 2.35 | 4.50 |
|  | NIM | 1 | 1.01 | 1.33 | 2.25 |
| **H20** | PyTorch | 1 | 3.06 | 6.51 | 12.31 |
|  | vLLM-Omni | 1 | 1.73 | 4.92 | 10.73 |
|  | Diffusers | 1 | 3.00 | 6.00 | 10.00 |
|  | TensorRT-LLM | 1 | 1.94 | 5.55 | 11.56 |
|  | NIM | 1 | 1.18 | 2.54 | 4.70 |
| **H100 NVL** | PyTorch | 1 | 2.77 | 2.83 | 4.21 |
|  | vLLM-Omni | 1 | 1.55 | 1.92 | 3.43 |
|  | Diffusers | 1 | 3.00 | 3.00 | 4.00 |
|  | TensorRT-LLM | — |  |  |  |
|  | NIM | 1 | 1.14 | 1.15 | 1.72 |
| **H200 NVL** | PyTorch | 1 | 2.75 | 2.85 | 3.58 |
|  | vLLM-Omni | 1 | 1.53 | 1.58 | 2.81 |
|  | Diffusers | 1 | 3.00 | 3.00 | 4.00 |
|  | TensorRT-LLM | — |  |  |  |
|  | NIM | 1 | 1.12 | 1.16 | 1.46 |
| **H100 80GB HBM3** | PyTorch | 1 | 3.01 | 3.01 | 3.45 |
|  | vLLM-Omni | 1 | 1.61 | 1.53 | 2.61 |
|  | Diffusers | 1 | 3.00 | 3.00 | 4.00 |
|  | TensorRT-LLM | 1 | 1.02 | 1.30 | 2.63 |
|  | NIM | 1 | 1.10 | 1.24 | 1.51 |
| **H200 141GB HBM3** | PyTorch | 1 | 2.96 | 3.04 | 3.28 |
|  | vLLM-Omni | 1 | 1.57 | 1.52 | 2.60 |
|  | Diffusers | 1 | 3.00 | 3.00 | 4.00 |
|  | TensorRT-LLM | 1 | 1.05 | 1.30 | 2.60 |
|  | NIM | 1 | 1.17 | 1.18 | 1.42 |
| **B200** | PyTorch | 1 | 2.68 | 2.75 | 2.87 |
|  | vLLM-Omni | 1 | 1.49 | 1.20 | 1.76 |
|  | Diffusers | 1 | 3.00 | 3.20 | 3.00 |
|  | TensorRT-LLM | 1 | 0.86 | 0.88 | 1.40 |
|  | NIM | 1 | 0.99 | 1.06 | 1.02 |
| **B300** | PyTorch | — |  |  |  |
|  | vLLM-Omni | 1 | 1.97 | 1.81 | 2.34 |
|  | Diffusers | 1 | 3.86 | 3.86 | 3.99 |
|  | TensorRT-LLM | 1 | 1.36 | 1.30 | 1.40 |
|  | NIM | 1 | 1.45 | 1.50 | 1.48 |

<sub>Notes:
1. OSS times measured on identical workloads (same seed, sampler settings, prompt).
2. Multi-GPU configurations use tensor parallelism, except TensorRT-LLM, which may also split across CFG and Ulysses dimensions.
3. vLLM-Omni numbers are for the upcoming public release in the vLLM-Omni repo and are subject to change before GA; the H100 NVL values in particular are pre-release.
4. Diffusers numbers use the HuggingFace `diffusers` integration without custom CUDA graphs. Single-GPU values come from PBR `#308202` and PBR `#308451`; multi-GPU values from PBR `#308918`.
5. PyTorch numbers report average generation (sampling) time from OSS inference benchmarking.
6. At 256p, multi-GPU configurations on B300 may underperform single-GPU because of small-workload tensor-parallel overhead, so single-GPU is the recommended deployment at this resolution.
7. NIM numbers use FP8 latency profiles with offload disabled and report `Avg. Generation Time (s)`, excluding request overhead and MP4 encoding. Runs use concurrency 1, three measured requests, and 189 video frames or one image frame.</sub>

## Additional NIM configurations

Average generation time in seconds for GPU counts not shown in the primary tables above. These runs use the same FP8 latency profiles with offload disabled (see note 7).

| GPU | Modality | GPUs | 256p | 480p | 720p |
|---|---|:-:|---:|---:|---:|
| RTX PRO 6000 Blackwell | t2v | 4 | 3.45 | 25.41 | 90.81 |
| RTX PRO 6000 Blackwell | t2v | 8 | 2.83 | 17.04 | 49.33 |
| H20 | t2v | 4 | 5.88 | 53.77 | 203.35 |
| H20 | t2v | 8 | 3.84 | 28.51 | 105.11 |
| H100 NVL | t2v | 4 | 2.50 | 24.35 | 102.48 |
| H100 NVL | t2v | 8 | 2.46 | 12.96 | 48.50 |
| H200 NVL | t2v | 4 | 2.62 | 16.95 | 62.77 |
| H200 NVL | t2v | 8 | 2.14 | 10.30 | 33.53 |
| H100 80GB HBM3 | t2v | 4 | 2.24 | 13.78 | 48.35 |
| H100 80GB HBM3 | t2v | 8 | 1.81 | 7.77 | 26.14 |
| H200 141GB HBM3 | t2v | 4 | 2.12 | 13.65 | 48.75 |
| H200 141GB HBM3 | t2v | 8 | 1.85 | 7.56 | 25.78 |
| B200 | t2v | 4 | 1.57 | 7.54 | 25.97 |
| B200 | t2v | 8 | 1.56 | 4.41 | 14.50 |
| B300 | t2v | 4 | 2.48 | 7.36 | 24.41 |
| B300 | t2v | 8 | 2.50 | 4.82 | 13.80 |
| RTX PRO 6000 Blackwell | i2v | 4 | 3.50 | 25.54 | 90.98 |
| RTX PRO 6000 Blackwell | i2v | 8 | 2.98 | 16.03 | 49.97 |
| H20 | i2v | 4 | 5.97 | 53.73 | 203.76 |
| H20 | i2v | 8 | 3.96 | 28.66 | 105.06 |
| H100 NVL | i2v | 4 | 2.61 | 24.29 | 99.98 |
| H100 NVL | i2v | 8 | 2.53 | 13.18 | 48.97 |
| H200 NVL | i2v | 4 | 2.72 | 17.07 | 62.81 |
| H200 NVL | i2v | 8 | 2.32 | 10.58 | 33.63 |
| H100 80GB HBM3 | i2v | 4 | 2.29 | 13.88 | 48.33 |
| H100 80GB HBM3 | i2v | 8 | 1.93 | 7.92 | 26.37 |
| H200 141GB HBM3 | i2v | 4 | 2.19 | 13.72 | 48.66 |
| H200 141GB HBM3 | i2v | 8 | 1.95 | 7.69 | 26.00 |
| B200 | i2v | 4 | 1.62 | 7.62 | 26.13 |
| B200 | i2v | 8 | 1.71 | 4.59 | 14.50 |
| B300 | i2v | 4 | 2.48 | 7.45 | 24.56 |
| B300 | i2v | 8 | 2.62 | 4.97 | 14.20 |
| RTX PRO 6000 Blackwell | t2i | 4 | 0.90 | 0.92 | 1.13 |
| RTX PRO 6000 Blackwell | t2i | 8 | 0.89 | 0.94 | 1.00 |
| H20 | t2i | 4 | 1.06 | 1.13 | 1.61 |
| H20 | t2i | 8 | 1.09 | 2.98 | 1.18 |
| H100 NVL | t2i | 4 | 0.96 | 0.98 | 1.00 |
| H100 NVL | t2i | 8 | 0.97 | 2.72 | 1.00 |
| H200 NVL | t2i | 4 | 0.92 | 0.98 | 0.98 |
| H200 NVL | t2i | 8 | 1.03 | 2.81 | 1.04 |
| H100 80GB HBM3 | t2i | 4 | 0.98 | 1.00 | 1.08 |
| H100 80GB HBM3 | t2i | 8 | 1.02 | 2.93 | 1.07 |
| H200 141GB HBM3 | t2i | 4 | 1.04 | 0.99 | 1.06 |
| H200 141GB HBM3 | t2i | 8 | 1.09 | 2.93 | 1.10 |
| B200 | t2i | 4 | 0.87 | 0.91 | 0.89 |
| B200 | t2i | 8 | 0.90 | 1.02 | 0.91 |
| B300 | t2i | 4 | 1.36 | 1.40 | 1.37 |
| B300 | t2i | 8 | 1.38 | 1.45 | 1.47 |

## Additional audiovisual generation

### Video-to-Video (v2v)

A text prompt and source video condition a generated continuation or transformation.

| GPU | Engine | GPUs | 256p | 480p | 720p |
|---|---|:-:|---:|---:|---:|
| **RTX PRO 6000 Blackwell** | PyTorch | 1 | 13.73 | 180.54 | 785.41 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 1 | 24.19 | 135.85 | 432.18 |
|  | TensorRT-LLM | 1 | 10.36 | 102.50 | 370.24 |
| **H20** | PyTorch | 1 | 30.44 | 257.79 | 930.96 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 1 | 44.88 | 287.55 | 974.75 |
|  | TensorRT-LLM | 1 | 27.10 | 250.57 | 908.07 |
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
| **H200 141GB HBM3** | PyTorch | 1 | 7.48 | 60.23 | 214.03 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 1 | 18.93 | 85.52 | 266.84 |
|  | TensorRT-LLM | 1 | 6.59 | 61.40 | 229.48 |
| **B200** | PyTorch | 1 | 4.52 | 32.53 | 114.16 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 1 | 16.93 | 52.32 | 137.54 |
|  | TensorRT-LLM | 1 | 3.54 | 28.37 | 100.29 |
| **B300** | PyTorch | — |  |  |  |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 1 | 16.14 | 50.74 | 131.11 |
|  | TensorRT-LLM | 1 | 3.44 | 26.63 | 91.40 |

### Text-to-Audio-and-Video (t2av)

A text prompt produces synchronized video and sound.

| GPU | Engine | GPUs | 256p | 480p | 720p |
|---|---|:-:|---:|---:|---:|
| **RTX PRO 6000 Blackwell** | PyTorch | 1 | 14.13 | 180.98 | 794.74 |
|  | vLLM-Omni | 1 | 12.25 | 106.02 | 373.10 |
|  | Diffusers | 1 | 13.00 | 113.00 | 393.00 |
|  | TensorRT-LLM | 1 | 11.35 | 103.28 | 374.75 |
| **H20** | PyTorch | 1 | 30.82 | 256.41 | 922.35 |
|  | vLLM-Omni | 1 | 29.43 | 260.84 | 937.93 |
|  | Diffusers | 1 | 32.00 | 263.00 | 932.60 |
|  | TensorRT-LLM | 1 | 29.38 | 254.88 | 913.33 |
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
| **H200 141GB HBM3** | PyTorch | 1 | 7.55 | 60.17 | 211.59 |
|  | vLLM-Omni | 1 | 7.13 | 58.27 | 209.29 |
|  | Diffusers | 1 | 9.00 | 69.00 | 237.00 |
|  | TensorRT-LLM | 1 | 6.93 | 60.77 | 233.06 |
| **B200** | PyTorch | 1 | 4.63 | 32.60 | 113.89 |
|  | vLLM-Omni | 1 | 4.29 | 32.20 | 107.74 |
|  | Diffusers | 1 | 7.00 | 37.00 | 117.00 |
|  | TensorRT-LLM | 1 | 3.77 | 28.64 | 101.28 |
| **B300** | PyTorch | — |  |  |  |
|  | vLLM-Omni | 1 | 5.59 | 33.15 | 104.92 |
|  | Diffusers | 1 | 6.46 | 39.66 | 122.33 |
|  | TensorRT-LLM | 1 | 3.65 | 26.83 | 91.85 |

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
| **H20** | PyTorch | 1 | 31.34 | 259.06 | 924.39 |
|  | vLLM-Omni | 1 | 30.36 | 265.37 | 948.09 |
|  | Diffusers | 1 | 32.00 | 263.00 | 934.80 |
|  | TensorRT-LLM | 1 | 29.95 | 259.43 | 924.85 |
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
| **H200 141GB HBM3** | PyTorch | 1 | 7.66 | 60.55 | 214.82 |
|  | vLLM-Omni | 1 | 7.55 | 59.73 | 212.73 |
|  | Diffusers | 1 | 9.00 | 68.00 | 241.00 |
|  | TensorRT-LLM | 1 | 7.37 | 62.87 | 231.61 |
| **B200** | PyTorch | 1 | 4.62 | 32.82 | 113.85 |
|  | vLLM-Omni | 1 | 4.58 | 33.36 | 110.05 |
|  | Diffusers | 1 | 7.20 | 37.00 | 118.00 |
|  | TensorRT-LLM | 1 | 4.04 | 29.77 | 103.28 |
| **B300** | PyTorch | — |  |  |  |
|  | vLLM-Omni | 1 | 6.01 | 34.39 | 108.59 |
|  | Diffusers | 1 | 15.35 | 50.19 | 131.92 |
|  | TensorRT-LLM | 1 | 4.03 | 28.10 | 94.57 |

## Transfer generation

Transfer conditions a video-to-video generation on a structural control hint
extracted from the source video. Each control is reported separately because
the hint changes how much of the frame the model must synthesise.

| GPU | Transfer control | Engine | GPUs | 256p | 480p | 720p |
|---|---|---|:-:|---:|---:|---:|
| **RTX PRO 6000 Blackwell** | blur | PyTorch | 1 |  |  |  |
|  |  | vLLM-Omni | 1 | 40.11 | 288.60 | 1061.44 |
|  |  | Diffusers | 1 | 30.95 | 217.40 | 735.43 |
|  |  | TensorRT-LLM | 1 | 25.62 | 249.50 | 935.86 |
|  |  | NIM | 1 |  |  |  |
|  | depth | PyTorch | 1 |  |  |  |
|  |  | vLLM-Omni | 1 | 31.56 | 291.76 | 1090.33 |
|  |  | Diffusers | 1 | 36.52 | 225.34 | 745.70 |
|  |  | TensorRT-LLM | 1 | 26.13 | 253.94 | 942.69 |
|  |  | NIM | 1 |  |  |  |
|  | edge | PyTorch | 1 |  |  |  |
|  |  | vLLM-Omni | 1 | 30.64 | 287.82 | 1080.82 |
|  |  | Diffusers | 1 | 31.25 | 217.90 | 735.35 |
|  |  | TensorRT-LLM | 1 | 25.67 | 250.54 | 938.79 |
|  |  | NIM | 1 |  |  |  |
|  | multi_control | PyTorch | 1 |  |  |  |
|  |  | vLLM-Omni | 1 | 73.36 | 1060.11 | 3988.93 |
|  |  | Diffusers | 1 |  |  |  |
|  |  | TensorRT-LLM | 1 | 35.07 | 423.61 | 1745.86 |
|  |  | NIM | 1 |  |  |  |
|  | seg | PyTorch | 1 |  |  |  |
|  |  | vLLM-Omni | 1 | 31.50 | 289.01 | 1083.64 |
|  |  | Diffusers | 1 | 32.89 | 220.10 | 738.51 |
|  |  | TensorRT-LLM | 1 | 25.85 | 250.47 | 937.18 |
|  |  | NIM | 1 |  |  |  |
|  | wsm | PyTorch | 1 |  |  |  |
|  |  | vLLM-Omni | 1 | 26.12 | 225.70 | 814.25 |
|  |  | Diffusers | 1 | 29.84 | 176.06 | 565.32 |
|  |  | TensorRT-LLM | 1 | 12.95 | 114.91 | 407.62 |
|  |  | NIM | 1 |  |  |  |
| **H20** | blur | PyTorch | 1 |  |  |  |
|  |  | vLLM-Omni | 1 | 85.93 | 625.60 | 2280.10 |
|  |  | Diffusers | 1 | 73.61 | 491.22 | 1726.61 |
|  |  | TensorRT-LLM | 1 | 63.10 | 614.14 | 2308.16 |
|  |  | NIM | 1 |  |  |  |
|  | depth | PyTorch | 1 |  |  |  |
|  |  | vLLM-Omni | 1 | 66.69 | 636.40 | 2367.37 |
|  |  | Diffusers | 1 | 86.84 | 510.15 | 1753.96 |
|  |  | TensorRT-LLM | 1 | 64.44 | 622.79 | 2328.65 |
|  |  | NIM | 1 |  |  |  |
|  | edge | PyTorch | 1 |  |  |  |
|  |  | vLLM-Omni | 1 | 64.75 | 626.76 | 2321.00 |
|  |  | Diffusers | 1 | 74.68 | 492.78 | 1728.26 |
|  |  | TensorRT-LLM | 1 | 63.92 | 615.09 | 2310.55 |
|  |  | NIM | 1 |  |  |  |
|  | multi_control | PyTorch | 8 |  |  |  |
|  |  | vLLM-Omni | 8 | 60.40 |  | 2622.35 |
|  |  | Diffusers | 8 |  |  |  |
|  |  | TensorRT-LLM | 8 | 0.64 | 0.54 | 0.51 |
|  |  | NIM | 8 |  |  |  |
|  | seg | PyTorch | 1 |  |  |  |
|  |  | vLLM-Omni | 1 | 65.84 | 623.52 | 2341.06 |
|  |  | Diffusers | 1 | 77.50 | 496.77 | 1734.77 |
|  |  | TensorRT-LLM | 1 | 63.56 | 616.74 | 2314.38 |
|  |  | NIM | 1 |  |  |  |
|  | wsm | PyTorch | 1 |  |  |  |
|  |  | vLLM-Omni | 1 | 53.66 | 488.58 | 1753.23 |
|  |  | Diffusers | 1 | 70.42 | 398.67 | 1319.24 |
|  |  | TensorRT-LLM | 1 | 32.09 | 284.19 | 1001.59 |
|  |  | NIM | 1 |  |  |  |
| **H100 NVL** | blur | PyTorch | 1 |  |  |  |
|  |  | vLLM-Omni | 1 | 27.53 | 184.34 | 684.44 |
|  |  | Diffusers | 1 | 26.97 | 224.61 | 775.61 |
|  |  | TensorRT-LLM | 1 |  |  |  |
|  |  | NIM | 1 |  |  |  |
|  | depth | PyTorch | 1 |  |  |  |
|  |  | vLLM-Omni | 1 | 20.73 | 191.19 | 683.55 |
|  |  | Diffusers | 1 | 31.63 | 233.03 | 610.83 |
|  |  | TensorRT-LLM | 1 |  |  |  |
|  |  | NIM | 1 |  |  |  |
|  | edge | PyTorch | 1 |  |  |  |
|  |  | vLLM-Omni | 1 | 20.32 | 188.64 | 674.49 |
|  |  | Diffusers | 1 | 34.73 | 173.04 | 602.04 |
|  |  | TensorRT-LLM | 1 |  |  |  |
|  |  | NIM | 1 |  |  |  |
|  | multi_control | PyTorch | 1 |  |  |  |
|  |  | vLLM-Omni | 1 | 50.16 | 674.18 | 2999.04 |
|  |  | Diffusers | 1 |  |  |  |
|  |  | TensorRT-LLM | 1 |  |  |  |
|  |  | NIM | 1 |  |  |  |
|  | seg | PyTorch | 1 |  |  |  |
|  |  | vLLM-Omni | 1 | 20.87 | 190.35 | 680.37 |
|  |  | Diffusers | 1 | 28.55 | 174.78 | 777.58 |
|  |  | TensorRT-LLM | 1 |  |  |  |
|  |  | NIM | 1 |  |  |  |
|  | wsm | PyTorch | 1 |  |  |  |
|  |  | vLLM-Omni | 1 | 17.17 | 148.32 | 507.19 |
|  |  | Diffusers | 1 | 26.17 | 139.82 | 594.21 |
|  |  | TensorRT-LLM | 1 |  |  |  |
|  |  | NIM | 1 |  |  |  |
| **H100 80GB HBM3** | blur | PyTorch | 1 |  |  |  |
|  |  | vLLM-Omni | 1 | 20.40 | 133.27 | 476.11 |
|  |  | Diffusers | 1 | 21.06 | 131.86 | 454.46 |
|  |  | TensorRT-LLM | 1 | 15.20 | 149.57 | 575.59 |
|  |  | NIM | 1 |  |  |  |
|  | depth | PyTorch | 1 |  |  |  |
|  |  | vLLM-Omni | 1 | 15.75 | 135.11 | 488.18 |
|  |  | Diffusers | 1 | 24.61 | 135.88 | 457.43 |
|  |  | TensorRT-LLM | 1 | 15.45 | 149.96 | 580.66 |
|  |  | NIM | 1 |  |  |  |
|  | edge | PyTorch | 1 |  |  |  |
|  |  | vLLM-Omni | 1 | 15.33 | 133.00 | 483.75 |
|  |  | Diffusers | 1 | 21.28 | 132.06 | 447.24 |
|  |  | TensorRT-LLM | 1 | 15.28 | 148.99 | 572.91 |
|  |  | NIM | 1 |  |  |  |
|  | multi_control | PyTorch | 1 |  |  |  |
|  |  | vLLM-Omni | 1 | 36.94 | 477.12 | 2122.18 |
|  |  | Diffusers | 1 |  |  |  |
|  |  | TensorRT-LLM | 1 | 21.50 | 254.29 | 1072.98 |
|  |  | NIM | 1 |  |  |  |
|  | seg | PyTorch | 1 |  |  |  |
|  |  | vLLM-Omni | 1 | 16.04 | 133.82 | 484.97 |
|  |  | Diffusers | 1 | 22.54 | 133.24 | 456.55 |
|  |  | TensorRT-LLM | 1 | 15.42 | 145.46 | 576.57 |
|  |  | NIM | 1 |  |  |  |
|  | wsm | PyTorch | 1 |  |  |  |
|  |  | vLLM-Omni | 1 | 13.03 | 105.20 | 365.05 |
|  |  | Diffusers | 1 | 20.25 | 106.79 | 342.79 |
|  |  | TensorRT-LLM | 1 | 7.96 | 68.97 | 248.27 |
|  |  | NIM | 1 |  |  |  |
| **H200 NVL** | blur | PyTorch | 1 |  |  |  |
|  |  | vLLM-Omni | 1 | 21.94 | 156.49 | 586.19 |
|  |  | Diffusers | 1 | 22.82 | 152.13 | 535.83 |
|  |  | TensorRT-LLM | 1 |  |  |  |
|  |  | NIM | 1 |  |  |  |
|  | depth | PyTorch | 1 |  |  |  |
|  |  | vLLM-Omni | 1 | 16.93 | 159.81 | 597.37 |
|  |  | Diffusers | 1 | 26.80 | 156.98 | 541.71 |
|  |  | TensorRT-LLM | 1 |  |  |  |
|  |  | NIM | 1 |  |  |  |
|  | edge | PyTorch | 1 |  |  |  |
|  |  | vLLM-Omni | 1 | 16.46 | 157.29 | 592.05 |
|  |  | Diffusers | 1 | 23.06 | 152.54 | 533.87 |
|  |  | TensorRT-LLM | 1 |  |  |  |
|  |  | NIM | 1 |  |  |  |
|  | multi_control | PyTorch | 1 |  |  |  |
|  |  | vLLM-Omni | 1 | 40.28 | 587.45 | 2792.29 |
|  |  | Diffusers | 1 |  |  |  |
|  |  | TensorRT-LLM | 1 |  |  |  |
|  |  | NIM | 1 |  |  |  |
|  | seg | PyTorch | 1 |  |  |  |
|  |  | vLLM-Omni | 1 | 17.13 | 157.81 | 594.45 |
|  |  | Diffusers | 1 | 24.12 | 153.68 | 537.15 |
|  |  | TensorRT-LLM | 1 |  |  |  |
|  |  | NIM | 1 |  |  |  |
|  | wsm | PyTorch | 1 |  |  |  |
|  |  | vLLM-Omni | 1 | 13.97 | 122.69 | 442.62 |
|  |  | Diffusers | 1 | 21.90 | 123.08 | 407.57 |
|  |  | TensorRT-LLM | 1 |  |  |  |
|  |  | NIM | 1 |  |  |  |
| **H200 141GB HBM3** | blur | PyTorch | 1 |  |  |  |
|  |  | vLLM-Omni | 1 | 20.78 | 134.95 | 492.24 |
|  |  | Diffusers | 1 | 20.63 | 128.41 | 450.79 |
|  |  | TensorRT-LLM | 1 | 15.11 | 149.55 | 580.80 |
|  |  | NIM | 1 |  |  |  |
|  | depth | PyTorch | 1 |  |  |  |
|  |  | vLLM-Omni | 1 | 16.00 | 138.25 | 497.13 |
|  |  | Diffusers | 1 | 24.01 | 134.51 | 457.91 |
|  |  | TensorRT-LLM | 1 | 15.42 | 152.64 | 587.03 |
|  |  | NIM | 1 |  |  |  |
|  | edge | PyTorch | 1 |  |  |  |
|  |  | vLLM-Omni | 1 | 15.26 | 136.41 | 492.75 |
|  |  | Diffusers | 1 | 20.87 | 130.99 | 443.92 |
|  |  | TensorRT-LLM | 1 | 14.95 | 148.90 | 586.42 |
|  |  | NIM | 1 |  |  |  |
|  | multi_control | PyTorch | 1 |  |  |  |
|  |  | vLLM-Omni | 1 | 38.09 | 500.91 | 2316.90 |
|  |  | Diffusers | 1 |  |  |  |
|  |  | TensorRT-LLM | 1 | 21.65 | 259.30 | 1095.79 |
|  |  | NIM | 1 |  |  |  |
|  | seg | PyTorch | 1 |  |  |  |
|  |  | vLLM-Omni | 1 | 16.03 | 136.90 | 493.89 |
|  |  | Diffusers | 1 | 21.84 | 130.12 | 453.51 |
|  |  | TensorRT-LLM | 1 | 15.31 | 148.26 | 583.48 |
|  |  | NIM | 1 |  |  |  |
|  | wsm | PyTorch | 1 |  |  |  |
|  |  | vLLM-Omni | 1 | 12.95 | 109.39 | 371.24 |
|  |  | Diffusers | 1 | 19.65 | 106.00 | 344.09 |
|  |  | TensorRT-LLM | 1 | 7.90 | 68.71 | 252.43 |
|  |  | NIM | 1 |  |  |  |
| **B200** | blur | PyTorch | 1 |  |  |  |
|  |  | vLLM-Omni | 1 | 13.26 | 72.95 | 257.39 |
|  |  | Diffusers | 1 | 12.69 | 70.78 | 226.51 |
|  |  | TensorRT-LLM | 1 | 7.73 | 68.21 | 252.69 |
|  |  | NIM | 1 |  |  |  |
|  | depth | PyTorch | 1 |  |  |  |
|  |  | vLLM-Omni | 1 | 10.55 | 73.56 | 264.11 |
|  |  | Diffusers | 1 | 14.79 | 73.23 | 229.54 |
|  |  | TensorRT-LLM | 1 | 7.92 | 68.77 | 248.98 |
|  |  | NIM | 1 |  |  |  |
|  | edge | PyTorch | 1 |  |  |  |
|  |  | vLLM-Omni | 1 | 10.35 | 74.00 | 261.85 |
|  |  | Diffusers | 1 | 12.76 | 71.04 | 223.24 |
|  |  | TensorRT-LLM | 1 | 7.78 | 68.29 | 248.51 |
|  |  | NIM | 1 |  |  |  |
|  | multi_control | PyTorch | 1 |  |  |  |
|  |  | vLLM-Omni | 1 | 23.27 | 257.99 | 1111.03 |
|  |  | Diffusers | 1 |  |  |  |
|  |  | TensorRT-LLM | 1 | 10.72 | 112.58 | 455.44 |
|  |  | NIM | 1 |  |  |  |
|  | seg | PyTorch | 1 |  |  |  |
|  |  | vLLM-Omni | 1 | 11.10 | 73.35 | 262.72 |
|  |  | Diffusers | 1 | 13.54 | 71.21 | 223.41 |
|  |  | TensorRT-LLM | 1 | 7.99 | 68.45 | 248.75 |
|  |  | NIM | 1 |  |  |  |
|  | wsm | PyTorch | 1 |  |  |  |
|  |  | vLLM-Omni | 1 | 9.41 | 59.04 | 195.99 |
|  |  | Diffusers | 1 | 12.16 | 59.29 | 172.60 |
|  |  | TensorRT-LLM | 1 | 4.25 | 32.24 | 108.75 |
|  |  | NIM | 1 |  |  |  |
| **B300** | blur | PyTorch | 1 |  |  |  |
|  |  | vLLM-Omni | 1 | 13.85 | 73.35 | 240.39 |
|  |  | Diffusers | 1 | 12.39 | 66.13 | 207.59 |
|  |  | TensorRT-LLM | 1 | 7.67 | 62.95 | 228.34 |
|  |  | NIM | 1 |  |  |  |
|  | depth | PyTorch | 1 |  |  |  |
|  |  | vLLM-Omni | 1 | 11.11 | 71.59 | 247.16 |
|  |  | Diffusers | 1 | 14.26 | 70.00 | 211.12 |
|  |  | TensorRT-LLM | 1 | 7.67 | 63.68 | 226.26 |
|  |  | NIM | 1 |  |  |  |
|  | edge | PyTorch | 1 |  |  |  |
|  |  | vLLM-Omni | 1 | 13.05 | 72.93 | 245.16 |
|  |  | Diffusers | 1 | 12.48 | 67.68 | 207.95 |
|  |  | TensorRT-LLM | 1 | 7.58 | 63.10 | 228.36 |
|  |  | NIM | 1 |  |  |  |
|  | multi_control | PyTorch | 1 |  |  |  |
|  |  | vLLM-Omni | 1 | 26.79 | 243.97 | 1026.91 |
|  |  | Diffusers | 1 |  |  |  |
|  |  | TensorRT-LLM | 1 | 10.59 | 105.33 | 414.58 |
|  |  | NIM | 1 |  |  |  |
|  | seg | PyTorch | 1 |  |  |  |
|  |  | vLLM-Omni | 1 | 14.38 | 71.85 | 246.04 |
|  |  | Diffusers | 1 | 13.43 | 68.85 | 210.33 |
|  |  | TensorRT-LLM | 1 | 7.64 | 64.13 | 228.37 |
|  |  | NIM | 1 |  |  |  |
|  | wsm | PyTorch | 1 |  |  |  |
|  |  | vLLM-Omni | 1 | 10.06 | 58.97 | 185.08 |
|  |  | Diffusers | 1 | 11.95 | 55.41 | 162.28 |
|  |  | TensorRT-LLM | 1 | 4.23 | 30.44 | 101.17 |
|  |  | NIM | 1 |  |  |  |

<sub>Transfer notes:
1. Values are average generation latency in seconds; lower is better. Diffusers numbers come from PBR `#308587`, vLLM-Omni from PBR `#308574`, and TensorRT-LLM from PBR `#308000`.
2. The **GPUs** column gives the number of GPUs behind every value in that row, and is the same for every engine in a GPU block so the rows can be compared directly.
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
|  | TensorRT-LLM | — |  |  |  |
| **H20** | PyTorch | 1 | 31.47 | 31.16 | 31.45 |
|  | vLLM-Omni | 1 | 4.82 | 31.36 | 87.55 |
|  | Diffusers | 1 | 5.00 | 28.00 | 28.20 |
|  | TensorRT-LLM | — |  |  |  |
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
| **H200 141GB HBM3** | PyTorch | 1 | 7.77 | 7.76 | 7.81 |
|  | vLLM-Omni | 1 | 1.33 | 7.72 | 20.65 |
|  | Diffusers | 1 | 2.00 | 8.00 | 8.00 |
|  | TensorRT-LLM | — |  |  |  |
| **B200** | PyTorch | 1 | 4.75 | 4.75 | 4.73 |
|  | vLLM-Omni | 1 | 0.94 | 4.93 | 12.00 |
|  | Diffusers | 1 | 2.00 | 5.00 | 5.00 |
|  | TensorRT-LLM | — |  |  |  |
| **B300** | PyTorch | — |  |  |  |
|  | vLLM-Omni | 1 | 1.09 | 4.86 | 11.70 |
|  | Diffusers | 1 | 20.87 | 24.49 | 26.44 |
|  | TensorRT-LLM | — |  |  |  |

### Forward Dynamics — Camera

| GPU | Engine | GPUs | 256p | 480p | 720p |
|---|---|:-:|---:|---:|---:|
| **RTX PRO 6000 Blackwell** | PyTorch | 1 | 16.82 | 16.82 | 16.81 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | — |  |  |  |
|  | TensorRT-LLM | — |  |  |  |
| **H20** | PyTorch | 1 | 31.40 | 31.10 | 31.40 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | — |  |  |  |
|  | TensorRT-LLM | — |  |  |  |
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
| **H200 141GB HBM3** | PyTorch | 1 | 7.73 | 7.75 | 7.75 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | — |  |  |  |
|  | TensorRT-LLM | — |  |  |  |
| **B200** | PyTorch | 1 | 4.70 | 4.74 | 4.71 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | — |  |  |  |
|  | TensorRT-LLM | — |  |  |  |
| **B300** | PyTorch | — |  |  |  |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | — |  |  |  |
|  | TensorRT-LLM | — |  |  |  |

### Forward Dynamics — Robot

| GPU | Engine | GPUs | 256p | 480p | 720p |
|---|---|:-:|---:|---:|---:|
| **RTX PRO 6000 Blackwell** | PyTorch | 1 | 3.74 | 3.73 | 3.72 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 1 | 2.00 | 3.00 | 3.00 |
|  | TensorRT-LLM | — |  |  |  |
| **H20** | PyTorch | 1 | 7.00 | 7.01 | 7.01 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 1 | 3.00 | 7.00 | 7.00 |
|  | TensorRT-LLM | — |  |  |  |
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
| **H200 141GB HBM3** | PyTorch | 1 | 1.93 | 1.93 | 1.93 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 1 | 2.00 | 2.00 | 2.00 |
|  | TensorRT-LLM | — |  |  |  |
| **B200** | PyTorch | 1 | 1.28 | 1.30 | 1.31 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 1 | 2.00 | 2.00 | 2.00 |
|  | TensorRT-LLM | — |  |  |  |
| **B300** | PyTorch | — |  |  |  |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 1 | 22.65 | 20.75 | 18.10 |
|  | TensorRT-LLM | — |  |  |  |

### Inverse Dynamics — Autonomous Vehicle (AV)

| GPU | Engine | GPUs | 256p | 480p | 720p |
|---|---|:-:|---:|---:|---:|
| **RTX PRO 6000 Blackwell** | PyTorch | 1 | 16.74 | 16.75 | 16.75 |
|  | vLLM-Omni | 1 | 2.10 | 12.96 | 36.10 |
|  | Diffusers | 1 | 2.00 | 12.00 | 12.00 |
|  | TensorRT-LLM | — |  |  |  |
| **H20** | PyTorch | 1 | 31.22 | 30.99 | 31.22 |
|  | vLLM-Omni | 1 | 4.84 | 31.36 | 87.50 |
|  | Diffusers | 1 | 5.00 | 28.00 | 28.00 |
|  | TensorRT-LLM | — |  |  |  |
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
| **H200 141GB HBM3** | PyTorch | 1 | 7.51 | 7.64 | 7.51 |
|  | vLLM-Omni | 1 | 1.36 | 7.69 | 20.65 |
|  | Diffusers | 1 | 2.00 | 8.00 | 8.00 |
|  | TensorRT-LLM | — |  |  |  |
| **B200** | PyTorch | 1 | 4.60 | 4.63 | 4.58 |
|  | vLLM-Omni | 1 | 0.92 | 5.10 | 12.29 |
|  | Diffusers | 1 | 2.00 | 5.00 | 5.00 |
|  | TensorRT-LLM | — |  |  |  |
| **B300** | PyTorch | — |  |  |  |
|  | vLLM-Omni | 1 | 1.36 | 5.75 | 11.89 |
|  | Diffusers | 1 | 8.76 | 17.79 | 17.99 |
|  | TensorRT-LLM | — |  |  |  |

### Inverse Dynamics — Robot

| GPU | Engine | GPUs | 256p | 480p | 720p |
|---|---|:-:|---:|---:|---:|
| **RTX PRO 6000 Blackwell** | PyTorch | 1 | 3.70 | 3.70 | 3.70 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 1 | 2.00 | 3.00 | 3.00 |
|  | TensorRT-LLM | — |  |  |  |
| **H20** | PyTorch | 1 | 6.94 | 6.94 | 6.89 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 1 | 3.00 | 7.00 | 7.00 |
|  | TensorRT-LLM | — |  |  |  |
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
| **H200 141GB HBM3** | PyTorch | 1 | 1.90 | 1.90 | 1.91 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 1 | 2.00 | 2.20 | 2.00 |
|  | TensorRT-LLM | — |  |  |  |
| **B200** | PyTorch | 1 | 1.28 | 1.27 | 1.28 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 1 | 2.00 | 2.00 | 2.00 |
|  | TensorRT-LLM | — |  |  |  |
| **B300** | PyTorch | — |  |  |  |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 1 | 9.28 | 12.40 | 10.86 |
|  | TensorRT-LLM | — |  |  |  |

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
1. PyTorch values are average generation (sampling) latency in seconds from PBR `#308197`; lower is better.
2. PyTorch values use `CUDA_GRAPH=No` and the `latency` automatic-sharding preset.
3. vLLM-Omni audiovisual values for `t2av`/`t2vs` and `i2av`/`i2vs` are from PBR `#308195`. vLLM-Omni action values are from PBR `#308481`.
4. Action vLLM-Omni rows report one GPU because the 8-GPU Ulysses runs failed sequence-length divisibility checks.
5. Policy-DROID vLLM-Omni uses the `nvidia/Cosmos3-Nano-Policy-DROID` checkpoint at 480p on one GPU only. It is reported in its own table and is not comparable to the PyTorch policy-robot row.
6. Diffusers rows are intentionally empty reservations for future benchmark campaigns.
7. The **GPUs** column gives the number of GPUs behind every value in that row.
8. Empty cells indicate unmeasured combinations, not unsupported combinations.</sub>