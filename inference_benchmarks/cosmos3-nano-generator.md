# Cosmos3-Nano Generator Inference Benchmarks

[Back to the inference benchmark index](../inference_benchmarks.md)

These tables report **Cosmos3-Nano Generator** latency in seconds. Lower is better. Empty cells mean that a run has not been completed for that GPU, engine, resolution, or GPU count; they do not indicate that a combination is unsupported.

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

The primary t2v, i2v, and t2i tables preserve the previously published benchmark campaigns across PyTorch, vLLM-Omni, Diffusers, and NIM. Those tables use BF16 precision, batch size 1, and matched prompts, seeds, and sampler settings where documented. Video workloads follow the standard Cosmos3 generation profile of 189 frames at 24 FPS unless a resolution tier limits frame count.

The additional audiovisual and action tables come from three internal benchmark reports. PyTorch values are from PBR `#308197`, **Cosmos3-Generator OSS Inference Benchmarking 32B and 8B (189 frames)**: average generation (sampling) latency from the native OSS path, using **CUDA Graphs disabled** and the **latency** automatic-sharding preset. vLLM-Omni values for text-to-audio-and-video (`t2av`/`t2vs`) and image-to-audio-and-video (`i2av`/`i2vs`) are from PBR `#308195`. vLLM-Omni action values are from PBR `#308481`, **Cosmos3-Generator vLLM-Omni Inference Benchmarking (action)**, measured with the `vllm/vllm-omni:cosmos3` image and the official action cookbook samples; `DIFFUSION_ATTENTION_BACKEND=TORCH_SDPA` was not set. Forward-dynamics cells are the mean of `av_forward`, `av_left`, and `av_right`; inverse-dynamics cells are the mean of `av_inverse_0` and `av_inverse_1`. That action sweep reports 1/2/4 GPUs only: 8-GPU Ulysses runs failed a sequence-length divisibility check, so `/8` action cells stay empty. The 2-GPU columns are omitted to keep the published table layout. Policy-DROID is a separate checkpoint and was measured only at 480p on one GPU. Super was not measured on H20, H100 NVL, or H100 80GB HBM3. Values are rounded to two decimal places.

These reports establish the reported timing matrix but do not expose every prompt and action payload in this repository. The linked public recipes explain modality behavior and provide representative payloads; their example-specific frame counts and action chunk sizes should not be treated as the exact internal benchmark inputs.

## Workload definitions

| Workload | Input | Output |
|---|---|---|
| Video-to-video (`v2v`) | Text prompt and source video | Generated video |
| Text-to-audio-and-video (`t2av`) | Text prompt | Synchronized video and sound |
| Video-to-audio-and-video (`v2av`) | Text prompt and source video | Generated video with synchronized sound |
| Image-to-audio-and-video (`i2av`) | Text prompt and source image | Generated video with synchronized sound |
| Forward dynamics | Initial visual observation and an action trajectory | Future-observation rollout video |
| Inverse dynamics | Observed video | Recovered action trajectory; some serving integrations also return video |
| Policy | Initial visual observation, instruction, and optional state | Predicted action trajectory and, for general Generator paths, a rollout video |
| Policy-DROID | Multiview wrist/exterior observations for the DROID embodiment | Predicted action chunk from the Policy-DROID checkpoint |

The PBR uses `t2av`, `v2av`, and `i2av`; some public recipes call the same sound-producing modes `t2vs`, `v2vs`, and `i2vs`. See the [audiovisual cookbook](../cookbooks/cosmos3/generator/audiovisual/README.md) for generation inputs and the [action cookbook](../cookbooks/cosmos3/generator/action/README.md) for action representations and output contracts. vLLM-Omni request shapes are maintained in the [Nano recipe](https://github.com/vllm-project/vllm-omni/blob/main/recipes/cosmos3/Cosmos3-Nano.md). Public recipes cover more embodiments than this PBR; the action tables below intentionally use only its measured domain rows.

## Primary vision generation

### Text-to-Video (t2v)

| GPU | Engine | 256p/1 | 256p/4 | 256p/8 | 480p/1 | 480p/4 | 480p/8 | 720p/1 | 720p/4 | 720p/8 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **RTX PRO 6000 Blackwell** | PyTorch | 13.95 | 6.46 | 4.90 | 180.81 | 57.23 | 35.71 | 786.37 | 225.45 | 127.57 |
| | vLLM-Omni | 10.65 | 5.06 | 3.78 | 105.61 | 35.93 | 23.76 | 369.67 | 114.30 | 68.66 |
| | Diffusers | 11.20 | | | 112.00 | | | 392.00 | | |
| | NIM | 7.93 | 4.69 | 4.00 | 82.69 | 33.57 | 24.49 | 318.69 | 107.83 | 68.50 |
| **H20** | PyTorch | 30.57 | 11.25 | 7.87 | 257.51 | 80.18 | 50.49 | 931.39 | 268.88 | 157.71 |
| | vLLM-Omni | 28.58 | 10.20 | 7.70 | 256.97 | 77.42 | 47.53 | 929.81 | 260.75 | 148.46 |
| | Diffusers | 30.20 | | | 258.00 | | | 926.00 | | |
| | NIM | 17.81 | 7.84 | 6.14 | 192.07 | 62.80 | 41.67 | 771.37 | 223.24 | 132.68 |
| **H100 NVL** | PyTorch | 10.03 | 4.27 | 3.95 | 84.12 | 29.18 | 21.46 | 297.27 | 94.15 | 61.63 |
| | vLLM-Omni | 9.25 | 3.68 | 3.15 | 80.75 | 27.48 | 18.77 | 311.13 | 88.25(*) | 54.01(*) |
| | Diffusers | 11.00 | | | 90.00 | | | 324.20 | | |
| | NIM | 7.09 | 3.77 | 3.65 | 68.01 | 25.67 | 20.51 | 267.73 | 86.18 | 57.38 |
| **H200 NVL** | PyTorch | 8.17 | 3.75 | 3.14 | 69.79 | 24.23 | 15.10 | 244.39 | 77.35 | 45.70 |
| | vLLM-Omni | 7.44 | 3.27 | 2.33 | 64.58 | 21.31 | 12.92 | 240.05 | 69.63 | 39.17 |
| | Diffusers | 9.00 | | | 74.00 | | | 276.20 | | |
| | NIM | 5.87 | 3.38 | 3.00 | 57.10 | 21.74 | 15.04 | 229.63 | 71.32 | 43.34 |
| **H100 80GB HBM3** | PyTorch | 7.61 | 3.50 | 3.17 | 59.83 | 21.23 | 14.37 | 207.78 | 66.94 | 41.81 |
| | vLLM-Omni | 6.97 | 3.45 | 3.49 | 58.17 | 19.95 | 13.46 | 202.29 | 62.82 | 37.80 |
| | Diffusers | 9.00 | | | 68.00 | | | 240.00 | | |
| | NIM | 5.72 | 3.36 | 3.12 | 51.73 | 20.26 | 14.82 | 199.46 | 65.32 | 41.66 |
| **H200 141GB HBM3** | PyTorch | 7.53 | 3.34 | 3.19 | 60.18 | 20.84 | 13.97 | 214.28 | 67.48 | 41.26 |
| | vLLM-Omni | 6.79 | 3.25 | 3.42 | 58.14 | 19.77 | 12.97 | 208.36 | 63.27 | 37.49 |
| | Diffusers | 9.00 | | | 67.00 | | | 239.60 | | |
| | NIM | 5.72 | 3.26 | 3.10 | 51.90 | 20.09 | 14.61 | 200.63 | 64.57 | 40.70 |
| **B200** | PyTorch | 4.56 | 2.78 | 2.79 | 33.20 | 13.20 | 9.69 | 114.85 | 39.75 | 26.27 |
| | vLLM-Omni | 4.03 | 2.43 | 3.49 | 32.04 | 12.63 | 10.09 | 107.84 | 35.29 | 22.87 |
| | Diffusers | 7.00 | | | 36.80 | | | 117.00 | | |
| | NIM | 3.72 | 2.74 | 2.89 | 26.68 | 12.66 | 10.10 | 93.33 | 35.04 | 24.64 |
| **B300** | PyTorch | | | | | | | | | |
| | vLLM-Omni | 4.46 | 4.11 | 5.44 | 32.18 | 13.83 | 11.57 | 102.10 | 35.68 | 24.33 |
| | Diffusers | 39.40 | | | 63.40 | | | 139.40 | | |
| | NIM | 4.51 | 3.54 | 4.11 | 27.49 | 14.51 | 11.86 | 90.39 | 37.28 | 26.19 |

### Image-to-Video (i2v)

| GPU | Engine | 256p/1 | 256p/4 | 256p/8 | 480p/1 | 480p/4 | 480p/8 | 720p/1 | 720p/4 | 720p/8 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **RTX PRO 6000 Blackwell** | PyTorch | 14.08 | 6.51 | 4.92 | 182.14 | 57.43 | 35.74 | 788.80 | 226.25 | 127.79 |
| | vLLM-Omni | 11.04 | 5.48 | 4.24 | 107.77 | 38.05 | 25.95 | 375.01 | 119.27 | 73.57 |
| | Diffusers | 12.00 | | | 112.00 | | | 397.00 | | |
| | NIM | 8.23 | 5.28 | 4.64 | 84.58 | 35.74 | 26.63 | 326.96 | 112.27 | 73.31 |
| **H20** | PyTorch | 31.36 | 11.39 | 7.97 | 257.10 | 80.53 | 50.63 | 933.07 | 268.99 | 158.10 |
| | vLLM-Omni | 29.50 | 11.26 | 8.64 | 261.56 | 81.93 | 52.06 | 940.16 | 271.37 | 158.76 |
| | Diffusers | 31.00 | | | 258.00 | | | 925.00 | | |
| | NIM | 18.67 | 9.04 | 7.38 | 195.10 | 67.64 | 46.30 | 774.88 | 233.92 | 143.15 |
| **H100 NVL** | PyTorch | 10.19 | 4.31 | 3.99 | 84.50 | 28.69 | 21.52 | 298.57 | 95.76 | 60.58 |
| | vLLM-Omni | 9.62 | 4.11 | 3.63 | 82.61 | 29.35 | 20.73 | 286.33 | 92.23(*) | 58.02(*) |
| | Diffusers | 11.00 | | | 91.00 | | | 325.20 | | |
| | NIM | 7.39 | 4.36 | 4.28 | 69.39 | 27.67 | 22.43 | 272.29 | 90.26 | 61.55 |
| **H200 NVL** | PyTorch | 8.27 | 3.76 | 3.09 | 69.99 | 24.22 | 15.12 | 246.62 | 77.69 | 45.99 |
| | vLLM-Omni | 7.83 | 3.69 | 2.78 | 66.39 | 22.93 | 14.58 | 243.52 | 73.26 | 42.86 |
| | Diffusers | 9.00 | | | 74.00 | | | 275.20 | | |
| | NIM | 6.25 | 3.97 | 3.60 | 58.64 | 23.33 | 16.75 | 232.47 | 75.13 | 47.22 |
| **H100 80GB HBM3** | PyTorch | 7.64 | 3.47 | 3.21 | 59.95 | 21.40 | 14.43 | 207.87 | 67.52 | 41.66 |
| | vLLM-Omni | 7.37 | 3.81 | 3.97 | 59.77 | 21.68 | 15.12 | 205.97 | 66.52 | 41.51 |
| | Diffusers | 9.00 | | | 68.00 | | | 239.80 | | |
| | NIM | 6.08 | 3.89 | 3.71 | 53.02 | 22.20 | 16.61 | 202.59 | 69.02 | 45.16 |
| **H200 141GB HBM3** | PyTorch | 7.65 | 3.37 | 3.17 | 60.51 | 21.01 | 14.07 | 214.80 | 67.14 | 41.00 |
| | vLLM-Omni | 7.28 | 3.63 | 3.83 | 59.64 | 21.35 | 14.67 | 209.65 | 66.65 | 40.77 |
| | Diffusers | 9.00 | | | 67.20 | | | 240.00 | | |
| | NIM | 6.04 | 3.80 | 3.65 | 53.25 | 21.89 | 16.23 | 203.66 | 68.30 | 44.29 |
| **B200** | PyTorch | 4.60 | 2.77 | 2.81 | 33.08 | 13.07 | 9.66 | 113.90 | 40.01 | 26.58 |
| | vLLM-Omni | 4.33 | 2.77 | 3.84 | 33.09 | 13.79 | 11.39 | 110.19 | 37.76 | 25.68 |
| | Diffusers | | | | | | | 116.00 | | |
| | NIM | 4.05 | 3.43 | 3.60 | 27.72 | 14.02 | 11.43 | 95.57 | 37.76 | 27.29 |
| **B300** | PyTorch | | | | | | | | | |
| | vLLM-Omni | 5.61 | 4.67 | 5.90 | 33.45 | 15.06 | 13.13 | 104.75 | 38.27 | 26.87 |
| | Diffusers | 28.60 | | | 65.60 | | | 139.60 | | |
| | NIM | 4.50 | 4.96 | 5.03 | 28.90 | 16.06 | 13.25 | 92.59 | 39.49 | 29.00 |

### Text-to-Image (t2i)

| GPU | Engine | 256p/1 | 256p/4 | 256p/8 | 480p/1 | 480p/4 | 480p/8 | 720p/1 | 720p/4 | 720p/8 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **RTX PRO 6000 Blackwell** | PyTorch | 2.99 | 2.36 | 2.38 | 4.51 | 2.46 | 2.48 | 7.12 | 3.18 | 2.70 |
| | vLLM-Omni | 1.59 | 1.54 | 1.59 | 2.87 | 1.55 | 1.81 | 4.99 | 2.32 | 1.96 |
| | Diffusers | 2.00 | | | 4.00 | | | 5.00 | | |
| **H20** | PyTorch | 3.06 | 2.65 | 2.74 | 6.51 | 2.84 | 2.89 | 12.31 | 4.28 | 3.06 |
| | vLLM-Omni | 1.73 | 2.46 | 3.22 | 4.92 | 2.57 | 7.24 | 10.73 | 4.24 | 3.59 |
| | Diffusers | 3.00 | | | 6.00 | | | 10.00 | | |
| **H100 NVL** | PyTorch | 2.77 | 2.45 | 2.57 | 2.83 | 2.56 | 2.51 | 4.21 | 2.57 | 2.64 |
| | vLLM-Omni | 1.55 | 1.75 | 1.91 | 1.92 | 1.81 | 10.82 | 3.44 | 1.83 | 1.90 |
| | Diffusers | 3.00 | | | 3.00 | | | 4.00 | | |
| **H200 NVL** | PyTorch | 2.75 | 2.42 | 2.50 | 2.85 | 2.52 | 2.62 | 3.58 | 2.62 | 2.64 |
| | vLLM-Omni | 1.53 | 2.01 | 1.96 | 1.58 | 1.91 | 17.71 | 2.81 | 1.94 | 1.94 |
| | Diffusers | 3.00 | | | 3.00 | | | 4.00 | | |
| **H100 80GB HBM3** | PyTorch | 3.01 | 2.66 | 2.56 | 3.01 | 2.59 | 2.75 | 3.45 | 2.73 | 2.77 |
| | vLLM-Omni | 1.61 | 2.45 | 3.18 | 1.53 | 2.35 | 7.02 | 2.61 | 2.45 | 3.03 |
| | Diffusers | 3.00 | | | 3.00 | | | 4.00 | | |
| **H200 141GB HBM3** | PyTorch | 2.96 | 2.59 | 2.70 | 3.04 | 2.78 | 2.77 | 3.28 | 2.84 | 2.77 |
| | vLLM-Omni | 1.57 | 2.38 | 3.16 | 1.52 | 2.37 | 7.05 | 2.60 | 2.33 | 3.20 |
| | Diffusers | 3.00 | | | 3.00 | | | 4.00 | | |
| **B200** | PyTorch | 2.68 | 2.39 | 2.59 | 2.75 | 2.43 | 2.56 | 2.87 | 2.58 | 2.62 |
| | vLLM-Omni | 1.49 | 2.21 | 3.27 | 1.20 | 2.05 | 7.58 | 1.77 | 2.20 | 3.41 |
| | Diffusers | | | | | | | 3.00 | | |
| **B300** | PyTorch | | | | | | | | | |
| | vLLM-Omni | 1.97 | 4.52 | 5.82 | 1.81 | 4.16 | 71.19 | 2.34 | 4.09 | 5.62 |
| | Diffusers | 36.20 | | | | | | 41.00 | | |

<sub>Notes:
1. All times measured on identical workloads (same seed, sampler settings, prompt).
2. 4×/8× GPU configurations use tensor parallelism.
3. vLLM-Omni numbers are for the upcoming public release in the vLLM-Omni repo; subject to change before GA. Values marked with (*) are pre-release vLLM-Omni measurements on H100 NVL and may change before GA.
4. Diffusers numbers use the HuggingFace `diffusers` integration without custom CUDA graphs; reported at 256p/1, 480p/1, and 720p/1 (single-GPU only).
5. PyTorch numbers report average generation (sampling) time from OSS inference benchmarking.
6. At 256p, multi-GPU configurations on B300 may underperform single-GPU due to small-workload TP overhead; single-GPU is recommended at this resolution.
7. NIM numbers use latency profiles with FP8 precision and report end-to-end `Request Latency s`, including request processing, video generation, output encoding, and returning the response.</sub>

## Additional audiovisual generation

### Video-to-Video (v2v)

A text prompt and source video condition a generated continuation or transformation.

| GPU | Engine | 256p/1 | 256p/4 | 256p/8 | 480p/1 | 480p/4 | 480p/8 | 720p/1 | 720p/4 | 720p/8 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **RTX PRO 6000 Blackwell** | PyTorch | 13.73 | 6.49 | 4.88 | 180.54 | 57.44 | 35.65 | 785.41 | 224.94 | 127.19 |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |
| **H20** | PyTorch | 30.44 | 11.35 | 7.94 | 257.79 | 80.52 | 50.56 | 930.96 | 268.82 | 157.81 |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |
| **H100 NVL** | PyTorch | 10.18 | 4.29 | 3.95 | 81.73 | 28.86 | 21.40 | 297.83 | 94.37 | 61.31 |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |
| **H200 NVL** | PyTorch | 8.10 | 3.73 | 3.12 | 69.70 | 24.17 | 15.14 | 250.33 | 76.59 | 45.94 |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |
| **H100 80GB HBM3** | PyTorch | 7.50 | 3.47 | 3.16 | 59.75 | 21.10 | 14.29 | 207.36 | 67.05 | 41.54 |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |
| **H200 141GB HBM3** | PyTorch | 7.48 | 3.35 | 3.11 | 60.23 | 20.99 | 13.94 | 214.03 | 66.90 | 41.06 |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |
| **B200** | PyTorch | 4.52 | 2.74 | 2.78 | 32.53 | 13.06 | 9.68 | 114.16 | 39.67 | 26.33 |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |
| **B300** | PyTorch |  |  |  |  |  |  |  |  |  |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |

### Text-to-Audio-and-Video (t2av)

A text prompt produces synchronized video and sound.

| GPU | Engine | 256p/1 | 256p/4 | 256p/8 | 480p/1 | 480p/4 | 480p/8 | 720p/1 | 720p/4 | 720p/8 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **RTX PRO 6000 Blackwell** | PyTorch | 14.13 | 6.47 | 4.89 | 180.98 | 57.21 | 35.77 | 794.74 | 225.73 | 127.53 |
| | vLLM-Omni | 12.25 | 5.32 | 4.19 | 106.02 | 36.11 | 24.02 | 373.10 | 115.06 | 69.19 |
| | Diffusers | | | | | | | | | |
| **H20** | PyTorch | 30.82 | 11.21 | 7.87 | 256.41 | 80.22 | 50.51 | 922.35 | 268.73 | 158.03 |
| | vLLM-Omni | 29.43 | 10.74 | 7.88 | 260.84 | 78.67 | 48.02 | 937.93 | 262.83 | 148.26 |
| | Diffusers | | | | | | | | | |
| **H100 NVL** | PyTorch | 9.99 | 4.27 | 3.95 | 82.32 | 28.50 | 21.46 | 297.88 | 94.45 | 60.25 |
| | vLLM-Omni | 9.92 | 3.85 | 3.26 | 81.57 | 27.15 | 18.93 | 291.91 | 90.29 | 55.66 |
| | Diffusers | | | | | | | | | |
| **H200 NVL** | PyTorch | 8.20 | 3.75 | 3.08 | 69.88 | 24.30 | 15.17 | 250.46 | 78.00 | 45.75 |
| | vLLM-Omni | 7.82 | 3.43 | 2.35 | 65.42 | 21.42 | 12.93 | 244.60 | 70.56 | 39.95 |
| | Diffusers | | | | | | | | | |
| **H100 80GB HBM3** | PyTorch | 7.60 | 3.43 | 3.18 | 59.09 | 21.18 | 14.32 | 207.87 | 66.74 | 41.71 |
| | vLLM-Omni | 7.34 | 3.54 | 3.43 | 58.65 | 20.10 | 13.47 | 203.92 | 63.22 | 38.45 |
| | Diffusers | | | | | | | | | |
| **H200 141GB HBM3** | PyTorch | 7.55 | 3.33 | 3.13 | 60.17 | 20.86 | 13.90 | 211.59 | 67.02 | 41.24 |
| | vLLM-Omni | 7.13 | 3.45 | 3.35 | 58.27 | 19.62 | 13.11 | 209.29 | 63.53 | 37.92 |
| | Diffusers | | | | | | | | | |
| **B200** | PyTorch | 4.63 | 2.81 | 2.81 | 32.60 | 13.17 | 9.64 | 113.89 | 39.59 | 26.33 |
| | vLLM-Omni | 4.29 | 2.51 | 3.47 | 32.20 | 12.70 | 10.08 | 107.74 | 36.10 | 23.83 |
| | Diffusers | | | | | | | | | |
| **B300** | PyTorch |  |  |  |  |  |  |  |  |  |
| | vLLM-Omni | 5.59 | 4.21 | 5.54 | 33.15 | 14.10 | 11.90 | 104.92 | 36.00 | 24.92 |
| | Diffusers | | | | | | | | | |

### Video-to-Audio-and-Video (v2av)

A text prompt and source video produce transformed video with synchronized sound.

| GPU | Engine | 256p/1 | 256p/4 | 256p/8 | 480p/1 | 480p/4 | 480p/8 | 720p/1 | 720p/4 | 720p/8 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **RTX PRO 6000 Blackwell** | PyTorch | 13.73 | 6.45 | 4.88 | 180.72 | 57.30 | 35.73 | 786.11 | 224.99 | 127.45 |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |
| **H20** | PyTorch | 30.42 | 11.35 | 7.94 | 257.83 | 80.48 | 50.55 | 921.82 | 268.81 | 158.07 |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |
| **H100 NVL** | PyTorch | 9.91 | 4.35 | 3.95 | 81.96 | 29.02 | 21.53 | 297.51 | 94.64 | 60.36 |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |
| **H200 NVL** | PyTorch | 8.10 | 3.73 | 3.09 | 69.59 | 24.39 | 15.08 | 246.51 | 76.46 | 46.27 |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |
| **H100 80GB HBM3** | PyTorch | 7.49 | 3.46 | 3.17 | 59.86 | 21.25 | 14.21 | 207.55 | 67.24 | 41.63 |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |
| **H200 141GB HBM3** | PyTorch | 7.49 | 3.35 | 3.10 | 60.15 | 21.00 | 13.93 | 214.44 | 67.13 | 41.05 |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |
| **B200** | PyTorch | 4.51 | 2.76 | 2.82 | 32.62 | 13.05 | 9.61 | 113.72 | 39.28 | 26.18 |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |
| **B300** | PyTorch |  |  |  |  |  |  |  |  |  |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |

### Image-to-Audio-and-Video (i2av)

A text prompt and source image produce video with synchronized sound.

| GPU | Engine | 256p/1 | 256p/4 | 256p/8 | 480p/1 | 480p/4 | 480p/8 | 720p/1 | 720p/4 | 720p/8 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **RTX PRO 6000 Blackwell** | PyTorch | 14.08 | 6.48 | 4.91 | 181.64 | 57.25 | 35.69 | 787.81 | 226.71 | 127.85 |
| | vLLM-Omni | 12.61 | 5.72 | 4.53 | 108.05 | 38.30 | 26.20 | 378.04 | 120.08 | 74.23 |
| | Diffusers | | | | | | | | | |
| **H20** | PyTorch | 31.34 | 11.36 | 7.97 | 259.06 | 80.55 | 50.63 | 924.39 | 269.21 | 157.97 |
| | vLLM-Omni | 30.36 | 11.60 | 8.91 | 265.37 | 83.11 | 52.70 | 948.09 | 272.93 | 158.76 |
| | Diffusers | | | | | | | | | |
| **H100 NVL** | PyTorch | 10.45 | 4.31 | 3.97 | 84.37 | 28.74 | 21.46 | 298.61 | 94.02 | 60.57 |
| | vLLM-Omni | 10.34 | 4.30 | 3.72 | 83.21 | 29.20 | 20.82 | 295.65 | 94.44 | 59.83 |
| | Diffusers | | | | | | | | | |
| **H200 NVL** | PyTorch | 8.29 | 3.76 | 3.08 | 71.20 | 24.34 | 15.05 | 248.35 | 76.82 | 46.40 |
| | vLLM-Omni | 8.24 | 3.85 | 2.81 | 67.23 | 23.05 | 14.65 | 249.41 | 74.44 | 43.50 |
| | Diffusers | | | | | | | | | |
| **H100 80GB HBM3** | PyTorch | 7.73 | 3.46 | 3.11 | 59.13 | 21.24 | 14.30 | 207.93 | 67.50 | 41.75 |
| | vLLM-Omni | 7.74 | 3.96 | 4.08 | 60.27 | 21.81 | 15.26 | 207.53 | 66.74 | 41.92 |
| | Diffusers | | | | | | | | | |
| **H200 141GB HBM3** | PyTorch | 7.66 | 3.35 | 3.14 | 60.55 | 20.98 | 13.99 | 214.82 | 67.35 | 41.21 |
| | vLLM-Omni | 7.55 | 3.90 | 3.77 | 59.73 | 21.35 | 14.67 | 212.73 | 66.78 | 41.03 |
| | Diffusers | | | | | | | | | |
| **B200** | PyTorch | 4.62 | 2.78 | 2.80 | 32.82 | 13.09 | 9.69 | 113.85 | 39.76 | 26.56 |
| | vLLM-Omni | 4.58 | 2.83 | 3.85 | 33.36 | 13.89 | 11.14 | 110.05 | 38.82 | 26.30 |
| | Diffusers | | | | | | | | | |
| **B300** | PyTorch |  |  |  |  |  |  |  |  |  |
| | vLLM-Omni | 6.01 | 4.76 | 6.09 | 34.39 | 15.45 | 13.15 | 108.59 | 38.62 | 27.59 |
| | Diffusers | | | | | | | | | |

## Action generation

Forward dynamics is reported separately for AV, camera, and robot inputs. Inverse dynamics and policy are reported for AV and robot because those are the only domain rows in PBR `#308197`; no camera row is inferred. For each domain, PyTorch is populated from that report. vLLM-Omni currently covers forward dynamics AV and inverse dynamics AV from PBR `#308481`. Camera, robot-FD, robot-ID, policy-AV, and policy-robot vLLM-Omni rows remain reserved. **Policy — DROID** is a separate checkpoint (`nvidia/Cosmos3-Nano-Policy-DROID`) and is not mixed with the policy-robot table.

### Forward Dynamics — Autonomous Vehicle (AV)

| GPU | Engine | 256p/1 | 256p/4 | 256p/8 | 480p/1 | 480p/4 | 480p/8 | 720p/1 | 720p/4 | 720p/8 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **RTX PRO 6000 Blackwell** | PyTorch | 16.87 | 12.13 | 8.96 | 16.86 | 12.18 | 8.95 | 16.85 | 12.14 | 8.94 |
| | vLLM-Omni | 2.08 | 1.46 |  | 12.94 | 7.06 |  | 36.19 | 16.58 |  |
| | Diffusers | | | | | | | | | |
| **H20** | PyTorch | 31.47 | 19.97 | 13.77 | 31.16 | 19.91 | 13.78 | 31.45 | 19.95 | 13.78 |
| | vLLM-Omni | 4.82 | 2.36 |  | 31.36 | 12.83 |  | 87.55 | 32.09 |  |
| | Diffusers | | | | | | | | | |
| **H100 NVL** | PyTorch | 10.32 | 7.36 | 6.46 | 10.66 | 7.35 | 6.31 | 10.64 | 7.43 | 6.32 |
| | vLLM-Omni | 1.71 | 1.46 |  | 10.58 | 5.63 |  | 28.26 | 13.61 |  |
| | Diffusers | | | | | | | | | |
| **H200 NVL** | PyTorch | 8.30 | 6.18 | 4.41 | 8.31 | 6.16 | 4.39 | 8.33 | 6.14 | 4.39 |
| | vLLM-Omni | 1.43 | 1.50 |  | 8.21 | 3.83 |  | 22.78 | 9.37 |  |
| | Diffusers | | | | | | | | | |
| **H100 80GB HBM3** | PyTorch | 7.80 | 5.69 | 4.30 | 7.87 | 5.68 | 4.34 | 7.79 | 5.68 | 4.30 |
| | vLLM-Omni | 1.33 | 1.51 |  | 7.74 | 3.77 |  | 20.60 | 9.09 |  |
| | Diffusers | | | | | | | | | |
| **H200 141GB HBM3** | PyTorch | 7.77 | 5.51 | 4.16 | 7.76 | 5.56 | 4.18 | 7.81 | 5.54 | 4.16 |
| | vLLM-Omni | 1.33 | 1.49 |  | 7.72 | 3.63 |  | 20.65 | 8.79 |  |
| | Diffusers | | | | | | | | | |
| **B200** | PyTorch | 4.75 | 3.88 | 3.38 | 4.75 | 3.85 | 3.37 | 4.73 | 3.87 | 3.36 |
| | vLLM-Omni | 0.94 | 1.20 |  | 4.93 | 2.84 |  | 12.00 | 5.84 |  |
| | Diffusers | | | | | | | | | |
| **B300** | PyTorch |  |  |  |  |  |  |  |  |  |
| | vLLM-Omni | 1.09 | 2.30 |  | 4.86 | 3.56 |  | 11.70 | 6.31 |  |
| | Diffusers | | | | | | | | | |

### Forward Dynamics — Camera

| GPU | Engine | 256p/1 | 256p/4 | 256p/8 | 480p/1 | 480p/4 | 480p/8 | 720p/1 | 720p/4 | 720p/8 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **RTX PRO 6000 Blackwell** | PyTorch | 16.82 | 12.01 | 8.80 | 16.82 | 12.01 | 8.80 | 16.81 | 12.01 | 8.81 |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |
| **H20** | PyTorch | 31.40 | 19.68 | 13.52 | 31.10 | 19.69 | 13.50 | 31.40 | 19.75 | 13.52 |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |
| **H100 NVL** | PyTorch | 10.61 | 7.35 | 6.19 | 10.32 | 7.27 | 6.31 | 10.57 | 7.35 | 6.25 |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |
| **H200 NVL** | PyTorch | 8.24 | 6.08 | 4.32 | 8.26 | 6.09 | 4.32 | 8.26 | 6.09 | 4.33 |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |
| **H100 80GB HBM3** | PyTorch | 7.78 | 5.64 | 4.24 | 7.83 | 5.61 | 4.21 | 7.75 | 5.60 | 4.21 |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |
| **H200 141GB HBM3** | PyTorch | 7.73 | 5.44 | 4.08 | 7.75 | 5.44 | 4.09 | 7.75 | 5.49 | 4.08 |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |
| **B200** | PyTorch | 4.70 | 3.79 | 3.23 | 4.74 | 3.78 | 3.27 | 4.71 | 3.78 | 3.23 |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |
| **B300** | PyTorch |  |  |  |  |  |  |  |  |  |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |

### Forward Dynamics — Robot

| GPU | Engine | 256p/1 | 256p/4 | 256p/8 | 480p/1 | 480p/4 | 480p/8 | 720p/1 | 720p/4 | 720p/8 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **RTX PRO 6000 Blackwell** | PyTorch | 3.74 | 3.15 | 2.62 | 3.73 | 3.14 | 2.63 | 3.72 | 3.13 | 2.59 |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |
| **H20** | PyTorch | 7.00 | 4.78 | 3.59 | 7.01 | 4.79 | 3.59 | 7.01 | 4.78 | 3.59 |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |
| **H100 NVL** | PyTorch | 2.49 | 2.15 | 2.18 | 2.50 | 2.16 | 2.12 | 2.55 | 2.11 | 2.15 |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |
| **H200 NVL** | PyTorch | 2.08 | 2.14 | 2.11 | 2.09 | 2.03 | 2.08 | 2.09 | 2.03 | 2.09 |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |
| **H100 80GB HBM3** | PyTorch | 2.00 | 2.14 | 2.15 | 2.01 | 2.14 | 2.17 | 1.98 | 2.15 | 2.25 |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |
| **H200 141GB HBM3** | PyTorch | 1.93 | 2.14 | 2.30 | 1.93 | 2.14 | 2.15 | 1.93 | 2.11 | 2.21 |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |
| **B200** | PyTorch | 1.28 | 1.89 | 1.99 | 1.30 | 1.91 | 2.00 | 1.31 | 1.91 | 1.94 |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |
| **B300** | PyTorch |  |  |  |  |  |  |  |  |  |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |

### Inverse Dynamics — Autonomous Vehicle (AV)

| GPU | Engine | 256p/1 | 256p/4 | 256p/8 | 480p/1 | 480p/4 | 480p/8 | 720p/1 | 720p/4 | 720p/8 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **RTX PRO 6000 Blackwell** | PyTorch | 16.74 | 12.05 | 8.79 | 16.75 | 12.02 | 8.78 | 16.75 | 12.02 | 8.78 |
| | vLLM-Omni | 2.10 | 1.47 |  | 12.96 | 7.09 |  | 36.10 | 16.61 |  |
| | Diffusers | | | | | | | | | |
| **H20** | PyTorch | 31.22 | 19.66 | 13.52 | 30.99 | 19.68 | 13.52 | 31.22 | 19.70 | 13.53 |
| | vLLM-Omni | 4.84 | 2.39 |  | 31.36 | 12.81 |  | 87.50 | 32.08 |  |
| | Diffusers | | | | | | | | | |
| **H100 NVL** | PyTorch | 10.47 | 7.13 | 6.16 | 10.45 | 7.16 | 6.26 | 10.44 | 7.10 | 6.14 |
| | vLLM-Omni | 1.74 | 1.48 |  | 10.54 | 5.61 |  | 28.16 |  |  |
| | Diffusers | | | | | | | | | |
| **H200 NVL** | PyTorch | 8.12 | 5.96 | 4.19 | 8.13 | 5.97 | 4.20 | 8.10 | 5.98 | 4.20 |
| | vLLM-Omni | 1.46 | 1.46 |  | 8.20 | 3.80 |  | 22.34 | 9.32 |  |
| | Diffusers | | | | | | | | | |
| **H100 80GB HBM3** | PyTorch | 7.62 | 5.54 | 4.15 | 7.63 | 5.52 | 4.16 | 7.68 | 5.53 | 4.14 |
| | vLLM-Omni | 1.39 | 1.50 |  | 7.72 | 3.75 |  | 20.55 | 9.05 |  |
| | Diffusers | | | | | | | | | |
| **H200 141GB HBM3** | PyTorch | 7.51 | 5.36 | 4.00 | 7.64 | 5.34 | 3.99 | 7.51 | 5.37 | 3.99 |
| | vLLM-Omni | 1.36 | 1.48 |  | 7.69 | 3.60 |  | 20.65 | 8.71 |  |
| | Diffusers | | | | | | | | | |
| **B200** | PyTorch | 4.60 | 3.74 | 3.20 | 4.63 | 3.73 | 3.19 | 4.58 | 3.76 | 3.23 |
| | vLLM-Omni | 0.92 | 1.55 |  | 5.10 | 2.83 |  | 12.29 | 5.92 |  |
| | Diffusers | | | | | | | | | |
| **B300** | PyTorch |  |  |  |  |  |  |  |  |  |
| | vLLM-Omni | 1.36 | 2.55 |  | 5.75 | 4.04 |  | 11.89 | 6.67 |  |
| | Diffusers | | | | | | | | | |

### Inverse Dynamics — Robot

| GPU | Engine | 256p/1 | 256p/4 | 256p/8 | 480p/1 | 480p/4 | 480p/8 | 720p/1 | 720p/4 | 720p/8 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **RTX PRO 6000 Blackwell** | PyTorch | 3.70 | 3.12 | 2.61 | 3.70 | 3.11 | 2.60 | 3.70 | 3.11 | 2.57 |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |
| **H20** | PyTorch | 6.94 | 4.72 | 3.53 | 6.94 | 4.72 | 3.52 | 6.89 | 4.71 | 3.52 |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |
| **H100 NVL** | PyTorch | 2.53 | 2.12 | 2.09 | 2.47 | 2.07 | 2.09 | 2.54 | 2.11 | 2.09 |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |
| **H200 NVL** | PyTorch | 2.05 | 1.99 | 2.09 | 2.05 | 2.01 | 2.07 | 2.05 | 2.05 | 2.05 |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |
| **H100 80GB HBM3** | PyTorch | 1.96 | 2.09 | 2.16 | 1.96 | 2.07 | 2.18 | 1.95 | 2.17 | 2.13 |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |
| **H200 141GB HBM3** | PyTorch | 1.90 | 2.06 | 2.10 | 1.90 | 2.07 | 2.13 | 1.91 | 2.12 | 2.09 |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |
| **B200** | PyTorch | 1.28 | 1.91 | 1.91 | 1.27 | 1.87 | 1.97 | 1.28 | 1.86 | 1.91 |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |
| **B300** | PyTorch |  |  |  |  |  |  |  |  |  |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |

### Policy — Autonomous Vehicle (AV)

| GPU | Engine | 256p/1 | 256p/4 | 256p/8 | 480p/1 | 480p/4 | 480p/8 | 720p/1 | 720p/4 | 720p/8 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **RTX PRO 6000 Blackwell** | PyTorch | 16.88 | 12.13 | 8.91 | 16.89 | 12.16 | 8.93 | 16.91 | 12.13 | 8.93 |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |
| **H20** | PyTorch | 31.51 | 19.90 | 13.77 | 31.51 | 19.94 | 13.79 | 31.50 | 19.97 | 13.78 |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |
| **H100 NVL** | PyTorch | 10.66 | 7.34 | 6.27 | 10.65 | 7.41 | 6.33 | 10.35 | 7.33 | 6.27 |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |
| **H200 NVL** | PyTorch | 8.31 | 6.17 | 4.38 | 8.32 | 6.16 | 4.39 | 8.31 | 6.14 | 4.38 |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |
| **H100 80GB HBM3** | PyTorch | 7.82 | 5.68 | 4.31 | 7.82 | 5.67 | 4.33 | 7.80 | 5.67 | 4.32 |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |
| **H200 141GB HBM3** | PyTorch | 7.76 | 5.52 | 4.15 | 7.69 | 5.58 | 4.16 | 7.66 | 5.51 | 4.16 |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |
| **B200** | PyTorch | 4.75 | 3.89 | 3.34 | 4.77 | 3.88 | 3.34 | 4.74 | 3.87 | 3.28 |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |
| **B300** | PyTorch |  |  |  |  |  |  |  |  |  |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |

### Policy — Robot

| GPU | Engine | 256p/1 | 256p/4 | 256p/8 | 480p/1 | 480p/4 | 480p/8 | 720p/1 | 720p/4 | 720p/8 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **RTX PRO 6000 Blackwell** | PyTorch | 3.74 | 3.14 | 2.60 | 3.74 | 3.15 | 2.62 | 3.74 | 3.14 | 2.59 |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |
| **H20** | PyTorch | 6.95 | 4.79 | 3.60 | 7.00 | 4.78 | 3.59 | 7.01 | 4.78 | 3.60 |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |
| **H100 NVL** | PyTorch | 2.50 | 2.11 | 2.12 | 2.51 | 2.11 | 2.13 | 2.55 | 2.17 | 2.16 |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |
| **H200 NVL** | PyTorch | 2.11 | 2.03 | 2.10 | 2.09 | 2.15 | 2.10 | 2.09 | 2.02 | 2.11 |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |
| **H100 80GB HBM3** | PyTorch | 1.99 | 2.12 | 2.20 | 1.99 | 2.16 | 2.19 | 1.99 | 2.11 | 2.23 |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |
| **H200 141GB HBM3** | PyTorch | 1.93 | 2.10 | 2.21 | 1.93 | 2.10 | 2.16 | 1.94 | 2.11 | 2.10 |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |
| **B200** | PyTorch | 1.30 | 1.91 | 1.95 | 1.30 | 1.92 | 1.91 | 1.30 | 1.90 | 2.00 |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |
| **B300** | PyTorch |  |  |  |  |  |  |  |  |  |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |

### Policy — DROID

These rows report **Cosmos3-Nano-Policy-DROID**, not the general Cosmos3-Nano checkpoint used in the other action tables. vLLM-Omni only supports this modality at **480p on one GPU** (multi-GPU is disabled, and resolution is fixed for the multiview wrist/exterior inputs). PyTorch and Diffusers cells are reserved.

| GPU | Engine | 256p/1 | 256p/4 | 256p/8 | 480p/1 | 480p/4 | 480p/8 | 720p/1 | 720p/4 | 720p/8 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **RTX PRO 6000 Blackwell** | PyTorch | | | | | | | | | |
| | vLLM-Omni | | | | 3.82 | | | | | |
| | Diffusers | | | | | | | | | |
| **H20** | PyTorch | | | | | | | | | |
| | vLLM-Omni | | | | 8.51 | | | | | |
| | Diffusers | | | | | | | | | |
| **H100 NVL** | PyTorch | | | | | | | | | |
| | vLLM-Omni | | | | 2.94 | | | | | |
| | Diffusers | | | | | | | | | |
| **H200 NVL** | PyTorch | | | | | | | | | |
| | vLLM-Omni | | | | 2.34 | | | | | |
| | Diffusers | | | | | | | | | |
| **H100 80GB HBM3** | PyTorch | | | | | | | | | |
| | vLLM-Omni | | | | 2.23 | | | | | |
| | Diffusers | | | | | | | | | |
| **H200 141GB HBM3** | PyTorch | | | | | | | | | |
| | vLLM-Omni | | | | 2.21 | | | | | |
| | Diffusers | | | | | | | | | |
| **B200** | PyTorch | | | | | | | | | |
| | vLLM-Omni | | | | 1.47 | | | | | |
| | Diffusers | | | | | | | | | |
| **B300** | PyTorch | | | | | | | | | |
| | vLLM-Omni | | | | 2.15 | | | | | |
| | Diffusers | | | | | | | | | |

<sub>Additional-modality notes:
1. PyTorch values are average generation (sampling) latency in seconds from PBR `#308197`; lower is better.
2. PyTorch values use `CUDA_GRAPH=No` and the `latency` automatic-sharding preset.
3. vLLM-Omni audiovisual values for `t2av`/`t2vs` and `i2av`/`i2vs` are from PBR `#308195`. vLLM-Omni action values are from PBR `#308481`.
4. Action vLLM-Omni `/8` cells are empty because 8-GPU Ulysses runs failed sequence-length divisibility checks. Two documented action misses are also left blank: Super inverse-dynamics AV at 720p/4 on H200 141GB HBM3 (OOM), and Nano inverse-dynamics AV at 720p/4 on H100 NVL (CUDA/NCCL failure).
5. Policy-DROID vLLM-Omni uses the `nvidia/Cosmos3-Nano-Policy-DROID` checkpoint at 480p/1 only. It is reported in its own table and is not comparable to the PyTorch policy-robot row.
6. Diffusers rows are intentionally empty reservations for future benchmark campaigns.
7. The `/1`, `/4`, and `/8` suffixes denote the number of GPUs used by the benchmark run.
8. Empty cells indicate unmeasured combinations, not unsupported combinations.</sub>
