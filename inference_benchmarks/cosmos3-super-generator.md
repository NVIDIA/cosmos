# Cosmos3-Super Generator Inference Benchmarks

[Back to the inference benchmark index](../inference_benchmarks.md)

These tables report **Cosmos3-Super Generator** latency in seconds. Lower is better. Empty cells mean that a run has not been completed for that GPU, engine, resolution, or GPU count; they do not indicate that a combination is unsupported.

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

## Benchmark methodology

The primary t2v, i2v, and t2i tables preserve the previously published benchmark campaigns across PyTorch, vLLM-Omni, Diffusers, and NIM. Those tables use BF16 precision, batch size 1, and matched prompts, seeds, and sampler settings where documented. Video workloads follow the standard Cosmos3 generation profile of 189 frames at 24 FPS unless a resolution tier limits frame count.

The additional audiovisual and action tables come from PBR `#308197`, **Cosmos3-Generator OSS Inference Benchmarking 32B and 8B (189 frames)**. PyTorch values are average generation (sampling) latency from the native OSS path, using **CUDA Graphs disabled** and the **latency** automatic-sharding preset. Columns combine the 256p, 480p, and 720p tiers with 1, 4, or 8 GPUs. Values are rounded to two decimal places. The workbook does not include B300, so B300 is retained as an empty reservation.

The PBR establishes the reported timing matrix but does not expose every prompt and action payload in this repository. The linked public recipes explain modality behavior and provide representative payloads; their example-specific frame counts and action chunk sizes should not be treated as the exact internal benchmark inputs.

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

The PBR uses `t2av`, `v2av`, and `i2av`; some public recipes call the same sound-producing modes `t2vs`, `v2vs`, and `i2vs`. See the [audiovisual cookbook](../cookbooks/cosmos3/generator/audiovisual/README.md) for generation inputs and the [action cookbook](../cookbooks/cosmos3/generator/action/README.md) for action representations and output contracts. vLLM-Omni request shapes are maintained in the [Super recipe](https://github.com/vllm-project/vllm-omni/blob/main/recipes/cosmos3/Cosmos3-Super.md). Public recipes cover more embodiments than this PBR; the action tables below intentionally use only its measured domain rows.

## Primary vision generation

### Text-to-Video (t2v)

| GPU | Engine | 256p/1 | 256p/4 | 256p/8 | 480p/1 | 480p/4 | 480p/8 | 720p/1 | 720p/4 | 720p/8 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **RTX PRO 6000 Blackwell** | PyTorch |  | 65.10 | 66.04 |  | 201.16 | 118.90 |  | 789.03 | 427.16 |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |
| | NIM |  | 12.65 | 13.99 |  | 104.25 | 99.05 |  | 350.74 | 286.02 |
| **H20** | PyTorch | | 40.74 | 27.72 | | 276.24 | 152.46 | | 930.45 | 492.41 |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |
| | NIM |  | 20.07 | 12.95 |  | 192.45 | 110.71 |  | 734.37 | 395.56 |
| **H100 NVL** | PyTorch |  | 20.73 | 16.83 |  | 101.27 | 64.14 |  | 330.04 | 186.19 |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |
| | NIM |  | 8.77 | 12.73 |  | 73.37 | 66.07 |  | 267.64 | 197.32 |
| **H200 NVL** | PyTorch | | 24.90 | 24.78 | | 82.19 | 47.69 | | 267.66 | 142.35 |
| | vLLM-Omni | 27.54 | | 5.06 | 252.33 | | 36.66 | 911.49 | 245.51 | 123.85 |
| | Diffusers | 33.00 | | | 286.80 | | | 1036.00 | | |
| | NIM | 17.13 | 6.79 | 4.43 | 200.00 | 58.55 | 32.87 | 811.41 | 223.00 | 117.98 |
| **H100 80GB HBM3** | PyTorch | | | | | | | | | |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |
| | NIM |  | 6.98 | 5.89 |  | 55.10 | 35.52 |  | 198.13 | 114.92 |
| **H200 141GB HBM3** | PyTorch |  | 14.82 | 11.82 |  | 70.27 | 41.78 |  | 224.43 | 123.49 |
| | vLLM-Omni | 25.61 | | 5.87 | 219.11 | | 35.26 | 769.63 | 212.30 | 111.94 |
| | Diffusers | 31.00 | | | 251.60 | | | 886.20 | | |
| | NIM | 15.95 | 6.14 | 4.28 | 174.71 | 52.94 | 30.94 | 695.89 | 194.34 | 106.16 |
| **B200** | PyTorch | 14.66 | 5.59 | 4.09 | 114.38 | 35.73 | 21.39 | 407.50 | 118.38 | 65.93 |
| | vLLM-Omni | 13.84 | | 4.76 | 114.08 | | 22.09 | 390.28 | 113.31 | 62.11 |
| | Diffusers | | | | 127.20 | | | 414.40 | | |
| | NIM | 9.09 | 4.26 | 3.38 | 82.39 | 27.83 | 17.74 | 314.68 | 92.25 | 53.43 |
| **B300** | PyTorch | | | | | | | | | |
| | vLLM-Omni | 14.57 | | 6.68 | 109.03 | | 22.67 | 366.66 | 108.58 | 60.73 |
| | Diffusers | 54.20 | | | 155.40 | | | 424.80 | | |
| | NIM | 9.67 | 5.23 | 5.19 | 79.73 | 28.97 | 18.39 | 292.35 | 92.31 | 54.07 |

### Image-to-Video (i2v)

| GPU | Engine | 256p/1 | 256p/4 | 256p/8 | 480p/1 | 480p/4 | 480p/8 | 720p/1 | 720p/4 | 720p/8 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **RTX PRO 6000 Blackwell** | PyTorch |  | 65.42 | 66.54 |  | 202.26 | 118.95 |  | 795.14 | 427.96 |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |
| | NIM |  | 13.17 | 14.48 |  | 106.50 | 100.23 |  | 356.60 | 289.93 |
| **H20** | PyTorch | | 41.37 | 28.22 | | 278.23 | 153.19 | | 931.74 | 491.93 |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |
| | NIM |  | 21.16 | 14.06 |  | 196.86 | 114.49 |  | 745.12 | 405.68 |
| **H100 NVL** | PyTorch |  | 20.85 | 16.96 |  | 99.56 | 64.17 |  | 331.40 | 186.47 |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |
| | NIM |  | 9.32 | 13.30 |  | 75.31 | 67.34 |  | 271.46 | 201.39 |
| **H200 NVL** | PyTorch |  | 24.81 | 24.70 |  | 82.51 | 47.60 |  | 269.85 | 141.62 |
| | vLLM-Omni | 27.90 | | 5.52 | 254.29 | | 38.51 | 915.05 | 248.89 | 127.32 |
| | Diffusers | 33.00 | | | 287.20 | | | 1034.60 | | |
| | NIM | 17.51 | 7.43 | 5.04 | 201.45 | 60.15 | 34.48 | 817.35 | 226.38 | 121.35 |
| **H100 80GB HBM3** | PyTorch | | | | | | | | | |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |
| | NIM |  | 7.50 | 6.49 |  | 56.33 | 36.81 |  | 200.97 | 118.77 |
| **H200 141GB HBM3** | PyTorch |  | 14.87 | 11.80 |  | 70.45 | 42.10 |  | 224.36 | 123.57 |
| | vLLM-Omni | 25.47 | | 6.32 | 220.70 | | 36.90 | 766.33 | 215.03 | 117.52 |
| | Diffusers | 31.00 | | | 249.20 | | | 879.20 | | |
| | NIM | 16.39 | 6.74 | 4.90 | 175.95 | 54.46 | 32.51 | 699.13 | 197.96 | 109.55 |
| **B200** | PyTorch | 14.71 | 5.63 | 4.12 | 112.40 | 35.70 | 21.25 | 397.31 | 117.98 | 65.91 |
| | vLLM-Omni | 14.13 | | 5.31 | 115.17 | | 23.26 | 393.02 | 115.69 | 64.82 |
| | Diffusers | 19.20 | | | | | | 414.80 | | |
| | NIM | 9.36 | 4.83 | 4.12 | 83.19 | 29.09 | 19.14 | 316.76 | 94.65 | 55.92 |
| **B300** | PyTorch | | | | | | | | | |
| | vLLM-Omni | 14.14 | | 7.19 | 111.42 | | 23.91 | 368.73 | 111.41 | 63.25 |
| | Diffusers | 54.20 | | | 151.80 | | | 425.00 | | |
| | NIM | 9.73 | 5.58 | 5.94 | 80.51 | 30.17 | 20.62 | 294.11 | 93.77 | 56.76 |

### Text-to-Image (t2i)

| GPU | Engine | 256p/1 | 256p/4 | 256p/8 | 480p/1 | 480p/4 | 480p/8 | 720p/1 | 720p/4 | 720p/8 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **RTX PRO 6000 Blackwell** | PyTorch | | 93.20 | 93.18 | | 92.22 | 93.39 | | 92.89 | 93.47 |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |
| **H20** | PyTorch |  | 14.58 | 14.18 |  | 19.93 | 16.46 |  | 26.26 | 20.92 |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |
| **H100 NVL** | PyTorch | | 19.73 | 19.86 | | 19.78 | 19.80 | | 20.68 | 19.87 |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |
| **H200 NVL** | PyTorch |  | 32.51 | 32.86 |  | 32.53 | 33.05 |  | 32.64 | 33.16 |
| | vLLM-Omni | 2.73 | | 3.20 | 6.28 | | 17.71 | 11.02 | 4.22 | 3.26 |
| | Diffusers | 5.00 | | | 8.00 | | | 12.00 | | |
| **H100 80GB HBM3** | PyTorch | | | | | | | | | |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |
| **H200 141GB HBM3** | PyTorch |  | 13.62 | 13.48 |  | 13.33 | 13.53 |  | 13.78 | 13.50 |
| | vLLM-Omni | 2.83 | | 4.50 | 5.70 | | 7.16 | 10.24 | 4.23 | 4.43 |
| | Diffusers | 5.00 | | | 8.00 | | | 11.00 | | |
| **B200** | PyTorch | 4.51 | 4.10 | 4.27 | 4.78 | 4.13 | 4.48 | 7.25 | 4.28 | 4.65 |
| | vLLM-Omni | 2.32 | | 4.58 | 3.29 | | 9.10 | 6.02 | 3.09 | 4.43 |
| | Diffusers | | | | | | | 8.00 | | |
| **B300** | PyTorch | | | | | | | | | |
| | vLLM-Omni | 5.05 | | 7.62 | 3.79 | | 72.65 | 7.08 | 5.79 | 7.24 |
| | Diffusers | 38.80 | | | 39.40 | | | 40.40 | | |

<sub>Notes:
1. All times measured on identical workloads (same seed, sampler settings, prompt).
2. 4×/8× GPU configurations use tensor parallelism.
3. vLLM-Omni numbers are for the upcoming public release in the vLLM-Omni repo; subject to change before GA. Current vLLM-Omni coverage is B200 at 720p.
4. Diffusers numbers use the HuggingFace `diffusers` integration without custom CUDA graphs; reported at 256p/1, 480p/1, and 720p/1 (single-GPU only).
5. At 256p, multi-GPU configurations on B300 may underperform single-GPU due to small-workload TP overhead; single-GPU is recommended at this resolution.
6. PyTorch numbers report average generation (sampling) time from OSS inference benchmarking.
7. NIM numbers use latency profiles with FP8 precision and report end-to-end `Request Latency s`, including request processing, video generation, output encoding, and returning the response.</sub>

## Additional audiovisual generation

### Video-to-Video (v2v)

A text prompt and source video condition a generated continuation or transformation.

| GPU | Engine | 256p/1 | 256p/4 | 256p/8 | 480p/1 | 480p/4 | 480p/8 | 720p/1 | 720p/4 | 720p/8 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **RTX PRO 6000 Blackwell** | PyTorch |  | 64.80 | 66.31 |  | 201.59 | 118.92 |  | 787.37 | 427.19 |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |
| **H20** | PyTorch |  | 40.93 | 27.98 |  | 277.39 | 153.02 |  | 931.12 | 491.75 |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |
| **H100 NVL** | PyTorch |  | 20.75 | 16.79 |  | 99.57 | 63.84 |  | 328.83 | 183.64 |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |
| **H200 NVL** | PyTorch |  | 24.87 | 24.74 |  | 82.22 | 47.60 |  | 268.16 | 142.23 |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |
| **H100 80GB HBM3** | PyTorch |  |  |  |  |  |  |  |  |  |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |
| **H200 141GB HBM3** | PyTorch |  | 14.66 | 11.76 |  | 70.52 | 41.85 |  | 223.86 | 123.09 |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |
| **B200** | PyTorch | 14.30 | 5.59 | 4.07 | 111.49 | 35.68 | 21.43 | 395.97 | 117.46 | 65.62 |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |
| **B300** | PyTorch |  |  |  |  |  |  |  |  |  |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |

### Text-to-Audio-and-Video (t2av)

A text prompt produces synchronized video and sound.

| GPU | Engine | 256p/1 | 256p/4 | 256p/8 | 480p/1 | 480p/4 | 480p/8 | 720p/1 | 720p/4 | 720p/8 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **RTX PRO 6000 Blackwell** | PyTorch |  | 65.38 | 65.79 |  | 202.34 | 118.87 |  | 788.88 | 429.78 |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |
| **H20** | PyTorch |  | 40.69 | 27.73 |  | 277.33 | 152.53 |  | 930.63 | 492.38 |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |
| **H100 NVL** | PyTorch |  | 20.78 | 16.86 |  | 99.28 | 64.17 |  | 329.26 | 183.81 |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |
| **H200 NVL** | PyTorch |  | 24.88 | 24.79 |  | 82.49 | 47.67 |  | 267.86 | 142.06 |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |
| **H100 80GB HBM3** | PyTorch |  |  |  |  |  |  |  |  |  |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |
| **H200 141GB HBM3** | PyTorch |  | 15.05 | 11.74 |  | 70.09 | 41.49 |  | 223.96 | 123.35 |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |
| **B200** | PyTorch | 14.52 | 5.64 | 4.07 | 112.99 | 35.93 | 21.57 | 395.20 | 118.67 | 65.93 |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |
| **B300** | PyTorch |  |  |  |  |  |  |  |  |  |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |

### Video-to-Audio-and-Video (v2av)

A text prompt and source video produce transformed video with synchronized sound.

| GPU | Engine | 256p/1 | 256p/4 | 256p/8 | 480p/1 | 480p/4 | 480p/8 | 720p/1 | 720p/4 | 720p/8 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **RTX PRO 6000 Blackwell** | PyTorch |  | 64.67 | 66.10 |  | 202.31 | 118.88 |  | 785.77 | 428.64 |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |
| **H20** | PyTorch |  | 40.89 | 28.00 |  | 276.26 | 152.78 |  | 930.81 | 492.63 |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |
| **H100 NVL** | PyTorch |  | 20.65 | 16.79 |  | 99.90 | 64.69 |  | 323.95 | 183.62 |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |
| **H200 NVL** | PyTorch |  | 24.88 | 24.73 |  | 82.36 | 47.67 |  | 268.78 | 143.06 |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |
| **H100 80GB HBM3** | PyTorch |  |  |  |  |  |  |  |  |  |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |
| **H200 141GB HBM3** | PyTorch |  | 14.85 | 11.74 |  | 70.42 | 41.59 |  | 223.23 | 123.06 |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |
| **B200** | PyTorch | 14.28 | 5.60 | 4.07 | 111.84 | 35.64 | 21.48 | 406.66 | 117.16 | 65.30 |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |
| **B300** | PyTorch |  |  |  |  |  |  |  |  |  |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |

### Image-to-Audio-and-Video (i2av)

A text prompt and source image produce video with synchronized sound.

| GPU | Engine | 256p/1 | 256p/4 | 256p/8 | 480p/1 | 480p/4 | 480p/8 | 720p/1 | 720p/4 | 720p/8 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **RTX PRO 6000 Blackwell** | PyTorch |  | 65.54 | 66.27 |  | 202.65 | 119.05 |  | 790.09 | 429.26 |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |
| **H20** | PyTorch |  | 41.33 | 28.11 |  | 277.13 | 153.33 |  | 931.18 | 493.51 |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |
| **H100 NVL** | PyTorch |  | 20.83 | 16.85 |  | 99.62 | 64.19 |  | 329.80 | 186.66 |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |
| **H200 NVL** | PyTorch |  | 24.81 | 24.69 |  | 82.65 | 47.65 |  | 269.74 | 142.18 |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |
| **H100 80GB HBM3** | PyTorch |  |  |  |  |  |  |  |  |  |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |
| **H200 141GB HBM3** | PyTorch |  | 15.09 | 11.82 |  | 69.80 | 41.95 |  | 224.65 | 123.42 |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |
| **B200** | PyTorch | 14.84 | 5.64 | 4.12 | 113.32 | 35.84 | 21.70 | 407.30 | 117.95 | 65.20 |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |
| **B300** | PyTorch |  |  |  |  |  |  |  |  |  |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |

## Action generation

Forward dynamics is reported separately for AV, camera, and robot inputs. Inverse dynamics and policy are reported for AV and robot because those are the only domain rows in this PBR; no camera row is inferred. For each domain, PyTorch is populated from the PBR while vLLM-Omni and Diffusers rows are reserved for future measurements.

### Forward Dynamics — Autonomous Vehicle (AV)

| GPU | Engine | 256p/1 | 256p/4 | 256p/8 | 480p/1 | 480p/4 | 480p/8 | 720p/1 | 720p/4 | 720p/8 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **RTX PRO 6000 Blackwell** | PyTorch |  | 79.50 | 59.39 |  | 61.21 | 59.31 |  | 61.15 | 59.35 |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |
| **H20** | PyTorch |  | 67.73 | 41.51 |  | 67.83 | 41.47 |  | 67.85 | 41.42 |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |
| **H100 NVL** | PyTorch |  | 28.53 | 21.00 |  | 28.69 | 21.00 |  | 28.56 | 21.04 |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |
| **H200 NVL** | PyTorch |  | 24.40 | 22.66 |  | 24.38 | 22.67 |  | 24.46 | 22.65 |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |
| **H100 80GB HBM3** | PyTorch |  |  |  |  |  |  |  |  |  |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |
| **H200 141GB HBM3** | PyTorch |  | 19.88 | 14.41 |  | 19.87 | 14.41 |  | 19.88 | 14.41 |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |
| **B200** | PyTorch | 13.59 | 9.29 | 6.33 | 13.55 | 9.29 | 6.37 | 13.86 | 9.32 | 6.34 |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |
| **B300** | PyTorch |  |  |  |  |  |  |  |  |  |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |

### Forward Dynamics — Camera

| GPU | Engine | 256p/1 | 256p/4 | 256p/8 | 480p/1 | 480p/4 | 480p/8 | 720p/1 | 720p/4 | 720p/8 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **RTX PRO 6000 Blackwell** | PyTorch |  | 60.75 | 59.62 |  | 60.69 | 59.39 |  | 60.61 | 59.55 |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |
| **H20** | PyTorch |  | 67.86 | 41.13 |  | 67.89 | 41.10 |  | 67.88 | 41.11 |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |
| **H100 NVL** | PyTorch |  | 28.73 | 21.05 |  | 28.76 | 20.98 |  | 28.65 | 21.00 |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |
| **H200 NVL** | PyTorch |  | 24.33 | 22.55 |  | 24.35 | 22.54 |  | 24.37 | 22.54 |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |
| **H100 80GB HBM3** | PyTorch |  |  |  |  |  |  |  |  |  |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |
| **H200 141GB HBM3** | PyTorch |  | 19.85 | 14.30 |  | 19.92 | 14.30 |  | 20.29 | 14.29 |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |
| **B200** | PyTorch | 13.73 | 9.32 | 6.27 | 13.72 | 9.27 | 6.27 | 13.58 | 9.28 | 6.27 |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |
| **B300** | PyTorch |  |  |  |  |  |  |  |  |  |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |

### Forward Dynamics — Robot

| GPU | Engine | 256p/1 | 256p/4 | 256p/8 | 480p/1 | 480p/4 | 480p/8 | 720p/1 | 720p/4 | 720p/8 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **RTX PRO 6000 Blackwell** | PyTorch |  | 75.52 | 56.35 |  | 55.04 | 56.38 |  | 54.88 | 56.01 |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |
| **H20** | PyTorch |  | 20.93 | 15.60 |  | 20.91 | 15.61 |  | 20.78 | 15.59 |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |
| **H100 NVL** | PyTorch |  | 13.61 | 12.90 |  | 13.68 | 12.90 |  | 13.57 | 12.90 |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |
| **H200 NVL** | PyTorch |  | 20.25 | 20.16 |  | 20.25 | 20.15 |  | 20.25 | 20.15 |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |
| **H100 80GB HBM3** | PyTorch |  |  |  |  |  |  |  |  |  |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |
| **H200 141GB HBM3** | PyTorch |  | 9.34 | 8.74 |  | 9.34 | 8.74 |  | 9.33 | 8.70 |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |
| **B200** | PyTorch | 3.50 | 2.91 | 2.98 | 3.49 | 3.06 | 3.00 | 3.47 | 2.96 | 3.00 |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |
| **B300** | PyTorch |  |  |  |  |  |  |  |  |  |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |

### Inverse Dynamics — Autonomous Vehicle (AV)

| GPU | Engine | 256p/1 | 256p/4 | 256p/8 | 480p/1 | 480p/4 | 480p/8 | 720p/1 | 720p/4 | 720p/8 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **RTX PRO 6000 Blackwell** | PyTorch |  | 60.92 | 59.20 |  | 60.87 | 59.14 |  | 60.95 | 59.07 |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |
| **H20** | PyTorch |  | 67.37 | 41.17 |  | 67.09 | 40.96 |  | 67.37 | 41.06 |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |
| **H100 NVL** | PyTorch |  | 28.42 | 20.76 |  | 28.39 | 21.05 |  | 28.41 | 20.69 |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |
| **H200 NVL** | PyTorch |  | 24.16 | 22.40 |  | 24.17 | 22.40 |  | 24.18 | 22.42 |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |
| **H100 80GB HBM3** | PyTorch |  |  |  |  |  |  |  |  |  |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |
| **H200 141GB HBM3** | PyTorch |  | 20.05 | 14.30 |  | 19.68 | 14.32 |  | 19.91 | 14.16 |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |
| **B200** | PyTorch | 13.38 | 9.08 | 6.11 | 13.37 | 9.13 | 6.15 | 13.36 | 9.08 | 6.11 |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |
| **B300** | PyTorch |  |  |  |  |  |  |  |  |  |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |

### Inverse Dynamics — Robot

| GPU | Engine | 256p/1 | 256p/4 | 256p/8 | 480p/1 | 480p/4 | 480p/8 | 720p/1 | 720p/4 | 720p/8 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **RTX PRO 6000 Blackwell** | PyTorch |  | 55.15 | 56.28 |  | 55.12 | 56.33 |  | 54.83 | 56.00 |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |
| **H20** | PyTorch |  | 20.71 | 15.51 |  | 20.83 | 15.52 |  | 20.84 | 15.51 |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |
| **H100 NVL** | PyTorch |  | 13.51 | 12.51 |  | 13.54 | 12.51 |  | 13.50 | 12.51 |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |
| **H200 NVL** | PyTorch |  | 20.21 | 20.11 |  | 20.18 | 20.10 |  | 20.19 | 20.12 |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |
| **H100 80GB HBM3** | PyTorch |  |  |  |  |  |  |  |  |  |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |
| **H200 141GB HBM3** | PyTorch |  | 9.30 | 8.68 |  | 9.28 | 8.68 |  | 9.29 | 8.65 |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |
| **B200** | PyTorch | 3.42 | 2.89 | 2.96 | 3.49 | 3.02 | 2.92 | 3.41 | 2.90 | 3.03 |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |
| **B300** | PyTorch |  |  |  |  |  |  |  |  |  |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |

### Policy — Autonomous Vehicle (AV)

| GPU | Engine | 256p/1 | 256p/4 | 256p/8 | 480p/1 | 480p/4 | 480p/8 | 720p/1 | 720p/4 | 720p/8 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **RTX PRO 6000 Blackwell** | PyTorch |  | 61.03 | 59.66 |  | 61.19 | 59.51 |  | 61.04 | 59.49 |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |
| **H20** | PyTorch |  | 67.86 | 41.54 |  | 67.56 | 41.51 |  | 67.62 | 41.49 |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |
| **H100 NVL** | PyTorch |  | 28.61 | 21.02 |  | 28.77 | 21.00 |  | 28.54 | 21.31 |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |
| **H200 NVL** | PyTorch |  | 24.40 | 22.64 |  | 24.46 | 22.63 |  | 24.47 | 22.64 |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |
| **H100 80GB HBM3** | PyTorch |  |  |  |  |  |  |  |  |  |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |
| **H200 141GB HBM3** | PyTorch |  | 20.31 | 14.54 |  | 19.87 | 14.41 |  | 20.14 | 14.52 |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |
| **B200** | PyTorch | 13.71 | 9.33 | 6.36 | 13.73 | 9.31 | 6.34 | 13.60 | 9.32 | 6.34 |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |
| **B300** | PyTorch |  |  |  |  |  |  |  |  |  |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |

### Policy — Robot

| GPU | Engine | 256p/1 | 256p/4 | 256p/8 | 480p/1 | 480p/4 | 480p/8 | 720p/1 | 720p/4 | 720p/8 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **RTX PRO 6000 Blackwell** | PyTorch |  | 55.13 | 56.31 |  | 55.47 | 56.29 |  | 54.92 | 55.95 |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |
| **H20** | PyTorch |  | 20.90 | 15.60 |  | 20.88 | 15.60 |  | 20.92 | 15.62 |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |
| **H100 NVL** | PyTorch |  | 13.57 | 12.56 |  | 13.58 | 12.55 |  | 13.60 | 12.54 |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |
| **H200 NVL** | PyTorch |  | 20.25 | 20.16 |  | 20.26 | 20.17 |  | 20.26 | 20.16 |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |
| **H100 80GB HBM3** | PyTorch |  |  |  |  |  |  |  |  |  |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |
| **H200 141GB HBM3** | PyTorch |  | 9.33 | 8.72 |  | 9.35 | 8.75 |  | 9.34 | 8.70 |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |
| **B200** | PyTorch | 3.48 | 2.95 | 3.07 | 3.49 | 3.02 | 2.98 | 3.48 | 2.92 | 3.05 |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |
| **B300** | PyTorch |  |  |  |  |  |  |  |  |  |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |

<sub>Additional-modality notes:
1. All reported values are average PyTorch generation (sampling) latency in seconds; lower is better.
2. PyTorch values use `CUDA_GRAPH=No` and the `latency` automatic-sharding preset.
3. The `/1`, `/4`, and `/8` suffixes denote the number of GPUs used by the benchmark run.
4. vLLM-Omni and Diffusers rows are intentionally empty reservations for future benchmark campaigns.
5. Empty cells indicate unmeasured combinations, not unsupported combinations.</sub>
