# Inference Benchmarks

These tables collect inference benchmarks for Cosmos3. **Generator** sections measure diffusion-path latency (image and video generation via i2v, t2i, and t2v) across PyTorch, vLLM-Omni, and Diffusers. The **Reasoner** section measures VLM serving metrics (TTFT, request latency, throughput) for text outputs from vision and text inputs via vLLM.

Generator results are published incrementally from internal benchmark runs. **Empty cells mean that combination has not been measured yet** — not that it is unsupported. See the notes under each table for workload details and data-source definitions.

## Table of Contents

- [Cosmos3-Nano Generator](#cosmos3-nano-generator)
  - [Text-to-Video (t2v)](#text-to-video-t2v)
  - [Image-to-Video (i2v)](#image-to-video-i2v)
  - [Text-to-Image (t2i)](#text-to-image-t2i)
- [Cosmos3-Super Generator](#cosmos3-super-generator)
  - [Text-to-Video (t2v)](#text-to-video-t2v-1)
  - [Image-to-Video (i2v)](#image-to-video-i2v-1)
  - [Text-to-Image (t2i)](#text-to-image-t2i-1)
- [Cosmos3-Nano Reasoner](#cosmos3-nano-reasoner)
  - [RTX PRO 6000 Blackwell](#rtx-pro-6000-blackwell)
  - [H20](#h20)
  - [H100 NVL](#h100-nvl)
  - [H200 NVL](#h200-nvl)
  - [H100 80GB HBM3 (SXM)](#h100-80gb-hbm3-sxm)
  - [H200 141GB HBM3](#h200-141gb-hbm3)
  - [B200](#b200)
  - [B300](#b300)
- [Cosmos3-Super Reasoner](#cosmos3-super-reasoner)
  - [RTX PRO 6000 Blackwell](#rtx-pro-6000-blackwell-1)
  - [H20](#h20-1)
  - [H100 NVL](#h100-nvl-1)
  - [H200 NVL](#h200-nvl-1)
  - [H200 141GB HBM3](#h200-141gb-hbm3-1)
  - [B200](#b200-1)
  - [B300](#b300-1)

## Cosmos3-Nano Generator

These tables report **Cosmos3-Nano** generator latency in seconds for **image-to-video (i2v)**, **text-to-image (t2i)**, and **text-to-video (t2v)** - the primary vision-generation modes of the omni-model. Benchmarks use BF16 precision, batch size 1, and identical prompts, seeds, and sampler settings across engines where noted below. Video workloads follow the standard Cosmos3 generation profile (189 frames at 24 FPS unless a resolution tier limits frame count).

Three integration paths are compared. **PyTorch** reports average generation (sampling) time from OSS reference inference with CUDA Graphs enabled where supported. **vLLM-Omni** reports total pipeline time at **720p** on supported GPUs. **Diffusers** reports end-to-end generation time through the Hugging Face `Cosmos3OmniPipeline` without custom CUDA graphs at **256p/1**, **480p/1**, and **720p/1** (320×192, 832×480, and 1280×720). Empty cells indicate a run has not been completed for that GPU, engine, resolution, or tensor-parallel width - tables are filled in as benchmark campaigns finish.

### Text-to-Video (t2v)

| GPU | Engine | 256p/1 | 256p/4 | 256p/8 | 480p/1 | 480p/4 | 480p/8 | 720p/1 | 720p/4 | 720p/8 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **RTX PRO 6000 Blackwell** | PyTorch | | | | | | | 786.37 | 225.45 | |
| | vLLM-Omni | | | | | | | 369.67 | 114.30 | 68.66 |
| | Diffusers | 11.20 | | | 112.00 | | | 392.00 | | |
| **H20** | PyTorch | | | | | | | 931.39 | 268.88 | 157.71 |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | 30.20 | | | 258.00 | | | 926.00 | | |
| **H100 NVL** | PyTorch | | | 3.95 | 84.12 | | | 297.27 | 94.15 | 61.63 |
| | vLLM-Omni | | | | | | | 311.13 | 88.25(*) | 54.01(*) |
| | Diffusers | 11.00 | | | 90.00 | | | 324.20 | | |
| **H200 NVL** | PyTorch | | | | | | | 244.39 | 77.35 | 45.70 |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | 9.00 | | | 74.00 | | | 276.20 | | |
| **H100 80GB HBM3** | PyTorch | 7.61 | | | 59.83 | | | 207.78 | | |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | 9.00 | | | 68.00 | | | 240.00 | | |
| **H200 141GB HBM3** | PyTorch | | 3.34 | 3.19 | | | 13.97 | 214.28 | 67.48 | 41.26 |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | 9.00 | | | 67.00 | | | 239.60 | | |
| **B200** | PyTorch | 4.56 | 2.78 | 2.79 | | | | 114.85 | 39.75 | 26.27 |
| | vLLM-Omni | | | | | | | 107.84 | 35.29 | 22.87 |
| | Diffusers | 7.00 | | | 36.80 | | | 117.00 | | |
| **B300** | PyTorch | | | | | | | | | |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | 39.40 | | | 63.40 | | | 139.40 | | |

### Image-to-Video (i2v)

| GPU | Engine | 256p/1 | 256p/4 | 256p/8 | 480p/1 | 480p/4 | 480p/8 | 720p/1 | 720p/4 | 720p/8 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **RTX PRO 6000 Blackwell** | PyTorch | | | | | | | 788.80 | 226.25 | 127.79 |
| | vLLM-Omni | | | | | | | 375.01 | 119.27 | 73.57 |
| | Diffusers | 12.00 | | | 112.00 | | | 397.00 | | |
| **H20** | PyTorch | | | | | | | | | 158.10 |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | 31.00 | | | 258.00 | | | 925.00 | | |
| **H100 NVL** | PyTorch | | | 3.99 | 84.50 | 28.69 | | 298.57 | 95.76 | 60.58 |
| | vLLM-Omni | | | | | | | 286.33 | 92.23(*) | 58.02(*) |
| | Diffusers | 11.00 | | | 91.00 | | | 325.20 | | |
| **H200 NVL** | PyTorch | | | | | | | 246.62 | 77.69 | 45.99 |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | 9.00 | | | 74.00 | | | 275.20 | | |
| **H100 80GB HBM3** | PyTorch | 7.64 | | | | | | 207.87 | | |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | 9.00 | | | 68.00 | | | 239.80 | | |
| **H200 141GB HBM3** | PyTorch | | 3.37 | 3.17 | | | 14.07 | 214.80 | 67.14 | 41.00 |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | 9.00 | | | 67.20 | | | 240.00 | | |
| **B200** | PyTorch | 4.60 | 2.77 | 2.81 | | | 9.66 | 113.90 | 40.01 | 26.58 |
| | vLLM-Omni | | | | | | | 110.19 | 37.76 | 25.68 |
| | Diffusers | | | | | | | 116.00 | | |
| **B300** | PyTorch | | | | | | | | | |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | 28.60 | | | 65.60 | | | 139.60 | | |

### Text-to-Image (t2i)

| GPU | Engine | 256p/1 | 256p/4 | 256p/8 | 480p/1 | 480p/4 | 480p/8 | 720p/1 | 720p/4 | 720p/8 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **RTX PRO 6000 Blackwell** | PyTorch | | | | | | | 7.12 | 3.18 | 2.70 |
| | vLLM-Omni | | | | | | | 4.99 | 2.32 | 1.96 |
| | Diffusers | 2.00 | | | 4.00 | | | 5.00 | | |
| **H20** | PyTorch | | | | | | | | | |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | 3.00 | | | 6.00 | | | 10.00 | | |
| **H100 NVL** | PyTorch | | 2.45 | | | | | 4.21 | 2.57 | 2.64 |
| | vLLM-Omni | | | | | | | 3.44 | 1.83 | 1.90 |
| | Diffusers | 3.00 | | | 3.00 | | | 4.00 | | |
| **H200 NVL** | PyTorch | | | | | | | 3.58 | 2.62 | 2.64 |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | 3.00 | | | 3.00 | | | 4.00 | | |
| **H100 80GB HBM3** | PyTorch | 3.01 | | | | | | 3.45 | | |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | 3.00 | | | 3.00 | | | 4.00 | | |
| **H200 141GB HBM3** | PyTorch | | 2.59 | 2.70 | | 2.78 | | 3.28 | 2.84 | 2.77 |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | 3.00 | | | 3.00 | | | 4.00 | | |
| **B200** | PyTorch | | | 2.59 | 2.75 | | 2.56 | 2.87 | 2.58 | 2.62 |
| | vLLM-Omni | | | | | | | 1.77 | 2.20 | 3.41 |
| | Diffusers | | | | | | | 3.00 | | |
| **B300** | PyTorch | | | | | | | | | |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | 36.20 | | | | | | 41.00 | | |

<sub>Notes:
1. All times measured on identical workloads (same seed, sampler settings, prompt).
2. 4×/8× GPU configurations use tensor parallelism.
3. vLLM-Omni numbers are for the upcoming public release in the vLLM-Omni repo; subject to change before GA. Values marked with (*) are pre-release vLLM-Omni measurements on H100 NVL and may change before GA.
4. Diffusers numbers use the HuggingFace `diffusers` integration without custom CUDA graphs; reported at 256p/1, 480p/1, and 720p/1 (single-GPU only).
5. PyTorch numbers report average generation (sampling) time from OSS inference benchmarking.
6. At 256p, multi-GPU configurations on B300 may underperform single-GPU due to small-workload TP overhead; single-GPU is recommended at this resolution.</sub>

## Cosmos3-Super Generator

These tables report **Cosmos3-Super** generator latency in seconds for **image-to-video (i2v)**, **text-to-image (t2i)**, and **text-to-video (t2v)**. The 32B checkpoint targets higher-quality world generation; expect longer runtimes than Nano at the same resolution. Benchmarks use BF16 precision, batch size 1, and matched prompts, seeds, and sampler settings. Video workloads follow the standard Cosmos3 profile (189 frames at 24 FPS where applicable).

As with Nano, three engines are tracked: **PyTorch** (OSS generation/sampling time), **vLLM-Omni** (total pipeline time at 720p on supported GPUs), and **Diffusers** (Hugging Face `Cosmos3OmniPipeline` end-to-end time at **256p/1**, **480p/1**, and **720p/1** — 320×192, 832×480, and 1280×720). Super coverage is narrower than Nano in early releases - for example, vLLM-Omni and Diffusers runs exist primarily on B200 and select H200 configurations. **Empty cells are pending measurements**, not unsupported configurations.

### Text-to-Video (t2v)

| GPU | Engine | 256p/1 | 256p/4 | 256p/8 | 480p/1 | 480p/4 | 480p/8 | 720p/1 | 720p/4 | 720p/8 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **RTX PRO 6000 Blackwell** | PyTorch | | | | | | | | | 427.16 |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |
| **H20** | PyTorch | | | | | | | | | 492.41 |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |
| **H100 NVL** | PyTorch | | | | | 101.27 | | | 330.04 | 186.19 |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |
| **H200 NVL** | PyTorch | | | | | | | | 258.34 | 139.37 |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | 33.00 | | | 286.80 | | | 1036.00 | | |
| **H100 80GB HBM3** | PyTorch | | | | | | | | | |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |
| **H200 141GB HBM3** | PyTorch | | | | | 70.27 | 41.78 | | 224.43 | 123.49 |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | 31.00 | | | 251.60 | | | 886.20 | | |
| **B200** | PyTorch | | 5.59 | | 114.38 | 35.73 | 21.39 | 407.50 | 118.38 | 65.93 |
| | vLLM-Omni | | | | | | | 390.28 | 113.31 | 62.11 |
| | Diffusers | | | | 127.20 | | | 414.40 | | |
| **B300** | PyTorch | | | | | | | | | |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | 54.20 | | | 155.40 | | | 424.80 | | |

### Image-to-Video (i2v)

| GPU | Engine | 256p/1 | 256p/4 | 256p/8 | 480p/1 | 480p/4 | 480p/8 | 720p/1 | 720p/4 | 720p/8 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **RTX PRO 6000 Blackwell** | PyTorch | | | | | | | | 795.14 | |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |
| **H20** | PyTorch | | | | | | | | 931.74 | |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |
| **H100 NVL** | PyTorch | | | 16.96 | | | | | 331.40 | 186.47 |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |
| **H200 NVL** | PyTorch | | | | | | | | | 138.31 |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | 33.00 | | | 287.20 | | | 1034.60 | | |
| **H100 80GB HBM3** | PyTorch | | | | | | | | | |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |
| **H200 141GB HBM3** | PyTorch | | | | | | | | 224.36 | 123.57 |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | 31.00 | | | 249.20 | | | 879.20 | | |
| **B200** | PyTorch | 14.71 | 5.63 | 4.12 | | | 21.25 | 397.31 | 117.98 | 65.91 |
| | vLLM-Omni | | | | | | | 393.02 | 115.69 | 64.82 |
| | Diffusers | 19.20 | | | | | | 414.80 | | |
| **B300** | PyTorch | | | | | | | | | |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | 54.20 | | | 151.80 | | | 425.00 | | |

### Text-to-Image (t2i)

| GPU | Engine | 256p/1 | 256p/4 | 256p/8 | 480p/1 | 480p/4 | 480p/8 | 720p/1 | 720p/4 | 720p/8 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **RTX PRO 6000 Blackwell** | PyTorch | | | | | | | | 92.61 | 93.11 |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |
| **H20** | PyTorch | | | | | | | | | |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |
| **H100 NVL** | PyTorch | | 19.73 | 19.86 | | | | | 20.68 | 19.87 |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |
| **H200 NVL** | PyTorch | | | | | | | | 32.64 | |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | 5.00 | | | 8.00 | | | 12.00 | | |
| **H100 80GB HBM3** | PyTorch | | | | | | | | | |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | | | | | | | | | |
| **H200 141GB HBM3** | PyTorch | | 13.62 | | | 13.33 | | | 13.78 | 13.50 |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | 5.00 | | | 8.00 | | | 11.00 | | |
| **B200** | PyTorch | | 4.10 | 4.27 | 4.78 | 4.13 | | 7.25 | 4.28 | 4.65 |
| | vLLM-Omni | | | | | | | 6.02 | 3.09 | 4.43 |
| | Diffusers | | | | | | | 8.00 | | |
| **B300** | PyTorch | | | | | | | | | |
| | vLLM-Omni | | | | | | | | | |
| | Diffusers | 38.80 | | | 39.40 | | | 40.40 | | |

<sub>Notes:
1. All times measured on identical workloads (same seed, sampler settings, prompt).
2. 4×/8× GPU configurations use tensor parallelism.
3. vLLM-Omni numbers are for the upcoming public release in the vLLM-Omni repo; subject to change before GA. Current vLLM-Omni coverage is B200 at 720p.
4. Diffusers numbers use the HuggingFace `diffusers` integration without custom CUDA graphs; reported at 256p/1, 480p/1, and 720p/1 (single-GPU only).
5. At 256p, multi-GPU configurations on B300 may underperform single-GPU due to small-workload TP overhead; single-GPU is recommended at this resolution.
6. PyTorch numbers report average generation (sampling) time from OSS inference benchmarking.</sub>

## Cosmos3-Nano Reasoner

These tables report **Cosmos3-Nano** reasoner serving performance through **vLLM**. Unlike the generator sections, Reasoner benchmarks measure **text understanding and generation latency** - time to first token (TTFT) in milliseconds, end-to-end request latency in milliseconds, and token throughput under concurrent load - not diffusion sampling time. Workloads vary input sequence length, output sequence length, and video frame rate to reflect common captioning, VQA, and video-understanding request profiles.

All runs use the **`nvidia/Cosmos3-Nano`** checkpoint. Metrics are collected with the AIPerf client at client-side concurrency levels of 1, 64, 128, and 256. Each GPU section below contains four workload tables (Input 50 / Output 1 or 100 / Video 1 or 2 FPS). Lower is better for latency metrics; higher is better for throughput.

### RTX PRO 6000 Blackwell

#### Input 50 / Output 1 / Video 1 FPS

| Metric | Concurrency 1 | Concurrency 64 | Concurrency 128 | Concurrency 256 |
|---|---:|---:|---:|---:|
| Time To First Token (ms) | 187.59 | 5826.84 | 9742.43 | 19541.84 |
| Request Latency (ms) | 187.59 | 5826.84 | 9742.43 | 19541.84 |
| Request Count (requests) | 50 | 320 | 256 | 512 |
| Request Throughput (Req/s) | 5.29 | 9.95 | 9.97 | 9.89 |
| Output Token Throughput (Tok/s) | 5.29 | 9.95 | 9.97 | 9.89 |

#### Input 50 / Output 1 / Video 2 FPS

| Metric | Concurrency 1 | Concurrency 64 | Concurrency 128 | Concurrency 256 |
|---|---:|---:|---:|---:|
| Time To First Token (ms) | 316.90 | 12223.00 | 20364.04 | 40929.42 |
| Request Latency (ms) | 316.90 | 12223.00 | 20364.04 | 40929.42 |
| Request Count (requests) | 50 | 320 | 256 | 512 |
| Request Throughput (Req/s) | 3.14 | 4.73 | 4.75 | 4.71 |
| Output Token Throughput (Tok/s) | 3.14 | 4.73 | 4.75 | 4.71 |

#### Input 50 / Output 100 / Video 1 FPS

| Metric | Concurrency 1 | Concurrency 64 | Concurrency 128 | Concurrency 256 |
|---|---:|---:|---:|---:|
| Time To First Token (ms) | 186.46 | 2280.43 | 4627.08 | 14419.32 |
| Request Latency (ms) | 1402.12 | 9309.93 | 18541.90 | 39202.74 |
| Request Count (requests) | 50 | 320 | 256 | 512 |
| Request Throughput (Req/s) | 0.71 | 6.85 | 6.82 | 6.22 |
| Output Token Throughput (Tok/s) | 71.22 | 684.76 | 682.18 | 622.49 |

#### Input 50 / Output 100 / Video 2 FPS

| Metric | Concurrency 1 | Concurrency 64 | Concurrency 128 | Concurrency 256 |
|---|---:|---:|---:|---:|
| Time To First Token (ms) | 315.77 | 3248.72 | 13795.45 | 44476.55 |
| Request Latency (ms) | 1553.53 | 18532.34 | 37994.05 | 71534.87 |
| Request Count (requests) | 50 | 320 | 256 | 512 |
| Request Throughput (Req/s) | 0.64 | 3.44 | 3.22 | 3.15 |
| Output Token Throughput (Tok/s) | 64.28 | 343.79 | 322.21 | 314.62 |

### H20

#### Input 50 / Output 1 / Video 1 FPS

| Metric | Concurrency 1 | Concurrency 64 | Concurrency 128 | Concurrency 256 |
|---|---:|---:|---:|---:|
| Time To First Token (ms) | 358.62 | 14953.90 | 25086.30 | 49549.94 |
| Request Latency (ms) | 358.62 | 14953.90 | 25086.30 | 49549.94 |
| Request Count (requests) | 50 | 320 | 256 | 512 |
| Request Throughput (Req/s) | 2.77 | 3.88 | 3.87 | 3.90 |
| Output Token Throughput (Tok/s) | 2.77 | 3.88 | 3.87 | 3.90 |

#### Input 50 / Output 1 / Video 2 FPS

| Metric | Concurrency 1 | Concurrency 64 | Concurrency 128 | Concurrency 256 |
|---|---:|---:|---:|---:|
| Time To First Token (ms) | 648.48 | 30604.91 | 51364.19 | 101597.85 |
| Request Latency (ms) | 648.48 | 30604.91 | 51364.19 | 101597.85 |
| Request Count (requests) | 50 | 320 | 256 | 512 |
| Request Throughput (Req/s) | 1.54 | 1.89 | 1.89 | 1.90 |
| Output Token Throughput (Tok/s) | 1.54 | 1.89 | 1.89 | 1.90 |

#### Input 50 / Output 100 / Video 1 FPS

| Metric | Concurrency 1 | Concurrency 64 | Concurrency 128 | Concurrency 256 |
|---|---:|---:|---:|---:|
| Time To First Token (ms) | 360.40 | 6607.80 | 10973.05 | 29404.60 |
| Request Latency (ms) | 1026.97 | 18990.48 | 37514.21 | 74287.25 |
| Request Count (requests) | 50 | 320 | 256 | 512 |
| Request Throughput (Req/s) | 0.97 | 3.37 | 3.40 | 3.33 |
| Output Token Throughput (Tok/s) | 97.14 | 336.55 | 339.57 | 332.93 |

#### Input 50 / Output 100 / Video 2 FPS

| Metric | Concurrency 1 | Concurrency 64 | Concurrency 128 | Concurrency 256 |
|---|---:|---:|---:|---:|
| Time To First Token (ms) | 646.27 | 8329.82 | 29036.81 | 86145.80 |
| Request Latency (ms) | 1331.00 | 37577.68 | 74291.62 | 136416.07 |
| Request Count (requests) | 50 | 320 | 256 | 512 |
| Request Throughput (Req/s) | 0.75 | 1.70 | 1.67 | 1.67 |
| Output Token Throughput (Tok/s) | 75.00 | 170.03 | 167.36 | 167.12 |

### H100 NVL

#### Input 50 / Output 1 / Video 1 FPS

| Metric | Concurrency 1 | Concurrency 64 | Concurrency 128 | Concurrency 256 |
|---|---:|---:|---:|---:|
| Time To First Token (ms) | 170.69 | 6527.13 | 10726.80 | 21881.52 |
| Request Latency (ms) | 170.69 | 6527.13 | 10726.80 | 21881.52 |
| Request Count (requests) | 50 | 320 | 256 | 512 |
| Request Throughput (Req/s) | 5.81 | 8.88 | 9.05 | 8.83 |
| Output Token Throughput (Tok/s) | 5.81 | 8.88 | 9.05 | 8.83 |

#### Input 50 / Output 1 / Video 2 FPS

| Metric | Concurrency 1 | Concurrency 64 | Concurrency 128 | Concurrency 256 |
|---|---:|---:|---:|---:|
| Time To First Token (ms) | 303.65 | 13480.29 | 22431.93 | 44352.53 |
| Request Latency (ms) | 303.65 | 13480.29 | 22431.93 | 44352.53 |
| Request Count (requests) | 50 | 320 | 256 | 512 |
| Request Throughput (Req/s) | 3.27 | 4.29 | 4.31 | 4.35 |
| Output Token Throughput (Tok/s) | 3.27 | 4.29 | 4.31 | 4.35 |

#### Input 50 / Output 100 / Video 1 FPS

| Metric | Concurrency 1 | Concurrency 64 | Concurrency 128 | Concurrency 256 |
|---|---:|---:|---:|---:|
| Time To First Token (ms) | 172.66 | 2890.81 | 5022.58 | 13929.58 |
| Request Latency (ms) | 867.35 | 9192.19 | 18061.43 | 35151.09 |
| Request Count (requests) | 50 | 320 | 256 | 512 |
| Request Throughput (Req/s) | 1.15 | 6.94 | 7.02 | 6.95 |
| Output Token Throughput (Tok/s) | 115.03 | 694.37 | 702.48 | 695.12 |

#### Input 50 / Output 100 / Video 2 FPS

| Metric | Concurrency 1 | Concurrency 64 | Concurrency 128 | Concurrency 256 |
|---|---:|---:|---:|---:|
| Time To First Token (ms) | 296.99 | 3572.31 | 13808.03 | 41101.98 |
| Request Latency (ms) | 1009.41 | 18030.41 | 35239.25 | 64485.08 |
| Request Count (requests) | 50 | 320 | 256 | 512 |
| Request Throughput (Req/s) | 0.99 | 3.54 | 3.48 | 3.50 |
| Output Token Throughput (Tok/s) | 98.87 | 353.81 | 348.37 | 350.07 |

### H200 NVL

#### Input 50 / Output 1 / Video 1 FPS

| Metric | Concurrency 1 | Concurrency 64 | Concurrency 128 | Concurrency 256 |
|---|---:|---:|---:|---:|
| Time To First Token (ms) | 142.79 | 3614.37 | 6050.58 | 12094.34 |
| Request Latency (ms) | 142.79 | 3614.37 | 6050.58 | 12094.34 |
| Request Count (requests) | 50 | 320 | 256 | 512 |
| Request Throughput (Req/s) | 6.92 | 16.04 | 16.08 | 15.96 |
| Output Token Throughput (Tok/s) | 6.92 | 16.04 | 16.08 | 15.96 |

#### Input 50 / Output 1 / Video 2 FPS

| Metric | Concurrency 1 | Concurrency 64 | Concurrency 128 | Concurrency 256 |
|---|---:|---:|---:|---:|
| Time To First Token (ms) | 228.84 | 7569.46 | 12515.91 | 25646.99 |
| Request Latency (ms) | 228.84 | 7569.46 | 12515.91 | 25646.99 |
| Request Count (requests) | 50 | 320 | 256 | 512 |
| Request Throughput (Req/s) | 4.34 | 7.62 | 7.71 | 7.48 |
| Output Token Throughput (Tok/s) | 4.34 | 7.62 | 7.71 | 7.48 |

#### Input 50 / Output 100 / Video 1 FPS

| Metric | Concurrency 1 | Concurrency 64 | Concurrency 128 | Concurrency 256 |
|---|---:|---:|---:|---:|
| Time To First Token (ms) | 142.23 | 1948.06 | 3180.20 | 5271.37 |
| Request Latency (ms) | 770.15 | 5284.58 | 10054.55 | 19831.69 |
| Request Count (requests) | 50 | 320 | 256 | 512 |
| Request Throughput (Req/s) | 1.30 | 12.07 | 12.60 | 12.71 |
| Output Token Throughput (Tok/s) | 129.53 | 1206.86 | 1259.60 | 1270.44 |

#### Input 50 / Output 100 / Video 2 FPS

| Metric | Concurrency 1 | Concurrency 64 | Concurrency 128 | Concurrency 256 |
|---|---:|---:|---:|---:|
| Time To First Token (ms) | 227.40 | 2718.13 | 5522.05 | 17729.47 |
| Request Latency (ms) | 862.92 | 10249.14 | 19775.33 | 39089.75 |
| Request Count (requests) | 50 | 320 | 256 | 512 |
| Request Throughput (Req/s) | 1.16 | 6.22 | 6.38 | 6.18 |
| Output Token Throughput (Tok/s) | 115.63 | 621.97 | 638.33 | 618.21 |

### H100 80GB HBM3 (SXM)

#### Input 50 / Output 1 / Video 1 FPS

| Metric | Concurrency 1 | Concurrency 64 | Concurrency 128 | Concurrency 256 |
|---|---:|---:|---:|---:|
| Time To First Token (ms) | 145.52 | 3332.72 | 5608.41 | 11133.76 |
| Request Latency (ms) | 145.52 | 3332.72 | 5608.41 | 11133.76 |
| Request Count (requests) | 50 | 320 | 256 | 512 |
| Request Throughput (Req/s) | 6.78 | 17.41 | 17.36 | 17.38 |
| Output Token Throughput (Tok/s) | 6.78 | 17.41 | 17.36 | 17.38 |

#### Input 50 / Output 1 / Video 2 FPS

| Metric | Concurrency 1 | Concurrency 64 | Concurrency 128 | Concurrency 256 |
|---|---:|---:|---:|---:|
| Time To First Token (ms) | 228.80 | 6876.25 | 11556.33 | 22836.32 |
| Request Latency (ms) | 228.80 | 6876.25 | 11556.33 | 22836.32 |
| Request Count (requests) | 50 | 320 | 256 | 512 |
| Request Throughput (Req/s) | 4.34 | 8.42 | 8.38 | 8.46 |
| Output Token Throughput (Tok/s) | 4.34 | 8.42 | 8.38 | 8.46 |

#### Input 50 / Output 100 / Video 1 FPS

| Metric | Concurrency 1 | Concurrency 64 | Concurrency 128 | Concurrency 256 |
|---|---:|---:|---:|---:|
| Time To First Token (ms) | 143.36 | 1720.73 | 2906.00 | 9000.63 |
| Request Latency (ms) | 865.56 | 5251.56 | 9818.73 | 18353.75 |
| Request Count (requests) | 50 | 320 | 256 | 512 |
| Request Throughput (Req/s) | 1.15 | 12.14 | 12.87 | 12.83 |
| Output Token Throughput (Tok/s) | 115.24 | 1213.61 | 1286.89 | 1282.48 |

#### Input 50 / Output 100 / Video 2 FPS

| Metric | Concurrency 1 | Concurrency 64 | Concurrency 128 | Concurrency 256 |
|---|---:|---:|---:|---:|
| Time To First Token (ms) | 231.81 | 2295.44 | 8767.78 | 23214.88 |
| Request Latency (ms) | 967.49 | 9738.18 | 18061.79 | 33190.08 |
| Request Count (requests) | 50 | 320 | 256 | 512 |
| Request Throughput (Req/s) | 1.03 | 6.54 | 6.53 | 6.60 |
| Output Token Throughput (Tok/s) | 103.12 | 653.91 | 653.39 | 659.92 |

### H200 141GB HBM3

#### Input 50 / Output 1 / Video 1 FPS

| Metric | Concurrency 1 | Concurrency 64 | Concurrency 128 | Concurrency 256 |
|---|---:|---:|---:|---:|
| Time To First Token (ms) | 142.45 | 3363.01 | 5656.58 | 11271.28 |
| Request Latency (ms) | 142.45 | 3363.01 | 5656.58 | 11271.28 |
| Request Count (requests) | 50 | 320 | 256 | 512 |
| Request Throughput (Req/s) | 6.93 | 17.25 | 17.21 | 17.17 |
| Output Token Throughput (Tok/s) | 6.93 | 17.25 | 17.21 | 17.17 |

#### Input 50 / Output 1 / Video 2 FPS

| Metric | Concurrency 1 | Concurrency 64 | Concurrency 128 | Concurrency 256 |
|---|---:|---:|---:|---:|
| Time To First Token (ms) | 229.70 | 6932.55 | 11640.68 | 23173.10 |
| Request Latency (ms) | 229.70 | 6932.55 | 11640.68 | 23173.10 |
| Request Count (requests) | 50 | 320 | 256 | 512 |
| Request Throughput (Req/s) | 4.31 | 8.35 | 8.32 | 8.33 |
| Output Token Throughput (Tok/s) | 4.31 | 8.35 | 8.32 | 8.33 |

#### Input 50 / Output 100 / Video 1 FPS

| Metric | Concurrency 1 | Concurrency 64 | Concurrency 128 | Concurrency 256 |
|---|---:|---:|---:|---:|
| Time To First Token (ms) | 143.25 | 2060.05 | 2839.97 | 4713.83 |
| Request Latency (ms) | 711.77 | 4965.09 | 9364.20 | 18325.25 |
| Request Count (requests) | 50 | 320 | 256 | 512 |
| Request Throughput (Req/s) | 1.40 | 12.84 | 13.53 | 13.75 |
| Output Token Throughput (Tok/s) | 140.02 | 1284.38 | 1352.41 | 1374.57 |

#### Input 50 / Output 100 / Video 2 FPS

| Metric | Concurrency 1 | Concurrency 64 | Concurrency 128 | Concurrency 256 |
|---|---:|---:|---:|---:|
| Time To First Token (ms) | 228.24 | 2715.85 | 5015.19 | 15991.33 |
| Request Latency (ms) | 807.07 | 9341.21 | 18285.65 | 35043.34 |
| Request Count (requests) | 50 | 320 | 256 | 512 |
| Request Throughput (Req/s) | 1.24 | 6.82 | 6.90 | 6.90 |
| Output Token Throughput (Tok/s) | 123.55 | 682.33 | 690.26 | 689.50 |

### B200

#### Input 50 / Output 1 / Video 1 FPS

| Metric | Concurrency 1 | Concurrency 64 | Concurrency 128 | Concurrency 256 |
|---|---:|---:|---:|---:|
| Time To First Token (ms) | 115.55 | 1661.57 | 2819.22 | 5550.74 |
| Request Latency (ms) | 115.55 | 1661.57 | 2819.22 | 5550.74 |
| Request Count (requests) | 50 | 320 | 256 | 512 |
| Request Throughput (Req/s) | 8.53 | 34.96 | 34.72 | 34.94 |
| Output Token Throughput (Tok/s) | 8.53 | 34.96 | 34.72 | 34.94 |

#### Input 50 / Output 1 / Video 2 FPS

| Metric | Concurrency 1 | Concurrency 64 | Concurrency 128 | Concurrency 256 |
|---|---:|---:|---:|---:|
| Time To First Token (ms) | 168.95 | 3410.27 | 5699.88 | 11422.16 |
| Request Latency (ms) | 168.95 | 3410.27 | 5699.88 | 11422.16 |
| Request Count (requests) | 50 | 320 | 256 | 512 |
| Request Throughput (Req/s) | 5.86 | 16.98 | 17.01 | 16.93 |
| Output Token Throughput (Tok/s) | 5.86 | 16.98 | 17.01 | 16.93 |

#### Input 50 / Output 100 / Video 1 FPS

| Metric | Concurrency 1 | Concurrency 64 | Concurrency 128 | Concurrency 256 |
|---|---:|---:|---:|---:|
| Time To First Token (ms) | 115.27 | 1106.35 | 2111.97 | 2549.79 |
| Request Latency (ms) | 553.01 | 2736.53 | 5001.20 | 9279.25 |
| Request Count (requests) | 50 | 320 | 256 | 512 |
| Request Throughput (Req/s) | 1.80 | 23.28 | 25.23 | 27.01 |
| Output Token Throughput (Tok/s) | 180.16 | 2328.01 | 2523.07 | 2701.08 |

#### Input 50 / Output 100 / Video 2 FPS

| Metric | Concurrency 1 | Concurrency 64 | Concurrency 128 | Concurrency 256 |
|---|---:|---:|---:|---:|
| Time To First Token (ms) | 166.52 | 1914.24 | 2596.36 | 7277.89 |
| Request Latency (ms) | 622.36 | 4881.38 | 9220.30 | 17548.01 |
| Request Count (requests) | 50 | 320 | 256 | 512 |
| Request Throughput (Req/s) | 1.60 | 13.04 | 13.63 | 13.87 |
| Output Token Throughput (Tok/s) | 160.11 | 1303.92 | 1362.49 | 1386.99 |

### B300

#### Input 50 / Output 1 / Video 1 FPS

| Metric | Concurrency 1 | Concurrency 64 | Concurrency 128 | Concurrency 256 |
|---|---:|---:|---:|---:|
| Time To First Token (ms) | 80.70 | 1617.12 | 2742.68 | 5421.22 |
| Request Latency (ms) | 80.70 | 1617.12 | 2742.68 | 5421.22 |
| Request Count (requests) | 50 | 320 | 256 | 511 |
| Request Throughput (Req/s) | 12.24 | 35.92 | 35.69 | 35.79 |
| Output Token Throughput (Tok/s) | 12.24 | 35.92 | 35.69 | 35.79 |

#### Input 50 / Output 1 / Video 2 FPS

| Metric | Concurrency 1 | Concurrency 64 | Concurrency 128 | Concurrency 256 |
|---|---:|---:|---:|---:|
| Time To First Token (ms) | 126.25 | 3304.76 | 5551.12 | 11054.47 |
| Request Latency (ms) | 126.25 | 3304.76 | 5551.12 | 11054.47 |
| Request Count (requests) | 50 | 320 | 256 | 512 |
| Request Throughput (Req/s) | 7.86 | 17.53 | 17.49 | 17.49 |
| Output Token Throughput (Tok/s) | 7.86 | 17.53 | 17.49 | 17.49 |

#### Input 50 / Output 100 / Video 1 FPS

| Metric | Concurrency 1 | Concurrency 64 | Concurrency 128 | Concurrency 256 |
|---|---:|---:|---:|---:|
| Time To First Token (ms) | 83.19 | 1070.93 | 1444.68 | 2739.50 |
| Request Latency (ms) | 490.11 | 2657.06 | 4750.02 | 8975.21 |
| Request Count (requests) | 50 | 320 | 256 | 512 |
| Request Throughput (Req/s) | 2.03 | 23.96 | 26.57 | 27.92 |
| Output Token Throughput (Tok/s) | 203.29 | 2396.35 | 2657.14 | 2791.79 |

#### Input 50 / Output 100 / Video 2 FPS

| Metric | Concurrency 1 | Concurrency 64 | Concurrency 128 | Concurrency 256 |
|---|---:|---:|---:|---:|
| Time To First Token (ms) | 129.22 | 1602.00 | 2404.48 | 6982.58 |
| Request Latency (ms) | 550.02 | 4684.78 | 8813.83 | 16813.61 |
| Request Count (requests) | 50 | 320 | 256 | 512 |
| Request Throughput (Req/s) | 1.81 | 13.59 | 14.25 | 14.47 |
| Output Token Throughput (Tok/s) | 181.25 | 1358.62 | 1425.02 | 1447.23 |

<sub>Notes:
1. Source: vLLM inference benchmarking for `nvidia/Cosmos3-Nano`; AIPerf client was used as the benchmarking tool.
2. Hardware: results are grouped by GPU product (RTX PRO 6000 Blackwell, H20, H100 NVL, H200 NVL, H100 80GB HBM3 SXM, H200 141GB HBM3, B200, B300). All metrics are averages for a number of requests.
3. **Time To First Token (TTFT)** measures latency until the first output token is emitted. **Request Latency** is end-to-end time per request. For single-token outputs (Output 1), TTFT and request latency are identical.
4. **Request Throughput** is completed requests per second. **Output Token Throughput** is generated tokens per second (for Output 1 workloads, the two throughputs match).
5. Concurrency is the number of simultaneous client requests issued by AIPerf, not tensor-parallel GPU count.</sub>
## Cosmos3-Super Reasoner

These tables report **Cosmos3-Super** reasoner serving performance through **vLLM**. Unlike the generator sections, Reasoner benchmarks measure **text understanding and generation latency** - time to first token (TTFT) in milliseconds, end-to-end request latency in milliseconds, and token throughput under concurrent load - not diffusion sampling time. Workloads vary input sequence length, output sequence length, and video frame rate to reflect common captioning, VQA, and video-understanding request profiles.

All runs use the **`nvidia/Cosmos3-Super`** checkpoint. Metrics are collected with the AIPerf client at client-side concurrency levels of 1, 64, 128, and 256. Each GPU section below contains four workload tables (Input 50 / Output 1 or 100 / Video 1 or 2 FPS). Lower is better for latency metrics; higher is better for throughput. **Empty cells indicate a run has not been completed** for that GPU, workload, or concurrency level.

### RTX PRO 6000 Blackwell

#### Input 50 / Output 1 / Video 1 FPS

| Metric | Concurrency 1 | Concurrency 64 | Concurrency 128 | Concurrency 256 |
|---|---:|---:|---:|---:|
| Time To First Token (ms) | 534.73 | 24781.47 | 41467.45 | 82626.36 |
| Request Latency (ms) | 534.73 | 24781.47 | 41467.45 | 82626.36 |
| Request Count (requests) | 50 | 320 | 256 | 509 |
| Request Throughput (Req/s) | 1.86 | 2.34 | 2.34 | 2.32 |
| Output Token Throughput (Tok/s) | 1.86 | 2.34 | 2.34 | 2.32 |

#### Input 50 / Output 1 / Video 2 FPS

| Metric | Concurrency 1 | Concurrency 64 | Concurrency 128 | Concurrency 256 |
|---|---:|---:|---:|---:|
| Time To First Token (ms) | 978.78 | 51145.61 | 85476.50 |  |
| Request Latency (ms) | 978.78 | 51145.61 | 85476.50 |  |
| Request Count (requests) | 50 | 320 | 256 |  |
| Request Throughput (Req/s) | 1.02 | 1.13 | 1.13 |  |
| Output Token Throughput (Tok/s) | 1.02 | 1.13 | 1.13 |  |

#### Input 50 / Output 100 / Video 1 FPS

| Metric | Concurrency 1 | Concurrency 64 | Concurrency 128 | Concurrency 256 |
|---|---:|---:|---:|---:|
| Time To First Token (ms) | 530.47 | 25094.90 | 54400.80 | 117849.75 |
| Request Latency (ms) | 5225.79 | 40064.22 | 69193.77 | 133019.50 |
| Request Count (requests) | 50 | 320 | 256 | 512 |
| Request Throughput (Req/s) | 0.19 | 1.51 | 1.50 | 1.51 |
| Output Token Throughput (Tok/s) | 19.12 | 151.27 | 149.69 | 151.15 |

#### Input 50 / Output 100 / Video 2 FPS

| Metric | Concurrency 1 | Concurrency 64 | Concurrency 128 | Concurrency 256 |
|---|---:|---:|---:|---:|
| Time To First Token (ms) | 981.55 |  | 114704.46 |  |
| Request Latency (ms) | 5716.49 |  | 130177.35 |  |
| Request Count (requests) | 50 |  | 256 |  |
| Request Throughput (Req/s) | 0.17 |  | 0.77 |  |
| Output Token Throughput (Tok/s) | 17.49 |  | 77.00 |  |

### H20

#### Input 50 / Output 1 / Video 1 FPS

| Metric | Concurrency 1 | Concurrency 64 | Concurrency 128 | Concurrency 256 |
|---|---:|---:|---:|---:|
| Time To First Token (ms) | 1241.58 |  | 108912.73 |  |
| Request Latency (ms) | 1241.58 |  | 108912.73 |  |
| Request Count (requests) | 50 |  | 256 |  |
| Request Throughput (Req/s) | 0.80 |  | 0.89 |  |
| Output Token Throughput (Tok/s) | 0.80 |  | 0.89 |  |

#### Input 50 / Output 1 / Video 2 FPS

| Metric | Concurrency 1 | Concurrency 64 | Concurrency 128 | Concurrency 256 |
|---|---:|---:|---:|---:|
| Time To First Token (ms) | 2399.91 |  |  |  |
| Request Latency (ms) | 2399.91 |  |  |  |
| Request Count (requests) | 50 |  |  |  |
| Request Throughput (Req/s) | 0.42 |  |  |  |
| Output Token Throughput (Tok/s) | 0.42 |  |  |  |

#### Input 50 / Output 100 / Video 1 FPS

| Metric | Concurrency 1 | Concurrency 64 | Concurrency 128 | Concurrency 256 |
|---|---:|---:|---:|---:|
| Time To First Token (ms) | 1230.95 |  | 108988.21 |  |
| Request Latency (ms) | 3523.74 |  | 135784.46 |  |
| Request Count (requests) | 50 |  | 256 |  |
| Request Throughput (Req/s) | 0.28 |  | 0.78 |  |
| Output Token Throughput (Tok/s) | 28.36 |  | 77.81 |  |

#### Input 50 / Output 100 / Video 2 FPS

| Metric | Concurrency 1 | Concurrency 64 | Concurrency 128 | Concurrency 256 |
|---|---:|---:|---:|---:|
| Time To First Token (ms) | 2380.85 |  |  |  |
| Request Latency (ms) | 4707.46 |  |  |  |
| Request Count (requests) | 50 |  |  |  |
| Request Throughput (Req/s) | 0.21 |  |  |  |
| Output Token Throughput (Tok/s) | 21.22 |  |  |  |

### H100 NVL

#### Input 50 / Output 1 / Video 1 FPS

| Metric | Concurrency 1 | Concurrency 64 | Concurrency 128 | Concurrency 256 |
|---|---:|---:|---:|---:|
| Time To First Token (ms) | 521.87 | 27004.42 | 45688.95 | 90353.80 |
| Request Latency (ms) | 521.87 | 27004.42 | 45688.95 | 90353.80 |
| Request Count (requests) | 50 | 320 | 256 | 512 |
| Request Throughput (Req/s) | 1.91 | 2.15 | 2.13 | 2.14 |
| Output Token Throughput (Tok/s) | 1.91 | 2.15 | 2.13 | 2.14 |

#### Input 50 / Output 1 / Video 2 FPS

| Metric | Concurrency 1 | Concurrency 64 | Concurrency 128 | Concurrency 256 |
|---|---:|---:|---:|---:|
| Time To First Token (ms) | 993.48 | 55409.83 | 92484.18 |  |
| Request Latency (ms) | 993.48 | 55409.83 | 92484.18 |  |
| Request Count (requests) | 50 | 320 | 256 |  |
| Request Throughput (Req/s) | 1.00 | 1.05 | 1.05 |  |
| Output Token Throughput (Tok/s) | 1.00 | 1.05 | 1.05 |  |

#### Input 50 / Output 100 / Video 1 FPS

| Metric | Concurrency 1 | Concurrency 64 | Concurrency 128 | Concurrency 256 |
|---|---:|---:|---:|---:|
| Time To First Token (ms) | 508.97 | 26567.65 | 54861.82 | 116733.31 |
| Request Latency (ms) | 3119.76 | 39090.16 | 67203.81 | 129435.49 |
| Request Count (requests) | 50 | 320 | 256 | 512 |
| Request Throughput (Req/s) | 0.32 | 1.54 | 1.55 | 1.54 |
| Output Token Throughput (Tok/s) | 32.03 | 153.87 | 154.58 | 154.04 |

#### Input 50 / Output 100 / Video 2 FPS

| Metric | Concurrency 1 | Concurrency 64 | Concurrency 128 | Concurrency 256 |
|---|---:|---:|---:|---:|
| Time To First Token (ms) | 999.10 | 49069.05 | 116178.34 |  |
| Request Latency (ms) | 3638.23 | 59084.57 | 128875.27 |  |
| Request Count (requests) | 50 | 320 | 256 |  |
| Request Throughput (Req/s) | 0.27 | 1.00 | 0.77 |  |
| Output Token Throughput (Tok/s) | 27.46 | 100.14 | 77.38 |  |

### H200 NVL

#### Input 50 / Output 1 / Video 1 FPS

| Metric | Concurrency 1 | Concurrency 64 | Concurrency 128 | Concurrency 256 |
|---|---:|---:|---:|---:|
| Time To First Token (ms) | 357.69 | 16243.11 | 26759.35 | 54470.31 |
| Request Latency (ms) | 357.69 | 16243.11 | 26759.35 | 54470.31 |
| Request Count (requests) | 50 | 319 | 254 | 510 |
| Request Throughput (Req/s) | 2.78 | 3.56 | 3.58 | 3.53 |
| Output Token Throughput (Tok/s) | 2.78 | 3.56 | 3.58 | 3.53 |

#### Input 50 / Output 1 / Video 2 FPS

| Metric | Concurrency 1 | Concurrency 64 | Concurrency 128 | Concurrency 256 |
|---|---:|---:|---:|---:|
| Time To First Token (ms) | 641.05 | 33640.68 | 56090.59 | 111965.21 |
| Request Latency (ms) | 641.05 | 33640.68 | 56090.59 | 111965.21 |
| Request Count (requests) | 50 | 320 | 255 | 510 |
| Request Throughput (Req/s) | 1.56 | 1.72 | 1.72 | 1.71 |
| Output Token Throughput (Tok/s) | 1.56 | 1.72 | 1.72 | 1.71 |

#### Input 50 / Output 100 / Video 1 FPS

| Metric | Concurrency 1 | Concurrency 64 | Concurrency 128 | Concurrency 256 |
|---|---:|---:|---:|---:|
| Time To First Token (ms) | 348.88 | 5805.63 | 16053.68 | 48411.16 |
| Request Latency (ms) | 2385.95 | 21240.62 | 40187.13 | 75354.93 |
| Request Count (requests) | 50 | 320 | 256 | 512 |
| Request Throughput (Req/s) | 0.42 | 3.01 | 3.06 | 2.99 |
| Output Token Throughput (Tok/s) | 41.87 | 300.56 | 305.57 | 298.96 |

#### Input 50 / Output 100 / Video 2 FPS

| Metric | Concurrency 1 | Concurrency 64 | Concurrency 128 | Concurrency 256 |
|---|---:|---:|---:|---:|
| Time To First Token (ms) | 640.14 | 13800.40 | 47514.83 | 110513.13 |
| Request Latency (ms) | 2692.46 | 41460.80 | 74683.53 | 138991.66 |
| Request Count (requests) | 50 | 320 | 256 | 512 |
| Request Throughput (Req/s) | 0.37 | 1.52 | 1.51 | 1.52 |
| Output Token Throughput (Tok/s) | 37.12 | 151.97 | 151.42 | 152.07 |

### H200 141GB HBM3

#### Input 50 / Output 1 / Video 1 FPS

| Metric | Concurrency 1 | Concurrency 64 | Concurrency 128 | Concurrency 256 |
|---|---:|---:|---:|---:|
| Time To First Token (ms) | 327.21 | 14045.01 | 23809.00 | 46893.25 |
| Request Latency (ms) | 327.21 | 14045.01 | 23809.00 | 46893.25 |
| Request Count (requests) | 50 | 320 | 256 | 507 |
| Request Throughput (Req/s) | 3.04 | 4.14 | 4.09 | 4.08 |
| Output Token Throughput (Tok/s) | 3.04 | 4.14 | 4.09 | 4.08 |

#### Input 50 / Output 1 / Video 2 FPS

| Metric | Concurrency 1 | Concurrency 64 | Concurrency 128 | Concurrency 256 |
|---|---:|---:|---:|---:|
| Time To First Token (ms) | 592.19 | 28769.68 | 48595.04 | 95884.55 |
| Request Latency (ms) | 592.19 | 28769.68 | 48595.04 | 95884.55 |
| Request Count (requests) | 50 | 320 | 256 | 512 |
| Request Throughput (Req/s) | 1.68 | 2.01 | 1.99 | 2.01 |
| Output Token Throughput (Tok/s) | 1.68 | 2.01 | 1.99 | 2.01 |

#### Input 50 / Output 100 / Video 1 FPS

| Metric | Concurrency 1 | Concurrency 64 | Concurrency 128 | Concurrency 256 |
|---|---:|---:|---:|---:|
| Time To First Token (ms) | 327.73 | 5374.55 | 14328.53 | 42108.40 |
| Request Latency (ms) | 2254.07 | 18553.56 | 35613.84 | 65558.20 |
| Request Count (requests) | 50 | 320 | 256 | 512 |
| Request Throughput (Req/s) | 0.44 | 3.44 | 3.44 | 3.43 |
| Output Token Throughput (Tok/s) | 44.31 | 344.02 | 344.42 | 343.29 |

#### Input 50 / Output 100 / Video 2 FPS

| Metric | Concurrency 1 | Concurrency 64 | Concurrency 128 | Concurrency 256 |
|---|---:|---:|---:|---:|
| Time To First Token (ms) | 592.97 | 11969.34 | 41372.57 | 95995.33 |
| Request Latency (ms) | 2533.93 | 36021.00 | 65053.92 | 120751.01 |
| Request Count (requests) | 50 | 320 | 256 | 512 |
| Request Throughput (Req/s) | 0.39 | 1.75 | 1.74 | 1.75 |
| Output Token Throughput (Tok/s) | 39.43 | 174.83 | 173.72 | 174.85 |

### B200

#### Input 50 / Output 1 / Video 1 FPS

| Metric | Concurrency 1 | Concurrency 64 | Concurrency 128 | Concurrency 256 |
|---|---:|---:|---:|---:|
| Time To First Token (ms) | 212.10 | 6902.51 | 11412.35 | 22707.11 |
| Request Latency (ms) | 212.10 | 6902.51 | 11412.35 | 22707.11 |
| Request Count (requests) | 50 | 320 | 256 | 512 |
| Request Throughput (Req/s) | 4.68 | 8.41 | 8.52 | 8.52 |
| Output Token Throughput (Tok/s) | 4.68 | 8.41 | 8.52 | 8.52 |

#### Input 50 / Output 1 / Video 2 FPS

| Metric | Concurrency 1 | Concurrency 64 | Concurrency 128 | Concurrency 256 |
|---|---:|---:|---:|---:|
| Time To First Token (ms) | 350.30 | 13909.70 | 23275.71 | 46780.29 |
| Request Latency (ms) | 350.30 | 13909.70 | 23275.71 | 46780.29 |
| Request Count (requests) | 50 | 320 | 256 | 510 |
| Request Throughput (Req/s) | 2.84 | 4.16 | 4.16 | 4.10 |
| Output Token Throughput (Tok/s) | 2.84 | 4.16 | 4.16 | 4.10 |

#### Input 50 / Output 100 / Video 1 FPS

| Metric | Concurrency 1 | Concurrency 64 | Concurrency 128 | Concurrency 256 |
|---|---:|---:|---:|---:|
| Time To First Token (ms) | 212.40 | 2723.57 | 5574.27 | 16228.42 |
| Request Latency (ms) | 1552.87 | 9594.94 | 17572.41 | 34293.88 |
| Request Count (requests) | 50 | 320 | 256 | 512 |
| Request Throughput (Req/s) | 0.64 | 6.65 | 7.21 | 6.97 |
| Output Token Throughput (Tok/s) | 64.30 | 664.83 | 721.29 | 696.84 |

#### Input 50 / Output 100 / Video 2 FPS

| Metric | Concurrency 1 | Concurrency 64 | Concurrency 128 | Concurrency 256 |
|---|---:|---:|---:|---:|
| Time To First Token (ms) | 325.21 | 3821.22 | 15872.01 | 42120.75 |
| Request Latency (ms) | 1686.82 | 17970.15 | 34042.27 | 61444.78 |
| Request Count (requests) | 50 | 320 | 256 | 512 |
| Request Throughput (Req/s) | 0.59 | 3.55 | 3.52 | 3.60 |
| Output Token Throughput (Tok/s) | 59.21 | 354.77 | 352.10 | 360.15 |

### B300

#### Input 50 / Output 1 / Video 1 FPS

| Metric | Concurrency 1 | Concurrency 64 | Concurrency 128 | Concurrency 256 |
|---|---:|---:|---:|---:|
| Time To First Token (ms) | 176.24 | 6665.86 | 11233.68 | 22238.90 |
| Request Latency (ms) | 176.24 | 6665.86 | 11233.68 | 22238.90 |
| Request Count (requests) | 50 | 320 | 256 | 510 |
| Request Throughput (Req/s) | 5.64 | 8.71 | 8.67 | 8.65 |
| Output Token Throughput (Tok/s) | 5.64 | 8.71 | 8.67 | 8.65 |

#### Input 50 / Output 1 / Video 2 FPS

| Metric | Concurrency 1 | Concurrency 64 | Concurrency 128 | Concurrency 256 |
|---|---:|---:|---:|---:|
| Time To First Token (ms) | 301.49 | 13570.83 | 22688.87 | 45276.68 |
| Request Latency (ms) | 301.49 | 13570.83 | 22688.87 | 45276.68 |
| Request Count (requests) | 50 | 320 | 255 | 512 |
| Request Throughput (Req/s) | 3.31 | 4.26 | 4.26 | 4.26 |
| Output Token Throughput (Tok/s) | 3.31 | 4.26 | 4.26 | 4.26 |

#### Input 50 / Output 100 / Video 1 FPS

| Metric | Concurrency 1 | Concurrency 64 | Concurrency 128 | Concurrency 256 |
|---|---:|---:|---:|---:|
| Time To First Token (ms) | 175.83 | 2491.56 | 4999.71 | 8916.09 |
| Request Latency (ms) | 1492.46 | 9254.00 | 17203.60 | 33189.36 |
| Request Count (requests) | 50 | 320 | 256 | 512 |
| Request Throughput (Req/s) | 0.67 | 6.89 | 7.37 | 7.61 |
| Output Token Throughput (Tok/s) | 66.93 | 689.29 | 736.64 | 761.12 |

#### Input 50 / Output 100 / Video 2 FPS

| Metric | Concurrency 1 | Concurrency 64 | Concurrency 128 | Concurrency 256 |
|---|---:|---:|---:|---:|
| Time To First Token (ms) | 303.52 | 3655.15 | 8924.49 | 30494.72 |
| Request Latency (ms) | 1637.02 | 17223.78 | 33088.08 | 62798.50 |
| Request Count (requests) | 50 | 320 | 256 | 512 |
| Request Throughput (Req/s) | 0.61 | 3.70 | 3.82 | 3.83 |
| Output Token Throughput (Tok/s) | 61.02 | 370.09 | 382.18 | 383.31 |

<sub>Notes:
1. Source: vLLM inference benchmarking for `nvidia/Cosmos3-Super`; AIPerf client was used as the benchmarking tool.
2. Hardware: results are grouped by GPU product (RTX PRO 6000 Blackwell, H20, H100 NVL, H200 NVL, H200 141GB HBM3, B200, B300). All metrics are averages for a number of requests.
3. **Time To First Token (TTFT)** measures latency until the first output token is emitted. **Request Latency** is end-to-end time per request. For single-token outputs (Output 1), TTFT and request latency are identical.
4. **Request Throughput** is completed requests per second. **Output Token Throughput** is generated tokens per second (for Output 1 workloads, the two throughputs match).
5. Concurrency is the number of simultaneous client requests issued by AIPerf, not tensor-parallel GPU count.</sub>
