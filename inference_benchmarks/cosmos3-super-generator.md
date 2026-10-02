# Cosmos3-Super Generator Inference Benchmarks

[Back to the inference benchmark index](../inference_benchmarks.md)

These tables report **Cosmos3-Super Generator** latency in seconds. Lower is better.

Every row reports one GPU configuration, named in the **GPUs** column, and all three resolutions in that row come from it. The configuration is one GPU when the model fits on a single device at its highest resolution, 720p, and eight GPUs when it does not. Cosmos3-Super fits on a single GPU on B200 and B300, and on H200 NVL and H200 141GB HBM3 for every engine except PyTorch; on RTX PRO 6000 Blackwell, H20, H100 NVL, and H100 80GB HBM3 it is reported at eight GPUs. Because the configuration varies between rows, read the **GPUs** column before comparing latencies down a column.

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

The additional audiovisual and action tables come from three internal benchmark reports. PyTorch values are from PBR `#308197`, **Cosmos3-Generator OSS Inference Benchmarking 32B and 8B (189 frames)**: average generation (sampling) latency from the native OSS path, using **CUDA Graphs disabled** and the **latency** automatic-sharding preset. vLLM-Omni values for text-to-audio-and-video (`t2av`/`t2vs`) and image-to-audio-and-video (`i2av`/`i2vs`) are from PBR `#308195`. vLLM-Omni action values are from PBR `#308481`, **Cosmos3-Generator vLLM-Omni Inference Benchmarking (action)**, measured with the `vllm/vllm-omni:cosmos3` image and the official action cookbook samples; `DIFFUSION_ATTENTION_BACKEND=TORCH_SDPA` was not set. Forward-dynamics cells are the mean of `av_forward`, `av_left`, and `av_right`; inverse-dynamics cells are the mean of `av_inverse_0` and `av_inverse_1`. That action sweep covers 1, 2, and 4 GPUs only, because its 8-GPU Ulysses runs failed a sequence-length divisibility check; those rows therefore report the single-GPU configuration. Policy-DROID is a separate checkpoint and was measured only at 480p on one GPU. Super was not measured on H20, H100 NVL, or H100 80GB HBM3. Diffusers values come from three further reports: PBR `#308202` (all modalities) and PBR `#308451` (video-to-video) on one GPU and PBR `#308918` for 4- and 8-GPU runs. Where PBR `#308918` offers several tensor- and context-parallel splits at the same GPU count, the fastest is published. Its 1-GPU numbers are not used, because that sweep ran a different denoising-step budget than the single-GPU reports. Values are rounded to two decimal places.

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

The PBR uses `t2av`, `v2av`, and `i2av`; some public recipes call the same sound-producing modes `t2vs`, `v2vs`, and `i2vs`. See the [audiovisual cookbook](../cookbooks/cosmos3/generator/audiovisual/README.md) for generation inputs and the [action cookbook](../cookbooks/cosmos3/generator/action/README.md) for action representations and output contracts. vLLM-Omni request shapes are maintained in the [Super recipe](https://github.com/vllm-project/vllm-omni/blob/main/recipes/cosmos3/Cosmos3-Super.md). Public recipes cover more embodiments than this PBR; the action tables below intentionally use only its measured domain rows.

## Primary vision generation

### Text-to-Video (t2v)

| GPU | Engine | GPUs | 256p | 480p | 720p |
|---|---|:-:|---:|---:|---:|
| **RTX PRO 6000 Blackwell** | PyTorch | 8 | 66.04 | 118.90 | 427.16 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 8 | 25.39 | 154.25 | 437.57 |
|  | NIM | 8 | 13.99 | 99.05 | 286.02 |
| **H20** | PyTorch | 8 | 27.72 | 152.46 | 492.41 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 8 | 26.66 | 161.49 | 517.37 |
|  | NIM | 8 | 12.95 | 110.71 | 395.56 |
| **H100 NVL** | PyTorch | 8 | 16.83 | 64.14 | 186.19 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 8 | 24.21 | 108.70 | 322.28 |
|  | NIM | 8 | 12.73 | 66.07 | 197.32 |
| **H200 NVL** | PyTorch | 8 | 24.78 | 47.69 | 142.35 |
|  | vLLM-Omni | 1 | 27.54 | 252.33 | 911.49 |
|  | Diffusers | 1 | 33.00 | 286.80 | 1036.00 |
|  | NIM | 1 | 17.13 | 200.00 | 811.41 |
| **H100 80GB HBM3** | PyTorch | — |  |  |  |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 8 | 19.40 | 56.39 | 174.57 |
|  | NIM | 8 | 5.89 | 35.52 | 114.92 |
| **H200 141GB HBM3** | PyTorch | 8 | 11.82 | 41.78 | 123.49 |
|  | vLLM-Omni | 1 | 25.61 | 219.11 | 769.63 |
|  | Diffusers | 1 | 31.00 | 251.60 | 886.20 |
|  | NIM | 1 | 15.95 | 174.71 | 695.89 |
| **B200** | PyTorch | 1 | 14.66 | 114.38 | 407.50 |
|  | vLLM-Omni | 1 | 13.84 | 114.08 | 390.28 |
|  | Diffusers | 1 | 19.00 | 127.20 | 414.40 |
|  | NIM | 1 | 9.09 | 82.39 | 314.68 |
| **B300** | PyTorch | — |  |  |  |
|  | vLLM-Omni | 1 | 14.57 | 109.03 | 366.66 |
|  | Diffusers | 1 | 17.73 | 125.70 | 398.94 |
|  | NIM | 1 | 9.67 | 79.73 | 292.35 |

### Image-to-Video (i2v)

| GPU | Engine | GPUs | 256p | 480p | 720p |
|---|---|:-:|---:|---:|---:|
| **RTX PRO 6000 Blackwell** | PyTorch | 8 | 66.54 | 118.95 | 427.96 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 8 | 26.29 | 156.16 | 540.65 |
|  | NIM | 8 | 14.48 | 100.23 | 289.93 |
| **H20** | PyTorch | 8 | 28.22 | 153.19 | 491.93 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 8 | 27.65 | 165.91 | 530.44 |
|  | NIM | 8 | 14.06 | 114.49 | 405.68 |
| **H100 NVL** | PyTorch | 8 | 16.96 | 64.17 | 186.47 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 8 | 25.45 | 112.36 | 325.56 |
|  | NIM | 8 | 13.30 | 67.34 | 201.39 |
| **H200 NVL** | PyTorch | 8 | 24.70 | 47.60 | 141.62 |
|  | vLLM-Omni | 1 | 27.90 | 254.29 | 915.05 |
|  | Diffusers | 1 | 33.00 | 287.20 | 1034.60 |
|  | NIM | 1 | 17.51 | 201.45 | 817.35 |
| **H100 80GB HBM3** | PyTorch | — |  |  |  |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 8 | 19.98 | 55.63 | 184.81 |
|  | NIM | 8 | 6.49 | 36.81 | 118.77 |
| **H200 141GB HBM3** | PyTorch | 8 | 11.80 | 42.10 | 123.57 |
|  | vLLM-Omni | 1 | 25.47 | 220.70 | 766.33 |
|  | Diffusers | 1 | 31.00 | 249.20 | 879.20 |
|  | NIM | 1 | 16.39 | 175.95 | 699.13 |
| **B200** | PyTorch | 1 | 14.71 | 112.40 | 397.31 |
|  | vLLM-Omni | 1 | 14.13 | 115.17 | 393.02 |
|  | Diffusers | 1 | 19.20 | 127.00 | 414.80 |
|  | NIM | 1 | 9.36 | 83.19 | 316.76 |
| **B300** | PyTorch | — |  |  |  |
|  | vLLM-Omni | 1 | 14.14 | 111.42 | 368.73 |
|  | Diffusers | 1 | 25.21 | 132.23 | 410.81 |
|  | NIM | 1 | 9.73 | 80.51 | 294.11 |

### Text-to-Image (t2i)

| GPU | Engine | GPUs | 256p | 480p | 720p |
|---|---|:-:|---:|---:|---:|
| **RTX PRO 6000 Blackwell** | PyTorch | 8 | 93.18 | 93.39 | 93.47 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 8 | 16.90 | 17.52 | 16.73 |
| **H20** | PyTorch | 8 | 14.18 | 16.46 | 20.92 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 8 | 18.29 | 18.48 | 18.46 |
| **H100 NVL** | PyTorch | 8 | 19.86 | 19.80 | 19.87 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 8 | 16.56 | 16.60 | 16.88 |
| **H200 NVL** | PyTorch | 8 | 32.86 | 33.05 | 33.16 |
|  | vLLM-Omni | 1 | 2.73 | 6.28 | 11.02 |
|  | Diffusers | 1 | 5.00 | 8.00 | 12.00 |
| **H100 80GB HBM3** | PyTorch | — |  |  |  |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 8 | 17.97 | 18.03 | 19.28 |
| **H200 141GB HBM3** | PyTorch | 8 | 13.48 | 13.53 | 13.50 |
|  | vLLM-Omni | 1 | 2.83 | 5.70 | 10.24 |
|  | Diffusers | 1 | 5.00 | 8.00 | 11.00 |
| **B200** | PyTorch | 1 | 4.51 | 4.78 | 7.25 |
|  | vLLM-Omni | 1 | 2.32 | 3.29 | 6.02 |
|  | Diffusers | 1 | 4.40 | 6.00 | 8.00 |
| **B300** | PyTorch | — |  |  |  |
|  | vLLM-Omni | 1 | 5.05 | 3.79 | 7.08 |
|  | Diffusers | 1 | 6.49 | 6.76 | 7.11 |

<sub>Notes:
1. All times measured on identical workloads (same seed, sampler settings, prompt).
2. Multi-GPU configurations use tensor parallelism.
3. vLLM-Omni numbers are for the upcoming public release in the vLLM-Omni repo; subject to change before GA. Current vLLM-Omni coverage is B200 at 720p.
4. Diffusers numbers use the HuggingFace `diffusers` integration without custom CUDA graphs, and were measured on a single GPU only.
5. At 256p, multi-GPU configurations on B300 may underperform single-GPU because of small-workload tensor-parallel overhead, so single-GPU is the recommended deployment at this resolution.
6. PyTorch numbers report average generation (sampling) time from OSS inference benchmarking.
7. NIM numbers use latency profiles with FP8 precision and report end-to-end `Request Latency s`, including request processing, video generation, output encoding, and returning the response.</sub>

## Additional audiovisual generation

### Video-to-Video (v2v)

A text prompt and source video condition a generated continuation or transformation.

| GPU | Engine | GPUs | 256p | 480p | 720p |
|---|---|:-:|---:|---:|---:|
| **RTX PRO 6000 Blackwell** | PyTorch | 8 | 66.31 | 118.92 | 427.19 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 8 | 28.39 | 158.14 | 545.45 |
| **H20** | PyTorch | 8 | 27.98 | 153.02 | 491.75 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 8 | 28.18 | 167.95 | 532.68 |
| **H100 NVL** | PyTorch | 8 | 16.79 | 63.84 | 183.64 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 8 | 25.01 | 112.47 | 348.31 |
| **H200 NVL** | PyTorch | 8 | 24.74 | 47.60 | 142.23 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 1 | 40.73 | 316.00 | 1093.16 |
| **H100 80GB HBM3** | PyTorch | — |  |  |  |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 8 | 21.61 | 57.74 | 188.43 |
| **H200 141GB HBM3** | PyTorch | 8 | 11.76 | 41.85 | 123.09 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 1 | 41.17 | 269.44 | 916.41 |
| **B200** | PyTorch | 1 | 14.30 | 111.49 | 395.97 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 1 | 26.30 | 142.61 | 437.27 |
| **B300** | PyTorch | — |  |  |  |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 1 | 25.86 | 139.04 | 416.98 |

### Text-to-Audio-and-Video (t2av)

A text prompt produces synchronized video and sound.

| GPU | Engine | GPUs | 256p | 480p | 720p |
|---|---|:-:|---:|---:|---:|
| **RTX PRO 6000 Blackwell** | PyTorch | 8 | 65.79 | 118.87 | 429.78 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | — |  |  |  |
| **H20** | PyTorch | 8 | 27.73 | 152.53 | 492.38 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | — |  |  |  |
| **H100 NVL** | PyTorch | 8 | 16.86 | 64.17 | 183.81 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | — |  |  |  |
| **H200 NVL** | PyTorch | 8 | 24.79 | 47.67 | 142.06 |
|  | vLLM-Omni | 1 | 29.07 | 260.51 | 916.51 |
|  | Diffusers | 1 | 35.00 | 291.40 | 1034.20 |
| **H100 80GB HBM3** | PyTorch | — |  |  |  |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | — |  |  |  |
| **H200 141GB HBM3** | PyTorch | 8 | 11.74 | 41.49 | 123.35 |
|  | vLLM-Omni | 1 | 27.21 | 221.67 | 763.74 |
|  | Diffusers | 1 | 33.00 | 253.40 | 874.60 |
| **B200** | PyTorch | 1 | 14.52 | 112.99 | 395.20 |
|  | vLLM-Omni | 1 | 14.77 | 115.26 | 388.51 |
|  | Diffusers | 1 | 20.00 | 128.00 | 422.00 |
| **B300** | PyTorch | — |  |  |  |
|  | vLLM-Omni | 1 | 15.26 | 113.34 | 372.74 |
|  | Diffusers | 1 | 18.94 | 126.86 | 402.37 |

### Video-to-Audio-and-Video (v2av)

A text prompt and source video produce transformed video with synchronized sound.

| GPU | Engine | GPUs | 256p | 480p | 720p |
|---|---|:-:|---:|---:|---:|
| **RTX PRO 6000 Blackwell** | PyTorch | 8 | 66.10 | 118.88 | 428.64 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | — |  |  |  |
| **H20** | PyTorch | 8 | 28.00 | 152.78 | 492.63 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | — |  |  |  |
| **H100 NVL** | PyTorch | 8 | 16.79 | 64.69 | 183.62 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | — |  |  |  |
| **H200 NVL** | PyTorch | 8 | 24.73 | 47.67 | 143.06 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 1 | 48.42 | 320.70 | 1093.26 |
| **H100 80GB HBM3** | PyTorch | — |  |  |  |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | — |  |  |  |
| **H200 141GB HBM3** | PyTorch | 8 | 11.74 | 41.59 | 123.06 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 1 | 43.23 | 270.19 | 909.99 |
| **B200** | PyTorch | 1 | 14.28 | 111.84 | 406.66 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 1 | 30.26 | 141.75 | 446.20 |
| **B300** | PyTorch | — |  |  |  |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 1 | 30.36 | 138.17 | 420.63 |

### Image-to-Audio-and-Video (i2av)

A text prompt and source image produce video with synchronized sound.

| GPU | Engine | GPUs | 256p | 480p | 720p |
|---|---|:-:|---:|---:|---:|
| **RTX PRO 6000 Blackwell** | PyTorch | 8 | 66.27 | 119.05 | 429.26 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | — |  |  |  |
| **H20** | PyTorch | 8 | 28.11 | 153.33 | 493.51 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | — |  |  |  |
| **H100 NVL** | PyTorch | 8 | 16.85 | 64.19 | 186.66 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | — |  |  |  |
| **H200 NVL** | PyTorch | 8 | 24.69 | 47.65 | 142.18 |
|  | vLLM-Omni | 1 | 29.42 | 262.20 | 922.18 |
|  | Diffusers | 1 | 35.00 | 292.20 | 1037.00 |
| **H100 80GB HBM3** | PyTorch | — |  |  |  |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | — |  |  |  |
| **H200 141GB HBM3** | PyTorch | 8 | 11.82 | 41.95 | 123.42 |
|  | vLLM-Omni | 1 | 27.58 | 220.57 | 774.15 |
|  | Diffusers | 1 | 33.00 | 256.00 | 880.40 |
| **B200** | PyTorch | 1 | 14.84 | 113.32 | 407.30 |
|  | vLLM-Omni | 1 | 15.15 | 116.54 | 396.11 |
|  | Diffusers | 1 | 20.00 | 128.00 | 423.00 |
| **B300** | PyTorch | — |  |  |  |
|  | vLLM-Omni | 1 | 16.46 | 114.02 | 373.17 |
|  | Diffusers | 1 | 29.16 | 134.46 | 414.22 |

## Action generation

Forward dynamics is reported separately for AV, camera, and robot inputs. Inverse dynamics and policy are reported for AV and robot because those are the only domain rows in PBR `#308197`; no camera row is inferred. For each domain, PyTorch is populated from that report. vLLM-Omni currently covers forward dynamics AV and inverse dynamics AV from PBR `#308481`. Camera, robot-FD, robot-ID, policy-AV, and policy-robot vLLM-Omni rows remain reserved. **Policy — DROID** is a separate checkpoint and is not mixed with the policy-robot table; this Super page has no Policy-DROID measurements.

### Forward Dynamics — Autonomous Vehicle (AV)

| GPU | Engine | GPUs | 256p | 480p | 720p |
|---|---|:-:|---:|---:|---:|
| **RTX PRO 6000 Blackwell** | PyTorch | 8 | 59.39 | 59.31 | 59.35 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 8 | 11.99 | 22.93 | 23.68 |
| **H20** | PyTorch | 8 | 41.51 | 41.47 | 41.42 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 8 | 11.40 | 26.98 | 28.79 |
| **H100 NVL** | PyTorch | 8 | 21.00 | 21.00 | 21.04 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 8 | 10.30 | 19.29 | 20.05 |
| **H200 NVL** | PyTorch | 8 | 22.66 | 22.67 | 22.65 |
|  | vLLM-Omni | 1 | 4.33 | 28.78 | 82.08 |
|  | Diffusers | 1 | 27.17 | 53.19 | 51.89 |
| **H100 80GB HBM3** | PyTorch | — |  |  |  |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 8 | 10.34 | 12.89 | 13.45 |
| **H200 141GB HBM3** | PyTorch | 8 | 14.41 | 14.41 | 14.41 |
|  | vLLM-Omni | 1 | 3.99 | 26.04 | 71.61 |
|  | Diffusers | 1 | 26.51 | 47.34 | 50.20 |
| **B200** | PyTorch | 1 | 13.59 | 13.55 | 13.86 |
|  | vLLM-Omni | 1 | 2.37 | 14.59 | 38.41 |
|  | Diffusers | 1 | 4.00 | 16.00 | 16.00 |
| **B300** | PyTorch | — |  |  |  |
|  | vLLM-Omni | 1 | 2.85 | 14.20 | 36.96 |
|  | Diffusers | 1 | 22.18 | 35.41 | 38.82 |

### Forward Dynamics — Camera

| GPU | Engine | GPUs | 256p | 480p | 720p |
|---|---|:-:|---:|---:|---:|
| **RTX PRO 6000 Blackwell** | PyTorch | 8 | 59.62 | 59.39 | 59.55 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | — |  |  |  |
| **H20** | PyTorch | 8 | 41.13 | 41.10 | 41.11 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | — |  |  |  |
| **H100 NVL** | PyTorch | 8 | 21.05 | 20.98 | 21.00 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | — |  |  |  |
| **H200 NVL** | PyTorch | 8 | 22.55 | 22.54 | 22.54 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | — |  |  |  |
| **H100 80GB HBM3** | PyTorch | — |  |  |  |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | — |  |  |  |
| **H200 141GB HBM3** | PyTorch | 8 | 14.30 | 14.30 | 14.29 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | — |  |  |  |
| **B200** | PyTorch | 1 | 13.73 | 13.72 | 13.58 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | — |  |  |  |
| **B300** | PyTorch | — |  |  |  |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | — |  |  |  |

### Forward Dynamics — Robot

| GPU | Engine | GPUs | 256p | 480p | 720p |
|---|---|:-:|---:|---:|---:|
| **RTX PRO 6000 Blackwell** | PyTorch | 8 | 56.35 | 56.38 | 56.01 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 8 | 9.40 | 10.60 | 10.82 |
| **H20** | PyTorch | 8 | 15.60 | 15.61 | 15.59 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 8 | 8.95 | 10.24 | 10.65 |
| **H100 NVL** | PyTorch | 8 | 12.90 | 12.90 | 12.90 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 8 | 8.03 | 8.49 | 8.61 |
| **H200 NVL** | PyTorch | 8 | 20.16 | 20.15 | 20.15 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 1 | 22.21 | 28.53 | 27.35 |
| **H100 80GB HBM3** | PyTorch | — |  |  |  |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 8 | 10.27 | 9.04 | 9.25 |
| **H200 141GB HBM3** | PyTorch | 8 | 8.74 | 8.74 | 8.70 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 1 | 22.02 | 26.45 | 26.38 |
| **B200** | PyTorch | 1 | 3.50 | 3.49 | 3.47 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 1 | 3.00 | 5.00 | 5.00 |
| **B300** | PyTorch | — |  |  |  |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 1 | 22.21 | 21.81 | 24.51 |

### Inverse Dynamics — Autonomous Vehicle (AV)

| GPU | Engine | GPUs | 256p | 480p | 720p |
|---|---|:-:|---:|---:|---:|
| **RTX PRO 6000 Blackwell** | PyTorch | 8 | 59.20 | 59.14 | 59.07 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 8 | 11.15 | 23.33 | 23.28 |
| **H20** | PyTorch | 8 | 41.17 | 40.96 | 41.06 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 8 | 11.33 | 26.66 | 28.46 |
| **H100 NVL** | PyTorch | 8 | 20.76 | 21.05 | 20.69 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 8 | 9.72 | 18.67 | 20.67 |
| **H200 NVL** | PyTorch | 8 | 22.40 | 22.40 | 22.42 |
|  | vLLM-Omni | 1 | 4.36 | 28.21 | 80.85 |
|  | Diffusers | 1 | 15.22 | 44.50 | 43.51 |
| **H100 80GB HBM3** | PyTorch | — |  |  |  |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 8 | 10.49 | 12.73 | 13.30 |
| **H200 141GB HBM3** | PyTorch | 8 | 14.30 | 14.32 | 14.16 |
|  | vLLM-Omni | 1 | 4.03 | 25.99 | 71.45 |
|  | Diffusers | 1 | 16.58 | 39.75 | 41.71 |
| **B200** | PyTorch | 1 | 13.38 | 13.37 | 13.36 |
|  | vLLM-Omni | 1 | 2.42 | 14.69 | 38.74 |
|  | Diffusers | 1 | 4.00 | 16.00 | 16.00 |
| **B300** | PyTorch | — |  |  |  |
|  | vLLM-Omni | 1 | 2.69 | 14.44 | 37.70 |
|  | Diffusers | 1 | 15.13 | 26.24 | 28.51 |

### Inverse Dynamics — Robot

| GPU | Engine | GPUs | 256p | 480p | 720p |
|---|---|:-:|---:|---:|---:|
| **RTX PRO 6000 Blackwell** | PyTorch | 8 | 56.28 | 56.33 | 56.00 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 8 | 9.59 | 10.90 | 8.71 |
| **H20** | PyTorch | 8 | 15.51 | 15.52 | 15.51 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 8 | 9.48 | 10.17 | 10.65 |
| **H100 NVL** | PyTorch | 8 | 12.51 | 12.51 | 12.51 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 8 | 7.92 | 8.46 | 8.79 |
| **H200 NVL** | PyTorch | 8 | 20.11 | 20.10 | 20.12 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 1 | 12.09 | 16.87 | 18.74 |
| **H100 80GB HBM3** | PyTorch | — |  |  |  |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 8 | 8.71 | 8.93 | 9.08 |
| **H200 141GB HBM3** | PyTorch | 8 | 8.68 | 8.68 | 8.65 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 1 | 25.77 | 16.25 | 16.29 |
| **B200** | PyTorch | 1 | 3.42 | 3.49 | 3.41 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 1 | 3.00 | 5.00 | 5.00 |
| **B300** | PyTorch | — |  |  |  |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 1 | 9.65 | 15.08 | 15.21 |

### Policy — Autonomous Vehicle (AV)

| GPU | Engine | GPUs | 256p | 480p | 720p |
|---|---|:-:|---:|---:|---:|
| **RTX PRO 6000 Blackwell** | PyTorch | 8 | 59.66 | 59.51 | 59.49 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 8 | 12.11 | 22.60 | 25.76 |
| **H20** | PyTorch | 8 | 41.54 | 41.51 | 41.49 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 8 | 11.35 | 26.61 | 28.81 |
| **H100 NVL** | PyTorch | 8 | 21.02 | 21.00 | 21.31 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 8 | 9.77 | 18.80 | 20.12 |
| **H200 NVL** | PyTorch | 8 | 22.64 | 22.63 | 22.64 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 1 | 15.35 | 41.89 | 45.40 |
| **H100 80GB HBM3** | PyTorch | — |  |  |  |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 8 | 10.55 | 13.02 | 13.52 |
| **H200 141GB HBM3** | PyTorch | 8 | 14.54 | 14.41 | 14.52 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 1 | 14.71 | 41.30 | 40.35 |
| **B200** | PyTorch | 1 | 13.71 | 13.73 | 13.60 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 1 | 4.00 | 16.00 | 16.00 |
| **B300** | PyTorch | — |  |  |  |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 1 | 15.29 | 28.62 | 28.87 |

### Policy — Robot

| GPU | Engine | GPUs | 256p | 480p | 720p |
|---|---|:-:|---:|---:|---:|
| **RTX PRO 6000 Blackwell** | PyTorch | 8 | 56.31 | 56.29 | 55.95 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 8 | 9.57 | 10.08 | 8.76 |
| **H20** | PyTorch | 8 | 15.60 | 15.60 | 15.62 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 8 | 9.03 | 10.19 | 10.73 |
| **H100 NVL** | PyTorch | 8 | 12.56 | 12.55 | 12.54 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 8 | 8.05 | 8.60 | 8.89 |
| **H200 NVL** | PyTorch | 8 | 20.16 | 20.17 | 20.16 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 1 | 13.82 | 16.92 | 17.03 |
| **H100 80GB HBM3** | PyTorch | — |  |  |  |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 8 | 8.71 | 9.09 | 9.12 |
| **H200 141GB HBM3** | PyTorch | 8 | 8.72 | 8.75 | 8.70 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 1 | 11.90 | 16.13 | 14.72 |
| **B200** | PyTorch | 1 | 3.48 | 3.49 | 3.48 |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 1 | 3.00 | 5.00 | 5.00 |
| **B300** | PyTorch | — |  |  |  |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | 1 | 10.57 | 15.31 | 15.33 |

### Policy — DROID

PBR `#308481` measured Policy-DROID only for Cosmos3-Nano-Policy-DROID. Super has no Policy-DROID row in that sweep; this table is reserved.

| GPU | Engine | GPUs | 256p | 480p | 720p |
|---|---|:-:|---:|---:|---:|
| **RTX PRO 6000 Blackwell** | PyTorch | — |  |  |  |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | — |  |  |  |
| **H20** | PyTorch | — |  |  |  |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | — |  |  |  |
| **H100 NVL** | PyTorch | — |  |  |  |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | — |  |  |  |
| **H200 NVL** | PyTorch | — |  |  |  |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | — |  |  |  |
| **H100 80GB HBM3** | PyTorch | — |  |  |  |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | — |  |  |  |
| **H200 141GB HBM3** | PyTorch | — |  |  |  |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | — |  |  |  |
| **B200** | PyTorch | — |  |  |  |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | — |  |  |  |
| **B300** | PyTorch | — |  |  |  |
|  | vLLM-Omni | — |  |  |  |
|  | Diffusers | — |  |  |  |

<sub>Additional-modality notes:
1. PyTorch values are average generation (sampling) latency in seconds from PBR `#308197`; lower is better.
2. PyTorch values use `CUDA_GRAPH=No` and the `latency` automatic-sharding preset.
3. vLLM-Omni audiovisual values for `t2av`/`t2vs` and `i2av`/`i2vs` are from PBR `#308195`. vLLM-Omni action values are from PBR `#308481`.
4. Action vLLM-Omni rows report one GPU because the 8-GPU Ulysses runs failed sequence-length divisibility checks.
5. Policy-DROID is reported in its own table. PBR `#308481` has no Super Policy-DROID measurement.
6. Diffusers rows are intentionally empty reservations for future benchmark campaigns.
7. The **GPUs** column gives the number of GPUs behind every value in that row.
8. Empty cells indicate unmeasured combinations, not unsupported combinations.</sub>