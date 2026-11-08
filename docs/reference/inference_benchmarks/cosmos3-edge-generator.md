# Cosmos3-Edge Generator Inference Benchmarks

[Back to the inference benchmark index](../inference_benchmarks.md)

See [measurement settings and reproduction limits](benchmark_methodology.md) for runtime-specific timing, precision, and workload details.

These tables report **Cosmos3-Edge** Generator latency in seconds. Lower latency is better, and empty cells have no publishable measurement at the stated configuration; they do not establish lack of support.

Cosmos3-Edge supports **256p and 480p**, so 480p is its highest resolution. Every row reports one GPU configuration, named in the **GPUs** column: one GPU where the model fits on a single device at 480p, and eight GPUs where it does not. Cosmos3-Edge fits on a single GPU or one integrated computing platform at 480p, so all populated rows report one. Read the **GPUs** column and the timing notes before comparing engines.

The historical PyTorch and vLLM-Omni video benchmarks generate **121 frames**. The SGLang sweep does not retain its full frame/step budget alongside the scalar timings. The datacenter vLLM-Omni, PyTorch, Diffusers, SGLang, and TensorRT-LLM values report generation latency. Historical embedded vLLM-Omni rows use end-to-end latency and are not directly comparable to generation-only timings.

## Table of Contents

- [Text-to-Image (t2i)](#text-to-image-t2i)
- [Text-to-Video (t2v)](#text-to-video-t2v)
- [Image-to-Video (i2v)](#image-to-video-i2v)
- [Video-to-Video (v2v)](#video-to-video-v2v)
- [Action generation](#action-generation)

## Text-to-Image (t2i)

| GPU or Platform | Engine | GPUs | 256p | 480p |
|---|---|:-:|---:|---:|
| **B200 SXM 192 GB** | PyTorch | 1 |  |  |
|  | vLLM-Omni | — |  |  |
|  | TensorRT-LLM | 1 |  |  |
|  | Diffusers | 1 |  |  |
|  | SGLang | 1 |  |  |
| **B300** | PyTorch | 1 |  |  |
|  | vLLM-Omni | — |  |  |
|  | TensorRT-LLM | 1 |  |  |
|  | Diffusers | 1 |  |  |
|  | SGLang | 1 |  |  |
| **H200 SXM 141 GB** | PyTorch | 1 |  |  |
|  | vLLM-Omni | — |  |  |
|  | TensorRT-LLM | 1 |  |  |
|  | Diffusers | 1 |  |  |
|  | SGLang | 1 |  |  |
| **H200 NVL** | PyTorch | 1 | 2.23 | 2.32 |
|  | vLLM-Omni | — |  |  |
|  | TensorRT-LLM | — |  |  |
|  | Diffusers | 1 |  |  |
| **H100 SXM 80 GB** | PyTorch | 1 |  |  |
|  | vLLM-Omni | — |  |  |
|  | TensorRT-LLM | 1 |  |  |
|  | Diffusers | 1 |  |  |
|  | SGLang | 1 |  |  |
| **H100 NVL 96 GB** | PyTorch | 1 | 2.59 | 2.47 |
|  | vLLM-Omni | — |  |  |
|  | TensorRT-LLM | — |  |  |
|  | Diffusers | 1 |  | 2.66 |
| **H20 SXM 96 GB** | PyTorch | 1 |  |  |
|  | vLLM-Omni | — |  |  |
|  | TensorRT-LLM | 1 |  |  |
|  | Diffusers | 1 |  |  |
|  | SGLang | 1 |  |  |
| **RTX PRO 6000 Blackwell Server Edition** | PyTorch | 1 |  |  |
|  | vLLM-Omni | — |  |  |
|  | TensorRT-LLM | 1 |  |  |
|  | Diffusers | 1 |  |  |
|  | SGLang | 1 |  |  |
| **DGX Station** | PyTorch | 1 | 3.45 | 3.32 |
|  | vLLM-Omni | — |  |  |
|  | TensorRT-LLM | — |  |  |
| **DGX Spark** | PyTorch | 1 | 7.73 | 8.87 |
|  | vLLM-Omni | — |  |  |
|  | TensorRT-LLM | — |  |  |
| **Jetson AGX Thor T5000, 128 GB, MAXN** | PyTorch | — |  |  |
|  | vLLM-Omni | — |  |  |
|  | TensorRT-LLM | — |  |  |
| **Jetson T3000, 32 GB, 1100 MHz** | PyTorch | — |  |  |
|  | vLLM-Omni | — |  |  |
|  | TensorRT-LLM | — |  |  |
| **Jetson T2000, 16 GB, 702 MHz, THOR_NANO** | PyTorch | — |  |  |
|  | vLLM-Omni | — |  |  |
|  | TensorRT-LLM | — |  |  |

## Text-to-Video (t2v)

| GPU or Platform | Engine | GPUs | 256p | 480p |
|---|---|:-:|---:|---:|
| **B200 SXM 192 GB** | PyTorch | 1 |  | 7.83 |
|  | vLLM-Omni | 1 |  |  |
|  | TensorRT-LLM | 1 |  | 7.84 |
|  | Diffusers | 1 |  | 13.12 |
|  | SGLang | 1 |  | 6.66 |
| **B300** | PyTorch | 1 |  | 7.09 |
|  | vLLM-Omni | 1 |  |  |
|  | TensorRT-LLM | 1 |  | 7.22 |
|  | Diffusers | 1 |  |  |
|  | SGLang | 1 |  |  |
| **H200 SXM 141 GB** | PyTorch | 1 | 2.66 | 12.81 |
|  | vLLM-Omni | 1 |  | 13.11 |
|  | TensorRT-LLM | 1 |  | 14.89 |
|  | Diffusers | 1 |  | 20.80 |
|  | SGLang | 1 |  | 10.59 |
| **H200 NVL** | PyTorch | 1 | 2.88 | 14.49 |
|  | vLLM-Omni | 1 | 2.05 | 14.11 |
|  | TensorRT-LLM | — |  |  |
|  | Diffusers | 1 | 3.91 | 22.81 |
| **H100 SXM 80 GB** | PyTorch | 1 | 2.75 | 13.25 |
|  | vLLM-Omni | 1 |  | 13.30 |
|  | TensorRT-LLM | 1 |  | 15.09 |
|  | Diffusers | 1 |  |  |
|  | SGLang | 1 |  |  |
| **H100 NVL 96 GB** | PyTorch | 1 | 3.35 | 17.12 |
|  | vLLM-Omni | 1 |  | 17.29 |
|  | TensorRT-LLM | — |  |  |
|  | Diffusers | 1 |  | 26.90 |
| **H20 SXM 96 GB** | PyTorch | 1 | 10.51 | 54.60 |
|  | vLLM-Omni | 1 | 5.73 | 53.98 |
|  | TensorRT-LLM | 1 | 6.91 | 59.19 |
|  | Diffusers | 1 | 9.49 | 69.45 |
|  | SGLang | 1 | 5.71 | 46.01 |
| **RTX PRO 6000 Blackwell Server Edition** | PyTorch | 1 | 4.30 | 22.72 |
|  | vLLM-Omni | — |  |  |
|  | TensorRT-LLM | 1 | 2.77 | 26.13 |
|  | Diffusers | 1 | 4.44 | 34.50 |
|  | SGLang | 1 | 2.29 | 19.81 |
| **DGX Station** | PyTorch | 1 | 2.93 | 6.76 |
|  | vLLM-Omni | 1 | 3.44 | 7.09 |
|  | TensorRT-LLM | — |  |  |
| **DGX Spark** | PyTorch | 1 | 20.23 | 108.78 |
|  | vLLM-Omni | 1 | 11.46 | 89.80 |
|  | TensorRT-LLM | — |  |  |
| **Jetson AGX Thor T5000, 128 GB, MAXN** | PyTorch | — |  |  |
|  | vLLM-Omni | — |  |  |
|  | TensorRT-LLM | — |  |  |
| **Jetson T3000, 32 GB, 1100 MHz** | PyTorch | — |  |  |
|  | vLLM-Omni | — |  |  |
|  | TensorRT-LLM | — |  |  |
| **Jetson T2000, 16 GB, 702 MHz, THOR_NANO** | PyTorch | — |  |  |
|  | vLLM-Omni | — |  |  |
|  | TensorRT-LLM | — |  |  |

## Image-to-Video (i2v)

| GPU or Platform | Engine | GPUs | 256p | 480p |
|---|---|:-:|---:|---:|
| **B200 SXM 192 GB** | PyTorch | 1 |  | 7.45 |
|  | vLLM-Omni | 1 |  |  |
|  | TensorRT-LLM | 1 |  | 8.55 |
|  | Diffusers | 1 |  | 13.49 |
|  | SGLang | 1 |  | 6.78 |
| **B300** | PyTorch | 1 |  | 6.84 |
|  | vLLM-Omni | 1 |  |  |
|  | TensorRT-LLM | 1 |  | 8.04 |
|  | Diffusers | 1 |  |  |
|  | SGLang | 1 |  |  |
| **H200 SXM 141 GB** | PyTorch | 1 | 2.59 | 12.31 |
|  | vLLM-Omni | 1 |  | 13.13 |
|  | TensorRT-LLM | 1 | 1.99 | 15.89 |
|  | Diffusers | 1 |  |  |
|  | SGLang | 1 |  |  |
| **H200 NVL** | PyTorch | 1 | 2.83 | 14.07 |
|  | vLLM-Omni | 1 |  | 14.15 |
|  | TensorRT-LLM | — |  |  |
|  | Diffusers | 1 |  | 23.13 |
| **H100 SXM 80 GB** | PyTorch | 1 | 2.71 | 12.68 |
|  | vLLM-Omni | 1 |  | 13.33 |
|  | TensorRT-LLM | 1 |  | 15.95 |
|  | Diffusers | 1 |  |  |
|  | SGLang | 1 |  |  |
| **H100 NVL 96 GB** | PyTorch | 1 | 3.31 | 16.42 |
|  | vLLM-Omni | 1 |  | 17.33 |
|  | TensorRT-LLM | — |  |  |
|  | Diffusers | 1 |  | 27.94 |
| **H20 SXM 96 GB** | PyTorch | 1 | 10.49 | 52.83 |
|  | vLLM-Omni | 1 | 5.75 | 53.94 |
|  | TensorRT-LLM | 1 | 7.50 | 62.08 |
|  | Diffusers | 1 | 10.09 | 72.40 |
|  | SGLang | 1 | 5.82 | 46.39 |
| **RTX PRO 6000 Blackwell Server Edition** | PyTorch | 1 | 4.25 | 21.92 |
|  | vLLM-Omni | — |  |  |
|  | TensorRT-LLM | 1 | 3.09 | 27.55 |
|  | Diffusers | 1 | 4.71 | 35.73 |
|  | SGLang | 1 | 2.37 | 19.97 |
| **DGX Station** | PyTorch | 1 |  |  |
|  | vLLM-Omni | 1 | 3.49 | 7.13 |
|  | TensorRT-LLM | — |  |  |
| **DGX Spark** | PyTorch | 1 |  |  |
|  | vLLM-Omni | 1 | 11.36 | 89.41 |
|  | TensorRT-LLM | — |  |  |
| **Jetson AGX Thor T5000, 128 GB, MAXN** | PyTorch | — |  |  |
|  | vLLM-Omni | — |  |  |
|  | TensorRT-LLM | — |  |  |
| **Jetson T3000, 32 GB, 1100 MHz** | PyTorch | — |  |  |
|  | vLLM-Omni | — |  |  |
|  | TensorRT-LLM | — |  |  |
| **Jetson T2000, 16 GB, 702 MHz, THOR_NANO** | PyTorch | — |  |  |
|  | vLLM-Omni | — |  |  |
|  | TensorRT-LLM | — |  |  |

## Video-to-Video (v2v)

| GPU or Platform | Engine | GPUs | 256p | 480p |
|---|---|:-:|---:|---:|
| **B200 SXM 192 GB** | PyTorch | 1 |  | 7.28 |
|  | vLLM-Omni | — |  |  |
|  | TensorRT-LLM | 1 |  | 7.94 |
|  | Diffusers | 1 |  | 14.25 |
| **B300** | PyTorch | 1 |  |  |
|  | vLLM-Omni | — |  |  |
|  | TensorRT-LLM | 1 |  | 7.35 |
|  | Diffusers | 1 |  |  |
| **H200 SXM 141 GB** | PyTorch | 1 | 2.57 | 11.95 |
|  | vLLM-Omni | — |  |  |
|  | TensorRT-LLM | 1 |  | 15.09 |
|  | Diffusers | 1 |  | 22.42 |
| **H200 NVL** | PyTorch | 1 | 2.71 | 13.53 |
|  | vLLM-Omni | — |  |  |
|  | TensorRT-LLM | — |  |  |
|  | Diffusers | 1 | 4.49 | 23.96 |
| **H100 SXM 80 GB** | PyTorch | 1 | 2.68 | 12.32 |
|  | vLLM-Omni | — |  |  |
|  | TensorRT-LLM | 1 |  | 15.28 |
|  | Diffusers | 1 |  | 23.52 |
| **H100 NVL 96 GB** | PyTorch | 1 | 2.99 | 15.91 |
|  | vLLM-Omni | — |  |  |
|  | TensorRT-LLM | — |  |  |
|  | Diffusers | 1 | 4.76 | 28.84 |
| **H20 SXM 96 GB** | PyTorch | 1 | 9.17 | 50.89 |
|  | vLLM-Omni | — |  |  |
|  | TensorRT-LLM | 1 | 7.04 | 60.03 |
|  | Diffusers | 1 | 10.47 | 73.35 |
| **RTX PRO 6000 Blackwell Server Edition** | PyTorch | 1 | 3.75 | 21.15 |
|  | vLLM-Omni | — |  |  |
|  | TensorRT-LLM | 1 | 2.86 | 26.21 |
|  | Diffusers | 1 | 5.07 | 36.61 |
| **DGX Station** | PyTorch | 1 | 2.90 | 6.17 |
|  | vLLM-Omni | — |  |  |
|  | TensorRT-LLM | — |  |  |
| **DGX Spark** | PyTorch | 1 | 17.54 | 101.66 |
|  | vLLM-Omni | — |  |  |
|  | TensorRT-LLM | — |  |  |
| **Jetson AGX Thor T5000, 128 GB, MAXN** | PyTorch | — |  |  |
|  | vLLM-Omni | — |  |  |
|  | TensorRT-LLM | — |  |  |
| **Jetson T3000, 32 GB, 1100 MHz** | PyTorch | — |  |  |
|  | vLLM-Omni | — |  |  |
|  | TensorRT-LLM | — |  |  |
| **Jetson T2000, 16 GB, 702 MHz, THOR_NANO** | PyTorch | — |  |  |
|  | vLLM-Omni | — |  |  |
|  | TensorRT-LLM | — |  |  |

<sub>Notes:
1. The **GPUs** column gives the number of GPUs behind every value in that row. All current measurements use one GPU or one integrated computing platform.
2. Values are average end-to-end or generation latency in seconds; lower is better.
3. TensorRT-LLM values come from a TensorRT-LLM campaign that covers the six datacenter GPUs only; the DGX and Jetson platforms were not part of that campaign.
4. vLLM-Omni and PyTorch were measured for i2v only, so their rows are empty in the other three sections.
5. Video measurements generate **121 output frames**.
6. PyTorch values report average generation latency rather than diffusion-only latency.
7. Diffusers and NIM have no Cosmos3-Edge coverage yet, so they are not listed.</sub>

## Action generation

### Forward Dynamics — Autonomous Vehicle (AV)

| GPU | Engine | GPUs | 256p | 480p |
|---|---|:-:|---:|---:|
| **B200** | SGLang | 1 | 0.49 | 1.53 |
| **B300** | SGLang | 1 | 0.46 | 1.37 |
| **H100 80GB HBM3** | SGLang | 1 | 0.52 | 2.53 |
| **H20** | SGLang | 1 | 1.56 | 10.10 |
| **H200 141GB HBM3** | SGLang | 1 | 0.50 | 2.49 |
| **RTX PRO 6000 Blackwell** | SGLang | 1 | 0.69 | 4.39 |

### Forward Dynamics — Camera

| GPU | Engine | GPUs | 256p | 480p |
|---|---|:-:|---:|---:|
| **B200** | SGLang | 1 |  | 1.56 |
| **B300** | SGLang | 1 |  | 1.40 |
| **H100 80GB HBM3** | SGLang | 1 |  | 2.58 |
| **H20** | SGLang | 1 | 1.57 | 10.05 |
| **H200 141GB HBM3** | SGLang | 1 |  | 2.50 |
| **RTX PRO 6000 Blackwell** | SGLang | 1 | 0.69 | 4.41 |

### Forward Dynamics — Robot

| GPU | Engine | GPUs | 256p | 480p |
|---|---|:-:|---:|---:|
| **B200** | SGLang | 1 |  |  |
| **B300** | SGLang | 1 | 0.34 |  |
| **H100 80GB HBM3** | SGLang | 1 |  |  |
| **H20** | SGLang | 1 | 0.48 | 2.30 |
| **H200 141GB HBM3** | SGLang | 1 |  |  |
| **RTX PRO 6000 Blackwell** | SGLang | 1 | 0.31 | 1.05 |

### Inverse Dynamics — Autonomous Vehicle (AV)

| GPU | Engine | GPUs | 256p | 480p |
|---|---|:-:|---:|---:|
| **B200** | SGLang | 1 | 2.39 | 2.37 |
| **B300** | SGLang | 1 | 2.14 | 2.16 |
| **H100 80GB HBM3** | SGLang | 1 | 3.75 | 3.75 |
| **H20** | SGLang | 1 | 14.47 | 14.51 |
| **H200 141GB HBM3** | SGLang | 1 | 3.64 | 3.68 |
| **RTX PRO 6000 Blackwell** | SGLang | 1 | 6.39 | 6.39 |

### Inverse Dynamics — Robot

| GPU | Engine | GPUs | 256p | 480p |
|---|---|:-:|---:|---:|
| **B200** | SGLang | 1 |  | 1.02 |
| **B300** | SGLang | 1 |  |  |
| **H100 80GB HBM3** | SGLang | 1 |  | 1.19 |
| **H20** | SGLang | 1 | 3.93 | 3.94 |
| **H200 141GB HBM3** | SGLang | 1 |  | 1.17 |
| **RTX PRO 6000 Blackwell** | SGLang | 1 | 1.90 | 1.90 |
