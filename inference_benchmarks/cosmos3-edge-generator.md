# Cosmos3-Edge Generator Inference Benchmarks

[Back to the inference benchmark index](../inference_benchmarks.md)

These tables report **Cosmos3-Edge** Generator latency in seconds. Lower latency is better, and empty cells indicate that a run has not been completed rather than an unsupported combination.

Cosmos3-Edge supports **256p and 480p**, so 480p is its highest resolution. Every row reports one GPU configuration, named in the **GPUs** column: one GPU where the model fits on a single device at 480p, and eight GPUs where it does not. Cosmos3-Edge fits on a single GPU or one integrated computing platform at 480p, so all populated rows report one. Within a platform's block every engine reports the same configuration, so the rows can be compared directly.

Video benchmarks generate **121 frames**. vLLM-Omni values report end-to-end latency, PyTorch values report average generation latency, and TensorRT-LLM values report average generation latency from PBR `#308000`.

## Table of Contents

- [Text-to-Image (t2i)](#text-to-image-t2i)
- [Text-to-Video (t2v)](#text-to-video-t2v)
- [Image-to-Video (i2v)](#image-to-video-i2v)
- [Video-to-Video (v2v)](#video-to-video-v2v)

## Text-to-Image (t2i)

| GPU or Platform | Engine | GPUs | 256p | 480p |
|---|---|:-:|---:|---:|
| **B200 SXM 192 GB** | PyTorch | — |  |  |
|  | vLLM-Omni | — |  |  |
|  | TensorRT-LLM | 1 | 0.52 | 0.53 |
| **B300** | PyTorch | — |  |  |
|  | vLLM-Omni | — |  |  |
|  | TensorRT-LLM | 1 | 0.81 | 0.82 |
| **H200 SXM 141 GB** | PyTorch | — |  |  |
|  | vLLM-Omni | — |  |  |
|  | TensorRT-LLM | 1 | 0.58 | 0.58 |
| **H200 NVL** | PyTorch | — |  |  |
|  | vLLM-Omni | — |  |  |
|  | TensorRT-LLM | — |  |  |
| **H100 SXM 80 GB** | PyTorch | — |  |  |
|  | vLLM-Omni | — |  |  |
|  | TensorRT-LLM | 1 | 0.58 | 0.59 |
| **H100 NVL 96 GB** | PyTorch | — |  |  |
|  | vLLM-Omni | — |  |  |
|  | TensorRT-LLM | — |  |  |
| **H20 SXM 96 GB** | PyTorch | — |  |  |
|  | vLLM-Omni | — |  |  |
|  | TensorRT-LLM | 1 | 0.79 | 1.43 |
| **RTX PRO 6000 Blackwell Server Edition** | PyTorch | — |  |  |
|  | vLLM-Omni | — |  |  |
|  | TensorRT-LLM | 1 | 0.57 | 0.66 |
| **DGX Station** | PyTorch | — |  |  |
|  | vLLM-Omni | — |  |  |
|  | TensorRT-LLM | — |  |  |
| **DGX Spark** | PyTorch | — |  |  |
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
| **B200 SXM 192 GB** | PyTorch | — |  |  |
|  | vLLM-Omni | — |  |  |
|  | TensorRT-LLM | 1 | 1.05 | 7.84 |
| **B300** | PyTorch | — |  |  |
|  | vLLM-Omni | — |  |  |
|  | TensorRT-LLM | 1 | 1.00 | 7.22 |
| **H200 SXM 141 GB** | PyTorch | — |  |  |
|  | vLLM-Omni | — |  |  |
|  | TensorRT-LLM | 1 | 1.77 | 14.89 |
| **H200 NVL** | PyTorch | — |  |  |
|  | vLLM-Omni | — |  |  |
|  | TensorRT-LLM | — |  |  |
| **H100 SXM 80 GB** | PyTorch | — |  |  |
|  | vLLM-Omni | — |  |  |
|  | TensorRT-LLM | 1 | 1.83 | 15.09 |
| **H100 NVL 96 GB** | PyTorch | — |  |  |
|  | vLLM-Omni | — |  |  |
|  | TensorRT-LLM | — |  |  |
| **H20 SXM 96 GB** | PyTorch | — |  |  |
|  | vLLM-Omni | — |  |  |
|  | TensorRT-LLM | 1 | 6.91 | 59.19 |
| **RTX PRO 6000 Blackwell Server Edition** | PyTorch | — |  |  |
|  | vLLM-Omni | — |  |  |
|  | TensorRT-LLM | 1 | 2.77 | 26.13 |
| **DGX Station** | PyTorch | — |  |  |
|  | vLLM-Omni | — |  |  |
|  | TensorRT-LLM | — |  |  |
| **DGX Spark** | PyTorch | — |  |  |
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

## Image-to-Video (i2v)

| GPU or Platform | Engine | GPUs | 256p | 480p |
|---|---|:-:|---:|---:|
| **B200 SXM 192 GB** | PyTorch | 1 |  | 7.45 |
|  | vLLM-Omni | 1 |  | 7.09 |
|  | TensorRT-LLM | 1 | 1.23 | 8.55 |
| **B300** | PyTorch | 1 |  | 6.84 |
|  | vLLM-Omni | 1 |  | 8.74 |
|  | TensorRT-LLM | 1 | 1.42 | 8.04 |
| **H200 SXM 141 GB** | PyTorch | 1 |  | 12.31 |
|  | vLLM-Omni | 1 |  | 13.13 |
|  | TensorRT-LLM | 1 | 1.99 | 15.89 |
| **H200 NVL** | PyTorch | 1 |  | 14.07 |
|  | vLLM-Omni | 1 |  | 14.15 |
|  | TensorRT-LLM | — |  |  |
| **H100 SXM 80 GB** | PyTorch | 1 |  | 12.68 |
|  | vLLM-Omni | 1 |  | 13.33 |
|  | TensorRT-LLM | 1 | 2.08 | 15.95 |
| **H100 NVL 96 GB** | PyTorch | 1 |  | 16.42 |
|  | vLLM-Omni | 1 |  | 17.33 |
|  | TensorRT-LLM | — |  |  |
| **H20 SXM 96 GB** | PyTorch | 1 |  | 52.83 |
|  | vLLM-Omni | 1 |  | 53.94 |
|  | TensorRT-LLM | 1 | 7.50 | 62.08 |
| **RTX PRO 6000 Blackwell Server Edition** | PyTorch | 1 |  | 21.92 |
|  | vLLM-Omni | — |  |  |
|  | TensorRT-LLM | 1 | 3.09 | 27.55 |
| **DGX Station** | PyTorch | 1 |  | 6.31 |
|  | vLLM-Omni | 1 |  | 7.13 |
|  | TensorRT-LLM | — |  |  |
| **DGX Spark** | PyTorch | 1 |  | 103.36 |
|  | vLLM-Omni | 1 |  | 89.41 |
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
| **B200 SXM 192 GB** | PyTorch | — |  |  |
|  | vLLM-Omni | — |  |  |
|  | TensorRT-LLM | 1 | 1.15 | 7.94 |
| **B300** | PyTorch | — |  |  |
|  | vLLM-Omni | — |  |  |
|  | TensorRT-LLM | 1 | 1.24 | 7.35 |
| **H200 SXM 141 GB** | PyTorch | — |  |  |
|  | vLLM-Omni | — |  |  |
|  | TensorRT-LLM | 1 | 1.91 | 15.09 |
| **H200 NVL** | PyTorch | — |  |  |
|  | vLLM-Omni | — |  |  |
|  | TensorRT-LLM | — |  |  |
| **H100 SXM 80 GB** | PyTorch | — |  |  |
|  | vLLM-Omni | — |  |  |
|  | TensorRT-LLM | 1 | 1.95 | 15.28 |
| **H100 NVL 96 GB** | PyTorch | — |  |  |
|  | vLLM-Omni | — |  |  |
|  | TensorRT-LLM | — |  |  |
| **H20 SXM 96 GB** | PyTorch | — |  |  |
|  | vLLM-Omni | — |  |  |
|  | TensorRT-LLM | 1 | 7.04 | 60.03 |
| **RTX PRO 6000 Blackwell Server Edition** | PyTorch | — |  |  |
|  | vLLM-Omni | — |  |  |
|  | TensorRT-LLM | 1 | 2.86 | 26.21 |
| **DGX Station** | PyTorch | — |  |  |
|  | vLLM-Omni | — |  |  |
|  | TensorRT-LLM | — |  |  |
| **DGX Spark** | PyTorch | — |  |  |
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
3. TensorRT-LLM values come from PBR `#308000`, which covers the six datacenter GPUs only; the DGX and Jetson platforms were not part of that campaign.
4. vLLM-Omni and PyTorch were measured for i2v only, so their rows are empty in the other three sections.
5. Video measurements generate **121 output frames**.
6. PyTorch values report average generation latency rather than diffusion-only latency.
7. Diffusers and NIM have no Cosmos3-Edge coverage yet, so they are not listed.</sub>
