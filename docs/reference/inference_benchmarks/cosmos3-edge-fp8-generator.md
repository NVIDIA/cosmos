# Cosmos3-Edge-FP8 Generator Inference Benchmarks

[Back to the inference benchmark index](../inference_benchmarks.md)

These tables report **Cosmos3-Edge-FP8** request latency in seconds through **vLLM-Omni**. Lower is better. Each row uses **one GPU or one integrated computing platform**. The existing resolution columns are retained; only 480p was measured in this sweep.

## Measurement settings

- Runtime: `vllm/vllm-omni:v0.31.0rc1`; one configured warmup request, followed by an individual measured execution. Cold-start measurements are excluded.
- Video: 832×480, 121 frames at 24 FPS, 35 denoising steps, guidance 4.0, flow shift 3.0.
- Image: 832×480, 50 steps, guidance 4.0, flow shift 3.0.
- Policy: base Edge model with DROID inputs, 30 steps and guidance 1.0; measurements await validation of their conditioning canvas and action chunk. These rows do not represent the separately trained Policy-DROID model.
- Precision: FP8 with the first and last three denoising steps retained in higher precision; `VLLM_OMNI_USE_QUACK_FP8=0`. Embedded video/image seeds are 42 and policy seed is 0; Thor uses JetPack 7.2.1, MAXN, and `jetson_clocks`.
- Compilation on the T3000 emulation: `MAX_JOBS=2`; warmup completes kernel compilation before measurement.
- Attention: container default on datacenter and embedded systems; `CUDNN_ATTN` on DGX Spark, DGX Station, and RTX PRO 5000. No fastest-backend selection is made for the datacenter or embedded rows.
- Timing: video and policy use server-reported inference time. Image uses client request time. They include different response overheads and must not be compared with generation-only timings on the other Generator pages.

See [measurement settings and reproduction limits](benchmark_methodology.md). Empty cells have no validated measurement; they do not establish lack of support. No v2v, audiovisual, or transfer measurements are available for this FP8 sweep.

## Text-to-Video (t2v)

| GPU or Platform | Engine | GPUs | 256p | 480p |
|---|---|:-:|---:|---:|
| **H100 80GB HBM3** | vLLM-Omni | 1 |  | 9.28 |
| **H200 141GB HBM3** | vLLM-Omni | 1 |  | 9.15 |
| **RTX PRO 6000 Blackwell** | vLLM-Omni | 1 |  | 16.04 |
| **H20** | vLLM-Omni | 1 |  | 36.67 |
| **DGX Station** | vLLM-Omni | 1 |  | 4.19 |
| **DGX Spark** | vLLM-Omni | 1 |  | 75.07 |
| **RTX PRO 5000 Blackwell 72GB** | vLLM-Omni | 1 |  | 27.71 |
| **Jetson AGX Thor T5000** | vLLM-Omni | 1 |  | 62.71 |
| **Jetson AGX Thor T5000 as T3000** | vLLM-Omni | 1 |  | 135.10 |

## Image-to-Video (i2v)

| GPU or Platform | Engine | GPUs | 256p | 480p |
|---|---|:-:|---:|---:|
| **H100 80GB HBM3** | vLLM-Omni | 1 |  | 9.28 |
| **H200 141GB HBM3** | vLLM-Omni | 1 |  | 9.34 |
| **RTX PRO 6000 Blackwell** | vLLM-Omni | 1 |  | 16.09 |
| **H20** | vLLM-Omni | 1 |  | 36.74 |
| **DGX Station** | vLLM-Omni | 1 |  | 4.27 |
| **DGX Spark** | vLLM-Omni | 1 |  | 75.04 |
| **RTX PRO 5000 Blackwell 72GB** | vLLM-Omni | 1 |  | 27.97 |
| **Jetson AGX Thor T5000** | vLLM-Omni | 1 |  | 62.97 |
| **Jetson AGX Thor T5000 as T3000** | vLLM-Omni | 1 |  | 137.80 |

## Text-to-Image (t2i)

| GPU or Platform | Engine | GPUs | 256p | 480p |
|---|---|:-:|---:|---:|
| **H100 80GB HBM3** | vLLM-Omni | 1 |  | 1.41 |
| **H200 141GB HBM3** | vLLM-Omni | 1 |  | 1.42 |
| **RTX PRO 6000 Blackwell** | vLLM-Omni | 1 |  |  |
| **H20** | vLLM-Omni | 1 |  | 1.47 |
| **DGX Station** | vLLM-Omni | 1 |  | 2.58 |
| **DGX Spark** | vLLM-Omni | 1 |  | 2.83 |
| **RTX PRO 5000 Blackwell 72GB** | vLLM-Omni | 1 |  |  |
| **Jetson AGX Thor T5000** | vLLM-Omni | 1 |  | 4.58 |
| **Jetson AGX Thor T5000 as T3000** | vLLM-Omni | 1 |  | 5.89 |

## Policy — Base Edge with DROID Inputs

| GPU or Platform | Engine | GPUs | 256p | 480p |
|---|---|:-:|---:|---:|
| **H100 80GB HBM3** | vLLM-Omni | 1 |  |  |
| **H200 141GB HBM3** | vLLM-Omni | 1 |  |  |
| **RTX PRO 6000 Blackwell** | vLLM-Omni | 1 |  |  |
| **H20** | vLLM-Omni | 1 |  |  |
| **DGX Station** | vLLM-Omni | 1 |  |  |
| **DGX Spark** | vLLM-Omni | 1 |  |  |
| **RTX PRO 5000 Blackwell 72GB** | vLLM-Omni | 1 |  |  |
| **Jetson AGX Thor T5000** | vLLM-Omni | 1 |  |  |
| **Jetson AGX Thor T5000 as T3000** | vLLM-Omni | 1 |  |  |
