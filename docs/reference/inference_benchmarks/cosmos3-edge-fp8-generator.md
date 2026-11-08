# Cosmos3-Edge-FP8 Generator Inference Benchmarks

[Back to the inference benchmark index](../inference_benchmarks.md)

These tables report **Cosmos3-Edge-FP8** request latency in seconds through **vLLM-Omni**. Lower is better. Each row uses **one GPU or one integrated computing platform**. The existing resolution columns are retained; only 480p was measured in this sweep.

## Known anomalies and measurement gaps

- **FP8 is not consistently faster for images.** The matched RTX PRO 6000 campaign reports FP8 t2i about 4× slower than BF16 across multiple hosts. On RTX PRO 5000 the enterprise comparison is 5.79 s FP8 versus 1.69 s BF16 (3.43×). These comparisons use the matched controls, not the historical BF16 table.
- `FLASHINFER_ATTN` failed in this container, and `FLASH_ATTN` failed on Blackwell. The published Blackwell measurements therefore use cuDNN attention. The first switch into FP8 adds about 6 s once; one identical warmup request excludes that cold cost from these measurements.
- On the T3000 emulation, first-use FP8 kernel compilation exhausted memory with the default parallel compilation. The measured setup caps compilation with `MAX_JOBS=2`. Generator FP8 did not load on Orin in this software stack; the Reasoner's Orin weight-only path does not establish Generator compatibility.
- **Policy payloads are not fully aligned across platforms.** Embedded policy inputs use a 736×544 conditioning canvas; chunk descriptions in the datacenter and embedded studies are inconsistent. Keep these as separate campaign timings pending confirmation of the actual action tensor, chunk, and rollout settings. Do not infer policy GPU rankings from the common resolution-tier label.

See the [anomaly review](benchmark_anomaly_review.md) before making precision or cross-platform comparisons.

## Measurement settings

- Runtime: `vllm/vllm-omni:v0.31.0rc1`; one configured warmup request, followed by an individual measured execution. Cold-start measurements are excluded.
- Video: 832×480, 121 frames at 24 FPS, 35 denoising steps, guidance 4.0, flow shift 3.0.
- Image: 832×480, 50 steps, guidance 4.0, flow shift 3.0.
- Policy: base Edge model with DROID inputs, nominal chunk size 32, 30 steps, guidance 1.0; actual cross-campaign chunk/canvas parity is unresolved as noted above. These rows do not represent the separately trained Policy-DROID model.
- Precision: FP8 with the first and last three denoising steps retained in higher precision; `VLLM_OMNI_USE_QUACK_FP8=0`. Embedded video/image seeds are 42 and policy seed is 0; Thor uses JetPack 7.2.1, MAXN, and `jetson_clocks`.
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
| **RTX PRO 6000 Blackwell** | vLLM-Omni | 1 |  | 4.10 |
| **H20** | vLLM-Omni | 1 |  | 1.47 |
| **DGX Station** | vLLM-Omni | 1 |  | 2.58 |
| **DGX Spark** | vLLM-Omni | 1 |  | 2.83 |
| **RTX PRO 5000 Blackwell 72GB** | vLLM-Omni | 1 |  | 5.79 |
| **Jetson AGX Thor T5000** | vLLM-Omni | 1 |  | 4.58 |
| **Jetson AGX Thor T5000 as T3000** | vLLM-Omni | 1 |  | 5.89 |

## Policy — Base Edge with DROID Inputs

| GPU or Platform | Engine | GPUs | 256p | 480p |
|---|---|:-:|---:|---:|
| **H100 80GB HBM3** | vLLM-Omni | 1 |  | 1.56 |
| **H200 141GB HBM3** | vLLM-Omni | 1 |  | 1.52 |
| **RTX PRO 6000 Blackwell** | vLLM-Omni | 1 |  | 2.97 |
| **H20** | vLLM-Omni | 1 |  | 4.77 |
| **DGX Station** | vLLM-Omni | 1 |  | 1.22 |
| **DGX Spark** | vLLM-Omni | 1 |  | 9.64 |
| **RTX PRO 5000 Blackwell 72GB** | vLLM-Omni | 1 |  | 4.07 |
| **Jetson AGX Thor T5000** | vLLM-Omni | 1 |  | 8.44 |
| **Jetson AGX Thor T5000 as T3000** | vLLM-Omni | 1 |  | 17.63 |
