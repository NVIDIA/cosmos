# Cosmos3-Nano-FP8 Generator Inference Benchmarks

[Back to the inference benchmark index](../inference_benchmarks.md)

See [measurement settings and reproduction limits](benchmark_methodology.md) for runtime-specific timing, precision, and workload details.

These tables report **Cosmos3-Nano-FP8** Generator latency in seconds, for the FP8 quantized Cosmos3-Nano checkpoint. Lower is better. The BF16 numbers for the same model are on the [Cosmos3-Nano page](cosmos3-nano-generator.md).

Every row reports one GPU configuration, named in the **GPUs** column, and all three resolutions in that row come from it. The configuration is one GPU when the model fits on a single device at its highest resolution, 720p, and eight GPUs when it does not. The **GPUs** column is authoritative: engines can use different counts when their memory requirements differ. Most rows report one GPU; the **GPUs** column identifies any memory-driven exceptions.

Empty cells have no publishable measurement at the stated configuration. They can reflect an unmeasured workload, an unsuccessful run, or a measurement awaiting validation; they do not establish lack of support.

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
  - [Forward Dynamics — Autonomous Vehicle (AV)](#forward-dynamics--autonomous-vehicle-av)
  - [Forward Dynamics — Camera](#forward-dynamics--camera)
  - [Forward Dynamics — Robot](#forward-dynamics--robot)
  - [Inverse Dynamics — Autonomous Vehicle (AV)](#inverse-dynamics--autonomous-vehicle-av)
  - [Inverse Dynamics — Robot](#inverse-dynamics--robot)
  - [Inverse Dynamics — Camera](#inverse-dynamics--camera)
  - [Policy — Autonomous Vehicle (AV)](#policy--autonomous-vehicle-av)
  - [Policy — Robot](#policy--robot)

## Benchmark methodology

PyTorch values report average generation latency. The PyTorch FP8 recipe differs from the mixed-step recipe used by other engines, so those values are not directly comparable; primary rows retain the **PyTorch\*** marker. Diffusers, vLLM-Omni, and SGLang include measured 480p and 720p vision, audiovisual, and transfer workloads, including multi-GPU configurations where available. Action values are included only when the embodiment is identified. NIM rows use FP8 latency profiles. The base-model page may select a different GPU count; compare only matching configurations. GB300 and enterprise platforms appear where measured. No TensorRT-LLM FP8 measurements are included. Values are rounded to two decimal places. See the linked measurement settings for attention backends, parallel splits, and reproduction limits.

These measurements establish the reported timing matrix but do not expose every prompt and action payload in this repository. The linked public recipes explain modality behavior and provide representative payloads; their example-specific frame counts and action chunk sizes should not be treated as the exact internal benchmark inputs.

## Workload definitions

| Workload | Input | Output |
|---|---|---|
| Text-to-image (`t2i`) | Text prompt | Generated image |
| Text-to-video (`t2v`) | Text prompt | Generated video |
| Image-to-video (`i2v`) | Text prompt and source image | Generated video |
| Video-to-video (`v2v`) | Text prompt and source video | Generated video |
| Text-to-audio-and-video (`t2av`) | Text prompt | Synchronized video and sound |
| Video-to-audio-and-video (`v2av`) | Text prompt and source video | Generated video with synchronized sound |
| Image-to-audio-and-video (`i2av`) | Text prompt and source image | Generated video with synchronized sound |
| Transfer video-to-video (`transfer`) | Text prompt, source video, and one control hint | Generated video steered by that hint |
| Forward dynamics | Initial visual observation and an action trajectory | Future-observation rollout video |
| Inverse dynamics | Observed video | Recovered action trajectory; some serving integrations also return video |
| Policy | Initial visual observation, instruction, and optional state | Predicted action trajectory and, for general Generator paths, a rollout video |

The benchmark uses `t2av`, `v2av`, and `i2av`; some public recipes call the same sound-producing modes `t2vs`, `v2vs`, and `i2vs`. See the [audiovisual cookbook](https://nvda.ws/4zd0fw5) for generation inputs and the [action cookbook](https://nvda.ws/4dUeBsJ) for action representations and output contracts.

## Primary vision generation

### Text-to-Video (t2v)

| GPU | Engine | GPUs | 256p | 480p | 720p |
|---|---|:-:|---:|---:|---:|
| **RTX PRO 6000 Blackwell** | PyTorch\* | 1 | 4.71 | 47.34 | 173.02 |
|  | Diffusers | 1 |  | 112.53 | 385.89 |
|  | NIM | 1 | 5.06 | 46.32 | 170.57 |
|  | vLLM-Omni | 1 |  | 80.74 | 311.43 |
|  | SGLang | 1 |  | 87.73 | 333.63 |
| **H20** | PyTorch\* | 1 | 11.82 | 112.34 | 427.62 |
|  | Diffusers | 1 |  | 237.56 | 866.18 |
|  | NIM | 1 | 11.78 | 86.02 | 282.28 |
|  | vLLM-Omni | 1 |  | 207.82 | 790.11 |
|  | SGLang | 1 |  | 255.24 | 921.37 |
| **H100 80GB HBM3** | PyTorch\* | 1 | 3.12 | 26.01 | 94.01 |
|  | Diffusers | 1 |  | 69.55 | 237.29 |
|  | NIM | 1 | 3.34 | 24.71 | 86.16 |
|  | vLLM-Omni | 1 |  | 55.78 | 189.83 |
|  | SGLang | 1 |  | 48.84 | 181.32 |
| **H200 141GB HBM3** | PyTorch\* | 1 | 3.09 | 26.59 | 94.57 |
|  | Diffusers | 1 |  | 68.32 | 237.18 |
|  | NIM | 1 | 3.15 | 24.62 | 85.55 |
|  | vLLM-Omni | 1 |  | 55.89 | 188.94 |
|  | SGLang | 1 |  | 48.40 | 181.06 |
| **B200** | PyTorch\* | 1 | 1.97 | 14.85 | 51.85 |
|  | Diffusers | 1 |  | 39.70 | 124.10 |
|  | NIM | 1 | 2.00 | 13.71 | 49.86 |
|  | vLLM-Omni | 1 |  | 22.94 | 85.58 |
|  | SGLang | 1 |  | 29.34 | 113.82 |
| **B300** | PyTorch\* | 1 | 1.99 | 13.68 | 46.41 |
|  | Diffusers | 1 |  | 37.61 | 115.14 |
|  | NIM | 1 | 1.98 | 13.10 | 45.76 |
|  | vLLM-Omni | 1 |  | 21.75 | 79.52 |
|  | SGLang | 1 |  | 23.63 | 84.74 |
| **GB300** | PyTorch\* | 1 |  |  |  |
|  | Diffusers | 1 |  | 34.53 | 97.30 |
|  | NIM | 1 | 2.52 | 11.09 | 39.07 |
|  | vLLM-Omni | 1 |  | 18.93 | 67.68 |
|  | SGLang | 1 |  | 22.78 | 77.94 |
| **DGX Spark** | NIM | 1 | 36.78 | 358.61 | 1360.13 |
|  | Diffusers | 1 |  | 522.73 | 1730.23 |
| **RTX PRO 5000 Blackwell 72GB** | NIM | 1 | 13.15 | 143.06 | 557.53 |
|  | Diffusers | 1 |  | 198.30 | 692.91 |
| **DGX Station** | Diffusers | 1 |  | 33.86 | 102.37 |

### Image-to-Video (i2v)

| GPU | Engine | GPUs | 256p | 480p | 720p |
|---|---|:-:|---:|---:|---:|
| **RTX PRO 6000 Blackwell** | PyTorch\* | 1 | 4.50 | 45.96 | 169.15 |
|  | Diffusers | 1 |  | 114.53 | 390.81 |
|  | NIM | 1 | 4.90 | 46.43 | 171.34 |
|  | vLLM-Omni | 1 |  | 80.83 | 311.89 |
|  | SGLang | 1 |  | 88.36 | 338.59 |
| **H20** | PyTorch\* | 1 | 11.19 | 110.19 | 419.99 |
|  | Diffusers | 1 |  | 240.91 | 875.66 |
|  | NIM | 1 | 11.67 | 86.66 | 282.41 |
|  | vLLM-Omni | 1 |  | 207.92 | 789.78 |
|  | SGLang | 1 |  | 256.22 | 924.15 |
| **H100 80GB HBM3** | PyTorch\* | 1 | 2.96 | 25.04 | 91.65 |
|  | Diffusers | 1 |  | 71.06 | 241.53 |
|  | NIM | 1 | 3.39 | 24.65 | 86.60 |
|  | vLLM-Omni | 1 |  | 58.99 | 186.41 |
|  | SGLang | 1 |  | 48.87 | 181.90 |
| **H200 141GB HBM3** | PyTorch\* | 1 | 2.94 | 25.56 | 92.44 |
|  | Diffusers | 1 |  | 69.69 | 237.95 |
|  | NIM | 1 | 3.16 | 24.67 | 87.67 |
|  | vLLM-Omni | 1 |  | 56.66 | 189.77 |
|  | SGLang | 1 |  | 48.34 | 181.79 |
| **B200** | PyTorch\* | 1 | 1.87 | 14.24 | 50.24 |
|  | Diffusers | 1 |  | 40.69 | 126.07 |
|  | NIM | 1 | 1.95 | 13.76 | 49.16 |
|  | vLLM-Omni | 1 |  | 23.30 | 84.36 |
|  | SGLang | 1 |  | 30.00 | 116.19 |
| **B300** | PyTorch\* | 1 | 1.78 | 13.08 | 44.94 |
|  | Diffusers | 1 |  | 38.35 | 117.18 |
|  | NIM | 1 | 2.07 | 13.63 | 46.81 |
|  | vLLM-Omni | 1 |  | 21.75 | 79.36 |
|  | SGLang | 1 |  | 23.97 | 85.10 |
| **GB300** | PyTorch\* | 1 |  |  |  |
|  | Diffusers | 1 |  | 34.96 | 99.37 |
|  | NIM | 1 | 1.96 | 11.25 | 39.32 |
|  | vLLM-Omni | 1 |  | 18.80 | 67.61 |
|  | SGLang | 1 |  | 22.92 | 77.57 |
| **DGX Spark** | NIM | 1 | 36.57 | 354.86 | 1350.29 |
|  | Diffusers | 1 |  | 533.41 | 1739.15 |
| **RTX PRO 5000 Blackwell 72GB** | NIM | 1 | 13.24 | 143.25 | 557.87 |
|  | Diffusers | 1 |  | 201.55 | 700.33 |
| **DGX Station** | Diffusers | 1 |  | 34.50 | 104.17 |

### Text-to-Image (t2i)

| GPU | Engine | GPUs | 256p | 480p | 720p |
|---|---|:-:|---:|---:|---:|
| **RTX PRO 6000 Blackwell** | PyTorch\* | 1 | 0.97 | 1.02 | 1.50 |
|  | Diffusers | 1 |  | 7.29 | 7.83 |
|  | NIM | 1 | 0.72 | 1.00 | 1.48 |
|  | vLLM-Omni | 1 |  | 6.61 | 6.59 |
|  | SGLang | 1 |  | 1.97 | 2.82 |
| **H20** | PyTorch\* | 1 | 1.27 | 1.74 | 3.25 |
|  | Diffusers | 1 |  | 9.94 | 13.56 |
|  | NIM | 1 | 1.09 | 1.61 | 2.80 |
|  | vLLM-Omni | 1 |  | 15.13 | 17.09 |
|  | SGLang | 1 |  | 3.15 | 7.27 |
| **H100 80GB HBM3** | PyTorch\* | 1 | 1.16 | 1.14 | 1.21 |
|  | Diffusers | 1 |  | 7.50 | 7.68 |
|  | NIM | 1 | 1.02 | 1.04 | 1.15 |
|  | vLLM-Omni | 1 |  | 13.11 | 13.15 |
|  | SGLang | 1 |  | 1.61 | 1.73 |
| **H200 141GB HBM3** | PyTorch\* | 1 | 1.12 | 1.14 | 1.21 |
|  | Diffusers | 1 |  | 7.44 | 7.74 |
|  | NIM | 1 | 1.02 | 1.05 | 1.08 |
|  | vLLM-Omni | 1 |  | 13.40 | 13.22 |
|  | SGLang | 1 |  | 1.57 | 1.67 |
| **B200** | PyTorch\* | 1 | 0.96 | 0.92 | 0.97 |
|  | Diffusers | 1 |  | 6.82 | 6.94 |
|  | NIM | 1 | 0.58 | 0.63 | 0.67 |
|  | vLLM-Omni | 1 |  | 5.38 | 5.42 |
|  | SGLang | 1 |  | 1.62 | 1.70 |
| **B300** | PyTorch\* | 1 | 0.95 | 0.95 | 1.51 |
|  | Diffusers | 1 |  | 10.28 | 6.71 |
|  | NIM | 1 | 0.72 | 0.84 | 0.87 |
|  | vLLM-Omni | 1 |  | 5.11 | 7.16 |
|  | SGLang | 1 |  | 2.41 | 1.51 |
| **GB300** | PyTorch\* | 1 |  |  |  |
|  | Diffusers | 1 |  | 11.79 | 13.19 |
|  | NIM | 1 | 1.70 | 1.69 | 1.71 |
|  | vLLM-Omni | 1 |  | 12.27 | 6.89 |
|  | SGLang | 1 |  | 1.67 | 1.74 |
| **DGX Spark** | NIM | 1 | 2.34 | 3.87 | 7.47 |
|  | Diffusers | 1 |  | 32.73 | 40.74 |
| **RTX PRO 5000 Blackwell 72GB** | NIM | 1 | 1.63 | 2.08 | 3.45 |
|  | Diffusers | 1 |  | 9.79 | 11.46 |
| **DGX Station** | Diffusers | 1 |  | 9.68 | 9.66 |

<sub>Notes:
1. Measurements come from separate campaigns; identical prompt, seed, step budget, and timing boundaries are not established across engines. See the measurement settings before comparing rows.
2. Multi-GPU layouts use the recorded tensor, CFG, context, Ring, or Ulysses splits; GPU count alone does not specify the layout.
3. Values are average generation latency in seconds; lower is better.
4. The **GPUs** column gives the number of GPUs behind every value in that row; check it before comparing engines.
5. Diffusers FP8 was measured at 480p and 720p; the GPUs column distinguishes single- and multi-GPU results.
6. FP8 is faster than BF16 for the PyTorch path throughout. For the Diffusers path it only pays off on long video workloads: short ones are slower in FP8, `t2i` consistently so by roughly a factor of two, which suggests a fixed quantization overhead that the short runs cannot amortize. Compare against the BF16 page before choosing a precision for a short workload.
7. \* PyTorch values were measured with a different FP8 recipe than the other engines on this page, which run the first and last three denoising steps in W8A16 and the middle steps in W8A8. PyTorch results are therefore optimistic and cannot be compared directly with the other values.
8. NIM numbers use NIM 2.0.0 on DGX Spark and RTX PRO 5000, and NIM 2.1.0 (`nvcr.io/nim/nvidia/cosmos3:2.1.0`) on the other displayed platforms and report average generation time, excluding request processing and output encoding; step budgets and other workload differences still limit comparisons with other backends.</sub>

## Additional audiovisual generation

### Video-to-Video (v2v)

| GPU | Engine | GPUs | 256p | 480p | 720p |
|---|---|:-:|---:|---:|---:|
| **RTX PRO 6000 Blackwell** | PyTorch | 1 | 4.40 | 47.39 | 175.87 |
|  | Diffusers | 1 |  | 115.33 | 391.90 |
|  | vLLM-Omni | 1 |  | 18.09 | 55.69 |
|  | SGLang | 1 |  | 93.60 | 345.71 |
| **H20** | PyTorch | 1 | 10.99 | 112.59 | 437.65 |
|  | Diffusers | 1 |  | 243.99 | 877.03 |
|  | vLLM-Omni | 1 |  | 55.06 | 145.66 |
|  | SGLang | 1 |  | 264.08 | 936.39 |
| **H100 80GB HBM3** | PyTorch | 1 | 2.93 | 25.89 | 95.20 |
|  | Diffusers | 1 |  | 72.01 | 241.38 |
|  | vLLM-Omni | 1 |  | 20.59 | 41.83 |
|  | SGLang | 1 |  | 54.28 | 187.99 |
| **H200 141GB HBM3** | PyTorch | 1 | 2.90 | 26.51 | 96.80 |
|  | Diffusers | 1 |  | 70.17 | 239.24 |
|  | vLLM-Omni | 1 |  | 20.11 | 42.44 |
|  | SGLang | 1 |  | 53.71 | 187.58 |
| **B200** | PyTorch | 1 | 1.85 | 14.69 | 52.19 |
|  | Diffusers | 1 |  | 41.72 | 127.42 |
|  | vLLM-Omni | 1 |  | 6.25 | 16.59 |
|  | SGLang | 1 |  | 34.23 | 118.98 |
| **B300** | PyTorch | 1 | 1.92 | 13.61 | 47.32 |
|  | Diffusers | 1 |  | 39.99 | 118.73 |
|  | vLLM-Omni | 1 |  | 6.51 | 15.44 |
|  | SGLang | 1 |  | 30.55 | 91.23 |
| **GB300** | PyTorch | 1 |  |  |  |
|  | Diffusers | 1 |  | 35.97 | 100.27 |
|  | vLLM-Omni | 1 |  | 6.83 | 13.60 |
|  | SGLang | 1 |  | 26.44 | 81.94 |
| **DGX Spark** | NIM | 1 | 37.20 | 360.62 | 1371.19 |
|  | Diffusers | 1 |  | 530.90 | 1751.65 |
|  | PyTorch | 1 | 38.40 | 375.00 | 1393.20 |
| **RTX PRO 5000 Blackwell 72GB** | NIM | 1 | 13.58 | 143.11 | 557.23 |
|  | Diffusers | 1 |  | 202.54 | 700.88 |
| **DGX Station** | Diffusers | 1 |  | 35.20 | 104.94 |
|  | PyTorch | 1 | 2.50 | 19.50 | 70.30 |

### Text-to-Audio-and-Video (t2av)

| GPU | Engine | GPUs | 256p | 480p | 720p |
|---|---|:-:|---:|---:|---:|
| **RTX PRO 6000 Blackwell** | PyTorch | 1 | 7.67 | 50.58 | 178.94 |
|  | Diffusers | 1 |  | 113.70 | 391.60 |
|  | vLLM-Omni | 1 |  | 82.19 | 316.45 |
|  | SGLang | 1 |  | 89.52 | 339.67 |
| **H20** | PyTorch | 1 | 17.97 | 120.72 | 442.78 |
|  | Diffusers | 1 |  | 240.59 | 873.73 |
|  | vLLM-Omni | 1 |  | 210.85 | 800.56 |
|  | SGLang | 1 |  | 260.35 | 934.68 |
| **H100 80GB HBM3** | PyTorch | 1 | 4.88 | 28.50 | 97.19 |
|  | Diffusers | 1 |  | 70.46 | 239.14 |
|  | vLLM-Omni | 1 |  | 59.05 | 184.65 |
|  | SGLang | 1 |  | 48.99 | 181.17 |
| **H200 141GB HBM3** | PyTorch | 1 | 4.79 | 28.21 | 99.38 |
|  | Diffusers | 1 |  | 68.99 | 237.11 |
|  | vLLM-Omni | 1 |  | 57.31 | 190.89 |
|  | SGLang | 1 |  | 48.72 | 182.85 |
| **B200** | PyTorch | 1 | 3.10 | 16.48 | 54.04 |
|  | Diffusers | 1 |  | 40.71 | 124.89 |
|  | vLLM-Omni | 1 |  | 23.43 | 85.20 |
|  | SGLang | 1 |  | 30.12 | 114.57 |
| **B300** | PyTorch | 1 | 3.12 | 15.17 | 49.50 |
|  | Diffusers | 1 |  | 38.13 | 116.44 |
|  | vLLM-Omni | 1 |  | 22.03 | 80.44 |
|  | SGLang | 1 |  | 24.22 | 85.80 |
| **GB300** | PyTorch | 1 |  |  |  |
|  | Diffusers | 1 |  | 34.89 | 100.22 |
|  | vLLM-Omni | 1 |  | 19.00 | 70.86 |
|  | SGLang | 1 |  | 23.44 | 78.95 |
| **DGX Spark** | Diffusers | 1 |  | 531.97 | 1742.20 |
| **DGX Station** | Diffusers | 1 |  | 34.70 | 103.12 |
| **RTX PRO 5000 Blackwell 72GB** | Diffusers | 1 |  | 205.63 | 697.97 |

### Video-to-Audio-and-Video (v2av)

| GPU | Engine | GPUs | 256p | 480p | 720p |
|---|---|:-:|---:|---:|---:|
| **RTX PRO 6000 Blackwell** | PyTorch | — |  |  |  |
|  | Diffusers | 1 |  | 116.76 | 397.24 |
| **H20** | PyTorch | — |  |  |  |
|  | Diffusers | 1 |  | 245.82 | 886.39 |
| **H100 80GB HBM3** | PyTorch | 1 |  |  |  |
|  | Diffusers | 1 |  | 73.07 | 243.26 |
| **H200 141GB HBM3** | PyTorch | 1 |  |  |  |
|  | Diffusers | 1 |  | 71.80 | 241.17 |
| **B200** | PyTorch | 1 |  |  |  |
|  | Diffusers | 1 |  | 42.28 | 126.29 |
| **B300** | PyTorch | 1 |  |  |  |
|  | Diffusers | 1 |  | 39.94 | 120.27 |
| **GB300** | PyTorch | 1 |  |  |  |
|  | Diffusers | 1 |  | 36.41 | 101.35 |
| **DGX Spark** | Diffusers | 1 |  | 538.48 | 1750.77 |
| **DGX Station** | Diffusers | 1 |  | 35.95 | 105.87 |
| **RTX PRO 5000 Blackwell 72GB** | Diffusers | 1 |  | 209.92 | 706.11 |

### Image-to-Audio-and-Video (i2av)

| GPU | Engine | GPUs | 256p | 480p | 720p |
|---|---|:-:|---:|---:|---:|
| **RTX PRO 6000 Blackwell** | PyTorch | 1 | 7.30 | 48.80 | 174.71 |
|  | Diffusers | 1 |  | 115.84 | 395.85 |
|  | vLLM-Omni | 1 |  | 82.12 | 316.20 |
|  | SGLang | 1 |  | 89.60 | 340.06 |
| **H20** | PyTorch | 1 | 17.08 | 115.28 | 428.45 |
|  | Diffusers | 1 |  | 244.80 | 884.24 |
|  | vLLM-Omni | 1 |  | 210.65 | 800.51 |
|  | SGLang | 1 |  | 261.07 | 934.87 |
| **H100 80GB HBM3** | PyTorch | 1 | 4.63 | 27.33 | 94.44 |
|  | Diffusers | 1 |  | 72.07 | 243.07 |
|  | vLLM-Omni | 1 |  | 57.66 | 191.12 |
|  | SGLang | 1 |  | 49.68 | 184.14 |
| **H200 141GB HBM3** | PyTorch | 1 | 4.54 | 27.44 | 97.66 |
|  | Diffusers | 1 |  | 70.18 | 241.70 |
|  | vLLM-Omni | 1 |  | 56.88 | 192.38 |
|  | SGLang | 1 |  | 48.88 | 184.08 |
| **B200** | PyTorch | 1 | 2.96 | 15.67 | 52.20 |
|  | Diffusers | 1 |  | 41.66 | 126.91 |
|  | vLLM-Omni | 1 |  | 23.27 | 84.73 |
|  | SGLang | 1 |  | 30.60 | 114.85 |
| **B300** | PyTorch | 1 | 2.81 | 14.60 | 47.27 |
|  | Diffusers | 1 |  | 39.34 | 118.23 |
|  | vLLM-Omni | 1 |  | 22.21 | 80.03 |
|  | SGLang | 1 |  | 24.36 | 86.01 |
| **GB300** | PyTorch | 1 |  |  |  |
|  | Diffusers | 1 |  | 36.51 | 101.85 |
|  | vLLM-Omni | 1 |  | 18.93 | 68.67 |
|  | SGLang | 1 |  | 23.21 | 79.21 |
| **DGX Spark** | Diffusers | 1 |  | 546.37 | 1769.13 |
| **DGX Station** | Diffusers | 1 |  | 36.14 | 105.17 |
| **RTX PRO 5000 Blackwell 72GB** | Diffusers | 1 |  | 208.98 | 705.77 |

## Transfer generation

Transfer conditions a video-to-video generation on a structural control hint
extracted from the source video. Each control is reported separately because
the hint changes how much of the frame the model must synthesise.

| GPU | Transfer control | Engine | GPUs | 256p | 480p | 720p |
|---|---|---|:-:|---:|---:|---:|
| **RTX PRO 6000 Blackwell** | blur | PyTorch | 1 | 15.92 | 99.23 | 369.40 |
|  |  | Diffusers | 1 |  | 278.59 | 973.75 |
|  |  | vLLM-Omni | 1 |  | 204.92 | 794.17 |
|  |  | SGLang | 1 |  | 209.31 | 834.12 |
|  | depth | PyTorch | 1 | 15.34 | 102.88 | 379.90 |
|  |  | Diffusers | 1 |  | 289.70 | 989.44 |
|  |  | vLLM-Omni | 1 |  | 208.37 | 816.16 |
|  |  | SGLang | 1 |  | 213.53 | 843.51 |
|  | edge | PyTorch | 1 | 13.27 | 99.61 | 374.57 |
|  |  | Diffusers | 1 |  | 279.34 | 975.04 |
|  |  | vLLM-Omni | 1 |  | 204.28 | 804.38 |
|  |  | SGLang | 1 |  | 209.77 | 834.57 |
|  | multi_control | PyTorch | — |  |  |  |
|  |  | Diffusers | — |  |  |  |
|  |  | vLLM-Omni | 1 |  | 608.48 | 2644.63 |
|  |  | SGLang | 1 |  | 365.88 | 1588.91 |
|  | seg | PyTorch | 1 | 14.54 | 104.96 | 378.47 |
|  |  | Diffusers | 1 |  | 281.97 | 977.23 |
|  |  | vLLM-Omni | 1 |  | 205.45 | 809.54 |
|  |  | SGLang | 1 |  | 210.79 | 837.61 |
|  | wsm | PyTorch | 1 | 12.81 | 78.87 | 280.91 |
|  |  | Diffusers | 1 |  | 134.04 | 430.19 |
|  |  | vLLM-Omni | 1 |  | 93.30 | 345.93 |
|  |  | SGLang | 1 |  | 93.63 | 353.94 |
| **H20** | blur | PyTorch | 1 | 36.48 | 238.41 | 918.21 |
|  |  | Diffusers | 1 |  | 605.34 | 2225.57 |
|  |  | vLLM-Omni | 1 |  | 515.01 | 2003.77 |
|  |  | SGLang | 1 |  | 614.77 | 2315.58 |
|  | depth | PyTorch | 1 | 34.13 | 243.68 | 951.21 |
|  |  | Diffusers | 1 |  | 629.74 | 2262.63 |
|  |  | vLLM-Omni | 1 |  | 523.71 | 2059.26 |
|  |  | SGLang | 1 |  | 625.37 | 2335.41 |
|  | edge | PyTorch | 1 | 29.64 | 238.38 | 947.97 |
|  |  | Diffusers | 1 |  | 606.33 | 2229.61 |
|  |  | vLLM-Omni | 1 |  | 515.47 | 2037.47 |
|  |  | SGLang | 1 |  | 614.98 | 2312.76 |
|  | multi_control | PyTorch | 1 | 36.57 | 385.10 | 1678.80 |
|  |  | Diffusers | 1 |  |  |  |
|  |  | vLLM-Omni | 1 |  | 1514.45 | 6693.33 |
|  |  | SGLang | 1 |  | 1043.10 |  |
|  | seg | PyTorch | 1 | 31.20 | 239.56 | 992.60 |
|  |  | Diffusers | 1 |  | 611.72 | 2237.51 |
|  |  | vLLM-Omni | 1 |  | 517.07 | 2042.07 |
|  |  | SGLang | 1 |  | 617.89 | 2319.15 |
|  | wsm | PyTorch | 1 | 28.15 | 186.34 | 707.32 |
|  |  | Diffusers | 1 |  | 287.51 | 969.79 |
|  |  | vLLM-Omni | 1 |  | 241.43 | 874.83 |
|  |  | SGLang | 1 |  | 280.08 | 995.06 |
| **H100 80GB HBM3** | blur | PyTorch | 1 | 10.37 | 55.94 | 198.76 |
|  |  | Diffusers | 1 |  | 171.90 | 600.30 |
|  |  | vLLM-Omni | 1 |  | 139.38 | 468.63 |
|  |  | SGLang | 1 |  | 116.74 | 451.48 |
|  | depth | PyTorch | 1 | 9.83 | 57.44 | 207.16 |
|  |  | Diffusers | 1 |  | 178.89 | 606.60 |
|  |  | vLLM-Omni | 1 |  | 124.68 | 480.71 |
|  |  | SGLang | 1 |  | 119.08 | 458.12 |
|  | edge | PyTorch | 1 | 8.49 | 55.27 | 204.24 |
|  |  | Diffusers | 1 |  | 172.75 | 597.01 |
|  |  | vLLM-Omni | 1 |  | 123.11 | 473.86 |
|  |  | SGLang | 1 |  | 116.13 | 445.96 |
|  | multi_control | PyTorch | 1 | 10.51 | 91.48 | 367.08 |
|  |  | Diffusers | 1 |  |  |  |
|  |  | vLLM-Omni | 1 |  | 367.61 | 1493.09 |
|  |  | SGLang | 1 |  | 201.57 | 845.27 |
|  | seg | PyTorch | 1 | 10.03 | 56.62 | 214.56 |
|  |  | Diffusers | 1 |  | 174.09 | 601.13 |
|  |  | vLLM-Omni | 1 |  | 135.17 | 482.92 |
|  |  | SGLang | 1 |  | 118.16 | 453.28 |
|  | wsm | PyTorch | 1 | 8.33 | 44.52 | 153.26 |
|  |  | Diffusers | 1 |  | 84.48 | 264.67 |
|  |  | vLLM-Omni | 1 |  | 70.87 | 213.20 |
|  |  | SGLang | 1 |  | 53.24 | 193.99 |
| **H200 141GB HBM3** | blur | PyTorch | 1 | 10.32 | 55.96 | 205.31 |
|  |  | Diffusers | 1 |  | 169.37 | 598.78 |
|  |  | vLLM-Omni | 1 |  | 138.38 | 474.07 |
|  |  | SGLang | 1 |  | 115.85 | 445.77 |
|  | depth | PyTorch | 1 | 9.70 | 58.92 | 211.20 |
|  |  | Diffusers | 1 |  | 177.22 | 603.37 |
|  |  | vLLM-Omni | 1 |  | 125.30 | 476.57 |
|  |  | SGLang | 1 |  | 116.76 | 452.18 |
|  | edge | PyTorch | 1 | 8.60 | 56.78 | 208.28 |
|  |  | Diffusers | 1 |  | 169.21 | 594.04 |
|  |  | vLLM-Omni | 1 |  | 123.38 | 483.20 |
|  |  | SGLang | 1 |  | 114.87 | 448.04 |
|  | multi_control | PyTorch | 1 | 10.46 | 92.81 | 382.20 |
|  |  | Diffusers | 1 |  |  |  |
|  |  | vLLM-Omni | 1 |  | 361.34 | 1531.91 |
|  |  | SGLang | 1 |  | 199.37 | 850.73 |
|  | seg | PyTorch | 1 | 9.68 | 58.18 | 209.54 |
|  |  | Diffusers | 1 |  | 170.62 | 601.64 |
|  |  | vLLM-Omni | 1 |  | 136.76 | 485.36 |
|  |  | SGLang | 1 |  | 117.61 | 447.82 |
|  | wsm | PyTorch | 1 | 8.39 | 45.06 | 154.98 |
|  |  | Diffusers | 1 |  | 82.92 | 263.74 |
|  |  | vLLM-Omni | 1 |  | 71.05 | 214.79 |
|  |  | SGLang | 1 |  | 52.21 | 192.67 |
| **B200** | blur | PyTorch | 1 | 7.02 | 32.23 | 110.23 |
|  |  | Diffusers | 1 |  | 94.88 | 298.51 |
|  |  | vLLM-Omni | 1 |  | 58.32 | 216.19 |
|  |  | SGLang | 1 |  | 72.25 | 286.77 |
|  | depth | PyTorch | 1 | 6.68 | 33.48 | 113.49 |
|  |  | Diffusers | 1 |  | 98.03 | 307.64 |
|  |  | vLLM-Omni | 1 |  | 59.42 | 223.11 |
|  |  | SGLang | 1 |  | 75.16 | 289.87 |
|  | edge | PyTorch | 1 | 5.79 | 31.53 | 111.67 |
|  |  | Diffusers | 1 |  | 96.11 | 302.90 |
|  |  | vLLM-Omni | 1 |  | 59.45 | 221.08 |
|  |  | SGLang | 1 |  | 72.35 | 290.96 |
|  | multi_control | PyTorch | — |  |  |  |
|  |  | Diffusers | — |  |  |  |
|  |  | vLLM-Omni | 1 |  | 169.80 | 705.44 |
|  |  | SGLang | 1 |  | 128.49 | 547.55 |
|  | seg | PyTorch | 1 | 6.65 | 33.21 | 114.39 |
|  |  | Diffusers | 1 |  | 97.13 | 301.15 |
|  |  | vLLM-Omni | 1 |  | 60.19 | 222.51 |
|  |  | SGLang | 1 |  | 74.84 | 287.19 |
|  | wsm | PyTorch | 1 | 5.78 | 26.19 | 85.06 |
|  |  | Diffusers | 1 |  | 48.75 | 138.10 |
|  |  | vLLM-Omni | 1 |  | 28.24 | 97.58 |
|  |  | SGLang | 1 |  | 32.56 | 124.42 |
| **B300** | blur | PyTorch | 1 | 8.30 | 31.29 | 101.38 |
|  |  | Diffusers | 1 |  | 89.40 | 278.49 |
|  |  | vLLM-Omni | 1 |  | 56.52 | 204.65 |
|  |  | SGLang | 1 |  | 58.16 | 212.04 |
|  | depth | PyTorch | 1 | 7.81 | 32.60 | 105.09 |
|  |  | Diffusers | 1 |  | 93.48 | 282.93 |
|  |  | vLLM-Omni | 1 |  | 57.33 | 211.39 |
|  |  | SGLang | 1 |  | 59.13 | 214.33 |
|  | edge | PyTorch | 1 | 6.08 | 31.68 | 103.18 |
|  |  | Diffusers | 1 |  | 89.60 | 277.25 |
|  |  | vLLM-Omni | 1 |  | 56.66 | 208.71 |
|  |  | SGLang | 1 |  | 58.23 | 210.99 |
|  | multi_control | PyTorch | — |  |  |  |
|  |  | Diffusers | — |  |  |  |
|  |  | vLLM-Omni | 1 |  | 163.75 | 666.65 |
|  |  | SGLang | 1 |  | 98.05 | 389.30 |
|  | seg | PyTorch | 1 | 7.14 | 32.89 | 106.26 |
|  |  | Diffusers | 1 |  | 89.99 | 280.71 |
|  |  | vLLM-Omni | 1 |  | 57.52 | 209.82 |
|  |  | SGLang | 1 |  | 59.67 | 212.82 |
|  | wsm | PyTorch | 1 | 5.89 | 25.42 | 78.99 |
|  |  | Diffusers | 1 |  | 46.64 | 129.77 |
|  |  | vLLM-Omni | 1 |  | 27.39 | 92.13 |
|  |  | SGLang | 1 |  | 27.12 | 93.27 |
| **GB300** | blur | PyTorch | 1 |  |  |  |
|  |  | Diffusers | 1 |  | 80.61 | 236.36 |
|  |  | vLLM-Omni | 1 |  | 48.37 | 175.02 |
|  |  | SGLang | 1 |  | 53.52 | 189.19 |
|  | depth | PyTorch | 1 |  |  |  |
|  |  | Diffusers | 1 |  | 81.57 | 238.78 |
|  |  | vLLM-Omni | 1 |  | 48.44 | 178.93 |
|  |  | SGLang | 1 |  | 53.06 | 192.77 |
|  | edge | PyTorch | 1 |  |  |  |
|  |  | Diffusers | 1 |  | 80.54 | 235.31 |
|  |  | vLLM-Omni | 1 |  | 48.34 | 176.42 |
|  |  | SGLang | 1 |  | 53.23 | 190.56 |
|  | multi_control | PyTorch | — |  |  |  |
|  |  | Diffusers | — |  |  |  |
|  |  | vLLM-Omni | 1 |  | 140.85 | 564.76 |
|  |  | SGLang | 1 |  | 88.88 | 345.68 |
|  | seg | PyTorch | 1 |  |  |  |
|  |  | Diffusers | 1 |  | 81.67 | 237.79 |
|  |  | vLLM-Omni | 1 |  | 47.24 | 178.06 |
|  |  | SGLang | 1 |  | 53.71 | 193.46 |
|  | wsm | PyTorch | 1 |  |  |  |
|  |  | Diffusers | 1 |  | 42.26 | 110.70 |
|  |  | vLLM-Omni | 1 |  | 23.99 | 77.77 |
|  |  | SGLang | 1 |  | 25.07 | 84.15 |
| **DGX Spark** | blur | Diffusers | 1 |  | 1284.48 | 4269.37 |
| **DGX Spark** | depth | Diffusers | 1 |  | 1340.97 | 4346.09 |
|  | depth | NIM | 1 |  | 118.52 |  |
| **DGX Spark** | edge | PyTorch | 1 | 118.90 | 903.10 | 3632.50 |
|  | edge | Diffusers | 1 |  | 1286.19 | 4263.48 |
|  | edge | NIM | 1 |  | 114.30 |  |
| **DGX Spark** | seg | Diffusers | 1 |  | 1296.84 | 4285.75 |
|  | seg | NIM | 1 |  | 116.26 |  |
| **DGX Spark** | wsm | Diffusers | 1 |  | 632.78 | 1933.05 |
|  | wsm | NIM | 1 |  | 71.25 |  |
| **DGX Station** | blur | Diffusers | 1 |  | 81.03 | 246.79 |
| **DGX Station** | depth | Diffusers | 1 |  | 84.35 | 250.34 |
| **DGX Station** | edge | PyTorch | 1 | 12.30 | 48.90 | 178.40 |
|  | edge | Diffusers | 1 |  | 81.33 | 246.52 |
| **DGX Station** | seg | Diffusers | 1 |  | 82.17 | 247.90 |
| **DGX Station** | wsm | Diffusers | 1 |  | 41.86 | 114.47 |
| **RTX PRO 5000 Blackwell 72GB** | blur | Diffusers | 1 |  | 496.50 | 1751.76 |
| **RTX PRO 5000 Blackwell 72GB** | depth | Diffusers | 1 |  | 516.83 | 1780.13 |
|  | depth | NIM | 1 |  | 46.93 |  |
| **RTX PRO 5000 Blackwell 72GB** | edge | Diffusers | 1 |  | 498.43 | 1754.28 |
|  | edge | NIM | 1 |  | 45.56 |  |
| **RTX PRO 5000 Blackwell 72GB** | seg | Diffusers | 1 |  | 502.19 | 1758.95 |
|  | seg | NIM | 1 |  | 46.29 |  |
| **RTX PRO 5000 Blackwell 72GB** | wsm | Diffusers | 1 |  | 239.88 | 774.30 |
|  | wsm | NIM | 1 |  | 28.19 |  |

<sub>Transfer notes:
1. Control hints follow the `extra_params` names: `blur`, `depth`, `edge`, `multi_control`, `seg`, and `wsm`. `multi_control` combines several hints in one request.
2. The PyTorch campaign reports `segmentation`, which is the same control as `seg` in the Diffusers campaign.
3. Diffusers FP8 has no `multi_control` measurements.</sub>

## Action generation

### Forward Dynamics — Autonomous Vehicle (AV)

| GPU | Engine | GPUs | 256p | 480p | 720p |
|---|---|:-:|---:|---:|---:|
| **RTX PRO 6000 Blackwell** | PyTorch | 1 | 5.80 | 5.78 | 5.80 |
|  | Diffusers | 1 |  | 16.34 | 17.31 |
|  | SGLang | 1 |  | 8.97 | 27.06 |
| **H20** | PyTorch | 1 | 13.44 | 13.32 | 13.44 |
|  | Diffusers | 1 |  | 32.45 | 34.29 |
|  | SGLang | 1 |  | 28.17 | 80.21 |
| **H100 80GB HBM3** | PyTorch | 1 | 3.54 | 3.53 | 3.53 |
|  | Diffusers | 1 |  | 11.68 | 12.20 |
|  | SGLang | 1 |  | 5.52 | 15.57 |
| **H200 141GB HBM3** | PyTorch | 1 | 3.45 | 3.47 | 3.46 |
|  | Diffusers | 1 |  | 11.19 | 11.73 |
|  | SGLang | 1 |  | 5.34 | 15.61 |
| **B200** | PyTorch | 1 | 2.16 | 2.16 | 2.16 |
|  | Diffusers | 1 |  | 8.05 | 8.42 |
|  | SGLang | 1 |  | 3.27 | 9.35 |
| **B300** | PyTorch | 1 | 2.11 | 2.09 | 2.10 |
|  | Diffusers | 1 |  | 8.83 | 8.96 |
|  | SGLang | 1 |  | 2.93 | 7.79 |
| **GB300** | PyTorch | 1 |  |  |  |
|  | Diffusers | 1 |  | 7.71 | 7.64 |
|  | SGLang | 1 |  | 2.78 | 7.52 |
| **DGX Spark** | NIM | 1 | 5.98 | 41.33 | 119.42 |
|  | Diffusers | 1 |  | 80.13 | 84.19 |
| **RTX PRO 5000 Blackwell 72GB** | NIM | 1 | 2.24 | 15.28 | 46.28 |
|  | Diffusers | 1 |  | 26.74 | 27.98 |
| **DGX Station** | Diffusers | 1 |  | 7.04 | 7.31 |

### Forward Dynamics — Camera

| GPU | Engine | GPUs | 256p | 480p | 720p |
|---|---|:-:|---:|---:|---:|
| **RTX PRO 6000 Blackwell** | PyTorch | 1 | 5.47 | 5.49 | 5.48 |
|  | Diffusers | 1 |  |  |  |
|  | SGLang | 1 |  | 19.31 | 56.18 |
| **H20** | PyTorch | 1 | 12.69 | 12.57 | 12.58 |
|  | Diffusers | 1 |  |  |  |
|  | SGLang | 1 |  | 28.22 | 80.15 |
| **H100 80GB HBM3** | PyTorch | 1 | 3.38 | 3.37 | 3.36 |
|  | Diffusers | 1 |  |  |  |
|  | SGLang | 1 |  | 5.53 | 15.76 |
| **H200 141GB HBM3** | PyTorch | 1 | 3.29 | 3.30 | 3.29 |
|  | Diffusers | 1 |  |  |  |
|  | SGLang | 1 |  | 5.37 | 15.58 |
| **B200** | PyTorch | 1 | 2.06 | 2.06 | 2.06 |
|  | Diffusers | 1 |  |  |  |
|  | SGLang | 1 |  | 3.34 | 9.48 |
| **B300** | PyTorch | 1 | 1.98 | 1.98 | 2.06 |
|  | Diffusers | 1 |  |  |  |
|  | SGLang | 1 |  | 2.96 | 7.80 |
| **GB300** | SGLang | 1 |  | 2.84 | 7.48 |

### Forward Dynamics — Robot

| GPU | Engine | GPUs | 256p | 480p | 720p |
|---|---|:-:|---:|---:|---:|
| **RTX PRO 6000 Blackwell** | PyTorch | 1 | 1.28 | 1.28 | 1.28 |
|  | Diffusers | 1 |  | 5.43 | 5.63 |
| **H20** | PyTorch | 1 | 2.85 | 2.84 | 2.84 |
|  | Diffusers | 1 |  | 9.62 | 10.08 |
| **H100 80GB HBM3** | PyTorch | 1 | 0.85 | 0.85 | 0.85 |
|  | Diffusers | 1 |  | 4.63 | 4.79 |
| **H200 141GB HBM3** | PyTorch | 1 | 0.84 | 0.84 | 0.84 |
|  | Diffusers | 1 |  | 4.58 | 4.69 |
| **B200** | PyTorch | 1 | 0.61 | 0.61 | 0.60 |
|  | Diffusers | 1 |  | 4.09 | 4.35 |
| **B300** | PyTorch | 1 | 0.87 | 0.88 | 0.60 |
|  | Diffusers | 1 |  | 3.78 | 5.98 |
| **GB300** | PyTorch | 1 |  |  |  |
|  | Diffusers | 1 |  | 5.70 | 6.32 |
| **DGX Spark** | Diffusers | 1 |  | 26.28 | 27.98 |
| **DGX Station** | Diffusers | 1 |  | 5.20 | 5.22 |
| **RTX PRO 5000 Blackwell 72GB** | Diffusers | 1 |  | 7.83 | 8.28 |

### Inverse Dynamics — Autonomous Vehicle (AV)

| GPU | Engine | GPUs | 256p | 480p | 720p |
|---|---|:-:|---:|---:|---:|
| **RTX PRO 6000 Blackwell** | PyTorch | 1 | 9.02 | 9.01 | 9.00 |
|  | Diffusers | 1 |  | 16.23 | 17.11 |
|  | SGLang | 1 |  | 7.98 | 50.85 |
| **H20** | PyTorch | 1 | 21.15 | 21.36 | 21.15 |
|  | Diffusers | 1 |  | 32.38 | 33.92 |
|  | SGLang | 1 |  | 25.15 | 73.75 |
| **H100 80GB HBM3** | PyTorch | 1 | 5.45 | 5.44 | 5.44 |
|  | Diffusers | 1 |  | 11.61 | 12.21 |
|  | SGLang | 1 |  | 5.14 | 14.97 |
| **H200 141GB HBM3** | PyTorch | 1 | 5.32 | 5.39 | 5.32 |
|  | Diffusers | 1 |  | 11.10 | 11.65 |
|  | SGLang | 1 |  | 5.07 | 14.99 |
| **B200** | PyTorch | 1 | 3.25 | 3.24 | 3.24 |
|  | Diffusers | 1 |  | 8.00 | 8.27 |
|  | SGLang | 1 |  | 3.29 | 9.69 |
| **B300** | PyTorch | 1 | 3.15 | 3.14 | 3.13 |
|  | Diffusers | 1 |  | 8.90 | 8.95 |
|  | SGLang | 1 |  | 3.15 | 8.07 |
| **GB300** | PyTorch | 1 |  |  |  |
|  | Diffusers | 1 |  | 7.85 | 7.88 |
|  | SGLang | 1 |  | 2.49 | 7.20 |
| **DGX Spark** | NIM | 1 | 6.94 | 45.62 | 128.84 |
|  | Diffusers | 1 |  | 79.53 | 83.68 |
| **RTX PRO 5000 Blackwell 72GB** | NIM | 1 | 3.11 | 17.24 | 49.90 |
|  | Diffusers | 1 |  | 26.57 | 27.88 |
| **DGX Station** | Diffusers | 1 |  | 6.99 | 7.32 |

### Inverse Dynamics — Robot

| GPU | Engine | GPUs | 256p | 480p | 720p |
|---|---|:-:|---:|---:|---:|
| **RTX PRO 6000 Blackwell** | PyTorch | 1 | 1.86 | 1.85 | 1.85 |
|  | Diffusers | 1 |  | 5.48 | 5.65 |
| **H20** | PyTorch | 1 | 4.32 | 4.28 | 4.31 |
|  | Diffusers | 1 |  | 9.60 | 10.09 |
| **H100 80GB HBM3** | PyTorch | 1 | 1.21 | 1.22 | 1.22 |
|  | Diffusers | 1 |  | 4.75 | 4.79 |
| **H200 141GB HBM3** | PyTorch | 1 | 1.19 | 1.19 | 1.19 |
|  | Diffusers | 1 |  | 4.64 | 4.72 |
| **B200** | PyTorch | 1 | 0.84 | 0.86 | 0.84 |
|  | Diffusers | 1 |  | 4.02 | 4.18 |
| **B300** | PyTorch | 1 | 0.87 | 0.84 | 0.84 |
|  | Diffusers | 1 |  | 5.91 | 6.03 |
| **GB300** | PyTorch | 1 |  |  |  |
|  | Diffusers | 1 |  | 6.57 | 6.08 |
| **DGX Spark** | Diffusers | 1 |  | 26.40 | 28.00 |
| **DGX Station** | Diffusers | 1 |  | 5.22 | 5.39 |
| **RTX PRO 5000 Blackwell 72GB** | Diffusers | 1 |  | 7.82 | 8.28 |

### Policy — Autonomous Vehicle (AV)

| GPU | Engine | GPUs | 256p | 480p | 720p |
|---|---|:-:|---:|---:|---:|
| **RTX PRO 6000 Blackwell** | PyTorch | 1 | 5.80 | 5.79 | 5.80 |
|  | Diffusers | 1 |  |  |  |
| **H20** | PyTorch | 1 | 13.44 | 13.44 | 13.44 |
|  | Diffusers | 1 |  |  |  |
| **H100 80GB HBM3** | PyTorch | 1 | 3.54 | 3.54 | 3.54 |
|  | Diffusers | 1 |  |  |  |
| **H200 141GB HBM3** | PyTorch | 1 | 3.51 | 3.49 | 3.51 |
|  | Diffusers | 1 |  |  |  |
| **B200** | PyTorch | 1 | 2.17 | 2.17 | 2.14 |
|  | Diffusers | 1 |  |  |  |
| **B300** | PyTorch | 1 | 2.14 | 2.13 | 2.16 |
|  | Diffusers | 1 |  |  |  |
| **DGX Spark** | NIM | 1 | 6.06 | 41.52 | 119.82 |
| **RTX PRO 5000 Blackwell 72GB** | NIM | 1 | 2.28 | 15.34 | 46.40 |

### Policy — Robot

| GPU | Engine | GPUs | 256p | 480p | 720p |
|---|---|:-:|---:|---:|---:|
| **RTX PRO 6000 Blackwell** | PyTorch | 1 | 1.28 | 1.29 | 1.24 |
|  | Diffusers | 1 |  |  |  |
| **H20** | PyTorch | 1 | 2.75 | 2.87 | 2.75 |
|  | Diffusers | 1 |  |  |  |
| **H100 80GB HBM3** | PyTorch | 1 | 0.84 | 0.84 | 0.85 |
|  | Diffusers | 1 |  |  |  |
| **H200 141GB HBM3** | PyTorch | 1 | 0.83 | 0.82 | 0.85 |
|  | Diffusers | 1 |  |  |  |
| **B200** | PyTorch | 1 | 0.60 | 0.61 | 0.62 |
|  | Diffusers | 1 |  |  |  |
| **B300** | PyTorch | 1 | 0.89 | 0.85 | 0.86 |
|  | Diffusers | 1 |  |  |  |

<sub>Action and additional-modality notes:
1. Forward-dynamics cells for the AV domain are the mean of the forward, left, and right trajectories; inverse-dynamics cells are the mean of the two inverse trajectories.
2. The Diffusers FP8 campaign covers forward and inverse dynamics for the AV and Robot domains only, so the Camera and Policy rows carry PyTorch values alone.
3. Policy-DROID is a separate checkpoint and was not part of either FP8 campaign.</sub>

### Inverse Dynamics — Camera

| GPU | Engine | GPUs | 256p | 480p | 720p |
|---|---|:-:|---:|---:|---:|
| **B200** | SGLang | 1 |  | 3.99 | 9.12 |
| **B300** | SGLang | 1 |  | 3.67 | 8.03 |
| **GB300** | SGLang | 1 |  | 3.14 | 6.81 |
| **H100 80GB HBM3** | SGLang | 1 |  | 5.95 | 14.56 |
| **H20** | SGLang | 1 |  | 25.89 | 73.30 |
| **H200 141GB HBM3** | SGLang | 1 |  | 5.79 | 14.57 |
| **RTX PRO 6000 Blackwell** | SGLang | 1 |  | 8.76 | 24.62 |
