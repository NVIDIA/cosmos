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
| **RTX PRO 6000 Blackwell** | PyTorch\* | 1 | 4.71 |  |  |
|  | Diffusers | 1 |  |  |  |
|  | NIM | 1 | 5.06 |  |  |
|  | vLLM-Omni | 1 |  | 80.74 | 311.43 |
|  | SGLang | 1 |  | 87.73 | 333.63 |
| **H20** | PyTorch\* | 1 | 11.82 |  |  |
|  | Diffusers | 1 |  |  |  |
|  | NIM | 1 | 11.78 |  |  |
|  | vLLM-Omni | 1 |  |  |  |
|  | SGLang | 1 |  |  |  |
| **H100 80GB HBM3** | PyTorch\* | 1 | 3.12 |  |  |
|  | Diffusers | 1 |  |  |  |
|  | NIM | 1 | 3.34 |  |  |
|  | vLLM-Omni | 1 |  |  |  |
|  | SGLang | 1 |  | 48.84 |  |
| **H200 141GB HBM3** | PyTorch\* | 1 | 3.09 |  |  |
|  | Diffusers | 1 |  |  |  |
|  | NIM | 1 | 3.15 |  |  |
|  | vLLM-Omni | 1 |  |  |  |
|  | SGLang | 1 |  | 48.40 |  |
| **B200** | PyTorch\* | 1 | 1.97 |  |  |
|  | Diffusers | 1 |  |  |  |
|  | NIM | 1 | 2.00 |  |  |
|  | vLLM-Omni | 1 |  | 22.94 | 85.58 |
|  | SGLang | 1 |  |  |  |
| **B300** | PyTorch\* | 1 | 1.99 |  |  |
|  | Diffusers | 1 |  |  |  |
|  | NIM | 1 |  |  |  |
|  | vLLM-Omni | 1 |  | 21.75 | 79.52 |
|  | SGLang | 1 |  | 23.63 | 84.74 |
| **GB300** | PyTorch\* | 1 |  |  |  |
|  | Diffusers | 1 |  |  |  |
|  | NIM | 1 |  |  |  |
|  | vLLM-Omni | 1 |  | 18.93 | 67.68 |
|  | SGLang | 1 |  |  | 77.94 |
| **DGX Spark** | NIM | 1 | 36.78 | 358.61 | 1360.13 |
|  | Diffusers | 1 |  | 522.73 | 1730.23 |
| **RTX PRO 5000 Blackwell 72GB** | NIM | 1 | 13.15 | 143.06 | 557.53 |
|  | Diffusers | 1 |  | 198.30 | 692.91 |
| **DGX Station** | Diffusers | 1 |  | 33.86 | 102.37 |

### Image-to-Video (i2v)

| GPU | Engine | GPUs | 256p | 480p | 720p |
|---|---|:-:|---:|---:|---:|
| **RTX PRO 6000 Blackwell** | PyTorch\* | 1 | 4.50 |  |  |
|  | Diffusers | 1 |  |  |  |
|  | NIM | 1 | 4.90 |  |  |
|  | vLLM-Omni | 1 |  | 80.83 | 311.89 |
|  | SGLang | 1 |  | 88.36 |  |
| **H20** | PyTorch\* | 1 | 11.19 |  |  |
|  | Diffusers | 1 |  |  |  |
|  | NIM | 1 | 11.67 |  |  |
|  | vLLM-Omni | 1 |  |  |  |
|  | SGLang | 1 |  |  |  |
| **H100 80GB HBM3** | PyTorch\* | 1 | 2.96 |  |  |
|  | Diffusers | 1 |  |  |  |
|  | NIM | 1 | 3.39 |  |  |
|  | vLLM-Omni | 1 |  |  |  |
|  | SGLang | 1 |  | 48.87 |  |
| **H200 141GB HBM3** | PyTorch\* | 1 | 2.94 |  |  |
|  | Diffusers | 1 |  |  |  |
|  | NIM | 1 | 3.16 |  |  |
|  | vLLM-Omni | 1 |  |  |  |
|  | SGLang | 1 |  | 48.34 |  |
| **B200** | PyTorch\* | 1 | 1.87 |  |  |
|  | Diffusers | 1 |  |  |  |
|  | NIM | 1 |  |  |  |
|  | vLLM-Omni | 1 |  | 23.30 | 84.36 |
|  | SGLang | 1 |  |  |  |
| **B300** | PyTorch\* | 1 | 1.78 |  |  |
|  | Diffusers | 1 |  |  |  |
|  | NIM | 1 |  |  |  |
|  | vLLM-Omni | 1 |  | 21.75 | 79.36 |
|  | SGLang | 1 |  | 23.97 | 85.10 |
| **GB300** | PyTorch\* | 1 |  |  |  |
|  | Diffusers | 1 |  |  |  |
|  | NIM | 1 | 1.96 |  |  |
|  | vLLM-Omni | 1 |  | 18.80 | 67.61 |
|  | SGLang | 1 |  |  | 77.57 |
| **DGX Spark** | NIM | 1 | 36.57 | 354.86 | 1350.29 |
|  | Diffusers | 1 |  | 533.41 | 1739.15 |
| **RTX PRO 5000 Blackwell 72GB** | NIM | 1 | 13.24 | 143.25 | 557.87 |
|  | Diffusers | 1 |  | 201.55 | 700.33 |
| **DGX Station** | Diffusers | 1 |  | 34.50 | 104.17 |

### Text-to-Image (t2i)

| GPU | Engine | GPUs | 256p | 480p | 720p |
|---|---|:-:|---:|---:|---:|
| **RTX PRO 6000 Blackwell** | PyTorch\* | 1 | 0.97 |  |  |
|  | Diffusers | 1 |  |  |  |
|  | NIM | 1 | 0.72 |  |  |
|  | vLLM-Omni | 1 |  |  |  |
|  | SGLang | 1 |  |  |  |
| **H20** | PyTorch\* | 1 | 1.27 |  |  |
|  | Diffusers | 1 |  |  |  |
|  | NIM | 1 | 1.09 |  |  |
|  | vLLM-Omni | 1 |  |  |  |
|  | SGLang | 1 |  |  |  |
| **H100 80GB HBM3** | PyTorch\* | 1 | 1.16 |  |  |
|  | Diffusers | 1 |  |  |  |
|  | NIM | 1 | 1.02 |  |  |
|  | vLLM-Omni | 1 |  |  |  |
|  | SGLang | 1 |  |  |  |
| **H200 141GB HBM3** | PyTorch\* | 1 | 1.12 |  |  |
|  | Diffusers | 1 |  |  |  |
|  | NIM | 1 | 1.02 |  |  |
|  | vLLM-Omni | 1 |  |  |  |
|  | SGLang | 1 |  |  |  |
| **B200** | PyTorch\* | 1 | 0.96 |  |  |
|  | Diffusers | 1 |  |  |  |
|  | NIM | 1 |  |  |  |
|  | vLLM-Omni | 1 |  |  |  |
|  | SGLang | 1 |  |  |  |
| **B300** | PyTorch\* | 1 | 0.95 |  |  |
|  | Diffusers | 1 |  |  |  |
|  | NIM | 1 |  |  |  |
|  | vLLM-Omni | 1 |  |  |  |
|  | SGLang | 1 |  |  |  |
| **GB300** | PyTorch\* | 1 |  |  |  |
|  | Diffusers | 1 |  |  |  |
|  | NIM | 1 |  |  |  |
|  | vLLM-Omni | 1 |  |  |  |
|  | SGLang | 1 |  |  |  |
| **DGX Spark** | NIM | 1 | 2.34 |  |  |
|  | Diffusers | 1 |  |  |  |
| **RTX PRO 5000 Blackwell 72GB** | NIM | 1 | 1.63 |  |  |
|  | Diffusers | 1 |  |  |  |
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
| **RTX PRO 6000 Blackwell** | PyTorch | 1 | 4.40 |  |  |
|  | Diffusers | 1 |  |  |  |
|  | vLLM-Omni | 1 |  |  |  |
|  | SGLang | 1 |  |  |  |
| **H20** | PyTorch | 1 | 10.99 |  |  |
|  | Diffusers | 1 |  |  |  |
|  | vLLM-Omni | 1 |  |  |  |
|  | SGLang | 1 |  |  |  |
| **H100 80GB HBM3** | PyTorch | 1 | 2.93 |  |  |
|  | Diffusers | 1 |  |  |  |
|  | vLLM-Omni | 1 |  |  |  |
|  | SGLang | 1 |  |  |  |
| **H200 141GB HBM3** | PyTorch | 1 | 2.90 |  |  |
|  | Diffusers | 1 |  |  |  |
|  | vLLM-Omni | 1 |  |  |  |
|  | SGLang | 1 |  |  |  |
| **B200** | PyTorch | 1 | 1.85 |  |  |
|  | Diffusers | 1 |  |  |  |
|  | vLLM-Omni | 1 |  |  |  |
|  | SGLang | 1 |  |  |  |
| **B300** | PyTorch | 1 | 1.92 |  |  |
|  | Diffusers | 1 |  |  |  |
|  | vLLM-Omni | 1 |  |  |  |
|  | SGLang | 1 |  |  |  |
| **GB300** | PyTorch | 1 |  |  |  |
|  | Diffusers | 1 |  |  |  |
|  | vLLM-Omni | 1 |  |  |  |
|  | SGLang | 1 |  |  |  |
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
| **RTX PRO 6000 Blackwell** | PyTorch | 1 | 7.67 |  |  |
|  | Diffusers | 1 |  |  |  |
|  | vLLM-Omni | 1 |  | 82.19 | 316.45 |
|  | SGLang | 1 |  | 89.52 | 339.67 |
| **H20** | PyTorch | 1 | 17.97 |  |  |
|  | Diffusers | 1 |  | 240.59 | 873.73 |
|  | vLLM-Omni | 1 |  | 210.85 | 800.56 |
|  | SGLang | 1 |  |  |  |
| **H100 80GB HBM3** | PyTorch | 1 | 4.88 |  |  |
|  | Diffusers | 1 |  |  |  |
|  | vLLM-Omni | 1 |  |  | 184.65 |
|  | SGLang | 1 |  | 48.99 | 181.17 |
| **H200 141GB HBM3** | PyTorch | 1 | 4.79 |  |  |
|  | Diffusers | 1 |  |  |  |
|  | vLLM-Omni | 1 |  |  | 190.89 |
|  | SGLang | 1 |  | 48.72 | 182.85 |
| **B200** | PyTorch | 1 | 3.10 |  |  |
|  | Diffusers | 1 |  |  |  |
|  | vLLM-Omni | 1 |  | 23.43 | 85.20 |
|  | SGLang | 1 |  | 30.12 |  |
| **B300** | PyTorch | 1 | 3.12 |  |  |
|  | Diffusers | 1 |  |  |  |
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
| **RTX PRO 6000 Blackwell** | PyTorch | 1 | 7.30 |  |  |
|  | Diffusers | 1 |  |  |  |
|  | vLLM-Omni | 1 |  | 82.12 | 316.20 |
|  | SGLang | 1 |  | 89.60 | 340.06 |
| **H20** | PyTorch | 1 | 17.08 |  |  |
|  | Diffusers | 1 |  |  |  |
|  | vLLM-Omni | 1 |  | 210.65 | 800.51 |
|  | SGLang | 1 |  |  |  |
| **H100 80GB HBM3** | PyTorch | 1 | 4.63 |  |  |
|  | Diffusers | 1 |  |  |  |
|  | vLLM-Omni | 1 |  |  |  |
|  | SGLang | 1 |  | 49.68 | 184.14 |
| **H200 141GB HBM3** | PyTorch | 1 | 4.54 |  |  |
|  | Diffusers | 1 |  |  |  |
|  | vLLM-Omni | 1 |  |  | 192.38 |
|  | SGLang | 1 |  | 48.88 | 184.08 |
| **B200** | PyTorch | 1 | 2.96 |  |  |
|  | Diffusers | 1 |  |  |  |
|  | vLLM-Omni | 1 |  | 23.27 | 84.73 |
|  | SGLang | 1 |  | 30.60 |  |
| **B300** | PyTorch | 1 | 2.81 |  |  |
|  | Diffusers | 1 |  |  |  |
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
| **RTX PRO 6000 Blackwell** | blur | PyTorch | 1 | 15.92 |  |  |
|  |  | Diffusers | 1 |  |  |  |
|  |  | vLLM-Omni | 1 |  |  |  |
|  |  | SGLang | 1 |  |  |  |
|  | depth | PyTorch | 1 | 15.34 |  |  |
|  |  | Diffusers | 1 |  |  |  |
|  |  | vLLM-Omni | 1 |  |  |  |
|  |  | SGLang | 1 |  |  |  |
|  | edge | PyTorch | 1 | 13.27 |  |  |
|  |  | Diffusers | 1 |  |  |  |
|  |  | vLLM-Omni | 1 |  |  |  |
|  |  | SGLang | 1 |  |  |  |
|  | multi_control | PyTorch | — |  |  |  |
|  |  | Diffusers | — |  |  |  |
|  |  | vLLM-Omni | 1 |  | 608.48 | 2644.63 |
|  |  | SGLang | 1 |  | 365.88 | 1588.91 |
|  | seg | PyTorch | 1 | 14.54 |  |  |
|  |  | Diffusers | 1 |  |  |  |
|  |  | vLLM-Omni | 1 |  | 205.45 |  |
|  |  | SGLang | 1 |  |  |  |
|  | wsm | PyTorch | 1 | 12.81 | 78.87 | 280.91 |
|  |  | Diffusers | 1 |  | 134.04 | 430.19 |
|  |  | vLLM-Omni | 1 |  | 93.30 | 345.93 |
|  |  | SGLang | 1 |  | 93.63 | 353.94 |
| **H20** | blur | PyTorch | 1 | 36.48 |  |  |
|  |  | Diffusers | 1 |  |  |  |
|  |  | vLLM-Omni | 1 |  |  |  |
|  |  | SGLang | 1 |  |  |  |
|  | depth | PyTorch | 1 | 34.13 |  |  |
|  |  | Diffusers | 1 |  |  |  |
|  |  | vLLM-Omni | 1 |  |  |  |
|  |  | SGLang | 1 |  |  |  |
|  | edge | PyTorch | 1 | 29.64 |  |  |
|  |  | Diffusers | 1 |  |  |  |
|  |  | vLLM-Omni | 1 |  |  |  |
|  |  | SGLang | 1 |  |  |  |
|  | multi_control | PyTorch | 1 | 36.57 |  |  |
|  |  | Diffusers | 1 |  |  |  |
|  |  | vLLM-Omni | 1 |  |  |  |
|  |  | SGLang | 1 |  |  |  |
|  | seg | PyTorch | 1 | 31.20 |  |  |
|  |  | Diffusers | 1 |  |  |  |
|  |  | vLLM-Omni | 1 |  |  |  |
|  |  | SGLang | 1 |  |  |  |
|  | wsm | PyTorch | 1 | 28.15 | 186.34 | 707.32 |
|  |  | Diffusers | 1 |  | 287.51 | 969.79 |
|  |  | vLLM-Omni | 1 |  | 241.43 | 874.83 |
|  |  | SGLang | 1 |  | 280.08 | 995.06 |
| **H100 80GB HBM3** | blur | PyTorch | 1 | 10.37 |  |  |
|  |  | Diffusers | 1 |  |  |  |
|  |  | vLLM-Omni | 1 |  |  |  |
|  |  | SGLang | 1 |  |  |  |
|  | depth | PyTorch | 1 | 9.83 |  |  |
|  |  | Diffusers | 1 |  |  |  |
|  |  | vLLM-Omni | 1 |  |  |  |
|  |  | SGLang | 1 |  |  |  |
|  | edge | PyTorch | 1 | 8.49 |  |  |
|  |  | Diffusers | 1 |  |  |  |
|  |  | vLLM-Omni | 1 |  |  |  |
|  |  | SGLang | 1 |  |  |  |
|  | multi_control | PyTorch | 1 | 10.51 |  |  |
|  |  | Diffusers | 1 |  |  |  |
|  |  | vLLM-Omni | 1 |  |  |  |
|  |  | SGLang | 1 |  |  |  |
|  | seg | PyTorch | 1 | 10.03 |  |  |
|  |  | Diffusers | 1 |  |  |  |
|  |  | vLLM-Omni | 1 |  |  |  |
|  |  | SGLang | 1 |  |  |  |
|  | wsm | PyTorch | 1 | 8.33 | 44.52 | 153.26 |
|  |  | Diffusers | 1 |  | 84.48 | 264.67 |
|  |  | vLLM-Omni | 1 |  | 70.87 | 213.20 |
|  |  | SGLang | 1 |  | 53.24 | 193.99 |
| **H200 141GB HBM3** | blur | PyTorch | 1 | 10.32 |  |  |
|  |  | Diffusers | 1 |  |  |  |
|  |  | vLLM-Omni | 1 |  |  |  |
|  |  | SGLang | 1 |  |  |  |
|  | depth | PyTorch | 1 | 9.70 |  |  |
|  |  | Diffusers | 1 |  |  |  |
|  |  | vLLM-Omni | 1 |  |  |  |
|  |  | SGLang | 1 |  | 116.76 |  |
|  | edge | PyTorch | 1 | 8.60 |  |  |
|  |  | Diffusers | 1 |  |  |  |
|  |  | vLLM-Omni | 1 |  |  |  |
|  |  | SGLang | 1 |  |  |  |
|  | multi_control | PyTorch | 1 | 10.46 |  |  |
|  |  | Diffusers | 1 |  |  |  |
|  |  | vLLM-Omni | 1 |  |  |  |
|  |  | SGLang | 1 |  |  |  |
|  | seg | PyTorch | 1 | 9.68 |  |  |
|  |  | Diffusers | 1 |  |  |  |
|  |  | vLLM-Omni | 1 |  |  |  |
|  |  | SGLang | 1 |  |  |  |
|  | wsm | PyTorch | 1 | 8.39 | 45.06 | 154.98 |
|  |  | Diffusers | 1 |  | 82.92 | 263.74 |
|  |  | vLLM-Omni | 1 |  | 71.05 | 214.79 |
|  |  | SGLang | 1 |  | 52.21 | 192.67 |
| **B200** | blur | PyTorch | 1 |  |  |  |
|  |  | Diffusers | 1 |  |  |  |
|  |  | vLLM-Omni | 1 |  | 58.32 | 216.19 |
|  |  | SGLang | 1 |  |  |  |
|  | depth | PyTorch | 1 |  |  |  |
|  |  | Diffusers | 1 |  |  |  |
|  |  | vLLM-Omni | 1 |  | 59.42 | 223.11 |
|  |  | SGLang | 1 |  |  |  |
|  | edge | PyTorch | 1 |  |  |  |
|  |  | Diffusers | 1 |  |  |  |
|  |  | vLLM-Omni | 1 |  | 59.45 | 221.08 |
|  |  | SGLang | 1 |  |  |  |
|  | multi_control | PyTorch | — |  |  |  |
|  |  | Diffusers | — |  |  |  |
|  |  | vLLM-Omni | 1 |  | 169.80 | 705.44 |
|  |  | SGLang | 1 |  | 128.49 | 547.55 |
|  | seg | PyTorch | 1 |  |  |  |
|  |  | Diffusers | 1 |  |  |  |
|  |  | vLLM-Omni | 1 |  | 60.19 | 222.51 |
|  |  | SGLang | 1 |  |  |  |
|  | wsm | PyTorch | 1 | 5.78 | 26.19 | 85.06 |
|  |  | Diffusers | 1 |  | 48.75 | 138.10 |
|  |  | vLLM-Omni | 1 |  | 28.24 | 97.58 |
|  |  | SGLang | 1 |  | 32.56 | 124.42 |
| **B300** | blur | PyTorch | 1 |  |  |  |
|  |  | Diffusers | 1 |  |  |  |
|  |  | vLLM-Omni | 1 |  | 56.52 |  |
|  |  | SGLang | 1 |  | 58.16 |  |
|  | depth | PyTorch | 1 |  |  |  |
|  |  | Diffusers | 1 |  |  |  |
|  |  | vLLM-Omni | 1 |  | 57.33 |  |
|  |  | SGLang | 1 |  | 59.13 |  |
|  | edge | PyTorch | 1 |  |  |  |
|  |  | Diffusers | 1 |  |  |  |
|  |  | vLLM-Omni | 1 |  | 56.66 |  |
|  |  | SGLang | 1 |  | 58.23 |  |
|  | multi_control | PyTorch | — |  |  |  |
|  |  | Diffusers | — |  |  |  |
|  |  | vLLM-Omni | 1 |  | 163.75 | 666.65 |
|  |  | SGLang | 1 |  | 98.05 | 389.30 |
|  | seg | PyTorch | 1 |  |  |  |
|  |  | Diffusers | 1 |  |  |  |
|  |  | vLLM-Omni | 1 |  | 57.52 | 209.82 |
|  |  | SGLang | 1 |  | 59.67 |  |
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
| **DGX Spark** | depth | Diffusers | 1 |  |  | 4346.09 |
|  | depth | NIM | 1 |  |  |  |
| **DGX Spark** | edge | PyTorch | 1 | 118.90 |  | 3632.50 |
|  | edge | Diffusers | 1 |  |  | 4263.48 |
|  | edge | NIM | 1 |  |  |  |
| **DGX Spark** | seg | Diffusers | 1 |  |  | 4285.75 |
|  | seg | NIM | 1 |  |  |  |
| **DGX Spark** | wsm | Diffusers | 1 |  |  | 1933.05 |
|  | wsm | NIM | 1 |  |  |  |
| **DGX Station** | blur | Diffusers | 1 |  | 81.03 | 246.79 |
| **DGX Station** | depth | Diffusers | 1 |  | 84.35 | 250.34 |
| **DGX Station** | edge | PyTorch | 1 | 12.30 | 48.90 | 178.40 |
|  | edge | Diffusers | 1 |  | 81.33 | 246.52 |
| **DGX Station** | seg | Diffusers | 1 |  | 82.17 | 247.90 |
| **DGX Station** | wsm | Diffusers | 1 |  | 41.86 | 114.47 |
| **RTX PRO 5000 Blackwell 72GB** | blur | Diffusers | 1 |  | 496.50 | 1751.76 |
| **RTX PRO 5000 Blackwell 72GB** | depth | Diffusers | 1 |  |  | 1780.13 |
|  | depth | NIM | 1 |  |  |  |
| **RTX PRO 5000 Blackwell 72GB** | edge | Diffusers | 1 |  |  | 1754.28 |
|  | edge | NIM | 1 |  |  |  |
| **RTX PRO 5000 Blackwell 72GB** | seg | Diffusers | 1 |  |  | 1758.95 |
|  | seg | NIM | 1 |  |  |  |
| **RTX PRO 5000 Blackwell 72GB** | wsm | Diffusers | 1 |  |  | 774.30 |
|  | wsm | NIM | 1 |  |  |  |

<sub>Transfer notes:
1. Control hints follow the `extra_params` names: `blur`, `depth`, `edge`, `multi_control`, `seg`, and `wsm`. `multi_control` combines several hints in one request.
2. The PyTorch campaign reports `segmentation`, which is the same control as `seg` in the Diffusers campaign.
3. Diffusers FP8 has no `multi_control` measurements.</sub>

## Action generation

### Forward Dynamics — Autonomous Vehicle (AV)

| GPU | Engine | GPUs | 256p | 480p | 720p |
|---|---|:-:|---:|---:|---:|
| **RTX PRO 6000 Blackwell** | PyTorch | 1 | 5.80 |  |  |
|  | Diffusers | 1 |  |  |  |
|  | SGLang | 1 |  | 8.97 |  |
| **H20** | PyTorch | 1 | 13.44 |  |  |
|  | Diffusers | 1 |  |  |  |
|  | SGLang | 1 |  |  |  |
| **H100 80GB HBM3** | PyTorch | 1 | 3.54 |  |  |
|  | Diffusers | 1 |  |  |  |
|  | SGLang | 1 |  |  |  |
| **H200 141GB HBM3** | PyTorch | 1 | 3.45 |  |  |
|  | Diffusers | 1 |  |  |  |
|  | SGLang | 1 |  |  |  |
| **B200** | PyTorch | 1 | 2.16 |  |  |
|  | Diffusers | 1 |  |  |  |
|  | SGLang | 1 |  |  |  |
| **B300** | PyTorch | 1 | 2.11 |  |  |
|  | Diffusers | 1 |  |  |  |
|  | SGLang | 1 |  |  |  |
| **GB300** | PyTorch | 1 |  |  |  |
|  | Diffusers | 1 |  |  | 7.64 |
|  | SGLang | 1 |  |  | 7.52 |
| **DGX Spark** | NIM | 1 | 5.98 | 41.33 | 119.42 |
|  | Diffusers | 1 |  | 80.13 | 84.19 |
| **RTX PRO 5000 Blackwell 72GB** | NIM | 1 | 2.24 | 15.28 | 46.28 |
|  | Diffusers | 1 |  | 26.74 | 27.98 |
| **DGX Station** | Diffusers | 1 |  | 7.04 | 7.31 |

### Forward Dynamics — Camera

| GPU | Engine | GPUs | 256p | 480p | 720p |
|---|---|:-:|---:|---:|---:|
| **RTX PRO 6000 Blackwell** | PyTorch | 1 | 5.47 |  |  |
|  | Diffusers | 1 |  |  |  |
|  | SGLang | 1 |  |  |  |
| **H20** | PyTorch | 1 | 12.69 |  |  |
|  | Diffusers | 1 |  |  |  |
|  | SGLang | 1 |  |  |  |
| **H100 80GB HBM3** | PyTorch | 1 | 3.38 | 3.37 |  |
|  | Diffusers | 1 |  |  |  |
|  | SGLang | 1 |  | 5.53 |  |
| **H200 141GB HBM3** | PyTorch | 1 | 3.29 | 3.30 |  |
|  | Diffusers | 1 |  |  |  |
|  | SGLang | 1 |  | 5.37 |  |
| **B200** | PyTorch | 1 | 2.06 | 2.06 |  |
|  | Diffusers | 1 |  |  |  |
|  | SGLang | 1 |  | 3.34 |  |
| **B300** | PyTorch | 1 | 1.98 | 1.98 |  |
|  | Diffusers | 1 |  |  |  |
|  | SGLang | 1 |  | 2.96 |  |
| **GB300** | SGLang | 1 |  | 2.84 | 7.48 |

### Forward Dynamics — Robot

| GPU | Engine | GPUs | 256p | 480p | 720p |
|---|---|:-:|---:|---:|---:|
| **RTX PRO 6000 Blackwell** | PyTorch | 1 | 1.28 |  |  |
|  | Diffusers | 1 |  |  |  |
| **H20** | PyTorch | 1 | 2.85 |  |  |
|  | Diffusers | 1 |  |  |  |
| **H100 80GB HBM3** | PyTorch | 1 | 0.85 |  |  |
|  | Diffusers | 1 |  |  |  |
| **H200 141GB HBM3** | PyTorch | 1 | 0.84 |  |  |
|  | Diffusers | 1 |  |  |  |
| **B200** | PyTorch | 1 |  |  |  |
|  | Diffusers | 1 |  |  |  |
| **B300** | PyTorch | 1 |  |  |  |
|  | Diffusers | 1 |  |  |  |
| **GB300** | PyTorch | 1 |  |  |  |
|  | Diffusers | 1 |  |  |  |
| **DGX Spark** | Diffusers | 1 |  | 26.28 | 27.98 |
| **DGX Station** | Diffusers | 1 |  | 5.20 | 5.22 |
| **RTX PRO 5000 Blackwell 72GB** | Diffusers | 1 |  | 7.83 | 8.28 |

### Inverse Dynamics — Autonomous Vehicle (AV)

| GPU | Engine | GPUs | 256p | 480p | 720p |
|---|---|:-:|---:|---:|---:|
| **RTX PRO 6000 Blackwell** | PyTorch | 1 | 9.02 | 9.01 |  |
|  | Diffusers | 1 |  |  |  |
|  | SGLang | 1 |  |  |  |
| **H20** | PyTorch | 1 | 21.15 | 21.36 |  |
|  | Diffusers | 1 |  | 32.38 |  |
|  | SGLang | 1 |  | 25.15 |  |
| **H100 80GB HBM3** | PyTorch | 1 | 5.45 |  |  |
|  | Diffusers | 1 |  |  |  |
|  | SGLang | 1 |  |  |  |
| **H200 141GB HBM3** | PyTorch | 1 | 5.32 |  |  |
|  | Diffusers | 1 |  |  |  |
|  | SGLang | 1 |  |  |  |
| **B200** | PyTorch | 1 | 3.25 |  |  |
|  | Diffusers | 1 |  |  |  |
|  | SGLang | 1 |  |  |  |
| **B300** | PyTorch | 1 | 3.15 |  |  |
|  | Diffusers | 1 |  |  |  |
|  | SGLang | 1 |  |  |  |
| **GB300** | PyTorch | 1 |  |  |  |
|  | Diffusers | 1 |  |  | 7.88 |
|  | SGLang | 1 |  |  | 7.20 |
| **DGX Spark** | NIM | 1 | 6.94 | 45.62 | 128.84 |
|  | Diffusers | 1 |  | 79.53 | 83.68 |
| **RTX PRO 5000 Blackwell 72GB** | NIM | 1 | 3.11 | 17.24 | 49.90 |
|  | Diffusers | 1 |  | 26.57 | 27.88 |
| **DGX Station** | Diffusers | 1 |  | 6.99 | 7.32 |

### Inverse Dynamics — Robot

| GPU | Engine | GPUs | 256p | 480p | 720p |
|---|---|:-:|---:|---:|---:|
| **RTX PRO 6000 Blackwell** | PyTorch | 1 | 1.86 |  |  |
|  | Diffusers | 1 |  |  |  |
| **H20** | PyTorch | 1 | 4.32 |  |  |
|  | Diffusers | 1 |  |  |  |
| **H100 80GB HBM3** | PyTorch | 1 | 1.21 |  |  |
|  | Diffusers | 1 |  |  |  |
| **H200 141GB HBM3** | PyTorch | 1 | 1.19 |  |  |
|  | Diffusers | 1 |  |  |  |
| **B200** | PyTorch | 1 | 0.84 |  |  |
|  | Diffusers | 1 |  |  |  |
| **B300** | PyTorch | 1 | 0.87 |  |  |
|  | Diffusers | 1 |  |  |  |
| **GB300** | PyTorch | 1 |  |  |  |
|  | Diffusers | 1 |  |  | 6.08 |
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
| **H100 80GB HBM3** | PyTorch | 1 |  | 0.84 | 0.85 |
|  | Diffusers | 1 |  |  |  |
| **H200 141GB HBM3** | PyTorch | 1 |  | 0.82 | 0.85 |
|  | Diffusers | 1 |  |  |  |
| **B200** | PyTorch | 1 |  |  |  |
|  | Diffusers | 1 |  |  |  |
| **B300** | PyTorch | 1 |  |  |  |
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
