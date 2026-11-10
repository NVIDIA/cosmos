# Cosmos3-Super-FP8 Generator Inference Benchmarks

[Back to the inference benchmark index](../inference_benchmarks.md)

See [measurement settings and reproduction limits](benchmark_methodology.md) for runtime-specific timing, precision, and workload details.

These tables report **Cosmos3-Super-FP8** Generator latency in seconds, for the FP8 quantized Cosmos3-Super checkpoint. Lower is better. The BF16 numbers for the same model are on the [Cosmos3-Super page](cosmos3-super-generator.md).

Every row reports one GPU configuration, named in the **GPUs** column, and all three resolutions in that row come from it. The configuration is one GPU when the model fits on a single device at its highest resolution, 720p, and eight GPUs when it does not. The **GPUs** column is authoritative: engines can use different counts when their memory requirements differ. Super FP8 memory requirements vary by runtime and workload; retain the count printed in each row rather than assuming every engine fits on one device.

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
| **RTX PRO 6000 Blackwell** | PyTorch\* | 1 | 15.27 | 154.79 | 589.57 |
|  | Diffusers | 1 |  |  |  |
|  | NIM | 1 | 15.50 | 151.90 | 597.67 |
| **H20** | PyTorch\* | 1 | 37.03 | 381.98 | 1505.14 |
|  | Diffusers | 1 |  | 857.07 | 3127.03 |
|  | NIM | 1 |  |  |  |
|  | vLLM-Omni | 8 |  | 115.72 | 403.84 |
|  | SGLang | 8 |  | 248.39 |  |
| **H100 80GB HBM3** | PyTorch\* | 1 | 9.15 | 83.30 |  |
|  | Diffusers | 1 |  | 234.65 | 830.32 |
|  | NIM | 1 | 16.53 | 85.05 | 295.86 |
|  | vLLM-Omni | 8 |  | 40.31 | 111.00 |
|  | SGLang | 1 |  | 184.02 |  |
| **H200 141GB HBM3** | PyTorch\* | 1 | 9.00 | 85.98 | 322.54 |
|  | Diffusers | 1 |  | 231.20 | 825.50 |
|  | NIM | 1 | 9.75 | 83.09 | 298.16 |
|  | vLLM-Omni | 1 |  | 179.48 | 662.76 |
|  | SGLang | 1 |  | 177.51 | 662.13 |
| **B200** | PyTorch\* | 1 | 5.25 | 45.67 | 168.91 |
|  | Diffusers | 1 |  | 130.78 | 405.87 |
|  | NIM | 1 | 5.43 | 47.71 | 175.45 |
|  | vLLM-Omni | 1 |  | 80.83 | 300.82 |
|  | SGLang | 1 |  | 110.99 | 431.23 |
| **B300** | PyTorch\* | 1 | 4.92 | 42.14 | 149.55 |
|  | Diffusers | 1 |  | 120.69 | 381.36 |
|  | NIM | 1 | 5.42 | 45.96 | 164.10 |
|  | vLLM-Omni | 1 |  | 77.19 | 282.94 |
|  | SGLang | 1 |  | 89.71 | 313.16 |
| **GB300** | PyTorch\* | 1 |  |  |  |
|  | Diffusers | 1 |  | 109.65 | 325.76 |
|  | NIM | 1 | 5.38 | 38.64 | 139.55 |
|  | vLLM-Omni | 1 |  | 66.14 | 239.50 |
|  | SGLang | 1 |  | 84.53 | 288.04 |

### Image-to-Video (i2v)

| GPU | Engine | GPUs | 256p | 480p | 720p |
|---|---|:-:|---:|---:|---:|
| **RTX PRO 6000 Blackwell** | PyTorch\* | 1 | 15.21 | 154.24 | 587.19 |
|  | Diffusers | 1 |  |  |  |
|  | NIM | 1 | 14.34 | 151.67 | 595.21 |
| **H20** | PyTorch\* | 1 | 37.09 | 380.01 | 1501.45 |
|  | Diffusers | 1 |  | 861.80 | 3135.16 |
|  | NIM | 1 |  |  |  |
|  | vLLM-Omni | 8 |  | 120.74 | 409.69 |
|  | SGLang | 8 |  | 258.14 | 773.46 |
| **H100 80GB HBM3** | PyTorch\* | 8 | 5.75 | 19.08 | 55.33 |
|  | Diffusers | 8 |  | 57.11 | 173.46 |
|  | NIM | 8 |  |  |  |
|  | vLLM-Omni | 8 |  | 40.32 | 111.52 |
|  | SGLang | 8 |  | 26.78 | 93.40 |
| **H200 141GB HBM3** | PyTorch\* | 1 | 9.02 | 85.61 | 318.65 |
|  | Diffusers | 1 |  | 234.20 | 819.74 |
|  | NIM | 1 | 9.27 | 83.86 | 300.45 |
|  | vLLM-Omni | 1 |  | 184.31 | 669.31 |
|  | SGLang | 1 |  | 177.71 | 662.92 |
| **B200** | PyTorch\* | 1 | 5.18 | 45.24 | 167.93 |
|  | Diffusers | 1 |  | 129.97 | 408.14 |
|  | NIM | 1 | 5.18 | 47.54 | 180.21 |
|  | vLLM-Omni | 1 |  | 80.90 | 300.40 |
|  | SGLang | 1 |  | 112.78 | 426.17 |
| **B300** | PyTorch\* | 1 | 4.96 | 41.10 | 151.74 |
|  | Diffusers | 1 |  | 120.90 | 383.28 |
|  | NIM | 1 | 5.27 | 46.03 | 164.42 |
|  | vLLM-Omni | 1 |  | 77.21 | 282.37 |
|  | SGLang | 1 |  | 90.04 | 313.72 |
| **GB300** | PyTorch\* | 1 |  |  |  |
|  | Diffusers | 1 |  | 108.89 | 327.89 |
|  | NIM | 1 | 4.60 | 39.01 | 137.93 |
|  | vLLM-Omni | 1 |  | 65.71 | 241.39 |
|  | SGLang | 1 |  | 85.05 | 283.31 |

### Text-to-Image (t2i)

| GPU | Engine | GPUs | 256p | 480p | 720p |
|---|---|:-:|---:|---:|---:|
| **RTX PRO 6000 Blackwell** | PyTorch\* | 1 | 1.75 | 2.76 | 5.13 |
|  | Diffusers | 1 |  |  |  |
|  | NIM | 1 | 1.11 | 2.03 | 3.49 |
| **H20** | PyTorch\* | 1 | 2.33 | 5.73 | 12.01 |
|  | Diffusers | 1 |  | 38.60 | 53.04 |
|  | NIM | 1 |  |  |  |
|  | vLLM-Omni | 1 |  | 27.65 | 39.18 |
|  | SGLang | 8 |  | 4.89 | 9.04 |
| **H100 80GB HBM3** | PyTorch\* | 1 | 1.84 | 1.92 | 2.94 |
|  | Diffusers | 1 |  | 21.47 | 24.36 |
|  | NIM | 1 | 7.05 | 7.29 | 8.20 |
|  | vLLM-Omni | 8 |  | 16.10 | 15.72 |
|  | SGLang | 1 |  | 4.81 | 7.16 |
| **H200 141GB HBM3** | PyTorch\* | 1 | 1.88 | 1.85 | 2.98 |
|  | Diffusers | 1 |  | 19.90 | 23.20 |
|  | NIM | 1 | 1.68 | 1.74 | 2.53 |
|  | vLLM-Omni | 1 |  | 17.24 | 15.69 |
|  | SGLang | 1 |  | 4.59 | 6.60 |
| **B200** | PyTorch\* | 1 | 1.48 | 1.47 | 1.85 |
|  | Diffusers | 1 |  | 17.38 | 19.21 |
|  | NIM | 1 | 0.93 | 0.95 | 1.36 |
|  | vLLM-Omni | 1 |  | 9.92 | 9.56 |
|  | SGLang | 1 |  | 3.99 | 4.44 |
| **B300** | PyTorch\* | 1 | 2.37 | 2.29 | 2.47 |
|  | Diffusers | 1 |  | 16.78 | 18.40 |
|  | NIM | 1 | 1.39 | 1.44 | 1.49 |
|  | vLLM-Omni | 1 |  | 10.99 | 12.61 |
|  | SGLang | 1 |  | 3.57 | 5.46 |
| **GB300** | PyTorch\* | 1 |  |  |  |
|  | Diffusers | 1 |  | 22.73 | 20.98 |
|  | NIM | 1 | 1.84 | 2.93 | 2.82 |
|  | vLLM-Omni | 1 |  | 20.95 | 12.36 |
|  | SGLang | 1 |  | 2.91 | 3.05 |

<sub>Notes:
1. Measurements come from separate campaigns; identical prompt, seed, step budget, and timing boundaries are not established across engines. See the measurement settings before comparing rows.
2. Multi-GPU layouts use the recorded tensor, CFG, context, Ring, or Ulysses splits; GPU count alone does not specify the layout.
3. Values are average generation latency in seconds; lower is better.
4. The **GPUs** column gives the number of GPUs behind every value in that row; check it before comparing engines.
5. Diffusers FP8 was measured at 480p and 720p; the GPUs column distinguishes single- and multi-GPU results.
6. FP8 is faster than BF16 for the PyTorch path throughout. For the Diffusers path it only pays off on long video workloads: short ones are slower in FP8, `t2i` consistently so by roughly a factor of two, which suggests a fixed quantization overhead that the short runs cannot amortize. Compare against the BF16 page before choosing a precision for a short workload.
7. \* PyTorch values were measured with a different FP8 recipe than the other engines on this page, which run the first and last three denoising steps in W8A16 and the middle steps in W8A8. PyTorch results are therefore optimistic and cannot be compared directly with the other values.
8. NIM numbers are from NIM 2.1.0 (`nvcr.io/nim/nvidia/cosmos3:2.1.0`) and report average generation time, excluding request processing and output encoding; step budgets and other workload differences still limit comparisons with other backends.</sub>

## Additional audiovisual generation

### Video-to-Video (v2v)

| GPU | Engine | GPUs | 256p | 480p | 720p |
|---|---|:-:|---:|---:|---:|
| **RTX PRO 6000 Blackwell** | PyTorch | 1 | 14.60 | 159.65 | 611.60 |
|  | Diffusers | 1 |  |  |  |
| **H20** | PyTorch | 1 | 35.78 | 396.97 | 1551.74 |
|  | Diffusers | 1 |  | 867.44 | 3136.15 |
|  | vLLM-Omni | 1 |  | 180.68 |  |
|  | SGLang | 8 |  | 266.54 | 777.14 |
| **H100 80GB HBM3** | PyTorch | 8 | 5.79 | 19.97 | 57.83 |
|  | Diffusers | 8 |  | 59.43 | 176.82 |
|  | vLLM-Omni | 8 |  | 19.15 | 32.52 |
|  | SGLang | 8 |  | 32.01 | 99.04 |
| **H200 141GB HBM3** | PyTorch | 1 | 8.74 | 89.08 | 334.16 |
|  | Diffusers | 1 |  | 233.86 | 829.09 |
|  | vLLM-Omni | 1 |  | 49.19 | 131.24 |
|  | SGLang | 1 |  | 185.63 | 672.13 |
| **B200** | PyTorch | 1 | 4.95 | 46.10 | 174.67 |
|  | Diffusers | 1 |  | 132.66 | 410.66 |
|  | vLLM-Omni | 1 |  | 20.80 | 57.75 |
|  | SGLang | 1 |  | 115.34 | 437.29 |
| **B300** | PyTorch | 1 | 4.74 | 43.03 | 157.26 |
|  | Diffusers | 1 |  | 123.34 | 384.10 |
|  | vLLM-Omni | 1 |  | 20.11 | 55.19 |
|  | SGLang | 1 |  | 96.54 | 319.63 |
| **GB300** | PyTorch | 1 |  |  |  |
|  | Diffusers | 1 |  | 109.88 | 325.10 |
|  | vLLM-Omni | 1 |  | 17.62 | 47.55 |
|  | SGLang | 1 |  | 89.32 | 287.23 |
| **DGX Spark** | PyTorch | 1 | 127.70 | 1291.00 | 4886.80 |
| **DGX Station** | PyTorch | 1 | 7.30 | 66.40 | 242.80 |

### Text-to-Audio-and-Video (t2av)

| GPU | Engine | GPUs | 256p | 480p | 720p |
|---|---|:-:|---:|---:|---:|
| **RTX PRO 6000 Blackwell** | PyTorch | 1 | 26.29 | 167.44 | 607.06 |
|  | Diffusers | 1 |  |  |  |
| **H20** | PyTorch | 1 | 61.37 | 414.65 | 1539.41 |
|  | Diffusers | 1 |  | 871.38 | 3151.04 |
|  | vLLM-Omni | 1 |  | 760.68 |  |
|  | SGLang | 8 |  | 255.12 | 769.52 |
| **H100 80GB HBM3** | PyTorch | 8 | 6.27 | 20.73 | 58.41 |
|  | Diffusers | 8 |  | 57.22 | 174.67 |
|  | vLLM-Omni | 8 |  | 41.14 | 113.09 |
|  | SGLang | 8 |  | 26.91 | 93.46 |
| **H200 141GB HBM3** | PyTorch | 1 | 15.05 | 92.13 | 335.54 |
|  | Diffusers | 1 |  | 237.46 | 821.20 |
|  | vLLM-Omni | 1 |  | 185.16 | 676.32 |
|  | SGLang | 1 |  | 179.47 | 669.38 |
| **B200** | PyTorch | 1 | 8.82 | 49.69 | 173.99 |
|  | Diffusers | 1 |  | 131.90 | 410.84 |
|  | vLLM-Omni | 1 |  | 81.56 | 304.12 |
|  | SGLang | 1 |  | 114.42 | 429.47 |
| **B300** | PyTorch | 1 | 8.56 | 44.96 | 157.74 |
|  | Diffusers | 1 |  | 122.56 | 382.21 |
|  | vLLM-Omni | 1 |  | 78.09 | 286.12 |
|  | SGLang | 1 |  | 91.10 | 316.84 |
| **GB300** | PyTorch | 1 |  |  |  |
|  | Diffusers | 1 |  | 109.41 | 320.27 |
|  | vLLM-Omni | 1 |  | 67.77 | 243.13 |
|  | SGLang | 1 |  | 85.77 | 286.02 |

### Video-to-Audio-and-Video (v2av)

| GPU | Engine | GPUs | 256p | 480p | 720p |
|---|---|:-:|---:|---:|---:|
| **RTX PRO 6000 Blackwell** | PyTorch | — |  |  |  |
|  | Diffusers | — |  |  |  |
| **H20** | PyTorch | — |  |  |  |
|  | Diffusers | 1 |  | 877.63 | 3164.49 |
| **H100 80GB HBM3** | PyTorch | 1 |  |  |  |
|  | Diffusers | 1 |  | 240.63 |  |
| **H200 141GB HBM3** | PyTorch | 1 |  |  |  |
|  | Diffusers | 1 |  | 238.03 | 826.01 |
| **B200** | PyTorch | 1 |  |  |  |
|  | Diffusers | 1 |  | 133.87 | 413.17 |
| **B300** | PyTorch | 1 |  |  |  |
|  | Diffusers | 1 |  | 123.61 | 386.33 |
| **GB300** | PyTorch | 1 |  |  |  |
|  | Diffusers | 1 |  | 110.98 | 334.54 |

### Image-to-Audio-and-Video (i2av)

| GPU | Engine | GPUs | 256p | 480p | 720p |
|---|---|:-:|---:|---:|---:|
| **RTX PRO 6000 Blackwell** | PyTorch | 1 | 25.77 | 165.51 | 602.27 |
|  | Diffusers | 1 |  |  |  |
| **H20** | PyTorch | 1 | 59.59 | 408.98 | 1511.82 |
|  | Diffusers | 1 |  | 876.43 | 3162.37 |
|  | vLLM-Omni | 8 |  | 122.41 | 408.81 |
|  | SGLang | 8 |  | 274.41 | 774.99 |
| **H100 80GB HBM3** | PyTorch | 8 | 6.08 | 19.64 | 55.85 |
|  | Diffusers | 8 |  | 58.57 | 177.88 |
|  | vLLM-Omni | 8 |  | 41.40 | 112.61 |
|  | SGLang | 8 |  | 27.08 | 93.59 |
| **H200 141GB HBM3** | PyTorch | 1 | 14.59 | 90.77 | 330.04 |
|  | Diffusers | 1 |  | 238.95 | 831.19 |
|  | vLLM-Omni | 1 |  | 187.15 | 668.31 |
|  | SGLang | 1 |  | 179.45 | 669.74 |
| **B200** | PyTorch | 1 | 8.52 | 48.68 | 172.03 |
|  | Diffusers | 1 |  | 132.97 | 418.44 |
|  | vLLM-Omni | 1 |  | 81.45 | 303.50 |
|  | SGLang | 1 |  | 112.34 | 430.26 |
| **B300** | PyTorch | 1 | 8.30 | 44.71 | 155.68 |
|  | Diffusers | 1 |  | 122.66 | 384.16 |
|  | vLLM-Omni | 1 |  | 78.01 | 284.95 |
|  | SGLang | 1 |  | 91.25 | 316.78 |
| **GB300** | PyTorch | 1 |  |  |  |
|  | Diffusers | 1 |  | 112.33 | 324.87 |
|  | vLLM-Omni | 1 |  | 67.00 | 242.28 |
|  | SGLang | 1 |  | 86.73 | 287.81 |

## Transfer generation

Transfer conditions a video-to-video generation on a structural control hint
extracted from the source video. Each control is reported separately because
the hint changes how much of the frame the model must synthesise.

| GPU | Transfer control | Engine | GPUs | 256p | 480p | 720p |
|---|---|---|:-:|---:|---:|---:|
| **RTX PRO 6000 Blackwell** | blur | PyTorch | 1 | 56.31 | 348.18 | 1312.06 |
|  |  | Diffusers | 1 |  |  |  |
|  | depth | PyTorch | 1 | 53.67 | 360.23 | 1350.49 |
|  |  | Diffusers | 1 |  |  |  |
|  | edge | PyTorch | 1 | 45.73 | 347.50 | 1329.40 |
|  |  | Diffusers | 1 |  |  |  |
|  | multi_control | PyTorch | — |  |  |  |
|  |  | Diffusers | — |  |  |  |
|  | seg | PyTorch | 1 | 49.86 | 367.16 | 1402.75 |
|  |  | Diffusers | 1 |  |  |  |
|  | wsm | PyTorch | 1 | 43.38 | 275.08 | 999.98 |
|  |  | Diffusers | 1 |  |  |  |
| **H20** | blur | PyTorch | 1 | 132.90 | 857.75 | 3317.89 |
|  |  | Diffusers | 1 |  | 2247.21 | 8125.02 |
|  |  | vLLM-Omni | 8 |  | 457.20 | 1682.74 |
|  |  | SGLang | 8 |  | 1042.64 | 3066.56 |
|  | depth | PyTorch | 1 | 125.40 | 888.61 | 3434.99 |
|  |  | Diffusers | 1 |  | 2351.47 | 8276.35 |
|  |  | vLLM-Omni | 8 |  | 448.59 | 1719.67 |
|  |  | SGLang | 8 |  | 1030.18 | 3094.80 |
|  | edge | PyTorch | 1 | 107.20 | 867.13 | 3423.93 |
|  |  | Diffusers | 1 |  | 2259.33 | 8140.63 |
|  |  | vLLM-Omni | 8 |  | 450.65 | 1715.00 |
|  | multi_control | PyTorch | 1 | 129.52 | 1399.05 | 5934.07 |
|  |  | Diffusers | 1 |  |  |  |
|  |  | vLLM-Omni | 8 |  | 2751.63 | 11941.85 |
|  |  | SGLang | 8 |  | 1689.40 | 5848.36 |
|  | seg | PyTorch | 1 | 117.22 | 912.61 | 3473.61 |
|  |  | Diffusers | 1 |  | 2277.51 | 8167.97 |
|  |  | vLLM-Omni | 8 |  | 444.44 | 1709.93 |
|  |  | SGLang | 8 |  | 1044.19 | 3051.08 |
|  | wsm | PyTorch | 1 | 100.62 | 677.55 | 2530.15 |
|  |  | Diffusers | 1 |  | 1063.41 | 3534.60 |
|  |  | vLLM-Omni | 8 |  | 187.57 | 656.15 |
|  |  | SGLang | 8 |  | 421.87 | 1183.89 |
| **H100 80GB HBM3** | blur | PyTorch | 8 | 25.70 | 74.88 | 211.62 |
|  |  | Diffusers | 8 |  | 128.84 | 388.64 |
|  |  | vLLM-Omni | 8 |  | 150.24 | 427.94 |
|  |  | SGLang | 8 |  | 107.61 | 395.42 |
|  | depth | PyTorch | 8 | 24.13 | 76.10 | 219.20 |
|  |  | Diffusers | 8 |  | 131.38 | 388.54 |
|  |  | vLLM-Omni | 8 |  | 132.75 | 434.82 |
|  |  | SGLang | 8 |  | 108.29 | 397.52 |
|  | edge | PyTorch | 8 | 23.24 | 73.96 | 216.16 |
|  |  | Diffusers | 8 |  | 121.83 | 344.05 |
|  |  | vLLM-Omni | 8 |  | 125.40 | 432.44 |
|  |  | SGLang | 8 |  | 106.62 | 391.44 |
|  | multi_control | PyTorch | 8 | 26.52 | 110.00 | 380.70 |
|  |  | Diffusers | 8 |  |  |  |
|  |  | vLLM-Omni | 8 |  | 685.22 | 3124.26 |
|  |  | SGLang | 8 |  | 194.63 | 760.03 |
|  | seg | PyTorch | 8 | 27.41 | 83.47 | 236.85 |
|  |  | Diffusers | 8 |  | 128.81 | 389.87 |
|  |  | vLLM-Omni | 8 |  | 126.98 | 427.91 |
|  |  | SGLang | 8 |  | 109.71 | 393.68 |
|  | wsm | PyTorch | 1 | 25.48 | 149.06 |  |
|  |  | Diffusers | 1 |  | 289.78 | 923.57 |
|  |  | vLLM-Omni | 8 |  | 64.83 | 173.55 |
|  |  | SGLang | 1 |  | 202.39 |  |
| **H200 141GB HBM3** | blur | PyTorch | 1 | 32.51 | 192.22 | 727.48 |
|  |  | Diffusers | 1 |  | 603.23 | 2115.64 |
|  |  | vLLM-Omni | 1 |  | 475.52 | 1688.94 |
|  |  | SGLang | 1 |  | 433.26 | 1633.84 |
|  | depth | PyTorch | 1 | 30.58 | 200.84 | 741.97 |
|  |  | Diffusers | 1 |  | 621.13 | 2137.67 |
|  |  | vLLM-Omni | 1 |  | 470.09 | 1734.87 |
|  |  | SGLang | 1 |  | 434.56 | 1672.19 |
|  | edge | PyTorch | 1 | 26.71 | 195.05 | 730.81 |
|  |  | Diffusers | 1 |  | 598.72 | 2100.45 |
|  |  | vLLM-Omni | 1 |  | 455.75 | 1713.62 |
|  |  | SGLang | 1 |  | 434.24 | 1654.36 |
|  | multi_control | PyTorch | 1 | 31.90 | 316.92 | 1289.45 |
|  |  | Diffusers | 1 |  |  |  |
|  |  | vLLM-Omni | 1 |  | 1305.86 | 5518.02 |
|  |  | SGLang | 1 |  | 757.62 | 3143.94 |
|  | seg | PyTorch | 1 | 29.32 | 204.51 | 736.27 |
|  |  | Diffusers | 1 |  | 602.62 | 2122.42 |
|  |  | vLLM-Omni | 1 |  | 472.94 | 1728.71 |
|  |  | SGLang | 1 |  | 429.20 | 1662.72 |
|  | wsm | PyTorch | 1 | 25.08 | 155.47 | 542.31 |
|  |  | Diffusers | 1 |  | 285.48 | 923.68 |
|  |  | vLLM-Omni | 1 |  | 217.05 | 752.29 |
|  |  | SGLang | 1 |  | 195.57 | 720.40 |
| **B200** | blur | PyTorch | 1 | 19.26 | 103.45 | 374.36 |
|  |  | Diffusers | 1 |  | 326.36 | 1025.97 |
|  |  | vLLM-Omni | 1 |  | 209.45 | 784.33 |
|  |  | SGLang | 1 |  | 272.79 | 1078.48 |
|  | depth | PyTorch | 1 | 18.14 | 107.49 | 385.52 |
|  |  | Diffusers | 1 |  | 334.83 | 1044.04 |
|  |  | vLLM-Omni | 1 |  | 208.92 | 792.97 |
|  |  | SGLang | 1 |  | 275.02 | 1114.32 |
|  | edge | PyTorch | 1 | 15.76 | 103.47 | 374.81 |
|  |  | Diffusers | 1 |  | 327.94 | 1029.37 |
|  |  | vLLM-Omni | 1 |  | 205.02 | 786.53 |
|  |  | SGLang | 1 |  | 273.17 | 1086.87 |
|  | multi_control | PyTorch | — |  |  |  |
|  |  | Diffusers | — |  |  |  |
|  |  | vLLM-Omni | 1 |  | 598.61 | 2545.35 |
|  |  | SGLang | 1 |  | 470.03 | 2076.88 |
|  | seg | PyTorch | 1 | 17.60 | 109.95 | 400.23 |
|  |  | Diffusers | 1 |  | 326.72 | 1029.82 |
|  |  | vLLM-Omni | 1 |  | 206.12 | 786.58 |
|  |  | SGLang | 1 |  | 269.78 | 1091.80 |
|  | wsm | PyTorch | 1 | 15.36 | 82.54 | 285.69 |
|  |  | Diffusers | 1 |  | 161.39 | 469.48 |
|  |  | vLLM-Omni | 1 |  | 94.77 | 334.73 |
|  |  | SGLang | 1 |  | 122.90 | 459.00 |
| **B300** | blur | PyTorch | 1 | 18.27 | 94.98 | 331.00 |
|  |  | Diffusers | 1 |  | 301.73 | 956.39 |
|  |  | vLLM-Omni | 1 |  | 197.58 | 728.72 |
|  |  | SGLang | 1 |  | 213.84 | 775.33 |
|  | depth | PyTorch | 1 | 17.29 | 100.31 | 342.30 |
|  |  | Diffusers | 1 |  | 315.60 | 974.65 |
|  |  | vLLM-Omni | 1 |  | 200.14 | 750.46 |
|  |  | SGLang | 1 |  | 217.86 | 782.96 |
|  | edge | PyTorch | 1 | 15.61 | 95.30 | 343.94 |
|  |  | Diffusers | 1 |  | 303.64 | 956.04 |
|  |  | vLLM-Omni | 1 |  | 197.63 | 740.53 |
|  |  | SGLang | 1 |  | 213.45 | 775.70 |
|  | multi_control | PyTorch | — |  |  |  |
|  |  | Diffusers | — |  |  |  |
|  |  | vLLM-Omni | 1 |  | 576.47 | 2352.55 |
|  |  | SGLang | 1 |  | 358.25 | 1417.59 |
|  | seg | PyTorch | 1 | 18.14 | 102.72 | 358.43 |
|  |  | Diffusers | 1 |  | 304.87 | 961.79 |
|  |  | vLLM-Omni | 1 |  | 197.63 | 741.35 |
|  |  | SGLang | 1 |  | 215.38 | 781.10 |
|  | wsm | PyTorch | 1 | 15.66 | 76.04 | 258.05 |
|  |  | Diffusers | 1 |  | 150.02 | 430.41 |
|  |  | vLLM-Omni | 1 |  | 90.83 | 316.34 |
|  |  | SGLang | 1 |  | 97.90 | 342.83 |
| **GB300** | blur | PyTorch | 1 |  |  |  |
|  |  | Diffusers | 1 |  | 268.46 | 806.60 |
|  |  | vLLM-Omni | 1 |  | 166.65 | 619.52 |
|  |  | SGLang | 1 |  | 200.79 | 700.28 |
|  | depth | PyTorch | 1 |  |  |  |
|  |  | Diffusers | 1 |  | 271.61 | 822.16 |
|  |  | vLLM-Omni | 1 |  | 169.84 | 635.19 |
|  |  | SGLang | 1 |  | 200.97 | 715.86 |
|  | edge | PyTorch | 1 |  |  |  |
|  |  | Diffusers | 1 |  | 260.92 | 811.23 |
|  |  | vLLM-Omni | 1 |  | 163.57 | 628.87 |
|  |  | SGLang | 1 |  | 200.28 | 692.75 |
|  | multi_control | PyTorch | — |  |  |  |
|  |  | Diffusers | — |  |  |  |
|  |  | vLLM-Omni | 1 |  | 480.61 | 2006.45 |
|  |  | SGLang | 1 |  | 333.49 | 1257.74 |
|  | seg | PyTorch | 1 |  |  |  |
|  |  | Diffusers | 1 |  | 268.54 | 804.36 |
|  |  | vLLM-Omni | 1 |  | 166.96 | 624.94 |
|  |  | SGLang | 1 |  | 202.57 | 693.74 |
|  | wsm | PyTorch | 1 |  |  |  |
|  |  | Diffusers | 1 |  | 134.03 | 365.92 |
|  |  | vLLM-Omni | 1 |  | 77.12 | 268.45 |
|  |  | SGLang | 1 |  | 94.33 | 310.54 |
| **DGX Spark** | edge | PyTorch | 1 | 445.80 | 3247.40 | 12642.00 |
| **DGX Station** | edge | PyTorch | 1 | 26.00 | 169.60 | 625.50 |

<sub>Transfer notes:
1. Control hints follow the `extra_params` names: `blur`, `depth`, `edge`, `multi_control`, `seg`, and `wsm`. `multi_control` combines several hints in one request.
2. The PyTorch campaign reports `segmentation`, which is the same control as `seg` in the Diffusers campaign.
3. Diffusers FP8 has no `multi_control` measurements.</sub>

## Action generation

### Forward Dynamics — Autonomous Vehicle (AV)

| GPU | Engine | GPUs | 256p | 480p | 720p |
|---|---|:-:|---:|---:|---:|
| **RTX PRO 6000 Blackwell** | PyTorch | 1 | 16.71 | 16.73 | 16.72 |
|  | Diffusers | 1 |  |  |  |
| **H20** | PyTorch | 1 | 40.31 | 40.29 | 40.28 |
|  | Diffusers | 1 |  | 105.28 | 107.86 |
|  | SGLang | 8 |  |  | 170.99 |
| **H100 80GB HBM3** | PyTorch | 1 | 9.82 | 9.80 | 9.83 |
|  | Diffusers | 1 |  | 33.43 | 33.83 |
|  | SGLang | 1 |  | 19.28 | 57.03 |
| **H200 141GB HBM3** | PyTorch | 1 | 9.48 | 9.63 | 9.48 |
|  | Diffusers | 1 |  | 32.07 | 32.65 |
|  | SGLang | 1 |  | 18.58 | 54.29 |
| **B200** | PyTorch | 1 | 5.42 | 5.43 | 5.42 |
|  | Diffusers | 1 |  | 21.96 | 22.05 |
|  | SGLang | 1 |  | 11.11 | 32.95 |
| **B300** | PyTorch | 1 | 5.07 | 5.06 | 5.19 |
|  | Diffusers | 1 |  | 21.31 | 21.84 |
|  | SGLang | 1 |  | 9.82 | 27.21 |
| **GB300** | PyTorch | 1 |  |  |  |
|  | Diffusers | 1 |  | 19.76 | 20.41 |
|  | SGLang | 1 |  | 9.55 | 26.28 |

### Forward Dynamics — Camera

| GPU | Engine | GPUs | 256p | 480p | 720p |
|---|---|:-:|---:|---:|---:|
| **RTX PRO 6000 Blackwell** | PyTorch | 1 | 15.77 | 15.75 | 15.78 |
|  | Diffusers | 1 |  |  |  |
| **H20** | PyTorch | 1 | 38.35 | 37.96 | 37.97 |
|  | Diffusers | 1 |  |  |  |
|  | SGLang | 8 |  | 64.03 | 168.06 |
| **H100 80GB HBM3** | PyTorch | 1 | 9.25 | 9.28 | 9.25 |
|  | Diffusers | 1 |  |  |  |
|  | SGLang | 1 |  | 19.36 | 57.20 |
| **H200 141GB HBM3** | PyTorch | 1 | 8.99 | 8.99 | 8.99 |
|  | Diffusers | 1 |  |  |  |
|  | SGLang | 1 |  | 18.61 | 53.82 |
| **B200** | PyTorch | 1 | 5.14 | 5.15 | 5.12 |
|  | Diffusers | 1 |  |  |  |
|  | SGLang | 1 |  | 11.17 | 32.33 |
| **B300** | PyTorch | 1 | 4.82 | 4.86 | 4.80 |
|  | Diffusers | 1 |  |  |  |
|  | SGLang | 1 |  | 9.84 | 27.27 |
| **GB300** | SGLang | 1 |  | 9.53 | 25.82 |

### Forward Dynamics — Robot

| GPU | Engine | GPUs | 256p | 480p | 720p |
|---|---|:-:|---:|---:|---:|
| **RTX PRO 6000 Blackwell** | PyTorch | 1 | 3.58 | 3.58 | 3.58 |
|  | Diffusers | 1 |  |  |  |
| **H20** | PyTorch | 1 | 8.49 | 8.50 | 8.50 |
|  | Diffusers | 1 |  | 32.34 | 32.93 |
| **H100 80GB HBM3** | PyTorch | 1 | 2.14 | 2.14 | 2.14 |
|  | Diffusers | 1 |  | 13.90 | 14.04 |
| **H200 141GB HBM3** | PyTorch | 1 | 2.09 | 2.10 | 2.10 |
|  | Diffusers | 1 |  | 13.20 | 13.35 |
| **B200** | PyTorch | 1 | 1.27 | 1.26 | 1.28 |
|  | Diffusers | 1 |  | 10.60 | 10.71 |
| **B300** | PyTorch | 1 | 1.30 | 1.32 | 1.31 |
|  | Diffusers | 1 |  | 10.24 | 11.15 |
| **GB300** | PyTorch | 1 |  |  |  |
|  | Diffusers | 1 |  | 10.71 | 12.12 |

### Inverse Dynamics — Autonomous Vehicle (AV)

| GPU | Engine | GPUs | 256p | 480p | 720p |
|---|---|:-:|---:|---:|---:|
| **RTX PRO 6000 Blackwell** | PyTorch | 1 | 28.37 | 28.36 | 28.29 |
|  | Diffusers | 1 |  |  |  |
| **H20** | PyTorch | 1 | 68.74 | 68.70 | 68.72 |
|  | Diffusers | 1 |  | 105.06 | 106.81 |
|  | SGLang | 8 |  | 59.13 | 173.53 |
| **H100 80GB HBM3** | PyTorch | 1 | 16.56 | 16.51 | 16.51 |
|  | Diffusers | 1 |  | 33.15 | 33.73 |
|  | SGLang | 1 |  | 19.03 | 54.54 |
| **H200 141GB HBM3** | PyTorch | 1 | 15.96 | 15.97 | 15.95 |
|  | Diffusers | 1 |  | 31.93 | 32.37 |
|  | SGLang | 1 |  | 18.49 | 53.75 |
| **B200** | PyTorch | 1 | 9.01 | 9.03 | 9.02 |
|  | Diffusers | 1 |  | 21.73 | 21.84 |
|  | SGLang | 1 |  | 11.10 | 33.13 |
| **B300** | PyTorch | 1 | 8.51 | 8.49 | 8.51 |
|  | Diffusers | 1 |  | 20.85 | 20.99 |
|  | SGLang | 1 |  | 9.93 | 28.32 |
| **GB300** | PyTorch | 1 |  |  |  |
|  | Diffusers | 1 |  | 19.50 | 20.02 |
|  | SGLang | 1 |  | 9.35 | 25.51 |

### Inverse Dynamics — Robot

| GPU | Engine | GPUs | 256p | 480p | 720p |
|---|---|:-:|---:|---:|---:|
| **RTX PRO 6000 Blackwell** | PyTorch | 1 | 5.72 | 5.72 | 5.73 |
|  | Diffusers | 1 |  |  |  |
| **H20** | PyTorch | 1 | 13.84 | 13.73 | 13.71 |
|  | Diffusers | 1 |  | 32.52 | 32.98 |
| **H100 80GB HBM3** | PyTorch | 1 | 3.39 | 3.39 | 3.39 |
|  | Diffusers | 1 |  | 13.91 | 14.02 |
| **H200 141GB HBM3** | PyTorch | 1 | 3.37 | 3.32 | 3.32 |
|  | Diffusers | 1 |  | 13.10 | 13.38 |
| **B200** | PyTorch | 1 | 1.97 | 1.97 | 1.96 |
|  | Diffusers | 1 |  | 10.57 | 10.70 |
| **B300** | PyTorch | 1 | 1.95 | 1.90 | 1.97 |
|  | Diffusers | 1 |  | 10.95 | 11.30 |
| **GB300** | PyTorch | 1 |  |  |  |
|  | Diffusers | 1 |  | 12.74 | 13.76 |

### Policy — Autonomous Vehicle (AV)

| GPU | Engine | GPUs | 256p | 480p | 720p |
|---|---|:-:|---:|---:|---:|
| **RTX PRO 6000 Blackwell** | PyTorch | 1 | 16.74 | 16.75 | 16.73 |
|  | Diffusers | 1 |  |  |  |
| **H20** | PyTorch | 1 | 40.58 | 40.29 | 40.59 |
|  | Diffusers | 1 |  |  |  |
| **H100 80GB HBM3** | PyTorch | 1 | 9.80 | 9.83 | 9.81 |
|  | Diffusers | 1 |  |  |  |
| **H200 141GB HBM3** | PyTorch | 1 | 9.62 | 9.61 | 9.62 |
|  | Diffusers | 1 |  |  |  |
| **B200** | PyTorch | 1 | 5.42 | 5.33 | 5.40 |
|  | Diffusers | 1 |  |  |  |
| **B300** | PyTorch | 1 | 5.20 | 5.18 | 5.20 |
|  | Diffusers | 1 |  |  |  |

### Policy — Robot

| GPU | Engine | GPUs | 256p | 480p | 720p |
|---|---|:-:|---:|---:|---:|
| **RTX PRO 6000 Blackwell** | PyTorch | 1 | 3.42 | 3.41 | 3.42 |
|  | Diffusers | 1 |  |  |  |
| **H20** | PyTorch | 1 | 8.17 | 8.11 | 8.10 |
|  | Diffusers | 1 |  |  |  |
| **H100 80GB HBM3** | PyTorch | 1 | 2.05 | 2.05 | 2.04 |
|  | Diffusers | 1 |  |  |  |
| **H200 141GB HBM3** | PyTorch | 1 | 2.04 | 2.00 | 2.04 |
|  | Diffusers | 1 |  |  |  |
| **B200** | PyTorch | 1 | 1.23 | 1.21 | 1.24 |
|  | Diffusers | 1 |  |  |  |
| **B300** | PyTorch | 1 | 1.19 | 1.18 | 1.18 |
|  | Diffusers | 1 |  |  |  |

<sub>Action and additional-modality notes:
1. Forward-dynamics cells for the AV domain are the mean of the forward, left, and right trajectories; inverse-dynamics cells are the mean of the two inverse trajectories.
2. The Diffusers FP8 campaign covers forward and inverse dynamics for the AV and Robot domains only, so the Camera and Policy rows carry PyTorch values alone.
3. Policy-DROID is a separate checkpoint and was not part of either FP8 campaign.</sub>

### Inverse Dynamics — Camera

| GPU | Engine | GPUs | 256p | 480p | 720p |
|---|---|:-:|---:|---:|---:|
| **B200** | SGLang | 1 |  | 11.88 | 32.00 |
| **B300** | SGLang | 1 |  | 11.31 | 27.37 |
| **GB300** | SGLang | 1 |  | 9.92 | 25.55 |
| **H100 80GB HBM3** | SGLang | 1 |  | 19.70 | 54.45 |
| **H20** | SGLang | 8 |  | 62.76 | 171.28 |
| **H200 141GB HBM3** | SGLang | 1 |  | 19.06 | 52.72 |
