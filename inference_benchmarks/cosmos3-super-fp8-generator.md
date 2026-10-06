# Cosmos3-Super-FP8 Generator Inference Benchmarks

[Back to the inference benchmark index](../inference_benchmarks.md)

These tables report **Cosmos3-Super-FP8** Generator latency in seconds, for the FP8 quantized Cosmos3-Super checkpoint. Lower is better. The BF16 numbers for the same model are on the [Cosmos3-Super page](cosmos3-super-generator.md).

Every row reports one GPU configuration, named in the **GPUs** column, and all three resolutions in that row come from it. The configuration is one GPU when the model fits on a single device at its highest resolution, 720p, and eight GPUs when it does not. Within one GPU's block every engine reports the same configuration, so a column can be read straight across the runtimes. Cosmos3-Super-FP8 fits on a single GPU except on H100 80GB HBM3, where the 720p runs need eight.

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
- [Transfer generation](#transfer-generation)
- [Action generation](#action-generation)
  - [Forward Dynamics — Autonomous Vehicle (AV)](#forward-dynamics--autonomous-vehicle-av)
  - [Forward Dynamics — Camera](#forward-dynamics--camera)
  - [Forward Dynamics — Robot](#forward-dynamics--robot)
  - [Inverse Dynamics — Autonomous Vehicle (AV)](#inverse-dynamics--autonomous-vehicle-av)
  - [Inverse Dynamics — Robot](#inverse-dynamics--robot)
  - [Policy — Autonomous Vehicle (AV)](#policy--autonomous-vehicle-av)
  - [Policy — Robot](#policy--robot)

## Benchmark methodology

PyTorch values are from PBR `#309161`, **FP8 Cosmos3-Nano/Super (regular non-distilled) Generator PyTorch Inference Benchmarking**, reporting average generation latency over datacenter configurations at 1, 4, and 8 GPUs. Diffusers values are from PBR `#309434`, **FP8 Cosmos3-Nano/Super Generator Diffusers Inference Benchmarking**, measured on a single GPU only and at 480p and 720p only, which is why the 256p column is empty for Diffusers and why its rows are empty wherever a block reports eight GPUs. Neither OSS campaign covered H100 NVL or H200 NVL; those GPUs have NIM results only in the primary tables. GB300 appears only in the Diffusers campaign. vLLM-Omni and TensorRT-LLM have no FP8 results here. Values are rounded to two decimal places.

[NIM](../cookbooks/cosmos3/nim/README.md) results cover t2v, i2v, and t2i. NIM reports `Avg. Generation Time (s)` using FP8 latency profiles with offload disabled, excluding request overhead and MP4 encoding. Runs use concurrency 1, three measured requests, and 189 video frames or one image frame. The same measurements are also shown on the [Super page](cosmos3-super-generator.md), which includes [additional GPU counts and profiles](cosmos3-super-generator.md#additional-nim-configurations). Precision alone does not establish identical checkpoints or benchmark settings across backends.

These reports establish the reported timing matrix but do not expose every prompt and action payload in this repository. The linked public recipes explain modality behavior and provide representative payloads; their example-specific frame counts and action chunk sizes should not be treated as the exact internal benchmark inputs.

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

The PBR uses `t2av`, `v2av`, and `i2av`; some public recipes call the same sound-producing modes `t2vs`, `v2vs`, and `i2vs`. See the [audiovisual cookbook](../cookbooks/cosmos3/generator/audiovisual/README.md) for generation inputs and the [action cookbook](../cookbooks/cosmos3/generator/action/README.md) for action representations and output contracts.

## Primary vision generation

### Text-to-Video (t2v)

| GPU | Engine | GPUs | 256p | 480p | 720p |
|---|---|:-:|---:|---:|---:|
| **RTX PRO 6000 Blackwell** | PyTorch | 1 | 15.27 | 154.79 | 589.57 |
|  | Diffusers | 1 |  | 386.15 |  |
|  | NIM | 1 | 26.91 | 291.97 | 1152.93 |
| **H20** | PyTorch | 1 | 37.03 | 381.98 | 1505.14 |
|  | Diffusers | 1 |  |  |  |
|  | NIM | 1 | 67.93 | 711.10 | 2815.02 |
| **H100 80GB HBM3** | PyTorch | 1 | 9.15 | 83.30 |  |
|  | Diffusers | 1 |  | 234.65 | 830.32 |
|  | NIM | 1 |  |  |  |
| **H200 141GB HBM3** | PyTorch | 1 | 9.00 | 85.98 | 322.54 |
|  | Diffusers | 1 |  | 231.20 | 825.50 |
|  | NIM | 1 | 17.51 | 170.11 | 639.87 |
| **B200** | PyTorch | 1 | 5.25 | 45.67 | 168.91 |
|  | Diffusers | 1 |  | 130.78 | 405.87 |
|  | NIM | 1 | 9.39 | 94.04 | 346.99 |
| **B300** | PyTorch | 1 | 4.92 | 42.14 | 149.55 |
|  | Diffusers | 1 |  | 120.69 | 381.36 |
|  | NIM | 1 | 9.29 | 88.67 | 322.94 |
| **GB300** | PyTorch | 1 |  |  |  |
|  | Diffusers | 1 |  | 109.65 | 325.76 |
|  | NIM | 1 |  |  |  |
| **H100 NVL** | NIM | 1 | 24.54 | 267.92 | 937.49 |
| **H200 NVL** | NIM | 1 | 19.32 | 222.13 | 840.59 |

### Image-to-Video (i2v)

| GPU | Engine | GPUs | 256p | 480p | 720p |
|---|---|:-:|---:|---:|---:|
| **RTX PRO 6000 Blackwell** | PyTorch | 1 | 15.21 | 154.24 | 587.19 |
|  | Diffusers | 1 |  |  |  |
|  | NIM | 1 | 26.96 | 292.49 | 1157.15 |
| **H20** | PyTorch | 1 | 37.09 | 380.01 | 1501.45 |
|  | Diffusers | 1 |  |  |  |
|  | NIM | 1 | 68.26 | 712.78 | 2815.06 |
| **H100 80GB HBM3** | PyTorch | 8 | 5.75 | 19.08 | 55.33 |
|  | Diffusers | 8 |  |  |  |
|  | NIM | 8 |  |  |  |
| **H200 141GB HBM3** | PyTorch | 1 | 9.02 | 85.61 | 318.65 |
|  | Diffusers | 1 |  | 234.20 | 819.74 |
|  | NIM | 1 | 17.53 | 170.44 | 639.64 |
| **B200** | PyTorch | 1 | 5.18 | 45.24 | 167.93 |
|  | Diffusers | 1 |  | 129.97 | 408.14 |
|  | NIM | 1 | 9.42 | 92.08 | 340.06 |
| **B300** | PyTorch | 1 | 4.96 | 41.10 | 151.74 |
|  | Diffusers | 1 |  | 120.90 |  |
|  | NIM | 1 | 9.27 | 90.00 | 318.53 |
| **GB300** | PyTorch | 1 |  |  |  |
|  | Diffusers | 1 |  | 108.89 | 327.89 |
|  | NIM | 1 |  |  |  |
| **H100 NVL** | NIM | 1 | 24.59 | 266.77 | 939.10 |
| **H200 NVL** | NIM | 1 | 19.42 | 222.95 | 840.66 |

### Text-to-Image (t2i)

| GPU | Engine | GPUs | 256p | 480p | 720p |
|---|---|:-:|---:|---:|---:|
| **RTX PRO 6000 Blackwell** | PyTorch | 1 | 1.75 | 2.76 | 5.13 |
|  | Diffusers | 1 |  |  |  |
|  | NIM | 1 | 2.04 | 4.02 | 7.37 |
| **H20** | PyTorch | 1 | 2.33 | 5.73 | 12.01 |
|  | Diffusers | 1 |  |  |  |
|  | NIM | 1 | 2.22 | 10.18 | 18.16 |
| **H100 80GB HBM3** | PyTorch | 1 | 1.84 | 1.92 | 2.94 |
|  | Diffusers | 1 |  | 21.47 | 24.36 |
|  | NIM | 1 |  |  |  |
| **H200 141GB HBM3** | PyTorch | 1 | 1.88 | 1.85 | 2.98 |
|  | Diffusers | 1 |  | 19.90 | 23.20 |
|  | NIM | 1 | 1.98 | 3.08 | 4.79 |
| **B200** | PyTorch | 1 | 1.48 | 1.47 | 1.85 |
|  | Diffusers | 1 |  | 17.38 | 19.21 |
|  | NIM | 1 | 1.68 | 1.73 | 2.68 |
| **B300** | PyTorch | 1 | 2.37 | 2.29 | 2.47 |
|  | Diffusers | 1 |  |  | 18.40 |
|  | NIM | 1 | 2.56 | 2.55 | 2.80 |
| **GB300** | PyTorch | 1 |  |  |  |
|  | Diffusers | 1 |  | 22.73 | 20.98 |
|  | NIM | 1 |  |  |  |
| **H100 NVL** | NIM | 1 | 1.91 | 3.73 | 6.50 |
| **H200 NVL** | NIM | 1 | 1.93 | 3.25 | 5.15 |

<sub>Notes:
1. OSS times measured on identical workloads (same seed, sampler settings, prompt).
2. Multi-GPU configurations use tensor parallelism.
3. Values are average generation latency in seconds; lower is better.
4. The **GPUs** column gives the number of GPUs behind every value in that row, and is the same for every engine in a GPU block so the rows can be compared directly.
5. Diffusers FP8 was measured on one GPU at 480p and 720p only.
6. FP8 is faster than BF16 for the PyTorch path throughout. For the Diffusers path it only pays off on long video workloads: short ones are slower in FP8, `t2i` consistently so by roughly a factor of two, which suggests a fixed quantization overhead that the short runs cannot amortize. Compare against the BF16 page before choosing a precision for a short workload.</sub>

## Additional audiovisual generation

### Video-to-Video (v2v)

| GPU | Engine | GPUs | 256p | 480p | 720p |
|---|---|:-:|---:|---:|---:|
| **RTX PRO 6000 Blackwell** | PyTorch | 1 | 14.60 | 159.65 | 611.60 |
|  | Diffusers | 1 |  |  |  |
| **H20** | PyTorch | 1 | 35.78 | 396.97 | 1551.74 |
|  | Diffusers | 1 |  |  |  |
| **H100 80GB HBM3** | PyTorch | 8 | 5.79 | 19.97 | 57.83 |
|  | Diffusers | 8 |  |  |  |
| **H200 141GB HBM3** | PyTorch | 1 | 8.74 | 89.08 | 334.16 |
|  | Diffusers | 1 |  | 233.86 | 829.09 |
| **B200** | PyTorch | 1 | 4.95 | 46.10 | 174.67 |
|  | Diffusers | 1 |  | 132.66 | 410.66 |
| **B300** | PyTorch | 1 | 4.74 | 43.03 | 157.26 |
|  | Diffusers | 1 |  | 123.34 | 384.10 |
| **GB300** | PyTorch | 1 |  |  |  |
|  | Diffusers | 1 |  | 109.88 | 325.10 |

### Text-to-Audio-and-Video (t2av)

| GPU | Engine | GPUs | 256p | 480p | 720p |
|---|---|:-:|---:|---:|---:|
| **RTX PRO 6000 Blackwell** | PyTorch | 1 | 26.29 | 167.44 | 607.06 |
|  | Diffusers | 1 |  |  |  |
| **H20** | PyTorch | 1 | 61.37 | 414.65 | 1539.41 |
|  | Diffusers | 1 |  |  |  |
| **H100 80GB HBM3** | PyTorch | 8 | 6.27 | 20.73 | 58.41 |
|  | Diffusers | 8 |  |  |  |
| **H200 141GB HBM3** | PyTorch | 1 | 15.05 | 92.13 | 335.54 |
|  | Diffusers | 1 |  | 237.46 | 821.20 |
| **B200** | PyTorch | 1 | 8.82 | 49.69 | 173.99 |
|  | Diffusers | 1 |  | 131.90 | 410.84 |
| **B300** | PyTorch | 1 | 8.56 | 44.96 | 157.74 |
|  | Diffusers | 1 |  |  | 382.21 |
| **GB300** | PyTorch | 1 |  |  |  |
|  | Diffusers | 1 |  | 109.41 | 320.27 |

### Video-to-Audio-and-Video (v2av)

| GPU | Engine | GPUs | 256p | 480p | 720p |
|---|---|:-:|---:|---:|---:|
| **RTX PRO 6000 Blackwell** | PyTorch | — |  |  |  |
|  | Diffusers | — |  |  |  |
| **H20** | PyTorch | — |  |  |  |
|  | Diffusers | — |  |  |  |
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
|  | Diffusers | 1 |  |  |  |
| **H100 80GB HBM3** | PyTorch | 8 | 6.08 | 19.64 | 55.85 |
|  | Diffusers | 8 |  |  |  |
| **H200 141GB HBM3** | PyTorch | 1 | 14.59 | 90.77 | 330.04 |
|  | Diffusers | 1 |  | 238.95 | 831.19 |
| **B200** | PyTorch | 1 | 8.52 | 48.68 | 172.03 |
|  | Diffusers | 1 |  | 132.97 | 418.44 |
| **B300** | PyTorch | 1 | 8.30 | 44.71 | 155.68 |
|  | Diffusers | 1 |  | 122.66 | 384.16 |
| **GB300** | PyTorch | 1 |  |  |  |
|  | Diffusers | 1 |  | 112.33 | 324.87 |

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
|  |  | Diffusers | 1 |  |  |  |
|  | depth | PyTorch | 1 | 125.40 | 888.61 | 3434.99 |
|  |  | Diffusers | 1 |  |  |  |
|  | edge | PyTorch | 1 | 107.20 | 867.13 | 3423.93 |
|  |  | Diffusers | 1 |  |  |  |
|  | multi_control | PyTorch | 1 | 129.52 | 1399.05 | 5934.07 |
|  |  | Diffusers | 1 |  |  |  |
|  | seg | PyTorch | 1 | 117.22 | 912.61 | 3473.61 |
|  |  | Diffusers | 1 |  |  |  |
|  | wsm | PyTorch | 1 | 100.62 | 677.55 | 2530.15 |
|  |  | Diffusers | 1 |  |  |  |
| **H100 80GB HBM3** | blur | PyTorch | 8 | 25.70 | 74.88 | 211.62 |
|  |  | Diffusers | 8 |  |  |  |
|  | depth | PyTorch | 8 | 24.13 | 76.10 | 219.20 |
|  |  | Diffusers | 8 |  |  |  |
|  | edge | PyTorch | 8 | 23.24 | 73.96 | 216.16 |
|  |  | Diffusers | 8 |  |  |  |
|  | multi_control | PyTorch | 8 | 26.52 | 110.00 | 380.70 |
|  |  | Diffusers | 8 |  |  |  |
|  | seg | PyTorch | 8 | 27.41 | 83.47 | 236.85 |
|  |  | Diffusers | 8 |  |  |  |
|  | wsm | PyTorch | 1 | 25.48 | 149.06 |  |
|  |  | Diffusers | 1 |  | 289.78 | 923.57 |
| **H200 141GB HBM3** | blur | PyTorch | 1 | 32.51 | 192.22 | 727.48 |
|  |  | Diffusers | 1 |  | 603.23 | 2115.64 |
|  | depth | PyTorch | 1 | 30.58 | 200.84 | 741.97 |
|  |  | Diffusers | 1 |  | 621.13 | 2137.67 |
|  | edge | PyTorch | 1 | 26.71 | 195.05 | 730.81 |
|  |  | Diffusers | 1 |  | 598.72 | 2100.45 |
|  | multi_control | PyTorch | 1 | 31.90 | 316.92 | 1289.45 |
|  |  | Diffusers | 1 |  |  |  |
|  | seg | PyTorch | 1 | 29.32 | 204.51 | 736.27 |
|  |  | Diffusers | 1 |  | 602.62 | 2122.42 |
|  | wsm | PyTorch | 1 | 25.08 | 155.47 | 542.31 |
|  |  | Diffusers | 1 |  | 285.48 | 923.68 |
| **B200** | blur | PyTorch | 1 | 19.26 | 103.45 | 374.36 |
|  |  | Diffusers | 1 |  | 326.36 | 1025.97 |
|  | depth | PyTorch | 1 | 18.14 | 107.49 | 385.52 |
|  |  | Diffusers | 1 |  | 334.83 | 1044.04 |
|  | edge | PyTorch | 1 | 15.76 | 103.47 | 374.81 |
|  |  | Diffusers | 1 |  | 327.94 | 1029.37 |
|  | multi_control | PyTorch | — |  |  |  |
|  |  | Diffusers | — |  |  |  |
|  | seg | PyTorch | 1 | 17.60 | 109.95 | 400.23 |
|  |  | Diffusers | 1 |  | 326.72 | 1029.82 |
|  | wsm | PyTorch | 1 | 15.36 | 82.54 | 285.69 |
|  |  | Diffusers | 1 |  | 161.39 | 469.48 |
| **B300** | blur | PyTorch | 1 | 18.27 | 94.98 | 331.00 |
|  |  | Diffusers | 1 |  | 301.73 | 956.39 |
|  | depth | PyTorch | 1 | 17.29 | 100.31 | 342.30 |
|  |  | Diffusers | 1 |  | 315.60 | 974.65 |
|  | edge | PyTorch | 1 | 15.61 | 95.30 | 343.94 |
|  |  | Diffusers | 1 |  | 303.64 | 956.04 |
|  | multi_control | PyTorch | — |  |  |  |
|  |  | Diffusers | — |  |  |  |
|  | seg | PyTorch | 1 | 18.14 | 102.72 | 358.43 |
|  |  | Diffusers | 1 |  | 304.87 | 961.79 |
|  | wsm | PyTorch | 1 | 15.66 | 76.04 | 258.05 |
|  |  | Diffusers | 1 |  | 150.02 | 430.41 |
| **GB300** | blur | PyTorch | 1 |  |  |  |
|  |  | Diffusers | 1 |  | 268.46 | 806.60 |
|  | depth | PyTorch | 1 |  |  |  |
|  |  | Diffusers | 1 |  | 271.61 | 822.16 |
|  | edge | PyTorch | 1 |  |  |  |
|  |  | Diffusers | 1 |  | 260.92 | 811.23 |
|  | multi_control | PyTorch | — |  |  |  |
|  |  | Diffusers | — |  |  |  |
|  | seg | PyTorch | 1 |  |  |  |
|  |  | Diffusers | 1 |  | 268.54 | 804.36 |
|  | wsm | PyTorch | 1 |  |  |  |
|  |  | Diffusers | 1 |  | 134.03 | 365.92 |

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
|  | Diffusers | 1 |  |  |  |
| **H100 80GB HBM3** | PyTorch | 1 | 9.82 | 9.80 | 9.83 |
|  | Diffusers | 1 |  | 33.43 | 33.83 |
| **H200 141GB HBM3** | PyTorch | 1 | 9.48 | 9.63 | 9.48 |
|  | Diffusers | 1 |  | 32.07 | 32.65 |
| **B200** | PyTorch | 1 | 5.42 | 5.43 | 5.42 |
|  | Diffusers | 1 |  | 21.96 | 22.05 |
| **B300** | PyTorch | 1 | 5.07 | 5.06 | 5.19 |
|  | Diffusers | 1 |  |  | 21.84 |
| **GB300** | PyTorch | 1 |  |  |  |
|  | Diffusers | 1 |  | 19.76 | 20.41 |

### Forward Dynamics — Camera

| GPU | Engine | GPUs | 256p | 480p | 720p |
|---|---|:-:|---:|---:|---:|
| **RTX PRO 6000 Blackwell** | PyTorch | 1 | 15.77 | 15.75 | 15.78 |
|  | Diffusers | 1 |  |  |  |
| **H20** | PyTorch | 1 | 38.35 | 37.96 | 37.97 |
|  | Diffusers | 1 |  |  |  |
| **H100 80GB HBM3** | PyTorch | 1 | 9.25 | 9.28 | 9.25 |
|  | Diffusers | 1 |  |  |  |
| **H200 141GB HBM3** | PyTorch | 1 | 8.99 | 8.99 | 8.99 |
|  | Diffusers | 1 |  |  |  |
| **B200** | PyTorch | 1 | 5.14 | 5.15 | 5.12 |
|  | Diffusers | 1 |  |  |  |
| **B300** | PyTorch | 1 | 4.82 | 4.86 | 4.80 |
|  | Diffusers | 1 |  |  |  |

### Forward Dynamics — Robot

| GPU | Engine | GPUs | 256p | 480p | 720p |
|---|---|:-:|---:|---:|---:|
| **RTX PRO 6000 Blackwell** | PyTorch | 1 | 3.58 | 3.58 | 3.58 |
|  | Diffusers | 1 |  | 17.42 |  |
| **H20** | PyTorch | 1 | 8.49 | 8.50 | 8.50 |
|  | Diffusers | 1 |  |  |  |
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
|  | Diffusers | 1 |  |  | 51.58 |
| **H20** | PyTorch | 1 | 68.74 | 68.70 | 68.72 |
|  | Diffusers | 1 |  |  |  |
| **H100 80GB HBM3** | PyTorch | 1 | 16.56 | 16.51 | 16.51 |
|  | Diffusers | 1 |  | 33.15 | 33.73 |
| **H200 141GB HBM3** | PyTorch | 1 | 15.96 | 15.97 | 15.95 |
|  | Diffusers | 1 |  | 31.93 | 32.37 |
| **B200** | PyTorch | 1 | 9.01 | 9.03 | 9.02 |
|  | Diffusers | 1 |  | 21.73 | 21.84 |
| **B300** | PyTorch | 1 | 8.51 | 8.49 | 8.51 |
|  | Diffusers | 1 |  | 20.85 | 20.99 |
| **GB300** | PyTorch | 1 |  |  |  |
|  | Diffusers | 1 |  | 19.50 | 20.02 |

### Inverse Dynamics — Robot

| GPU | Engine | GPUs | 256p | 480p | 720p |
|---|---|:-:|---:|---:|---:|
| **RTX PRO 6000 Blackwell** | PyTorch | 1 | 5.72 | 5.72 | 5.73 |
|  | Diffusers | 1 |  |  |  |
| **H20** | PyTorch | 1 | 13.84 | 13.73 | 13.71 |
|  | Diffusers | 1 |  |  |  |
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
