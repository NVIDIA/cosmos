# Cosmos3-Nano-FP8 Generator Inference Benchmarks

[Back to the inference benchmark index](../inference_benchmarks.md)

These tables report **Cosmos3-Nano-FP8** Generator latency in seconds, for the FP8 quantized Cosmos3-Nano checkpoint. Lower is better. The BF16 numbers for the same model are on the [Cosmos3-Nano page](cosmos3-nano-generator.md).

Every row reports one GPU configuration, named in the **GPUs** column, and all three resolutions in that row come from it. The configuration is one GPU when the model fits on a single device at its highest resolution, 720p, and eight GPUs when it does not. Within one GPU's block every engine reports the same configuration, so a column can be read straight across the runtimes. All populated rows report one GPU.

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

[NIM](../cookbooks/cosmos3/nim/README.md) results cover t2v, i2v, and t2i. NIM reports `Avg. Generation Time (s)` using FP8 latency profiles with offload disabled, excluding request overhead and MP4 encoding. Runs use concurrency 1, three measured requests, and 189 video frames or one image frame. The same measurements are also shown on the [Nano page](cosmos3-nano-generator.md), which includes [additional GPU counts and profiles](cosmos3-nano-generator.md#additional-nim-configurations). Precision alone does not establish identical checkpoints or benchmark settings across backends.

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
| **RTX PRO 6000 Blackwell** | PyTorch | 1 | 4.71 | 47.34 | 173.02 |
|  | Diffusers | 1 |  |  |  |
|  | NIM | 1 | 7.24 | 81.35 | 313.63 |
| **H20** | PyTorch | 1 | 11.82 | 112.34 | 427.62 |
|  | Diffusers | 1 |  |  |  |
|  | NIM | 1 | 18.78 | 195.35 | 776.37 |
| **H100 80GB HBM3** | PyTorch | 1 | 3.12 | 26.01 | 94.01 |
|  | Diffusers | 1 |  | 69.55 | 237.29 |
|  | NIM | 1 | 5.35 | 46.08 | 173.20 |
| **H200 141GB HBM3** | PyTorch | 1 | 3.09 | 26.59 | 94.57 |
|  | Diffusers | 1 |  | 68.32 | 237.18 |
|  | NIM | 1 | 5.02 | 46.93 | 177.23 |
| **B200** | PyTorch | 1 | 1.97 | 14.85 | 51.85 |
|  | Diffusers | 1 |  | 39.70 | 124.10 |
|  | NIM | 1 | 2.94 | 25.03 | 91.62 |
| **B300** | PyTorch | 1 | 1.99 | 13.68 | 46.41 |
|  | Diffusers | 1 |  | 37.61 | 115.14 |
|  | NIM | 1 | 2.86 | 23.67 | 85.52 |
| **GB300** | PyTorch | 1 |  |  |  |
|  | Diffusers | 1 |  | 34.53 | 97.30 |
|  | NIM | 1 |  |  |  |
| **H100 NVL** | NIM | 1 | 6.70 | 67.09 | 252.22 |
| **H200 NVL** | NIM | 1 | 5.42 | 56.85 | 227.34 |

### Image-to-Video (i2v)

| GPU | Engine | GPUs | 256p | 480p | 720p |
|---|---|:-:|---:|---:|---:|
| **RTX PRO 6000 Blackwell** | PyTorch | 1 | 4.50 | 45.96 | 169.15 |
|  | Diffusers | 1 |  |  |  |
|  | NIM | 1 | 7.27 | 81.50 | 313.64 |
| **H20** | PyTorch | 1 | 11.19 | 110.19 | 419.99 |
|  | Diffusers | 1 |  |  |  |
|  | NIM | 1 | 18.69 | 195.87 | 776.48 |
| **H100 80GB HBM3** | PyTorch | 1 | 2.96 | 25.04 | 91.65 |
|  | Diffusers | 1 |  | 71.06 | 241.53 |
|  | NIM | 1 | 5.37 | 46.34 | 173.14 |
| **H200 141GB HBM3** | PyTorch | 1 | 2.94 | 25.56 | 92.44 |
|  | Diffusers | 1 |  | 69.69 | 237.95 |
|  | NIM | 1 | 5.07 | 46.95 | 177.57 |
| **B200** | PyTorch | 1 | 1.87 | 14.24 | 50.24 |
|  | Diffusers | 1 |  | 40.69 | 126.07 |
|  | NIM | 1 | 2.98 | 25.59 | 91.65 |
| **B300** | PyTorch | 1 | 1.78 | 13.08 | 44.94 |
|  | Diffusers | 1 |  | 38.35 | 117.18 |
|  | NIM | 1 | 2.88 | 23.74 | 85.50 |
| **GB300** | PyTorch | 1 |  |  |  |
|  | Diffusers | 1 |  | 34.96 | 99.37 |
|  | NIM | 1 |  |  |  |
| **H100 NVL** | NIM | 1 | 6.76 | 67.50 | 251.21 |
| **H200 NVL** | NIM | 1 | 5.43 | 57.02 | 226.12 |

### Text-to-Image (t2i)

| GPU | Engine | GPUs | 256p | 480p | 720p |
|---|---|:-:|---:|---:|---:|
| **RTX PRO 6000 Blackwell** | PyTorch | 1 | 0.97 | 1.02 | 1.50 |
|  | Diffusers | 1 |  | 7.29 |  |
|  | NIM | 1 | 1.01 | 1.33 | 2.25 |
| **H20** | PyTorch | 1 | 1.27 | 1.74 | 3.25 |
|  | Diffusers | 1 |  |  |  |
|  | NIM | 1 | 1.18 | 2.54 | 4.70 |
| **H100 80GB HBM3** | PyTorch | 1 | 1.16 | 1.14 | 1.21 |
|  | Diffusers | 1 |  | 7.50 | 7.68 |
|  | NIM | 1 | 1.10 | 1.24 | 1.51 |
| **H200 141GB HBM3** | PyTorch | 1 | 1.12 | 1.14 | 1.21 |
|  | Diffusers | 1 |  | 7.44 | 7.74 |
|  | NIM | 1 | 1.17 | 1.18 | 1.42 |
| **B200** | PyTorch | 1 | 0.96 | 0.92 | 0.97 |
|  | Diffusers | 1 |  | 6.82 | 6.94 |
|  | NIM | 1 | 0.99 | 1.06 | 1.02 |
| **B300** | PyTorch | 1 | 0.95 | 0.95 | 1.51 |
|  | Diffusers | 1 |  | 10.28 | 6.71 |
|  | NIM | 1 | 1.45 | 1.50 | 1.48 |
| **GB300** | PyTorch | 1 |  |  |  |
|  | Diffusers | 1 |  | 11.79 | 13.19 |
|  | NIM | 1 |  |  |  |
| **H100 NVL** | NIM | 1 | 1.14 | 1.15 | 1.72 |
| **H200 NVL** | NIM | 1 | 1.12 | 1.16 | 1.46 |

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
| **RTX PRO 6000 Blackwell** | PyTorch | 1 | 4.40 | 47.39 | 175.87 |
|  | Diffusers | 1 |  |  |  |
| **H20** | PyTorch | 1 | 10.99 | 112.59 | 437.65 |
|  | Diffusers | 1 |  |  |  |
| **H100 80GB HBM3** | PyTorch | 1 | 2.93 | 25.89 | 95.20 |
|  | Diffusers | 1 |  | 72.01 | 241.38 |
| **H200 141GB HBM3** | PyTorch | 1 | 2.90 | 26.51 | 96.80 |
|  | Diffusers | 1 |  | 70.17 | 239.24 |
| **B200** | PyTorch | 1 | 1.85 | 14.69 | 52.19 |
|  | Diffusers | 1 |  | 41.72 | 127.42 |
| **B300** | PyTorch | 1 | 1.92 | 13.61 | 47.32 |
|  | Diffusers | 1 |  | 39.99 | 118.73 |
| **GB300** | PyTorch | 1 |  |  |  |
|  | Diffusers | 1 |  | 35.97 | 100.27 |

### Text-to-Audio-and-Video (t2av)

| GPU | Engine | GPUs | 256p | 480p | 720p |
|---|---|:-:|---:|---:|---:|
| **RTX PRO 6000 Blackwell** | PyTorch | 1 | 7.67 | 50.58 | 178.94 |
|  | Diffusers | 1 |  |  |  |
| **H20** | PyTorch | 1 | 17.97 | 120.72 | 442.78 |
|  | Diffusers | 1 |  |  |  |
| **H100 80GB HBM3** | PyTorch | 1 | 4.88 | 28.50 | 97.19 |
|  | Diffusers | 1 |  | 70.46 | 239.14 |
| **H200 141GB HBM3** | PyTorch | 1 | 4.79 | 28.21 | 99.38 |
|  | Diffusers | 1 |  | 68.99 | 237.11 |
| **B200** | PyTorch | 1 | 3.10 | 16.48 | 54.04 |
|  | Diffusers | 1 |  | 40.71 | 124.89 |
| **B300** | PyTorch | 1 | 3.12 | 15.17 | 49.50 |
|  | Diffusers | 1 |  | 38.13 | 116.44 |
| **GB300** | PyTorch | 1 |  |  |  |
|  | Diffusers | 1 |  | 34.89 | 100.22 |

### Video-to-Audio-and-Video (v2av)

| GPU | Engine | GPUs | 256p | 480p | 720p |
|---|---|:-:|---:|---:|---:|
| **RTX PRO 6000 Blackwell** | PyTorch | — |  |  |  |
|  | Diffusers | — |  |  |  |
| **H20** | PyTorch | — |  |  |  |
|  | Diffusers | — |  |  |  |
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

### Image-to-Audio-and-Video (i2av)

| GPU | Engine | GPUs | 256p | 480p | 720p |
|---|---|:-:|---:|---:|---:|
| **RTX PRO 6000 Blackwell** | PyTorch | 1 | 7.30 | 48.80 | 174.71 |
|  | Diffusers | 1 |  |  |  |
| **H20** | PyTorch | 1 | 17.08 | 115.28 | 428.45 |
|  | Diffusers | 1 |  |  |  |
| **H100 80GB HBM3** | PyTorch | 1 | 4.63 | 27.33 | 94.44 |
|  | Diffusers | 1 |  | 72.07 | 243.07 |
| **H200 141GB HBM3** | PyTorch | 1 | 4.54 | 27.44 | 97.66 |
|  | Diffusers | 1 |  | 70.18 | 241.70 |
| **B200** | PyTorch | 1 | 2.96 | 15.67 | 52.20 |
|  | Diffusers | 1 |  | 41.66 | 126.91 |
| **B300** | PyTorch | 1 | 2.81 | 14.60 | 47.27 |
|  | Diffusers | 1 |  | 39.34 |  |
| **GB300** | PyTorch | 1 |  |  |  |
|  | Diffusers | 1 |  | 36.51 | 101.85 |

## Transfer generation

Transfer conditions a video-to-video generation on a structural control hint
extracted from the source video. Each control is reported separately because
the hint changes how much of the frame the model must synthesise.

| GPU | Transfer control | Engine | GPUs | 256p | 480p | 720p |
|---|---|---|:-:|---:|---:|---:|
| **RTX PRO 6000 Blackwell** | blur | PyTorch | 1 | 15.92 | 99.23 | 369.40 |
|  |  | Diffusers | 1 |  |  |  |
|  | depth | PyTorch | 1 | 15.34 | 102.88 | 379.90 |
|  |  | Diffusers | 1 |  |  |  |
|  | edge | PyTorch | 1 | 13.27 | 99.61 | 374.57 |
|  |  | Diffusers | 1 |  |  |  |
|  | multi_control | PyTorch | — |  |  |  |
|  |  | Diffusers | — |  |  |  |
|  | seg | PyTorch | 1 | 14.54 | 104.96 | 378.47 |
|  |  | Diffusers | 1 |  |  |  |
|  | wsm | PyTorch | 1 | 12.81 | 78.87 | 280.91 |
|  |  | Diffusers | 1 |  |  | 430.19 |
| **H20** | blur | PyTorch | 1 | 36.48 | 238.41 | 918.21 |
|  |  | Diffusers | 1 |  |  |  |
|  | depth | PyTorch | 1 | 34.13 | 243.68 | 951.21 |
|  |  | Diffusers | 1 |  |  |  |
|  | edge | PyTorch | 1 | 29.64 | 238.38 | 947.97 |
|  |  | Diffusers | 1 |  |  |  |
|  | multi_control | PyTorch | 1 | 36.57 | 385.10 | 1678.80 |
|  |  | Diffusers | 1 |  |  |  |
|  | seg | PyTorch | 1 | 31.20 | 239.56 | 992.60 |
|  |  | Diffusers | 1 |  |  |  |
|  | wsm | PyTorch | 1 | 28.15 | 186.34 | 707.32 |
|  |  | Diffusers | 1 |  |  |  |
| **H100 80GB HBM3** | blur | PyTorch | 1 | 10.37 | 55.94 | 198.76 |
|  |  | Diffusers | 1 |  | 171.90 | 600.30 |
|  | depth | PyTorch | 1 | 9.83 | 57.44 | 207.16 |
|  |  | Diffusers | 1 |  | 178.89 | 606.60 |
|  | edge | PyTorch | 1 | 8.49 | 55.27 | 204.24 |
|  |  | Diffusers | 1 |  | 172.75 | 597.01 |
|  | multi_control | PyTorch | 1 | 10.51 | 91.48 | 367.08 |
|  |  | Diffusers | 1 |  |  |  |
|  | seg | PyTorch | 1 | 10.03 | 56.62 | 214.56 |
|  |  | Diffusers | 1 |  | 174.09 | 601.13 |
|  | wsm | PyTorch | 1 | 8.33 | 44.52 | 153.26 |
|  |  | Diffusers | 1 |  | 84.48 | 264.67 |
| **H200 141GB HBM3** | blur | PyTorch | 1 | 10.32 | 55.96 | 205.31 |
|  |  | Diffusers | 1 |  | 169.37 | 598.78 |
|  | depth | PyTorch | 1 | 9.70 | 58.92 | 211.20 |
|  |  | Diffusers | 1 |  | 177.22 | 603.37 |
|  | edge | PyTorch | 1 | 8.60 | 56.78 | 208.28 |
|  |  | Diffusers | 1 |  | 169.21 | 594.04 |
|  | multi_control | PyTorch | 1 | 10.46 | 92.81 | 382.20 |
|  |  | Diffusers | 1 |  |  |  |
|  | seg | PyTorch | 1 | 9.68 | 58.18 | 209.54 |
|  |  | Diffusers | 1 |  | 170.62 | 601.64 |
|  | wsm | PyTorch | 1 | 8.39 | 45.06 | 154.98 |
|  |  | Diffusers | 1 |  | 82.92 | 263.74 |
| **B200** | blur | PyTorch | 1 | 7.02 | 32.23 | 110.23 |
|  |  | Diffusers | 1 |  | 94.88 | 298.51 |
|  | depth | PyTorch | 1 | 6.68 | 33.48 | 113.49 |
|  |  | Diffusers | 1 |  | 98.03 | 307.64 |
|  | edge | PyTorch | 1 | 5.79 | 31.53 | 111.67 |
|  |  | Diffusers | 1 |  | 96.11 | 302.90 |
|  | multi_control | PyTorch | — |  |  |  |
|  |  | Diffusers | — |  |  |  |
|  | seg | PyTorch | 1 | 6.65 | 33.21 | 114.39 |
|  |  | Diffusers | 1 |  | 97.13 | 301.15 |
|  | wsm | PyTorch | 1 | 5.78 | 26.19 | 85.06 |
|  |  | Diffusers | 1 |  | 48.75 | 138.10 |
| **B300** | blur | PyTorch | 1 | 8.30 | 31.29 | 101.38 |
|  |  | Diffusers | 1 |  |  | 278.49 |
|  | depth | PyTorch | 1 | 7.81 | 32.60 | 105.09 |
|  |  | Diffusers | 1 |  | 93.48 | 282.93 |
|  | edge | PyTorch | 1 | 6.08 | 31.68 | 103.18 |
|  |  | Diffusers | 1 |  | 89.60 | 277.25 |
|  | multi_control | PyTorch | — |  |  |  |
|  |  | Diffusers | — |  |  |  |
|  | seg | PyTorch | 1 | 7.14 | 32.89 | 106.26 |
|  |  | Diffusers | 1 |  | 89.99 | 280.71 |
|  | wsm | PyTorch | 1 | 5.89 | 25.42 | 78.99 |
|  |  | Diffusers | 1 |  | 46.64 | 129.77 |
| **GB300** | blur | PyTorch | 1 |  |  |  |
|  |  | Diffusers | 1 |  | 80.61 | 236.36 |
|  | depth | PyTorch | 1 |  |  |  |
|  |  | Diffusers | 1 |  | 81.57 | 238.78 |
|  | edge | PyTorch | 1 |  |  |  |
|  |  | Diffusers | 1 |  | 80.54 | 235.31 |
|  | multi_control | PyTorch | — |  |  |  |
|  |  | Diffusers | — |  |  |  |
|  | seg | PyTorch | 1 |  |  |  |
|  |  | Diffusers | 1 |  | 81.67 | 237.79 |
|  | wsm | PyTorch | 1 |  |  |  |
|  |  | Diffusers | 1 |  | 42.26 | 110.70 |

<sub>Transfer notes:
1. Control hints follow the `extra_params` names: `blur`, `depth`, `edge`, `multi_control`, `seg`, and `wsm`. `multi_control` combines several hints in one request.
2. The PyTorch campaign reports `segmentation`, which is the same control as `seg` in the Diffusers campaign.
3. Diffusers FP8 has no `multi_control` measurements.</sub>

## Action generation

### Forward Dynamics — Autonomous Vehicle (AV)

| GPU | Engine | GPUs | 256p | 480p | 720p |
|---|---|:-:|---:|---:|---:|
| **RTX PRO 6000 Blackwell** | PyTorch | 1 | 5.80 | 5.78 | 5.80 |
|  | Diffusers | 1 |  |  |  |
| **H20** | PyTorch | 1 | 13.44 | 13.32 | 13.44 |
|  | Diffusers | 1 |  |  |  |
| **H100 80GB HBM3** | PyTorch | 1 | 3.54 | 3.53 | 3.53 |
|  | Diffusers | 1 |  | 11.68 | 12.20 |
| **H200 141GB HBM3** | PyTorch | 1 | 3.45 | 3.47 | 3.46 |
|  | Diffusers | 1 |  | 11.19 | 11.73 |
| **B200** | PyTorch | 1 | 2.16 | 2.16 | 2.16 |
|  | Diffusers | 1 |  | 8.05 | 8.42 |
| **B300** | PyTorch | 1 | 2.11 | 2.09 | 2.10 |
|  | Diffusers | 1 |  | 8.83 | 8.96 |
| **GB300** | PyTorch | 1 |  |  |  |
|  | Diffusers | 1 |  | 7.71 | 7.64 |

### Forward Dynamics — Camera

| GPU | Engine | GPUs | 256p | 480p | 720p |
|---|---|:-:|---:|---:|---:|
| **RTX PRO 6000 Blackwell** | PyTorch | 1 | 5.47 | 5.49 | 5.48 |
|  | Diffusers | 1 |  |  |  |
| **H20** | PyTorch | 1 | 12.69 | 12.57 | 12.58 |
|  | Diffusers | 1 |  |  |  |
| **H100 80GB HBM3** | PyTorch | 1 | 3.38 | 3.37 | 3.36 |
|  | Diffusers | 1 |  |  |  |
| **H200 141GB HBM3** | PyTorch | 1 | 3.29 | 3.30 | 3.29 |
|  | Diffusers | 1 |  |  |  |
| **B200** | PyTorch | 1 | 2.06 | 2.06 | 2.06 |
|  | Diffusers | 1 |  |  |  |
| **B300** | PyTorch | 1 | 1.98 | 1.98 | 2.06 |
|  | Diffusers | 1 |  |  |  |

### Forward Dynamics — Robot

| GPU | Engine | GPUs | 256p | 480p | 720p |
|---|---|:-:|---:|---:|---:|
| **RTX PRO 6000 Blackwell** | PyTorch | 1 | 1.28 | 1.28 | 1.28 |
|  | Diffusers | 1 |  |  | 5.63 |
| **H20** | PyTorch | 1 | 2.85 | 2.84 | 2.84 |
|  | Diffusers | 1 |  |  |  |
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

### Inverse Dynamics — Autonomous Vehicle (AV)

| GPU | Engine | GPUs | 256p | 480p | 720p |
|---|---|:-:|---:|---:|---:|
| **RTX PRO 6000 Blackwell** | PyTorch | 1 | 9.02 | 9.01 | 9.00 |
|  | Diffusers | 1 |  |  |  |
| **H20** | PyTorch | 1 | 21.15 | 21.36 | 21.15 |
|  | Diffusers | 1 |  |  |  |
| **H100 80GB HBM3** | PyTorch | 1 | 5.45 | 5.44 | 5.44 |
|  | Diffusers | 1 |  | 11.61 | 12.21 |
| **H200 141GB HBM3** | PyTorch | 1 | 5.32 | 5.39 | 5.32 |
|  | Diffusers | 1 |  | 11.10 | 11.65 |
| **B200** | PyTorch | 1 | 3.25 | 3.24 | 3.24 |
|  | Diffusers | 1 |  | 8.00 | 8.27 |
| **B300** | PyTorch | 1 | 3.15 | 3.14 | 3.13 |
|  | Diffusers | 1 |  | 8.90 | 8.95 |
| **GB300** | PyTorch | 1 |  |  |  |
|  | Diffusers | 1 |  | 7.85 | 7.88 |

### Inverse Dynamics — Robot

| GPU | Engine | GPUs | 256p | 480p | 720p |
|---|---|:-:|---:|---:|---:|
| **RTX PRO 6000 Blackwell** | PyTorch | 1 | 1.86 | 1.85 | 1.85 |
|  | Diffusers | 1 |  |  | 5.65 |
| **H20** | PyTorch | 1 | 4.32 | 4.28 | 4.31 |
|  | Diffusers | 1 |  |  |  |
| **H100 80GB HBM3** | PyTorch | 1 | 1.21 | 1.22 | 1.22 |
|  | Diffusers | 1 |  | 4.75 | 4.79 |
| **H200 141GB HBM3** | PyTorch | 1 | 1.19 | 1.19 | 1.19 |
|  | Diffusers | 1 |  | 4.64 | 4.72 |
| **B200** | PyTorch | 1 | 0.84 | 0.86 | 0.84 |
|  | Diffusers | 1 |  | 4.02 | 4.18 |
| **B300** | PyTorch | 1 | 0.87 | 0.84 | 0.84 |
|  | Diffusers | 1 |  |  | 6.03 |
| **GB300** | PyTorch | 1 |  |  |  |
|  | Diffusers | 1 |  | 6.57 | 6.08 |

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
