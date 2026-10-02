# Cosmos3-Edge Generator Inference Benchmarks

[Back to the inference benchmark index](../inference_benchmarks.md)

These tables report **Cosmos3-Edge** Generator latency in seconds for **image-to-video (i2v)**. Lower latency is better, and empty cells indicate that a run has not been completed.

Cosmos3-Edge supports **256p and 480p**, so 480p is its highest resolution. Every row reports one GPU configuration, named in the **GPUs** column: one GPU where the model fits on a single device at 480p, and eight GPUs where it does not. Cosmos3-Edge fits on a single GPU or one integrated computing platform at 480p, so all populated rows report one. Video benchmarks generate **121 frames**. vLLM-Omni values report end-to-end latency, while PyTorch values report average generation latency.

### vLLM-Omni

| GPU or Platform | GPUs | 256p | 480p |
|---|:-:|---:|---:|
| B200 SXM 192 GB | 1 |  | 7.09 |
| B300 | 1 |  | 8.74 |
| H200 SXM 141 GB | 1 |  | 13.13 |
| H200 NVL | 1 |  | 14.15 |
| H100 SXM 80 GB | 1 |  | 13.33 |
| H100 NVL 96 GB | 1 |  | 17.33 |
| H20 SXM 96 GB | 1 |  | 53.94 |
| RTX PRO 6000 Blackwell Server Edition | — |  |  |
| DGX Station | 1 |  | 7.13 |
| DGX Spark | 1 |  | 89.41 |
| Jetson AGX Thor T5000, 128 GB, MAXN | — |  |  |
| Jetson T3000, 32 GB, 1100 MHz | — |  |  |
| Jetson T2000, 16 GB, 702 MHz, THOR_NANO | — |  |  |

### PyTorch

| GPU or Platform | GPUs | 256p | 480p |
|---|:-:|---:|---:|
| B200 SXM 192 GB | 1 |  | 7.45 |
| B300 | 1 |  | 6.84 |
| H200 SXM 141 GB | 1 |  | 12.31 |
| H200 NVL | 1 |  | 14.07 |
| H100 SXM 80 GB | 1 |  | 12.68 |
| H100 NVL 96 GB | 1 |  | 16.42 |
| H20 SXM 96 GB | 1 |  | 52.83 |
| RTX PRO 6000 Blackwell Server Edition | 1 |  | 21.92 |
| DGX Station | 1 |  | 6.31 |
| DGX Spark | 1 |  | 103.36 |
| Jetson AGX Thor T5000, 128 GB, MAXN | — |  |  |
| Jetson T3000, 32 GB, 1100 MHz | — |  |  |

<sub>Notes:
1. The **GPUs** column gives the number of GPUs behind every value in that row. All current measurements use one GPU or one integrated computing platform.
2. Values are average end-to-end or generation latency in seconds; lower is better.
3. Cosmos3-Edge supports 256p and 480p. Every benchmark run to date is at 480p, so the 256p column is reserved for a future campaign rather than unsupported.
4. Video measurements (i2v) generate **121 output frames**.
5. PyTorch values report average generation latency rather than diffusion-only latency.</sub>
