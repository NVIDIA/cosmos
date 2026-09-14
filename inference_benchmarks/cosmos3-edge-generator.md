# Cosmos3-Edge Generator Inference Benchmarks

[Back to the inference benchmark index](../inference_benchmarks.md)

These tables report **Cosmos3-Edge** Generator latency in seconds for **image-to-video (i2v)**. Measurements use one GPU or one integrated computing platform. Lower latency is better, and empty cells indicate that a run has not been completed.

Unless otherwise noted, visual-generation benchmarks use **480p resolution**. Video benchmarks generate **121 frames**. vLLM-Omni values report end-to-end latency, while PyTorch values report average generation latency.

### vLLM-Omni

| GPU or Platform | Image-to-Video |
|---|---:|
| B200 SXM 192 GB | 7.09 |
| B300 | 8.74 |
| H200 SXM 141 GB | 13.13 |
| H200 NVL | 14.15 |
| H100 SXM 80 GB | 13.33 |
| H100 NVL 96 GB | 17.33 |
| H20 SXM 96 GB | 53.94 |
| RTX PRO 6000 Blackwell Server Edition |  |
| DGX Station | 7.13 |
| DGX Spark | 89.41 |
| Jetson AGX Thor T5000, 128 GB, MAXN |  |
| Jetson T3000, 32 GB, 1100 MHz |  |
| Jetson T2000, 16 GB, 702 MHz, THOR_NANO |  |

### PyTorch

| GPU or Platform | Image-to-Video |
|---|---:|
| B200 SXM 192 GB | 7.45 |
| B300 | 6.84 |
| H200 SXM 141 GB | 12.31 |
| H200 NVL | 14.07 |
| H100 SXM 80 GB | 12.68 |
| H100 NVL 96 GB | 16.42 |
| H20 SXM 96 GB | 52.83 |
| RTX PRO 6000 Blackwell Server Edition | 21.92 |
| DGX Station | 6.31 |
| DGX Spark | 103.36 |
| Jetson AGX Thor T5000, 128 GB, MAXN |  |
| Jetson T3000, 32 GB, 1100 MHz |  |

<sub>Notes:
1. All measurements use one GPU or one integrated computing platform.
2. Values are average end-to-end or generation latency in seconds; lower is better.
3. Unless otherwise specified, visual-generation measurements use **480p resolution**.
4. Video measurements (i2v) generate **121 output frames**.
5. PyTorch values report average generation latency rather than diffusion-only latency.</sub>
