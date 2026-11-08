# Cosmos3-Edge-FP8 Reasoner Inference Benchmarks

[Back to the inference benchmark index](../inference_benchmarks.md)

These tables measure text output through **vLLM** and, in separate sections, **NIM**, with FP8 weights. Latencies are milliseconds; request and output-token throughput are per second. Lower latency and higher throughput are better. Each system uses **one GPU or one integrated platform**; concurrency is the number of simultaneous client requests.

The vLLM workload uses 50 text input tokens and one 374×374, 30-second synthetic video: 30 frames at 1 FPS or 60 frames at 2 FPS. Requested output length is 1 or 100 tokens. KV-cache precision is separate from model-weight precision. Blank cells have no validated published measurement and must not be read as lack of support.

See [measurement settings and reproduction limits](benchmark_methodology.md). Displayed values retain the source's available precision; trailing zeros do not imply additional measurement accuracy.

Datacenter runs use vLLM 0.28.0. Enterprise measurements cover RTX PRO 5000, DGX Spark, and DGX Station. Embedded runs use vLLM 0.31.1rc1, AIPerf 0.11.0, and JetPack 7.2.1 with MAXN and `jetson_clocks`. The T3000 row is a T5000 flashed with the T3000 emulation configuration (32 GB, 8 CPUs, 12 SMs).

On Thor, set `VLLM_DISABLED_KERNELS=FlashInferFP8ScaledMMLinearKernel` and `VLLM_USE_FLASHINFER_SAMPLER=0` to select the measured CUTLASS FP8 path. Orin uses weight-only W8A16. GPU memory utilization is 0.7 on Orin, 0.6 on T5000, and 0.5 on the T3000 emulation. The embedded vLLM serving arguments are `--dtype auto --kv-cache-dtype auto --tensor-parallel-size 1 --max-model-len 16384 --reasoning-parser qwen3 --trust-remote-code --no-enable-prefix-caching --limit-mm-per-prompt '{"video": 1}' --media-io-kwargs '{"video":{"num_frames": -1}}'`.

**Successful-request accounting:** some single-token runs returned empty replies. Their metrics summarize requests that completed with content; the request-count row records that population. Those rows do not establish a zero-error serving rate. Output-100 runs and single-token runs must be considered separately.

## Known anomalies and comparison limits

- The DGX Spark vLLM output-100 / 1-FPS throughput jump from 85.40 Tok/s at concurrency 128 to 170.30 Tok/s at concurrency 256 was reproduced in the measured campaign. It is retained as an unexplained scaling discontinuity, not smoothed away.
- Edge NIM uses video-token pruning rate **0.6** with `vidcom2`, GPU memory utilization **0.7**, and maximum batched tokens **24,300**. This changes the vision workload relative to vLLM and is a confounder in runtime speedup claims.
- Thor NIM's decode path drops the last frame; at 8 FPS it decodes 239 of 240 submitted frames and reduces the per-frame spatial budget. VP9/CPU decoding is separately labelled. Nominal video FPS alone does not establish identical processed token counts.
- Some errored NIM runs have valid-request-only metrics in the source. This page keeps passed runs only, so those cells are deliberately withheld rather than unmeasured.

See the [anomaly review](benchmark_anomaly_review.md).

## Datacenter

### H100 80GB HBM3

#### Input 50 / Output 1 / Video 1 FPS

| Metric | Concurrency 1 | Concurrency 64 | Concurrency 128 | Concurrency 256 |
|---|---:|---:|---:|---:|
| Time To First Token (ms) | 107.32 |  | 2001.51 | 3909.29 |
| Request Latency (ms) | 107.32 |  | 2001.51 | 3909.29 |
| Request Throughput (Req/s) | 9.17 |  | 48.71 | 49.60 |
| Output Token Throughput (Tok/s) | 9.17 |  | 48.71 | 49.60 |

#### Input 50 / Output 1 / Video 2 FPS

| Metric | Concurrency 1 | Concurrency 64 | Concurrency 128 | Concurrency 256 |
|---|---:|---:|---:|---:|
| Time To First Token (ms) | 165.42 | 2766.77 | 4673.78 | 9367.14 |
| Request Latency (ms) | 165.42 | 2766.77 | 4673.78 | 9367.14 |
| Request Throughput (Req/s) | 5.98 | 20.88 | 20.71 | 20.60 |
| Output Token Throughput (Tok/s) | 5.98 | 20.88 | 20.71 | 20.60 |

#### Input 50 / Output 100 / Video 1 FPS

| Metric | Concurrency 1 | Concurrency 64 | Concurrency 128 | Concurrency 256 |
|---|---:|---:|---:|---:|
| Time To First Token (ms) | 108.10 | 559.24 | 1043.72 | 4593.92 |
| Request Latency (ms) | 317.74 | 2563.08 | 4864.05 | 8953.01 |
| Request Throughput (Req/s) | 3.13 | 24.66 | 25.36 | 25.55 |
| Output Token Throughput (Tok/s) | 312.64 | 2464.50 | 2533.25 | 2553.81 |

#### Input 50 / Output 100 / Video 2 FPS

| Metric | Concurrency 1 | Concurrency 64 | Concurrency 128 | Concurrency 256 |
|---|---:|---:|---:|---:|
| Time To First Token (ms) | 164.03 | 855.47 | 5005.28 | 13111.67 |
| Request Latency (ms) | 393.74 | 5381.15 | 9762.12 | 18118.99 |
| Request Throughput (Req/s) | 2.52 | 11.73 | 11.77 | 11.79 |
| Output Token Throughput (Tok/s) | 252.34 | 1173.02 | 1176.52 | 1178.38 |


### H200 141GB HBM3

#### Input 50 / Output 1 / Video 1 FPS

| Metric | Concurrency 1 | Concurrency 64 | Concurrency 128 | Concurrency 256 |
|---|---:|---:|---:|---:|
| Time To First Token (ms) | 106.24 | 1156.80 | 1976.93 |  |
| Request Latency (ms) | 106.24 | 1156.80 | 1976.93 |  |
| Request Throughput (Req/s) | 9.26 | 50.06 | 49.33 |  |
| Output Token Throughput (Tok/s) | 9.26 | 50.06 | 49.33 |  |

#### Input 50 / Output 1 / Video 2 FPS

| Metric | Concurrency 1 | Concurrency 64 | Concurrency 128 | Concurrency 256 |
|---|---:|---:|---:|---:|
| Time To First Token (ms) | 162.05 | 2767.54 | 4646.28 | 9087.14 |
| Request Latency (ms) | 162.05 | 2767.54 | 4646.28 | 9087.14 |
| Request Throughput (Req/s) | 6.10 | 20.87 | 20.82 | 21.23 |
| Output Token Throughput (Tok/s) | 6.10 | 20.87 | 20.82 | 21.23 |

#### Input 50 / Output 100 / Video 1 FPS

| Metric | Concurrency 1 | Concurrency 64 | Concurrency 128 | Concurrency 256 |
|---|---:|---:|---:|---:|
| Time To First Token (ms) | 102.99 | 661.99 | 1143.81 | 3386.64 |
| Request Latency (ms) | 290.79 | 2300.87 | 4457.08 | 8610.34 |
| Request Throughput (Req/s) | 3.41 | 27.51 | 27.80 | 27.71 |
| Output Token Throughput (Tok/s) | 341.18 | 2749.96 | 2777.34 | 2769.39 |

#### Input 50 / Output 100 / Video 2 FPS

| Metric | Concurrency 1 | Concurrency 64 | Concurrency 128 | Concurrency 256 |
|---|---:|---:|---:|---:|
| Time To First Token (ms) | 160.58 | 954.63 | 3803.04 | 11292.26 |
| Request Latency (ms) | 363.42 | 4835.77 | 9551.40 | 17654.95 |
| Request Throughput (Req/s) | 2.73 | 13.08 | 12.58 | 12.57 |
| Output Token Throughput (Tok/s) | 273.38 | 1307.27 | 1256.96 | 1256.31 |


### H20

#### Input 50 / Output 1 / Video 1 FPS

| Metric | Concurrency 1 | Concurrency 64 | Concurrency 128 | Concurrency 256 |
|---|---:|---:|---:|---:|
| Time To First Token (ms) | 164.66 | 4320.69 | 7251.94 | 14448.84 |
| Request Latency (ms) | 164.66 | 4320.69 | 7251.94 | 14448.84 |
| Request Throughput (Req/s) | 6.00 | 13.39 | 13.36 | 13.36 |
| Output Token Throughput (Tok/s) | 6.00 | 13.39 | 13.36 | 13.36 |

#### Input 50 / Output 1 / Video 2 FPS

| Metric | Concurrency 1 | Concurrency 64 | Concurrency 128 | Concurrency 256 |
|---|---:|---:|---:|---:|
| Time To First Token (ms) | 304.71 | 10466.74 |  | 34991.21 |
| Request Latency (ms) | 304.71 | 10466.74 |  | 34991.21 |
| Request Throughput (Req/s) | 3.26 | 5.52 |  | 5.51 |
| Output Token Throughput (Tok/s) | 3.26 | 5.52 |  | 5.51 |

#### Input 50 / Output 100 / Video 1 FPS

| Metric | Concurrency 1 | Concurrency 64 | Concurrency 128 | Concurrency 256 |
|---|---:|---:|---:|---:|
| Time To First Token (ms) | 163.60 | 1416.49 | 3395.32 | 12957.73 |
| Request Latency (ms) | 378.55 | 7263.86 | 14810.21 | 29096.98 |
| Request Throughput (Req/s) | 2.63 | 8.77 | 8.47 | 8.19 |
| Output Token Throughput (Tok/s) | 262.42 | 876.05 | 846.26 | 819.07 |

#### Input 50 / Output 100 / Video 2 FPS

| Metric | Concurrency 1 | Concurrency 64 | Concurrency 128 | Concurrency 256 |
|---|---:|---:|---:|---:|
| Time To First Token (ms) | 306.02 | 2520.46 | 14669.80 | 41130.89 |
| Request Latency (ms) | 551.74 | 17029.24 | 32764.47 | 60768.68 |
| Request Throughput (Req/s) | 1.81 | 3.73 | 3.66 | 3.63 |
| Output Token Throughput (Tok/s) | 180.23 | 372.81 | 365.36 | 362.74 |


### RTX PRO 6000 Blackwell

#### Input 50 / Output 1 / Video 1 FPS

| Metric | Concurrency 1 | Concurrency 64 | Concurrency 128 | Concurrency 256 |
|---|---:|---:|---:|---:|
| Time To First Token (ms) | 120.52 | 1911.14 | 3268.45 |  |
| Request Latency (ms) | 120.52 | 1911.14 | 3268.45 |  |
| Request Throughput (Req/s) | 8.18 | 30.13 | 29.65 |  |
| Output Token Throughput (Tok/s) | 8.18 | 30.13 | 29.65 |  |

#### Input 50 / Output 1 / Video 2 FPS

| Metric | Concurrency 1 | Concurrency 64 | Concurrency 128 | Concurrency 256 |
|---|---:|---:|---:|---:|
| Time To First Token (ms) | 193.24 | 4786.41 |  | 16052.71 |
| Request Latency (ms) | 193.24 | 4786.41 |  | 16052.71 |
| Request Throughput (Req/s) | 5.13 | 12.05 |  | 11.98 |
| Output Token Throughput (Tok/s) | 5.13 | 12.05 |  | 11.98 |

#### Input 50 / Output 100 / Video 1 FPS

| Metric | Concurrency 1 | Concurrency 64 | Concurrency 128 | Concurrency 256 |
|---|---:|---:|---:|---:|
| Time To First Token (ms) | 120.98 | 686.96 | 1882.00 | 8890.10 |
| Request Latency (ms) | 405.29 | 4658.97 | 10223.88 | 21081.70 |
| Request Throughput (Req/s) | 2.45 | 13.62 | 12.12 | 11.13 |
| Output Token Throughput (Tok/s) | 245.38 | 1361.79 | 1211.42 | 1111.83 |

#### Input 50 / Output 100 / Video 2 FPS

| Metric | Concurrency 1 | Concurrency 64 | Concurrency 128 | Concurrency 256 |
|---|---:|---:|---:|---:|
| Time To First Token (ms) | 195.27 | 1274.77 | 8850.49 | 26278.82 |
| Request Latency (ms) | 508.56 | 10610.40 | 20842.79 | 39274.74 |
| Request Throughput (Req/s) | 1.96 | 5.96 | 5.64 | 5.54 |
| Output Token Throughput (Tok/s) | 195.69 | 595.21 | 563.31 | 553.34 |


## Enterprise

### RTX PRO 5000 Blackwell 72GB

#### Input 50 / Output 1 / Video 1 FPS

| Metric | Concurrency 1 | Concurrency 64 | Concurrency 128 | Concurrency 256 |
|---|---:|---:|---:|---:|
| Time To First Token (ms) | 100.66 | 3354.34 | 5627.30 | 11293.15 |
| Request Latency (ms) | 100.66 | 3354.34 | 5627.30 | 11293.15 |
| Request Count (requests) | 50 | 320 | 256 | 512 |
| Request Throughput (Req/s) | 9.71 | 17.16 | 17.10 | 17.00 |
| Output Token Throughput (Tok/s) | 9.71 | 17.16 | 17.10 | 17.00 |

#### Input 50 / Output 1 / Video 2 FPS

| Metric | Concurrency 1 | Concurrency 64 | Concurrency 128 | Concurrency 256 |
|---|---:|---:|---:|---:|
| Time To First Token (ms) | 197.93 | 8547.20 | 14141.65 | 28897.19 |
| Request Latency (ms) | 197.93 | 8547.20 | 14141.65 | 28897.19 |
| Request Count (requests) | 50 | 320 | 256 | 512 |
| Request Throughput (Req/s) | 5.00 | 6.74 | 6.79 | 6.63 |
| Output Token Throughput (Tok/s) | 5.00 | 6.74 | 6.79 | 6.63 |

#### Input 50 / Output 100 / Video 1 FPS

| Metric | Concurrency 1 | Concurrency 64 | Concurrency 128 | Concurrency 256 |
|---|---:|---:|---:|---:|
| Time To First Token (ms) | 99.66 | 919.23 | 3850.68 | 16391.67 |
| Request Latency (ms) | 407.24 | 7465.64 | 15434.92 | 29438.56 |
| Request Count (requests) | 50 | 320 | 256 | 512 |
| Request Throughput (Req/s) | 2.43 | 8.51 | 7.92 | 7.70 |
| Output Token Throughput (Tok/s) | 243.16 | 850.75 | 791.49 | 769.24 |

#### Input 50 / Output 100 / Video 2 FPS

| Metric | Concurrency 1 | Concurrency 64 | Concurrency 128 | Concurrency 256 |
|---|---:|---:|---:|---:|
| Time To First Token (ms) | 197.78 | 3299.64 | 17742.89 | 44315.20 |
| Request Latency (ms) | 545.44 | 17295.58 | 31353.93 | 58871.04 |
| Request Count (requests) | 50 | 320 | 256 | 512 |
| Request Throughput (Req/s) | 1.82 | 3.64 | 3.63 | 3.60 |
| Output Token Throughput (Tok/s) | 181.93 | 363.22 | 362.34 | 360.02 |


### DGX Spark

#### Input 50 / Output 1 / Video 1 FPS

| Metric | Concurrency 1 | Concurrency 64 | Concurrency 128 | Concurrency 256 |
|---|---:|---:|---:|---:|
| Time To First Token (ms) | 210.91 | 9942.39 | 16533.82 | 32679.75 |
| Request Latency (ms) | 210.91 | 9942.39 | 16533.82 | 32679.75 |
| Request Count (requests) | 50 | 320 | 256 | 512 |
| Request Throughput (Req/s) | 4.71 | 5.81 | 5.83 | 5.90 |
| Output Token Throughput (Tok/s) | 4.71 | 5.81 | 5.83 | 5.90 |

#### Input 50 / Output 1 / Video 2 FPS

| Metric | Concurrency 1 | Concurrency 64 | Concurrency 128 | Concurrency 256 |
|---|---:|---:|---:|---:|
| Time To First Token (ms) | 466.81 | 23496.54 | 39697.16 | 78848.54 |
| Request Latency (ms) | 466.81 | 23496.54 | 39697.16 | 78848.54 |
| Request Count (requests) | 50 | 320 | 256 | 512 |
| Request Throughput (Req/s) | 2.13 | 2.46 | 2.43 | 2.44 |
| Output Token Throughput (Tok/s) | 2.13 | 2.46 | 2.43 | 2.44 |

#### Input 50 / Output 100 / Video 1 FPS

| Metric | Concurrency 1 | Concurrency 64 | Concurrency 128 | Concurrency 256 |
|---|---:|---:|---:|---:|
| Time To First Token (ms) | 213.17 | 25008.31 | 80650.64 | 97058.95 |
| Request Latency (ms) | 1437.40 | 74826.95 | 129088.36 | 121437.58 |
| Request Count (requests) | 50 | 320 | 256 | 512 |
| Request Throughput (Req/s) | 0.69 | 0.84 | 0.85 | 1.70 |
| Output Token Throughput (Tok/s) | 69.46 | 83.67 | 85.44 | 170.26 |

#### Input 50 / Output 100 / Video 2 FPS

| Metric | Concurrency 1 | Concurrency 64 | Concurrency 128 | Concurrency 256 |
|---|---:|---:|---:|---:|
| Time To First Token (ms) | 467.30 | 82773.04 | 96353.83 | 209086.41 |
| Request Latency (ms) | 1874.14 | 127891.29 | 120942.90 | 234143.61 |
| Request Count (requests) | 50 | 320 | 256 | 512 |
| Request Throughput (Req/s) | 0.53 | 0.47 | 0.86 | 0.85 |
| Output Token Throughput (Tok/s) | 53.29 | 47.43 | 85.65 | 85.39 |


### DGX Station

#### Input 50 / Output 1 / Video 1 FPS

| Metric | Concurrency 1 | Concurrency 64 | Concurrency 128 | Concurrency 256 |
|---|---:|---:|---:|---:|
| Time To First Token (ms) | 66.81 | 889.23 | 1406.58 | 2743.83 |
| Request Latency (ms) | 66.81 | 889.23 | 1406.58 | 2743.83 |
| Request Count (requests) | 50 | 320 | 256 | 512 |
| Request Throughput (Req/s) | 14.80 | 66.06 | 71.50 | 73.04 |
| Output Token Throughput (Tok/s) | 14.80 | 66.06 | 71.50 | 73.04 |

#### Input 50 / Output 1 / Video 2 FPS

| Metric | Concurrency 1 | Concurrency 64 | Concurrency 128 | Concurrency 256 |
|---|---:|---:|---:|---:|
| Time To First Token (ms) | 65.58 | 1696.00 | 2858.41 | 6026.97 |
| Request Latency (ms) | 65.58 | 1696.00 | 2858.41 | 6026.97 |
| Request Count (requests) | 50 | 320 | 256 | 512 |
| Request Throughput (Req/s) | 15.05 | 34.48 | 34.73 | 32.68 |
| Output Token Throughput (Tok/s) | 15.05 | 34.48 | 34.73 | 32.68 |

#### Input 50 / Output 100 / Video 1 FPS

| Metric | Concurrency 1 | Concurrency 64 | Concurrency 128 | Concurrency 256 |
|---|---:|---:|---:|---:|
| Time To First Token (ms) | 65.44 | 688.12 | 1254.03 | 2169.38 |
| Request Latency (ms) | 232.81 | 1692.57 | 3086.60 | 5482.81 |
| Request Count (requests) | 50 | 320 | 256 | 512 |
| Request Throughput (Req/s) | 4.26 | 37.64 | 40.85 | 45.39 |
| Output Token Throughput (Tok/s) | 425.65 | 3761.68 | 4082.39 | 4536.51 |

#### Input 50 / Output 100 / Video 2 FPS

| Metric | Concurrency 1 | Concurrency 64 | Concurrency 128 | Concurrency 256 |
|---|---:|---:|---:|---:|
| Time To First Token (ms) | 68.16 | 518.76 | 1237.57 | 3588.64 |
| Request Latency (ms) | 235.47 | 2810.61 | 4980.92 | 8809.02 |
| Request Count (requests) | 50 | 320 | 256 | 512 |
| Request Throughput (Req/s) | 4.22 | 22.61 | 24.93 | 27.05 |
| Output Token Throughput (Tok/s) | 421.37 | 2259.75 | 2491.40 | 2703.66 |


## Embedded

### Jetson AGX Orin 64GB

#### Input 50 / Output 1 / Video 1 FPS

| Metric | Concurrency 1 | Concurrency 64 | Concurrency 128 | Concurrency 256 | Concurrency 512 |
|---|---:|---:|---:|---:|---:|
| Time To First Token (ms) | 812.33 | 43753.89 | 73222.69 | 145921.59 | 291603.11 |
| Request Latency (ms) | 812.33 | 43753.89 | 73222.69 | 145921.59 | 291603.11 |
| Request Count (requests) | 50 | 320 | 256 | 512 | 1023 |
| Request Throughput (Req/s) | 1.23 | 1.32 | 1.32 | 1.32 | 1.32 |
| Output Token Throughput (Tok/s) | 1.23 | 1.32 | 1.32 | 1.32 | 1.32 |

#### Input 50 / Output 1 / Video 2 FPS

| Metric | Concurrency 1 | Concurrency 64 | Concurrency 128 | Concurrency 256 | Concurrency 512 |
|---|---:|---:|---:|---:|---:|
| Time To First Token (ms) | 1757.34 | 98711.99 | 164267.43 | 327606.78 | 653333.12 |
| Request Latency (ms) | 1757.34 | 98711.99 | 164267.43 | 327606.78 | 653333.12 |
| Request Count (requests) | 49 | 315 | 251 | 497 | 999 |
| Request Throughput (Req/s) | 0.56 | 0.58 | 0.58 | 0.57 | 0.57 |
| Output Token Throughput (Tok/s) | 0.56 | 0.58 | 0.58 | 0.57 | 0.57 |

#### Input 50 / Output 100 / Video 1 FPS

| Metric | Concurrency 1 | Concurrency 64 | Concurrency 128 | Concurrency 256 | Concurrency 512 |
|---|---:|---:|---:|---:|---:|
| Time To First Token (ms) | 817.83 | 31654.68 | 100620.12 | 241675.75 | 535671.35 |
| Request Latency (ms) | 2407.67 | 89557.87 | 156871.89 | 301554.12 | 598835.07 |
| Request Count (requests) | 50 | 320 | 256 | 512 | 1024 |
| Request Throughput (Req/s) | 0.41 | 0.70 | 0.71 | 0.69 | 0.67 |
| Output Token Throughput (Tok/s) | 41.46 | 69.88 | 70.84 | 68.95 | 66.48 |

#### Input 50 / Output 100 / Video 2 FPS

| Metric | Concurrency 1 | Concurrency 64 | Concurrency 128 | Concurrency 256 | Concurrency 512 |
|---|---:|---:|---:|---:|---:|
| Time To First Token (ms) | 1762.85 | 120421.74 | 252330.75 | 549698.36 | 1176329.76 |
| Request Latency (ms) | 3645.41 | 184590.29 | 315616.99 | 615247.28 | 1245089.85 |
| Request Count (requests) | 50 | 320 | 256 | 512 | 1024 |
| Request Throughput (Req/s) | 0.27 | 0.33 | 0.33 | 0.33 | 0.31 |
| Output Token Throughput (Tok/s) | 27.37 | 32.89 | 33.11 | 32.54 | 31.30 |


### Jetson AGX Thor T5000

#### Input 50 / Output 1 / Video 1 FPS

| Metric | Concurrency 1 | Concurrency 64 | Concurrency 128 | Concurrency 256 | Concurrency 512 |
|---|---:|---:|---:|---:|---:|
| Time To First Token (ms) | 193.78 | 9650.28 | 16055.54 | 31976.86 | 63704.48 |
| Request Latency (ms) | 193.78 | 9650.28 | 16055.54 | 31976.86 | 63704.48 |
| Request Count (requests) | 50 | 319 | 255 | 510 | 1021 |
| Request Throughput (Req/s) | 5.12 | 5.99 | 5.99 | 6.00 | 6.03 |
| Output Token Throughput (Tok/s) | 5.12 | 5.99 | 5.99 | 6.00 | 6.03 |

#### Input 50 / Output 1 / Video 2 FPS

| Metric | Concurrency 1 | Concurrency 64 | Concurrency 128 | Concurrency 256 | Concurrency 512 |
|---|---:|---:|---:|---:|---:|
| Time To First Token (ms) | 479.97 | 25910.62 | 42817.44 | 86059.36 | 170667.61 |
| Request Latency (ms) | 479.97 | 25910.62 | 42817.44 | 86059.36 | 170667.61 |
| Request Count (requests) | 48 | 306 | 251 | 501 | 983 |
| Request Throughput (Req/s) | 1.99 | 2.15 | 2.20 | 2.20 | 2.17 |
| Output Token Throughput (Tok/s) | 1.99 | 2.15 | 2.20 | 2.20 | 2.17 |

#### Input 50 / Output 100 / Video 1 FPS

| Metric | Concurrency 1 | Concurrency 64 | Concurrency 128 | Concurrency 256 | Concurrency 512 |
|---|---:|---:|---:|---:|---:|
| Time To First Token (ms) | 205.66 | 11256.59 | 36606.36 | 87705.16 | 191782.51 |
| Request Latency (ms) | 1300.70 | 32729.19 | 57856.45 | 109724.99 | 214373.25 |
| Request Count (requests) | 50 | 320 | 256 | 512 | 1024 |
| Request Throughput (Req/s) | 0.77 | 1.90 | 1.89 | 1.88 | 1.86 |
| Output Token Throughput (Tok/s) | 76.69 | 189.72 | 189.19 | 188.20 | 186.20 |

#### Input 50 / Output 100 / Video 2 FPS

| Metric | Concurrency 1 | Concurrency 64 | Concurrency 128 | Concurrency 256 | Concurrency 512 |
|---|---:|---:|---:|---:|---:|
| Time To First Token (ms) | 497.48 | 41910.55 | 86630.36 | 188991.84 | 394208.55 |
| Request Latency (ms) | 1808.68 | 64489.49 | 108646.98 | 211607.53 | 417127.17 |
| Request Count (requests) | 50 | 320 | 256 | 512 | 1024 |
| Request Throughput (Req/s) | 0.55 | 0.94 | 0.95 | 0.94 | 0.94 |
| Output Token Throughput (Tok/s) | 55.17 | 93.62 | 95.20 | 94.37 | 93.86 |


### Jetson AGX Thor T3000 (emulated)

#### Input 50 / Output 1 / Video 1 FPS

| Metric | Concurrency 1 | Concurrency 64 | Concurrency 128 | Concurrency 256 | Concurrency 512 |
|---|---:|---:|---:|---:|---:|
| Time To First Token (ms) | 371.62 | 19670.29 | 32707.28 | 65389.65 | 130786.58 |
| Request Latency (ms) | 371.62 | 19670.29 | 32707.28 | 65389.65 | 130786.58 |
| Request Count (requests) | 50 | 320 | 252 | 511 | 1023 |
| Request Throughput (Req/s) | 2.68 | 2.94 | 2.89 | 2.94 | 2.94 |
| Output Token Throughput (Tok/s) | 2.68 | 2.94 | 2.89 | 2.94 | 2.94 |

#### Input 50 / Output 1 / Video 2 FPS

| Metric | Concurrency 1 | Concurrency 64 | Concurrency 128 | Concurrency 256 | Concurrency 512 |
|---|---:|---:|---:|---:|---:|
| Time To First Token (ms) | 998.11 | 54576.02 | 91001.12 | 181486.26 | 361954.79 |
| Request Latency (ms) | 998.11 | 54576.02 | 91001.12 | 181486.26 | 361954.79 |
| Request Count (requests) | 48 | 305 | 255 | 493 | 989 |
| Request Throughput (Req/s) | 0.96 | 1.01 | 1.06 | 1.02 | 1.02 |
| Output Token Throughput (Tok/s) | 0.96 | 1.01 | 1.06 | 1.02 | 1.02 |

#### Input 50 / Output 100 / Video 1 FPS

| Metric | Concurrency 1 | Concurrency 64 | Concurrency 128 | Concurrency 256 | Concurrency 512 |
|---|---:|---:|---:|---:|---:|
| Time To First Token (ms) | 384.30 | 34144.26 | 64521.15 | 134795.05 | 262205.23 |
| Request Latency (ms) | 1605.57 | 43033.59 | 73344.69 | 143703.49 | 271132.01 |
| Request Count (requests) | 50 | 320 | 256 | 512 | 1024 |
| Request Throughput (Req/s) | 0.62 | 1.38 | 1.37 | 1.37 | 1.43 |
| Output Token Throughput (Tok/s) | 62.16 | 137.81 | 136.90 | 136.75 | 143.24 |

#### Input 50 / Output 100 / Video 2 FPS

| Metric | Concurrency 1 | Concurrency 64 | Concurrency 128 | Concurrency 256 | Concurrency 512 |
|---|---:|---:|---:|---:|---:|
| Time To First Token (ms) | 1006.76 | 89091.45 | 170350.59 | 320286.27 | 647204.40 |
| Request Latency (ms) | 2441.39 | 99508.17 | 180553.54 | 330749.34 | 657701.63 |
| Request Count (requests) | 50 | 320 | 256 | 512 | 1024 |
| Request Throughput (Req/s) | 0.41 | 0.59 | 0.54 | 0.59 | 0.59 |
| Output Token Throughput (Tok/s) | 40.84 | 58.78 | 54.30 | 58.71 | 58.69 |




## NIM serving

These are separate **NIM 2.1.0** measurements, using `nvcr.io/nim/nvidia/cosmos3:2.1.0`, one integrated platform, `NIM_MODEL_TYPE=reasoner`, `NIM_MODEL_VARIANT=edge`, and `NIM_PRECISION=fp8`. The H.264/mp4 workload uses NVDEC; the separately labelled VP9 workload uses CPU decoding through OpenCV. The request contains 50 text tokens and a 30-second video at the stated FPS. Only passed runs are included; failed cells remain blank. These results use a different video-processing path from the vLLM tables.


**H.264 / NVDEC**

### Jetson AGX Thor T5000

#### Input 50 / Output 1 / Video 1 FPS

| Metric | Concurrency 1 | Concurrency 8 | Concurrency 16 | Concurrency 32 | Concurrency 64 |
|---|---:|---:|---:|---:|---:|
| Time To First Token (ms) | 76.19 |  |  | 2233.34 |  |
| Request Latency (ms) | 76.19 |  |  | 2233.34 |  |
| Request Count (requests) | 50 |  |  | 160 |  |
| Request Throughput (Req/s) | 12.91 |  |  | 12.99 |  |
| Output Token Throughput (Tok/s) | 12.91 |  |  | 12.99 |  |

#### Input 50 / Output 1 / Video 2 FPS

| Metric | Concurrency 1 | Concurrency 8 | Concurrency 16 | Concurrency 32 | Concurrency 64 |
|---|---:|---:|---:|---:|---:|
| Time To First Token (ms) | 165.62 |  |  | 5074.28 |  |
| Request Latency (ms) | 165.62 |  |  | 5074.28 |  |
| Request Count (requests) | 50 |  |  | 160 |  |
| Request Throughput (Req/s) | 5.99 |  |  | 5.71 |  |
| Output Token Throughput (Tok/s) | 5.99 |  |  | 5.71 |  |

#### Input 50 / Output 1 / Video 4 FPS

| Metric | Concurrency 1 | Concurrency 8 | Concurrency 16 | Concurrency 32 | Concurrency 64 |
|---|---:|---:|---:|---:|---:|
| Time To First Token (ms) | 433.70 | 3443.94 | 6668.17 | 13314.42 |  |
| Request Latency (ms) | 433.70 | 3443.94 | 6668.17 | 13314.42 |  |
| Request Count (requests) | 50 | 50 | 80 | 160 |  |
| Request Throughput (Req/s) | 2.30 | 2.17 | 2.18 | 2.17 |  |
| Output Token Throughput (Tok/s) | 2.30 | 2.17 | 2.18 | 2.17 |  |

#### Input 50 / Output 1 / Video 8 FPS

| Metric | Concurrency 1 | Concurrency 8 | Concurrency 16 | Concurrency 32 | Concurrency 64 |
|---|---:|---:|---:|---:|---:|
| Time To First Token (ms) | 768.08 | 6253.72 | 12145.30 | 24179.98 | 48564.82 |
| Request Latency (ms) | 768.08 | 6253.72 | 12145.30 | 24179.98 | 48564.82 |
| Request Count (requests) | 50 | 50 | 80 | 160 | 320 |
| Request Throughput (Req/s) | 1.30 | 1.19 | 1.20 | 1.20 | 1.19 |
| Output Token Throughput (Tok/s) | 1.30 | 1.19 | 1.20 | 1.20 | 1.19 |

#### Input 50 / Output 100 / Video 1 FPS

| Metric | Concurrency 1 | Concurrency 8 | Concurrency 16 | Concurrency 32 | Concurrency 64 |
|---|---:|---:|---:|---:|---:|
| Time To First Token (ms) | 95.68 | 323.11 | 471.69 | 678.22 | 1184.78 |
| Request Latency (ms) | 1111.99 | 1985.76 | 3006.12 | 4922.50 | 8908.84 |
| Request Count (requests) | 50 | 50 | 80 | 160 | 320 |
| Request Throughput (Req/s) | 0.90 | 3.77 | 5.25 | 6.38 | 7.01 |
| Output Token Throughput (Tok/s) | 89.74 | 376.63 | 524.47 | 637.99 | 701.04 |

#### Input 50 / Output 100 / Video 2 FPS

| Metric | Concurrency 1 | Concurrency 8 | Concurrency 16 | Concurrency 32 | Concurrency 64 |
|---|---:|---:|---:|---:|---:|
| Time To First Token (ms) | 176.95 | 863.43 | 1162.61 | 1433.59 | 2193.61 |
| Request Latency (ms) | 1286.02 | 3145.99 | 5367.14 | 9608.24 | 18044.57 |
| Request Count (requests) | 50 | 50 | 80 | 160 | 320 |
| Request Throughput (Req/s) | 0.78 | 2.41 | 2.95 | 3.29 | 3.50 |
| Output Token Throughput (Tok/s) | 77.49 | 240.64 | 295.29 | 328.59 | 349.02 |

#### Input 50 / Output 100 / Video 4 FPS

| Metric | Concurrency 1 | Concurrency 8 | Concurrency 16 | Concurrency 32 | Concurrency 64 |
|---|---:|---:|---:|---:|---:|
| Time To First Token (ms) | 474.19 | 2251.66 | 3163.11 | 3600.65 | 5725.44 |
| Request Latency (ms) | 1656.93 | 6155.39 | 11369.79 | 21655.10 | 41772.13 |
| Request Count (requests) | 50 | 50 | 80 | 160 | 320 |
| Request Throughput (Req/s) | 0.60 | 1.24 | 1.40 | 1.47 | 1.52 |
| Output Token Throughput (Tok/s) | 60.25 | 124.22 | 139.94 | 146.71 | 151.86 |

#### Input 50 / Output 100 / Video 8 FPS

| Metric | Concurrency 1 | Concurrency 8 | Concurrency 16 | Concurrency 32 | Concurrency 64 |
|---|---:|---:|---:|---:|---:|
| Time To First Token (ms) | 824.17 | 3913.93 | 4987.63 | 6729.77 | 11692.71 |
| Request Latency (ms) | 2101.37 | 10003.65 | 19043.31 | 36345.47 | 71869.23 |
| Request Count (requests) | 50 | 50 | 80 | 160 | 320 |
| Request Throughput (Req/s) | 0.48 | 0.77 | 0.83 | 0.87 | 0.88 |
| Output Token Throughput (Tok/s) | 47.53 | 77.28 | 83.43 | 86.83 | 88.09 |


### Jetson AGX Thor T3000 (emulated)

#### Input 50 / Output 1 / Video 1 FPS

| Metric | Concurrency 1 | Concurrency 8 | Concurrency 16 | Concurrency 32 | Concurrency 64 |
|---|---:|---:|---:|---:|---:|
| Time To First Token (ms) | 144.11 | 1072.71 |  |  |  |
| Request Latency (ms) | 144.11 | 1072.71 |  |  |  |
| Request Count (requests) | 50 | 50 |  |  |  |
| Request Throughput (Req/s) | 6.87 | 7.02 |  |  |  |
| Output Token Throughput (Tok/s) | 6.87 | 7.02 |  |  |  |

#### Input 50 / Output 1 / Video 2 FPS

| Metric | Concurrency 1 | Concurrency 8 | Concurrency 16 | Concurrency 32 | Concurrency 64 |
|---|---:|---:|---:|---:|---:|
| Time To First Token (ms) | 312.42 | 2537.75 | 4952.81 | 9881.43 |  |
| Request Latency (ms) | 312.42 | 2537.75 | 4952.81 | 9881.43 |  |
| Request Count (requests) | 50 | 50 | 80 | 160 |  |
| Request Throughput (Req/s) | 3.18 | 3.02 | 3.05 | 3.06 |  |
| Output Token Throughput (Tok/s) | 3.18 | 3.02 | 3.05 | 3.06 |  |

#### Input 50 / Output 100 / Video 1 FPS

| Metric | Concurrency 1 | Concurrency 8 | Concurrency 16 | Concurrency 32 | Concurrency 64 |
|---|---:|---:|---:|---:|---:|
| Time To First Token (ms) | 157.40 | 675.64 | 1021.08 | 1250.40 | 1946.60 |
| Request Latency (ms) | 1308.63 | 2858.19 | 4645.11 | 8232.13 | 15262.21 |
| Request Count (requests) | 50 | 50 | 80 | 160 | 320 |
| Request Throughput (Req/s) | 0.76 | 2.63 | 3.42 | 3.86 | 4.14 |
| Output Token Throughput (Tok/s) | 76.23 | 262.71 | 340.32 | 385.42 | 413.97 |

#### Input 50 / Output 100 / Video 2 FPS

| Metric | Concurrency 1 | Concurrency 8 | Concurrency 16 | Concurrency 32 | Concurrency 64 |
|---|---:|---:|---:|---:|---:|
| Time To First Token (ms) | 315.79 | 1704.44 | 3098.06 | 5407.81 | 16577.08 |
| Request Latency (ms) | 1542.82 | 5030.09 | 8627.49 | 16200.44 | 31336.51 |
| Request Count (requests) | 50 | 50 | 80 | 160 | 320 |
| Request Throughput (Req/s) | 0.65 | 1.52 | 1.85 | 1.97 | 1.96 |
| Output Token Throughput (Tok/s) | 64.67 | 152.04 | 184.85 | 196.89 | 195.81 |


**VP9 / CPU decoding**

### Jetson AGX Thor T5000

#### Input 50 / Output 100 / Video 1 FPS

| Metric | Concurrency 1 |
|---|---:|
| Time To First Token (ms) | 88.96 |
| Request Latency (ms) | 1127.42 |
| Request Count (requests) | 50 |
| Request Throughput (Req/s) | 0.89 |
| Output Token Throughput (Tok/s) | 88.50 |
