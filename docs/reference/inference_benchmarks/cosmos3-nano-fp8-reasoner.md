# Cosmos3-Nano-FP8 Reasoner Inference Benchmarks

[Back to the inference benchmark index](../inference_benchmarks.md)

These tables measure text output through **vLLM** and, in separate sections, **NIM**, with FP8 weights. Latencies are milliseconds; request and output-token throughput are per second. Lower latency and higher throughput are better. Each system uses **one GPU or one integrated platform**; concurrency is the number of simultaneous client requests.

The vLLM workload uses 50 text input tokens and one 374×374, 30-second synthetic video: 30 frames at 1 FPS or 60 frames at 2 FPS. Requested output length is 1 or 100 tokens. KV-cache precision is separate from model-weight precision. Blank cells have no validated published measurement and must not be read as lack of support.

See [measurement settings and reproduction limits](benchmark_methodology.md). Displayed values retain the source's available precision; trailing zeros do not imply additional measurement accuracy.

The synthetic-video workload is served with AIPerf. Exact runtime builds and complete prompt payloads were not retained with every measurement; these results describe the recorded configuration rather than a fully pinned reproduction environment. Some Super single-token runs returned empty replies with the `qwen3` parser, and some H20 runs timed out. Missing results are left blank. Where single-token metrics are available despite empty replies, they describe the successful requests shown by the request-count row and do not establish a zero-error serving rate.

## Known anomalies and comparison limits

NIM RTX PRO 6000 Blackwell, input 100 / output 1 / video 4 FPS, rises from 3.94 to 8.37 requests/s at concurrency 128 to 256 after a near plateau at concurrency 64 to 128. Request latency falls from 26,336.40 to 24,627.21 ms. These source-matching values need scheduling, batch-composition, and run-duration checks; doubling concurrency alone does not establish an erroneous result. Separate NIM Reasoner profiles and some datacenter/GB300 source grids remain incomplete. See the [anomaly review](benchmark_anomaly_review.md).

## Datacenter

### B200

#### Input 50 / Output 1 / Video 1 FPS

| Metric | Concurrency 1 | Concurrency 64 | Concurrency 128 | Concurrency 256 |
|---|---:|---:|---:|---:|
| Time To First Token (ms) | 127.70 | 1126.40 | 1996.10 | 3800.40 |
| Request Latency (ms) | 127.70 | 1126.40 | 1996.10 | 3800.40 |
| Request Count (requests) | 50 | 320 | 256 | 512 |
| Request Throughput (Req/s) | 7.70 | 51.80 | 50.00 | 51.80 |
| Output Token Throughput (Tok/s) | 8.00 | 52.00 | 50.00 | 52.00 |

#### Input 50 / Output 1 / Video 2 FPS

| Metric | Concurrency 1 | Concurrency 64 | Concurrency 128 | Concurrency 256 |
|---|---:|---:|---:|---:|
| Time To First Token (ms) | 152.70 | 2306.10 | 3968.90 | 7767.60 |
| Request Latency (ms) | 152.70 | 2306.10 | 3968.90 | 7767.60 |
| Request Count (requests) | 50 | 320 | 256 | 512 |
| Request Throughput (Req/s) | 6.50 | 25.00 | 24.70 | 25.10 |
| Output Token Throughput (Tok/s) | 6.00 | 25.00 | 25.00 | 25.00 |

#### Input 50 / Output 100 / Video 1 FPS

| Metric | Concurrency 1 | Concurrency 64 | Concurrency 128 | Concurrency 256 |
|---|---:|---:|---:|---:|
| Time To First Token (ms) | 137.80 | 957.70 | 1629.90 | 2999.80 |
| Request Latency (ms) | 449.00 | 2091.40 | 3733.10 | 6851.90 |
| Request Count (requests) | 50 | 320 | 256 | 512 |
| Request Throughput (Req/s) | 2.20 | 28.90 | 34.10 | 37.00 |
| Output Token Throughput (Tok/s) | 222.00 | 2886.00 | 3405.00 | 3697.00 |


### B300

#### Input 50 / Output 1 / Video 1 FPS

| Metric | Concurrency 1 | Concurrency 64 | Concurrency 128 | Concurrency 256 |
|---|---:|---:|---:|---:|
| Time To First Token (ms) | 96.40 | 1160.20 | 2048.40 | 3725.20 |
| Request Latency (ms) | 96.40 | 1160.20 | 2048.40 | 3725.20 |
| Request Count (requests) | 50 | 320 | 256 | 512 |
| Request Throughput (Req/s) | 10.30 | 50.70 | 48.80 | 53.30 |
| Output Token Throughput (Tok/s) | 10.00 | 51.00 | 49.00 | 53.00 |

#### Input 50 / Output 1 / Video 2 FPS

| Metric | Concurrency 1 | Concurrency 64 | Concurrency 128 | Concurrency 256 |
|---|---:|---:|---:|---:|
| Time To First Token (ms) | 115.20 | 2175.30 | 3756.50 | 7220.70 |
| Request Latency (ms) | 115.20 | 2175.30 | 3756.50 | 7220.70 |
| Request Count (requests) | 50 | 320 | 256 | 512 |
| Request Throughput (Req/s) | 8.60 | 26.70 | 25.80 | 26.90 |
| Output Token Throughput (Tok/s) | 9.00 | 27.00 | 26.00 | 27.00 |

#### Input 50 / Output 100 / Video 1 FPS

| Metric | Concurrency 1 | Concurrency 64 | Concurrency 128 | Concurrency 256 |
|---|---:|---:|---:|---:|
| Time To First Token (ms) | 108.50 | 1092.00 | 1673.10 | 2970.90 |
| Request Latency (ms) | 414.70 | 2190.00 | 3699.80 | 6887.90 |
| Request Count (requests) | 50 | 320 | 256 | 512 |
| Request Throughput (Req/s) | 2.40 | 27.50 | 34.40 | 33.40 |
| Output Token Throughput (Tok/s) | 240.00 | 2749.00 | 3435.00 | 3335.00 |


### H20

#### Input 50 / Output 1 / Video 1 FPS

| Metric | Concurrency 1 | Concurrency 64 | Concurrency 128 | Concurrency 256 |
|---|---:|---:|---:|---:|
| Time To First Token (ms) | 243.50 | 8060.50 | 13550.80 | 26907.40 |
| Request Latency (ms) | 243.50 | 8060.50 | 13550.80 | 26907.40 |
| Request Count (requests) | 50 | 320 | 256 | 512 |
| Request Throughput (Req/s) | 4.10 | 7.20 | 7.20 | 7.20 |
| Output Token Throughput (Tok/s) | 4.00 | 7.00 | 7.00 | 7.00 |

#### Input 50 / Output 1 / Video 2 FPS

| Metric | Concurrency 1 | Concurrency 64 | Concurrency 128 | Concurrency 256 |
|---|---:|---:|---:|---:|
| Time To First Token (ms) | 424.20 | 17074.70 | 28612.30 | 57130.30 |
| Request Latency (ms) | 424.20 | 17074.70 | 28612.30 | 57130.30 |
| Request Count (requests) | 50 | 320 | 256 | 512 |
| Request Throughput (Req/s) | 2.30 | 3.40 | 3.40 | 3.40 |
| Output Token Throughput (Tok/s) | 2.00 | 3.00 | 3.00 | 3.00 |

#### Input 50 / Output 100 / Video 1 FPS

| Metric | Concurrency 1 | Concurrency 64 | Concurrency 128 | Concurrency 256 |
|---|---:|---:|---:|---:|
| Time To First Token (ms) | 242.00 | 4217.60 | 6331.90 | 14186.70 |
| Request Latency (ms) | 672.60 | 10940.10 | 21757.50 | 43833.30 |
| Request Count (requests) | 50 | 320 | 256 | 512 |
| Request Throughput (Req/s) | 1.50 | 5.80 | 5.80 | 5.70 |
| Output Token Throughput (Tok/s) | 148.00 | 584.00 | 584.00 | 569.00 |


### H100 80GB HBM3

#### Input 50 / Output 1 / Video 1 FPS

| Metric | Concurrency 1 | Concurrency 64 | Concurrency 128 |
|---|---:|---:|---:|
| Time To First Token (ms) | 130.20 | 2186.80 | 3707.50 |
| Request Latency (ms) | 130.20 | 2186.80 | 3707.50 |
| Request Count (requests) | 50 | 320 | 256 |
| Request Throughput (Req/s) | 7.60 | 26.60 | 26.30 |
| Output Token Throughput (Tok/s) | 8.00 | 27.00 | 26.00 |

#### Input 50 / Output 1 / Video 2 FPS

| Metric | Concurrency 1 | Concurrency 64 | Concurrency 128 |
|---|---:|---:|---:|
| Time To First Token (ms) | 192.70 | 4603.40 | 7709.70 |
| Request Latency (ms) | 192.70 | 4603.40 | 7709.70 |
| Request Count (requests) | 50 | 320 | 256 |
| Request Throughput (Req/s) | 5.10 | 12.60 | 12.60 |
| Output Token Throughput (Tok/s) | 5.00 | 13.00 | 13.00 |

#### Input 50 / Output 100 / Video 1 FPS

| Metric | Concurrency 1 | Concurrency 64 | Concurrency 128 |
|---|---:|---:|---:|
| Time To First Token (ms) | 129.50 | 1437.70 | 2150.40 |
| Request Latency (ms) | 575.80 | 3696.00 | 6761.40 |
| Request Count (requests) | 50 | 320 | 256 |
| Request Throughput (Req/s) | 1.70 | 17.20 | 18.70 |
| Output Token Throughput (Tok/s) | 173.00 | 1724.00 | 1867.00 |


## Enterprise

### DGX Spark

#### Input 50 / Output 1 / Video 1 FPS

| Metric | Concurrency 1 | Concurrency 64 | Concurrency 128 | Concurrency 256 |
|---|---:|---:|---:|---:|
| Time To First Token (ms) | 332.50 | 16537.70 | 27161.00 | 54829.50 |
| Request Latency (ms) | 332.50 | 16537.70 | 27161.00 | 54829.50 |
| Request Count (requests) | 50 | 320 | 256 | 512 |
| Request Throughput (Req/s) | 3.00 | 3.50 | 3.60 | 3.50 |
| Output Token Throughput (Tok/s) | 3.00 | 3.50 | 3.60 | 3.50 |

#### Input 50 / Output 1 / Video 2 FPS

| Metric | Concurrency 1 | Concurrency 64 | Concurrency 128 | Concurrency 256 |
|---|---:|---:|---:|---:|
| Time To First Token (ms) | 654.90 | 35324.40 | 58009.40 | 115832.30 |
| Request Latency (ms) | 654.90 | 35324.40 | 58009.40 | 115832.30 |
| Request Count (requests) | 50 | 320 | 256 | 512 |
| Request Throughput (Req/s) | 1.50 | 1.60 | 1.70 | 1.70 |
| Output Token Throughput (Tok/s) | 1.50 | 1.60 | 1.70 | 1.70 |

#### Input 50 / Output 100 / Video 1 FPS

| Metric | Concurrency 1 | Concurrency 64 | Concurrency 128 | Concurrency 256 |
|---|---:|---:|---:|---:|
| Time To First Token (ms) | 336.10 | 4876.40 | 39855.40 |  |
| Request Latency (ms) | 4418.50 | 49985.80 | 103975.70 |  |
| Request Count (requests) | 50 | 320 | 256 |  |
| Request Throughput (Req/s) | 0.20 | 1.30 | 1.20 |  |
| Output Token Throughput (Tok/s) | 22.60 | 126.60 | 115.40 |  |


### DGX Station

#### Input 50 / Output 1 / Video 1 FPS

| Metric | Concurrency 1 | Concurrency 64 | Concurrency 128 | Concurrency 256 |
|---|---:|---:|---:|---:|
| Time To First Token (ms) | 73.60 | 921.10 | 1569.30 | 3290.00 |
| Request Latency (ms) | 73.60 | 921.10 | 1569.30 | 3290.00 |
| Request Count (requests) | 50 | 320 | 256 | 512 |
| Request Throughput (Req/s) | 13.50 | 63.60 | 63.20 | 60.60 |
| Output Token Throughput (Tok/s) | 13.50 | 63.60 | 63.20 | 60.60 |

#### Input 50 / Output 1 / Video 2 FPS

| Metric | Concurrency 1 | Concurrency 64 | Concurrency 128 | Concurrency 256 |
|---|---:|---:|---:|---:|
| Time To First Token (ms) | 72.90 | 1883.70 | 3228.90 | 6297.10 |
| Request Latency (ms) | 72.90 | 1883.70 | 3228.90 | 6297.10 |
| Request Count (requests) | 50 | 320 | 256 | 512 |
| Request Throughput (Req/s) | 13.60 | 30.70 | 30.10 | 30.90 |
| Output Token Throughput (Tok/s) | 13.60 | 30.70 | 30.10 | 30.90 |

#### Input 50 / Output 100 / Video 1 FPS

| Metric | Concurrency 1 | Concurrency 64 | Concurrency 128 | Concurrency 256 |
|---|---:|---:|---:|---:|
| Time To First Token (ms) | 85.40 | 703.40 | 1266.90 | 2409.20 |
| Request Latency (ms) | 389.10 | 1759.70 | 3144.20 | 5804.00 |
| Request Count (requests) | 50 | 320 | 256 | 512 |
| Request Throughput (Req/s) | 2.60 | 36.30 | 40.40 | 43.60 |
| Output Token Throughput (Tok/s) | 256.00 | 3626.20 | 4039.20 | 4355.60 |

#### Input 50 / Output 100 / Video 2 FPS

| Metric | Concurrency 1 | Concurrency 64 | Concurrency 128 | Concurrency 256 |
|---|---:|---:|---:|---:|
| Time To First Token (ms) | 80.60 | 1246.60 | 1752.70 | 2790.20 |
| Request Latency (ms) | 393.50 | 3092.30 | 5754.90 | 10811.10 |
| Request Count (requests) | 50 | 320 | 256 | 512 |
| Request Throughput (Req/s) | 2.50 | 20.60 | 22.00 | 23.20 |
| Output Token Throughput (Tok/s) | 253.10 | 2062.40 | 2197.70 | 2319.80 |


## GB300 comparison — qwen3 parser

### GB300

#### Input 50 / Output 1 / Video 1 FPS

| Metric | Concurrency 1 | Concurrency 64 | Concurrency 128 | Concurrency 256 |
|---|---:|---:|---:|---:|
| Time To First Token (ms) | 82.20 | 905.60 | 1624.30 | 3003.20 |
| Request Latency (ms) | 82.20 | 905.60 | 1624.30 | 3003.20 |
| Request Count (requests) | 50 | 320 | 256 | 512 |
| Request Throughput (Req/s) | 12.10 | 65.20 | 62.30 | 60.80 |
| Output Token Throughput (Tok/s) | 12.00 | 65.00 | 62.00 | 61.00 |

#### Input 50 / Output 1 / Video 2 FPS

| Metric | Concurrency 1 | Concurrency 64 | Concurrency 128 | Concurrency 256 |
|---|---:|---:|---:|---:|
| Time To First Token (ms) | 91.60 | 1862.50 | 3214.70 | 6257.40 |
| Request Latency (ms) | 91.60 | 1862.50 | 3214.70 | 6257.40 |
| Request Count (requests) | 50 | 320 | 256 | 512 |
| Request Throughput (Req/s) | 10.80 | 30.90 | 28.30 | 31.10 |
| Output Token Throughput (Tok/s) | 11.00 | 31.00 | 28.00 | 31.00 |

#### Input 50 / Output 100 / Video 1 FPS

| Metric | Concurrency 1 | Concurrency 64 | Concurrency 128 | Concurrency 256 |
|---|---:|---:|---:|---:|
| Time To First Token (ms) | 91.80 | 828.60 | 1269.60 | 2617.20 |
| Request Latency (ms) | 412.30 | 1980.30 | 3088.50 | 6350.10 |
| Request Count (requests) | 50 | 320 | 256 | 512 |
| Request Throughput (Req/s) | 2.40 | 32.20 | 41.10 | 39.90 |
| Output Token Throughput (Tok/s) | 242.00 | 3222.00 | 4109.00 |  |


### B300

#### Input 50 / Output 1 / Video 1 FPS

| Metric | Concurrency 1 | Concurrency 64 | Concurrency 128 | Concurrency 256 |
|---|---:|---:|---:|---:|
| Time To First Token (ms) | 96.40 | 1160.20 | 2048.40 | 3725.20 |
| Request Latency (ms) | 96.40 | 1160.20 | 2048.40 | 3725.20 |
| Request Count (requests) | 50 | 320 | 256 | 512 |
| Request Throughput (Req/s) | 10.30 | 50.70 | 48.80 | 53.30 |
| Output Token Throughput (Tok/s) | 10.00 | 51.00 | 49.00 | 53.00 |

#### Input 50 / Output 1 / Video 2 FPS

| Metric | Concurrency 1 | Concurrency 64 | Concurrency 128 | Concurrency 256 |
|---|---:|---:|---:|---:|
| Time To First Token (ms) | 115.20 | 2175.30 | 3756.50 | 7220.70 |
| Request Latency (ms) | 115.20 | 2175.30 | 3756.50 | 7220.70 |
| Request Count (requests) | 50 | 320 | 256 | 512 |
| Request Throughput (Req/s) | 8.60 | 26.70 | 25.80 | 26.90 |
| Output Token Throughput (Tok/s) | 9.00 | 27.00 | 26.00 | 27.00 |

#### Input 50 / Output 100 / Video 1 FPS

| Metric | Concurrency 1 | Concurrency 64 | Concurrency 128 | Concurrency 256 |
|---|---:|---:|---:|---:|
| Time To First Token (ms) | 108.50 | 1092.00 | 1673.10 | 2970.90 |
| Request Latency (ms) | 414.70 | 2190.00 | 3699.80 | 6887.90 |
| Request Count (requests) | 50 | 320 | 256 | 512 |
| Request Throughput (Req/s) | 2.40 | 27.50 | 34.40 | 33.40 |
| Output Token Throughput (Tok/s) | 240.00 | 2749.00 | 3435.00 | 3335.00 |


The available datacenter and GB300 tables do not yet expose every model/workload combination. Unavailable source cells have not been reconstructed from speedup summaries or other models.



## NIM serving

These FP8 measurements use NIM 1.7.1.rc3 with one GPU, two synthetic-video workloads: 100 text input tokens / 1 output token with 608×320 video at 4 FPS, and 1,000 input tokens / 100 output tokens with 1312×736 video at 2 FPS. Both videos last 10 seconds. All sampled frames are retained, pruning is zero, KV and multimodal caches are disabled, and strict output length is enabled. The two workloads target approximately 4K and 9K visual tokens respectively. These tables are separate from the vLLM 50-token, 30-second workload. The measured NIM build is prerelease software; results with release builds can differ.


### B300

#### Input 100 / Output 1 / Video 4 FPS

| Metric | Concurrency 1 | Concurrency 8 | Concurrency 16 | Concurrency 32 | Concurrency 64 | Concurrency 128 | Concurrency 256 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Time To First Token (ms) | 810.05 | 3713.85 | 4618.77 | 6403.89 | 8686.51 | 11985.89 | 23263.69 |
| Request Latency (ms) | 810.05 | 3713.85 | 4618.77 | 6403.89 | 8686.51 | 11985.89 | 23263.69 |
| Request Count (requests) | 10 | 16 | 32 | 64 | 128 | 256 | 512 |
| Request Throughput (Req/s) | 1.23 | 2.05 | 3.26 | 4.57 | 6.46 | 8.34 | 8.07 |
| Output Token Throughput (Tok/s) | 1.23 | 2.05 | 3.26 | 4.57 | 6.46 | 8.34 | 8.07 |

#### Input 1000 / Output 100 / Video 2 FPS

| Metric | Concurrency 1 | Concurrency 8 | Concurrency 16 | Concurrency 32 | Concurrency 64 | Concurrency 128 | Concurrency 256 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Time To First Token (ms) | 1222.27 | 4990.84 | 7134.10 | 11161.76 | 16151.49 | 40393.37 | 69978.16 |
| Request Latency (ms) | 1557.04 | 5794.34 | 8640.07 | 13921.45 | 22988.78 | 43016.72 | 71751.07 |
| Request Count (requests) | 10 | 16 | 32 | 64 | 128 | 256 | 512 |
| Request Throughput (Req/s) | 0.64 | 1.28 | 1.70 | 2.05 | 2.70 | 2.36 | 2.60 |
| Output Token Throughput (Tok/s) | 64.14 | 128.02 | 169.59 | 204.69 | 270.49 | 236.09 | 260.03 |


### B200

#### Input 100 / Output 1 / Video 4 FPS

| Metric | Concurrency 1 | Concurrency 8 | Concurrency 16 | Concurrency 32 | Concurrency 64 | Concurrency 128 | Concurrency 256 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Time To First Token (ms) | 580.86 | 2614.20 | 3403.35 | 4522.26 | 6440.32 | 11379.03 | 22779.10 |
| Request Latency (ms) | 580.86 | 2614.20 | 3403.35 | 4522.26 | 6440.32 | 11379.03 | 22779.10 |
| Request Count (requests) | 10 | 16 | 32 | 64 | 128 | 256 | 512 |
| Request Throughput (Req/s) | 1.71 | 2.89 | 4.32 | 6.28 | 8.14 | 8.53 | 8.45 |
| Output Token Throughput (Tok/s) | 1.71 | 2.89 | 4.32 | 6.28 | 8.14 | 8.53 | 8.45 |

#### Input 1000 / Output 100 / Video 2 FPS

| Metric | Concurrency 1 | Concurrency 8 | Concurrency 16 | Concurrency 32 | Concurrency 64 | Concurrency 128 | Concurrency 256 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Time To First Token (ms) | 938.52 | 3549.86 | 5007.85 | 8053.35 | 9649.33 | 22778.75 | 45605.21 |
| Request Latency (ms) | 1283.03 | 4648.92 | 6640.98 | 11576.41 | 19389.96 | 33260.34 | 57018.59 |
| Request Count (requests) | 10 | 16 | 32 | 64 | 128 | 256 | 512 |
| Request Throughput (Req/s) | 0.78 | 1.71 | 2.24 | 2.64 | 3.24 | 3.52 | 3.78 |
| Output Token Throughput (Tok/s) | 77.81 | 171.28 | 224.05 | 264.04 | 323.89 | 352.11 | 378.21 |


### H100 80GB HBM3

#### Input 100 / Output 1 / Video 4 FPS

| Metric | Concurrency 1 | Concurrency 8 | Concurrency 16 | Concurrency 32 | Concurrency 64 | Concurrency 128 | Concurrency 256 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Time To First Token (ms) | 714.24 | 3235.08 | 4178.95 | 6004.23 | 8841.76 | 14661.86 | 27977.78 |
| Request Latency (ms) | 714.24 | 3235.08 | 4178.95 | 6004.23 | 8841.76 | 14661.86 | 27977.78 |
| Request Count (requests) | 10 | 16 | 32 | 64 | 128 | 256 | 512 |
| Request Throughput (Req/s) | 1.39 | 2.33 | 3.52 | 4.68 | 5.97 | 6.89 | 7.09 |
| Output Token Throughput (Tok/s) | 1.39 | 2.33 | 3.52 | 4.68 | 5.97 | 6.89 | 7.09 |

#### Input 1000 / Output 100 / Video 2 FPS

| Metric | Concurrency 1 | Concurrency 8 | Concurrency 16 | Concurrency 32 | Concurrency 64 | Concurrency 128 | Concurrency 256 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Time To First Token (ms) | 1120.68 | 4546.67 | 6498.66 | 10375.19 | 13530.42 | 32797.89 | 70642.07 |
| Request Latency (ms) | 1618.98 | 6240.33 | 9735.78 | 16800.21 | 28966.84 | 50632.96 | 89237.13 |
| Request Count (requests) | 10 | 16 | 32 | 64 | 128 | 256 | 512 |
| Request Throughput (Req/s) | 0.62 | 1.23 | 1.60 | 1.87 | 2.16 | 2.29 | 2.40 |
| Output Token Throughput (Tok/s) | 61.68 | 123.30 | 160.09 | 186.86 | 215.91 | 228.69 | 240.30 |


### H100 NVL

#### Input 100 / Output 1 / Video 4 FPS

| Metric | Concurrency 1 | Concurrency 8 | Concurrency 16 | Concurrency 32 | Concurrency 64 | Concurrency 128 | Concurrency 256 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Time To First Token (ms) | 661.16 | 3307.89 | 4490.25 | 7347.67 | 12859.30 | 23579.74 | 42650.06 |
| Request Latency (ms) | 661.16 | 3307.89 | 4490.25 | 7347.67 | 12859.30 | 23579.74 | 42650.06 |
| Request Count (requests) | 10 | 16 | 32 | 64 | 128 | 256 | 512 |
| Request Throughput (Req/s) | 1.50 | 2.26 | 3.17 | 3.78 | 4.18 | 4.52 | 4.75 |
| Output Token Throughput (Tok/s) | 1.50 | 2.26 | 3.17 | 3.78 | 4.18 | 4.52 | 4.75 |

#### Input 1000 / Output 100 / Video 2 FPS

| Metric | Concurrency 1 | Concurrency 8 | Concurrency 16 | Concurrency 32 | Concurrency 64 | Concurrency 128 | Concurrency 256 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Time To First Token (ms) | 1137.81 | 4852.19 | 7878.98 | 13162.60 | 18208.05 | 47128.00 | 104803.14 |
| Request Latency (ms) | 1622.16 | 7135.18 | 11819.80 | 21407.79 | 38851.10 | 70453.38 | 129092.26 |
| Request Count (requests) | 10 | 16 | 32 | 64 | 128 | 256 | 512 |
| Request Throughput (Req/s) | 0.62 | 1.12 | 1.30 | 1.45 | 1.61 | 1.66 | 1.69 |
| Output Token Throughput (Tok/s) | 61.51 | 111.76 | 129.76 | 144.96 | 161.02 | 165.68 | 168.89 |


### H200 141GB HBM3

#### Input 100 / Output 1 / Video 4 FPS

| Metric | Concurrency 1 | Concurrency 8 | Concurrency 16 | Concurrency 32 | Concurrency 64 | Concurrency 128 | Concurrency 256 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Time To First Token (ms) | 698.61 | 3219.82 | 4118.04 | 5837.14 | 9013.31 | 15088.12 | 28922.02 |
| Request Latency (ms) | 698.61 | 3219.82 | 4118.04 | 5837.14 | 9013.31 | 15088.12 | 28922.02 |
| Request Count (requests) | 10 | 16 | 32 | 64 | 128 | 256 | 512 |
| Request Throughput (Req/s) | 1.42 | 2.34 | 3.58 | 4.76 | 5.80 | 6.59 | 6.81 |
| Output Token Throughput (Tok/s) | 1.42 | 2.34 | 3.58 | 4.76 | 5.80 | 6.59 | 6.81 |

#### Input 1000 / Output 100 / Video 2 FPS

| Metric | Concurrency 1 | Concurrency 8 | Concurrency 16 | Concurrency 32 | Concurrency 64 | Concurrency 128 | Concurrency 256 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Time To First Token (ms) | 1111.66 | 4710.34 | 6744.06 | 10058.34 | 13236.71 | 31223.79 | 67960.39 |
| Request Latency (ms) | 1525.63 | 6266.61 | 9817.52 | 16251.39 | 28255.95 | 48101.83 | 85687.45 |
| Request Count (requests) | 10 | 16 | 32 | 64 | 128 | 256 | 512 |
| Request Throughput (Req/s) | 0.65 | 1.23 | 1.59 | 1.93 | 2.22 | 2.41 | 2.50 |
| Output Token Throughput (Tok/s) | 65.38 | 122.84 | 159.00 | 193.29 | 221.63 | 241.17 | 250.18 |


### H200 NVL

#### Input 100 / Output 1 / Video 4 FPS

| Metric | Concurrency 1 | Concurrency 8 | Concurrency 16 | Concurrency 32 | Concurrency 64 | Concurrency 128 | Concurrency 256 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Time To First Token (ms) | 656.87 | 2936.53 | 3848.57 | 5911.99 | 8912.31 | 15892.62 | 40480.77 |
| Request Latency (ms) | 656.87 | 2936.53 | 3848.57 | 5911.99 | 8912.31 | 15892.62 | 40480.77 |
| Request Count (requests) | 10 | 16 | 32 | 64 | 128 | 256 | 512 |
| Request Throughput (Req/s) | 1.51 | 2.57 | 3.78 | 4.73 | 5.91 | 6.39 | 5.21 |
| Output Token Throughput (Tok/s) | 1.51 | 2.57 | 3.78 | 4.73 | 5.91 | 6.39 | 5.21 |

#### Input 1000 / Output 100 / Video 2 FPS

| Metric | Concurrency 1 | Concurrency 8 | Concurrency 16 | Concurrency 32 | Concurrency 64 | Concurrency 128 | Concurrency 256 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Time To First Token (ms) | 1046.35 | 4259.48 | 6182.73 | 10160.95 | 13379.34 | 40149.29 | 90894.78 |
| Request Latency (ms) | 1501.70 | 6207.20 | 9404.00 | 17108.45 | 30387.26 | 58538.56 | 111048.63 |
| Request Count (requests) | 10 | 16 | 32 | 64 | 128 | 256 | 512 |
| Request Throughput (Req/s) | 0.66 | 1.24 | 1.62 | 1.83 | 2.06 | 1.99 | 1.97 |
| Output Token Throughput (Tok/s) | 66.49 | 124.14 | 162.28 | 182.54 | 205.91 | 199.22 | 196.58 |


### RTX PRO 6000 Blackwell

#### Input 100 / Output 1 / Video 4 FPS

| Metric | Concurrency 1 | Concurrency 8 | Concurrency 16 | Concurrency 32 | Concurrency 64 | Concurrency 128 | Concurrency 256 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Time To First Token (ms) | 685.45 | 3606.36 | 5739.20 | 8814.70 | 13492.81 | 26336.40 | 24627.21 |
| Request Latency (ms) | 685.45 | 3606.36 | 5739.20 | 8814.70 | 13492.81 | 26336.40 | 24627.21 |
| Request Count (requests) | 10 | 16 | 32 | 64 | 128 | 256 | 512 |
| Request Throughput (Req/s) | 1.45 | 2.07 | 2.54 | 3.16 | 3.89 | 3.94 | 8.37 |
| Output Token Throughput (Tok/s) | 1.45 | 2.07 | 2.54 | 3.16 | 3.89 | 3.94 | 8.37 |

#### Input 1000 / Output 100 / Video 2 FPS

| Metric | Concurrency 1 | Concurrency 8 | Concurrency 16 | Concurrency 32 | Concurrency 64 | Concurrency 128 | Concurrency 256 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Time To First Token (ms) | 1785.81 | 5309.84 | 12145.39 | 21652.43 | 22939.67 | 51486.35 | 130540.84 |
| Request Latency (ms) | 2607.86 | 8479.48 | 17488.79 | 31015.53 | 48800.22 | 80472.87 | 161526.48 |
| Request Count (requests) | 10 | 16 | 32 | 64 | 128 | 256 | 512 |
| Request Throughput (Req/s) | 0.38 | 0.94 | 0.87 | 0.96 | 1.28 | 1.44 | 1.34 |
| Output Token Throughput (Tok/s) | 38.29 | 93.75 | 87.01 | 96.26 | 128.14 | 143.62 | 134.40 |
