# Cosmos3-Super-FP8 Reasoner Inference Benchmarks

[Back to the inference benchmark index](../inference_benchmarks.md)

These tables measure text output through **vLLM** and, in separate sections, **NIM**, with FP8 weights. Latencies are milliseconds; request and output-token throughput are per second. Lower latency and higher throughput are better. Each system uses **one GPU or one integrated platform**; concurrency is the number of simultaneous client requests.

The vLLM workload uses 50 text input tokens and one 374×374, 30-second synthetic video: 30 frames at 1 FPS or 60 frames at 2 FPS. Requested output length is 1 or 100 tokens. KV-cache precision is separate from model-weight precision. Blank cells have no validated published measurement and must not be read as lack of support.

See [measurement settings and reproduction limits](benchmark_methodology.md). Displayed values retain the source's available precision; trailing zeros do not imply additional measurement accuracy.

The synthetic-video workload is served with AIPerf. Exact runtime builds and complete prompt payloads were not retained with every measurement; these results describe the recorded configuration rather than a fully pinned reproduction environment. Some Super single-token runs returned empty replies with the `qwen3` parser, and some H20 runs timed out. Missing results are left blank. Where single-token metrics are available despite empty replies, they describe the successful requests shown by the request-count row and do not establish a zero-error serving rate.

## Enterprise

### DGX Spark

#### Input 50 / Output 1 / Video 1 FPS

| Metric | Concurrency 1 | Concurrency 64 | Concurrency 128 | Concurrency 256 |
|---|---:|---:|---:|---:|
| Time To First Token (ms) | 1245.10 | 2464.60 |  |  |
| Request Latency (ms) | 1245.10 | 2464.60 |  |  |
| Request Count (requests) | 50 | 50 |  |  |
| Request Throughput (Req/s) | 0.80 | 0.40 |  |  |
| Output Token Throughput (Tok/s) | 0.80 | 0.40 |  |  |


### DGX Station

#### Input 50 / Output 1 / Video 1 FPS

| Metric | Concurrency 1 | Concurrency 64 | Concurrency 128 | Concurrency 256 |
|---|---:|---:|---:|---:|
| Time To First Token (ms) |  | 155.80 | 124.90 | 2077.10 |
| Request Latency (ms) |  | 155.80 | 940.20 | 5422.40 |
| Request Count (requests) |  | 50 | 50 | 320 |
| Request Throughput (Req/s) |  | 6.40 | 1.10 | 11.80 |
| Output Token Throughput (Tok/s) |  | 6.40 | 106.10 | 1176.90 |

#### Input 50 / Output 1 / Video 2 FPS

| Metric | Concurrency 1 | Concurrency 64 | Concurrency 128 | Concurrency 256 |
|---|---:|---:|---:|---:|
| Time To First Token (ms) | 2994.40 | 4941.50 | 157.30 | 2250.40 |
| Request Latency (ms) | 10001.80 | 18874.80 | 989.60 | 10069.90 |
| Request Count (requests) | 256 | 512 | 50 | 320 |
| Request Throughput (Req/s) | 12.70 | 13.40 | 1.00 | 6.30 |
| Output Token Throughput (Tok/s) | 1272.40 | 1344.40 | 100.80 | 633.80 |


The available datacenter and GB300 tables do not yet expose every model/workload combination. Unavailable source cells have not been reconstructed from speedup summaries or other models.



## NIM serving

These FP8 measurements use NIM 1.7.1.rc3 with one GPU, two synthetic-video workloads: 100 text input tokens / 1 output token with 608×320 video at 4 FPS, and 1,000 input tokens / 100 output tokens with 1312×736 video at 2 FPS. Both videos last 10 seconds. All sampled frames are retained, pruning is zero, KV and multimodal caches are disabled, and strict output length is enabled. The two workloads target approximately 4K and 9K visual tokens respectively. These tables are separate from the vLLM 50-token, 30-second workload. The measured NIM build is prerelease software; results with release builds can differ.


### B300

#### Input 100 / Output 1 / Video 4 FPS

| Metric | Concurrency 1 | Concurrency 8 | Concurrency 16 | Concurrency 32 | Concurrency 64 | Concurrency 128 | Concurrency 256 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Time To First Token (ms) | 717.91 | 3271.66 | 4834.25 | 8260.48 | 11860.95 | 20177.71 | 36242.77 |
| Request Latency (ms) | 717.91 | 3271.66 | 4834.25 | 8260.48 | 11860.95 | 20177.71 | 36242.77 |
| Request Count (requests) | 10 | 16 | 32 | 64 | 128 | 256 | 512 |
| Request Throughput (Req/s) | 1.38 | 2.30 | 3.00 | 3.38 | 4.46 | 5.07 | 5.76 |
| Output Token Throughput (Tok/s) | 1.38 | 2.30 | 3.00 | 3.38 | 4.46 | 5.07 | 5.76 |

#### Input 1000 / Output 100 / Video 2 FPS

| Metric | Concurrency 1 | Concurrency 8 | Concurrency 16 | Concurrency 32 | Concurrency 64 | Concurrency 128 | Concurrency 256 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Time To First Token (ms) | 1484.46 | 7051.90 | 7884.78 | 15238.11 | 18531.09 | 45256.75 | 110709.69 |
| Request Latency (ms) | 2359.79 | 9127.30 | 11966.28 | 21313.92 | 38159.22 | 67382.83 | 134538.04 |
| Request Count (requests) | 10 | 16 | 32 | 64 | 128 | 256 | 512 |
| Request Throughput (Req/s) | 0.42 | 0.82 | 1.22 | 1.33 | 1.64 | 1.72 | 1.62 |
| Output Token Throughput (Tok/s) | 42.34 | 81.59 | 121.51 | 132.61 | 164.02 | 172.30 | 161.97 |


### B200

#### Input 100 / Output 1 / Video 4 FPS

| Metric | Concurrency 1 | Concurrency 8 | Concurrency 16 | Concurrency 32 | Concurrency 64 | Concurrency 128 | Concurrency 256 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Time To First Token (ms) | 690.47 | 3285.86 | 4713.28 | 7088.39 | 11281.23 | 19871.81 | 32118.96 |
| Request Latency (ms) | 690.47 | 3285.86 | 4713.28 | 7088.39 | 11281.23 | 19871.81 | 32118.96 |
| Request Count (requests) | 10 | 16 | 32 | 64 | 128 | 256 | 512 |
| Request Throughput (Req/s) | 1.44 | 2.28 | 3.04 | 3.87 | 4.69 | 5.40 | 6.32 |
| Output Token Throughput (Tok/s) | 1.44 | 2.28 | 3.04 | 3.87 | 4.69 | 5.40 | 6.32 |

#### Input 1000 / Output 100 / Video 2 FPS

| Metric | Concurrency 1 | Concurrency 8 | Concurrency 16 | Concurrency 32 | Concurrency 64 | Concurrency 128 | Concurrency 256 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Time To First Token (ms) | 1215.21 | 4825.43 | 7777.64 | 11782.63 | 16265.18 | 43080.33 | 94536.64 |
| Request Latency (ms) | 2098.06 | 7465.19 | 12336.80 | 19662.59 | 37720.93 | 67499.72 | 120061.04 |
| Request Count (requests) | 10 | 16 | 32 | 64 | 128 | 256 | 512 |
| Request Throughput (Req/s) | 0.48 | 1.07 | 1.26 | 1.52 | 1.66 | 1.72 | 1.79 |
| Output Token Throughput (Tok/s) | 47.61 | 106.69 | 126.01 | 152.39 | 166.16 | 171.83 | 178.60 |


### H100 80GB HBM3

#### Input 100 / Output 1 / Video 4 FPS

| Metric | Concurrency 1 | Concurrency 8 | Concurrency 16 | Concurrency 32 | Concurrency 64 | Concurrency 128 | Concurrency 256 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Time To First Token (ms) | 955.50 | 4473.73 | 6576.63 | 11108.36 | 17727.46 | 25757.25 | 41005.96 |
| Request Latency (ms) | 955.50 | 4473.73 | 6576.63 | 11108.36 | 17727.46 | 25757.25 | 41005.96 |
| Request Count (requests) | 10 | 16 | 32 | 64 | 128 | 256 | 512 |
| Request Throughput (Req/s) | 1.04 | 1.62 | 2.09 | 2.37 | 3.16 | 4.18 | 5.02 |
| Output Token Throughput (Tok/s) | 1.04 | 1.62 | 2.09 | 2.37 | 3.16 | 4.18 | 5.02 |

#### Input 1000 / Output 100 / Video 2 FPS

| Metric | Concurrency 1 | Concurrency 8 | Concurrency 16 | Concurrency 32 | Concurrency 64 | Concurrency 128 | Concurrency 256 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Time To First Token (ms) | 1811.79 | 7345.16 | 11728.54 | 16963.34 | 43233.02 | 96887.94 | 204411.01 |
| Request Latency (ms) | 3280.24 | 12569.09 | 21241.25 | 39068.40 | 68038.21 | 122999.38 | 231195.16 |
| Request Count (requests) | 10 | 16 | 32 | 64 | 128 | 256 | 512 |
| Request Throughput (Req/s) | 0.30 | 0.63 | 0.75 | 0.80 | 0.85 | 0.87 | 0.88 |
| Output Token Throughput (Tok/s) | 30.46 | 63.35 | 74.83 | 79.74 | 84.95 | 87.15 | 88.15 |


### H100 NVL

#### Input 100 / Output 1 / Video 4 FPS

| Metric | Concurrency 1 | Concurrency 8 | Concurrency 16 | Concurrency 32 | Concurrency 64 | Concurrency 128 | Concurrency 256 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Time To First Token (ms) | 964.99 | 5426.26 | 8782.87 | 16026.15 | 29193.74 | 57223.88 | 111039.93 |
| Request Latency (ms) | 964.99 | 5426.26 | 8782.87 | 16026.15 | 29193.74 | 57223.88 | 111039.93 |
| Request Count (requests) | 10 | 16 | 32 | 64 | 128 | 256 | 512 |
| Request Throughput (Req/s) | 1.03 | 1.30 | 1.53 | 1.62 | 1.73 | 1.75 | 1.78 |
| Output Token Throughput (Tok/s) | 1.03 | 1.30 | 1.53 | 1.62 | 1.73 | 1.75 | 1.78 |

#### Input 1000 / Output 100 / Video 2 FPS

| Metric | Concurrency 1 | Concurrency 8 | Concurrency 16 | Concurrency 32 | Concurrency 64 | Concurrency 128 | Concurrency 256 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Time To First Token (ms) | 2201.47 | 9179.24 | 16467.37 | 26415.18 | 60021.16 | 156370.68 | 345233.63 |
| Request Latency (ms) | 3567.70 | 15948.53 | 29604.68 | 53554.38 | 116120.54 | 217159.45 | 407528.55 |
| Request Count (requests) | 10 | 16 | 32 | 64 | 128 | 256 | 512 |
| Request Throughput (Req/s) | 0.28 | 0.50 | 0.54 | 0.59 | 0.52 | 0.51 | 0.51 |
| Output Token Throughput (Tok/s) | 28.01 | 49.99 | 53.79 | 59.33 | 51.97 | 50.77 | 51.23 |


### H200 141GB HBM3

#### Input 100 / Output 1 / Video 4 FPS

| Metric | Concurrency 1 | Concurrency 8 | Concurrency 16 | Concurrency 32 | Concurrency 64 | Concurrency 128 | Concurrency 256 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Time To First Token (ms) | 885.15 | 4646.18 | 6663.79 | 10561.10 | 18917.34 | 33237.00 | 48397.58 |
| Request Latency (ms) | 885.15 | 4646.18 | 6663.79 | 10561.10 | 18917.34 | 33237.00 | 48397.58 |
| Request Count (requests) | 10 | 16 | 32 | 64 | 128 | 256 | 512 |
| Request Throughput (Req/s) | 1.12 | 1.58 | 2.08 | 2.49 | 2.71 | 3.30 | 4.41 |
| Output Token Throughput (Tok/s) | 1.12 | 1.58 | 2.08 | 2.49 | 2.71 | 3.30 | 4.41 |

#### Input 1000 / Output 100 / Video 2 FPS

| Metric | Concurrency 1 | Concurrency 8 | Concurrency 16 | Concurrency 32 | Concurrency 64 | Concurrency 128 | Concurrency 256 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Time To First Token (ms) | 1719.70 | 7464.76 | 11689.78 | 15488.53 | 27158.70 | 79073.01 | 178293.80 |
| Request Latency (ms) | 2870.53 | 12201.00 | 20305.25 | 36625.29 | 69615.75 | 125941.25 | 227562.45 |
| Request Count (requests) | 10 | 16 | 32 | 64 | 128 | 256 | 512 |
| Request Throughput (Req/s) | 0.35 | 0.65 | 0.76 | 0.87 | 0.90 | 0.92 | 0.94 |
| Output Token Throughput (Tok/s) | 34.81 | 65.32 | 76.44 | 86.64 | 90.00 | 91.86 | 94.20 |


### H200 NVL

#### Input 100 / Output 1 / Video 4 FPS

| Metric | Concurrency 1 | Concurrency 8 | Concurrency 16 | Concurrency 32 | Concurrency 64 | Concurrency 128 | Concurrency 256 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Time To First Token (ms) | 848.97 | 4316.24 | 6740.26 | 11509.78 | 21065.49 | 37023.03 | 85305.10 |
| Request Latency (ms) | 848.97 | 4316.24 | 6740.26 | 11509.78 | 21065.49 | 37023.03 | 85305.10 |
| Request Count (requests) | 10 | 16 | 32 | 64 | 128 | 256 | 512 |
| Request Throughput (Req/s) | 1.17 | 1.66 | 2.01 | 2.27 | 2.41 | 2.94 | 2.34 |
| Output Token Throughput (Tok/s) | 1.17 | 1.66 | 2.01 | 2.27 | 2.41 | 2.94 | 2.34 |

#### Input 1000 / Output 100 / Video 2 FPS

| Metric | Concurrency 1 | Concurrency 8 | Concurrency 16 | Concurrency 32 | Concurrency 64 | Concurrency 128 | Concurrency 256 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Time To First Token (ms) | 1767.92 | 7504.01 | 11893.50 | 18218.74 | 30931.06 | 90345.58 | 222182.49 |
| Request Latency (ms) | 3012.22 | 12774.85 | 21856.18 | 41774.32 | 79188.94 | 145184.64 | 280287.29 |
| Request Count (requests) | 10 | 16 | 32 | 64 | 128 | 256 | 512 |
| Request Throughput (Req/s) | 0.33 | 0.62 | 0.73 | 0.76 | 0.79 | 0.80 | 0.77 |
| Output Token Throughput (Tok/s) | 33.17 | 62.38 | 72.80 | 75.68 | 79.12 | 79.57 | 76.97 |


### RTX PRO 6000 Blackwell

#### Input 100 / Output 1 / Video 4 FPS

| Metric | Concurrency 1 | Concurrency 8 | Concurrency 16 | Concurrency 32 | Concurrency 64 | Concurrency 128 | Concurrency 256 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Time To First Token (ms) | 1155.02 | 6392.83 | 8922.88 | 16486.59 | 31072.56 | 39291.37 | 107727.40 |
| Request Latency (ms) | 1155.02 | 6392.83 | 8922.88 | 16486.59 | 31072.56 | 39291.37 | 107727.40 |
| Request Count (requests) | 10 | 16 | 32 | 64 | 128 | 256 | 512 |
| Request Throughput (Req/s) | 0.86 | 1.12 | 1.51 | 1.58 | 1.78 | 2.73 | 1.81 |
| Output Token Throughput (Tok/s) | 0.86 | 1.12 | 1.51 | 1.58 | 1.78 | 2.73 | 1.81 |

#### Input 1000 / Output 100 / Video 2 FPS

| Metric | Concurrency 1 | Concurrency 8 | Concurrency 16 | Concurrency 32 | Concurrency 64 | Concurrency 128 | Concurrency 256 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Time To First Token (ms) | 2562.00 | 11312.93 | 18772.59 | 31207.81 | 58294.27 | 154541.71 | 335024.78 |
| Request Latency (ms) | 5266.03 | 20122.89 | 34745.03 | 62078.70 | 114956.18 | 214889.88 | 397353.06 |
| Request Count (requests) | 10 | 16 | 32 | 64 | 128 | 256 | 512 |
| Request Throughput (Req/s) | 0.19 | 0.40 | 0.44 | 0.50 | 0.52 | 0.52 | 0.53 |
| Output Token Throughput (Tok/s) | 18.98 | 39.54 | 43.63 | 49.81 | 52.46 | 51.81 | 52.58 |
