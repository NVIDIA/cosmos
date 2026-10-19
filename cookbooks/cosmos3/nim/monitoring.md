<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: OpenMDW-1.1 -->

# Monitor the Cosmos3 NIM with Prometheus and Grafana

Scrape the NIM's Prometheus metrics and chart them in Grafana. The steps apply
to both the Generator and the Reasoner.

## Prerequisites

A running NIM. Confirm that the metrics endpoint returns Prometheus text:

```bash
export NIM_URL=${NIM_URL:-http://localhost:8000}
curl -fsS "$NIM_URL/v1/metrics" | head
```

## Prometheus

Create `prometheus.yml`:

```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: cosmos3-nim
    metrics_path: /v1/metrics
    scrape_interval: 15s
    static_configs:
      - targets: ["cosmos3-generator:8000"]
  - job_name: cosmos3-generator-triton
    metrics_path: /metrics
    scrape_interval: 15s
    static_configs:
      - targets: ["cosmos3-generator:8002"]
```

The NIM serves metrics at `/v1/metrics` on the HTTP API port `8000`, not at
the Prometheus default `/metrics`. The `cosmos3-generator-triton` job scrapes
Generator request metrics from the internal Triton port `8002`, reachable on
the shared Docker network without publishing it; omit this job for a Reasoner.

Run Prometheus with the config mounted read-only. This example uses a shared
Docker network; attach the NIM container to it. The examples use the Generator
container name from [Deployment](deployment.md#launch-generator); for a Reasoner,
use `cosmos3-reasoner` instead.

```bash
docker network create cosmos3-monitoring   # skip if it exists
docker network connect cosmos3-monitoring cosmos3-generator

docker run -d --name prometheus \
  --network cosmos3-monitoring \
  -p 9090:9090 \
  -v "$PWD/prometheus.yml:/etc/prometheus/prometheus.yml:ro" \
  prom/prometheus:v2.53.0
```

To scrape a NIM published on the host instead, use `--network host`, drop
`-p 9090:9090`, and set the target to `localhost:8000`.

Open `http://localhost:9090/targets` (**Status > Targets**). The
`cosmos3-nim` job, and for a Generator the `cosmos3-generator-triton`
job, must show `UP`.

| Symptom | Cause |
|---------|-------|
| Target `DOWN`, `404` | `metrics_path` is not `/v1/metrics`. |
| Target `DOWN`, connection refused or no such host | Prometheus cannot reach the NIM. Put both containers on the same Docker network and use the container name, or use host networking. |

## Grafana

```bash
docker run -d --name grafana \
  --network cosmos3-monitoring \
  -p 3000:3000 \
  grafana/grafana:11.1.0
```

Open `http://localhost:3000`, sign in as `admin` / `admin`, and change the
password when prompted. Then add the data source:

1. Go to **Connections > Data sources > Add data source**.
2. Select **Prometheus**.
3. Set **Prometheus server URL** to `http://prometheus:9090` (the container
   name on the shared network).
4. Select **Save & test**.

Optionally, provision the data source instead. Save the file as
`grafana-datasource.yml` and add
`-v "$PWD/grafana-datasource.yml:/etc/grafana/provisioning/datasources/prometheus.yml:ro"`
to the `docker run` command:

```yaml
apiVersion: 1
datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
```

Use **Explore** to browse the metrics Prometheus has collected, then build
dashboard panels from them. Each runtime exposes different metric families:

| Runtime | Endpoint | Metric families |
|---------|----------|-----------------|
| Generator | `:8000/v1/metrics` | GPU, process, and Python gauges only, no request metrics: `gpu_utilization`, `gpu_memory_used_bytes`, `gpu_power_usage_watts`, `process_resident_memory_bytes` |
| Generator | `:8002/metrics` (Triton) | Per-model request metrics, labelled by `model` such as `Cosmos3Omni` and `Qwen3GuardFilter`: `nv_inference_count`, `nv_inference_request_success`, `nv_inference_request_failure`, `nv_inference_request_duration_us`, `nv_inference_queue_duration_us`; plus `nv_gpu_*` and `nv_cpu_*` gauges |
| Reasoner | `:8000/v1/metrics` | vLLM request and cache metrics: `vllm:time_to_first_token_seconds`, `vllm:e2e_request_latency_seconds`, `vllm:num_requests_running`, `vllm:num_requests_waiting`, `vllm:kv_cache_usage_perc`; plus `gpu_*` gauges |

Metric names can change between releases; check your service's output before
building dashboards or alerts.

## Kubernetes

For [Helm](helm.md) deployments, set `metrics.serviceMonitor.enabled: true`
to create a Prometheus Operator `ServiceMonitor` that scrapes `/v1/metrics` on
the `http-api` port. Generator request metrics on port `8002` need an
additional scrape target.

## Security

The NIM does not authenticate requests. Do not expose `/v1/metrics`,
Prometheus, or Grafana outside a trusted network. Change Grafana's default
`admin` credentials before use.
