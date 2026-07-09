<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# TAR and VANTAGE Benchmarks for Cosmos3 Model

This repository is an **NVIDIA-customized [VLMEvalKit](https://github.com/open-compass/VLMEvalKit)** that
releases the NVIDIA Metropolis "smart infra" benchmarks and makes the cosmos-reasoner scores on them
**publicly reproducible**. The reproduction kit lives in [`cosmos_eval/`](cosmos_eval/); the stock
VLMEvalKit inference + scoring engine it drives is the rest of the tree (upstream toolkit docs:
[`README-vlmevalkit.md`](README-vlmevalkit.md)).

The kit ships, for each benchmark, the exact `run.py --config` settings the internal evaluation uses —
split into a shared dataset layer (`cosmos_eval/data/`) and a per-family model layer
(`cosmos_eval/models/`) — plus a parallel launcher (`cosmos_eval/run_all.py`) and a standalone score
reporter (`cosmos_eval/parse_score.py`).

You deploy the model behind an OpenAI-compatible endpoint, point the launcher at it, and it composes
each config, runs the stock vlmevalkit `run.py`, and reports the headline score per benchmark. No
extra evaluation logic lives in the kit — inference and scoring are vlmevalkit's; the kit only composes
the inputs and reads the outputs.

> **This rollout ships the NVIDIA Metropolis "smart infra" benchmarks — `AETCBench_all` (the official
> **TAR** / Traffic Anomaly Reasoning benchmark) + the 8 `VANTAGE_*` (9 total).**
>
> **⚠️ The datasets are not publicly available yet** — both the VANTAGE and TAR
> challenges are still ongoing. **We will update this code as soon as the data and evaluation are ready.**
> Until then these configs are published as a reference; `run_all` runs every benchmark in the manifest.

```
.                          # NVIDIA-customized VLMEvalKit (stock inference + scoring engine)
├── README.md              # this file
├── README-vlmevalkit.md   # upstream VLMEvalKit toolkit docs
├── run.py                 # stock vlmevalkit entrypoint (driven once per benchmark)
└── cosmos_eval/                       # the reproduction kit
    ├── data/<domain>/<bench>.json     # dataset section per benchmark (model-independent)
    ├── models/<family>.json           # per-family model layer (day 1: cosmos)
    ├── manifest.json                  # per-bench run.py CLI flags
    ├── run_all.py                     # parallel launcher (compose → run.py → parse_score)
    └── parse_score.py                 # standalone score reporter
```

---

## 1. Environment setup

Use a clean virtual environment with **Python 3.10+**, run from the **repo root** (the directory that
holds this README):

```bash
python -m venv .venv && source .venv/bin/activate

# vlmevalkit's own dependencies
pip install -r requirements.txt

# Extra packages required at runtime but not pinned by requirements.txt:
#   einops, accelerate   -> imported when vlmevalkit loads its dataset modules
#   setuptools<81        -> jieba imports pkg_resources, which setuptools>=81 drops
#                           and Python 3.12 venvs omit
#   evaluate             -> TAR (AETCBench_all) scoring metric
pip install einops accelerate "setuptools<81" evaluate

# install vlmevalkit itself so `import vlmeval` resolves
pip install -e .

# TAR (AETCBench_all) only: also install `nvdataset` — NVIDIA's INTERNAL DSS client. It fetches
# the TAR (and cosmos-DVC) data and requires the NVDATASET_TENANTID / NGC_API_KEY credentials in §3.
```

Sanity check:

```bash
python -c "import vlmeval; print('vlmeval OK')"
```

---

## 2. Deploy the model endpoint (vLLM)

Serve the model behind an OpenAI-compatible `/v1/chat/completions` endpoint (see also
**[Reasoner with vLLM](https://github.com/NVIDIA/cosmos/tree/main#reasoner-with-vllm)**):

```bash
vllm serve nvidia/Cosmos3-Nano \
  --async-scheduling \
  --allowed-local-media-path / \
  --port 8080 \
  --max-model-len 128000 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.9 \
  --enable-chunked-prefill \
  --mm-processor-cache-gb 0 \
  --mm-encoder-tp-mode data \
  --media-io-kwargs '{"video": {"num_frames": -1, "fps": -1}}'
```

The `--media-io-kwargs` / `--mm-*` flags let the server honor the per-benchmark frame sampling the
configs request, and `--allowed-local-media-path /` lets it read local media. (Raise
`--tensor-parallel-size` for larger models.) Any OpenAI-compatible server works (vLLM, SGLang, TGI, a
hosted NIM, …) as long as it answers `POST /v1/chat/completions` and `GET /v1/models`; the wrapper
health-checks the endpoint and auto-detects the served model id from `/v1/models`.

---

## 3. Environment variables

```bash
# Model under test
export COSMOS_API_BASE=http://<host>:8080/v1/chat/completions
export COSMOS_MODEL=<served-model-id>      # or any placeholder; auto-detected from /v1/models
export COSMOS_API_KEY=<key-or-EMPTY>

export NVDATASET_TENANTID=<dss-tenant>   # DSS tenant for the nvdataset client
export NGC_API_KEY=<ngc-key>             # NGC key for the nvdataset client
```

`run_all.py` substitutes `${COSMOS_MODEL}` / `${COSMOS_API_BASE}` into each composed config at launch.

---

## 4. Run

From the vlmevalkit repo root:

```bash
# all benchmarks in the manifest
python cosmos_eval/run_all.py --model cosmos --concurrency 2 --work-dir ./out

# a specific few
python cosmos_eval/run_all.py --benchmarks VANTAGE_VQA,AETCBench_all --work-dir ./out
```

Per benchmark, the launcher composes `model_conf = {class} ∪ defaults ∪ benchmarks[bench]` and
`dataset_conf = data[bench] ∪ {model_family}`, writes `<work-dir>/_configs/<bench>.json`, runs
`run.py --config … --verbose --save-eval-results` with the manifest's CLI flags (one `run.py`
subprocess per benchmark, `--concurrency` at a time), then reports the score.

**Useful flags**

| Flag | Effect |
|---|---|
| `--benchmarks A,B` | run only these benchmarks (default: all in the manifest) |
| `--concurrency N` | up to N benchmarks (each its own `run.py`) at once |
| `--dry-run` | print the `run.py` commands and exit |
| `--export-configs DIR` | compose + write each `--config` JSON to `DIR` (placeholders kept), then exit |
| `--import-configs DIR` | run from a dir of (possibly hand-edited) configs instead of composing |

---

## 5. Data & scoring

These benchmarks' datasets are being published to Hugging Face:
- **VANTAGE** (`nvidia/PhysicalAI-VANTAGE-Bench`) — test inputs are released; ground truth is withheld,
  so scores come from the **VANTAGE leaderboard**.
- **TAR — Traffic Anomaly Reasoning** (config key `AETCBench_all`; `nvidia/PhysicalAI-Traffic-Anomaly-Reasoning`) —
  test answers are redacted; submit predictions to the **AI City Challenge** evaluation server.

`run_all` produces per-benchmark predictions (and a local score where ground truth is available); follow
each benchmark's leaderboard instructions to obtain official scores.

---

## 6. Confirm the results

After a run, each benchmark has its own output dir and a parsed score in the summary table:

```
=== cosmos_eval summary ===
  AETCBench_all       ok         <score>
  VANTAGE_2DGrounding ok         <score>
  ...
  9/9 ok, 0 failed
```

`run.py` writes its evaluation output to `<work-dir>/<bench>/<model>/T*/<model>_<bench>.{dict,df}.eval.json`.
Report any single output directly with the standalone reporter:

```bash
python cosmos_eval/parse_score.py --work-dir ./out/VANTAGE_VQA --dataset VANTAGE_VQA
# VANTAGE_VQA  Overall: <score>
```

`parse_score.py` prints the headline **Overall** (0–100) plus every native scalar sub-score key the
benchmark's eval JSON contains. It reads both eval-output shapes vlmevalkit emits
(`*.dict.eval.json`, `*.df.eval.json`). The scores it reports match the internal evaluation pipeline,
benchmark for benchmark.

Each benchmark dir also keeps `run.log` (full `run.py` output) for debugging; the composed config
used is under `<work-dir>/_configs/<bench>.json`.

---

## 7. Benchmarks & headline metrics

`parse_score.py` reports each benchmark's headline as **Overall** (0–100) plus all native sub-scores.
The headline metric and the eval-JSON key it reads, per benchmark:

| Benchmark | Modality | Headline metric | eval-JSON key |
|---|---|---|---|
| `VANTAGE_2DPointing` | image | accuracy | (generic) |
| `VANTAGE_2DGrounding` | image | Mean_IoU | `Mean_IoU` |
| `VANTAGE_Astro2D` | image | F1 @ IoU=0.5 | `f1` |
| `VANTAGE_VQA` | video | accuracy | `accuracy` |
| `VANTAGE_EventVerification` | video | macro-avg F1 | `macro avg--f1-score` |
| `VANTAGE_Temporal` | video | overall IoU | `overall.iou` |
| `VANTAGE_DVC` | video | SODA_c (BERTScore-based) | `overall.SODA_c` |
| `VANTAGE_SOT` | video | mean success-AUC | `Overall` |
| `AETCBench_all` (TAR) | video | weighted_mean | `weighted_mean` |

---

## 8. Tips for reliable runs

- **Concurrency vs. replicas:** each benchmark already issues ~`api_nproc` parallel requests, so a high
  `--concurrency` (benchmarks in parallel) can overload a single GPU replica. Keep it modest on one
  replica and scale up by adding replicas.
- **Resume:** outputs are isolated under `<work-dir>/<bench>/`; if a run is interrupted, just re-run the
  affected `--benchmarks`.
- **Verify before trusting a score:** scan each benchmark's `run.log` for dropped or empty-input warnings.
