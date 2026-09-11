# FAQ & Troubleshooting

## Which CUDA version should I use?

CUDA 13 (recommended) or 12.8. Your system CUDA and PyTorch's CUDA major version must match — check with `nvidia-smi` and `python -c "import torch; print(torch.version.cuda)"`.

## Which base container should I use?

NVIDIA NGC PyTorch: `nvcr.io/nvidia/pytorch:25.09-py3` (CUDA 13) or `nvcr.io/nvidia/pytorch:25.06-py3` (CUDA 12).

## `torch.cuda.is_available()` is `False` ("The NVIDIA driver on your system is too old")

The installed `torch` targets newer CUDA than your driver — `uv pip install torch` defaults to `cu130`. Install a matching build: `uv pip install --torch-backend=auto torch torchvision`, or pin (e.g. `--torch-backend=cu128`). For `uv sync` workflows use `COSMOS3_UV_GROUP=cu128-train`; for vLLM pair `cu128` with `vllm==0.19.1`.

## Import fails with `libxcb.so.1: cannot open shared object file`

Headless servers and minimal containers are missing system graphics libraries a dependency links against:

```bash
apt-get install -y libxcb1 libgl1 libglib2.0-0
```

## `uv` errors on install or `sync`

Requires `uv >= 0.11.3` (enforced via `pyproject.toml`). Older versions fail to parse the project config or don't recognize `--torch-backend=cu130`. Upgrade: `uv self update`.

## vLLM reports DeepGEMM unavailable

```bash
export VLLM_USE_DEEP_GEMM=0
```

## Generation seems to hang

It probably isn't: the first run downloads a multi-GB checkpoint, and diffusion runs through every inference step before producing any output. Long per-step times are expected.

## Training: OOM, NCCL hangs, slow throughput

Training troubleshooting lives with the trainer — see the [Cosmos Framework docs](https://github.com/NVIDIA/cosmos-framework/tree/main/docs) (parallelism degrees, mixed precision, and environment variables are the usual levers).

---

Not answered here? [Open a discussion](https://github.com/NVIDIA/cosmos/discussions).
