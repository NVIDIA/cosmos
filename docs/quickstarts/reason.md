# Quickstart: Reason

**Goal:** text answers about images and video — captioning, temporal localization, grounding, embodied task planning, physical plausibility.
**Two main paths:** NIM (fastest, prebuilt container) or vLLM (pip-installable, more control). Both expose an OpenAI-compatible chat API. Also available: **Transformers** for Python-first research and **TensorRT-LLM** serving — see the [reasoner cookbooks](https://github.com/NVIDIA/cosmos/tree/main/cookbooks/cosmos3/reasoner/).

Zero-install preview: the [browser playground](https://build.nvidia.com/nvidia/cosmos3-nano-reasoner) runs this same NIM.

## Path A · NIM container (recommended)

Prebuilt and optimized — skips all CUDA/vLLM version pairing. Needs an [NGC API key](https://catalog.ngc.nvidia.com/orgs/nim/teams/nvidia/containers/cosmos3-reasoner) and a one-time `docker login nvcr.io` (username `$oauthtoken`, password = your key).

```bash
export LOCAL_NIM_CACHE=~/.cache/nim && mkdir -p "$LOCAL_NIM_CACHE"

docker run -it --rm --name=nvidia-cosmos3-reasoner \
  --runtime=nvidia --gpus all --shm-size=32GB \
  -e NGC_API_KEY=$NGC_API_KEY \
  -e NIM_MODEL_SIZE=nano \
  -v "$LOCAL_NIM_CACHE:/opt/nim/.cache" \
  -u $(id -u) -p 8000:8000 \
  nvcr.io/nim/nvidia/cosmos3-reasoner:1.7.0
```

`NIM_MODEL_SIZE=nano` serves `nvidia/cosmos3-nano-reasoner`; `super` serves the 64B `nvidia/cosmos3-super-reasoner`.

## Path B · vLLM

```bash
uv venv --python 3.13 --seed --managed-python && source .venv/bin/activate
uv pip install --torch-backend=cu130 "vllm==0.21.0" \
  "vllm-cosmos3 @ git+https://github.com/NVIDIA/cosmos-framework.git#subdirectory=packages/vllm-cosmos3"
```

vLLM version and torch backend are paired: `cu130` ↔ `vllm==0.21.0`, `cu128` ↔ `vllm==0.19.1`. `--torch-backend=auto` is not reliable here.

```bash
vllm serve nvidia/Cosmos3-Nano \
  --hf-overrides '{"architectures": ["Cosmos3ReasonerForConditionalGeneration"]}' \
  --async-scheduling \
  --allowed-local-media-path / \
  --port 8000
```

If your build reports DeepGEMM unavailable: `export VLLM_USE_DEEP_GEMM=0` first. Multi-GPU flags (`--tensor-parallel-size`, `--mm-encoder-tp-mode data`, …): [reasoner cookbook README](https://github.com/NVIDIA/cosmos/blob/main/cookbooks/cosmos3/README.md).

## Query it

Either path serves `http://127.0.0.1:8000/v1`:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="not-used")
response = client.chat.completions.create(
    model="nvidia/cosmos3-nano-reasoner",   # Path B: "nvidia/Cosmos3-Nano"
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": [
            {"type": "video_url", "video_url": {"url": "https://download.samplelib.com/mp4/sample-5s.mp4"}},
            {"type": "text", "text": "List the notable events with approximate timestamps."},
        ]},
    ],
    max_tokens=256,
    extra_body={"media_io_kwargs": {"video": {"fps": 4.0}}},
)
print(response.choices[0].message.content)
```

Images: JPG/PNG via URL or base64 data URI (`image_url`). Videos: MP4 via URL, base64, or pre-decoded frames (`video_url`). Message conventions are Qwen3-VL-compatible.

**Explicit reasoning:** append to the user prompt —

```text
Answer the question using the following format:

<think>
Your reasoning.
</think>

Write your final answer immediately after the </think> tag.
```

## What to try

Captioning, temporal localization, embodied next-action prediction, 2D grounding (JSON boxes), describe-anything, action chain-of-thought, physical-plausibility classification — each with a runnable example in the [reasoner cookbooks](https://github.com/NVIDIA/cosmos/tree/main/cookbooks/cosmos3/reasoner/). Prompting patterns per task: the [reasoner prompt guide](https://github.com/NVIDIA/cosmos/blob/main/cookbooks/cosmos3/reasoner/reasoner_prompt_guide.md). Sampling defaults: [models reference](../reference/models.md).

## Next steps

- Production deployment & parallelism → [Serve](serve.md)
- Fine-tune the Reasoner on your domain → [Cosmos Framework](https://github.com/NVIDIA/cosmos-framework)
- API details → [NIM API reference](https://docs.nvidia.com/nim/vision-language-models/1.7.0/examples/cosmos-reason3/api.html)
