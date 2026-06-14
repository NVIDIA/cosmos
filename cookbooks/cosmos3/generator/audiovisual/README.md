# Cosmos3 Generator Audiovisual Examples

Generate images and video (with optional audio) from text or image prompts with
`Cosmos3-Nano` and `Cosmos3-Super`, across three inference backends. Sample
prompts live under [`assets/`](./assets).

Environment setup for every backend is centralized in the shared
[Cosmos3 cookbooks environment setup](../../README.md) guide; each backend below
links to the section you need. The quickstarts are minimal text-to-video examples
to get one generation running per backend — run them from this folder.

Generator requires the Guardrail. Request access to the gated
[nvidia/Cosmos-1.0-Guardrail](https://huggingface.co/nvidia/Cosmos-1.0-Guardrail)
HF repository before running these examples. To disable the guardrail, set
`enable_safety_checker=False` (Diffusers), `guardrails: false` (vLLM-Omni
`extra_params`/`extra_args`), or `--no-guardrails` (Cosmos Framework).

## Run with Cosmos Framework

### Quickstart

Set up the environment: [Cosmos Framework setup](../../README.md#cosmos-framework).
Activate the framework venv created during setup (`packages/cosmos3/.venv` at the
repo root), or call its `torchrun` by path. The notebook builds a full inference
payload JSON (not the raw prompt asset alone); build one the same way, then run
from this folder:

```bash
python3 - <<'PY'
import json
from pathlib import Path

prompt = json.dumps(
    json.load(open("assets/prompts/text2video/robot_kitchen.json")),
    ensure_ascii=True,
    separators=(",", ":"),
)
negative = json.dumps(
    json.load(open("assets/negative_prompts/text2video/neg_prompt.json")),
    ensure_ascii=True,
    separators=(",", ":"),
)
payload = {
    "model_mode": "text2video",
    "name": "robot_kitchen",
    "prompt": prompt,
    "negative_prompt": negative,
    "enable_sound": False,
    "num_steps": 35,
    "guidance": 6.0,
    "shift": 10.0,
    "fps": 24,
    "num_frames": 189,
    "resolution": "720",
    "aspect_ratio": "16,9",
    "seed": 0,
}
Path("/tmp/cosmos3_t2v_payload.json").write_text(json.dumps(payload, indent=2) + "\n")
PY

torchrun --nproc-per-node=1 \
  -m cosmos_framework.scripts.inference \
  --parallelism-preset=throughput \
  -i /tmp/cosmos3_t2v_payload.json \
  -o /tmp/cosmos3_t2v_framework \
  --checkpoint-path Cosmos3-Nano \
  --seed=0
```

To run **Cosmos3-Super** instead, set `--checkpoint-path Cosmos3-Super` and use
more GPUs via `--nproc-per-node`.

### Notebook walkthrough

[`run_with_cosmos_framework.ipynb`](./run_with_cosmos_framework.ipynb) is the full
tutorial for the native PyTorch backend: it covers every use case — text-to-image,
text-to-video, image-to-video, with audio on or off — and includes the detailed,
environment-aware setup and visualization for each generation.

## Run with Diffusers

### Quickstart

Set up the environment: [Diffusers setup](../../README.md#diffusers).
Run a text-to-video generation with `Cosmos3OmniPipeline`:

```python
import json
import torch
from diffusers import Cosmos3OmniPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from diffusers.utils import export_to_video

prompt = json.load(open("assets/prompts/text2video/robot_kitchen.json"))
negative = json.load(open("assets/negative_prompts/text2video/neg_prompt.json"))

pipe = Cosmos3OmniPipeline.from_pretrained(
    "nvidia/Cosmos3-Nano", torch_dtype=torch.bfloat16, device_map="cuda"
)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=10.0)

result = pipe(
    prompt=json.dumps(prompt),
    negative_prompt=json.dumps(negative),
    image=None,
    num_frames=189,
    height=720,
    width=1280,
    fps=24,
    num_inference_steps=35,
    guidance_scale=6.0,
    enable_sound=False,
    add_resolution_template=False,
    add_duration_template=False,
    generator=torch.Generator(device="cuda").manual_seed(1234),
)
export_to_video(result.video, "/tmp/cosmos3_t2v_diffusers.mp4", fps=24)
```

To run **Cosmos3-Super** instead, load the larger checkpoint:
`Cosmos3OmniPipeline.from_pretrained("nvidia/Cosmos3-Super", ...)`.

### Notebook walkthrough

[`run_with_diffusers.ipynb`](./run_with_diffusers.ipynb) is the full tutorial for
the Diffusers backend: it provisions a dedicated venv, then walks through
text-to-image, text-to-video, and image-to-video generation (with and without
audio) using `Cosmos3OmniPipeline`, including how to preview the generated media.

## Run with vLLM-Omni

### Quickstart

Set up the environment and start the server:
[vLLM-Omni setup](../../README.md#vllm-omni) (Docker recommended). Run the
Docker command from this folder (`COSMOS3_WORKDIR` defaults to the current
directory) with **`COSMOS3_HOST_PORT=8000`** unless you already have another
server on that port.

Send a text-to-video request with the OpenAI-compatible video API:

```python
import json
from pathlib import Path

import requests

prompt = json.load(open("assets/prompts/text2video/robot_kitchen.json"))
negative = json.load(open("assets/negative_prompts/text2video/neg_prompt.json"))

response = requests.post(
    "http://localhost:8000/v1/videos/sync",
    data={
        "prompt": json.dumps(prompt),
        "negative_prompt": json.dumps(negative),
        "size": "1280x720",
        "num_frames": "189",
        "fps": "24",
        "num_inference_steps": "35",
        "guidance_scale": "6.0",
        "flow_shift": "10.0",
        "seed": "0",
        "extra_params": json.dumps(
            {
                "use_resolution_template": False,
                "use_duration_template": False,
                "guardrails": True,
            }
        ),
    },
    headers={"Accept": "video/mp4"},
)
response.raise_for_status()
Path("/tmp/cosmos3_t2v.mp4").write_bytes(response.content)
```

For image-to-video, post to the same endpoint with an image under
`files={"input_reference": ...}`. For audio, add `"generate_sound": "true"`.

### Notebook walkthrough

[`run_with_vllm_omni.ipynb`](./run_with_vllm_omni.ipynb) is the full tutorial for
the vLLM-Omni backend: it walks through text-to-image, text-to-video, and
image-to-video requests with audio on or off. Server launch options (Nano and
Super, tensor parallelism, layerwise offload, and CFG-parallel variants) live in
the [shared environment setup guide](../../README.md#vllm-omni).
