# Cosmos3 Reasoner Examples

Run the Cosmos3 Reasoner (vision-language reasoning over images and video) across
multiple inference backends. Sample inputs live under [`assets/`](./assets).

Environment setup for every backend is centralized in the shared
[Cosmos3 cookbooks environment setup](../README.md) guide; each backend below
links to the section you need.

## Run with Cosmos Framework

### Quickstart

Set up the environment: [Cosmos Framework setup](../README.md#cosmos-framework).
This produces the framework venv at `packages/cosmos3/.venv`; run the commands
below from that checkout (`cd packages/cosmos3`).

Create a Reasoner input file. The `enable_sound=false` field is intentional — it
avoids a strict argument-validation failure in the current Reasoner path:

```bash
mkdir -p outputs/cookbooks/cosmos3/reasoner/inputs
cat > outputs/cookbooks/cosmos3/reasoner/inputs/robot_image.json <<'JSON'
{
  "model_mode": "reasoner",
  "name": "robot_image",
  "prompt": "Describe what is happening in this image in one sentence.",
  "vision_path": "https://github.com/nvidia-cosmos/cosmos-dependencies/raw/refs/heads/assets/cosmos3/inputs/vision/robot_153.jpg",
  "enable_sound": false
}
JSON
```

Run Nano on a single GPU:

```bash
COSMOS_TRAINING=false CUDA_VISIBLE_DEVICES=0 \
MASTER_ADDR=127.0.0.1 MASTER_PORT=29501 RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 \
.venv/bin/python -m cosmos_framework.scripts.inference \
  --parallelism-preset=latency \
  -i outputs/cookbooks/cosmos3/reasoner/inputs/robot_image.json \
  -o outputs/cookbooks/cosmos3/reasoner/nano/cosmos_framework_image \
  --checkpoint-path Cosmos3-Nano \
  --seed=0 \
  --benchmark
```

The generated text is written to
`outputs/cookbooks/cosmos3/reasoner/nano/cosmos_framework_image/robot_image/reasoner_text.txt`.

### Notebook walkthrough

[`run_with_cosmos_framework.ipynb`](./run_with_cosmos_framework.ipynb) is the full
tutorial. It writes text and image smoke tests, then walks through image
capability sections — detailed captioning, robot task planning, 2D grounding,
describe-anything, and action-trajectory prompts — rendering the prompt, media
input, model output, and any parsed boxes or trajectories together for review. It
also shows how to scale from **Nano** (single GPU, shown above) to **Super**
(4 GPUs via `.venv/bin/torchrun`). Video examples live in the vLLM notebook,
because the current Cosmos Framework Reasoner path expects image inputs through
`vision_path`.

## Run with vLLM

### Quickstart

Set up the environment and start the server:
[vLLM setup](../README.md#vllm) (install) and
[Start the server](../README.md#start-the-server) (launch commands).

The quickstart below uses **Cosmos3-Nano** on port 8000. The
[`run_with_vllm.ipynb`](./run_with_vllm.ipynb) notebook defaults to
**Cosmos3-Super** on port **8001** — use that launch command from the env setup
guide and point the client at `http://localhost:8001/v1`.

Once the server is ready, query it with the OpenAI client:

```python
import openai

image_url = (
    "https://github.com/nvidia-cosmos/cosmos-dependencies/raw/refs/heads/"
    "assets/cosmos3/inputs/vision/robot_153.jpg"
)

client = openai.OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")

response = client.chat.completions.create(
    model=client.models.list().data[0].id,
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": "Caption the image in detail."},
            ],
        }
    ],
    max_tokens=4096,
    seed=0,
)

print(response.choices[0].message.content)
```

### Notebook walkthrough

[`run_with_vllm.ipynb`](./run_with_vllm.ipynb) uses the **Cosmos3-Super** launch
from the [environment setup guide](../README.md#start-the-server) and walks
through many more image and video examples: detailed captioning,
VQA, temporal localization, embodied reasoning, common-sense reasoning, 2D
grounding, describe-anything, action CoT trajectories, driving scenes,
physical-plausibility, and situation understanding. To run a different model,
change only the **server launch** command and the client `base_url`; every
example resolves the model dynamically
(`MODEL = client.models.list().data[0].id`), so the prompts work unchanged.

## Run with NIM

### Quickstart

Set up the environment: [NIM setup](../README.md#nim). That launches the
prebuilt [Cosmos 3 Reasoner NIM](https://catalog.ngc.nvidia.com/orgs/nim/teams/nvidia/containers/cosmos3-reasoner)
container, which serves the same OpenAI-compatible chat-completions API as the
vLLM path on port 8000 — without the vLLM/CUDA setup. You can also try this same
NIM interactively in your browser on the
[cosmos3-nano-reasoner build page](https://build.nvidia.com/nvidia/cosmos3-nano-reasoner).

Once the server is up, query it with the OpenAI client (the served model name is
`nvidia/cosmos3-nano-reasoner`, or `nvidia/cosmos3-super-reasoner` for Super):

```python
import openai

image_url = (
    "https://github.com/nvidia-cosmos/cosmos-dependencies/raw/refs/heads/"
    "assets/cosmos3/inputs/vision/robot_153.jpg"
)

client = openai.OpenAI(api_key="not-used", base_url="http://127.0.0.1:8000/v1")

response = client.chat.completions.create(
    model="nvidia/cosmos3-nano-reasoner",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": "Caption the image in detail."},
            ],
        }
    ],
    max_tokens=4096,
)

print(response.choices[0].message.content)
```

Video inputs use the `video_url` content type and accept `media_io_kwargs`
frame-sampling controls through `extra_body`
(`{"media_io_kwargs": {"video": {"fps": 4.0}}}`). See the
[Cosmos Reason 3 NIM API reference](https://docs.nvidia.com/nim/vision-language-models/1.7.0/examples/cosmos-reason3/api.html)
and the [NIM section in the top-level README](../../../README.md#reasoner-with-nim)
for the full request reference.

### Notebook walkthrough

[`run_with_nim.ipynb`](./run_with_nim.ipynb) is the NIM counterpart to the vLLM
notebook: it launches the NIM container, waits for readiness, and then runs the
same image and video examples — detailed captioning, VQA, temporal localization,
embodied reasoning, common-sense reasoning, 2D grounding, describe-anything,
action CoT trajectories, driving scenes, physical-plausibility, and situation
understanding. Because the container does not see the host filesystem, local
assets are sent as base64 data URIs; video frame sampling is controlled with
`media_io_kwargs` while spatial resolution is controlled with
`mm_processor_kwargs`. Each example resolves the served model dynamically
(`client.models.list()`), so the prompts work unchanged for both the `nano` and
`super` sizes.

## Run with Transformers

Support for Transformers-based Reasoner inference is coming soon — see
[Transformers setup](../README.md#transformers-coming-soon).
