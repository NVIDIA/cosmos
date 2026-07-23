# Cosmos3 Generator Audiovisual Examples

Generate images and video (with optional audio) from text, image or video prompts with
`Cosmos3-Nano`, `Cosmos3-Super`, `Cosmos3-Edge`, and the published four-step
distilled Cosmos3-Super students across Cosmos Framework, Diffusers, vLLM-Omni,
and NIM backends. Sample prompts live under [`assets/`](./assets).

> **NIM scope:** `Cosmos3-Generator` NIM currently exposes Text2Video and
> Image2Video only. It does not expose text-to-image, video-to-video,
> sound/audio generation, action modes, or transfer controls.

Environment setup for every backend is centralized in the shared
[Cosmos3 cookbooks environment setup](../../README.md) guide; each backend below
links to the section you need. The quickstarts are minimal text-to-video examples
to get one generation running per backend — run them from this folder.

Generator requires the Guardrail. Request access to the gated
[nvidia/Cosmos-1.0-Guardrail](https://huggingface.co/nvidia/Cosmos-1.0-Guardrail)
HF repository before running these examples. To disable the guardrail, set
`enable_safety_checker=False` (Diffusers), `guardrails: false` (vLLM-Omni
`extra_params`/`extra_args`), or `--no-guardrails` (Cosmos Framework). For
Generator NIM set `NIM_ENABLE_TEXT_GUARDRAILS=0 NIM_ENABLE_VIDEO_GUARDRAILS=0`.

NIM backends use NGC authentication instead of Hugging Face login; see the
[Generator NIM setup](../../README.md#generator-nim) for details.

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

To run **Cosmos3-Edge** instead, set `--checkpoint-path Cosmos3-Edge`. Edge has
no audio modules, so keep `"enable_sound": False` in the payload. Edge supports
only 256p/480p, so use Edge's generation settings in the payload:
`"resolution": "480"`, `"num_frames": 121`, and `"fps": 24`.

### Notebook walkthrough

[`run_with_cosmos_framework.ipynb`](./run_with_cosmos_framework.ipynb) is the full
tutorial for the native PyTorch backend: it covers every use case — text-to-image,
text-to-video, image-to-video, with audio on or off — and includes the detailed,
environment-aware setup and visualization for each generation. It also includes
four-GPU inference examples for `nvidia/Cosmos3-Super-Text2Image-4Step` and
`nvidia/Cosmos3-Super-Image2Video-4Step`.

### Distillation training recipe

[`distill/README.md`](./distill/README.md) documents the short T2I and I2V DMD2
training, resume, and student-only export workflow. The first supported topology
is exactly 8 GB200 nodes with 4 GPUs per node. This is an integration smoke
recipe, not a production reproduction recipe.

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
For video-to-video, upload a source video under `input_reference` and choose the
clean conditioning frames through `extra_params`:

```python
from pathlib import Path

source_video = Path("../action/assets/videos/av_0.mp4").resolve()
with source_video.open("rb") as video_file:
    response = requests.post(
        "http://localhost:8000/v1/videos/sync",
        data={
            "prompt": "Continue the same driving scene with smooth natural motion.",
            "negative_prompt": "blurry, distorted, low quality, jittery, deformed",
            "size": "832x480",
            "num_frames": "61",
            "fps": "10",
            "num_inference_steps": "35",
            "guidance_scale": "6.0",
            "flow_shift": "10.0",
            "seed": "2222",
            "extra_params": json.dumps(
                {
                    "use_resolution_template": False,
                    "use_duration_template": False,
                    "guardrails": True,
                    "condition_frame_indexes_vision": [0, 1],
                    "condition_video_keep": "first",
                }
            ),
        },
        files={"input_reference": (source_video.name, video_file, "video/mp4")},
        headers={"Accept": "video/mp4"},
    )
response.raise_for_status()
Path("/tmp/cosmos3_v2v.mp4").write_bytes(response.content)
```

### Notebook walkthrough

[`run_with_vllm_omni.ipynb`](./run_with_vllm_omni.ipynb) is the full tutorial for
the vLLM-Omni backend: it walks through text-to-image, text-to-video, and
image-to-video requests with audio on or off plus standard video-to-video
requests. Server launch options (Nano and
Super, tensor parallelism, layerwise offload, and CFG-parallel variants) live in
the [shared environment setup guide](../../README.md#vllm-omni).

## Run with NIM

### Quickstart

Set up the environment: [Generator NIM setup](../../README.md#generator-nim).
`Cosmos3-Generator` NIM is a prebuilt NGC container that serves Text2Video and
Image2Video through `POST /v1/infer`. It returns JSON with a base64-encoded MP4
in `b64_video`; unlike vLLM-Omni, it does not use `/v1/videos/sync` and does not
return MP4 bytes directly.

Authenticate Docker to NGC once:

```bash
export NGC_API_KEY=<your_key>
echo "$NGC_API_KEY" | docker login nvcr.io --username '$oauthtoken' --password-stdin
```

Launch **Cosmos3-Nano** (default model size, FP8, latency profile):

```bash
export LOCAL_NIM_CACHE="${LOCAL_NIM_CACHE:-$HOME/.cache/nim}"
mkdir -p "$LOCAL_NIM_CACHE"
chmod -R 777 "$LOCAL_NIM_CACHE" 2>/dev/null || true

docker run -it --rm --name cosmos3-generator \
  --runtime=nvidia \
  --gpus all \
  --shm-size=32GB \
  --ulimit nofile=65536:65536 \
  -e NGC_API_KEY="$NGC_API_KEY" \
  -v "$LOCAL_NIM_CACHE:/opt/nim/.cache" \
  -p 8000:8000 \
  nvcr.io/nim/nvidia/cosmos3-generator:1.0.0
```

Launch **Cosmos3-Super** by adding `NIM_MODEL_SIZE=super`:

```bash
docker run -it --rm --name cosmos3-generator \
  --runtime=nvidia \
  --gpus all \
  --shm-size=32GB \
  --ulimit nofile=65536:65536 \
  -e NGC_API_KEY="$NGC_API_KEY" \
  -e NIM_MODEL_SIZE=super \
  -e NIM_PRECISION=fp8 \
  -e NIM_PERF_PROFILE=latency \
  -v "$LOCAL_NIM_CACHE:/opt/nim/.cache" \
  -p 8000:8000 \
  nvcr.io/nim/nvidia/cosmos3-generator:1.0.0
```

Wait until the readiness endpoint returns success:

```bash
curl -fsS http://127.0.0.1:8000/v1/health/ready
```

Send a Text2Video request. The NIM infers T2V from a non-empty `prompt` with no
`image` field:

```bash
curl -sS -X POST http://127.0.0.1:8000/v1/infer \
  -H 'Accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "A humanoid robot walks through a futuristic warehouse, inspecting shelves of mechanical components. Photorealistic, cinematic lighting.",
    "seed": 42,
    "guidance_scale": 6.0,
    "steps": 35,
    "resolution": "256",
    "num_output_frames": 25,
    "fps": 24.0
  }' | jq -r '.b64_video' | base64 -d > /tmp/cosmos3_nim_t2v.mp4
```

Send an Image2Video request with a local image encoded as a data URI:

```bash
IMG_B64=$(base64 -w 0 assets/images/image2video/humanoid_robot.jpg)
cat > /tmp/cosmos3_nim_i2v.json <<EOF
{
  "prompt": "The humanoid robot performs a controlled standing backflip in a modern living room, then lands steadily on both feet.",
  "image": "data:image/jpeg;base64,${IMG_B64}",
  "seed": 123,
  "guidance_scale": 6.0,
  "steps": 35,
  "resolution": "256",
  "num_output_frames": 25,
  "fps": 24.0
}
EOF

curl -sS -X POST http://127.0.0.1:8000/v1/infer \
  -H 'Accept: application/json' \
  -H 'Content-Type: application/json' \
  -d @/tmp/cosmos3_nim_i2v.json | jq -r '.b64_video' | base64 -d > /tmp/cosmos3_nim_i2v.mp4
```

Key request fields and constraints:

| Field | Constraint / default |
| --- | --- |
| `prompt` | Required for T2V; optional when `image` is provided; max 20000 chars |
| `negative_prompt` | Optional; omitted means the server uses the Cosmos3 default |
| `image` | I2V conditioning image; raw base64, `data:image/...;base64,...`, or public URL when URL inputs are enabled |
| `guidance_scale` | `1.0` to `7.0`, default `6.0` |
| `steps` | `1` to `100`, default `35` |
| `resolution` | `256`, `480`, `720`, optionally with `_16_9`, `_1_1`, `_9_16`, `_4_3`, or `_3_4` |
| `num_output_frames` | Must follow the 4k+1 cadence (`25, 29, 33, ...`); caps: `256 <= 397`, `480 <= 297`, `720 <= 197` |
| `fps` | `1.0` to `60.0`, recommended `10` to `30`, default `24.0` |

### Notebook walkthrough

[`run_with_nim.ipynb`](./run_with_nim.ipynb) launches the NIM container,
waits for readiness, inspects the service endpoints, sends T2V and I2V requests,
decodes `b64_video`, and previews the generated MP4 files inline.

### Limitations

`Cosmos3-Generator` NIM currently exposes **Text2Video** and **Image2Video** only.
It does **not** expose text-to-image, video-to-video, sound/audio generation,
action modes, or transfer controls. For those broader Generator API workflows,
use vLLM-Omni or Cosmos Framework as appropriate.
