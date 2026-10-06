# Cosmos3 Generator Audiovisual Examples

Generate images and video (with optional audio) from text, image or video prompts with
`Cosmos3-Nano`, `Cosmos3-Super`, `Cosmos3-Edge`, and the published four-step
distilled Cosmos3-Super students across Cosmos Framework, Diffusers, vLLM-Omni, TRT-LLM
and NIM backends. Sample prompts live under [`assets/`](./assets).

> **NIM scope:** [Cosmos3 Certified NIM](../../nim/README.md) supports
> text-to-image, text-to-video, image-to-video, and video-to-video,
> plus [Action](../../nim/action.md) and [Transfer](../../nim/transfer.md).
> Sound/audio generation is not exposed by the public Generator API.

Environment setup for every backend is centralized in the shared
[Cosmos3 cookbooks environment setup](../../README.md) guide; each backend below
links to the section you need. The quickstarts are minimal text-to-video examples
to get one generation running per backend — run them from this folder.

For backends other than NIM, request access to the gated
[nvidia/Cosmos-1.0-Guardrail](https://huggingface.co/nvidia/Cosmos-1.0-Guardrail)
HF repository before running these examples. To disable the guardrail, set
`enable_safety_checker=False` (Diffusers), `TRTLLM_DISABLE_COSMOS3_GUARDRAILS=1`
or `use_guardrails: false` through `extra_params` (TensorRT-LLM),
`guardrails: false` (vLLM-Omni `extra_params`/`extra_args`), or
`--no-guardrails` (Cosmos Framework). For NIM, see
[Guardrails](../../nim/operations.md#guardrails).

NIM uses NGC authentication instead of Hugging Face login; see the
[NIM setup](../../nim/deployment.md) for details.

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

To run **Cosmos3-Edge** instead, load `nvidia/Cosmos3-Edge` and use its
single-GPU, no-audio settings: `height=480`, `width=832`, `num_frames=121`,
`num_inference_steps=20`, `guidance_scale=5.0`, and `flow_shift=8.0`.

### Notebook walkthrough

[`run_with_diffusers.ipynb`](./run_with_diffusers.ipynb) is the full tutorial for
the Diffusers backend: it provisions a dedicated venv, then walks through
text-to-image, text-to-video, image-to-video, and video-to-video generation (with
and without audio) using `Cosmos3OmniPipeline`, including a dedicated Cosmos3-Edge
section with a 480p image-to-video example and preview.

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

## Run with TensorRT-LLM

### Quickstart

Set up the environment and start the server:
[TensorRT-LLM setup](../../README.md#tensorrt-llm-generator). The notebook targets the
OpenAI-compatible VisualGen API served by `trtllm-serve`.

Send a text-to-video request with the synchronous video API:

```python
import json
from pathlib import Path

import requests

prompt = json.load(open("assets/prompts/text2video/robot_kitchen.json"))
negative = json.load(open("assets/negative_prompts/text2video/neg_prompt.json"))

response = requests.post(
    "http://localhost:8000/v1/videos/sync",
    json={
        "prompt": json.dumps(prompt, ensure_ascii=True, separators=(",", ":")),
        "negative_prompt": json.dumps(negative, ensure_ascii=True, separators=(",", ":")),
        "size": "1280x720",
        "seconds": 189 / 24,
        "fps": 24,
        "num_frames": 189,
        "num_inference_steps": 35,
        "guidance_scale": 6.0,
        "max_sequence_length": 4096,
        "seed": 0,
        "format": "mp4",
        "response_format": "file",
        "extra_params": {
            "use_resolution_template": False,
            "use_duration_template": False,
            "use_system_prompt": False,
            "use_guardrails": True,
        },
    },
    headers={"Accept": "video/mp4"},
)
response.raise_for_status()
if (
    "video/mp4" not in response.headers.get("content-type", "")
    or response.content[4:8] != b"ftyp"
):
    raise RuntimeError("TensorRT-LLM did not return browser-compatible MP4")
Path("/tmp/cosmos3_t2v_trtllm.mp4").write_bytes(response.content)
```

For image-to-video, post multipart form data to the same endpoint with the
reference image under `image_reference`. To generate synchronized audio for a
text-to-video or image-to-video request, add `"enable_audio": True` to
`extra_params`. Keep `ffmpeg` installed in the server environment so TensorRT-LLM
can mux the generated audio into MP4. Explicit `format=mp4` makes a missing
encoder fail early instead of returning browser-incompatible AVI.

For video-to-video, upload an MP4 reference under `video_reference`.
TensorRT-LLM forwards the encoded bytes to
the Cosmos3 workers, which decode the conditioning window with NVDEC:

```python
import json
from pathlib import Path

import requests

source_video = Path("assets/videos/car_driving_plain.mp4").resolve()
v2v_prompt = json.load(open("assets/prompts/image2video/car_driving.json"))
v2v_negative = json.load(open("assets/negative_prompts/image2video/neg_prompt.json"))

with source_video.open("rb") as video_file:
    response = requests.post(
        "http://localhost:8000/v1/videos/sync",
        data={
            "prompt": json.dumps(v2v_prompt, ensure_ascii=True, separators=(",", ":")),
            "negative_prompt": json.dumps(v2v_negative, ensure_ascii=True, separators=(",", ":")),
            "size": "1280x720",
            "num_frames": "189",
            "fps": "24",
            "num_inference_steps": "35",
            "guidance_scale": "6.0",
            "max_sequence_length": "4096",
            "seed": "0",
            "format": "mp4",
            "response_format": "file",
            "extra_params": json.dumps(
                {
                    "use_resolution_template": False,
                    "use_duration_template": False,
                    "use_system_prompt": True,
                    "use_guardrails": True,
                    "condition_video_latent_indexes": [0, 1],
                    "condition_video_keep": "first",
                },
                separators=(",", ":"),
            ),
        },
        files={"video_reference": (source_video.name, video_file, "video/mp4")},
        headers={"Accept": "video/mp4"},
    )
response.raise_for_status()
if (
    "video/mp4" not in response.headers.get("content-type", "")
    or response.content[4:8] != b"ftyp"
):
    raise RuntimeError("TensorRT-LLM did not return browser-compatible MP4")
Path("/tmp/cosmos3_v2v_trtllm.mp4").write_bytes(response.content)
```

`condition_video_latent_indexes` identifies clean latent frames in the output;
with `[0, 1]`, TensorRT-LLM consumes the first five pixel frames from the input.
Set `condition_video_keep` to `"last"` to condition on the corresponding tail
window instead.

For text-to-image, send JSON to `/v1/images/generations` with `format=png`,
`response_format=b64_json`, and `extra_params.output_type="image"`, then
base64-decode `data[0].b64_json`. This produces a PNG image instead of wrapping
one frame in a video container.

To run **Cosmos3-Edge** instead, serve `nvidia/Cosmos3-Edge` on a single GPU with
no config override (`trtllm-serve nvidia/Cosmos3-Edge --port 8000`) and send
Edge's 480p-native shape: `"size": "832x480"`, `"num_frames": 121`,
`"num_inference_steps": 50`, and `"guidance_scale": 5.0`. Text-to-image is a
native image request: post to `/v1/images/generations` with
`"output_type": "image"` in `extra_params`, Edge's native `"size": "640x640"`,
and `"guidance_scale": 4.0`. That flag selects the image path; without it the
server runs video mode and defaults the negative prompt to Cosmos3's video
negative prompt, whose motion and frame-to-frame artifact terms do not apply to
a still. The images API carries no frame or frame-rate fields. Flow shift (3.0)
rides the checkpoint-declared native flow schedule, so requests do not send it.
TensorRT-LLM serves Edge for text-to-image, text-to-video, and image-to-video
only: Edge has no audio tower, so `enable_audio` is unavailable, its action
weights are not served by this pipeline, and video-to-video is validated for Nano
and Super. Edge support landed in TensorRT-LLM PR
[#16773](https://github.com/NVIDIA/TensorRT-LLM/pull/16773).

The TRT-LLM notebook always sends model-specific `extra_params`, so use a
TensorRT-LLM release with the Cosmos3 VisualGen API schema. The notebook sets
request-level `max_sequence_length=4096` for longer structured JSON prompts.

### Notebook walkthrough

[`run_with_trt_llm.ipynb`](./run_with_trt_llm.ipynb) is the full tutorial for the
TensorRT-LLM backend: it walks through text-to-image, text-to-video and
image-to-video with or without synchronized audio, and video-to-video requests
against an already-running VisualGen server. It also includes a dedicated
Cosmos3-Edge section with 480p text-to-image, text-to-video, and image-to-video
examples. Server launch options (Nano, Super, and Edge, FP8 dynamic
quantization, CFG parallelism, Ulysses, and parallel VAE) live in the
[shared environment setup guide](../../README.md#tensorrt-llm-generator).

### Distilled 4-step notebook

[`run_distilled_with_trt_llm.ipynb`](./run_distilled_with_trt_llm.ipynb) is the
tutorial for the two published DMD2-distilled Cosmos3-Super students,
`nvidia/Cosmos3-Super-Text2Image-4Step` and
`nvidia/Cosmos3-Super-Image2Video-4Step`. Each serves on a single GPU and runs a
fixed four-step stochastic schedule with classifier-free guidance baked into the
weights, so requests omit `num_inference_steps` and `guidance_scale` entirely
and let the server read both from the checkpoint; sending a conflicting value is
rejected rather than clamped. Requests also leave `use_system_prompt` unset so
the image-to-video student's `default_use_system_prompt: true` applies. These
students cover text-to-image and image-to-video only. Server launch commands
live in the
[shared environment setup guide](../../README.md#tensorrt-llm-generator).

## Run with NIM

Use [NIM](../../nim/README.md) with `NIM_MODEL_TYPE=generator`. Follow the
[deployment guide](../../nim/deployment.md)
to select a compatible model and latency or throughput profile, then use the
[generation guide](../../nim/generation.md) for text-to-image, text-to-video,
image-to-video, and video-to-video requests through `POST /v1/infer`.

Requests select the task explicitly with `model_mode`. The guide includes
[Python examples](../../nim/examples/) for general-purpose and four-step
models, response decoding, and saving generated media.
