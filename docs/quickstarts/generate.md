# Quickstart: Generate

**Goal:** your first generated image, video, or video-with-sound.
**Path:** Hugging Face Diffusers — the Python-first route for research and prototyping.
**Hardware:** one NVIDIA GPU (Ampere/Hopper/Blackwell), Linux. Nano (16B) fits a single modern GPU in BF16.

Want an API server instead of Python? → [Serve](serve.md). Want to explore interactively? → [audiovisual cookbook](https://github.com/NVIDIA/cosmos/blob/main/cookbooks/cosmos3/generator/audiovisual/run_with_diffusers.ipynb).

## 1 · Environment

```bash
uv venv --python 3.13 --seed --managed-python
source .venv/bin/activate
uv pip install --torch-backend=auto \
  "diffusers @ git+https://github.com/huggingface/diffusers.git" \
  accelerate av cosmos_guardrail huggingface_hub imageio imageio-ffmpeg \
  torch torchvision transformers
```

`--torch-backend=auto` matches the CUDA build of `torch` to your driver. Without it, uv pulls the newest CUDA wheel (`cu130`), which fails on pre-CUDA-13 drivers with "The NVIDIA driver on your system is too old". Pin explicitly if you prefer (e.g. `--torch-backend=cu128`). Full matrix: [FAQ](../reference/faq.md#which-cuda-version-should-i-use).

Authenticate with Hugging Face — checkpoints are gated, and Generator runs also need access to the gated [nvidia/Cosmos-1.0-Guardrail](https://huggingface.co/nvidia/Cosmos-1.0-Guardrail) repo:

```bash
uvx hf@latest auth login   # or: export HF_TOKEN=<your_token>
```

Set `HF_HOME` if you want the multi-GB checkpoint cache on a bigger disk.

## 2 · Generate

```python
import torch
from diffusers import Cosmos3OmniPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from diffusers.utils import export_to_video

pipe = Cosmos3OmniPipeline.from_pretrained(
    "nvidia/Cosmos3-Nano",
    torch_dtype=torch.bfloat16,
    device_map="cuda",
)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=10.0)

result = pipe(
    prompt="A mobile robot navigates a warehouse aisle and stops at a shelf.",
    negative_prompt="",
    num_frames=189,          # ≈7.9 s at 24 FPS
    height=720, width=1280,
    fps=24,
    num_inference_steps=35,
    guidance_scale=6.0,
    enable_sound=False,
    generator=torch.Generator(device="cuda").manual_seed(1234),
)
export_to_video(result.video, "cosmos3_t2v.mp4", fps=24, macro_block_size=1)
```

The first run downloads Cosmos3-Nano; diffusion runs through every inference step before producing output, so long step times are expected — it isn't a hang.

## 3 · Modes

| Mode | How |
|---|---|
| Text → image | `num_frames=1`; returns a PIL image |
| Text → video | as above; 189 frames ≈ 7.9 s at 24 FPS |
| Image → video | pass `image=` with a conditioning frame |
| Text → video + sound | `enable_sound=True` (checkpoints with sound modules); stereo AAC 48 kHz |

Runnable examples of every mode: [Cosmos 3 Diffusers docs](https://huggingface.co/docs/diffusers/main/en/api/pipelines/cosmos3).

## Next steps

- Resolutions, aspect ratios, frame counts, prompt-length guidance → [models reference](../reference/models.md)
- Action-conditioned generation (policy, forward/inverse dynamics) → [action cookbooks](https://github.com/NVIDIA/cosmos/tree/main/cookbooks/cosmos3/generator/action/)
- Something failed → [FAQ](../reference/faq.md)
