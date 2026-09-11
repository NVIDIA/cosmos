# Quickstart: Serve

**Goal:** a production OpenAI-compatible endpoint.
**Pick your surface:** Generator (video/image/sound/action out) → **vLLM-Omni** (below) or **SGLang**. Reasoner (text out) → **vLLM**, **TensorRT-LLM**, or **NIM** (see [Reason](reason.md)).

All Generator serving paths at a glance:

| Path | Modes | Notes |
|---|---|---|
| vLLM-Omni (below) | T2I, T2V, I2V, V2V, sound, action | Broadest coverage; Docker image or pip |
| [SGLang Diffusion](https://docs.sglang.io/cookbook/diffusion/Cosmos/Cosmos3) | T2I, T2V, I2V, action (fd/id) | `sglang serve --model-path nvidia/Cosmos3-Nano`; requires SGLang `main` until support lands in a stable release; disable guardrails with `SGLANG_DISABLE_COSMOS3_GUARDRAILS=1` |
| TensorRT-LLM (VisualGen) | T2I, T2V, I2V, V2V, sound | See the [TensorRT-LLM generator cookbook](https://github.com/NVIDIA/cosmos/blob/main/cookbooks/cosmos3/generator/audiovisual/run_with_trt_llm.ipynb) |
| Generator NIM | T2V, I2V only | Prebuilt NGC container; `POST /v1/infer`, returns JSON `b64_video` — see the [NIM generator cookbook](https://github.com/NVIDIA/cosmos/blob/main/cookbooks/cosmos3/generator/audiovisual/run_with_nim.ipynb) |

## Generator with vLLM-Omni

Serves the full Cosmos 3 checkpoint — reasoner path, diffusion path, and media tokenizers — behind an HTTP API.

```bash
docker run --runtime nvidia --gpus all \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v "$(pwd):/workspace" \
  -p 8000:8000 --ipc=host \
  vllm/vllm-omni:cosmos3 \
  vllm serve nvidia/Cosmos3-Nano \
  --omni \
  --model-class-name Cosmos3OmniDiffusersPipeline \
  --allowed-local-media-path / \
  --port 8000 \
  --init-timeout 1800
```

Cosmos 3 checkpoints can exceed the default init timeout — keep `--init-timeout 1800` on every `vllm serve`. The server prints `Application startup complete.` when ready. Pip install from `main` instead of Docker: [vLLM-Omni recipes](https://github.com/vllm-project/vllm-omni/tree/main/recipes/cosmos3).

## Endpoints

| Mode | Endpoint | Notes |
|---|---|---|
| Text → image | `POST /v1/images/generations` | returns base64 PNG |
| Text → video | `POST /v1/videos/sync` | blocks, returns MP4 bytes |
| Image → video | `POST /v1/videos/sync` | conditioning image via `input_reference` |
| Video → video | `POST /v1/videos/sync` | choose clean conditioning frames |
| Video + sound | `POST /v1/videos/sync` | `generate_sound=true` |
| Policy / inverse dynamics | `POST /v1/videos` (async) | returns video + predicted action chunk |
| Forward dynamics | `POST /v1/videos/sync` | image + action chunk → video |

Example request:

```bash
curl -sS -X POST http://localhost:8000/v1/videos/sync \
  --form-string "prompt=A small warehouse robot moves a blue box across a clean floor." \
  --form-string "size=1280x720" \
  --form-string "num_frames=189" --form-string "fps=24" \
  --form-string "num_inference_steps=35" --form-string "guidance_scale=6.0" \
  --form-string "flow_shift=10.0" --form-string "seed=0" \
  -o output.mp4
```

Use `--form-string` (not `-F`) for text fields — `-F` treats `;` as a content-type separator and silently truncates. Use `guidance_scale` for CFG, never `true_cfg_scale`. Full request-field table: [Videos API](https://docs.vllm.ai/projects/vllm-omni/en/latest/serving/videos_api/) · [Image API](https://docs.vllm.ai/projects/vllm-omni/en/latest/serving/image_generation_api/).

**Action modes** condition on an embodiment via `extra_params`: `action_mode` (`policy` / `inverse_dynamics` / `forward_dynamics`), `domain_name` (e.g. `bridge_orig_lerobot`, `av`, `camera_pose`), `raw_action_dim`, `action_chunk_size`, and `action_path` for forward dynamics (the server must be able to read that path — mount it and cover it with `--allowed-local-media-path`).

## Scaling up

| Flag | Use |
|---|---|
| `--tensor-parallel-size N` | split weights across N GPUs (needed for Super, e.g. 4) |
| `--enable-layerwise-offload` | CPU↔GPU block offload; saves VRAM, costs latency + CPU RAM |
| `--cfg-parallel-size 2` | positive/negative CFG branches on separate GPUs |
| `--ulysses-degree 2` | sequence parallelism |

Provision GPUs ≥ the product of enabled degrees. Measured latency/throughput per config: [benchmarks](../reference/benchmarks.md).

## Guardrails

Cosmos 3 ships guardrails that screen prompts and blur faces, **on by default** (Generator paths also require access to the gated [Cosmos-1.0-Guardrail](https://huggingface.co/nvidia/Cosmos-1.0-Guardrail) HF repo). Each backend has its own toggle:

| Backend | Disable with |
|---|---|
| vLLM-Omni | `extra_params={"guardrails": false}` per request, or the deploy config below |
| Diffusers | `enable_safety_checker=False` |
| Cosmos Framework | `--no-guardrails` |
| TensorRT-LLM | `TRTLLM_DISABLE_COSMOS3_GUARDRAILS=1` or `use_guardrails: false` in `extra_params` |
| SGLang | `SGLANG_DISABLE_COSMOS3_GUARDRAILS=1` |
| Generator NIM | `NIM_ENABLE_TEXT_GUARDRAILS=0 NIM_ENABLE_VIDEO_GUARDRAILS=0` |

To disable vLLM-Omni server-wide (guardrail models never load; per-request overrides can't re-enable):

```yaml
# no_guardrails.yaml
async_chunk: false
stages:
  - stage_id: 0
    max_num_seqs: 1
    enforce_eager: true
    trust_remote_code: true
    model_class_name: Cosmos3OmniDiffusersPipeline
    model_config:
      guardrails: false
      offload_guardrail_models: false
```

```bash
vllm serve nvidia/Cosmos3-Nano --omni \
  --model-class-name Cosmos3OmniDiffusersPipeline \
  --deploy-config no_guardrails.yaml --port 8000 --init-timeout 1800
```

Review the [limitations](https://github.com/NVIDIA/cosmos/blob/main/README.md#limitations--safety) before deploying without guardrails.
