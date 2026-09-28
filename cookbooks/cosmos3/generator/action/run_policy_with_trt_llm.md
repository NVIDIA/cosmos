# Cosmos3-Edge-Policy-DROID server with TensorRT-LLM

**DRAFT — not yet run end-to-end.** This serves [`nvidia/Cosmos3-Edge-Policy-DROID`](https://huggingface.co/nvidia/Cosmos3-Edge-Policy-DROID) through TensorRT-LLM. Run the [one-shot notebook](./run_policy_with_trt_llm.ipynb) to inspect the HTTP response. RoboLab integration needs an adapter to TensorRT-LLM's request and response format.

## Table of Contents

- [Policy Server](#policy-server)
- [One-Shot Client](#one-shot-client)

## Policy Server

Build the TensorRT-LLM container from a checkout containing [PR #18463](https://github.com/NVIDIA/TensorRT-LLM/pull/18463), which merged into `main` on 5 Sep 2026:

```bash
git clone https://github.com/NVIDIA/TensorRT-LLM.git
cd TensorRT-LLM

make -C docker release_build IMAGE_TAG=edge-policy-droid
```

Launch the container and start the policy server. `--enable_visual_gen` selects
the video-generation route; without it, TensorRT-LLM selects its language-model
loader for this checkpoint. The server reads the checkpoint's policy defaults,
including its 32-action chunk, 15 FPS, state conditioning, and sampler settings:

```bash
: "${HF_TOKEN:?Set HF_TOKEN to an authorized Hugging Face token}"

docker run -d --name cosmos3-trtllm-policy-notebook \
  --runtime nvidia --gpus '"device=0"' \
  -e HF_HOME=/root/.cache/huggingface \
  -e HF_TOKEN \
  -e TRTLLM_DISABLE_COSMOS3_GUARDRAILS=1 \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -p 8000:8000 --ipc=host \
  tensorrt_llm/release:edge-policy-droid \
  trtllm-serve nvidia/Cosmos3-Edge-Policy-DROID --enable_visual_gen --port 8000

# Wait until this returns 200 before running a client against it.
curl -i http://localhost:8000/health
```

To inspect startup logs:

```bash
docker logs -f cosmos3-trtllm-policy-notebook
```

## One-Shot Client

Run [`run_policy_with_trt_llm.ipynb`](./run_policy_with_trt_llm.ipynb) from the
`cosmos` repository root after the server reports healthy. The notebook composes
the checked-in DROID camera frames, sends a structured task prompt and an
8-value current state to `POST /v1/videos/sync`, and saves the returned
`safetensors` payload and a video preview.

The notebook reads the first current-state row from the same checked-in DROID
sample. Use the live robot's current model-space state for robot control.
RoboLab's existing Cosmos Framework client needs a transport adapter to call
TensorRT-LLM; this guide does not claim a simulation run.
