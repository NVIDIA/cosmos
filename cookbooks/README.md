# Cookbooks

Runnable notebooks for every Cosmos 3 capability. Each is self-contained; pick by what you want to do, then by which backend you run.

## Generator

| Notebook | Workflows | Backend |
|---|---|---|
| `cosmos3/generator/audiovisual/run_with_diffusers.ipynb` | T2I · T2V · I2V, each ± sound | Diffusers |
| `cosmos3/generator/audiovisual/run_with_cosmos_framework.ipynb` | Same, via `cosmos_framework.scripts.inference` | Cosmos Framework |
| `cosmos3/generator/audiovisual/run_with_vllm_omni.ipynb` | T2I · T2V · I2V · V2V ± sound, OpenAI-compatible server | vLLM-Omni |
| `cosmos3/generator/audiovisual/run_with_sglang.ipynb` | T2I · T2V · I2V ± sound | SGLang |
| `cosmos3/generator/audiovisual/run_with_trt_llm.ipynb` | T2I · T2V · I2V ± sound · V2V | TensorRT-LLM (VisualGen) |
| `cosmos3/generator/audiovisual/run_with_nim.ipynb` | T2V · I2V via `/v1/infer` | Generator NIM |
| `cosmos3/generator/action/run_fd_with_{cosmos_framework,diffusers,vllm_omni,sglang}.ipynb` | Forward dynamics (AV, DROID, UMI, human hand pose) | Framework · Diffusers · vLLM-Omni · SGLang |
| `cosmos3/generator/action/run_id_with_{cosmos_framework,diffusers,vllm_omni,sglang}.ipynb` | Inverse dynamics (AV ego-motion) | Framework · Diffusers · vLLM-Omni · SGLang |
| `cosmos3/generator/action/run_policy_with_{diffusers,vllm_omni,sglang}.ipynb` (+ framework guide) | Action policy rollouts | Diffusers · vLLM-Omni · SGLang · Framework |
| `cosmos3/generator/transfer/run_video_transfer_with_{cosmos_framework,diffusers,vllm_omni}.ipynb` | Video transfer: edge, blur, depth, segmentation, world-scenario controls | Framework · Diffusers · vLLM-Omni |

## Reasoner

| Notebook | Workflows | Backend |
|---|---|---|
| `cosmos3/reasoner/run_with_cosmos_framework.ipynb` | Captioning, task planning, grounding, describe-anything, action trajectories | Cosmos Framework |
| `cosmos3/reasoner/run_with_transformers.ipynb` | Python-first reasoning with Hugging Face Transformers | Transformers |
| `cosmos3/reasoner/run_with_vllm.ipynb` | Full reasoning suite incl. temporal localization, physical plausibility (Super on 4 GPUs by default) | vLLM |
| `cosmos3/reasoner/run_with_tensorrt_llm.ipynb` | Same suite; 1-GPU Nano and 4-GPU Super launches | TensorRT-LLM |
| `cosmos3/reasoner/run_with_nim.ipynb` | Same suite against the prebuilt NIM container | NIM |

## Post-training

| Cookbook | Covers |
|---|---|
| `cosmos3/generator/audiovisual/finetune/` | Vision SFT recipes (Nano full, Super LoRA, Edge full) |
| `cosmos3/generator/action/finetune/` | DROID action-policy SFT |
| `cosmos3/reasoner/finetune/` | LLaVA-OneVision alignment, VideoPhy-2 plausibility SFT, TAO agent skills |
| `cosmos3/generator/audiovisual/distill/` | DMD2 distillation → the `-4Step` checkpoints |

Prompting the Reasoner well: [`cosmos3/reasoner/reasoner_prompt_guide.md`](cosmos3/reasoner/reasoner_prompt_guide.md). Evaluation suites (PAIBench, Physics-IQ, RBench, UniGenBench, VLMEvalKit) live in [`evaluation/`](../evaluation/).

Environment setup shared by all cookbooks (backends, CUDA pairing, guardrail toggles): `cosmos3/README.md`.
