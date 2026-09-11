# Cosmos Documentation

**Quickstarts** are task-oriented — copy-paste your way to a result. **Reference** is lookup material.

## Quickstarts

| Guide | You'll end up with |
|---|---|
| [Generate](quickstarts/generate.md) | Images, video, and synchronized sound from text/image prompts (Diffusers) |
| [Reason](quickstarts/reason.md) | Text answers about images & video — captions, grounding, planning (NIM or vLLM) |
| [Serve](quickstarts/serve.md) | A production OpenAI-compatible endpoint for Generator or Reasoner |
| [Post-train](https://github.com/NVIDIA/cosmos-framework) | A post-trained Cosmos 3 checkpoint on your own data — SFT, distillation, RL (in Cosmos Framework; come back here to evaluate) |
| [Evaluate](https://github.com/NVIDIA/cosmos/tree/main/evaluation/) | Benchmark scores: PAIBench-C/G, Physics-IQ, RBench, UniGenBench (Generator); VLMEvalKit (Reasoner) |

Not sure which surface you need? **Reasoner** takes text+vision in, text out. **Generator** takes text/vision/sound/action in, vision/sound/action out. Details: [models reference](reference/models.md).

## Reference

| Page | Contents |
|---|---|
| [Models](reference/models.md) | Model family, I/O specs, generation settings, action embodiments, sampling defaults |
| [Benchmarks](reference/benchmarks.md) | Speed: latency & serving throughput by GPU, engine, resolution (quality benchmarks: [`evaluation/`](https://github.com/NVIDIA/cosmos/tree/main/evaluation/)) |
| [FAQ](reference/faq.md) | Troubleshooting: CUDA/driver pairing, gated checkpoints, install errors, common pitfalls |

Setup, training, and framework internals are documented in [Cosmos Framework](https://github.com/NVIDIA/cosmos-framework/tree/main/docs).

## Runnable notebooks

Every capability has a notebook in [`cookbooks/`](https://github.com/NVIDIA/cosmos/tree/main/cookbooks/) — generation (audiovisual, action, transfer), reasoning, forward/inverse dynamics, across Diffusers, Cosmos Framework, vLLM-Omni, vLLM, SGLang, TensorRT-LLM, and NIM.
