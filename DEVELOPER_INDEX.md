<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: OpenMDW-1.1 -->

# Cosmos Developer Index

<!-- INTRO:START -->
**A head start for whatever world you’re building.** Train and run your world models faster with NVIDIA and NVLabs technology for video world models—from datasets and training libraries to fast inference runtimes and new research methods. Cosmos is an open platform that gives you a foundation to build on. Grab a useful component, accelerate your current pipeline, or build something new with Cosmos models. Take as much or as little as you need. **Explore 36 projects and 9 dataset highlights by capability below.**
<!-- INTRO:END -->

Jump to: [Data](#data--curate-and-reconstruct) · [Train](#train--start-with-cosmos-or-adapt-your-own-model) · [Accelerate](#accelerate--generate-faster) · [Deploy](#deploy--act-and-ship-safely) · [Dataset highlights](#dataset-highlights)

Browse by capability, or use Find on this page. Each project links to its upstream implementation, dataset, or research page. Check the upstream documentation for supported models, setup instructions, and terms for code, weights, data, and services.

## Selected highlights

<!-- RESULTS:START -->
- **120× faster video generation with SANA-Video 2.0.** July 2026: 720p/5s in 13.06s vs. Wan 2.2-A14B at 1556s, both on one H100. SANA-Video 2.0 5B + Sol-Engine replaces the model and pipeline; quality is not identical. [SANA / SANA-Sprint / SANA-WM](https://github.com/NVlabs/Sana) · [Result](https://arxiv.org/html/2607.21553v1)

- **32.5× faster Cosmos3-Super generation with VisualGen.** Updated October 2026: Cosmos3-Super, 189 frames at 720p, 35 steps: 383.1s on one B200 (table; Figure 8 says GB200) vs. 11.78s on a 72-GPU GB200 NVL72 rack. Multi-GPU latency scaling. [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) · [Result](https://github.com/NVIDIA/TensorRT-LLM/blob/57aef0357b9a07b4afcab5459f605fa851135238/docs/source/blogs/tech_blog/blog25_Scaling_Video_Generation_Across_NVL72_Rack_with_TensorRT-LLM.md)

- **27.7× faster MiniMax-H3 video generation with H3 Super Acceleration (Sol Engine).** August 2026: 10s of 1344x768 video, 414.1s with the published SGLang baseline (GPU not stated) vs. 14.931s on one GB200. Sol Engine alone: 132.5s (~3.1x). 4-step H3 LoRA draft + 3-step LTX refinement changes sampling and quality; total sums timed stages, excluding loading/warmup. [SANA / Sol-Engine](https://github.com/NVlabs/Sana/tree/sol-engine) · [Result](https://nvlabs.github.io/Sana/Sol-Engine/H3-Super-Acceleration/)

- **4.58× faster Cosmos3-Nano serving with TensorRT/Dynamo-Triton.** September 2026: Cosmos3-Nano, 189 frames at 720p, 35 steps: 156.6s on one GPU vs. 34.2s on eight. 1-to-8 GPU scaling on the same TensorRT/Dynamo-Triton stack; GPU model unspecified. Excludes loading/MP4 encoding; sampled-frame checks pass, outputs are not identical. [TensorRT](https://github.com/NVIDIA/TensorRT) · [Result](https://developer.nvidia.com/blog/simplifying-model-serving-across-multiple-gpus-with-nvidia-tensorrt-multi-device-integration-in-nvidia-dynamo-triton/)
<!-- RESULTS:END -->

Results reported by the project teams for the stated workloads. Follow the source for the full comparison, hardware, and quality measurements.

<!-- EXAMPLES:START -->
## Data — curate and reconstruct

| Project | What you can do | Research |
|---|---|---|
| [Cosmos Curator](https://github.com/NVIDIA/cosmos-curator) | Video dataset preparation through splitting, annotation, filtering, and deduplication. |  |
| [Cosmos Xenna](https://github.com/nvidia-cosmos/cosmos-xenna) | Orchestrate distributed video-curation stages with Ray, including custom video-captioning pipelines. **Limited maintenance.** |  |
| [NVIDIA DALI](https://github.com/NVIDIA/DALI) | Load, decode, and augment images, video, and audio for GPU training pipelines. |  |
| [nvImageCodec](https://github.com/NVIDIA/nvImageCodec) | Decode image batches on the GPU and feed PyTorch data loaders through reusable codec and zero-copy interfaces. |  |
| [Isaac Lab](https://github.com/isaac-sim/IsaacLab) | Render and save simulated RGB, depth, and segmentation sequences for visual training data. | [Paper](https://arxiv.org/abs/2511.04831) |
| [GRAIL](https://github.com/NVlabs/GRAIL) | Generate human-object interaction videos and reconstruct metric 4D trajectories for humanoid training data. **Non-commercial use.** | [Paper](https://arxiv.org/abs/2606.05160) |
| [SIL-Wheel](https://github.com/nv-tlabs/sil-wheel) | Search and annotate Physical AI videos using captions, embeddings, trajectories, and object filters to build training and evaluation slices. |  |
| [Lyra](https://github.com/nv-tlabs/lyra) | Generate camera-controlled videos and reconstruct explorable 3D scenes, with Lyra 1 and Lyra 2 implementations. | [Lyra 1 paper](https://arxiv.org/abs/2509.19296) · [Lyra 2 paper](https://arxiv.org/abs/2604.13036) |
| [ViPE](https://github.com/nv-tlabs/vipe) | Estimate camera poses and dense depth from videos for geometric annotation and Lyra video-training data preparation. | [Paper](https://arxiv.org/abs/2508.10934) |
| [Cosmos Data Factory (CDF) augmentation](https://github.com/NVIDIA/cdf-augmentation) | Augment videos through deployed Cosmos and compatible generation endpoints, with optional captioning and output checks. |  |
| [Cosmos Data Factory (CDF) auto-labeling](https://github.com/NVIDIA/cdf-auto-labeling) | Produce temporal video captions, tracked-object annotations and reusable labeling artifacts through cookbook-driven pipelines. |  |
| [FLORA](https://github.com/NVlabs/Flora) | Segment robot and human demonstration videos into action intervals and generate contextual captions, with optional outcome and process reward labels. **Non-commercial research use.** | [Paper](https://arxiv.org/abs/2610.31779) |
| [ACCV on-demand video decoder](https://github.com/NVIDIA/ACCV-Lab) | Decode sampled video frames on demand with NVDEC and PyTorch data-loader examples for video training data. |  |
| [CASCADE devkit](https://github.com/nv-tlabs/cascade-devkit) | Query driving-video annotations by entities, time and causal relationships to select and inspect scenario slices. **Published datasets require separate access.** |  |

## Train — start with Cosmos or adapt your own model

| Project | What you can do | Research |
|---|---|---|
| [Cosmos 3](https://github.com/NVIDIA/cosmos) | Start from Cosmos world models with supported video generation and action-conditioned physical-AI capabilities. | [Paper](https://arxiv.org/abs/2606.02800) |
| [Cosmos Framework](https://github.com/NVIDIA/cosmos-framework) | Train, post-train (SFT, DPO, GRPO, PPO, DiffusionNFT), and serve supported Cosmos world models with distributed training, dataset adapters, and model-specific inference paths. |  |
| [Transformer Engine](https://github.com/NVIDIA/TransformerEngine) | Use mixed-precision transformer layers and fused attention for transformer training on NVIDIA GPUs. |  |
| [NVIDIA OmniDreams](https://github.com/nv-tlabs/omni-dreams) | Post-train the OmniDreams driving world model with student initialization, bidirectional-teacher, and self-forcing distillation recipes; run inference in FlashDreams. |  |
| [LidarVerse](https://github.com/nv-tlabs/lidarverse) | Generate, forecast, and resimulate LiDAR sequences with a Cosmos-Predict2-derived diffusion model and released PAI fine-tuning recipes. | [Paper](https://arxiv.org/abs/2610.19201) |

## Accelerate — generate faster

| Project | What you can do | Research |
|---|---|---|
| [FastGen](https://github.com/NVlabs/FastGen) | Distill diffusion models for faster image and video generation using reusable few-step training methods. | [DMD2 paper (one supported method)](https://arxiv.org/abs/2405.14867) |
| [rCM / Causal-rCM](https://github.com/NVlabs/rcm) | Distill video diffusion models into few-step generators, including causal autoregressive recipes. | [Paper 1](https://arxiv.org/abs/2510.08431) · [Paper 2](https://arxiv.org/abs/2606.25473) |
| [SANA / SANA-Sprint / SANA-WM](https://github.com/NVlabs/Sana) | Explore efficient image and video models, few-step distillation, and camera-controlled world-model training. | [SANA-WM paper](https://arxiv.org/abs/2605.15178) · [SANA-Sprint paper](https://arxiv.org/abs/2503.09641) |
| [LongLive](https://github.com/NVlabs/LongLive) | Train and distill autoregressive video generators with parallel and low-precision execution paths. | [LongLive 2.0 paper](https://arxiv.org/abs/2605.18739) · [LongLive paper](https://arxiv.org/abs/2509.22622) |
| [NVIDIA Model Optimizer](https://github.com/NVIDIA/Model-Optimizer) | Quantize supported diffusion transformers and video VAEs, and explore diffusion caching and distillation APIs. |  |
| [TensorRT diffusion examples](https://github.com/NVIDIA/TensorRT/tree/release/11.2/demo/Diffusion) | Explore TensorRT diffusion examples for supported image and video models; the documented Cosmos-Predict2 Video2World path runs on the PyTorch backend only. |  |
| [FlashDreams / Ludus Renderer](https://github.com/NVIDIA/flashdreams) | Serve interactive video/world models and render HD-map conditioning with Ludus Renderer, a CUDA-native F-theta renderer for OmniDreams. |  |
| [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) | Run supported Wan, Cosmos, and other video diffusion models through VisualGen, with quantization, attention optimizations, and step caching. |  |
| [CUTLASS / CuTe DSL](https://github.com/NVIDIA/cutlass) | Build matrix and attention kernels for diffusion-transformer computation with CUDA templates and the CuTe DSL. |  |
| [FlashInfer (via TensorRT-LLM VisualGen)](https://github.com/flashinfer-ai/flashinfer) | Use FlashInfer's dense and quantized (MXFP8/NVFP4) attention kernels through the FLASHINFER attention backend in TensorRT-LLM VisualGen. Community project. |  |
| [AIPerf](https://github.com/ai-dynamo/aiperf) | Benchmark video-generation endpoints, including SGLang's asynchronous video API, with latency and throughput reports. |  |
| [CV-CUDA](https://github.com/CVCUDA/CV-CUDA) | Move image and video preprocessing onto the GPU with reusable computer-vision operators and Python/C++ interfaces. |  |
| [Context-Matched Distillation](https://github.com/nv-tlabs/cmd) | Distill Cosmos-Predict2.5 into causal video generators with short, long, and camera-conditioned inference paths. | [Paper](https://arxiv.org/abs/2608.13391) |

## Deploy — act and ship safely

| Project | What you can do | Research |
|---|---|---|
| [AlpaSim](https://github.com/NVlabs/alpasim) | Run camera-based closed-loop driving scenarios as a simulation reference for physical-AI evaluation. |  |
| [Cosmos Evaluator](https://github.com/NVIDIA/cosmos-evaluator) | Automated grading of Cosmos-generated video (hallucination, object, attribute, and physical-plausibility checks). |  |
| [NVIDIA Dynamo](https://github.com/ai-dynamo/dynamo) | Serve video diffusion models through documented vLLM-Omni, SGLang, FastVideo, and experimental TensorRT-LLM worker paths. |  |
| [PhyProbe](https://research.nvidia.com/labs/cosmos-lab/phyprobe/) | Score physical-consistency violations in generated videos with a frozen video encoder and a lightweight scoring head, without prompts or reference clips. | [Paper](https://arxiv.org/abs/2609.38377) |

## Dataset highlights

Selected NVIDIA datasets and simulation assets for world-model development. Browse the [full NVIDIA dataset catalog on Hugging Face](https://huggingface.co/nvidia/datasets) for more.

| Resource | What it’s useful for |
|---|---|
| [NVIDIA SimReady Assets](https://huggingface.co/datasets/nvidia/simready-assets) | Simulation-ready 3D assets for building scenes and environments. |
| [Physical AI Autonomous Vehicles](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles) | Real driving recordings with camera, LiDAR, and radar data for scenario mining, reconstruction, and AV model development. **Gated dataset access.** |
| [Physical AI NuRec — Driving](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles-NuRec) | Reconstructed driving scenes (USDZ, XODR maps, surface meshes) for camera-based AV simulation and testing with NVIDIA NuRec, including CARLA. **Gated dataset access.** |
| [Physical AI NuRec — Robotics](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-NuRec) | Reconstructed environments for realistic rendering and mobile-robot data generation in Isaac Sim. **Gated dataset access.** |
| [Synthetic Autonomous Driving Scenarios](https://huggingface.co/datasets/nvidia/PhysicalAI-WorldModel-Synthetic-Autonomous-Driving-Scenarios) | Multi-camera synthetic driving videos with captions, covering rare traffic interactions and varied weather and lighting. |
| [Synthetic Embodied Robot Scenes](https://huggingface.co/datasets/nvidia/PhysicalAI-WorldModel-Synthetic-Embodied-Robot-Scenes) | Simulated manipulation, collision, and humanoid-motion videos for robot world-model training across different embodiments. |
| [Synthetic Warehouse Operations Scenes](https://huggingface.co/datasets/nvidia/PhysicalAI-WorldModel-Synthetic-Warehouse-Operations-Scenes) | Annotated multi-view warehouse videos covering routine operations, collisions, near misses, and fires. |
| [Synthetic Physical Interaction Scenes](https://huggingface.co/datasets/nvidia/PhysicalAI-WorldModel-Synthetic-Physical-Interaction-Scenes) | Simulated collisions and object dynamics with physics annotations, depth, segmentation, and captions for world-model training. |
| [Cosmos3 Driving Simulation Evaluation](https://huggingface.co/datasets/nvidia/Cosmos3-Driving-Simulation-Evaluation) | Scene-control videos and matching prompts for evaluating multi-view driving generation. Camera footage is not included. |
<!-- EXAMPLES:END -->

## Pick a project. Start building.

Keep building your existing model: explore [FastGen's distillation methods](https://github.com/NVlabs/FastGen) or [Model Optimizer's diffusion examples](https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/diffusers). For Cosmos walkthroughs, start with the [Cosmos cookbooks](COOKBOOK.md). Each project provides its own implementation and setup instructions; check its terms for code, weights, and data.
