<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: OpenMDW-1.1 -->

# Cosmos Cookbook

Start from what you want to achieve. Every recipe here leverages Cosmos to help you reach an objective.

<!--
Maintainer notes (not rendered):
- Every URL lives once, in the link block at the bottom. To point a recipe at its entry skill in
  NVIDIA/cosmos-framework, change its one line there.
- Keep Framework links pinned to the selected snapshot. Before public release, verify the matching
  files in NVIDIA/cosmos-framework. If export rewrites history, update the pins together to the
  verified public release commit; do not substitute a moving main branch.
- Recipes still pending in Framework PR #129 are not part of this snapshot. Replace their Coming soon
  labels only after selecting a revision containing their entry files.
- One short name per recipe. The section supplies the verb.
- Industry and GPU-budget codes are the legend's codes, nothing else. Use the same codes in each skill's
  metadata so agents can filter on them.
- ANY means the recipe is not tied to one industry. MFG covers manufacturing and other industrial sites.
- GPU budget lists every size a documented full run needs (smoke tests do not count).
  A blank cell means the recipe does not state it yet.
- The animated NEW badge after "Simulation" is assets/brand/badge-new.svg. Keep `&nbsp;` before it: a plain
  space changes the heading anchor to #simulation- and breaks every link to #simulation.
  README.md and docs/reference/models.md use the same badge; `git grep badge-new.svg` lists every use.
  Remove them all, and this note, once Simulation is no longer new.
- Recipe links use the commit of [Framework tag `GA-final-2026-10-19` (`be540e4`)][framework-snapshot].
-->

**Jump to:** [Inference](#inference) · [Simulation](#simulation) · [Post-training](#post-training) · [Distillation](#distillation) · [Evaluation](#evaluation)

<details>
<summary>Legend: industry and GPU-budget codes</summary>

| Code | Industry | Code | GPU budget |
| :-: | --- | :-: | --- |
| `ANY` | any industry | `S` | 1 GPU, up to 96 GB |
| `AV` | autonomous vehicles | `M` | 1 GPU over 96 GB, or 2–8 GPUs |
| `ROB` | robotics | `L` | 9–64 GPUs |
| `INFRA` | smart infrastructure | `XL` | more than 64 GPUs |
| `MFG` | manufacturing and industrial sites |  |  |
| `AG` | agriculture |  |  |
| `MED` | healthcare |  |  |

</details>

## Inference

| Recipe | Industry | GPU budget |
| --- | :-: | :-: |
| **[Generation][gen]**<br><sub>Text-to-image, text-to-video, image-to-video, video-to-video, with optional audio</sub> | `ANY` | `S` `M` |
| **[Reasoning][reason]**<br><sub>Captioning, VQA, grounding, temporal localization, physical plausibility</sub> | `ANY` `INFRA` | `S` `M` |
| **[Action][action]**<br><sub>Forward dynamics, inverse dynamics, policy</sub> | `AV` `ROB` | `S` |
| **[Transfer][transfer]**<br><sub>Spatial controls: edge, blur, depth, segmentation, world-scenario map, multi-control</sub> | `ANY` `AV` | `S` `M` |
| **[NIM deployment][nim]**<br><sub>Choose a run time (generation/reasoning), run accelerated inference from one container · [support matrix][nim-matrix]</sub> | `ANY` | `S` `M` |

## Simulation&nbsp;<a href="#simulation"><img src="assets/brand/badge-new.svg" alt="New"></a>

| Recipe | Industry | GPU budget |
| --- | :-: | :-: |
| **[Multi-view multi-sensor driving SDG][transfer-auto]**<br><sub>Fine-tune Cosmos on driving data for multi-view + LiDAR joint generation · [Try example checkpoint][try-auto]</sub> | `AV` | `L` |
| **[Autoregressive driving simulation][sim-auto]**<br><sub>Build autoregressive multi-view driving simulator · [Try example checkpoint][try-auto]</sub> | `AV` | `M` |
| **[Autoregressive bimanual simulation][sim-bimanual]**<br><sub>Build interactive neural simulator for robot policy· [Try example checkpoint][try-bimanual]</sub> | `ROB` | `M` |
| **[Autoregressive sim-to-real transfer][sim-transfer]**<br><sub>Build autoregressive sim-to-real transfer model · [Try example checkpoint][try-sim-transfer]</sub> | `ROB` | `M` |
| **Surgical simulation**<br><sub>*Coming soon*</sub> | `MED` |  |

## Post-training

| Recipe | Industry | GPU budget |
| --- | :-: | :-: |
| **[Generation][ft-gen]**<br><sub>Post-train with your captioned video: full SFT or LoRA</sub> | `ANY` | `M` |
| **[Reasoning][ft-reason]**<br><sub>Alignment SFT and physical-plausibility SFT</sub> | `ANY` | `M` |
| **[Action policy][ft-policy]**<br><sub>Post-train a policy, then evaluate it closed-loop: DROID, LIBERO, RoboCasa</sub> | `ROB` | `L` `XL` |
| **[Forward dynamics][ft-fd]**<br><sub>Post-train a forward-dynamics model on DROID</sub> | `ROB` | `M` |
| **Worker safety**<br><sub>*Coming soon* · Flag hazards in industrial video</sub> | `MFG` | `S` |
| **Action Policy (Light)**<br><sub>*Coming soon* · Post-train a LIBERO-10 manipulation policy on one GPU, evaluate it closed-loop</sub> | `ROB` | `M` |

## Distillation

| Recipe | Industry | GPU budget |
| --- | :-: | :-: |
| **[4-step students][distill]**<br><sub>4-step distillation guide, text-to-image, image-to-video</sub> | `ANY` | `L` |

## Evaluation

| Benchmarks | Industry | GPU budget |
| --- | :-: | :-: |
| **Generation**<br><sub>[PAI-Bench-G][paibench-g] video · [PAI-Bench-C][paibench-c] controlled video · [PhysicsIQ][physics-iq] physics · [UniGenBench][unigenbench] images · [RBench][rbench] robotics</sub> | `ANY` `ROB` | `M` |
| **Reasoning**<br><sub>[TAR & VANTAGE][vlmevalkit] with VLMEvalKit, on any OpenAI-compatible endpoint</sub> | `INFRA` | `S` |

---

[framework-snapshot]: https://github.com/NVIDIA/cosmos-framework/commit/be540e4750cb8fe13d297ec0d32ab8909123f26d
[gen]: https://github.com/NVIDIA/cosmos-framework/blob/be540e4750cb8fe13d297ec0d32ab8909123f26d/.agents/skills/cookbooks/cosmos3/generator/audiovisual/README.md
[reason]: https://github.com/NVIDIA/cosmos-framework/blob/be540e4750cb8fe13d297ec0d32ab8909123f26d/.agents/skills/cookbooks/cosmos3/reasoner/README.md
[action]: https://github.com/NVIDIA/cosmos-framework/blob/be540e4750cb8fe13d297ec0d32ab8909123f26d/.agents/skills/cookbooks/cosmos3/generator/action/README.md
[transfer]: https://github.com/NVIDIA/cosmos-framework/blob/be540e4750cb8fe13d297ec0d32ab8909123f26d/.agents/skills/cookbooks/cosmos3/generator/transfer/README.md
[nim]: https://github.com/NVIDIA/cosmos-framework/blob/be540e4750cb8fe13d297ec0d32ab8909123f26d/.agents/skills/cookbooks/cosmos3/nim/README.md
[nim-matrix]: https://github.com/NVIDIA/cosmos-framework/blob/be540e4750cb8fe13d297ec0d32ab8909123f26d/.agents/skills/cookbooks/cosmos3/nim/support-matrix.md
[transfer-auto]: https://github.com/NVIDIA/cosmos-framework/tree/be540e4750cb8fe13d297ec0d32ab8909123f26d/.agents/skills/cosmos3-nano-transfer-auto
[try-auto]: https://github.com/NVIDIA/cosmos-framework/blob/be540e4750cb8fe13d297ec0d32ab8909123f26d/.agents/skills/cookbooks/cosmos3/simulation/auto/README.md
[sim-auto]: https://github.com/NVIDIA/cosmos-framework/tree/be540e4750cb8fe13d297ec0d32ab8909123f26d/.agents/skills/cosmos3-nano-sim-auto
[sim-bimanual]: https://github.com/NVIDIA/cosmos-framework/tree/be540e4750cb8fe13d297ec0d32ab8909123f26d/.agents/skills/cosmos3-nano-sim-bimanual
[try-bimanual]: https://github.com/NVIDIA/cosmos-framework/tree/be540e4750cb8fe13d297ec0d32ab8909123f26d/.agents/skills/cookbooks/cosmos3/simulation/bimanual
[sim-transfer]: https://github.com/NVIDIA/cosmos-framework/tree/be540e4750cb8fe13d297ec0d32ab8909123f26d/.agents/skills/cosmos3-nano-sim-transfer
[try-sim-transfer]: https://github.com/NVIDIA/cosmos-framework/tree/be540e4750cb8fe13d297ec0d32ab8909123f26d/.agents/skills/cookbooks/cosmos3/simulation/transfer
[ft-gen]: https://github.com/NVIDIA/cosmos-framework/blob/be540e4750cb8fe13d297ec0d32ab8909123f26d/.agents/skills/cookbooks/cosmos3/generator/audiovisual/finetune/README.md
[ft-reason]: https://github.com/NVIDIA/cosmos-framework/blob/be540e4750cb8fe13d297ec0d32ab8909123f26d/.agents/skills/cookbooks/cosmos3/reasoner/finetune/README.md
[ft-policy]: https://github.com/NVIDIA/cosmos-framework/blob/be540e4750cb8fe13d297ec0d32ab8909123f26d/.agents/skills/cookbooks/cosmos3/generator/action/finetune/README.md
[ft-fd]: https://github.com/NVIDIA/cosmos-framework/blob/be540e4750cb8fe13d297ec0d32ab8909123f26d/.agents/skills/cookbooks/cosmos3/generator/action/finetune/action_fd_droid_posttrain.md
[distill]: https://github.com/NVIDIA/cosmos-framework/blob/be540e4750cb8fe13d297ec0d32ab8909123f26d/.agents/skills/cookbooks/cosmos3/generator/audiovisual/distill/README.md
[paibench-g]: https://github.com/NVIDIA/cosmos-framework/blob/be540e4750cb8fe13d297ec0d32ab8909123f26d/.agents/skills/evaluation/cosmos3/generator/paibench_g/README.md
[paibench-c]: https://github.com/NVIDIA/cosmos-framework/blob/be540e4750cb8fe13d297ec0d32ab8909123f26d/.agents/skills/evaluation/cosmos3/generator/paibench_c/README.md
[physics-iq]: https://github.com/NVIDIA/cosmos-framework/blob/be540e4750cb8fe13d297ec0d32ab8909123f26d/.agents/skills/evaluation/cosmos3/generator/physics_iq/README.md
[unigenbench]: https://github.com/NVIDIA/cosmos-framework/blob/be540e4750cb8fe13d297ec0d32ab8909123f26d/.agents/skills/evaluation/cosmos3/generator/unigenbench/run_with_cosmos_framework.ipynb
[rbench]: https://github.com/NVIDIA/cosmos-framework/blob/be540e4750cb8fe13d297ec0d32ab8909123f26d/.agents/skills/evaluation/cosmos3/generator/rbench/README.md
[vlmevalkit]: https://github.com/NVIDIA/cosmos-framework/blob/be540e4750cb8fe13d297ec0d32ab8909123f26d/.agents/skills/evaluation/cosmos3/reasoner/vlmevalkit/README.md
