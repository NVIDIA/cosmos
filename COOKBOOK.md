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
- Recipe links point at the `main` branch of [Cosmos Framework][framework-main].
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

[framework-main]: https://github.com/NVIDIA/cosmos-framework/tree/main
[gen]: https://nvda.ws/4z3OmbC
[reason]: https://nvda.ws/4z2ezYl
[action]: https://nvda.ws/4dUeBsJ
[transfer]: https://nvda.ws/4emDe1e
[nim]: https://nvda.ws/4iztHGw
[nim-matrix]: https://nvda.ws/3U5M3G0
[transfer-auto]: https://nvda.ws/4yzgXVk
[try-auto]: https://nvda.ws/4iz2Iek
[sim-auto]: https://nvda.ws/4BjzZBt
[sim-bimanual]: https://nvda.ws/4hYEJFc
[try-bimanual]: https://nvda.ws/4yCZHP1
[sim-transfer]: https://nvda.ws/4sgWzqt
[try-sim-transfer]: https://nvda.ws/47xlyMI
[ft-gen]: https://nvda.ws/4idFCsq
[ft-reason]: https://nvda.ws/4zCYwji
[ft-policy]: https://nvda.ws/4yAmivs
[ft-fd]: https://nvda.ws/4jA1AIw
[distill]: https://nvda.ws/487r6O2
[paibench-g]: https://nvda.ws/4ze9JXK
[paibench-c]: https://nvda.ws/4z9nsPW
[physics-iq]: https://nvda.ws/4i7hIyB
[unigenbench]: https://nvda.ws/4z8byWu
[rbench]: https://nvda.ws/4z7zVU4
[vlmevalkit]: https://nvda.ws/47DwqIW
