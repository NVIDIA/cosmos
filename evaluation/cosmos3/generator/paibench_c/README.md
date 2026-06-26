<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: OpenMDW-1.1 -->

# Cosmos3 PAI-Bench-C (Conditional Generation) Reproduction

End-to-end recipe for reproducing the PAI-Bench Conditional Generation (PAI-Bench-C) benchmark
with Cosmos3 (Nano or Super) using the native Cosmos Framework PyTorch entrypoint
(`python -m cosmos_framework.scripts.inference`) and the public
[`physical-ai-bench`](https://github.com/SHI-Labs/physical-ai-bench) evaluation library.

PAI-Bench-C evaluates video generation conditioned on spatial control signals across 600 tasks
and four control modalities:

- **edge** — Canny edge map conditioning (`canny/task_XXXX.mp4`); scored with Canny F1 score.
- **blur** — Blurred reference conditioning (`blur/task_XXXX.mp4`); scored with SSIM.
- **depth** — Depth map conditioning (`depth_vids/task_XXXX.mp4`); scored with scale-invariant RSME (si-RMSE).
- **seg** — Segmentation map conditioning (`sam2_vids/task_XXXX.mp4`); scored with mIoU / recall.

All modalities also report DOVER video quality. Each modality generates 121 frames at 30 FPS, 720p.

## Quick Start

```bash
# Set your HuggingFace token once (needed for gated model and dataset downloads)
export HF_TOKEN=hf_...

# Quick smoke-test: 4 tasks, edge modality, Cosmos3-Nano (default)
bash run_paibench_c.sh

# Full 600-task run, Cosmos3-Nano (run once per modality)
PAIBENCH_C_NUM_SAMPLES=600 PAIBENCH_C_CHECKPOINT=Cosmos3-Nano PAIBENCH_C_MODALITIES=edge  bash run_paibench_c.sh
PAIBENCH_C_NUM_SAMPLES=600 PAIBENCH_C_CHECKPOINT=Cosmos3-Nano PAIBENCH_C_MODALITIES=blur  bash run_paibench_c.sh
PAIBENCH_C_NUM_SAMPLES=600 PAIBENCH_C_CHECKPOINT=Cosmos3-Nano PAIBENCH_C_MODALITIES=depth bash run_paibench_c.sh
PAIBENCH_C_NUM_SAMPLES=600 PAIBENCH_C_CHECKPOINT=Cosmos3-Nano PAIBENCH_C_MODALITIES=seg   bash run_paibench_c.sh

# Full 600-task run, Cosmos3-Super (run once per modality)
PAIBENCH_C_NUM_SAMPLES=600 PAIBENCH_C_CHECKPOINT=Cosmos3-Super PAIBENCH_C_MODALITIES=edge  bash run_paibench_c.sh
PAIBENCH_C_NUM_SAMPLES=600 PAIBENCH_C_CHECKPOINT=Cosmos3-Super PAIBENCH_C_MODALITIES=blur  bash run_paibench_c.sh
PAIBENCH_C_NUM_SAMPLES=600 PAIBENCH_C_CHECKPOINT=Cosmos3-Super PAIBENCH_C_MODALITIES=depth bash run_paibench_c.sh
PAIBENCH_C_NUM_SAMPLES=600 PAIBENCH_C_CHECKPOINT=Cosmos3-Super PAIBENCH_C_MODALITIES=seg   bash run_paibench_c.sh

# Skip dependency install on subsequent runs
SKIP_INSTALL=1 bash run_paibench_c.sh

# Generation only — skip evaluation
SKIP_EVAL=1 bash run_paibench_c.sh

# Evaluation only — skip generation (videos already exist)
SKIP_GEN=1 bash run_paibench_c.sh
```

Alternatively, open `run_with_cosmos_framework.ipynb` for an interactive step-by-step walkthrough
with inline video previews and metric display.

## Files

- `run_paibench_c.sh` — self-contained bash script; mirrors every step of the notebook.
- `run_with_cosmos_framework.ipynb` — interactive notebook (demo case + four full-sweep cells).
- `assets/prompts.json` — 600 task entries, each with:
  - `caption` — the actual generation prompt used in the internal evaluation run (fully upsampled JSON description).
  - `video_path`, `canny_path`, `blur_path`, `depth_path`, `seg_path` — relative paths within the HF dataset to the GT video and each control-signal video.
  - `negative_prompt` — shared negative prompt.

## Dataset

The GT videos and pre-computed control signals come from the Hugging Face dataset
[`shi-labs/physical-ai-bench-conditional-generation`](https://huggingface.co/datasets/shi-labs/physical-ai-bench-conditional-generation),
cloned via `git clone` (Git LFS). Captions are read from `assets/prompts.json`.

Dataset layout (under `$PAIBENCH_C_DATASET_ROOT`):

```
physical-ai-bench-conditional-generation/
├── videos/           # 600 GT reference videos (task_0000.mp4 … task_0599.mp4)
├── canny/            # pre-computed Canny edge maps     (control input: edge)
├── blur/             # pre-computed blur references     (control input: blur)
├── depth_vids/       # pre-computed depth maps          (control input: depth)
├── sam2_vids/        # pre-computed SAM2 seg maps       (control input: seg)
├── depth_npzs/       # pre-cached GT depth arrays      (eval: ignored — GT depth recomputed on the fly)
├── sam2_pkls/        # pre-cached GT seg masks          (eval: ignored — GT seg recomputed on the fly)
├── captions/         # per-task captions (task_XXXX.json)
└── metadata.csv      # task index with paths and base captions
```

## Sampling Settings

| Setting          | edge | blur | depth | seg |
| ---------------- | ---: | ---: | ----: | --: |
| num_frames       |  121 |  121 |   121 | 121 |
| fps              |   30 |   30 |    30 |  30 |
| resolution       | 720p | 720p |  720p | 720p |
| num_steps        |   50 |   50 |    50 |  50 |
| guidance         |  3.0 |  3.0 |   3.0 | 3.0 |
| control_guidance |  1.5 |  1.5 |   1.5 | 2.0 |
| seed             | 2026 | 2026 |  2026 | 2026 |

## Reference Scores

The primary metric for each modality is **bold**. All other metrics are cross-modality diagnostics.
SI-RMSE is lower-is-better; all others are higher-is-better.

> All scores were computed using [`trungtpham/pai-bench-c-cosmos3`](https://github.com/trungtpham/pai-bench-c-cosmos3), which includes fixes from [SHI-Labs/physical-ai-bench PR #8](https://github.com/SHI-Labs/physical-ai-bench/pull/8) for depth SI-RMSE outlier blow-up and DOVER score bias. The script auto-clones this repo. Once PR #8 merges into `SHI-Labs/physical-ai-bench:main`, that will become the default.
>
> GT segmentation masks and GT depth maps are recomputed on the fly from the raw GT videos (`--force_recompute_gt_seg --force_recompute_gt_depth`) rather than using the pre-cached pkl/npz files bundled with the dataset. Predicted segmentation is always computed from the original pred video bytes (no lossy re-encode). Both choices match the internal imaginaire4 evaluation pipeline.


### Cosmos3-Nano (600 tasks)

| Modality | **Primary metric** | Canny F1 | Blur SSIM | Depth SI-RMSE ↓ | Seg mIoU | Seg Recall | DOVER |
| -------- | -----------------: | -------: | --------: | --------------: | -------: | ---------: | ----: |
| **edge** | Canny F1 **0.491** | 0.491 | 0.853 | 0.556 | 0.765 | 0.704 | 10.14 |
| **blur** | Blur SSIM **0.902** | 0.292 | 0.902 | 0.458 | 0.770 | 0.700 | 9.91 |
| **depth** | Depth si-RMSE **0.610** | 0.190 | 0.748 | 0.610 | 0.738 | 0.683 | 10.75 |
| **seg** | Seg mIoU **0.753** | 0.154 | 0.708 | 0.789 | 0.753 | 0.676 | 10.60 |

### Cosmos3-Super (600 tasks)

> Set `PAIBENCH_C_CHECKPOINT=Cosmos3-Super` to reproduce these scores.

| Modality | **Primary metric** | Canny F1 | Blur SSIM | Depth SI-RMSE ↓ | Seg mIoU | Seg Recall | DOVER |
| -------- | -----------------: | -------: | --------: | --------------: | -------: | ---------: | ----: |
| **edge** | Canny F1 **0.502** | 0.502 | 0.863 | 0.529 | 0.764 | 0.713 | 10.09 |
| **blur** | Blur SSIM **0.906** | 0.306 | 0.906 | 0.437 | 0.770 | 0.705 | 9.98 |
| **depth** | Depth SI-RMSE **0.560** | 0.200 | 0.758 | 0.560 | 0.756 | 0.687 | 10.62 |
| **seg** | Seg mIoU **0.751** | 0.159 | 0.709 | 0.747 | 0.751 | 0.683 | 10.62 |


## Notes
The reference scores above may differ slightly from those reported in the Cosmos3 paper. The paper scores were computed with the internal evaluation pipeline; the scores here use the public [`trungtpham/pai-bench-c-cosmos3`](https://github.com/trungtpham/pai-bench-c-cosmos3) library. Minor numerical differences can also arise from library version changes (PyTorch, Transformers, SAM2).

## Changes to PAI-Bench-C

The reference scores above were produced with the fork
[`trungtpham/pai-bench-c-cosmos3`](https://github.com/trungtpham/pai-bench-c-cosmos3)
rather than upstream `SHI-Labs/physical-ai-bench:main`.
The fork applies nine bug fixes:

| Area | Root cause | Fix |
| ---- | ---------- | --- |
| **Depth si-RMSE — resize** | Predicted depth resized with `cv2.INTER_NEAREST`, introducing quantisation error. | Use `INTER_AREA` when downsampling, `INTER_LINEAR` when upsampling (`metrics_canny_blur_depth.py`). |
| **Depth si-RMSE — outlier cap** | A single extreme per-pixel residual could dominate the mean-squared error and blow up SI-RMSE for an entire video. | Cap per-pixel residuals at ±10.0 before squaring (`per_pixel_error_cap=10.0` in `compute_depth_error_video_sirmse`). |
| **DOVER — deterministic** | DOVER's spatial patch sampler uses `torch.randint` / `np.random.randint` without a fixed seed, causing score variation between runs. | Seed `torch.manual_seed(0)` and `np.random.seed(0)` inside `get_technical_view()`, saving and restoring global RNG state. |
| **DOVER — original bytes** | `dover_single_task` decoded the video to a numpy array then re-encoded it to MP4 bytes via imageio (lossy H.264, CRF ≈ 28) before passing to DOVER, introducing a systematic ~1.1 point downward bias. | Read raw file bytes directly with `Path(pred_video_file).read_bytes()` — no re-encode. |
| **Canny F1 — RGB→GRAY** | `cv2.Canny` was called on raw RGB frames; OpenCV treated them as BGR, swapping the R/B channel weights in the implicit grayscale conversion. | Explicitly convert with `cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)` before calling `cv2.Canny` (`video_to_canny_and_blur.py`). |
| **Seg mIoU / Recall** | Empty masks (objects SAM2 failed to track in any frame) were included in the IoU denominator, deflating scores. | Add `is_mostly_inside()` filter in `segmentation_metrics.py` to drop fully-empty masks before matching. |
| **Seg — SAM2 version** | Bundled `third_party/Grounded-SAM-2` ships an older SAM2 with different mask-propagation logic. | Pin `sam2==1.1.0` in `pyproject.toml`; force-reinstall after the editable install so site-packages wins. Also add `frames_tracked_per_obj` to `pack2tensor()` (required by `sam2>=1.1.0`). |
| **Seg — GroundingDINO API** | `transformers>=5.0` renamed `threshold=` to `box_threshold=` in `post_process_grounded_object_detection`; the old keyword was silently ignored, disabling box filtering. | Pin `transformers==4.46.3`; use `box_threshold=` at the call site in `grounded_sam_v2.py`. |
| **Seg — deterministic sampling** | `sample_points_from_masks()` used the global `np.random.choice`, causing run-to-run recall variance of ~1 object. | Seed a per-mask `np.random.default_rng` from the mask's pixel coordinates for reproducible point sampling. |

## Requirements

- 4-GPU Linux node (configurable via `COSMOS3_NUM_GPUS`, default 4)
- `git`
- HuggingFace account with access to the [Cosmos3 model family](https://huggingface.co/collections/nvidia/cosmos-3-68302292b80adf90f68ae96b); set `HF_TOKEN` before running

The script automatically installs `uv`, `git-lfs`, and the [`physical-ai-bench`](https://github.com/SHI-Labs/physical-ai-bench) evaluation library on first run.
