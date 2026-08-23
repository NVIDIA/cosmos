# Cosmos3-Nano Action-Policy Fine-Tuning (SFT)

This example demonstrates supervised fine-tuning (SFT) of [Cosmos3-Nano](https://huggingface.co/nvidia/Cosmos3-Nano) into a robot action policy, using the action-policy recipe from [cosmos-framework](https://github.com/NVIDIA/cosmos-framework). Three embodiments are covered:

- **DROID** — reproduces [Cosmos3-Nano-Policy-DROID](https://huggingface.co/nvidia/Cosmos3-Nano-Policy-DROID): trained on real-robot DROID data, evaluated on the RoboLab simulation benchmark.
- **LIBERO-10** — reproduces the Cosmos3 paper's LIBERO-10 results: trained and evaluated on the LIBERO-10 simulation benchmark.
- **RoboCasa** — mobile-base kitchen manipulation: trained on any RoboCasa task split (target or pretrain, atomic or composite) and evaluated closed-loop in held-out kitchens.

| Recipe | Launch shell | Base model | Dataset |
| --- | --- | --- | --- |
| Policy-DROID SFT (Nano) | `launch_sft_action_policy_droid_nano.sh` | Cosmos3-Nano | [Cosmos3-DROID](https://huggingface.co/datasets/nvidia/Cosmos3-DROID) success split |
| Policy-LIBERO-10 SFT (Nano) | `launch_sft_action_policy_libero_10_nano.sh` | Cosmos3-Nano | [LIBERO_LeRobot_v3](https://huggingface.co/datasets/nvidia/LIBERO_LeRobot_v3) `libero_10` |
| Policy-LIBERO-all SFT (Nano) | `launch_sft_action_policy_libero_all_nano.sh` | Cosmos3-Nano | [LIBERO_LeRobot_v3](https://huggingface.co/datasets/nvidia/LIBERO_LeRobot_v3) all 4 suites |
| Policy-RoboCasa SFT (Nano) | `launch_sft_action_policy_robocasa_nano.sh` | Cosmos3-Nano | [RoboCasa](https://robocasa.ai) target or pretrain splits (LeRobot v3.0) |

The DROID recipe uses the registered `action_policy_droid_nano` experiment: `joint_pos` 8-D actions, proprioceptive state, `concat_view` 480p video, chunk length 32, episode-shuffle streaming, JSON-formatted action prompts (`format_prompt_as_json=True`), and the optional `keep_ranges_1_0_1.json` window filter. The reference reproduction runs lr 2e-4 (cosine, cycle 100000), generator loss_scale 10, global batch 8192 (HSDP 32x8 = 256 ranks; GB200 reference, 64 nodes x 4), for 10000 iters. The action prompt is serialized as JSON at both train and eval time, so evaluation must use the matching JSON prompt format.

The LIBERO recipe uses `frame_wise_relative` rot6d 10-D actions, `quantile_rot` normalization, `concat_view` (third-person + wrist) at 20 fps, lr 5e-5 / warmup 500 / cycle 16000, global batch 2048 (HSDP 2x8). To match the LIBERO-10 results reported in Cosmos3, we provide two presets:

- **libero-10** — `action_policy_libero_nano` + `launch_sft_action_policy_libero_10_nano.sh`; trains on `libero_10` alone (max_iter 2000).
- **libero-all** — `action_policy_libero_all_nano` + `launch_sft_action_policy_libero_all_nano.sh`; equal mix of all 4 LIBERO suites, which needs longer training (max_iter 5000).

The RoboCasa recipe uses the registered `action_policy_robocasa_nano` experiment. It is the
first of the three with a **mobile base**, so the action contract is wider than the arm-only
10-D used by LIBERO. RoboCasa's native base command is passed through unchanged
(`base_encoding="raw"`, 15-D):

```
[base_motion(4), control_mode(1), eef_pos(3), eef_rot6d(6), gripper(1)]
```

Observation is `camera_set="left_wrist"` — `agentview_left` and `eye_in_hand` concatenated
horizontally at their native 256x256 each — plus end-effector proprioception as a clean
condition token (`use_state=True`). `action_normalization=None`, because the state token is
absolute and delta statistics do not apply. Chunk length 32 at 20 fps, lr 5e-5 / warmup 500 /
cycle 16000, global batch 2048 (HSDP 8x2 = 16 ranks; 4 GB200 nodes x 4 GPUs).

The recipe is independent of which RoboCasa tasks it trains on. `TASK_SET` selects the split:

| `TASK_SET` | Tasks | RoboCasa split |
| --- | --- | --- |
| `target_atomic` (default) | 18 | `target/atomic` |
| `pretrain_atomic` | 65 | `pretrain/atomic` |
| `pretrain_all` | 300 | `pretrain` atomic + composite |

The names resolve to the task tuples in cosmos-framework's `robocasa_lerobot_dataset.py`,
which is the single source of truth. Larger splits need a longer schedule — see the examples below.

## Prerequisites

1. **Install cosmos-framework.** This recipe drives `cosmos_framework.scripts.train`, so install a cosmos-framework checkout first — follow the shared [cosmos-framework setup](../../../README.md#cosmos-framework) (clone into `packages/cosmos3`, then `uv sync --all-extras --group=cu130-train`; use `cu128-train` on a CUDA 12.x driver).
2. **Recommended container.** For a curated CUDA + PyTorch base, NVIDIA recommends starting from the NGC PyTorch container **`nvcr.io/nvidia/pytorch:25.09-py3`** (CUDA 13; use **`:25.06-py3`** for a CUDA 12.8 driver). See the cosmos-framework [setup guide](https://github.com/NVIDIA/cosmos-framework/blob/main/docs/setup.md#recommended-base-image).
3. **Activate** the cosmos-framework venv so `cosmos_framework` is importable: `source <path-to>/packages/cosmos3/.venv/bin/activate`.
4. **Hugging Face access.** The base model, Wan2.2 VAE, Cosmos3-DROID dataset, and `keep_ranges_1_0_1.json` filter are hosted on Hugging Face — authenticate once with `uvx hf@latest auth login` (or export `HF_TOKEN`) and accept any model/dataset terms first.
5. **Weights & Biases.** The TOML follows the canonical cosmos-framework example and defaults to `wandb_mode="online"`. Export `WANDB_API_KEY` for online logging, or add `job.wandb_mode=disabled` to `EXTRA_TAIL_OVERRIDES` for a local/no-W&B run.
6. **Prepare the Cosmos3-DROID dataset.** The full Cosmos3-DROID dataset is large, so this cookbook expects you to stage it explicitly. By default the launcher looks for `data/Cosmos3-DROID/success/meta/info.json` under this folder. Set `DATASET_PATH=/path/to/Cosmos3-DROID/success` to use another filesystem.
7. **Run from this directory** (`cookbooks/cosmos3/generator/action/finetune/`). Converted checkpoints, the Wan2.2 VAE, the `keep_ranges_1_0_1.json` filter, and run outputs default to `checkpoints/`, `data/`, and `outputs/` here (all git-ignored).

## Quick start

Stage the dataset, then run the launcher:

```shell
# One possible layout. The dataset is large; put it on a filesystem with enough space.
uvx hf@latest download \
    nvidia/Cosmos3-DROID \
    --repo-type dataset \
    --local-dir data/Cosmos3-DROID

bash launch_sft_action_policy_droid_nano.sh
```

The launcher is a complete local wrapper for the public cookbook:

- checks that the Cosmos3-DROID success split exists
- downloads `Wan2.2_VAE.pth` if needed
- converts `Cosmos3-Nano` to a local DCP checkpoint if needed
- downloads `keep_ranges_1_0_1.json` if needed
- launches training with `action_policy_droid_nano.toml`

The script intentionally stays close to the `cosmos-framework` example launcher: `DATASET_PATH`
is bridged to `DROID_ROOT`, `BASE_CHECKPOINT_PATH` and `WAN_VAE_PATH` are exported for the TOML,
`EXTRA_TAIL_OVERRIDES` is appended after `--`, and `NPROC_PER_NODE` / `NNODES` / `NODE_RANK` /
`MASTER_ADDR` / `MASTER_PORT` map directly to `torchrun`. The extra cookbook behavior is the
reasoner-style local prep above and enabling the released `keep_ranges_1_0_1.json` filter by default.

Paths are fixed at the top of the script (under this git-ignored folder) — edit them there or export env vars to relocate data and checkpoints:

```shell
export DATASET_PATH=/scratch/Cosmos3-DROID/success
export BASE_CHECKPOINT_PATH=/scratch/checkpoints/Cosmos3-Nano
export WAN_VAE_PATH=/scratch/checkpoints/wan22_vae/Wan2.2_VAE.pth
export FILTER_PATH=/scratch/droid/keep_ranges_1_0_1.json
bash launch_sft_action_policy_droid_nano.sh
```

The committed TOML pins the GB200 reference shape (HSDP 32x8 = 256 ranks, global
batch 8192). To run on a single 8-GPU node — e.g. a short smoke test — drop the
replicate degree to 1 alongside the iteration/batch knobs:

```shell
export EXTRA_TAIL_OVERRIDES=" \
  job.wandb_mode=disabled \
  trainer.max_iter=10 \
  checkpoint.save_iter=10 \
  model.config.parallelism.data_parallel_replicate_degree=1 \
  dataloader_train.max_samples_per_batch=32 \
"
bash launch_sft_action_policy_droid_nano.sh
```

To reproduce the reference at full global batch 8192 on fewer GPUs, keep
`data_parallel_replicate_degree=1` and raise `trainer.grad_accum_iter` (32 on one
8-GPU node) instead of shrinking the batch.

## LIBERO quick start

Each launcher stages its dataset (auto-downloaded if missing), downloads the Wan
VAE, converts the base checkpoint, and trains.

**libero-10:**

```shell
bash launch_sft_action_policy_libero_10_nano.sh
```

- downloads `nvidia/LIBERO_LeRobot_v3` `libero_10` to `data/LIBERO_LeRobot_v3/libero_10` if missing
- launches training with `action_policy_libero_10_nano.toml` (max_iter 2000)

**libero-all:**

```shell
bash launch_sft_action_policy_libero_all_nano.sh
```

- downloads all 4 suites of `nvidia/LIBERO_LeRobot_v3` to `data/LIBERO_LeRobot_v3` if missing
- launches training with `action_policy_libero_all_nano.toml` (max_iter 5000; it needs longer to converge)

Both download `Wan2.2_VAE.pth` and convert `Cosmos3-Nano` to a local DCP checkpoint if needed.
Relocate inputs via env vars, or run a short smoke test:

```shell
export LIBERO_ROOT=/scratch/LIBERO_LeRobot_v3/libero_10   # libero-all preset: the parent dir, /scratch/LIBERO_LeRobot_v3
export EXTRA_TAIL_OVERRIDES="job.wandb_mode=disabled trainer.max_iter=10 checkpoint.save_iter=10 dataloader_train.max_samples_per_batch=32"
bash launch_sft_action_policy_libero_10_nano.sh
```

Checkpoints are saved every 500 iters.

## RoboCasa quick start

Get the RoboCasa dataset from [robocasa.ai](https://robocasa.ai), convert the split you want,
then train. The example below uses `target/atomic`; `SRC_ROOT` and `TASK_SET` select any other.

The converter ships with cosmos-framework:

```shell
CONVERT=$(python -c 'import cosmos_framework, pathlib
print(pathlib.Path(cosmos_framework.__file__).parent / "scripts/convert_robocasa_to_lerobot_v30.sh")')

# RoboCasa publishes LeRobot v2.1; the pinned lerobot is v3.0-only and rejects it.
# Conversion runs on copies -- the released dataset is never modified -- and is idempotent.
SRC_ROOT=/path/to/robocasa/datasets/v1.0/target/atomic \
    bash "$CONVERT"     # -> data/robocasa_v30/<task>/<date>/lerobot/

bash launch_sft_action_policy_robocasa_nano.sh
```

Other splits: convert the matching source directory, point `ROBOCASA_ROOT` at it, and give the
larger corpus a longer schedule.

```shell
# 65 pretrain/atomic tasks. The converter takes every task it finds under SRC_ROOT.
SRC_ROOT=/path/to/robocasa/datasets/v1.0/pretrain/atomic V30_ROOT=$PWD/data/robocasa_pretrain \
    bash "$CONVERT"

TASK_SET=pretrain_atomic ROBOCASA_ROOT=$PWD/data/robocasa_pretrain \
    bash launch_sft_action_policy_robocasa_nano.sh

# 300 pretrain tasks (atomic + composite) -- convert both splits into one root, 40k steps
SRC_ROOT=/path/to/robocasa/datasets/v1.0/pretrain/composite V30_ROOT=$PWD/data/robocasa_pretrain \
    bash "$CONVERT"

TASK_SET=pretrain_all ROBOCASA_ROOT=$PWD/data/robocasa_pretrain \
    EXTRA_TAIL_OVERRIDES="trainer.max_iter=40000 scheduler.cycle_lengths=[40000] scheduler.warm_up_steps=[1250]" \
    bash launch_sft_action_policy_robocasa_nano.sh
```

The launcher resolves `TASK_SET` to its task tuple, verifies every task is staged before
starting, and names any that are missing; the loader itself skips absent tasks with a warning
rather than failing.

Relocate inputs via env vars, or run a short smoke test:

```shell
export ROBOCASA_ROOT=/scratch/robocasa_v30
# The TOML pins HSDP 8x2 = 16 ranks (4 nodes x 4 GPUs). On a single 8-GPU node, drop the
# replicate degree to 1 as well, or torchrun starts 8 processes for a 16-rank layout.
export EXTRA_TAIL_OVERRIDES=" \
  job.wandb_mode=disabled \
  trainer.max_iter=10 \
  checkpoint.save_iter=10 \
  model.config.parallelism.data_parallel_replicate_degree=1 \
  dataloader_train.max_samples_per_batch=8 \
"
bash launch_sft_action_policy_robocasa_nano.sh
```

Checkpoints are saved every 1000 iters.

### Closed-loop evaluation

RoboCasa is scored by rolling the policy out in simulation, not by held-out loss. The evaluator
lives in cosmos-framework at `cosmos_framework/simulation/robocasa/closed_loop_eval.py`, alongside
the LIBERO one.

It drives the simulator against a policy server
(`cosmos_framework.scripts.action_policy_server_robocasa`) over HTTP, so it needs **two Python
environments**: robosuite/robocasa and cosmos-framework have conflicting numpy/mujoco pins and
cannot share a venv. The server runs in the cosmos-framework venv; the evaluator runs in a
separate venv built per the [RoboCasa install guide](https://robocasa.ai). Both processes run
together — start the server first and leave it up:

```shell
# --- process 1, cosmos-framework venv -------------------------------------------------
# Serve the DCP checkpoint directly; no safetensors export is needed for evaluation.
RUN_DIR=outputs/train/<project>/<group>/<name>
CKPT=$RUN_DIR/checkpoints/$(cat "$RUN_DIR/checkpoints/latest_checkpoint.txt")

python -m cosmos_framework.scripts.action_policy_server_robocasa \
    --checkpoint-path "$CKPT/model" --config-file "$RUN_DIR/config.yaml" \
    --port 8912 --num-steps 10 --guidance 3.0 --fps 20

# --- process 2, robosuite/robocasa venv -----------------------------------------------
# cosmos-framework is NOT installed in this venv (that is the point of splitting them), but
# the evaluator only imports pure-python modules from it, so a checkout on PYTHONPATH is
# enough. --dataset-dir is the ORIGINAL RoboCasa export, not the converted copy: the
# simulator reads scene XML from it.
MUJOCO_GL=egl PYTHONPATH=/path/to/cosmos-framework \
python -m cosmos_framework.simulation.robocasa.closed_loop_eval \
    --server-url http://127.0.0.1:8912 \
    --dataset-dir /path/to/robocasa/datasets/v1.0/target/atomic/<task>/<date>/lerobot \
    --output-dir eval_out --num-test-episodes 20 \
    --action-horizon 32 --camera-set left_wrist \
    --use-state --use-base-action --base-encoding raw \
    --seed 0 --image-size 256 --cam-size 256
```

The evaluation contract must match the training recipe — a mismatch does not raise, it just
degrades the policy. For this recipe: `--action-horizon 32` (equals `chunk_length`),
`--camera-set left_wrist`, `--use-state`, and `--use-base-action --base-encoding raw` for the
15-D contract.

The server derives that same 15-D width from the checkpoint's experiment config, so the two
sides agree without being configured twice; `GET /info` reports the width in effect. Pass
`--raw-action-dim` only to override it, e.g. when serving a checkpoint whose config is not
available.

`smoke_test_robocasa_eval.sh` runs this exact pair end to end on one task for a couple of
rollouts. Use it to check the handshake before launching a full evaluation:

```shell
RUN_DIR=outputs/train/<project>/<group>/<name> \
FRAMEWORK_ROOT=/path/to/cosmos-framework \
SIM_PYTHON=/path/to/robocasa-venv/bin/python \
DATASET_DIR=/path/to/robocasa/datasets/v1.0/target/atomic/<task>/<date>/lerobot \
    bash smoke_test_robocasa_eval.sh
```

The policy server instantiates the text-safety guardrail, which downloads the gated
[`nvidia/Cosmos-Guardrail1`](https://huggingface.co/nvidia/Cosmos-Guardrail1) repo (~7 GB). Accept
its terms and authenticate (`uvx hf@latest auth login`) before the first evaluation, and do not
set `HF_HUB_OFFLINE=1` on that run.

## Outputs

Training writes to `outputs/train/<project>/<group>/<name>/`:

- `checkpoints/iter_<N>/` — DCP checkpoint (model / optim / scheduler / trainer state); `checkpoints/latest_checkpoint.txt` names the newest.
- `config.yaml`, launch metadata, logs, and one directory per registered callback.

## Export to Hugging Face safetensors

```shell
RUN_DIR=outputs/train/<project>/<group>/<name>
CKPT=$RUN_DIR/checkpoints/$(cat "$RUN_DIR/checkpoints/latest_checkpoint.txt")
python -m cosmos_framework.scripts.export_model \
    --checkpoint-path "$CKPT" \
    --config-file "$RUN_DIR/config.yaml" \
    -o "$RUN_DIR/model"
```

Use the exported `$RUN_DIR/model` with the [Cosmos3-Nano-Policy-DROID inference cookbook](../run_policy_with_cosmos_framework.md).

## Convert to Diffusers

Convert the export into a Diffusers pipeline:

```shell
python -m cosmos_framework.scripts.convert_model_to_diffusers \
    --checkpoint-path "$RUN_DIR/model" -o "$RUN_DIR/diffusers"
```

The input is the Hugging Face directory produced by `export_model` above (not a raw DCP checkpoint), so run the export first. See the [Export and Convert Checkpoints](../../../../../README.md#export-and-convert-checkpoints) overview for details.

## Advanced configuration

These recipes are intentionally minimal. For the full post-training reference — raw `torchrun`, resuming, every TOML field, the `keep_ranges_1_0_1.json` filter, and advanced HSDP/multi-node parallelism — see the canonical cosmos-framework docs:

- [Cosmos3-Nano-Policy-DROID post-training guide](https://github.com/NVIDIA/cosmos-framework/blob/main/docs/action_policy_droid_posttrain.md)
- [Post-Training (SFT) guide](https://github.com/NVIDIA/cosmos-framework/blob/main/docs/training.md)
- [SFT structured-TOML config reference](https://github.com/NVIDIA/cosmos-framework/blob/main/docs/sft_config.md)
- [environment variables](https://github.com/NVIDIA/cosmos-framework/blob/main/docs/environment_variables.md) · [FAQ / OOM during SFT](https://github.com/NVIDIA/cosmos-framework/blob/main/docs/faq.md)

> SFT here is a multi-GPU `torchrun` job, so this cookbook ships as a launch script + this README rather than a one-click notebook.
