# Cosmos3 Vision Generator Fine-Tuning (SFT)

Supervised fine-tuning (SFT) of the Cosmos3 video generator on your own captioned video data. Tested on 8×H100 (80 GB).

| Recipe | Launch shell | Base model | Dataset |
| --- | --- | --- | --- |
| Vision SFT (full) | `launch_sft_vision_nano.sh` | Cosmos3-Nano | [BridgeData2-Subset-Synthetic-Captions](https://huggingface.co/datasets/nvidia/BridgeData2-Subset-Synthetic-Captions) |
| Vision SFT (LoRA) | `launch_sft_vision_super.sh` | Cosmos3-Super | same as above |
| Vision SFT (full) | `launch_sft_vision_edge.sh` | Cosmos3-Edge | same as above |

All recipes train on structured-JSON captions (`caption_json`, the model's native prompt format), so training stays aligned with inference. The Cosmos3-Edge recipe is a full fine-tune of the compact 2B dense Nemotron backbone (no audio/sound tokenizer); at 2B it also fits a 4-GPU node.

## Prerequisites

1. **Install the framework.** These recipes drive `cosmos_framework.scripts.train`, so install a cosmos-framework checkout first — follow the shared [Cosmos Framework setup](../../../README.md#cosmos-framework) (clone into `packages/cosmos3`, then `uv sync --all-extras --group=cu130-train`; use `cu128-train` on a CUDA 12.x driver).
2. **Recommended container.** For a curated CUDA + PyTorch base, NVIDIA recommends starting from the NGC PyTorch container **`nvcr.io/nvidia/pytorch:25.09-py3`** (CUDA 13; use **`:25.06-py3`** for a CUDA 12.8 driver). See the framework [setup guide](https://github.com/NVIDIA/cosmos-framework/blob/main/docs/setup.md#recommended-base-image).
3. **Activate** the framework venv so `cosmos_framework` is importable: `source <path-to>/packages/cosmos3/.venv/bin/activate`.
4. **Hugging Face access.** Some assets are license-gated — accept terms on the dataset/model pages and authenticate once with `uvx hf@latest auth login` (or export `HF_TOKEN`).
5. **Run from this directory** (`cookbooks/cosmos3/generator/audiovisual/finetune/`). Downloads, converted checkpoints, and run outputs default to `data/`, `checkpoints/`, and `outputs/` here (all git-ignored).

## Quick start

Each launcher is a complete recipe — run it from this folder and it downloads the dataset, fetches the Wan2.2 VAE, converts the base checkpoint, then runs 8-GPU training (the download/convert steps are skipped if their outputs already exist):

```shell
bash launch_sft_vision_nano.sh      # full SFT on Cosmos3-Nano
# or
bash launch_sft_vision_super.sh     # LoRA SFT on Cosmos3-Super
# or
bash launch_sft_vision_edge.sh      # full SFT on Cosmos3-Edge (2B; also fits a 4-GPU node)
```

Paths are fixed at the top of each script (under this git-ignored folder) — edit them there to put data or checkpoints on another filesystem.

These recipes default to 8 GPUs. On a 4-GPU node (e.g. GB200×4), set `--nproc_per_node=4` on the `torchrun` line in the launch script.

## LoRA fine-tuning

LoRA freezes the backbone and trains small low-rank adapters on the generation-pathway attention projections, so optimizer state is adapter-sized rather than backbone-sized — that is what lets the 32B Super tier fit a node budget a full fine-tune of that size would not.

Enabling it is a **TOML-level switch**: no code change and no separate LoRA experiment to register. `[job].experiment` stays whatever registered experiment you started from (`vision_sft_nano`, `vision_sft_edge`, …); change `[job].name` so the run gets its own output directory.

**[`launch_sft_vision_super.sh`](launch_sft_vision_super.sh) is already a LoRA recipe** — read it and its TOML, [`toml/sft_config/vision_sft_super.toml`](toml/sft_config/vision_sft_super.toml), alongside this section for a known-good set of values. To convert another recipe, copy its TOML and touch these four places:

| TOML section | Change | Why |
| --- | --- | --- |
| `[model]` | `lora_enabled = true`, plus `lora_rank` / `lora_alpha` / `lora_target_modules` | injects the adapters before FSDP wraps the network |
| `[optimizer]` | `keys_to_select = ["lora_"]`, and raise `lr` | trains the adapters only; they start at zero and are a tiny fraction of the parameters, so they tolerate — and want — a larger LR than a full fine-tune |
| `[checkpoint]` | add `"lora_"` to `keys_to_skip_loading` | the base checkpoint has no adapter tensors; let them initialize fresh instead of failing the load |
| `[model.ema]` / `[model.compile]` | disable both | EMA over a frozen backbone is wasted memory, and LoRA injection subclasses `nn.Linear`, so leave `torch.compile` off |

The `[model]` block ends up looking like the following — the numbers are the Super recipe's, shown as an example to start from rather than a required setting:

```toml
lora_enabled        = true
lora_rank           = 16     # e.g. — larger rank buys capacity at the cost of more trainable params
lora_alpha          = 32     # e.g. — commonly around 2x rank
lora_target_modules = "q_proj_moe_gen,k_proj_moe_gen,v_proj_moe_gen,o_proj_moe_gen"
```

`lora_target_modules` matches a bare name against exact child-module names, and any target containing a `.` against the full module path. The four MoE-gen projections above are the generation pathway and exist in every MoT block on all tiers (dense-FFN Edge included), so that line carries over unchanged — and because `_moe_gen` is part of the leaf name itself, they can't collide with the understanding tower.

Extending beyond attention needs more care: the gen-path FFN is `mlp_moe_gen`, but its children are named exactly like the understanding tower's (`mlp.up_proj`, …), so a bare `up_proj` would adapt both. Write those targets path-qualified instead — `mlp_moe_gen.up_proj`, and so on — and check the module tree first, since the FFN's children differ between the dense and MoE tiers.

For the LR, take the recipe's full-FT value as the starting point and go up — for example, the Super LoRA recipe runs `5e-4` where the full fine-tunes use `1e-4`.

Finally, copy the matching launch shell and point `--sft-toml` at the new TOML. Everything else in the recipe (data, VAE, checkpoint conversion), the outputs, and the [Hugging Face export](#export-to-hugging-face-safetensors) are unchanged.

## Outputs

Training writes to `outputs/train/<project>/<group>/<name>/`:

- `checkpoints/iter_<N>/` — DCP checkpoint (model / optim / scheduler / trainer state); `checkpoints/latest_checkpoint.txt` names the newest.
- `config.yaml`, launch metadata, logs, and one directory per registered callback.

## Export to Hugging Face safetensors

```shell
RUN_DIR=outputs/train/<project>/<group>/<name>
CKPT=$RUN_DIR/checkpoints/$(cat "$RUN_DIR/checkpoints/latest_checkpoint.txt")
python -m cosmos_framework.scripts.export_model \
    --checkpoint-path "$CKPT" --config-file "$RUN_DIR/config.yaml" -o "$RUN_DIR/model"
```

Use the exported `$RUN_DIR/model` with the [audiovisual inference cookbook](../README.md).

## Convert to Diffusers

Convert the export into a Diffusers pipeline:

```shell
python -m cosmos_framework.scripts.convert_model_to_diffusers \
    --checkpoint-path "$RUN_DIR/model" -o "$RUN_DIR/diffusers"
```

The input is the Hugging Face directory produced by `export_model` above (not a raw DCP checkpoint), so run the export first. See the [Export and Convert Checkpoints](../../../../../README.md#export-and-convert-checkpoints) overview for details.

## Advanced configuration

These recipes are intentionally minimal. For the full post-training reference — raw `torchrun`, resuming, every TOML field, parallelism / LoRA / EMA knobs, and the generator↔reasoner remap — see the canonical framework docs:

- [Post-Training (SFT) guide](https://github.com/NVIDIA/cosmos-framework/blob/main/docs/training.md)
- [SFT structured-TOML config reference](https://github.com/NVIDIA/cosmos-framework/blob/main/docs/sft_config.md)
- [JSONL dataset format](https://github.com/NVIDIA/cosmos-framework/blob/main/docs/dataset_jsonl.md) · [environment variables](https://github.com/NVIDIA/cosmos-framework/blob/main/docs/environment_variables.md) · [FAQ / OOM during SFT](https://github.com/NVIDIA/cosmos-framework/blob/main/docs/faq.md)

> SFT here is a multi-GPU `torchrun` job, so these cookbooks ship as launch scripts + this README rather than a one-click notebook.
