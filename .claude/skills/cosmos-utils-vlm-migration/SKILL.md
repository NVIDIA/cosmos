---
name: cosmos-utils-vlm-migration
description: >
  Redirect edits, patches, or PRs that target pre-2026-05-18 paths under
  cosmos_training/cosmos/utils/. Covers (a) the vfm/vlm consolidation — utils/vfm/vlm/,
  utils/vfm/fused_adam.py, utils/vlm/compute_flops_qwen3vl.py — and (b) the follow-up
  cleanup pass that removed dead subsystems: utils/one_logger/, utils/optim_instantiate.py,
  utils/configs/lr_scheduler.py, utils/training_telemetry/context_managers.py,
  utils/vlm/flop_calculator.py, utils/env_parsers/customization_env_parser.py,
  utils/env_parsers/inference_env_parser.py, and the load_config(enable_one_logger=...)
  parameter. Use this skill whenever a diff, cherry-pick, rebase, code-review suggestion,
  blame trail, or external snippet references the old paths/imports, OR when applying
  any upstream change that touches utils/. Triggers on: "cherry-pick", "rebase",
  "apply patch", "port this change", "merge upstream", "from cosmos.utils.vfm.vlm",
  "from cosmos.utils.vfm.fused_adam", "from cosmos.utils.one_logger",
  "compute_flops_qwen3vl", "FlopCalculator", "optim_instantiate", "enable_one_logger",
  "OneLoggerCallback", "CustomizationEnvParser", "InferenceEnvParser", or any edit whose
  target file path is under utils/vfm/vlm/, utils/one_logger/, utils/configs/, or any
  other path listed in the redirect table below. Use proactively before applying any
  change to these areas of the repo.
---

# Cosmos utils/ refactor + cleanup — pre-2026-05-18 → post-refactor mapping

On 2026-05-18 several related changes landed in `cosmos_training/cosmos/utils/`:

1. The duplicated `utils/vfm/vlm/` tree was **merged into `utils/vlm/`** (feature union).
2. `utils/vfm/fused_adam.py` was **promoted to top-level `utils/fused_adam.py`** (DTensor-aware version).
3. A follow-up cleanup pass **deleted ~2400 lines** of dead/orphan code across the utils
   tree: the broken `utils/one_logger/` subsystem, `utils/optim_instantiate.py`,
   `utils/configs/`, `utils/training_telemetry/context_managers.py`,
   `utils/vlm/flop_calculator.py`, and two `utils/env_parsers/*` files.

Any change targeting these old paths must be **rewritten** against the new layout before
it can be applied — the old files have been deleted from HEAD.

## Hard rules

1. **Never** restore deleted files. If a patch tries to add back any of:
   `utils/vfm/vlm/*`, `utils/vfm/fused_adam.py`, `utils/vlm/compute_flops_qwen3vl.py`,
   `utils/vlm/flop_calculator.py`, `utils/one_logger/*`, `utils/optim_instantiate.py`,
   `utils/configs/lr_scheduler.py`, `utils/training_telemetry/context_managers.py`,
   `utils/env_parsers/customization_env_parser.py`,
   `utils/env_parsers/inference_env_parser.py` — it's stale; redirect or drop.
2. **Always** verify the new file already contains the equivalent feature before adding
   it. The merged files are a *superset* of both forks' behavior; the change you're
   porting may already be present.
3. If you can't find an equivalent symbol in the new location, stop and ask — silent
   feature loss is worse than the consolidation churn.
4. **For deletions that originated as dead code** (one_logger, optim_instantiate,
   flop_calculator, env_parsers, etc.), do not re-introduce them just because a patch
   asks. Verify the requested behavior is genuinely needed first; the original code was
   broken or orphan when deleted.

## Path redirect table

### Consolidated (merged into new location)

| Old import / file path | New import / file path | Notes |
|---|---|---|
| `cosmos.utils.vfm.vlm.constant` | `cosmos.utils.vlm.constant` | identical names exported |
| `cosmos.utils.vfm.vlm.create_position_ids` | `cosmos.utils.vlm.create_position_ids` | `get_position_ids`, `get_rope_index_qwen3_vl` |
| `cosmos.utils.vfm.vlm.optimizer` | `cosmos.utils.vlm.optimizer` | `OptimizerConfig`, `build_optimizers`, `build_lr_schedulers` |
| `cosmos.utils.vfm.vlm.pretrained_models_downloader` | `cosmos.utils.vlm.pretrained_models_downloader` | `maybe_download_hf_model_from_s3`, `parallel_download_s3_prefix_to_dir`, `s3_dir_exists`, `has_model_weights`, `_load_s3_credentials`, `_download_from_hf_hub` |
| `cosmos.utils.vfm.fused_adam` | `cosmos.utils.fused_adam` | `FusedAdam` (DTensor-aware) |
| `cosmos.utils.vlm.compute_flops_qwen3vl` | `cosmos.tools.flops.qwen3_vl` | `compute_qwen3vl_flops_from_config` now accepts `is_causal` (defaults to True). Numeric output verified bit-identical when `is_causal=False` against the deleted local impl. The only caller (`utils/vlm/flop_calculator.py`) has since been deleted as well, so this redirect is mostly historical. |

### Deleted entirely (do NOT re-add)

| Old import / file path | Why removed |
|---|---|
| `cosmos.utils.vfm.vlm.flop_calculator` | Preserved during initial vfm/vlm→vlm merge; confirmed zero in-tree refs and removed. `FlopCalculator` class no longer exists anywhere. |
| `cosmos.utils.vlm.flop_calculator` | Same class, post-merge location. Also deleted. |
| `cosmos.utils.one_logger.*` (whole subdir, 5 files + README) | `one_logger_override_utils.py` imported `OneLoggerCallback` from `cosmos.utils.callback`, but that class never existed in the post-imaginaire4 tree (also not in the `one-logger` PyPI package). The only call site (`load_config(..., enable_one_logger=True)`) silently swallowed the `ImportError`, so OneLogger has effectively been off for the duration. The whole subdir + the `enable_one_logger` parameter on `load_config` were removed. |
| `cosmos.utils.optim_instantiate` (`get_regular_param_group`, `get_base_optimizer`, `get_base_scheduler`) | Superseded by per-pipeline optimizer builders in `utils/vfm/optimizer.py` and `utils/vlm/optimizer.py`. Zero importers when deleted. |
| `cosmos.utils.configs.lr_scheduler` (`LambdaLinearSchedulerConfig`) | LazyCall wrapper around `LambdaLinearScheduler`. Zero importers when deleted. The implementation it wrapped — `cosmos.utils.functional.lr_scheduler.LambdaLinearScheduler` — is still alive. Whole `utils/configs/` subdir gone (it only had this one file). |
| `cosmos.utils.training_telemetry.context_managers` | Orphan. The live entry point `utils.training_telemetry.telemetry` uses `utils.py` and lazy-imports `TelemetryCallback` from `callback.py`; neither path touches the context_managers file. Zero external **or** internal cross-refs at delete time. |
| `cosmos.utils.env_parsers.customization_env_parser` (`CustomizationEnvParser`) | Inference-side AWS Fleet/Lambda env vars (FT_AWS_*, LAMBDA_STAGE, FLEET_FUNCTION). Zero importers in the training tree. |
| `cosmos.utils.env_parsers.inference_env_parser` (`InferenceEnvParser`) | Inference deployment env vars (TRT_ENABLED, NIM_DEPLOYMENT, PORT, MODEL_MODULE, …). Zero importers in the training tree. |

### Signature changes

| Function | Change |
|---|---|
| `cosmos.utils.config.load_config(config_path, opts, enable_one_logger=...)` | The `enable_one_logger` keyword argument was removed when `utils/one_logger/` was deleted. New signature: `load_config(config_path: str, opts: list[str]) -> Config`. Drop the kwarg from any patch that adds it back. |

## Feature mapping inside merged files (what came from where)

If a backport touches a specific symbol/feature, this tells you whether the new file
already has it.

### `utils/vlm/optimizer.py` — `OptimizerConfig`
Contains the union of both forks:
- Legacy named freeze flags: `freeze_vision_encoder`, `freeze_mm_projector`, `freeze_llm` (both forks)
- `freeze_llm_moe_gates: bool = False` — was vlm-only (declared but not yet referenced in code as of merge)
- `trainable_params: Optional[list[str]] = None` — was vfm/vlm-only; regex whitelist; enforced in `__attrs_post_init__`
- `frozen_params: Optional[list[str]] = None` — was vfm/vlm-only; regex blacklist; mutually exclusive with `trainable_params`
- `betas` now wrapped in `tuple(...)` inside `build_optimizers` — was vfm/vlm-only bugfix

### `utils/vlm/pretrained_models_downloader.py`
Contains the union:
- `resolve_hf_model_store(credentials, bucket)` — was vlm-only; maps checkpoint-store creds to permanent HF model store
- `_load_s3_credentials(credential_path)` — was vfm/vlm-only; env-var-aware via `cosmos.utils.easy_io.backends.auto_auth` (replaces raw `json.load(open(...))`)
- `_download_from_hf_hub(model_name_or_path, include_model_weights)` — was vfm/vlm-only; HF Hub fallback when no S3 creds
- `_stream_download` (inside `parallel_download_s3_prefix_to_dir`) — was vlm-only; bypasses ETag validation for GCS-compatible buckets
- `maybe_download_hf_model_from_s3` body:
  - Local-dir short-circuit (`if os.path.isdir(model_name_or_path)`) — was vfm/vlm-only
  - No-credentials → `_download_from_hf_hub` branch — was vfm/vlm-only
  - `not INTERNAL` → `CheckpointConfig.maybe_from_uri` + `download_checkpoint_v2` branch — was vfm/vlm-only
  - Cache check accepts `vocab.json` OR `tokenizer.json` — was vlm-only (vfm/vlm checked only `vocab.json`)

### `utils/vlm/flop_calculator.py` — DELETED
Initially merged from vfm/vlm/. Subsequently determined to have zero in-tree
references (the dynamic batcher this was built for never wired it up here) and
deleted on 2026-05-18. The bit-identical FLOP numeric verification still holds
for `cosmos.tools.flops.qwen3_vl.compute_qwen3vl_flops_from_config(..., is_causal=False)`
if you ever need to rebuild this calculator.

### `utils/vlm/create_position_ids.py`, `utils/vlm/constant.py`
The vfm/vlm version was adopted wholesale. Logic-identical to the prior `utils/vlm/`
version — only docstrings, type annotations, and `Optional[T]` → `T | None` differ.

### `utils/fused_adam.py` (was `utils/vfm/fused_adam.py`)
DTensor-aware via `cosmos.utils.misc.get_local_tensor_if_DTensor`. For non-DTensor params
(the only kind the old top-level `utils/fused_adam.py` handled), behavior is equivalent:
`get_local_tensor_if_DTensor(x)` is a no-op for regular tensors. TE import path is
`transformer_engine_torch as tex` (unchanged from top-level pre-refactor).

## NOT consolidated — two `fused_adam.py` remain by design

`utils/fused_adam.py` and `utils/vlm/fused_adam.py` both still exist. They are **not**
duplicates:

- `utils/fused_adam.py`: imports `transformer_engine_torch as tex`, uses
  `cosmos.utils.misc.get_local_tensor_if_DTensor`.
- `utils/vlm/fused_adam.py`: imports `transformer_engine as te`, uses
  `te.pytorch.optimizers.multi_tensor_adam`, with an inlined `get_local_tensor_if_DTensor`.

These differ in their TE module path. Unifying them requires verifying that
`transformer_engine_torch.multi_tensor_adam*` and
`te.pytorch.optimizers.multi_tensor_adam*` resolve to equivalent CUDA kernels at the
runtime TE version. **Do not unify without that runtime verification.**

## Import sites that were redirected on 2026-05-18

These already point at the new paths in HEAD; if you see an external patch still using
the old paths, redirect:

**vfm/vlm consolidation:**
- `cosmos/model/vfm/vlm_model.py` (3 import sites)
- `cosmos/model/vfm/algorithm/loss/cross_entropy.py`
- `cosmos/data/vfm/augmentors/vlm/tokenize_data.py`
- `cosmos/data/vfm/processors/base.py`
- `cosmos/data/vfm/processors/__init__.py`
- `cosmos/utils/vfm/optimizer.py` (the `FusedAdam` lazy import)

**one_logger removal:**
- `cosmos/utils/config.py` — `load_config` lost the `enable_one_logger` parameter and the gated lazy-import block
- `scripts/train.py` — dropped `enable_one_logger=True` kwarg
- `cosmos/data/vfm/action/compute_action_stats.py` — dropped `enable_one_logger=False` kwarg

## Workflow when applying any change to the utils tree

1. **Read the patch target path.** If it matches an entry in the redirect table
   (consolidated), rewrite the path before applying. If it matches an entry in the
   "Deleted entirely" table, the patch is targeting code that no longer exists — drop
   it or escalate.
2. **Check feature mapping above.** If the change adds/modifies a feature listed under
   "Feature mapping inside merged files," confirm the merged file's current state — the
   change may already be present (in which case it's a no-op), partially present (so you
   need to merge carefully), or absent (port it).
3. **For anything calling `compute_qwen3vl_flops_from_config`:** if the change touches
   the FLOP computation, re-run the equivalence check (see `[[utils-vfm-vlm-forks]]`
   memory for context) before assuming the dynamic batcher calibration still holds.
4. **For `load_config` call sites:** if a patch passes `enable_one_logger=...`, drop
   that kwarg — the parameter was removed.
5. **Never** create a new `utils/vfm/vlm/`, `utils/one_logger/`, or `utils/configs/`
   directory, and never restore the deleted files listed above. If a patch can't be
   cleanly applied to the new layout, stop and ask the user.

## Related memory

`[[utils-vfm-vlm-forks]]` in the project memory captures the consolidation history,
the reasoning behind the leftover `utils/vlm/fused_adam.py`, and the follow-up
deletions in this same session.
