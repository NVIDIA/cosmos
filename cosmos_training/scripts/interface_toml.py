# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Translate ``toml/*.toml`` style configs into Hydra overrides.

The interface schema (see ``toml/vfm_example.toml`` /
``toml/vlm_example.toml``) is a flat, snake_case TOML with top-level
sections ``[job]``, ``[train]``, ``[train.train_policy]``, ``[train.ckpt]``,
``[policy]``, ``[policy.parallelism]``, ``[logging]``.

Two mapping dicts route the same interface key to the appropriate Hydra path
depending on the variant inferred from ``--config``:

* ``INTERFACE_TO_HYDRA_VFM_VLM`` — when ``--config`` is under ``configs/base/vlm/...``.
* ``INTERFACE_TO_HYDRA_VFM_BASE`` — otherwise (vfm-base / MoT).

Each map keys on the flattened dotted-path of the interface TOML
(e.g. ``train.optm_betas``). Each value is a list of *rules*. A rule is:

* ``str`` — a Hydra dotted-path; the interface value is forwarded unchanged.
* ``(str, callable)`` — a Hydra dotted-path plus a one-argument transform. The
  callable may return ``SKIP`` to suppress emitting that override.

A single interface key may write to multiple Hydra paths. Unknown interface
keys (or keys whose rule list is empty) emit a WARN log and are otherwise
ignored, so partial migrations stay safe.
"""

from __future__ import annotations

import json
import tomllib
from collections.abc import Callable
from typing import Any, Union

from loguru import logger as logging

SKIP = object()  # transforms that return this sentinel skip emission entirely

Rule = Union[str, tuple[str, Callable[[Any], Any]]]
MappingDict = dict[str, list[Rule]]


# --- transforms ------------------------------------------------------------


def _is_async(value: Any) -> bool:
    return str(value).lower() == "async"


def _ckpt_type_sku(enabled: Any) -> str:
    # interface.toml: enable_checkpoint=true → ckpt_type=dcp; false → dummy (debug)
    return "dcp" if bool(enabled) else "dummy"


def _disable_wandb_if_missing(loggers: Any) -> Any:
    # interface.toml: [logging].logger = ["console", "wandb"]
    # Default callbacks SKU already wires up the wandb callback, so emit nothing in
    # the common case. If the user explicitly drops wandb, set the callback to null.
    if isinstance(loggers, (list, tuple)) and "wandb" in loggers:
        return SKIP
    return None  # → "null" override


def _as_list(value: Any) -> list[Any]:
    return [value]


# --------------------------------------------------------------------------- #
#  vfm-vlm — maps to configs.base.vlm.defaults.config.Config                  #
# --------------------------------------------------------------------------- #
INTERFACE_TO_HYDRA_VFM_VLM: MappingDict = {
    # ---------- §1. [job] ----------
    "job.project": ["job.project"],
    "job.group": ["job.group"],
    "job.name": ["job.name"],
    "job.wandb_mode": ["job.wandb_mode"],

    # ---------- §2. trainer loop ----------
    "train.max_iter": ["trainer.max_iter"],
    "train.logging_iter": ["trainer.logging_iter"],
    "train.run_validation": ["trainer.run_validation"],
    "train.upload_reproducible_setup": ["upload_reproducible_setup"],

    # ---------- §6. optimizer ----------
    "train.optm_name": ["optimizer.config.name"],
    "train.optm_fused": ["optimizer.config.fused"],
    # scheduler.{init_lr,end_lr,lr,lr_decay_iters} are OmegaConf interpolations
    # off optimizer.config.* / trainer.max_iter — see warmup_cosine_lr SKU at
    # configs/base/vlm/defaults/optimizer.py. Setting the optimizer side
    # propagates automatically.
    "train.optm_init_lr": ["optimizer.config.init_lr"],
    "train.optm_end_lr": ["optimizer.config.end_lr"],
    "train.optm_weight_decay": ["optimizer.config.weight_decay"],
    "train.optm_betas": ["optimizer.config.betas"],

    # ---------- §11. callbacks ----------
    "train.optm_grad_norm_clip": ["trainer.callbacks.grad_clip.clip_norm"],

    # ---------- §7. scheduler ----------
    "train.optm_warmup_steps": ["scheduler.warmup_iters"],
    # decay type swaps the scheduler SKU via Hydra defaults override.
    "train.optm_decay_type": [
        (
            "scheduler",
            lambda v: {"cosine": "warmup_cosine_lr"}.get(str(v).lower(), str(v)),
        ),
    ],

    # ---------- §5. precision / FSDP ----------
    # cosmos_training's TrainConfig has no `param_dtype` — model/activation dtype
    # lives on ParallelismConfig.precision (see configs/base/vlm/defaults/training.py
    # comment). Both the top-level policy.parallelism and the model.config mirror
    # must be set; the latter is what the model actually reads.
    "train.param_dtype": [
        "policy.parallelism.precision",
        "model.config.policy.parallelism.precision",
    ],
    "train.master_dtype": ["train.master_dtype"],
    "train.fsdp_reduce_dtype": ["train.fsdp_reduce_dtype"],
    "train.fsdp_offload": ["train.fsdp_offload"],
    "train.fsdp_reshard_after_forward": ["train.fsdp_reshard_after_forward"],

    # ---------- §5. compile ----------
    # cosmos_training: no `train.compile` field; torch.compile is on
    # ParallelismConfig.use_torch_compile.
    "train.compile": [
        "policy.parallelism.use_torch_compile",
        "model.config.policy.parallelism.use_torch_compile",
    ],

    # ---------- §5. global batch ----------
    "train.global_batch_per_dp": ["train.train_batch_per_replica"],

    # ---------- §8. train_policy ----------
    "train.train_policy.experiment": ["experiment"],
    # data_setting.max_batch_size drives the dataloader micro batch via
    # interpolation. The dataclass field train.train_policy.mini_batch is
    # metadata only; only the meaningful target is mirrored to keep the YAML
    # diff against CLI overrides clean.
    "train.train_policy.mini_batch": ["data_setting.max_batch_size"],
    "train.train_policy.dataloader_pool_size": ["dataloader_train.pool_size"],
    "train.train_policy.dataloader_num_workers": ["data_setting.num_data_workers"],
    "train.train_policy.dataloader_prefetch_factor": ["data_setting.data_prefetch_factor"],

    # ---------- §10. checkpoint ----------
    "train.ckpt.enable_checkpoint": [("ckpt_type", _ckpt_type_sku)],
    "train.ckpt.checkpoint_storage": ["checkpoint"],
    "train.ckpt.save_freq": ["checkpoint.save_iter"],
    "train.ckpt.save_mode": [("checkpoint.dcp_async_mode_enabled", _is_async)],
    "train.ckpt.load_path": ["checkpoint.load_path"],

    # ---------- §3. model ----------
    "policy.model_name_or_path": ["model.config.policy.model_name_or_path"],
    # model_safetensor_path fans out to all three pretrain_weights_path_*.
    "policy.model_safetensor_path": [
        "model.config.policy.pretrain_weights_path_vlm",
        "model.config.policy.pretrain_weights_path_llm",
        "model.config.policy.pretrain_weights_path_vit",
    ],
    "policy.model_max_length": ["model.config.policy.model_max_length"],
    # cosmos_training: no per-PolicyConfig flag; activation checkpointing lives
    # on ParallelismConfig.
    "policy.model_gradient_checkpointing": [
        "model.config.policy.parallelism.use_activation_checkpointing",
    ],

    # ---------- §4. parallelism ----------
    # cosmos_training uses long descriptive names rather than the i4 short
    # aliases. Unsupported axes (tp/ep/pp) are left empty so they WARN rather
    # than crash Hydra.
    "policy.parallelism.dp_shard_size": [
        "model.config.policy.parallelism.data_parallel_shard_degree",
    ],
    "policy.parallelism.dp_replicate_size": [
        "model.config.policy.parallelism.data_parallel_replicate_degree",
    ],
    "policy.parallelism.cp_size": [
        "model.config.policy.parallelism.context_parallel_shard_degree",
    ],
    "policy.parallelism.tp_size": [],
    "policy.parallelism.ep_size": [],
    "policy.parallelism.pp_size": [],

    # ---------- §11. logging ----------
    "logging.logger": [("trainer.callbacks.wandb", _disable_wandb_if_missing)],
}


# --------------------------------------------------------------------------- #
#  vfm-base — maps to configs.base.config.Config (MoT)                        #
# --------------------------------------------------------------------------- #
INTERFACE_TO_HYDRA_VFM_BASE: MappingDict = {
    # ------- [job] -------
    "job.project": ["job.project"],
    "job.group": ["job.group"],
    "job.name": ["job.name"],
    "job.wandb_mode": ["job.wandb_mode"],

    # ------- [train] (training loop) -------
    "train.max_iter": ["trainer.max_iter"],
    "train.logging_iter": ["trainer.logging_iter"],
    "train.run_validation": ["trainer.run_validation"],
    "train.upload_reproducible_setup": ["upload_reproducible_setup"],

    # ------- [train] (optimizer) -------
    # vfm-base optimizer is a build_optimizer LazyCall with hard-coded
    # optimizer_type per SKU (fusedadamw / adamw). Pick the SKU via Hydra
    # defaults override instead of trying to write to a non-existent `name`.
    "train.optm_name": [
        (
            "optimizer",
            lambda v: {"FusedAdam": "fusedadamw", "AdamW": "adamw"}.get(str(v), str(v).lower()),
        ),
    ],
    "train.optm_fused": ["optimizer.fused"],
    # init_lr/end_lr in vfm-base fold into scheduler f_start/f_min relative to
    # optimizer.lr — we can't compute without lr. Leave empty so partial
    # mappings stay safe.
    "train.optm_init_lr": [],
    "train.optm_end_lr": [],
    "train.optm_weight_decay": ["optimizer.weight_decay"],
    "train.optm_betas": ["optimizer.betas"],
    "train.optm_grad_norm_clip": ["trainer.callbacks.grad_clip.clip_norm"],

    # ------- [train] (scheduler) -------
    # vfm-base lambdacosine uses warm_up_steps as a LIST.
    "train.optm_warmup_steps": [("scheduler.warm_up_steps", _as_list)],
    # decay type for vfm-base flips the scheduler SKU; can't be expressed by
    # setting a scalar field. Leave empty; caller can extend.
    "train.optm_decay_type": [],

    # ------- [train] (precision / FSDP / compile) -------
    "train.param_dtype": ["model.config.parallelism.precision"],
    "train.compile": ["model.config.parallelism.use_torch_compile"],
    "train.master_dtype": [],
    "train.fsdp_reduce_dtype": [],
    "train.fsdp_offload": [],
    "train.fsdp_reshard_after_forward": [],

    # ------- [train] (batch) -------
    # vfm-base has no single batch-size knob: per-dataset batch_size lives
    # inside dataloader_train.dataloaders.<name>.... — leave empty.
    "train.global_batch_per_dp": [],

    # ------- [train.train_policy] -------
    "train.train_policy.experiment": ["experiment"],
    "train.train_policy.mini_batch": ["trainer.grad_accum_iter"],
    "train.train_policy.dataloader_pool_size": [],
    "train.train_policy.dataloader_num_workers": [],
    "train.train_policy.dataloader_prefetch_factor": [],

    # ------- [train.ckpt] -------
    "train.ckpt.enable_checkpoint": [("ckpt_type", _ckpt_type_sku)],
    "train.ckpt.checkpoint_storage": ["checkpoint"],
    "train.ckpt.save_freq": ["checkpoint.save_iter"],
    "train.ckpt.save_mode": [("checkpoint.dcp_async_mode_enabled", _is_async)],
    "train.ckpt.load_path": ["checkpoint.load_path"],

    # ------- [policy] -------
    # vfm-base swaps PolicyConfig for VLMConfig + parallelism flags.
    "policy.model_name_or_path": ["model.config.vlm_config.model_name"],
    "policy.model_safetensor_path": ["model.config.vlm_config.checkpoint_path"],
    # model_max_length has no scalar equivalent in vfm-base; SKU-driven.
    "policy.model_max_length": [],
    "policy.model_gradient_checkpointing": [
        "model.config.parallelism.use_activation_checkpointing",
    ],

    # ------- [policy.parallelism] -------
    "policy.parallelism.dp_shard_size": ["model.config.parallelism.data_parallel_shard_degree"],
    "policy.parallelism.dp_replicate_size": ["model.config.parallelism.data_parallel_replicate_degree"],
    "policy.parallelism.cp_size": ["model.config.parallelism.context_parallel_shard_degree"],
    # vfm-base has no tp, ep, pp — silently warn via empty list.
    "policy.parallelism.tp_size": [],
    "policy.parallelism.ep_size": [],
    "policy.parallelism.pp_size": [],

    # ------- [logging] -------
    "logging.logger": [("trainer.callbacks.wandb", _disable_wandb_if_missing)],
}


VARIANT_VLM = "vfm-vlm"
VARIANT_BASE = "vfm-base"

# Maps a top-level interface TOML ``config = "..."`` field to the paired
# ``configs/base/...config.py`` path. Lets a TOML self-declare its variant so
# launch scripts don't need to pass ``--config=...`` separately.
CONFIG_FIELD_TO_PY = {
    "vfm": "configs/base/config.py",
    "vlm": "configs/base/vlm/config.py",
}

# Top-level path keys the TOML may declare in place of CLI-tail overrides.
# train.py applies them post-load: the first three map to fixed attribute
# paths on the resolved Config; ``dataset_jsonl`` is routed by hasattr probe
# because different experiments expose jsonl_paths under different trees.
PATH_OVERRIDE_KEYS = ("checkpoint_path", "wan_vae_path", "dataset_jsonl", "model_path")

# Top-level keys consumed by the launcher, not by translate_interface_toml.
_LAUNCHER_DIRECTIVE_KEYS = ("config",) + PATH_OVERRIDE_KEYS


def read_config_py_from_toml(toml_path: str) -> str | None:
    """Return the ``configs/base/...config.py`` path declared by a TOML's
    top-level ``config`` field, or ``None`` if the field is absent.

    Raises ``ValueError`` if the field is present but not one of the known
    variant strings.
    """
    with open(toml_path, "rb") as f:
        raw = tomllib.load(f)
    variant = raw.get("config")
    if variant is None:
        return None
    if variant not in CONFIG_FIELD_TO_PY:
        raise ValueError(
            f"{toml_path}: top-level `config` = {variant!r} is not one of "
            f"{list(CONFIG_FIELD_TO_PY)}"
        )
    return CONFIG_FIELD_TO_PY[variant]


def read_path_overrides(toml_path: str) -> dict[str, Any]:
    """Return any ``PATH_OVERRIDE_KEYS`` top-level fields the TOML declares.

    Missing keys are simply absent from the result. train.py routes each one
    onto the resolved Config after ``load_config`` returns.
    """
    with open(toml_path, "rb") as f:
        raw = tomllib.load(f)
    return {k: raw[k] for k in PATH_OVERRIDE_KEYS if k in raw}


def _detect_variant(config_path: str | None) -> str:
    if config_path is None:
        raise ValueError("--toml with interface schema requires --config to choose a mapping dict")
    return VARIANT_VLM if "/vlm/" in config_path else VARIANT_BASE


def _select_mapping(variant: str) -> MappingDict:
    if variant == VARIANT_VLM:
        return INTERFACE_TO_HYDRA_VFM_VLM
    if variant == VARIANT_BASE:
        return INTERFACE_TO_HYDRA_VFM_BASE
    raise ValueError(f"unknown interface.toml variant: {variant!r}")


def _flatten(prefix: str, value: Any, out: dict[str, Any]) -> None:
    if isinstance(value, dict):
        for k, v in value.items():
            key = f"{prefix}.{k}" if prefix else k
            _flatten(key, v, out)
        return
    out[prefix] = value


def _to_override_literal(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, (int, float)):
        return str(value)
    return json.dumps(value, separators=(",", ":"))


def translate_interface_toml(toml_path: str, config_path: str | None) -> list[str]:
    """Return Hydra-style ``key=value`` override strings for the interface TOML."""
    with open(toml_path, "rb") as f:
        raw = tomllib.load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"Expected dict-like TOML, got {type(raw)}")
    # Top-level launcher directives (config variant + PATH_OVERRIDE_KEYS) are
    # consumed by read_config_py_from_toml() / read_path_overrides(); they
    # aren't Hydra overrides.
    for key in _LAUNCHER_DIRECTIVE_KEYS:
        raw.pop(key, None)

    variant = _detect_variant(config_path)
    mapping = _select_mapping(variant)
    logging.info(f"[interface.toml] variant={variant}, source={toml_path}")

    flat: dict[str, Any] = {}
    _flatten("", raw, flat)

    overrides: list[str] = []
    for key, value in flat.items():
        rules = mapping.get(key)
        if rules is None:
            logging.warning(f"[interface.toml] unknown key {key!r} (variant={variant}); ignoring")
            continue
        if not rules:
            logging.warning(
                f"[interface.toml] key {key!r} has no mapping for variant {variant}; ignoring"
            )
            continue
        for rule in rules:
            if isinstance(rule, str):
                target, transformed = rule, value
            else:
                target, transform = rule
                transformed = transform(value)
            if transformed is SKIP:
                continue
            overrides.append(f"{target}={_to_override_literal(transformed)}")
    return overrides
