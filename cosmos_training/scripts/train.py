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

import argparse
import os
import traceback
from typing import Any

import torch
from loguru import logger as logging

from cosmos.utils.config import Config, load_config, pretty_print_overrides
from cosmos.utils.lazy_config import LazyConfig, instantiate
from cosmos.utils.serialization import to_yaml
from cosmos.utils import distributed
from cosmos.utils.context_managers import data_loader_init, distributed_init, model_init
from cosmos.utils.launch import log_reproducible_setup
from cosmos.utils.training_telemetry import telemetry
from scripts.interface_toml import (
    read_config_py_from_toml,
    read_path_overrides,
    translate_interface_toml,
)

# ---------------------------------------------------------------------------
# --deterministic: mirrors launch_vfm.sh determinism settings.
# ---------------------------------------------------------------------------
# Two entry points because the work has to happen at two different points in the
# launch flow:
#   1. _setup_deterministic_env_and_backends() — at script entry, before any
#      CUDA init, so env vars (CUBLAS_WORKSPACE_CONFIG, FLASH_ATTENTION_DETERMINISTIC)
#      and torch backend flags take effect.
#   2. _apply_deterministic_config_overrides() — after load_config but before
#      config.freeze(), so the config mutations land before trainer.__init__
#      re-applies cudnn from config (imaginaire/trainer.py:125-126).
#
# PYTHONHASHSEED must be set externally (Python locks it at interpreter startup);
# we only warn when it's missing.


def _setup_deterministic_env_and_backends() -> None:
    """Set determinism env vars + torch backend flags. Call at script entry, pre-CUDA init."""
    if "PYTHONHASHSEED" not in os.environ:
        logging.warning(
            "PYTHONHASHSEED is not set; --deterministic is best-effort without it. "
            "For full reproducibility, prepend `PYTHONHASHSEED=42` (or any fixed value) "
            "to your launch command — Python's hash seed is fixed at interpreter startup "
            "and cannot be set retroactively."
        )
    os.environ["FLASH_ATTENTION_DETERMINISTIC"] = "1"
    # CUBLAS_WORKSPACE_CONFIG must be set before any CUBLAS init, hence script entry.
    # ":4096:8" is the value recommended by PyTorch's `torch.use_deterministic_algorithms`
    # docs for CUDA >= 10.2 — without it, deterministic cuBLAS GEMMs raise RuntimeError.
    # Refs:
    #   - https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html
    #   - https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(mode=True, warn_only=True)
    logging.info("Deterministic mode enabled.")


def _apply_deterministic_config_overrides(config: Config) -> None:
    """Apply config mutations. Call after load_config, before config.freeze().

    Forces:
      - trainer.cudnn.deterministic=True, trainer.cudnn.benchmark=False
      - trainer.seed=42 when at its default (0)
      - model.config.parallelism.use_torch_compile=False (any node with the key)
      - dataloader num_workers=0, prefetch_factor=None, dataset.detshuffle=True
        on every dataloader-shaped node in dataloader_train/dataloader_val.

    Only existing keys are mutated; projects without these fields are unaffected.
    """
    from omegaconf import DictConfig, ListConfig

    config.trainer.cudnn.deterministic = True
    config.trainer.cudnn.benchmark = False
    if config.trainer.seed == 0:
        config.trainer.seed = 42

    def _walk(cfg, mutations: dict) -> int:
        if cfg is None:
            return 0
        n = 0
        if isinstance(cfg, DictConfig):
            for k in list(cfg.keys()):
                if k in mutations:
                    target = mutations[k]
                    try:
                        if cfg[k] != target:
                            cfg[k] = target
                            n += 1
                    except Exception as e:
                        logging.warning(f"--deterministic: failed to set {k}={target!r}: {e}")
                    continue
                try:
                    v = cfg[k]
                except Exception:
                    continue
                if isinstance(v, (DictConfig, ListConfig)):
                    n += _walk(v, mutations)
        elif isinstance(cfg, ListConfig):
            for item in cfg:
                if isinstance(item, (DictConfig, ListConfig)):
                    n += _walk(item, mutations)
        return n

    # persistent_workers=False is needed alongside num_workers=0 — PyTorch's
    # DataLoader rejects (num_workers=0, persistent_workers=True) with
    # ValueError. Nested dataloaders (e.g. PackingDataLoader → RankPartitionedDataLoader)
    # pass the kwargs straight to torch.utils.data.DataLoader so they trip on this.
    dl_overrides = {
        "num_workers": 0,
        "prefetch_factor": None,
        "persistent_workers": False,
        "detshuffle": True,
    }
    n_dl = _walk(config.dataloader_train, dl_overrides) + _walk(config.dataloader_val, dl_overrides)
    # Force use_torch_compile=False: Blackwell FMHA must be forced to
    # non-deterministic mode due to an implementation limitation (no deterministic
    # FMHA kernel on Blackwell). torch.compile=True freezes kernel selection in
    # the compiled graph, so the per-kernel force cannot be applied — determinism
    # under --deterministic therefore requires the eager (non-compiled) path.
    n_tc = _walk(config.model, {"use_torch_compile": False})
    logging.info(
        f"--deterministic: applied {n_dl} dataloader override(s), "
        f"{n_tc} use_torch_compile override(s); trainer.seed={config.trainer.seed}"
    )


def _set_dataset_jsonl(config: Config, value: Any) -> str:
    """Route ``dataset_jsonl`` to whichever ``jsonl_paths`` attribute the
    resolved Config actually exposes.

    Different experiments expose the jsonl_paths list under different attribute
    trees of ``config.dataloader_train``:
      - ``data_source.jsonl_paths``                       (e.g. t2w_sft_8b_local_datapacker)
      - ``dataloader.datasets.video.dataset.jsonl_paths`` (e.g. mixed_modality_sft_8b)
    Returns the dotted attribute path that was actually set.
    """
    jsonl_list = [value] if isinstance(value, str) else list(value)
    dl = config.dataloader_train
    data_source = getattr(dl, "data_source", None)
    if data_source is not None and hasattr(data_source, "jsonl_paths"):
        data_source.jsonl_paths = jsonl_list
        return "dataloader_train.data_source.jsonl_paths"
    try:
        dataset = dl.dataloader.datasets.video.dataset
    except AttributeError:
        dataset = None
    if dataset is not None and hasattr(dataset, "jsonl_paths"):
        dataset.jsonl_paths = jsonl_list
        return "dataloader_train.dataloader.datasets.video.dataset.jsonl_paths"
    raise AttributeError(
        "dataset_jsonl: no known jsonl_paths attribute on config.dataloader_train "
        "(tried data_source.jsonl_paths and dataloader.datasets.video.dataset.jsonl_paths)"
    )


def _apply_path_overrides(config: Config, paths: dict[str, Any]) -> None:
    """Apply TOML top-level path keys to the resolved Config in-place.

    Each key has a fixed attribute target except ``dataset_jsonl``, which is
    routed by ``_set_dataset_jsonl`` based on the experiment's dataloader tree.
    """
    if not paths:
        return
    applied: list[str] = []
    if "checkpoint_path" in paths:
        config.checkpoint.load_path = paths["checkpoint_path"]
        applied.append("checkpoint.load_path")
    if "wan_vae_path" in paths:
        config.model.config.tokenizer.vae_path = paths["wan_vae_path"]
        applied.append("model.config.tokenizer.vae_path")
    if "model_path" in paths:
        config.model.config.policy.backbone.model_name = paths["model_path"]
        applied.append("model.config.policy.backbone.model_name")
    if "dataset_jsonl" in paths:
        applied.append(_set_dataset_jsonl(config, paths["dataset_jsonl"]))
    logging.info(f"[interface.toml] applied path overrides: {applied}")


@logging.catch(reraise=True)
@telemetry.monitor
def launch(config: Config, args: argparse.Namespace) -> None:
    # Need to initialize the distributed environment before calling config.validate() because it tries to synchronize
    # a buffer across ranks. If you don't do this, then you end up allocating a bunch of buffers on rank 0, and also that
    # check doesn't actually do anything.
    with distributed_init():
        distributed.init()

    # Apply --deterministic config-level overrides before validate/freeze/trainer-init
    # so (a) validate inspects the config the trainer will actually consume, and
    # (b) trainer.__init__ doesn't undo the script-level backends settings
    # (imaginaire/trainer.py:125-126 re-applies cudnn from config).
    if args.deterministic:
        _apply_deterministic_config_overrides(config)
    # Check that the config is valid
    config.validate()
    # Freeze the config so developers don't change it during training.
    config.freeze()  # type: ignore
    trainer = config.trainer.type(config)
    # Setup the miscellaneous stuff for reproducibility.
    log_reproducible_setup(config, args)

    if args.attach_vscode_debugger:
        print(f"RANK: {os.environ['RANK']}")
        if os.environ["RANK"] == "0":
            import debugpy  # noqa: T100

            debugpy.listen(3002)  # noqa: T100
            print("Waiting for debugger to attach. Listening on port 3002...")
            debugpy.wait_for_client()  # noqa: T100

    with model_init():
        model = instantiate(config.model, **config.model_init_kwargs())

    # Create the dataloaders.
    with data_loader_init():
        dataloader_train = instantiate(config.dataloader_train)
        dataloader_val = instantiate(config.dataloader_val)

    # Start training
    trainer.train(
        model,
        dataloader_train,
        dataloader_val,
    )


if __name__ == "__main__":
    # Usage:
    #   torchrun --nproc_per_node=1 -m scripts.train --config=configs/base/config.py
    #   torchrun --nproc_per_node=1 -m scripts.train --config=.../config.py --toml=interface.toml
    #   torchrun --nproc_per_node=1 -m scripts.train --toml=interface.toml
    # When --config and --toml are both given, --toml is treated as the flat
    # interface.toml schema (see toml/vfm_example.toml).
    # scripts/interface_toml.py translates it into Hydra overrides; the variant
    # (vfm-vlm vs vfm-base) is inferred from the --config path. When --toml is
    # alone, the TOML may self-declare the paired config.py via a top-level
    # ``config = "vfm"|"vlm"`` field (interface mode); otherwise it is loaded
    # as a full structured TOML (root _target_ etc.).

    # Get the config file from the input arguments.
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument(
        "--config",
        help="Path to base config (.py/.yaml/.toml). With --toml this is the Hydra base.",
        required=False,
    )
    parser.add_argument(
        "--toml",
        help=(
            "TOML path. Alone: full structured TOML config. With --config: interface.toml "
            "translated to Hydra overrides via scripts/interface_toml.py."
        ),
        required=False,
    )
    parser.add_argument(
        "opts",
        help="""
Modify config options at the end of the command. For Yacs configs, use
space-separated "PATH.KEY VALUE" pairs.
For python-based LazyConfig, use "path.key=value".
        """.strip(),
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--dryrun",
        action="store_true",
        help="Do a dry run without training. Useful for debugging the config.",
    )
    parser.add_argument(
        "--attach_vscode_debugger",
        action="store_true",
        help="Debug mode. Will start a debugpy server at 0.0.0.0:3002.",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help=(
            "Enable deterministic mode (mirrors launch_vfm.sh). Auto-applies env: "
            "CUBLAS_WORKSPACE_CONFIG=:4096:8, FLASH_ATTENTION_DETERMINISTIC=1; torch backends: "
            "cudnn.deterministic=True, cudnn.benchmark=False, "
            "use_deterministic_algorithms(warn_only=True); config: trainer.cudnn.{deterministic, "
            "benchmark}, trainer.seed=42 (when at default 0), "
            "model.config.parallelism.use_torch_compile=False, and for every dataloader in "
            "dataloader_train/dataloader_val: num_workers=0, prefetch_factor=None, "
            "dataset.detshuffle=True. PYTHONHASHSEED must be set externally (e.g. "
            "`PYTHONHASHSEED=42 torchrun ...`) since Python locks it in at interpreter startup."
        ),
    )
    args = parser.parse_args()

    if args.deterministic:
        _setup_deterministic_env_and_backends()

    # Path overrides declared at the top of the TOML (checkpoint_path,
    # wan_vae_path, dataset_jsonl, model_path) are not Hydra-style strings —
    # they're applied to the resolved Config object post-load.
    path_overrides: dict[str, Any] = {}
    if args.config and args.toml:
        config_path = args.config
        interface_overrides = translate_interface_toml(args.toml, args.config)
        args.opts = list(args.opts or []) + interface_overrides
        path_overrides = read_path_overrides(args.toml)
    elif args.toml:
        # If the TOML self-declares its paired config.py (top-level `config = "vfm"|"vlm"`),
        # treat it as an interface TOML; otherwise load it as a full structured TOML.
        toml_declared_config = read_config_py_from_toml(args.toml)
        if toml_declared_config is not None:
            config_path = toml_declared_config
            interface_overrides = translate_interface_toml(args.toml, config_path)
            args.opts = list(args.opts or []) + interface_overrides
            path_overrides = read_path_overrides(args.toml)
        else:
            config_path = args.toml
    elif args.config:
        config_path = args.config
    else:
        parser.error("Provide --config and/or --toml.")

    config = load_config(config_path, args.opts)
    _apply_path_overrides(config, path_overrides)

    if args.dryrun:
        logging.info(
            "Config:\n" + config.pretty_print(use_color=True) + "\n" + pretty_print_overrides(args.opts, use_color=True)
        )
        os.makedirs(config.job.path_local, exist_ok=True)
        try:
            to_yaml(config, f"{config.job.path_local}/config.yaml")
        except Exception:
            logging.error("to_yaml failed, falling back to LazyConfig.save_yaml:")
            logging.error(f"Traceback: {traceback.format_exc()}")
            LazyConfig.save_yaml(config, f"{config.job.path_local}/config.yaml")
        print(f"{config.job.path_local}/config.yaml")
    else:
        # Launch the training job.
        launch(config, args)
