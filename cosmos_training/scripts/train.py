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

import torch
from loguru import logger as logging

from cosmos.utils.config import Config, load_config, pretty_print_overrides
from cosmos.utils.lazy_config import LazyConfig, instantiate
from cosmos.utils.serialization import to_yaml
from cosmos.utils import distributed
from cosmos.utils.context_managers import data_loader_init, distributed_init, model_init
from cosmos.utils.launch import log_reproducible_setup
from cosmos.utils.training_telemetry import telemetry

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
    # Usage: torchrun --nproc_per_node=1 -m scripts.train --config=projects/<project>/configs/config.py

    # Get the config file from the input arguments.
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--config", help="Path to the config file", required=False)
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

    config = load_config(args.config, args.opts, enable_one_logger=True)

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
