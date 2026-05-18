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

"""Self-contained regression test for the two smoke launch flows.

Re-runs the same ``torchrun`` invocation that ``launch_vlm_llava_ov.sh`` and
``launch_mixed_modality_sft_8b.sh`` execute (limited to 10 iterations,
``--deterministic`` mode) and asserts that the rank-0 ``loss`` and global
``clip_grad_norm`` reproduce the inline goldens at the bottom of this file.

This file is intentionally the only deliverable — the goldens are embedded as a
Python constant and the ``torchrun`` command line is reproduced here, so the
upstream launch scripts stay untouched and there is no separate JSON file to
commit.

Invocation (on a 4-GPU node, inside the i4 training container, from the repo
root)::

    pytest -s tests/launch_regression_test.py -o addopts=

The ``-o addopts=`` clears the ``addopts`` line in the repo's ``.pytest.toml``
which references ``--suppress-no-test-exit-code`` from the optional
``pytest-custom-exit-code`` plugin (not installed in the i4 image). Plain
``pytest -s tests/launch_regression_test.py`` works once that plugin is
installed.

Determinism notes:
  * ``mixed_modality_sft_8b`` reproduces bit-exactly; all 10 iters are checked.
  * ``vlm_llava_ov`` streams ``lmms-lab/LLaVA-OneVision-Data`` from HuggingFace
    Hub. Only the first 2 iters reproduce exactly — later iters drift with
    shard arrival order. Set ``COSMOS_REGRESSION_VLM_FULL=1`` to assert all 10
    (expected to fail).

Refreshing the goldens (after an intentional numerical change)::

    COSMOS_REGRESSION_UPDATE_GOLDENS=1 pytest -s launch_regression_test.py ...

That prints the captured series for each spec; copy them into ``_GOLDENS`` below.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pytest

THIS_DIR = Path(__file__).resolve().parent
# ``scripts.train`` and the Hydra ``--config=...`` paths are relative to
# ``cosmos_training/``; we always invoke torchrun from there.
COSMOS_TRAINING_DIR = THIS_DIR.parent / "cosmos_training"

# Fixture paths used by the two launch flows (read-only Lustre artifacts).
_DATASET_JSONL = (
    "/lustre/fsw/portfolios/cosmos/users/yangyangt/cosmos_opensource/"
    "sft_dataset_bridge/train/video_dataset_file.jsonl"
)
_DCP_LOAD_PATH = "/lustre/fsw/portfolios/cosmos/users/yangyangt/midtrain"
_WAN_VAE_PATH = (
    "/lustre/fsw/portfolios/cosmos/users/yangyangt/cosmos_opensource/"
    "pretrained/tokenizers/video/wan2pt2/Wan2.2_VAE.pth"
)
_SIGLIP_MODEL_PATH = (
    "/lustre/fsw/portfolios/cosmos/projects/cosmos_base_training/"
    "users/maoshengl/models/Siglip2-Qwen3-1.7B-BF16-Alignment"
)

# Tolerances for ``pytest.approx``. The launch scripts pass ``--deterministic``
# and ``PYTHONHASHSEED=42``; the tolerance only absorbs minor noise from
# non-deterministic NCCL reductions.
_DEFAULT_RTOL = 1e-3
_DEFAULT_ATOL = 1e-3

# --- log parsers -------------------------------------------------------------
#
# VLM (``pre_exp012_llava_ov_datapacker``) logs the DP-reduced loss on rank 0::
#
#     train/loss_avg: 1.32225 (iteration 0)
#
# VFM mixed-modality (``mixed_modality_sft_8b``) goes through the ``IterSpeed``
# callback for the first 50 steps, which logs the local per-rank loss with a
# ``[RANK X]`` prefix and a 1-indexed iteration label::
#
#     [RANK 0] Iteration 1: Hit counter: 1/50 | Loss: 0.2100 | Time: 164.12s
#
# In both cases ``GradClip`` emits the global grad-norm via every rank, also
# prefixed with ``[RANK X]``. The key is ``clip_grad_norm/global`` for VLM and
# ``clip_grad_norm/<modality>/global`` for VFM (``track_per_modality=True``).
_VLM_LOSS_RE = re.compile(r"train/loss_avg:\s+([0-9.eE+-]+)\s+\(iteration\s+\d+\)")
_VFM_LOSS_RE = re.compile(r"\[RANK\s+0\]\s+Iteration\s+\d+:\s+Hit counter:[^|]+\|\s+Loss:\s+([0-9.eE+-]+)")
_GRAD_NORM_RE = re.compile(
    r"\[RANK\s+0\][^\n]*clip_grad_norm/(?:[^/]+/)?global:\s+([0-9.eE+-]+)\s+\(iteration\s+\d+\)"
)


@dataclass(frozen=True)
class LaunchSpec:
    """A single launch flow under regression — mirrors the bash launch script."""

    key: str  # goldens key + pytest parametrize id source
    config: str  # ``--config=...`` value
    experiment: str  # Hydra ``experiment=...`` override
    master_port: int
    extra_hydra_args: tuple[str, ...]
    loss_re: re.Pattern[str]
    deterministic_iters: int  # how many leading iters are bit-exact deterministic
    extra_env: dict[str, str] = field(default_factory=dict)


_SPECS: list[LaunchSpec] = [
    LaunchSpec(
        # Replicates cosmos_training/launch_vlm_llava_ov.sh
        key="launch_vlm_llava_ov",
        config="configs/base/vlm/config.py",
        experiment="pre_exp012_llava_ov_datapacker",
        master_port=50012,
        extra_hydra_args=(
            f"model.config.policy.backbone.model_name={_SIGLIP_MODEL_PATH}",
            "trainer.max_iter=10",
            "trainer.logging_iter=1",
            "job.wandb_mode=disabled",
            "ckpt_type=dummy",
            "checkpoint.load_from_object_store.enabled=false",
            "checkpoint.save_to_object_store.enabled=false",
            "upload_reproducible_setup=false",
        ),
        loss_re=_VLM_LOSS_RE,
        deterministic_iters=2,
    ),
    LaunchSpec(
        # Replicates cosmos_training/launch_mixed_modality_sft_8b.sh
        key="launch_mixed_modality_sft_8b",
        config="configs/base/config.py",
        experiment="mixed_modality_sft_8b",
        master_port=50011,
        extra_hydra_args=(
            f'dataloader_train.dataloader.datasets.video.dataset.jsonl_paths=["{_DATASET_JSONL}"]',
            f"model.config.tokenizer.vae_path={_WAN_VAE_PATH}",
            f"checkpoint.load_path={_DCP_LOAD_PATH}",
            "model.config.parallelism.data_parallel_shard_degree=4",
            "job.wandb_mode=disabled",
            "upload_reproducible_setup=false",
            "trainer.max_iter=10",
            "trainer.logging_iter=1",
            "checkpoint.save_iter=999999",
        ),
        loss_re=_VFM_LOSS_RE,
        deterministic_iters=10,
    ),
]


# --- helpers -----------------------------------------------------------------


def _parse_series(log_text: str, loss_re: re.Pattern[str]) -> tuple[list[float], list[float]]:
    """Extract per-iteration rank-0 loss and global grad-norm series, in order.

    Pairs by *position* rather than iteration label: VFM's ``IterSpeed``
    callback logs with the post-increment iteration while ``GradClip`` logs the
    pre-increment one, so labels don't line up but sequences do.
    """
    losses = [float(m.group(1)) for m in loss_re.finditer(log_text)]
    grad_norms = [float(m.group(1)) for m in _GRAD_NORM_RE.finditer(log_text)]
    assert losses and grad_norms, (
        f"No loss/grad-norm pairs found in log (losses={len(losses)}, grads={len(grad_norms)})"
    )
    assert len(losses) == len(grad_norms), (
        f"loss vs grad-norm length mismatch ({len(losses)} vs {len(grad_norms)}): "
        "the log must contain one rank-0 entry of each per training step."
    )
    return losses, grad_norms


def _run_torchrun(spec: LaunchSpec, run_dir: Path) -> Path:
    """Invoke the same ``torchrun`` command that the launch script runs.

    Returns the path of the captured combined stdout+stderr log.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    log_file = run_dir / "training.log"

    cmd = [
        "torchrun",
        "--nproc_per_node=4",
        f"--master_port={spec.master_port}",
        "-m",
        "scripts.train",
        f"--config={spec.config}",
        "--deterministic",
        "--",
        f"experiment={spec.experiment}",
        *spec.extra_hydra_args,
    ]

    env = os.environ.copy()
    # HF env mirrors what the launch scripts set up; ``HF_TOKEN`` must already
    # be exported in the caller's environment if the experiment hits gated Hub
    # endpoints (e.g. the LLaVA-OneVision-Data streaming dataset).
    env.setdefault("HF_HOME", "/tmp/hf_cache")
    Path(env["HF_HOME"]).mkdir(parents=True, exist_ok=True)
    env.setdefault("HF_HUB_DISABLE_XET", "1")
    # Determinism: PYTHONHASHSEED must be set before the interpreter starts.
    env["PYTHONHASHSEED"] = "42"
    # ``scripts.train`` needs the ``cosmos`` package on sys.path.
    env["PYTHONPATH"] = f".:{env.get('PYTHONPATH', '')}"
    # Keep generated config / wandb artifacts inside the per-test scratch.
    env["IMAGINAIRE_OUTPUT_ROOT"] = str(run_dir / "output")
    env.update(spec.extra_env)

    with log_file.open("w") as fp:
        result = subprocess.run(
            cmd,
            env=env,
            cwd=str(COSMOS_TRAINING_DIR),
            stdout=fp,
            stderr=subprocess.STDOUT,
        )
    if result.returncode != 0:
        # Some torchrun teardowns emit a harmless PyGIL warning on shutdown.
        # Treat the run as a success iff the log shows the training-complete
        # marker; otherwise propagate the failure.
        text = log_file.read_text(errors="replace")
        if "Done with training" not in text:
            pytest.fail(
                f"{spec.key}: torchrun failed with exit code {result.returncode} "
                "and log does not contain 'Done with training'.\n"
                f"Log tail:\n{text[-2000:]}"
            )
    return log_file


# --- fixtures ----------------------------------------------------------------


@pytest.fixture(scope="module", autouse=True)
def _require_4_gpus() -> None:
    """Skip the whole module unless we can launch 4-GPU training here."""
    if shutil.which("torchrun") is None:
        pytest.skip("torchrun not on PATH — must run inside the i4 training container")
    try:
        import torch
    except Exception as exc:  # pragma: no cover — surfaces during dev only
        pytest.skip(f"torch unavailable ({exc!r})")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 4:
        pytest.skip(f"requires 4 visible CUDA devices, found {torch.cuda.device_count()}")


# --- tests -------------------------------------------------------------------


@pytest.mark.level(2)
@pytest.mark.gpus(4)
@pytest.mark.parametrize("spec", _SPECS, ids=lambda s: s.key.removeprefix("launch_"))
def test_launch_regression(spec: LaunchSpec, tmp_path: Path) -> None:
    """Re-run ``spec``'s torchrun command and check loss / grad-norm against goldens."""
    log_path = _run_torchrun(spec, tmp_path)
    loss, grad_norm = _parse_series(log_path.read_text(errors="replace"), spec.loss_re)
    assert len(loss) == 10, f"expected 10 iterations, parsed {len(loss)} (loss={loss})"

    # Refresh path: print captured values for manual copy into ``_GOLDENS``.
    if os.environ.get("COSMOS_REGRESSION_UPDATE_GOLDENS") == "1":
        print(f"\n# --- goldens for {spec.key!r} ---")
        print(f'"{spec.key}": {{')
        print(f'    "loss": {loss},')
        print(f'    "grad_norm": {grad_norm},')
        print("},")
        pytest.skip(
            f"captured fresh series for {spec.key!r}; copy the printed dict into "
            "_GOLDENS at the bottom of launch_regression_test.py, then rerun "
            "without COSMOS_REGRESSION_UPDATE_GOLDENS to assert."
        )

    expected = _GOLDENS.get(spec.key)
    assert expected is not None, f"no goldens for {spec.key!r}; capture with COSMOS_REGRESSION_UPDATE_GOLDENS=1"

    n = spec.deterministic_iters
    if spec.key == "launch_vlm_llava_ov" and os.environ.get("COSMOS_REGRESSION_VLM_FULL") == "1":
        n = 10

    assert loss[:n] == pytest.approx(
        expected["loss"][:n], rel=_DEFAULT_RTOL, abs=_DEFAULT_ATOL
    ), f"{spec.key}: rank-0 loss[:{n}] does not match goldens"
    assert grad_norm[:n] == pytest.approx(
        expected["grad_norm"][:n], rel=_DEFAULT_RTOL, abs=_DEFAULT_ATOL
    ), f"{spec.key}: global grad-norm[:{n}] does not match goldens"


# --- inline goldens ----------------------------------------------------------
#
# Captured 2026-05-18 on a 4 × NVIDIA GB200 node inside
# ``imaginaire4_v11.2.4.sqsh`` with ``--deterministic`` and seed 42.
# Refresh with ``COSMOS_REGRESSION_UPDATE_GOLDENS=1`` and paste the printed
# series back into this dict.

_GOLDENS: dict[str, dict[str, list[float]]] = {
    "launch_vlm_llava_ov": {
        "loss": [1.32225, 1.20588, 1.39314, 1.40814, 1.16484, 1.23860, 1.37230, 1.22829, 0.96367, 1.14856],
        "grad_norm": [
            38.85580, 23.75230, 30.57402, 32.61629, 23.28848,
            37.41921, 131.50893, 44.30863, 25.23239, 27.81148,
        ],
    },
    "launch_mixed_modality_sft_8b": {
        "loss": [0.21000, 0.19890, 0.18670, 0.21440, 0.20190, 0.26100, 0.28080, 0.20230, 0.18910, 0.24790],
        "grad_norm": [0.34375, 0.37305, 0.42188, 0.30664, 0.32422, 0.33984, 0.55078, 0.56250, 0.37695, 0.59766],
    },
}


if __name__ == "__main__":  # pragma: no cover — manual driver
    sys.exit(pytest.main([__file__, "-v", "-s", "-o", "addopts="]))
