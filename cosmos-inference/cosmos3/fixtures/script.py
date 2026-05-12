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

"""Inference/training script test fixtures.

Used by 'tests/scripts_test.py'.
"""

import os
import re
import shutil
import subprocess
import warnings
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Callable

import numpy as np
import pydantic
import pytest

from cosmos3.common.args import MEDIA_EXTENSIONS, ResolvedFilePath
from cosmos3.common.init import get_free_port
from cosmos3.fixtures.args import Level, NumGpus
from cosmos3._src.imaginaire.utils.checkpoint_db import HF_VERSION
from cosmos3._src.imaginaire.utils.easy_io import easy_io

INPUT_DIR = Path("inputs").absolute()
OUTPUT_DIR = Path("outputs").absolute()


class ScriptConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid")

    script: ResolvedFilePath
    """Script path."""
    use_tmp_input_dir: bool = False
    """If set, use a per-test temp directory for INPUT_DIR."""
    levels: tuple[Level, ...] = (0,)
    """Test levels."""
    gpus: tuple[NumGpus, NumGpus, NumGpus] = (0, 1, 1)
    """Number of GPUs for each level."""
    marks: tuple[pytest.MarkDecorator | pytest.Mark, ...] = ()
    """Additional pytest marks."""

    golden_psnr: pydantic.PositiveFloat = 14.0
    """Golden comparison PSNR threshold in dB."""

    training: bool = False
    """Enable training features."""
    internal: bool = False
    """Enable internal (nvidia-only) features."""

    before_script: Callable[["ScriptRunner"], None] | None = None
    """Function to run before the script."""
    after_script: Callable[["ScriptRunner"], None] | None = None
    """Function to run after the script."""

    @property
    def name(self) -> str:
        return self.script.stem

    def get_marks(self, level: int) -> list[pytest.MarkDecorator | pytest.Mark]:
        marks = list(self.marks)
        if level not in self.levels:
            marks.append(pytest.mark.manual)
        marks.append(pytest.mark.gpus(self.gpus[level]))
        return marks


@dataclass(kw_only=True, frozen=True)
class ScriptRunner:
    request: pytest.FixtureRequest
    tmp_path_factory: pytest.TempPathFactory
    tmp_path: Path
    level: int = 0

    @cached_property
    def output_name(self) -> str:
        test_name = self.request.node.name
        if "[" in test_name and "]" in test_name:
            base_part, param_part = test_name.split("[", 1)
            param_part = param_part.rstrip("]").replace("/", "_").replace("-", "_")
            sanitized_name = f"{base_part}_{param_part}"
        else:
            sanitized_name = test_name.replace("/", "_").replace("-", "_")
        return sanitized_name

    @cached_property
    def input_dir(self) -> Path:
        return INPUT_DIR

    @cached_property
    def tmp_input_dir(self) -> Path:
        return self.tmp_path / "inputs"

    @cached_property
    def output_dir(self) -> Path:
        return OUTPUT_DIR / "pytest" / self.output_name

    @cached_property
    def golden_dir(self) -> Path:
        return INPUT_DIR / "outputs/pytest" / self.output_name

    def _get_env(
        self,
        cfg: ScriptConfig,
        *,
        torchrun_args: list[str] | None = None,
        inference_args: list[str] | None = None,
        train_args: list[str] | None = None,
        train_overrides: list[str] | None = None,
    ) -> dict[str, str]:
        if torchrun_args is None:
            torchrun_args = []
        if inference_args is None:
            inference_args = []
        if train_args is None:
            train_args = []
        if train_overrides is None:
            train_overrides = []

        num_gpus = os.environ["NUM_GPUS"]
        master_port = get_free_port()
        env = dict(os.environ)
        env |= {
            "COSMOS_INTERNAL": "1" if cfg.internal else "0",
            "COSMOS_TRAINING": "1" if cfg.training or cfg.internal else "0",
            "INPUT_DIR": f"{self.tmp_input_dir if cfg.use_tmp_input_dir else self.input_dir}",
            "OUTPUT_DIR": f"{self.output_dir}",
            "TMP_DIR": f"{self.tmp_path}/tmp",
            "MASTER_PORT": str(master_port),
            "HF_VERSION": HF_VERSION,
            "TORCHRUN_ARGS": " ".join(
                [
                    f"--nproc_per_node={num_gpus}",
                    f"--master_port={master_port}",
                    *torchrun_args,
                ]
            ),
            "INFERENCE_ARGS": " ".join(
                [
                    "--seed=0",
                    "--debug",
                    *inference_args,
                ]
            ),
            "TRAIN_ARGS": " ".join(
                [
                    *train_args,
                ]
            ),
            "TRAIN_OVERRIDES": " ".join(
                [
                    "job.wandb_mode=disabled",
                    f"model.config.parallelism.data_parallel_shard_degree={num_gpus}",
                    *train_overrides,
                ]
            ),
        }
        if not cfg.internal:
            # Disable S3 checkpoints
            env |= {
                "IMAGINAIRE_CACHE_DIR": "/invalid",
            }
        return env

    def get_env(self, cfg: ScriptConfig, level: int) -> dict[str, str]:
        match level:
            case 0:
                return self._get_env(cfg) | {"COSMOS_SMOKE": "1"}
            case 1:
                return self._get_env(
                    cfg,
                    inference_args=[
                        "--no-guardrails",
                    ],
                    train_overrides=[
                        "trainer.max_iter=5",
                    ],
                )
            case 2:
                return self._get_env(
                    cfg,
                    inference_args=[
                        "--guardrails",
                    ],
                    train_overrides=[
                        "trainer.max_iter=20",
                    ],
                )
            case _:
                assert False, "unreachable"

    def run(self, cfg: ScriptConfig, level: int):
        object.__setattr__(self, "level", level)  # frozen dataclass, but level is set per call
        shutil.rmtree(self.output_dir, ignore_errors=True)
        if cfg.before_script:
            cfg.before_script(self)
        subprocess.check_call(
            ["bash", "-euxo", "pipefail", str(cfg.script)],
            cwd=self.request.config.rootpath,
            env=self.get_env(cfg, level),
        )
        if cfg.after_script:
            cfg.after_script(self)

        if False:
            _check_golden_dir(self.output_dir, self.golden_dir, min_psnr=cfg.golden_psnr)


def script_test(configs: list[ScriptConfig]) -> Callable[[type], type]:
    names = set()
    for cfg in configs:
        if cfg.name in names:
            raise ValueError(f"Duplicate script name: {cfg.name}")
        names.add(cfg.name)

    def decorator(cls: type) -> type:
        @pytest.fixture
        def script_runner(
            self, request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory, tmp_path: Path
        ) -> ScriptRunner:
            return ScriptRunner(request=request, tmp_path_factory=tmp_path_factory, tmp_path=tmp_path)

        setattr(cls, "script_runner", script_runner)

        @pytest.mark.level(0)
        @pytest.mark.parametrize("cfg", [pytest.param(cfg, id=cfg.name, marks=cfg.get_marks(0)) for cfg in configs])
        def test_level_0(self, cfg: ScriptConfig, script_runner: ScriptRunner):
            script_runner.run(cfg, 0)

        setattr(cls, "test_level_0", test_level_0)

        @pytest.mark.level(1)
        @pytest.mark.parametrize("cfg", [pytest.param(cfg, id=cfg.name, marks=cfg.get_marks(1)) for cfg in configs])
        def test_level_1(self, cfg: ScriptConfig, script_runner: ScriptRunner):
            script_runner.run(cfg, 1)

        setattr(cls, "test_level_1", test_level_1)

        @pytest.mark.level(2)
        @pytest.mark.parametrize("cfg", [pytest.param(cfg, id=cfg.name, marks=cfg.get_marks(2)) for cfg in configs])
        def test_level_2(self, cfg: ScriptConfig, script_runner: ScriptRunner):
            script_runner.run(cfg, 2)

        setattr(cls, "test_level_2", test_level_2)
        return cls

    return decorator


def _extract_bash_commands(md_file: Path) -> list[str]:
    content = md_file.read_text()
    pattern = r"```(bash|shell)([^\n]*)\n(.*?)```"
    matches = re.findall(pattern, content, re.DOTALL)
    scripts = []
    for lang, attrs, block_content in matches:
        if "exclude=true" in attrs.lower():
            continue

        lines = []
        for line in block_content.strip().split("\n"):
            if line.strip() and not line.strip().startswith("#"):
                line = line.split("#")[0].rstrip()
                # Replace --nproc_per_node with dynamic NUM_GPUS value
                line = re.sub(r"--nproc_per_node=\d+", "--nproc_per_node=$NUM_GPUS", line)
                line = re.sub(r"--master_port=\d+", "--master_port=$MASTER_PORT", line)
                if line:
                    lines.append(line)

        if lines:
            script = "\n".join(lines)
            scripts.append(script)

    return scripts


def _array_to_float(array: np.ndarray) -> np.ndarray:
    if np.issubdtype(array.dtype, np.floating):
        assert np.min(array) >= 0.0 and np.max(array) <= 1.0
        return array
    if array.dtype == np.uint8:
        return array / 255.0
    raise NotImplementedError(f"Unsupported dtype: {array.dtype}")


def _compute_psnr(array1: np.ndarray, array2: np.ndarray) -> float:
    """Compare PSNR between two arrays."""
    array1 = _array_to_float(array1)
    array2 = _array_to_float(array2)
    overall_mse = ((array1 - array2) ** 2).mean()
    return 10 * np.log10(1.0 / overall_mse) if overall_mse > 0 else float("inf")


def _check_golden_file(output_path: Path, golden_path: Path, /, min_psnr: float) -> None:
    output_array, _output_meta = easy_io.load(output_path)
    assert isinstance(output_array, np.ndarray)
    golden_array, _golden_meta = easy_io.load(golden_path)
    assert isinstance(golden_array, np.ndarray)
    psnr = _compute_psnr(output_array, golden_array)
    if psnr < min_psnr:
        warnings.warn(
            f"FAIL: Golden PSNR {psnr:.2f} dB is less than minimum {min_psnr:.2f} dB for file '{output_path}'"
        )
    else:
        print(f"PASS: Golden PSNR {psnr:.2f} dB is greater than minimum {min_psnr:.2f} dB for file '{output_path}'")


def _check_golden_dir(output_dir: Path, golden_dir: Path, /, min_psnr: float) -> None:
    if not golden_dir.exists():
        warnings.warn(f"Golden directory '{golden_dir}' does not exist")
        return
    for dirpath, _dirnames, filenames in os.walk(golden_dir):
        for filename in filenames:
            golden_path = Path(dirpath) / filename
            output_path = output_dir / golden_path.relative_to(golden_dir)
            if output_path.suffix not in MEDIA_EXTENSIONS:
                continue
            if not output_path.exists():
                warnings.warn(f"File '{output_path}' missing in output directory")
                continue
            _check_golden_file(output_path, golden_path, min_psnr=min_psnr)
