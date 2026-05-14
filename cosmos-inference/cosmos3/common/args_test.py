# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import json
import os
from pathlib import Path

from cosmos3.args import DEFAULT_CHECKPOINT, DEFAULT_CHECKPOINT_NAME
from cosmos3.common.args import CheckpointConfig, CheckpointOverrides, download_file

CHECKPOINTS: dict[str, CheckpointConfig] = {
    DEFAULT_CHECKPOINT_NAME: DEFAULT_CHECKPOINT,
}


def test_download_file(tmp_path: Path):
    download_url_1 = (
        "https://github.com/nvidia-cosmos/cosmos-dependencies/raw/refs/heads/assets/cosmos3/inputs/vision/robot_153.jpg"
    )
    file_size_1 = 279410

    download_url_2 = "https://github.com/nvidia-cosmos/cosmos-dependencies/raw/refs/heads/assets/cosmos3/inputs/vision/bus_terminal.jpg"
    file_size_2 = 1283715

    # Download file
    download_path = Path(
        download_file(
            download_url_1,
            tmp_path,
            "robot_welding",
        )
    )
    assert download_path.stat().st_size == file_size_1
    meta_path = Path(f"{download_path}.meta")
    assert json.loads(meta_path.read_text()) == {
        "url": download_url_1,
    }
    cache_path = download_path.resolve()

    # Same file should be noop
    download_path = Path(download_file(str(download_path), tmp_path, "robot_welding"))
    assert download_path.resolve() == cache_path

    # Copy file
    copy_path = Path(download_file(str(download_path), tmp_path, "robot_welding_copy"))
    assert copy_path.stat().st_size == file_size_1
    assert copy_path.resolve() == cache_path

    # Re-download should be noop
    copy_path = Path(download_file(str(download_path), tmp_path, "robot_welding_copy"))
    assert copy_path.resolve() == cache_path

    # Force re-download
    os.remove(meta_path)
    download_path = Path(download_file(download_url_1, tmp_path, "robot_welding"))
    assert download_path.resolve() != cache_path
    assert download_path.stat().st_size == file_size_1

    # Different file should overwrite
    download_path = Path(
        download_file(
            download_url_2,
            tmp_path,
            "robot_welding",
        )
    )
    assert download_path.stat().st_size == file_size_2
    assert json.loads(Path(f"{download_path}.meta").read_text()) == {
        "url": download_url_2,
    }


def test_parse_checkpoint_path(tmp_path: Path):
    # Named checkpoint
    args = CheckpointOverrides(
        checkpoint_path=DEFAULT_CHECKPOINT_NAME,
    ).build_checkpoint(checkpoints=CHECKPOINTS)
    assert args.checkpoint_type == "hf"
    assert args.experiment == ""

    # Local HF checkpoint
    hf_path = tmp_path / "hf"
    hf_path.mkdir(parents=True, exist_ok=True)
    (hf_path / "config.json").touch()
    (hf_path / "model.safetensors").touch()
    args = CheckpointOverrides(
        checkpoint_path=str(hf_path),
    ).build_checkpoint(checkpoints=CHECKPOINTS)
    assert args.checkpoint_type == "hf"
    assert args.experiment == ""

    # Local DCP checkpoint
    dcp_path = (
        tmp_path
        / "cosmos3_vfm/t2i_mot_0p6b_qwen3_vl_ablations/t2i_mot_exp000_015_qwen3_0p6b_256res_frozen_llm_lr_4e4_large_seq_no_llm_qknorm_gcp_bs8k_baseline_long_run_1e4/checkpoints/iter_000700000/model"
    )
    dcp_path.mkdir(parents=True, exist_ok=True)
    (dcp_path / ".metadata").touch()
    (dcp_path / "__0_0.distcp").touch()
    args = CheckpointOverrides(
        checkpoint_path=str(dcp_path),
    ).build_checkpoint(checkpoints=CHECKPOINTS)
    assert args.checkpoint_type == "dcp"
    assert (
        args.experiment
        == "t2i_mot_exp000_015_qwen3_0p6b_256res_frozen_llm_lr_4e4_large_seq_no_llm_qknorm_gcp_bs8k_baseline_long_run_1e4"
    )

    # S3 DCP checkpoint
    args = CheckpointOverrides(
        checkpoint_path=f"s3://bucket/cosmos3_vfm/t2i_mot_0p6b_qwen3_vl_ablations/t2i_mot_exp000_015_qwen3_0p6b_256res_frozen_llm_lr_4e4_large_seq_no_llm_qknorm_gcp_bs8k_baseline_long_run_1e4/checkpoints/iter_000700000/model",
    ).build_checkpoint(checkpoints=CHECKPOINTS)
    assert args.checkpoint_type == "dcp"
    assert (
        args.experiment
        == "t2i_mot_exp000_015_qwen3_0p6b_256res_frozen_llm_lr_4e4_large_seq_no_llm_qknorm_gcp_bs8k_baseline_long_run_1e4"
    )
    assert args.config_file == "cosmos3/_src/vfm/configs/base/config.py"
