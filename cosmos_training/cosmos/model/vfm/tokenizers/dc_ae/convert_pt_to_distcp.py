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

import os
from dataclasses import dataclass

import torch
import torch.distributed.checkpoint
from omegaconf import MISSING, OmegaConf


@dataclass
class ConvertPTToDistCPConfig:
    input_path: str = MISSING
    output_dir: str = MISSING


def main():
    torch.distributed.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    cfg = OmegaConf.to_object(OmegaConf.merge(OmegaConf.structured(ConvertPTToDistCPConfig), OmegaConf.from_cli()))
    checkpoint = torch.load(cfg.input_path, map_location="cpu")
    torch.distributed.checkpoint.save(checkpoint, checkpoint_id=cfg.output_dir)


if __name__ == "__main__":
    main()


"""
python cosmos/model/vfm/tokenizers/dc_ae/convert_pt_to_distcp.py \
    input_path=checkpoints/cosmos_8b_wan22_vae/iter_000020000/model_ema_fp32.pt \
    output_dir=checkpoints/cosmos_8b_wan22_vae/iter_000020000/model_dcp_from_torch_save
"""
