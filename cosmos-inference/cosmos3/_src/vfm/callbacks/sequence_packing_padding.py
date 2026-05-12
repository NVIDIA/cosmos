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

import torch
import wandb

import cosmos3._src.vfm.datasets.sequence_packing as sequence_packing
from cosmos3._src.imaginaire.callbacks.every_n import EveryN
from cosmos3._src.imaginaire.model import ImaginaireModel
from cosmos3._src.imaginaire.trainer import ImaginaireTrainer


class SequencePackingPadding(EveryN):
    """
    Callback that saves lengths to which und and gen sequences are padded. This information will be used
    to compute FLOPs done during training.

    Args:
        every_n (int): Frequency with which callback is run during training.
    """

    def __init__(self, every_n: int = 500):
        super().__init__(every_n=every_n, step_size=1, barrier_after_run=False, run_at_start=True)

    def every_n_impl(
        self,
        trainer: ImaginaireTrainer,
        model: ImaginaireModel,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int,
    ) -> None:
        if wandb.run:
            log_dict = {
                "SequencePackingPadding/max_causal_len_image_batch": sequence_packing.MAX_CAUSAL_LEN_IMAGE_BATCH,
                "SequencePackingPadding/max_full_len_image_batch": sequence_packing.MAX_FULL_LEN_IMAGE_BATCH,
                "SequencePackingPadding/max_causal_len_video_batch": sequence_packing.MAX_CAUSAL_LEN_VIDEO_BATCH,
                "SequencePackingPadding/max_full_len_video_batch": sequence_packing.MAX_FULL_LEN_VIDEO_BATCH,
            }
            modality = "video"
            if "is_image_batch" in output_batch:
                modality = "image" if output_batch["is_image_batch"] else "video"
            if "und_token_length" in output_batch:
                log_dict[f"SequencePackingPadding/und_token_length_{modality}"] = output_batch["und_token_length"]
            if "gen_token_length" in output_batch:
                log_dict[f"SequencePackingPadding/gen_token_length_{modality}"] = output_batch["gen_token_length"]
            if "action_token_length" in output_batch:
                log_dict[f"SequencePackingPadding/action_token_length"] = output_batch["action_token_length"]
            if "sound_token_length" in output_batch:
                log_dict[f"SequencePackingPadding/sound_token_length"] = output_batch["sound_token_length"]
            if "vision_token_length" in output_batch:
                log_dict[f"SequencePackingPadding/vision_token_length"] = output_batch["vision_token_length"]

            wandb.log(
                log_dict,
                step=iteration,
            )
