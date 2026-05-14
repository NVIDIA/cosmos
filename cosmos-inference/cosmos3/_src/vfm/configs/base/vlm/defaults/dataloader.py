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

from torch.utils.data import DataLoader

from cosmos3._src.imaginaire.lazy_config import LazyCall as L
from cosmos3._src.imaginaire.utils.config_helper import ConfigStore
from cosmos3._src.vfm.processors import build_processor
from projects.cosmos3.vlm.datasets.collate_fn import custom_collate
from projects.cosmos3.vlm.datasets.debug_data_qwen import DebugQwenDataset
from projects.cosmos3.vlm.datasets.dummy_data_qwen import DummyQwenDataset


# Debug dataset
def create_debug_dataloader_config_qwen(
    num_images, loss_on_completion_only: bool = True, use_dummy_image: bool = False
):
    return L(DataLoader)(
        dataset=L(DebugQwenDataset)(
            tokenizer=L(build_processor)(
                tokenizer_type="${model.config.policy.model_name_or_path}",
                credentials="${checkpoint.load_from_object_store.credentials}",
                bucket="${checkpoint.load_from_object_store.bucket}",
            ),
            num_images=num_images,
            seq_len="${model.config.policy.model_max_length}",
            image_token_len="${model.config.policy.qwen_max_video_token_length}",
            # use_dummy_image=use_dummy_image,
        ),
        num_workers=8,
        prefetch_factor=4,
        batch_size=1,
        sampler=None,
        persistent_workers=False,
        pin_memory=True,
        collate_fn=custom_collate,
    )


def create_dummy_dataloader_config_qwen():
    return L(DataLoader)(
        dataset=L(DummyQwenDataset)(
            tokenizer=L(build_processor)(
                tokenizer_type="${model.config.policy.model_name_or_path}",
                credentials="${checkpoint.load_from_object_store.credentials}",
                bucket="${checkpoint.load_from_object_store.bucket}",
            ),
            num_visual_tokens="${model.config.policy.qwen_max_video_token_length}",
            total_tokens="${model.config.policy.model_max_length}",
            batch_size="${dataloader_train.batch_size}",
        ),
        num_workers=8,
        prefetch_factor=4,
        batch_size=1,
        sampler=None,
        persistent_workers=False,
        pin_memory=True,
        collate_fn=custom_collate,
    )


def register_data_debug():
    cs = ConfigStore.instance()
    for split in ["train", "val"]:
        cs.store(
            group=f"data_{split}",
            package=f"dataloader_{split}",
            name="debug_image_data_qwen",  # This data is from pixtral model output, expected to have low loss ~1.4
            node=create_debug_dataloader_config_qwen(1),
        )
        cs.store(
            group=f"data_{split}",
            package=f"dataloader_{split}",
            name="dummy_image_data_qwen",
            node=create_dummy_dataloader_config_qwen(),
        )


def register_data():
    register_data_debug()
