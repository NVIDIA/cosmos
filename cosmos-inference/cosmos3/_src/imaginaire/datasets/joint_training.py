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

"""Utility funcitons to use joint dataloader for training."""

from typing import Dict, Iterator  # For multiview training

import torch

import cosmos3._src.imaginaire.config
import cosmos3._src.imaginaire.datasets.webdataset.dataloader
from cosmos3._src.imaginaire.config import Config
from cosmos3._src.imaginaire.lazy_config import instantiate
from cosmos3._src.imaginaire.utils import log


def create_dataloader_dict(
    config: Config, dataloader_train: cosmos3._src.imaginaire.datasets.webdataset.dataloader.DataLoader
) -> Dict:
    """Create the dataloader dictionary.

    Example config:

    ```
    config:
      joint_train:
          data_sample_prob:
              dataloader_train: 0.5 # sampling probability for default dataloader
              dataloader_1: 0.2 # sampling probability for dataloader_1
              dataloader_2: 0.3 # sampling probability for dataloader_2
          dataloader_1:
              ... # dataloader config for dataloader_1
          dataloader_2:
              ... # dataloader config for dataloader_2
    ```

    Args:
        config (Config): The config object for the Imaginaire codebase.

    Returns:
        dict: The dataloader dictionary.
    """
    dataloader_list = list(config.joint_train.data_sample_prob.keys())

    dataloader_dict = {}
    for dataloader_name in dataloader_list:
        if dataloader_name == "dataloader_train":
            continue
        log.info(
            f"Creating dataloader: {dataloader_name}, sampling probability: {config.joint_train.data_sample_prob[dataloader_name]}"
        )
        dataloader_dict[dataloader_name] = iter(instantiate(getattr(config.joint_train, dataloader_name)))
    dataloader_dict["dataloader_train"] = iter(dataloader_train)
    return dataloader_dict


def data_batch_iterator(dataloader_dict: Dict, data_sample_prob: Dict) -> Iterator[Dict]:
    """Sample data batches continuously from the dataloader dictionary based on sampling probabilities."""
    dataloader_list = list(data_sample_prob.keys())

    while True:
        selected_dataloader_id = torch.multinomial(
            torch.tensor([data_sample_prob[dataloader_name] for dataloader_name in dataloader_list]), 1
        ).item()
        selected_dataloader_name = dataloader_list[selected_dataloader_id]
        selected_dataloader = dataloader_dict[selected_dataloader_name]

        try:
            data_batch = next(selected_dataloader)
        except StopIteration:
            # Reinitialize the iterator for the selected dataloader once it is exhausted
            dataloader_dict[dataloader_list[selected_dataloader_id]] = iter(selected_dataloader)
            data_batch = next(dataloader_dict[dataloader_list[selected_dataloader_id]])
        data_batch["dataloader_name"] = selected_dataloader_name
        yield data_batch


def init_and_wrap_data_loaders(config: Config, dataloader_train: torch.utils.data.DataLoader) -> Dict:
    """Wrap the dataloaders for multiview training.

    Args:
        config (Config): The config object for the Imaginaire codebase.
        dataloader_train (torch.utils.data.DataLoader): The training data loader.

    Returns:
        dict: The dataloader dictionary.
    """
    # Create the dataloader dictionary with multiple dataloaders
    dataloader_dict = create_dataloader_dict(config, dataloader_train)

    # Create the data batch iterator sample from the dataloader dictionary based on sampling probabilities
    dataloader_train = data_batch_iterator(dataloader_dict, config.joint_train.data_sample_prob)
    return dataloader_train
