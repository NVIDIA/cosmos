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

from __future__ import annotations

from typing import Optional, Union

import torch
import torch.nn as nn

from cosmos3._src.imaginaire.functional.batch_ops import batch_mul
from cosmos3._src.imaginaire.utils.count_params import count_params


class AbstractEmbModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self._is_trainable = None
        self._dropout_rate = None
        self._input_key = None
        self._return_dict = False

    @property
    def is_trainable(self) -> bool:
        return self._is_trainable

    @property
    def dropout_rate(self) -> Union[float, torch.Tensor]:
        return self._dropout_rate

    @property
    def input_key(self) -> str:
        return self._input_key

    @property
    def is_return_dict(self) -> bool:
        return self._return_dict

    @is_trainable.setter
    def is_trainable(self, value: bool) -> None:
        self._is_trainable = value

    @dropout_rate.setter
    def dropout_rate(self, value: Union[float, torch.Tensor]) -> None:
        self._dropout_rate = value

    @input_key.setter
    def input_key(self, value: str) -> None:
        self._input_key = value

    @is_return_dict.setter
    def is_return_dict(self, value: bool) -> None:
        self._return_dict = value

    @is_trainable.deleter
    def is_trainable(self) -> None:
        del self._is_trainable

    @dropout_rate.deleter
    def dropout_rate(self) -> None:
        del self._dropout_rate

    @input_key.deleter
    def input_key(self) -> None:
        del self._input_key

    @is_return_dict.deleter
    def is_return_dict(self) -> None:
        del self._return_dict

    def random_dropout_input(
        self, in_tensor: torch.Tensor, dropout_rate: Optional[float] = None, key: Optional[str] = None
    ) -> torch.Tensor:
        del key
        dropout_rate = dropout_rate if dropout_rate is not None else self.dropout_rate
        return batch_mul(
            torch.bernoulli((1.0 - dropout_rate) * torch.ones(in_tensor.shape[0])).type_as(in_tensor),
            in_tensor,
        )

    def details(self) -> str:
        return ""

    def summary(self) -> str:
        input_key = self.input_key if self.input_key is not None else getattr(self, "input_keys", None)
        return (
            f"{self.__class__.__name__} \n\tinput key: {input_key}"
            f"\n\tParam count: {count_params(self, False)} \n\tTrainable: {self.is_trainable}"
            f"\n\tDropout rate: {self.dropout_rate}"
            f"\n\t{self.details()}"
        )
