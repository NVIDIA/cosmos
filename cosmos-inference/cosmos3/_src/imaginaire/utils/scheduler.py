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

import math
from typing import List

import torch


class WarmupLambdaLR(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, warmup, last_epoch=-1, verbose=False):
        # Define the lambda function based on the warmup period
        self.warmup = warmup

        def lr_lambda(epoch):
            # Increase lr linearly for the first 'warmup' epochs
            if epoch < warmup:
                return float(epoch + 1) / warmup
            # After 'warmup' epochs, keep lr constant
            return 1.0

        # Initialize the parent class with the generated lr_lambda
        super(WarmupLambdaLR, self).__init__(optimizer, lr_lambda, last_epoch)


# cosine lr decay scheduler with warmup from https://github.com/karpathy/nanoGPT/blob/master/train.py#L228
class WarmupCosineLR(torch.optim.lr_scheduler.LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_iters: int,
        lr_decay_iters: int,
        min_lr: float,
        last_epoch: int = -1,
    ):
        self.warmup_iters = warmup_iters
        self.lr_decay_iters = lr_decay_iters
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        # 1) linear warmup for warmup_iters steps
        if self.last_epoch < self.warmup_iters:
            return [base_lr * self.last_epoch / self.warmup_iters for base_lr in self.base_lrs]
        # 2) if it > lr_decay_iters, return min learning rate
        if self.last_epoch > self.lr_decay_iters:
            return [self.min_lr for _ in self.base_lrs]
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (self.last_epoch - self.warmup_iters) / (self.lr_decay_iters - self.warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
        return [self.min_lr + coeff * (base_lr - self.min_lr) for base_lr in self.base_lrs]
