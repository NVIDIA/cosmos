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

import torch
import torch.distributed as dist
import torch.utils.data
import wandb

from cosmos.model._base import ImaginaireModel
from cosmos.utils import distributed
from cosmos.utils.callback import Callback


class DataStatsCallback(Callback):
    def __init__(
        self,
        logging_iter_multipler: int = 1,
        save_s3: bool = False,
    ) -> None:
        super().__init__()
        self.logging_iter_multipler = logging_iter_multipler
        assert self.logging_iter_multipler > 0, "logging_iter_multipler should be greater than 0"
        self.save_s3 = save_s3
        self.wandb_extra_tag = f"@{logging_iter_multipler}" if logging_iter_multipler > 1 else ""
        self.name = "data_stats" + self.wandb_extra_tag
        self.data_freq_current = {}
        self.data_freq_acc = {}
        self.avg_num_assistant_tokens = []
        self.avg_num_real_tokens = []
        self.max_num_real_tokens = []
        self.min_num_real_tokens = []

        # Per-dataset token length tracking
        self.dataset_token_lengths = {}  # dataset_name -> list of avg_num_real_tokens
        self.dataset_assistant_tokens = {}  # dataset_name -> list of avg_num_assistant_tokens
        self.num_log_current = 0
        self.total_count_acc = {}

    def on_training_step_end(
        self,
        model: ImaginaireModel,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> None:
        self.num_log_current += 1
        dataset_name = data_batch.get("dataset_name", "default")

        # Handle case where dataset_name gets batched into a list
        if isinstance(dataset_name, list):

            assert len(dataset_name) == 1, "dataset_name should be a list of 1"
            dataset_name = dataset_name[0]

        if dataset_name in ["default"] and "__url__" in data_batch:
            # try to get the name from url
            dataset_name = "/".join(data_batch["__url__"][0].split("/")[:-1])

        if dataset_name not in self.data_freq_current:
            self.data_freq_current[dataset_name] = torch.tensor(0, device="cuda")  # []
        self.data_freq_current[dataset_name] += 1

        if dataset_name not in self.data_freq_acc:
            self.data_freq_acc[dataset_name] = torch.tensor(0, device="cuda")  # []
        self.data_freq_acc[dataset_name] += 1

        if "avg_num_assistant_tokens" in output_batch:
            self.avg_num_assistant_tokens.append(output_batch["avg_num_assistant_tokens"])
            # Track per-dataset assistant tokens
            if dataset_name not in self.dataset_assistant_tokens:
                self.dataset_assistant_tokens[dataset_name] = []
            self.dataset_assistant_tokens[dataset_name].append(output_batch["avg_num_assistant_tokens"])
        if "avg_num_real_tokens" in output_batch:
            self.avg_num_real_tokens.append(output_batch["avg_num_real_tokens"])
            # Track per-dataset token lengths
            if dataset_name not in self.dataset_token_lengths:
                self.dataset_token_lengths[dataset_name] = []
            self.dataset_token_lengths[dataset_name].append(output_batch["avg_num_real_tokens"])
        if "max_num_real_tokens" in output_batch:
            self.max_num_real_tokens.append(output_batch["max_num_real_tokens"])
        if "min_num_real_tokens" in output_batch:
            self.min_num_real_tokens.append(output_batch["min_num_real_tokens"])

        if iteration % (self.config.trainer.logging_iter * self.logging_iter_multipler) == 0:
            # Step 1: Gather all dataset names across ranks
            local_dataset_names = list(self.data_freq_current.keys())
            all_dataset_names = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(all_dataset_names, local_dataset_names)

            # Step 2: Create the union of all dataset names
            union_dataset_names = set()
            for names in all_dataset_names:
                union_dataset_names.update(names)
            union_dataset_names = sorted(list(union_dataset_names))

            # Step 3: For any missing dataset name, add dummy _LossRecord with NaN loss
            for dataset_name in union_dataset_names:
                if dataset_name not in self.data_freq_acc:
                    self.data_freq_acc[dataset_name] = torch.tensor(0, device="cuda")  # []
                if dataset_name not in self.data_freq_current:
                    self.data_freq_current[dataset_name] = torch.tensor(0, device="cuda")  # []

            # Step 4: Calculate the total count of each dataset across all ranks
            total_count_current = {}
            for dataset_name in union_dataset_names:
                acc_tensor = self.data_freq_acc[dataset_name].clone()
                current_tensor = self.data_freq_current[dataset_name].clone()

                dist.all_reduce(acc_tensor, op=dist.ReduceOp.SUM)
                dist.all_reduce(current_tensor, op=dist.ReduceOp.SUM)

                self.total_count_acc[dataset_name] = acc_tensor.item()
                total_count_current[dataset_name] = current_tensor.item()

            if distributed.is_rank0() and wandb.run is not None:
                info = {}
                if len(self.avg_num_assistant_tokens) > 0:
                    info["data_stats_tokens/avg_num_assistant_tokens"] = sum(self.avg_num_assistant_tokens) / len(
                        self.avg_num_assistant_tokens
                    )
                    self.avg_num_assistant_tokens = []
                if len(self.avg_num_real_tokens) > 0:
                    info["data_stats_tokens/avg_num_real_tokens"] = sum(self.avg_num_real_tokens) / len(
                        self.avg_num_real_tokens
                    )
                    self.avg_num_real_tokens = []
                if len(self.max_num_real_tokens) > 0:
                    info["data_stats_tokens/max_num_real_tokens"] = max(self.max_num_real_tokens)
                    self.max_num_real_tokens = []

                if len(self.min_num_real_tokens) > 0:
                    info["data_stats_tokens/min_num_real_tokens"] = min(self.min_num_real_tokens)
                    self.min_num_real_tokens = []

                # Log per-dataset average token lengths
                for dataset_name in union_dataset_names:
                    if dataset_name in self.dataset_token_lengths and len(self.dataset_token_lengths[dataset_name]) > 0:
                        avg_token_length = sum(self.dataset_token_lengths[dataset_name]) / len(
                            self.dataset_token_lengths[dataset_name]
                        )
                        info[f"data_stats_avg_tokens_per_dataset/{dataset_name}"] = avg_token_length
                    if (
                        dataset_name in self.dataset_assistant_tokens
                        and len(self.dataset_assistant_tokens[dataset_name]) > 0
                    ):
                        avg_assistant_tokens = sum(self.dataset_assistant_tokens[dataset_name]) / len(
                            self.dataset_assistant_tokens[dataset_name]
                        )
                        info[f"data_stats_avg_assistant_tokens_per_dataset/{dataset_name}"] = avg_assistant_tokens

                # Reset per-dataset token lengths after logging
                self.dataset_token_lengths = {}
                self.dataset_assistant_tokens = {}

                # Log the valid count per dataset
                for dataset_name in union_dataset_names:
                    info[f"data_stats_count_acc/{dataset_name}"] = self.total_count_acc[dataset_name]
                    info[f"data_stats_count_current/{dataset_name}"] = total_count_current[dataset_name]
                self.num_log_current = 0

                wandb.log(info, step=iteration)

                # Create a table of the data stats, columns: Dataset, Accumulated frequency, Current frequency, Accumulated Count, Current Count
                table_html = "<table><tr><th>Dataset</th><th>Accumulated frequency</th><th>Current frequency</th><th>Accumulated Count</th><th>Current Count</th></tr>"
                total_count_acc_sum = sum(self.total_count_acc.values())
                total_count_current_sum = sum(total_count_current.values())
                # Sort union_dataset_names by total_count_acc, from highest to lowest
                union_dataset_names = sorted(union_dataset_names, key=lambda x: self.total_count_acc[x], reverse=True)
                acc_freq_list = []
                current_freq_list = []
                for name in union_dataset_names:
                    acc_freq = self.total_count_acc[name] / total_count_acc_sum
                    acc_freq_list.append(acc_freq)
                    current_freq = total_count_current[name] / total_count_current_sum
                    current_freq_list.append(current_freq)
                    table_html += f"<tr><td>{name}</td><td>{acc_freq}</td><td>{current_freq}</td><td>{self.total_count_acc[name]}</td><td>{total_count_current[name]}</td></tr>"
                # Sum over all dataset for each column
                acc_freq_sum = sum(acc_freq_list)
                current_freq_sum = sum(current_freq_list)
                table_html += f"<tr><td>Total ({len(union_dataset_names)})</td><td>{acc_freq_sum}</td><td>{current_freq_sum}</td><td>{total_count_acc_sum}</td><td>{total_count_current_sum}</td></tr>"

                table_html += "</table>"
                wandb.log({"table_data_stats/html": wandb.Html(table_html)}, step=iteration)
            # Reset self.data_freq_current
            self.data_freq_current = {k: v * 0 for k, v in self.data_freq_current.items()}
        if (
            distributed.is_rank0()
            and wandb.run is not None
            and iteration in [100, 1000, 2000, 5000, 15000, 30000]
            and len(self.total_count_acc)
        ):
            # log a table of the total_count_acc
            # Sort self.total_count_acc by value, from highest to lowest
            sorted_total_count_acc = sorted(self.total_count_acc.items(), key=lambda x: x[1], reverse=True)
            table = wandb.Table(data=[[k, v] for k, v in sorted_total_count_acc], columns=["Dataset", "Count"])

            wandb.log(
                {
                    f"data_counts_bar_{iteration:09d}": wandb.plot.bar(
                        table, "Dataset", "Count", title=f"Count per Dataset iter {iteration:09d}"
                    )
                },
                step=iteration,
            )
