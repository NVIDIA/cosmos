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

VERSION = "1.0.2"
MINIMAL_SCHEMA_CALLBACK_ARGS = {
    "initialize": {
        "required": {
            "enable_for_current_rank",
            "one_logger_project",
            "one_logger_run_name",
            "one_logger_async",
            "log_every_n_train_iterations",
            "app_tag_run_name",
            "app_tag_run_version",
            "world_size",
            "global_batch_size",
            "batch_size",
            "app_tag",
            "train_iterations_target",
            "train_samples_target",
            "is_baseline_run",
            "is_train_iterations_enabled",
            "is_validation_iterations_enabled",
            "is_test_iterations_enabled",
            "is_save_checkpoint_enabled",
            "is_log_throughput_enabled",
        },
        "optional": {
            "micro_batch_size",
            "summary_data_schema_version",
            "app_run_type",
            "app_start_time",
            "app_metrics_feature_tags",
            "seq_length",
            "metadata",
            "barrier",
        },
    },
    "on_train_start": {
        "required": {"train_iterations_start"},
        "optional": {"train_samples_start"},
    },
    "on_app_start": {"optional": {"app_start_time"}},
    "on_app_end": {"optional": {"app_finish_time"}},
    "on_model_init_start": {"optional": {"app_model_init_start_time"}},
    "on_model_init_end": {"optional": {"app_model_init_finish_time"}},
    "on_dataloader_init_start": {"optional": {"app_build_dataiters_start_time"}},
    "on_dataloader_init_end": {"optional": {"app_build_dataiters_finish_time"}},
    "on_load_checkpoint_start": {"optional": {"load_checkpoint_start_time"}},
    "on_load_checkpoint_end": {"optional": {"load_checkpoint_finish_time"}},
    "on_optimizer_init_start": {"optional": {"app_build_optimizer_start_time"}},
    "on_optimizer_init_end": {"optional": {"app_build_optimizer_finish_time"}},
}

THROUGHPUT_CALLBACK_ARGS = {
    "initialize": {
        "optional": {"flops_per_sample"},
    },
}

CHECKPOINT_CALLBACK_ARGS = {
    "initialize": {
        "required": {"save_checkpoint_strategy"},
    },
    "on_save_checkpoint_start": {
        "required": {"global_step"},
    },
    "on_save_checkpoint_end": {
        "required": {"global_step"},
    },
    "on_save_checkpoint_success": {
        "required": {"global_step"},
    },
}
