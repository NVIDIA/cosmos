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

"""Evaluation entrypoint for (Action) models with local dataset."""

from cosmos3.common.init import init_script, is_rank0

init_script()

import json

import pydantic
import torch
import tyro

from cosmos3.args import OmniSetupOverrides
from cosmos3.common.args import SampleOutputs, SetupOverrides, tyro_cli
from cosmos3.common.init import init_output_dir
from cosmos3.dataset import DatasetArgs, create_dataset
from cosmos3.scripts.eval_utils import aggregate_metrics, compute_sample_metrics, extract_gt_action, extract_gt_video
from cosmos3._src.imaginaire.utils import log


class EvalArgs(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid", use_attribute_docstrings=True)

    setup: SetupOverrides = OmniSetupOverrides.model_construct()
    """Model and parallelism configuration."""
    dataset: DatasetArgs = DatasetArgs()
    """Dataset loading configuration."""
    compute_metrics: bool = True
    """Compute per-sample metrics and write metrics.json sidecars + metrics_aggregate.json."""


def eval_dataset(args: EvalArgs) -> list[SampleOutputs]:
    """Run dataset inference: load dataset in memory, run inference, save outputs."""
    if args.setup.output_dir is None:
        raise ValueError("'output_dir' is required")

    setup = args.setup.build_setup()
    init_output_dir(setup.output_dir)
    log.debug(f"{args.__class__.__name__}({args})")

    samples = create_dataset(
        args.dataset,
        config_args=setup,
    )
    log.info(f"Loaded {len(samples)} samples in memory")

    pipe = setup.get_inference_cls().create(setup)

    output_dir = setup.output_dir
    all_outputs: list[SampleOutputs] = []
    for i, (sample_args, data_batch) in enumerate(samples):
        assert sample_args.name
        sample_args.output_dir = output_dir / sample_args.name
        sample_args = sample_args.build_sample(model_config=pipe.model_config)
        log.info(f"[{i + 1}/{len(samples)}] Processing: {sample_args.name}")

        gt_video: torch.Tensor | None = None
        gt_action: torch.Tensor | None = None
        if args.compute_metrics:
            gt_video = extract_gt_video(data_batch)
            gt_action = extract_gt_action(data_batch)

        batch_outputs = pipe.generate_batch([sample_args], data_batch)
        all_outputs.extend(batch_outputs)

        if args.compute_metrics and batch_outputs:
            sample_output = batch_outputs[0]
            if sample_output.status == "success":
                metrics = compute_sample_metrics(
                    sample_args.name,
                    gt_video,
                    gt_action,
                    sample_output,
                    sample_args.output_dir,
                    sample_args.vision_extension,
                )
                (sample_args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True))
                log.info(f"Metrics for {sample_args.name}: {metrics}")

    if setup.benchmark and is_rank0():
        benchmark_file = output_dir / "benchmark.json"
        benchmark_file.write_text(json.dumps(pipe.get_timer_results(), indent=2, sort_keys=True))
        log.success(f"Saved benchmark to '{benchmark_file}'")

    if args.compute_metrics and is_rank0():
        aggregate = aggregate_metrics(output_dir)
        aggregate_file = output_dir / "metrics_aggregate.json"
        aggregate_file.write_text(json.dumps(aggregate, indent=2, sort_keys=True))
        log.success(f"Saved aggregated metrics to '{aggregate_file}'")

    return all_outputs


def main() -> None:
    args = tyro_cli(
        EvalArgs,
        description=__doc__,
        config=(
            tyro.conf.OmitArgPrefixes,
            tyro.conf.CascadeSubcommandArgs,
            tyro.conf.OmitSubcommandPrefixes,
        ),
    )
    eval_dataset(args)


if __name__ == "__main__":
    main()
