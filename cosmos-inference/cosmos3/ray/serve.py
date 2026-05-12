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

import asyncio
import importlib.metadata
import os
import random
import socket
import subprocess
import uuid
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Sequence, cast

import fastapi
import pydantic
import ray
import ray.serve
import ray.serve.handle
import torch
import tyro
from fastapi.staticfiles import StaticFiles
from ray.serve.config import AutoscalingConfig

from cosmos3.args import OmniSampleOverrides, OmniSetupArgs, OmniSetupOverrides
from cosmos3.common.args import ResolvedPath, SampleOutputs, tyro_cli
from cosmos3._src.imaginaire.utils import log

if TYPE_CHECKING:
    from ray.actor import ActorClass
    from ray.serve.batching import _LazyBatchQueueWrapper

torch.set_grad_enabled(False)

fastapi_app = fastapi.FastAPI()

DEFAULT_MAX_BATCH_SIZE = 1
DEFAULT_BATCH_WAIT_TIMEOUT_S = 0.01
DEFAULT_AUTOSCALING_CONFIG = AutoscalingConfig()


def _get_environment_info() -> dict[str, Any]:
    try:
        commit_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        commit_sha = None
    return {
        "cosmos3_version": importlib.metadata.version("cosmos3"),
        "commit_sha": commit_sha,
    }


class OmniAppArgs(pydantic.BaseModel):
    output_dir: Annotated[ResolvedPath, tyro.conf.arg(aliases=("-o",))]


class OmniModelDeploymentUserConfig(pydantic.BaseModel):
    max_batch_size: int = DEFAULT_MAX_BATCH_SIZE
    batch_wait_timeout_s: float = DEFAULT_BATCH_WAIT_TIMEOUT_S


class OmniModelDeploymentArgs(pydantic.BaseModel):
    setup: OmniSetupOverrides
    world_size: int | None = None

    user_config: OmniModelDeploymentUserConfig = pydantic.Field(default_factory=OmniModelDeploymentUserConfig)
    autoscaling_config: dict[str, Any] = pydantic.Field(default_factory=lambda: DEFAULT_AUTOSCALING_CONFIG.dict())


class OmniRouterDeploymentArgs(pydantic.BaseModel):
    app: OmniAppArgs
    models: dict[str, OmniModelDeploymentArgs] = pydantic.Field(min_length=1)


@ray.remote(num_gpus=1)
class OmniModelWorker:
    def __init__(self, setup_args: OmniSetupArgs, *, rank: int, world_size: int):
        self.setup_args = setup_args

        # Ray sets CUDA_VISIBLE_DEVICES, so LOCAL_WORLD_SIZE is always 1.
        local_rank = 0
        local_world_size = 1

        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(local_rank)
        os.environ["LOCAL_WORLD_SIZE"] = str(local_world_size)

    def get_address_and_port(self) -> tuple[str, int]:
        ip = ray.util.get_node_ip_address()
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            port = s.getsockname()[1]
        return ip, port

    def setup_distributed(self, master_addr: str, master_port: int) -> bool:
        from cosmos3.inference import OmniInference
        from cosmos3._src.imaginaire.utils import distributed

        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = str(master_port)
        distributed.init()

        self.pipe = OmniInference.create(self.setup_args)
        return True

    def generate_batch(self, sample_overrides_list: Sequence[OmniSampleOverrides]) -> list[SampleOutputs]:
        # Create batches
        sample_args_list = [args.build_sample(model_config=self.pipe.model.config) for args in sample_overrides_list]
        batches = self.pipe.create_batches(sample_args_list)

        # Generate batches
        sample_outputs_list: list[SampleOutputs] = []
        for sample_args_list, data_batch in batches:
            sample_outputs_list.extend(self.pipe.generate_batch(sample_args_list, data_batch))
        return sample_outputs_list


@ray.serve.deployment(
    health_check_timeout_s=5 * 60,  # Model loading can take a while
    autoscaling_config=DEFAULT_AUTOSCALING_CONFIG,
)
class OmniModelDeployment:
    def __init__(self, setup_args: OmniSetupArgs):
        self.setup_args = setup_args
        world_size = setup_args.world_size

        self.pg = ray.util.placement_group([{"GPU": 1, "CPU": 1} for _ in range(world_size)])
        ray.get(self.pg.ready())

        # Spawn the workers
        self.workers = [
            cast("ActorClass", OmniModelWorker)
            .options(placement_group=self.pg)
            .remote(
                rank=i,
                world_size=world_size,
                setup_args=setup_args,
            )
            for i in range(world_size)
        ]

        master_addr, master_port = ray.get(self.workers[0].get_address_and_port.remote())
        ray.get([w.setup_distributed.remote(master_addr, master_port) for w in self.workers])

    def reconfigure(self, user_config_dict: dict[str, Any]):
        user_config = OmniModelDeploymentUserConfig.model_validate(user_config_dict)

        _generate_batch = cast("_LazyBatchQueueWrapper", self._generate_batch)
        _generate_batch.set_max_batch_size(user_config.max_batch_size)
        _generate_batch.set_batch_wait_timeout_s(user_config.batch_wait_timeout_s)

    async def generate(self, sample_overrides: OmniSampleOverrides) -> SampleOutputs:
        result = await self._generate_batch(sample_overrides)
        return result

    @ray.serve.batch(
        max_batch_size=DEFAULT_MAX_BATCH_SIZE,
        batch_wait_timeout_s=DEFAULT_BATCH_WAIT_TIMEOUT_S,
    )
    async def _generate_batch(self, sample_overrides_list: list[OmniSampleOverrides]) -> list[SampleOutputs]:
        tasks = [worker.generate_batch.remote(sample_overrides_list) for worker in self.workers]
        # Wait for all workers to complete.
        results = await asyncio.wait_for(
            asyncio.gather(*tasks),  # type: ignore
            timeout=300.0,
        )

        sample_outputs_list = cast(list[SampleOutputs], results[0])
        sample_outputs_list = [
            obj.map_files(lambda p: p.relative_to(self.setup_args.output_dir)) for obj in sample_outputs_list
        ]
        return sample_outputs_list


@ray.serve.deployment(num_replicas=1)
@ray.serve.ingress(fastapi_app)
class OmniRouterDeployment:
    def __init__(self, models: dict[str, ray.serve.handle.DeploymentHandle], output_dir: Path):
        output_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = output_dir
        self.models = models
        fastapi_app.mount("/outputs", StaticFiles(directory=output_dir), name="outputs")

    @fastapi_app.get("/models")
    async def list_models(self) -> list[str]:
        return list(self.models.keys())

    @fastapi_app.get("/info")
    async def info(self) -> dict[str, Any]:
        return {
            "environment": _get_environment_info(),
            "models": list(self.models.keys()),
            "output_dir": str(self.output_dir),
        }

    @fastapi_app.post("/generate")
    async def generate(self, sample_args: OmniSampleOverrides) -> SampleOutputs:
        if sample_args.model:
            if sample_args.model not in self.models:
                raise fastapi.HTTPException(
                    status_code=404,
                    detail=f"Model '{sample_args.model}' not found. Available models: {list(self.models.keys())}",
                )
            handle = self.models[sample_args.model]
        else:
            handle = next(iter(self.models.values()))
        sample_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sample_uuid = uuid.uuid4().hex
        sample_args.output_dir = self.output_dir / f"generate_{sample_timestamp}_{sample_uuid[:4]}"
        if sample_args.seed is None:
            sample_args.seed = random.randint(0, 10000)
        sample_args.download(sample_args.output_dir / "inputs")

        result: SampleOutputs = await handle.generate.remote(sample_args)
        return result


def router_app_builder(router_args_dict: dict) -> ray.serve.Application:
    router_args = OmniRouterDeploymentArgs.model_validate(router_args_dict)

    # Build models
    models = {}
    for model_name, model_args in router_args.models.items():
        model_args.setup.output_dir = router_args.app.output_dir
        setup_args = model_args.setup.build_setup(
            world_size=model_args.world_size,
        )
        log.info(f"{model_name}: {setup_args.__class__.__name__}({setup_args})")
        models[model_name] = (
            cast(ray.serve.Deployment, OmniModelDeployment)
            .options(
                name=model_name,
                autoscaling_config=model_args.autoscaling_config,
                user_config=model_args.user_config.model_dump(),
            )
            .bind(setup_args)
        )
    return cast(ray.serve.Deployment, OmniRouterDeployment).bind(
        models,
        output_dir=router_args.app.output_dir,
    )


def main():
    model_args = tyro_cli(OmniModelDeploymentArgs, description=__doc__, config=(tyro.conf.OmitArgPrefixes,))
    if model_args.setup.output_dir is None:
        raise ValueError("An output directory is required. Set it with -o <path> or --output-dir <path>.")

    ray.init()  # type: ignore
    if model_args.world_size is None:
        available_resources = ray.available_resources()  # type: ignore
        available_gpus = int(available_resources.get("GPU", 0))
        model_args.world_size = max(1, available_gpus)

    app_args = OmniAppArgs(
        output_dir=model_args.setup.output_dir,
    )
    router_args = OmniRouterDeploymentArgs(app=app_args, models={"": model_args})
    router_app = router_app_builder(router_args.model_dump())
    ray.serve.run(router_app, name="cosmos3_omni", blocking=True)


if __name__ == "__main__":
    main()
