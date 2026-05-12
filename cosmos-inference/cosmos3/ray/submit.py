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

from pathlib import Path
from typing import Annotated

import pydantic
import requests
import tyro

from cosmos3.args import OmniSampleOverrides
from cosmos3.common.args import ResolvedPath, SampleOutputs, tyro_cli
from cosmos3._src.imaginaire.utils import log


class Args(pydantic.BaseModel):
    output_dir: Annotated[ResolvedPath, tyro.conf.arg(aliases=("-o",))]
    """Output directory."""
    input_files: Annotated[list[ResolvedPath], tyro.conf.arg(aliases=("-i",))]
    """Path to the inference parameter file(s).

    If multiple files are provided, the model will be loaded once and all the samples will be run sequentially.
    
    Accepts glob patterns (e.g. `inputs/*.json`).
    """
    overrides: OmniSampleOverrides = pydantic.Field(default_factory=OmniSampleOverrides)
    """Overrides for the inference parameters."""

    host: str = "localhost"
    """The hostname of the Ray Serve endpoint."""
    port: int = 8000
    """The port of the Ray Serve endpoint."""

    timeout: int = 600
    """The timeout for the request in seconds."""

    @property
    def server(self) -> str:
        return f"http://{self.host}:{self.port}"


def handle_response(response: requests.Response) -> None:
    try:
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        try:
            detail = response.json().get("detail")
        except Exception:
            detail = None
        if detail:
            raise ValueError(f"{detail}") from e
        raise e


def submit(args: Args):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sample_args_list = OmniSampleOverrides.from_files(args.input_files, overrides=args.overrides)


    for i_sample, sample_args in enumerate(sample_args_list):
        assert sample_args.name
        log.info(f"[{i_sample + 1}/{len(sample_args_list)}] Submitting sample '{sample_args.name}'")
        log.debug(f"{sample_args.__class__.__name__}({sample_args})")

        sample_dir = args.output_dir / sample_args.name
        sample_dir.mkdir(parents=True, exist_ok=True)
        sample_args_file = sample_dir / "sample_args.json"
        sample_args_file.write_text(sample_args.model_dump_json())
        log.info(f"Saved sample args to '{sample_args_file}'")

        response = requests.post(
            f"{args.server}/generate", json=sample_args.model_dump(mode="json"), timeout=args.timeout
        )
        handle_response(response)
        response_dict = response.json()
        sample_outputs = SampleOutputs.model_validate(response_dict)

        # Download files
        def download_file(remote_file: Path) -> Path:
            response = requests.get(f"{args.server}/outputs/{remote_file}", timeout=args.timeout)
            handle_response(response)
            local_file = sample_dir.joinpath(*remote_file.parts[1:])
            local_file.write_bytes(response.content)
            return local_file

        sample_outputs = sample_outputs.map_files(download_file)

        sample_outputs_file = sample_dir / "sample_outputs.json"
        sample_outputs_file.write_text(sample_outputs.model_dump_json())
        log.info(f"Saved sample outputs to '{sample_outputs_file}'")


def main():
    args = tyro_cli(Args, description=__doc__, config=(tyro.conf.OmitArgPrefixes,))
    submit(args)


if __name__ == "__main__":
    main()
