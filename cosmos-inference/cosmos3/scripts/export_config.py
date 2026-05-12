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

"""Export config to yaml file."""

from cosmos3.common.init import init_script

init_script(
    env={
        "COSMOS_TRAINING": "1",
        "COSMOS_DEVICE": "cpu",
    }
)

from typing import Annotated

import tyro

from cosmos3.common.args import ConfigOverrides, ResolvedPath, tyro_cli
from cosmos3.common.config import InvalidMode


class Args(ConfigOverrides):
    output_file: Annotated[ResolvedPath, tyro.conf.arg(aliases=("-o",))]

    invalid: InvalidMode = "error"
    """How to handle unknown field types."""
    config_key: str | None = None
    """Config key to export."""


def export_config(args: Args):
    config_args = args.build_config()
    if args.output_file.suffix not in [".yaml", ".yml"]:
        raise ValueError("Output file must have a .yaml or .yml extension")

    from cosmos3.common.config import serialize_config

    config = config_args.load_config()

    if args.config_key:
        for k in args.config_key.split("."):
            config = getattr(config, k)

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    serialize_config(config, args.output_file, invalid=args.invalid)
    print(f"Saved config to {args.output_file}")


def main():
    args = tyro_cli(Args, description=__doc__)
    export_config(args)


if __name__ == "__main__":
    main()
