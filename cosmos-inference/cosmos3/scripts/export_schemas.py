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

"""Export config defaults and schemas."""

from cosmos3.common.init import init_script

init_script(default_env={"COSMOS_TRAINING": "0"})

import argparse
import json
import pathlib
import textwrap
import typing

import pydantic
import yaml

from cosmos3.args import OmniSampleOverrides, OmniSetupOverrides

MODELS: list[tuple[type[pydantic.BaseModel], dict[str, typing.Any]]] = [
    (OmniSetupOverrides, {}),
    (OmniSampleOverrides, {}),
]


def _nested_model_cls(field: pydantic.fields.FieldInfo) -> type[pydantic.BaseModel] | None:
    """Return the BaseModel subclass backing *field*, if any."""
    ann = field.annotation
    if isinstance(ann, type) and issubclass(ann, pydantic.BaseModel):
        return ann
    for arg in typing.get_args(ann):
        if isinstance(arg, type) and issubclass(arg, pydantic.BaseModel):
            return arg
    return None


def _commented_yaml(model_cls: type, data: dict) -> str:
    lines: list[str] = []
    for name, field in sorted(model_cls.model_fields.items()):
        if name not in data:
            continue
        if field.description:
            for dl in field.description.strip().splitlines():
                s = dl.strip()
                lines.append(f"# {s}" if s else "#")
        value = data[name]
        nested = _nested_model_cls(field)
        if nested and isinstance(value, dict):
            lines.append(f"{name}:")
            lines.append(textwrap.indent(_commented_yaml(nested, value).rstrip("\n"), "  "))
        else:
            lines.append(yaml.dump({name: value}, default_flow_style=False, sort_keys=False).rstrip())
    return "\n".join(lines) + "\n"


def export_schemas(output_dir: pathlib.Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for cls, defaults in MODELS:
        data = cls.model_construct(**defaults).model_dump(mode="json")
        (output_dir / f"{cls.__name__}.yaml").write_text(_commented_yaml(cls, data))
        (output_dir / f"{cls.__name__}.schema.json").write_text(json.dumps(cls.model_json_schema(), indent=2) + "\n")
        print(f"Saved {cls.__name__} -> {output_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", type=pathlib.Path, default="schemas", help="Output directory")
    args = parser.parse_args()
    export_schemas(args.output.absolute())


if __name__ == "__main__":
    main()
