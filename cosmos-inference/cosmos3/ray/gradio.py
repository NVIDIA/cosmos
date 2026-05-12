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

import json
import time
import typing
from functools import partial
from pathlib import Path
from typing import Any, Literal, Union, cast

import gradio as gr
import httpx
import pydantic

from cosmos3.args import OmniSampleOverrides
from cosmos3.common.args import (
    MEDIA_EXTENSIONS,
    ResolvedPath,
    SampleOutputs,
    tyro_cli,
)

INPUTS_DIR = Path(__file__).parents[2] / "inputs"


class Args(pydantic.BaseModel):
    host: str = "localhost"
    """The hostname to bind the Gradio server to."""
    port: int = 8080
    """The port to bind the Gradio server to. If None, a random port will be used."""
    server_host: str = "localhost"
    """The hostname of the Ray Serve endpoint."""
    server_port: int = 8000
    """The port of the Ray Serve endpoint."""
    server_output_dir: ResolvedPath = Path("outputs/ray_serve")
    """Server output directory."""

    timeout: int = 600
    """The timeout for the request in seconds."""

    @property
    def server(self) -> str:
        return f"http://{self.server_host}:{self.server_port}"


def get_info(args: Args) -> dict[str, Any]:
    response = httpx.get(f"{args.server}/info", timeout=args.timeout)
    response.raise_for_status()
    return response.json()


def build_components(
    model: type[pydantic.BaseModel], include: dict[str, dict[str, Any] | None]
) -> dict[str, gr.components.Component]:
    components: dict[str, gr.components.Component] = {}
    for name, kwargs in include.items():
        field = model.model_fields[name]

        default_val = field.get_default(call_default_factory=True)

        min_val, max_val = None, None
        for meta in field.metadata:
            if hasattr(meta, "ge"):
                min_val = meta.ge
            if hasattr(meta, "gt"):
                min_val = meta.gt + 1
            if hasattr(meta, "le"):
                max_val = meta.le
            if hasattr(meta, "lt"):
                max_val = meta.lt - 1

        kwargs = dict(
            value=default_val,
            label=name.replace("_", " ").title(),
        ) | (kwargs or {})
        kwargs = cast(dict[str, Any], kwargs)

        annotation = field.annotation
        is_optional = False
        if typing.get_origin(annotation) is Union:
            args = typing.get_args(annotation)
            is_optional = type(None) in args
            args = [arg for arg in args if arg is not type(None)]
            if len(args) == 1:
                annotation = args[0]
        if typing.get_origin(annotation) is Literal:
            choices = list(typing.get_args(annotation))
            if is_optional:
                choices.insert(0, None)
            kwargs.setdefault("choices", choices)
            components[name] = gr.Dropdown(**kwargs)
        elif annotation in (int, float):
            if min_val is not None:
                kwargs.setdefault("minimum", min_val)
            if max_val is not None:
                kwargs.setdefault("maximum", max_val)
            if "minimum" in kwargs and "maximum" in kwargs:
                components[name] = gr.Slider(**kwargs)
            else:
                components[name] = gr.Number(**kwargs)
        elif annotation == bool:
            components[name] = gr.Checkbox(**kwargs)
        else:
            components[name] = gr.Textbox(**kwargs)
    return components


COMPONENTS: dict[str, dict[str, Any] | None] = {
    "prompt": dict(
        lines=3,
    ),
    "resolution": None,
    "aspect_ratio": None,
    "fps": None,
    "num_frames": None,
    "seed": None,
}
EXCLUDE_FIELDS = [
    "model",
    "name",
    "output_dir",
    "prompt_path",
    "tensors_file",
    "pickle_file",
]


def _load_sample_outputs(args: Args, sample_outputs: SampleOutputs):
    if sample_outputs.status == "success":
        media_output = [
            (f"{args.server_output_dir}/{p}", p.name)
            for p in sample_outputs.outputs[0].files
            if p.suffix.lower() in MEDIA_EXTENSIONS
        ]
    else:
        media_output = []
    return media_output


async def generate(*inputs, args: Args):
    try:
        model, *values, extra_json = inputs
        request_dict = {k: v for k, v in zip(COMPONENTS, values, strict=True) if v}
        name = str(round(time.time()))
        request_dict = request_dict | json.loads(extra_json) | {"model": model, "name": name}
        sample_args = OmniSampleOverrides.model_validate(request_dict)

        async with httpx.AsyncClient(timeout=args.timeout) as http_client:
            response = await http_client.post(
                f"{args.server}/generate",
                json=sample_args.model_dump(mode="json"),
            )
        response.raise_for_status()

        response_dict = response.json()
        sample_outputs = SampleOutputs.model_validate(response_dict)
        media_output = _load_sample_outputs(args, sample_outputs)
        return media_output, request_dict, response_dict
    except Exception as e:
        raise gr.Error(f"Error generating: {str(e)}")


def load_input(input_name: str, examples: dict[str, Path]):
    if input_name:
        input_file = examples[input_name]
        sample_args_list = OmniSampleOverrides.from_files([input_file])
        assert len(sample_args_list) == 1
        sample_args = sample_args_list[0]
        if sample_args.prompt_path:
            sample_args.prompt = Path(sample_args.prompt_path).read_text().strip()
    else:
        sample_args = OmniSampleOverrides(name="")

    updates = []
    updates.append(sample_args.model or gr.update())
    for comp_name in COMPONENTS:
        updates.append(getattr(sample_args, comp_name))
    extra_data = sample_args.model_dump_json(indent=2, exclude={*COMPONENTS, *EXCLUDE_FIELDS})
    updates.append(extra_data)
    return tuple(updates)


def ui_builder(args: Args) -> gr.Blocks:
    info = get_info(args)
    available_models = info["models"]
    if len(available_models) == 0:
        raise ValueError("No models available")

    default_example = "t2i"
    examples: dict[str, Path] = {}
    for p in INPUTS_DIR.rglob("*.json"):
        if "internal" in p.parts:
            continue
        if p.stem in examples:
            raise ValueError(f"Duplicate example file: {p}")
        examples[p.stem] = p
    assert default_example in examples

    with gr.Blocks(title="Cosmos3 Omni Generator") as ui:
        gr.Markdown("# Cosmos3 Omni Generator")
        with gr.Accordion("Environment", open=False):
            gr.JSON(value=info["environment"])

        example_dropdown = gr.Dropdown(
            choices=["", *sorted(examples.keys())],
            value=default_example,
            label="Input",
        )

        with gr.Row():
            with gr.Column(scale=1):
                model_input = gr.Dropdown(value=available_models[0], choices=available_models, label="Model")

                generate_btn = gr.Button("Generate", variant="primary")

                components = build_components(OmniSampleOverrides, COMPONENTS)

                with gr.Accordion("Extra Arguments", open=False):
                    extra_json = OmniSampleOverrides(name="").model_dump_json(
                        indent=2,
                        exclude={*COMPONENTS, *EXCLUDE_FIELDS},
                    )
                    extra_input = gr.Code(
                        extra_json,
                        language="json",
                        lines=10,
                    )

            with gr.Column(scale=1):
                media_output = gr.Gallery(label="Media", allow_preview=True)

                with gr.Accordion("Request", open=False):
                    request_output = gr.JSON()

                with gr.Accordion("Response", open=False):
                    response_output = gr.JSON()

        load_input_kwargs = dict(
            fn=partial(load_input, examples=examples),
            inputs=[example_dropdown],
            outputs=[model_input, *components.values(), extra_input],
        )
        example_dropdown.change(**load_input_kwargs)
        ui.load(**load_input_kwargs)

        generate_btn.click(
            fn=partial(generate, args=args),
            inputs=[model_input, *components.values(), extra_input],
            outputs=[media_output, request_output, response_output],
        )

    return ui


def main():
    args = tyro_cli(Args, description=__doc__)
    ui = ui_builder(args)
    ui.launch(server_name=args.host, server_port=args.port, allowed_paths=[str(args.server_output_dir)])


if __name__ == "__main__":
    main()
