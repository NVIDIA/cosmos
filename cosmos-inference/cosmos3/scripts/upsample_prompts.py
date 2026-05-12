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
import re
from pathlib import Path
from typing import Annotated

import openai
import pydantic
import tyro
from tqdm import tqdm

from cosmos3.args import OmniSampleOverrides
from cosmos3._src.imaginaire.utils import log

_PACKAGE_DIR = Path(__file__).parents[1].absolute()


class Args(pydantic.BaseModel):
    input_files: Annotated[list[Path], tyro.conf.arg(aliases=("-i",))]
    """Path to the input sample argument files."""
    output_dir: Annotated[Path, tyro.conf.arg(aliases=("-o",))]
    """Output directory."""

    server: str = "http://localhost:8000/v1"
    """The URL of the API server."""
    model: str | None = None
    """The model to use.
    
    If not provided, the first model in the list will be used.
    """

    max_workers: int = 16
    """Maximum number of concurrent requests to the API."""
    max_retries: int = 5
    """Maximum number of retries for each request."""

    debug: bool = False
    """If True, enable debug outputs."""


def _extract_xml_tag(text: str, tag: str) -> str | None:
    pattern = rf"<{tag}>\s*(.*?)\s*</{tag}>"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


async def _process_sample(
    args: Args,
    client: openai.AsyncOpenAI,
    sample_overrides: OmniSampleOverrides,
    prompt_template: str,
):
    assert args.model
    assert sample_overrides.name
    assert sample_overrides.prompt

    sample_overrides.output_dir = args.output_dir / sample_overrides.name

    prompt = prompt_template.replace(r"{caption}", sample_overrides.prompt)

    for i_retry in range(args.max_retries):
        # Send request
        try:
            response = await client.chat.completions.create(
                model=args.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2048,
                temperature=0.7,
                top_p=0.8,
                extra_body={"top_k": 20, "min_p": 0.0},
            )
        except Exception as e:
            log.warning(f"[{i_retry + 1}/{args.max_retries}] API Error for {sample_overrides.name}: {e}")
            await asyncio.sleep(1)  # Backoff before retrying
            continue

        if args.debug:
            retry_dir = sample_overrides.output_dir / f"{i_retry}"
            retry_dir.mkdir(parents=True, exist_ok=True)
            (retry_dir / "response.json").write_text(response.model_dump_json())

        assert len(response.choices) == 1
        choice = response.choices[0]
        if choice.finish_reason != "stop" or not choice.message.content:
            log.warning(f"[{i_retry + 1}/{args.max_retries}] Invalid response for {sample_overrides.name}")
            continue

        # Extract final prompt
        text = choice.message.content.strip()
        final_prompt = _extract_xml_tag(text, "final_prompt")
        if final_prompt is None:
            log.warning(
                f"[{i_retry + 1}/{args.max_retries}] Failed to extract final prompt for {sample_overrides.name}"
            )
            continue

        # Save
        sample_overrides.prompt = final_prompt
        sample_overrides.output_dir.mkdir(parents=True, exist_ok=True)
        (sample_overrides.output_dir / "sample_args.json").write_text(sample_overrides.model_dump_json())
        return
    log.warning(f"Failed to get response for {sample_overrides.name}")


async def process_sample(
    args: Args,
    client: openai.AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    sample_overrides: OmniSampleOverrides,
    prompt_template: str,
):
    async with semaphore:
        return await _process_sample(args, client, sample_overrides, prompt_template)


async def upsample_prompts(args: Args):
    sample_overrides_list = OmniSampleOverrides.from_files(args.input_files)
    prompt_template = (_PACKAGE_DIR / "defaults/prompt_upsampler.txt").read_text()

    client = openai.AsyncOpenAI(
        api_key="EMPTY",
        base_url=args.server,
        timeout=3600,
    )
    if not args.model:
        models = await client.models.list()
        args.model = models.data[0].id
        log.info(f"Using model: {args.model}")

    semaphore = asyncio.Semaphore(args.max_workers)

    tasks = [
        process_sample(
            args=args,
            client=client,
            semaphore=semaphore,
            sample_overrides=sample_overrides,
            prompt_template=prompt_template,
        )
        for sample_overrides in sample_overrides_list
    ]
    for result in tqdm(asyncio.as_completed(tasks), desc="Upsampling", total=len(sample_overrides_list)):
        await result


def main():
    args = tyro.cli(Args, description=__doc__)
    asyncio.run(upsample_prompts(args))


if __name__ == "__main__":
    main()
