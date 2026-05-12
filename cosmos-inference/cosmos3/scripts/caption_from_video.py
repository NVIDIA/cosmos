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

"""Generate dense narrative captions from video files using a Vision-Language Model.

Each video is passed directly to a VLM server via a ``video_url`` content part
using a ``file://`` path.  A structured prompt template guides the VLM through
a two-phase captioning process (scene analysis → dense narrative rewrite).

The VLM server must support the OpenAI chat-completions API with vision and
must be started with ``--allowed-local-media-path`` pointing to the root of
your video storage so that it can read video files by path.  Compatible servers
include vLLM serving Qwen2-VL / Qwen3-VL, LLaVA-Next-Video, etc.

Example usage::

    # Caption videos listed in a JSONL file (each line has {"name": ..., "vision_path": ...})
    python -m cosmos3.scripts.caption_from_video \
        -i samples.jsonl -o /output/captions \
        --server http://localhost:8000/v1

    # Caption a single video directly
    python -m cosmos3.scripts.caption_from_video \
        --video /path/to/video.mp4 -o /output/captions \
        --server http://localhost:8000/v1

    # Caption a directory of videos
    python -m cosmos3.scripts.caption_from_video \
        --video /path/to/videos/ -o /output/captions \
        --server http://localhost:8000/v1
"""

import asyncio
import re
from pathlib import Path
from typing import Annotated

import openai
import pydantic
import tyro
from tqdm import tqdm

from cosmos3.args import OmniSampleOverrides
from cosmos3.common.args import VIDEO_EXTENSIONS
from cosmos3._src.imaginaire.utils import log

_PACKAGE_DIR = Path(__file__).parents[1].absolute()


class Args(pydantic.BaseModel):
    input_files: Annotated[list[Path] | None, tyro.conf.arg(aliases=("-i",))] = None
    """Path to input sample argument files (JSON/JSONL).
    Each entry should have at least 'name' and 'vision_path' fields.
    Mutually exclusive with --video."""

    video: Annotated[Path | None, tyro.conf.arg(aliases=("-v",))] = None
    """Path to a single video file or a directory of videos.
    Mutually exclusive with --input_files."""

    output_dir: Annotated[Path, tyro.conf.arg(aliases=("-o",))]
    """Output directory for generated captions."""

    server: str = "http://localhost:8000/v1"
    """The URL of the OpenAI-compatible VLM API server."""
    model: str | None = None
    """The model to use. If not provided, the first model served will be used."""

    max_workers: int = 16
    """Maximum number of concurrent requests to the API."""
    max_retries: int = 5
    """Maximum number of retries for each request."""

    prompt_template_path: Path | None = None
    """Path to a custom prompt template. Defaults to the built-in video_captioner.txt."""

    debug: bool = False
    """If True, save raw API responses for debugging."""


def _extract_xml_tag(text: str, tag: str) -> str | None:
    pattern = rf"<{tag}>\s*(.*?)\s*</{tag}>"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def _build_vlm_messages(
    video_path: Path,
    prompt_template: str,
) -> list[dict]:
    """Build an OpenAI-compatible multimodal message with a video file URL + text prompt.

    The vLLM server must be started with ``--allowed-local-media-path`` so it
    can read the video directly from the shared filesystem.
    """
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "video_url",
                    "video_url": {"url": f"file://{video_path.absolute()}"},
                },
                {"type": "text", "text": prompt_template},
            ],
        }
    ]


async def _process_single(
    args: Args,
    client: openai.AsyncOpenAI,
    name: str,
    video_path: Path,
    prompt_template: str,
):
    assert args.model

    output_dir = args.output_dir / name
    messages = _build_vlm_messages(video_path, prompt_template)

    for i_retry in range(args.max_retries):
        try:
            response = await client.chat.completions.create(
                model=args.model,
                messages=messages,
                max_tokens=2048,
                temperature=0.7,
                top_p=0.8,
                extra_body={"top_k": 20, "min_p": 0.0},
            )
        except Exception as e:
            log.warning(f"[{i_retry + 1}/{args.max_retries}] API Error for {name}: {e}")
            await asyncio.sleep(1)
            continue

        if args.debug:
            retry_dir = output_dir / f"{i_retry}"
            retry_dir.mkdir(parents=True, exist_ok=True)
            (retry_dir / "response.json").write_text(response.model_dump_json())

        assert len(response.choices) == 1
        choice = response.choices[0]
        if choice.finish_reason != "stop" or not choice.message.content:
            log.warning(f"[{i_retry + 1}/{args.max_retries}] Invalid response for {name}")
            continue

        text = choice.message.content.strip()
        final_prompt = _extract_xml_tag(text, "final_prompt")
        if final_prompt is None:
            log.warning(f"[{i_retry + 1}/{args.max_retries}] Failed to extract final prompt for {name}")
            continue

        output_dir.mkdir(parents=True, exist_ok=True)

        sample_overrides = OmniSampleOverrides(
            name=name,
            prompt=final_prompt,
            vision_path=str(video_path),
            output_dir=output_dir,
        )
        (output_dir / "sample_args.json").write_text(sample_overrides.model_dump_json())

        (output_dir / "caption.txt").write_text(final_prompt)
        return

    log.warning(f"Failed to get caption for {name}")


async def _process_with_semaphore(
    args: Args,
    client: openai.AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    name: str,
    video_path: Path,
    prompt_template: str,
):
    async with semaphore:
        return await _process_single(args, client, name, video_path, prompt_template)


def _collect_video_items(args: Args) -> list[tuple[str, Path]]:
    """Return a list of (name, video_path) pairs from the CLI arguments."""
    items: list[tuple[str, Path]] = []

    if args.input_files:
        sample_overrides_list = OmniSampleOverrides.from_files(args.input_files)
        for s in sample_overrides_list:
            if not s.vision_path:
                log.warning(f"Skipping '{s.name}': no vision_path")
                continue
            vp = Path(s.vision_path)
            if vp.suffix.lower() not in VIDEO_EXTENSIONS:
                log.warning(f"Skipping '{s.name}': vision_path is not a video ({vp.suffix})")
                continue
            items.append((s.name or vp.stem, vp))

    elif args.video:
        if args.video.is_dir():
            for vp in sorted(args.video.iterdir()):
                if vp.suffix.lower() in VIDEO_EXTENSIONS:
                    items.append((vp.stem, vp))
        elif args.video.is_file():
            items.append((args.video.stem, args.video))
        else:
            raise FileNotFoundError(f"Video path does not exist: {args.video}")

    if not items:
        raise ValueError("No video inputs found. Provide --input_files (-i) or --video (-v).")
    return items


async def caption_from_video(args: Args):
    if args.input_files and args.video:
        raise ValueError("Provide either --input_files or --video, not both.")

    if args.prompt_template_path:
        prompt_template = args.prompt_template_path.read_text()
    else:
        prompt_template = (_PACKAGE_DIR / "defaults/video_captioner.txt").read_text()

    items = _collect_video_items(args)

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
        _process_with_semaphore(
            args=args,
            client=client,
            semaphore=semaphore,
            name=name,
            video_path=video_path,
            prompt_template=prompt_template,
        )
        for name, video_path in items
    ]
    for result in tqdm(asyncio.as_completed(tasks), desc="Captioning", total=len(tasks)):
        await result


def main():
    args = tyro.cli(Args, description=__doc__)
    asyncio.run(caption_from_video(args))


if __name__ == "__main__":
    main()
