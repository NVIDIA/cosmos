# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Run a Reasoner image or video Chat Completions request."""

import argparse
import os
from pathlib import Path

from common import media_to_data_url
from openai import OpenAI

NIM_URL = os.environ.get("NIM_URL", "http://localhost:8000").rstrip("/")
ASSETS = Path(__file__).resolve().parents[2] / "reasoner" / "assets"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=("image", "video"), default="image")
    case = parser.parse_args().case

    client = OpenAI(base_url=f"{NIM_URL}/v1", api_key="not-used", timeout=1800)
    model = client.models.list().data[0].id

    if case == "image":
        media = {
            "type": "image_url",
            "image_url": {"url": media_to_data_url(ASSETS / "robot_153.jpg")},
        }
        prompt = "Caption the image in detail."
    else:
        media = {
            "type": "video_url",
            "video_url": {"url": media_to_data_url(ASSETS / "video_caption.mp4")},
        }
        prompt = "Describe the video in detail."

    request = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [media, {"type": "text", "text": prompt}],
            }
        ],
        "max_tokens": 4096,
    }
    if case == "image":
        request["seed"] = 0
    else:
        request["extra_body"] = {"media_io_kwargs": {"video": {"fps": 4.0}}}

    response = client.chat.completions.create(**request)
    print(response.choices[0].message.content)


if __name__ == "__main__":
    main()
