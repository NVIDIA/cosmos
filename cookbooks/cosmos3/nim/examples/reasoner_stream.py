# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Stream one Reasoner image-caption response to the terminal."""

import os
from pathlib import Path

from common import media_to_data_url
from openai import OpenAI

NIM_URL = os.environ.get("NIM_URL", "http://localhost:8000").rstrip("/")
IMAGE = Path(__file__).resolve().parents[2] / "reasoner" / "assets" / "robot_153.jpg"


def main() -> None:
    client = OpenAI(base_url=f"{NIM_URL}/v1", api_key="not-used", timeout=1800)
    stream = client.chat.completions.create(
        model=client.models.list().data[0].id,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": media_to_data_url(IMAGE)},
                    },
                    {"type": "text", "text": "Caption the image in detail."},
                ],
            }
        ],
        max_tokens=4096,
        seed=0,
        stream=True,
    )

    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
    print()


if __name__ == "__main__":
    main()
