# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Run a Reasoner image-caption request through the Responses API."""

import os
from pathlib import Path

from common import media_to_data_url, require_runtime
from openai import OpenAI

NIM_URL = os.environ.get("NIM_URL", "http://localhost:8000").rstrip("/")
IMAGE = Path(__file__).resolve().parents[2] / "reasoner" / "assets" / "robot_153.jpg"


def main() -> None:
    require_runtime(
        NIM_URL,
        expected_runtime="reasoner",
        expected_endpoint="/v1/chat/completions",
    )
    client = OpenAI(base_url=f"{NIM_URL}/v1", api_key="not-used", timeout=1800)
    response = client.responses.create(
        model=client.models.list().data[0].id,
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_image",
                        "detail": "auto",
                        "image_url": media_to_data_url(IMAGE),
                    },
                    {"type": "input_text", "text": "Caption the image in detail."},
                ],
            }
        ],
        max_output_tokens=4096,
        store=False,
    )
    print(response.output_text)


if __name__ == "__main__":
    main()
