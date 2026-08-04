# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Generate an image with a selected super-t2i-4step profile."""

import os
from pathlib import Path

import requests
from common import decode_image

NIM_URL = os.environ.get("NIM_URL", "http://localhost:8000").rstrip("/")
OUTPUT = Path(__file__).parent / "outputs" / "t2i_4step.jpg"


def main() -> None:
    # Start the NIM with NIM_MODEL_VARIANT=super-t2i-4step. The profile owns
    # num_inference_steps, guidance_scale, and flow_shift, so this request
    # omits all three.
    request = {
        "model_mode": "text2image",
        "prompt": (
            "A white robotic arm drapes sapphire satin over a dress mannequin "
            "in a softly lit fashion studio, photorealistic editorial style."
        ),
        "negative_prompt": "",
        "resolution": "720_1_1",
        "num_frames": 1,
        "fps": 24.0,
        "seed": 0,
    }

    response = requests.post(f"{NIM_URL}/v1/infer", json=request, timeout=1800)
    response.raise_for_status()

    OUTPUT.parent.mkdir(exist_ok=True)
    OUTPUT.write_bytes(decode_image(response.json()["b64_image"]))
    print(f"Saved image to {OUTPUT}")


if __name__ == "__main__":
    main()
