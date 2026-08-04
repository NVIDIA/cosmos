# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Generate an image from a text prompt with the Generator runtime."""

import os
from pathlib import Path

import requests
from common import decode_image

NIM_URL = os.environ.get("NIM_URL", "http://localhost:8000").rstrip("/")
OUTPUT = Path(__file__).parent / "outputs" / "t2i.jpg"


def main() -> None:
    request = {
        "model_mode": "text2image",
        "prompt": (
            "Photorealistic fashion-studio scene shot at eye level: a sleek "
            "white-and-aluminum robotic arm enters from the upper left, its "
            "precision two-finger gripper delicately pinching the edge of a "
            "length of deep sapphire blue satin and draping it over a beige "
            "headless dress mannequin mounted on a chrome stand at the center "
            "of the frame. The satin forms deliberate diagonal pleats across "
            "the torso and cascades down to pool luxuriously on the polished "
            "pale concrete floor. A few silver dressmaker pins catch pinpoint "
            "highlights along the right side seam. A large softbox from the "
            "right casts smooth, creamy specular highlights along the satin "
            "folds and gentle, soft shadows beneath the mannequin's waist. In "
            "the softly blurred background, a wooden cutting table holds a "
            "coiled yellow measuring tape, and chrome garment racks recede "
            "into bokeh against clean neutral cream walls. The mood is precise, "
            "futuristic, and quietly elegant—an editorial vision of "
            "robotic-assisted couture."
        ),
        "negative_prompt": "",
        "resolution": "720_1_1",
        "num_frames": 1,
        "fps": 24.0,
        "num_inference_steps": 50,
        "guidance_scale": 4.0,
        "flow_shift": 3.0,
        "seed": 0,
    }

    response = requests.post(f"{NIM_URL}/v1/infer", json=request, timeout=1800)
    response.raise_for_status()

    OUTPUT.parent.mkdir(exist_ok=True)
    OUTPUT.write_bytes(decode_image(response.json()["b64_image"]))
    print(f"Saved image to {OUTPUT}")


if __name__ == "__main__":
    main()
