# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import argparse
import json
import os
from collections.abc import Iterable

import torch
from PIL import Image

from cosmos3._src.imaginaire.auxiliary.guardrail.common.core import (
    GUARDRAIL1_CHECKPOINT,
    ContentSafetyGuardrail,
    GuardrailRunner,
)
from cosmos3._src.imaginaire.auxiliary.guardrail.common.io_utils import get_video_filepaths, read_video
from cosmos3._src.imaginaire.auxiliary.guardrail.video_content_safety_filter.model import ModelConfig, VideoSafetyModel
from cosmos3._src.imaginaire.auxiliary.guardrail.video_content_safety_filter.vision_encoder import SigLIPEncoder
from cosmos3._src.imaginaire.utils import log, misc

# Define the class index to class name mapping for multi-class classification
CLASS_IDX_TO_NAME = {
    0: "Safe",
    1: "Sexual_Content",
    3: "Drugs",
    4: "Child_Abuse",
    5: "Hate_and_Harassment",
    6: "Self-Harm",
}

CUTOFF_UNSAFE_FRAMES_PERCENT = 10  # 10% of frames are unsafe, then the video is unsafe


class VideoContentSafetyFilter(ContentSafetyGuardrail):
    def __init__(
        self,
        offload_model_to_cpu: bool = True,
    ) -> None:
        """Video content safety filter model.

        Args:
            checkpoint_dir (str): Path to the checkpoint directory.
            offload_model_to_cpu (bool, optional): Whether to offload the model to CPU. Defaults to True.
        """
        self.offload_model = offload_model_to_cpu
        self.dtype = torch.float32
        self.checkpoint_dir = os.path.join(GUARDRAIL1_CHECKPOINT.download(), "video_content_safety_filter")

        # Use ModelConfig directly for inference configuration
        model_config = ModelConfig(input_size=1152, num_classes=7)

        # Load the multi-class classifier and initialize the SigLIP encoder
        self.model = VideoSafetyModel(model_config)
        safety_filter_local_path = os.path.join(self.checkpoint_dir, "safety_filter.pt")
        checkpoint = torch.load(safety_filter_local_path, map_location=torch.device("cpu"), weights_only=True)
        self.model.load_state_dict(checkpoint["model"])
        self.encoder = SigLIPEncoder(device="cuda", dtype=self.dtype)
        if offload_model_to_cpu:
            self.encoder.to("cpu")
            self.model = self.model.to("cpu", dtype=self.dtype).eval()
            log.debug("Moved video content safety filter to CPU")
        else:
            self.encoder.to("cuda")
            self.model = self.model.to("cuda", dtype=self.dtype).eval()
            log.debug("Moved video content safety filter to GPU")

    @torch.inference_mode()
    def __infer(self, pil_image: Image.Image) -> int:
        """Infer the class of the image."""
        image_embs = self.encoder.encode_image(pil_image)
        logits = self.model.network(image_embs)
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        predicted_class = int(torch.argmax(probabilities, dim=-1).item())
        return predicted_class

    def _to_cuda_if_offload(self):
        if self.offload_model:
            self.encoder = self.encoder.to("cuda")
            self.model = self.model.to("cuda")
            log.debug("Move video content safety filter to GPU")

    def _to_cpu_if_offload(self):
        if self.offload_model:
            self.encoder = self.encoder.to("cpu")
            self.model = self.model.to("cpu")
            log.debug("Offload video content safety filter to CPU")

    def is_safe_file(self, filepath: str) -> bool:
        """Check if the video file is safe."""
        video_data = read_video(filepath)

        # Sample frames at 2 FPS
        sample_rate = 2  # frames per second
        frame_interval = int(video_data.fps / sample_rate)
        frame_numbers = list(range(0, int(video_data.fps * video_data.duration), frame_interval))
        frames = [video_data.frames[frame_number] for frame_number in frame_numbers]
        return self.is_safe_frames(frames)

    def is_safe_frames(self, frames: Iterable) -> bool:
        """Check if video frames are safe. Populates ``self.last_diagnostics`` as a side effect;
        single-instance serial use only (not thread-safe)."""
        is_safe = True
        frame_scores: list[dict] = []
        unsafe_frame_count = 0
        total_frame_count = 0

        self._to_cuda_if_offload()
        for frame_number, frame in enumerate(frames):
            total_frame_count += 1
            try:
                pil_image = Image.fromarray(frame)
                predicted_class = self.__infer(pil_image)
                class_name = CLASS_IDX_TO_NAME.get(predicted_class, "Unknown")
                frame_scores.append({"frame_number": frame_number, "class": class_name})

                # If any frame considered in the list of unsafe categories, mark the video as unsafe
                if class_name != "Safe" and class_name in CLASS_IDX_TO_NAME.values():
                    log.warning(f"Unsafe frame detected in frame_number {frame_number}: {class_name}")
                    unsafe_frame_count += 1

            except Exception as e:
                log.warning(f"Warning: Failed to run safety classifier on frame_number {frame_number}. Exception: {e}")
                continue

        unsafe_ratio = unsafe_frame_count / total_frame_count if total_frame_count else 0.0
        if unsafe_ratio > (CUTOFF_UNSAFE_FRAMES_PERCENT / 100):
            is_safe = False
            log.warning(
                f"Unsafe frame count {unsafe_frame_count} is greater than {CUTOFF_UNSAFE_FRAMES_PERCENT}% of total frames {total_frame_count}"
            )

        # .get(..., "Safe") guards against future callers appending partial entries; "Safe" is filtered out.
        unsafe_categories = sorted({s.get("class", "Safe") for s in frame_scores if s.get("class", "Safe") != "Safe"})
        self.last_diagnostics: dict = {
            "unsafe_frames": unsafe_frame_count,
            "total_frames": total_frame_count,
            "unsafe_ratio": unsafe_ratio,
            "unsafe_categories": unsafe_categories,
            "cutoff_percent": CUTOFF_UNSAFE_FRAMES_PERCENT,
        }

        video_data = {
            "is_safe": is_safe,
            "frame_scores": frame_scores,
        }
        self._to_cpu_if_offload()
        log.debug(f"Frames data: {json.dumps(video_data, indent=4)}")
        return is_safe

    def _format_block_message(self) -> str:
        """Build a diagnostic message for the most recent unsafe classification."""
        d = getattr(self, "last_diagnostics", None)
        if not d:
            return "unsafe content detected"
        return (
            f"unsafe content detected: "
            f"{d['unsafe_frames']}/{d['total_frames']} frames "
            f"({d['unsafe_ratio']:.1%}) exceed the {d['cutoff_percent']}% cutoff; "
            f"categories={d['unsafe_categories']}"
        )

    def is_safe(self, input: str | Iterable) -> tuple[bool, str]:
        if isinstance(input, str):
            is_safe = self.is_safe_file(input)
            return is_safe, "safe video detected" if is_safe else self._format_block_message()
        elif isinstance(input, Iterable):
            is_safe = self.is_safe_frames(input)
            return is_safe, "safe frames detected" if is_safe else self._format_block_message()
        else:
            raise ValueError(f"Input type {type(input)} not supported.")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="Path containing input videos")
    return parser.parse_args()


def main(args):
    filepaths = get_video_filepaths(args.input_dir)
    if not filepaths:
        log.error(f"No video files found in directory: {args.input_dir}")
        return

    video_filter = VideoContentSafetyFilter()
    runner = GuardrailRunner(safety_models=[video_filter], generic_safe_msg="Video is safe")

    for filepath in filepaths:
        with misc.timer("video content safety filter"):
            _ = runner.run_safety_check(filepath)


if __name__ == "__main__":
    args = parse_args()
    main(args)
