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
import os
import warnings

import numpy as np
import torch
from retinaface.data import cfg_re50
from retinaface.layers.functions.prior_box import PriorBox
from retinaface.models.retinaface import RetinaFace
from tqdm import tqdm

from cosmos3._src.imaginaire.auxiliary.guardrail.common.core import (
    GUARDRAIL1_CHECKPOINT,
    GuardrailRunner,
    PostprocessingGuardrail,
)
from cosmos3._src.imaginaire.auxiliary.guardrail.common.io_utils import (
    get_video_filepaths,
    read_video,
    save_video,
)
from cosmos3._src.imaginaire.auxiliary.guardrail.face_blur_filter.blur_utils import pixelate_face
from cosmos3._src.imaginaire.auxiliary.guardrail.face_blur_filter.retinaface_utils import (
    decode_batch,
    filter_detected_boxes,
    load_model,
)
from cosmos3._src.imaginaire.utils import log, misc

# RetinaFace model constants from https://github.com/biubug6/Pytorch_Retinaface/blob/master/detect.py
TOP_K = 5_000
KEEP_TOP_K = 750
NMS_THRESHOLD = 0.4


class RetinaFaceFilter(PostprocessingGuardrail):
    def __init__(
        self,
        batch_size: int = 1,
        confidence_threshold: float = 0.7,
        offload_model_to_cpu: bool = True,
    ) -> None:
        """
        Initialize the RetinaFace model for face detection and blurring.

        Args:
            checkpoint: Path to the RetinaFace checkpoint file
            batch_size: Batch size for RetinaFace inference and processing
            confidence_threshold: Minimum confidence score to consider a face detection
            offload_model_to_cpu (bool, optional): Whether to offload the model to CPU. Defaults to True.
        """
        self.checkpoint = f"{GUARDRAIL1_CHECKPOINT.download()}/face_blur_filter/Resnet50_Final.pth"
        self.cfg = cfg_re50
        self.batch_size = batch_size
        self.confidence_threshold = confidence_threshold
        self.dtype = torch.float32
        self.offload_model = offload_model_to_cpu

        # Disable loading ResNet pretrained weights
        self.cfg["pretrain"] = False
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.net = RetinaFace(cfg=self.cfg, phase="test")

        # Load from RetinaFace pretrained checkpoint
        if not offload_model_to_cpu:
            self.net = load_model(self.net, self.checkpoint, False)
            self.net.to("cuda", dtype=self.dtype).eval()
            log.debug("Moved face blur filter to GPU")
        else:
            self.net = load_model(self.net, self.checkpoint, True)
            self.net.to("cpu", dtype=self.dtype).eval()
            log.debug("Moved face blur filter to CPU")

    def preprocess_frames(self, frames: np.ndarray) -> torch.Tensor:
        """Preprocess a sequence of frames for face detection.

        Args:
            frames: Input frames

        Returns:
            Preprocessed frames tensor
        """
        with torch.no_grad():
            frames_tensor = torch.from_numpy(frames).to("cuda", dtype=self.dtype)  # Shape: [T, H, W, C]
            frames_tensor = frames_tensor.permute(0, 3, 1, 2)  # Shape: [T, C, H, W]
            frames_tensor = frames_tensor[:, [2, 1, 0], :, :]  # RGB to BGR to match RetinaFace model input
            means = torch.tensor([104.0, 117.0, 123.0], device="cuda", dtype=self.dtype).view(1, 3, 1, 1)
            frames_tensor = frames_tensor - means  # Subtract mean BGR values for each channel
            return frames_tensor

    def blur_detected_faces(
        self,
        frames: np.ndarray,
        batch_loc: torch.Tensor,
        batch_conf: torch.Tensor,
        prior_data: torch.Tensor,
        scale: torch.Tensor,
        min_size: tuple[int] = (20, 20),
    ) -> list[np.ndarray]:
        """Blur detected faces in a batch of frames using RetinaFace predictions.

        Args:
            frames: Input frames
            batch_loc: Batched location predictions
            batch_conf: Batched confidence scores
            prior_data: Prior boxes for the video
            scale: Scale factor for resizing detections
            min_size: Minimum size of a detected face region in pixels

        Returns:
            Processed frames with pixelated faces
        """
        with torch.no_grad():
            batch_boxes = decode_batch(batch_loc, prior_data, self.cfg["variance"])
            batch_boxes = batch_boxes * scale

        blurred_frames = []
        for i, boxes in enumerate(batch_boxes):
            boxes = boxes.detach().cpu().numpy()
            scores = batch_conf[i, :, 1].detach().cpu().numpy()

            filtered_boxes = filter_detected_boxes(
                boxes,
                scores,
                confidence_threshold=self.confidence_threshold,
                nms_threshold=NMS_THRESHOLD,
                top_k=TOP_K,
                keep_top_k=KEEP_TOP_K,
            )

            frame = frames[i]
            for box in filtered_boxes:
                x1, y1, x2, y2 = map(int, box)
                # Ignore bounding boxes smaller than the minimum size
                if x2 - x1 < min_size[0] or y2 - y1 < min_size[1]:
                    continue
                max_h, max_w = frame.shape[:2]
                face_roi = frame[max(y1, 0) : min(y2, max_h), max(x1, 0) : min(x2, max_w)]
                blurred_face = pixelate_face(face_roi)
                frame[max(y1, 0) : min(y2, max_h), max(x1, 0) : min(x2, max_w)] = blurred_face
            blurred_frames.append(frame)

        return blurred_frames

    def postprocess(self, frames: np.ndarray) -> np.ndarray:
        """Blur faces in a sequence of frames.

        Args:
            frames: Input frames

        Returns:
            Processed frames with pixelated faces
        """
        if self.offload_model:
            self.net = self.net.to("cuda")
            log.debug("Move face blur filter to GPU")

        num_frames = len(frames)
        processed_batches = []
        prior_data, scale = None, None

        for i in range(0, num_frames, self.batch_size):
            # Get batch of frames from numpy array (stays on CPU)
            batch_frames = frames[i : i + self.batch_size]

            # Preprocess just this batch on GPU
            batch_tensor = self.preprocess_frames(batch_frames)
            h, w = batch_tensor.shape[-2:]

            with torch.no_grad():
                # Generate priors for the video
                if prior_data is None:
                    priorbox = PriorBox(self.cfg, image_size=(h, w))
                    priors = priorbox.forward()
                    priors = priors.to("cuda", dtype=self.dtype)
                    prior_data = priors.data

                # Get scale for resizing detections
                if scale is None:
                    scale = torch.Tensor([w, h, w, h])
                    scale = scale.to("cuda", dtype=self.dtype)

                batch_loc, batch_conf, _ = self.net(batch_tensor)

            # Blur detected faces in this batch
            processed_batches.append(self.blur_detected_faces(batch_frames, batch_loc, batch_conf, prior_data, scale))

            # Free GPU memory for this batch
            del batch_tensor, batch_loc, batch_conf

        processed_frames = [frame for batch in processed_batches for frame in batch]
        if self.offload_model:
            self.net = self.net.to("cpu")
            log.debug("Offload face blur filter to CPU")
        return np.array(processed_frames)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="Path containing input videos")
    parser.add_argument("--output_dir", type=str, required=True, help="Path for saving processed videos")
    return parser.parse_args()


def main(args):
    filepaths = get_video_filepaths(args.input_dir)
    if not filepaths:
        log.error(f"No video files found in directory: {args.input_dir}")
        return

    face_blur = RetinaFaceFilter()
    postprocessing_runner = GuardrailRunner(postprocessors=[face_blur])
    os.makedirs(args.output_dir, exist_ok=True)

    for filepath in tqdm(filepaths):
        video_data = read_video(filepath)
        with misc.timer("face blur filter"):
            frames = postprocessing_runner.postprocess(video_data.frames)

        output_path = os.path.join(args.output_dir, os.path.basename(filepath))
        save_video(output_path, frames, video_data.fps)


if __name__ == "__main__":
    args = parse_args()
    main(args)
