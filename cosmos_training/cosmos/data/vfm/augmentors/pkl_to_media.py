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

"""Augmentors for handling video loading from pickled bytes."""

import io
import pickle as pkl
import random
import re
from typing import Dict, Optional

import torch
from PIL import Image, UnidentifiedImageError
from qwen_vl_utils.vision_process import smart_nframes, smart_resize
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from cosmos.data.imaginaire.webdataset.augmentors.augmentor import Augmentor
from cosmos.utils import log
from cosmos.utils.vfm.video_preprocess import tensor_to_pil_images

Image.MAX_IMAGE_PIXELS = 933120000
_VIDEO_EXTENSIONS = "mp4 avi webm mov".split()

VIDEO_DECODER_OPTIONS = {}


def token_to_pixels(token_length: int, patch_size: int = 14, temporal_patch_size: int = 2) -> int:
    """Convert token length to pixels based on patch size and temporal patch size."""

    merged_patch_size = patch_size * 2
    return token_length * merged_patch_size**2 * temporal_patch_size


def pixels_to_token(pixels: int, patch_size: int = 14, temporal_patch_size: int = 2) -> int:
    """Convert pixels to token length based on patch size and temporal patch size."""

    merged_patch_size = patch_size * 2
    return pixels // merged_patch_size**2 // temporal_patch_size


def _video_decoder_qwen_func(
    key: str,
    data: bytes,
    min_fps_thres: int = 4,
    max_fps_thres: int = 60,
    target_fps: float = 2.0,
    min_video_token_length: int = 16,
    max_video_token_length: int = 8192,
    num_threads: int = 0,
    random_augmentation: bool = False,
    fps_random_range: list[float] = [0.5, 1.5],
    max_video_token_length_random_range: list[float] = [0.75, 1.25],
    frame_count_random_range: Optional[list[int]] = None,
    start_frame: Optional[int] = None,
    end_frame: Optional[int] = None,
    **kwargs,
) -> dict | None:
    """Actual video decoder function.

    Args:
        key (str): Video file name/key
        data (bytes): Video binary data
        min_fps_thres (int, optional): Minimum FPS threshold. Defaults to 4.
        max_fps_thres (int, optional): Maximum FPS threshold. Defaults to 60.
        target_fps (float, optional): Target FPS. Defaults to 2.0.
        min_video_token_length (int, optional): Minimum token length. Defaults to 16.
        max_video_token_length (int, optional): Maximum token length. Defaults to 8192.
        num_threads (int, optional): Number of threads for decord. Defaults to 0.
        random_augmentation (bool, optional): Whether to randomize the FPS and max_video_token_length. Defaults to False.
        fps_random_range (list[float], optional): Random FPS range. Defaults to [10.0, 24.0].
        max_video_token_length_random_range (list[float], optional): Random max_video_token_length range. Defaults to [0.75, 1.25].
        frame_count_random_range (list[int], optional): Random frame count range. If provided, take priority over fps_random_range.
        start_frame (Optional[int], optional): Start frame. Defaults to None. If both start_frame and end_frame are provided, the video will be decoded from start_frame to end_frame.
        end_frame (Optional[int], optional): End frame. Defaults to None. If both start_frame and end_frame are provided, the video will be decoded from start_frame to end_frame.

    Raises:
        ValueError: Video fps lower than 1, skipping
        ValueError: Video fps lower than min_fps_thres, skipping
        ValueError: Video fps higher than max_fps_thres, skipping

    Returns:
        dict | None: Dictionary with video frames tensor and target FPS
    """
    import decord

    # Check video extension
    extension = re.sub(r".*[.]", "", key)
    if extension.lower() not in _VIDEO_EXTENSIONS:
        return None

    # Read video
    video_buffer = io.BytesIO(data)
    video_reader = decord.VideoReader(video_buffer, num_threads=num_threads)
    total_frames, video_fps = len(video_reader), video_reader.get_avg_fps()

    if start_frame is not None and end_frame is not None:
        total_frames = end_frame - start_frame

    if video_fps < 1:
        raise ValueError("Video fps lower than 1, skipping")
    if video_fps < min_fps_thres:
        raise ValueError(f"Video fps {video_fps} lower than {min_fps_thres}, skipping")
    if video_fps > max_fps_thres:
        raise ValueError(f"Video fps {video_fps} higher than {max_fps_thres}, skipping")

    if random_augmentation:
        if frame_count_random_range is not None:
            # Random number of frames
            min_frames_range, max_frames_range = frame_count_random_range
            min_frames_range = min(min_frames_range, total_frames)
            max_frames_range = min(max_frames_range, total_frames)
            target_frames = random.uniform(min_frames_range, max_frames_range)
            target_fps = target_frames / total_frames * video_fps
        else:
            # randomize fps
            target_fps = (
                random.uniform(fps_random_range[0], fps_random_range[1]) * target_fps
                if random.random() < 0.5
                else target_fps
            )
        # randomize max_video_token_length
        max_video_token_length = int(
            random.uniform(max_video_token_length_random_range[0], max_video_token_length_random_range[1])
            * max_video_token_length
        )
        log.debug(f"random_augmentation: max_video_token_length: {max_video_token_length}, target_fps: {target_fps}")

    patch_size = 14
    min_height_width = 56  # https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2_vl/image_processing_qwen2_vl.py#L57
    temporal_patch_size = 2
    min_pixels: int = token_to_pixels(min_video_token_length, patch_size, temporal_patch_size)
    max_pixels: int = token_to_pixels(max_video_token_length, patch_size, temporal_patch_size)
    max_frames: int = max_pixels // (min_height_width) ** 2 // temporal_patch_size

    # sample based on target fps
    nframes = smart_nframes(dict(fps=target_fps), total_frames=total_frames, video_fps=video_fps)
    nframes = min(nframes, max_frames)
    if start_frame is not None and end_frame is not None:
        idx = torch.linspace(start_frame, end_frame - 1, nframes).round().long().tolist()  # [nframes]
    else:
        idx = torch.linspace(0, total_frames - 1, nframes).round().long().tolist()  # [nframes]
    video_frames = video_reader.get_batch(idx).asnumpy()
    video_frames = torch.tensor(video_frames).permute(0, 3, 1, 2)  # [T,C,H,W]
    sample_fps = nframes / max(total_frames, 1e-6) * video_fps

    # recompute max_pixels based on number of sampled frames
    nframes, _, height, width = video_frames.shape
    max_pixels = max_pixels // nframes
    resized_height, resized_width = smart_resize(
        height,
        width,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    video_frames = transforms.functional.resize(
        video_frames,
        [resized_height, resized_width],
        interpolation=InterpolationMode.BICUBIC,
        antialias=True,
    ).float()
    video_frames = video_frames.permute(1, 0, 2, 3)  # [C,T,H,W]

    # Clean up
    video_reader.seek(0)  # set video reader point back to 0 to clean up cache
    del video_reader  # delete the reader to avoid memory leak

    return dict(videos=video_frames, fps=sample_fps)


class PKLToMedia(Augmentor):
    """
    Converts PKL bytes stored in a data dictionary into media.

    Handles input formats for the specified input key:
        A dictionary mapping media names (str) to bytes objects.

    The output format is a dictionary mapping names to their respective decoded objects:
    Input dict[str, bytes] -> Output dict[str, torch.Tensor | PIL.Image]

    Corrupted or non-decodable bytes are skipped with a warning.
    """

    def __init__(
        self,
        input_key: str = "media",
        output_key: str = "media",
        min_fps_thres: int = 4,
        max_fps_thres: int = 60,
        target_fps: float = 4.0,
        min_video_token_length: int = 16,
        max_video_token_length: int = 8192,
        num_threads: int = 0,
        random_augmentation: bool = False,
        is_input_in_dict: bool = False,
        use_start_frame_end_frame: bool = False,
        frame_count_random_range: Optional[list[int]] = None,
    ) -> None:
        """
        Args:
            input_key (str): Key in the data_dict containing video/image data.
            output_key (str): Key to store the resulting video frame tensors or PIL images.
            min_fps_thres (int): Minimum FPS threshold for video decoding.
            max_fps_thres (int): Maximum FPS threshold for video decoding.
            target_fps (float): Target FPS for video decoding.
            min_video_token_length (int): Minimum token length for video decoding.
            max_video_token_length (int): Maximum token length for video decoding.
            num_threads (int): Number of threads for video decoding.
            random_augmentation (bool): Whether to apply random augmentation during decoding.
            is_input_in_dict (bool): Whether the input key is in the data_dict instead of pkl files. (For cosmos predict2 videos)
            use_start_frame_end_frame (bool): Whether to use start_frame and end_frame to decode the video. (For cosmos predict2 videos)
            frame_count_random_range (list[int], optional): Random frame count range. Defaults to None.
        """
        self.input_key = input_key
        self.output_key = output_key
        self.video_decoder_params = {
            "min_fps_thres": min_fps_thres,
            "max_fps_thres": max_fps_thres,
            "target_fps": target_fps,
            "min_video_token_length": min_video_token_length,
            "max_video_token_length": max_video_token_length,
            "num_threads": num_threads,
            "random_augmentation": random_augmentation,
            "frame_count_random_range": frame_count_random_range,
        }
        self.is_input_in_dict = is_input_in_dict
        self.use_start_frame_end_frame = use_start_frame_end_frame

    def _bytes_to_video_frames(self, video_bytes: bytes, identifier: str = "video") -> Optional[Dict]:
        """Converts video bytes to video frame tensors using the video decoder."""
        try:
            result = _video_decoder_qwen_func(
                key=f"{identifier}.mp4",  # Add .mp4 extension for the decoder
                data=video_bytes,
                **self.video_decoder_params,
            )
            result["videos"] = tensor_to_pil_images(result["videos"])  # 3,T,H,W -> list of PIL images
            if result is not None:
                return result
            else:
                log.warning(f"Skipping item '{identifier}': Video decoder returned None.")
                return None
        except Exception as e:
            log.warning(f"Skipping item '{identifier}': Error decoding video bytes: {e}")
            return None

    def _bytes_to_pil(self, image_bytes: bytes, identifier: str = "image") -> Optional[Image.Image]:
        """Converts a single bytes object to a PIL Image."""
        try:
            with io.BytesIO(image_bytes) as stream:
                img = Image.open(stream)
                img.load()  # Verify the image data
                return img.convert("RGB")  # Convert to standard RGB format
        except UnidentifiedImageError:
            log.warning(f"Skipping item '{identifier}': Cannot identify image file from bytes.")
        except Exception as e:
            log.warning(f"Skipping item '{identifier}': Error decoding image bytes: {e}")
        return None

    def __call__(self, data_dict: Dict) -> Dict:
        """
        Processes the data_dict to convert video/image bytes to their respective formats.

        Args:
            data_dict (Dict): The input data dictionary.

        Returns:
            Dict: The modified data dictionary with video frame tensors and/or PIL images.
        """
        input_key = self.input_key
        output_key = self.output_key

        if input_key not in data_dict:
            log.debug(
                f"Input key '{input_key}' not found in data_dict. Skipping PKLToMedia. Available keys: {data_dict.keys()}"
            )
            return data_dict

        if not self.is_input_in_dict:
            data = pkl.loads(data_dict[input_key])
        else:
            data = data_dict[input_key]

        output_data = {}

        if isinstance(data, dict):
            for name, item in data.items():
                if isinstance(item, bytes):
                    # Determine if this is video or image based on the key name
                    if "video" in name.lower():
                        # Decode as video
                        result = self._bytes_to_video_frames(item, identifier=f"{input_key}['{name}']")
                        if result:
                            output_data[name] = result
                    elif "image" in name.lower():
                        # Decode as image
                        result = self._bytes_to_pil(item, identifier=f"{input_key}['{name}']")
                        if result:
                            output_data[name] = result
                    else:
                        log.warning(
                            f"Skipping item with key '{name}' in '{input_key}': Key does not contain 'video' or 'image'."
                        )
                else:
                    log.warning(f"Skipping item with key '{name}' in '{input_key}': Expected bytes, got {type(item)}.")
        else:
            raise ValueError(
                f"Input key '{input_key}' has unsupported type {type(data)}. "
                f"Expected dict[str, bytes] for video/image data."
            )

        # Add the processed data and optionally remove the input key
        data_dict[output_key] = output_data
        if input_key != output_key:
            del data_dict[input_key]

        return data_dict
