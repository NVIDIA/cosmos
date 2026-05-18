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

"""Augmentors for handling video loading from pickled bytes.
Copied from projects/cosmos/reason1/datasets/augmentors/bytes_to_media.py
Changes:
    1: fully support start frame end frame, s.t. we could remove the  projects/cosmos/reason1/datasets/augmentors/bytes_to_media.py class for predict2 video support
    2: add processor in init, as we need to read the processing config during the video decoding process
"""

import io
import pickle as pkl
from typing import Dict, Optional

from PIL import Image, UnidentifiedImageError

from cosmos.data.imaginaire.webdataset.augmentors.augmentor import Augmentor
from cosmos.utils import log
from cosmos.data.vfm.vlm.video_decoder_qwen import _video_decoder_qwen_func
from cosmos.data.vfm.processors.qwen3vl_processor import Qwen3VLProcessor
from cosmos.utils.vfm.video_preprocess import tensor_to_pil_images


class BytesToMedia(Augmentor):
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
        is_input_pickle_byptes: bool = True,
        use_start_frame_end_frame: bool = False,
        frame_count_random_range: Optional[list[int]] = None,
        processor: Qwen3VLProcessor = None,
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
            is_input_pickle_byptes (bool): Whether the input key is in the data_dict instead of pkl files. (For cosmos predict2 videos)
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
        self.is_input_pickle_byptes = is_input_pickle_byptes
        self.use_start_frame_end_frame = use_start_frame_end_frame
        self.processor = processor

    def _bytes_to_video_frames(
        self, video_bytes: bytes, identifier: str = "video", start_frame: int = None, end_frame: int = None
    ) -> Optional[Dict]:
        """Converts video bytes to video frame tensors using the video decoder."""
        try:
            result = _video_decoder_qwen_func(
                key=f"{identifier}.mp4",  # Add .mp4 extension for the decoder
                data=video_bytes,
                processor=self.processor,
                start_frame=start_frame,
                end_frame=end_frame,
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

    def _perhaps_unpickle_image_bytes(self, image_bytes: bytes) -> bytes:
        """Unpickles the image bytes if it's double-pickled."""
        if image_bytes[:3] == b"\x80\x04\x95":
            nested_data = pkl.loads(image_bytes)
            if isinstance(nested_data, dict) and "image" in nested_data:
                image_bytes = nested_data["image"]
            else:
                image_bytes = nested_data
        return image_bytes

    def _bytes_to_pil(self, image_bytes: bytes, identifier: str = "image") -> Optional[Image.Image]:
        """Converts a single bytes object to a PIL Image."""
        image_bytes = self._perhaps_unpickle_image_bytes(image_bytes)
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
                f"Input key '{input_key}' not found in data_dict. Skipping BytesToMedia. Available keys: {data_dict.keys()}"
            )
            return data_dict

        raw = data_dict[input_key]
        if self.is_input_pickle_byptes:
            if isinstance(raw, bytes):
                # Old webdataset (<1.0.2): .pkl files not auto-decoded, raw bytes arrive here
                data = pkl.loads(raw)
            elif isinstance(raw, dict):
                # New webdataset (>=1.0.2): basichandlers runs as default post-handler,
                # auto-decoding .media.pkl before this augmentor — use directly
                data = raw
            else:
                raise ValueError(f"Input key '{input_key}' has unexpected type {type(raw)}; expected bytes or dict.")
        else:
            data = raw
        output_data = {}

        if isinstance(data, dict):
            for name, item in data.items():
                if isinstance(item, bytes):
                    # Determine if this is video or image based on the key name
                    if ("video" in name.lower() or ".mp4" in name.lower()) and not self.use_start_frame_end_frame:
                        # Decode as video
                        result = self._bytes_to_video_frames(item, identifier=f"{input_key}['{name}']")
                        if result:
                            output_data[name] = result
                    elif ("video" in name.lower() or ".mp4" in name.lower()) and self.use_start_frame_end_frame:
                        assert "start_frame" in data_dict.keys() and "end_frame" in data_dict.keys(), (
                            f"start_frame and end_frame are not in data_dict.keys(): {data_dict.keys()}"
                        )
                        start_frame = data_dict["start_frame"]
                        end_frame = data_dict["end_frame"]
                        result = self._bytes_to_video_frames(
                            item, identifier=f"{input_key}['{name}']", start_frame=start_frame, end_frame=end_frame
                        )
                        if result:
                            output_data[name] = result

                    elif (
                        "image" in name.lower()
                        or ".jpg" in name.lower()
                        or ".jpeg" in name.lower()
                        or ".png" in name.lower()
                    ):
                        # Decode as image
                        result = self._bytes_to_pil(item, identifier=f"{input_key}['{name}']")
                        if result:
                            output_data[name] = result
                    else:
                        log.warning(
                            f"Skipping item with key '{name}' in '{input_key}': Key does not contain 'video', '.mp4', '.jpg', '.jpeg', '.png', or 'image'."
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
