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

from typing import IO, Any, Dict, Tuple

import imageio
import imageio.v3 as iio_v3
import numpy as np
import torch

from cosmos.utils import log
from cosmos.utils.easy_io.handlers.base import BaseFileHandler


class ImageioVideoHandler(BaseFileHandler):
    str_like = False

    def load_from_fileobj(
        self, file: IO[bytes], format: str = "mp4", mode: str = "rgb", **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Load video from a file-like object using imageio.v3 with specified format and color mode.

        Parameters:
            file (IO[bytes]): A file-like object containing video data.
            format (str): Format of the video file (default 'mp4').
            mode (str): Color mode of the video, 'rgb' or 'gray' (default 'rgb').

        Returns:
            tuple: A tuple containing an array of video frames and metadata about the video.
        """
        file.seek(0)

        # The plugin argument in v3 replaces the format argument in v2
        plugin = kwargs.pop("plugin", "pyav")

        # Load all frames at once using v3 API
        video_frames = iio_v3.imread(file, plugin=plugin, **kwargs)

        # Handle grayscale conversion if needed
        if mode == "gray":
            import cv2

            if len(video_frames.shape) == 4:  # (frames, height, width, channels)
                gray_frames = []
                for frame in video_frames:
                    gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                    gray_frame = np.expand_dims(gray_frame, axis=2)  # Keep dimensions consistent
                    gray_frames.append(gray_frame)
                video_frames = np.array(gray_frames)

        # Extract metadata
        # Note: iio_v3.imread doesn't return metadata directly like v2 did
        # We need to extract it separately
        file.seek(0)
        metadata = self._extract_metadata(file, plugin=plugin)

        return video_frames, metadata

    def _extract_metadata(self, file: IO[bytes], plugin: str = "pyav") -> Dict[str, Any]:
        """
        Extract metadata from a video file.

        Parameters:
            file (IO[bytes]): File-like object containing video data.
            plugin (str): Plugin to use for reading.

        Returns:
            dict: Video metadata.
        """
        try:
            # Create a generator to read frames and metadata
            metadata = iio_v3.immeta(file, plugin=plugin)

            # Add some standard fields similar to v2 metadata format
            if "fps" not in metadata and "duration" in metadata:
                # Read the first frame to get shape information
                file.seek(0)
                first_frame = iio_v3.imread(file, plugin=plugin, index=0)
                metadata["size"] = first_frame.shape[1::-1]  # (width, height)
                metadata["source_size"] = metadata["size"]

                # Create a consistent metadata structure with v2
                metadata["plugin"] = plugin
                if "codec" not in metadata:
                    metadata["codec"] = "unknown"
                if "pix_fmt" not in metadata:
                    metadata["pix_fmt"] = "unknown"

                # Calculate nframes if possible
                if "fps" in metadata and "duration" in metadata:
                    metadata["nframes"] = int(metadata["fps"] * metadata["duration"])
                else:
                    metadata["nframes"] = float("inf")

            return metadata

        except Exception as e:
            # Fallback to basic metadata
            return {
                "plugin": plugin,
                "nframes": float("inf"),
                "codec": "unknown",
                "fps": 30.0,  # Default values
                "duration": 0,
                "size": (0, 0),
            }

    def dump_to_fileobj(
        self,
        obj: np.ndarray | torch.Tensor,
        file: IO[bytes],
        format: str = "mp4",  # pylint: disable=redefined-builtin
        fps: int = 17,
        quality: int = 5,
        ffmpeg_params=None,
        **kwargs,
    ):
        """
        Save an array of video frames to a file-like object using imageio.

        Parameters:
            obj (Union[np.ndarray, torch.Tensor]): An array of frames to be saved as video.
            file (IO[bytes]): A file-like object to which the video data will be written.
            format (str): Format of the video file (default 'mp4').
            fps (int): Frames per second of the output video (default 17).
            quality (int): Quality of the video (0-10, default 5).
            ffmpeg_params (list): Additional parameters to pass to ffmpeg.

        """
        if isinstance(obj, torch.Tensor):
            assert obj.dtype == torch.uint8, "Tensor must be of type uint8"
            obj = obj.cpu().numpy()
        h, w = obj.shape[1:-1]

        # Default ffmpeg params that ensure width and height are set
        default_ffmpeg_params = ["-s", f"{w}x{h}"]

        # Use provided ffmpeg_params if any, otherwise use defaults
        final_ffmpeg_params = ffmpeg_params if ffmpeg_params is not None else default_ffmpeg_params

        mimsave_kwargs = {
            "fps": fps,
            "quality": quality,
            "macro_block_size": 1,
            "ffmpeg_params": final_ffmpeg_params,
            "output_params": ["-f", "mp4"],
        }
        # Update with any other kwargs
        mimsave_kwargs.update(kwargs)
        log.debug(f"mimsave_kwargs: {mimsave_kwargs}")

        imageio.mimsave(file, obj, format, **mimsave_kwargs)

    def dump_to_str(self, obj, **kwargs):
        raise NotImplementedError
