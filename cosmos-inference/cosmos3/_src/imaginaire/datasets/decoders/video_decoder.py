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

import io
import math
import re
from random import randint
from typing import Callable, List, Tuple

import numpy as np
import torch
from PIL import Image

from cosmos3._src.imaginaire.utils import log

Image.MAX_IMAGE_PIXELS = 933120000
_VIDEO_EXTENSIONS = "mp4 avi webm mov".split()

VIDEO_DECODER_OPTIONS = {}


def video_decoder_register(key):
    def decorator(func):
        VIDEO_DECODER_OPTIONS[key] = func
        return func

    return decorator


@video_decoder_register("video_decoder_metadata")
def video_decoder_metadata(num_threads, **kwargs):
    """
    Video decoder using the video's native fps
    """
    import decord

    def video_decoder(key: str, data: bytes):
        extension = re.sub(r".*[.]", "", key)
        if extension.lower() not in _VIDEO_EXTENSIONS:
            return None
        video_buffer = io.BytesIO(data)
        reader = decord.VideoReader(video_buffer, num_threads=num_threads)
        num_frames = len(reader)
        video_fps = int(np.round(reader.get_avg_fps()))
        length_in_s = float(num_frames) / float(video_fps)
        bitrate = video_buffer.getbuffer().nbytes * 8 / length_in_s
        video_frames = reader.get_batch([0]).asnumpy()
        video_frames = torch.from_numpy(video_frames).permute(3, 0, 1, 2)  # (T, H, W, C) -> (C, T, H, W)
        return video_frames, {"fps": video_fps, "num_frames": num_frames, "bitrate": bitrate}

    return video_decoder


@video_decoder_register("video_decoder_w_controlled_fps")
def video_decoder_w_controlled_fps(
    sequence_length: int = 34,
    chunk_size: int = 0,
    use_fps_control: bool = False,
    min_fps_thres: int = 4,
    max_fps_thres: int = 30,
    sampling_reweighting: bool = False,
    sampling_reweighting_factor: int = 1,
    num_threads=4,
    limit_fps_range: bool = False,
    save_raw: bool = False,
):
    """
    Video decoder using with fps control.
    This function samples videos with fps in the range [min_fps_thres, max_fps_thres].
    We adjust the fps range if min and max fps cannot be supported to get the sequence length with desired chunk size.

    Parameters:
    - sequence_length (int) : Number of frames returned by the function
    - chunk_size (int): How the video is divided into chunks. Only return frames within a chunk. chunk_size=0 means we use full video length. Defaults to 0.
    - min_fps_thres (int): Minimum fps threshold to sample from.
    - max_fps_thres (int): Maximum fps threshold to sample from.
    - sampling_reweighting (bool): If False, sample fps weights uniformly. If True, reweight sampling distrubution.
    - sampling_reweighting_factor (int): The fps sampling distribution reweighting factor. If sampling_reweighting_factor  > 1, sample more on lower fps side.
    - num_thread (int): Number of threads for decord.
    - save_raw (bool): If True, will also return entire raw video in data_dict key "video_raw_bytes", alongside with the video frames. Only enable this for visualization and debug.
    """
    import decord

    def video_decoder(
        key: str,
        data: bytes,
    ):
        extension = re.sub(r".*[.]", "", key)
        if extension.lower() not in _VIDEO_EXTENSIONS:
            return None

        video_buffer = io.BytesIO(data)
        video_reader = decord.VideoReader(video_buffer, num_threads=num_threads)
        num_target_frames = sequence_length if sequence_length > 0 else len(video_reader)
        num_orig_frames = len(video_reader)

        # Obtain the number of chunks
        if chunk_size == 0:
            curr_chunk_size = num_orig_frames
        else:
            curr_chunk_size = chunk_size
        num_chunks = max(num_orig_frames // curr_chunk_size, 1)

        # Checks to ensure that number of target frames we need is present in the video / chunk.
        if num_target_frames > curr_chunk_size:
            raise ValueError(
                f"Specified sequence_length {num_target_frames} exceeds curr_chunk_size {curr_chunk_size}, num_orig_frames={num_orig_frames}, chunk_size={chunk_size}"
            )

        if num_target_frames > num_orig_frames:
            raise ValueError(
                f"Specified sequence_length {num_target_frames} exceeds num frames in video {num_orig_frames}."
            )

        # Now obtain min and max fps that we can use within this chunk
        video_fps = int(np.round(video_reader.get_avg_fps()))

        if video_fps < 1:
            raise ValueError("Video fps lower than 1, skipping")
        if limit_fps_range:
            if video_fps < min_fps_thres:
                raise ValueError(f"Video fps {video_fps} lower than {min_fps_thres}, skipping")
            if video_fps > max_fps_thres:
                raise ValueError(f"Video fps {video_fps} larger than {max_fps_thres}, skipping")

        # Check if the last chunk has separate window
        # This happens only if remainder frames >= curr_chunk_size / 2 [data annotation was done this way]
        # Else this is used as a part of previous window.
        num_frames_in_last_chunk = num_orig_frames - num_chunks * curr_chunk_size
        if num_frames_in_last_chunk >= int(0.5 * curr_chunk_size):
            if num_frames_in_last_chunk > num_target_frames:
                num_chunks += 1

        # Sample which chunk to use
        chunk_index = randint(0, num_chunks - 1)

        if chunk_index == num_chunks - 1:
            # For the last chunk, use all of the remaining frames
            num_samples_in_chunk = num_orig_frames - chunk_index * curr_chunk_size
        else:
            # Else use only the chunk size
            num_samples_in_chunk = curr_chunk_size

        if use_fps_control:
            # When fps control is provided, sample random fps.
            min_fps = max(min_fps_thres, math.ceil(video_fps * float(num_target_frames) / float(num_samples_in_chunk)))
            max_fps = min(max_fps_thres, video_fps)

            # Randomly sample a target fps in the range of (min_fps, max_fps)
            if max_fps > min_fps:
                fps_selections = list(range(min_fps, max_fps + 1))

                # Sample reweighting favors the smaller fps more
                if sampling_reweighting:
                    dist = [1 / (float(pp) ** sampling_reweighting_factor) for pp in fps_selections]
                    target_fps = np.random.choice(fps_selections, 1, p=[pp / sum(dist) for pp in dist])
                else:
                    target_fps = np.random.choice(fps_selections, 1)
            else:
                target_fps = max_fps

        else:
            # If not, use native fps
            target_fps = video_fps

        # stride used for subsampling video
        stride = int(video_fps / target_fps)

        # This is the actual target fps we obtain after subsampling
        target_fps = video_fps / stride

        # Select the frame start index and frame end index
        chunk_frame_start = chunk_index * curr_chunk_size
        if num_samples_in_chunk <= num_target_frames * stride:
            raise ValueError(
                f"Decoded video not long enough, num_samples_in_chunk={num_samples_in_chunk}, num_target_frames={num_target_frames}, stride={stride}, video_fps={video_fps}, target_fps={target_fps}, min_fps_thres={min_fps_thres}, max_fps_thres={max_fps_thres}, use_fps_control={use_fps_control}"
            )
        # Start index is randomly selected in the chunk
        frame_start = chunk_frame_start + int(
            np.random.choice(num_samples_in_chunk - int(num_target_frames * stride), 1)
        )
        frame_end = frame_start + num_target_frames * stride

        # Subsample the frames
        if "depth" in key:
            frame_start = video_decoder.frame_start
            frame_end = video_decoder.frame_end
            stride = video_decoder.stride
            chunk_index = video_decoder.chunk_index
        else:
            video_decoder.frame_start = frame_start
            video_decoder.frame_end = frame_end
            video_decoder.stride = stride
            video_decoder.chunk_index = chunk_index
        video_frames = video_reader.get_batch(np.arange(frame_start, frame_end, stride).tolist()).asnumpy()

        # Return the frames and metadata
        if num_target_frames is not None and video_frames.shape[0] < num_target_frames:
            raise ValueError("Decoded video not long enough, skipping")
        video_frames = torch.from_numpy(video_frames).permute(3, 0, 1, 2)  # (T, H, W, C) -> (C, T, H, W)
        video_reader.seek(0)  # set video reader point back to 0 to clean up cache
        del video_reader  # delete the reader to avoid memory leak

        ret_dict = {
            "video": video_frames,
            "fps": float(target_fps),
            "num_frames": video_frames.shape[1],
            "chunk_index": chunk_index,
            "frame_start": frame_start,
            "frame_end": frame_end,
            "stride": stride,
            "orig_num_frames": num_orig_frames,
        }
        if save_raw:
            ret_dict["video_raw_bytes"] = data
        return ret_dict

    return video_decoder


@video_decoder_register("video_decoder_for_kd_dataset")
def video_decoder_for_kd_dataset(
    sequence_length: int = 34,
    num_threads: int = 4,
    save_raw: bool = False,
    **kwargs,
):
    """
    Video decoder for Knowledge Distillation dataset.
    This function reads in the raw video frames, without any fps control.

    Parameters:
    - sequence_length (int) : Number of frames returned by the function
    - num_thread (int): Number of threads for decord.
    - save_raw (bool): If True, will also return entire raw video in data_dict key "video_raw_bytes", alongside with the video frames. Only enable this for visualization and debug.
    """
    import decord

    def video_decoder(
        key: str,
        data: bytes,
    ):
        extension = re.sub(r".*[.]", "", key)
        if extension.lower() not in _VIDEO_EXTENSIONS:
            return None

        video_buffer = io.BytesIO(data)
        video_reader = decord.VideoReader(video_buffer, num_threads=num_threads)
        num_target_frames = sequence_length if sequence_length > 0 else len(video_reader)
        num_orig_frames = len(video_reader)
        assert num_target_frames == num_orig_frames, (
            "Number of target frames must be equal to the number of original frames"
        )

        # Now obtain min and max fps that we can use within this chunk
        video_fps = int(np.round(video_reader.get_avg_fps()))
        assert video_fps == 24, "Generated video FPS should be 24"

        # Sample which chunk to use
        chunk_index = 0
        frame_start = 0
        stride = 1
        frame_end = frame_start + num_target_frames * stride
        video_frames = video_reader.get_batch(np.arange(frame_start, frame_end, stride).tolist()).asnumpy()

        # Return the frames and metadata
        if num_target_frames is not None and video_frames.shape[0] < num_target_frames:
            raise ValueError("Decoded video not long enough, skipping")
        video_frames = torch.from_numpy(video_frames).permute(3, 0, 1, 2)  # (T, H, W, C) -> (C, T, H, W)
        video_reader.seek(0)  # set video reader point back to 0 to clean up cache
        del video_reader  # delete the reader to avoid memory leak

        ret_dict = {
            "video": video_frames,
            "fps": float(video_fps),
            "num_frames": video_frames.shape[1],
            "chunk_index": chunk_index,
            "frame_start": frame_start,
            "frame_end": frame_end,
            "stride": stride,
            "orig_num_frames": num_orig_frames,
        }
        if save_raw:
            ret_dict["video_raw_bytes"] = data
        return ret_dict

    return video_decoder


@video_decoder_register("video_decoder_basic")
def video_decoder_basic(
    sequence_length: int = 25,
    use_fps_control: bool = False,
    min_fps_thres: int = 4,
    max_fps_thres: int = 30,
    num_threads=4,
    **kwargs,
) -> Callable[[str, bytes], dict[str, torch.Tensor | int]]:
    """Basic video decoder for a specified sequence length.

    If loaded video has fewer frames than requested, temporally pads with the last frame.
    Optionally, allows subsampling video with a variable FPS in [`min_fps_thres` .. `max_fps_thres`].

    Args:
        sequence_length (int) : The number of frames to sample from the loaded video.
        use_fps_control (bool) : Controls whether to temporally subsample.
        min_fps_thres (int): Minimum FPS threshold to sample from.
        max_fps_thres (int): Maximum FPS threshold to sample from.
        num_thread (int): Number of threads for the decord.

    Returns:
        Returns a callable that returns a dictionary of:
            - The sampled video(torch.Tensor, torch.uint8), layout (C, T, H, W).
            - The FPS (int) of the sample.
    """
    import decord

    def video_decoder(
        key: str,
        data: bytes,
    ) -> dict[str, torch.Tensor | int]:
        extension = re.sub(r".*[.]", "", key)
        if extension.lower() not in _VIDEO_EXTENSIONS:
            return None

        video_buffer = io.BytesIO(data)
        video_reader = decord.VideoReader(video_buffer, num_threads=num_threads)

        # video and request metadata.
        num_target_frames = sequence_length if sequence_length > 0 else len(video_reader)
        num_orig_frames = len(video_reader)
        assert num_orig_frames > 0, "Video has no frames."
        video_fps = max(1, int(video_reader.get_avg_fps() + 0.5))

        if use_fps_control:
            # When fps control is provided, sample random fps.
            min_fps = max(min_fps_thres, math.ceil(video_fps * float(num_target_frames) / float(num_orig_frames)))
            max_fps = min(max_fps_thres, video_fps)

            # If frame range is valid, sample random fps in the range of (min_fps, max_fps)
            if max_fps > min_fps:
                fps_selections = list(range(min_fps, max_fps + 1))
                target_fps = np.random.choice(fps_selections, 1)
            else:
                target_fps = max_fps
        else:
            target_fps = video_fps

        # This is the actual target fps we obtain after subsampling.
        stride = int(video_fps / target_fps)
        target_fps = video_fps / stride
        num_target_stride_frames = int(num_target_frames * stride)

        # Start index is randomly selected in the
        valid_length = max(num_orig_frames - num_target_stride_frames, 1)
        frame_start = np.random.choice(valid_length, 1)
        frame_end = min(frame_start + num_target_stride_frames, num_orig_frames)
        frame_indices = np.arange(frame_start, frame_end, stride).tolist()

        # Grab the frames.
        video_frames = video_reader.get_batch(frame_indices).asnumpy()

        # If sampled frames are less than requested, pad with the last frame via replication
        if video_frames.shape[0] < num_target_frames:
            pad_size = num_target_frames - video_frames.shape[0]
            video_frames = np.pad(video_frames, ((0, pad_size), (0, 0), (0, 0), (0, 0)), mode="edge")

        video_frames = torch.from_numpy(video_frames).permute(3, 0, 1, 2)  # (T, H, W, C) -> (C, T, H, W)
        video_reader.seek(0)  # set video reader point back to 0 to clean up cache
        del video_reader  # delete the reader to avoid memory leak
        return {
            "video": video_frames,
            "fps": float(target_fps),
        }

    return video_decoder


@video_decoder_register("video_decoder_still_padding")
def video_decoder_still_padding(
    sequence_length: int = 25,
    use_fps_control: bool = False,
    min_fps_thres: int = 4,
    max_fps_thres: int = 30,
    num_threads=4,
    sampling_reweighting: bool = False,
    sampling_reweighting_factor: int = 1,
    limit_fps_range: bool = False,
    **kwargs,
) -> Callable[[str, bytes], dict[str, torch.Tensor | int]]:
    """Video decoder for a specified sequence length.

    If loaded video has fewer frames than requested, temporally pads with the last frame.
    Optionally, allows subsampling video with a variable FPS in [`min_fps_thres` .. `max_fps_thres`].

    Args:
        sequence_length (int) : The number of frames to sample from the loaded video.
        use_fps_control (bool) : Controls whether to temporally subsample.
        min_fps_thres (int): Minimum FPS threshold to sample from.
        max_fps_thres (int): Maximum FPS threshold to sample from.
        num_thread (int): Number of threads for the decord.

    Returns:
        Returns a callable that returns a dictionary of:
            - The sampled video(torch.Tensor, torch.uint8), layout (C, T, H, W).
            - number of video frames
            - frame_start
            - frame_end
    """
    import decord

    def video_decoder(
        key: str,
        data: bytes,
    ) -> dict[str, torch.Tensor | int]:
        extension = re.sub(r".*[.]", "", key)
        if extension.lower() not in _VIDEO_EXTENSIONS:
            return None

        video_buffer = io.BytesIO(data)
        video_reader = decord.VideoReader(video_buffer, num_threads=num_threads)

        # video and request metadata.
        num_target_frames = sequence_length if sequence_length > 0 else len(video_reader)
        num_orig_frames = len(video_reader)
        assert num_orig_frames > 0, "Video has no frames."

        if num_target_frames > num_orig_frames:
            log.warning(
                f"Specified sequence_length {num_target_frames} exceeds num frames in video {num_orig_frames}. Padding last frame"
            )
            # Grab the frames.
            video_frames = video_reader.get_batch(range(num_orig_frames)).asnumpy()

            # Pad with the last frame via replication
            pad_size = num_target_frames - video_frames.shape[0]
            video_frames = np.pad(video_frames, ((0, pad_size), (0, 0), (0, 0), (0, 0)), mode="edge")

            video_frames = torch.from_numpy(video_frames).permute(3, 0, 1, 2)  # (T, H, W, C) -> (C, T, H, W)
            video_reader.seek(0)  # set video reader point back to 0 to clean up cache
            del video_reader  # delete the reader to avoid memory leak
            return {
                "video": video_frames,
                "frame_start": 0,
                "frame_end": num_orig_frames,
                "num_frames": video_frames.shape[1],
            }

        video_fps = max(1, int(video_reader.get_avg_fps() + 0.5))

        if video_fps < 1:
            raise ValueError("Video fps lower than 1, skipping")
        if limit_fps_range:
            if video_fps < min_fps_thres:
                raise ValueError(f"Video fps {video_fps} lower than {min_fps_thres}, skipping")
            if video_fps > max_fps_thres:
                raise ValueError(f"Video fps {video_fps} larger than {max_fps_thres}, skipping")

        if use_fps_control:
            # When fps control is provided, sample random fps.
            min_fps = max(min_fps_thres, math.ceil(video_fps * float(num_target_frames) / float(num_orig_frames)))
            max_fps = min(max_fps_thres, video_fps)

            # If frame range is valid, sample random fps in the range of (min_fps, max_fps)
            if max_fps > min_fps:
                fps_selections = list(range(min_fps, max_fps + 1))

                # Sample reweighting favors the smaller fps more
                if sampling_reweighting:
                    dist = [1 / (float(pp) ** sampling_reweighting_factor) for pp in fps_selections]
                    target_fps = np.random.choice(fps_selections, 1, p=[pp / sum(dist) for pp in dist])
                else:
                    target_fps = np.random.choice(fps_selections, 1)
            else:
                target_fps = max_fps
        else:
            target_fps = video_fps

        # This is the actual target fps we obtain after subsampling.
        stride = int(video_fps / target_fps)
        target_fps = video_fps / stride
        num_target_stride_frames = int(num_target_frames * stride)

        # Start index is randomly selected in the
        valid_length = max(num_orig_frames - num_target_stride_frames, 1)
        frame_start = np.random.choice(valid_length, 1)
        frame_end = min(frame_start + num_target_stride_frames, num_orig_frames)
        frame_indices = np.arange(frame_start, frame_end, stride).tolist()

        # Grab the frames.
        video_frames = video_reader.get_batch(frame_indices).asnumpy()

        # If sampled frames are less than requested, pad with the last frame via replication
        if video_frames.shape[0] < num_target_frames:
            pad_size = num_target_frames - video_frames.shape[0]
            video_frames = np.pad(video_frames, ((0, pad_size), (0, 0), (0, 0), (0, 0)), mode="edge")

        video_frames = torch.from_numpy(video_frames).permute(3, 0, 1, 2)  # (T, H, W, C) -> (C, T, H, W)
        video_reader.seek(0)  # set video reader point back to 0 to clean up cache
        del video_reader  # delete the reader to avoid memory leak
        return {
            "video": video_frames,
            "frame_start": frame_start,
            "frame_end": frame_end,
            "num_frames": video_frames.shape[1],
        }

    return video_decoder


def video_decoder_w_lower_fps_get_indices(
    num_orig_frames: int,
    video_fps: int,
    min_fps_thres: int,
    max_fps_thres: int,
    sequence_length: int,
) -> Tuple[List[int], float]:
    """Generates frame indices for video sampling with FPS control.

    This function determines valid stride lengths for sampling frames from a video,
    preferring lower FPS (larger strides) when multiple options are available.
    It returns both the selected frame indices and the resulting FPS.

    Args:
        num_orig_frames: Total number of frames in the original video.
        video_fps: Original video frames per second.
        min_fps_thres: Minimum allowed frames per second.
        max_fps_thres: Maximum allowed frames per second.
        sequence_length: Number of frames to sample.

    Returns:
        A tuple containing:
            - list[int]: Frame indices to sample from the original video.
            - float: The resulting frames per second after sampling.

    Raises:
        ValueError: If no valid stride options are available given the constraints.
        ValueError: If input parameters are invalid (e.g., negative values).
    """
    # Validate input parameters
    if num_orig_frames <= 0:
        raise ValueError("num_orig_frames must be positive")
    if video_fps <= 0:
        raise ValueError("video_fps must be positive")
    if min_fps_thres <= 0:
        raise ValueError("min_fps_thres must be positive")
    if max_fps_thres < min_fps_thres:
        raise ValueError("max_fps_thres must be greater than or equal to min_fps_thres")
    if sequence_length <= 1:
        raise ValueError("sequence_length must be greater than 1")
    if sequence_length > num_orig_frames:
        raise ValueError("sequence_length cannot be greater than num_orig_frames")

    # Calculate stride range
    min_stride = 1
    max_stride = (num_orig_frames - 1) // (sequence_length - 1)

    valid_strides = []
    for stride in range(min_stride, max_stride + 1):
        # Check if we can get sequence_length frames with this stride
        if (num_orig_frames - stride * (sequence_length - 1)) > 0:
            new_fps = video_fps / stride
            if min_fps_thres <= new_fps <= max_fps_thres:
                valid_strides.append(stride)

    if not valid_strides:
        raise ValueError(
            f"No valid stride options available for the given constraints. "
            f"stride range = [{min_stride}, {max_stride}]; "
            f"original FPS = {video_fps}; "
            f"sequence_length = {sequence_length}; "
            f"min_fps_thres = {min_fps_thres}; "
            f"max_fps_thres = {max_fps_thres}; "
            f"original num_frames = {num_orig_frames}"
        )

    # Select stride with weighted probability
    if len(valid_strides) >= 2:
        stride_choices = valid_strides[-2:]  # Taking last two as they're the largest
        weights = [0.01, 0.99]  # [smaller_stride, larger_stride]
        selected_stride = np.random.choice(stride_choices, p=weights)
    else:
        selected_stride = valid_strides[0]

    # Calculate the maximum valid start index and random start frame
    max_start_idx = num_orig_frames - (sequence_length - 1) * selected_stride
    frame_start = np.random.randint(0, max_start_idx)

    # Generate frame indices
    frame_indices = [frame_start + i * selected_stride for i in range(sequence_length)]
    return frame_indices, video_fps / selected_stride


@video_decoder_register("video_decoder_w_lower_fps")
def video_decoder_w_lower_fps(
    chunk_size: int = 0,
    sequence_length: int = 34,
    min_fps_thres: int = 4,
    max_fps_thres: int = 30,
    num_threads: int = 4,
    return_frame_indices: bool = False,
    **kwargs,
) -> dict:
    """
    Simplified video decoder with FPS control and frame sampling.

    Args:
        key: Video file name/key
        data: Video binary data
        min_fps_thres: Minimum FPS threshold
        max_fps_thres: Maximum FPS threshold
        sequence_length: Number of frames to return
        num_threads: Number of threads for decord
        limit_fps_range: Whether to enforce FPS limits
        return_frame_indices: Whether to return frame indices

    Returns:
        dict with video frames tensor and target FPS
    """
    import decord

    del kwargs  # Unused

    def video_decoder(
        key: str,
        data: bytes,
    ) -> dict[str, torch.Tensor | int]:
        # Check video extension
        extension = re.sub(r".*[.]", "", key)
        if extension.lower() not in _VIDEO_EXTENSIONS:
            return None

        # Read video
        video_buffer = io.BytesIO(data)
        video_reader = decord.VideoReader(video_buffer, num_threads=num_threads)
        num_target_frames = sequence_length if sequence_length > 0 else len(video_reader)

        # Get video metadata
        num_orig_frames = len(video_reader)
        video_fps = int(np.round(video_reader.get_avg_fps()))

        # Basic validations
        # Obtain the number of chunks
        if chunk_size == 0:
            curr_chunk_size = num_orig_frames
        else:
            curr_chunk_size = chunk_size
        num_chunks = max(num_orig_frames // curr_chunk_size, 1)

        # Checks to ensure that number of target frames we need is present in the video / chunk.
        if num_target_frames > curr_chunk_size:
            raise ValueError("Specified sequence_length exceeds curr_chunk_size.")

        if num_target_frames > num_orig_frames:
            raise ValueError(
                f"Specified sequence_length {num_target_frames} exceeds num frames in video {num_orig_frames}."
            )

        if video_fps < 1:
            raise ValueError("Video fps lower than 1, skipping")
        if video_fps < min_fps_thres:
            raise ValueError(f"Video fps {video_fps} lower than {min_fps_thres}, skipping")

        # Check if the last chunk has separate window
        # This happens only if remainder frames >= curr_chunk_size / 2 [data annotation was done this way]
        # Else this is used as a part of previous window.
        num_frames_in_last_chunk = num_orig_frames - num_chunks * curr_chunk_size
        if num_frames_in_last_chunk >= int(0.5 * curr_chunk_size):
            if num_frames_in_last_chunk > num_target_frames:
                num_chunks += 1

        # Sample which chunk to use
        chunk_index = randint(0, num_chunks - 1)

        if chunk_index == num_chunks - 1:
            # For the last chunk, use all of the remaining frames
            num_samples_cur_chunk = num_orig_frames - chunk_index * curr_chunk_size
        else:
            # Else use only the chunk size
            num_samples_cur_chunk = curr_chunk_size
        idx_first_in_cur_chunk = chunk_index * curr_chunk_size

        frame_indices, adjusted_fps = video_decoder_w_lower_fps_get_indices(
            num_orig_frames=num_samples_cur_chunk,
            video_fps=video_fps,
            min_fps_thres=min_fps_thres,
            max_fps_thres=max_fps_thres,
            sequence_length=num_target_frames,
        )
        frame_indices = [idx_first_in_cur_chunk + idx for idx in frame_indices]

        # Sample frames
        video_frames = video_reader.get_batch(frame_indices).asnumpy()
        video_frames = torch.from_numpy(video_frames).permute(3, 0, 1, 2)  # (T, H, W, C) -> (C, T, H, W)

        # Clean up
        video_reader.seek(0)
        del video_reader

        output = {
            "video": video_frames,
            "fps": float(adjusted_fps),
            "orig_fps": video_fps,
            "frame_start": frame_indices[0],
            "frame_end": frame_indices[-1],
            "num_frames": video_frames.shape[1],
            "orig_num_frames": num_orig_frames,
            "chunk_index": chunk_index,
        }
        if return_frame_indices:
            output["frame_indices"] = frame_indices
        return output

    return video_decoder


@video_decoder_register("video_naive_bytes")
def video_naive_bytes(*args, **kwargs):
    """
    do nothing, just return the video bytes
    """
    del args, kwargs

    def video_decoder(
        key: str,
        data: bytes,
    ):
        extension = re.sub(r".*[.]", "", key)
        if extension.lower() not in _VIDEO_EXTENSIONS:
            return None

        return data

    return video_decoder


def construct_video_decoder(
    video_decoder_name: str = "video_decoder_w_controlled_fps",
    sequence_length: int = 34,
    chunk_size: int = 0,
    use_fps_control: bool = False,
    min_fps_thres: int = 4,
    max_fps_thres: int = 24,
    sampling_reweighting: bool = False,
    sampling_reweighting_factor: int = 1,
    num_threads=4,
    limit_fps_range: bool = False,
    # if true, video decoder will additionally save the raw video (alongside with processed frames) to the data_dict
    # set to true for inference/debugging
    save_raw: bool = False,
):
    return VIDEO_DECODER_OPTIONS[video_decoder_name](
        sequence_length=sequence_length,
        chunk_size=chunk_size,
        use_fps_control=use_fps_control,
        min_fps_thres=min_fps_thres,
        max_fps_thres=max_fps_thres,
        sampling_reweighting=sampling_reweighting,
        sampling_reweighting_factor=sampling_reweighting_factor,
        num_threads=num_threads,
        limit_fps_range=limit_fps_range,
        save_raw=save_raw,
    )


def construct_video_decoder_metadata(
    num_threads=4,
):
    return VIDEO_DECODER_OPTIONS["video_decoder_metadata"](num_threads=num_threads)
