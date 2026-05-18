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

"""Shared helpers for local datasets (S3, video decoding, aspect ratio)."""

import io
import json
import subprocess
import time
from collections.abc import Generator
from pathlib import Path
from typing import Any

import numpy as np
from boto3.s3.transfer import TransferConfig
from botocore.config import Config

from cosmos.utils import log

client_config = Config(
    response_checksum_validation="when_required",
    request_checksum_calculation="when_required",
    connect_timeout=10,
    read_timeout=5,
)
transfer_config = TransferConfig(use_threads=True, max_concurrency=8, multipart_chunksize=8 * 1024 * 1024)


def parse_s3_url(s3_url: str) -> tuple[str, str]:
    s3_url = s3_url.removeprefix("s3://")
    bucket, key = s3_url.split("/", 1)
    return bucket, key


def download_from_s3(s3_client: Any, s3_url: str, max_tries: int = 20) -> bytes | None:
    """Download a file from S3."""
    if not s3_url.startswith("s3://"):
        return Path(s3_url).read_bytes()
    tries = 0
    while True:
        tries += 1
        try:
            bucket, key = parse_s3_url(s3_url)
            buffer = io.BytesIO()
            s3_client.download_fileobj(Bucket=bucket, Key=key, Fileobj=buffer, Config=transfer_config)
            data = buffer.getvalue()
            return data
        except Exception as e:
            log.error(f"Error downloading from S3 (try {tries}): {e}\n{s3_url}")
        if tries >= max_tries:
            return None
        time.sleep(1)


def get_video_metadata(video_path: str) -> dict:
    """
    Get video metadata using ffprobe.

    Args:
        video_path: Path to the video file

    Returns:
        Dictionary containing width, height, fps, and total_frames
    """
    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_streams",
        "-select_streams",
        "v:0",
        video_path,
    ]
    result = subprocess.run(cmd, stdin=subprocess.DEVNULL, capture_output=True, check=True, text=True)
    probe_data = json.loads(result.stdout)

    # Decode output
    stream = probe_data["streams"][0]
    width = int(stream["width"])
    height = int(stream["height"])
    fps_parts = stream["r_frame_rate"].split("/")
    video_fps = float(fps_parts[0]) / float(fps_parts[1])
    if "nb_frames" in stream:
        total_frames = int(stream["nb_frames"])
    else:
        duration = float(stream.get("duration") or 0)
        total_frames = int(duration * video_fps)

    return dict(width=width, height=height, fps=video_fps, total_frames=total_frames)


def ffmpeg_decode_video(
    video_path: str, scale_hw: tuple[int, int] | None = None, num_threads: int = 1
) -> Generator[np.ndarray, None, None]:
    """
    Decode video frames using ffmpeg and yield HWC uint8 RGB frames.

    Args:
        video_path: Path to the video file
        scale_hw: Tuple of width and height to scale the video to (default: None)

    Yields:
        np.ndarray: HWC uint8 RGB frames
    """
    if scale_hw is None:
        metadata = get_video_metadata(video_path)
        out_width = metadata["width"]
        out_height = metadata["height"]
    else:
        out_height, out_width = scale_hw

    # Calculate frame size in bytes
    frame_size = out_width * out_height * 3  # 3 channels (RGB)

    # Build ffmpeg command to decode and output raw RGB frames
    ffmpeg_cmd = [
        "ffmpeg",
        "-loglevel",
        "quiet",
        "-threads",
        str(num_threads),
        "-filter_threads",
        str(num_threads),
        "-filter_complex_threads",
        str(num_threads),
        "-i",
        video_path,
        "-threads",
        str(num_threads),
        "-filter_threads",
        str(num_threads),
        "-filter_complex_threads",
        str(num_threads),
        "-pix_fmt",
        "rgb24",
        "-sws_flags",
        "bicubic+accurate_rnd",  # lanczos too much ringing on graphics
        *(["-vf", f"scale={scale_hw[1]}:{scale_hw[0]}"] if scale_hw else []),  # WH
        "-f",
        "rawvideo",
        "-vsync",
        "0",
        "-",
    ]

    process = subprocess.Popen(
        ffmpeg_cmd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,  # Set to None to print errors
        bufsize=-1,
    )

    try:
        while True:
            raw_frame = process.stdout.read(frame_size)

            if len(raw_frame) != frame_size:
                assert len(raw_frame) == 0, f"Incomplete frame: {len(raw_frame)} bytes"
                break

            frame = np.frombuffer(raw_frame, dtype=np.uint8)
            frame = frame.reshape((out_height, out_width, 3))

            yield frame
    finally:
        process.stdout.close()
        process.wait()


def get_aspect_ratio(width: int, height: int) -> str:
    """Compute aspect ratio bucket from width and height."""
    ratio = width / height

    if ratio < 0.65:
        return "9,16"  # 0.5625
    elif ratio < 0.88:
        return "3,4"  # 0.75
    elif ratio < 1.16:
        return "1,1"  # 1.0
    elif ratio < 1.55:
        return "4,3"  # 1.3333
    else:
        return "16,9"  # 1.7778


def save_video_frames_to_mp4(
    frames: np.ndarray | Any,
    output_path: str,
    fps: float = 24.0,
    overlay_frame_id: bool = False,
    fps_to_show: float | None = None,
) -> None:
    """Encode video frames to MP4 using FFmpeg.

    Args:
        frames: Video frames as numpy (T, H, W, 3) or torch tensor (C, T, H, W), uint8.
        output_path: Path for the output .mp4 file.
        fps: Output video frame rate.
        overlay_frame_id: If True, draw frame index (0, 1, ...) on each frame via FFmpeg drawtext.
        fps_to_show: If provided, draw the FPS value on the video instead of the actual FPS.
    """
    cpu_fn = getattr(frames, "cpu", None)
    if callable(cpu_fn):
        frames = cpu_fn().numpy()  # type: ignore[union-attr]
    frames = np.asarray(frames, dtype=np.uint8)
    if frames.ndim == 4 and frames.shape[0] == 3:
        # CTHW -> THWC
        frames = np.transpose(frames, (1, 2, 3, 0))
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError("frames must be (T, H, W, 3) or (C, T, H, W) uint8")
    t, h, w, _ = frames.shape
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{w}x{h}",
        "-r",
        str(fps),
        "-i",
        "pipe:0",
    ]
    if overlay_frame_id:
        # %{n} = frame index (0-based); add fps and resolution as literal text
        drawtext_frame = "drawtext=text='%{n}':x=10:y=10:fontsize=24:fontcolor=white:box=1:boxcolor=black@0.6"
        drawtext_fps = (
            f"drawtext=text='fps: {fps_to_show or fps}':x=10:y=40:fontsize=24:fontcolor=white:box=1:boxcolor=black@0.6"
        )
        drawtext_res = f"drawtext=text='{w}x{h}':x=10:y=70:fontsize=24:fontcolor=white:box=1:boxcolor=black@0.6"
        cmd += ["-vf", ",".join([drawtext_frame, drawtext_fps, drawtext_res])]
    cmd += [
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        output_path,
    ]
    process = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    _, stderr = process.communicate(input=frames.tobytes())
    if process.returncode != 0:
        log.error(f"FFmpeg failed: {stderr.decode()}")
        raise RuntimeError(f"FFmpeg exited with {process.returncode}")
