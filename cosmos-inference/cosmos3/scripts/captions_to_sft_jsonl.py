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
"""Convert caption_from_video.py output directories into the SFT training JSONL format.

The SFT dataset loader (sft_dataset.py) expects each JSONL line to have:
  uuid, duration, width, height, vision_path, t2w_windows

where t2w_windows is a list of dicts with start_frame, end_frame, and a
caption field.  The default key is "caption", which sft_dataset.py
recognises as a generic fallback.  Videos longer than 61 s are filtered
by the loader, so they are skipped here with a warning.

Usage
-----
    python -m cosmos3.scripts.captions_to_sft_jsonl \
        --captions-dir outputs/captions \
        --videos-dir outputs/videos \
        -o outputs/my_dataset.jsonl

    # With a custom caption key (default: caption):
    python -m cosmos3.scripts.captions_to_sft_jsonl \
        --captions-dir outputs/captions \
        --videos-dir outputs/videos \
        -o outputs/my_dataset.jsonl \
        --caption-key qwen3_235b_dense
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Annotated

import tyro

_MAX_DURATION = 61.0  # seconds; matches hard-coded limit in sft_dataset.py
_MIN_FRAMES = 61  # matches min_frames=61 in get_sft_dataset()
_VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv", ".webm")


def _find_video(videos_dir: Path, name: str) -> Path | None:
    for ext in _VIDEO_EXTENSIONS:
        candidate = videos_dir / f"{name}{ext}"
        if candidate.exists():
            return candidate
    return None


def _get_video_metadata(video_path: Path) -> dict:
    """Return fps, duration, width, height, total_frames via ffprobe."""
    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_streams",
        "-show_format",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {video_path}: {result.stderr}")
    data = json.loads(result.stdout)

    video_stream = next(
        (s for s in data["streams"] if s["codec_type"] == "video"),
        None,
    )
    if video_stream is None:
        raise RuntimeError(f"No video stream found in {video_path}")

    fps_str = video_stream.get("avg_frame_rate", "30/1")
    fps_num, fps_den = map(int, fps_str.split("/"))
    fps = fps_num / max(fps_den, 1)

    duration = float(data["format"]["duration"])
    width = video_stream["width"]
    height = video_stream["height"]

    # nb_frames may be absent; fall back to duration * fps
    total_frames = int(video_stream.get("nb_frames") or round(duration * fps))

    return {
        "fps": fps,
        "duration": duration,
        "width": width,
        "height": height,
        "total_frames": total_frames,
    }


def main(
    captions_dir: Annotated[
        Path, tyro.conf.arg(help="Directory containing per-video caption subdirectories (each with a caption.txt).")
    ],
    videos_dir: Annotated[Path, tyro.conf.arg(help="Directory containing video files named <clip_name>.<ext>.")],
    output: Annotated[Path, tyro.conf.arg(aliases=("-o",), help="Output JSONL path.")],
    caption_key: str = "caption",
) -> None:
    """Build an SFT JSONL from caption.txt files and a videos directory."""
    caption_files = sorted(captions_dir.glob("*/caption.txt"))
    if not caption_files:
        print(f"No caption.txt files found under {captions_dir}", file=sys.stderr)
        sys.exit(1)

    records = []
    skipped = 0

    for caption_path in caption_files:
        name = caption_path.parent.name
        caption = caption_path.read_text().strip()

        if not caption:
            print(f"  SKIP {name}: empty caption.txt")
            skipped += 1
            continue

        video_path = _find_video(videos_dir, name)
        if video_path is None:
            print(f"  SKIP {name}: no video found in {videos_dir} for name '{name}'")
            skipped += 1
            continue

        try:
            meta = _get_video_metadata(video_path)
        except Exception as e:
            print(f"  SKIP {name}: ffprobe error — {e}")
            skipped += 1
            continue

        if meta["duration"] > _MAX_DURATION:
            print(
                f"  SKIP {name}: duration {meta['duration']:.1f}s > {_MAX_DURATION}s "
                "(sft_dataset.py would filter this out)"
            )
            skipped += 1
            continue

        if meta["total_frames"] < _MIN_FRAMES:
            print(
                f"  SKIP {name}: only {meta['total_frames']} frames < {_MIN_FRAMES} "
                "(sft_dataset.py would filter this out)"
            )
            skipped += 1
            continue

        try:
            vision_path = str(video_path.resolve().relative_to(videos_dir.resolve().parent))
        except ValueError:
            vision_path = str(video_path)

        record = {
            "uuid": name,
            "duration": meta["duration"],
            "width": meta["width"],
            "height": meta["height"],
            "vision_path": vision_path,
            "t2w_windows": [
                {
                    "start_frame": 0,
                    "end_frame": meta["total_frames"] - 1,
                    "temporal_interval": 1,
                    caption_key: caption,
                }
            ],
        }
        records.append(record)
        print(f"  OK  {name}: {meta['duration']:.1f}s, {meta['total_frames']} frames, {meta['width']}x{meta['height']}")

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    print(f"\nWrote {len(records)} records → {output}")
    if skipped:
        print(f"Skipped {skipped} videos")
    if not records:
        print("ERROR: No valid records written.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    tyro.cli(main)
