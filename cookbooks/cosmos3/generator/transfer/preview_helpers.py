# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Preview helpers for the transfer cookbook notebook (importable from any preview cell)."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

TRANSFER_CONTROLS = ("edge", "blur", "depth", "seg", "wsm")


def _ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        pass
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    raise RuntimeError(
        "ffmpeg not found. Use the Cosmos Framework venv from notebook §5, or install "
        "imageio-ffmpeg (`pip install imageio-ffmpeg`), or ensure `ffmpeg` is on PATH."
    )


def _transfer_root() -> Path:
    root = os.environ.get("COSMOS3_TRANSFER_ROOT")
    if root:
        return Path(root).resolve()
    here = Path(__file__).resolve().parent
    if (here / "specs").is_dir():
        return here
    for path in [Path.cwd().resolve(), *Path.cwd().resolve().parents]:
        candidate = path / "cookbooks" / "cosmos3" / "generator" / "transfer"
        if (candidate / "specs").is_dir():
            return candidate
    return here


def _specs_dir() -> Path:
    return _transfer_root() / "specs"


def _output_root() -> Path:
    out = os.environ.get("COSMOS3_TRANSFER_OUTPUT_ROOT")
    if out:
        return Path(out).resolve()
    return (_transfer_root() / "outputs" / "notebooks").resolve()


def resolve_spec_path(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (_specs_dir() / p).resolve()


def load_transfer_spec(control: str) -> dict:
    if control not in TRANSFER_CONTROLS:
        raise ValueError(f"control must be one of {TRANSFER_CONTROLS}, got {control!r}")
    spec_path = _specs_dir() / f"{control}.json"
    if not spec_path.is_file():
        raise FileNotFoundError(f"missing spec: {spec_path}")
    return json.loads(spec_path.read_text())


def make_preview(src: Path, crf: int = 28) -> Path:
    preview = src.with_name(f"{src.stem}_preview.mp4")
    if not preview.exists() or preview.stat().st_mtime < src.stat().st_mtime:
        subprocess.run(
            [
                _ffmpeg_exe(),
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(src),
                "-c:v",
                "libx264",
                "-crf",
                str(crf),
                "-preset",
                "veryfast",
                "-an",
                "-pix_fmt",
                "yuv420p",
                str(preview),
            ],
            check=True,
        )
    return preview


def preview_transfer(control: str, *, model: str | None = None) -> None:
    """Preview control input and generated output for *control*.

    *model* selects which output directory to read — e.g. ``Cosmos3-Nano`` or
    ``Cosmos3-Super``.  Output is read from
    ``<output_root>/<model>/transfer_<control>/vision.mp4``.
    Defaults to the ``COSMOS3_MODEL`` environment variable, falling back to
    ``Cosmos3-Nano``.
    """
    resolved_model = model or os.environ.get("COSMOS3_MODEL", "Cosmos3-Nano")
    spec = load_transfer_spec(control)
    control_path = resolve_spec_path(spec[control]["control_path"])
    vision_path = _output_root() / resolved_model / f"transfer_{control}" / "vision.mp4"
    if not control_path.is_file():
        raise FileNotFoundError(f"missing control video: {control_path}")
    if not vision_path.is_file():
        raise FileNotFoundError(f"missing output: {vision_path} (run {control} inference first)")

    try:
        from IPython.display import Video, display
    except ImportError:
        display = None
        Video = None

    for label, src in [("control", control_path), ("generated", vision_path)]:
        preview = make_preview(src)
        print(
            f"{control} {label}: {src.name} "
            f"({src.stat().st_size // 1024} KB -> {preview.stat().st_size // 1024} KB preview)"
        )
        if display is not None and Video is not None:
            display(Video(str(preview), embed=True))
        else:
            print(f"  preview path: {preview}")


def preview_multi_control(*, model: str | None = None) -> None:
    """Preview computed control signals and the generated output for multi-control transfer.

    *model* selects which output directory to read (e.g. ``Cosmos3-Nano`` or ``Cosmos3-Super``).
    Defaults to the ``COSMOS3_MODEL`` environment variable, falling back to ``Cosmos3-Nano``.
    """
    resolved_model = model or os.environ.get("COSMOS3_MODEL", "Cosmos3-Nano")
    out_dir = _output_root() / resolved_model / "transfer_multi_control"
    vision_path = out_dir / "vision.mp4"
    if not vision_path.is_file():
        raise FileNotFoundError(
            f"missing output: {vision_path}\n"
            "Run the multi-control inference cell first (§19)."
        )

    try:
        from IPython.display import Video, display
    except ImportError:
        display = None
        Video = None

    control_videos = [
        ("edge control (computed on-the-fly)", out_dir / "control_edge.mp4"),
        ("blur control (computed on-the-fly)", out_dir / "control_blur.mp4"),
        ("generated output", vision_path),
    ]

    for label, src in control_videos:
        if not src.is_file():
            print(f"[skip] {label}: {src} not found")
            continue
        preview = make_preview(src)
        print(
            f"{label}: {src.name} "
            f"({src.stat().st_size // 1024} KB -> {preview.stat().st_size // 1024} KB preview)"
        )
        if display is not None and Video is not None:
            display(Video(str(preview), embed=True))
        else:
            print(f"  preview path: {preview}")
