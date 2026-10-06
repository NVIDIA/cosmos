from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from diffusers import CosmosActionCondition
from diffusers.utils import load_image, load_video

RESOLUTIONS = {
    "240p": (240, 432),
    "480p": (480, 848),
    "720p": (720, 1280),
}
ACTION_TIERS = {
    "240p": 256,
    "480p": 480,
    "720p": 720,
}


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _summarize_trace(path: Path) -> dict[str, Any]:
    samples = [json.loads(line) for line in path.read_text().splitlines() if line]
    return {
        "samples": len(samples),
        "peak_allocated_bytes": max((sample.get("global_peak_allocated_bytes", 0) for sample in samples), default=0),
        "peak_reserved_bytes": max((sample.get("global_peak_reserved_bytes", 0) for sample in samples), default=0),
        "peak_device_used_bytes": max(
            (sample.get("global_peak_device_used_bytes", 0) for sample in samples), default=0
        ),
    }


def _write_sound(result: dict[str, Any], path: Path) -> dict[str, Any]:
    from scipy.io import wavfile

    waveform = result["sound"].detach().float().cpu().numpy()
    if waveform.ndim == 3 and waveform.shape[0] == 1:
        waveform = waveform[0]
    if waveform.ndim != 2:
        raise ValueError(f"expected Cosmos sound [channels, samples], got {tuple(waveform.shape)}")
    sampling_rate = int(result["sampling_rate"])
    wavfile.write(path, sampling_rate, waveform.T)
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "channels": int(waveform.shape[0]),
        "samples": int(waveform.shape[1]),
        "sampling_rate": sampling_rate,
    }


def _common_video_args(device: str, height: int, width: int, seed: int) -> dict[str, Any]:
    return {
        "num_frames": 189,
        "height": height,
        "width": width,
        "fps": 24.0,
        "num_inference_steps": 35,
        "guidance_scale": 6.0,
        "add_resolution_template": False,
        "add_duration_template": False,
        "generator": torch.Generator(device=device).manual_seed(seed),
    }


def _prepare_call(
    modality: str,
    model: Path,
    device: str,
    height: int,
    width: int,
    resolution: str,
    v2v_assets: Path,
    transfer_assets: Path,
    action_video: Path,
) -> tuple[dict[str, Any], dict[str, Any], int]:
    assets = model / "assets"
    negative_prompt = _load_json(assets / "negative_prompt.json")
    inputs: dict[str, Any] = {
        "negative_prompt_json": str((assets / "negative_prompt.json").resolve()),
        "negative_prompt_sha256": _sha256(assets / "negative_prompt.json"),
    }

    if modality in {"t2v", "i2v", "t2vs", "i2vs"}:
        prompt_name = (
            "example_t2vs_prompt.json"
            if modality == "t2vs"
            else ("example_i2v_prompt.json" if modality in {"i2v", "i2vs"} else "example_t2v_prompt.json")
        )
        prompt_path = assets / prompt_name
        call = _common_video_args(device, height, width, seed=0 if modality.endswith("s") else 123)
        if modality == "i2v":
            call["generator"] = torch.Generator(device=device).manual_seed(1111)
        call.update(
            {
                "prompt": json.dumps(_load_json(prompt_path)),
                "negative_prompt": json.dumps(negative_prompt),
            }
        )
        inputs.update({"prompt_json": str(prompt_path.resolve()), "prompt_sha256": _sha256(prompt_path)})
        if modality in {"i2v", "i2vs"}:
            image_path = assets / "example_i2v_input.jpg"
            call["image"] = load_image(str(image_path))
            inputs.update({"image": str(image_path.resolve()), "image_sha256": _sha256(image_path)})
        if modality in {"t2vs", "i2vs"}:
            call["enable_sound"] = True
            call["output"] = ["videos", "sound", "sampling_rate"]
        else:
            call["output"] = "videos"
        return call, inputs, 24

    if modality == "v2v":
        video_path = v2v_assets / "input_robot_pouring.mp4"
        prompt_path = v2v_assets / "input_prompt.json"
        negative_path = v2v_assets / "input_negative_prompt.json"
        call = _common_video_args(device, height, width, seed=0)
        call.update(
            {
                "prompt": json.dumps(_load_json(prompt_path)),
                "negative_prompt": json.dumps(_load_json(negative_path)),
                "video": load_video(str(video_path)),
                "condition_frame_indexes_vision": [0, 1],
                "condition_video_keep": "first",
                "output": "videos",
            }
        )
        inputs = {
            "video": str(video_path.resolve()),
            "video_sha256": _sha256(video_path),
            "prompt_json": str(prompt_path.resolve()),
            "prompt_sha256": _sha256(prompt_path),
            "negative_prompt_json": str(negative_path.resolve()),
            "negative_prompt_sha256": _sha256(negative_path),
        }
        return call, inputs, 24

    if modality == "transfer":
        video_path = transfer_assets / "edge" / "control_edge.mp4"
        prompt_path = transfer_assets / "edge" / "prompt.json"
        negative_path = transfer_assets / "negative_prompt.json"
        call = {
            "prompt": json.dumps(_load_json(prompt_path)),
            "negative_prompt": json.dumps(_load_json(negative_path)),
            "control_videos": {"edge": load_video(str(video_path))},
            "num_frames": 121,
            "height": height,
            "width": width,
            "fps": 30.0,
            "num_inference_steps": 35,
            "guidance_scale": 3.0,
            "control_guidance": 1.5,
            "generator": torch.Generator(device=device).manual_seed(0),
            "output": "videos",
        }
        inputs = {
            "control_video": str(video_path.resolve()),
            "control_video_sha256": _sha256(video_path),
            "prompt_json": str(prompt_path.resolve()),
            "prompt_sha256": _sha256(prompt_path),
            "negative_prompt_json": str(negative_path.resolve()),
            "negative_prompt_sha256": _sha256(negative_path),
        }
        return call, inputs, 30

    if modality == "action":
        tier = ACTION_TIERS[resolution]
        call = {
            "prompt": "Put the pot to the left of the purple item.",
            "action": CosmosActionCondition(
                mode="policy",
                chunk_size=16,
                domain_name="bridge_orig_lerobot",
                resolution_tier=tier,
                video=load_video(str(action_video)),
                view_point="ego_view",
            ),
            "fps": 5,
            "num_inference_steps": 30,
            "guidance_scale": 1.0,
            "use_system_prompt": False,
            "generator": torch.Generator(device=device).manual_seed(0),
            "output": ["videos", "action"],
        }
        inputs = {"video": str(action_video.resolve()), "video_sha256": _sha256(action_video)}
        return call, inputs, 5

    raise AssertionError(f"unhandled modality: {modality}")
