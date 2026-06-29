#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Encode egocentric hand-pose annotations into the raw 57D Action format.

This script is intentionally source-neutral. It demonstrates the hand-pose
action contract using a folder layout like:

    example_root/
      videos/<sample_id>.mp4
      captions/<sample_id>.json
      cameras/<sample_id>.json
      human_annotation/<sample_id>.json

The raw 57D layout is:

    [camera, right_wrist, right_fingertips, left_wrist, left_fingertips]

where camera and wrist pose blocks are [translation(3), rot6d(6)], and each
fingertip block is 5 * xyz in the frame-local wrist coordinate frame.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    from cosmos_framework.data.vfm.action.action_processing import (
        ActionProcessor,
        ActionNormalizationMethod,
        load_action_stats,
        resolve_action_normalization,
    )
    from cosmos_framework.data.vfm.action.pose_utils import (
        build_abs_pose_from_components,
        pose_abs_to_rel,
        pose_rel_to_abs,
    )
except ModuleNotFoundError as exc:
    if exc.name != "cosmos_framework":
        raise
    raise ModuleNotFoundError(
        "This script requires cosmos-framework. Install it and activate its environment, "
        "or run with PYTHONPATH pointing to a cosmos-framework checkout."
    ) from exc

NUM_JOINTS = 21
WRIST = 0
THUMB_TIP = 4
INDEX_TIP = 8
MIDDLE_TIP = 12
RING_TIP = 16
PINKY_TIP = 20
FINGERTIP_JOINTS = (THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP)
DEFAULT_EXAMPLE_ROOT = Path(__file__).resolve().parents[1] / "assets" / "egocentric_hand_action_example"

# Source-specific wrist-frame correction applied as:
#     wrist_pose_for_action = wrist_pose_from_source @ WRIST_FRAME_ALIGN
#
# The ESCALE example is already in the convention used by this script, so this
# stays identity. If your data uses a different wrist-local coordinate frame,
# edit the 3x3 rotation block so the transformed wrist frame has:
#   +X: from thumb side toward pinky side
#   +Y: outward from the palm
#   +Z: from wrist toward fingertips
# Keep the translation column zero unless your wrist origin needs a known offset.
WRIST_FRAME_ALIGN = np.eye(4, dtype=np.float32)  # [4,4]


def _as_float32_array(value: Any, name: str, shape_suffix: tuple[int, ...] | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)  # [...]
    if shape_suffix is not None and array.shape[-len(shape_suffix) :] != shape_suffix:
        raise ValueError(f"{name} must end with shape {shape_suffix}, got {array.shape}")
    return array  # [...]


def build_pose_matrix(translation: np.ndarray, quat_xyzw: np.ndarray) -> np.ndarray:
    translation = _as_float32_array(translation, "translation", (3,))  # [T,3]
    quat_xyzw = _as_float32_array(quat_xyzw, "quat_xyzw", (4,))  # [T,4]
    return build_abs_pose_from_components(translation, quat_xyzw, "quat_xyzw").astype(np.float32)  # [T,4,4]


def invert_pose_matrix(pose: np.ndarray) -> np.ndarray:
    pose = _as_float32_array(pose, "pose", (4, 4))  # [T,4,4]
    return np.linalg.inv(pose).astype(np.float32)  # [T,4,4]


def build_fingertips_in_wrist_frame(hand_keypoints: np.ndarray, wrist_cam: np.ndarray) -> np.ndarray:
    hand_keypoints = _as_float32_array(hand_keypoints, "hand_keypoints", (NUM_JOINTS, 3))  # [T+1,21,3]
    wrist_cam = _as_float32_array(wrist_cam, "wrist_cam", (4, 4))  # [T+1,4,4]
    future_fingertips = hand_keypoints[1:, FINGERTIP_JOINTS, :]  # [T,5,3]
    ones = np.ones((*future_fingertips.shape[:-1], 1), dtype=np.float32)  # [T,5,1]
    future_fingertips_h = np.concatenate([future_fingertips, ones], axis=-1)  # [T,5,4]
    wrist_inv = invert_pose_matrix(wrist_cam[1:])  # [T,4,4]
    fingertips_wrist = np.einsum("tij,tnj->tni", wrist_inv, future_fingertips_h)[..., :3]  # [T,5,3]
    return fingertips_wrist.reshape(fingertips_wrist.shape[0], -1).astype(np.float32)  # [T,15]


def build_hand_wrist_cam(
    hand_keypoints: np.ndarray,
    ee_pose: np.ndarray,
    wrist_position_source: str,
) -> np.ndarray:
    hand_keypoints = _as_float32_array(hand_keypoints, "hand_keypoints", (NUM_JOINTS, 3))  # [T+1,21,3]
    ee_pose = _as_float32_array(ee_pose, "ee_pose", (7,))  # [T+1,7]
    wrist_quat = ee_pose[:, :4]  # [T+1,4]
    if wrist_position_source == "keypoint":
        wrist_pos = hand_keypoints[:, WRIST, :]  # [T+1,3]
    elif wrist_position_source == "ee_pose":
        wrist_pos = ee_pose[:, 4:]  # [T+1,3]
    else:
        raise ValueError(f"Unsupported wrist_position_source: {wrist_position_source!r}")
    wrist_cam = build_pose_matrix(wrist_pos, wrist_quat)  # [T+1,4,4]
    return wrist_cam @ WRIST_FRAME_ALIGN[None, :, :]  # [T+1,4,4]


def build_raw_57d_action(
    camera_world2cam: np.ndarray,
    left_keypoints: np.ndarray,
    right_keypoints: np.ndarray,
    left_ee_pose: np.ndarray,
    right_ee_pose: np.ndarray,
    wrist_position_source: str,
    pose_convention: str,
) -> np.ndarray:
    camera_world2cam = _as_float32_array(camera_world2cam, "camera_world2cam", (7,))  # [T+1,7]
    cam_w2c = build_pose_matrix(camera_world2cam[:, 4:], camera_world2cam[:, :4])  # [T+1,4,4]
    cam_c2w = invert_pose_matrix(cam_w2c)  # [T+1,4,4]

    right_wrist_cam = build_hand_wrist_cam(right_keypoints, right_ee_pose, wrist_position_source)  # [T+1,4,4]
    left_wrist_cam = build_hand_wrist_cam(left_keypoints, left_ee_pose, wrist_position_source)  # [T+1,4,4]
    right_wrist_world = cam_c2w @ right_wrist_cam  # [T+1,4,4]
    left_wrist_world = cam_c2w @ left_wrist_cam  # [T+1,4,4]

    cam_rel = pose_abs_to_rel(cam_c2w, rotation_format="rot6d", pose_convention=pose_convention)  # [T,9]
    right_wrist_rel = pose_abs_to_rel(
        right_wrist_world,
        rotation_format="rot6d",
        pose_convention=pose_convention,
    )  # [T,9]
    left_wrist_rel = pose_abs_to_rel(
        left_wrist_world,
        rotation_format="rot6d",
        pose_convention=pose_convention,
    )  # [T,9]
    right_fingertips = build_fingertips_in_wrist_frame(right_keypoints, right_wrist_cam)  # [T,15]
    left_fingertips = build_fingertips_in_wrist_frame(left_keypoints, left_wrist_cam)  # [T,15]

    return np.concatenate(
        [cam_rel, right_wrist_rel, right_fingertips, left_wrist_rel, left_fingertips],
        axis=-1,
    ).astype(np.float32)  # [T,57]


def decompose_raw_57d_action(action: np.ndarray) -> dict[str, np.ndarray]:
    action = _as_float32_array(action, "action", (57,))  # [T,57]
    return {
        "camera": action[:, 0:9],  # [T,9]
        "right_wrist": action[:, 9:18],  # [T,9]
        "right_fingertips": action[:, 18:33].reshape(action.shape[0], len(FINGERTIP_JOINTS), 3),  # [T,5,3]
        "left_wrist": action[:, 33:42],  # [T,9]
        "left_fingertips": action[:, 42:57].reshape(action.shape[0], len(FINGERTIP_JOINTS), 3),  # [T,5,3]
    }


def _pose_error_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    pred = _as_float32_array(pred, "pred", (4, 4))  # [T,4,4]
    target = _as_float32_array(target, "target", (4, 4))  # [T,4,4]
    trans_l2 = np.linalg.norm(pred[:, :3, 3] - target[:, :3, 3], axis=-1)  # [T]
    rot_delta = np.swapaxes(pred[:, :3, :3], -1, -2) @ target[:, :3, :3]  # [T,3,3]
    trace = np.trace(rot_delta, axis1=1, axis2=2)  # [T]
    cos_angle = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)  # [T]
    rot_deg = np.degrees(np.arccos(cos_angle))  # [T]
    return {
        "translation_l2_max": float(trans_l2.max(initial=0.0)),
        "translation_l2_mean": float(trans_l2.mean() if trans_l2.size else 0.0),
        "rotation_deg_max": float(rot_deg.max(initial=0.0)),
        "rotation_deg_mean": float(rot_deg.mean() if rot_deg.size else 0.0),
    }


def verify_raw_57d_roundtrip(
    action: np.ndarray,
    camera_world2cam: np.ndarray,
    left_keypoints: np.ndarray,
    right_keypoints: np.ndarray,
    left_ee_pose: np.ndarray,
    right_ee_pose: np.ndarray,
    wrist_position_source: str,
    pose_convention: str,
) -> dict[str, Any]:
    """Decode the raw action and compare against the original pose arrays."""
    parts = decompose_raw_57d_action(action)
    cam_w2c = build_pose_matrix(camera_world2cam[:, 4:], camera_world2cam[:, :4])  # [T+1,4,4]
    target_cam_c2w = invert_pose_matrix(cam_w2c)  # [T+1,4,4]
    target_right_wrist_cam = build_hand_wrist_cam(
        right_keypoints,
        right_ee_pose,
        wrist_position_source,
    )  # [T+1,4,4]
    target_left_wrist_cam = build_hand_wrist_cam(
        left_keypoints,
        left_ee_pose,
        wrist_position_source,
    )  # [T+1,4,4]
    target_right_wrist_world = target_cam_c2w @ target_right_wrist_cam  # [T+1,4,4]
    target_left_wrist_world = target_cam_c2w @ target_left_wrist_cam  # [T+1,4,4]

    pred_cam_c2w = pose_rel_to_abs(
        parts["camera"],
        rotation_format="rot6d",
        pose_convention=pose_convention,
        initial_pose=target_cam_c2w[0],
        normalize_rotation=False,
    ).astype(np.float32)  # [T+1,4,4]
    pred_right_wrist_world = pose_rel_to_abs(
        parts["right_wrist"],
        rotation_format="rot6d",
        pose_convention=pose_convention,
        initial_pose=target_right_wrist_world[0],
        normalize_rotation=False,
    ).astype(np.float32)  # [T+1,4,4]
    pred_left_wrist_world = pose_rel_to_abs(
        parts["left_wrist"],
        rotation_format="rot6d",
        pose_convention=pose_convention,
        initial_pose=target_left_wrist_world[0],
        normalize_rotation=False,
    ).astype(np.float32)  # [T+1,4,4]

    pred_cam_w2c = invert_pose_matrix(pred_cam_c2w)  # [T+1,4,4]
    pred_right_wrist_cam = pred_cam_w2c @ pred_right_wrist_world  # [T+1,4,4]
    pred_left_wrist_cam = pred_cam_w2c @ pred_left_wrist_world  # [T+1,4,4]

    right_fingertips_h = np.concatenate(
        [parts["right_fingertips"], np.ones((*parts["right_fingertips"].shape[:-1], 1), dtype=np.float32)],
        axis=-1,
    )  # [T,5,4]
    left_fingertips_h = np.concatenate(
        [parts["left_fingertips"], np.ones((*parts["left_fingertips"].shape[:-1], 1), dtype=np.float32)],
        axis=-1,
    )  # [T,5,4]
    pred_right_tips_cam = np.einsum("tij,tnj->tni", pred_right_wrist_cam[1:], right_fingertips_h)[..., :3]  # [T,5,3]
    pred_left_tips_cam = np.einsum("tij,tnj->tni", pred_left_wrist_cam[1:], left_fingertips_h)[..., :3]  # [T,5,3]
    target_right_tips_cam = right_keypoints[1:, FINGERTIP_JOINTS, :]  # [T,5,3]
    target_left_tips_cam = left_keypoints[1:, FINGERTIP_JOINTS, :]  # [T,5,3]
    right_tip_l2 = np.linalg.norm(pred_right_tips_cam - target_right_tips_cam, axis=-1)  # [T,5]
    left_tip_l2 = np.linalg.norm(pred_left_tips_cam - target_left_tips_cam, axis=-1)  # [T,5]

    return {
        "camera_c2w": _pose_error_metrics(pred_cam_c2w, target_cam_c2w),
        "right_wrist_world": _pose_error_metrics(pred_right_wrist_world, target_right_wrist_world),
        "left_wrist_world": _pose_error_metrics(pred_left_wrist_world, target_left_wrist_world),
        "right_fingertips_camera_l2_max": float(right_tip_l2.max(initial=0.0)),
        "right_fingertips_camera_l2_mean": float(right_tip_l2.mean() if right_tip_l2.size else 0.0),
        "left_fingertips_camera_l2_max": float(left_tip_l2.max(initial=0.0)),
        "left_fingertips_camera_l2_mean": float(left_tip_l2.mean() if left_tip_l2.size else 0.0),
    }


def load_example(root: Path, sample_id: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    annotation = json.loads((root / "human_annotation" / f"{sample_id}.json").read_text())
    camera = json.loads((root / "cameras" / f"{sample_id}.json").read_text())
    caption = json.loads((root / "captions" / f"{sample_id}.json").read_text())
    return annotation, camera, caption


def get_video_frame_count(video_path: Path) -> int | None:
    if not video_path.exists():
        return None
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=nb_frames",
        "-of",
        "default=nw=1:nk=1",
        str(video_path),
    ]
    try:
        output = subprocess.check_output(cmd, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return int(output) if output.isdigit() else None


def preprocess_model_action(
    raw_action: np.ndarray,
    stats_path: Path,
    stats_key: str,
    normalization_method: ActionNormalizationMethod,
    max_action_dim: int,
) -> np.ndarray:
    raw_action = _as_float32_array(raw_action, "raw_action", (57,))  # [T,57]
    stats_np = load_action_stats(str(stats_path), stats_key=stats_key)
    stats: dict[str, torch.Tensor] = {}
    for key, value in stats_np.items():
        stats[key] = torch.from_numpy(value).float()  # [D]
    normalizer = resolve_action_normalization(normalization_method, stats)
    processor = ActionProcessor(max_action_dim=max_action_dim)
    raw_action_tensor = torch.from_numpy(raw_action).float()  # [T,57]
    processed = processor.preprocess_action(
        {},
        raw_action_tensor,
        action_normalizer=normalizer,
    )  # dict
    processed_action = processed["action"]  # [T,D_model]
    return processed_action.detach().cpu().numpy().astype(np.float32)  # [T,D_model]


def pick_default_sample_id(root: Path) -> str:
    candidates = sorted((root / "human_annotation").glob("*.json"))
    if not candidates:
        raise FileNotFoundError(f"No annotation JSON files found under {root / 'human_annotation'}")
    return candidates[0].stem


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--example-root", type=Path, default=DEFAULT_EXAMPLE_ROOT)
    parser.add_argument("--sample-id", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("hand_pose_57d_outputs"))
    parser.add_argument("--pose-convention", choices=["backward_framewise", "backward_anchored"], default="backward_framewise")
    parser.add_argument("--wrist-position-source", choices=["keypoint", "ee_pose"], default="keypoint")
    parser.add_argument("--normalizer-stats", type=Path, default=None)
    parser.add_argument("--normalizer-stats-key", type=str, default="global_raw")
    parser.add_argument(
        "--action-normalization",
        choices=["quantile", "quantile_rot", "meanstd", "minmax"],
        default="quantile_rot",
    )
    parser.add_argument("--max-action-dim", type=int, default=64)
    parser.add_argument("--skip-roundtrip-check", action="store_true")
    args = parser.parse_args()

    root = args.example_root
    sample_id = args.sample_id or pick_default_sample_id(root)
    annotation, camera, caption = load_example(root, sample_id)

    num_frames = int(annotation["num_frames"])
    left_keypoints = _as_float32_array(annotation["left_hand"]["hand_keypoints"], "left_hand.hand_keypoints", (NUM_JOINTS, 3))  # [N,21,3]
    right_keypoints = _as_float32_array(annotation["right_hand"]["hand_keypoints"], "right_hand.hand_keypoints", (NUM_JOINTS, 3))  # [N,21,3]
    left_ee_pose = _as_float32_array(annotation["left_ee_pose"], "left_ee_pose", (7,))  # [N,7]
    right_ee_pose = _as_float32_array(annotation["right_ee_pose"], "right_ee_pose", (7,))  # [N,7]
    camera_world2cam = _as_float32_array(camera["camera"]["pose_world2cam"], "camera.pose_world2cam", (7,))  # [N,7]

    if not all(array.shape[0] == num_frames for array in [left_keypoints, right_keypoints, left_ee_pose, right_ee_pose, camera_world2cam]):
        raise ValueError("All pose arrays must have num_frames rows.")

    left_wrist_pos_delta = np.linalg.norm(left_keypoints[:, WRIST, :] - left_ee_pose[:, 4:], axis=1)  # [N]
    right_wrist_pos_delta = np.linalg.norm(right_keypoints[:, WRIST, :] - right_ee_pose[:, 4:], axis=1)  # [N]
    raw_action = build_raw_57d_action(
        camera_world2cam=camera_world2cam,
        left_keypoints=left_keypoints,
        right_keypoints=right_keypoints,
        left_ee_pose=left_ee_pose,
        right_ee_pose=right_ee_pose,
        wrist_position_source=args.wrist_position_source,
        pose_convention=args.pose_convention,
    )  # [N-1,57]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_dir / f"{sample_id}_raw_action_57d.npy"
    np.save(raw_path, raw_action)

    model_action_path = None
    model_action = None
    if args.normalizer_stats is not None:
        model_action = preprocess_model_action(
            raw_action,
            args.normalizer_stats,
            args.normalizer_stats_key,
            args.action_normalization,
            args.max_action_dim,
        )  # [N-1,D_model]
        model_action_path = args.output_dir / f"{sample_id}_model_action_{args.max_action_dim}d.npy"
        np.save(model_action_path, model_action)

    roundtrip = None
    if not args.skip_roundtrip_check:
        roundtrip = verify_raw_57d_roundtrip(
            action=raw_action,
            camera_world2cam=camera_world2cam,
            left_keypoints=left_keypoints,
            right_keypoints=right_keypoints,
            left_ee_pose=left_ee_pose,
            right_ee_pose=right_ee_pose,
            wrist_position_source=args.wrist_position_source,
            pose_convention=args.pose_convention,
        )

    video_frames = get_video_frame_count(root / "videos" / f"{sample_id}.mp4")
    caption_text = caption.get("vlm_pipeline", {}).get("long") or caption.get("vlm_pipeline", {}).get("medium")

    metadata = {
        "sample_id": sample_id,
        "num_pose_frames": num_frames,
        "num_action_frames": int(raw_action.shape[0]),
        "video_frames": video_frames,
        "raw_action_shape": list(raw_action.shape),
        "raw_action_path": str(raw_path),
        "model_action_path": str(model_action_path) if model_action_path else None,
        "layout": "[camera(9), right_wrist(9), right_fingertips(15), left_wrist(9), left_fingertips(15)]",
        "pose_convention": args.pose_convention,
        "wrist_position_source": args.wrist_position_source,
        "wrist_frame_align": WRIST_FRAME_ALIGN.tolist(),
        "wrist_keypoint_vs_ee_translation_l2_m": {
            "left_mean": float(left_wrist_pos_delta.mean()),
            "left_max": float(left_wrist_pos_delta.max()),
            "right_mean": float(right_wrist_pos_delta.mean()),
            "right_max": float(right_wrist_pos_delta.max()),
        },
        "roundtrip_check": roundtrip,
        "caption": caption_text,
    }
    metadata_path = args.output_dir / f"{sample_id}_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")

    print(f"sample_id: {sample_id}")
    print(f"pose frames: {num_frames}")
    print(f"video frames: {video_frames}")
    if video_frames is not None and abs(video_frames - num_frames) > 1:
        print("warning: video and pose frame counts differ by more than one frame")
    print(
        "wrist position source: "
        f"{args.wrist_position_source} "
        f"(left keypoint-vs-ee mean {float(left_wrist_pos_delta.mean()):.4f} m, "
        f"right {float(right_wrist_pos_delta.mean()):.4f} m)"
    )
    print(f"raw action: {raw_action.shape} -> {raw_path}")
    if model_action_path is not None:
        print(f"model action: {model_action.shape} -> {model_action_path}")
    if roundtrip is not None:
        print("round-trip check against source annotations:")
        print(f"  camera c2w: {roundtrip['camera_c2w']}")
        print(f"  right wrist world: {roundtrip['right_wrist_world']}")
        print(f"  left wrist world: {roundtrip['left_wrist_world']}")
        print(
            "  fingertip camera L2 max/mean: "
            f"right {roundtrip['right_fingertips_camera_l2_max']:.3e}/"
            f"{roundtrip['right_fingertips_camera_l2_mean']:.3e}, "
            f"left {roundtrip['left_fingertips_camera_l2_max']:.3e}/"
            f"{roundtrip['left_fingertips_camera_l2_mean']:.3e}"
        )
    print(f"metadata: {metadata_path}")


if __name__ == "__main__":
    main()
