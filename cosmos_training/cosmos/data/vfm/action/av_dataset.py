"""AV Dataset for Action training.

This module provides an IterableDataset for AV data
that loads S3 storage containing tar files with video, action trajectories,
and route waypoints.

Data format expected:
    s3://bucket/path/*.tar -> pkl files containing:
        - video: mp4 bytes
        - action: pickled dict with:
            - history_xyz: (history_length, 3) tensor - position history
            - history_quat: (history_length, 4) tensor - quaternion history
            - future_xyz: (future_length, 3) tensor - position future
            - future_quat: (future_length, 4) tensor - quaternion future
        - route: pickled numpy array of shape (num_waypoints, 3) - route waypoints in ego frame

Action format: 7D pose per timestep [x, y, z, qw, qx, qy, qz] (3 position + 4 quaternion)
"""

import io
import json
import math
import pickle
import random
import tarfile
from typing import Iterator, Literal

import numpy as np
import torch
import torchvision
import torchvision.transforms.functional as F
from scipy.spatial.transform import Rotation
from torch.utils.data import IterableDataset

# import torch.multiprocessing
# torch.multiprocessing.set_sharing_strategy("file_system")
from cosmos.utils import log
from cosmos.utils.easy_io import easy_io
from cosmos.data.vfm.action.camera_dataset import get_target_size_and_crop
from cosmos.data.vfm.action.domain_utils import get_domain_id
from cosmos.data.vfm.action.pose_utils import (
    RotationConvention,
    build_abs_pose_from_components,
    pose_abs_to_rel,
)


def decode_video_bytes(
    video_bytes: bytes,
    resolution: str | None = None,
    history_len: float | None = None,
    future_len: float | None = None,
    original_history_steps: int | None = None,
) -> tuple[torch.Tensor, float]:
    """Decode video from mp4 bytes using torchvision.io.

    Args:
        video_bytes: Raw mp4 video bytes.
        resolution: Target resolution for video frames (e.g. "256", "480"). If None, keeps original resolution.
        history_len: Desired history length in seconds. Used with future_len to slice video.
        future_len: Desired future length in seconds. Used with history_len to slice video.
        original_history_steps: Number of frames in the original history portion of the video.

    Returns:
        Tuple of (video tensor in (C, T, H, W) uint8 format, original fps).

    Note:
        The video structure is [history_frames | future_frames]. When slicing:
        - History portion: take last (history_len * fps) frames from video[:original_history_steps]
        - Future portion: take first (future_len * fps) frames from video[original_history_steps:]
        This mirrors the slicing in process_action_trajectory.
    """
    # Write bytes to a temporary file for torchvision.io.read_video
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as tmp_file:
        tmp_file.write(video_bytes)
        tmp_file.flush()

        # Read video using torchvision.io
        # Returns: (video_frames, audio_frames, info)
        # video_frames shape: (T, H, W, C) uint8
        video_frames, _, info = torchvision.io.read_video(tmp_file.name, pts_unit="sec")

    original_fps = info.get("video_fps", 30.0)

    # Slice video to match history_len and future_len
    # Video structure: [history_frames | future_frames]
    if original_history_steps is None:
        raise ValueError("original_history_steps is required to slice video")

    # Split video at history/future boundary
    history_video = video_frames[:original_history_steps]
    future_video = video_frames[original_history_steps:]

    # Slice history (take last N frames)
    if history_len is not None:
        history_steps = int(history_len * original_fps)
        if history_steps > history_video.shape[0]:
            raise ValueError(
                f"Requested history_len={history_len}s ({history_steps} frames at {original_fps}Hz) "
                f"exceeds available history video ({history_video.shape[0]} frames)"
            )
        history_video = history_video[-history_steps:]

    # Slice future (take first N frames)
    if future_len is not None:
        future_steps = int(future_len * original_fps)
        if future_steps > future_video.shape[0]:
            raise ValueError(
                f"Requested future_len={future_len}s ({future_steps} frames at {original_fps}Hz) "
                f"exceeds available future video ({future_video.shape[0]} frames)"
            )
        future_video = future_video[:future_steps]

    # Concatenate sliced portions
    video_frames = torch.cat([history_video, future_video], dim=0)  # [T,H,W,C]

    # Convert from (T, H, W, C) to (T, C, H, W)
    video_tensor = video_frames.permute(0, 3, 1, 2)  # [T,C,H,W]

    # Resize and Crop if resolution is provided
    if resolution is not None:
        T, C, H, W = video_tensor.shape
        # get_target_size_and_crop expects (resolution, current_H, current_W)
        new_H, new_W, target_canvas_H, target_canvas_W = get_target_size_and_crop(resolution, H, W)

        # Resize if needed
        if new_H != H or new_W != W:
            video_tensor = F.resize(
                video_tensor, [new_H, new_W], interpolation=F.InterpolationMode.BICUBIC, antialias=True
            )

        # Center Crop
        if new_H != target_canvas_H or new_W != target_canvas_W:
            video_tensor = F.center_crop(video_tensor, [target_canvas_H, target_canvas_W])

    # Convert to uint8 if not already
    if video_tensor.dtype != torch.uint8:
        video_tensor = video_tensor.to(torch.uint8)

    # Convert from (T, C, H, W) to (C, T, H, W)
    video_tensor = video_tensor.permute(1, 0, 2, 3)  # [C,T,H,W]

    return video_tensor, original_fps


# 3x3 rotation from car convention (x=forward, y=left, z=up)
# to OpenCV convention (x=right, y=down, z=forward).
# Mapping: new_x = -old_y, new_y = -old_z, new_z = old_x
CAR_TO_OPENCV_ROTATION = np.array(
    [[0, -1, 0], [0, 0, -1], [1, 0, 0]],
    dtype=np.float32,
)


def process_action_trajectory(
    action_data: dict,
    history_len: float | None = None,
    future_len: float | None = None,
    fps: int = 10,
    rotation_format: Literal["9D", "rot6d", "quat_xyzw", "euler_xyz"] = "9D",
    pose_convention: Literal["backward_anchored", "backward_framewise"] = ("backward_framewise"),
    scale: float = 1.0,
    rotation_scale: float = 1.0,
    max_translation_norm: float | None = None,
    align_opencv_pose: bool = False,
):
    """Process action trajectories from action data dict.

    Args:
        action_data: Dict with:
            - history_xyz: (T_hist, 3) tensor - position history
            - history_quat: (T_hist, 4) tensor - quaternion history
            - future_xyz: (T_fut, 3) tensor - position future
            - future_quat: (T_fut, 4) tensor - quaternion future
        history_len: Desired history length in seconds.
        future_len: Desired future length in seconds.
        fps: Frames per second, used to compute number of steps from time durations.
        align_opencv_pose: If True, transform poses from car convention
            (x=forward, y=left, z=up) to OpenCV convention (x=right, y=down, z=forward).
            NOTE: av_v2_* data is already in OpenCV convention, DO NOT apply this transformation!

    Returns:
        Tuple of (history_action, future_action).
        Both actions are torch.Tensor of shape (T, 7) in [x, y, z, qw, qx, qy, qz] format.

    Note:
        History steps = history_len * fps, same as future steps = future_len * fps.
        For example, with history_len=1.0s and fps=10, we get 10 history steps.
    """
    # Extract and ensure tensors
    history_xyz = action_data["history_xyz"]
    history_quat = action_data["history_quat"]
    future_xyz = action_data["future_xyz"]
    future_quat = action_data["future_quat"]

    # Convert to tensors if needed
    if not isinstance(history_xyz, torch.Tensor):
        history_xyz = torch.tensor(history_xyz, dtype=torch.float32)  # [T_hist,3]
    if not isinstance(history_quat, torch.Tensor):
        history_quat = torch.tensor(history_quat, dtype=torch.float32)  # [T_hist,4]
    if not isinstance(future_xyz, torch.Tensor):
        future_xyz = torch.tensor(future_xyz, dtype=torch.float32)  # [T_fut,3]
    if not isinstance(future_quat, torch.Tensor):
        future_quat = torch.tensor(future_quat, dtype=torch.float32)  # [T_fut,4]

    # Slice history to desired length (take the last N steps)
    if history_len is not None:
        history_steps = int(history_len * fps)
        available_history = history_xyz.shape[0]
        if history_steps > available_history:
            raise ValueError(
                f"Requested history_len={history_len}s ({history_steps} steps at {fps}Hz) "
                f"exceeds available history ({available_history} steps)"
            )
        history_xyz = history_xyz[-history_steps:]
        history_quat = history_quat[-history_steps:]

    # Slice future to desired length (take the first N steps)
    if future_len is not None:
        future_steps = int(future_len * fps)
        available_future = future_xyz.shape[0]
        if future_steps > available_future:
            raise ValueError(
                f"Requested future_len={future_len}s ({future_steps} steps at {fps}Hz) "
                f"exceeds available future ({available_future} steps)"
            )
        future_xyz = future_xyz[:future_steps]
        future_quat = future_quat[:future_steps]

    # Concatenate to form full trajectory
    # history_xyz: (T_hist, 3)
    # history_quat: (T_hist, 4) [w, x, y, z]
    all_xyz = torch.cat([history_xyz, future_xyz], dim=0)  # [T,3]
    all_quat = torch.cat([history_quat, future_quat], dim=0)  # [T,4]

    poses_abs = build_abs_pose_from_components(
        all_xyz,
        all_quat,
        "quat_wxyz",
    )

    if align_opencv_pose:
        poses_abs[:, :3, :3] = poses_abs[:, :3, :3] @ CAR_TO_OPENCV_ROTATION.T

    actions = pose_abs_to_rel(
        poses_abs,
        rotation_format=rotation_format,
        pose_convention=pose_convention,
        translation_scale=scale,
        rotation_scale=rotation_scale,
    )

    if max_translation_norm is not None:
        trans_norms = np.linalg.norm(actions[:, :3], axis=1)
        if trans_norms.max() > max_translation_norm:
            return None

    actions = torch.from_numpy(actions)  # [T-1,action_dim]

    # Split back
    # history_action has one less action than history_xyz because the first action is the initial pose
    history_action = actions[: len(history_xyz) - 1]
    future_action = actions[len(history_xyz) - 1 :]

    return history_action, future_action


def add_route_noise(
    route: torch.Tensor,
    lat_noise_range: float = 0.0,
    long_noise_range: float = 0.0,
    point_wise_noise: float = 0.0,
) -> torch.Tensor:
    """Add noise to route waypoints for data augmentation.

    Applies two types of noise:
    1. Uniform lateral/longitudinal shift (same offset for all waypoints in a sample)
    2. Per-point Gaussian noise (independent per waypoint)

    Both noise types leave the z-axis unchanged. NaN waypoints (padding) are preserved.

    Args:
        route: (T, 3) tensor of route waypoints in XYZ.
        lat_noise_range: Half-range for uniform lateral (Y) noise.
        long_noise_range: Half-range for uniform longitudinal (X) noise.
        point_wise_noise: Standard deviation of per-point Gaussian noise.

    Returns:
        Noisy route tensor of shape (T, 3).
    """
    if lat_noise_range > 0 or long_noise_range > 0:
        shift = torch.rand(3) * torch.tensor([2 * long_noise_range, 2 * lat_noise_range, 0.0]) - torch.tensor(  # [3]
            [long_noise_range, lat_noise_range, 0.0]
        )
        route = route + shift[None, :]

    if point_wise_noise > 0:
        noise = torch.randn(route.shape[0], 3) * point_wise_noise  # [T,3]
        noise[..., -1] = 0.0
        route = route + noise

    return route


def apply_route_dropout(
    route: torch.Tensor,
    dropout_rate: float = 0.5,
    tail_dropout_rate: float = 0.3,
) -> torch.Tensor:
    """Apply dropout masking to route waypoints for data augmentation.

    Three dropout behaviours, applied sequentially:
    1. With probability ``dropout_rate``, mask **all** waypoints.
       Otherwise, randomly mask the first K waypoints (K ~ Uniform(0, T)).
    2. With probability ``tail_dropout_rate``, additionally mask waypoints
       from a random index in [T//2, T) onward.

    Masked waypoints are set to NaN so downstream code can detect padding
    via ``torch.isnan``.

    Args:
        route: (T, 3) tensor of route waypoints.
        dropout_rate: Probability of fully disabling the route.
        tail_dropout_rate: Probability of additional tail dropout.

    Returns:
        Route tensor with dropout applied, shape (T, 3).
    """
    T = route.shape[0]
    mask = torch.isnan(route[..., 0])  # [T] existing padding

    if random.uniform(0, 1) < dropout_rate:
        dropout_mask = torch.ones(T, dtype=torch.bool)  # [T]
    else:
        dropout_idx = random.randint(0, T)
        dropout_mask = torch.arange(T) < dropout_idx  # [T]

    if random.uniform(0, 1) < tail_dropout_rate:
        tail_idx = random.randint(T // 2, T - 1) if T > 1 else 0
        tail_dropout_mask = torch.arange(T) >= tail_idx  # [T]
        dropout_mask = dropout_mask | tail_dropout_mask

    mask = mask | dropout_mask
    route = route.clone()
    route[mask] = float("nan")
    return route


def _classify_displacement(dx: float, dy: float, move_threshold: float = 0.1) -> str:
    """Classify a 2D displacement vector into a direction label.

    Uses the angle of the displacement in ego frame (X=forward, Y=left) to
    determine the driving direction.

    Args:
        dx: Forward displacement (positive = forward).
        dy: Lateral displacement (positive = left).
        move_threshold: Minimum displacement magnitude (meters) to count as movement.

    Returns:
        One of: "go forward", "turn left", "turn right", "go backward", "stay".
    """
    dist = math.sqrt(dx * dx + dy * dy)
    if dist < move_threshold:
        return "stay"

    angle_deg = math.degrees(math.atan2(dy, dx))

    if -45 <= angle_deg <= 45:
        return "go forward"
    elif 45 < angle_deg <= 135:
        return "turn left"
    elif -135 <= angle_deg < -45:
        return "turn right"
    else:
        return "go backward"


def classify_trajectory_to_text(
    trajectory: torch.Tensor,
    move_threshold: float = 0.1,
    min_segment_steps: int = 2,
) -> str:
    """Classify a trajectory in ego frame into a brief semantic text description.

    Classifies each consecutive point pair independently, groups consecutive
    identical labels, filters out noisy short groups, and joins distinct
    phases with "then".

    Works with any (T, 3) path in ego frame — route waypoints or pose
    positions returned by :func:`compute_future_trajectory_in_ego_frame`.

    Args:
        trajectory: (T, 3) tensor of positions in ego frame (X=forward, Y=left, Z=up).
            The first point is treated as the starting position.
        move_threshold: Minimum per-step displacement (meters) to count as movement.
        min_segment_steps: Minimum consecutive steps required for a direction label to
            be kept; shorter runs are treated as noise and dropped.

    Returns:
        A description such as "go forward", "stay then go forward",
        "turn left then go forward", or "stay" when the trajectory is empty
        or all NaN.
    """
    valid_mask = ~torch.isnan(trajectory[:, 0])
    valid_pts = trajectory[valid_mask]

    if len(valid_pts) < 2:
        return "stay"

    # Classify every consecutive point pair
    step_labels: list[str] = []
    for i in range(len(valid_pts) - 1):
        dx = valid_pts[i + 1, 0].item() - valid_pts[i, 0].item()
        dy = valid_pts[i + 1, 1].item() - valid_pts[i, 1].item()
        step_labels.append(_classify_displacement(dx, dy, move_threshold))

    # Group consecutive identical labels with their counts
    groups: list[tuple[str, int]] = []
    for label in step_labels:
        if groups and label == groups[-1][0]:
            groups[-1] = (label, groups[-1][1] + 1)
        else:
            groups.append((label, 1))

    # Filter out groups shorter than min_segment_steps to suppress noise
    if len(groups) > 1:
        filtered = [(lbl, cnt) for lbl, cnt in groups if cnt >= min_segment_steps]
        if not filtered:
            # All groups are short — keep the longest one
            filtered = [max(groups, key=lambda g: g[1])]
        groups = filtered

    # Deduplicate consecutive identical labels (may arise after filtering)
    result = [groups[0][0]]
    for label, _ in groups[1:]:
        if label != result[-1]:
            result.append(label)

    return " then ".join(result)


def compute_future_trajectory_in_ego_frame(
    action_data: dict,
    history_len: float | None = None,
    future_len: float | None = None,
    fps: int = 10,
) -> torch.Tensor:
    """Compute future trajectory positions in the ego coordinate frame.

    Transforms absolute future xyz positions so that the origin is the last
    history pose and axes align with ego frame (X=forward, Y=left, Z=up).

    Args:
        action_data: Dict with keys ``history_xyz``, ``history_quat``,
            ``future_xyz`` (and optionally ``future_quat``).
        history_len: History length in seconds for slicing.  If *None*, uses all.
        future_len: Future length in seconds for slicing.  If *None*, uses all.
        fps: Frames per second.

    Returns:
        (T, 3) float tensor of future positions in ego frame.
    """
    history_xyz = action_data["history_xyz"]
    history_quat = action_data["history_quat"]
    future_xyz = action_data["future_xyz"]

    if not isinstance(history_xyz, torch.Tensor):
        history_xyz = torch.tensor(history_xyz, dtype=torch.float32)
    if not isinstance(history_quat, torch.Tensor):
        history_quat = torch.tensor(history_quat, dtype=torch.float32)
    if not isinstance(future_xyz, torch.Tensor):
        future_xyz = torch.tensor(future_xyz, dtype=torch.float32)

    # Slice to match the requested durations
    if history_len is not None:
        history_steps = int(history_len * fps)
        history_xyz = history_xyz[-history_steps:]
        history_quat = history_quat[-history_steps:]
    if future_len is not None:
        future_steps = int(future_len * fps)
        future_xyz = future_xyz[:future_steps]

    # Current pose = last history frame
    current_pos = history_xyz[-1]  # (3,)
    current_quat_wxyz = history_quat[-1]  # (4,) [w, x, y, z]

    # Scipy expects [x, y, z, w]
    quat_xyzw = current_quat_wxyz[[1, 2, 3, 0]].numpy()
    rot_world_to_ego = Rotation.from_quat(quat_xyzw).inv()

    # Translate then rotate into ego frame
    future_rel = (future_xyz - current_pos[None, :]).numpy()
    future_ego = rot_world_to_ego.apply(future_rel).astype(np.float32)

    return torch.from_numpy(future_ego)


class AVDataset(IterableDataset):
    """AV dataset that reads tar files from S3 using wdinfo.json."""

    def __init__(
        self,
        root: str | list[str] = "s3://nv-00-10206-robot/cosmos3_action_data/av_v2_02182026_wdinfo/",
        credential_path: str = "credentials/gcp_training.secret",
        resolution: str | None = None,
        fps: int = 10,
        mode: str = "policy",
        embodiment_type: str = "av",
        split: str = "train",
        seed: int = 0,
        shuffle: bool = True,
        history_len: float | None = None,
        future_len: float | None = None,
        rotation_format: RotationConvention = "rot9d",
        pose_convention: Literal["backward_anchored", "backward_framewise"] = ("backward_framewise"),
        route_lat_noise_range: float = 0.0,
        route_long_noise_range: float = 0.0,
        route_point_wise_noise: float = 0.0,
        route_dropout: bool = False,
        route_dropout_rate: float = 0.0,
        route_tail_dropout_rate: float = 0.0,
        include_route_in_prompt: bool = True,
        use_semantic_route_prompt: bool = False,
        translation_scale: float = 1.0,
        rotation_scale: float = 1.0,
        max_action_translation_norm: float | None = None,
        align_opencv_pose: bool = False,
        # When True, use a separate domain ID for inverse dynamics / policy modes
        # so that DomainAwareLinear learns different projections for anchored (conditioning)
        # vs framewise (generation) action representations.
        mode_aware_domain: bool = False,
        inv_embodiment_type: str = "av_inv",
    ):
        """Initialize AVDataset.

        Args:
            root: S3 path (or list of S3 paths) to wdinfo directories containing train/val subdirectories with wdinfo.json files.
            credential_path: Path to JSON file containing S3 credentials.
            resolution: Target resolution for video frames (e.g. "256", "480"). If None, keeps original resolution.
            fps: Target frames per second for video and actions.
            mode: Training mode ('policy', 'forward_dynamics', 'inverse_dynamics', 'image2video', 'joint').
            embodiment_type: Embodiment type for domain ID.
            split: Dataset split ('train', 'val', or 'full').
            seed: Random seed for shuffling.
            shuffle: Whether to shuffle tar files during iteration (for training).
            history_len: Desired history length in seconds. If None, uses all available history.
            future_len: Desired future length in seconds. If None, uses all available future.
            rotation_format: Rotation convention for actions (e.g. "rot9d", "rot6d", "euler_xyz").
            pose_convention: Pose format for actions (e.g. "backward_framewise", "backward_framewise").
            route_lat_noise_range: Half-range for uniform lateral (Y) noise on route waypoints.
            route_long_noise_range: Half-range for uniform longitudinal (X) noise on route waypoints.
            route_point_wise_noise: Std-dev of per-waypoint Gaussian noise on route.
            route_dropout: Whether to apply random waypoint dropout on route during training.
            route_dropout_rate: Probability of fully masking the route (used when route_dropout=True).
            route_tail_dropout_rate: Probability of additional tail dropout (used when route_dropout=True).
            include_route_in_prompt: Whether to include route waypoints as text in the prompt.
            use_semantic_route_prompt: When True and include_route_in_prompt is True, replace raw
                coordinate waypoints with a brief semantic description (e.g. "go forward then turn left").
            translation_scale: Scale factor applied to the translation block of the encoded action.
            rotation_scale: Scale factor applied to the rotation block of the encoded action
                (uniform scalar, preserves rotation-block geometry). Pass the same value to
                `pose_rel_to_abs` when decoding.
            max_action_translation_norm: If set, discard the sample when any per-frame
                scaled translation L2 norm exceeds this value.  Acts as an outlier
                filter to prevent loss spikes from extreme camera motion.
            align_opencv_pose: If True, transform pose rotations from car body-frame
                convention (x=forward, y=left, z=up) to OpenCV camera convention
                (x=right, y=down, z=forward) before computing relative actions.
            mode_aware_domain: When True, inverse_dynamics/policy modes use a separate domain ID.
            inv_embodiment_type: Embodiment type string for the inverse domain ID.
        """
        super().__init__()

        if isinstance(root, str):
            root = [root]
        self.roots = [r.rstrip("/") for r in root]
        self.credential_path = credential_path
        self.resolution = resolution
        self.fps = fps
        self.mode = mode
        self.split = split.lower().strip()
        self.seed = seed
        self.shuffle = shuffle
        self._epoch = 0
        self.history_len = history_len
        self.future_len = future_len
        self.rotation_format: RotationConvention = rotation_format
        self.pose_convention: Literal["absolute", "backward_anchored", "backward_framewise"] = pose_convention
        self.route_lat_noise_range = route_lat_noise_range
        self.route_long_noise_range = route_long_noise_range
        self.route_point_wise_noise = route_point_wise_noise
        self.route_dropout = route_dropout
        self.route_dropout_rate = route_dropout_rate
        self.route_tail_dropout_rate = route_tail_dropout_rate
        self.include_route_in_prompt = include_route_in_prompt
        self.use_semantic_route_prompt = use_semantic_route_prompt
        self.translation_scale = translation_scale
        self.rotation_scale = rotation_scale
        self.max_action_translation_norm = max_action_translation_norm
        self.align_opencv_pose = align_opencv_pose
        # Get domain ID for this embodiment
        self.domain_id = get_domain_id(embodiment_type)
        self.mode_aware_domain = mode_aware_domain
        self.domain_id_inv = get_domain_id(inv_embodiment_type) if mode_aware_domain else self.domain_id

        # Validate mode
        valid_modes = ["joint", "forward_dynamics", "inverse_dynamics", "policy", "image2video"]
        if mode not in valid_modes:
            raise ValueError(f"mode must be one of {valid_modes}, got {mode}")

        # Validate split
        if self.split not in {"train", "val", "valid", "validation", "eval", "test", "full"}:
            raise ValueError(f"Unsupported {split=}. Use train/val/full.")

        # Validate S3 roots
        for r in self.roots:
            if not r.startswith("s3://"):
                raise ValueError(f"root must be an S3 path starting with 's3://', got: {r}")

        # Configure S3 backend using easy_io
        self._setup_s3_backend()

        # Load tar files from wdinfo.json
        self._tar_files: list[str] = []
        self._total_key_count: int = 0
        self._chunk_size: int = 10

        self._load_wdinfo()

        log.info(
            f"Initialized AVDataset: root={self.roots}, split={self.split}, "
            f"resolution={resolution}, fps={fps}, mode={mode}, "
            f"num_tar_files={len(self._tar_files)}, "
            f"total_samples={self._total_key_count}"
        )

    def _setup_s3_backend(self) -> None:
        """Configure the easy_io S3 backend. Called in __init__ and __iter__ for worker processes."""
        easy_io.set_s3_backend(
            backend_args={
                "backend": "s3",
                "path_mapping": None,
                "s3_credential_path": self.credential_path,
            }
        )

    def _load_wdinfo(self) -> None:
        """Load wdinfo.json for the current split from all roots and build tar file list.

        Supports two directory layouts per root:
          - Split-based: ``{root}/train/wdinfo.json``, ``{root}/val/wdinfo.json``
          - Flat: ``{root}/wdinfo.json`` (treated as train-only)

        Split-based paths are tried first; the flat path is used as a fallback
        only when no split-based wdinfo was found and the requested split
        includes "train".
        """
        self._tar_files = []
        self._total_key_count = 0

        for root in self.roots:
            bucket = root.replace("s3://", "").split("/")[0]

            # Determine which splits we need
            if self.split in {"val", "valid", "validation", "eval", "test"}:
                target_splits = ["val"]
            elif self.split == "train":
                target_splits = ["train"]
            elif self.split == "full":
                target_splits = ["train", "val"]
            else:
                raise ValueError(f"Unsupported split: {self.split}")

            # Try split-based layout first ({root}/train/wdinfo.json, {root}/val/wdinfo.json)
            wdinfo_entries: list[tuple[str, dict]] = []
            for split_name in target_splits:
                split_path = f"{root}/{split_name}/wdinfo.json"
                try:
                    wdinfo_entries.append((split_path, json.loads(easy_io.get(split_path))))
                except Exception:
                    pass

            # Fall back to flat layout ({root}/wdinfo.json, treated as train-only)
            if not wdinfo_entries and "train" in target_splits:
                flat_path = f"{root}/wdinfo.json"
                try:
                    wdinfo_entries.append((flat_path, json.loads(easy_io.get(flat_path))))
                except Exception:
                    pass

            if not wdinfo_entries:
                log.warning(f"No wdinfo.json found for root={root}, split={self.split}")

            for wdinfo_path, wdinfo in wdinfo_entries:
                log.info(f"Loading wdinfo from: {wdinfo_path}")

                # Extract metadata
                self._chunk_size = wdinfo.get("chunk_size", 10)
                data_root = wdinfo.get("root", "")
                data_list = wdinfo.get("data_list", [])

                if not data_list:
                    log.warning(f"No tar files found in wdinfo: {wdinfo_path}")
                    continue

                # Reconstruct full S3 paths for tar files
                tar_root = f"s3://{bucket}/{data_root}".rstrip("/")
                tar_paths = [f"{tar_root}/{filename}" for filename in data_list]
                self._tar_files.extend(tar_paths)

                # Accumulate total sample count
                self._total_key_count += wdinfo.get("total_key_count", len(data_list) * self._chunk_size)

                log.info(
                    f"Loaded {len(data_list)} tar files from wdinfo, "
                    f"with {wdinfo.get('total_key_count', len(data_list) * self._chunk_size)} samples"
                )

        if not self._tar_files:
            raise RuntimeError(f"No tar files found in wdinfo at {self.roots}")

    def __len__(self) -> int:
        """Return the estimated number of samples in the current split."""
        return self._total_key_count

    def _process_sample(self, pkl_data: dict, key: str, global_idx: int) -> dict | None:
        """Process a single sample from pkl data.

        Args:
            pkl_data: Dictionary with 'video' (bytes) and 'action' (pickled dict).
            key: Sample key (basename without .pkl).
            global_idx: Global sample index for __key__.

        Returns:
            Processed sample dictionary, or None if the sample should be discarded.
        """
        # Extract video bytes
        video_bytes = pkl_data.get("video")
        if video_bytes is None:
            raise RuntimeError(f"No video found for key {key}")

        # Extract action data
        action_bytes = pkl_data.get("action")
        if action_bytes is None:
            raise RuntimeError(f"Missing action data for key {key}")

        action_data = pickle.loads(action_bytes)

        # Extract route data
        route_bytes = pkl_data.get("route")
        if route_bytes is not None:
            route_data = pickle.loads(route_bytes)
            if not isinstance(route_data, torch.Tensor):
                route = torch.tensor(route_data, dtype=torch.float32)  # [num_waypoints,3]
            else:
                route = route_data.float()  # [num_waypoints,3]
        else:
            log.warning(f"No route found for key {key}")
            route = torch.full((20, 3), float("nan"))  # [20,3]

        # Apply route augmentations during training
        if self.split == "train":
            route = add_route_noise(
                route,
                lat_noise_range=self.route_lat_noise_range,
                long_noise_range=self.route_long_noise_range,
                point_wise_noise=self.route_point_wise_noise,
            )
            if self.route_dropout:
                route = apply_route_dropout(
                    route,
                    dropout_rate=self.route_dropout_rate,
                    tail_dropout_rate=self.route_tail_dropout_rate,
                )

        # Get original history frame count for video slicing
        original_history_steps = len(action_data["history_xyz"])

        # Decode video
        video, _ = decode_video_bytes(
            video_bytes,
            resolution=self.resolution,
            history_len=self.history_len,
            future_len=self.future_len,
            original_history_steps=original_history_steps,
        )

        # Determine mode for this sample
        if self.mode == "joint":
            mode = random.choice(["forward_dynamics", "inverse_dynamics", "policy"])
            # mode = random.choice(["policy", "image2video"])
        else:
            mode = self.mode

        # Process actions
        action_result = process_action_trajectory(
            action_data,
            history_len=self.history_len,
            future_len=self.future_len,
            fps=self.fps,
            rotation_format=self.rotation_format,
            pose_convention=self.pose_convention,
            scale=self.translation_scale,
            rotation_scale=self.rotation_scale,
            max_translation_norm=self.max_action_translation_norm,
            align_opencv_pose=self.align_opencv_pose,
        )
        if action_result is None:
            return None
        history_action, future_action = action_result

        # Combine and pad actions
        combined_action = torch.cat([history_action, future_action], dim=0)  # [T_hist+T_fut,action_dim]

        # FPS as tensor
        fps_tensor = torch.tensor(self.fps, dtype=torch.long)  # scalar

        # Key as tensor
        key_tensor = torch.tensor([global_idx], dtype=torch.long)  # [1]

        # Compute actual history/future lengths from data
        actual_history_length = history_action.shape[0]
        actual_future_length = future_action.shape[0]

        # Generate prompt based on actual data lengths
        history_duration = actual_history_length / self.fps
        future_duration = actual_future_length / self.fps

        prompt = "You are an autonomous vehicle planning system. "
        if self.include_route_in_prompt and mode == "policy":  # only include route in prompt for policy mode
            if self.use_semantic_route_prompt:
                future_ego = compute_future_trajectory_in_ego_frame(
                    action_data, self.history_len, self.future_len, self.fps
                )
                trajectory_desc = classify_trajectory_to_text(future_ego)
                prompt += f"Please {trajectory_desc}. "
            else:
                num_waypoints = route.shape[0]
                waypoints_str = ", ".join(
                    "nan" if torch.isnan(wp[0]) else f"({wp[0]:.2f}, {wp[1]:.2f}, {wp[2]:.2f})" for wp in route
                )
                prompt += (
                    f"The navigation route has {num_waypoints} waypoints "
                    f"(XYZ in ego frame with X=forward, Y=left, Z=up): "
                    f"[{waypoints_str}]. A nan waypoint means that waypoint is not available. "
                )
        # prompt += f"Predict the future {future_duration:.1f}s action trajectory at {self.fps}Hz."

        # Select domain ID: use inverse domain for generation modes when mode_aware_domain is on
        if self.mode_aware_domain and mode in ["inverse_dynamics", "policy"]:
            domain_id = self.domain_id_inv
        else:
            domain_id = self.domain_id

        sample = {
            "video": video,
            "action": combined_action,
            "action_history": history_action,
            "action_future": future_action,
            "route": route,
            "conditioning_fps": fps_tensor,
            "prompt": prompt,
            "ai_caption": prompt,
            "mode": mode,
            "__key__": key_tensor,
            "domain_id": torch.tensor(domain_id, dtype=torch.long),
            "history_length": actual_history_length,
            "future_length": actual_future_length,
            "viewpoint": "ego_view",
        }
        return sample

    def __iter__(self) -> Iterator[dict[str, torch.Tensor | str | int]]:
        """Iterate over the dataset, loading tar files from S3."""
        # Re-configure S3 backend in case this is running in a worker process after unpickling
        self._setup_s3_backend()

        # Optionally shuffle tar files for training
        tar_files = list(self._tar_files)
        if self.shuffle:
            rng = random.Random(self.seed + self._epoch)
            rng.shuffle(tar_files)
            self._epoch += 1

        global_idx = 0

        for tar_path in tar_files:
            try:
                # Read tar file bytes using easy_io
                tar_bytes = easy_io.get(tar_path)

                with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:*") as tar:
                    for member in tar.getmembers():
                        if not member.name.endswith(".pkl"):
                            continue

                        try:
                            # Extract and process the sample
                            f_member = tar.extractfile(member)
                            if f_member is None:
                                log.warning(f"Failed to extract {member.name} from {tar_path}")
                                continue

                            try:
                                pkl_data = pickle.load(f_member)
                            finally:
                                f_member.close()

                            key = member.name.rsplit(".", 1)[0]

                            sample = self._process_sample(pkl_data, key, global_idx)
                            if sample is not None:
                                yield sample
                                global_idx += 1

                        except Exception as e:
                            log.warning(f"Failed to process sample {member.name} from {tar_path}: {e}")
                            continue

            except Exception as e:
                log.warning(f"Failed to read tar file {tar_path}: {e}")
                continue


# PYTHONPATH=. python cosmos/data/vfm/action/av_dataset.py
if __name__ == "__main__":
    import json as _json
    import os
    import time

    import torchvision

    from cosmos.data.vfm.action.pose_utils import pose_rel_to_abs

    _ACTION_SCALE = 1.35
    _ROTATION_SCALE = 1.0
    _ROTATION_FORMAT = "rot6d"
    _POSE_CONVENTION = "backward_framewise"

    dataset = AVDataset(
        root=[
            # "s3://nv-00-10206-robot/cosmos3_action_data/av_02182026_wdinfo/",
            # "s3://nv-00-10206-robot/cosmos3_action_data/av_03292026_wdinfo/",
            "s3://nv-00-10206-robot/cosmos3_action_data/av_v2_02182026_wdinfo/",
            "s3://nv-00-10206-robot/cosmos3_action_data/av_v2_03292026_wdinfo/",
        ],
        split="train",
        shuffle=True,
        fps=10,
        mode="inverse_dynamics",
        history_len=0.1,
        future_len=6.0,
        rotation_format=_ROTATION_FORMAT,
        pose_convention=_POSE_CONVENTION,
        translation_scale=_ACTION_SCALE,
        rotation_scale=_ROTATION_SCALE,
        resolution="480",
        include_route_in_prompt=False,
        use_semantic_route_prompt=False,
        # align_opencv_pose=False,
    )
    dataset_iter = iter(dataset)
    os.makedirs("temp", exist_ok=True)

    for i in range(5):
        print(f"==================== Sample {i} ====================")
        _t0 = time.time()
        data = next(dataset_iter)
        _t1 = time.time()
        print(f"{'Loading time':<25}: {_t1 - _t0:.2f}s")

        print(f"{'video shape':<25}: {data['video'].shape}")  # [C,T,H,W]
        print(f"{'action shape':<25}: {data['action'].shape}")  # [T,action_dim]
        print(f"{'action_history shape':<25}: {data['action_history'].shape}")
        print(f"{'action_future shape':<25}: {data['action_future'].shape}")
        print(f"{'route shape':<25}: {data['route'].shape}")
        print(f"{'history_length':<25}: {data['history_length']}")
        print(f"{'future_length':<25}: {data['future_length']}")
        print(f"{'conditioning_fps':<25}: {data['conditioning_fps'].item()}")
        print(f"{'mode':<25}: {data['mode']}")
        print(f"{'domain_id':<25}: {data['domain_id'].item()}")
        print(f"{'prompt':<25}: {data['prompt']}")

        # save video
        video = data["video"].permute(1, 0, 2, 3)  # [C,T,H,W] -> [T,C,H,W]
        video_path = f"temp/av_sample_{i}.mp4"
        torchvision.io.write_video(
            video_path, video.permute(0, 2, 3, 1).numpy(), fps=data["conditioning_fps"].item()
        )  # expects (T, H, W, C)
        print(f"Saved video to {video_path}")

        # reconstruct absolute poses from relative actions and save as json
        camera_poses = pose_rel_to_abs(
            data["action"].float().numpy(),
            rotation_format=_ROTATION_FORMAT,
            pose_convention=_POSE_CONVENTION,
            translation_scale=_ACTION_SCALE,
            rotation_scale=_ROTATION_SCALE,
        )
        pose_path = f"temp/av_sample_{i}_camera.json"
        with open(pose_path, "w") as f:
            _json.dump(camera_poses.tolist(), f)
        print(f"Saved camera poses to {pose_path}")
