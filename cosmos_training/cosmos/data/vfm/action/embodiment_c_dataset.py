"""Embodiment C datasets: multi-task LeRobot with instruction annotations from meta/info.json.

Provides gripper variants.

Action representation:
    - **Relative FK-pose**: calibrated head camera pose + left/right
      gripper-base wrist poses are computed via URDF forward kinematics from
      ``observation.state``. Gripper wrist rotations are then converted through
      the dataset ``_to_opencv`` mapping for action/viewer display. The viewer
      can optionally request source-state FK link poses for direct mesh playback.

View modes:
    - **ego_view**: single ``observation.images.top_head`` camera.
    - **concat_view**: top-head view on top, left/right wrist views resized and
      concatenated horizontally on the bottom (like DROID).
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import torch
import torch.nn.functional as F
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

from cosmos.utils import log
from cosmos.data.vfm.action.embodiment_c_dataset_config import (
    CUSTOM_GRIPPER_SUBPATHS,
    DEFAULT_CUSTOM_GRIPPER_EXT_ROOT,
    DEFAULT_CUSTOM_GRIPPER_ROOT,
    DEFAULT_OFFSHELF_GRIPPER_ROOT,
    OFFSHELF_GRIPPER_SUBPATHS,
)
from cosmos.data.vfm.action.embodiment_c_fk import (
    AGIBOT_GEAR_GRIPPER_TO_OPENCV_BY_WRIST,
    apply_agibot_gripper_to_opencv,
    compute_fk_transforms_batch,
    compute_link_poses_batch,
    convert_gripper_state_to_open_fraction,
    extract_fk_transforms_from_link_poses,
)
from cosmos.data.vfm.action.embodiment_c_spec import (
    AGIBOT_GEAR_EXT_STATE_LEFT_HAND_SLICE,
    AGIBOT_GEAR_EXT_STATE_RIGHT_HAND_SLICE,
    AGIBOT_GEAR_GRIPPER_NORMALIZER_EMBODIMENT_TYPE,
    AGIBOT_GEAR_VIDEO_KEY,
    get_embodiment_c_kind_spec,
)
from cosmos.data.vfm.action.cosmos3_action_lerobot import (
    ActionNormalization,
    ActionSpec,
    BaseActionLeRobotDataset,
    Gripper,
    Pos,
    Rot,
    build_action_spec,
    split_episode_ids,
)
from cosmos.data.vfm.action.pose_utils import PoseConvention, pose_abs_to_rel
from cosmos.data.vfm.action.viewpoint_utils import Viewpoint

# Camera keys for concat_view.
_TOP_HEAD_KEY = AGIBOT_GEAR_VIDEO_KEY
_HAND_LEFT_KEY = "observation.images.hand_left"
_HAND_RIGHT_KEY = "observation.images.hand_right"


def _resolve_root_paths(
    root: str | list[str] | tuple[str, ...],
    expand_root_fn: Callable[[str], list[str]],
) -> list[str]:
    """Normalize a root argument into the list of concrete shard roots."""

    if isinstance(root, str):
        return expand_root_fn(root)
    resolved_roots: list[str] = []
    for item in root:
        resolved_roots.extend(expand_root_fn(item))
    return resolved_roots


def _build_delta_timestamps(video_key: str, chunk_length: int, dt: float) -> dict[str, list[float]]:
    """Build the LeRobot delta-timestamp layout for AgiBot samples."""

    return {
        video_key: [index * dt for index in range(0, chunk_length + 1)],
        "observation.state": [index * dt for index in range(0, chunk_length + 1)],
    }


def _load_instruction_segments(lerobot_root: str) -> dict[str, list[dict[str, Any]]]:
    """Load instruction_segments from meta/info.json for a task path."""
    info_path = Path(lerobot_root) / "meta" / "info.json"
    if not info_path.is_file():
        return {}
    try:
        with open(info_path) as f:
            info = json.load(f)
        segments = info.get("instruction_segments", {})
        return segments if isinstance(segments, dict) else {}
    except Exception as e:
        log.warning(f"EmbodimentCGripperDataset: failed to load {info_path}: {e}")
        return {}


def _extract_high_level_instruction(entry: Any) -> str | None:
    """Extract the episode-level instruction text from one metadata entry."""

    if isinstance(entry, str):
        text = entry.strip()
        return text or None
    if isinstance(entry, dict):
        value = entry.get("high_level_instruction")
        if isinstance(value, str):
            text = value.strip()
            return text or None
    return None


def _load_high_level_instructions(lerobot_root: str) -> dict[str, str]:
    """Load high_level_instruction from meta/info.json for a task path."""

    info_path = Path(lerobot_root) / "meta" / "info.json"
    if not info_path.is_file():
        return {}
    try:
        with open(info_path) as f:
            info = json.load(f)
        raw_instructions = info.get("high_level_instruction", {})
        if not isinstance(raw_instructions, dict):
            return {}
        high_level_instructions: dict[str, str] = {}
        for episode_key, entry in raw_instructions.items():
            instruction = _extract_high_level_instruction(entry)
            if instruction is not None:
                high_level_instructions[str(episode_key)] = instruction
        return high_level_instructions
    except Exception as e:
        log.warning(f"EmbodimentCGripperDataset: failed to load high_level_instruction from {info_path}: {e}")
        return {}


def _coerce_frame_index(value: Any) -> int | None:
    """Convert a frame index value to ``int`` when possible."""

    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _get_instruction_frame_bounds(segment: dict[str, Any]) -> tuple[int, int] | None:
    """Extract inclusive ``[start,end]`` frame bounds from one instruction segment."""

    start = _coerce_frame_index(segment.get("start_frame_index"))
    end = _coerce_frame_index(segment.get("end_frame_index", segment.get("success_frame_index")))
    if start is None or end is None or end < start:
        return None
    return start, end


def _get_gripper_state_slices(embodiment_type: str, kind_spec: Any) -> tuple[slice, slice]:
    """Return ``(right,left)`` state slices for scalar AgiBot gripper positions."""

    if embodiment_type == "embodiment_c_gripper_ext":
        return AGIBOT_GEAR_EXT_STATE_RIGHT_HAND_SLICE, AGIBOT_GEAR_EXT_STATE_LEFT_HAND_SLICE
    state_hand_slice = kind_spec.state_hand_slice
    return (
        slice(state_hand_slice.start + 1, state_hand_slice.start + 2),
        slice(state_hand_slice.start, state_hand_slice.start + 1),
    )


def _build_instruction_episode_spans(
    *,
    episodes: Any,
    episode_ids: list[int],
    instruction_segments: dict[str, list[dict[str, Any]]],
    high_level_instructions: dict[str, str],
    chunk_length: int,
    sample_stride: int = 1,
) -> tuple[list[tuple[int, int, int]], dict[int, int], int, int, int, int]:
    """Build valid spans from instruction segments, with episode-level fallback."""

    assert sample_stride >= 1, f"sample_stride must be >= 1, got {sample_stride}"

    dataset_from_index = list(episodes["dataset_from_index"])
    dataset_to_index = list(episodes["dataset_to_index"])
    length = list(episodes["length"])

    spans: list[tuple[int, int, int]] = []
    episode_start_rows: dict[int, int] = {}
    valid_count = 0
    sample_count = 0
    missing_episode_count = 0
    fallback_episode_count = 0

    for episode_id in episode_ids:
        episode_start = int(dataset_from_index[episode_id])
        episode_stop = int(dataset_to_index[episode_id])
        episode_length = int(length[episode_id])
        episode_start_rows[episode_id] = episode_start
        sample_count += episode_length

        raw_segments = instruction_segments.get(str(episode_id), [])
        if not raw_segments:
            missing_episode_count += 1
            if str(episode_id) in high_level_instructions:
                max_episode_frame = min(episode_length - 1, episode_stop - episode_start - 1)
                valid_len = max_episode_frame - chunk_length + 1
                if valid_len > 0:
                    strided_valid_len = (valid_len + sample_stride - 1) // sample_stride
                    spans.append((episode_id, episode_start, strided_valid_len))
                    valid_count += strided_valid_len
                    fallback_episode_count += 1
            continue

        max_episode_frame = min(episode_length - 1, episode_stop - episode_start - 1)
        for segment in raw_segments:
            bounds = _get_instruction_frame_bounds(segment)
            if bounds is None:
                continue

            segment_start, segment_end = bounds
            segment_start = max(0, segment_start)
            segment_end = min(max_episode_frame, segment_end)
            valid_len = segment_end - segment_start - chunk_length + 1
            if valid_len <= 0:
                continue

            strided_valid_len = (valid_len + sample_stride - 1) // sample_stride
            spans.append((episode_id, episode_start + segment_start, strided_valid_len))
            valid_count += strided_valid_len

    return spans, episode_start_rows, valid_count, sample_count, missing_episode_count, fallback_episode_count


class EmbodimentCGripperDataset(BaseActionLeRobotDataset):
    """Embodiment C Gripper dataset with deferred source registration.

    Sources are registered by ``_register_sources()`` which is called by
    ``ActionUnifiedIterableDataset.assign_worker()`` during training, or
    explicitly for standalone/eval use. Instruction segments and episode-level
    high-level instructions are loaded alongside each source.

    Action layout:
        ``[head_cam_delta(9), right_hand_delta(9), right_gripper(1),
          left_hand_delta(9), left_gripper(1)]``  → 29 dims for gripper.

    Concat view layout:
        ┌──────────────────┐
        │    top_head      │  (H, W)
        ├─────────┬────────┤
        │ hand_L  │ hand_R │  (H/2, W/2) each
        └─────────┴────────┘
    """

    def __init__(
        self,
        root: str | list[str] | tuple[str, ...] = DEFAULT_OFFSHELF_GRIPPER_ROOT,
        fps: float = 10.0,
        chunk_length: int = 16,
        split_seed: int = 42,
        split_val_ratio: float = 0.005,
        split: str = "train",
        mode: str = "joint",
        pose_convention: PoseConvention = "backward_framewise",
        embodiment_type: str = "embodiment_c_gripper",
        tolerance_s: float = 1e-3,
        video_key: str = AGIBOT_GEAR_VIDEO_KEY,
        action_normalization: ActionNormalization | None = None,
        viewpoint: Viewpoint = "concat_view",
        rotation_format: Literal["rot6d"] = "rot6d",
        max_loaded_datasets: int = 32,
        skip_video_loading: bool = False,
        sample_stride: int = 1,
        enable_fast_init: bool = False,
        fast_init_max_workers: int = 64,
        return_agibot_link_poses: bool = False,
    ) -> None:
        super().__init__(
            fps=fps,
            chunk_length=chunk_length,
            split_seed=split_seed,
            split_val_ratio=split_val_ratio,
            split=split,
            mode=mode,
            embodiment_type=embodiment_type,
            viewpoint=viewpoint,
            pose_convention=pose_convention,
            rotation_format=rotation_format,
            action_normalization=action_normalization,
            tolerance_s=tolerance_s,
            max_loaded_datasets=max_loaded_datasets,
            skip_video_loading=skip_video_loading,
            sample_stride=sample_stride,
            enable_fast_init=enable_fast_init,
            fast_init_max_workers=fast_init_max_workers,
        )
        self._video_key = video_key
        self._kind_spec = get_embodiment_c_kind_spec(embodiment_type)
        self._is_concat_view = viewpoint == "concat_view"
        self._to_opencv: dict[str, np.ndarray] = AGIBOT_GEAR_GRIPPER_TO_OPENCV_BY_WRIST
        self._return_agibot_link_poses: bool = return_agibot_link_poses

        self._all_shard_roots = _resolve_root_paths(root, self._expand_root)
        if not self._all_shard_roots:
            raise ValueError(
                "EmbodimentCGripperDataset: no task directories found under root. "
                "Point root to a dir with task_* subdirs, or to a single LeRobot path with meta/info.json."
            )

        frame_ts = [index * self._dt for index in range(0, self._chunk_length + 1)]
        self._delta_timestamps = _build_delta_timestamps(self._video_key, self._chunk_length, self._dt)
        if self._is_concat_view:
            self._delta_timestamps[_HAND_LEFT_KEY] = frame_ts
            self._delta_timestamps[_HAND_RIGHT_KEY] = frame_ts

        self._instruction_segments: list[dict[str, list[dict[str, Any]]]] = []
        self._high_level_instructions: list[dict[str, str]] = []
        self._task_root_paths: list[str] = []
        self._episode_start_rows: list[dict[int, int]] = []

    def _normalizer_filename(self) -> str:
        """Resolve Embodiment C gripper stats, sharing them with compatible ext data."""

        if self._embodiment_type == "embodiment_c_gripper_ext":
            return (
                f"{AGIBOT_GEAR_GRIPPER_NORMALIZER_EMBODIMENT_TYPE}_{self._pose_convention}_{self._rotation_format}.json"
            )
        return super()._normalizer_filename()

    def _expand_root(self, root: str) -> list[str]:
        """Expand a base root path to task-specific subpaths."""
        root_name = Path(root.rstrip("/")).name
        if root_name == Path(DEFAULT_OFFSHELF_GRIPPER_ROOT).name:
            return [os.path.join(root, sub) for sub in OFFSHELF_GRIPPER_SUBPATHS]
        if root_name == Path(DEFAULT_CUSTOM_GRIPPER_ROOT).name:
            return [os.path.join(root, sub) for sub in CUSTOM_GRIPPER_SUBPATHS]
        return [root]

    def _register_sources(self, indices: list[int] | None = None) -> None:
        if indices is None:
            indices = list(range(len(self._all_shard_roots)))
        for idx in indices:
            path = self._all_shard_roots[idx]
            instruction_segments = _load_instruction_segments(path)
            high_level_instructions = _load_high_level_instructions(path)
            self._task_root_paths.append(path)
            self._instruction_segments.append(instruction_segments)
            self._high_level_instructions.append(high_level_instructions)
            try:
                self._register_source(
                    root=path,
                    delta_timestamps=self._delta_timestamps,
                    tolerance_s=self._tolerance_s,
                    dataset_label=Path(path).name,
                )
            except Exception as e:
                self._task_root_paths.pop()
                self._instruction_segments.pop()
                self._high_level_instructions.pop()
                log.warning(f"EmbodimentCGripperDataset: failed to load LeRobotDatasetMetadata at {path}: {e}")
                continue

    def _append_index_records(
        self,
        *,
        meta: LeRobotDatasetMetadata,
        ds_idx: int,
        dataset_label: str | None = None,
    ) -> None:
        """Populate index records using only frame starts that stay inside instruction segments."""

        episode_ids = split_episode_ids(
            total_episodes=meta.total_episodes,
            seed=self._split_seed,
            val_ratio=self._split_val_ratio,
            split=self._split,
        )
        instruction_segments = self._instruction_segments[ds_idx]
        high_level_instructions = self._high_level_instructions[ds_idx]
        episode_spans, episode_start_rows, valid_count, sample_count, missing_episode_count, fallback_episode_count = (
            _build_instruction_episode_spans(
                episodes=meta.episodes,
                episode_ids=episode_ids,
                instruction_segments=instruction_segments,
                high_level_instructions=high_level_instructions,
                chunk_length=self._chunk_length,
                sample_stride=self._sample_stride,
            )
        )

        while len(self._episode_start_rows) <= ds_idx:
            self._episode_start_rows.append({})
        self._episode_start_rows[ds_idx] = episode_start_rows

        class_name = self.__class__.__name__
        label = f" [{dataset_label}]" if dataset_label else ""
        log.info(f"{class_name}{label}: split={self._split}, num episodes={len(episode_ids)}")
        if sample_count > 0:
            log.info(
                f"{class_name}{label}: kept {valid_count} / {sample_count} "
                f"({100 * valid_count / sample_count:.2f} %) instruction-filtered samples"
            )
        if missing_episode_count > 0:
            if fallback_episode_count == missing_episode_count:
                log.info(
                    f"{class_name}{label}: missing instruction_segments for {missing_episode_count} / "
                    f"{len(episode_ids)} episodes; using high_level_instruction fallback for all of them"
                )
            else:
                log.warning(
                    f"{class_name}{label}: missing instruction_segments for {missing_episode_count} / "
                    f"{len(episode_ids)} episodes; high_level_instruction fallback covered "
                    f"{fallback_episode_count} of them"
                )

        for episode_id, sample_start, valid_len in episode_spans:
            self._episode_records.append((ds_idx, sample_start, valid_len, episode_id))
            self._num_valid_indices += valid_len
            self._episode_cum_ends.append(self._num_valid_indices)

    def _get_ai_caption(self, ds_idx: int, episode_id: int, frame_index_in_episode: int) -> str:
        segments = self._instruction_segments[ds_idx]
        high_level_instructions = self._high_level_instructions[ds_idx]
        ep_key = str(episode_id)
        for seg in segments.get(ep_key, []):
            bounds = _get_instruction_frame_bounds(seg)
            if bounds is None:
                continue
            start, end = bounds
            if start <= frame_index_in_episode <= end:
                if "instruction" not in seg:
                    raise ValueError(
                        f"EmbodimentCGripperDataset: segment for episode_id={episode_id} "
                        f"frame_index_in_episode={frame_index_in_episode} missing 'instruction' key."
                    )
                return seg["instruction"]
        if ep_key in high_level_instructions:
            return high_level_instructions[ep_key]
        raise ValueError(
            f"EmbodimentCGripperDataset: no instruction annotation found for episode_id={episode_id} "
            f"frame_index_in_episode={frame_index_in_episode} (ds_idx={ds_idx})."
        )

    # -- FK-based action construction ----------------------------------------

    def _build_fk_action(self, sample: dict[str, Any]) -> tuple[torch.Tensor, dict[str, Any]]:
        """Build relative FK-pose action plus absolute initial poses for reconstruction.

        Uses ``observation.state`` across the full ``T+1`` conditioning window to
        compute absolute FK transforms, converts them to relative ``rot6d`` SE(3)
        deltas, then concatenates hand/gripper values from the next observed
        state frame for each delta. The source ``action`` column is intentionally
        not requested or read: for Embodiment C, the training/viewer action is the
        difference between consecutive observed states.

        Returns:
            ``(action, extras)`` where ``action`` has shape ``(T, action_dim)``
            and ``extras`` carries the absolute initial poses for the head and
            gripper-base wrist trajectories. When ``return_agibot_link_poses`` is
            enabled, ``extras`` also carries native URDF link poses for direct
            viewer mesh animation.

        Scalar hand actions are emitted in the shared viewer/action convention:
        ``0.0`` means closed and ``1.0`` means open.
        ``convert_gripper_state_to_open_fraction`` maps observed AgiBot gripper
        state, including URDF ``[-pi/4,0]`` radians or actuator-close
        ``[0,120]`` degrees, into that same open-fraction range.
        """
        obs_state = sample["observation.state"]  # [T+1,S]
        states_np = obs_state.detach().cpu().numpy().astype(np.float32, copy=False)  # [T+1,S]
        action_steps = int(states_np.shape[0] - 1)
        if action_steps <= 0:
            raise ValueError(f"{self.__class__.__name__}: observation.state must contain at least 2 frames.")
        link_poses = None
        if self._return_agibot_link_poses:
            link_poses = compute_link_poses_batch(states_np, self._embodiment_type)  # {name:[T+1,4,4]}
            native_fk = extract_fk_transforms_from_link_poses(link_poses)  # {name:[T+1,4,4]}
        else:
            native_fk = compute_fk_transforms_batch(states_np, self._embodiment_type)  # {name:[T+1,4,4]}
        fk = apply_agibot_gripper_to_opencv(native_fk, self._to_opencv)  # {name:[T+1,4,4]}
        pose_convention = cast(PoseConvention, self._pose_convention)
        head_rel = pose_abs_to_rel(fk["head_camera"], rotation_format="rot6d", pose_convention=pose_convention)  # [T,9]
        right_rel = pose_abs_to_rel(
            fk["right_wrist"], rotation_format="rot6d", pose_convention=pose_convention
        )  # [T,9]
        left_rel = pose_abs_to_rel(fk["left_wrist"], rotation_format="rot6d", pose_convention=pose_convention)  # [T,9]
        # Normalize observed AgiBot gripper state to the viewer/action
        # convention: 0.0=closed, 1.0=open. Actuator-close state values use
        # 0=open and 120=closed, so small open-state jitter stays near 1.0 open.
        right_state_slice, left_state_slice = _get_gripper_state_slices(self._embodiment_type, self._kind_spec)
        right_hand = convert_gripper_state_to_open_fraction(states_np[1:, right_state_slice])  # [T,1]
        left_hand = convert_gripper_state_to_open_fraction(states_np[1:, left_state_slice])  # [T,1]
        action_np = np.concatenate(
            [
                head_rel,  # [T,9]
                right_rel,  # [T,9]
                right_hand,  # [T,1|6]
                left_rel,  # [T,9]
                left_hand,  # [T,1|6]
            ],
            axis=-1,
        ).astype(np.float32, copy=False)  # [T,A]
        extras = {
            "initial_pose": torch.from_numpy(fk["head_camera"][0].copy()).float(),  # [4,4]
            "initial_pose_right": torch.from_numpy(fk["right_wrist"][0].copy()).float(),  # [4,4]
            "initial_pose_left": torch.from_numpy(fk["left_wrist"][0].copy()).float(),  # [4,4]
        }
        if link_poses is not None:
            agibot_link_poses = {
                link_name: torch.from_numpy(poses.copy()).float() for link_name, poses in link_poses.items()
            }  # {name:[T+1,4,4]}
            extras["agibot_link_poses"] = agibot_link_poses
        return torch.from_numpy(action_np).float(), extras  # [T,A]

    # -- Multi-view composition ----------------------------------------------

    def _compose_multi_view(self, sample: dict[str, Any]) -> torch.Tensor:
        """Compose top-head, left-hand, and right-hand views into a single frame.

        Layout (per frame):
            ┌──────────────────┐
            │     top_head     │   (H, W)
            ├─────────┬────────┤
            │ hand_L  │ hand_R │   (H/2, W/2) each
            └─────────┴────────┘

        Left and right hand cameras are downscaled by 2× so they tile to the
        same width as the top-head view.  Output height is 3H/2.

        Returns:
            Composited video tensor in raw LeRobot ``(T, C, H_out, W)`` float format.
        """
        top = sample[_TOP_HEAD_KEY]  # [T,C,H,W]
        left = sample[_HAND_LEFT_KEY]  # [T,C,H_l,W_l]
        right = sample[_HAND_RIGHT_KEY]  # [T,C,H_r,W_r]

        _, _, h_top, w_top = top.shape
        half_h, half_w = h_top // 2, w_top // 2

        left = F.interpolate(left, size=(half_h, half_w), mode="bilinear", align_corners=False)  # [T,C,H/2,W/2]
        right = F.interpolate(right, size=(half_h, half_w), mode="bilinear", align_corners=False)  # [T,C,H/2,W/2]
        bottom = torch.cat([left, right], dim=-1)  # [T,C,H/2,W]

        composite = torch.cat([top, bottom], dim=-2)  # [T,C,3H/2,W]
        return composite  # [T,C,3H/2,W]

    # -- __getitem__ ---------------------------------------------------------

    def _resolve_sample(self, idx: int) -> tuple[str, int, int, int, dict[str, Any]]:
        """Resolve a flat index to one dataset row and its associated metadata."""

        mode = self._choose_mode()
        dataset_idx, row_idx, episode_id, _ = self._resolve_index(int(idx))
        frame_index_in_episode = row_idx - self._episode_start_rows[dataset_idx][episode_id]
        sample = self._get_dataset(dataset_idx)[row_idx]
        return mode, dataset_idx, episode_id, frame_index_in_episode, sample

    def __getitem__(self, idx: int) -> dict[str, Any]:
        mode, dataset_idx, episode_id, frame_index_in_episode, sample = self._resolve_sample(idx)
        ai_caption = self._get_ai_caption(dataset_idx, episode_id, frame_index_in_episode)

        # Respect skip_video_loading (used by action-stats / eval scripts that
        # only need ``action``).  Base class strips ``observation.image*`` from
        # delta_timestamps in this mode, so ``sample`` does not contain
        # ``self._video_key`` and direct indexing would KeyError.  Mirrors the
        # droid / robomind pattern.
        if self._skip_video_loading:
            video = None
        # Video: concat_view or single ego_view.
        elif self._is_concat_view:
            video = self._compose_multi_view(sample)
        else:
            video = sample[self._video_key]  # [T,C,H,W]

        # Action: relative FK-pose based.
        action, action_extras = self._build_fk_action(sample)

        extras: dict[str, Any] = {**action_extras}
        if self._is_concat_view:
            extras["additional_view_description"] = (
                "The top row shows the head-mounted camera view looking down at the workspace. "
                "The bottom row contains two horizontally concatenated wrist-mounted camera views: "
                "the left hand camera on the left and the right hand camera on the right."
            )

        result = self._build_result(mode=mode, video=video, action=action, ai_caption=ai_caption, **extras)
        result["__episode_id__"] = episode_id
        result["__task_root__"] = self._task_root_paths[dataset_idx]
        return result

    def _build_action_spec(self) -> ActionSpec:
        """Embodiment C gripper bimanual layout (29D).

        ``[head_pos+rot6d (9) | right_pos+rot6d (9) | right_gripper (1)
                              | left_pos+rot6d  (9) | left_gripper  (1)]``

        All three SE(3) blocks (head camera + both wrists) participate in
        idle-frame detection: a chunk only counts as idle if the head is
        steady AND both arms are at rest AND both grippers are unchanged.
        Override this method (or use ``Reserved`` for head dims) if you want
        head motion to be ignored by idle detection.
        """
        return build_action_spec(
            Pos(prefix="head"),
            Rot("rot6d", prefix="head"),
            Pos(prefix="right"),
            Rot("rot6d", prefix="right"),
            Gripper(prefix="right"),
            Pos(prefix="left"),
            Rot("rot6d", prefix="left"),
            Gripper(prefix="left"),
        )

    @property
    def action_dim(self) -> int:
        return 29


class EmbodimentCGripperExtDataset(EmbodimentCGripperDataset):
    """Embodiment C Gripper Extended dataset for custom tasks.

    The source episodes use a different state layout (``state=[94]``) compared
    to the standard gripper (``state=[32]``). Source action columns are not used;
    emitted FK-pose actions are derived from consecutive observed states.

    Key layout differences handled here:

    **State (94-dim):**
        - Arm joint positions at ``[54:68]`` (vs ``[0:14]`` in standard).
        - Head yaw/pitch at ``[82:84]`` (vs ``[16]``/``[17]``).
        - Waist pitch/lift at ``[84:86]`` (vs ``[18]``/``[19]``).
        - Mobile-base position/quaternion at ``[86:93]``.

    The FK engine (``_extract_joint_values_from_state``) handles the state
    remapping for the ``embodiment_c_gripper_ext`` embodiment type and folds
    mobile-base motion into the head/gripper-base poses. The emitted FK-pose
    action is the same 29-dim layout as standard gripper:
    ``[head(9), right(9), right_grip(1), left(9), left_grip(1)]``.
    """

    def __init__(
        self,
        root: str | list[str] | tuple[str, ...] = DEFAULT_CUSTOM_GRIPPER_EXT_ROOT,
        embodiment_type: str = "embodiment_c_gripper_ext",
        **kwargs: Any,
    ) -> None:
        super().__init__(root=root, embodiment_type=embodiment_type, **kwargs)
