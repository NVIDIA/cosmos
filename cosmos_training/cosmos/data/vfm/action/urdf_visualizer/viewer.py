#!/usr/bin/env python
"""Interactive 3D viewer for robot action datasets.

Uses the unified 57D action representation: every dataset declares one explicit
raw ``ActionFormat`` (9D/10D/20D/57D), which is converted to
``UnifiedAction(action_57d, mask)`` before rendering.

**57D layout**: ``[ego(9) | R_wrist(9) | R_fingers(15) | L_wrist(9) | L_fingers(15)]``

Dependencies::

    pip install viser mujoco pin

Usage:
    # Use each dataset's declared raw action format:
    uv run python cosmos/data/vfm/action/urdf_visualizer/viewer.py --share

    # Override the raw action format explicitly:
    uv run python cosmos/data/vfm/action/urdf_visualizer/viewer.py --action-format 57d --share
"""

from __future__ import annotations

import argparse
import importlib
import os
import random
import sys
import time as _time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

from cosmos.utils import log
from cosmos.data.vfm.action.urdf_visualizer.unified_action import ActionFormat

_REPO_ROOT = str(Path(__file__).resolve().parents[6])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# ── Dataset Registry ──────────────────────────────────────────────────────────


@dataclass
class DatasetEntry:
    """Metadata for a dataset available in the viewer."""

    name: str
    robot_name: str
    max_finger_width: float
    fps: int
    pose_convention: str = "backward_framewise"
    camera_fov_deg: float = 60.0
    camera_aspect: float = 4 / 3
    dataset_class: str = ""
    dataset_kwargs: dict[str, Any] = field(default_factory=dict)
    action_format: ActionFormat = ActionFormat.SINGLE_ARM_10D
    dual_base_left: np.ndarray | None = None
    dual_base_right: np.ndarray | None = None
    robot_embodiment_type: str | None = None
    to_unified_fn: str | None = None  # "module.path:function" for custom action conversion


def _lazycfg_to_entry(
    cfg: Any,
    *,
    robot_name: str = "",
    max_finger_width: float = 0.0,
    fps: int = 10,
    action_format: ActionFormat = ActionFormat.SINGLE_ARM_10D,
    camera_fov_deg: float = 60.0,
    camera_aspect: float = 4 / 3,
    dual_base_left: np.ndarray | None = None,
    dual_base_right: np.ndarray | None = None,
    robot_embodiment_type: str | None = None,
    to_unified_fn: str | None = None,
    viewer_overrides: dict[str, Any] | None = None,
) -> DatasetEntry:
    """Build a viewer dataset entry from a v1p2 ``LazyCall(dataset_entry)`` config."""

    from omegaconf import OmegaConf

    from cosmos.utils.lazy_config.registry import convert_target_to_string

    ds_cfg = cfg.dataset
    target = ds_cfg._target_
    dataset_class = target if isinstance(target, str) else convert_target_to_string(target)
    dataset_kwargs = {
        key: value for key, value in OmegaConf.to_container(ds_cfg, resolve=False).items() if key != "_target_"
    }
    dataset_kwargs["action_normalization"] = None
    if viewer_overrides is not None:
        dataset_kwargs.update(viewer_overrides)

    pose_convention = str(dataset_kwargs.get("pose_convention", "backward_framewise"))
    cfg_dict = OmegaConf.to_container(cfg, resolve=False)
    dataset_name = str(cfg_dict.get("name", "unknown")) if isinstance(cfg_dict, dict) else "unknown"
    return DatasetEntry(
        name=dataset_name,
        robot_name=robot_name,
        max_finger_width=max_finger_width,
        fps=fps,
        pose_convention=pose_convention,
        camera_fov_deg=camera_fov_deg,
        camera_aspect=camera_aspect,
        dataset_class=dataset_class,
        dataset_kwargs=dataset_kwargs,
        action_format=action_format,
        dual_base_left=dual_base_left,
        dual_base_right=dual_base_right,
        robot_embodiment_type=robot_embodiment_type,
        to_unified_fn=to_unified_fn,
    )


def _build_datasets() -> dict[str, DatasetEntry]:
    """Build the viewer dataset registry from ``action_datasets_v1p2``."""

    from experiments.action.midtrain_config.action_datasets_v1p2 import (
        DATASET_EMBODIMENT_C_GRIPPER_480,
        DATASET_EMBODIMENT_C_GRIPPER_EXT_480,
        DATASET_AGIBOTWORLD_BETA_480,
        DATASET_AV_480,
        DATASET_BRIDGE_480,
        DATASET_CAMERA_480,
        DATASET_DROID_480,
        DATASET_FRACTAL_256,
        DATASET_HAND_POSE_480,
        DATASET_ROBOMIND_FRANKA_480,
        DATASET_ROBOMIND_FRANKA_DUAL_480,
        DATASET_ROBOMIND_UR_480,
        DATASET_UMI_256,
    )

    raw_action_override = {"action_normalization": None}
    credential_override = {"credential_path": "credentials/gcp_training.secret"}
    agibot_unifier = (
        "cosmos.data.vfm.action.urdf_visualizer.unified_action:to_unified_from_embodiment_c_fk_action"
    )

    return {
        "fractal": _lazycfg_to_entry(
            DATASET_FRACTAL_256,
            robot_name="google_robot",
            max_finger_width=0.05,
            fps=3,
            action_format=ActionFormat.SINGLE_ARM_10D,
            camera_fov_deg=69.0,
            camera_aspect=320 / 256,
            viewer_overrides=raw_action_override,
        ),
        "droid": _lazycfg_to_entry(
            DATASET_DROID_480,
            robot_name="franka_panda",
            max_finger_width=0.08,
            fps=15,
            action_format=ActionFormat.SINGLE_ARM_10D,
            viewer_overrides=raw_action_override,
        ),
        "bridge": _lazycfg_to_entry(
            DATASET_BRIDGE_480,
            robot_name="widowx",
            max_finger_width=0.06,
            fps=5,
            action_format=ActionFormat.SINGLE_ARM_10D,
            viewer_overrides=raw_action_override,
        ),
        "robomind_franka": _lazycfg_to_entry(
            DATASET_ROBOMIND_FRANKA_480,
            robot_name="franka_panda",
            max_finger_width=0.08,
            fps=10,
            action_format=ActionFormat.SINGLE_ARM_10D,
            viewer_overrides=raw_action_override,
        ),
        "robomind_franka_dual": _lazycfg_to_entry(
            DATASET_ROBOMIND_FRANKA_DUAL_480,
            robot_name="franka_panda",
            max_finger_width=0.08,
            fps=10,
            action_format=ActionFormat.DUAL_ARM_20D,
            dual_base_left=np.array(
                [[1, 0, 0, 0.0], [0, 1, 0, 0.3], [0, 0, 1, 0.0], [0, 0, 0, 1.0]],
                dtype=np.float32,
            ),
            dual_base_right=np.array(
                [[1, 0, 0, 0.0], [0, 1, 0, -0.3], [0, 0, 1, 0.0], [0, 0, 0, 1.0]],
                dtype=np.float32,
            ),
            viewer_overrides=raw_action_override,
        ),
        "robomind_ur": _lazycfg_to_entry(
            DATASET_ROBOMIND_UR_480,
            robot_name="ur5e",
            max_finger_width=0.085,
            fps=10,
            action_format=ActionFormat.SINGLE_ARM_10D,
            viewer_overrides=raw_action_override,
        ),
        "umi": _lazycfg_to_entry(
            DATASET_UMI_256,
            robot_name="",
            max_finger_width=0.08,
            fps=10,
            action_format=ActionFormat.SINGLE_ARM_10D,
            viewer_overrides={**raw_action_override, "normalizer_dir": ""},
        ),
        "camera": _lazycfg_to_entry(
            DATASET_CAMERA_480,
            robot_name="",
            max_finger_width=0.0,
            fps=10,
            action_format=ActionFormat.EGO_9D,
            viewer_overrides={**raw_action_override, **credential_override},
        ),
        "av": _lazycfg_to_entry(
            DATASET_AV_480,
            robot_name="",
            max_finger_width=0.0,
            fps=10,
            action_format=ActionFormat.EGO_9D,
            viewer_overrides={**raw_action_override, **credential_override},
        ),
        "hand_pose": _lazycfg_to_entry(
            DATASET_HAND_POSE_480,
            robot_name="",
            max_finger_width=0.0,
            fps=15,
            action_format=ActionFormat.UNIFIED_57D,
            viewer_overrides={**raw_action_override, "return_overlay_data": True},
        ),
        "embodiment_c_gripper": _lazycfg_to_entry(
            DATASET_EMBODIMENT_C_GRIPPER_480,
            robot_name="embodiment_c",
            max_finger_width=0.12,
            fps=10,
            camera_fov_deg=69.0,
            camera_aspect=640 / 480,
            viewer_overrides={**raw_action_override, "return_agibot_link_poses": True},
            robot_embodiment_type="embodiment_c_gripper",
            to_unified_fn=agibot_unifier,
        ),
        "embodiment_c_gripper_ext": _lazycfg_to_entry(
            DATASET_EMBODIMENT_C_GRIPPER_EXT_480,
            robot_name="embodiment_c",
            max_finger_width=0.12,
            fps=10,
            camera_fov_deg=69.0,
            camera_aspect=640 / 480,
            viewer_overrides={**raw_action_override, "return_agibot_link_poses": True},
            robot_embodiment_type="embodiment_c_gripper_ext",
            to_unified_fn=agibot_unifier,
        ),
        "agibotworld_beta": _lazycfg_to_entry(
            DATASET_AGIBOTWORLD_BETA_480,
            robot_name="embodiment_c",
            max_finger_width=0.12,
            fps=10,
            camera_fov_deg=69.0,
            camera_aspect=640 / 480,
            viewer_overrides={**raw_action_override, "return_agibot_link_poses": True},
            robot_embodiment_type="embodiment_c_gripper",
            to_unified_fn=agibot_unifier,
        ),
    }


DATASETS: dict[str, DatasetEntry] = {}


# ── Dataset Creation ──────────────────────────────────────────────────────────


def _create_dataset(entry: DatasetEntry, chunk_length: int):
    """Instantiate a dataset class for the given entry."""
    import importlib
    import inspect

    module_path, class_name = entry.dataset_class.rsplit(".", 1)
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)

    kwargs = dict(entry.dataset_kwargs)
    kwargs["chunk_length"] = chunk_length
    kwargs["split"] = "full"
    kwargs["mode"] = "policy"
    kwargs["enable_fast_init"] = True

    # UMI: factory function
    if callable(cls) and not inspect.isclass(cls):
        _OMEGACONF_BLOCKLIST = {"chunk_length", "split", "action_normalization", "enable_fast_init"}
        kwargs = {k: v for k, v in kwargs.items() if k not in _OMEGACONF_BLOCKLIST}
        kwargs["eager_load"] = True
        return cls(**kwargs)

    sig = inspect.signature(cls.__init__)
    valid_params = set(sig.parameters.keys()) - {"self"}
    has_var_keyword = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    if not has_var_keyword:
        kwargs = {k: v for k, v in kwargs.items() if k in valid_params}

    dataset = cls(**kwargs)

    if hasattr(dataset, "_register_sources"):
        dataset._register_sources()
        if hasattr(dataset, "__len__") and len(dataset) == 0:
            raise RuntimeError(f"{entry.name}: registered sources but found no valid samples")

    from torch.utils.data import IterableDataset as _IterableBase

    if isinstance(dataset, _IterableBase):
        dataset = _IterableToMapDataset(dataset)

    return dataset


def _resolve_action_format(
    entry: DatasetEntry,
    action_format_override: ActionFormat | None,
) -> ActionFormat:
    """Return the explicit raw action format for one dataset load."""
    return action_format_override if action_format_override is not None else entry.action_format


@lru_cache(maxsize=None)
def _load_symbol(target: str):
    """Import and cache one ``module:symbol`` reference."""

    module_name, symbol_name = target.split(":", maxsplit=1)
    module = importlib.import_module(module_name)
    return getattr(module, symbol_name)


def _format_sample_text(value: Any, max_chars: int | None = None) -> str:
    """Format optional sample text for the viewer info panel."""

    if value is None:
        return ""
    text = value if isinstance(value, str) else str(value)
    if not text:
        return ""
    if max_chars is None:
        return text
    return text[:max_chars]


def _build_viewer_idle_action_spec(action_format: ActionFormat) -> Any:
    """Build a fallback idle-frame spec from the viewer-declared action format."""

    from cosmos.data.vfm.action.action_spec import Gripper, Pos, Rot, build_action_spec

    if action_format is ActionFormat.EGO_9D:
        return build_action_spec(Pos(prefix="ego"), Rot("rot6d", prefix="ego"))
    if action_format is ActionFormat.SINGLE_ARM_10D:
        return build_action_spec(Pos(), Rot("rot6d"), Gripper())
    if action_format is ActionFormat.DUAL_ARM_20D:
        return build_action_spec(
            Pos(prefix="left"),
            Rot("rot6d", prefix="left"),
            Gripper(prefix="left"),
            Pos(prefix="right"),
            Rot("rot6d", prefix="right"),
            Gripper(prefix="right"),
        )
    if action_format is ActionFormat.UNIFIED_57D:
        return build_action_spec(
            Pos(prefix="ego"),
            Rot("rot6d", prefix="ego"),
            Pos(prefix="right_wrist"),
            Rot("rot6d", prefix="right_wrist"),
            Pos(dim=15, prefix="right_fingers"),
            Pos(prefix="left_wrist"),
            Rot("rot6d", prefix="left_wrist"),
            Pos(dim=15, prefix="left_fingers"),
        )
    raise ValueError(f"Unsupported action format for idle-frame detection: {action_format}")


def _compute_viewer_idle_frames(
    action: Any,
    dataset: Any,
    action_format: ActionFormat,
) -> torch.Tensor | None:
    """Compute idle frames for a viewer sample when the dataset did not provide them."""

    action_spec = getattr(dataset, "action_spec", None)
    compute_idle_frames_method = getattr(dataset, "_compute_idle_frames", None)
    if action_spec is not None and compute_idle_frames_method is not None:
        return compute_idle_frames_method(action)

    from cosmos.data.vfm.action.pose_utils import compute_idle_frames

    spec = _build_viewer_idle_action_spec(action_format)
    try:
        idle_frames = compute_idle_frames(action, spec)
    except (TypeError, ValueError) as error:
        log.warning(f"Viewer idle-frame detection skipped for {action_format.value}: {error}")
        return None
    return torch.tensor(idle_frames, dtype=torch.long)  # []


@lru_cache(maxsize=1)
def _get_viewer_idle_frames_augmentor() -> Any:
    """Return the caption augmentor used by the viewer idle-frame path."""

    from cosmos.data.vfm.augmentors.idle_frames_text_info import IdleFramesTextInfo

    return IdleFramesTextInfo(
        input_keys=["ai_caption", "idle_frames", "action"],
        output_keys=["ai_caption"],
        args={
            "caption_key": "ai_caption",
            "idle_frames_key": "idle_frames",
            "action_key": "action",
            "dropout_rate": 0.0,
            "enabled": True,
        },
    )


def _enable_viewer_idle_frames(sample: dict[str, Any], dataset: Any, action_format: ActionFormat) -> dict[str, Any]:
    """Populate idle-frame metadata and append text in the direct viewer data path."""

    updated_sample = sample
    idle_frames = updated_sample.get("idle_frames")
    action = updated_sample.get("action")
    if idle_frames is None and action is not None:
        idle_frames = _compute_viewer_idle_frames(action, dataset, action_format)
        if idle_frames is not None:
            updated_sample = dict(updated_sample)
            updated_sample["idle_frames"] = idle_frames

    if idle_frames is None:
        return updated_sample

    updated_sample = dict(updated_sample)
    caption = updated_sample.get("ai_caption")
    if isinstance(caption, dict):
        updated_sample["ai_caption"] = dict(caption)

    augmented_sample = _get_viewer_idle_frames_augmentor()(updated_sample)
    return updated_sample if augmented_sample is None else augmented_sample


class _IterableToMapDataset:
    """Wraps an IterableDataset into a random-access dataset with lazy loading."""

    _MAX_CACHE = 200

    def __init__(self, iterable_dataset, max_samples: int | None = None):
        self._iter = iter(iterable_dataset)
        self._samples: list[dict] = []
        self._exhausted = False
        self._max = max_samples or self._MAX_CACHE
        self._ds_name = iterable_dataset.__class__.__name__
        log.info(f"Lazy wrapper: {self._ds_name} (max {self._max})")

    def _fetch_up_to(self, idx: int) -> bool:
        while len(self._samples) <= idx and not self._exhausted and len(self._samples) < self._max:
            try:
                self._samples.append(next(self._iter))
                log.info(f"{self._ds_name}: fetched sample {len(self._samples) - 1}")
            except StopIteration:
                self._exhausted = True
                log.info(f"{self._ds_name}: exhausted at {len(self._samples)}")
        return idx < len(self._samples)

    def __len__(self):
        return max(len(self._samples), 1)

    def __getitem__(self, idx):
        if self._fetch_up_to(idx):
            return self._samples[idx]
        if self._samples:
            return self._samples[idx % len(self._samples)]
        raise IndexError(f"{self._ds_name}: no samples available")


# ── Viewer ────────────────────────────────────────────────────────────────────


def _collect_scene_points(state) -> np.ndarray:
    """Collect all visible trajectory positions for camera fitting."""
    points: list[np.ndarray] = []
    for poses in (state.ego_poses, state.right_poses, state.left_poses):
        if poses is not None and len(poses) > 0:
            points.append(poses[:, :3, 3].astype(np.float32))
    if not points:
        return np.zeros((1, 3), dtype=np.float32)
    return np.concatenate(points, axis=0)


def _get_observation_up_direction(state, view_forward: np.ndarray) -> np.ndarray:
    """Estimate a stable viewer up-direction from the observation camera poses."""
    if state.ego_poses is None or len(state.ego_poses) == 0:
        return np.array([0.0, 0.0, 1.0], dtype=np.float32)

    # Ego poses are camera-to-world transforms in OpenCV camera convention,
    # where image up corresponds to the negative camera Y axis.
    camera_up = -state.ego_poses[:, :3, 1].astype(np.float32)
    reference = camera_up[0]
    aligned_up = camera_up.copy()
    for idx in range(1, len(aligned_up)):
        if float(np.dot(aligned_up[idx], reference)) < 0.0:
            aligned_up[idx] *= -1.0

    up_direction = aligned_up.mean(axis=0)
    up_direction -= view_forward * float(np.dot(up_direction, view_forward))
    up_norm = float(np.linalg.norm(up_direction))
    if up_norm < 1e-6:
        return np.array([0.0, 0.0, 1.0], dtype=np.float32)
    return up_direction / up_norm


def _get_observation_forward_direction(state) -> np.ndarray | None:
    """Estimate a stable viewer forward direction from the observation camera poses."""
    if state.ego_poses is None or len(state.ego_poses) == 0:
        return None

    # Ego poses are camera-to-world transforms in OpenCV camera convention,
    # where the optical axis points along the positive camera Z axis.
    camera_forward = state.ego_poses[:, :3, 2].astype(np.float32)
    reference = camera_forward[0]
    aligned_forward = camera_forward.copy()
    for idx in range(1, len(aligned_forward)):
        if float(np.dot(aligned_forward[idx], reference)) < 0.0:
            aligned_forward[idx] *= -1.0

    forward_direction = aligned_forward.mean(axis=0)
    forward_norm = float(np.linalg.norm(forward_direction))
    if forward_norm < 1e-6:
        return None
    return forward_direction / forward_norm


def _reset_camera_to_trajectory(client, state, camera_fov_deg: float) -> None:
    """Frame one client's viewport using the current trajectory extent."""
    points = _collect_scene_points(state)
    center = points.mean(axis=0)
    extent = points - center[None, :]
    radius = float(np.linalg.norm(extent, axis=1).max()) if len(points) > 0 else 0.0
    radius = max(radius, 0.15)

    fov_rad = float(np.deg2rad(camera_fov_deg))
    fit_distance = radius / max(np.tan(fov_rad / 2.0), 0.35)
    view_forward = _get_observation_forward_direction(state)
    if view_forward is None:
        view_dir = np.array([1.0, -1.0, 0.7], dtype=np.float32)
        view_dir /= np.linalg.norm(view_dir)
        camera_position = center + view_dir * max(fit_distance * 1.35, 0.5)
        view_forward = center - camera_position
        view_forward /= np.linalg.norm(view_forward)
    else:
        camera_position = center - view_forward * max(fit_distance * 1.35, 0.5)
    view_forward = center - camera_position
    view_forward /= np.linalg.norm(view_forward)
    up_direction = _get_observation_up_direction(state, view_forward)
    camera = client.camera

    # Camera state arrives asynchronously from the browser. Wait briefly so we can
    # update only this client's viewport instead of broadcasting a global reset target.
    deadline = _time.time() + 1.0
    while getattr(camera._state, "update_timestamp", 0.0) == 0.0 and _time.time() < deadline:
        _time.sleep(0.01)
    if getattr(camera._state, "update_timestamp", 0.0) == 0.0:
        return

    camera.fov = fov_rad
    camera.up_direction = tuple(up_direction.tolist())
    camera.look_at = tuple(center.tolist())
    # Setting position also translates look_at, so restore the target afterwards.
    camera.position = tuple(camera_position.tolist())
    camera.look_at = tuple(center.tolist())
    client.flush()


def launch_viewer(
    port: int = 8013,
    share: bool = False,
    chunk_length: int = 16,
    action_format_override: ActionFormat | None = None,
) -> None:
    """Launch the interactive dataset viewer."""
    global DATASETS
    import threading as _threading

    import viser

    from cosmos.data.vfm.action.urdf_visualizer.unified_action import (
        build_scene_state,
        get_video_from_sample,
        to_unified,
    )
    from cosmos.data.vfm.action.urdf_visualizer.unified_renderer import UnifiedRenderer

    server = viser.ViserServer(host="0.0.0.0", port=port)
    if not DATASETS:
        DATASETS = _build_datasets()
    datasets = DATASETS
    dataset_cache: dict[str, Any] = {}
    dataset_locks: dict[str, Any] = {}
    dataset_cache_lock = _threading.Lock()
    sessions_lock = _threading.Lock()

    def _get_dataset_lock(cache_key: str) -> Any:
        """Return the per-dataset lock for a cache key."""
        with dataset_cache_lock:
            lock = dataset_locks.get(cache_key)
            if lock is None:
                lock = _threading.Lock()
                dataset_locks[cache_key] = lock
            return lock

    @dataclass
    class ViewerSession:
        client: Any
        renderer: Any
        time_slider: Any
        speed_slider: Any
        play_button: Any
        action_text: Any
        show: dict[str, bool | float]
        load_lock: Any = field(default_factory=_threading.Lock)
        playing: bool = False
        last_frame_time: float = 0.0

    sessions: dict[int, ViewerSession] = {}

    @server.on_client_connect
    def _(client) -> None:
        client.scene.reset()
        client.scene.set_up_direction("+z")
        client.gui.reset()

        renderer = UnifiedRenderer(client)

        with client.gui.add_folder("Dataset"):
            ds_dropdown = client.gui.add_dropdown(
                "Dataset", options=list(datasets.keys()), initial_value=list(datasets.keys())[0]
            )
            ep_input = client.gui.add_number("Episode", initial_value=0, min=0, step=1)
            random_button = client.gui.add_button("🎲 Random episode")
            status_text = client.gui.add_markdown("*Ready*")
            info_text = client.gui.add_markdown("")

        with client.gui.add_folder("Display", expand_by_default=False):
            show_robot = client.gui.add_checkbox("Show robot mesh", initial_value=True)
            show_frames = client.gui.add_checkbox("Show wrist frames", initial_value=True)
            show_traj = client.gui.add_checkbox("Show trajectory", initial_value=True)
            show_fingertips = client.gui.add_checkbox("Show fingertips", initial_value=True)
            show_ego = client.gui.add_checkbox("Show ego camera", initial_value=True)
            axis_scale = client.gui.add_slider("Axis scale", min=0.1, max=20.0, step=0.1, initial_value=1.0)

        robot_frame_toggle_handles: dict[str, Any] = {}
        robot_frame_toggle_folder = client.gui.add_folder("Robot Frame Toggles", expand_by_default=False)
        with robot_frame_toggle_folder:
            robot_frame_toggle_status = client.gui.add_markdown("*Load an episode to choose robot frame coordinates.*")

        with client.gui.add_folder("Playback"):
            time_slider = client.gui.add_slider("Time", min=0, max=1, step=1, initial_value=0)
            play_button = client.gui.add_button("▶ Play")
            speed_slider = client.gui.add_slider("Speed (fps)", min=1, max=30, step=1, initial_value=3)

        cam_panel = client.gui.add_image(np.zeros((64, 64, 3), dtype=np.uint8))
        renderer.set_video_panel(cam_panel)

        with client.gui.add_folder("Action (57D)"):
            action_text = client.gui.add_markdown("*No episode loaded*")

        show = {
            "frames": True,
            "traj": True,
            "fingertips": True,
            "ego": True,
            "robot": True,
            "robot_frame_filters": {},
            "axis_scale": 1.0,
        }
        session = ViewerSession(
            client=client,
            renderer=renderer,
            time_slider=time_slider,
            speed_slider=speed_slider,
            play_button=play_button,
            action_text=action_text,
            show=show,
        )

        def _update_action_text(t: int) -> None:
            """Update the 57D action display for one client."""
            txt = renderer.format_action_text(t)
            action_text.content = txt if txt else "*No data*"

        def _clear_robot_frame_toggles() -> None:
            """Remove the dynamic per-frame toggle controls."""
            for handle in robot_frame_toggle_handles.values():
                handle.remove()
            robot_frame_toggle_handles.clear()

        def _rebuild_robot_frame_toggles() -> None:
            """Rebuild the GUI toggles for the currently loaded robot frames."""
            _clear_robot_frame_toggles()
            selectors = renderer.get_robot_frame_selectors()
            if not selectors:
                show["robot_frame_filters"] = {}
                robot_frame_toggle_status.content = "*No robot frame coordinates available for this episode.*"
                return

            prev_filters = cast(dict[str, bool], show.get("robot_frame_filters", {}))
            show["robot_frame_filters"] = {
                selector_key: prev_filters.get(selector_key, False) for selector_key, _ in selectors
            }
            robot_frame_toggle_status.content = "*Choose which robot frame coordinates to show.*"

            with robot_frame_toggle_folder:
                for selector_key, label in selectors:
                    checkbox = client.gui.add_checkbox(
                        label,
                        initial_value=cast(dict[str, bool], show["robot_frame_filters"])[selector_key],
                    )
                    robot_frame_toggle_handles[selector_key] = checkbox

                    @checkbox.on_update
                    def _(_, selector_key: str = selector_key, checkbox: Any = checkbox) -> None:
                        cast(dict[str, bool], show["robot_frame_filters"])[selector_key] = bool(checkbox.value)
                        renderer.update(time_slider.value, show)

        def do_load_episode() -> None:
            t_start = _time.time()
            ds_name = ds_dropdown.value
            entry = datasets[ds_name]
            effective_action_format = _resolve_action_format(entry, action_format_override)
            ep_idx = max(int(ep_input.value), 0)
            cache_key = ds_name

            status_text.content = f"⏳ Loading {ds_name} episode {ep_idx}..."

            try:
                with _get_dataset_lock(cache_key):
                    dataset: Any
                    with dataset_cache_lock:
                        dataset = cast(Any, dataset_cache.get(cache_key))
                    if dataset is None:
                        status_text.content = f"⏳ Creating {ds_name} dataset..."
                        dataset = _create_dataset(entry, chunk_length)
                        with dataset_cache_lock:
                            dataset_cache[cache_key] = dataset
                    to_opencv = getattr(dataset, "_to_opencv", np.eye(3, dtype=np.float32))

                    n_total = len(dataset)
                    if ep_idx >= n_total:
                        if isinstance(dataset, _IterableToMapDataset) and not dataset._exhausted:
                            pass
                        else:
                            ep_idx = n_total - 1
                            ep_input.value = ep_idx

                    sample: Any = _enable_viewer_idle_frames(dataset[ep_idx], dataset, effective_action_format)

                action_tensor = sample["action"]
                action_raw = (
                    action_tensor.numpy() if isinstance(action_tensor, torch.Tensor) else np.asarray(action_tensor)
                )

                uses_dual_initial_pose = effective_action_format is ActionFormat.DUAL_ARM_20D

                initial_pose_t = sample.get("initial_pose")
                if initial_pose_t is None:
                    initial_pose = np.eye(4, dtype=np.float32)
                elif isinstance(initial_pose_t, torch.Tensor):
                    initial_pose = initial_pose_t.numpy().astype(np.float32)
                else:
                    initial_pose = np.asarray(initial_pose_t, dtype=np.float32)

                initial_pose_right_t = sample.get("initial_pose_right")
                initial_pose_left_t = sample.get("initial_pose_left")
                initial_pose_right = None
                initial_pose_left = None
                if initial_pose_right_t is not None:
                    initial_pose_right = (
                        initial_pose_right_t.numpy().astype(np.float32)
                        if isinstance(initial_pose_right_t, torch.Tensor)
                        else np.asarray(initial_pose_right_t, dtype=np.float32)
                    )
                if initial_pose_left_t is not None:
                    initial_pose_left = (
                        initial_pose_left_t.numpy().astype(np.float32)
                        if isinstance(initial_pose_left_t, torch.Tensor)
                        else np.asarray(initial_pose_left_t, dtype=np.float32)
                    )
                if uses_dual_initial_pose:
                    if initial_pose_left is None:
                        initial_pose_left = initial_pose

                if entry.to_unified_fn:
                    converter = _load_symbol(entry.to_unified_fn)
                    import inspect as _inspect

                    params = _inspect.signature(converter).parameters
                    embodiment_type = entry.robot_embodiment_type or str(
                        entry.dataset_kwargs.get("embodiment_type", "")
                    )
                    if "embodiment_type" in params:
                        unified = converter(sample, embodiment_type=embodiment_type)
                    elif "kind" in params:
                        unified = converter(action_raw, kind="gripper")
                    else:
                        unified = converter(action_raw)
                    raw_action_label = "custom"
                else:
                    unified = to_unified(action_raw, action_format=effective_action_format)
                    raw_action_label = effective_action_format.value
                state = build_scene_state(
                    unified,
                    initial_pose=initial_pose,
                    initial_pose_right=initial_pose_right,
                    initial_pose_left=initial_pose_left,
                    right_base_pose=entry.dual_base_right,
                    left_base_pose=entry.dual_base_left,
                    pose_convention=entry.pose_convention,
                    sample=sample,
                )
                state.video = get_video_from_sample(sample)

                # Inject FK joint configs when the dataset provides them (e.g. UR).
                jc = sample.get("joint_configs")
                if jc is not None:
                    state.joint_configs = (
                        jc.numpy().astype(np.float32)
                        if isinstance(jc, torch.Tensor)
                        else np.asarray(jc, dtype=np.float32)
                    )
                status_text.content = "⏳ Loading robot animation..."
                renderer.load(state, entry, to_opencv=to_opencv)
                _rebuild_robot_frame_toggles()
                _reset_camera_to_trajectory(client, state, entry.camera_fov_deg)

                T = state.T
                time_slider.max = max(T, 1)
                time_slider.value = 0

                ai_caption_text = _format_sample_text(sample.get("ai_caption", ""), max_chars=160)
                debug_caption_text = _format_sample_text(sample.get("debug_caption", ""))
                t_total = _time.time() - t_start
                info_text.content = (
                    f"**{ds_name.upper()}** — Episode {ep_idx}\n\n"
                    + (f"Task: {ai_caption_text}\n\n" if ai_caption_text else "")
                    + (f"Debug: {debug_caption_text}\n\n" if debug_caption_text else "")
                    + (
                        f"Steps: {T} | Raw: {raw_action_label} ({action_raw.shape[-1]}D) → 57D | "
                        f"Robot: {entry.robot_name or '—'} | FPS: {entry.fps}"
                    )
                )
                status_text.content = f"✅ Loaded in {t_total:.1f}s"
                log.info(f"Loaded {ds_name} ep {ep_idx}: {ai_caption_text[:60]}, {T} steps, {t_total:.1f}s")

                renderer.update(0, show)
                renderer.update_axis_scale(axis_scale.value)
                _update_action_text(0)
                session.last_frame_time = _time.time()

            except Exception as e:
                status_text.content = f"❌ Load failed: {e}"
                log.error(f"Load failed for client {client.client_id}: {e}")
                import traceback

                traceback.print_exc()

        def _do_load_threaded() -> None:
            if not session.load_lock.acquire(blocking=False):
                return

            def _run() -> None:
                try:
                    do_load_episode()
                finally:
                    session.load_lock.release()

            _threading.Thread(target=_run, daemon=True).start()

        @ds_dropdown.on_update
        def _(_) -> None:
            _do_load_threaded()

        @ep_input.on_update
        def _(_) -> None:
            _do_load_threaded()

        @random_button.on_click
        def _(_) -> None:
            ds_name = ds_dropdown.value
            cache_key = ds_name
            with _get_dataset_lock(cache_key):
                with dataset_cache_lock:
                    ds = dataset_cache.get(cache_key)
                if ds is None:
                    ep_input.value = 0
                elif isinstance(ds, _IterableToMapDataset):
                    ep_input.value = len(ds._samples)
                else:
                    ep_input.value = random.randint(0, max(len(ds) - 1, 0))
            _do_load_threaded()

        @time_slider.on_update
        def _(_) -> None:
            renderer.update(time_slider.value, show)
            _update_action_text(time_slider.value)

        @show_robot.on_update
        def _(_) -> None:
            show["robot"] = show_robot.value
            renderer.update(time_slider.value, show)

        @show_frames.on_update
        def _(_) -> None:
            show["frames"] = show_frames.value
            renderer.update(time_slider.value, show)

        @show_traj.on_update
        def _(_) -> None:
            show["traj"] = show_traj.value
            renderer.update(time_slider.value, show)

        @show_fingertips.on_update
        def _(_) -> None:
            show["fingertips"] = show_fingertips.value
            renderer.update(time_slider.value, show)

        @show_ego.on_update
        def _(_) -> None:
            show["ego"] = show_ego.value
            renderer.update(time_slider.value, show)

        @axis_scale.on_update
        def _(_) -> None:
            show["axis_scale"] = axis_scale.value
            renderer.update_axis_scale(axis_scale.value)

        @play_button.on_click
        def _(_) -> None:
            session.playing = not session.playing
            session.last_frame_time = _time.time()
            play_button.label = "⏸ Pause" if session.playing else "▶ Play"

        with sessions_lock:
            sessions[client.client_id] = session
        _do_load_threaded()

    @server.on_client_disconnect
    def _(client) -> None:
        with sessions_lock:
            sessions.pop(client.client_id, None)

    # ── Share URL ──
    log.info(f"✅ Viewer ready at http://0.0.0.0:{port}")
    if share:
        share_url = server.request_share_url()
        if share_url:
            log.info(f"🌐 Share URL: {share_url}")
            try:
                with open(os.path.expanduser("~/share_url.txt"), "w") as f:
                    f.write(share_url + "\n")
            except Exception:
                pass

    # ── Main Loop ──
    try:
        while True:
            now = _time.time()
            with sessions_lock:
                active_sessions = list(sessions.values())
            for session in active_sessions:
                renderer = session.renderer
                if not session.playing or renderer.state is None:
                    continue
                frame_period = 1.0 / max(float(session.speed_slider.value), 1.0)
                if now - session.last_frame_time < frame_period:
                    continue
                t = session.time_slider.value
                t = (t + 1) % max(renderer.state.T, 1)
                session.time_slider.value = t
                renderer.update(t, session.show)
                txt = renderer.format_action_text(t)
                session.action_text.content = txt if txt else "*No data*"
                session.last_frame_time = now
            _time.sleep(0.02)
    except KeyboardInterrupt:
        log.info("Shutting down.")


def main():
    parser = argparse.ArgumentParser(description="Action dataset viewer (unified 57D)")
    parser.add_argument("--port", type=int, default=8013)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--chunk-length", type=int, default=16)
    parser.add_argument(
        "--action-format",
        choices=[fmt.value for fmt in ActionFormat],
        default=None,
        help="Optional override for the dataset-declared raw action format",
    )
    args = parser.parse_args()
    launch_viewer(
        port=args.port,
        share=args.share,
        chunk_length=args.chunk_length,
        action_format_override=ActionFormat(args.action_format) if args.action_format is not None else None,
    )


if __name__ == "__main__":
    main()
