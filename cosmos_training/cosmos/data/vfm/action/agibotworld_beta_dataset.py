"""AgiBotWorld-Beta dataset with FK-pose actions and multi-view support.

Uses the same calibrated G1/omnipicker URDF forward kinematics as Embodiment C to produce
relative SE(3) delta actions for head camera and left/right gripper-base wrists,
concatenated with gripper open-fraction values.

Action layout (29 dims, same as GEAR gripper):
    ``[head_cam_delta(9), right_wrist_delta(9), right_gripper(1),
      left_wrist_delta(9), left_gripper(1)]``

View modes:
    - **ego_view**: single ``observation.images.head`` camera.
    - **concat_view**: head view on top, left/right wrist views resized and
      concatenated horizontally on the bottom.
"""

from __future__ import annotations

from typing import Any, Literal, cast

import numpy as np
import torch
import torch.nn.functional as F

from cosmos.data.vfm.action.embodiment_c_fk import (
    AGIBOT_GEAR_GRIPPER_TO_OPENCV_BY_WRIST,
    apply_agibot_gripper_to_opencv,
    apply_robot_base_motion_to_poses,
    compute_fk_transforms_batch,
    compute_link_poses_batch,
    convert_gripper_state_to_open_fraction,
    extract_fk_transforms_from_link_poses,
)
from cosmos.data.vfm.action.embodiment_c_spec import AGIBOT_GEAR_GRIPPER_NORMALIZER_EMBODIMENT_TYPE
from cosmos.data.vfm.action.agibotworld_beta_dataset_config import LEROBOT_ROOTS
from cosmos.data.vfm.action.cosmos3_action_lerobot import (
    ActionNormalization,
    ActionSpec,
    BaseActionLeRobotDataset,
    Gripper,
    Pos,
    Rot,
    build_action_spec,
)
from cosmos.data.vfm.action.pose_utils import PoseConvention, pose_abs_to_rel
from cosmos.data.vfm.action.viewpoint_utils import Viewpoint

# Video keys.
_HEAD_KEY = "observation.images.head"
_HAND_LEFT_KEY = "observation.images.hand_left"
_HAND_RIGHT_KEY = "observation.images.hand_right"

# State observation keys needed for FK.
_ROBOT_POSITION_KEY = "observation.states.robot.position"
_ROBOT_ORIENTATION_KEY = "observation.states.robot.orientation"
_STATE_KEYS = [
    "observation.states.effector.position",
    "observation.states.joint.position",
    "observation.states.head.position",
    "observation.states.waist.position",
    _ROBOT_POSITION_KEY,
    _ROBOT_ORIENTATION_KEY,
]

_BASE_ROOT = (
    "/lustre/fsw/portfolios/cosmos/projects/cosmos_base_training/"
    "cosmos3_action_datasets/AgiBotWorld-Beta_20260102/agibotworld"
)


def _resolve_root_paths(root: str | list[str] | tuple[str, ...] | None) -> list[str]:
    """Normalize a root argument into concrete AgiBotWorld-Beta LeRobot roots."""

    if root is None:
        return [f"{_BASE_ROOT}/{subpath}" for subpath in LEROBOT_ROOTS]
    if not isinstance(root, str):
        return [str(path) for path in root]
    return [f"{root}/{subpath}" for subpath in LEROBOT_ROOTS]


def _split_task_for_caption(task: str) -> tuple[str, str]:
    """Split AgiBotWorld task text into train caption and debug-only detail."""

    ai_caption, separator, debug_caption = task.partition("|")
    if not separator:
        return task.strip(), ""
    return ai_caption.strip(), debug_caption.strip()


def _assemble_embodiment_c_state(
    effector_pos: np.ndarray,
    joint_pos: np.ndarray,
    head_pos: np.ndarray,
    waist_pos: np.ndarray,
) -> np.ndarray:
    """Assemble a GEAR-compatible flat state vector from Beta's decomposed fields.

    Args:
        effector_pos: ``(N, 2)`` gripper positions as ``[left_gripper, right_gripper]``.
        joint_pos: ``(N, 14)`` arm joint positions (7 left + 7 right).
        head_pos: ``(N, 2)`` head joints as ``[head_yaw, head_pitch]``.
        waist_pos: ``(N, 2)`` waist joints as ``[body_pitch, lift_body]``.

    Returns:
        State array of shape ``(N, 20)`` with GEAR gripper layout:
        ``[arm_joints(14), gripper(2), head_yaw, head_pitch, waist_pitch, waist_lift]``

    Note:
        Beta ``waist_pos = [body_pitch, lift_body]`` maps directly into the
        standard GEAR body/head block:
        ``state[16:20] = [head_yaw, head_pitch, body_pitch, lift_body]``.
    """
    # Body/head block: [head_yaw, head_pitch, waist_pitch, waist_lift]
    body_head = np.stack(
        [head_pos[:, 0], head_pos[:, 1], waist_pos[:, 0], waist_pos[:, 1]],
        axis=-1,
    )  # [N,4]

    return np.concatenate([joint_pos, effector_pos, body_head], axis=-1).astype(np.float32)  # [N,20]


class AgiBotWorldBetaDataset(BaseActionLeRobotDataset):
    """AgiBotWorld-Beta dataset with FK-pose actions matching GEAR gripper format.

    Concat view layout:
        ┌──────────────────┐
        │      head        │  (H, W)
        ├─────────┬────────┤
        │ hand_L  │ hand_R │  (H/2, W/2) each
        └─────────┴────────┘
    """

    def __init__(
        self,
        root: str | list[str] | tuple[str, ...] | None = None,
        fps: float = 10.0,
        chunk_length: int = 16,
        split_seed: int = 42,
        split_val_ratio: float = 0.05,
        split: str = "train",
        action_normalization: ActionNormalization | None = None,
        mode: str = "joint",
        viewpoint: Viewpoint = "concat_view",
        pose_convention: PoseConvention = "backward_framewise",
        rotation_format: Literal["rot6d"] = "rot6d",
        tolerance_s: float = 3e-4,
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
            embodiment_type="agibotworld",
            viewpoint=viewpoint,
            action_normalization=action_normalization,
            pose_convention=pose_convention,
            rotation_format=rotation_format,
            tolerance_s=tolerance_s,
            max_loaded_datasets=max_loaded_datasets,
            skip_video_loading=skip_video_loading,
            sample_stride=sample_stride,
            enable_fast_init=enable_fast_init,
            fast_init_max_workers=fast_init_max_workers,
        )

        self._is_concat_view = viewpoint == "concat_view"
        self._to_opencv: dict[str, np.ndarray] = AGIBOT_GEAR_GRIPPER_TO_OPENCV_BY_WRIST
        self._return_agibot_link_poses: bool = return_agibot_link_poses

        self._all_shard_roots = _resolve_root_paths(root)

        # T+1 frames for observations (states + video). Source action columns
        # are intentionally not requested; emitted actions are state deltas.
        frame_ts_obs = [i * self._dt for i in range(self._chunk_length + 1)]

        self._delta_timestamps = {
            _HEAD_KEY: frame_ts_obs,
        }
        # State keys for FK (T+1 frames).
        for key in _STATE_KEYS:
            self._delta_timestamps[key] = frame_ts_obs
        # Multi-view cameras.
        if self._is_concat_view:
            self._delta_timestamps[_HAND_LEFT_KEY] = frame_ts_obs
            self._delta_timestamps[_HAND_RIGHT_KEY] = frame_ts_obs

    def _normalizer_filename(self) -> str:
        """Use the shared Embodiment C gripper FK-pose stats for Beta data."""

        return f"{AGIBOT_GEAR_GRIPPER_NORMALIZER_EMBODIMENT_TYPE}_{self._pose_convention}_{self._rotation_format}.json"

    # -- FK-based action construction ----------------------------------------

    def _build_fk_action(self, sample: dict[str, Any]) -> tuple[torch.Tensor, dict[str, Any]]:
        """Build relative FK-pose action plus absolute initial poses.

        Assembles GEAR-compatible flat state from Beta's decomposed observation
        fields, runs calibrated G1/omnipicker URDF forward kinematics, and converts to framewise
        relative SE(3) deltas with rot6d rotation blocks.

        Returns:
            ``(action, extras)`` where ``action`` is ``(T, 29)`` and ``extras``
            carries absolute initial poses for reconstruction.

        Beta ``observation.states.effector.position`` stores left/right scalar
        gripper positions. Before these are concatenated into the
        GEAR-compatible 29D action, they are normalized to the shared
        viewer/action convention:
        ``0.0`` means closed and ``1.0`` means open. AgiBot actuator-close
        state values use ``0=open`` and ``120=closed``; angle-valued gripper
        states are scaled by ``convert_gripper_state_to_open_fraction``.
        """

        # Assemble GEAR-compatible flat state from decomposed Beta fields.
        effector_pos = sample["observation.states.effector.position"].detach().cpu().numpy()  # [T+1,2]
        joint_pos = sample["observation.states.joint.position"].detach().cpu().numpy()  # [T+1,14]
        head_pos = sample["observation.states.head.position"].detach().cpu().numpy()  # [T+1,2]
        waist_pos = sample["observation.states.waist.position"].detach().cpu().numpy()  # [T+1,2]
        robot_pos = sample[_ROBOT_POSITION_KEY].detach().cpu().numpy()  # [T+1,3]
        robot_quat = sample[_ROBOT_ORIENTATION_KEY].detach().cpu().numpy()  # [T+1,4]
        num_state_steps = min(
            effector_pos.shape[0],
            joint_pos.shape[0],
            head_pos.shape[0],
            waist_pos.shape[0],
            robot_pos.shape[0],
            robot_quat.shape[0],
        )
        if num_state_steps < 2:
            raise ValueError(f"{self.__class__.__name__}: state observations must contain at least 2 frames.")
        effector_pos = effector_pos[:num_state_steps].astype(np.float32, copy=False)  # [T+1,2]
        joint_pos = joint_pos[:num_state_steps].astype(np.float32, copy=False)  # [T+1,14]
        head_pos = head_pos[:num_state_steps].astype(np.float32, copy=False)  # [T+1,2]
        waist_pos = waist_pos[:num_state_steps].astype(np.float32, copy=False)  # [T+1,2]
        robot_pos = robot_pos[:num_state_steps].astype(np.float32, copy=False)  # [T+1,3]
        robot_quat = robot_quat[:num_state_steps].astype(np.float32, copy=False)  # [T+1,4]
        states_np = _assemble_embodiment_c_state(effector_pos, joint_pos, head_pos, waist_pos)  # [T+1,20]

        # Forward kinematics → absolute 4×4 transforms; gripper rotations are
        # first lifted by observed mobile-base motion, then converted through
        # _to_opencv for action/viewer display.
        link_poses = None
        if self._return_agibot_link_poses:
            link_poses = compute_link_poses_batch(states_np, "embodiment_c_gripper")  # {name:[T+1,4,4]}
            link_poses = apply_robot_base_motion_to_poses(link_poses, robot_pos, robot_quat)  # {name:[T+1,4,4]}
            native_fk = extract_fk_transforms_from_link_poses(link_poses)  # {name:[T+1,4,4]}
        else:
            native_fk = compute_fk_transforms_batch(states_np, "embodiment_c_gripper")  # {name:[T+1,4,4]}
            native_fk = apply_robot_base_motion_to_poses(native_fk, robot_pos, robot_quat)  # {name:[T+1,4,4]}
        fk = apply_agibot_gripper_to_opencv(native_fk, self._to_opencv)  # {name:[T+1,4,4]}

        # Relative SE(3) deltas.
        pose_convention = cast(PoseConvention, self._pose_convention)
        head_rel = pose_abs_to_rel(fk["head_camera"], rotation_format="rot6d", pose_convention=pose_convention)  # [T,9]
        right_rel = pose_abs_to_rel(
            fk["right_wrist"], rotation_format="rot6d", pose_convention=pose_convention
        )  # [T,9]
        left_rel = pose_abs_to_rel(fk["left_wrist"], rotation_format="rot6d", pose_convention=pose_convention)  # [T,9]

        # Gripper open fractions: Beta observed effector[0]=left,
        # effector[1]=right.
        # The converter standardizes URDF-angle and actuator-close-degree
        # encodings to 0.0=closed, 1.0=open for viewer consistency.
        right_gripper = convert_gripper_state_to_open_fraction(effector_pos[1:, 1:2])  # [T,1]
        left_gripper = convert_gripper_state_to_open_fraction(effector_pos[1:, 0:1])  # [T,1]

        # Concatenate in GEAR action order.
        action_np = np.concatenate(
            [head_rel, right_rel, right_gripper, left_rel, left_gripper],
            axis=-1,
        ).astype(np.float32)  # [T,29]

        extras = {
            "initial_pose": torch.from_numpy(fk["head_camera"][0].copy()).float(),  # [4,4]
            "initial_pose_right": torch.from_numpy(fk["right_wrist"][0].copy()).float(),  # [4,4]
            "initial_pose_left": torch.from_numpy(fk["left_wrist"][0].copy()).float(),  # [4,4]
        }
        if link_poses is not None:
            extras["agibot_link_poses"] = {
                link_name: torch.from_numpy(poses.copy()).float() for link_name, poses in link_poses.items()
            }  # {name:[T+1,4,4]}
        return torch.from_numpy(action_np).float(), extras  # [T,29]

    # -- Multi-view composition ----------------------------------------------

    def _compose_multi_view(self, sample: dict[str, Any]) -> torch.Tensor:
        """Compose head, left-hand, and right-hand views into a single frame.

        Returns:
            Composited video tensor in raw LeRobot ``(T, C, H_out, W)`` float format.
        """
        top = sample[_HEAD_KEY]  # [T,C,H,W]
        left = sample[_HAND_LEFT_KEY]  # [T,C,H_l,W_l]
        right = sample[_HAND_RIGHT_KEY]  # [T,C,H_r,W_r]

        _, _, h_top, w_top = top.shape
        half_h, half_w = h_top // 2, w_top // 2

        left = F.interpolate(left, size=(half_h, half_w), mode="bilinear", align_corners=False)  # [T,C,H/2,W/2]
        right = F.interpolate(right, size=(half_h, half_w), mode="bilinear", align_corners=False)  # [T,C,H/2,W/2]
        bottom = torch.cat([left, right], dim=-1)  # [T,C,H/2,W]

        composite = torch.cat([top, bottom], dim=-2)  # [T,C,3H/2,W]
        return composite  # [T,C,3H/2,W]

    def __getitem__(self, idx: int) -> dict[str, Any]:
        mode, _, _, sample = self._fetch_sample(idx)

        ai_caption, debug_caption = _split_task_for_caption(sample["task"])

        if self._skip_video_loading:
            video = None
        elif self._is_concat_view:
            video = self._compose_multi_view(sample)
        else:
            video = sample[_HEAD_KEY]  # [T,C,H,W]

        # Action: FK-pose based (same layout as GEAR gripper).
        action, action_extras = self._build_fk_action(sample)

        extras: dict[str, Any] = {**action_extras}
        if self._is_concat_view:
            extras["additional_view_description"] = (
                "The top row shows the head-mounted camera view looking down at the workspace. "
                "The bottom row contains two horizontally concatenated wrist-mounted camera views: "
                "the left hand camera on the left and the right hand camera on the right."
            )
        if debug_caption:
            extras["debug_caption"] = debug_caption

        return self._build_result(mode=mode, video=video, action=action, ai_caption=ai_caption, **extras)

    def _build_action_spec(self) -> ActionSpec:
        """AgiBotWorld-Beta bimanual layout (29D, same as GEAR gripper).

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
