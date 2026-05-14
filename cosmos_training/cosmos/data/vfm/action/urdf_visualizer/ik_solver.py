"""Robot-agnostic IK solver using pinocchio + MuJoCo.

Supports any robot loaded from MJCF (Google Robot, Franka Panda, WidowX, etc).
Auto-detects EE frame, arm vs finger joints, and uses multi-start random seeding.

The solver:
1. Auto-discovers the EE frame from a list of candidate names
2. Determines which joints are "arm" joints (actuated for IK) vs "finger" joints
3. Uses multi-start random sampling to avoid local minima
4. Optionally sets finger joint angles from gripper opening fractions
"""

from functools import lru_cache
from typing import Any

import numpy as np

from cosmos.utils import log
from cosmos.data.vfm.action.embodiment_c_spec import (
    AGIBOT_GEAR_GRIPPER_OPEN_ANGLE_RAD,
    AGIBOT_GEAR_HEAD_CAMERA_LINK_NAME,
    AGIBOT_GEAR_LEFT_EE_LINK_NAME,
    AGIBOT_GEAR_LEFT_GRIPPER_JOINT_MIMICS,
    AGIBOT_GEAR_RIGHT_EE_LINK_NAME,
    AGIBOT_GEAR_RIGHT_GRIPPER_JOINT_MIMICS,
)
from cosmos.data.vfm.action.urdf_visualizer.robot_scene_model import (
    get_ee_frame_candidates,
    get_mujoco_to_pinocchio_world_transform,
    get_robot_config,
    get_urdf_path,
    get_visual_geom_ids,
    resolve_robot_name_from_mjcf,
)

# ── IK Solver ────────────────────────────────────────────────────────────────


def _find_ee_frame(model, robot_name: str | None = None) -> int | None:
    """Find the end-effector frame ID by trying candidate names.

    If robot_name is given and its config specifies ``ee_frame``, that name
    is tried first before falling through to the generic candidates.
    """
    cfg = get_robot_config(robot_name) if robot_name else {}
    override = cfg.get("ee_frame")
    if override:
        fid = model.getFrameId(override)
        if fid < model.nframes:
            return fid
        log.warning(f"Configured ee_frame '{override}' not found in Pinocchio model")

    for name in get_ee_frame_candidates():
        fid = model.getFrameId(name)
        if fid < model.nframes:
            return fid
    log.warning(f"Could not find EE frame by name for robot '{robot_name}', skipping IK")
    return None


@lru_cache(maxsize=1)
def _get_agibot_model() -> tuple[Any, dict[str, int]]:
    """Load the AgiBot URDF model and cache the key frame IDs."""

    import pinocchio as pin  # pyright: ignore[reportMissingImports]

    urdf_path = get_urdf_path("embodiment_c")
    if urdf_path is None:
        raise FileNotFoundError("AgiBot URDF path is unavailable")

    model = pin.buildModelFromUrdf(urdf_path)
    frame_ids = {
        "head": model.getFrameId(AGIBOT_GEAR_HEAD_CAMERA_LINK_NAME),
        "left": model.getFrameId(AGIBOT_GEAR_LEFT_EE_LINK_NAME),
        "right": model.getFrameId(AGIBOT_GEAR_RIGHT_EE_LINK_NAME),
    }
    for frame_name, frame_id in frame_ids.items():
        if frame_id >= model.nframes:
            raise ValueError(f"AgiBot frame {frame_name!r} missing from Pinocchio model")
    return model, frame_ids


def _set_agibot_gripper_joint_configs(
    model: Any,
    joint_configs: np.ndarray,
    *,
    left_gripper_openings: np.ndarray | None,
    right_gripper_openings: np.ndarray | None,
) -> None:
    """Write AgiBot omnipicker finger joint angles from open fractions."""

    side_specs = (
        ("left", left_gripper_openings, AGIBOT_GEAR_LEFT_GRIPPER_JOINT_MIMICS),
        ("right", right_gripper_openings, AGIBOT_GEAR_RIGHT_GRIPPER_JOINT_MIMICS),
    )
    lower = model.lowerPositionLimit
    upper = model.upperPositionLimit
    num_steps = int(joint_configs.shape[0])

    for side, openings, joint_mimics in side_specs:
        if openings is None:
            continue
        if len(openings) != num_steps:
            log.warning(
                f"AgiBot {side} gripper openings length {len(openings)} does not match IK frames {num_steps}; "
                "leaving finger joints at neutral."
            )
            continue

        joint_indices: list[tuple[int, float, float]] = []
        for joint_name, multiplier, offset in joint_mimics:
            joint_id = model.getJointId(joint_name)
            if joint_id >= model.njoints:
                log.warning(f"AgiBot {side} gripper joint '{joint_name}' not found in Pinocchio model")
                continue
            joint_indices.append((model.idx_qs[joint_id], multiplier, offset))

        for step_idx, opening in enumerate(openings):
            primary_angle = -float(np.clip(opening, 0.0, 1.0)) * AGIBOT_GEAR_GRIPPER_OPEN_ANGLE_RAD
            for q_idx, multiplier, offset in joint_indices:
                joint_value = multiplier * primary_angle + offset
                joint_configs[step_idx, q_idx] = np.clip(joint_value, lower[q_idx], upper[q_idx])


def solve_agibot_trajectory_ik(
    head_camera_poses: np.ndarray | None,
    left_wrist_poses: np.ndarray | None,
    right_wrist_poses: np.ndarray | None,
    left_gripper_openings: np.ndarray | None = None,
    right_gripper_openings: np.ndarray | None = None,
    max_iter: int = 200,
    dt: float = 0.2,
    damp: float = 1e-4,
    pos_tol_m: float = 5e-3,
    rot_tol_rad: float = 5e-2,
) -> np.ndarray | None:
    """Solve full-body AgiBot IK from calibrated head-camera and gripper-base trajectories."""

    import pinocchio as pin  # pyright: ignore[reportMissingImports]

    targets: list[np.ndarray] = [
        poses for poses in (head_camera_poses, left_wrist_poses, right_wrist_poses) if poses is not None
    ]
    if not targets:
        return None

    num_steps = int(targets[0].shape[0])
    if any(int(poses.shape[0]) != num_steps for poses in targets):
        raise ValueError("AgiBot IK requires head/left/right trajectories to have matching lengths")

    model, frame_ids = _get_agibot_model()
    data = model.createData()
    lower = model.lowerPositionLimit.copy()
    upper = model.upperPositionLimit.copy()

    head_targets = head_camera_poses.astype(np.float32, copy=False) if head_camera_poses is not None else None
    left_targets = left_wrist_poses.astype(np.float32, copy=False) if left_wrist_poses is not None else None
    right_targets = right_wrist_poses.astype(np.float32, copy=False) if right_wrist_poses is not None else None

    task_specs: list[tuple[str, np.ndarray, int, float]] = []
    if head_targets is not None:
        task_specs.append(("head", head_targets, frame_ids["head"], 0.75))
    if left_targets is not None:
        task_specs.append(("left", left_targets, frame_ids["left"], 1.0))
    if right_targets is not None:
        task_specs.append(("right", right_targets, frame_ids["right"], 1.0))

    joint_configs = np.empty((num_steps, model.nq), dtype=np.float32)
    q = pin.neutral(model)
    rot_weight = 0.35

    for step_idx in range(num_steps):
        best_q = q.copy()
        best_total = float("inf")
        best_pos = float("inf")
        best_rot = float("inf")

        for _ in range(max_iter):
            pin.forwardKinematics(model, data, q)
            pin.updateFramePlacements(model, data)

            errors: list[np.ndarray] = []
            jacobians: list[np.ndarray] = []
            max_pos_error = 0.0
            max_rot_error = 0.0

            for _, poses, frame_id, task_weight in task_specs:
                target = poses[step_idx]
                placement = data.oMf[frame_id]
                pos_err = target[:3, 3] - placement.translation
                rot_err = pin.log3(target[:3, :3] @ placement.rotation.T)
                max_pos_error = max(max_pos_error, float(np.linalg.norm(pos_err)))
                max_rot_error = max(max_rot_error, float(np.linalg.norm(rot_err)))

                err6 = np.concatenate([pos_err, rot_weight * rot_err]).astype(np.float64, copy=False)
                errors.append(task_weight * err6)

                jacobian = pin.computeFrameJacobian(model, data, q, frame_id, pin.LOCAL_WORLD_ALIGNED).copy()
                jacobian[3:, :] *= rot_weight
                jacobians.append(task_weight * jacobian)

            error_vec = np.concatenate(errors, axis=0)
            total_error = float(np.linalg.norm(error_vec))
            if total_error < best_total:
                best_total = total_error
                best_q = q.copy()
                best_pos = max_pos_error
                best_rot = max_rot_error

            if max_pos_error < pos_tol_m and max_rot_error < rot_tol_rad:
                break

            stacked_jacobian = np.concatenate(jacobians, axis=0)
            normal = stacked_jacobian @ stacked_jacobian.T + damp * np.eye(stacked_jacobian.shape[0])
            velocity = stacked_jacobian.T @ np.linalg.solve(normal, error_vec)
            q = pin.integrate(model, q, velocity * dt)
            q = np.clip(q, lower, upper)

        q = best_q.copy()
        joint_configs[step_idx] = q.astype(np.float32, copy=False)
        log.info(f"AgiBot IK frame {step_idx}: max_pos={best_pos * 1000:.2f}mm, max_rot={np.degrees(best_rot):.2f}deg")

    _set_agibot_gripper_joint_configs(
        model,
        joint_configs,
        left_gripper_openings=left_gripper_openings,
        right_gripper_openings=right_gripper_openings,
    )
    return joint_configs


def compute_agibot_link_poses_batch_from_configs(
    joint_configs: np.ndarray,
    link_names: list[str],
) -> dict[str, np.ndarray]:
    """Compute AgiBot URDF link poses from solved full-body joint configurations."""

    import pinocchio as pin  # pyright: ignore[reportMissingImports]

    if joint_configs.size == 0:
        return {link_name: np.empty((0, 4, 4), dtype=np.float32) for link_name in link_names}

    model, _ = _get_agibot_model()
    data = model.createData()
    frame_ids = {link_name: model.getFrameId(link_name) for link_name in link_names}
    for link_name, frame_id in frame_ids.items():
        if frame_id >= model.nframes:
            raise ValueError(f"AgiBot link frame {link_name!r} missing from Pinocchio model")

    num_steps = int(joint_configs.shape[0])
    link_poses = {link_name: np.empty((num_steps, 4, 4), dtype=np.float32) for link_name in link_names}

    for step_idx in range(num_steps):
        q = joint_configs[step_idx].astype(np.float64, copy=False)
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        for link_name, frame_id in frame_ids.items():
            placement = data.oMf[frame_id]
            transform = np.eye(4, dtype=np.float32)
            transform[:3, :3] = placement.rotation.astype(np.float32, copy=False)
            transform[:3, 3] = placement.translation.astype(np.float32, copy=False)
            link_poses[link_name][step_idx] = transform

    return link_poses


def solve_trajectory_ik(
    mjcf_path: str,
    world_ee_positions: np.ndarray,
    gripper_openings: np.ndarray | None = None,
    world_ee_orientations: np.ndarray | None = None,
    robot_name: str | None = None,
    max_random_samples: int = 50_000,
    seed: int = 42,
) -> np.ndarray | None:
    """Solve IK for a sequence of world-space EE poses (robot-agnostic).

    Args:
        mjcf_path: Path to the MJCF XML file.
        world_ee_positions: (T, 3) target EE positions.
        gripper_openings: (T,) gripper opening fractions [0=closed, 1=open].
        world_ee_orientations: (T, 3, 3) target EE rotation matrices.
        robot_name: Robot name for config lookup (optional, inferred from path).
        max_random_samples: Number of random configs to try for initial seed.
        seed: Random seed.

    Returns:
        (T, nq) joint configurations, or None if IK fails.
    """
    import pinocchio as pin

    model, cfg = _build_pinocchio_model(mjcf_path, robot_name)
    log.info(f"IK: pinocchio model nq={model.nq}")
    data = model.createData()
    ee_id = _find_ee_frame(model, robot_name)
    if ee_id is None:
        return None
    ee_name = model.frames[ee_id].name
    log.info(f"IK: using EE frame '{ee_name}' (id={ee_id}), nq={model.nq}")

    T = len(world_ee_positions)
    use_6dof = world_ee_orientations is not None and len(world_ee_orientations) == T

    # Apply TCP offset: dataset EE poses may be at the TCP (e.g. ee_gripper_link)
    # while the Pinocchio frame is at the kinematic link origin (e.g. gripper_link).
    # Convert target TCP poses → IK link-frame targets:
    #   p_link = p_tcp - R_tcp @ tcp_offset
    tcp_offset = cfg.get("tcp_offset")
    if tcp_offset is not None:
        tcp_offset = np.asarray(tcp_offset, dtype=np.float32)
        log.info(f"IK: applying TCP offset {tcp_offset} to target positions")
        world_ee_positions = world_ee_positions.copy()
        for t in range(T):
            if use_6dof:
                world_ee_positions[t] -= world_ee_orientations[t] @ tcp_offset
            else:
                world_ee_positions[t] -= tcp_offset

    lower = model.lowerPositionLimit.copy()
    upper = model.upperPositionLimit.copy()

    # Determine arm joints vs finger joints
    # After model reduction, arm joints are first, fingers follow
    n_arm = cfg.get("n_arm_joints", model.nq - 2)  # default: all but last 2 are arm
    n_finger = model.nq - n_arm

    log.info(f"IK: {n_arm} arm joints + {n_finger} finger joints")

    # ── 6-DoF CLIK (position + orientation) ──
    def _ik_6dof(target_pos, target_rot, q_init, max_iter=800, eps_pos=5e-5, eps_rot=1e-3, dt=0.1, damp=1e-4):
        q = q_init.copy()
        best_q = q.copy()
        best_total = float("inf")
        stall_count = 0
        rot_weight = 1.0

        for it in range(max_iter):
            pin.forwardKinematics(model, data, q)
            pin.updateFramePlacements(model, data)

            pos_err = target_pos - data.oMf[ee_id].translation
            pos_norm = np.linalg.norm(pos_err)
            R_err = target_rot @ data.oMf[ee_id].rotation.T
            rot_err = pin.log3(R_err)
            rot_norm = np.linalg.norm(rot_err)

            if pos_norm < eps_pos and rot_norm < eps_rot:
                return q, pos_norm, rot_norm, it + 1

            total = pos_norm + 0.05 * rot_norm
            if total < best_total:
                best_total = total
                best_q = q.copy()
                stall_count = 0
            else:
                stall_count += 1

            if stall_count > 50 and rot_weight > 0.1:
                rot_weight *= 0.8
                stall_count = 0
            if stall_count > 150:
                break

            err6 = np.concatenate([pos_err, rot_weight * rot_err])
            J = pin.computeFrameJacobian(model, data, q, ee_id, pin.LOCAL_WORLD_ALIGNED).copy()
            J[3:, :] *= rot_weight
            # Zero out finger joint columns
            J[:, n_arm:] = 0
            JJt = J @ J.T + damp * np.eye(6)
            v = J.T @ np.linalg.solve(JJt, err6)
            v[n_arm:] = 0  # don't move finger joints

            q = pin.integrate(model, q, v * dt)
            q = np.clip(q, lower, upper)

        pin.forwardKinematics(model, data, best_q)
        pin.updateFramePlacements(model, data)
        pos_norm = np.linalg.norm(target_pos - data.oMf[ee_id].translation)
        rot_norm = np.linalg.norm(pin.log3(target_rot @ data.oMf[ee_id].rotation.T))
        return best_q, pos_norm, rot_norm, max_iter

    # ── 3-DoF CLIK (position only) ──
    def _ik_3dof(target_pos, q_init, max_iter=500, eps=5e-5, dt=0.15, damp=1e-4):
        q = q_init.copy()
        for it in range(max_iter):
            pin.forwardKinematics(model, data, q)
            pin.updateFramePlacements(model, data)
            err = target_pos - data.oMf[ee_id].translation
            if np.linalg.norm(err) < eps:
                return q, np.linalg.norm(err), it + 1
            J = pin.computeFrameJacobian(model, data, q, ee_id, pin.LOCAL_WORLD_ALIGNED)[:3]
            J[:, n_arm:] = 0
            v = J.T @ np.linalg.solve(J @ J.T + damp * np.eye(3), err)
            v[n_arm:] = 0
            q = pin.integrate(model, q, v * dt)
            q = np.clip(q, lower, upper)
        return q, np.linalg.norm(err), max_iter

    def _solve_full_trajectory(seed_q):
        configs = []
        max_pe = 0.0
        max_re = 0.0
        q = seed_q.copy()
        for t in range(T):
            if use_6dof:
                q, pe, re, _ = _ik_6dof(world_ee_positions[t], world_ee_orientations[t], q)
                max_pe = max(max_pe, float(pe))
                max_re = max(max_re, float(re))
            else:
                q, pe, _ = _ik_3dof(world_ee_positions[t], q)
                max_pe = max(max_pe, float(pe))
            configs.append(q.copy())
        return np.array(configs), max_pe, max_re

    # ── Multi-start seed search ──
    # For robots with a base rotation joint (Google Robot torso, WidowX waist),
    # search multiple rotation basins. For Franka (no base rotation freedom),
    # use a single wider search.
    base_joint_range = upper[0] - lower[0]
    if base_joint_range > 4.0:
        # Wide base rotation — split into basins (Google Robot, WidowX)
        n_basins = 4
        basin_size = base_joint_range / n_basins
        basins = []
        for i in range(n_basins):
            b_lo = lower[0] + i * basin_size
            b_hi = lower[0] + (i + 1) * basin_size
            basins.append((b_lo, b_hi))
    else:
        # No wide base rotation — single basin (Franka)
        basins = [(lower[0], upper[0])]

    samples_per_basin = max_random_samples // len(basins)
    target0_pos = world_ee_positions[0]
    target0_rot = world_ee_orientations[0] if use_6dof else None

    best_overall_configs = None
    best_overall_score = float("inf")
    best_basin_info = ""
    best_max_pe = 0.0
    best_max_re = 0.0

    for basin_idx, (b_lo, b_hi) in enumerate(basins):
        rng = np.random.RandomState(seed + basin_idx)
        basin_lower = lower.copy()
        basin_upper = upper.copy()
        basin_lower[0] = max(lower[0], b_lo)
        basin_upper[0] = min(upper[0], b_hi)

        # Find best seed in this basin
        basin_best_q = pin.neutral(model)
        basin_best_q[0] = (b_lo + b_hi) / 2
        basin_best_score = float("inf")

        for _ in range(samples_per_basin):
            q = rng.uniform(basin_lower, basin_upper)
            pin.forwardKinematics(model, data, q)
            pin.updateFramePlacements(model, data)
            pos_err = np.linalg.norm(data.oMf[ee_id].translation - target0_pos)

            if target0_rot is not None:
                rot_err = np.linalg.norm(pin.log3(target0_rot.T @ data.oMf[ee_id].rotation))
                score = pos_err + 0.3 * rot_err
            else:
                score = pos_err

            if score < basin_best_score:
                basin_best_score = score
                basin_best_q = q.copy()
                if pos_err < 0.005 and (target0_rot is None or rot_err < 0.1):
                    break

        if basin_best_score > 0.5:
            continue

        configs, max_pe, max_re = _solve_full_trajectory(basin_best_q)
        traj_score = max_pe + 0.05 * max_re
        log.info(
            f"  Basin [{b_lo:+.1f}, {b_hi:+.1f}]: seed_score={basin_best_score:.4f}, "
            f"traj max_pos={max_pe * 1000:.1f}mm, max_rot={np.degrees(max_re):.1f}°, "
            f"j0={basin_best_q[0]:+.2f}rad"
        )

        if traj_score < best_overall_score:
            best_overall_score = traj_score
            best_overall_configs = configs
            best_basin_info = f"j0_basin=[{b_lo:+.1f},{b_hi:+.1f}], seed_j0={basin_best_q[0]:+.2f}rad"
            best_max_pe = max_pe
            best_max_re = max_re

    if best_overall_configs is None:
        log.warning("IK failed: no basin converged")
        return None

    configs = best_overall_configs
    if use_6dof:
        log.info(
            f"IK solved ({T} frames, 6-DoF): max_pos={best_max_pe * 1000:.2f}mm, max_rot={np.degrees(best_max_re):.1f}° [{best_basin_info}]"
        )
    else:
        log.info(f"IK solved ({T} frames, 3-DoF): max_pos={best_max_pe * 1000:.2f}mm [{best_basin_info}]")

    # ── Set finger joints from gripper openings ──
    if n_finger > 0:
        finger_min = cfg.get("finger_min", lower[n_arm])
        finger_max = cfg.get("finger_max", upper[n_arm])
    else:
        finger_min = 0.0
        finger_max = 0.0
    close_is_max = cfg.get("finger_close_is_max", True)
    finger_joint_names = cfg.get("finger_joint_names")

    if gripper_openings is not None and len(gripper_openings) == T:
        # Find finger joint indices by name if specified (e.g., Robotiq driver joints)
        if finger_joint_names:
            # Use Pinocchio joint name lookup
            finger_indices = []
            for fjn in finger_joint_names:
                # Pinocchio joint names include the joint name from MJCF
                jid = model.getJointId(fjn)
                if jid < model.njoints:
                    # Pinocchio joint index → qpos index
                    qi = model.idx_qs[jid]
                    finger_indices.append(qi)
                else:
                    log.warning(f"Finger joint '{fjn}' not found in Pinocchio model")
            if not finger_indices:
                finger_indices = list(range(n_arm, n_arm + n_finger))
        else:
            finger_indices = list(range(n_arm, n_arm + n_finger))

        for t in range(T):
            g = float(np.clip(gripper_openings[t], 0.0, 1.0))
            if close_is_max:
                # Robotiq/Google Robot: high angle = closed
                angle = finger_max - g * (finger_max - finger_min)
            else:
                # Franka/WidowX: high value = open
                angle = finger_min + g * (finger_max - finger_min)

            for ji in finger_indices:
                # WidowX right finger is negative range
                if lower[ji] < 0 and upper[ji] < 0:
                    configs[t, ji] = -(finger_min + g * (finger_max - finger_min))
                else:
                    configs[t, ji] = angle
        log.info(
            f"Finger joints set from gripper openings ({gripper_openings.min():.2f} to {gripper_openings.max():.2f})"
        )

    return configs


def _build_pinocchio_model(mjcf_path: str, robot_name: str | None = None):
    """Build a pinocchio model, reducing to arm+finger joints if configured.

    For URDF models (SimplerEnv), builds from URDF and locks non-arm joints.
    For MJCF models (Menagerie), builds from MJCF and locks non-arm joints.
    Shared by solve_trajectory_ik and compute_fk_ee_poses.
    """
    import pinocchio as pin

    cfg = get_robot_config(robot_name) if robot_name else {}
    urdf_path = get_urdf_path(robot_name) if robot_name else None

    if urdf_path:
        full_model = pin.buildModelFromUrdf(urdf_path)
    else:
        full_model = pin.buildModelFromMJCF(mjcf_path)

    # Reduce model to arm + finger joints only (lock base, wheels, etc.)
    arm_joint_names = cfg.get("arm_joints", [])
    finger_jnames = cfg.get("finger_joint_names", [])
    keep_names = set(arm_joint_names) | set(finger_jnames)

    if keep_names:
        lock_ids = [ji for ji in range(1, full_model.njoints) if full_model.names[ji] not in keep_names]
        if lock_ids:
            q_ref = pin.neutral(full_model)
            model = pin.buildReducedModel(full_model, lock_ids, q_ref)
        else:
            model = full_model
    else:
        model = full_model

    return model, cfg


def compute_fk_ee_poses(
    mjcf_path: str,
    joint_configs: np.ndarray,
    robot_name: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Run FK and return EE positions and orientations."""
    import pinocchio as pin

    model, _ = _build_pinocchio_model(mjcf_path, robot_name)
    data = model.createData()
    ee_id = _find_ee_frame(model, robot_name)
    if ee_id is None:
        log.warning(f"Skipping FK — no EE frame found for robot '{robot_name}'")
        T = len(joint_configs)
        return np.zeros((T, 3)), np.zeros((T, 3, 3))

    T = len(joint_configs)
    fk_positions = np.zeros((T, 3))
    fk_orientations = np.zeros((T, 3, 3))

    for t in range(T):
        pin.forwardKinematics(model, data, joint_configs[t])
        pin.updateFramePlacements(model, data)
        fk_positions[t] = data.oMf[ee_id].translation.copy()
        fk_orientations[t] = data.oMf[ee_id].rotation.copy()

    return fk_positions, fk_orientations


def verify_ik_with_fk(
    mjcf_path: str,
    joint_configs: np.ndarray,
    target_positions: np.ndarray,
    target_orientations: np.ndarray | None = None,
) -> dict:
    """Verify IK solution by running FK and comparing to targets."""
    import pinocchio as pin

    fk_pos, fk_rot = compute_fk_ee_poses(mjcf_path, joint_configs)
    if fk_pos is None:
        return None

    T = len(joint_configs)
    pos_errors_mm = np.linalg.norm(fk_pos - target_positions, axis=1) * 1000

    rot_errors_deg = None
    if target_orientations is not None:
        rot_errors_deg = np.zeros(T)
        for t in range(T):
            R_err = target_orientations[t].T @ fk_rot[t]
            angle = np.linalg.norm(pin.log3(R_err))
            rot_errors_deg[t] = np.degrees(angle)

    summary = f"FK Verification ({T} frames): pos mean={pos_errors_mm.mean():.2f}mm max={pos_errors_mm.max():.2f}mm"
    if rot_errors_deg is not None:
        summary += f", rot mean={rot_errors_deg.mean():.1f}° max={rot_errors_deg.max():.1f}°"

    return {
        "fk_positions": fk_pos,
        "fk_orientations": fk_rot,
        "pos_errors_mm": pos_errors_mm,
        "rot_errors_deg": rot_errors_deg,
        "summary": summary,
    }


def compute_mujoco_geom_transforms(
    mjcf_path: str,
    joint_configs: np.ndarray,
) -> tuple[
    list[list[tuple[np.ndarray, np.ndarray]]],
    list[tuple[np.ndarray, np.ndarray]] | None,
    list[tuple[np.ndarray, np.ndarray]] | None,
    dict[str, list[tuple[np.ndarray, np.ndarray]]] | None,
]:
    """Compute MuJoCo geom/body/site transforms in the Pinocchio-aligned world.

    Also extracts camera site pose, EE body pose, and named body/site frames.

    Important: some MJCFs (notably MuJoCo Menagerie's Google Robot) include a
    fixed ``worldbody -> root_body`` transform that Pinocchio's
    ``buildModelFromMJCF()`` omits. Dataset poses and IK targets already live in
    Pinocchio's root-free world, so we explicitly remove that MuJoCo root
    transform here before returning any MuJoCo-derived poses.

    Returns:
        (all_geom_transforms, camera_poses_or_None, ee_poses_or_None, robot_frames_or_None)
        - all_geom_transforms: list of per-frame geom transforms [(pos, mat), ...]
        - camera_poses: list of (pos, mat) per frame for 'camera_site', or None if no site.
        - ee_poses: list of (pos, mat) per frame for the EE body, or None.
        - robot_frames: dict mapping ``body:<name>`` / ``site:<name>`` to per-frame poses.
    """
    import mujoco

    model = mujoco.MjModel.from_xml_path(mjcf_path)
    data = mujoco.MjData(model)

    visual_geom_ids = get_visual_geom_ids(model)

    # Determine which robot config applies (by matching MJCF filename)
    robot_name = resolve_robot_name_from_mjcf(mjcf_path)
    cfg = get_robot_config(robot_name) if robot_name is not None else {}

    # Find camera_site if it exists
    camera_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "camera_site")
    has_camera_site = camera_site_id >= 0
    if has_camera_site:
        body_id = model.site_bodyid[camera_site_id]
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or "?"
        log.info(f"Found camera_site (id={camera_site_id}) on body '{body_name}'")

    # Also look up camera body (e.g. zed_mini for DROID)
    camera_body_id = -1
    camera_body_name = cfg.get("camera_body")
    if camera_body_name and not has_camera_site:
        camera_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, camera_body_name)
        if camera_body_id >= 0:
            log.info(f"Found camera body '{camera_body_name}' (id={camera_body_id})")

    all_transforms = []
    camera_poses = [] if (has_camera_site or camera_body_id >= 0) else None
    robot_frame_specs = []
    for body_id in range(1, model.nbody):
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
        if body_name:
            robot_frame_specs.append(("body", body_name, body_id))
    for site_id in range(model.nsite):
        site_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, site_id) or ""
        if site_name:
            robot_frame_specs.append(("site", site_name, site_id))
    robot_frames = {f"{kind}:{name}": [] for kind, name, _ in robot_frame_specs} if robot_frame_specs else None

    # Find EE body for extracting FK-derived EE pose
    ee_body_id = -1
    ee_override = cfg.get("ee_frame")
    ee_candidates = get_ee_frame_candidates(robot_name)
    for candidate in ee_candidates:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, candidate)
        if bid >= 0:
            ee_body_id = bid
            break
    ee_poses = [] if ee_body_id >= 0 else None

    # Find driver joint indices for robots with finger_joint_names
    finger_joint_names = cfg.get("finger_joint_names", [])
    arm_jnames = cfg.get("arm_joints", []) if cfg else []
    # Indices to pin during constraint settling: arm joints + driver joints
    n_arm = cfg.get("n_arm_joints", 7) if cfg else 7
    if arm_jnames:
        # Use name-based mapping for arm joint indices
        pin_indices = []
        for jn in arm_jnames:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
            if jid >= 0:
                pin_indices.append(model.jnt_qposadr[jid])
    else:
        pin_indices = list(range(n_arm))
    if finger_joint_names:
        for fjn in finger_joint_names:
            for ji in range(model.njnt):
                jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, ji) or ""
                if jname == fjn:
                    qi = model.jnt_qposadr[ji]
                    pin_indices.append(qi)

    # Build mapping from IK output (pinocchio reduced model) to MuJoCo qpos indices.
    # When a URDF is used with model reduction, the IK output has only arm+finger
    # joints in pinocchio order. We need to map those to MuJoCo's qpos order.
    arm_jnames = cfg.get("arm_joints", []) if cfg else []
    pin_to_mj_map = None  # None = direct mapping (qpos[:len(q)] = q)
    if arm_jnames:
        # Build ordered list: arm joints first, then finger joints
        ordered_jnames = list(arm_jnames) + list(finger_joint_names)
        mj_indices = []
        for jn in ordered_jnames:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
            if jid >= 0:
                mj_indices.append(model.jnt_qposadr[jid])
        if mj_indices:
            pin_to_mj_map = mj_indices

    # If joint_configs has more columns than n_arm, the extra column is a
    # normalized gripper signal (raw UR: 0=open, 1=closed). Robotiq ctrl
    # matches: 0=open, 255=closed → ctrl = raw * finger_max (no inversion).
    finger_max = cfg.get("finger_max", 0.0) if cfg else 0.0
    has_gripper_ctrl = finger_max > 0.0 and model.nu > n_arm and joint_configs.shape[1] > n_arm

    def _apply_world_correction(pos: np.ndarray, mat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Map one MuJoCo world pose into the root-free Pinocchio world."""
        corrected_pos = world_correction[:3, :3] @ pos + world_correction[:3, 3]
        corrected_mat = world_correction[:3, :3] @ mat
        return corrected_pos.astype(np.float32), corrected_mat.astype(np.float32)

    for q in joint_configs:
        if pin_to_mj_map and len(q) == len(pin_to_mj_map):
            # Full arm+finger mapping from IK (e.g. Franka): use all joints.
            data.qpos[:] = 0
            for i, mi in enumerate(pin_to_mj_map):
                data.qpos[mi] = q[i]
        else:
            # For robots with a separate gripper ctrl signal (e.g. UR5e),
            # only write arm joints — the 7th column is a raw gripper value,
            # not a qpos DOF. For other robots the pinocchio output already
            # includes finger joints in the trailing columns; write them all.
            n_set = n_arm if has_gripper_ctrl else len(q)
            data.qpos[:n_set] = q[:n_set]
        mujoco.mj_forward(model, data)

        # For robots with equality constraints (e.g., Robotiq 4-bar linkage),
        # step physics to let constraints resolve the passive linkage joints.
        if model.neq > 0:
            data.qvel[:] = 0
            data.ctrl[:] = 0
            # Raw UR gripper maps directly to Robotiq ctrl: 0=open, 255=closed.
            if has_gripper_ctrl:
                data.ctrl[-1] = float(q[n_arm]) * finger_max
            gripper_ctrl_val = float(data.ctrl[-1]) if model.nu > 0 else 0.0
            saved = data.qpos[pin_indices].copy()
            for _ in range(200):
                mujoco.mj_step(model, data)
                data.qpos[pin_indices] = saved
                if model.nu > 0:
                    data.ctrl[-1] = gripper_ctrl_val  # keep gripper ctrl during settling
                data.qvel[:] = 0
            mujoco.mj_forward(model, data)
        world_correction = get_mujoco_to_pinocchio_world_transform(model, data, robot_name)

        frame_transforms = []
        for gi in visual_geom_ids:
            pos = data.geom_xpos[gi].copy()
            mat = data.geom_xmat[gi].reshape(3, 3).copy()
            frame_transforms.append(_apply_world_correction(pos, mat))
        all_transforms.append(frame_transforms)

        # Extract camera site pose
        if has_camera_site:
            cam_pos = data.site_xpos[camera_site_id].copy()
            cam_mat = data.site_xmat[camera_site_id].reshape(3, 3).copy()
            camera_poses.append(_apply_world_correction(cam_pos, cam_mat))
        elif camera_body_id >= 0:
            cam_pos = data.xpos[camera_body_id].copy()
            cam_mat = data.xmat[camera_body_id].reshape(3, 3).copy()
            camera_poses.append(_apply_world_correction(cam_pos, cam_mat))

        # Extract EE body pose
        if ee_body_id >= 0:
            ee_pos = data.xpos[ee_body_id].copy()
            ee_mat = data.xmat[ee_body_id].reshape(3, 3).copy()
            ee_poses.append(_apply_world_correction(ee_pos, ee_mat))

        if robot_frames is not None:
            for kind, name, frame_id in robot_frame_specs:
                if kind == "body":
                    pos = data.xpos[frame_id].copy()
                    mat = data.xmat[frame_id].reshape(3, 3).copy()
                else:
                    pos = data.site_xpos[frame_id].copy()
                    mat = data.site_xmat[frame_id].reshape(3, 3).copy()
                robot_frames[f"{kind}:{name}"].append(_apply_world_correction(pos, mat))

    return all_transforms, camera_poses, ee_poses, robot_frames
