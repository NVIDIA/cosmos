"""Hand-pose dataset constants: data roots, dataset keys, and skeleton topology."""

import numpy as np

# ---------------------------------------------------------------------------
# Embodiment_a LeRobot parquet keys
# ---------------------------------------------------------------------------
HAND_RIGHT_POSITION_KEY = "observation.state.hand_right_cam"
HAND_RIGHT_ROTATION_KEY = "observation.state.hand_right_cam_rotation"
HAND_LEFT_POSITION_KEY = "observation.state.hand_left_cam"
HAND_LEFT_ROTATION_KEY = "observation.state.hand_left_cam_rotation"
CAM_POSITION_KEY = "observation.state.camera_position"
CAM_ROTATION_KEY = "observation.state.camera_rotation"

# ---------------------------------------------------------------------------
# Skeleton topology
# ---------------------------------------------------------------------------
NUM_JOINTS = 21
QUAT_DIM_PER_JOINT = 4  # quaternion (qx, qy, qz, qw) as stored in parquet
MAT_DIM_PER_JOINT = 9  # flattened 3x3 rotation matrix (internal working format)
WRIST_JOINT_IDX = 0
FINGERTIP_JOINT_IDXS = (4, 8, 12, 16, 20)

ROTATION_FORMAT_DIM: dict[str, int] = {"rot9d": 9, "rot6d": 6, "euler_xyz": 3}

# ---------------------------------------------------------------------------
# Subtask-name filters for ``skip_no_action`` (exact / prefix / substring).
# Mirrors FilterConfig in
# pipelines/customers/dht/action_gen/run_extract_subtask_clips.py.
# Matching is done against the normalized name (underscores → spaces,
# stripped, lowercased).
# ---------------------------------------------------------------------------
NO_ACTION_SKIP_LABELS: tuple[str, ...] = ("no action", "no actions")
NO_ACTION_SKIP_LABEL_PREFIXES: tuple[str, ...] = ("hold", "adjust")
NO_ACTION_SKIP_LABEL_SUBSTRINGS: tuple[str, ...] = ("idle",)

# ---------------------------------------------------------------------------
# Wrist-frame alignment (Embodiment_a)
# ---------------------------------------------------------------------------
# 90° CCW rotation about local Z so that the wrist-local frame becomes:
#   X = thumb→pinky,  Y = palm normal (outward),  Z = wrist→fingertips
WRIST_FRAME_ALIGN_EMBODIMENT_A = np.array(
    [[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
    dtype=np.float32,
)

WRIST_FRAME_ALIGN_EMBODIMENT_A_IDENTITY = np.eye(4, dtype=np.float32)
# WRIST_FRAME_ALIGN_EMBODIMENT_A = np.array(
#     [[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
#     dtype=np.float32,
# )
# WRIST_FRAME_ALIGN_EMBODIMENT_A = WRIST_FRAME_ALIGN_EMBODIMENT_A_IDENTITY

# ---------------------------------------------------------------------------
# Embodiment_a sharded dataset roots
# ---------------------------------------------------------------------------
_EMBODIMENT_A_BASE = "/lustre/fsw/portfolios/cosmos/projects/cosmos_base_training/cosmos3_action_datasets/embodiment_a"
EMBODIMENT_A_FEB08_500HR = [f"{_EMBODIMENT_A_BASE}/feb_08_500hr_lerobot_no_bframes/shard_{i:02d}" for i in range(10)]
EMBODIMENT_A_FEB15_1500HR = [f"{_EMBODIMENT_A_BASE}/feb_15_1500hr_lerobot_no_bframes/shard_{i:02d}" for i in range(31)]
EMBODIMENT_A_FEB23_1000HR = [f"{_EMBODIMENT_A_BASE}/feb_23_1000hr_lerobot_no_bframes/shard_{i:02d}" for i in range(21)]
EMBODIMENT_A_MAR02_1000HR = [f"{_EMBODIMENT_A_BASE}/mar_02_1000hr_lerobot/shard_{i:02d}" for i in range(21)]
EMBODIMENT_A_MAR09_4000HR = [f"{_EMBODIMENT_A_BASE}/mar_09_4000hr_lerobot/shard_{i:02d}" for i in range(81)]
EMBODIMENT_A_MAR16_7000HR = [f"{_EMBODIMENT_A_BASE}/mar_16_7000hr_lerobot/shard_{i:02d}" for i in range(140)]
EMBODIMENT_A_MAR30_9000HR = [f"{_EMBODIMENT_A_BASE}/mar_30_9000hr_lerobot/shard_{i:02d}" for i in range(180)]
EMBODIMENT_A_APR03 = [f"{_EMBODIMENT_A_BASE}/apr_03_lerobot/shard_{i:02d}" for i in range(181)]
EMBODIMENT_A_APR06_10000HR = [f"{_EMBODIMENT_A_BASE}/apr_06_10000hr_lerobot/shard_{i:02d}" for i in range(202)]
EMBODIMENT_A_ALL = (
    EMBODIMENT_A_FEB08_500HR
    + EMBODIMENT_A_FEB15_1500HR
    + EMBODIMENT_A_FEB23_1000HR
    + EMBODIMENT_A_MAR02_1000HR
    + EMBODIMENT_A_MAR09_4000HR
    + EMBODIMENT_A_MAR16_7000HR
    + EMBODIMENT_A_MAR30_9000HR
    + EMBODIMENT_A_APR03
    + EMBODIMENT_A_APR06_10000HR
)

# ---------------------------------------------------------------------------
# Registry of all hand-pose datasets for hand-only experiments
# ---------------------------------------------------------------------------
HAND_POSE_DATASETS: dict[str, str | list[str]] = {
    "embodiment_a_feb08_500hr": EMBODIMENT_A_FEB08_500HR,
    "embodiment_a_feb15_1500hr": EMBODIMENT_A_FEB15_1500HR,
    "embodiment_a_feb23_1000hr": EMBODIMENT_A_FEB23_1000HR,
    "embodiment_a_mar02_1000hr": EMBODIMENT_A_MAR02_1000HR,
    "embodiment_a_mar09_4000hr": EMBODIMENT_A_MAR09_4000HR,
    "embodiment_a_mar16_7000hr": EMBODIMENT_A_MAR16_7000HR,
    "embodiment_a_mar30_9000hr": EMBODIMENT_A_MAR30_9000HR,
    "embodiment_a_apr03": EMBODIMENT_A_APR03,
    "embodiment_a_apr06_10000hr": EMBODIMENT_A_APR06_10000HR,
    "embodiment_a_all": EMBODIMENT_A_ALL,
    "embodiment_a_500hr_legacy_single": "/lustre/fsw/portfolios/dir/projects/dir_cosmos_base_lustre/qianlim/datasets/embodiment_a/feb_08_500hr_lerobot",
    "vitra_ego4d": [
        f"/lustre/fsw/portfolios/dir/projects/dir_cosmos_base_lustre/cosmos3_action_datasets/VITRA/ego4d/{res}"
        for res in ("810x1080", "1440x1080", "1920x1080", "1920x1440", "2560x1440", "2560x1920")
    ],
    "hwb_egoverse_eval_set_v0p1": "/lustre/fsw/portfolios/dir/projects/dir_cosmos_base_lustre/cosmos3_action_datasets/egocentric_eval_hwb/HWB_egoverse_v0p1_lerobot_v30/",
    "hwb_egoverse_eval_set_v0p2": "/lustre/fsw/portfolios/dir/projects/dir_cosmos_base_lustre/cosmos3_action_datasets/egocentric_eval_hwb/HWB_v0p2_lerobot_v3/",
}
