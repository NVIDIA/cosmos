from hydra.core.config_store import ConfigStore

from cosmos.utils.lazy_config import LazyCall as L
from configs.base.experiment.action.midtrain_config.action_datasets_v1p2 import (
    DATASET_AV_480,
    DATASET_CAMERA_256,
    DATASET_CAMERA_480,
    DATASET_CAMERA_720,
)
from configs.base.experiment.action.pretrained_config.cosmos3_8b import make_8b_experiment
from cosmos.data.vfm.action.av_dataset import AVDataset
from cosmos.data.vfm.action.camera_dataset_sharded import CAMERA_WDINFOS, CameraDatasetSharded
from cosmos.data.vfm.action.unified_dataset import dataset_entry

cs = ConfigStore.instance()

_RECIPE_V5_CKPT = (
    "cosmos3_vfm/t2w_mot_8b_qwen3_vl_runs/t2w_mot_exp302_000_qwen3_vl_8b_multires_recipe_v5/checkpoints/iter_000005500/"
)
_RECIPE_V7_CKPT = (
    "cosmos3_vfm/t2w_mot_8b_qwen3_vl_runs/t2w_mot_exp302_000_qwen3_vl_8b_multires_recipe_v7/checkpoints/iter_000030000/"
)
_MIDTRAIN_CKPT = "cosmos3_vfm/t2w_mot_8b_qwen3_vl_runs/t2w_mot_exp506_000_qwen3_vl_8b_multires_recipe_midtraining_v1/checkpoints/iter_000050000/"

_COMPILE_TOKENIZER = dict(
    enabled=True,
    warmup_resolutions=["256", "480", "720"],
)


def _make_exp(name: str, datasets: list) -> dict:
    """Create an 8B experiment with common camera/AV overrides applied."""
    exp = make_8b_experiment(name, datasets=datasets, num_workers=4, training_iterations=100_000)
    exp["job"]["group"] = "uva_camera"
    return exp


### framewise experiments
# camera-only joint
camera_8b_joint_300 = _make_exp(
    "camera_8b_joint_300",
    [DATASET_CAMERA_256, DATASET_CAMERA_480, DATASET_CAMERA_720],
)
camera_8b_joint_300["checkpoint"]["load_path"] = _MIDTRAIN_CKPT
cs.store(group="experiment", package="_global_", name="camera_8b_joint_300", node=camera_8b_joint_300)

# camera-only fd
DATASET_CAMERA_256_FD = L(dataset_entry)(
    name="camera_256_20260501",
    dataset=L(CameraDatasetSharded)(
        wdinfo_paths=[
            CAMERA_WDINFOS["pretrained_clips_260307_100k_filtered"],
            CAMERA_WDINFOS["pretrained_clips_260325_500k_01_filtered"],
            CAMERA_WDINFOS["pretrained_clips_260325_500k_02_filtered"],
            CAMERA_WDINFOS["pretrained_clips_260313_10s_100k_filtered"],
        ],
        split="train",
        shuffle=True,
        fix_caption=False,
        wdinfo_resolution="gt256",
        max_frames=400,
        translation_scale=10,
        max_action_translation_norm=10,
        mode="forward_dynamics",
        rotation_format="rot6d",
        pose_convention="backward_framewise",
    ),
    ratio=1,
    resolution="256",
)

DATASET_CAMERA_480_FD = L(dataset_entry)(
    name="camera_480_20260501",
    dataset=L(CameraDatasetSharded)(
        wdinfo_paths=[
            CAMERA_WDINFOS["pretrained_clips_260307_100k_filtered"],
            CAMERA_WDINFOS["pretrained_clips_260325_500k_01_filtered"],
            CAMERA_WDINFOS["pretrained_clips_260325_500k_02_filtered"],
            CAMERA_WDINFOS["pretrained_clips_260313_10s_100k_filtered"],
        ],
        split="train",
        shuffle=True,
        fix_caption=False,
        wdinfo_resolution="gt480",  # 480, 720, resize to 480
        max_frames=300,
        translation_scale=10,
        max_action_translation_norm=10,
        mode="forward_dynamics",
        rotation_format="rot6d",
        pose_convention="backward_framewise",
    ),
    ratio=1,
    resolution="480",
)

DATASET_CAMERA_720_FD = L(dataset_entry)(
    name="camera_720_20260501",
    dataset=L(CameraDatasetSharded)(
        wdinfo_paths=[
            CAMERA_WDINFOS["pretrained_clips_260307_100k_filtered"],
            CAMERA_WDINFOS["pretrained_clips_260325_500k_01_filtered"],
            CAMERA_WDINFOS["pretrained_clips_260325_500k_02_filtered"],
            CAMERA_WDINFOS["pretrained_clips_260313_10s_100k_filtered"],
        ],
        split="train",
        shuffle=True,
        fix_caption=False,
        wdinfo_resolution="gt720",  # only 720
        max_frames=200,
        translation_scale=10,
        max_action_translation_norm=10,
        mode="forward_dynamics",
        rotation_format="rot6d",
        pose_convention="backward_framewise",
    ),
    ratio=1,
    resolution="720",
)
camera_8b_fd_300 = _make_exp(
    "camera_8b_fd_300",
    [DATASET_CAMERA_256_FD, DATASET_CAMERA_480_FD, DATASET_CAMERA_720_FD],
)
camera_8b_fd_300["checkpoint"]["load_path"] = _MIDTRAIN_CKPT
cs.store(group="experiment", package="_global_", name="camera_8b_fd_300", node=camera_8b_fd_300)

# av-only joint
av_8b_joint_fw_011 = _make_exp(
    "av_8b_joint_fw_011",
    [DATASET_AV_480],
)
cs.store(group="experiment", package="_global_", name="av_8b_joint_fw_011", node=av_8b_joint_fw_011)

# camera-av joint
camera_av_8b_joint_fw_012 = _make_exp(
    "camera_av_8b_joint_fw_012",
    [DATASET_CAMERA_480, DATASET_AV_480],
)
cs.store(group="experiment", package="_global_", name="camera_av_8b_joint_fw_012", node=camera_av_8b_joint_fw_012)

# camera-av joint, continued from fw_012 iter 60k (dataset updated inplace)
camera_av_8b_joint_fw_013 = _make_exp(
    "camera_av_8b_joint_fw_013",
    [DATASET_CAMERA_480, DATASET_AV_480],
)
camera_av_8b_joint_fw_013["checkpoint"]["load_path"] = (
    "cosmos3_action/uva_camera/camera_av_8b_joint_fw_012/checkpoints/iter_000060000/"
)
camera_av_8b_joint_fw_013["checkpoint"]["load_training_state"] = False
camera_av_8b_joint_fw_013["checkpoint"]["keys_to_skip_loading"] = []
cs.store(group="experiment", package="_global_", name="camera_av_8b_joint_fw_013", node=camera_av_8b_joint_fw_013)

### new 16f framewise experiments

DATASET_AV_480_16F = L(dataset_entry)(
    name="av_480",
    dataset=L(AVDataset)(
        root=[
            "s3://nv-00-10206-robot/cosmos3_action_data/av_v2_02182026_wdinfo/",
            "s3://nv-00-10206-robot/cosmos3_action_data/av_v2_03292026_wdinfo/",
        ],
        split="train",
        fps=10,
        mode="joint",
        history_len=0.1,
        future_len=1.6,
        rotation_format="rot6d",
        pose_convention="backward_framewise",
        translation_scale=1.35,
        resolution="480",
        credential_path="${job.cluster.object_store_credential_data}",
        shuffle=True,
        include_route_in_prompt=True,
        use_semantic_route_prompt=True,
    ),
    ratio=1,
    resolution="480",
)

# av-only joint
av_8b_16f_joint_fw_011 = _make_exp(
    "av_8b_16f_joint_fw_011",
    [DATASET_AV_480_16F],
)
cs.store(group="experiment", package="_global_", name="av_8b_16f_joint_fw_011", node=av_8b_16f_joint_fw_011)


DATASET_CAMERA_480_16F = L(dataset_entry)(
    name="camera_480",
    dataset=L(CameraDatasetSharded)(
        wdinfo_paths=[
            CAMERA_WDINFOS["pretrained_clips_260307_100k_filtered"],
            CAMERA_WDINFOS["pretrained_clips_260325_500k_01_filtered"],
            CAMERA_WDINFOS["pretrained_clips_260325_500k_02_filtered"],
            CAMERA_WDINFOS["pretrained_clips_260313_10s_100k_filtered"],
        ],
        split="train",
        shuffle=True,
        fix_caption=False,
        wdinfo_resolution="gt480",  # 480, 720, resize to 480
        max_frames=17,
        translation_scale=50,
        mode="joint",
        rotation_format="rot6d",
        pose_convention="backward_framewise",
        max_action_translation_norm=10,
    ),
    ratio=1,
    resolution="480",
)

camera_8b_16f_joint_fw_thresh_011 = _make_exp(
    "camera_8b_16f_joint_fw_thresh_011",
    [DATASET_CAMERA_480_16F],
)
cs.store(
    group="experiment",
    package="_global_",
    name="camera_8b_16f_joint_fw_thresh_011",
    node=camera_8b_16f_joint_fw_thresh_011,
)

# ablate scale
DATASET_CAMERA_480_16F_THRESH_S30 = L(dataset_entry)(
    name="camera_480",
    dataset=L(CameraDatasetSharded)(
        wdinfo_paths=[
            CAMERA_WDINFOS["pretrained_clips_260307_100k_filtered"],
            CAMERA_WDINFOS["pretrained_clips_260325_500k_01_filtered"],
            CAMERA_WDINFOS["pretrained_clips_260325_500k_02_filtered"],
            CAMERA_WDINFOS["pretrained_clips_260313_10s_100k_filtered"],
        ],
        split="train",
        shuffle=True,
        fix_caption=False,
        wdinfo_resolution="gt480",  # 480, 720, resize to 480
        max_frames=17,
        translation_scale=30,
        mode="joint",
        rotation_format="rot6d",
        pose_convention="backward_framewise",
        max_action_translation_norm=10,
    ),
    ratio=1,
    resolution="480",
)

camera_8b_16f_joint_fw_s30_011 = _make_exp(
    "camera_8b_16f_joint_fw_s30_011",
    [DATASET_CAMERA_480_16F_THRESH_S30],
)
cs.store(
    group="experiment",
    package="_global_",
    name="camera_8b_16f_joint_fw_s30_011",
    node=camera_8b_16f_joint_fw_s30_011,
)


DATASET_CAMERA_480_16F_THRESH_S10 = L(dataset_entry)(
    name="camera_480",
    dataset=L(CameraDatasetSharded)(
        wdinfo_paths=[
            CAMERA_WDINFOS["pretrained_clips_260307_100k_filtered"],
            CAMERA_WDINFOS["pretrained_clips_260325_500k_01_filtered"],
            CAMERA_WDINFOS["pretrained_clips_260325_500k_02_filtered"],
            CAMERA_WDINFOS["pretrained_clips_260313_10s_100k_filtered"],
        ],
        split="train",
        shuffle=True,
        fix_caption=False,
        wdinfo_resolution="gt480",  # 480, 720, resize to 480
        max_frames=17,
        translation_scale=10,
        mode="joint",
        rotation_format="rot6d",
        pose_convention="backward_framewise",
        max_action_translation_norm=10,
    ),
    ratio=1,
    resolution="480",
)

camera_8b_16f_joint_fw_s10_011 = _make_exp(
    "camera_8b_16f_joint_fw_s10_011",
    [DATASET_CAMERA_480_16F_THRESH_S10],
)
cs.store(
    group="experiment",
    package="_global_",
    name="camera_8b_16f_joint_fw_s10_011",
    node=camera_8b_16f_joint_fw_s10_011,
)

DATASET_CAMERA_480_16F_THRESH_S5 = L(dataset_entry)(
    name="camera_480",
    dataset=L(CameraDatasetSharded)(
        wdinfo_paths=[
            CAMERA_WDINFOS["pretrained_clips_260307_100k_filtered"],
            CAMERA_WDINFOS["pretrained_clips_260325_500k_01_filtered"],
            CAMERA_WDINFOS["pretrained_clips_260325_500k_02_filtered"],
            CAMERA_WDINFOS["pretrained_clips_260313_10s_100k_filtered"],
        ],
        split="train",
        shuffle=True,
        fix_caption=False,
        wdinfo_resolution="gt480",  # 480, 720, resize to 480
        max_frames=17,
        translation_scale=5,
        mode="joint",
        rotation_format="rot6d",
        pose_convention="backward_framewise",
        max_action_translation_norm=10,
    ),
    ratio=1,
    resolution="480",
)

camera_8b_16f_joint_fw_s5_011 = _make_exp(
    "camera_8b_16f_joint_fw_s5_011",
    [DATASET_CAMERA_480_16F_THRESH_S5],
)
cs.store(
    group="experiment",
    package="_global_",
    name="camera_8b_16f_joint_fw_s5_011",
    node=camera_8b_16f_joint_fw_s5_011,
)


DATASET_CAMERA_480_16F_THRESH_S1 = L(dataset_entry)(
    name="camera_480",
    dataset=L(CameraDatasetSharded)(
        wdinfo_paths=[
            CAMERA_WDINFOS["pretrained_clips_260307_100k_filtered"],
            CAMERA_WDINFOS["pretrained_clips_260325_500k_01_filtered"],
            CAMERA_WDINFOS["pretrained_clips_260325_500k_02_filtered"],
            CAMERA_WDINFOS["pretrained_clips_260313_10s_100k_filtered"],
        ],
        split="train",
        shuffle=True,
        fix_caption=False,
        wdinfo_resolution="gt480",  # 480, 720, resize to 480
        max_frames=17,
        translation_scale=1,
        mode="joint",
        rotation_format="rot6d",
        pose_convention="backward_framewise",
        max_action_translation_norm=10,
    ),
    ratio=1,
    resolution="480",
)

camera_8b_16f_joint_fw_s1_011 = _make_exp(
    "camera_8b_16f_joint_fw_s1_011",
    [DATASET_CAMERA_480_16F_THRESH_S1],
)
cs.store(
    group="experiment",
    package="_global_",
    name="camera_8b_16f_joint_fw_s1_011",
    node=camera_8b_16f_joint_fw_s1_011,
)

### camera fd-only experiments

DATASET_CAMERA_FD_480 = L(dataset_entry)(
    name="camera_480",
    dataset=L(CameraDatasetSharded)(
        wdinfo_paths=[
            CAMERA_WDINFOS["pretrained_clips_260307_100k_filtered"],
            CAMERA_WDINFOS["pretrained_clips_260325_500k_01_filtered"],
            CAMERA_WDINFOS["pretrained_clips_260325_500k_02_filtered"],
            CAMERA_WDINFOS["pretrained_clips_260313_10s_100k_filtered"],
        ],
        split="train",
        shuffle=True,
        fix_caption=False,
        wdinfo_resolution="gt480",  # 480, 720, resize to 480
        max_frames=149,
        translation_scale=10,
        max_action_translation_norm=10,
        mode="forward_dynamics",
        rotation_format="rot6d",
        pose_convention="backward_framewise",
    ),
    ratio=1,
    resolution="480",
)

camera_8b_fd_fw_step1_001 = _make_exp(
    "camera_8b_fd_fw_step1_001",
    [DATASET_CAMERA_FD_480],
)
cs.store(
    group="experiment",
    package="_global_",
    name="camera_8b_fd_fw_step1_001",
    node=camera_8b_fd_fw_step1_001,
)

DATASET_CAMERA_FD_LONG_256 = L(dataset_entry)(
    name="camera_256",
    dataset=L(CameraDatasetSharded)(
        wdinfo_paths=[
            CAMERA_WDINFOS["pretrained_clips_260307_100k_filtered"],
            CAMERA_WDINFOS["pretrained_clips_260325_500k_01_filtered"],
            CAMERA_WDINFOS["pretrained_clips_260325_500k_02_filtered"],
            CAMERA_WDINFOS["pretrained_clips_260313_10s_100k_filtered"],
        ],
        split="train",
        shuffle=True,
        fix_caption=False,
        wdinfo_resolution="gt256",
        max_frames=400,
        translation_scale=10,
        max_action_translation_norm=10,
        mode="forward_dynamics",
        rotation_format="rot6d",
        pose_convention="backward_framewise",
    ),
    ratio=1,
    resolution="256",
)

DATASET_CAMERA_FD_LONG_480 = L(dataset_entry)(
    name="camera_480",
    dataset=L(CameraDatasetSharded)(
        wdinfo_paths=[
            CAMERA_WDINFOS["pretrained_clips_260307_100k_filtered"],
            CAMERA_WDINFOS["pretrained_clips_260325_500k_01_filtered"],
            CAMERA_WDINFOS["pretrained_clips_260325_500k_02_filtered"],
            CAMERA_WDINFOS["pretrained_clips_260313_10s_100k_filtered"],
        ],
        split="train",
        shuffle=True,
        fix_caption=False,
        wdinfo_resolution="gt480",  # 480, 720, resize to 480
        max_frames=300,
        translation_scale=10,
        max_action_translation_norm=10,
        mode="forward_dynamics",
        rotation_format="rot6d",
        pose_convention="backward_framewise",
    ),
    ratio=1,
    resolution="480",
)

DATASET_CAMERA_FD_LONG_720 = L(dataset_entry)(
    name="camera_720",
    dataset=L(CameraDatasetSharded)(
        wdinfo_paths=[
            CAMERA_WDINFOS["pretrained_clips_260307_100k_filtered"],
            CAMERA_WDINFOS["pretrained_clips_260325_500k_01_filtered"],
            CAMERA_WDINFOS["pretrained_clips_260325_500k_02_filtered"],
            CAMERA_WDINFOS["pretrained_clips_260313_10s_100k_filtered"],
        ],
        split="train",
        shuffle=True,
        fix_caption=False,
        wdinfo_resolution="gt720",  # 480, 720, resize to 480
        max_frames=200,
        translation_scale=10,
        max_action_translation_norm=10,
        mode="forward_dynamics",
        rotation_format="rot6d",
        pose_convention="backward_framewise",
    ),
    ratio=1,
    resolution="720",
)

camera_8b_fd_long_001 = _make_exp(
    "camera_8b_fd_long_001",
    [
        DATASET_CAMERA_FD_LONG_256,
        DATASET_CAMERA_FD_LONG_480,
        DATASET_CAMERA_FD_LONG_720,
    ],
)
camera_8b_fd_long_001["checkpoint"]["load_path"] = (
    "cosmos3_action/uva_camera/camera_8b_fd_fw_step1_001/checkpoints/iter_000075000/"
)
camera_8b_fd_long_001["checkpoint"]["load_training_state"] = False
camera_8b_fd_long_001["checkpoint"]["keys_to_skip_loading"] = []
cs.store(
    group="experiment",
    package="_global_",
    name="camera_8b_fd_long_001",
    node=camera_8b_fd_long_001,
)

### AV ID-only experiments
DATASET_AV_ID_480 = L(dataset_entry)(
    name="av_id_480",
    dataset=L(AVDataset)(
        root=[
            "s3://nv-00-10206-robot/cosmos3_action_data/av_v2_02182026_wdinfo/",
            "s3://nv-00-10206-robot/cosmos3_action_data/av_v2_03292026_wdinfo/",
        ],
        split="train",
        fps=10,
        mode="inverse_dynamics",
        history_len=0.1,
        future_len=6.0,
        rotation_format="rot6d",
        pose_convention="backward_framewise",
        translation_scale=1.35,
        max_action_translation_norm=10,
        resolution="480",
        credential_path="${job.cluster.object_store_credential_data}",
        shuffle=True,
        include_route_in_prompt=False,
        use_semantic_route_prompt=False,
        align_opencv_pose=False,
    ),
    ratio=1,
    resolution="480",
)

av_8b_id_fw_step1_003 = _make_exp(
    "av_8b_id_fw_step1_003",
    [DATASET_AV_ID_480],
)
cs.store(group="experiment", package="_global_", name="av_8b_id_fw_step1_003", node=av_8b_id_fw_step1_003)


DATASET_AV_JOINT_480 = L(dataset_entry)(
    name="av_joint_480",
    dataset=L(AVDataset)(
        root=[
            "s3://nv-00-10206-robot/cosmos3_action_data/av_v2_02182026_wdinfo/",
            "s3://nv-00-10206-robot/cosmos3_action_data/av_v2_03292026_wdinfo/",
        ],
        split="train",
        fps=10,
        mode="joint",
        history_len=0.1,
        future_len=6.0,
        rotation_format="rot6d",
        pose_convention="backward_framewise",
        translation_scale=1.35,
        max_action_translation_norm=10,
        resolution="480",
        credential_path="${job.cluster.object_store_credential_data}",
        shuffle=True,
        include_route_in_prompt=False,
        use_semantic_route_prompt=False,
        align_opencv_pose=False,
    ),
    ratio=1,
    resolution="480",
)

av_8b_joint_fw_step1_003 = _make_exp(
    "av_8b_joint_fw_step1_003",
    [DATASET_AV_JOINT_480],
)
cs.store(group="experiment", package="_global_", name="av_8b_joint_fw_step1_003", node=av_8b_joint_fw_step1_003)

# same exp with 1 node
av_8b_joint_fw_step1_004 = _make_exp(
    "av_8b_joint_fw_step1_004",
    [DATASET_AV_JOINT_480],
)
cs.store(group="experiment", package="_global_", name="av_8b_joint_fw_step1_004", node=av_8b_joint_fw_step1_004)

### Rotation format and scale experiments

DATASET_AV_JOINT_480_AA = L(dataset_entry)(
    name="av_joint_480",
    dataset=L(AVDataset)(
        root=[
            "s3://nv-00-10206-robot/cosmos3_action_data/av_v2_02182026_wdinfo/",
            "s3://nv-00-10206-robot/cosmos3_action_data/av_v2_03292026_wdinfo/",
        ],
        split="train",
        fps=10,
        mode="joint",
        history_len=0.1,
        future_len=6.0,
        rotation_format="axisangle",
        pose_convention="backward_framewise",
        translation_scale=1.35,
        rotation_scale=1.35 * 50,
        max_action_translation_norm=10,
        resolution="480",
        credential_path="${job.cluster.object_store_credential_data}",
        shuffle=True,
        include_route_in_prompt=False,
        use_semantic_route_prompt=False,
        align_opencv_pose=False,
    ),
    ratio=1,
    resolution="480",
)

av_8b_joint_fwaa_001 = _make_exp(
    "av_8b_joint_fwaa_001",
    [DATASET_AV_JOINT_480_AA],
)
cs.store(group="experiment", package="_global_", name="av_8b_joint_fwaa_001", node=av_8b_joint_fwaa_001)

DATASET_AV_ID_480_AA = L(dataset_entry)(
    name="av_id_480",
    dataset=L(AVDataset)(
        root=[
            "s3://nv-00-10206-robot/cosmos3_action_data/av_v2_02182026_wdinfo/",
            "s3://nv-00-10206-robot/cosmos3_action_data/av_v2_03292026_wdinfo/",
        ],
        split="train",
        fps=10,
        mode="inverse_dynamics",
        history_len=0.1,
        future_len=6.0,
        rotation_format="axisangle",
        pose_convention="backward_framewise",
        translation_scale=1.35,
        rotation_scale=1.35 * 50,
        max_action_translation_norm=10,
        resolution="480",
        credential_path="${job.cluster.object_store_credential_data}",
        shuffle=True,
        include_route_in_prompt=False,
        use_semantic_route_prompt=False,
        align_opencv_pose=False,
    ),
    ratio=1,
    resolution="480",
)

av_8b_id_fwaa_001 = _make_exp(
    "av_8b_id_fwaa_001",
    [DATASET_AV_ID_480_AA],
)
cs.store(group="experiment", package="_global_", name="av_8b_id_fwaa_001", node=av_8b_id_fwaa_001)

DATASET_CAMERA_JOINT_480_AA = L(dataset_entry)(
    name="camera_480",
    dataset=L(CameraDatasetSharded)(
        wdinfo_paths=[
            CAMERA_WDINFOS["pretrained_clips_260307_100k_filtered"],
            CAMERA_WDINFOS["pretrained_clips_260325_500k_01_filtered"],
            CAMERA_WDINFOS["pretrained_clips_260325_500k_02_filtered"],
            CAMERA_WDINFOS["pretrained_clips_260313_10s_100k_filtered"],
        ],
        split="train",
        shuffle=True,
        fix_caption=False,
        wdinfo_resolution="gt480",  # 480, 720, resize to 480
        max_frames=149,
        translation_scale=10,
        rotation_scale=10 * 8,
        max_action_translation_norm=10,
        mode="joint",
        rotation_format="axisangle",
        pose_convention="backward_framewise",
    ),
    ratio=1,
    resolution="480",
)

camera_8b_joint_fwaa_001 = _make_exp(
    "camera_8b_joint_fwaa_001",
    [DATASET_CAMERA_JOINT_480_AA],
)
cs.store(
    group="experiment",
    package="_global_",
    name="camera_8b_joint_fwaa_001",
    node=camera_8b_joint_fwaa_001,
)

DATASET_CAMERA_FD_480_AA = L(dataset_entry)(
    name="camera_fd_480",
    dataset=L(CameraDatasetSharded)(
        wdinfo_paths=[
            CAMERA_WDINFOS["pretrained_clips_260307_100k_filtered"],
            CAMERA_WDINFOS["pretrained_clips_260325_500k_01_filtered"],
            CAMERA_WDINFOS["pretrained_clips_260325_500k_02_filtered"],
            CAMERA_WDINFOS["pretrained_clips_260313_10s_100k_filtered"],
        ],
        split="train",
        shuffle=True,
        fix_caption=False,
        wdinfo_resolution="gt480",  # 480, 720, resize to 480
        max_frames=149,
        translation_scale=10,
        rotation_scale=10 * 8,
        max_action_translation_norm=10,
        mode="forward_dynamics",
        rotation_format="axisangle",
        pose_convention="backward_framewise",
    ),
    ratio=1,
    resolution="480",
)

camera_8b_fd_fwaa_001 = _make_exp(
    "camera_8b_fd_fwaa_001",
    [DATASET_CAMERA_FD_480_AA],
)
cs.store(
    group="experiment",
    package="_global_",
    name="camera_8b_fd_fwaa_001",
    node=camera_8b_fd_fwaa_001,
)

DATASET_CAMERA_FD_480_AA_WOROTSCL = L(dataset_entry)(
    name="camera_fd_480",
    dataset=L(CameraDatasetSharded)(
        wdinfo_paths=[
            CAMERA_WDINFOS["pretrained_clips_260307_100k_filtered"],
            CAMERA_WDINFOS["pretrained_clips_260325_500k_01_filtered"],
            CAMERA_WDINFOS["pretrained_clips_260325_500k_02_filtered"],
            CAMERA_WDINFOS["pretrained_clips_260313_10s_100k_filtered"],
        ],
        split="train",
        shuffle=True,
        fix_caption=False,
        wdinfo_resolution="gt480",  # 480, 720, resize to 480
        max_frames=149,
        translation_scale=10,
        rotation_scale=10,
        max_action_translation_norm=10,
        mode="forward_dynamics",
        rotation_format="axisangle",
        pose_convention="backward_framewise",
    ),
    ratio=1,
    resolution="480",
)

camera_8b_fd_fwaa_worotscl_001 = _make_exp(
    "camera_8b_fd_fwaa_worotscl_001",
    [DATASET_CAMERA_FD_480_AA_WOROTSCL],
)
cs.store(
    group="experiment",
    package="_global_",
    name="camera_8b_fd_fwaa_worotscl_001",
    node=camera_8b_fd_fwaa_worotscl_001,
)

### SDG data ablation
# DATASET_CAMERA_PRETRAIN_480 = L(dataset_entry)(
#     name="camera_480",
#     dataset=L(CameraDatasetSharded)(
#         wdinfo_paths=[
#             CAMERA_WDINFOS["pretrained_clips_260307_100k_filtered"],
#             CAMERA_WDINFOS["pretrained_clips_260325_500k_01_filtered"],
#             CAMERA_WDINFOS["pretrained_clips_260325_500k_02_filtered"],
#             CAMERA_WDINFOS["pretrained_clips_260313_10s_100k_filtered"],
#         ],
#         split="train",
#         shuffle=True,
#         fix_caption=False,
#         wdinfo_resolution="gt480",  # 480, 720, resize to 480
#         max_frames=61,
#         translation_scale=50,
#         max_action_translation_norm=10,
#         mode="joint",
#         rotation_format="6D",
#         rel_pose_format="backward_framewise",
#     ),
#     ratio=1,
#     resolution="480",
# )

# DATASET_CAMERA_ENDEAVOR_480 = L(dataset_entry)(
#     name="camera_endeavor_480",
#     dataset=L(CameraDatasetSharded)(
#         wdinfo_paths=[
#             CAMERA_WDINFOS["endeavor_forever_480"],
#         ],
#         split="train",
#         shuffle=True,
#         fix_caption=False,
#         wdinfo_resolution="gt480",  # 480, 720, resize to 480
#         max_frames=61,
#         translation_scale=50,
#         max_action_translation_norm=10,
#         mode="joint",
#         rotation_format="6D",
#         rel_pose_format="backward_framewise",
#     ),
#     ratio=1,
#     resolution="480",
# )

# DATASET_CAMERA_SYNHUMAN_480 = L(dataset_entry)(
#     name="camera_synhuman_480",
#     dataset=L(CameraDatasetSharded)(
#         wdinfo_paths=[
#             CAMERA_WDINFOS["synhuman_20260223_480"],
#         ],
#         split="train",
#         shuffle=True,
#         fix_caption=False,
#         wdinfo_resolution="gt480",  # 480, 720, resize to 480
#         max_frames=61,
#         translation_scale=50,
#         max_action_translation_norm=10,
#         mode="joint",
#         rotation_format="6D",
#         rel_pose_format="backward_framewise",
#     ),
#     ratio=1,
#     resolution="480",
# )
