# -----------------------------------------------------------------------------
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# This codebase constitutes NVIDIA proprietary technology and is strictly
# confidential. Any unauthorized reproduction, distribution, or disclosure
# of this code, in whole or in part, outside NVIDIA is strictly prohibited
# without prior written consent.
#
# For inquiries regarding the use of this code in other NVIDIA proprietary
# projects, please contact the Deep Imagination Research Team at
# dir@exchange.nvidia.com.
# ------

"""Hand-pose Action experiments built on top of the centralized pretrained configs.

Base configs are constructed via ``make_2b_experiment`` / ``make_8b_experiment``
from the pretrained-config module and then patched with hand-pose-specific
overrides (dataloaders, action dims, tokenizer durations, …).

Example (single node interactive run)::

    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. torchrun --nproc_per_node=1 \
        --master_port=12347 scripts/train.py \
        --config=configs/base/config.py \
        -- experiment=embodiment_a_500hrNew_camWristRel_fd job.wandb_mode=disabled
"""

from typing import Any

from hydra.core.config_store import ConfigStore

from cosmos.utils.lazy_config import LazyCall as L
from configs.base.experiment.action.pretrained_config.cosmos3_2b import make_2b_experiment
from configs.base.experiment.action.pretrained_config.cosmos3_8b import make_8b_experiment
from cosmos.data.vfm.action.dataloaders import InfiniteDataLoader
from cosmos.data.vfm.action.hand_pose_dataset import HandPoseDataset
from cosmos.data.vfm.action.hand_pose_dataset_config import HAND_POSE_DATASETS, EMBODIMENT_A_ALL
from cosmos.data.vfm.action.unified_dataset import dataset_entry, wrap_dataset
from cosmos.data.vfm.joint_dataloader import IterativeJointDataLoader

# --------------------------------------------------------------------------
# Data roots & experiment factory
# --------------------------------------------------------------------------

_EMBODIMENT_A_500HR_UPDATED_ROOT_FIXED = HAND_POSE_DATASETS["embodiment_a_feb08_500hr"]
_VITRA_EGO4D_ROOT = HAND_POSE_DATASETS["vitra_ego4d"]


def _make_hand_pose_experiment(
    name: str,
    base: str = "hand_pose_exp_8b_base_config",
    max_action_dim: int = 256,
    save_iter: int | None = 500,
    mode: str = "forward_dynamics",
    pose_convention: str = "backward_framewise",
    chunk_length: int = 72,
    root: str | list[str] = _EMBODIMENT_A_500HR_UPDATED_ROOT_FIXED,
    keypoint_option: str = "wrist_plus_fingers",
    rotation_format: str = "rot9d",
    intra_episode_val_ratio: float = 0.0,
    max_episodes: int | None = None,
    snap_to_subtask: bool = True,
    skip_no_action: bool = True,
    max_subtasks_per_episode: int | None = 5,
) -> dict:
    """Build a complete hand-pose posttrain experiment config.

    Creates a Hydra experiment node with matched train/val dataloaders that
    differ only in ``split``.  Defaults match ``HandPoseDataset`` class defaults.
    """
    ds_kwargs: dict[str, Any] = dict(
        mode=mode,
        pose_convention=pose_convention,
        chunk_length=chunk_length,
        root=root,
        keypoint_option=keypoint_option,
        rotation_format=rotation_format,
        intra_episode_val_ratio=intra_episode_val_ratio,
        snap_to_subtask=snap_to_subtask,
        skip_no_action=skip_no_action,
    )
    if max_episodes is not None:
        ds_kwargs["max_episodes"] = max_episodes
    if max_subtasks_per_episode is not None:
        ds_kwargs["max_subtasks_per_episode"] = max_subtasks_per_episode

    def dataloader_override(split: str) -> dict:
        ds_list = [
            L(dataset_entry)(
                name="hand_pose",
                dataset=L(HandPoseDataset)(split=split, **ds_kwargs),
                ratio=1.0,
                resolution="480",
            ),
        ]
        return {"dataloaders": {"uva_data": {"dataloader": {"dataset": {"list_of_datasets": ds_list}}}}}

    node: dict[str, Any] = dict(
        defaults=[f"/experiment/{base}", "_self_"],
        model=dict(config=dict(max_action_dim=max_action_dim)),
        dataloader_train=dataloader_override("train"),
        dataloader_val=dataloader_override("val"),
        job=dict(name=name),
    )
    if save_iter is not None:
        node["checkpoint"] = dict(save_iter=save_iter)
    return node


def _hand_pose_base_dataloader(split: str) -> Any:
    """Default hand-pose dataloader for base configs (train or val)."""
    is_train = split == "train"
    loader_kwargs: dict[str, Any] = dict(
        batch_size=4 if is_train else 1,
        shuffle=is_train,
        num_workers=4 if is_train else 0,
        pin_memory=True,
        drop_last=True,
    )
    if is_train:
        loader_kwargs["seed"] = 42

    return L(IterativeJointDataLoader)(
        dataloaders={
            "uva_data": dict(
                dataloader=L(InfiniteDataLoader)(
                    dataset=L(wrap_dataset)(
                        list_of_datasets=[
                            L(dataset_entry)(
                                name="hand_pose",
                                dataset=L(HandPoseDataset)(chunk_length=72, split=split),
                                ratio=1.0,
                                resolution="480",
                            ),
                        ],
                        tokenizer_config="${model.config.vlm_config.tokenizer}",
                        cfg_dropout_rate=0.1 if is_train else 0.0,
                        max_action_dim="${model.config.max_action_dim}",
                        shard_across_workers=True,
                    ),
                    **loader_kwargs,
                ),
                ratio=1,
            ),
        },
        tokenizer_spatial_compression_factor="${model.config.tokenizer.spatial_compression_factor}",
        tokenizer_temporal_compression_factor="${model.config.tokenizer.temporal_compression_factor}",
        patch_spatial="${model.config.diffusion_expert_config.patch_spatial}",
        max_sequence_length="${model.config.max_num_tokens_after_packing}",
    )


# ==========================================================================
# Base configs (built from centralized pretrained-config factories)
# ==========================================================================

# ---- 2B: 480p, unified_3d_mrope, 300 frames ----
hand_pose_exp_2b_base_config = make_2b_experiment(
    "hand_pose_exp_2b_base_config",
    datasets=[],
    max_action_dim=64,
    training_iterations=50_000,
)
hand_pose_exp_2b_base_config["job"]["group"] = "hand_pose"
hand_pose_exp_2b_base_config["model"]["config"]["max_action_dim"] = 64
hand_pose_exp_2b_base_config["model"]["config"]["num_embodiment_domains"] = 32
hand_pose_exp_2b_base_config["model"]["config"]["tokenizer"]["encode_exact_durations"] = [17, 61, 73, 93]
hand_pose_exp_2b_base_config["dataloader_train"] = _hand_pose_base_dataloader("train")
hand_pose_exp_2b_base_config["dataloader_val"] = _hand_pose_base_dataloader("val")

# ---- 8B: 480p single-res, unified_3d_mrope with modality offset ----
hand_pose_exp_8b_base_config = make_8b_experiment(
    "hand_pose_exp_8b_base_config",
    datasets=[],
    max_action_dim=64,
    training_iterations=50_000,
)
hand_pose_exp_8b_base_config["job"]["group"] = "hand_pose"
hand_pose_exp_8b_base_config["model"]["config"]["resolution"] = "480"
hand_pose_exp_8b_base_config["trainer"]["callbacks"]["compile_tokenizer"]["enabled"] = True
hand_pose_exp_8b_base_config["trainer"]["callbacks"]["compile_tokenizer"]["warmup_resolutions"] = ["480"]
hand_pose_exp_8b_base_config["dataloader_train"] = _hand_pose_base_dataloader("train")
hand_pose_exp_8b_base_config["dataloader_val"] = _hand_pose_base_dataloader("val")

# ==========================================================================
# Reusable dataset entries for multi-domain joint training.
# ==========================================================================

_HAND_POSE_DATASET_DEFAULTS: dict[str, object] = dict(
    root=_EMBODIMENT_A_500HR_UPDATED_ROOT_FIXED,
    chunk_length=72,
    pose_convention="backward_framewise",
    keypoint_option="wrist_plus_finger_tips",
    rotation_format="rot6d",
    snap_to_subtask=True,
    skip_no_action=True,
    max_subtasks_per_episode=5,
)

DATASET_HAND_POSE_480 = L(dataset_entry)(
    name="hand_pose",
    dataset=L(HandPoseDataset)(split="train", mode="forward_dynamics", **_HAND_POSE_DATASET_DEFAULTS),
    ratio=1.0,
    resolution="480",
)

DATASET_HAND_POSE_480_I2V = L(dataset_entry)(
    name="hand_pose",
    dataset=L(HandPoseDataset)(split="train", mode="image2video", **_HAND_POSE_DATASET_DEFAULTS),
    ratio=1.0,
    resolution="480",
)

DATASET_HAND_POSE_480_JOINT = L(dataset_entry)(
    name="hand_pose",
    dataset=L(HandPoseDataset)(split="train", mode="joint", **_HAND_POSE_DATASET_DEFAULTS),
    ratio=1.0,
    resolution="480",
)

# ==========================================================================
# Posttrain ablation experiments
#
# Experiment names keep legacy "cam*" prefixes for W&B / checkpoint continuity;
# actual pose_convention values use the unified names.
#
# Axes varied across experiments:
#   Pose convention  – backward_anchored, backward_framewise
#   Keypoints        – wrist_plus_fingers (all 21, default) · wrist_plus_finger_tips
#   Rotation format  – rot9d (default) · rot6d
#   Model size       – 2B · 8B
# ==========================================================================

# -- Group 1: backward_anchored ± egocam (8B) -----------------------------
embodiment_a_500hrNew_camAnchored_fd_8b_32node = _make_hand_pose_experiment(
    "embodiment_a_500hrNew_camAnchored_fd_8b_32node",
    pose_convention="backward_anchored",
)
embodiment_a_500hrNew_camAnchoredWithCamPose_fd_8b_32node = _make_hand_pose_experiment(
    "embodiment_a_500hrNew_camAnchoredWithCamPose_fd_8b_32node",
    pose_convention="backward_anchored",
)
embodiment_a_500hrNew_camAnchoredWithCamPose_rot6D_fd_8b_32node = _make_hand_pose_experiment(
    "embodiment_a_500hrNew_camAnchoredWithCamPose_rot6D_fd_8b_32node",
    pose_convention="backward_anchored",
    rotation_format="rot6d",
)
embodiment_a_500hrNew_camAnchoredWithCamPose_wristAndFingerTips_rot6D_fd_8b_32node = _make_hand_pose_experiment(
    "embodiment_a_500hrNew_camAnchoredWithCamPose_wristAndFingerTips_rot6D_fd_8b_32node",
    max_action_dim=64,
    pose_convention="backward_anchored",
    keypoint_option="wrist_plus_finger_tips",
    rotation_format="rot6d",
)

# -- Group 2: backward_framewise (8B) -----------------------------
embodiment_a_500hrNew_camWristRelWithCamPose_fd_8b_32node = _make_hand_pose_experiment(
    "embodiment_a_500hrNew_camWristRelWithCamPose_fd_8b_32node",
    pose_convention="backward_framewise",
)
embodiment_a_500hrNew_camWristRelWithCamPose_wristAndFingerTips_fd_8b_32node = _make_hand_pose_experiment(
    "embodiment_a_500hrNew_camWristRelWithCamPose_wristAndFingerTips_fd_8b_32node",
    pose_convention="backward_framewise",
    keypoint_option="wrist_plus_finger_tips",
)
embodiment_a_500hrNew_camWristRelWithCamPose_wristAndFingerTips_rot6D_fd_8b_32node = _make_hand_pose_experiment(
    "embodiment_a_500hrNew_camWristRelWithCamPose_wristAndFingerTips_rot6D_fd_8b_32node",
    max_action_dim=64,
    pose_convention="backward_framewise",
    keypoint_option="wrist_plus_finger_tips",
    rotation_format="rot6d",
)
embodiment_a_500hrNew_camWristRelWithCamPose_wristAndFingerTips_deltaTminus1_fd_8b_32node = _make_hand_pose_experiment(
    "embodiment_a_500hrNew_camWristRelWithCamPose_wristAndFingerTips_deltaTminus1_fd_8b_32node",
    pose_convention="backward_framewise",
    keypoint_option="wrist_plus_finger_tips",
)

# -- Group 3: backward_framewise (midtraining default, 57D, 8B) ------------
embodiment_a_500hrNew_backwardFramewise_fingerTips_rot6D_fd_8b_maxep10 = _make_hand_pose_experiment(
    "embodiment_a_500hrNew_backwardFramewise_fingerTips_rot6D_fd_8b_maxep10",
    max_action_dim=64,
    pose_convention="backward_framewise",
    keypoint_option="wrist_plus_finger_tips",
    rotation_format="rot6d",
    max_episodes=10,
)
embodiment_a_500hrNew_backwardFramewise_fingerTips_rot6D_id_8b_maxep10 = _make_hand_pose_experiment(
    "embodiment_a_500hrNew_backwardFramewise_fingerTips_rot6D_id_8b_maxep10",
    max_action_dim=64,
    mode="inverse_dynamics",
    pose_convention="backward_framewise",
    keypoint_option="wrist_plus_finger_tips",
    rotation_format="rot6d",
    max_episodes=10,
)
embodiment_a_all_backwardFramewise_fingerTips_rot6D_fd_8b_32node = _make_hand_pose_experiment(
    "embodiment_a_all_backwardFramewise_fingerTips_rot6D_fd_8b_32node",
    max_action_dim=64,
    pose_convention="backward_framewise",
    keypoint_option="wrist_plus_finger_tips",
    rotation_format="rot6d",
    root=EMBODIMENT_A_ALL,
)

# -- Group 4: vitra_ego4d backward_framewise (57D, 8B, overfit) -------------
vitra_ego4d_backwardFramewise_fingerTips_rot6D_fd_8b_maxep10 = _make_hand_pose_experiment(
    "vitra_ego4d_backwardFramewise_fingerTips_rot6D_fd_8b_maxep10",
    max_action_dim=64,
    pose_convention="backward_framewise",
    keypoint_option="wrist_plus_finger_tips",
    rotation_format="rot6d",
    chunk_length=16,
    root=_VITRA_EGO4D_ROOT,
    max_episodes=10,
)
vitra_ego4d_backwardFramewise_fingerTips_rot6D_id_8b_maxep10 = _make_hand_pose_experiment(
    "vitra_ego4d_backwardFramewise_fingerTips_rot6D_id_8b_maxep10",
    max_action_dim=64,
    mode="inverse_dynamics",
    pose_convention="backward_framewise",
    keypoint_option="wrist_plus_finger_tips",
    rotation_format="rot6d",
    chunk_length=16,
    root=_VITRA_EGO4D_ROOT,
    max_episodes=10,
)

# -- test all data -----------------------------
embodiment_a_all_camAnchored_fd_8b_32node = _make_hand_pose_experiment(
    "embodiment_a_500hrNew_camAnchored_fd_8b_32node",
    pose_convention="backward_anchored",
    root=EMBODIMENT_A_ALL,
)

# ==========================================================================
# Register all experiments
# ==========================================================================

cs = ConfigStore.instance()

_ALL_EXPERIMENTS: list[tuple[str, dict]] = [
    ("hand_pose_exp_2b_base_config", hand_pose_exp_2b_base_config),
    ("hand_pose_exp_8b_base_config", hand_pose_exp_8b_base_config),
    ("embodiment_a_500hrNew_camAnchored_fd_8b_32node", embodiment_a_500hrNew_camAnchored_fd_8b_32node),
    ("embodiment_a_500hrNew_camAnchoredWithCamPose_fd_8b_32node", embodiment_a_500hrNew_camAnchoredWithCamPose_fd_8b_32node),
    (
        "embodiment_a_500hrNew_camAnchoredWithCamPose_rot6D_fd_8b_32node",
        embodiment_a_500hrNew_camAnchoredWithCamPose_rot6D_fd_8b_32node,
    ),
    (
        "embodiment_a_500hrNew_camAnchoredWithCamPose_wristAndFingerTips_rot6D_fd_8b_32node",
        embodiment_a_500hrNew_camAnchoredWithCamPose_wristAndFingerTips_rot6D_fd_8b_32node,
    ),
    ("embodiment_a_500hrNew_camWristRelWithCamPose_fd_8b_32node", embodiment_a_500hrNew_camWristRelWithCamPose_fd_8b_32node),
    (
        "embodiment_a_500hrNew_camWristRelWithCamPose_wristAndFingerTips_fd_8b_32node",
        embodiment_a_500hrNew_camWristRelWithCamPose_wristAndFingerTips_fd_8b_32node,
    ),
    (
        "embodiment_a_500hrNew_camWristRelWithCamPose_wristAndFingerTips_rot6D_fd_8b_32node",
        embodiment_a_500hrNew_camWristRelWithCamPose_wristAndFingerTips_rot6D_fd_8b_32node,
    ),
    (
        "embodiment_a_500hrNew_camWristRelWithCamPose_wristAndFingerTips_deltaTminus1_fd_8b_32node",
        embodiment_a_500hrNew_camWristRelWithCamPose_wristAndFingerTips_deltaTminus1_fd_8b_32node,
    ),
    (
        "embodiment_a_500hrNew_backwardFramewise_fingerTips_rot6D_fd_8b_maxep10",
        embodiment_a_500hrNew_backwardFramewise_fingerTips_rot6D_fd_8b_maxep10,
    ),
    (
        "embodiment_a_500hrNew_backwardFramewise_fingerTips_rot6D_id_8b_maxep10",
        embodiment_a_500hrNew_backwardFramewise_fingerTips_rot6D_id_8b_maxep10,
    ),
    (
        "embodiment_a_all_backwardFramewise_fingerTips_rot6D_fd_8b_32node",
        embodiment_a_all_backwardFramewise_fingerTips_rot6D_fd_8b_32node,
    ),
    (
        "vitra_ego4d_backwardFramewise_fingerTips_rot6D_fd_8b_maxep10",
        vitra_ego4d_backwardFramewise_fingerTips_rot6D_fd_8b_maxep10,
    ),
    (
        "vitra_ego4d_backwardFramewise_fingerTips_rot6D_id_8b_maxep10",
        vitra_ego4d_backwardFramewise_fingerTips_rot6D_id_8b_maxep10,
    ),
    ("embodiment_a_all_camAnchored_fd_8b_32node", embodiment_a_all_camAnchored_fd_8b_32node),
]

for _name, _node in _ALL_EXPERIMENTS:
    cs.store(_name, _node, group="experiment", package="_global_")
