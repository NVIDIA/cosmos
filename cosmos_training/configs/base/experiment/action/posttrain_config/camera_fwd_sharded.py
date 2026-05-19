# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from hydra.core.config_store import ConfigStore

from cosmos.utils.lazy_config import LazyCall as L
from configs.base.experiment.action.pretrained_config.cosmos3_8b import make_8b_experiment
from cosmos.data.vfm.action.camera_dataset_sharded import CAMERA_WDINFOS, CameraDatasetSharded
from cosmos.data.vfm.action.dataloaders import InfiniteDataLoader
from cosmos.data.vfm.action.unified_dataset import dataset_entry, wrap_dataset
from cosmos.data.vfm.joint_dataloader import IterativeJointDataLoader

cs = ConfigStore.instance()

RECIPE_V5_CHECKPOINT = (
    "cosmos3_vfm/t2w_mot_8b_qwen3_vl_runs/t2w_mot_exp302_000_qwen3_vl_8b_multires_recipe_v5/checkpoints/iter_000005500/"
)

_COMPILE_TOKENIZER_CONFIG = dict(
    enabled=True,
    warmup_resolutions=["256", "480", "720"],
)


def _apply_camera_overrides(exp: dict) -> dict:
    """Apply overrides common to all camera forward-dynamics sharded experiments."""
    exp["job"]["group"] = "uva_camera"
    exp["model"]["config"]["max_action_dim"] = 32
    exp["model"]["config"]["num_embodiment_domains"] = 32
    exp["trainer"]["callbacks"]["compile_tokenizer"] = _COMPILE_TOKENIZER_CONFIG.copy()
    exp["trainer"]["compile_config"]["recompile_limit"] = 100
    return exp


def _make_multires_dataloader(datasets: list) -> L:
    """Build an IterativeJointDataLoader for multi-resolution camera datasets."""
    return L(IterativeJointDataLoader)(
        dataloaders={
            "camera_data": dict(
                dataloader=L(InfiniteDataLoader)(
                    dataset=L(wrap_dataset)(
                        list_of_datasets=datasets,
                        tokenizer_config="${model.config.vlm_config.tokenizer}",
                        cfg_dropout_rate=0.1,
                        max_action_dim="${model.config.max_action_dim}",
                    ),
                    batch_size=4,
                    num_workers=4,
                    pin_memory=True,
                    drop_last=True,
                    use_deterministic_seed=False,
                    in_order=False,
                    multiprocessing_context="spawn",
                ),
                ratio=1,
            ),
        },
        tokenizer_spatial_compression_factor="${model.config.tokenizer.spatial_compression_factor}",
        tokenizer_temporal_compression_factor="${model.config.tokenizer.temporal_compression_factor}",
        patch_spatial="${model.config.diffusion_expert_config.patch_spatial}",
        max_sequence_length="${model.config.max_num_tokens_after_packing}",
    )


# ---------------------------------------------------------------------------
# Experiment 1: 8b multires (modality_offset checkpoint, 480p, no waver)
# s3://nv-00-10206-checkpoint-experiments/cosmos3_vfm/t2w_mot_8b_qwen3_vl_runs/t2w_mot_exp302_003_qwen3_vl_8b_multires_modality_offset/checkpoints/iter_000006750/
# ---------------------------------------------------------------------------
camera_8b_480res_fd_sharded_v2 = _apply_camera_overrides(
    make_8b_experiment("camera_8b_480res_fd_sharded_v2", datasets=[], max_action_dim=32, training_iterations=100_000)
)
camera_8b_480res_fd_sharded_v2["model"]["config"]["resolution"] = "480"
del camera_8b_480res_fd_sharded_v2["model"]["config"]["rectified_flow_training_config"]["train_time_video_distribution"]
camera_8b_480res_fd_sharded_v2["checkpoint"]["save_iter"] = 500
camera_8b_480res_fd_sharded_v2["checkpoint"]["load_path"] = (
    "cosmos3_vfm/t2w_mot_8b_qwen3_vl_runs/"
    "t2w_mot_exp302_003_qwen3_vl_8b_multires_modality_offset/"
    "checkpoints/iter_000006750/"
)
camera_8b_480res_fd_sharded_v2["dataloader_train"] = L(IterativeJointDataLoader)(
    dataloaders={
        "camera_data": dict(
            dataloader=L(InfiniteDataLoader)(
                dataset=L(wrap_dataset)(
                    list_of_datasets=L(CameraDatasetSharded)(
                        wdinfo_paths=[
                            CAMERA_WDINFOS["tartanair_480"],
                            CAMERA_WDINFOS["endeavor_forever_480"],
                            CAMERA_WDINFOS["synhuman_20251218_480"],
                            CAMERA_WDINFOS["pretrained_clips_260131_10k_480"],
                        ],
                        split="train",
                        shuffle=True,
                        fix_caption=True,
                        max_frames=149,
                        mode="forward_dynamics",
                        rotation_format="rot6d",
                        pose_convention="backward_anchored",
                    ),
                    tokenizer_config="${model.config.vlm_config.tokenizer}",
                    cfg_dropout_rate=0.1,
                    max_action_dim="${model.config.max_action_dim}",
                ),
                batch_size=4,
                shuffle=False,
                num_workers=4,
                pin_memory=True,
                drop_last=True,
                in_order=False,
                seed=0,
            ),
            ratio=1,
        ),
    },
    tokenizer_spatial_compression_factor="${model.config.tokenizer.spatial_compression_factor}",
    tokenizer_temporal_compression_factor="${model.config.tokenizer.temporal_compression_factor}",
    patch_spatial="${model.config.diffusion_expert_config.patch_spatial}",
    max_sequence_length="${model.config.max_num_tokens_after_packing}",
)
camera_8b_480res_fd_sharded_v2["dataloader_val"] = L(IterativeJointDataLoader)(
    dataloaders={
        "camera_data": dict(
            dataloader=L(InfiniteDataLoader)(
                dataset=L(wrap_dataset)(
                    list_of_datasets=L(CameraDatasetSharded)(
                        wdinfo_paths=[
                            CAMERA_WDINFOS["tartanair_480"],
                            CAMERA_WDINFOS["endeavor_forever_480"],
                            CAMERA_WDINFOS["synhuman_20251218_480"],
                            CAMERA_WDINFOS["pretrained_clips_260131_10k_480"],
                        ],
                        split="val",
                        shuffle=False,
                        fix_caption=True,
                        max_frames=149,
                        mode="forward_dynamics",
                        rotation_format="rot6d",
                        pose_convention="backward_anchored",
                    ),
                    tokenizer_config="${model.config.vlm_config.tokenizer}",
                    cfg_dropout_rate=0.1,
                    max_action_dim="${model.config.max_action_dim}",
                ),
                batch_size=1,
                shuffle=False,
                num_workers=0,
                pin_memory=True,
                drop_last=True,
                in_order=False,
                seed=0,
            ),
            ratio=1,
        ),
    },
    tokenizer_spatial_compression_factor="${model.config.tokenizer.spatial_compression_factor}",
    tokenizer_temporal_compression_factor="${model.config.tokenizer.temporal_compression_factor}",
    patch_spatial="${model.config.diffusion_expert_config.patch_spatial}",
    max_sequence_length="${model.config.max_num_tokens_after_packing}",
)

cs.store(
    group="experiment",
    package="_global_",
    name="camera_8b_480res_fd_sharded_v2",
    node=camera_8b_480res_fd_sharded_v2,
)


# ---------------------------------------------------------------------------
# Experiment 2: 480p, recipe_v5, pretrain10k with captions
# base model:
# s3://nv-00-10206-checkpoint-experiments/cosmos3_vfm/t2w_mot_8b_qwen3_vl_runs/t2w_mot_exp302_000_qwen3_vl_8b_multires_recipe_v5/checkpoints/iter_000005500/model/
# ---------------------------------------------------------------------------
camera_8b_480res_fd_sharded_v3_pretrain10k_wcap = _apply_camera_overrides(
    make_8b_experiment(
        "camera_8b_480res_fd_sharded_v3_pretrain10k_wcap", datasets=[], max_action_dim=32, training_iterations=100_000
    )
)
camera_8b_480res_fd_sharded_v3_pretrain10k_wcap["model"]["config"]["resolution"] = "480"
camera_8b_480res_fd_sharded_v3_pretrain10k_wcap["checkpoint"]["save_iter"] = 1000
camera_8b_480res_fd_sharded_v3_pretrain10k_wcap["checkpoint"]["load_path"] = RECIPE_V5_CHECKPOINT
camera_8b_480res_fd_sharded_v3_pretrain10k_wcap["dataloader_train"] = L(IterativeJointDataLoader)(
    dataloaders={
        "camera_data": dict(
            dataloader=L(InfiniteDataLoader)(
                dataset=L(wrap_dataset)(
                    list_of_datasets=L(CameraDatasetSharded)(
                        wdinfo_paths=[
                            # CAMERA_WDINFOS["tartanair_480"],
                            # CAMERA_WDINFOS["endeavor_forever_480"],
                            # CAMERA_WDINFOS["synhuman_20251218_480"],
                            CAMERA_WDINFOS["pretrained_clips_260131_10k_480"],
                        ],
                        split="train",
                        shuffle=True,
                        fix_caption=False,
                        max_frames=149,
                        mode="forward_dynamics",
                        rotation_format="rot6d",
                        pose_convention="backward_anchored",
                        discard_varying_intrinsics=False,
                    ),
                    tokenizer_config="${model.config.vlm_config.tokenizer}",
                    cfg_dropout_rate=0.1,
                    max_action_dim="${model.config.max_action_dim}",
                ),
                batch_size=4,
                shuffle=False,
                num_workers=4,
                pin_memory=True,
                drop_last=True,
                in_order=False,
                seed=0,
            ),
            ratio=1,
        ),
    },
    tokenizer_spatial_compression_factor="${model.config.tokenizer.spatial_compression_factor}",
    tokenizer_temporal_compression_factor="${model.config.tokenizer.temporal_compression_factor}",
    patch_spatial="${model.config.diffusion_expert_config.patch_spatial}",
    max_sequence_length="${model.config.max_num_tokens_after_packing}",
)
camera_8b_480res_fd_sharded_v3_pretrain10k_wcap["dataloader_val"] = L(IterativeJointDataLoader)(
    dataloaders={
        "camera_data": dict(
            dataloader=L(InfiniteDataLoader)(
                dataset=L(wrap_dataset)(
                    list_of_datasets=L(CameraDatasetSharded)(
                        wdinfo_paths=[
                            # CAMERA_WDINFOS["tartanair_480"],
                            # CAMERA_WDINFOS["endeavor_forever_480"],
                            # CAMERA_WDINFOS["synhuman_20251218_480"],
                            CAMERA_WDINFOS["pretrained_clips_260131_10k_480"],
                        ],
                        split="val",
                        shuffle=False,
                        fix_caption=False,
                        max_frames=149,
                        mode="forward_dynamics",
                        rotation_format="rot6d",
                        pose_convention="backward_anchored",
                        discard_varying_intrinsics=False,
                    ),
                    tokenizer_config="${model.config.vlm_config.tokenizer}",
                    cfg_dropout_rate=0.1,
                    max_action_dim="${model.config.max_action_dim}",
                ),
                batch_size=1,
                shuffle=False,
                num_workers=0,
                pin_memory=True,
                drop_last=True,
                in_order=False,
                seed=0,
            ),
            ratio=1,
        ),
    },
    tokenizer_spatial_compression_factor="${model.config.tokenizer.spatial_compression_factor}",
    tokenizer_temporal_compression_factor="${model.config.tokenizer.temporal_compression_factor}",
    patch_spatial="${model.config.diffusion_expert_config.patch_spatial}",
    max_sequence_length="${model.config.max_num_tokens_after_packing}",
)

cs.store(
    group="experiment",
    package="_global_",
    name="camera_8b_480res_fd_sharded_v3_pretrain10k_wcap",
    node=camera_8b_480res_fd_sharded_v3_pretrain10k_wcap,
)


# ---------------------------------------------------------------------------
# Dataset constants: pretrain 100k filtered
# ---------------------------------------------------------------------------
DATASET_CAMERA_P100K_FILTERED_256 = L(dataset_entry)(
    name="camera_p100k_filtered_256",
    dataset=L(CameraDatasetSharded)(
        wdinfo_paths=[
            CAMERA_WDINFOS["pretrained_clips_260307_100k_filtered"],
        ],
        split="train",
        shuffle=True,
        fix_caption=False,
        wdinfo_resolution="all",  # 256, 480, 720, resize to 256
        max_frames=400,
        mode="forward_dynamics",
        rotation_format="rot6d",
        pose_convention="backward_anchored",
        discard_varying_intrinsics=False,
    ),
    ratio=3,
    resolution="256",
)

DATASET_CAMERA_P100K_FILTERED_480 = L(dataset_entry)(
    name="camera_p100k_filtered_480",
    dataset=L(CameraDatasetSharded)(
        wdinfo_paths=[
            CAMERA_WDINFOS["pretrained_clips_260307_100k_filtered"],
        ],
        split="train",
        shuffle=True,
        fix_caption=False,
        wdinfo_resolution="gt480",  # 480, 720, resize to 480
        max_frames=300,
        mode="forward_dynamics",
        rotation_format="rot6d",
        pose_convention="backward_anchored",
        discard_varying_intrinsics=False,
    ),
    ratio=2,
    resolution="480",
)

DATASET_CAMERA_P100K_FILTERED_720 = L(dataset_entry)(
    name="camera_p100k_filtered_720",
    dataset=L(CameraDatasetSharded)(
        wdinfo_paths=[
            CAMERA_WDINFOS["pretrained_clips_260307_100k_filtered"],
        ],
        split="train",
        shuffle=True,
        fix_caption=False,
        wdinfo_resolution="gt720",  # only 720
        max_frames=200,
        mode="forward_dynamics",
        rotation_format="rot6d",
        pose_convention="backward_anchored",
        discard_varying_intrinsics=False,
    ),
    ratio=1,
    resolution="720",
)


# ---------------------------------------------------------------------------
# Experiment 3: pretrain 100k filtered (720p, recipe_v5)
# ---------------------------------------------------------------------------
camera_8b_fd_sharded_v3_p100k_filtered = _apply_camera_overrides(
    make_8b_experiment(
        "camera_8b_fd_sharded_v3_p100k_filtered", datasets=[], max_action_dim=32, training_iterations=100_000
    )
)
camera_8b_fd_sharded_v3_p100k_filtered["checkpoint"]["save_iter"] = 1000
camera_8b_fd_sharded_v3_p100k_filtered["checkpoint"]["load_path"] = RECIPE_V5_CHECKPOINT
camera_8b_fd_sharded_v3_p100k_filtered["dataloader_train"] = _make_multires_dataloader(
    [
        DATASET_CAMERA_P100K_FILTERED_256,
        DATASET_CAMERA_P100K_FILTERED_480,
        DATASET_CAMERA_P100K_FILTERED_720,
    ]
)

cs.store(
    group="experiment",
    package="_global_",
    name="camera_8b_fd_sharded_v3_p100k_filtered",
    node=camera_8b_fd_sharded_v3_p100k_filtered,
)


# ---------------------------------------------------------------------------
# Dataset constants: pretrain 100k filtered framewise
# ---------------------------------------------------------------------------
DATASET_CAMERA_P100K_FILTERED_FRAMEWISE_256 = L(dataset_entry)(
    name="camera_p100k_filtered_framewise_256",
    dataset=L(CameraDatasetSharded)(
        wdinfo_paths=[
            CAMERA_WDINFOS["pretrained_clips_260307_100k_filtered"],
        ],
        split="train",
        shuffle=True,
        fix_caption=False,
        wdinfo_resolution="all",  # 256, 480, 720, resize to 256
        max_frames=400,
        mode="forward_dynamics",
        rotation_format="rot6d",
        pose_convention="backward_framewise",
        discard_varying_intrinsics=False,
    ),
    ratio=3,
    resolution="256",
)

DATASET_CAMERA_P100K_FILTERED_FRAMEWISE_480 = L(dataset_entry)(
    name="camera_p100k_filtered_framewise_480",
    dataset=L(CameraDatasetSharded)(
        wdinfo_paths=[
            CAMERA_WDINFOS["pretrained_clips_260307_100k_filtered"],
        ],
        split="train",
        shuffle=True,
        fix_caption=False,
        wdinfo_resolution="gt480",  # 480, 720, resize to 480
        max_frames=300,
        mode="forward_dynamics",
        rotation_format="rot6d",
        pose_convention="backward_framewise",
        discard_varying_intrinsics=False,
    ),
    ratio=2,
    resolution="480",
)

DATASET_CAMERA_P100K_FILTERED_FRAMEWISE_720 = L(dataset_entry)(
    name="camera_p100k_filtered_framewise_720",
    dataset=L(CameraDatasetSharded)(
        wdinfo_paths=[
            CAMERA_WDINFOS["pretrained_clips_260307_100k_filtered"],
        ],
        split="train",
        shuffle=True,
        fix_caption=False,
        wdinfo_resolution="gt720",  # only 720
        max_frames=200,
        mode="forward_dynamics",
        rotation_format="rot6d",
        pose_convention="backward_framewise",
        discard_varying_intrinsics=False,
    ),
    ratio=1,
    resolution="720",
)


# ---------------------------------------------------------------------------
# Experiment 4: pretrain 100k filtered framewise (720p, recipe_v5)
# ---------------------------------------------------------------------------
camera_8b_fd_sharded_v3_p100k_filtered_fw = _apply_camera_overrides(
    make_8b_experiment(
        "camera_8b_fd_sharded_v3_p100k_filtered_fw", datasets=[], max_action_dim=32, training_iterations=100_000
    )
)
camera_8b_fd_sharded_v3_p100k_filtered_fw["checkpoint"]["save_iter"] = 1000
camera_8b_fd_sharded_v3_p100k_filtered_fw["checkpoint"]["load_path"] = RECIPE_V5_CHECKPOINT
camera_8b_fd_sharded_v3_p100k_filtered_fw["dataloader_train"] = _make_multires_dataloader(
    [
        DATASET_CAMERA_P100K_FILTERED_FRAMEWISE_256,
        DATASET_CAMERA_P100K_FILTERED_FRAMEWISE_480,
        DATASET_CAMERA_P100K_FILTERED_FRAMEWISE_720,
    ]
)

cs.store(
    group="experiment",
    package="_global_",
    name="camera_8b_fd_sharded_v3_p100k_filtered_fw",
    node=camera_8b_fd_sharded_v3_p100k_filtered_fw,
)


# ---------------------------------------------------------------------------
# Dataset constants: pretrain 100k
# ---------------------------------------------------------------------------
DATASET_CAMERA_P100K_256 = L(dataset_entry)(
    name="camera_p100k_256",
    dataset=L(CameraDatasetSharded)(
        wdinfo_paths=[
            CAMERA_WDINFOS["pretrained_clips_260307_100k"],
        ],
        split="train",
        shuffle=True,
        fix_caption=False,
        wdinfo_resolution="all",  # 256, 480, 720, resize to 256
        max_frames=400,
        mode="forward_dynamics",
        rotation_format="rot6d",
        pose_convention="backward_anchored",
        discard_varying_intrinsics=False,
    ),
    ratio=3,
    resolution="256",
)

DATASET_CAMERA_P100K_480 = L(dataset_entry)(
    name="camera_p100k_480",
    dataset=L(CameraDatasetSharded)(
        wdinfo_paths=[
            CAMERA_WDINFOS["pretrained_clips_260307_100k"],
        ],
        split="train",
        shuffle=True,
        fix_caption=False,
        wdinfo_resolution="gt480",  # 480, 720, resize to 480
        max_frames=300,
        mode="forward_dynamics",
        rotation_format="rot6d",
        pose_convention="backward_anchored",
        discard_varying_intrinsics=False,
    ),
    ratio=2,
    resolution="480",
)

DATASET_CAMERA_P100K_720 = L(dataset_entry)(
    name="camera_p100k_720",
    dataset=L(CameraDatasetSharded)(
        wdinfo_paths=[
            CAMERA_WDINFOS["pretrained_clips_260307_100k"],
        ],
        split="train",
        shuffle=True,
        fix_caption=False,
        wdinfo_resolution="gt720",  # only 720
        max_frames=200,
        mode="forward_dynamics",
        rotation_format="rot6d",
        pose_convention="backward_anchored",
        discard_varying_intrinsics=False,
    ),
    ratio=1,
    resolution="720",
)


# ---------------------------------------------------------------------------
# Experiment 6: pretrain 100k (720p, recipe_v5)
# ---------------------------------------------------------------------------
camera_8b_fd_sharded_v3_p100k = _apply_camera_overrides(
    make_8b_experiment("camera_8b_fd_sharded_v3_p100k", datasets=[], max_action_dim=32, training_iterations=100_000)
)
camera_8b_fd_sharded_v3_p100k["checkpoint"]["save_iter"] = 1000
camera_8b_fd_sharded_v3_p100k["checkpoint"]["load_path"] = RECIPE_V5_CHECKPOINT
camera_8b_fd_sharded_v3_p100k["dataloader_train"] = _make_multires_dataloader(
    [
        DATASET_CAMERA_P100K_256,
        DATASET_CAMERA_P100K_480,
        DATASET_CAMERA_P100K_720,
    ]
)

cs.store(
    group="experiment",
    package="_global_",
    name="camera_8b_fd_sharded_v3_p100k",
    node=camera_8b_fd_sharded_v3_p100k,
)
