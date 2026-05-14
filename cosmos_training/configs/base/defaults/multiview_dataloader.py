# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""
Hydra ConfigStore registration for multiview dataloaders.

Registers named dataloader configs that can be referenced via Hydra overrides
(e.g. ``{override /data_train: video_control_mads_multiview_0823_gcs_720p_10fps_93frames_7views}``)
or used as templates for inline ``L(get_multiview_video_loader)(...)`` in
experiment configs.

Two naming conventions:

  **Transfer** (with control signal):
    ``video_control_{dataset}_{store}_{res}_{fps}_{frames}_{views}``

  **Predict** (no control signal):
    ``video_{dataset}_{store}_{res}_{fps}_{frames}_{views}``
"""

from hydra.core.config_store import ConfigStore

from cosmos.utils.lazy_config import LazyCall as L
from cosmos.data.vfm.multiview.multiview_data_source import (
    DEFAULT_CAMERAS,
    INDEX_TO_CAMERA_MAPPING,
    TRANSFER_CAPTION_KEY_MAPPING,
    TRANSFER_CONTROL_KEY_MAPPING,
    TRANSFER_VIDEO_KEY_MAPPING,
)
from cosmos.data.vfm.multiview.multiview_dataset import (
    MultiviewAugmentationConfig,
    get_multiview_video_loader,
)

# ---------------------------------------------------------------------------
# Camera view subsets
# ---------------------------------------------------------------------------

CAMERA_VIEW_CONFIGS: dict[str, tuple[str, ...]] = {
    "7views": DEFAULT_CAMERAS,
    "1view_front": ("camera_front_wide_120fov",),
    "4views": (
        "camera_front_wide_120fov",
        "camera_cross_right_120fov",
        "camera_rear_tele_30fov",
        "camera_cross_left_120fov",
    ),
}

# ---------------------------------------------------------------------------
# Grid dimensions
# ---------------------------------------------------------------------------

_TRANSFER_DATASETS = ["mads_multiview_0823"]
_OBJECT_STORES = ["gcs"]

_RESOLUTIONS: list[tuple[str, tuple[int, int]]] = [
    ("720p", (720, 1280)),
]

_FPS: list[tuple[str, int]] = [
    ("10fps", 1),  # MADS transfer data is already at 10 fps
]

_NUM_VIDEO_FRAMES: list[tuple[str, int]] = [
    ("29frames", 29),
    ("61frames", 61),
    ("93frames", 93),
]


def register_multiview_dataloaders() -> None:
    """Register all multiview dataloader configs with Hydra ConfigStore."""

    cs = ConfigStore.instance()

    # ----- Transfer dataloaders (with control signals) -----
    for dataset in _TRANSFER_DATASETS:
        for object_store in _OBJECT_STORES:
            for resolution_str, resolution_hw in _RESOLUTIONS:
                for fps_str, downsample_factor in _FPS:
                    for num_frames_str, num_frames in _NUM_VIDEO_FRAMES:
                        for views_str, camera_keys in CAMERA_VIEW_CONFIGS.items():
                            name = (
                                f"video_control_{dataset}_{object_store}_{resolution_str}_"
                                f"{fps_str}_{num_frames_str}_{views_str}"
                            )
                            cs.store(
                                group="data_train",
                                package="dataloader_train",
                                name=name,
                                node=L(get_multiview_video_loader)(
                                    dataset_name=dataset,
                                    is_train=True,
                                    augmentation_config=L(MultiviewAugmentationConfig)(
                                        resolution_hw=resolution_hw,
                                        fps_downsample_factor=downsample_factor,
                                        num_video_frames=num_frames,
                                        camera_keys=camera_keys,
                                        camera_video_key_mapping=TRANSFER_VIDEO_KEY_MAPPING,
                                        camera_caption_key_mapping=TRANSFER_CAPTION_KEY_MAPPING,
                                        camera_control_key_mapping=TRANSFER_CONTROL_KEY_MAPPING,
                                        position_to_camera_mapping=INDEX_TO_CAMERA_MAPPING,
                                        single_caption_camera_name="camera_front_wide_120fov",
                                    ),
                                ),
                            )

    # ----- Predict dataloaders (no control signals, for future use) -----
    # These use named keys (video_camera_front_wide_120fov, etc.) and need
    # different datasets (e.g. alpamayo_dec2024) with 30 fps native data.
    # Uncomment and add predict datasets to the catalog when needed.
    #
    # _PREDICT_DATASETS = ["alpamayo_dec2024"]
    # _PREDICT_FPS = [("10fps", 3), ("15fps", 2)]  # 30 fps native → downsample
    # for dataset in _PREDICT_DATASETS:
    #     for object_store in _OBJECT_STORES:
    #         for resolution_str, resolution_hw in _RESOLUTIONS:
    #             for fps_str, downsample_factor in _PREDICT_FPS:
    #                 for num_frames_str, num_frames in _NUM_VIDEO_FRAMES:
    #                     for views_str, camera_keys in CAMERA_VIEW_CONFIGS.items():
    #                         name = (
    #                             f"video_{dataset}_{object_store}_{resolution_str}_"
    #                             f"{fps_str}_{num_frames_str}_{views_str}"
    #                         )
    #                         cs.store(
    #                             group="data_train",
    #                             package="dataloader_train",
    #                             name=name,
    #                             node=L(get_multiview_video_loader)(
    #                                 dataset_name=dataset,
    #                                 is_train=True,
    #                                 augmentation_config=L(MultiviewAugmentationConfig)(
    #                                     resolution_hw=resolution_hw,
    #                                     fps_downsample_factor=downsample_factor,
    #                                     num_video_frames=num_frames,
    #                                     camera_keys=camera_keys,
    #                                     camera_video_key_mapping=PREDICT_VIDEO_KEY_MAPPING,
    #                                     camera_caption_key_mapping=PREDICT_CAPTION_KEY_MAPPING,
    #                                     camera_control_key_mapping=None,
    #                                     position_to_camera_mapping=None,
    #                                     single_caption_camera_name=None,
    #                                 ),
    #                             ),
    #                         )


# Auto-register on import
register_multiview_dataloaders()
