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

import os
import random
import time
from dataclasses import dataclass
from functools import partial
from typing import Callable, Literal

import numpy as np
import torch
import torch.distributed as dist
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader, IterableDataset
from torchcodec.decoders import VideoDecoder

from imaginaire.modules.camera import Camera
from cosmos.utils import log
from cosmos.utils.easy_io import easy_io
from cosmos.data.vfm.action.domain_utils import get_domain_id
from cosmos.data.vfm.action.pose_utils import (
    RotationConvention,
    pose_abs_to_rel,
)
from cosmos.data.vfm.action.unified_dataset import wrap_dataset
from cosmos.data.vfm.joint_dataloader import custom_collate_fn
from cosmos.data.vfm.utils import VIDEO_RES_SIZE_INFO

""" This load the cosmos3 camera-depth data from s3.
File structure:

s3://nv-00-10206-robot/cosmos3_action_data/<dataset_name>/v3/
- videos/
  - <uuid>.mp4
  ...
- captions/
  - <uuid>.json
  ...
- cameras/
  - <uuid>.json
  ...
meta.json

Each video contains N = 150 frames at 30fps.

For multi-view dataset, the structure is slightly different:
- videos/
  - <uuid>/
    - <view_name>.mp4
    ...

The meta.json is like:
{
    "scenes": [
        <uuid>,
        ...
    ]
}

The caption json is like:
{
    "Qwen3-VL-30B-A3B-Instruct": {
        "long": "...",
        "short": "...",  
        "medium": "..." 
    }
}

The camera json is like:
{
    "camera": {
        "focal_length": [fx, fy] * N,
        "principal_point": [cx, cy] * N,
        "pose_world2cam": [qx, qy, qz, qw, tx, ty, tz] * N
    }
}

Training datasets:
- tartanair: 2245 clips at 480x640
- endeavor_forever: 40555 clips at 720x1364
- synhuman_20251218: 24000 clips at 1080x1920
- drivesim: 41639 scenes (each 7 clips) at 1080x1920
- pretrain_camera_260131_10k: 42697 clips, various resolution (most are 720x1280), various fps (most are 30). ~half of the camera is static.
- MultiCamVideo: 13600*10 clips at 1280x1280, 30fps 
- SyncCamVideo: 6800*10 clips at 1280x1280, 30fps 


Testing datasets:
- camera_benchmark: 60 clips (various resolution)
- videos_camera_benchmark: 100 clips (1280x720, various fps)
"""


@dataclass
class DataConfig:
    bucket: str = "nv-00-10206-robot"
    credential_path: str = "credentials/gcp_training.secret"
    validate: bool = False
    validate_num: int = 100  # for each dataset
    caption_model: str = "Qwen3-VL-30B-A3B-Instruct"


    dataset_names: str = "tartanair"

    # Total video frames including the context frame (must be 4n+1 for VAE).
    # E.g. 93 = 1 context + 92 action frames.
    num_frames: int = 93

    # resolution group
    resolution: str = "256"

    # action mode
    mode: str = "forward_dynamics"  # camera-control is forward_dynamics
    embodiment_type: str = "camera_pose"
    # ablation caption

    fix_caption: bool = False
    fix_caption_text: str = "The camera moves in a scene."

    # ablation camera action format
    rotation_format: RotationConvention = "rot9d"
    pose_convention: Literal["backward_anchored", "backward_framewise"] = "backward_framewise"

    translation_scale: float = 1.0
    rotation_scale: float = 1.0

    # If set, downsample the video to this fps. Must be <= video fps (upsampling is forbidden).
    target_fps: float | None = None


def get_target_size_and_crop(resolution: str, current_H: int, current_W: int) -> tuple[int, int, int, int]:
    """Calculates resize dimensions and crop size for smallest-side resize + center crop."""
    target_resolutions = VIDEO_RES_SIZE_INFO[resolution]

    # Find closest supported aspect ratio to minimize cropping
    current_ar = current_W / current_H
    best_key = "1,1"
    min_diff = float("inf")

    for key in target_resolutions:
        w_r, h_r = map(int, key.split(","))
        target_ar = w_r / h_r
        diff = abs(current_ar - target_ar)
        if diff < min_diff:
            min_diff = diff
            best_key = key

    target_canvas_W, target_canvas_H = target_resolutions[best_key]

    # Resize logic (ResizeSmallestSideAspectPreserving)
    # We want the image to cover the target canvas completely.
    scaling_ratio = max(target_canvas_W / current_W, target_canvas_H / current_H)

    new_H = int(scaling_ratio * current_H + 0.5)
    new_W = int(scaling_ratio * current_W + 0.5)

    return new_H, new_W, target_canvas_H, target_canvas_W


class CameraDataset(IterableDataset):
    def __init__(self, conf: DataConfig):
        super().__init__()

        self.conf = conf
        self.domain_id = get_domain_id(conf.embodiment_type)

        easy_io.set_s3_backend(
            backend_args={
                "backend": "s3",
                "path_mapping": None,
                "s3_credential_path": self.conf.credential_path,
            }
        )

        self.uids = []
        dataset_names = conf.dataset_names.split(",")
        for dataset_name in dataset_names:
            path_uids = f"s3://{conf.bucket}/cosmos3_action_data/{dataset_name}/v3/meta.json"
            uids = easy_io.load(path_uids)["scenes"]  # list of uids

            # for benchmark dataset, do not split
            if dataset_name not in ["camera_benchmark", "videos_camera_benchmark"]:
                # train/test split
                assert self.conf.validate_num > 0 and len(uids) >= self.conf.validate_num
                stride = len(uids) // self.conf.validate_num
                val_indices = {i * stride for i in range(self.conf.validate_num)}
                if self.conf.validate:
                    uids = [uids[i] for i in sorted(val_indices)]
                else:
                    uids = [uids[i] for i in range(len(uids)) if i not in val_indices]

            for uid in uids:
                self.uids.append((dataset_name, uid))

        log.warning(f"Loaded {len(self.uids)} uids from {conf.dataset_names}")

    def __len__(self):
        return len(self.uids)

    def __iter__(self):
        if self.conf.validate:
            for i in range(len(self.uids)):
                sample = self.load_data(self.uids[i])
                if sample is None:
                    continue
                else:
                    yield sample
        else:
            # infinite random loop for training
            while True:
                indices = np.random.permutation(len(self.uids))
                for i in indices:
                    sample = self.load_data(self.uids[i])
                    if sample is None:
                        continue
                    else:
                        yield sample

    def load_data(self, uid: str) -> dict | None:
        """Load and preprocess a single sample.
        For multi-view dataset, we randomly load one view at each iteration.

        Args:
            uid: Unique identifier for the sample (e.g., 'westerndesert_Hard_P007')

        Returns:
            Dictionary with video, action, and metadata, or None if loading fails.
        """
        try:
            dataset_name, sample_name = uid

            if dataset_name == "drivesim":
                # multi-view dataset
                view_names = [
                    "camera_bev_from_behind_ego",
                    "camera_bev_looking_back_at_ego",
                    "camera_crossleft",
                    "camera_crossright",
                    "camera_frontwide",
                    "camera_rearleft",
                    "camera_rearright",
                ]
                view_name = random.choice(view_names) if not self.conf.validate else view_names[0]
                sample_name = os.path.join(sample_name, view_name)
            elif dataset_name in ["MultiCamVideo", "SyncCamVideo"]:
                # multi-view dataset
                view_names = [
                    "cam01",
                    "cam02",
                    "cam03",
                    "cam04",
                    "cam05",
                    "cam06",
                    "cam07",
                    "cam08",
                    "cam09",
                    "cam10",
                ]
                view_name = random.choice(view_names) if not self.conf.validate else view_names[0]
                sample_name = os.path.join(sample_name, view_name)

            video_path = f"s3://{self.conf.bucket}/cosmos3_action_data/{dataset_name}/v3/videos/{sample_name}.mp4"
            camera_path = f"s3://{self.conf.bucket}/cosmos3_action_data/{dataset_name}/v3/cameras/{sample_name}.json"
            caption_path = f"s3://{self.conf.bucket}/cosmos3_action_data/{dataset_name}/v3/captions/{sample_name}.json"

            # Load video bytes from S3
            video_bytes = easy_io.get(video_path)

            decoder = VideoDecoder(video_bytes, num_ffmpeg_threads=4)
            del video_bytes
            total_frames = decoder.metadata.num_frames
            video_fps = decoder.metadata.average_fps

            # Determine temporal stride for fps downsampling
            if self.conf.target_fps is not None:
                if self.conf.target_fps > video_fps:
                    raise ValueError(
                        f"target_fps ({self.conf.target_fps}) > video_fps ({video_fps}). Upsampling is not supported."
                    )
                stride = round(video_fps / self.conf.target_fps)
                effective_fps = video_fps / stride
            else:
                stride = 1
                effective_fps = video_fps

            # Sample consecutive frames
            if self.conf.num_frames == -1:
                # Load all available frames, aligned to VAE constraint (1 + 4*N total frames)
                num_strided_frames = (total_frames - 1) // stride + 1
                N = (num_strided_frames - 1) // 4
                if N < 1:
                    log.warning(f"Not enough frames for {uid}, total_frames: {total_frames}, stride: {stride}")
                    return None
                num_frames_to_load = 1 + 4 * N
                start_idx = 0
            else:
                num_frames_to_load = self.conf.num_frames
                num_raw_frames_needed = (num_frames_to_load - 1) * stride + 1

                if total_frames < num_raw_frames_needed:
                    log.warning(
                        f"Not enough frames to load for {uid}, total_frames: {total_frames}, num_raw_frames_needed: {num_raw_frames_needed}"
                    )
                    return None

                # Random start index for consecutive frame sampling
                if self.conf.validate:
                    start_idx = 0
                else:
                    start_idx = random.randint(0, total_frames - num_raw_frames_needed)

            num_raw_frames_needed = (num_frames_to_load - 1) * stride + 1
            frame_indices = list(range(start_idx, start_idx + num_raw_frames_needed, stride))

            # torchcodec returns [T,C,H,W] tensor
            frame_batch = decoder.get_frames_at(frame_indices)
            video_frames = frame_batch.data  # [T,C,H,W] uint8
            del decoder

            # Get target size and crop params
            T, C, H, W = video_frames.shape

            # temp: for pretrain_camera_260131_10k, we only use videos with at 720x1280 to enable batched training
            if dataset_name == "pretrain_camera_260131_10k":
                assert H == 720 and W == 1280, f"Expected resolution 720x1280, got {H}x{W}"

            new_H, new_W, target_canvas_H, target_canvas_W = get_target_size_and_crop(self.conf.resolution, H, W)

            # Resize if needed
            if new_H != H or new_W != W:
                video_frames = F.resize(
                    video_frames, [new_H, new_W], interpolation=F.InterpolationMode.BICUBIC, antialias=True
                )

            # Center Crop
            if new_H != target_canvas_H or new_W != target_canvas_W:
                video_frames = F.center_crop(video_frames, [target_canvas_H, target_canvas_W])

            # Convert to (C, T, H, W) format expected by model
            video = video_frames.permute(1, 0, 2, 3)  # [C,T,H,W]

            # Load camera data
            camera_data = easy_io.load(camera_path)
            w2c = np.array(camera_data["camera"]["pose_world2cam"]).reshape(-1, 7)  # [N,7]
            assert w2c.shape[0] == total_frames, f"Expected {total_frames} poses, got {w2c.shape[0]}"

            # Get w2c for the sampled frames (same stride as video frames)
            w2c = w2c[start_idx : start_idx + num_raw_frames_needed : stride]  # [num_frames,7]
            # Convert (qx,qy,qz,qw,tx,ty,tz) to [R|t] matrices
            w2c = Camera.extrinsic_params_to_matrices(w2c)  # [num_frames,3,4]
            w2c_homo = np.eye(4, dtype=np.float32)[None, :, :].repeat(w2c.shape[0], axis=0)  # [num_frames,4,4]
            w2c_homo[:, :3, :] = w2c
            c2w_homo = np.linalg.inv(w2c_homo)

            # Determine mode
            if self.conf.mode == "joint":
                mode = random.choice(["forward_dynamics", "inverse_dynamics", "policy"])
            else:
                mode = self.conf.mode

            # Using c2w_homo because pose_abs_to_rel expects c2w poses
            # The function handles w2c/c2w/anchored logic based on pose_convention
            action = pose_abs_to_rel(
                c2w_homo,
                rotation_format=self.conf.rotation_format,
                pose_convention=self.conf.pose_convention,
                translation_scale=self.conf.translation_scale,
                rotation_scale=self.conf.rotation_scale,
            )
            action = torch.from_numpy(action)  # [num_frames-1,action_dim]

            # Load caption data
            if self.conf.fix_caption:
                caption = self.conf.fix_caption_text
            else:
                caption_data = easy_io.load(caption_path)
                captions = caption_data[self.conf.caption_model]
                # Randomly select from long/short/medium captions
                if self.conf.validate:
                    caption = captions["long"]
                else:
                    selected_type = random.choice(["long", "short", "medium"])
                    caption = captions[selected_type]

            # FPS (must be scalar fps, not [fps])
            fps = torch.tensor(effective_fps, dtype=torch.float32)  # scalar

            # Build sample dict matching dummy_dataset format
            sample = {
                "video": video,  # [C,num_frames,H,W] uint8
                "action": action,  # [num_frames-1,max_action_dim] float32
                "conditioning_fps": fps,  # scalar float32
                "ai_caption": caption,
                "mode": mode,
                "__key__": torch.tensor([hash(uid) % (2**31)], dtype=torch.long),
                "domain_id": torch.tensor(self.domain_id, dtype=torch.long),
                "viewpoint": "ego_view",
            }
            return sample

        except Exception as e:
            log.warning(f"Error loading {uid}: {e}")
            return None  # skip this sample


# to align with other robot datasets interface.
class CameraDatasetPatched(CameraDataset):
    def __init__(
        self,
        dataset_names: str,
        split: Literal["train", "val"] = "train",
        num_frames: int = 93,
        resolution: str = "256",
        mode: str = "forward_dynamics",
        fix_caption: bool = False,
        rotation_format: RotationConvention = "rot9d",
        pose_convention: Literal["backward_anchored", "backward_framewise"] = ("backward_framewise"),
        translation_scale: float = 1.0,
        rotation_scale: float = 1.0,
        shuffle: bool = False,  # not used, decided by split
    ):
        super().__init__(
            conf=DataConfig(
                dataset_names=dataset_names,
                validate=split == "val",
                num_frames=num_frames,
                resolution=resolution,
                mode=mode,
                fix_caption=fix_caption,
                rotation_format=rotation_format,
                pose_convention=pose_convention,
                translation_scale=translation_scale,
                rotation_scale=rotation_scale,
            )
        )
        self.dataset_names = dataset_names
        self.split = split
        self.num_frames = num_frames
        self.resolution = resolution
        self.mode = mode
        self.fix_caption = fix_caption
        self.rotation_format = rotation_format
        self.pose_convention = pose_convention
        self.translation_scale = translation_scale
        self.rotation_scale = rotation_scale


def worker_init_fn(worker_id: int, seed: int = 0):
    try:
        rank = dist.get_rank()
    except Exception:
        rank = 0
    seed = seed + rank * 100000 + worker_id
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


class MixedDataLoader:
    def __init__(self, dataloaders: list[DataLoader], ratios: list[float]):
        self.dataloaders = dataloaders
        self.ratios = ratios

    def __len__(self) -> int:
        return sum(len(dl.dataset) for dl in self.dataloaders)

    def __iter__(self):
        iterators = [iter(dl) for dl in self.dataloaders]
        while True:
            choice = np.random.choice(len(self.dataloaders), p=self.ratios)
            yield next(iterators[choice])


def get_camera_dataloader(
    conf: dict[str, DataConfig],
    ratios: list[float] | None = None,
    batch_size: int = 1,
    num_workers: int = 8,
    seed: int | None = None,
    # wrap_dataset parameters for Action transform pipeline
    resolution: str | None = None,
    tokenizer_config: dict | None = None,
    cfg_dropout_rate: float = 0.0,
    max_action_dim: int = 32,
    collate_fn: Callable = custom_collate_fn,
):
    # random seed to make sure each training has different data order
    if seed is None:
        seed = random.randint(0, 1000000)

    # num_workers == 0 will not call worker_init_fn, so we need to seed the main process directly
    if num_workers == 0:
        try:
            rank = dist.get_rank()
        except Exception:
            rank = 0
        rank_seed = seed + rank * 100000
        np.random.seed(rank_seed)
        torch.manual_seed(rank_seed)
        random.seed(rank_seed)

    # filter out None configs (e.g. when overriding)
    conf = {k: v for k, v in conf.items() if v is not None}

    if len(conf) > 1:
        dataloaders = []
        dataset_lengths = []

        for i, dataset_name in enumerate(conf.keys()):
            # dataset wrapped with Action transform pipeline
            dataset = CameraDataset(conf=conf[dataset_name])
            dataset_lengths.append(len(dataset))
            wrapped = wrap_dataset(
                dataset,
                resolution=resolution,
                tokenizer_config=tokenizer_config,
                cfg_dropout_rate=cfg_dropout_rate,
                max_action_dim=max_action_dim,
            )

            # loader
            loader = DataLoader(
                dataset=wrapped,
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=True,
                worker_init_fn=partial(worker_init_fn, seed=seed),
                collate_fn=collate_fn,
            )
            dataloaders.append(loader)

        # calculate ratios if not provided
        if ratios is None:
            total_len = sum(dataset_lengths)
            ratios = [l / total_len for l in dataset_lengths]

        log.info(f"MixedDataset sizes: {dataset_lengths}, ratios: {ratios}")

        result = MixedDataLoader(dataloaders, ratios)
    else:
        # dataset wrapped with Action transform pipeline
        dataset = CameraDataset(conf=list(conf.values())[0])
        wrapped = wrap_dataset(
            dataset,
            resolution=resolution,
            tokenizer_config=tokenizer_config,
            cfg_dropout_rate=cfg_dropout_rate,
            max_action_dim=max_action_dim,
        )

        # loader
        result = DataLoader(
            dataset=wrapped,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            worker_init_fn=partial(worker_init_fn, seed=seed),
            collate_fn=collate_fn,
        )

        log.info(f"Dataset size: {len(dataset)}")

    return result


# PYTHONPATH=. python cosmos/data/vfm/action/camera_dataset.py
if __name__ == "__main__":
    import torchvision.io as io

    dataset = CameraDataset(conf=DataConfig(dataset_names="MultiCamVideo", validate=False, resolution="480"))
    dataset_iter = iter(dataset)

    for i in range(3):
        print(f"==================== Sample {i} ====================")
        _t0 = time.time()
        data = next(dataset_iter)
        _t1 = time.time()
        print(f"{'Loading time':<25}: {_t1 - _t0:.2f}s")

        print(f"==================== Sample {i} ====================")
        print(f"{'video shape':<25}: {data['video'].shape}")  # [C,T,H,W]
        print(f"{'action shape':<25}: {data['action'].shape}")  # [T,max_action_dim]
        print(f"{'conditioning_fps':<25}: {data['conditioning_fps'].item()}")
        print(f"{'mode':<25}: {data['mode']}")
        print(f"{'domain_id':<25}: {data['domain_id'].item()}")
        print(f"{'caption':<25}: {data['ai_caption']}...")

        # save video to local for debugging
        video = data["video"]
        video = video.permute(1, 0, 2, 3)  # [T,C,H,W]
        video_path = f"temp/camera_sample_{i}.mp4"
        io.write_video(
            video_path, video.permute(0, 2, 3, 1).numpy(), fps=data["conditioning_fps"].item()
        )  # expects (T, H, W, C)
        print(f"Saved video to {video_path}")
