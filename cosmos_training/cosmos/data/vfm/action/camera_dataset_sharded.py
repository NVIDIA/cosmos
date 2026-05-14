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

"""Sharded Camera Dataset for Action training.

Example data structure:

wdinfo:
s3://nv-00-10206-robot/cosmos3_action_data/camera_webdataset_wdinfo/
  tartanair/v3/resolution_480/aspect_ratio_5_4/duration_150/wdinfo_02132026.json

tarfiles:
s3://nv-00-10206-robot/cosmos3_action_data/camera_webdataset/
  tartanair/v3/resolution_480/aspect_ratio_5_4/duration_150/
    video/
      part_000000.tar
      part_000001.tar
      ...
    camera/
      part_000000.tar
      part_000001.tar
      ...
    metas/
      part_000000.tar
      part_000001.tar
      ...
"""

import io
import json
import os
import random
import re
import tarfile
from typing import Iterator, Literal

import numpy as np
import torch
from torch.utils.data import IterableDataset
from torchcodec.decoders import VideoDecoder

from imaginaire.modules.camera import Camera
from cosmos.utils import log
from cosmos.utils.easy_io import easy_io
from cosmos.data.vfm.action.domain_utils import get_domain_id
from cosmos.data.vfm.action.pose_utils import (
    RotationConvention,
    pose_abs_to_rel,
)

# all ready-to-use wdinfos
CAMERA_WDINFOS = {
    # ----- 5s datasets -----
    "tartanair_256": "s3://nv-00-10206-robot/cosmos3_action_data/camera_webdataset_wdinfo/tartanair/v1/resolution_256/aspect_ratio_5_4/duration_150/wdinfo_03022026.json",
    "tartanair_480": "s3://nv-00-10206-robot/cosmos3_action_data/camera_webdataset_wdinfo/tartanair/v1/resolution_544/aspect_ratio_23_17/duration_150/wdinfo_03022026.json",
    "endeavor_forever_256": "s3://nv-00-10206-robot/cosmos3_action_data/camera_webdataset_wdinfo/endeavor_forever/v1/resolution_192/aspect_ratio_5_3/duration_150/wdinfo_03032026.json",
    "endeavor_forever_480": "s3://nv-00-10206-robot/cosmos3_action_data/camera_webdataset_wdinfo/endeavor_forever/v1/resolution_480/aspect_ratio_26_15/duration_150/wdinfo_03032026.json",
    "synhuman_20251218_256": "s3://nv-00-10206-robot/cosmos3_action_data/camera_webdataset_wdinfo/synhuman_20251218/v1/resolution_192/aspect_ratio_5_3/duration_150/wdinfo_03032026.json",
    "synhuman_20251218_480": "s3://nv-00-10206-robot/cosmos3_action_data/camera_webdataset_wdinfo/synhuman_20251218/v1/resolution_480/aspect_ratio_26_15/duration_150/wdinfo_03032026.json",
    "pretrained_clips_260131_10k_256": "s3://nv-00-10206-robot/cosmos3_action_data/camera_webdataset_wdinfo/pretrained_clips_260131_10k/v3/resolution_192/aspect_ratio_5_3/duration_150/wdinfo_02172026.json",
    "pretrained_clips_260131_10k_480": "s3://nv-00-10206-robot/cosmos3_action_data/camera_webdataset_wdinfo/pretrained_clips_260131_10k/v3/resolution_480/aspect_ratio_26_15/duration_150/wdinfo_02172026.json",
    "synhuman_20260223_480": "s3://nv-00-10206-robot/cosmos3_action_data/camera_webdataset_wdinfo/synhuman_20260223/v1/resolution_480/aspect_ratio_26_15/duration_150/wdinfo_03032026.json",
    # ----- 10s datasets -----
    # pretrain video 100k (mixed resolution, 193k samples)
    "pretrained_clips_260307_100k": "s3://nv-00-10206-robot/cosmos3_action_data/camera_webdataset_wdinfo/pretrained_clips_260307_100k/v1p1/",
    # pretrain video 100k filtered (mixed resolution, 72k samples)
    "pretrained_clips_260307_100k_filtered": "s3://nv-00-10206-robot/cosmos3_action_data/camera_webdataset_wdinfo/pretrained_clips_260307_100k_filtered/v1p1/",
    "pretrained_clips_260325_500k_01_filtered": "s3://nv-00-10206-robot/cosmos3_action_data/camera_webdataset_wdinfo/pretrained_clips_260325_500k_01_filtered/v1p1/",
    "pretrained_clips_260325_500k_02_filtered": "s3://nv-00-10206-robot/cosmos3_action_data/camera_webdataset_wdinfo/pretrained_clips_260325_500k_02_filtered/v1p1/",
    "pretrained_clips_260313_10s_100k_filtered": "s3://nv-00-10206-robot/cosmos3_action_data/camera_webdataset_wdinfo/pretrained_clips_260313_10s_100k_filtered/v1p1/",
    # ----- 60s datasets -----
    # 60s duration
    "endeavor_forever_480_60s": "s3://nv-00-10206-robot/cosmos3_action_data/camera_webdataset_wdinfo/endeavor_forever/v1p2/resolution_480/aspect_ratio_26_15/duration_1800/wdinfo_03032026.json",  # only 1798 frames actually!
    "drivesim_480_60s": "s3://nv-00-10206-robot/cosmos3_action_data/camera_webdataset_wdinfo/drivesim/v1p2/resolution_480/aspect_ratio_26_15/duration_1800/wdinfo_03042026.json",  # 1749 frames
    # pretrain video 100k 59s filtered (mixed resolution, 16k samples)
    "pretrained_clips_260313_59s_100k_filtered": "s3://nv-00-10206-robot/cosmos3_action_data/camera_webdataset_wdinfo/pretrained_clips_260313_59s_100k_filtered/v1p2/",
    "pretrained_clips_260313_59s_100k_filtered_480": "s3://nv-00-10206-robot/cosmos3_action_data/camera_webdataset_wdinfo/pretrained_clips_260313_59s_100k_filtered/v1p2/resolution_480/aspect_ratio_16_9/frames_1700_1800/wdinfo_03242026.json",
}


class CameraDatasetSharded(IterableDataset):
    def __init__(
        self,
        wdinfo_paths: list[str] | None = None,
        bucket: str = "nv-00-10206-robot",
        credential_path: str = "credentials/gcp_training.secret",
        mode: str = "forward_dynamics",
        embodiment_type: str = "camera_pose",
        split: str = "train",
        seed: int = 0,
        shuffle: bool = True,
        rotation_format: RotationConvention = "rot6d",
        pose_convention: Literal["backward_anchored", "backward_framewise"] = ("backward_framewise"),
        fix_caption: bool = True,  # fix caption by default
        fix_caption_text: str = "The camera moves in a scene.",
        translation_scale: float = 1.0,
        rotation_scale: float = 1.0,
        discard_varying_intrinsics: bool = False,
        # wdinfo filtering
        wdinfo_resolution: Literal["all", "gt480", "gt720"] = "all",
        max_frames: int = -1,  # only truncate if exceed max_frames
        num_frames: int = -1,  # always truncate to num_frames
        # When True, use a separate domain ID for inverse dynamics / policy modes
        # so that DomainAwareLinear learns different projections for anchored (conditioning)
        # vs framewise (generation) action representations.
        mode_aware_domain: bool = False,
        inv_embodiment_type: str = "camera_pose_inv",
        max_action_translation_norm: float | None = None,
        # When True, decode the full video in one pass (used by KV-cache segment loaders).
        whole_video: bool = False,
        # Benchmark datasets: list of S3 dataset names for loading individual files
        # instead of tarfiles.  Activated automatically when split is val-like.
        benchmark_datasets: list[str] | None = ("videos_camera_benchmark",),
        # Caption model key used in the benchmark caption JSON files.
        benchmark_caption_model: str = "Qwen3-VL-30B-A3B-Instruct",
    ):
        super().__init__()

        self.wdinfo_paths = wdinfo_paths or []
        self.bucket = bucket
        self.credential_path = credential_path
        self.mode = mode
        self.split = split.lower().strip()
        self.seed = seed
        self.shuffle = shuffle
        self.rotation_format: RotationConvention = rotation_format
        self.pose_convention: Literal["absolute", "backward_anchored", "backward_framewise"] = pose_convention

        # hard-coded caption keys and weights for now
        caption_keys = {
            "qwen2p5_7b_caption": 0.7,
            "qwen2p5_7b_caption_short": 0.1,
            "qwen2p5_7b_caption_medium": 0.2,
        }
        self.caption_keys = list(caption_keys.keys())
        self.caption_weights = list(caption_keys.values())

        self.fix_caption = fix_caption
        self.fix_caption_text = fix_caption_text
        self.translation_scale = translation_scale
        self.rotation_scale = rotation_scale
        self.discard_varying_intrinsics = discard_varying_intrinsics
        self.max_action_translation_norm = max_action_translation_norm
        self.wdinfo_resolution = wdinfo_resolution
        self.max_frames = max_frames
        self.num_frames = num_frames
        self.whole_video = whole_video
        # Get domain ID for this embodiment
        self.domain_id = get_domain_id(embodiment_type)
        # When mode_aware_domain is True, inverse_dynamics/policy modes use a separate domain ID
        self.mode_aware_domain = mode_aware_domain
        self.domain_id_inv = get_domain_id(inv_embodiment_type) if mode_aware_domain else self.domain_id
        self.benchmark_caption_model = benchmark_caption_model

        # Validate mode
        valid_modes = ["joint", "forward_dynamics", "inverse_dynamics", "policy", "image2video"]
        if mode not in valid_modes:
            raise ValueError(f"mode must be one of {valid_modes}, got {mode}")

        # Validate split
        if self.split not in {"train", "val", "valid", "validation", "full"}:
            raise ValueError(f"Unsupported {split=}. Use train/val/full.")

        # Configure S3 backend using easy_io
        self._setup_s3_backend()

        # Validation always uses benchmark datasets
        self._benchmark_uids: list[tuple[str, str]] = []
        if self.split in {"val", "valid", "validation"}:
            if not benchmark_datasets:
                raise ValueError("benchmark_datasets is required for validation splits.")
            self._load_benchmark_uids(benchmark_datasets)

        # Load tar files from wdinfo.json (training / full splits only)
        self._tar_files: list[str] = []
        self._key_count: list[int] = []
        self._total_key_count: int = 0

        if self._benchmark_uids:
            log.info(
                f"Initialized CameraDatasetSharded (benchmark): "
                f"benchmark_datasets={benchmark_datasets}, "
                f"mode={mode}, split={self.split}, "
                f"num_samples={len(self._benchmark_uids)}"
            )
        else:
            if not self.wdinfo_paths:
                raise ValueError("Must provide wdinfo_paths for training.")

            # Resolve prefix paths into individual wdinfo JSON paths
            self.wdinfo_paths = self._resolve_wdinfo_paths(self.wdinfo_paths)

            # Filter wdinfo paths by resolution
            self.wdinfo_paths = self._filter_wdinfo_by_resolution(self.wdinfo_paths)

            self._load_wdinfo()

            log.info(
                f"Initialized CameraDatasetSharded: wdinfo_paths={self.wdinfo_paths}, "
                f"mode={mode}, split={self.split}, "
                f"num_tar_files={len(self._tar_files)}, "
                f"total_samples={self._total_key_count}"
            )

    def _setup_s3_backend(self) -> None:
        """Configure the easy_io S3 backend. Called in __init__ and __iter__ for worker processes."""
        easy_io.set_s3_backend(
            backend_args={
                "backend": "s3",
                "path_mapping": None,
                "s3_credential_path": self.credential_path,
            }
        )

    def _load_benchmark_uids(self, benchmark_datasets: list[str]) -> None:
        """Load UIDs from benchmark datasets stored as individual S3 files.

        Each benchmark dataset lives at ``s3://<bucket>/cosmos3_action_data/<name>/v3/``
        with a ``meta.json`` listing scene UIDs and per-UID ``videos/``, ``cameras/``,
        ``captions/`` directories.
        """
        for dataset_name in benchmark_datasets:
            meta_path = f"s3://{self.bucket}/cosmos3_action_data/{dataset_name}/v3/meta.json"
            try:
                uids = json.loads(easy_io.get(meta_path))["scenes"]
            except Exception as e:
                raise RuntimeError(f"Failed to load benchmark meta from {meta_path}: {e}") from e
            for uid in uids:
                self._benchmark_uids.append((dataset_name, uid))
            log.info(f"Loaded {len(uids)} benchmark UIDs from {dataset_name}")

    def _resolve_wdinfo_paths(self, wdinfo_paths: list[str]) -> list[str]:
        """Resolve wdinfo paths: direct .json paths are kept as-is,
        prefix/directory paths are walked recursively to find all .json files."""
        resolved: list[str] = []
        for path in wdinfo_paths:
            if path.endswith(".json"):
                resolved.append(path)
            else:
                log.info(f"Walking S3 prefix to discover wdinfo JSONs: {path}")
                relative_json_files = list(
                    easy_io.list_dir_or_file(path, list_dir=False, list_file=True, suffix=".json", recursive=True)
                )
                if not relative_json_files:
                    raise RuntimeError(f"No .json wdinfo files found under prefix: {path}")
                prefix = path.rstrip("/") + "/"
                for rel_path in sorted(relative_json_files):
                    resolved.append(prefix + rel_path)
                log.info(f"Discovered {len(relative_json_files)} wdinfo files under {path}")
        return resolved

    def _filter_wdinfo_by_resolution(self, wdinfo_paths: list[str]) -> list[str]:
        """Filter resolved wdinfo paths by minimum resolution.

        Resolution is extracted from the path pattern ``resolution_<N>``.
        For ``wdinfo_resolution="gtXXX"``, only paths with resolution >= XXX are kept.
        """
        if self.wdinfo_resolution == "all":
            return wdinfo_paths

        threshold_match = re.match(r"gt(\d+)", self.wdinfo_resolution)
        if not threshold_match:
            raise ValueError(
                f"Invalid wdinfo_resolution format: {self.wdinfo_resolution!r}. "
                "Expected 'all' or 'gtNNN' (e.g. 'gt480')."
            )
        min_resolution = int(threshold_match.group(1))

        filtered: list[str] = []
        for path in wdinfo_paths:
            res_match = re.search(r"resolution_(\d+)", path)
            if res_match:
                resolution = int(res_match.group(1))
                if resolution >= min_resolution:
                    filtered.append(path)
            else:
                log.warning(f"No resolution found in wdinfo path, including by default: {path}")
                filtered.append(path)

        log.info(f"Filtered wdinfo by resolution >= {min_resolution}: {len(filtered)}/{len(wdinfo_paths)} paths kept")
        if not filtered:
            raise RuntimeError(f"All wdinfo paths were filtered out by wdinfo_resolution={self.wdinfo_resolution!r}")
        return filtered

    def _load_wdinfo(self) -> None:
        self._tar_files = []
        self._key_count = []
        self._total_key_count = 0

        for wdinfo_path in self.wdinfo_paths:
            # log.info(f"Loading wdinfo from: {wdinfo_path}")

            try:
                wdinfo_bytes = easy_io.get(wdinfo_path)
                wdinfo = json.loads(wdinfo_bytes)
            except Exception as e:
                raise RuntimeError(f"Failed to load wdinfo from {wdinfo_path}: {e}") from e

            # Extract metadata
            data_root = wdinfo["root"].rstrip(
                "/"
            )  # e.g. cosmos3_action_data/camera_webdataset/tartanair/v3/resolution_256/aspect_ratio_5_4/duration_150/
            data_list = wdinfo["data_list"]  # ["part_000000.tar", "part_000001.tar", ...]
            key_count = wdinfo["data_list_key_count"]  # [20, 20, ...]

            # only for video tar files
            tar_paths = [f"s3://{self.bucket}/{data_root}/video/{tar_path}" for tar_path in data_list]

            # Reconstruct full S3 paths for tar files
            self._tar_files.extend(tar_paths)
            self._key_count.extend(key_count)

            # Accumulate total sample count
            self._total_key_count += sum(key_count)

            # log.info(f"Loaded {len(data_list)} tar files from wdinfo with {sum(key_count)} samples")

        if not self._tar_files:
            raise RuntimeError(f"No tar files found in wdinfo at {self.wdinfo_paths}")

    def __len__(self) -> int:
        if self._benchmark_uids:
            return len(self._benchmark_uids)
        return self._total_key_count

    def __iter__(self) -> Iterator[dict]:
        """Iterate over the dataset."""
        # Re-configure S3 backend in case this is running in a worker process after unpickling
        self._setup_s3_backend()

        if self._benchmark_uids:
            yield from self._iter_benchmark()
        else:
            yield from self._iter_tar_files()

    def _iter_benchmark(self) -> Iterator[dict]:
        """Iterate over benchmark datasets (individual S3 files)."""
        uids = list(self._benchmark_uids)
        if self.shuffle:
            random.shuffle(uids)

        skipped: list[dict[str, str]] = []
        yielded = 0

        for dataset_name, uid in uids:
            try:
                base = f"s3://{self.bucket}/cosmos3_action_data/{dataset_name}/v3"
                video_bytes = easy_io.get(f"{base}/videos/{uid}.mp4")
                camera_bytes = easy_io.get(f"{base}/cameras/{uid}.json")

                # Load caption from per-sample JSON (different format from tarfile metas)
                caption = None
                if not self.fix_caption:
                    caption_data = json.loads(easy_io.get(f"{base}/captions/{uid}.json"))
                    captions = caption_data[self.benchmark_caption_model]
                    caption = random.choice([captions["long"], captions["short"], captions["medium"]])

                sample = self._process_sample(video_bytes, camera_bytes, uid=uid, caption_override=caption)
                if sample:
                    yielded += 1
                    yield sample
                else:
                    log.warning(f"SKIPPED benchmark sample {dataset_name}/{uid}: _process_sample returned None")
                    skipped.append({"dataset": dataset_name, "uid": uid, "reason": "process_sample_returned_none"})

            except Exception as e:
                log.warning(f"SKIPPED benchmark sample {dataset_name}/{uid}: S3 load failed: {e}")
                skipped.append({"dataset": dataset_name, "uid": uid, "reason": f"s3_load_error: {e}"})
                continue

        log.info(f"Benchmark iteration complete: {yielded}/{len(uids)} samples yielded, {len(skipped)} skipped")
        if skipped:
            log.warning(f"Skipped benchmark samples: {skipped}")
        self._last_skipped_benchmark_samples = skipped

    def _iter_tar_files(self) -> Iterator[dict]:
        """Iterate over sharded tar files from S3."""
        tar_files = list(self._tar_files)
        if self.shuffle:
            random.shuffle(tar_files)

        for tar_path in tar_files:
            try:
                # Read tar file bytes using easy_io
                metas_path = tar_path.replace("/video/", "/metas/")
                camera_path = tar_path.replace("/video/", "/camera/")

                video_bytes = easy_io.get(tar_path)
                metas_bytes = easy_io.get(metas_path)
                camera_bytes = easy_io.get(camera_path)

                with (
                    tarfile.open(fileobj=io.BytesIO(video_bytes), mode="r") as video_tar,
                    tarfile.open(fileobj=io.BytesIO(metas_bytes), mode="r") as metas_tar,
                    tarfile.open(fileobj=io.BytesIO(camera_bytes), mode="r") as camera_tar,
                ):
                    # Map filenames to members for fast lookup
                    camera_members = {m.name: m for m in camera_tar.getmembers() if m.isfile()}
                    metas_members = {m.name: m for m in metas_tar.getmembers() if m.isfile()}

                    # Iterate over video files
                    for video_member in video_tar.getmembers():
                        uuid = os.path.splitext(os.path.basename(video_member.name))[0]
                        json_name = f"{uuid}.json"

                        if json_name not in camera_members or json_name not in metas_members:
                            raise ValueError(f"Missing metadata or camera for {uuid} in {tar_path}")

                        # Extract data
                        video_file_bytes = video_tar.extractfile(video_member).read()
                        camera_file_bytes = camera_tar.extractfile(camera_members[json_name]).read()
                        metas_file_bytes = metas_tar.extractfile(metas_members[json_name]).read()

                        sample = self._process_sample(
                            video_file_bytes,
                            camera_file_bytes,
                            uid=uuid,
                            metas_bytes=metas_file_bytes,
                        )
                        if sample:
                            yield sample
                            # DEBUG: only use 1 video per tar
                            # break

            except Exception as e:
                log.warning(f"Failed to read tar file {tar_path}: {e}")
                continue

    def _process_sample(
        self,
        video_bytes: bytes,
        camera_bytes: bytes,
        uid: str = "",
        *,
        metas_bytes: bytes | None = None,
        caption_override: str | None = None,
    ) -> dict | None:
        """Decode video frames, compute relative actions, and return a sample dict.

        When ``whole_video`` is True or ``num_frames == -1``, loads the full video from frame 0.
        Otherwise applies ``max_frames`` / ``num_frames`` clipping and sampling as in the base
        sharded loader. For the validation split, random clips use start index 0.

        Args:
            caption_override: When provided, use this caption directly instead of
                extracting from *metas_bytes*.  Used by the benchmark data path where
                captions live in a different JSON format.
        """
        try:
            decoder = VideoDecoder(video_bytes, num_ffmpeg_threads=4)
            total_frames = decoder.metadata.num_frames or 0
            video_fps = decoder.metadata.average_fps or 0.0
            whole_video = self.whole_video

            if whole_video:
                if total_frames == 0:
                    del decoder
                    log.warning(f"SKIPPED {uid}: whole_video=True but total_frames=0")
                    return None
                frame_indices = list(range(total_frames))
            elif self.max_frames > 0:
                # Use all frames at native fps (stride=1), capped by max_frames.
                # VAE compresses temporal dim by 4x with 1 condition frame,
                # so total video frames must be 1 + 4*N.
                num_video_frames = min(total_frames, self.max_frames)
                N = (num_video_frames - 1) // 4
                num_video_frames = 1 + 4 * N
                if num_video_frames < 2:
                    del decoder
                    log.warning(
                        f"SKIPPED {uid}: too few frames after VAE alignment "
                        f"(total_frames={total_frames}, max_frames={self.max_frames}, aligned={num_video_frames})"
                    )
                    return None
                frame_indices = list(range(num_video_frames))
            else:
                # No max_frames cap — use all frames, but still VAE-align to 1 + 4*N.
                N = (total_frames - 1) // 4
                num_video_frames = 1 + 4 * N
                if num_video_frames < 2:
                    del decoder
                    log.warning(
                        f"SKIPPED {uid}: too few frames after VAE alignment "
                        f"(total_frames={total_frames}, aligned={num_video_frames})"
                    )
                    return None
                frame_indices = list(range(num_video_frames))

            # If num_frames is set, always sample exactly num_frames from the available frames.
            if self.num_frames > 0 and not whole_video:
                available = len(frame_indices)
                N = (self.num_frames - 1) // 4
                target_frames = 1 + 4 * N
                if target_frames < 2 or available < target_frames:
                    del decoder
                    log.warning(
                        f"SKIPPED {uid}: not enough frames for num_frames={self.num_frames} "
                        f"(available={available}, target={target_frames})"
                    )
                    return None
                max_start = available - target_frames
                start = 0 if self.split != "train" else (random.randint(0, max_start) if max_start > 0 else 0)
                frame_indices = frame_indices[start : start + target_frames]

            # torchcodec returns [T,C,H,W] tensor
            frame_batch = decoder.get_frames_at(frame_indices)
            video_frames = frame_batch.data  # [T,C,H,W] uint8
            del decoder

            # Convert to [C,T,H,W] format expected by model
            video = video_frames.permute(1, 0, 2, 3)  # [T,C,H,W] -> [C,T,H,W]

            # Load camera data
            camera_data = json.loads(camera_bytes)

            # Special check for varying intrinsics (e.g., in synhuman_20260223 dataset)
            # If the intrinsic (fx, fy, cx, cy) changes during different frames, discard this sample.
            fl = np.array(camera_data["camera"]["focal_length"])
            pp = np.array(camera_data["camera"]["principal_point"])
            if self.discard_varying_intrinsics and (
                not np.allclose(fl, fl[0], atol=1e-5) or not np.allclose(pp, pp[0], atol=1e-5)
            ):
                log.warning(f"SKIPPED {uid}: varying intrinsics detected (discard_varying_intrinsics=True)")
                return None

            w2c = np.array(camera_data["camera"]["pose_world2cam"]).reshape(-1, 7)  # [N,7]
            if w2c.shape[0] <= frame_indices[-1]:
                log.warning(
                    f"SKIPPED {uid}: not enough camera poses "
                    f"(num_poses={w2c.shape[0]}, last_frame_idx={frame_indices[-1]})"
                )
                return None

            # Get w2c for the sampled frames
            w2c = w2c[frame_indices]  # [T,7]

            # Convert (qx,qy,qz,qw,tx,ty,tz) to [R|t] matrices
            w2c = Camera.extrinsic_params_to_matrices(w2c)  # [T,3,4]
            w2c_homo = np.eye(4, dtype=np.float32)[None, :, :].repeat(w2c.shape[0], axis=0)  # [T,4,4]
            w2c_homo[:, :3, :] = w2c
            c2w_homo = np.linalg.inv(w2c_homo)

            # Determine mode
            if self.mode == "joint":
                mode = random.choices(
                    ["forward_dynamics", "inverse_dynamics", "policy"],
                    weights=[0.8, 0.1, 0.1],
                    k=1,
                )[0]
            else:
                mode = self.mode

            action = pose_abs_to_rel(
                c2w_homo,
                rotation_format=self.rotation_format,
                pose_convention=self.pose_convention,
                translation_scale=self.translation_scale,
                rotation_scale=self.rotation_scale,
            )

            if self.max_action_translation_norm is not None and self.split == "train":
                trans_norms = np.linalg.norm(action[:, :3], axis=1)
                if trans_norms.max() > self.max_action_translation_norm:
                    log.warning(
                        f"SKIPPED {uid}: action translation norm too large "
                        f"(max={trans_norms.max():.4f}, threshold={self.max_action_translation_norm})"
                    )
                    return None

            action = torch.from_numpy(action)  # [num_frames,action_dim]

            # Load caption data
            if self.fix_caption:
                caption = self.fix_caption_text
            elif caption_override is not None:
                caption = caption_override
            else:
                metas_data = json.loads(metas_bytes)

                # Example: "t2w_windows": [{"end_frame": 150, "qwen2p5_7b_caption": "...", ...}]
                t2w_windows = metas_data["t2w_windows"]
                window = t2w_windows[0]
                caption_key = random.choices(self.caption_keys, weights=self.caption_weights, k=1)[0]
                caption = window[caption_key]

            # FPS
            fps = torch.tensor(video_fps, dtype=torch.float32)  # scalar

            # Select domain ID: use inverse domain for generation modes when mode_aware_domain is on
            if self.mode_aware_domain and mode in ["inverse_dynamics", "policy"]:
                domain_id = self.domain_id_inv
            else:
                domain_id = self.domain_id

            # Build sample dict
            sample = {
                "video": video,  # [C,T,H,W] uint8
                "action": action,  # [T-1,action_dim] float32
                "conditioning_fps": fps,  # scalar float32
                "ai_caption": caption,
                "mode": mode,
                "__key__": torch.tensor([hash(uid) % (2**31)], dtype=torch.long),
                "domain_id": torch.tensor(domain_id, dtype=torch.long),
                "viewpoint": "ego_view",
            }
            return sample

        except Exception as e:
            log.warning(f"Error processing sample {uid}: {e}")
            return None


# PYTHONPATH=. python cosmos/data/vfm/action/camera_dataset_sharded.py
if __name__ == "__main__":
    import time

    import torchvision

    dataset = CameraDatasetSharded(
        wdinfo_paths=[
            CAMERA_WDINFOS["pretrained_clips_260313_10s_100k_filtered"],
        ],
        split="train",
        shuffle=True,
        fix_caption=False,
        # wdinfo_resolution="gt720",
        # max_frames=200,
    )
    dataset_iter = iter(dataset)

    for i in range(10):
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
        print(f"{'caption':<25}: {data['ai_caption']}")

        # save video to local for debugging
        video = data["video"]
        video = video.permute(1, 0, 2, 3)  # [T,C,H,W]
        video_path = f"temp/camera_sample_{i}.mp4"
        torchvision.io.write_video(
            video_path, video.permute(0, 2, 3, 1).numpy(), fps=data["conditioning_fps"].item()
        )  # expects (T, H, W, C)
        print(f"Saved video to {video_path}")
