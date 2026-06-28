import os
import os.path as osp
import warnings
from abc import abstractmethod

import numpy as np
import portalocker
from PIL import Image

from vlmeval.smp import download_file, file_size, load, md5
from vlmeval.smp.file import LMUDataRoot
from vlmeval.smp.log import get_logger

logger = get_logger(__name__)


class VideoBaseDataset:

    MODALITY = 'VIDEO'

    def __init__(self,
                 dataset='MMBench-Video',
                 pack=False,
                 nframe=0,
                 fps=-1,
                 total_pixels=None,
                 **kwargs):
        # **kwargs absorbs profile-level fields (e.g. model_family) that
        # subclasses didn't consume in their own __init__; harmless here.
        try:
            import decord  # noqa: F401
        except Exception as e:
            logger.critical(f'{type(e)}: {e}')
            logger.critical('Please install decord via `pip install decord`.')

        self.dataset_name = dataset
        ret = self.prepare_dataset(dataset)
        assert ret is not None
        lmu_root = LMUDataRoot()
        self.frame_root = osp.join(lmu_root, 'images', dataset)
        os.makedirs(self.frame_root, exist_ok=True)
        self.frame_tmpl = 'frame-{}-of-{}.jpg'
        self.frame_tmpl_fps = 'frame-{}-of-{}-{}fps.jpg'

        self.data_root = ret['root']
        self.data_file = ret['data_file']
        self.data = load(self.data_file)
        if 'index' not in self.data:
            self.data['index'] = np.arange(len(self.data))

        assert 'question' in self.data and 'video' in self.data
        videos = list(set(self.data['video']))
        videos.sort()
        self.videos = videos
        self.pack = pack
        self.nframe = nframe
        self.fps = fps
        self.total_pixels = total_pixels

        if self.fps > 0 and self.nframe > 0:
            raise ValueError('fps and nframe should not be set at the same time')
        if self.fps <= 0 and self.nframe <= 0:
            logger.warning(
                'Neither fps nor nframe is set for video dataset. '
                'This is fine for VIDEO_LLM models (model-side processing), '
                'but will fail for non-VIDEO_LLM models that need dataset-side frame extraction.'
            )

    def __len__(self):
        return len(self.videos) if self.pack else len(self.data)

    def __getitem__(self, idx):
        if self.pack:
            assert idx < len(self.videos)
            sub_data = self.data[self.data['video'] == self.videos[idx]]
            return sub_data
        else:
            assert idx < len(self.data)
            return dict(self.data.iloc[idx])

    def frame_paths(self, video):
        frame_root = osp.join(self.frame_root, video)
        os.makedirs(frame_root, exist_ok=True)
        return [osp.join(frame_root, self.frame_tmpl.format(i, self.nframe)) for i in range(1, self.nframe + 1)]

    def frame_paths_fps(self, video, num_frames):
        frame_root = osp.join(self.frame_root, video)
        os.makedirs(frame_root, exist_ok=True)
        return [osp.join(frame_root,
                         self.frame_tmpl_fps.format(i, num_frames, self.fps)) for i in range(1, num_frames + 1)]

    def save_video_frames(self, video):
        import decord
        if self.fps > 0:
            vid_path = osp.join(self.data_root, video + '.mp4')

            # First, check if frames already exist (quick check without opening video)
            # We need to estimate the number of frames first
            lock_path = osp.join(self.frame_root, video + '.lock')

            # Try to get a quick estimate or use a reasonable default
            # We'll do a more thorough check inside the lock
            with portalocker.Lock(lock_path, 'w', timeout=300):
                # Re-check inside lock to avoid race condition
                # Open video INSIDE the lock to prevent concurrent decord access
                vid = decord.VideoReader(vid_path)

                # 计算视频的总帧数和总时长
                total_frames = len(vid)
                video_fps = vid.get_avg_fps()
                total_duration = total_frames / video_fps

                # 计算需要提取的总帧数
                required_frames = int(total_duration * self.fps)

                # 计算提取帧的间隔
                step_size = video_fps / self.fps

                # 计算提取帧的索引
                indices = [int(i * step_size) for i in range(required_frames)]

                # 提取帧并保存
                frame_paths = self.frame_paths_fps(video, len(indices))

                # Check if all frames already exist
                if np.all([osp.exists(p) for p in frame_paths]):
                    return frame_paths

                images = [vid[i].asnumpy() for i in indices]
                images = [Image.fromarray(arr) for arr in images]

                # Apply total_pixels constraint if set
                if self.total_pixels is not None and len(images) > 0:
                    images = self._resize_frames_to_fit_pixels(images, self.total_pixels)

                for im, pth in zip(images, frame_paths):
                    if not osp.exists(pth):
                        im.save(pth)
            return frame_paths

        else:
            frame_paths = self.frame_paths(video)
            flag = np.all([osp.exists(p) for p in frame_paths])
            if flag:
                return frame_paths
            lock_path = osp.join(self.frame_root, video + '.lock')
            with portalocker.Lock(lock_path, 'w', timeout=300):
                if np.all([osp.exists(p) for p in frame_paths]):
                    return frame_paths
                vid_path = osp.join(self.data_root, video + '.mp4')
                vid = decord.VideoReader(vid_path)
                step_size = len(vid) / (self.nframe + 1)
                indices = [int(i * step_size) for i in range(1, self.nframe + 1)]
                images = [vid[i].asnumpy() for i in indices]
                images = [Image.fromarray(arr) for arr in images]

                # Apply total_pixels constraint if set
                if self.total_pixels is not None and len(images) > 0:
                    images = self._resize_frames_to_fit_pixels(images, self.total_pixels)

                for im, pth in zip(images, frame_paths):
                    if not osp.exists(pth):
                        im.save(pth)
            return frame_paths

    def _resize_frames_to_fit_pixels(self, images, total_pixels):
        """
        Resize frames to fit within the total_pixels constraint.

        Args:
            images: List of PIL Image objects
            total_pixels: Maximum total pixels across all frames

        Returns:
            List of resized PIL Image objects
        """
        import math

        if len(images) == 0:
            return images

        # Calculate pixels per frame budget
        pixels_per_frame = total_pixels / len(images)

        resized_images = []
        for img in images:
            width, height = img.size
            current_pixels = width * height

            # Only resize if current pixels exceed budget
            if current_pixels > pixels_per_frame:
                # Calculate scale factor to fit within budget
                scale = math.sqrt(pixels_per_frame / current_pixels)
                new_width = int(width * scale)
                new_height = int(height * scale)

                # Ensure dimensions are at least 1
                new_width = max(1, new_width)
                new_height = max(1, new_height)

                img = img.resize((new_width, new_height), Image.LANCZOS)

            resized_images.append(img)

        return resized_images

    # Return a list of dataset names that are supported by this class, can override
    @classmethod
    def supported_datasets(cls):
        return ['MMBench-Video', 'Video-MME', 'MVBench', 'MVBench_MP4',
                'LongVideoBench', 'WorldSense', 'VDC', 'MovieChat1k', 'AV-SpeakerBench']

    # Given the prediction file, return the evaluation results in the format of a dictionary or pandas dataframe
    @abstractmethod
    def evaluate(self, eval_file, **judge_kwargs):
        pass

    @abstractmethod
    def build_prompt(self, idx):
        pass

    @abstractmethod
    def prepare_dataset(self, dataset):
        # The prepare_dataset function should return a dictionary containing:
        # `root` (directory that containing video files)
        # `data_file` (the TSV dataset file)
        pass

    def prepare_tsv(self, url, file_md5=None):
        data_root = LMUDataRoot()
        os.makedirs(data_root, exist_ok=True)
        update_flag = False
        file_name_legacy = url.split('/')[-1]
        file_name = f"{self.dataset_name}.tsv"
        data_path_legacy = os.path.join(data_root, file_name_legacy)
        data_path = os.path.join(data_root, file_name)

        self.data_path = data_path
        if os.path.exists(data_path):
            if file_md5 is None or md5(data_path) == file_md5:
                pass
            else:
                warnings.warn(f'The tsv file is in {data_root}, but the md5 does not match, will re-download')
                download_file(url, data_path)
                update_flag = True
        else:
            if os.path.exists(data_path_legacy) and (file_md5 is None or md5(data_path_legacy) == file_md5):
                warnings.warn(
                    'Due to a modification in #1055, the local target file name has changed. '
                    f'We detected the tsv file with legacy name {data_path_legacy} exists and will do the rename. '
                )
                import shutil
                shutil.move(data_path_legacy, data_path)
            else:
                download_file(url, data_path)
                update_flag = True

        if file_size(data_path, 'GB') > 1:
            local_path = data_path.replace('.tsv', '_local.tsv')
            if not os.path.exists(local_path) or os.environ.get('FORCE_LOCAL', None) or update_flag:
                from ..tools import LOCALIZE
                LOCALIZE(data_path, local_path)
            data_path = local_path
        return load(data_path)
