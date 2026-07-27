import os
import os.path as osp
import zipfile

import numpy as np
import pandas as pd
import portalocker
from huggingface_hub import snapshot_download
from PIL import Image

from vlmeval.smp import (dump, get_cache_path, get_file_extension, get_intermediate_file_path,
                         load, md5)
from vlmeval.utils import track_progress_rich
from .utils import build_judge
from .utils.qbench_video import (VQA_JUDGE_SYS_PROMPT, check_ans_mcq, check_ans_vqa,
                                 get_dimension_rating)
from .video_base import VideoBaseDataset
from .video_concat_dataset import ConcatVideoDataset

FAIL_MSG = 'Failed to obtain answer via API.'


class QBench_Video(ConcatVideoDataset):
    def __init__(self, dataset='QBench_Video', nframe=0, fps=-1):
        self.DATASET_SETS[dataset] = ['QBench_Video_MCQ', 'QBench_Video_VQA']
        super().__init__(dataset=dataset, nframe=nframe, fps=fps)

    @classmethod
    def supported_datasets(cls):
        return ['QBench_Video']

    def evaluate(self, eval_file, **judge_kwargs):
        result = super().evaluate(eval_file=eval_file, **judge_kwargs)
        score_file = get_intermediate_file_path(eval_file, '_acc')
        result.at['open_ended', 'acc'] /= 2
        dump(result, score_file)
        return result


class QBench_Video_MCQ(VideoBaseDataset):

    MD5 = '9d6760d75fa80aa9fd5e5cf1ea274ace'

    FRAMES_TMPL_SYS = """
You will receive {} distinct frames that have been uniformly sampled from a video sequence, arranged in the same temporal order as they appear in the video.
Please analyze these frames and answer the question based on your observations.
"""  # noqa: E501

    FRAMES_TMPL_SYS_4VIDEO_LLM = """
You will receive several distinct frames that have been uniformly sampled from a video sequence, arranged in the same temporal order as they appear in the video.
Please analyze these frames and answer the question based on your observations.
"""  # noqa: E501

    POST_PROMPT = """
Please answer the question in the following format: the uppercase letter of the correct answer option itself.
Please do not add any other answers beyond this.
"""

    TYPE = 'Video-MCQ'

    def __init__(self, dataset='qbenchvideo_single_MCQ', nframe=0, fps=-1):
        dataset_tsv_name = 'qbenchvideo_single_MCQ'
        super().__init__(dataset=dataset_tsv_name, nframe=nframe, fps=fps)

    @classmethod
    def supported_datasets(cls):
        return ['QBench_Video_MCQ']

    def prepare_dataset(self, dataset_name='qbenchvideo_single_MCQ', repo_id='zhangzicheng/Q-Bench-Video'):
        def check_integrity(pth):
            data_file = osp.join(pth, f'{dataset_name}.tsv')

            if not os.path.exists(data_file):
                return False

            if md5(data_file) != self.MD5:
                return False

            data = load(data_file)
            for idx, item in data.iterrows():
                if not osp.exists(os.path.normpath(osp.join(pth, item['video_path']))):
                    return False
            return True

        cache_path = get_cache_path(repo_id)
        if cache_path is not None and check_integrity(cache_path):
            dataset_path = cache_path
        else:
            def unzip_videos(pth):
                if not osp.exists(osp.join(pth, 'video')):
                    zip_file = osp.join(pth, 'video.zip')
                    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                        zip_ref.extractall(pth)
            dataset_path = snapshot_download(repo_id=repo_id, repo_type='dataset')
            unzip_videos(dataset_path)

        data_file = osp.join(dataset_path, f'{dataset_name}.tsv')
        return dict(root=dataset_path, data_file=data_file)

    def save_video_frames(self, line):
        video = line['video']
        vid_path = os.path.normpath(os.path.join(self.data_root, line['video_path']))
        import decord
        vid = decord.VideoReader(vid_path)
        video_info = {
            'fps': vid.get_avg_fps(),
            'n_frames': len(vid),
        }
        if self.nframe > 0 and self.fps < 0:
            step_size = len(vid) / (self.nframe + 1)
            indices = [int(i * step_size) for i in range(1, self.nframe + 1)]
            frame_paths = self.frame_paths(video)
        elif self.fps > 0:
            # not constrained by num_frames, get frames by fps
            total_duration = video_info['n_frames'] / video_info['fps']
            required_frames = int(total_duration * self.fps)
            step_size = video_info['fps'] / self.fps
            indices = [int(i * step_size) for i in range(required_frames)]
            frame_paths = self.frame_paths_fps(video, len(indices))

        flag = np.all([osp.exists(p) for p in frame_paths])

        if not flag:
            lock_path = osp.splitext(vid_path)[0] + '.lock'
            with portalocker.Lock(lock_path, 'w', timeout=30):
                if not np.all([osp.exists(p) for p in frame_paths]):
                    images = [vid[i].asnumpy() for i in indices]
                    images = [Image.fromarray(arr) for arr in images]
                    for im, pth in zip(images, frame_paths):
                        if not osp.exists(pth):
                            im.save(pth)

        return frame_paths

    def save_video_into_images(self, line):
        frame_paths = self.save_video_frames(line)
        return frame_paths

    def build_prompt(self, line, video_llm):
        if isinstance(line, int):
            assert line < len(self)
            line = self.data.iloc[line]

        # message = [dict(type='text', value=line['question'])]
        video_path = os.path.normpath(os.path.join(self.data_root, line['video_path']))
        if video_llm:
            message = [dict(type='text', value=self.FRAMES_TMPL_SYS_4VIDEO_LLM)]
            message.append(dict(type='text', value=line['question']))
            message.append(dict(type='video', value=video_path))
        else:
            img_frame_paths = self.save_video_into_images(line)
            message = [dict(type='text', value=self.FRAMES_TMPL_SYS.format(len(img_frame_paths)))]
            message.append(dict(type='text', value=line['question']))
            for im in img_frame_paths:
                message.append(dict(type='image', value=im))
        message.append(dict(type='text', value=self.POST_PROMPT))
        return message

    @classmethod
    def evaluate(self, eval_file, **judge_kwargs):
        assert get_file_extension(eval_file) in [
            'xlsx', 'json', 'tsv'
        ], 'data file should be an supported format (xlsx/json/tsv) file'

        tmp_file = get_intermediate_file_path(eval_file, '_tmp', 'pkl')
        score_file = get_intermediate_file_path(eval_file, '_score')

        if not osp.exists(score_file):
            # model = judge_kwargs.setdefault('model', 'exact_matching')

            res = {} if not osp.exists(tmp_file) else load(tmp_file)
            res = {k: v for k, v in res.items() if FAIL_MSG not in v}

            data = load(eval_file)
            data_un = data[~pd.isna(data['prediction'])]

            for idx in data['index']:
                ans = data.loc[data['index'] == idx, 'answer'].values[0]
                pred = data.loc[data['index'] == idx, 'prediction'].values[0]
                correct_choice = ans.split('.')[0].strip()
                correct_answer = ans.split('.')[1].strip()

                if FAIL_MSG in pred:
                    data.loc[idx, 'score'] = -1
                else:
                    data.loc[idx, 'score'] = int(check_ans_mcq(
                        pred, ans, correct_choice, correct_answer
                    ))

            rejected = [x for x in data['score'] if x == -1]

            print(
                f'Among {len(data)} questions, failed to obtain prediction for {len(data) - len(data_un)} questions, '
                f'failed to obtain the score for another {len(rejected)} questions. '
                f'Those questions will be counted as -1 score in ALL rating, and will not be counted in VALID rating.'
            )

            dump(data, score_file)

        rating = get_dimension_rating(score_file)
        return rating


class QBench_Video_VQA(VideoBaseDataset):

    MD5 = '49e6181b341c934d0b33ec78bdcc0a3d'

    FRAMES_TMPL_SYS = """
You will receive {} distinct frames that have been uniformly sampled from a video sequence, arranged in the same temporal order as they appear in the video.
Please analyze these frames and provide a detailed and accurate answer from the perspective of visual quality based on your observations.
"""  # noqa: E501

    FRAMES_TMPL_SYS_4VIDEO_LLM = """
You will receive several distinct frames that have been uniformly sampled from a video sequence, arranged in the same temporal order as they appear in the video.
Please analyze these frames and provide a detailed and accurate answer from the perspective of visual quality based on your observations.
"""  # noqa: E501

    TYPE = 'Video-VQA'
    DEFAULT_JUDGE = ['gpt-4o-0806', 'gpt-4o']

    def __init__(self, dataset='qbenchvideo_single_VQA', nframe=0, fps=-1):
        dataset_tsv_name = 'qbenchvideo_single_VQA'
        super().__init__(dataset=dataset_tsv_name, nframe=nframe, fps=fps)

    @classmethod
    def supported_datasets(cls):
        return ['QBench_Video_VQA']

    def prepare_dataset(self, dataset_name='qbenchvideo_single_VQA', repo_id='zhangzicheng/Q-Bench-Video'):
        def check_integrity(pth):
            data_file = osp.join(pth, f'{dataset_name}.tsv')

            if not os.path.exists(data_file):
                return False

            if md5(data_file) != self.MD5:
                return False

            data = load(data_file)
            for idx, item in data.iterrows():
                if not osp.exists(os.path.normpath(osp.join(pth, item['video_path']))):
                    return False
            return True

        cache_path = get_cache_path(repo_id)
        if cache_path is not None and check_integrity(cache_path):
            dataset_path = cache_path
        else:
            def unzip_videos(pth):
                if not osp.exists(osp.join(pth, 'video')):
                    zip_file = osp.join(pth, 'video.zip')
                    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                        zip_ref.extractall(pth)
            dataset_path = snapshot_download(repo_id=repo_id, repo_type='dataset')
            unzip_videos(dataset_path)

        data_file = osp.join(dataset_path, f'{dataset_name}.tsv')
        return dict(root=dataset_path, data_file=data_file)

    def save_video_frames(self, line):
        video = line['video']
        vid_path = os.path.normpath(os.path.join(self.data_root, line['video_path']))
        import decord
        vid = decord.VideoReader(vid_path)
        video_info = {
            'fps': vid.get_avg_fps(),
            'n_frames': len(vid),
        }
        if self.nframe > 0 and self.fps < 0:
            step_size = len(vid) / (self.nframe + 1)
            indices = [int(i * step_size) for i in range(1, self.nframe + 1)]
            frame_paths = self.frame_paths(video)
        elif self.fps > 0:
            # not constrained by num_frames, get frames by fps
            total_duration = video_info['n_frames'] / video_info['fps']
            required_frames = int(total_duration * self.fps)
            step_size = video_info['fps'] / self.fps
            indices = [int(i * step_size) for i in range(required_frames)]
            frame_paths = self.frame_paths_fps(video, len(indices))

        flag = np.all([osp.exists(p) for p in frame_paths])

        if not flag:
            lock_path = osp.splitext(vid_path)[0] + '.lock'
            with portalocker.Lock(lock_path, 'w', timeout=30):
                if not np.all([osp.exists(p) for p in frame_paths]):
                    images = [vid[i].asnumpy() for i in indices]
                    images = [Image.fromarray(arr) for arr in images]
                    for im, pth in zip(images, frame_paths):
                        if not osp.exists(pth):
                            im.save(pth)

        return frame_paths

    def save_video_into_images(self, line):
        frame_paths = self.save_video_frames(line)
        return frame_paths

    def build_prompt(self, line, video_llm):
        if isinstance(line, int):
            assert line < len(self)
            line = self.data.iloc[line]

        video_path = os.path.normpath(os.path.join(self.data_root, line['video_path']))
        if video_llm:
            message = [dict(type='text', value=self.FRAMES_TMPL_SYS_4VIDEO_LLM)]
            message.append(dict(type='text', value=line['question']))
            message.append(dict(type='video', value=video_path))
        else:
            img_frame_paths = self.save_video_into_images(line)
            message = [dict(type='text', value=self.FRAMES_TMPL_SYS.format(len(img_frame_paths)))]
            message.append(dict(type='text', value=line['question']))
            for im in img_frame_paths:
                message.append(dict(type='image', value=im))
        return message

    @classmethod
    def evaluate(self, eval_file, **judge_kwargs):
        model = judge_kwargs.setdefault('model', 'gpt-4o-0806')

        score_file = get_intermediate_file_path(eval_file, f'_{model}_score')
        tmp_file = get_intermediate_file_path(eval_file, f'_{model}', 'pkl')
        nproc = judge_kwargs.pop('nproc', 4)

        if not osp.exists(score_file):
            data = load(eval_file)
            model = build_judge(system_prompt=VQA_JUDGE_SYS_PROMPT, **judge_kwargs)
            lt = len(data)
            lines = [data.iloc[i] for i in range(lt)]
            tups = [(model, line) for line in lines]
            indices = [line['index'] for line in lines]

            ans = {}
            if osp.exists(tmp_file):
                ans = load(tmp_file)
            tups = [x for x, i in zip(tups, indices) if i not in ans]
            indices = [i for i in indices if i not in ans]

            if len(indices):
                _ = track_progress_rich(
                    check_ans_vqa,
                    tups,
                    nproc=nproc,
                    chunksize=nproc,
                    keys=indices,
                    save=tmp_file,
                )
            ans = load(tmp_file)
            for idx in ans:
                data.loc[data['index'] == idx, 'score'] = int(ans[idx].replace('Score:', '').strip())
            dump(data, score_file)

        rating = get_dimension_rating(score_file)
        return rating
