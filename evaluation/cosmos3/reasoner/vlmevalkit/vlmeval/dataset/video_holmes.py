import json
import os
import os.path as osp
import pickle
import warnings

import numpy as np
import pandas as pd
import portalocker
from huggingface_hub import snapshot_download
from PIL import Image

from vlmeval.smp import (dump, get_cache_path, get_file_extension, get_intermediate_file_path,
                         load, md5, modelscope_flag_set)
from .utils import DEBUG_MESSAGE, build_judge
from .video_base import VideoBaseDataset

FAIL_MSG = 'Failed to obtain answer via API.'


def unwrap_hf_pkl(pth, suffix='.mp4'):
    base_dir = os.path.join(pth, 'video_pkl/')
    target_dir = os.path.join(pth, 'video/')
    pickle_files = [os.path.join(base_dir, file) for file in os.listdir(base_dir)]
    pickle_files.sort()

    if not os.path.exists(target_dir):
        os.makedirs(target_dir, exist_ok=True)
        for pickle_file in pickle_files:
            with open(pickle_file, 'rb') as file:
                video_data = pickle.load(file)
            # For each video file in the pickle file, write its contents to a new mp4 file
            for video_name, video_content in video_data.items():
                output_path = os.path.join(target_dir, f'{video_name}{suffix}')
                with open(output_path, 'wb') as output_file:
                    output_file.write(video_content)
        print('The video file has been restored and stored from the pickle file.')
    else:
        print('The video file already exists.')


class Video_Holmes(VideoBaseDataset):

    MD5 = '85bdd91f9b29a99354c23b97ab7c113c'
    SYS = ''

    QUESTION_TMPL = """
    Based on the given video, reason and answer the single-choice question. Provide your reasoning between the <think> and </think> tags, and then give your final answer between the <answer> and </answer> tags. \
    The question is: {}. The options are: {}. \
    Your answer:
    """  # noqa: E501

    TYPE = 'Video-MCQ'
    DEFAULT_JUDGE = ['chatgpt-0125', 'gpt-4-0125']

    def __init__(self, dataset='Video_Holmes', nframe=32, fps=-1):
        super().__init__(dataset=dataset, nframe=nframe, fps=fps)
        self.dataset_name = dataset

    @classmethod
    def supported_datasets(cls):
        return ['Video_Holmes']

    def prepare_dataset(self, dataset_name='Video_Holmes', repo_id='TencentARC/Video-Holmes'):

        def check_integrity(pth):
            data_file = osp.join(pth, f'{dataset_name}.tsv')
            if not os.path.exists(data_file):
                return False

            if md5(data_file) != self.MD5:
                return False
            data = load(data_file)
            for video_pth in data['video_path']:
                if not osp.exists(osp.join(pth, video_pth)):
                    return False
            return True

        cache_path = get_cache_path(repo_id)
        if cache_path is not None and check_integrity(cache_path):
            dataset_path = cache_path
        else:
            def unzip_hf_zip(pth):
                import zipfile
                base_dir = pth
                target_dir = os.path.join(pth, 'video/')
                zip_files = [
                    os.path.join(base_dir, file) for file in os.listdir(base_dir)
                    if file == "videos.zip"
                ]
                zip_files.sort()

                if not os.path.exists(target_dir):
                    os.makedirs(target_dir, exist_ok=True)
                    for zip_file in zip_files:
                        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                            for member in zip_ref.namelist():
                                # Check if the member is a file (not a directory)
                                if not member.endswith('/'):
                                    # Extract the file to the specified directory
                                    source = zip_ref.open(member)
                                    target = open(os.path.join(target_dir, os.path.basename(member)), 'wb')
                                    with source, target:
                                        target.write(source.read())
                    print('The video file has been restored and stored from the zip file.')
                else:
                    print('The video file already exists.')

            def generate_tsv(pth):

                data_file = osp.join(pth, f'{dataset_name}.tsv')
                if os.path.exists(data_file) and md5(data_file) == self.MD5:
                    return

                with open(os.path.join(pth, 'test_Video-Holmes.json'), 'r', encoding='utf-8') as f:
                    data = json.load(f)

                rows = []

                for idx, item in enumerate(data):

                    video_id = item.get('video ID')
                    options = item.get('Options', {})
                    candidates = [f"{k}. {options.get(k, '')}".replace("'", "")
                                  for k in ['A', 'B', 'C', 'D', 'E', 'F'] if k in options]
                    row = {
                        'index': idx,
                        'video': video_id,
                        'video_path': f'./video/{video_id}.mp4',
                        'candidates': candidates,
                        'question': item.get('Question', ''),
                        'answer': item.get('Answer', ''),
                        'question_id': item.get('Question ID', ''),
                        'question_type': item.get('Question Type', ''),
                        'explanation': item.get('Explanation', ''),
                    }
                    rows.append(row)

                df = pd.DataFrame(rows)
                columns = ['index', 'video', 'video_path', 'candidates',
                           'question', 'answer', 'question_id', 'question_type', 'explanation']
                df = df[columns]
                df.to_csv(data_file, sep='\t', index=False)
                print("Generate tsv file OK")

            if modelscope_flag_set():
                from modelscope import dataset_snapshot_download
                dataset_path = dataset_snapshot_download(dataset_id=repo_id)
            else:
                dataset_path = snapshot_download(repo_id=repo_id, repo_type='dataset')

            unzip_hf_zip(dataset_path)
            generate_tsv(dataset_path)

        data_file = osp.join(dataset_path, f'{dataset_name}.tsv')
        return dict(data_file=data_file, root=dataset_path)

    def save_video_frames(self, video, video_llm=False):

        vid_path = osp.join(self.data_root, 'video', video + '.mp4')
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

        return frame_paths, indices, video_info

    def build_prompt(self, line, video_llm):

        if isinstance(line, int):
            assert line < len(self)
            line = self.data.iloc[line]

        frames, indices, video_info = self.save_video_frames(line['video'], video_llm)

        message = [dict(type='text', value=self.SYS)]

        if video_llm:
            message.append(dict(type='video', value=osp.join(self.data_root, 'video', line['video'] + '.mp4')))
        else:
            for im in frames:
                message.append(dict(type='image', value=im))

        text_prompt = self.QUESTION_TMPL.format(line['question'], line['candidates'])
        message.append(dict(type='text', value=text_prompt))
        # print(f"Build message OK {line[1]}")
        return message

    # It returns a dictionary
    @classmethod
    def evaluate(self, eval_file, **judge_kwargs):

        from .utils.videoholmes import extract_option, get_dimension_rating

        assert get_file_extension(eval_file) in ['xlsx', 'json', 'tsv'], 'data file should be an supported format (xlsx/json/tsv) file'  # noqa: E501

        tmp_file = get_intermediate_file_path(eval_file, '_tmp', 'pkl')
        tgt_file = get_intermediate_file_path(eval_file, '_rating', 'json')
        score_file = get_intermediate_file_path(eval_file, '_score')

        if not osp.exists(score_file):
            model = judge_kwargs.get('model', 'exact_matching')

            if model == 'exact_matching':
                model = None
            else:
                model = build_judge(**judge_kwargs)
                if not model.working():
                    warnings.warn('OPENAI API is not working properly, will use exact matching for evaluation')
                    warnings.warn(DEBUG_MESSAGE)
                    model = None
            res = {} if not osp.exists(tmp_file) else load(tmp_file)
            res = {k: v for k, v in res.items() if FAIL_MSG not in v}

            data = load(eval_file)
            data_un = data[~pd.isna(data['prediction'])]

            for idx in data['index']:
                ans = data.loc[data['index'] == idx, 'answer'].values[0]
                pred = str(data.loc[data['index'] == idx, 'prediction'].values[0])

                predicted_answer = extract_option(pred)

                data.loc[idx, 'score'] = int(predicted_answer == ans)

            rejected = [x for x in data['score'] if x == -1]

            print(
                f'Among {len(data)} questions, failed to obtain prediction for {len(data) - len(data_un)} questions, '
                f'failed to obtain the score for another {len(rejected)} questions. '
                f'Those questions will be counted as -1 score in ALL rating, and will not be counted in VALID rating.'
            )

            dump(data, score_file)

        rating = get_dimension_rating(score_file)
        dump(rating, tgt_file)
        return rating
