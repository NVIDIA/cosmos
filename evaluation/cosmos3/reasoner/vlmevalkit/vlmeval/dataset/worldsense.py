import json
import os
import os.path as osp
import warnings

import numpy as np
import pandas as pd
import portalocker
from huggingface_hub import snapshot_download
from PIL import Image

from vlmeval.smp import (dump, get_cache_path, get_file_extension, get_intermediate_file_path,
                         get_logger, load, md5, modelscope_flag_set)
from .utils import DEBUG_MESSAGE, build_judge
from .video_base import VideoBaseDataset

logger = get_logger(__name__)

FAIL_MSG = 'Failed to obtain answer via API.'


class WorldSense(VideoBaseDataset):

    MD5 = 'bfc25490be4080aa5494b883370b6b1f'

    BASE_SYS = 'Carefully watch this video and pay attention to every detail. '
    SYS = BASE_SYS + 'Based on your observations, select the best option that accurately addresses the question.'

    FRAMES_TMPL_NOSUB = """
These are the frames of a video. \
Select the best answer to the following multiple-choice question based on the video. \
Respond with only the letter (A, B, C, or D) of the correct option.
"""

    FRAMES_TMPL_SUB = """
These are the frames of a video. \
This video's subtitles are listed below:
{}
Select the best answer to the following multiple-choice question based on the video. \
Respond with only the letter (A, B, C, or D) of the correct option.
"""

    FRAMES_TMPL_AUDIO = """
These are the frames of a video and the corresponding audio. \
Select the best answer to the following multiple-choice question based on the video. \
Respond with only the letter (A, B, C, or D) of the correct option.
"""

    TYPE = 'Video-MCQ'
    DEFAULT_JUDGE = ['chatgpt-0125', 'gpt-4-0125']

    def __init__(self, dataset='WorldSense', use_subtitle=False, use_audio=False, nframe=0, fps=-1):
        super().__init__(dataset=dataset, nframe=nframe, fps=fps)
        self.use_subtitle = use_subtitle
        self.use_audio = use_audio
        self.dataset_name = dataset

        assert not (self.use_subtitle and self.use_audio), 'Cannot use both subtitle and audio at the same time.'

    @classmethod
    def supported_datasets(cls):
        return ['WorldSense']

    def prepare_dataset(self, dataset_name='WorldSense', repo_id='honglyhly/WorldSense'):

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

                from moviepy.editor import VideoFileClip
                base_dir = pth
                target_dir = os.path.join(pth, 'videos/')
                zip_files = [
                    os.path.join(base_dir, file) for file in os.listdir(base_dir)
                    if file.endswith('.zip') and file.startswith('worldsense_videos')
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

                subtitle_zip_file = os.path.join(base_dir, 'worldsense_subtitles.zip')
                subtitle_target_dir = os.path.join(base_dir, 'subtitles')

                if not os.path.exists(subtitle_target_dir):
                    os.makedirs(subtitle_target_dir, exist_ok=True)
                    with zipfile.ZipFile(subtitle_zip_file, 'r') as zip_ref:
                        for member in zip_ref.namelist():
                            # Check if the member is a file (not a directory)
                            if not member.endswith('/'):
                                # Extract the file to the specified directory
                                source = zip_ref.open(member)
                                target = open(os.path.join(subtitle_target_dir, os.path.basename(member)), 'wb')
                                with source, target:
                                    target.write(source.read())
                    print('The subtitle file has been restored and stored from the zip file.')
                else:
                    print('The subtitle file already exists.')

                audio_target_dir = os.path.join(base_dir, 'audios')
                if not os.path.exists(audio_target_dir):
                    os.makedirs(audio_target_dir, exist_ok=True)
                    videos = os.listdir(target_dir)
                    for video in videos:
                        video_path = os.path.join(target_dir, video)
                        audio_path = os.path.join(audio_target_dir, video.replace('.mp4', '.wav'))
                        video = VideoFileClip(video_path)
                        audio = video.audio
                        audio.write_audiofile(audio_path, verbose=False, logger=None)
                        video.close()
                        audio.close()
                    print('The audio file has been extracted from videos.')
                else:
                    print('The audio file already exists.')

            def generate_tsv(pth):

                data_file = osp.join(pth, f'{dataset_name}.tsv')
                print(data_file, md5(data_file))
                if os.path.exists(data_file) and md5(data_file) == self.MD5:
                    return

                with open(osp.join(pth, 'worldsense_qa.json'), 'rb') as file:
                    json_data = json.load(file)

                videos = list(json_data.keys())
                qa_index = 0
                data_list = []
                data_list.append([
                    'index', 'video', 'video_path', 'duration', 'domain', 'candidates',
                    'sub_category', 'audio_class', 'task_domain', 'task_type', 'subtitle_path',
                    'audio_path', 'video_caption', 'question', 'answer'
                ])
                for video in videos:
                    video_data = json_data[video]
                    tasks_data = list(video_data.keys())
                    tasks_data = [task for task in tasks_data if 'task' in task]
                    for _task in tasks_data:
                        task_list = []
                        _task_data = video_data[_task]
                        task_list.append(qa_index)
                        task_list.append(str(video))
                        task_list.append(f'./videos/{video}.mp4')
                        task_list.append(video_data['duration'])
                        task_list.append(video_data['domain'])
                        task_list.append(_task_data['candidates'])
                        task_list.append(video_data['sub_category'])
                        task_list.append(video_data['audio_class'])
                        task_list.append(_task_data['task_domain'])
                        task_list.append(_task_data['task_type'])
                        task_list.append(f'./subtitles/{video}.srt')
                        task_list.append(f'./audios/{video}.wav')
                        task_list.append(video_data['video_caption'])
                        task_list.append(_task_data['question'])
                        task_list.append(_task_data['answer'])

                        data_list.append(task_list)
                        qa_index += 1

                data_file = data_list
                data_file = pd.DataFrame(data_file[1:], columns=data_file[0])

                data_file.to_csv(osp.join(pth, f'{dataset_name}.tsv'), sep='\t', index=False)

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

        vid_path = osp.join(self.data_root, 'videos', video + '.mp4')
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

        if self.use_subtitle and os.path.exists(osp.join(self.data_root, line['subtitle_path'])):
            import pysubs2
            subs = pysubs2.load(osp.join(self.data_root, line['subtitle_path']), encoding='utf-8')
            subtitles = []

            if video_llm:
                n_frame_list = list(range(0, video_info['n_frames'], 1))
                indices = n_frame_list[0:-1:int(video_info['fps'])]
            for seleced_frame_id in indices:
                sub_text = ''
                cur_time = pysubs2.make_time(fps=video_info['fps'], frames=seleced_frame_id)
                for sub in subs:
                    if sub.start < cur_time and sub.end > cur_time:
                        sub_text = sub.text.replace('\\N', ' ')
                        break
                if sub_text.strip():
                    subtitles.append(sub_text)
            subtitles = '\n'.join(subtitles)
        else:
            subtitles = ''

        message = [dict(type='text', value=self.SYS)]
        if video_llm:
            message.append(dict(type='video', value=osp.join(self.data_root, 'videos', line['video'] + '.mp4')))
            if self.use_audio:
                message.append(dict(type='audio', value=osp.join(self.data_root, 'audios', line['video'] + '.wav')))
        else:
            for im in frames:
                message.append(dict(type='image', value=im))
            if self.use_audio:
                message.append(dict(type='audio', value=osp.join(self.data_root, 'audios', line['video'] + '.wav')))

        if self.use_subtitle:
            text_prompt = self.FRAMES_TMPL_SUB.format(subtitles)
        elif self.use_audio:
            text_prompt = self.FRAMES_TMPL_AUDIO
        else:
            text_prompt = self.FRAMES_TMPL_NOSUB
        message.append(dict(type='text', value=text_prompt))
        question_str = line['question'] + '\n' + '\n'.join(eval(line['candidates']))
        prompt = 'Question: {}\nAnswer: '.format(question_str)
        message.append(dict(type='text', value=prompt))
        return message

    # It returns a dictionary
    @classmethod
    def evaluate(self, eval_file, **judge_kwargs):
        from .utils.worldsense import (extract_characters_regex, extract_option,
                                       get_dimension_rating)

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

                if extract_characters_regex(pred) == '':
                    extract_pred = extract_option(
                        model,
                        data.loc[data['index'] == idx].to_dict(orient='records')[0],
                        'WorldSense'
                    )
                    data.loc[idx, 'score'] = int(extract_pred == ans)
                else:
                    data.loc[idx, 'score'] = int(extract_characters_regex(pred) == ans)

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
