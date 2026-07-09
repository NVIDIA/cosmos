import ast
import os
import os.path as osp
import warnings

import numpy as np
import pandas as pd
import portalocker
from huggingface_hub import snapshot_download
from PIL import Image

from vlmeval.smp import (dump, get_cache_path, get_file_extension, get_intermediate_file_path,
                         load, md5)
from .video_base import VideoBaseDataset


def _parse_multi_choice_response(response, all_choices):
    response = response or ""
    answer_prefixes = [
        "The best answer is",
        "The correct answer is",
        "The answer is",
        "The answer",
        "The best option is",
        "The correct option is",
        "Best answer:",
        "Best option:",
        "Answer:",
        "Option:",
        "The correct answer",
        "The correct option",
        "Based",
        "Correct answer",
        "\u261e",
        "<|im_end|>",
    ]
    for prefix in answer_prefixes:
        response = response.replace(prefix, "")

    # Strip optional <think>...</think> and <answer>...</answer> wrappers if present.
    import re
    think_match = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
    if think_match:
        response = response.replace(think_match.group(0), "")

    match_pred = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
    if match_pred:
        response = match_pred.group(1)

    response = response.strip()
    response = re.sub(r"[.,:!\"'`;\\/?`~@#\$%\^&\*\(\)\[\]\{\}\\|<>\n]", " ", response)
    tokens = response.split()

    for token in tokens:
        if token in all_choices or token.upper() in all_choices:
            return token.upper()

    # Fallback: pick the first valid choice to avoid empty return
    return all_choices[0] if all_choices else ""


class AVSpeakerBench(VideoBaseDataset):

    # MD5 of the generated TSV (set to None to skip checking when unknown)
    MD5 = "803f732fbff54c0d1891532ffb0c3979"

    BASE_SYS = 'Carefully watch and listen to the clip. '
    SYS = BASE_SYS + 'Based on your observations, select the best option that accurately addresses the question.'

    AUDIO_VISUAL_TMPL = """
Select the best answer to the following multiple-choice question based on the audiovisual clip.
Respond with only the letter (A, B, C, or D) of the correct option.
"""

    VISUAL_TMPL = """
Select the best answer to the following multiple-choice question based on the silent visual clip.
Rely on the visuals only and respond with the letter (A, B, C, or D).
"""

    AUDIO_TMPL = """
Select the best answer to the following multiple-choice question based on the audio clip.
Focus on the audio and respond with only the letter (A, B, C, or D).
"""

    TYPE = 'Video-MCQ'

    def __init__(self, dataset='AV-SpeakerBench', use_audio=True, audio_only=False, nframe=0, fps=-1):
        self.use_audio = use_audio
        self.audio_only = audio_only
        self.dataset_name = dataset

        assert not (audio_only and not use_audio), 'audio_only requires use_audio=True.'

        super().__init__(dataset=dataset, nframe=nframe, fps=fps)

    @classmethod
    def supported_datasets(cls):
        return ['AV-SpeakerBench']

    def prepare_dataset(self, dataset_name='AV-SpeakerBench', repo_id='plnguyen2908/AV-SpeakerBench'):

        def check_integrity(pth):
            data_file = osp.join(pth, f'{dataset_name}.tsv')
            if not osp.exists(data_file):
                return False
            if self.MD5 and md5(data_file) != self.MD5:
                return False
            data = load(data_file)
            for col in ['audio_visual_path', 'visual_path', 'audio_path', 'video_path']:
                if col in data:
                    for media_path in data[col]:
                        if pd.isna(media_path) or media_path == '':
                            continue
                        if not osp.exists(osp.join(pth, media_path)):
                            return False
            return True

        def unzip_hf_zip(pth):
            import zipfile
            base_dir = pth
            zip_files = [
                os.path.join(base_dir, file) for file in os.listdir(base_dir)
                if file.endswith('.zip')
            ]
            if not zip_files:
                return

            for zip_file in sorted(zip_files):
                with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                    for member in zip_ref.namelist():
                        if member.endswith('/'):
                            continue
                        parts = member.split('/')
                        fname = parts[-1]
                        first_dir = parts[0] if len(parts) > 1 else ''
                        target_dir = os.path.join(base_dir, first_dir) if first_dir else base_dir
                        os.makedirs(target_dir, exist_ok=True)
                        target_path = os.path.join(target_dir, fname)
                        if osp.exists(target_path):
                            continue
                        with zip_ref.open(member) as source, open(target_path, 'wb') as target:
                            target.write(source.read())

        branch = "vlm_eval_version"  # or commit hash / tag
        dataset_path = snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            revision=branch,
        )
        cache_path = get_cache_path(repo_id, branch="vlm_eval_version")

        if cache_path is not None:
            unzip_hf_zip(cache_path)
            if check_integrity(cache_path):
                dataset_path = cache_path

        if dataset_path is None:
            dataset_path = snapshot_download(repo_id=repo_id, repo_type='dataset')
            unzip_hf_zip(dataset_path)
            if not check_integrity(dataset_path):
                warnings.warn('Dataset integrity check failed after download; media files may be missing.')

        data_file = osp.join(dataset_path, 'test.tsv')

        return dict(data_file=data_file, root=dataset_path)

    def save_video_frames(self, video_path, video_id):
        vid_path = video_path
        if not osp.isabs(vid_path):
            vid_path = osp.join(self.data_root, vid_path)
        import decord
        vid = decord.VideoReader(vid_path)
        video_info = {
            'fps': vid.get_avg_fps(),
            'n_frames': len(vid),
        }
        if self.nframe > 0 and self.fps < 0:
            step_size = len(vid) / (self.nframe + 1)
            indices = [int(i * step_size) for i in range(1, self.nframe + 1)]
            frame_paths = self.frame_paths(video_id)
        elif self.fps > 0:
            total_duration = video_info['n_frames'] / video_info['fps']
            required_frames = int(total_duration * self.fps)
            step_size = video_info['fps'] / self.fps
            indices = [int(i * step_size) for i in range(required_frames)]
            frame_paths = self.frame_paths_fps(video_id, len(indices))

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

        if self.use_audio:
            video_path = line.get('audio_visual_path')
        else:
            video_path = line.get('visual_path')
        audio_path = line.get('audio_path')

        if not self.audio_only and not video_llm:
            frames, _, _ = self.save_video_frames(video_path, line['video'])
        else:
            frames = []

        message = [dict(type='text', value=self.SYS)]

        if not self.audio_only:
            if video_llm:
                message.append(
                    dict(
                        type='video',
                        value=osp.join(self.data_root, video_path)
                    )
                )
            else:
                for im in frames:
                    message.append(
                        dict(
                            type='image',
                            value=im
                        )
                    )

        else:
            message.append(dict(type='audio', value=osp.join(self.data_root, audio_path)))

        if self.audio_only:
            text_prompt = self.AUDIO_TMPL
        elif self.use_audio:
            text_prompt = self.AUDIO_VISUAL_TMPL
        else:
            text_prompt = self.VISUAL_TMPL
        message.append(dict(type='text', value=text_prompt))

        raw_choices = line.get('choices')
        if isinstance(raw_choices, str):
            try:
                choices = ast.literal_eval(raw_choices)
            except Exception:
                choices = raw_choices.split('\n')
        else:
            choices = list(raw_choices) if raw_choices is not None else []

        question_str = str(line['question']) + '\n' + '\n'.join(choices)
        prompt = f'{question_str}\nThe best answer is:'
        message.append(dict(type='text', value=prompt))
        return message

    # It returns a dictionary
    @classmethod
    def evaluate(self, eval_file, **judge_kwargs):
        assert get_file_extension(eval_file) in ['xlsx', 'json', 'tsv'], \
            'data file should be an supported format (xlsx/json/tsv) file'

        score_file = get_intermediate_file_path(eval_file, '_score')
        tgt_file = get_intermediate_file_path(eval_file, '_rating', 'json')

        if not osp.exists(score_file):
            data = load(eval_file)

            cnt_missing_pred = 0
            cnt_rejected = 0

            task_scores = {}

            for _, row in data.iterrows():
                if pd.isna(row.get('prediction', None)):
                    data.loc[data['index'] == row['index'], 'score'] = 0
                    cnt_missing_pred += 1
                    continue

                ans = str(row.get('answer', '')).strip().upper()
                raw_pred = str(row.get('prediction', ''))

                pred_label = _parse_multi_choice_response(raw_pred, ['A', 'B', 'C', 'D'])
                if pred_label == '':
                    cnt_rejected += 1
                    data.loc[data['index'] == row['index'], 'score'] = 0
                else:
                    data.loc[data['index'] == row['index'], 'score'] = int(pred_label == ans)

            valid = data[data['score'] >= 0]
            if len(valid):
                for task_id, group in valid.groupby('task_id') if 'task_id' in valid else []:
                    task_scores[str(task_id)] = float(group['score'].mean() * 100)
                overall = float(valid['score'].mean() * 100)
            else:
                overall = 0.0

            print(
                f'Among {len(data)} questions, failed to obtain prediction for {cnt_missing_pred} questions, '
                f'failed to parse prediction for another {cnt_rejected} questions.'
            )

            dump(data, score_file)

            rating = {'overall': overall}
            if len(task_scores):
                rating['by_task'] = task_scores
            dump(rating, tgt_file)
        else:
            rating = load(tgt_file)

        return rating
