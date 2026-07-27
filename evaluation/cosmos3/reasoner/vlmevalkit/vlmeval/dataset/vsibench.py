import ast
import fnmatch
import os
import pickle
import re
import warnings
from collections import OrderedDict

import decord
import numpy as np
import pandas as pd
from huggingface_hub import snapshot_download
from PIL import Image
from tqdm import tqdm

from vlmeval.smp import LMUDataRoot, get_cache_path, load, modelscope_flag_set
from .video_base import VideoBaseDataset


class VsiBench(VideoBaseDataset):
    """
    VSI-Bench.

    Reference:
      Thinking in Space: How Multimodal Large Language Models See, Remember, and Recall Spaces
      https://arxiv.org/abs/2412.14171
    """

    MD5 = ''
    TYPE = 'MCQ'
    MODALITY = 'VIDEO'

    OFFICAL_PRE_PROMPT = 'These are frames of a video.'
    OFFICAL_MCQ_POST_PROMPT = "Answer with the option's letter from the given choices directly."
    OFFICAL_VQA_POST_PROMPT = 'Answer briefly and directly in one float number.'

    LMUData_root = LMUDataRoot()

    DATASET_URL = {
        'VSI-Bench': 'https://huggingface.co/datasets/lmms-lab-si/EASI-Leaderboard-Data/resolve/main/VSI-Bench.tsv',  # noqa: E501
        'VSI-Bench-Debiased': 'https://huggingface.co/datasets/lmms-lab-si/EASI-Leaderboard-Data/resolve/main/VSI-Bench-Debiased.tsv',  # noqa: E501
    }
    DATASET_MD5 = {
        'VSI-Bench': '34544fd83241391d83eff087a1be7d83',
        'VSI-Bench-Debiased': '2a075fbc69a7725fe7f0718eafb7fca5',
    }

    def __init__(self, dataset, pack=False, nframe=0, fps=-1, sample_strategy='uniform_tail'):
        self.sample_strategy = sample_strategy
        valid_strategies = {'uniform_tail', 'uniform', 'chunk_center'}
        if sample_strategy not in valid_strategies:
            raise ValueError(f'[{dataset}] Unsupported sample_strategy \'{sample_strategy}\'')

        super().__init__(dataset=dataset, pack=pack, nframe=nframe, fps=fps)

    @classmethod
    def supported_datasets(cls):
        subsets = ['VSI-Bench', 'VSI-Bench-Debiased']
        return subsets

    def get_task_type(self, question_type):
        mcq_items = [
            'obj_appearance_order',
            'object_rel_direction_easy',
            'object_rel_direction_hard',
            'object_rel_direction_medium',
            'object_rel_distance',
            'route_planning',
        ]

        na_items = [
            'object_abs_distance',
            'object_size_estimation',
            'object_counting',
            'room_size_estimation',
        ]

        if question_type in mcq_items:
            return 'MCQ'
        elif question_type in na_items:
            return 'NA'
        else:
            raise ValueError(f'Unknown question type: {question_type}')

    def download_vsibench(self, repo_id='nyu-visionx/VSI-Bench'):
        cache_path = get_cache_path(repo_id)
        SENTINEL_NAME = '.vsibench_extracted'

        if (cache_path and os.path.isdir(cache_path)
                and os.path.isfile(os.path.join(cache_path, SENTINEL_NAME))):
            dataset_path = cache_path
        else:
            def _write_sentinel(sentinel_path, text='ok'):
                tmp = sentinel_path + '.tmp'
                with open(tmp, 'w', encoding='utf-8') as f:
                    f.write(text)
                os.replace(tmp, sentinel_path)

            def unzip_hf_zip(pth):
                import zipfile

                base_dir = pth
                zip_files = [
                    os.path.join(base_dir, f) for f in os.listdir(base_dir)
                    if f.endswith('.zip')
                ]
                zip_files.sort()

                for zip_file in tqdm(zip_files, desc='Unpacking Origin Data...'):
                    with zipfile.ZipFile(zip_file, 'r') as zf:
                        for info in zf.infolist():
                            if info.is_dir():
                                continue

                            rel = os.path.normpath(info.filename).lstrip('/\\')
                            dst = os.path.join(pth, rel)

                            absp = os.path.abspath(pth)
                            absd = os.path.abspath(dst)
                            if not absd.startswith(absp + os.sep):
                                raise RuntimeError(f'Unsafe path in zip: {info.filename}')

                            os.makedirs(os.path.dirname(dst), exist_ok=True)
                            with zf.open(info, 'r') as src, open(dst, 'wb') as out:
                                out.write(src.read())

                sentinel_path = os.path.join(pth, SENTINEL_NAME)
                _write_sentinel(sentinel_path, text='done')
                print('VsiBench data extracted to current directory with original layout.')

            if modelscope_flag_set():
                from modelscope import dataset_snapshot_download
                dataset_path = dataset_snapshot_download(dataset_id=repo_id)
            else:
                dataset_path = snapshot_download(repo_id=repo_id, repo_type='dataset')

            unzip_hf_zip(dataset_path)

        return dataset_path

    def prepare_dataset(self, dataset_name):
        url = self.DATASET_URL[dataset_name]
        md5 = self.DATASET_MD5[dataset_name]

        _ = super().prepare_tsv(url, md5)

        dataset_path = self.download_vsibench()
        self.dataset_path = dataset_path

        variant_data_file = os.path.join(self.LMUData_root, f'{dataset_name}.tsv')

        return dict(data_file=variant_data_file, root=dataset_path)

    def save_video_frames(self, video_path, video_llm=False):
        vid_path = os.path.join(self.data_root, video_path)

        vid = decord.VideoReader(vid_path)
        video_nframes = len(vid)
        video_fps = vid.get_avg_fps()
        video_info = {
            'fps': video_fps,
            'n_frames': video_nframes,
        }

        indices = []

        if self.nframe > 0 and self.fps < 0:
            if self.sample_strategy == 'uniform_tail':
                indices = np.linspace(0, video_nframes - 1, self.nframe, dtype=int).tolist()
                if (video_nframes - 1) != indices[-1]:
                    indices.append(video_nframes - 1)

            elif self.sample_strategy == 'uniform':
                indices = np.linspace(0, video_nframes - 1, self.nframe, dtype=int).tolist()

            elif self.sample_strategy == 'chunk_center':
                seg_size = float(video_nframes) / self.nframe
                indices = [int((seg_size / 2) + np.round(seg_size * i)) for i in range(self.nframe)]
                indices = np.clip(indices, 0, video_nframes - 1).tolist()

            else:
                raise ValueError(f'Unsupported sample strategy: {self.sample_strategy}')

            frame_paths = self.frame_paths(video_path)

        elif self.fps > 0:
            total_duration = video_nframes / video_fps
            required_frames = int(total_duration * self.fps)
            step_size = video_fps / self.fps
            indices = [int(i * step_size) for i in range(required_frames)]
            if self.sample_strategy == 'uniform_tail' and (video_nframes - 1) != indices[-1]:
                indices.append(video_nframes - 1)

            frame_paths = self.frame_paths_fps(video_path, len(indices))

        flag = np.all([os.path.exists(p) for p in frame_paths])

        if not flag:
            images = [vid[i].asnumpy() for i in indices]
            images = [Image.fromarray(arr) for arr in images]
            for im, pth in zip(images, frame_paths):
                if not os.path.exists(pth) and not video_llm:
                    im.save(pth)

        return frame_paths, indices, video_info

    def build_prompt(self, line, video_llm):
        if isinstance(line, int):
            assert line < len(self)
            line = self.data.iloc[line]

        question = line['question']
        question_type = str(line['question_type'])
        task_type = self.get_task_type(question_type)

        if task_type == 'MCQ':
            options = ast.literal_eval(line['candidates'])
            formatted_options = '\n'.join(options)

        # following VSI prompt format
        prompt_lst = [self.OFFICAL_PRE_PROMPT, question]
        if task_type == 'MCQ':
            prompt_lst += [formatted_options, self.OFFICAL_MCQ_POST_PROMPT]
        else:
            prompt_lst += [self.OFFICAL_VQA_POST_PROMPT]
        prompt = '\n'.join(prompt_lst)

        message = []

        if video_llm:
            message.append(dict(type='video', value=os.path.join(self.data_root, line['video'])))
        else:
            frames, _, _ = self.save_video_frames(line['video'], video_llm)
            for im in frames:
                message.append(dict(type='image', value=im))

        message.append(dict(type='text', value=prompt))

        return message

    def evaluate(self, eval_file, **judge_kwargs):
        from .utils.spatial_bench.cal_scores import (attach_score_cache, build_mcq_score_fn,
                                                     build_na_score_fn)
        from .utils.spatial_bench.tools.files import build_eval_paths, get_judge_tag_from_score_fn

        data = load(eval_file)
        data = data.sort_values(by='index')
        data['prediction'] = [str(x) for x in data['prediction']]

        data['task_type'] = data['question_type'].apply(self.get_task_type)
        mcq_data = data[data['task_type'] == 'MCQ'].copy()
        na_data = data[data['task_type'] == 'NA'].copy()

        splits = {'mcq': mcq_data, 'na': na_data}
        builders = {'mcq': build_mcq_score_fn, 'na': build_na_score_fn}

        # 1. build score_fns
        score_fns = {
            k: (builders[k](**judge_kwargs) if len(splits[k]) else None)
            for k in ('mcq', 'na')
        }

        # 2. pick a non-None score_fn to infer judge_tag
        score_fn_for_tag = score_fns.get('mcq') or score_fns.get('na')
        judge_tag = (
            get_judge_tag_from_score_fn(score_fn_for_tag)
            if score_fn_for_tag is not None
            else 'extract_matching'
        )
        result_file, xlsx_path, acc_tsv_path = build_eval_paths(eval_file, judge_tag)

        # 3. attach cache files for each sub-scorer
        for sub_tag, fn in score_fns.items():
            attach_score_cache(
                score_fn=fn,
                eval_file=eval_file,
                judge_tag=judge_tag,
                key_col='index',
                sub_tag=sub_tag,
            )

        # 4. run scoring
        mcq_scored = score_fns['mcq'](mcq_data) if score_fns['mcq'] else mcq_data
        na_scored = score_fns['na'](na_data) if score_fns['na'] else na_data

        summary = self._aggregate(mcq_scored, na_scored)

        try:
            to_dump = {
                'mcq_scored': mcq_scored,
                'na_scored': na_scored,
                'summary': summary,
            }
            with open(result_file, 'wb') as f:
                pickle.dump(to_dump, f)
            print(f'[save] result saved to {result_file}')
        except Exception as e:
            warnings.warn(f'[save] failed to save result to {result_file}: {e}')

        try:
            import pandas as pd

            frames = []
            if len(mcq_scored):
                df_mcq = mcq_scored.copy()
                df_mcq['task_type'] = 'MCQ'
                frames.append(df_mcq)
            if len(na_scored):
                df_na = na_scored.copy()
                df_na['task_type'] = 'NA'
                frames.append(df_na)

            if frames:
                merged = pd.concat(frames, axis=0, ignore_index=True)
            else:
                base_mcq_cols = list(mcq_data.columns) if len(mcq_data) else []
                base_na_cols = list(na_data.columns) if len(na_data) else []
                all_cols = list(dict.fromkeys(base_mcq_cols + base_na_cols + [
                    'pred_extracted', 'hit', 'MRA:.5:.95:.05', 'task_type',
                ]))
                merged = pd.DataFrame(columns=all_cols)

            prefer_front = [
                'index', 'question_type', 'task_type',
                'prediction', 'pred_extracted', 'answer',
                'hit', 'MRA:.5:.95:.05',
            ]
            ordered = [c for c in prefer_front if c in merged.columns] + \
                      [c for c in merged.columns if c not in prefer_front]
            merged = merged[ordered]

            with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
                merged.to_excel(writer, sheet_name='ALL', index=False)

            print(f'[save] extract & matching (merged) saved to {xlsx_path}')
        except Exception as e:
            warnings.warn(f'[save] failed to save merged extract xlsx to {xlsx_path}: {e}')

        try:
            acc_df = pd.DataFrame(
                [(k, v) for k, v in summary.items() if k not in ('tabulated_keys', 'tabulated_results')],
                columns=['metric', 'value'],
            )
            acc_df = acc_df.set_index('metric').T
            acc_df.to_csv(acc_tsv_path, sep='\t', index=False)
            print(f'[save] accuracy table saved to {acc_tsv_path}')
        except Exception as e:
            warnings.warn(f'[save] failed to save acc tsv to {acc_tsv_path}: {e}')

        print(f'[{self.dataset_name}] summary: {summary}')
        return summary

    def _aggregate(self, mcq_df: pd.DataFrame, na_df: pd.DataFrame) -> dict:
        output = {}

        if len(mcq_df):
            for qtype, sub in mcq_df.groupby('question_type'):
                output[f'{qtype}_accuracy'] = float(sub['hit'].mean())

        if len(na_df):
            for qtype, sub in na_df.groupby('question_type'):
                output[f'{qtype}_MRA:.5:.95:.05'] = float(sub['MRA:.5:.95:.05'].mean())

        rel_keys = [
            'object_rel_direction_easy_accuracy',
            'object_rel_direction_medium_accuracy',
            'object_rel_direction_hard_accuracy',
        ]
        if all(k in output for k in rel_keys):
            output['object_rel_direction_accuracy'] = (
                output.pop(rel_keys[0]) + output.pop(rel_keys[1]) + output.pop(rel_keys[2])
            ) / 3.0

        output['overall'] = float(sum(output.values()) / len(output)) if output else 0.0

        res = OrderedDict()
        res['overall'] = output['overall'] * 100.0

        ordered_qtypes = [
            'object_counting',
            'object_abs_distance',
            'object_size_estimation',
            'room_size_estimation',
            'object_rel_distance',
            'object_rel_direction',
            'route_planning',
            'obj_appearance_order',
        ]
        metrics_order = ['accuracy', 'MRA:.5:.95:.05']

        for qtype in ordered_qtypes:
            for metric in metrics_order:
                key = f'{qtype}_{metric}'
                if key in output:
                    res[key] = output[key] * 100.0

        tab_keys = ', '.join(list(res.keys()))
        tab_vals = ', '.join([f'{v:.3f}' for v in res.values()])
        print(f'Tabulated results: {tab_keys}')
        print(f'Tabulated results: {tab_vals}')

        res['tabulated_keys'] = tab_keys
        res['tabulated_results'] = tab_vals
        return res


class VsiSuperBase(VideoBaseDataset):
    MD5 = ''
    MODALITY = 'VIDEO'

    LMUData_root = LMUDataRoot()

    def __init__(self, dataset, pack=False, nframe=0, fps=-1):
        super().__init__(dataset=dataset, pack=pack, nframe=nframe, fps=fps)

    def download_vsisuper(self, repo_id, allow_patterns):
        cache_path = get_cache_path(repo_id)

        repo_name = repo_id.split('/')[1]

        # Extract duration from allow_patterns
        pattern = allow_patterns[0]
        basename = os.path.basename(pattern)
        m = re.search(r'(\d+)\s*mins?', basename)
        if not m:
            raise ValueError(
                f"Cannot parse duration from allow_patterns[0]={pattern!r}; "
            )
        duration = m.group(1)

        SENTINEL_NAME = f'.{repo_name}_{duration}_extracted'

        if (cache_path and os.path.isdir(cache_path)
                and os.path.isfile(os.path.join(cache_path, SENTINEL_NAME))):
            dataset_path = cache_path
        else:
            def _write_sentinel(sentinel_path, text='ok'):
                tmp = sentinel_path + '.tmp'
                with open(tmp, 'w', encoding='utf-8') as f:
                    f.write(text)
                os.replace(tmp, sentinel_path)

            def unzip_hf_zip(pth, only_zips=None):
                import zipfile

                base_dir = pth
                all_zips = [
                    f for f in os.listdir(base_dir)
                    if f.endswith('.zip')
                ]

                if only_zips:
                    filtered = []
                    for f in all_zips:
                        if any(fnmatch.fnmatch(f, pattern) for pattern in only_zips):
                            filtered.append(f)
                    zip_files = [os.path.join(base_dir, f) for f in filtered]
                else:
                    zip_files = [os.path.join(base_dir, f) for f in all_zips]

                zip_files.sort()

                for zip_file in tqdm(zip_files, desc=f'Unpacking Origin Data {only_zips}...'):
                    with zipfile.ZipFile(zip_file, 'r') as zf:
                        for info in zf.infolist():
                            if info.is_dir():
                                continue

                            rel = os.path.normpath(info.filename).lstrip('/\\')
                            dst = os.path.join(pth, rel)

                            absp = os.path.abspath(pth)
                            absd = os.path.abspath(dst)
                            if not absd.startswith(absp + os.sep):
                                raise RuntimeError(f'Unsafe path in zip: {info.filename}')

                            os.makedirs(os.path.dirname(dst), exist_ok=True)
                            with zf.open(info, 'r') as src, open(dst, 'wb') as out:
                                out.write(src.read())

                sentinel_path = os.path.join(pth, SENTINEL_NAME)
                _write_sentinel(sentinel_path, text='done')
                print(f'{repo_name} {duration}mins data extracted to current directory with original layout.')

            dataset_path = snapshot_download(
                repo_id=repo_id,
                repo_type='dataset',
                revision='main',
                allow_patterns=allow_patterns,
            )

            unzip_hf_zip(dataset_path, only_zips=allow_patterns)

        return dataset_path

    def save_video_frames(self, video_path, video_llm=False):
        vid_path = os.path.join(self.data_root, video_path)

        vid = decord.VideoReader(vid_path)
        video_nframes = len(vid)
        video_fps = vid.get_avg_fps()
        video_info = {
            'fps': video_fps,
            'n_frames': video_nframes,
        }

        indices = []

        if self.nframe > 0 and self.fps < 0:

            indices = np.linspace(0, video_nframes - 1, self.nframe, dtype=int).tolist()
            frame_paths = self.frame_paths(video_path)

        elif self.fps > 0:
            total_duration = video_nframes / video_fps
            required_frames = int(total_duration * self.fps)
            step_size = video_fps / self.fps
            indices = [int(i * step_size) for i in range(required_frames)]

            frame_paths = self.frame_paths_fps(video_path, len(indices))

        else:
            raise ValueError(
                f'save_video_frames: invalid configuration: nframe={self.nframe}, fps={self.fps}.'
                ' At least one of nframe>0 or fps>0 is required to determine frame sampling.'
            )

        flag = np.all([os.path.exists(p) for p in frame_paths])

        if not flag:
            images = [vid[i].asnumpy() for i in indices]
            images = [Image.fromarray(arr) for arr in images]
            for im, pth in zip(images, frame_paths):
                if not os.path.exists(pth) and not video_llm:
                    im.save(pth)

        return frame_paths, indices, video_info


class VsiSuperRecall(VsiSuperBase):
    """
    VSI-SUPER-Recall.

    Reference:
      Cambrian-S: Towards Spatial Supersensing in Video
      https://arxiv.org/abs/2511.04670
    """

    TYPE = 'MCQ'

    HF_ROOT_PATH = 'https://huggingface.co/datasets/lmms-lab-si/EASI-Leaderboard-Data/resolve/main'

    DATASET_URL = {
        'VsiSuperRecall_10mins': f'{HF_ROOT_PATH}/VsiSuperRecall_10mins.tsv',
        'VsiSuperRecall_30mins': f'{HF_ROOT_PATH}/VsiSuperRecall_30mins.tsv',
        'VsiSuperRecall_60mins': f'{HF_ROOT_PATH}/VsiSuperRecall_60mins.tsv',
        'VsiSuperRecall_120mins': f'{HF_ROOT_PATH}/VsiSuperRecall_120mins.tsv',
        'VsiSuperRecall_240mins': f'{HF_ROOT_PATH}/VsiSuperRecall_240mins.tsv'
    }

    DATASET_MD5 = {
        'VsiSuperRecall_10mins': 'bc914f85b41de2b4403e7172cf1d82e3',
        'VsiSuperRecall_30mins': '6f1581e2a8001efcd5ca060b6b8ac688',
        'VsiSuperRecall_60mins': '9db60ec28f5165f53812d4c579361a22',
        'VsiSuperRecall_120mins': '7f7448a69e45c473572fd1a44bc0619c',
        'VsiSuperRecall_240mins': '997889ecaf75f2765163f0fdde0cbce4'
    }

    def __init__(self, dataset, pack=False, nframe=0, fps=-1):
        super().__init__(dataset=dataset, pack=pack, nframe=nframe, fps=fps)
        self.duration = self.dataset_name.split('_')[1]

    @classmethod
    def supported_datasets(cls):
        video_duration = ['10', '30', '60', '120', '240']
        subsets = [f"VsiSuperRecall_{num}mins" for num in video_duration]

        return subsets

    def prepare_dataset(self, dataset_name):
        url = self.DATASET_URL[dataset_name]
        md5 = self.DATASET_MD5[dataset_name]

        _ = super().prepare_tsv(url, md5)

        data_duration = dataset_name.split('_')[1]

        dataset_path = self.download_vsisuper(
            repo_id='nyu-visionx/VSI-SUPER-Recall',
            allow_patterns=[f'{data_duration}.zip']
        )
        self.dataset_path = dataset_path

        data_file = os.path.join(self.LMUData_root, f'{dataset_name}.tsv')

        return dict(data_file=data_file, root=dataset_path)

    def build_prompt(self, line, video_llm):
        if isinstance(line, int):
            assert line < len(self)
            line = self.data.iloc[line]

        question = line['question'].strip()
        options = ast.literal_eval(line['options'])
        formatted_options = '\n'.join(options)

        # following VSI-SUPER-Recall prompt format from offical code base:
        # https://github.com/cambrian-mllm/cambrian-s/blob/main/lmms-eval/lmms_eval/tasks/cambrians_vsr/utils.py
        post_prompt = "\nAnswer with the option's letter from the given choices directly."
        prompt = question + '\nOptions:\n' + formatted_options + post_prompt

        message = []

        if video_llm:
            message.append(dict(type='video', value=os.path.join(self.data_root, line['video'])))
        else:
            frames, _, _ = self.save_video_frames(line['video'], video_llm)
            for im in frames:
                message.append(dict(type='image', value=im))

        message.append(dict(type='text', value=prompt))

        return message

    def evaluate(self, eval_file, **judge_kwargs):
        from .utils.spatial_bench.cal_scores import build_mcq_score_fn, eval_mcq_score

        # Select MCQ scoring function (rule-based or LLM-based) according to judge_kwargs['model'].
        score_fn = build_mcq_score_fn(**judge_kwargs)

        return eval_mcq_score(
            load_fn=load,
            eval_file=eval_file,
            score_fn=score_fn,
            group_col='question_type',
            order=None,
            dataset_name=getattr(self, 'dataset_name', f'VsiSuperRecall_{self.duration}')
        )


class VsiSuperCount(VsiSuperBase):
    """
    VSI-SUPER-Count.

    Reference:
      Cambrian-S: Towards Spatial Supersensing in Video
      https://arxiv.org/abs/2511.04670
    """

    TYPE = 'VQA'

    HF_ROOT_PATH = 'https://huggingface.co/datasets/lmms-lab-si/EASI-Leaderboard-Data/resolve/main'

    DATASET_URL = {
        'VsiSuperCount_10mins': f'{HF_ROOT_PATH}/VsiSuperCount_10mins.tsv',
        'VsiSuperCount_30mins': f'{HF_ROOT_PATH}/VsiSuperCount_30mins.tsv',
        'VsiSuperCount_60mins': f'{HF_ROOT_PATH}/VsiSuperCount_60mins.tsv',
        'VsiSuperCount_120mins': f'{HF_ROOT_PATH}/VsiSuperCount_120mins.tsv',
    }

    DATASET_MD5 = {
        'VsiSuperCount_10mins': '927ff0191d500561773b557bd2537d83',
        'VsiSuperCount_30mins': '5fa3af61ff0480b237d51c82bab5bee8',
        'VsiSuperCount_60mins': '7736e9a6130bd4f24e8255397e1395d0',
        'VsiSuperCount_120mins': '64302b2d1478153f362103e18fb71608',
    }

    def __init__(self, dataset, pack=False, nframe=0, fps=-1):
        super().__init__(dataset=dataset, pack=pack, nframe=nframe, fps=fps)
        self.duration = self.dataset_name.split('_')[1]

    @classmethod
    def supported_datasets(cls):
        video_duration = ['10', '30', '60', '120']
        subsets = [f"VsiSuperCount_{num}mins" for num in video_duration]

        return subsets

    def prepare_dataset(self, dataset_name):
        url = self.DATASET_URL[dataset_name]
        md5 = self.DATASET_MD5[dataset_name]

        data = super().prepare_tsv(url, md5)

        data_duration = dataset_name.split('_')[1]

        dataset_path = self.download_vsisuper(
            repo_id='nyu-visionx/VSI-SUPER-Count',
            allow_patterns=[f'{data_duration}*.zip']
        )
        self.dataset_path = dataset_path

        # NOTE: We intentionally drop all `*_streaming` samples here.
        # VSI-SUPER-Count streaming evaluation requires model-side changes
        # (e.g., episodic / streaming decoding, handling `query_times`, and
        # custom KV-cache scheduling). This cannot be implemented purely at
        # the benchmark/dataset level and would require modifying each
        # model’s inference / generate() logic, which is beyond the intended
        # design scope of VLMEvalKit. Therefore, this dataset class only
        # supports the non-streaming setting and filters out streaming rows.
        # We still keep the original TSV (with streaming + non-streaming)
        # in the repo for possible future streaming evaluation.
        if 'question_type' in data.columns:
            mask = ~data['question_type'].astype(str).str.contains('_streaming', na=False)
            data = data[mask]

            data_file = os.path.join(self.LMUData_root, f'{dataset_name}_wo_streaming.tsv')
            os.makedirs(os.path.dirname(data_file), exist_ok=True)
            data.to_csv(data_file, sep='\t', index=False)
        else:
            data_file = os.path.join(self.LMUData_root, f'{dataset_name}.tsv')

        return dict(data_file=data_file, root=dataset_path)

    def build_prompt(self, line, video_llm):
        if isinstance(line, int):
            assert line < len(self)
            line = self.data.iloc[line]

        question = line['question'].strip()

        # following VSI-SUPER-Count prompt format from offical code base:
        # https://github.com/cambrian-mllm/cambrian-s/blob/main/lmms-eval/lmms_eval/tasks/cambrians_vsc/utils.py
        pre_prompt = 'These are frames of a video.\n'
        post_prompt = "\nPlease answer the question using a single word or phrase."
        prompt = pre_prompt + question + post_prompt

        message = []

        if video_llm:
            message.append(dict(type='video', value=os.path.join(self.data_root, line['video'])))
        else:
            frames, _, _ = self.save_video_frames(line['video'], video_llm)
            for im in frames:
                message.append(dict(type='image', value=im))

        message.append(dict(type='text', value=prompt))

        return message

    def evaluate(self, eval_file, **judge_kwargs):
        from .utils.spatial_bench.cal_scores import attach_score_cache, build_na_score_fn
        from .utils.spatial_bench.tools.files import build_eval_paths, get_judge_tag_from_score_fn

        data = load(eval_file)
        data = data.sort_values(by='index')
        data['prediction'] = [str(x) for x in data['prediction']]

        # 1. build score_fn
        score_fn = build_na_score_fn(**judge_kwargs)

        # 2. build judge_tag
        judge_tag = (
            get_judge_tag_from_score_fn(score_fn)
            if score_fn is not None
            else 'extract_matching'
        )
        result_file, xlsx_path, acc_tsv_path = build_eval_paths(eval_file, judge_tag)

        # 3. attach cache files
        attach_score_cache(
            score_fn=score_fn,
            eval_file=eval_file,
            judge_tag=judge_tag,
            key_col='index',
            sub_tag='na',
        )

        # 4. run scoring
        na_scored = score_fn(data) if score_fn is not None else data
        summary = self._aggregate(na_scored)

        try:
            to_dump = {
                'na_scored': na_scored,
                'summary': summary,
            }
            with open(result_file, 'wb') as f:
                pickle.dump(to_dump, f)
            print(f'[save] result saved to {result_file}')
        except Exception as e:
            warnings.warn(f'[save] failed to save result to {result_file}: {e}')

        try:
            merged = na_scored.copy()

            prefer_front = [
                'index', 'question_type',
                'prediction', 'pred_extracted', 'answer',
                'MRA:.5:.95:.05',
            ]
            ordered = [c for c in prefer_front if c in merged.columns] + \
                      [c for c in merged.columns if c not in prefer_front]
            merged = merged[ordered]

            with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
                merged.to_excel(writer, sheet_name='ALL', index=False)

            print(f'[save] extract & matching (merged) saved to {xlsx_path}')
        except Exception as e:
            warnings.warn(f'[save] failed to save merged extract xlsx to {xlsx_path}: {e}')

        try:
            acc_df = pd.DataFrame(
                [(k, v) for k, v in summary.items()
                 if k not in ('tabulated_keys', 'tabulated_results')],
                columns=['metric', 'value'],
            )
            acc_df.to_csv(acc_tsv_path, sep='\t', index=False)
            print(f'[save] accuracy table saved to {acc_tsv_path}')
        except Exception as e:
            warnings.warn(f'[save] failed to save acc tsv to {acc_tsv_path}: {e}')

        print(f'[{self.dataset_name}] summary: {summary}')
        return summary

    def _aggregate(self, na_df: pd.DataFrame) -> dict:
        output = {}

        if len(na_df):
            for split_name, sub in na_df.groupby('question_type'):
                output[f'{split_name}_MRA:.5:.95:.05'] = float(
                    sub['MRA:.5:.95:.05'].mean()
                )

            output['overall'] = (
                float(sum(output.values()) / len(output))
                if output else 0.0
            )
        else:
            output['overall'] = 0.0

        res = OrderedDict()
        res['overall'] = output['overall'] * 100.0

        if len(na_df):
            for split_name in sorted(na_df['question_type'].unique()):
                key = f'{split_name}_MRA:.5:.95:.05'
                if key in output:
                    res[key] = output[key] * 100.0

        tab_keys = ', '.join(list(res.keys()))
        tab_vals = ', '.join([f'{v:.3f}' for v in res.values()])
        print(f'Tabulated results: {tab_keys}')
        print(f'Tabulated results: {tab_vals}')

        res['tabulated_keys'] = tab_keys
        res['tabulated_results'] = tab_vals
        return res
