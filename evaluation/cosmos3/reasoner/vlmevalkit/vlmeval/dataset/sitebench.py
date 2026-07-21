import ast
import json
import os
import string
import warnings
from collections import OrderedDict

import decord
import numpy as np
import pandas as pd
from huggingface_hub import snapshot_download
from PIL import Image
from tqdm import tqdm

from vlmeval.smp import LMUDataRoot, dump, get_cache_path, load, modelscope_flag_set, toliststr
from .image_mcq import ImageMCQDataset
from .video_base import VideoBaseDataset


class SiteBenchBase:
    """
    SITE-Bench.

    Reference:
      SITE: towards Spatial Intelligence Thorough Evaluation
      https://arxiv.org/abs/2505.05456
    """

    def __init__(self, *args, **kwargs) -> None:
        self.repo_id = 'franky-veteran/SITE-Bench'
        super().__init__(*args, **kwargs)

    def _task_category(self):
        return [
            'counting & existence',
            'object localization & positioning',
            '3d information understanding',
            'multi-view & cross-image reasoning',
            'spatial relationship reasoning',
            'movement prediction & navigation',
        ]

    def download_sitebench(self, repo_id='franky-veteran/SITE-Bench'):
        cache_path = get_cache_path(repo_id)
        SENTINEL_NAME = '.sitebench_extracted'

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
                print('SiteBench data extracted to current directory with original layout.')

            if modelscope_flag_set():
                from modelscope import dataset_snapshot_download
                dataset_path = dataset_snapshot_download(dataset_id=repo_id)
            else:
                dataset_path = snapshot_download(repo_id=repo_id, repo_type='dataset')

            unzip_hf_zip(dataset_path)

        return dataset_path

    def evaluate(self, eval_file, **kwargs):
        from .utils.spatial_bench.cal_scores import (attach_score_cache, build_mcq_score_fn,
                                                     compute_caa_score)
        from .utils.spatial_bench.tools.files import build_eval_paths, get_judge_tag_from_score_fn

        score_fn = build_mcq_score_fn(**kwargs)  # Select MCQ scoring func according to judge_kwargs['model'].
        judge_tag = get_judge_tag_from_score_fn(score_fn)

        result_file, xlsx_path, acc_tsv_path = build_eval_paths(eval_file, judge_tag)

        data = load(eval_file)
        if 'index' in data.columns:
            data = data.sort_values(by='index')
        data['prediction'] = [str(x) for x in data['prediction']]

        # compute per-sample hit (MCQ)
        attach_score_cache(
            score_fn=score_fn,
            eval_file=eval_file,
            judge_tag=judge_tag,
            key_col='index',
            sub_tag='mcq',
        )
        mcq_scored = score_fn(data.copy())

        cat_order = self._task_category()

        summary = OrderedDict()
        overall_acc = float(mcq_scored['hit'].mean()) if len(mcq_scored) else 0.0
        overall_caa = compute_caa_score(mcq_scored) if len(mcq_scored) else 0.0
        summary['overall_accuracy'] = overall_acc * 100.0
        summary['overall_caa'] = overall_caa * 100.0

        # per-category
        if 'category' in mcq_scored.columns:
            for cat in cat_order:
                sub = mcq_scored[mcq_scored['category'] == cat]
                if len(sub):
                    acc_cat = float(sub['hit'].mean())
                    caa_cat = compute_caa_score(sub)
                    summary[f'{cat}_accuracy'] = acc_cat * 100.0
                    summary[f'{cat}_caa'] = caa_cat * 100.0

        tab_keys = ', '.join(list(summary.keys()))
        tab_vals = ', '.join([f'{v:.3f}' for v in summary.values()])
        summary['tabulated_keys'] = tab_keys
        summary['tabulated_results'] = tab_vals

        try:
            import pickle
            with open(result_file, 'wb') as f:
                pickle.dump({'mcq_scored': mcq_scored, 'summary': summary}, f)
            print(f'[save] result saved to {result_file}')
        except Exception as e:
            warnings.warn(f'[save] failed to save result to {result_file}: {e}')

        try:
            prefer_front = [
                'index', 'question_type', 'category',
                'prediction', 'pred_extracted', 'answer', 'hit'
            ]
            merged = mcq_scored.copy()
            ordered = (
                [c for c in prefer_front if c in merged.columns]
                + [c for c in merged.columns if c not in prefer_front]
            )
            merged = merged[ordered]

            with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
                merged.to_excel(writer, sheet_name='ALL', index=False)
            print(f'[save] extract & matching saved to {xlsx_path}')
        except Exception as e:
            warnings.warn(f'[save] failed to save extract xlsx to {xlsx_path}: {e}')

        try:
            acc_df = pd.DataFrame(
                [(k, v) for k, v in summary.items() if k not in ('tabulated_keys', 'tabulated_results')],
                columns=['metric', 'value']
            )

            metric_order = ['overall_accuracy', 'overall_caa']
            metric_order += [f'{c}_accuracy' for c in cat_order if f'{c}_accuracy' in acc_df['metric'].values]
            metric_order += [f'{c}_caa' for c in cat_order if f'{c}_caa' in acc_df['metric'].values]

            metric_order += [k for k in acc_df['metric'].tolist() if k not in metric_order]

            acc_df = acc_df.set_index('metric').reindex(metric_order).reset_index()
            acc_df = acc_df.dropna(subset=['value'])
            acc_df.to_csv(acc_tsv_path, sep='\t', index=False)
            print(f'[save] accuracy/CAA table saved to {acc_tsv_path}')
        except Exception as e:
            warnings.warn(f'[save] failed to save acc tsv to {acc_tsv_path}: {e}')

        print(f'[{getattr(self, "dataset_name", "MCQ")}] summary: {summary}')
        return summary


class SiteBenchImage(SiteBenchBase, ImageMCQDataset):
    TYPE = 'MCQ'

    DATASET_URL = {
        'SiteBenchImage': 'https://huggingface.co/datasets/lmms-lab-si/EASI-Leaderboard-Data/resolve/main/SiteBenchImage.tsv'  # noqa: E501
    }

    DATASET_MD5 = {
        'SiteBenchImage': '59a2ada248b743c1d7b2f89dd5afcdc3'
    }

    def prepare_tsv(self, url, file_md5=None):
        data = super().prepare_tsv(url, file_md5)

        dataset_path = self.download_sitebench(self.repo_id)

        # === Transfer rel path to abs path ===
        if 'image_path' in data.columns:
            def fix_one(x: str):
                if not isinstance(x, str):
                    return x
                s = x.strip()
                s = os.path.expanduser(os.path.expandvars(s))

                if not dataset_path:
                    return os.path.normpath(s)
                return os.path.normpath(os.path.join(dataset_path, s.lstrip(r'\/')))

            def to_abs(p):
                if isinstance(p, list):
                    return [fix_one(xx) for xx in p]
                if isinstance(p, str) and p.strip().startswith('[') and p.strip().endswith(']'):
                    try:
                        lst = ast.literal_eval(p)
                        if isinstance(lst, list):
                            return [fix_one(xx) for xx in lst]
                    except Exception:
                        pass
                return fix_one(p)

            data['image_path'] = data['image_path'].map(to_abs)

        return data

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]

        if self.meta_only:
            tgt_path = toliststr(line['image_path'])
        else:
            tgt_path = self.dump_image(line)

        question = line['question']
        options = line['options']

        upper_letters = list(string.ascii_uppercase)

        if isinstance(options, str):
            try:
                options = json.loads(options)
            except Exception:
                options = ast.literal_eval(options)

        option_text = '\n'.join(
            f'{upper_letters[i]}: {options[i]}'
            for i in range(len(options))
        )

        prompt = ''
        if '<image>' not in question and '<image>' not in option_text:
            prompt = '<image>' * len(tgt_path) + '\n'

        # prompt format aligned with SITE paper
        prompt += 'Question: ' + question + '\n'
        prompt += 'Options:\n' + option_text + '\n'
        post_prompt = 'Give me the answer letter directly. The best answer is:'
        prompt += post_prompt

        msgs = self.build_msgs(tgt_path, prompt)
        return msgs

    @staticmethod
    def build_msgs(tgt_path, prompt):
        """
        Interlaced text and pictures.
        """
        images = tgt_path if isinstance(tgt_path, list) else [tgt_path]

        parts = prompt.split('<image>')
        segs = []

        for i, part in enumerate(parts):
            part = part.strip()
            if part:
                segs.append(dict(type='text', value=part))
            if (i != len(parts) - 1) and (i < len(images)):
                segs.append(dict(type='image', value=images[i]))

        return [s for s in segs if s['value']]


class SiteBenchVideo(SiteBenchBase, VideoBaseDataset):

    MD5 = ''

    TYPE = 'Video-MCQ'
    MODALITY = 'VIDEO'

    LMUData_root = LMUDataRoot()
    DATASET_URL = {
        'SiteBenchVideo': 'https://huggingface.co/datasets/lmms-lab-si/EASI-Leaderboard-Data/resolve/main/SiteBenchVideo.tsv'  # noqa: E501
    }
    DATASET_MD5 = {
        'SiteBenchVideo': 'bb2ac531fa83cf8280b23c25d738922d'
    }

    def __init__(self, dataset='SiteBenchVideo', pack=False, nframe=0, fps=-1):
        super().__init__(dataset=dataset, pack=pack, nframe=nframe, fps=fps)

    @classmethod
    def supported_datasets(cls):
        return ['SiteBenchVideo']

    def prepare_dataset(self, dataset_name='SiteBenchVideo'):
        data = super().prepare_tsv(
            self.DATASET_URL[self.dataset_name],
            self.DATASET_MD5[self.dataset_name]
        )

        dataset_path = self.download_sitebench(self.repo_id)
        self.dataset_path = dataset_path

        # === Transfer rel path to abs path ===
        if 'video_path' in data.columns:
            def fix_one(x: str):
                if not isinstance(x, str):
                    return x
                s = x.strip()
                s = os.path.expanduser(os.path.expandvars(s))

                if not dataset_path:
                    return os.path.normpath(s)
                return os.path.normpath(os.path.join(dataset_path, s.lstrip(r'\/')))

            def to_abs(p):
                if isinstance(p, list):
                    return [fix_one(xx) for xx in p]
                if isinstance(p, str) and p.strip().startswith('[') and p.strip().endswith(']'):
                    try:
                        lst = ast.literal_eval(p)
                        if isinstance(lst, list):
                            return [fix_one(xx) for xx in lst]
                    except Exception:
                        pass
                return fix_one(p)

            data['video'] = data['video_path'].map(to_abs)

        new_data_path = os.path.join(self.LMUData_root, 'SiteBenchVideo_abs_path.tsv')
        if not os.path.exists(new_data_path):
            dump(data, new_data_path)

        return dict(data_file=new_data_path, root=dataset_path)

    def save_video_frames(self, video, video_llm=False):
        vid_path = video
        rel_video_path = os.path.relpath(video, self.dataset_path)

        vid = decord.VideoReader(vid_path)
        video_nframes = len(vid)
        video_fps = vid.get_avg_fps()
        video_info = {
            'fps': video_fps,
            'n_frames': video_nframes,
        }

        if self.nframe > 0 and self.fps < 0:
            indices = np.linspace(0, video_nframes - 1, self.nframe, dtype=int).tolist()
            # Use os.path.relpath for robust relative path extraction
            frame_paths = self.frame_paths(rel_video_path)

        elif self.fps > 0:
            total_duration = video_nframes / video_fps
            required_frames = int(total_duration * self.fps)
            step_size = video_fps / self.fps

            indices = [int(i * step_size) for i in range(required_frames)]
            frame_paths = self.frame_paths_fps(rel_video_path, len(indices))

        missing = [
            (idx, pth) for idx, pth in zip(indices, frame_paths)
            if not os.path.exists(pth)
        ]

        if missing and not video_llm:
            for frame_idx, pth in missing:
                try:
                    frame_data = vid[frame_idx].asnumpy()
                    Image.fromarray(frame_data).save(pth)
                except Exception as e:
                    error_msg = f"Error saving frame {frame_idx} from {vid_path}: {str(e)}"
                    print(error_msg)

                    raise ValueError(error_msg) from e

        return frame_paths, indices, video_info

    def build_prompt(self, line, video_llm):
        if isinstance(line, int):
            assert line < len(self)
            line = self.data.iloc[line]

        question = line['question']
        raw_options = ast.literal_eval(line['candidates'])

        option_labels = list(string.ascii_uppercase)
        assert len(raw_options) <= len(option_labels), 'Too many options, extend option_labels if needed'

        options = [f'{label}: {opt}' for label, opt in zip(option_labels, raw_options)]
        formatted_options = '\n'.join(options)

        # video prompt from SITE paper
        pre_prompt = (
            'Select the best answer to the following multiple-choice question based on the video. '
            'Respond with only the letter of the correct option.'
        )
        post_prompt = 'Give me the answer letter directly. The best answer is:'

        question_text = 'Question: ' + question + '\n'
        option_text = 'Options:\n' + formatted_options + '\n'

        prompt = pre_prompt + '\n' + question_text + option_text + post_prompt

        message = []
        if video_llm:
            message.append(dict(type='video', value=line['video']))
        else:
            frames, _, _ = self.save_video_frames(line['video'], video_llm)
            for im in frames:
                message.append(dict(type='image', value=im))

        message.append(dict(type='text', value=prompt))
        return message
