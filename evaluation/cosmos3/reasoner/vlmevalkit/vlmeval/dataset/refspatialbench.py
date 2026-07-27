import ast
import os
from collections import defaultdict

import numpy as np
import pandas as pd
from huggingface_hub import snapshot_download
from PIL import Image

from ..smp import dump, get_cache_path, load, modelscope_flag_set, toliststr
from .image_vqa import ImageVQADataset
from .utils.spatial_bench.tools.utils import Point2DParser


class RefSpatialBench(ImageVQADataset):
    """
    RefSpatial-Bench.

    Reference:
      RefSpatial-Bench: A Benchmark for Multi-step Spatial Referring
      https://arxiv.org/abs/2506.04308
    """

    DATASET_URL = {
        'RefSpatial': 'https://huggingface.co/datasets/lmms-lab-si/EASI-Leaderboard-Data/resolve/main/RefSpatial.tsv',
        'RefSpatial_wo_unseen': 'https://huggingface.co/datasets/lmms-lab-si/EASI-Leaderboard-Data/resolve/main/RefSpatial_wo_unseen.tsv',  # noqa: E501
    }
    DATASET_MD5 = {
        'RefSpatial': 'd8c6d38bab73922aae4c19b027bbe4e5',
        'RefSpatial_wo_unseen': 'ad35a8b0e40878e2b4c9a74b93dd9011',
    }

    DEFAULT_FAMILY = 'cr'

    # Qwen3-VL convention: key `point_2d`, order [x, y]. Gemini/Gemma convention:
    # key `point` (no `_2d`), order [y, x]. Parser-side `Point2DParser` detects
    # the `point` (no-suffix) key and swaps yx→xy.
    _QWEN_PROMPT = (
        'Output the point coordinates in JSON format.\n'
        'For example:\n'
        '[\n'
        '  {"point_2d": [x, y], "label": "point_1"}\n'
        ']\n'
    )
    _GEMINI_PROMPT = (
        'Output the point coordinates in JSON format.\n'
        'For example:\n'
        '[\n'
        '  {"point": [y, x], "label": "point_1"}\n'
        ']\n'
    )
    # MiMo (Qwen2.5-VL backbone) emits absolute pixel coordinates, unlike
    # Qwen3-VL's 0-1000 normalized convention. Prompt key/order matches Qwen
    # (point_2d / [x, y]); only the value scale differs.
    _MIMO_PROMPT = (
        'Output the point coordinates in JSON format.\n'
        'For example:\n'
        '[\n'
        '  {"point_2d": [x, y], "label": "point_1"}\n'
        ']\n'
        'Coordinates are absolute pixel values, not normalized to 0-1000.\n'
    )
    _PROMPTS = {
        'cr':     _QWEN_PROMPT,
        'qwen3':  _QWEN_PROMPT,
        'gemini': _GEMINI_PROMPT,
        'gemma4': _GEMINI_PROMPT,
        'mimo':   _MIMO_PROMPT,
    }

    def __init__(self, dataset='RefSpatial', skip_noimg=False, **kwargs):
        self.model_family = kwargs.pop('model_family', self.DEFAULT_FAMILY)
        super().__init__(dataset=dataset, skip_noimg=skip_noimg, **kwargs)

    def _task_category(self):
        return ['location', 'placement', 'unseen']

    def prepare_tsv(self, url, file_md5=None, repo_id='BAAI/RefSpatial-Bench'):
        data = super().prepare_tsv(url, file_md5)

        SENTINEL_NAME = '.refspatial_extracted'
        cache_path = get_cache_path(repo_id)

        if (cache_path and os.path.isdir(cache_path)
                and os.path.isfile(os.path.join(cache_path, SENTINEL_NAME))):
            dataset_path = cache_path
        else:
            def _write_sentinel(sentinel_path, text='ok'):
                tmp = sentinel_path + '.tmp'
                with open(tmp, 'w', encoding='utf-8') as f:
                    f.write(text)
                os.replace(tmp, sentinel_path)

            if modelscope_flag_set():
                from modelscope import dataset_snapshot_download
                dataset_path = dataset_snapshot_download(dataset_id=repo_id)
            else:
                dataset_path = snapshot_download(repo_id=repo_id, repo_type='dataset')

            sentinel_path = os.path.join(dataset_path, SENTINEL_NAME)
            _write_sentinel(sentinel_path, text='done')

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
            data['mask_path'] = data['mask_path'].map(to_abs)

        return data

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]

        if self.meta_only:
            tgt_path = toliststr(line['image_path'])
        else:
            tgt_path = self.dump_image(line)

        question = line['question']

        post_prompt = self._PROMPTS.get(self.model_family, self._PROMPTS[self.DEFAULT_FAMILY])
        prompt = f'{question}' + post_prompt

        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=prompt))

        return msgs

    @staticmethod
    def _normalize_path(p):
        """
        Normalize a list-like or stringified list to a single path string.
        """
        if isinstance(p, (list, tuple)):
            return p[0]
        if isinstance(p, str) and p.startswith('[') and p.endswith(']'):
            try:
                v = ast.literal_eval(p)
                if isinstance(v, (list, tuple)) and v:
                    return v[0]
            except Exception:
                pass
        return p

    def parse_prediction(self, pred_text: str, width: int, height: int) -> np.ndarray:
        """
        Parse raw model output into pixel coordinates of shape (N, 2).

        Default: use Point2DParser (JSON/Python literal with 'point_2d'/'point'
        or fallback '(x, y)' / '(x0, y0, x1, y1)').

        Override this method if your model uses a different format.
        Always return pixel coordinates.
        """
        Point2DParser.log_hint(task_name=getattr(self, 'dataset_name', 'RefSpatial'))
        coord_scale = 'pixel' if self.model_family == 'mimo' else 'auto'
        return Point2DParser.parse(pred_text, width, height, output='pixel', coord_scale=coord_scale)

    def evaluate(self, eval_file, **judge_kwargs):
        from .utils.spatial_bench.tools.files import build_eval_paths

        data = load(eval_file)
        if 'index' in data.columns:
            data = data.sort_values(by='index')
        data['prediction'] = data['prediction'].astype(str)

        result_file, xlsx_path, acc_tsv_path = build_eval_paths(eval_file, judge_tag='Point2D')

        required = ['category', 'mask_path', 'prediction']
        for col in required:
            if col not in data.columns:
                raise ValueError(f"Column '{col}' not found in eval_file: {eval_file}")

        acc_all = []
        acc_by_cat = defaultdict(list)

        score_list = []
        num_points_list = []
        is_parsable_list = []

        for _, row in data.iterrows():
            cat = str(row['category']).lower()
            pred_text = row['prediction']

            mask_raw = row['mask_path']
            mask_path = self._normalize_path(mask_raw)
            mask_path = str(mask_path)

            if not os.path.exists(mask_path):
                print(f'[WARNING] mask not found: {mask_path}')
                continue

            mask = np.array(Image.open(mask_path)) / 255.0
            if mask.ndim == 3:
                mask = mask[:, :, 0]
            mask = (mask > 0).astype(np.uint8)

            acc = 0.0
            is_parsable = False
            num_points = 0

            try:
                points = self.parse_prediction(pred_text, mask.shape[1], mask.shape[0])
                is_parsable = True
                num_points = len(points)

                if len(points) > 0:
                    in_range = (
                        (points[:, 0] >= 0)
                        & (points[:, 0] < mask.shape[1])
                        & (points[:, 1] >= 0)
                        & (points[:, 1] < mask.shape[0])
                    )
                    if in_range.any():
                        vals = mask[points[in_range, 1], points[in_range, 0]]
                        # Out-of-range points count as 0
                        vals = np.concatenate([vals, np.zeros(points.shape[0] - in_range.sum())])
                        acc = float(vals.mean())

            except Exception as e:
                print(f'[WARN] failed to parse prediction: {pred_text} ({e})')

            acc_all.append(acc)
            acc_by_cat[cat].append(acc)

            score_list.append(acc)
            num_points_list.append(num_points)
            is_parsable_list.append(is_parsable)

        if not acc_all:
            raise ValueError('No valid accuracy computed; check eval_file format and mask_path.')

        overall = float(np.mean(acc_all))

        data['score'] = score_list
        data['num_points'] = num_points_list
        data['is_parsable'] = is_parsable_list

        dump(data, result_file)
        try:
            data.to_excel(xlsx_path, index=False)
        except Exception as e:
            print(f'[WARN] failed to save xlsx to {xlsx_path}: {e}')

        results = {'overall': overall}
        for cat in self._task_category():
            vals = acc_by_cat.get(cat, [])
            if vals:
                results[cat] = float(np.mean(vals))

        acc_df = pd.DataFrame([results])
        acc_df.to_csv(
            acc_tsv_path,
            sep='\t',
            index=False,
            float_format='%.6f',
        )

        return results
