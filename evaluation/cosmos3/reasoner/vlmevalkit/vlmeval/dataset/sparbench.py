import pickle
import warnings
from collections import OrderedDict

import numpy as np
import pandas as pd

from vlmeval.smp import load, toliststr
from .image_base import ImageBaseDataset
from .utils.spatial_bench.cal_scores import (attach_score_cache, build_mcq_score_fn,
                                             build_na_score_fn, mean_relative_accuracy)
from .utils.spatial_bench.tools.files import build_eval_paths, get_judge_tag_from_score_fn


class SparBench(ImageBaseDataset):
    """
    SPAR-Bench.

    Reference:
      From Flatland to Space: Teaching Vision-Language Models to Perceive and Reason in 3D
      https://arxiv.org/abs/2503.22976
    """

    TYPE = 'VQA'

    DATASET_URL = {
        'SparBench': 'https://huggingface.co/datasets/lmms-lab-si/EASI-Leaderboard-Data/resolve/main/SparBench.tsv',  # noqa: E501
        'SparBench_tiny': 'https://huggingface.co/datasets/lmms-lab-si/EASI-Leaderboard-Data/resolve/main/SparBench_tiny.tsv',  # noqa: E501
    }

    DATASET_MD5 = {
        'SparBench': 'cb1e5f5855c454241e0a70c3b8152976',
        'SparBench_tiny': 'c435e186064b795e0a9759aa60465a00',
    }

    TASK_TYPES = {
        'MCQ': [
            'obj_spatial_relation_oo',
            'obj_spatial_relation_oc_mv',
            'obj_spatial_relation_oo_mv',
            'spatial_imagination_oc',
            'spatial_imagination_oo',
            'spatial_imagination_oc_mv',
            'spatial_imagination_oo_mv',
            'position_matching',
            'camera_motion_infer',
            'distance_infer_center_oo',
            'distance_infer_center_oo_mv',
        ],
        'NA': [
            'depth_prediction_oc',
            'depth_prediction_oo',
            'distance_prediction_oc',
            'distance_prediction_oo',
            'depth_prediction_oc_mv',
            'depth_prediction_oo_mv',
            'distance_prediction_oo_mv',
            'distance_prediction_oc_mv',
        ],
        'SPECIAL': [
            'view_change_infer',
        ],
    }

    LOW_TASKS = [
        'depth_prediction_oc', 'depth_prediction_oc_mv',
        'depth_prediction_oo', 'depth_prediction_oo_mv',
        'distance_prediction_oc', 'distance_prediction_oc_mv',
        'distance_prediction_oo', 'distance_prediction_oo_mv',
    ]

    MIDDLE_TASKS = [
        'position_matching',
        'camera_motion_infer',
        'view_change_infer',
    ]

    HIGH_TASKS = [
        'distance_infer_center_oo', 'distance_infer_center_oo_mv',
        'obj_spatial_relation_oc_mv', 'obj_spatial_relation_oo', 'obj_spatial_relation_oo_mv',
        'spatial_imagination_oc', 'spatial_imagination_oc_mv',
        'spatial_imagination_oo', 'spatial_imagination_oo_mv',
    ]

    # used to strip suffix
    METRIC_SUFFIXES = (
        '_MRA:.5:.95:.05',
        '_accuracy',
        '_vci_metric',
    )

    @classmethod
    def get_task_type(cls, task: str) -> str:
        if task in cls.TASK_TYPES['MCQ']:
            return 'MCQ'
        if task in cls.TASK_TYPES['NA']:
            return 'NA'
        if task in cls.TASK_TYPES['SPECIAL']:
            return 'SPECIAL'
        raise ValueError(f'Unsupported SparBench task type: {task}')

    @staticmethod
    def _metric_base_task(metric_key: str) -> str | None:
        """
        Restore the base task name from the metric key.
        """
        if metric_key in ('overall', 'Low', 'Middle', 'High'):
            return None

        suffixes = [
            '_MRA:.5:.95:.05',
            '_accuracy',
            '_vci_metric',
        ]
        for suf in suffixes:
            if metric_key.endswith(suf):
                return metric_key[:-len(suf)]
        return None

    def build_prompt(self, line):
        if self.meta_only:
            tgt_path = toliststr(line['image_path'])
        else:
            tgt_path = self.dump_image(line)

        question = line['question']

        task = line['task']
        task_type = self.get_task_type(task)

        pre_prompt = ''

        if task_type == 'NA':
            post_prompt = 'Please answer the question using a single word or phrase.'
            prompt = pre_prompt + '\n' + question + '\n' + post_prompt

        elif task_type == 'MCQ':
            post_prompt = ''
            if task in ['position_matching', 'camera_motion_infer']:
                post_prompt = (
                    'The values represent the bounding box coordinates normalized to a 0-1000 scale, '
                    'with the top-left corner as the origin of the image.'
                )
            post_prompt2 = "Answer with the option's letter from the given choices directly."
            prompt = pre_prompt + '\n' + question + '\n' + post_prompt + '\n' + post_prompt2

        elif task_type == 'SPECIAL':
            post_prompt1 = ''
            post_prompt2 = ''
            prompt = pre_prompt + '\n' + question + '\n' + post_prompt1 + '\n' + post_prompt2

        else:
            raise ValueError(f'Unknown question type: {task}')

        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=prompt))

        return msgs

    def evaluate(self, eval_file, **judge_kwargs):
        data = load(eval_file)
        data = data.sort_values(by='index')
        data['prediction'] = [str(x) for x in data['prediction']]

        data['task_type'] = data['task'].apply(self.get_task_type)

        mcq_data = data[data['task_type'] == 'MCQ'].copy()
        na_data = data[data['task_type'] == 'NA'].copy()
        special_data = data[data['task_type'] == 'SPECIAL'].copy()

        splits = {'mcq': mcq_data, 'na': na_data, 'special': special_data}
        print(f'[split] MCQ={len(mcq_data)}, NA={len(na_data)}, SPECIAL={len(special_data)}')

        # 1. build score_fns
        builders = {
            'mcq': build_mcq_score_fn,
            'na': build_na_score_fn,
        }
        score_fns = {
            name: (builders[name](**judge_kwargs) if len(splits[name]) else None)
            for name in ('mcq', 'na')
        }

        # 2. pick a non-None score_fn to infer judge_tag
        primary_score_fn = score_fns.get('mcq') or score_fns.get('na')
        if primary_score_fn is not None:
            judge_tag = get_judge_tag_from_score_fn(primary_score_fn)
        else:
            judge_tag = 'extract_matching'

        result_file, xlsx_path, acc_tsv_path = build_eval_paths(eval_file, judge_tag)

        # 3. attach cache files for each sub-scorer
        for sub_tag, fn in score_fns.items():
            if fn is not None:
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
        sp_scored = (
            self.compute_special_score(special_data)
            if len(special_data)
            else special_data
        )

        summary = self._aggregate(mcq_scored, na_scored, sp_scored)

        print(f'[SparBench] summary: {summary}')

        # ---- save pkl dump ----
        try:
            to_dump = {
                'mcq_scored': mcq_scored,
                'na_scored': na_scored,
                'special_scored': sp_scored,
                'summary': summary,
            }
            with open(result_file, 'wb') as f:
                pickle.dump(to_dump, f)
            print(f'[save] result saved to {result_file}')
        except Exception as e:
            warnings.warn(f'[save] failed to save result to {result_file}: {e}')

        # ---- save extract_matching.xlsx ----
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

            if len(sp_scored):
                df_sp = sp_scored.copy()
                df_sp['task_type'] = 'SPECIAL'
                frames.append(df_sp)

            if frames:
                merged = pd.concat(frames, axis=0, ignore_index=True)
            else:
                merged = pd.DataFrame()

            prefer_front = [
                'index', 'task', 'task_type',
                'prediction', 'pred_extracted', 'answer',
                'hit', 'MRA:.5:.95:.05', 'vci_metric',
            ]
            ordered = [c for c in prefer_front if c in merged.columns] + \
                [c for c in merged.columns if c not in prefer_front]
            merged = merged[ordered]

            with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
                merged.to_excel(writer, sheet_name='ALL', index=False)

            print(f'[save] extract & matching saved to {xlsx_path}')

        except Exception as e:
            warnings.warn(f'[save] failed to save merged extract xlsx: {e}')

        # ---- save acc.tsv ----
        try:
            summary_clean = OrderedDict(
                (k, v) for k, v in summary.items()
                if k not in ('tabulated_keys', 'tabulated_results')
            )

            cols = list(summary_clean.keys())

            # overall / Low / Middle / High
            row_summary = {c: None for c in cols}
            for k in ('overall', 'Low', 'Middle', 'High'):
                if k in summary_clean:
                    row_summary[k] = summary_clean[k]

            # overall, Low, Middle, High && subtasks
            row_full = dict(summary_clean)

            acc_df = pd.DataFrame([row_summary, row_full], columns=cols)
            acc_df.to_csv(acc_tsv_path, sep='\t', index=False)
            print(f'[save] accuracy table saved to {acc_tsv_path}')

        except Exception as e:
            warnings.warn(f'[save] failed to save acc tsv: {e}')

        print(f'[{self.dataset_name}] summary: {summary}')
        return summary

    @staticmethod
    def _parse_instruction(instruction: str) -> dict[str, float]:
        # 'move_right:0.3,move_left:0.1' -> dict
        if instruction is None:
            return {}
        d = {}
        for item in str(instruction).split(','):
            item = item.strip()
            if not item or ':' not in item:
                continue
            k, v = item.split(':', 1)
            try:
                d[k.strip()] = float(v.strip())
            except Exception:
                pass
        return d

    @classmethod
    def _compute_vci_metric(cls, pred: str, answer: str) -> float:
        action_order = [
            ('move_right', 'move_left'),
            ('move_up', 'move_down'),
            ('move_forward', 'move_backward'),
            ('rotate_right', 'rotate_left'),
            ('rotate_up', 'rotate_down'),
        ]
        p = cls._parse_instruction(pred)
        g = cls._parse_instruction(answer)

        vals = []
        for a_pos, a_neg in action_order:
            pred_v = p.get(a_pos, 0.0) - p.get(a_neg, 0.0)
            gt_v = g.get(a_pos, 0.0) - g.get(a_neg, 0.0)
            vals.append(mean_relative_accuracy(pred_v, gt_v, 0.5, 0.95, 0.05))
        return float(np.mean(vals)) if len(vals) else 0.0

    def compute_special_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add column 'vci_metric'.
        """
        vals = []
        for _, r in df.iterrows():
            try:
                vals.append(self._compute_vci_metric(r['prediction'], r['answer']))
            except Exception:
                vals.append(0.0)
        df = df.copy()
        df['vci_metric'] = vals
        return df

    def _aggregate(self, mcq_df, na_df, sp_df) -> dict:
        task_metrics: dict[str, float] = {}

        # MCQ
        if len(mcq_df):
            for task, sub in mcq_df.groupby('task', sort=False):
                task_metrics[f'{task}_accuracy'] = float(sub['hit'].mean())

        # NA
        if len(na_df):
            for task, sub in na_df.groupby('task', sort=False):
                task_metrics[f'{task}_MRA:.5:.95:.05'] = float(sub['MRA:.5:.95:.05'].mean())

        # SPECIAL
        if len(sp_df):
            for task, sub in sp_df.groupby('task', sort=False):
                if 'vci_metric' in sub.columns:
                    task_metrics[f'{task}_vci_metric'] = float(sub['vci_metric'].mean())

        # overall
        if task_metrics:
            overall = float(np.mean(list(task_metrics.values())))
        else:
            overall = 0.0

        # base_task -> metric_key
        base_to_key: dict[str, str] = {}
        for k in task_metrics.keys():
            base = self._metric_base_task(k)
            if base is not None:
                base_to_key[base] = k

        def mean_group(task_list: list[str]) -> float:
            vals = []
            for t in task_list:
                key = base_to_key.get(t, None)
                if key is not None:
                    vals.append(task_metrics[key])
            return float(np.mean(vals)) if vals else 0.0

        low_val = mean_group(self.LOW_TASKS)
        mid_val = mean_group(self.MIDDLE_TASKS)
        high_val = mean_group(self.HIGH_TASKS)

        out = OrderedDict()
        out['overall'] = overall
        out['Low'] = low_val
        out['Middle'] = mid_val
        out['High'] = high_val

        # Low
        for t in self.LOW_TASKS:
            key = base_to_key.get(t, None)
            if key is not None:
                out[key] = task_metrics[key]

        # Middle
        for t in self.MIDDLE_TASKS:
            key = base_to_key.get(t, None)
            if key is not None:
                out[key] = task_metrics[key]

        # High
        for t in self.HIGH_TASKS:
            key = base_to_key.get(t, None)
            if key is not None:
                out[key] = task_metrics[key]

        return out
