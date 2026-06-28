import json

import pandas as pd

from vlmeval.smp import load, toliststr
from .image_vqa import ImageVQADataset


class ERQABench(ImageVQADataset):
    """
    ERQA.

    Reference:
      Gemini Robotics: Bringing AI into the Physical World
      https://arxiv.org/html/2503.20020v1
    """

    DATASET_URL = {
        'ERQA': 'https://huggingface.co/datasets/lmms-lab-si/EASI-Leaderboard-Data/resolve/main/ERQA.tsv',
    }
    DATASET_MD5 = {
        'ERQA': '40d45d4f0bb1852dbc68b28ee70d4ca5',
    }

    def _task_category(self):
        return [
            'Action Reasoning',
            'Multi-view Reasoning',
            'Pointing',
            'Spatial Reasoning',
            'State Estimation',
            'Task Reasoning',
            'Trajectory Reasoning',
            'Other'
        ]

    def build_prompt(self, line):
        """
        Images are interleaved into the question according to visual_indices.
        """
        if isinstance(line, int):
            line = self.data.iloc[line]

        # Resolve image paths
        if self.meta_only:
            tgt_paths = toliststr(line['image'])
        else:
            tgt_paths = self.dump_image(line)

        question = line['question']
        vis_raw = line.get('visual_indices', [])

        # Parse visual_indices → List[int]
        if isinstance(vis_raw, str):
            try:
                visual_indices = json.loads(vis_raw)
            except Exception:
                parts = [p.strip() for p in vis_raw.strip('[]').split(',') if p.strip()]
                visual_indices = [int(p) for p in parts] if parts else []
        elif isinstance(vis_raw, (list, tuple)):
            visual_indices = list(vis_raw)
        elif pd.isna(vis_raw):
            visual_indices = []
        else:
            visual_indices = []

        # Mismatch: put all images at the beginning
        if len(visual_indices) != len(tgt_paths):
            visual_indices = [0] * len(tgt_paths)

        img_idx_pairs = list(zip(tgt_paths, visual_indices))
        img_idx_pairs.sort(key=lambda x: x[1])

        msgs = []

        # All images first, then full question
        if not visual_indices or all(idx == 0 for idx in visual_indices):
            for p, _ in img_idx_pairs:
                msgs.append(dict(type='image', value=p))
            msgs.append(dict(type='text', value=question))
            return msgs

        # Interleave text and images by character index
        last_pos = 0
        for p, idx in img_idx_pairs:
            if idx == 0:
                msgs.append(dict(type='image', value=p))
            else:
                if idx <= len(question):
                    seg = question[last_pos:idx]
                    if seg:
                        msgs.append(dict(type='text', value=seg))
                    msgs.append(dict(type='image', value=p))
                    last_pos = idx
                else:
                    msgs.append(dict(type='image', value=p))

        if last_pos < len(question):
            seg = question[last_pos:]
            if seg:
                msgs.append(dict(type='text', value=seg))

        if not msgs:
            for p, _ in img_idx_pairs:
                msgs.append(dict(type='image', value=p))
            msgs.append(dict(type='text', value=question))

        return msgs

    def evaluate(self, eval_file, **judge_kwargs):
        from .utils.spatial_bench.cal_scores import build_mcq_score_fn, eval_mcq_score

        # Select MCQ scoring function (rule-based or LLM-based) according to judge_kwargs['model'].
        score_fn = build_mcq_score_fn(**judge_kwargs)

        return eval_mcq_score(
            load_fn=load,
            eval_file=eval_file,
            score_fn=score_fn,
            group_col='question_type',
            order=self._task_category(),
            dataset_name=getattr(self, 'dataset_name', 'ERQA')
        )
