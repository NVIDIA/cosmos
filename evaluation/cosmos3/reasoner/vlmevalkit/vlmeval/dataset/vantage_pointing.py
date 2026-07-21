"""
VANTAGE_2DPointing — image MCQ where each option is a candidate [x, y] pixel
coordinate. The model is asked for the letter (A/B/C/D) of the option whose
point lies inside the referenced object.

Lifted out of upstream image_mcq.py to keep the catch-all module focused, and
extended with an S3 fetch shim so the TSV + images are pulled from the
VANTAGE-bench stage on first use (the public HF release ships the TSV without
the `answer` column, which would make scoring impossible).
"""

import csv
import os.path as osp
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

from ..smp import LMUDataRoot, dump, load
from ..smp.file import get_file_extension, get_intermediate_file_path
from .image_mcq import ImageMCQDataset

# VANTAGE-bench S3 stage: TSV + images_annotated/. The TSV here has the
# `answer` and `target_point` columns that the public HF release omits.
S3_BUCKET = 'cosmos_understanding'
S3_PREFIX = 'benchmark/vantage_benchmark_hf_release_annotations/datasets/Metropolis2DPointing'


class VANTAGE_2DPointing(ImageMCQDataset):

    TYPE = 'MCQ'

    @classmethod
    def supported_datasets(cls):
        return ['VANTAGE_2DPointing']

    def __init__(self, dataset='VANTAGE_2DPointing', custom_prompt=None, data_root=None, **kwargs):
        self.verbose = kwargs.get('verbose', False)
        self.custom_prompt = custom_prompt
        self._data_root_override = data_root
        kwargs.pop('model_family', None)
        super().__init__(dataset=dataset, **kwargs)

    def load_data(self, dataset):
        """Override parent's DATASET_URL/prepare_tsv path. Pulls TSV + images
        from the VANTAGE-bench S3 stage on first call; subsequent calls reuse
        the local cache under LMUDataRoot()/datasets/<dataset>/."""
        if self._data_root_override is not None:
            local_dir = self._data_root_override
        else:
            local_dir = osp.join(LMUDataRoot(), 'datasets', dataset)
        # Upstream VANTAGE renamed the class to VANTAGE_2DPointing but kept the
        # data file at its original name Metropolis2DPointing.tsv on S3.
        data_file = osp.join(local_dir, 'Metropolis2DPointing.tsv')

        if not osp.exists(data_file):
            self._download_from_s3(local_dir)

        if not osp.exists(data_file):
            raise FileNotFoundError(
                f"VANTAGE_2DPointing TSV missing after S3 fetch: {data_file}"
            )

        self.img_root = local_dir
        return load(data_file)

    def _download_from_s3(self, local_dir: str) -> None:
        try:
            from s3fs import S3FileSystem
        except ImportError as e:
            raise ImportError(
                "s3fs is required for S3 access. Install with: pip install s3fs"
            ) from e

        Path(local_dir).parent.mkdir(parents=True, exist_ok=True)
        s3 = S3FileSystem(
            anon=False,
            profile='team-cosmos',
            client_kwargs={'endpoint_url': 'https://pdx.s8k.io'},
        )
        s3_path = f'{S3_BUCKET}/{S3_PREFIX}'
        print(f"Downloading VANTAGE_2DPointing from s3://{s3_path} to {local_dir} ...")
        s3.get(s3_path, str(local_dir), recursive=True)
        print(f"VANTAGE_2DPointing download complete: {local_dir}")

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]

        tgt_path = self.dump_image(line)
        if isinstance(tgt_path, list) and len(tgt_path) > 0:
            tgt_path = tgt_path[0]

        question = line['question']
        options = {cand: line[cand] for cand in 'ABCD' if cand in line and not pd.isna(line[cand])}

        # TSV options are absolute pixel coords (e.g. [1832, 721] for 1920x1080).
        # Models (cr2/cr3, qwen3-vl, qwen3.5) all use 0-1000 normalized internally;
        # normalize the option coords so the prompt matches their convention.
        from PIL import Image
        try:
            with Image.open(tgt_path) as im:
                W, H = im.size
        except Exception:
            W = H = None

        def _norm_option(item):
            if W is None or H is None:
                return item
            m = re.match(r'\s*\[?\s*(\d+)\s*,\s*(\d+)\s*\]?\s*$', str(item))
            if not m:
                return item
            x, y = int(m.group(1)), int(m.group(2))
            return f"{round(x * 1000 / W)}, {round(y * 1000 / H)}"

        options_prompt = 'Options (coordinates normalized to 0-1000 scale [x, y]):\n'
        for key, item in options.items():
            options_prompt += f'{key}. {_norm_option(item)}\n'

        if self.custom_prompt is not None:
            prompt = self.custom_prompt.format(question=question, options=options_prompt)
        else:
            prompt = (
                "Answer the following spatial grounding question based on the image.\n"
                f"Question: {question}\n"
                f"{options_prompt}"
                "Respond with ONLY the letter of the correct option (A, B, C, or D)."
            )

        return [
            dict(type='image', value=tgt_path),
            dict(type='text', value=prompt),
        ]

    def evaluate(self, eval_file, **judge_kwargs):
        assert get_file_extension(eval_file) in ['xlsx', 'json', 'tsv'], \
            'data file should be a supported format (xlsx/json/tsv) file'

        data = load(eval_file)
        verbose = judge_kwargs.get('verbose', False) or self.verbose
        results = {}
        category_stats = defaultdict(lambda: {"correct": 0, "total": 0})

        for _, row in data.iterrows():
            matching = self.data[self.data['index'].astype(str) == str(row['index'])]
            if len(matching) == 0:
                if verbose:
                    print(f"Warning: index {row['index']} not found in dataset, skipping")
                continue

            gt_item = matching.iloc[0]
            gt_answer = str(gt_item['answer']).strip().upper()
            pred_answer = extract_answer(row['prediction'])

            category = gt_item.get('category', 'General')
            if category == 'General' and 'question_id' in gt_item:
                qid = str(gt_item['question_id'])
                if '__' in qid:
                    category = qid.split('__')[1].split('_')[0]

            category_stats[category]['total'] += 1
            if pred_answer == gt_answer:
                category_stats[category]['correct'] += 1

        print("\nEvaluation Results:")
        print(f"{'Category':<30}{'Accuracy':<15}{'Correct':<10}{'Total':<10}")
        print("=" * 65)

        overall_correct = 0
        overall_total = 0

        for category in sorted(category_stats.keys()):
            stats = category_stats[category]
            accuracy = stats['correct'] / stats['total'] if stats['total'] > 0 else 0.0
            results[category] = {'acc': accuracy, 'correct': stats['correct'], 'total': stats['total']}
            overall_correct += stats['correct']
            overall_total += stats['total']
            print(f"{category:<30}{accuracy:<15.4f}{stats['correct']:<10}{stats['total']:<10}")

        overall_acc = overall_correct / overall_total if overall_total > 0 else 0.0
        results['Overall'] = {'acc': overall_acc, 'correct': overall_correct, 'total': overall_total}
        print(f"{'Overall':<30}{overall_acc:<15.4f}{overall_correct:<10}{overall_total:<10}")

        results_file = get_intermediate_file_path(eval_file, '_results', 'json')
        acc_file = get_intermediate_file_path(eval_file, '_acc', 'csv')
        dump(results, results_file)
        acc_summary = {'Overall': overall_acc}
        for k, v in results.items():
            if k != 'Overall':
                acc_summary[k] = v['acc']

        dump(pd.DataFrame([acc_summary]), acc_file)

        csv_path = get_intermediate_file_path(eval_file, '_metrics', 'csv')
        with open(csv_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Category", "Accuracy", "Correct", "Total"])
            for category, values in results.items():
                writer.writerow([category, f"{values['acc']:.4f}", values['correct'], values['total']])

        print(f"\nResults saved to: {results_file}, {acc_file}, {csv_path}")
        return acc_summary


def extract_answer(text):
    """Last A-D match wins, so chain-of-thought traces don't bias to the
    first-mentioned letter. Falls back to the first A-D char if no \\b match."""
    if pd.isna(text):
        return None
    text = str(text).strip()
    pattern = r'\b[A-D]\b|\([A-D]\)'
    matches = re.findall(pattern, text, re.IGNORECASE)
    if matches:
        return matches[-1].strip("()").upper()
    for char in text.upper():
        if char in 'ABCD':
            return char
    return None
