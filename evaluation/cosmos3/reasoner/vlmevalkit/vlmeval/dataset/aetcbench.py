import functools
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from vlmeval.dataset.utils.lvs import bert_score
from ..smp import *
from .video_base import VideoBaseDataset
logger = get_logger('AETCBench')

# All recognized task types and their filename patterns
TASK_TYPES = [
    'bcq', 'mcq', 'open_qa',
    'temporal_localization', 'causal_linkage',
    'scene_description', 'temporal_description', 'video_summarization',
    'bcq_openended', 'mcq_openended',
]

# DSS dataset names
DSS_TASKS_DATASET = 'AETC-Tasks'
DSS_VIDEOS_DATASET = 'AETC-Videos'


# Answer-format instructions per task type (appended to user query)
_ANSWER_INSTRUCTIONS = {
    'bcq': 'Answer with only Yes or No.',
    'bcq_openended': 'Answer with Yes or No, followed by a brief explanation.',
    'mcq': 'Choose the correct option by letter only.',
    'mcq_openended': 'Choose the correct option and provide a brief explanation.',
    "temporal_localization": "Provide the result in json format with 'mm:ss' for time depiction. Use keywords 'start', 'end' in the json output.",
}


def _build_user_query(task_type, item):
    """Build the user-facing prompt string from a task item.

    Mirrors the prompt format used in training (_build_conversation).
    Ground-truth answer and reasoning are intentionally excluded.
    """
    question = item.get('question', '')

    # Append MCQ options
    if task_type in ('mcq', 'mcq_openended'):
        options = item.get('options')
        if options:
            options_text = '\n'.join(f'{k}) {v}' for k, v in sorted(options.items()))
            question = f'{question}\n\n{options_text}'

    # Append answer-format instruction
    instruction = _ANSWER_INSTRUCTIONS.get(task_type)
    if instruction:
        question = f'{question}\n\n{instruction}'

    return question


def _format_reference_answer(task_type, item):
    """Format the ground-truth answer string for evaluation.

    Mirrors _format_answer from training code.
    """
    if task_type in (
        'open_qa', 'bcq_openended', 'mcq_openended',
        'video_summarization', 'scene_description',
        'temporal_description', 'causal_linkage',
    ):
        return item.get('answer') or ''

    if task_type == 'bcq':
        answer = item.get('answer', '')
        explanation = item.get('explanation', '')
        return f'{answer}. {explanation}' if explanation else answer

    if task_type == 'mcq':
        letter = item.get('answer', '')
        options = item.get('options', {})
        label = f'{letter}) {options[letter]}' if letter in options else letter
        explanation = item.get('explanation', '')
        return f'{label}. {explanation}' if explanation else label

    if task_type == 'temporal_localization':
        answer = item.get('answer')
        if answer:
            return json.dumps(answer)
        return ''

    return ''

class Evaluator:
    """Reference-based text metrics: BLEU, ROUGE, METEOR, BERTScore."""
    # NOTE: pending finalizing the metrics
    # EVAL_METRICS = ['bertscore', 'bleu', 'rouge', 'meteor']
    EVAL_METRICS = ['bertscore']
    def __init__(self):
        import evaluate
        if 'bleu' in self.EVAL_METRICS:
            self.bleu_metric = evaluate.load("bleu")
        if 'rouge' in self.EVAL_METRICS:
            self.rouge_metric = evaluate.load("rouge")
        if 'meteor' in self.EVAL_METRICS:
            self.meteor_metric = evaluate.load("meteor")
        if 'bertscore' in self.EVAL_METRICS:
            self.bertscore_metric = evaluate.load("bertscore")

    def __call__(self, references, candidates):
        results = {}
        if 'bleu' in self.EVAL_METRICS:
            bleu = self.bleu_metric.compute(predictions=candidates, references=references)
            results['bleu'] = bleu['bleu']
        if 'rouge' in self.EVAL_METRICS:
            rouge = self.rouge_metric.compute(predictions=candidates, references=references)
            results['rouge1'] = rouge['rouge1']
            results['rouge2'] = rouge['rouge2']
            results['rougeL'] = rouge['rougeL']
        if 'meteor' in self.EVAL_METRICS:
            meteor = self.meteor_metric.compute(predictions=candidates, references=references)
            results['meteor'] = meteor['meteor']    
        if 'bertscore' in self.EVAL_METRICS:
            bertscore = self.bertscore_metric.compute(predictions=candidates, references=references, lang='en', rescale_with_baseline=True)
            results['bertscore_f1'] = float(np.mean(bertscore['f1']))
            results['bertscore_precision'] = float(np.mean(bertscore['precision']))
            results['bertscore_recall'] = float(np.mean(bertscore['recall']))
        return results

def _parse_subdataset_from_path(rel_path):
    """Extract the top-level subdataset name from a relative path under AETC-Tasks/.

    e.g. 'so-tad/test/2045/task/bcq_aetc.json' -> 'so-tad'
    """
    parts = Path(rel_path).parts
    return parts[0] if parts else 'unknown'

def _preprocess_video(video_path, fps, max_pixels_per_frame, cache_dir):
    """Re-encode a video at target fps and resolution, caching the result.

    Args:
        video_path: Path to source video.
        fps: Target frames per second.
        max_pixels_per_frame: Max pixels (w*h) per frame. The video is
            scaled down (preserving aspect ratio) so that w*h <= this value.
            If None, no scaling is applied.
        cache_dir: Directory to store preprocessed videos.

    Returns:
        Path to the preprocessed video (str). Returns the original path
        if preprocessing is disabled (fps=None and max_pixels_per_frame=None).
    """
    import hashlib
    import subprocess

    if fps is None and max_pixels_per_frame is None:
        return video_path

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Deterministic cache key from source path + params
    key_str = f'{video_path}|fps={fps}|maxpix={max_pixels_per_frame}'
    key = hashlib.sha256(key_str.encode()).hexdigest()[:16]
    cache_path = cache_dir / f'{key}.mp4'
    if cache_path.exists():
        return str(cache_path)

    # Build ffmpeg filter chain
    vf_filters = []
    if fps is not None:
        vf_filters.append(f'fps={fps}')
    if max_pixels_per_frame is not None:
        # Scale down so w*h <= max_pixels_per_frame, preserving aspect ratio.
        # Use expression: if(gt(iw*ih, max), scale to fit, keep original)
        # sqrt(max / (iw*ih)) gives the uniform scale factor.
        mp = int(max_pixels_per_frame)
        vf_filters.append(
            f"scale='if(gt(iw*ih,{mp}),trunc(iw*sqrt({mp}/(iw*ih))/2)*2,iw)'"
            f":'if(gt(iw*ih,{mp}),trunc(ih*sqrt({mp}/(iw*ih))/2)*2,ih)'"
        )

    cmd = ['ffmpeg', '-i', str(video_path)]
    if vf_filters:
        cmd += ['-vf', ','.join(vf_filters)]
    cmd += [
        '-c:v', 'libx264',
        '-crf', '23',
        '-preset', 'fast',
        '-pix_fmt', 'yuv420p',
        '-an',  # drop audio
        '-threads', '1',
        '-loglevel', 'error',
        '-y',
        str(cache_path),
    ]
    subprocess.run(cmd, check=True)
    return str(cache_path)




class AETCScorer:
    """All scoring logic for AETCBench, separated for readability."""

    # Configurable weights for the overall score.
    # Keys must match the metric names returned by _eval_* methods.
    # Weights are normalized to sum to 1 at scoring time, so relative
    # magnitudes are what matter. Set a weight to 0 to exclude a metric.
    # Text-metric tasks get 4 sub-metrics (bertscore_f1/bleu/meteor/rougeL)
    # each at 0.25 so the task contributes 1.0 total, matching single-metric tasks.
    METRIC_WEIGHTS = {
        # NOTE: pending finalizing the weights
        'bcq_accuracy': 1.0,
        'mcq_accuracy': 1.0,
        'temporal_localization_miou': 1.0,
        'bcq_openended_bertscore_f1': 1.0,
        # 'bcq_openended_bleu': 1.0,
        # 'bcq_openended_meteor': 1.0,
        # 'bcq_openended_rougeL': 1.0,
        'mcq_openended_bertscore_f1': 1.0,
        # 'mcq_openended_bleu': 1.0,
        # 'mcq_openended_meteor': 1.0,
        # 'mcq_openended_rougeL': 1.0,
        'open_qa_bertscore_f1': 1.0,
        # 'open_qa_bleu': 1.0,
        # 'open_qa_meteor': 1.0,
        # 'open_qa_rougeL': 1.0,
        'causal_linkage_bertscore_f1': 1.0,
        # 'causal_linkage_bleu': 1.0,
        # 'causal_linkage_meteor': 1.0,
        # 'causal_linkage_rougeL': 1.0,
        'scene_description_bertscore_f1': 1.0,
        # 'scene_description_bleu': 1.0,
        # 'scene_description_meteor': 1.0,
        # 'scene_description_rougeL': 1.0,
        'temporal_description_bertscore_f1': 1.0,
        # 'temporal_description_bleu': 1.0,
        # 'temporal_description_meteor': 1.0,
        # 'temporal_description_rougeL': 1.0,
        'video_summarization_bertscore_f1': 1.0,
        # 'video_summarization_bleu': 1.0,
        # 'video_summarization_meteor': 1.0,
        # 'video_summarization_rougeL': 1.0,
    }

    EVAL_DISPATCH = {
        'bcq': '_eval_bcq',
        'bcq_openended': '_eval_bcq_openended',
        'mcq': '_eval_mcq',
        'mcq_openended': '_eval_mcq_openended',
        'open_qa': '_eval_open_qa',
        'temporal_localization': '_eval_temporal_localization',
        'causal_linkage': '_eval_causal_linkage',
        'scene_description': '_eval_scene_description',
        'temporal_description': '_eval_temporal_description',
        'video_summarization': '_eval_video_summarization',
    }

    def __init__(self, evaluator):
        self.evaluator = evaluator

    def score(self, data, **judge_kwargs):
        """Score all task types present in data.

        Args:
            data: pd.DataFrame with at least these columns:
                - task_type:  str, one of TASK_TYPES (e.g. 'bcq', 'mcq', ...)
                - prediction: str, raw model output text
                - answer:     str, formatted ground-truth (from _format_reference_answer)
            **judge_kwargs: forwarded to per-task evaluators (reserved for LLM judge config).

        Returns:
            dict[str, float]: flat mapping of metric_name -> score, e.g.
                {'bcq_accuracy': 0.85, 'mcq_accuracy': 0.72, ...}
        """
        metrics = {}
        for tt in data['task_type'].unique():
            subset = data[data['task_type'] == tt]
            method_name = self.EVAL_DISPATCH.get(tt)
            if method_name is None:
                logger.warning(f'No evaluator for task type {tt!r}, skipping')
                continue
            eval_fn = getattr(self, method_name)
            sub_metrics = eval_fn(subset, **judge_kwargs)
            metrics.update(sub_metrics)

        # Weighted overall score
        weights, scores = zip(*[
            (weight, metrics[name])
            for name, weight in self.METRIC_WEIGHTS.items()
            if weight > 0 and name in metrics])
        metrics['weighted_mean'] = np.average(scores, weights=weights)
   
        metrics['mean'] = np.mean(scores)

        return metrics

    # ---- extraction helpers ----

    @staticmethod
    def _extract_yesno_and_explanation(text):
        """Extract (yes_or_no, explanation) from free-form text.

        Returns:
            tuple: (answer, explanation) where answer is 'yes'/'no'/None
                   and explanation is a string or None.
        """
        if pd.isna(text) or not str(text).strip():
            return None, None
        text = str(text).strip()
        text_lower = text.lower()
        # Try leading yes/no followed by separator and optional explanation
        m = re.match(r'^(yes|no)\b[.,;:!\s]*(.*)$', text_lower, re.DOTALL)
        if m:
            answer = m.group(1)
            explanation = m.group(2).strip() or None
            return answer, explanation
        # Fallback: search anywhere
        m = re.search(r'\b(yes|no)\b', text_lower)
        if m:
            return m.group(1), None
        return None, None

    @staticmethod
    def _extract_letter_and_explanation(text):
        """Extract (letter, explanation) from free-form text.

        Returns:
            tuple: (letter, explanation) where letter is 'A'-'D'/None
                   and explanation is a string or None.
        """
        if pd.isna(text) or not str(text).strip():
            return None, None
        text = str(text).strip()
        # Leading patterns: "A", "A)", "A.", "(A)", "A:"
        m = re.match(r'^\(?([A-Za-z])\)?[).\s,:]+(.*)$', text, re.DOTALL)
        if m:
            letter = m.group(1).upper()
            explanation = m.group(2).strip() or None
            return letter, explanation
        # Single letter only
        m = re.match(r'^([A-Da-d])$', text.strip())
        if m:
            return m.group(1).upper(), None
        # Fallback: standalone A-D anywhere
        m = re.search(r'\b([A-D])\b', text)
        if m:
            return m.group(1).upper(), None
        return None, None

    @staticmethod
    def _gt_yesno(answer_str):
        """Extract ground-truth yes/no. Errors if GT doesn't start with Yes or No."""
        assert answer_str and str(answer_str).strip(), \
            f'GT answer is empty or missing: {answer_str!r}'
        first_word = str(answer_str).strip().lower().split('.')[0].split()[0]
        assert first_word in ('yes', 'no'), \
            f'GT answer does not start with Yes/No: {answer_str!r}'
        return first_word

    @staticmethod
    def _gt_letter(answer_str):
        """Extract ground-truth letter (e.g. 'D) ...'). Errors if GT doesn't match."""
        assert answer_str and str(answer_str).strip(), \
            f'GT answer is empty or missing: {answer_str!r}'
        m = re.match(r'^([A-Za-z])\)', str(answer_str).strip())
        assert m, f'GT answer does not match letter) format: {answer_str!r}'
        return m.group(1).upper()

    @staticmethod
    def _assert_nonempty_references(references, task_name):
        """Assert all GT references are non-empty strings."""
        for i, r in enumerate(references):
            assert r and str(r).strip(), \
                f'{task_name}: GT answer at index {i} is empty or missing: {r!r}'

    def _eval_text_metrics(self, data, task_name):
        """Run the reference-based evaluator and return bertscore_f1/bleu/meteor/rougeL."""
        references = data['answer'].tolist()
        candidates = data['prediction'].tolist()
        self._assert_nonempty_references(references, task_name)
        raw = self.evaluator(references, candidates)
        ret = {}
        if 'bertscore' in self.evaluator.EVAL_METRICS:
            ret[f'{task_name}_bertscore_f1'] = raw['bertscore_f1']
        if 'bleu' in self.evaluator.EVAL_METRICS:
            ret[f'{task_name}_bleu'] = raw['bleu']
        if 'meteor' in self.evaluator.EVAL_METRICS:
            ret[f'{task_name}_meteor'] = raw['meteor']
        if 'rougeL' in self.evaluator.EVAL_METRICS:
            ret[f'{task_name}_rougeL'] = raw['rougeL']
        return ret

    # ---- per-task evaluators ----

    def _eval_bcq(self, data, **judge_kwargs):
        """BCQ: exact yes/no match -> accuracy."""
        correct, total = 0, 0
        for _, row in data.iterrows():
            pred, _ = self._extract_yesno_and_explanation(row['prediction'])
            gt = self._gt_yesno(row['answer'])
            total += 1
            if pred == gt:
                correct += 1
        acc = correct / total if total > 0 else 0.0
        return {'bcq_accuracy': acc}

    def _eval_bcq_openended(self, data, **judge_kwargs):
        return self._eval_text_metrics(data, 'bcq_openended')

    def _eval_mcq(self, data, **judge_kwargs):
        """MCQ: exact letter match -> accuracy."""
        correct, total = 0, 0
        for _, row in data.iterrows():
            pred, _ = self._extract_letter_and_explanation(row['prediction'])
            gt = self._gt_letter(row['answer'])
            total += 1
            if pred == gt:
                correct += 1
        acc = correct / total if total > 0 else 0.0
        return {'mcq_accuracy': acc}

    def _eval_mcq_openended(self, data, **judge_kwargs):
        return self._eval_text_metrics(data, 'mcq_openended')

    def _eval_open_qa(self, data, **judge_kwargs):
        return self._eval_text_metrics(data, 'open_qa')

    @staticmethod
    def _parse_timestamp_to_seconds(ts):
        """Convert a MM:SS or HH:MM:SS timestamp string to seconds."""
        ts = str(ts).strip()
        parts = ts.split(':')
        if len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        elif len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        return float(ts)

    @staticmethod
    def _extract_json_from_text(text):
        """Extract JSON from text: try ```json ... ``` fenced block first, then direct parse."""
        if pd.isna(text) or not str(text).strip():
            return None
        text = str(text).strip()
        # Try fenced ```json ... ``` block
        m = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
        if m:
            try:
                json_object = json.loads(m.group(1))
                if (
                    isinstance(json_object, list)
                    and len(json_object) > 0
                    and isinstance(json_object[0], dict)
                    and json_object[0].get('start') is not None
                    and json_object[0].get('end') is not None
                ):
                    return json_object[0]
                else:
                    return json_object
            except json.JSONDecodeError:
                pass
        # Try direct JSON parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        return None

    def _eval_temporal_localization(self, data, **judge_kwargs):
        """Temporal localization: mean IoU over predictions that follow the output format.

        Predictions that can't be parsed as {"start": ..., "end": ...} are skipped
        (not counted as IoU=0) so the metric reflects format-compliant performance only.
        The number of skipped predictions is logged for visibility.
        """
        ious = []
        n_parse_fail = 0
        for _, row in data.iterrows():
            # Parse GT (stored as JSON string from _format_reference_answer)
            gt = self._extract_json_from_text(row['answer'])
            if gt is None:
                logger.warning(f'Failed to parse GT for temporal_localization: {row["answer"]!r}')
                continue
            # Parse prediction — skip entirely if non-compliant
            pred = self._extract_json_from_text(row['prediction'])
            if pred is None or 'start' not in pred or 'end' not in pred:
                n_parse_fail += 1
                continue
            try:
                gt_start = self._parse_timestamp_to_seconds(gt['start'])
                gt_end = self._parse_timestamp_to_seconds(gt['end'])
                pred_start = self._parse_timestamp_to_seconds(pred['start'])
                pred_end = self._parse_timestamp_to_seconds(pred['end'])
            except (KeyError, ValueError, TypeError):
                n_parse_fail += 1
                continue
            # Compute IoU
            inter_start = max(gt_start, pred_start)
            inter_end = min(gt_end, pred_end)
            intersection = max(0.0, inter_end - inter_start)
            union = max(0.0, (gt_end - gt_start) + (pred_end - pred_start) - intersection)
            iou = intersection / union if union > 0 else 0.0
            ious.append(iou)
        if n_parse_fail > 0:
            logger.warning(
                f'temporal_localization: {n_parse_fail}/{len(data)} predictions skipped (unparseable)'
            )
        miou = float(np.mean(ious)) if ious else 0.0
        return {'temporal_localization_miou': miou}

    def _eval_causal_linkage(self, data, **judge_kwargs):
        return self._eval_text_metrics(data, 'causal_linkage')

    def _eval_scene_description(self, data, **judge_kwargs):
        return self._eval_text_metrics(data, 'scene_description')

    def _eval_temporal_description(self, data, **judge_kwargs):
        return self._eval_text_metrics(data, 'temporal_description')

    def _eval_video_summarization(self, data, **judge_kwargs):
        return self._eval_text_metrics(data, 'video_summarization')


class AETCBench(VideoBaseDataset):
    TYPE = "VQA"
    def __init__(
        self,
        dataset='AETCBench',
        split='test',
        task='all',
        nframe=0,
        fps=4,
        total_pixels=8192 * 32 * 32,
        max_pixels=None,
        max_frames=None,
        preprocess_fps=None,
        preprocess_max_pixels=None,
    ):
        self.split = split
        self.task = task
        self.total_pixels = total_pixels
        self.max_pixels = max_pixels
        self.max_frames = max_frames
        self.preprocess_fps = preprocess_fps
        self.preprocess_max_pixels = preprocess_max_pixels
        super().__init__(
            dataset=dataset,
            nframe=nframe,
            fps=fps,
            total_pixels=total_pixels,
        )

    @classmethod
    def supported_datasets(cls):
        return ['AETCBench']

    # ------------------------------------------------------------------
    # Data download from DSS
    # ------------------------------------------------------------------

    def _download_from_dss(self, local_root_dir: Path):
        """Download AETC-Tasks and AETC-Videos from DSS with filtering.

        Uses nvdataset SDK.  Only downloads files matching the requested
        task type and split to avoid pulling the full dataset.

        Both datasets share the same scene directory structure so they
        overlay into a single tree:
            local_root_dir/{subdataset}/{scene_path}/raw/main.mp4
            local_root_dir/{subdataset}/{scene_path}/task/*.json
        """
        from nvdataset import NVDatasetClient
        from nvdataset.types import Filter, FilterOperator, Field

        if os.environ.get('NVDATASET_TENANTID') is None:
            raise ValueError('NVDATASET_TENANTID env var is not set')
        if os.environ.get('NGC_API_KEY') is None:
            raise ValueError('NGC_API_KEY env var is not set')

        client = NVDatasetClient()
        local_root_dir.mkdir(parents=True, exist_ok=True)

        # --- Download task annotations (filtered by task type) ---
        print(f'Downloading {DSS_TASKS_DATASET} (task={self.task}) ...')
        ds_tasks = client.load_dataset(DSS_TASKS_DATASET)
        ds_tasks.cache_local(
            str(local_root_dir),
            filters=[
                Filter(op=FilterOperator.CONTAINS, field=Field(name='key', value=f'ITS_Collision_Verification')),
                Filter(op=FilterOperator.CONTAINS, field=Field(name='key', value=f'_gemma4.json')), # currently, hacking with using gemma4
            ]
        )

        # --- Download videos for scenes that have task files ---
        # Collect the scene prefixes from downloaded tasks to filter videos
        scene_prefixes = set()
        for task_file in local_root_dir.rglob('*/task/*.json'):
            scene_dir = task_file.parent.parent
            scene_prefixes.add(scene_dir.relative_to(local_root_dir).as_posix())

        print(f'Downloading {DSS_VIDEOS_DATASET} for {len(scene_prefixes)} scenes ...')
        ds_videos = client.load_dataset(DSS_VIDEOS_DATASET)
        # Download scene by scene to avoid pulling the entire video dataset
        for prefix in sorted(scene_prefixes):
            video_filters = [
                Filter(op=FilterOperator.STARTS_WITH, field=Field(name='key', value=f'{prefix}/'))
            ]
            ds_videos.cache_local(str(local_root_dir), filters=video_filters)

        print(f'DSS download complete -> {local_root_dir}')

    # ------------------------------------------------------------------
    # Dataset preparation
    # ------------------------------------------------------------------

    def prepare_dataset(self, dataset_name='AETCBench'):
        cache_dir = LMUDataRoot()
        dataset_dir = Path(cache_dir) / 'videos' / 'AETCBench'
        dataset_dir.mkdir(parents=True, exist_ok=True)

        # Single merged tree from DSS download
        downloaded_dir = dataset_dir / 'AETC'
        if not downloaded_dir.exists():
            self._download_from_dss(downloaded_dir)
        tasks_root = downloaded_dir
        videos_root = downloaded_dir

        # Build the data table by walking task JSONs
        data_file = dataset_dir / f'{dataset_name}_{self.split}_{self.task}.tsv'
        if not data_file.exists():
            rows = self._walk_task_files(tasks_root, videos_root)
            if len(rows) == 0:
                raise RuntimeError(
                    f'No task items found under {tasks_root} '
                    f'for task={self.task}, split={self.split}'
                )
            df = pd.DataFrame(rows)
            df.to_csv(data_file, sep='\t', index=False)
            print(f'Built {len(df)} items -> {data_file}')
        else:
            print(f'Reusing cached data file {data_file}')

        return dict(root=str(videos_root), data_file=str(data_file))


    def _walk_task_files(self, tasks_root: Path, videos_root: Path):
        """Recursively find task JSONs and pair them with videos.

        Supports two layouts:
          - Merged: tasks_root == videos_root, each scene has raw/ + task/
          - Separate: tasks_root has task/ dirs, videos_root has raw/ dirs,
                      with matching relative paths.
        """
        rows = []
        # Walk task directories — look for any dir named 'task' at any depth
        all_task = sorted(list(tasks_root.rglob('task')))
        for task_dir in tqdm(all_task, desc="Walking task directories"):
            if not task_dir.is_dir():
                continue

            scene_dir_in_tasks = task_dir.parent
            scene_id = scene_dir_in_tasks.relative_to(tasks_root).as_posix()

            # Resolve video: look in videos_root at the same relative path
            raw_video_path = videos_root / scene_id / 'raw' / 'main.mp4'
            if not raw_video_path.exists():
                continue

            # Preprocess video if requested (resample fps, resize)
            if self.preprocess_fps is not None or self.preprocess_max_pixels is not None:
                preprocess_cache = Path(LMUDataRoot()) / 'videos' / 'AETCBench' / 'preprocessed'
                video_path = _preprocess_video(
                    str(raw_video_path),
                    fps=self.preprocess_fps,
                    max_pixels_per_frame=self.preprocess_max_pixels,
                    cache_dir=preprocess_cache,
                )
            else:
                video_path = str(raw_video_path)

            subdataset = _parse_subdataset_from_path(scene_id)

            # Filter task files by requested task type
            if self.task == 'all':
                task_files = sorted(task_dir.glob('*.json'))
            else:
                task_files = sorted(task_dir.glob(f'{self.task}_*.json'))

            for task_file in task_files:
                try:
                    with open(task_file, 'r') as f:
                        task_data = json.load(f)
                except (json.JSONDecodeError, OSError):
                    continue

                task_type = task_data.get('metadata', {}).get('type')
                if task_type is None:
                    print(f'Warning: no metadata.type in {task_file}, skipping')
                    continue
                if task_type not in TASK_TYPES:
                    print(f'Warning: unrecognized task type {task_type!r} in {task_file}, skipping')
                    continue

                items = task_data.get('items', [])
                metadata = task_data.get('metadata', {})
                # Derive annotation source from filename: bcq_aetc.json -> aetc
                stem = task_file.stem
                source = stem.rsplit('_', 1)[-1] if '_' in stem else 'unknown'

                for item_idx, item in enumerate(items):
                    user_query = _build_user_query(task_type, item)
                    answer = _format_reference_answer(task_type, item)
                    # reference_data: per-item dict with the item fields + file metadata
                    ref = dict(item=item, metadata=metadata)
                    rows.append(dict(
                        index=f'{scene_id}/{stem}#{item_idx}',
                        video=video_path,
                        question=user_query,
                        user_query=user_query,
                        answer=answer,
                        reference_data=json.dumps(ref),
                        task_type=task_type,
                        subdataset=subdataset,
                        source=source,
                        item_idx=item_idx,
                    ))

        print(f'Discovered {len(rows)} items across {len(set(r["subdataset"] for r in rows))} subdatasets')
        return rows

    # ------------------------------------------------------------------
    # Prompt building
    # ------------------------------------------------------------------

    def build_prompt(self, line, video_llm=True):
        if isinstance(line, int):
            line = self.data.iloc[line]

        video_path = line['video']
        user_query = line['user_query']

        msgs = []
        if video_llm:
            process_video_kwargs = {
                k: v for k, v in dict(
                    fps=self.fps,
                    total_pixels=self.total_pixels,
                    max_pixels=self.max_pixels,
                    max_frames=self.max_frames,
                ).items() if v is not None
            }
            if self.nframe > 0:
                process_video_kwargs['nframes'] = self.nframe
            msgs.append(dict(type='video', value=video_path, **process_video_kwargs))
        else:
            frames = self.save_video_frames(video_path)
            msgs.extend([dict(type='image', value=f) for f in frames])

        msgs.append(dict(type='text', value=user_query))
        return msgs

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    @functools.cached_property
    def evaluator(self):
        return Evaluator()

    def evaluate(self, eval_file, **judge_kwargs):
        data = load(eval_file)
        scorer = AETCScorer(evaluator=self.evaluator)
        metrics = scorer.score(data, **judge_kwargs)
        summary = pd.DataFrame([metrics])
        score_file = get_intermediate_file_path(eval_file, '_acc', 'csv')
        dump(summary, score_file)
        return summary

