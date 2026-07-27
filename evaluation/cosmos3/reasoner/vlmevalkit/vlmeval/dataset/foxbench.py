from __future__ import annotations
import json
import re
from pathlib import Path
from typing import Dict, List

import jieba
import nltk
import numpy as np
from nltk.metrics import f_measure, precision, recall
from nltk.translate import meteor_score
from rouge import Rouge

from vlmeval.smp import dump, load
from vlmeval.smp.file import get_intermediate_file_path
from vlmeval.smp.log import get_logger
from .image_base import ImageBaseDataset

logger = get_logger(__name__)


def _contain_chinese(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fa5]", text))


def _cal_ocr_metrics(pred: str, gt: str) -> Dict[str, float]:
    if _contain_chinese(gt) or _contain_chinese(pred):
        reference = jieba.lcut(gt)
        hypothesis = jieba.lcut(pred)
    else:
        reference = gt.split()
        hypothesis = pred.split()

    metrics = {
        "bleu": nltk.translate.bleu([reference], hypothesis),
        "meteor": meteor_score.meteor_score([reference], hypothesis),
    }

    reference_set = set(reference)
    hypothesis_set = set(hypothesis)
    metrics["f_measure"] = f_measure(reference_set, hypothesis_set)
    metrics["precision"] = precision(reference_set, hypothesis_set)
    metrics["recall"] = recall(reference_set, hypothesis_set)
    # Calculate edit distance normalized by max length, matching official implementation
    metrics["edit_dist"] = nltk.edit_distance(pred, gt) / max(len(pred), len(gt), 1)
    return metrics


def _cal_summary_metrics(pred: str, gt: str) -> Dict[str, float]:
    """Calculate ROUGE-L metrics, matching official implementation."""
    rouge = Rouge()
    # Official code: rouge.get_scores(hyps=hypothesis, refs=reference)
    # where hypothesis and reference are space-joined
    reference = ' '.join(gt.split())
    hypothesis = ' '.join(pred.split())
    res = rouge.get_scores(hyps=hypothesis, refs=reference)
    rl = res[0]["rouge-l"]
    return {"rouge_l_r": rl["r"], "rouge_l_p": rl["p"], "rouge_l_f": rl["f"]}


def _cal_qa_metric(pred: str, gt: str) -> Dict[str, float]:
    return {"accuracy": 1.0 if pred == gt else 0.0}


def _classify(question: str) -> str:
    """Classify task type based on question text.
    Following official implementation:
    - QA: questions about comparing or identifying content
    - Summary: summarization tasks
    - Translation: translation tasks (evaluated with OCR metrics)
    - OCR: all other tasks (default)
    """
    q = _normalize_question(question)
    q_lower = q.lower()
    # QA tasks: questions that expect specific answers
    if "which page's box contains more characters" in q_lower:
        return "qa"
    if "what is this in the box" in q_lower:
        return "qa"

    # Summary tasks
    if "summarize" in q_lower:
        return "summary"

    # Translation tasks (evaluated with OCR metrics per official code)
    if "translate" in q_lower:
        return "translation"

    # Default: OCR tasks (including box OCR, line OCR, page OCR, etc.)
    return "ocr"


def _normalize_question(question: str) -> str:
    if question is None:
        return ""
    parts = [segment.strip() for segment in str(question).strip().split("\n") if segment.strip()]
    return parts[-1] if parts else ""


class FoxBench(ImageBaseDataset):

    TYPE = "QA"
    MODALITY = "IMAGE"
    DATASET_URL = {'FoxBench': 'https://huggingface.co/datasets/EasonFan/fox_benchmark.tsv/resolve/main/fox_benchmark_data.tsv'}  # noqa: E501
    DATASET_MD5 = {'FoxBench': '153b88812ad58495558326bab9c90f8e'}

    def load_data(self, dataset):
        repo_root = Path(__file__).resolve().parents[4]
        candidates = [repo_root / "fox_benchmark_data.tsv", repo_root / "fox_bench.tsv"]
        for path in candidates:
            if path.exists():
                self.data_path = str(path)
                return load(path)

        data = super().load_data(dataset)
        self.data_path = getattr(self, "data_path", None)
        return data

    def build_prompt(self, line):
        return super().build_prompt(line)

    def evaluate(self, eval_file, **judge_kwargs):
        preds = load(eval_file)
        assert "prediction" in preds, "Prediction file must contain a `prediction` column"

        if not hasattr(self, "data_path") or self.data_path is None:
            self.load_data(self.dataset_name)
        refs = load(self.data_path)

        # Include sub_task if available in the data
        if "sub_task" in refs:
            ref_map = {
                str(idx): (q, ans, st)
                for idx, q, ans, st in zip(
                    refs["index"], refs["question"], refs["answer"], refs["sub_task"]
                )
            }
        else:
            ref_map = {
                str(idx): (q, ans, None)
                for idx, q, ans in zip(refs["index"], refs["question"], refs["answer"])
            }

        ocr_metrics: List[Dict[str, float]] = []
        translation_metrics: List[Dict[str, float]] = []
        summary_metrics: List[Dict[str, float]] = []
        qa_metrics: List[Dict[str, float]] = []
        details: List[Dict[str, object]] = []

        # Sub-task specific metrics
        sub_task_metrics: Dict[str, List[Dict[str, float]]] = {}

        # Evaluate each prediction against ground truth
        for idx, pred in zip(preds["index"], preds["prediction"]):
            idx_str = str(idx)
            if idx_str not in ref_map:
                continue

            ref_data = ref_map[idx_str]
            question, gt = ref_data[0], ref_data[1]
            sub_task = ref_data[2] if len(ref_data) > 2 else None

            normalized_question = _normalize_question(question)
            pred_text = "" if pred is None else str(pred)
            task = _classify(normalized_question)

            # Calculate metrics based on task type (matching official eval scripts)
            if task == "qa":
                # eval_qa_test.py: simple accuracy check
                m = _cal_qa_metric(pred_text, gt)
                qa_metrics.append(m)
            elif task == "summary":
                # eval_summary_test.py: ROUGE-L metrics
                m = _cal_summary_metrics(pred_text, gt)
                summary_metrics.append(m)
            elif task == "translation":
                # Translation evaluated with OCR metrics per official code
                m = _cal_ocr_metrics(pred_text, gt)
                translation_metrics.append(m)
            else:  # task == "ocr"
                # eval_ocr_test.py: BLEU, METEOR, F1, Precision, Recall, Edit Distance
                m = _cal_ocr_metrics(pred_text, gt)
                ocr_metrics.append(m)

            # Accumulate sub-task metrics if available
            if sub_task:
                if sub_task not in sub_task_metrics:
                    sub_task_metrics[sub_task] = []
                sub_task_metrics[sub_task].append(m)

            details.append({
                "index": idx_str,
                "task": task,
                "sub_task": sub_task,
                "question": normalized_question,
                "gt": gt,
                "pred": pred_text,
                **m,
            })

        def _avg(dicts: List[Dict[str, float]]) -> Dict[str, float]:
            if not dicts:
                return {}
            keys = dicts[0].keys()
            return {k: float(np.mean([d[k] for d in dicts])) for k in keys}

        # Calculate sub-task averages
        sub_task_scores = {st: _avg(metrics) for st, metrics in sub_task_metrics.items()}
        sub_task_counts = {st: len(metrics) for st, metrics in sub_task_metrics.items()}

        scores = {
            "ocr": _avg(ocr_metrics),
            "translation": _avg(translation_metrics),
            "summary": _avg(summary_metrics),
            "qa": _avg(qa_metrics),
            "counts": {
                "ocr": len(ocr_metrics),
                "translation": len(translation_metrics),
                "summary": len(summary_metrics),
                "qa": len(qa_metrics),
                "total": len(details),
            },
            "sub_tasks": sub_task_scores,
            "sub_task_counts": sub_task_counts,
        }

        score_file = get_intermediate_file_path(eval_file, "_foxbench_score", "json")
        dump(scores, score_file)
        detail_file = get_intermediate_file_path(eval_file, "_foxbench_detail", "json")
        dump(details, detail_file)
        logger.info(f'FoxBench successfully finished evaluating {eval_file}')
        logger.info(f'Results saved in {score_file}')
        logger.info('Score Summary:')
        logger.info(json.dumps(scores, ensure_ascii=False, indent=2))

        return scores
