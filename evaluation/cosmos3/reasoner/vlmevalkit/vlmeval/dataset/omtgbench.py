import os
import re
import zipfile
from typing import List, Tuple

import numpy as np
from huggingface_hub import snapshot_download
from scipy.optimize import linear_sum_assignment

from ..smp import LMUDataRoot, dump, get_cache_path, load
from .video_base import VideoBaseDataset


def parse_time_to_seconds(time_str: str) -> float:
    """Convert the time string into seconds"""
    time_str = time_str.strip()
    parts = time_str.split(":")

    if len(parts) == 1:
        return float(parts[0])
    elif len(parts) == 2:
        minutes, seconds = float(parts[0]), float(parts[1])
        return minutes * 60 + seconds
    elif len(parts) == 3:
        hours, minutes, seconds = float(parts[0]), float(parts[1]), float(parts[2])
        return hours * 3600 + minutes * 60 + seconds
    else:
        raise ValueError(f"Invalid time format: {time_str}")


def parse_time_intervals(text: str, strict: bool = False) -> List[List[float]]:
    """
    Parse all time intervals in the text and support multiple formats.
    Return: [[start, end], [start, end], ...] List of formats
    """
    intervals: List[List[float]] = []

    def add_interval(start_str: str, end_str: str, converter) -> None:
        try:
            start = converter(start_str.strip())
            end = converter(end_str.strip())
            if end > start:
                intervals.append([start, end])
        except (ValueError, TypeError):
            pass

    # 1: <time> 标签
    pattern_time_tag = r"<time>(\S+?)\s*-\s*(\S+?)\s*seconds?</time>"
    for s, e in re.findall(pattern_time_tag, text, re.IGNORECASE):
        add_interval(s, e, parse_time_to_seconds)
    if intervals:
        if strict:
            return sorted(intervals, key=lambda x: x[0])
        return sorted(intervals, key=lambda x: x[0])

    # 2: "X - Y seconds?"
    pattern_with_unit = r"(\d+(?::\d+(?:\.\d+)?)?(?:\.\d+)?)\s*-\s*(\d+(?::\d+(?:\.\d+)?)?(?:\.\d+)?)\s*seconds?"
    for s, e in re.findall(pattern_with_unit, text, re.IGNORECASE):
        add_interval(s, e, parse_time_to_seconds)
    if intervals:
        return sorted(intervals, key=lambda x: x[0])

    # 3: Other formats
    patterns = [
        (
            r"starts\s+at\s+(\S+?)(?:\s+seconds?)?\s+and\s+ends\s+at\s+(\S+?)(?:\s+seconds?)?",
            parse_time_to_seconds,
        ),
        (
            r"start\s+is\s+at\s+(\S+?)(?:\s+seconds?)?\s+and\s+(?:the\s+)?end\s+is\s+at\s+(\S+?)(?:\s+seconds?)?",
            parse_time_to_seconds,
        ),
        (r"(\d+(?:\.\d+)?)\s+to\s+(\d+(?:\.\d+)?)", float),
        (
            r"(?<!\d)(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)(?!\s*(?:seconds?|</time>))",
            float,
        ),
        (r"\[\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\]", float),
    ]

    for pattern, converter in patterns:
        for s, e in re.findall(pattern, text, re.IGNORECASE):
            add_interval(s, e, converter)
    unique_intervals = []
    seen = set()
    for interval in sorted(intervals, key=lambda x: x[0]):
        key = (interval[0], interval[1])
        if key not in seen:
            seen.add(key)
            unique_intervals.append(interval)
    return unique_intervals


def calculate_iou(seg1, seg2):
    """Calculating the IoU of two time segments"""
    s1, e1 = seg1
    s2, e2 = seg2
    inter = max(0, min(e1, e2) - max(s1, s2))
    union = (e1 - s1) + (e2 - s2) - inter
    return inter / union if union > 0 else 0.0


def merge_segments(segments):
    """Merge overlapping periods"""
    if not segments:
        return []
    sorted_segs = sorted(segments, key=lambda x: x[0])
    merged = [sorted_segs[0]]
    for s, e in sorted_segs[1:]:
        if s <= merged[-1][1]:  # 合并相邻或重叠
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return merged


def calculate_total_length(segments):
    """Calculate the total length of the time period after the merger"""
    return sum(e - s for s, e in segments)


def compute_one_to_many_metrics(
    pred_segments: List[Tuple[float, float]],
    gt_segments: List[Tuple[float, float]],
    iou_thresholds=[0.3, 0.5, 0.7],
):
    """
    Compute one-to-many matching metrics between predicted and ground truth segments.
    Args:
        pred_segments: List[[start, end]]
        gt_segments: List[[start, end]]
        iou_thresholds: List[float]

    Returns:
        dict: {
            "EtF1": float, "C-Acc": float, "tIoU": float,
            "tP@th": float, "tR@th": float,
            "tF1@th": float for each th in iou_thresholds
        }
    """
    c_acc = 1.0 if len(pred_segments) == len(gt_segments) else 0.0
    merged_pred = merge_segments(pred_segments)
    merged_gt = merge_segments(gt_segments)
    pred_len = calculate_total_length(merged_pred)
    gt_len = calculate_total_length(merged_gt)

    inter_len = sum(
        max(0, min(pe, ge) - max(ps, gs))
        for ps, pe in merged_pred
        for gs, ge in merged_gt
    )
    union_len = pred_len + gt_len - inter_len
    tIoU = inter_len / union_len if union_len > 0 else 0.0

    if len(pred_segments) == 0 or len(gt_segments) == 0:
        p_r_f1_value = 1.0 if len(pred_segments) == 0 and len(gt_segments) == 0 else 0.0
        results = {"C-Acc": c_acc, "tIoU": tIoU}
        for th in iou_thresholds:
            results.update(
                {
                    f"tP@{th}": p_r_f1_value,
                    f"tR@{th}": p_r_f1_value,
                    f"tF1@{th}": p_r_f1_value,
                }
            )
        results["EtF1"] = p_r_f1_value
        return results

    num_preds, num_gts = len(pred_segments), len(gt_segments)
    iou_matrix = np.array(
        [[calculate_iou(p, g) for g in gt_segments] for p in pred_segments]
    )
    pred_indices, gt_indices = linear_sum_assignment(-iou_matrix)

    results = {"C-Acc": c_acc, "tIoU": tIoU}
    matched_ious = [iou_matrix[i, j] for i, j in zip(pred_indices, gt_indices)]

    for th in iou_thresholds:
        tp = sum(1 for iou in matched_ious if iou >= th)
        precision = tp / num_preds
        recall = tp / num_gts
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        results.update({f"tP@{th}": precision, f"tR@{th}": recall, f"tF1@{th}": f1})
    EtF1 = c_acc * np.mean([results[f"tF1@{th}"] for th in iou_thresholds])
    results["EtF1"] = EtF1
    return results


class OMTGBench(VideoBaseDataset):
    TYPE = "Video-Temporal-Grounding"
    HF_REPO_ID = "insomnia7/omtg_bench"

    def __init__(self, dataset="OMTGBench", nframe=0, fps=2.0):
        super().__init__(dataset=dataset, nframe=nframe, fps=fps)

    @classmethod
    def supported_datasets(cls):
        return ["OMTGBench"]

    def prepare_dataset(self, dataset_name="OMTGBench"):
        cache_path = get_cache_path(self.HF_REPO_ID)
        if cache_path is None:
            cache_path = os.path.join(LMUDataRoot(), "OMTGBench")
        data_file = os.path.join(cache_path, f"{dataset_name}.tsv")
        video_root = os.path.join(cache_path, "videos")
        if not os.path.exists(data_file) or not os.path.exists(video_root):
            print(f"Downloading {dataset_name} from Hugging Face: {self.HF_REPO_ID}...")
            try:
                snapshot_download(
                    repo_id=self.HF_REPO_ID, repo_type="dataset", local_dir=cache_path
                )
            except Exception as e:
                print(f"Download failed: {e}")
                raise e

            zip_path = os.path.join(cache_path, "videos.zip")
            if os.path.exists(zip_path) and not os.path.exists(video_root):
                print(f"Extracting videos to {video_root}...")
                with zipfile.ZipFile(zip_path, "r") as zip_ref:
                    zip_ref.extractall(cache_path)

        return dict(root=video_root, data_file=data_file)

    def build_prompt(self, line, video_llm):
        if isinstance(line, int):
            line = self.data.iloc[line]
        message = []
        if video_llm:
            vid_path = os.path.join(self.data_root, line["video"])
            message.append(dict(type="video", value=vid_path))
        else:
            frames = self.save_video_frames(line["video"])
            for im in frames:
                message.append(dict(type="image", value=im))
        message.append(dict(type="text", value=line["question"]))
        return message

    def evaluate(self, eval_file, **judge_kwargs):
        data = load(eval_file)
        metrics = {
            "tIoU": [],
            "C-Acc": [],
            "EtF1": [],
        }
        iou_thresholds = [0.3, 0.5, 0.7]
        for th in iou_thresholds:
            metrics[f"tF1@{th}"] = []
            metrics[f"tP@{th}"] = []
            metrics[f"tR@{th}"] = []

        for i, row in data.iterrows():
            pred_text = str(row["prediction"])
            gt_text = str(row["answer"])
            pred_segs = parse_time_intervals(pred_text)
            gt_segs = parse_time_intervals(gt_text)
            print(
                f"Sample {i}: Predicted Segments: {pred_segs}, Ground Truth Segments: {gt_segs}"
            )
            one_to_many_metrics = compute_one_to_many_metrics(
                pred_segs, gt_segs, iou_thresholds=iou_thresholds
            )
            for k, v in one_to_many_metrics.items():
                metrics[k].append(v)

        results = {}
        for k, v in metrics.items():
            results[k] = np.mean(v) * 100  # to percentage
        print(f"Evaluation Results for {self.dataset_name}:")
        print(results)
        score_file = eval_file.replace(".xlsx", "_score.json")
        dump(results, score_file)
        return results
