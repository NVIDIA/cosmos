from typing import Any, Dict, List, Optional, Sequence, Tuple, Iterable
import json
import math
from datetime import datetime
from collections import defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment

# Suppress HuggingFace transformers warnings about uninitialized weights
# (RoBERTa pooler weights are not used by BERTScore)
import transformers
transformers.logging.set_verbosity_error()

from bert_score import score as bert_score

from vlmeval.smp import get_logger

eval_logger = get_logger('EVAL')

Event = Dict[str, Any]
Match = Tuple[int, int, float]  # (gt_index, pred_index, iou)


# ---------------------------------


def compute_bertscore_metrics(reference: str, candidate: str) -> Tuple[float, float, float]:
    """
    Compute BERTScore precision, recall, and F1 between reference and candidate text.

    Args:
        reference: Ground truth text.
        candidate: Predicted/generated text.

    Returns:
        Tuple of (precision, recall, f1_score)
    """
    try:
        eval_logger.debug(
            f"Computing BERTScore for reference length: {len(reference)}, candidate length: {len(candidate)}"
        )
        P, R, F1 = bert_score([candidate], [reference], lang="en", verbose=False)

        precision = float(P.item())
        recall = float(R.item())
        f1_score = float(F1.item())

        eval_logger.debug(f"BERTScore results - P: {precision:.4f}, R: {recall:.4f}, F1: {f1_score:.4f}")
        return precision, recall, f1_score
    except Exception as e:
        eval_logger.warning(f"Error computing BERTScore metrics: {e}")
        return 0.0, 0.0, 0.0


# ---------------------------------



def _time_to_seconds(ts: str) -> float:
    """
    Convert a timestamp string to seconds.
    
    Supports various formats that LLMs might generate:
    - HH:MM:SS.mmm (standard format)
    - HH:MM:SS:mmm (common LLM mistake - colon instead of period before ms)
    - HH:MM:SS (no milliseconds)
    - MM:SS.mmm or MM:SS (minutes:seconds only)
    - SS.mmm or SS (seconds only)
    """
    import re
    
    ts = ts.strip()
    
    # Fix common LLM mistake: HH:MM:SS:mmm -> HH:MM:SS.mmm
    # Match pattern where we have 4 colon-separated parts (the last being milliseconds)
    four_part_match = re.match(r'^(\d{1,2}):(\d{2}):(\d{2}):(\d{1,3})$', ts)
    if four_part_match:
        h, m, s, ms = four_part_match.groups()
        # Normalize milliseconds to 3 digits
        ms = ms.ljust(3, '0')[:3]
        ts = f"{h}:{m}:{s}.{ms}"
    
    # Try standard datetime formats
    for fmt in ("%H:%M:%S.%f", "%H:%M:%S", "%M:%S.%f", "%M:%S", "%S.%f", "%S"):
        try:
            dt = datetime.strptime(ts, fmt)
            # For formats without hours, datetime defaults hour to 1900 values
            # We need to handle this carefully
            if fmt.startswith("%H"):
                return dt.hour * 3600 + dt.minute * 60 + dt.second + dt.microsecond / 1e6
            elif fmt.startswith("%M"):
                return dt.minute * 60 + dt.second + dt.microsecond / 1e6
            else:
                return dt.second + dt.microsecond / 1e6
        except ValueError:
            continue
    
    # Last resort: try to parse with regex for flexible formats
    # Matches: HH:MM:SS.mmm, H:MM:SS.mmm, etc.
    flexible_match = re.match(
        r'^(\d{1,2}):(\d{1,2}):(\d{1,2})(?:[.:](\d{1,6}))?$', ts
    )
    if flexible_match:
        h, m, s, ms = flexible_match.groups()
        h, m, s = int(h), int(m), int(s)
        if ms:
            # Normalize to microseconds (6 digits)
            ms = ms.ljust(6, '0')[:6]
            ms = int(ms)
        else:
            ms = 0
        return h * 3600 + m * 60 + s + ms / 1e6
    
    raise ValueError(f"Unsupported timestamp format: {ts}")


def _to_seconds(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        stripped = value.strip()
        try:
            return float(stripped)
        except ValueError:
            return _time_to_seconds(stripped)
    raise TypeError(f"Unsupported timestamp type: {type(value)}")





def temporal_iou(
    gt_interval: Tuple[str, str], pred_interval: Tuple[str, str]
) -> float:
    """
    Compute temporal IoU between ground truth and prediction intervals.
    
    Returns 0.0 if any timestamp parsing fails (with error logged).
    """
    try:
        gt_start, gt_end = map(_to_seconds, gt_interval)
        pred_start, pred_end = map(_to_seconds, pred_interval)
    except (ValueError, TypeError) as e:
        eval_logger.warning(
            f"Failed to parse timestamps - GT: {gt_interval}, Pred: {pred_interval}. "
            f"Error: {e}. Returning IoU=0.0"
        )
        return 0.0

    intersection_start = max(gt_start, pred_start)
    intersection_end = min(gt_end, pred_end)
    intersection = max(0.0, intersection_end - intersection_start)

    union = max(gt_end, pred_end) - min(gt_start, pred_start)
    return 0.0 if union == 0 else intersection / union

def build_iou_matrix(
    gt_events: Iterable[Dict[str, Any]],
    pred_events: Iterable[Dict[str, Any]],
) -> Dict[str, np.ndarray]:
    gt_by_type = defaultdict(list)
    pred_by_type = defaultdict(list)

    for ev in gt_events:
        gt_by_type[ev["type"]].append(ev)

    for ev in pred_events:
        event_type = ev.get("event_type", ev.get("type"))
        if event_type:
            pred_by_type[event_type].append(ev)

    matrices = {}

    for ev_type, gt_list in gt_by_type.items():

        preds = pred_by_type[ev_type]
        matrix = np.zeros((len(gt_list), len(preds)))
        for i, gt in enumerate(gt_list):
            for j, pred in enumerate(preds):
                matrix[i, j] = temporal_iou((gt["start_time"], gt["end_time"]), (pred["start_time"], pred["end_time"]))

        matrices[ev_type] = matrix

    return matrices




def temporal_metrics_mapper(
    gt_events: Iterable[Dict[str, Any]],  # list of events: [{ type, start_time, end_time }, ... ]
    pred_events: Iterable[Dict[str, Any]],  # list of events: [{ type, start_time, end_time }, ... ]
    *,
    iou_threshold: float = 0.01
) -> Dict[str, Any]:
    # uses scipy implementation of Jonker-Volgenant algorithm to match events: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html

    # build IoU matrix
    iou_matrices = build_iou_matrix(gt_events, pred_events)

    gt_by_type = defaultdict(list)
    pred_by_type = defaultdict(list)

    for ev in gt_events:
        gt_by_type[ev["type"]].append(ev)
    for ev in pred_events:
        event_type = ev.get("event_type", ev.get("type"))
        if event_type:
            pred_by_type[event_type].append(ev)

    classwise_matches = defaultdict(list)
    classwise_unmatched_gt = defaultdict(list)
    classwise_unmatched_pred = defaultdict(list)


    for ev_type, gt_list in gt_by_type.items():
        preds = pred_by_type[ev_type]
        iou_matrix = iou_matrices[ev_type]

        eval_logger.debug(f"Event type: {ev_type}, GT count: {len(gt_list)}, Pred count: {len(preds)}")
        if len(gt_list) > 0 and len(preds) > 0:
            # Convert GT times to seconds for easier comparison with predictions
            gt_in_seconds = [(_to_seconds(g['start_time']), _to_seconds(g['end_time'])) for g in gt_list]
            eval_logger.debug(f"GT events for {ev_type} (in seconds): {gt_in_seconds}")
            eval_logger.debug(f"Pred events for {ev_type}: {[(p['start_time'], p['end_time']) for p in preds]}")
            eval_logger.debug(f"IoU matrix for {ev_type}:\n{iou_matrix}")
            eval_logger.debug(f"Max IoU: {iou_matrix.max()}, Min IoU: {iou_matrix.min()}")

        if not gt_list:
            classwise_unmatched_pred[ev_type].extend(preds)
            continue
        if not preds:
            classwise_unmatched_gt[ev_type].extend(gt_list)
            continue

        row_ind, col_ind = linear_sum_assignment(iou_matrix, maximize=True)

        matched_pred_indices = set()
        matched_gt_indices = set()

        for gt_idx, pred_idx in zip(row_ind, col_ind):
            iou = iou_matrix[gt_idx][pred_idx]
            eval_logger.debug(f"Matching GT[{gt_idx}] with Pred[{pred_idx}], IoU={iou:.4f}, threshold={iou_threshold}")
            if iou >= iou_threshold:
                classwise_matches[ev_type].append(
                    {
                        "type": ev_type,
                        "iou": iou,
                        "gt_event": gt_list[gt_idx],
                        "pred_event": preds[pred_idx],
                    }
                )
                matched_gt_indices.add(gt_idx)
                matched_pred_indices.add(pred_idx)

        for idx, gt in enumerate(gt_list):
            if idx not in matched_gt_indices:
                classwise_unmatched_gt[ev_type].append(gt)

        for idx, pred in enumerate(preds):
            if idx not in matched_pred_indices:
                classwise_unmatched_pred[ev_type].append(pred)


    all_classes = set(gt_by_type.keys()) | set(pred_by_type.keys())
    classwise_metrics = {}
    for ev_type in sorted(all_classes):
        tp = len(classwise_matches.get(ev_type, []))
        fp = len(classwise_unmatched_pred.get(ev_type, []))
        fn = len(classwise_unmatched_gt.get(ev_type, []))
        classwise_metrics[ev_type] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }
    # { type: { tp, fp, fn } ... }, [ { type: { iou, gt_event, pred_event } ... } ]
    return classwise_metrics, classwise_matches

#--------------------------------------------------

def parse_event_list(raw: str) -> Tuple[bool, List[Event], Optional[str]]:
    """
    Parse and validate an event list string.

    Returns:
        (is_valid, events, error_message)
    """
    try:
        parsed = json.loads(raw)
    except Exception as exc:  # noqa: BLE001
        return False, [], f"Invalid JSON: {exc}"

    if not isinstance(parsed, list):
        return False, [], "JSON root must be a list of events."

    events: List[Event] = []
    try:
        for idx, item in enumerate(parsed):
            if not isinstance(item, dict):
                raise ValueError(
                    f"Event at index {idx} is not an object: {item!r}"
                )
            events.append(item)
    except Exception as exc:  # noqa: BLE001
        return False, [], str(exc)
    return True, events, None


def calculate_score_per_video(
    pred: str,
    gt: str,
    iou_threshold: float = 0.5,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Calculate per-video detection metrics between predicted and ground-truth
    events, including temporal metrics and BERTScore metrics for event descriptions.

    Args:
        pred: JSON string of predicted events.
        gt: JSON string of ground-truth events.
        iou_threshold: IoU threshold for matching; defaults to 0.5.

    Returns:
        Tuple of:
            - classwise_metrics: { type: { tp, fp, fn } ... }
            - bert_metrics: { type: { bertscore_precision: [], bertscore_recall: [], bertscore_f1: [] } ... }
    """
    pred_valid, pred_events, pred_err = parse_event_list(pred)
    gt_valid, gt_events, gt_err = parse_event_list(gt)

    if not (pred_valid and gt_valid):
        return {}, {}

    classwise_metrics, classwise_matches = temporal_metrics_mapper(
        gt_events, pred_events, iou_threshold=iou_threshold
    )

    # Compute BERTScore for event descriptions from matched events
    bert_metrics: Dict[str, Dict[str, List[float]]] = {}
    for ev_type, matches in classwise_matches.items():
        if not matches:
            continue

        bert_metrics[ev_type] = {
            "bertscore_precision": [],
            "bertscore_recall": [],
            "bertscore_f1": []
        }

        for match in matches:
            gt_event = match["gt_event"]
            pred_event = match["pred_event"]

            gt_description = gt_event.get("description", "")
            pred_description = pred_event.get("description", "")

            if gt_description and pred_description:
                precision, recall, f1 = compute_bertscore_metrics(gt_description, pred_description)
                bert_metrics[ev_type]["bertscore_precision"].append(precision)
                bert_metrics[ev_type]["bertscore_recall"].append(recall)
                bert_metrics[ev_type]["bertscore_f1"].append(f1)
            else:
                eval_logger.warning(f"Missing description for event type {ev_type}")
                bert_metrics[ev_type]["bertscore_precision"].append(0.0)
                bert_metrics[ev_type]["bertscore_recall"].append(0.0)
                bert_metrics[ev_type]["bertscore_f1"].append(0.0)

    return classwise_metrics, bert_metrics


def lvs_aggregate_temporal_precision(results, metric_type="micro"):
    """
    Aggregation function for LVS temporal precision results.

    Args:
        results: a list of temporal metrics: { type: { tp, fp, fn } ... }
        metric_type: 'micro' or 'macro'

    Returns:
        Average temporal precision value
    """
    aggregated_metrics: defaultdict[str, defaultdict[str, list]] = defaultdict(
        lambda: defaultdict(list)
    )

    for result in results:
        for ev_type, metrics in result.items():
            for metric, value in metrics.items():
                aggregated_metrics[ev_type][metric].append(value)

    if metric_type == "micro":
        total_tp = sum(
            sum(event_metrics["tp"]) for event_metrics in aggregated_metrics.values()
        )
        total_fp = sum(
            sum(event_metrics["fp"]) for event_metrics in aggregated_metrics.values()
        )
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
        return precision
    elif metric_type == "macro":
        event_wise_precision = []
        for event_type in aggregated_metrics.keys():
            tp = sum(aggregated_metrics[event_type]["tp"])
            fp = sum(aggregated_metrics[event_type]["fp"])
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            event_wise_precision.append(precision)
        return np.mean(event_wise_precision) if event_wise_precision else 0.0
    else:
        raise ValueError(f"Invalid metric type: {metric_type}. Choose from 'micro' or 'macro'.")


def lvs_aggregate_temporal_recall(results, metric_type="micro"):
    """
    Aggregation function for LVS temporal recall results.

    Args:
        results: a list of temporal metrics: { type: { tp, fp, fn } ... }
        metric_type: 'micro' or 'macro'

    Returns:
        Average temporal recall value
    """
    aggregated_metrics: defaultdict[str, defaultdict[str, list]] = defaultdict(
        lambda: defaultdict(list)
    )

    for result in results:
        for ev_type, metrics in result.items():
            for metric, value in metrics.items():
                aggregated_metrics[ev_type][metric].append(value)

    if metric_type == "micro":
        total_tp = sum(sum(event_metrics["tp"]) for event_metrics in aggregated_metrics.values())
        total_fn = sum(sum(event_metrics["fn"]) for event_metrics in aggregated_metrics.values())
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
        return recall
    elif metric_type == "macro":
        event_wise_recall = []
        for event_type in aggregated_metrics.keys():
            tp = sum(aggregated_metrics[event_type]["tp"])
            fn = sum(aggregated_metrics[event_type]["fn"])
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            event_wise_recall.append(recall)
        return np.mean(event_wise_recall) if event_wise_recall else 0.0
    else:
        raise ValueError(f"Invalid metric type: {metric_type}. Choose from 'micro' or 'macro'.")


def lvs_aggregate_temporal_f1(results, metric_type="micro"):
    """
    Aggregation function for LVS temporal F1 results.

    Args:
        results: a list of temporal metrics: { type: { tp, fp, fn } ... }
        metric_type: 'micro' or 'macro'

    Returns:
        Average temporal F1 value
    """
    aggregated_metrics: defaultdict[str, defaultdict[str, list]] = defaultdict(
        lambda: defaultdict(list)
    )

    for result in results:
        for ev_type, metrics in result.items():
            for metric, value in metrics.items():
                aggregated_metrics[ev_type][metric].append(value)

    if metric_type == "micro":
        total_tp = sum(sum(event_metrics["tp"]) for event_metrics in aggregated_metrics.values())
        total_fp = sum(sum(event_metrics["fp"]) for event_metrics in aggregated_metrics.values())
        total_fn = sum(sum(event_metrics["fn"]) for event_metrics in aggregated_metrics.values())
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        return f1
    elif metric_type == "macro":
        event_wise_f1 = []
        for event_type in aggregated_metrics.keys():
            tp = sum(aggregated_metrics[event_type]["tp"])
            fp = sum(aggregated_metrics[event_type]["fp"])
            fn = sum(aggregated_metrics[event_type]["fn"])
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
            event_wise_f1.append(f1)
        return np.mean(event_wise_f1) if event_wise_f1 else 0.0
    else:
        raise ValueError(f"Invalid metric type: {metric_type}. Choose from 'micro' or 'macro'.")


def lvs_aggregate_bert_precision(results, metric_type="micro"):
    """
    Aggregation function for LVS event description BERTScore precision results.

    Args:
        results: a list of event description metrics:
                 { type: { bertscore_precision: [], bertscore_recall: [], bertscore_f1: [] } ... }
        metric_type: 'micro' or 'macro'

    Returns:
        Average event description precision value
    """
    all_precisions = []

    for result in results:
        for ev_type, metrics in result.items():
            if "bertscore_precision" in metrics:
                all_precisions.extend(metrics["bertscore_precision"])

    if not all_precisions:
        eval_logger.warning("No event description precision results to aggregate")
        return 0.0

    if metric_type == "micro":
        return np.mean(all_precisions)
    elif metric_type == "macro":
        # Group by event type
        type_precisions: Dict[str, List[float]] = defaultdict(list)
        for result in results:
            for ev_type, metrics in result.items():
                if "bertscore_precision" in metrics:
                    type_precisions[ev_type].extend(metrics["bertscore_precision"])

        # Calculate mean per type, then average across types
        type_means = [np.mean(precisions) for precisions in type_precisions.values() if precisions]
        return np.mean(type_means) if type_means else 0.0
    else:
        raise ValueError(f"Invalid metric type: {metric_type}. Choose from 'micro' or 'macro'.")


def lvs_aggregate_bert_recall(results, metric_type="micro"):
    """
    Aggregation function for LVS event description BERTScore recall results.

    Args:
        results: a list of event description metrics:
                 { type: { bertscore_precision: [], bertscore_recall: [], bertscore_f1: [] } ... }
        metric_type: 'micro' or 'macro'

    Returns:
        Average event description recall value
    """
    all_recalls = []

    for result in results:
        for ev_type, metrics in result.items():
            if "bertscore_recall" in metrics:
                all_recalls.extend(metrics["bertscore_recall"])

    if not all_recalls:
        eval_logger.warning("No event description recall results to aggregate")
        return 0.0

    if metric_type == "micro":
        return np.mean(all_recalls)
    elif metric_type == "macro":
        # Group by event type
        type_recalls: Dict[str, List[float]] = defaultdict(list)
        for result in results:
            for ev_type, metrics in result.items():
                if "bertscore_recall" in metrics:
                    type_recalls[ev_type].extend(metrics["bertscore_recall"])

        # Calculate mean per type, then average across types
        type_means = [np.mean(recalls) for recalls in type_recalls.values() if recalls]
        return np.mean(type_means) if type_means else 0.0
    else:
        raise ValueError(f"Invalid metric type: {metric_type}. Choose from 'micro' or 'macro'.")


def lvs_aggregate_bert_f1(results, metric_type="micro"):
    """
    Aggregation function for LVS event description BERTScore F1 results.

    Args:
        results: a list of event description metrics:
                 { type: { bertscore_precision: [], bertscore_recall: [], bertscore_f1: [] } ... }
        metric_type: 'micro' or 'macro'

    Returns:
        Average event description F1 value
    """
    all_f1s = []

    for result in results:
        for ev_type, metrics in result.items():
            if "bertscore_f1" in metrics:
                all_f1s.extend(metrics["bertscore_f1"])

    if not all_f1s:
        eval_logger.warning("No event description F1 results to aggregate")
        return 0.0

    if metric_type == "micro":
        return np.mean(all_f1s)
    elif metric_type == "macro":
        # Group by event type
        type_f1s: Dict[str, List[float]] = defaultdict(list)
        for result in results:
            for ev_type, metrics in result.items():
                if "bertscore_f1" in metrics:
                    type_f1s[ev_type].extend(metrics["bertscore_f1"])

        # Calculate mean per type, then average across types
        type_means = [np.mean(f1s) for f1s in type_f1s.values() if f1s]
        return np.mean(type_means) if type_means else 0.0
    else:
        raise ValueError(f"Invalid metric type: {metric_type}. Choose from 'micro' or 'macro'.")
