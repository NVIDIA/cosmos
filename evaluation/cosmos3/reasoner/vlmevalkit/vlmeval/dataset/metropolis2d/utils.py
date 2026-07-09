"""
Common utilities for 2D detection datasets.

This module contains shared functions for parsing, evaluation, and configuration
used by various 2D detection datasets (Metropolis2D, Astro2D, etc.).
"""

import json
import math
import os
import re
from typing import Any, Callable

import numpy as np
import yaml


def load_dataset_config(dataset_name, config_path=None, task='detection'):
    """
    Load dataset configuration from datasets.yaml.

    Args:
        dataset_name: Name of the dataset to look up
        config_path: Path to datasets.yaml. If None, uses default location.

    Returns:
        dict with dataset configuration or None if not found
    """
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), 'datasets.yaml')

    if not os.path.exists(config_path):
        return None

    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Look in 'detection' section
        if task in config:
            for dataset_cfg in config[task]:
                if dataset_cfg.get('name') == dataset_name:
                    return dataset_cfg

        return None
    except Exception:
        return None


def scale_bbox(bbox, height, width, scale_factor=1000, coord_scale='normalized'):
    """
    Scale the bounding box to the original image size.

    Args:
        bbox: The bounding box to scale.
        height: The height of the original image.
        width: The width of the original image.
        scale_factor: Normalization scale factor (used when coord_scale='normalized').
        coord_scale: How to interpret bbox values.
          'normalized' (default) — values are 0-`scale_factor` normalized; divide
                                   by scale_factor and multiply by image w/h.
          'pixel'                — values are already in absolute pixel coords;
                                   skip the rescale, just clamp to image bounds.
                                   Set explicitly by callers whose model emits
                                   pixel-scale coordinates (e.g. MiMo / Qwen2.5-VL).
    """
    if coord_scale == 'pixel':
        abs_x1, abs_y1, abs_x2, abs_y2 = (
            int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        )
    else:  # 'normalized'
        abs_x1, abs_y1, abs_x2, abs_y2 = (
            int(bbox[0] / scale_factor * width),
            int(bbox[1] / scale_factor * height),
            int(bbox[2] / scale_factor * width),
            int(bbox[3] / scale_factor * height)
        )

    # Clip the bounding box to the image size
    abs_x1 = max(abs_x1, 0)
    abs_y1 = max(abs_y1, 0)

    abs_x2 = min(abs_x2, width)
    abs_y2 = min(abs_y2, height)

    return abs_x1, abs_y1, abs_x2, abs_y2


def parse_kitti_label(label_path):
    """
    Parse KITTI format label file.

    KITTI format: type truncated occluded alpha bbox(x1,y1,x2,y2) dimensions(h,w,l) location(x,y,z) rotation_y [score]

    Returns:
        List of dicts with 'label' and 'bbox' (x1, y1, x2, y2 in pixel coordinates)
    """
    objects = []
    if not os.path.exists(label_path):
        return objects

    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 15:
                continue

            label = parts[0].lower()
            # Skip DontCare or other ignore labels
            if label in ['dontcare', 'misc', 'ignore']:
                continue

            # KITTI bbox format: x1, y1, x2, y2 (0-indexed, pixel coords)
            try:
                x1 = float(parts[4])
                y1 = float(parts[5])
                x2 = float(parts[6])
                y2 = float(parts[7])
            except (ValueError, IndexError):
                continue

            objects.append({
                'label': label,
                'bbox': [x1, y1, x2, y2]
            })

    return objects


# ---------------------------------------------------------------------------
# Multi-format bbox parsers — mirrored from
# vlmeval/dataset/utils/locate_anything_bench/parsers.py.
# Per-benchmark inline mirror (codebase convention) so future LA parser
# changes do not auto-propagate. LA tag form is intentionally omitted.
# ---------------------------------------------------------------------------

# Each parser returns: [(label, coords, is_point), ...]
# coords are 0-1000 normalized; for boxes: [x1,y1,x2,y2]; for points: [x,y].
_ParserResult = list[tuple[str, list[float], bool]]
_Parser = Callable[..., _ParserResult]  # (text: str, coord_scale: str = 'normalized') -> _ParserResult

_JSON_FENCE_PATTERN = re.compile(r'```(?:json)?\s*([\s\S]*?)```', re.IGNORECASE)


def _try_load_json(text: str) -> Any:
    """Best-effort JSON load — handles markdown fence, leading/trailing prose, single dict.

    str() coerces non-string inputs (NaN, int, None) so `re.search` and `.strip`
    below can't raise on a missing-but-non-string prediction.
    """
    if not text:
        return None
    text = str(text)
    fenced = _JSON_FENCE_PATTERN.search(text)
    payload = fenced.group(1).strip() if fenced else text.strip()

    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    try:
        obj, _ = decoder.raw_decode(payload)
        return obj
    except json.JSONDecodeError:
        pass

    for i, ch in enumerate(payload):
        if ch in '[{':
            try:
                obj, _ = decoder.raw_decode(payload[i:])
                return obj
            except json.JSONDecodeError:
                continue
    return None


def _coerce_to_list(parsed: Any) -> list:
    if isinstance(parsed, dict):
        return [parsed]
    if isinstance(parsed, list):
        return parsed
    return []


def _label_of(item: dict) -> str:
    label = item.get('label') or item.get('category') or item.get('name') or ''
    return label if isinstance(label, str) else str(label)


def _coerce_floats(seq: Any, n: int):
    if not isinstance(seq, (list, tuple)) or len(seq) != n:
        return None
    try:
        return [float(v) for v in seq]
    except (TypeError, ValueError):
        return None


def _in_range(coords, coord_scale: str = 'normalized') -> bool:
    """Range gate for parsed coords. 'normalized' enforces 0-1000 (filters
    noise from prose). 'pixel' drops the upper bound (MiMo / Qwen2.5-VL emit
    absolute pixel values that exceed 1000 on large images). NaN/Infinity are
    rejected under both modes — downstream `scale_bbox` casts via `int()`
    which raises OverflowError on Inf and ValueError on NaN."""
    if not all(math.isfinite(v) for v in coords):
        return False
    if coord_scale == 'pixel':
        return all(v >= 0 for v in coords)
    return all(0 <= v <= 1000 for v in coords)


def _parse_qwen_json(text: str, coord_scale: str = 'normalized') -> _ParserResult:
    """`[{"bbox_2d": [x1,y1,x2,y2], "label": "..."}, ...]` — coords 0-1000 xyxy."""
    items = _coerce_to_list(_try_load_json(text))
    results: _ParserResult = []
    for item in items:
        if not isinstance(item, dict):
            continue
        if 'bbox_2d' in item:
            coords = _coerce_floats(item.get('bbox_2d'), 4)
            if coords and _in_range(coords, coord_scale):
                results.append((_label_of(item), coords, False))
        elif 'point_2d' in item and 'box_2d' not in item:
            coords = _coerce_floats(item.get('point_2d'), 2)
            if coords and _in_range(coords, coord_scale):
                results.append((_label_of(item), coords, True))
    return results


def _parse_gemini_json(text: str, coord_scale: str = 'normalized') -> _ParserResult:
    """`[{"box_2d": [y1,x1,y2,x2], "label": "..."}, ...]` — yxyx swapped to xyxy."""
    items = _coerce_to_list(_try_load_json(text))
    results: _ParserResult = []
    for item in items:
        if not isinstance(item, dict):
            continue
        if 'box_2d' in item:
            coords = _coerce_floats(item.get('box_2d'), 4)
            if coords and _in_range(coords, coord_scale):
                y1, x1, y2, x2 = coords
                results.append((_label_of(item), [x1, y1, x2, y2], False))
        elif 'point_2d' in item:
            coords = _coerce_floats(item.get('point_2d'), 2)
            if coords and _in_range(coords, coord_scale):
                y, x = coords
                results.append((_label_of(item), [x, y], True))
    return results


def _parse_label_keyed_json(text: str, coord_scale: str = 'normalized') -> _ParserResult:
    """`[{"<label>": [x,y,x,y]}, ...]` or `[{"<label>": [[x,y,x,y], ...]}, ...]`.

    Single-key dicts where the key is a free-form label string and the value is
    either a single 4-tuple or a list of 4-tuples. Emitted by Cosmos-3 Super on
    2D grounding tasks when prompted with "JSON-like" wording. xyxy is assumed
    (same as Qwen-style).
    """
    items = _coerce_to_list(_try_load_json(text))
    results: _ParserResult = []
    for item in items:
        if not isinstance(item, dict) or len(item) != 1:
            continue
        label, val = next(iter(item.items()))
        if not isinstance(label, str):
            continue
        box = _coerce_floats(val, 4)
        if box is not None and _in_range(box, coord_scale):
            results.append((label, box, False))
            continue
        if isinstance(val, list):
            for sub in val:
                sub_box = _coerce_floats(sub, 4)
                if sub_box is not None and _in_range(sub_box, coord_scale):
                    results.append((label, sub_box, False))
    return results


def _parse_bare_array_json(text: str, coord_scale: str = 'normalized') -> _ParserResult:
    """Raw `[x1,y1,x2,y2]` / `[x,y]` / `[[..],[..]]` — no envelope, empty label.

    Assumes Qwen-style xyxy. A Gemini-family model emitting bare arrays would
    have its yxyx silently swapped — acceptable trade-off; runs LAST.
    """
    parsed = _try_load_json(text)
    if parsed is None:
        return []

    def _as_xy_box(seq):
        if not isinstance(seq, (list, tuple)) or len(seq) != 4:
            return None
        if not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in seq):
            return None
        coords = [float(v) for v in seq]
        return coords if _in_range(coords, coord_scale) else None

    def _as_xy_point(seq):
        if not isinstance(seq, (list, tuple)) or len(seq) != 2:
            return None
        if not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in seq):
            return None
        coords = [float(v) for v in seq]
        return coords if _in_range(coords, coord_scale) else None

    box = _as_xy_box(parsed)
    if box is not None:
        return [('', box, False)]
    pt = _as_xy_point(parsed)
    if pt is not None:
        return [('', pt, True)]

    if isinstance(parsed, list):
        results: _ParserResult = []
        for item in parsed:
            box = _as_xy_box(item)
            if box is not None:
                results.append(('', box, False))
                continue
            pt = _as_xy_point(item)
            if pt is not None:
                results.append(('', pt, True))
        return results
    return []


_ANGLEBOX_4TUP = re.compile(
    r'[\[<]\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,'
    r'\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*[\]>]'
)


def _parse_anglebox_tuples(text: str, coord_scale: str = 'normalized') -> _ParserResult:
    """Salvage 4-tuples wrapped in `[...]` or `<...>` or mismatched `[...>` / `<...]`.

    Cosmos3-Edge-Reasoner-0521 emits multi-box predictions where the first
    box uses `[...]` and subsequent boxes use `<...>`. Both delimiters land
    in the same character class. xyxy assumed (no yxyx swap) — Gemini-family
    models emit clean `box_2d` JSON and never reach this branch.
    """
    results: _ParserResult = []
    for m in _ANGLEBOX_4TUP.finditer(str(text)):
        coords = [float(x) for x in m.groups()]
        if _in_range(coords, coord_scale):
            results.append(('', coords, False))
    return results


_PARSERS: list[_Parser] = [
    _parse_qwen_json,         # `bbox_2d` (xyxy)
    _parse_gemini_json,       # `box_2d`  (yxyx → swap)
    _parse_label_keyed_json,  # `{"<label>": [x,y,x,y]}` or `{"<label>": [[..],..]}`
    _parse_anglebox_tuples,   # `[...]` / `<...>` / mismatched `[...>` 4-tuples (cosmos3-edge)
    _parse_bare_array_json,   # raw `[x,y,x,y]` / `[[..],..]` — empty label, runs LAST
]


def parse_bbox_2d_from_text(text: str, coord_scale: str = 'normalized') -> list:
    """Astro2D-shape: `[{'bbox_2d': [x1,y1,x2,y2]_norm, 'label': str, 'score': float}, ...]`.

    Dispatches across Qwen / Gemini / bare-array shapes via `_PARSERS`; returns
    the legacy dict-list shape so existing `evaluate()` callers stay unchanged.
    `coord_scale` is forwarded to each parser's range gate.
    """
    for parser in _PARSERS:
        tuples = parser(text, coord_scale=coord_scale)
        if tuples:
            return [
                {'bbox_2d': coords, 'label': label or 'person', 'score': 1.0}
                for label, coords, is_point in tuples if not is_point
            ]
    return []


def compute_2d_iou(box1, box2):
    """
    Compute IoU between two 2D bounding boxes.

    Args:
        box1, box2: [x1, y1, x2, y2] format

    Returns:
        IoU value (float)
    """
    xi1 = max(box1[0], box2[0])
    yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2])
    yi2 = min(box1[3], box2[3])
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = area1 + area2 - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


def compute_ap_coco(recalls, precisions):
    """
    Compute Average Precision using COCO-style all-point interpolation.

    The precision at each recall level r is interpolated by taking the maximum
    precision measured at a recall >= r. Then AP is computed as the area under
    this interpolated precision-recall curve.

    Args:
        recalls: numpy array of recall values (sorted in ascending order)
        precisions: numpy array of precision values corresponding to recalls

    Returns:
        AP value (float)
    """
    if len(recalls) == 0:
        return 0.0

    # Prepend (0, 1) and append (1, 0) to the PR curve
    recalls = np.concatenate([[0.0], recalls, [1.0]])
    precisions = np.concatenate([[1.0], precisions, [0.0]])

    # Make precision monotonically decreasing (from right to left)
    # This is the interpolation step: precision at recall r is max precision at recall >= r
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])

    # Find points where recall changes
    recall_change_indices = np.where(recalls[1:] != recalls[:-1])[0] + 1

    # Compute area under the interpolated PR curve
    ap = np.sum((recalls[recall_change_indices] - recalls[recall_change_indices - 1]) *
                precisions[recall_change_indices])

    return ap


def compute_ap_from_matches(tp_list, num_gt, confidence_scores):
    """
    Compute AP from detection matches using COCO-style calculation.

    Args:
        tp_list: list of 1s (true positive) or 0s (false positive) for each detection
        num_gt: total number of ground truth objects
        confidence_scores: confidence score for each detection

    Returns:
        AP value, precision array, recall array
    """
    if num_gt == 0:
        return 0.0, np.array([]), np.array([])

    if len(tp_list) == 0:
        return 0.0, np.array([]), np.array([])

    # Sort by confidence (descending)
    sorted_indices = np.argsort(-np.array(confidence_scores))
    tp_sorted = np.array(tp_list)[sorted_indices]

    # Cumulative sums
    tp_cumsum = np.cumsum(tp_sorted)
    fp_cumsum = np.cumsum(1 - tp_sorted)

    # Precision and recall at each threshold
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
    recalls = tp_cumsum / num_gt

    # Compute AP using COCO-style interpolation
    ap = compute_ap_coco(recalls, precisions)

    return ap, precisions, recalls


def evaluate_detections(pred_boxes, gt_boxes, iou_threshold=0.5):
    """
    Evaluate detections against ground truth using the given IoU threshold.

    Args:
        pred_boxes: List of predicted boxes, each with 'bbox' and optional 'score'
        gt_boxes: List of ground truth boxes with 'bbox'
        iou_threshold: IoU threshold for matching (default 0.5 for AP50)

    Returns:
        dict with 'tp', 'fp', 'fn', 'precision', 'recall'
    """
    if len(pred_boxes) == 0:
        return {
            'tp': 0,
            'fp': 0,
            'fn': len(gt_boxes),
            'precision': 0.0,
            'recall': 0.0
        }

    if len(gt_boxes) == 0:
        return {
            'tp': 0,
            'fp': len(pred_boxes),
            'fn': 0,
            'precision': 0.0,
            'recall': 1.0 if len(pred_boxes) == 0 else 0.0
        }

    # Sort predictions by score if available
    if 'score' in pred_boxes[0]:
        pred_boxes = sorted(pred_boxes, key=lambda x: x.get('score', 1.0), reverse=True)

    gt_matched = [False] * len(gt_boxes)
    tp = 0
    fp = 0

    for pred in pred_boxes:
        pred_bbox = pred['bbox']
        best_iou = 0.0
        best_gt_idx = -1

        for gt_idx, gt in enumerate(gt_boxes):
            if gt_matched[gt_idx]:
                continue
            iou = compute_2d_iou(pred_bbox, gt['bbox'])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_iou >= iou_threshold:
            tp += 1
            gt_matched[best_gt_idx] = True
        else:
            fp += 1

    fn = sum(1 for matched in gt_matched if not matched)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return {
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'precision': precision,
        'recall': recall
    }
