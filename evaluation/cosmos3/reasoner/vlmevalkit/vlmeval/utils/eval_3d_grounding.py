#!/usr/bin/env python3
"""
Evaluate 3D bounding box predictions against ground truth using IoU matching.

This module provides functionality to compare predicted 3D bounding boxes with ground truth
annotations. It uses the Hungarian algorithm for optimal matching and computes Intersection
over Union (IoU) metrics for evaluation.

The evaluation process:
1. Loads prediction and ground truth JSON files from specified directories
2. Matches files by filename between the two directories
3. For each matching file pair, performs optimal matching using Hungarian algorithm
4. Computes 3D IoU for matched boxes (only matching boxes with same labels)
5. Aggregates results across all files

Usage:
    python cosmos3/eval/eval_function.py <pred_dir> <gt_dir> [--iou-threshold 0.5] [--verbose]

    Example:
        python cosmos3/eval/eval_function.py output_eval/text_qwen30b output_eval/text_gt --iou-threshold 0.5 --verbose

Input JSON Format:
    Each JSON file should contain:
    {
        "annotations": [
            {
                "bbox_3d": [x, y, z, width, height, depth, roll, pitch, yaw],
                "label": "car"
            },
            ...
        ]
    }

    Note: Only the first 6 values (x, y, z, width, height, depth) are used for IoU calculation.
          Rotation (roll, pitch, yaw) is ignored for axis-aligned IoU computation.
"""

import argparse
import json
from pathlib import Path
from typing import Union

import numpy as np
from scipy.optimize import linear_sum_assignment


def load_boxes(json_path: Path) -> tuple[list[list[float]], list[str]]:
    """
    Load bbox_3d and label arrays from a JSON annotation file.

    Args:
        json_path: Path to JSON file containing annotations

    Returns:
        Tuple of (boxes, labels) where:
        - boxes: List of bbox_3d lists, each containing [x, y, z, width, height, depth, ...]
        - labels: List of label strings corresponding to each box

    Raises:
        FileNotFoundError: If the JSON file doesn't exist
        json.JSONDecodeError: If the file is not valid JSON
    """
    with open(json_path, "r") as f:
        data = json.load(f)
    boxes = [ann["bbox_3d"] for ann in data.get("annotations", [])]
    labels = [ann["label"] for ann in data.get("annotations", [])]
    return boxes, labels


def normalize_label(label: str) -> str:
    """
    Normalize label by extracting base name and removing suffixes.

    Removes numeric suffixes and underscores (e.g., "Car_5" → "car", "van_100" → "van").
    Converts to lowercase for case-insensitive matching.

    Args:
        label: Original label string (e.g., "Car_5", "van_100", "truck_1")

    Returns:
        Normalized label (lowercase, base name only)

    Examples:
        "Car_5" → "car"
        "van_100" → "van"
        "truck_1" → "truck"
        "pedestrian" → "pedestrian"
    """
    label_lower = label.lower().strip()

    # Split by underscore and take the first part (base name)
    # This handles cases like "Car_5", "van_100", "truck_1", etc.
    base_name = label_lower.split("_")[0]

    return base_name


def compute_3d_iou(box1: list[float], box2: list[float]) -> float:
    """
    Compute axis-aligned 3D Intersection over Union (IoU) ignoring rotation.

    This function computes the IoU of two 3D bounding boxes by treating them as
    axis-aligned boxes. Rotation information (roll, pitch, yaw) is ignored.

    Args:
        box1: First bounding box as [x, y, z, width, height, depth, ...]
        box2: Second bounding box as [x, y, z, width, height, depth, ...]

    Returns:
        IoU value between 0.0 and 1.0, where:
        - 0.0 means no overlap
        - 1.0 means perfect overlap
        - Returns 0.0 if union volume is zero

    Note:
        Only the first 6 values (center position and dimensions) are used.
        The boxes are assumed to be axis-aligned (rotation is ignored).
    """
    b1 = np.array(box1[:6])
    b2 = np.array(box2[:6])

    def to_min_max(b: np.ndarray) -> np.ndarray:
        """Convert center-size representation to min-max representation."""
        x, y, z, w, h, d = b
        return np.array([
            x - w / 2, x + w / 2,
            y - h / 2, y + h / 2,
            z - d / 2, z + d / 2
        ])

    mm1 = to_min_max(b1)
    mm2 = to_min_max(b2)

    inter_min = np.maximum(mm1[[0, 2, 4]], mm2[[0, 2, 4]])
    inter_max = np.minimum(mm1[[1, 3, 5]], mm2[[1, 3, 5]])
    inter = np.maximum(inter_max - inter_min, 0)
    inter_vol = np.prod(inter)

    vol1 = np.prod(mm1[[1, 3, 5]] - mm1[[0, 2, 4]])
    vol2 = np.prod(mm2[[1, 3, 5]] - mm2[[0, 2, 4]])

    union = vol1 + vol2 - inter_vol
    return float(inter_vol / union) if union > 0 else 0.0


def evaluate(
    pred_boxes: list[list[float]],
    pred_labels: list[str],
    gt_boxes: list[list[float]],
    gt_labels: list[str],
    iou_threshold: float = 0.5,
) -> dict[str, Union[float, list[float], int, str]]:
    """
    Evaluate predictions against ground truth using optimal matching (Hungarian algorithm).

    This function performs order-independent matching between predicted and ground truth
    boxes. It uses the Hungarian algorithm to find the optimal assignment that maximizes
    total IoU while ensuring:
    - Each ground truth box can match at most one prediction
    - Each prediction can match at most one ground truth
    - IoU matching ignores labels (matches purely based on IoU)

    Args:
        pred_boxes: List of predicted bounding boxes, each as [x, y, z, w, h, d, ...]
        pred_labels: List of labels for predicted boxes
        gt_boxes: List of ground truth bounding boxes, each as [x, y, z, w, h, d, ...]
        gt_labels: List of labels for ground truth boxes
        iou_threshold: IoU threshold for considering a match as correct (default: 0.5)

    Returns:
        Dictionary containing:
        - "matched_ious": List of IoU values for all matched pairs
        - "matched_labels": List of (gt_label, pred_label) tuples for matched pairs
        - "mean_iou": Mean IoU across all matched pairs
        - "iou_accuracy_percent": Percentage of ground truth boxes with IoU >= threshold (ignores label matching)
        - "label_accuracy_percent": Percentage of ground truth boxes with correct label match
        - "total_gt": Total number of ground truth boxes
        - "total_pred": Total number of predicted boxes
        - "matched_pairs": Number of successfully matched pairs

    Note:
        - If there are no ground truth boxes, accuracy is 100% if there are no predictions,
          otherwise 0%
        - IoU accuracy is calculated based purely on IoU values, ignoring label matches
        - Label matching extracts base names by removing suffixes (e.g., "Car_5" → "car", "van_100" → "van")
        - Labels are matched case-insensitively after normalization for label_accuracy_percent only
        - The matching is optimal in terms of maximizing total IoU
    """
    n_pred = len(pred_boxes)
    n_gt = len(gt_boxes)

    if n_gt == 0:
        return {
            "matched_ious": [],
            "matched_labels": [],
            "mean_iou": 0.0,
            "iou_accuracy_percent": 100.0 if n_pred == 0 else 0.0,
            "label_accuracy_percent": 100.0 if n_pred == 0 else 0.0,
            "total_gt": 0,
            "total_pred": n_pred,
            "matched_pairs": 0,
        }

    # Normalize labels for label accuracy calculation
    gt_labels_normalized = [normalize_label(label) for label in gt_labels]
    pred_labels_normalized = [normalize_label(label) for label in pred_labels]

    # Cost matrix (1 - IoU) - match purely based on IoU, ignoring labels
    cost = np.ones((n_gt, n_pred))

    for i in range(n_gt):
        for j in range(n_pred):
            iou = compute_3d_iou(gt_boxes[i], pred_boxes[j])
            cost[i][j] = 1 - iou        # cost = 1 - IoU

    # Run Hungarian matching based on IoU only (no label restriction)
    gt_idx, pred_idx = linear_sum_assignment(cost)

    matched_ious = []
    matched_labels = []
    correct = 0
    label_correct = 0

    for g, p in zip(gt_idx, pred_idx):
        iou = 1 - cost[g][p]
        matched_ious.append(iou)
        # Store original labels (not normalized) for reporting
        matched_labels.append((gt_labels[g], pred_labels[p]))

        # Check if normalized labels match for label accuracy
        if gt_labels_normalized[g] == pred_labels_normalized[p]:
            label_correct += 1

        # IoU accuracy ignores label matching - count if IoU >= threshold
        if iou >= iou_threshold:
            correct += 1

    accuracy = (correct / len(gt_boxes)) * 100 if len(gt_boxes) > 0 else 0.0
    label_accuracy = (label_correct / len(gt_boxes)) * 100 if len(gt_boxes) > 0 else 0.0

    return {
        "matched_ious": matched_ious,
        "matched_labels": matched_labels,
        "mean_iou": float(np.mean(matched_ious)) if matched_ious else 0.0,
        "iou_accuracy_percent": accuracy,
        "label_accuracy_percent": label_accuracy,
        "total_gt": len(gt_boxes),
        "total_pred": len(pred_boxes),
        "matched_pairs": len(matched_ious),
    }


def main() -> None:
    """
    Main function to compare prediction and ground truth JSON files file-by-file.

    This function:
    1. Parses command-line arguments
    2. Finds all JSON files in both prediction and ground truth directories
    3. Matches files by filename
    4. Evaluates each matching file pair
    5. Aggregates results and prints summary statistics

    Command-line arguments:
        pred_dir: Directory containing prediction JSON files
        gt_dir: Directory containing ground truth JSON files
        --iou-threshold: IoU threshold for correct matches (default: 0.5)
        --verbose: Print per-file results in addition to overall summary

    Output:
        Prints overall statistics including:
        - Number of files processed
        - Mean IoU across all matches
        - Overall accuracy percentage
        - Total ground truth and prediction counts

        If --verbose is set, also prints per-file results.

    Raises:
        ValueError: If either directory doesn't exist
    """
    parser = argparse.ArgumentParser(
        description="Evaluate 3D bounding box predictions against ground truth"
    )
    parser.add_argument(
        "pred_dir",
        type=str,
        help="Directory containing prediction JSON files",
    )
    parser.add_argument(
        "gt_dir",
        type=str,
        help="Directory containing ground truth JSON files",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="IoU threshold for considering a match correct (default: 0.5)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-file results",
    )

    args = parser.parse_args()

    pred_dir = Path(args.pred_dir)
    gt_dir = Path(args.gt_dir)

    if not pred_dir.exists():
        raise ValueError(f"Prediction directory does not exist: {pred_dir}")
    if not gt_dir.exists():
        raise ValueError(f"Ground truth directory does not exist: {gt_dir}")

    # Find all JSON files in both directories
    pred_files = {f.name: f for f in pred_dir.glob("*.json")}
    gt_files = {f.name: f for f in gt_dir.glob("*.json")}

    # Find common files
    common_files = set(pred_files.keys()) & set(gt_files.keys())
    pred_only = set(pred_files.keys()) - set(gt_files.keys())
    gt_only = set(gt_files.keys()) - set(pred_files.keys())

    if not common_files:
        print("No matching files found between the two directories!")
        print(f"Prediction-only files: {len(pred_only)}")
        print(f"Ground truth-only files: {len(gt_only)}")
        return

    print(f"Found {len(common_files)} matching files")
    if pred_only:
        print(f"Warning: {len(pred_only)} files only in prediction directory")
    if gt_only:
        print(f"Warning: {len(gt_only)} files only in ground truth directory")
    print()

    # Evaluate each file pair
    per_file_results = []

    for filename in sorted(common_files):
        pred_path = pred_files[filename]
        gt_path = gt_files[filename]

        try:
            pred_boxes, pred_labels = load_boxes(pred_path)
            gt_boxes, gt_labels = load_boxes(gt_path)

            result = evaluate(pred_boxes, pred_labels, gt_boxes, gt_labels, args.iou_threshold)
            result["filename"] = filename
            per_file_results.append(result)

            if args.verbose:
                print(f"{filename}:")
                print(f"  Mean IoU: {result['mean_iou']:.4f}")
                print(f"  IoU Accuracy: {result['iou_accuracy_percent']:.2f}%")
                print(f"  Label Accuracy: {result['label_accuracy_percent']:.2f}%")
                print(
                    f"  GT boxes: {result['total_gt']}, Pred boxes: {result['total_pred']}, "
                    f"Matched: {result['matched_pairs']}")
                print()

        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue

    # Aggregate results
    if per_file_results:
        all_matched_ious = []
        total_gt = 0
        total_pred = 0
        total_correct = 0
        total_label_correct = 0
        total_matched = 0

        for result in per_file_results:
            all_matched_ious.extend(result["matched_ious"])
            total_gt += result["total_gt"]
            total_pred += result["total_pred"]
            total_matched += result["matched_pairs"]
            # Calculate correct from accuracy
            correct = int((result["iou_accuracy_percent"] / 100.0) * result["total_gt"])
            total_correct += correct
            label_correct = int((result["label_accuracy_percent"] / 100.0) * result["total_gt"])
            total_label_correct += label_correct

        overall_result = {
            "num_files": len(per_file_results),
            "mean_iou": float(np.mean(all_matched_ious)) if all_matched_ious else 0.0,
            "iou_accuracy_percent": (total_correct / total_gt * 100) if total_gt > 0 else 0.0,
            "label_accuracy_percent": (total_label_correct / total_gt * 100) if total_gt > 0 else 0.0,
            "total_gt": total_gt,
            "total_pred": total_pred,
            "total_matched": total_matched,
            "total_correct": total_correct,
            "total_label_correct": total_label_correct,
        }

        print("=" * 60)
        print("Overall Results:")
        print("=" * 60)
        print(json.dumps(overall_result, indent=2))

        if args.verbose:
            print("\nPer-file results:")
            print(json.dumps(per_file_results, indent=2))


# add a function to evaluate the results
def evaluate_results():
    """
    Evaluate the results of the 3D grounding task.
    """
    # setup a dummy results
    dummy_results = {
        "pred_boxes": [[1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 5, 6, 7, 8, 9]],
        "pred_labels": ["car", "car", "car"],
        "gt_boxes": [[1, 2, 3, 41, 5, 6, 7, 8, 9], [1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 5, 6, 7, 8, 9]],
        "gt_labels": ["car", "van", "truck"],
    }
    iou_threshold = 0.5
    result = evaluate(
        dummy_results["pred_boxes"], dummy_results["pred_labels"],
        dummy_results["gt_boxes"], dummy_results["gt_labels"], iou_threshold)
    return result


if __name__ == "__main__":
    print(evaluate_results())
    # main()
